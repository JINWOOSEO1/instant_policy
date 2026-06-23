"""
Convert an alternative RGBD demo recording into the directory layout that
deploy_mujoco.py consumes via mujoco_scripts.result_io.load_demo_from_results().

Source layout (produced by collect_mug_rack_rgbd.py, extended with per-frame
EE pose and gripper state):

    {input_dir}/
      cameras.json                # OpenCV intrinsics + world->camera extrinsics
      frames/
        000000/
          rgb_0.png  depth_0.npy  # one rgb/depth pair per camera
          rgb_1.png  depth_1.npy
          rgb_2.png  depth_2.npy
          gripper_tcp_pose.npy    # 4x4 T_w_e (EE pose in world)
          gripper_open.npy        # scalar bool, True = open
        000001/ ...

The source has no segmentation masks, so object masks are produced interactively
with SAM2: click the object on each camera's first frame, then track it through
every frame -- identical to deploy_mujoco.py --sam2. The masked multi-view depth
is deprojected to world frame and merged into one object-only pointcloud per
frame.

Output layout (read by load_demo_from_results):

    results/{name}/demo/demo_{demo_index}/
      seg_pcd/0000.npy ...        # object-only world-frame pointclouds (N, 3)
      T_w_e/0000.npy ...          # 4x4 EE poses
      gripper_state.npy           # int array, 1 = open, 0 = closed

Usage:
    python -m mujoco_scripts.demo_data_inverter results/06181538_kettle --name kettle
"""

import argparse
import glob
import json
import os
import shutil
import tempfile

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from utils import downsample_pcd, transform_pcd
from mujoco_scripts.camera_utils import depth_to_pointcloud
from mujoco_scripts.demo_generation import (
    _setup_sam2_torch,
    interactive_mask_selection,
    sam2_cuda_extension_available,
    track_masks_with_video_predictor,
)
from mujoco_scripts.result_paths import (
    get_demo_gripper_state_path,
    get_demo_pose_dir,
    get_demo_seg_pcd_dir,
    get_object_root,
)


def load_camera_views(cameras_json_path):
    """Load ordered (intrinsic, extrinsic, name) tuples from cameras.json.

    `extrinsic` is the world->camera (OpenCV convention) transform, matching
    collect_mug_rack_rgbd.get_opencv_camera_matrices(). Its inverse maps
    camera-frame points back into the world frame.
    """
    with open(cameras_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    views = []
    for view in data["views"]:
        intrinsic = np.asarray(view["intrinsic"], dtype=np.float64)
        extrinsic = np.asarray(view["extrinsic"], dtype=np.float64)
        views.append((intrinsic, extrinsic, view.get("camera_name", "")))
    return views


def list_frame_dirs(input_dir):
    """Return frame directories under {input_dir}/frames sorted by index."""
    frames_root = os.path.join(input_dir, "frames")
    if not os.path.isdir(frames_root):
        raise FileNotFoundError(f"Frames directory not found: {frames_root}")

    frame_dirs = sorted(
        os.path.join(frames_root, name)
        for name in os.listdir(frames_root)
        if os.path.isdir(os.path.join(frames_root, name))
    )
    if not frame_dirs:
        raise FileNotFoundError(f"No frame directories found in {frames_root}")
    return frame_dirs


def make_camera_jpg_dir(frame_dirs, cam_idx):
    """Write one camera's RGB frames as an ordered jpg sequence for SAM2.

    SAM2VideoPredictor.init_state() expects a directory of frame images whose
    filenames sort into temporal order, so frames are named {order:05d}.jpg.
    """
    tmp_dir = tempfile.mkdtemp(prefix=f"sam2_cam{cam_idx}_")
    for order, frame_dir in enumerate(frame_dirs):
        rgb_path = os.path.join(frame_dir, f"rgb_{cam_idx}.png")
        bgr = cv2.imread(rgb_path)  # correct-colored image on disk -> BGR in memory
        if bgr is None:
            raise FileNotFoundError(f"RGB frame not found: {rgb_path}")
        cv2.imwrite(os.path.join(tmp_dir, f"{order:05d}.jpg"), bgr)
    return tmp_dir


def load_first_frame_rgb(frame_dir, cam_idx):
    """Load a frame's RGB image as an HxWx3 uint8 RGB array."""
    rgb_path = os.path.join(frame_dir, f"rgb_{cam_idx}.png")
    bgr = cv2.imread(rgb_path)
    if bgr is None:
        raise FileNotFoundError(f"RGB frame not found: {rgb_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def masked_depth_to_world_pcd(depth, mask, intrinsic, extrinsic):
    """Deproject masked depth into a world-frame pointcloud.

    `extrinsic` maps world->camera (OpenCV), so its inverse maps the deprojected
    camera-frame points into the world frame. No OpenGL flip is applied because
    the depth and extrinsic already share the OpenCV camera convention.
    """
    depth_masked = depth * mask.astype(np.float32)
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    pcd_cam = depth_to_pointcloud(depth_masked, fx, fy, cx, cy)
    if len(pcd_cam) == 0:
        return None
    return transform_pcd(pcd_cam, np.linalg.inv(extrinsic))


def build_sam2_masks(frame_dirs, views, sam2_config, sam2_ckpt):
    """Interactively segment + track each camera, returning per-camera masks.

    Returns {cam_idx: {frame_order: bool_mask}} and the list of temp jpg dirs
    created for tracking (so the caller can clean them up).
    """
    import torch  # noqa: F401  (configures SAM2 runtime + used downstream)
    from sam2_repo.sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2_repo.sam2.sam2_image_predictor import SAM2ImagePredictor

    _setup_sam2_torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    apply_postprocessing = sam2_cuda_extension_available()
    if not apply_postprocessing:
        print(
            "SAM2 CUDA extension `sam2._C` is not available. "
            "Disabling SAM2 post-processing to avoid runtime warnings."
        )

    print("Loading SAM2 image predictor...")
    image_predictor = SAM2ImagePredictor(
        build_sam2(
            sam2_config,
            sam2_ckpt,
            device=device,
            apply_postprocessing=apply_postprocessing,
        )
    )

    # Interactive keypoint selection on each camera's first frame.
    cam_points = {}
    cam_labels = {}
    for cam_idx, (_intrinsic, _extrinsic, cam_name) in enumerate(views):
        cam_label = f"cam{cam_idx} ({cam_name})" if cam_name else f"cam{cam_idx}"
        print(f"\n--- {cam_label}: select object keypoints ---")
        rgb = load_first_frame_rgb(frame_dirs[0], cam_idx)
        points, labels = interactive_mask_selection(rgb, image_predictor, cam_label)
        cam_points[cam_idx] = points
        cam_labels[cam_idx] = labels
        print(f"  {len(points)} keypoint(s) confirmed.")

    print("\nLoading SAM2 video predictor...")
    video_predictor = build_sam2_video_predictor(
        sam2_config,
        sam2_ckpt,
        device=device,
        apply_postprocessing=apply_postprocessing,
        vos_optimized=False,
    )

    cam_masks = {}
    tmp_dirs = []
    for cam_idx, (_intrinsic, _extrinsic, cam_name) in enumerate(views):
        cam_label = f"cam{cam_idx} ({cam_name})" if cam_name else f"cam{cam_idx}"
        print(f"\n--- Tracking {cam_label} ({len(frame_dirs)} frames) ---")
        jpg_dir = make_camera_jpg_dir(frame_dirs, cam_idx)
        tmp_dirs.append(jpg_dir)
        cam_masks[cam_idx] = track_masks_with_video_predictor(
            video_predictor,
            jpg_dir,
            cam_points[cam_idx],
            cam_labels[cam_idx],
            len(frame_dirs),
        )
        print(f"  Tracked {len(cam_masks[cam_idx])} masks.")

    return cam_masks, tmp_dirs


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Reformat an RGBD demo recording into deploy_mujoco's "
        "load_demo_from_results layout, segmenting the object with SAM2."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Source recording folder (contains cameras.json and frames/).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="kettle",
        help="Output object name -> results/{name}/ (default: kettle).",
    )
    parser.add_argument(
        "--demo_index",
        type=int,
        default=0,
        help="Demo index written under results/{name}/demo/demo_{index}.",
    )
    parser.add_argument(
        "--skip_frames",
        type=int,
        default=100,
        help="Number of leading frames to discard before processing (default: 100).",
    )
    parser.add_argument(
        "--tail_frames",
        type=int,
        default=100,
        help="Number of trailing frames to discard before processing (default: 100).",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.005,
        help="Voxel size for pointcloud downsampling (<=0 disables).",
    )
    parser.add_argument(
        "--mask_erosion",
        type=int,
        default=3,
        help="Erode SAM2 mask by this many pixels at boundaries (0 disables).",
    )
    parser.add_argument(
        "--sam2_config",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_s.yaml",
        help="SAM2 config path.",
    )
    parser.add_argument(
        "--sam2_ckpt",
        type=str,
        default="sam2_repo/checkpoints/sam2.1_hiera_small.pt",
        help="SAM2 checkpoint path.",
    )
    parser.add_argument(
        "--robot_base_pos",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Robot base position in source world frame (metres). "
             "When provided, all output seg_pcd and T_w_e are re-expressed "
             "in the robot-base frame.  Example: --robot_base_pos 0 -0.35 0.42",
    )
    parser.add_argument(
        "--robot_base_euler",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("RX", "RY", "RZ"),
        help="Robot base orientation in source world frame (radians, XYZ Euler). "
             "Used together with --robot_base_pos. "
             "Example: --robot_base_euler 0 0 1.5708",
    )
    args = parser.parse_args(argv)

    input_dir = args.input_dir.rstrip("/")
    cameras_json_path = os.path.join(input_dir, "cameras.json")
    if not os.path.exists(cameras_json_path):
        raise FileNotFoundError(f"cameras.json not found: {cameras_json_path}")

    # ── Robot-base frame transform ────────────────────────────────────────────
    if args.robot_base_pos is not None:
        R_w_base = Rotation.from_euler("xyz", args.robot_base_euler).as_matrix()
        T_w_base = np.eye(4)
        T_w_base[:3, :3] = R_w_base
        T_w_base[:3,  3] = args.robot_base_pos
        T_base_w = np.linalg.inv(T_w_base)
        print(
            f"Robot base pos={args.robot_base_pos}  euler={args.robot_base_euler} — "
            "output will be re-expressed in robot-base frame."
        )
    else:
        T_base_w = None

    views = load_camera_views(cameras_json_path)
    frame_dirs = list_frame_dirs(input_dir)
    num_views = len(views)
    if args.skip_frames > 0:
        print(f"Skipping first {args.skip_frames} frame(s).")
        frame_dirs = frame_dirs[args.skip_frames:]
    if args.tail_frames > 0:
        print(f"Skipping last {args.tail_frames} frame(s).")
        frame_dirs = frame_dirs[:-args.tail_frames]
    num_frames = len(frame_dirs)
    print(
        f"Loaded {num_views} camera view(s) and {num_frames} frame(s) "
        f"from {input_dir}."
    )

    # Prepare output directories.
    seg_pcd_dir = get_demo_seg_pcd_dir(args.name, args.demo_index)
    pose_dir = get_demo_pose_dir(args.name, args.demo_index)
    gripper_state_path = get_demo_gripper_state_path(args.name, args.demo_index)
    os.makedirs(seg_pcd_dir, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)

    # Segment the object with SAM2 (interactive keypoints + video tracking).
    cam_masks, tmp_dirs = build_sam2_masks(
        frame_dirs,
        views,
        args.sam2_config,
        args.sam2_ckpt,
    )

    try:
        print("\nGenerating segmented pointclouds...")
        gripper_states = []
        for order, frame_dir in enumerate(frame_dirs):
            # ── Segmented world-frame pointcloud (merge all cameras) ──────────
            cam_pcds = []
            for cam_idx, (intrinsic, extrinsic, _name) in enumerate(views):
                mask = cam_masks.get(cam_idx, {}).get(order)
                if mask is None or mask.sum() == 0:
                    continue
                if args.mask_erosion > 0:
                    kernel = np.ones((3, 3), np.uint8)
                    mask = cv2.erode(
                        mask.astype(np.uint8), kernel, iterations=args.mask_erosion
                    ).astype(bool)
                if mask.sum() == 0:
                    continue
                depth_path = os.path.join(frame_dir, f"depth_{cam_idx}.npy")
                depth = np.load(depth_path)
                pcd_world = masked_depth_to_world_pcd(depth, mask, intrinsic, extrinsic)
                if pcd_world is not None and len(pcd_world) > 0:
                    cam_pcds.append(pcd_world)

            if cam_pcds:
                pcd_merged = np.concatenate(cam_pcds, axis=0)
                if args.voxel_size > 0:
                    pcd_merged = downsample_pcd(pcd_merged, voxel_size=args.voxel_size)
            else:
                print(f"  Frame {order}: empty mask -- saving empty pointcloud.")
                pcd_merged = np.zeros((0, 3), dtype=np.float32)

            if T_base_w is not None and len(pcd_merged) > 0:
                pcd_merged = transform_pcd(pcd_merged, T_base_w)

            np.save(
                os.path.join(seg_pcd_dir, f"{order:04d}.npy"),
                pcd_merged.astype(np.float32),
            )

            # ── EE pose ───────────────────────────────────────────────────────
            T_w_e = np.load(os.path.join(frame_dir, "gripper_tcp_pose.npy"))
            if T_base_w is not None:
                T_w_e = T_base_w @ T_w_e
            np.save(os.path.join(pose_dir, f"{order:04d}.npy"), T_w_e)

            # ── Gripper state (1 = open, 0 = closed) ──────────────────────────
            gripper_open = bool(np.load(os.path.join(frame_dir, "gripper_open.npy")))
            gripper_states.append(int(gripper_open))

            if order % 50 == 0 or order == num_frames - 1:
                print(f"  Frame {order}/{num_frames}: {len(pcd_merged)} points")

        np.save(gripper_state_path, np.asarray(gripper_states, dtype=np.int32))
    finally:
        for tmp_dir in tmp_dirs:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    object_root = get_object_root(args.name)
    print(
        f"\nDone! Wrote {num_frames} frames to {object_root}.\n"
        f"  seg_pcd:       {seg_pcd_dir}\n"
        f"  T_w_e:         {pose_dir}\n"
        f"  gripper_state: {gripper_state_path}\n"
        f"Deploy with: python -m mujoco_scripts.deploy_mujoco "
        f"--object {args.name} --sam2"
    )


if __name__ == "__main__":
    main()
