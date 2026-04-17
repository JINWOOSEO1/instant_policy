"""
Demo generation for Instant Policy MuJoCo environments.
Supports rule-based and teleoperation demo collection,
with optional SAM2 or MuJoCo GT segmentation mask generation.

Usage:
    # Rule-based demo with GT masks (MuJoCo seg_renderer)
    python -m mujoco_scripts.demo_generation --object box --rule

    # Rule-based demo with SAM2 masks
    python -m mujoco_scripts.demo_generation --object box --rule --sam2

    # Teleop demo with GT masks
    python -m mujoco_scripts.demo_generation --object mug

    # Teleop demo with SAM2 masks
    python -m mujoco_scripts.demo_generation --object mug --sam2
"""

import argparse
import glob
import importlib
import os
import sys
import time

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from mujoco_scripts.camera_utils import load_camera_entries
from mujoco_scripts.result_paths import (
    get_demo_dir,
    get_demo_mask_dir,
    get_demo_pose_dir,
    get_demo_rgbd_dir,
    get_demo_root,
    get_object_root,
    resolve_demo_rgbd_dir,
)
from mujoco_scripts.simulation import MujocoEnv, mujoco_quat_to_mat, scipy_quat_to_mujoco


# ─── Object segmentation config ──────────────────────────────────────────────

OBJECT_GEOM_NAMES = {
    'mug': [
        # mug body
        'mug_bottom',
        'mug_wall_00', 'mug_wall_01', 'mug_wall_02', 'mug_wall_03',
        'mug_wall_04', 'mug_wall_05', 'mug_wall_06', 'mug_wall_07',
        'mug_wall_08', 'mug_wall_09', 'mug_wall_10', 'mug_wall_11',
        'mug_handle_top', 'mug_handle_mid', 'mug_handle_bot', 'mug_handle_low',
        # mug_rack body
        'rack_base', 'rack_post',
        'rack_branch_top_1', 'rack_branch_top_2',
        'rack_branch_mid_1', 'rack_branch_mid_2',
    ],
    'box': [
        # small_box body
        'small_box_geom',
        # target_box body
        'target_box_bottom',
        'target_box_wall_pos_x', 'target_box_wall_neg_x',
        'target_box_wall_pos_y', 'target_box_wall_neg_y',
    ],
    # box_2 reuses the same geom names as the box task variant.
    'box_2': [
        'small_box_geom',
        'target_box_bottom',
        'target_box_wall_pos_x', 'target_box_wall_neg_x',
        'target_box_wall_pos_y', 'target_box_wall_neg_y',
    ],
}


# ─── Trajectory utilities ────────────────────────────────────────────────────

def smoothstep(t):
    """Cubic easing for smoother waypoint interpolation."""
    t = np.clip(float(t), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def interpolate_linear(start, end, t):
    """Interpolate between two vectors with smooth easing."""
    alpha = smoothstep(t)
    return (1.0 - alpha) * start + alpha * end


def allocate_segment_frames(total_frames, weights):
    """Distribute total_frames across segments while preserving ratios."""
    weights = np.asarray(weights, dtype=np.float64)
    if total_frames < len(weights):
        raise ValueError('total_frames must be at least the number of segments')

    raw = weights / weights.sum() * total_frames
    frames = np.floor(raw).astype(np.int32)
    frames = np.maximum(frames, 1)

    deficit = total_frames - int(frames.sum())
    if deficit > 0:
        order = np.argsort(-(raw - np.floor(raw)))
        for idx in order[:deficit]:
            frames[idx] += 1
    elif deficit < 0:
        order = np.argsort(raw - np.floor(raw))
        for idx in order:
            if deficit == 0:
                break
            removable = frames[idx] - 1
            if removable <= 0:
                continue
            delta = min(removable, -deficit)
            frames[idx] -= delta
            deficit += delta

    if int(frames.sum()) != total_frames:
        raise RuntimeError('Failed to allocate rule trajectory frames')

    return frames.tolist()


def solve_ik_pose(
    model,
    data,
    ee_site_id,
    joint_range_low,
    joint_range_high,
    target_pos,
    target_quat_xyzw,
    *,
    step_size=0.5,
    damping=1e-4,
    max_iters=300,
    tol=1e-6,
):
    """Solve one TCP pose kinematically and report the residual error.

    The current simulation state is restored before returning, so this helper
    can be used safely for trajectory planning.
    """
    qpos_save = data.qpos.copy()
    qvel_save = data.qvel.copy()
    ctrl_save = data.ctrl.copy()

    target_quat_mj = scipy_quat_to_mujoco(target_quat_xyzw)

    for _ in range(max_iters):
        mujoco.mj_forward(model, data)
        current_pos = data.site_xpos[ee_site_id].copy()
        current_mat = data.site_xmat[ee_site_id].reshape(3, 3)

        err_pos = target_pos - current_pos
        err_rot = R.from_matrix(
            mujoco_quat_to_mat(target_quat_mj) @ current_mat.T
        ).as_rotvec()
        err = np.concatenate([err_pos, err_rot])

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, ee_site_id)
        J = np.vstack([jacp[:, :7], jacr[:, :7]])

        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), err)
        dq *= step_size
        data.qpos[:7] = np.clip(
            data.qpos[:7] + dq,
            joint_range_low,
            joint_range_high,
        )

        if np.max(np.abs(dq)) < tol:
            break

    mujoco.mj_forward(model, data)
    final_pos = data.site_xpos[ee_site_id].copy()
    final_mat = data.site_xmat[ee_site_id].reshape(3, 3)
    pos_err = np.linalg.norm(target_pos - final_pos)
    rot_err = np.linalg.norm(
        R.from_matrix(mujoco_quat_to_mat(target_quat_mj) @ final_mat.T).as_rotvec()
    )

    data.qpos[:] = qpos_save
    data.qvel[:] = qvel_save
    data.ctrl[:] = ctrl_save
    mujoco.mj_forward(model, data)

    return float(pos_err), float(rot_err)


def select_reachable_clearance_z(
    env,
    home_quat,
    grasp_pos,
    target_pos,
    preferred_clearance_z,
    min_clearance_z,
    *,
    num_candidates=40,
    num_path_samples=7,
    pos_tol=0.05,
    rot_tol=0.1,
):
    """Pick the highest transport height that remains kinematically reachable."""
    if preferred_clearance_z <= min_clearance_z:
        return float(min_clearance_z)

    candidate_zs = np.linspace(preferred_clearance_z, min_clearance_z, num_candidates)
    sample_ts = np.linspace(0.0, 1.0, num_path_samples)

    for candidate_z in candidate_zs:
        reachable = True
        for t in sample_ts:
            xy = (1.0 - t) * grasp_pos[:2] + t * target_pos[:2]
            sample_pos = np.array([xy[0], xy[1], candidate_z], dtype=np.float64)
            pos_err, rot_err = solve_ik_pose(
                env.model,
                env.data,
                env.ee_site_id,
                env.joint_range_low,
                env.joint_range_high,
                sample_pos,
                home_quat,
            )
            if pos_err > pos_tol or rot_err > rot_tol:
                reachable = False
                break

        if reachable:
            return float(candidate_z)

    return float(min_clearance_z)


# ─── Demo collection helpers ─────────────────────────────────────────────────

def prepare_output_dirs(object_name, demo_index):
    """Create the standard output directory structure."""
    object_root = get_object_root(object_name)
    demo_root = get_demo_root(object_name)
    demo_dir = get_demo_dir(object_name, demo_index)
    rgbd_dir = get_demo_rgbd_dir(object_name, demo_index)
    pose_dir = get_demo_pose_dir(object_name, demo_index)
    os.makedirs(object_root, exist_ok=True)
    os.makedirs(demo_root, exist_ok=True)
    os.makedirs(demo_dir, exist_ok=True)
    os.makedirs(rgbd_dir, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)
    return object_root, demo_root, demo_dir, rgbd_dir, pose_dir


def resolve_object_geom_ids(env, object_name):
    """Return GT segmentation geom IDs for the configured object."""
    if object_name not in OBJECT_GEOM_NAMES:
        raise ValueError(
            f'GT segmentation is not configured for object "{object_name}". '
            f'Available: {sorted(OBJECT_GEOM_NAMES.keys())}'
        )

    geom_ids = env.get_geom_ids_by_names(OBJECT_GEOM_NAMES[object_name])
    print(f'GT segmentation: object="{object_name}", {len(geom_ids)} geom(s)')
    return geom_ids


def record_frame(env, rgbd_dir, pose_dir, frame_idx, gripper_states,
                 mask_dir=None, geom_ids=None):
    """Save images, EE pose, and gripper state for one frame.

    If mask_dir and geom_ids are provided, also saves GT segmentation masks
    using the MuJoCo seg_renderer.
    """
    rgb_imgs = []
    for i, cam_name in enumerate(env.cam_names):
        rgb, depth = env.render_rgbd(cam_name)
        cam_dir = os.path.join(rgbd_dir, f'cam{i}')
        os.makedirs(cam_dir, exist_ok=True)
        rgb_path = os.path.join(cam_dir, f'{frame_idx:04d}.jpg')
        cv2.imwrite(
            rgb_path,
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        depth_path = os.path.join(cam_dir, f'{frame_idx:04d}_depth.npy')
        np.save(depth_path, depth)
        rgb_imgs.append(rgb)

        # Save GT segmentation mask when not using SAM2
        if mask_dir is not None and geom_ids is not None:
            mask = env.render_seg_mask(cam_name, geom_ids)
            mask_path = os.path.join(mask_dir, f'cam{i}_{frame_idx:04d}_mask.npy')
            np.save(mask_path, mask)

    tiled = np.hstack([cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in rgb_imgs])
    cv2.imshow('Camera Views', tiled)
    cv2.waitKey(1)

    T_w_e = env.get_ee_pose()
    np.save(os.path.join(pose_dir, f'{frame_idx:04d}.npy'), T_w_e)
    gripper_states.append(env.get_gripper_state())


def finalize_demo(demo_dir, rgbd_dir, pose_dir, frame_count, gripper_states, env):
    """Save final metadata and close visualization resources."""
    gripper_state_path = os.path.join(demo_dir, 'gripper_state.npy')
    np.save(gripper_state_path, np.array(gripper_states, dtype=np.int32))
    print(f'Saved {frame_count} frames to {demo_dir}/')
    print(f'  RGBD images: {rgbd_dir}/')
    print(f'  EE poses: {pose_dir}/')
    print(f'  Gripper states: {gripper_state_path}')

    cv2.destroyAllWindows()
    env.close()


# ─── Rule-based trajectory ───────────────────────────────────────────────────

def build_box_rule_trajectory(env, total_frames):
    """Build a reachability-aware pick-and-place plan for box tasks."""
    if total_frames < 8:
        raise ValueError('Rule-based box demo needs at least 8 frames')

    ee_pose = env.get_ee_pose()
    home_pos = ee_pose[:3, 3].copy()
    home_quat = R.from_matrix(ee_pose[:3, :3]).as_quat()

    small_box_center = env.get_geom_position('small_box_geom')
    target_box_bottom = env.get_geom_position('target_box_bottom')

    small_box_geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, 'small_box_geom')
    target_bottom_geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, 'target_box_bottom')
    target_wall_geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, 'target_box_wall_pos_x')
    box_half_height = env.model.geom_size[small_box_geom_id][2]
    target_bottom_half_height = env.model.geom_size[target_bottom_geom_id][2]
    target_wall_half_height = env.model.geom_size[target_wall_geom_id][2]

    pregrasp_pos = small_box_center + np.array([0.0, 0.0, 0.11])
    grasp_pos = small_box_center.copy()

    # Prefer the original tall transport arc when it is reachable, but fall
    # back to the highest reachable arc for taller scenes such as box_2.
    preferred_clearance_z = max(
        home_pos[2] + 0.06,
        small_box_center[2] + 0.20,
        target_box_bottom[2] + 0.20,
    )
    small_box_top_z = small_box_center[2] + box_half_height
    target_wall_top_z = (
        env.get_geom_position('target_box_wall_pos_x')[2] + target_wall_half_height
    )
    min_clearance_z = max(small_box_top_z, target_wall_top_z) + 0.08
    clearance_z = select_reachable_clearance_z(
        env,
        home_quat,
        grasp_pos,
        target_box_bottom,
        preferred_clearance_z,
        min_clearance_z,
    )
    if clearance_z < preferred_clearance_z - 1e-6:
        print(
            f'Adjusted rule transport height for {env.object_name}: '
            f'{preferred_clearance_z:.3f} -> {clearance_z:.3f}'
        )

    lift_pos = np.array([grasp_pos[0], grasp_pos[1], clearance_z])
    transport_pos = np.array([target_box_bottom[0], target_box_bottom[1], clearance_z])

    insert_height = target_box_bottom[2] + target_bottom_half_height + box_half_height + 0.003
    insert_pos = np.array([target_box_bottom[0], target_box_bottom[1], insert_height])
    retreat_pos = insert_pos + np.array([0.0, 0.0, 0.06])

    phase_specs = [
        ('approach_box', home_pos, pregrasp_pos, 255.0, 255.0),
        ('descend_to_grasp', pregrasp_pos, grasp_pos, 255.0, 255.0),
        ('close_gripper', grasp_pos, grasp_pos, 255.0, 0.0),
        ('lift_box', grasp_pos, lift_pos, 0.0, 0.0),
        ('move_to_target', lift_pos, transport_pos, 0.0, 0.0),
        ('lower_into_target', transport_pos, insert_pos, 0.0, 0.0),
        ('release_box', insert_pos, insert_pos, 0.0, 255.0),
        ('retreat_up', insert_pos, retreat_pos, 255.0, 255.0),
    ]
    phase_frames = allocate_segment_frames(total_frames, [15, 15, 10, 15, 20, 10, 10, 5])

    trajectory = []
    for (phase_name, start_pos, end_pos, start_grip, end_grip), num_frames in zip(
        phase_specs,
        phase_frames,
    ):
        for local_idx in range(num_frames):
            t = 1.0 if num_frames == 1 else local_idx / (num_frames - 1)
            trajectory.append({
                'phase': phase_name,
                'arm_pos': interpolate_linear(start_pos, end_pos, t),
                'arm_quat': home_quat.copy(),
                'gripper_val': float(interpolate_linear(
                    np.array([start_grip]),
                    np.array([end_grip]),
                    t,
                )[0]),
            })

    return trajectory


# ─── SAM2 mask generation ────────────────────────────────────────────────────

def _setup_sam2_torch():
    """Configure torch optimizations for SAM2 inference."""
    import torch
    import torch._inductor.config as inductor_cfg

    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    inductor_cfg.fx_graph_cache = True
    if hasattr(inductor_cfg, "fx_graph_remote_cache"):
        inductor_cfg.fx_graph_remote_cache = False


def sam2_cuda_extension_available():
    """Return whether the optional SAM2 CUDA extension is importable."""
    try:
        importlib.import_module("sam2._C")
        return True
    except Exception:
        return False


def show_fixed_window(window_name, image_bgr):
    """Show an image in a resizable OpenCV window with a fixed pixel size."""
    height, width = image_bgr.shape[:2]
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window_name, width, height)
    cv2.imshow(window_name, image_bgr)


class KeypointSelector:
    """Interactive OpenCV keypoint selection."""

    def __init__(self, window_name='Select Keypoints', display_scale=3):
        self.window_name = window_name
        self.display_scale = display_scale
        self.points = []
        self.labels = []  # 1 = foreground, 0 = background
        self.image = None
        self.display = None

    def mouse_callback(self, event, x, y, flags, param):
        x = min(x // self.display_scale, self.image.shape[1] - 1)
        y = min(y // self.display_scale, self.image.shape[0] - 1)

        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append([x, y])
            self.labels.append(1)
            self._draw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.points.append([x, y])
            self.labels.append(0)
            self._draw()

    def _draw(self):
        self.display = self.image.copy()
        for (x, y), label in zip(self.points, self.labels):
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            cv2.circle(self.display, (x, y), 5, color, -1)
            cv2.circle(self.display, (x, y), 7, color, 2)
        show_fixed_window(self.window_name, self._get_scaled_display())

    def _get_scaled_display(self):
        return cv2.resize(
            self.display,
            None,
            fx=self.display_scale,
            fy=self.display_scale,
            interpolation=cv2.INTER_NEAREST,
        )

    def select(self, image_rgb):
        """Show image, let user click points. Press 't' to confirm, 'r' to reset."""
        self.image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        self.display = self.image.copy()
        self.points = []
        self.labels = []

        show_fixed_window(self.window_name, self._get_scaled_display())
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print(f"  Left-click: foreground point | Right-click: background point")
        print(f"  Press 't' to confirm | Press 'r' to reset points")

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('t') and len(self.points) > 0:
                break
            elif key == ord('r'):
                self.points = []
                self.labels = []
                self.display = self.image.copy()
                show_fixed_window(self.window_name, self._get_scaled_display())
                print("  Points reset. Click again.")

        return np.array(self.points), np.array(self.labels)


def show_mask_overlay(image_rgb, mask, window_name='Mask Preview', display_scale=3):
    """Show mask overlaid on image."""
    mask = mask.astype(bool)
    overlay = image_rgb.copy()
    overlay[mask] = overlay[mask] * 0.5 + np.array([0, 255, 0]) * 0.5
    display = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)
    display = cv2.resize(
        display,
        None,
        fx=display_scale,
        fy=display_scale,
        interpolation=cv2.INTER_NEAREST,
    )
    show_fixed_window(window_name, display)


def interactive_mask_selection(image_rgb, predictor, cam_label, return_mask=False):
    """Closed-loop mask selection: click points, preview mask, accept or retry."""
    selector = KeypointSelector(window_name=f'{cam_label} - Select Keypoints')

    while True:
        points, labels = selector.select(image_rgb)

        predictor.set_image(image_rgb)
        masks, scores, logits = predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )

        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        show_mask_overlay(image_rgb, best_mask, window_name=f'{cam_label} - Mask Preview')

        print(f"  Mask score: {scores[best_idx]:.3f}")
        print(f"  Press 't' to accept | Press 'r' to retry with new points")

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('t'):
                cv2.destroyAllWindows()
                if return_mask:
                    return points, labels, best_mask
                return points, labels
            elif key == ord('r'):
                cv2.destroyWindow(f'{cam_label} - Mask Preview')
                break
            elif key == ord('q'):
                print("  Exiting.")
                sys.exit(0)


def track_masks_with_video_predictor(video_predictor, jpg_dir, points, labels, num_frames):
    """Track masks through video using SAM2VideoPredictor."""
    import torch

    state = video_predictor.init_state(video_path=jpg_dir)

    _, _, _ = video_predictor.add_new_points_or_box(
        state,
        frame_idx=0,
        obj_id=1,
        points=points,
        labels=labels,
    )

    masks = {}
    with torch.inference_mode():
        for frame_idx, obj_ids, mask_logits in video_predictor.propagate_in_video(state):
            mask = (mask_logits[0, 0] > 0).cpu().numpy()
            masks[frame_idx] = mask

    return masks


def generate_sam2_masks(object_name, demo_index, sam2_config, sam2_ckpt):
    """Generate masks using SAM2 (interactive keypoints + video tracking).

    Runs after demo recording is complete. Reads saved RGBD images,
    lets user select keypoints on first frame, then tracks through all frames.
    """
    import torch
    from PIL import Image
    from sam2_repo.sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2_repo.sam2.sam2_image_predictor import SAM2ImagePredictor

    _setup_sam2_torch()

    object_root = get_object_root(object_name)
    rgbd_dir = resolve_demo_rgbd_dir(object_name, demo_index)
    mask_dir = get_demo_mask_dir(object_name, demo_index)
    os.makedirs(mask_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    has_sam2_cuda_ext = sam2_cuda_extension_available()
    apply_postprocessing = has_sam2_cuda_ext

    if not has_sam2_cuda_ext:
        print(
            "SAM2 CUDA extension `sam2._C` is not available. "
            "Disabling SAM2 post-processing to avoid runtime warnings."
        )

    # Build SAM2 image predictor for interactive mask preview
    print('Loading SAM2 image predictor...')
    sam2_model = build_sam2(
        sam2_config,
        sam2_ckpt,
        device=device,
        apply_postprocessing=apply_postprocessing,
    )
    image_predictor = SAM2ImagePredictor(sam2_model)

    camera_entries = load_camera_entries(object_root, rgbd_dir=rgbd_dir)
    if not camera_entries:
        print(f'No camera metadata found in {object_root}')
        return

    # Interactive keypoint selection per camera
    cam_points = {}
    cam_labels = {}

    for camera in camera_entries:
        cam_idx = camera['index']
        cam_dir_name = camera['dir']
        cam_label = f'{cam_dir_name} ({camera["name"]})'
        print(f'\n--- {cam_label} ---')

        cam_dir = os.path.join(rgbd_dir, cam_dir_name)
        first_frame_path = os.path.join(cam_dir, '0000.jpg')
        if not os.path.exists(first_frame_path):
            print(f'  First frame not found: {first_frame_path}')
            continue

        image_rgb = np.array(Image.open(first_frame_path))

        points, labels = interactive_mask_selection(image_rgb, image_predictor, cam_label)
        cam_points[cam_idx] = points
        cam_labels[cam_idx] = labels
        print(f'  Selected {len(points)} points')

    # Build SAM2 video predictor for tracking
    print('\nLoading SAM2 video predictor...')
    video_predictor = build_sam2_video_predictor(
        sam2_config,
        sam2_ckpt,
        device=device,
        apply_postprocessing=apply_postprocessing,
        vos_optimized=False,
    )

    for camera in camera_entries:
        cam_idx = camera['index']
        if cam_idx not in cam_points:
            continue

        cam_dir_name = camera['dir']
        cam_label = f'{cam_dir_name} ({camera["name"]})'
        print(f'\n--- Tracking {cam_label} ---')

        cam_dir = os.path.join(rgbd_dir, cam_dir_name)
        jpg_files = sorted(glob.glob(os.path.join(cam_dir, '*.jpg')))
        num_frames = len(jpg_files)

        if num_frames == 0:
            print(f'  No frames found for {cam_label}')
            continue

        print(f'  Tracking {num_frames} frames...')
        masks = track_masks_with_video_predictor(
            video_predictor, cam_dir,
            cam_points[cam_idx], cam_labels[cam_idx],
            num_frames,
        )

        for frame_idx, mask in masks.items():
            mask_path = os.path.join(mask_dir, f'{cam_dir_name}_{frame_idx:04d}_mask.npy')
            np.save(mask_path, mask)

        print(f'  Saved {len(masks)} masks to {mask_dir}/')

    print('\nSAM2 mask generation done!')


# ─── Demo collection (rule) ──────────────────────────────────────────────────

def collect_rule_demo(args):
    """Collect a demonstration via a built-in rule trajectory."""
    if args.object != 'box':
        raise NotImplementedError(
            f'--rule is currently implemented only for --object box, got "{args.object}"'
        )

    object_root, demo_root, demo_dir, rgbd_dir, pose_dir = prepare_output_dirs(
        args.object,
        args.demo_index,
    )

    env = MujocoEnv(args.object)
    env.save_camera_params(object_root)
    env.launch_viewer()

    # Prepare GT mask output if not using SAM2
    mask_dir = None
    geom_ids = None
    if not args.sam2:
        mask_dir = get_demo_mask_dir(args.object, args.demo_index)
        os.makedirs(mask_dir, exist_ok=True)
        geom_ids = resolve_object_geom_ids(env, args.object)

    total_frames = min(args.max_frames, 120)
    if total_frames < 120:
        print(f'Rule demo compressed to {total_frames} frames because --max_frames={args.max_frames}.')

    planned_actions = build_box_rule_trajectory(env, total_frames)
    dt = 1.0 / args.fps
    sim_steps_per_frame = max(1, int(dt / env.model.opt.timestep))

    print(f'Running rule-based box demo for {len(planned_actions)} frames.')

    gripper_states = []
    frame = 0

    try:
        while frame < len(planned_actions) and env.viewer_is_running():
            t_start = time.time()
            action = planned_actions[frame]

            env.set_target(action['arm_pos'], action['arm_quat'], action['gripper_val'])
            env.step(n_substeps=sim_steps_per_frame)
            env.sync_viewer()
            record_frame(env, rgbd_dir, pose_dir, frame, gripper_states,
                         mask_dir=mask_dir, geom_ids=geom_ids)

            frame += 1
            if frame % 20 == 0 or frame == len(planned_actions):
                print(
                    f'Frame {frame}/{len(planned_actions)} '
                    f'({action["phase"]}, gripper={action["gripper_val"]:.1f})'
                )

            elapsed = time.time() - t_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print(f'\nRule demo interrupted at frame {frame}.')

    finalize_demo(demo_dir, rgbd_dir, pose_dir, frame, gripper_states, env)

    # Run SAM2 mask generation after recording if requested
    if args.sam2:
        generate_sam2_masks(args.object, args.demo_index,
                            args.sam2_config, args.sam2_ckpt)


# ─── Demo collection (teleop) ────────────────────────────────────────────────

def collect_teleop_demo(args):
    """Collect a demonstration via WebXR teleoperation."""
    from mujoco_scripts.webxr_control import TeleopPolicy

    object_root, demo_root, demo_dir, rgbd_dir, pose_dir = prepare_output_dirs(
        args.object,
        args.demo_index,
    )

    env = MujocoEnv(args.object)
    env.save_camera_params(object_root)

    # Prepare GT mask output if not using SAM2
    mask_dir = None
    geom_ids = None
    if not args.sam2:
        mask_dir = get_demo_mask_dir(args.object, args.demo_index)
        os.makedirs(mask_dir, exist_ok=True)
        geom_ids = resolve_object_geom_ids(env, args.object)

    # Start WebXR teleop policy
    policy = TeleopPolicy()
    print('Waiting for WebXR connection. Open the server URL on your phone.')
    print('Press "Start Episode" on the phone to arm the session.')
    print('Recording begins only when you press "Start Tracking".')
    policy.reset()

    env.launch_viewer()

    gripper_states = []
    recording_started = False
    frame = 0
    dt = 1.0 / args.fps
    sim_steps_per_frame = max(1, int(dt / env.model.opt.timestep))

    try:
        while frame < args.max_frames and env.viewer_is_running():
            t_start = time.time()

            obs = env.build_teleop_obs()
            action = policy.step(obs)

            if action == 'end_episode':
                if not recording_started:
                    print('Episode ended before tracking started. No frames recorded.')
                else:
                    print(f'Episode ended by user at frame {frame}.')
                break
            elif action == 'reset_env':
                if not recording_started:
                    print('Reset requested before tracking started.')
                else:
                    print('Reset requested.')
                break

            if action is None:
                env.sync_viewer()
            else:
                if not recording_started:
                    recording_started = True
                    print('Live tracking detected. Recording...')

                gripper_val = float(action['gripper_pos'][0]) * 255.0
                env.set_target(action['arm_pos'], action['arm_quat'], gripper_val)

                env.step(n_substeps=sim_steps_per_frame)
                env.sync_viewer()
                record_frame(env, rgbd_dir, pose_dir, frame, gripper_states,
                             mask_dir=mask_dir, geom_ids=geom_ids)

                frame += 1

                if frame % 50 == 0:
                    print(f'Frame {frame}/{args.max_frames}')

            elapsed = time.time() - t_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print(f'\nRecording interrupted at frame {frame}.')

    finalize_demo(demo_dir, rgbd_dir, pose_dir, frame, gripper_states, env)

    # Run SAM2 mask generation after recording if requested
    if args.sam2:
        generate_sam2_masks(args.object, args.demo_index,
                            args.sam2_config, args.sam2_ckpt)


# ─── Main ────────────────────────────────────────────────────────────────────

def collect_demo(args):
    """Dispatch to either teleop or rule-based demo collection."""
    if args.teleop:
        collect_teleop_demo(args)
    else:
        collect_rule_demo(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MuJoCo demo generation for Instant Policy')
    parser.add_argument('--object', type=str, default='mug',
                        help='Object name (loads asset/{object}.xml)')
    parser.add_argument('--demo_index', type=int, default=0,
                        help='Demo index to save under results/{object}/demo/demo_{index}')
    parser.add_argument('--teleop', action='store_true',
                        help='Generate a teleop demo instead of a rule-based demo')
    parser.add_argument('--fps', type=float, default=25.0,
                        help='Recording frame rate')
    parser.add_argument('--max_frames', type=int, default=2000,
                        help='Maximum number of frames to record (rule mode is capped at 120)')

    # Mask generation mode
    parser.add_argument('--sam2', action='store_true',
                        help='Use SAM2 for mask generation (interactive keypoints + video tracking)')
    parser.add_argument('--sam2_config', type=str,
                        default='configs/sam2.1/sam2.1_hiera_s.yaml',
                        help='SAM2 config file (relative to sam2 package)')
    parser.add_argument('--sam2_ckpt', type=str,
                        default='sam2_repo/checkpoints/sam2.1_hiera_small.pt',
                        help='SAM2 checkpoint path')

    args = parser.parse_args()

    if not args.sam2 and args.object not in OBJECT_GEOM_NAMES:
        parser.error(
            f'GT segmentation is not configured for --object {args.object}. '
            f'Available objects: {sorted(OBJECT_GEOM_NAMES.keys())}'
        )

    collect_demo(args)
