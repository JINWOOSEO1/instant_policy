"""
Demo generation for Instant Policy MuJoCo environments.
Supports rule-based and teleoperation demo collection,
with optional SAM2 or MuJoCo GT segmentation mask generation.

Usage:
    # Rule-based demo with GT masks (MuJoCo seg_renderer)
    python -m mujoco_scripts.demo_generation --object box

    # Rule-based demo with SAM2 masks
    python -m mujoco_scripts.demo_generation --object box --sam2

    # Rule-based mug-hanging demo with GT masks
    python -m mujoco_scripts.demo_generation --object mug_0

    # Teleop demo with GT masks
    python -m mujoco_scripts.demo_generation --object mug_0 --teleop

    # Teleop demo with SAM2 masks
    python -m mujoco_scripts.demo_generation --object mug_0 --sam2 --teleop
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
from mujoco_scripts.rule_trajectories import (
    build_box_rule_trajectory,
    build_mug_rule_trajectory,
)
from mujoco_scripts.simulation import MujocoEnv


# ─── Object segmentation config ──────────────────────────────────────────────

MUG_OBJECTS = {'mug_0', 'mug_1', 'mug_2', 'mug_3', 'mug_4', 'mug_3branch'}

OBJECT_GEOM_NAMES = {
    'mug_0': [
        # mug body
        'mug_bottom',
        'mug_wall_00', 'mug_wall_01', 'mug_wall_02', 'mug_wall_03',
        'mug_wall_04', 'mug_wall_05', 'mug_wall_06', 'mug_wall_07',
        'mug_wall_08', 'mug_wall_09', 'mug_wall_10', 'mug_wall_11',
        'mug_handle_top', 'mug_handle_mid', 'mug_handle_bot', 'mug_handle_low',
        # mug_rack body
        'rack_base', 'rack_post',
        'rack_branch_top_1', 'rack_branch_top_2',
        'rack_branch_bottom_1', 'rack_branch_bottom_2',
    ],
    'mug_1': [
        # mug body
        'mug_bottom',
        'mug_wall_00', 'mug_wall_01', 'mug_wall_02', 'mug_wall_03',
        'mug_wall_04', 'mug_wall_05', 'mug_wall_06', 'mug_wall_07',
        'mug_handle_top', 'mug_handle_upper', 'mug_handle_mid',
        'mug_handle_lower', 'mug_handle_bot',
        # mug_rack body
        'rack_base', 'rack_post',
        'rack_branch_top_1', 'rack_branch_top_2',
        'rack_branch_bottom_1', 'rack_branch_bottom_2',
    ],
    'mug_2': [
        # mug body
        'mug_bottom',
        'mug_wall_00', 'mug_wall_01', 'mug_wall_02', 'mug_wall_03',
        'mug_wall_04', 'mug_wall_05', 'mug_wall_06', 'mug_wall_07',
        'mug_wall_08', 'mug_wall_09', 'mug_wall_10', 'mug_wall_11',
        'mug_wall_taper_00', 'mug_wall_taper_01', 'mug_wall_taper_02', 'mug_wall_taper_03',
        'mug_wall_taper_04', 'mug_wall_taper_05', 'mug_wall_taper_06', 'mug_wall_taper_07',
        'mug_wall_taper_08', 'mug_wall_taper_09', 'mug_wall_taper_10', 'mug_wall_taper_11',
        'mug_handle_top', 'mug_handle_mid', 'mug_handle_bot', 'mug_handle_low',
        # mug_rack body
        'rack_base', 'rack_post',
        'rack_branch_top_1', 'rack_branch_top_2',
        'rack_branch_bottom_1', 'rack_branch_bottom_2',
    ],
    'mug_3': [
        # mug body
        'mug_bottom',
        'mug_wall_00', 'mug_wall_01', 'mug_wall_02', 'mug_wall_03',
        'mug_handle_top', 'mug_handle_mid', 'mug_handle_bot', 'mug_handle_low',
        # mug_rack body
        'rack_base', 'rack_post',
        'rack_branch_top_1', 'rack_branch_top_2',
        'rack_branch_bottom_1', 'rack_branch_bottom_2',
    ],
    'mug_4': [
        # mug body
        'mug_bottom',
        'mug_wall_00', 'mug_wall_01', 'mug_wall_02', 'mug_wall_03',
        'mug_handle_top', 'mug_handle_mid', 'mug_handle_bot', 'mug_handle_low',
        # mug_rack body
        'rack_base', 'rack_post',
        'rack_branch_top_1', 'rack_branch_top_2',
        'rack_branch_bottom_1', 'rack_branch_bottom_2',
    ],
    'mug_3branch': [
        # mug body
        'mug_bottom',
        'mug_wall_00', 'mug_wall_01', 'mug_wall_02', 'mug_wall_03',
        'mug_wall_04', 'mug_wall_05', 'mug_wall_06', 'mug_wall_07',
        'mug_wall_08', 'mug_wall_09', 'mug_wall_10', 'mug_wall_11',
        'mug_handle_top', 'mug_handle_mid', 'mug_handle_bot', 'mug_handle_low',
        # mug_rack body
        'rack_base', 'rack_post',
        'rack_branch_lower', 'rack_branch_middle', 'rack_branch_upper',
    ],
    'box': [
        # small_box body
        'small_box_geom',
        # target_box body
        'target_box_bottom',
        'target_box_wall_pos_x', 'target_box_wall_neg_x',
        'target_box_wall_pos_y', 'target_box_wall_neg_y',
    ],
}

TELEOP_PREVIEW_WINDOW = 'Teleop Preview'
DEFAULT_PREVIEW_CAMERA = 'cam_preview'
DEFAULT_PREVIEW_WIDTH = 640
DEFAULT_PREVIEW_HEIGHT = 480
DEFAULT_PREVIEW_DISPLAY_SCALE = 1.5
RULE_DEMO_FRAME_CAPS = {
    'box': 120,
    'mug_0': 200,
    'mug_1': 200,
    'mug_2': 200,
    'mug_3': 200,
    'mug_4': 200,
    'mug_3branch': 200,
}


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

        # Save GT segmentation mask when not using SAM2
        if mask_dir is not None and geom_ids is not None:
            mask = env.render_seg_mask(cam_name, geom_ids)
            mask_path = os.path.join(mask_dir, f'cam{i}_{frame_idx:04d}_mask.npy')
            np.save(mask_path, mask)

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


def show_teleop_preview(env, preview_renderer, cam_name):
    """Render and display a dedicated high-resolution teleop preview camera."""
    preview_renderer.update_scene(env.data, camera=cam_name)
    preview_rgb = preview_renderer.render()
    preview_bgr = cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR)
    preview_bgr = cv2.resize(
        preview_bgr,
        None,
        fx=DEFAULT_PREVIEW_DISPLAY_SCALE,
        fy=DEFAULT_PREVIEW_DISPLAY_SCALE,
        interpolation=cv2.INTER_LINEAR,
    )
    show_fixed_window(
        TELEOP_PREVIEW_WINDOW,
        preview_bgr,
    )
    cv2.waitKey(1)


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

        print("  Left-click: foreground point | Right-click: background point")
        print("  Press 't' to confirm | Press 'r' to reset points")

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
        print("  Press 't' to accept | Press 'r' to retry with new points")

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
    if args.object in ('box'):
        build_rule_trajectory = build_box_rule_trajectory
    elif args.object in MUG_OBJECTS:
        build_rule_trajectory = build_mug_rule_trajectory
    else:
        raise NotImplementedError(
            '--rule is currently implemented only for '
            f'--object box and mug-family objects, got "{args.object}"'
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

    rule_frame_cap = RULE_DEMO_FRAME_CAPS[args.object]
    total_frames = min(args.max_frames, rule_frame_cap)
    if total_frames < rule_frame_cap:
        print(
            f'Rule demo compressed to {total_frames} frames '
            f'because --max_frames={args.max_frames}.'
        )

    planned_actions = build_rule_trajectory(env, total_frames)
    dt = 1.0 / args.fps
    sim_steps_per_frame = max(1, int(dt / env.model.opt.timestep))

    print(f'Running rule-based {args.object} demo for {len(planned_actions)} frames.')

    gripper_states = []
    frame = 0

    try:
        while frame < len(planned_actions) and env.viewer_is_running():
            t_start = time.time()
            action = planned_actions[frame]

            env.set_target(action['arm_pos'], action['arm_quat'], action['gripper_val'])
            env.step(n_substeps=sim_steps_per_frame)
            env.sync_viewer()
            record_frame(
                env,
                rgbd_dir,
                pose_dir,
                frame,
                gripper_states,
                mask_dir=mask_dir,
                geom_ids=geom_ids,
            )

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
    preview_cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, args.preview_camera)
    if preview_cam_id == -1:
        raise ValueError(f'Preview camera "{args.preview_camera}" not found in asset/{args.object}.xml')
    offwidth = env.model.vis.global_.offwidth
    offheight = env.model.vis.global_.offheight
    if args.preview_width > offwidth or args.preview_height > offheight:
        raise ValueError(
            'Preview size '
            f'{args.preview_width}x{args.preview_height} exceeds offscreen framebuffer '
            f'{offwidth}x{offheight}. Reduce --preview_width/--preview_height or increase '
            'the XML <visual><global offwidth="..." offheight="..."/></visual> setting.'
        )
    preview_renderer = mujoco.Renderer(
        env.model,
        height=args.preview_height,
        width=args.preview_width,
    )

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
                show_teleop_preview(env, preview_renderer, args.preview_camera)
            else:
                if not recording_started:
                    recording_started = True
                    print('Live tracking detected. Recording...')

                gripper_val = float(action['gripper_pos'][0]) * 255.0
                env.set_target(action['arm_pos'], action['arm_quat'], gripper_val)

                env.step(n_substeps=sim_steps_per_frame)
                env.sync_viewer()
                show_teleop_preview(env, preview_renderer, args.preview_camera)
                record_frame(
                    env,
                    rgbd_dir,
                    pose_dir,
                    frame,
                    gripper_states,
                    mask_dir=mask_dir,
                    geom_ids=geom_ids,
                )

                frame += 1

                if frame % 50 == 0:
                    print(f'Frame {frame}/{args.max_frames}')

            elapsed = time.time() - t_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print(f'\nRecording interrupted at frame {frame}.')

    preview_renderer.close()
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
    parser.add_argument('--object', type=str, default='mug_0',
                        help='Object name (loads asset/{object}.xml)')
    parser.add_argument('--demo_index', type=int, default=0,
                        help='Demo index to save under results/{object}/demo/demo_{index}')
    parser.add_argument('--teleop', action='store_true',
                        help='Generate a teleop demo instead of a rule-based demo')
    parser.add_argument('--fps', type=float, default=25.0,
                        help='Recording frame rate')
    parser.add_argument('--max_frames', type=int, default=2000,
                        help='Maximum number of frames to record '
                             '(rule mode: box capped at 120, mug-family objects capped at 200)')
    parser.add_argument('--preview_camera', type=str, default=DEFAULT_PREVIEW_CAMERA,
                        help='Camera name used for the teleop preview window')
    parser.add_argument('--preview_width', type=int, default=DEFAULT_PREVIEW_WIDTH,
                        help='Width of the teleop preview render in pixels')
    parser.add_argument('--preview_height', type=int, default=DEFAULT_PREVIEW_HEIGHT,
                        help='Height of the teleop preview render in pixels')

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

    if args.preview_width <= 0 or args.preview_height <= 0:
        parser.error('--preview_width and --preview_height must be positive')

    if not args.sam2 and args.object not in OBJECT_GEOM_NAMES:
        parser.error(
            f'GT segmentation is not configured for --object {args.object}. '
            f'Available objects: {sorted(OBJECT_GEOM_NAMES.keys())}'
        )

    collect_demo(args)
