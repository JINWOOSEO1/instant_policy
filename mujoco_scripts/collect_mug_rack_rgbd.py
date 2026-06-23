"""
Collect a rule-based mug-rack RGBD sequence from fixed MuJoCo cameras.

The output layout is:

results/{MMDDHHMM}_mug_rack/
  cameras.json
  frames/
    000000/
      rgb_0.png
      depth_0.npy
      rgb_1.png
      depth_1.npy
      rgb_2.png
      depth_2.npy

Depth is saved as float32 meters in the OpenCV camera +Z direction. Values
outside (0, depth_max] are set to 0.
"""

import argparse
import json
import os
import time
from datetime import datetime

import cv2
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R


IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
DEFAULT_CAMERA_NAMES = ('cam_0', 'cam_1', 'cam_2')
MUG_OBJECTS = {'mug_0', 'mug_1', 'mug_2', 'mug_3', 'mug_4', 'mug_3branch'}
RULE_DEMO_FRAME_CAPS = {
    'mug_0': 200,
    'mug_1': 200,
    'mug_2': 200,
    'mug_3': 200,
    'mug_4': 200,
    'mug_3branch': 200,
}
SCENE_NOISE_STD = 0.02
SCENE_NOISE_CLIP = 0.03
MIN_OBJECT_CENTER_DISTANCE = 0.1
SCENE_MAX_RESAMPLES = 100


# Standalone rule trajectory helpers

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


def get_first_existing_geom_position(env, candidate_names):
    """Return the world position of the first geom name that exists in the model."""
    for name in candidate_names:
        geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id != -1:
            return env.data.geom_xpos[geom_id].copy(), name

    raise ValueError(
        f'None of the candidate geoms exist in the model: {candidate_names}'
    )


def build_mug_rule_trajectory(env, total_frames):
    """Build a rim-grasp and rack-hanging plan for mug tasks."""
    if total_frames < 8:
        raise ValueError('Rule-based mug demo needs at least 8 frames')

    ee_pose = env.get_ee_pose()
    home_pos = ee_pose[:3, 3].copy()
    home_quat = R.from_matrix(ee_pose[:3, :3]).as_quat()

    mug_center = env.get_body_position('source_object')
    target_branch, _ = get_first_existing_geom_position(
        env,
        [
            'rack_branch_middle',
            'rack_branch_bottom_2',
            'rack_branch_upper',
            'rack_branch_top_1',
            'rack_branch_lower',
            'rack_branch_bottom_1',
        ],
    )

    grasp_pos = mug_center + np.array([0.01, 0.06, 0.12])
    pregrasp_pos = grasp_pos + np.array([0.0, 0.0, 0.20])

    lift_pos = np.array([
        grasp_pos[0] - 0.02,
        grasp_pos[1],
        max(home_pos[2] - 0.001, grasp_pos[2] + 0.25),
    ])

    hang_pos = target_branch + np.array([-0.09, 0.0, 0.077])
    approach_branch_pos = hang_pos + np.array([0.0, -0.04, 0.05])

    phase_specs = [
        ('approach_rim', home_pos, pregrasp_pos, 255.0, 255.0),
        ('descend_to_rim', pregrasp_pos, grasp_pos, 255.0, 255.0),
        ('close_gripper', grasp_pos, grasp_pos, 255.0, 0.0),
        ('lift_mug', grasp_pos, lift_pos, 0.0, 0.0),
        ('move_to_prehang', lift_pos, approach_branch_pos, 0.0, 0.0),
        ('align_handle_to_branch', approach_branch_pos, hang_pos, 0.0, 0.0),
        ('settle_before_release', hang_pos, hang_pos, 0.0, 0.0),
        ('release_mug', hang_pos, hang_pos, 0.0, 255.0),
        ('hold_open_pose', hang_pos, hang_pos, 255.0, 255.0),
    ]
    phase_frames = allocate_segment_frames(total_frames, [30, 15, 10, 30, 40, 15, 5, 4, 3])

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


# Standalone MuJoCo environment

def scipy_quat_to_mujoco(quat_xyzw):
    """Convert scipy quaternion [x, y, z, w] to MuJoCo [w, x, y, z]."""
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


def mujoco_quat_to_mat(quat_wxyz):
    """Convert MuJoCo quaternion [w, x, y, z] to a 3x3 rotation matrix."""
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat_wxyz)
    return mat.reshape(3, 3)


class MujocoEnv:
    """Small standalone environment for rule-based mug-rack RGBD collection."""

    def __init__(self, object_name, height=128, width=128, cam_names=None,
                 asset_dir='asset'):
        self.object_name = object_name
        self.height = height
        self.width = width
        self.cam_names = list(cam_names or DEFAULT_CAMERA_NAMES)
        self.asset_dir = asset_dir

        scene_xml = os.path.join(asset_dir, f'{object_name}.xml')
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data = mujoco.MjData(self.model)

        self.target_object_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            'target_object',
        )
        self.target_object_base_pos = None
        if self.target_object_body_id != -1:
            self.target_object_base_pos = self.model.body_pos[self.target_object_body_id].copy()

        self.ee_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            'gripper_tcp',
        )
        if self.ee_site_id == -1:
            raise ValueError('Required site "gripper_tcp" not found in model')

        if self.model.njnt < 7:
            raise ValueError('Expected at least 7 arm joints for Panda IK control')
        self.joint_range_low = self.model.jnt_range[:7, 0].copy()
        self.joint_range_high = self.model.jnt_range[:7, 1].copy()

        self.rgb_renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.depth_renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.depth_renderer.enable_depth_rendering()

        self.target_pos = None
        self.target_quat = None
        self.gripper_val = 255.0
        self.viewer = None

        self.reset()

    def reset(self):
        """Reset to the scene_home keyframe if present, then randomize the scene."""
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, 'scene_home')
        if key_id != -1:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        else:
            mujoco.mj_resetData(self.model, self.data)
        self.randomize_scene()

        self.target_pos = self.data.site_xpos[self.ee_site_id].copy()
        ee_mat = self.data.site_xmat[self.ee_site_id].reshape(3, 3)
        self.target_quat = scipy_quat_to_mujoco(R.from_matrix(ee_mat).as_quat())
        self.gripper_val = float(self.data.ctrl[7]) if self.model.nu > 7 else 255.0

    def randomize_scene(self):
        """Apply the same XY randomization used by demo_generation.py."""
        source_joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            'source_object_joint',
        )

        if source_joint_id == -1 or self.target_object_body_id == -1 or self.target_object_base_pos is None:
            mujoco.mj_forward(self.model, self.data)
            return

        source_qpos_adr = self.model.jnt_qposadr[source_joint_id]
        source_base_pos = self.data.qpos[source_qpos_adr:source_qpos_adr + 3].copy()
        self.model.body_pos[self.target_object_body_id] = self.target_object_base_pos
        target_base_xy = self.target_object_base_pos[:2].copy()

        for _ in range(SCENE_MAX_RESAMPLES):
            source_noise = np.clip(
                np.random.normal(loc=0.0, scale=SCENE_NOISE_STD, size=2),
                -SCENE_NOISE_CLIP,
                SCENE_NOISE_CLIP,
            )
            target_noise = np.clip(
                np.random.normal(loc=0.0, scale=SCENE_NOISE_STD, size=2),
                -SCENE_NOISE_CLIP,
                SCENE_NOISE_CLIP,
            )

            source_xy = source_base_pos[:2] + source_noise
            target_xy = target_base_xy + target_noise
            if np.linalg.norm(source_xy - target_xy) >= MIN_OBJECT_CENTER_DISTANCE:
                self.data.qpos[source_qpos_adr:source_qpos_adr + 2] = source_xy
                self.model.body_pos[self.target_object_body_id, :2] = target_xy
                mujoco.mj_forward(self.model, self.data)
                return

        raise RuntimeError(
            'Failed to sample a valid randomized scene with sufficient object separation'
        )

    def get_ee_pose(self):
        """Get the 4x4 SE3 pose of the gripper TCP in world frame."""
        T = np.eye(4)
        T[:3, :3] = self.data.site_xmat[self.ee_site_id].reshape(3, 3)
        T[:3, 3] = self.data.site_xpos[self.ee_site_id]
        return T

    def get_body_position(self, body_name):
        """Get body position in world coordinates."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f'Body "{body_name}" not found in model')
        return self.data.xpos[body_id].copy()

    def set_target(self, target_pos, target_quat_xyzw, gripper_val):
        """Set the desired end-effector pose and gripper command."""
        self.target_pos = target_pos.copy()
        self.target_quat = scipy_quat_to_mujoco(target_quat_xyzw)
        self.gripper_val = np.clip(gripper_val, 0, 255)

    def step(self, n_substeps=1, converge=False, joint_tol=1e-3,
             max_ik_iters=100):
        """Run IK update(s) and physics steps."""
        iters = max_ik_iters if converge else 1
        for i in range(1, iters + 1):
            mujoco.mj_forward(self.model, self.data)
            dq = self._ik_step(self.target_pos, self.target_quat)
            q_target = np.clip(
                self.data.qpos[:7] + dq,
                self.joint_range_low,
                self.joint_range_high,
            )
            self.data.ctrl[:7] = q_target
            if self.model.nu > 7:
                self.data.ctrl[7] = self.gripper_val

            for _ in range(n_substeps):
                self.data.qfrc_applied[:7] = self.data.qfrc_bias[:7]
                mujoco.mj_step(self.model, self.data)

            if converge and np.max(np.abs(dq)) < joint_tol:
                return i
        if converge:
            return max_ik_iters
        return None

    def _ik_step(self, target_pos, target_quat_mj, step_size=0.5, damping=1e-4):
        """Damped least-squares IK for the first seven joints."""
        current_pos = self.data.site_xpos[self.ee_site_id].copy()
        current_mat = self.data.site_xmat[self.ee_site_id].reshape(3, 3)

        err_pos = target_pos - current_pos
        target_mat = mujoco_quat_to_mat(target_quat_mj)
        err_rot_mat = target_mat @ current_mat.T
        err_rot = R.from_matrix(err_rot_mat).as_rotvec()
        err = np.concatenate([err_pos, err_rot])

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
        J = np.vstack([jacp[:, :7], jacr[:, :7]])

        JJT = J @ J.T + damping * np.eye(6)
        dq = J.T @ np.linalg.solve(JJT, err)
        return dq * step_size

    def render_rgb(self, cam_name):
        """Render RGB from a fixed camera."""
        self.rgb_renderer.update_scene(self.data, camera=cam_name)
        return self.rgb_renderer.render()

    def render_depth(self, cam_name):
        """Render metric depth from a fixed camera."""
        self.depth_renderer.update_scene(self.data, camera=cam_name)
        return self.depth_renderer.render()

    def render_rgbd(self, cam_name):
        """Render both RGB and depth from a fixed camera."""
        return self.render_rgb(cam_name), self.render_depth(cam_name)

    def launch_viewer(self, azimuth=180.0, elevation=-25.0, distance=2.0,
                      lookat=None):
        """Launch a passive MuJoCo viewer."""
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.cam.azimuth = azimuth
        self.viewer.cam.elevation = elevation
        self.viewer.cam.distance = distance
        self.viewer.cam.lookat[:] = lookat or [0, 0, 0.3]
        return self.viewer

    def sync_viewer(self):
        """Sync the passive viewer if it is running."""
        if self.viewer is not None:
            self.viewer.sync()

    def viewer_is_running(self):
        """Return whether the passive viewer is still open."""
        return self.viewer is not None and self.viewer.is_running()

    def close(self):
        """Close viewer and renderers."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        self.rgb_renderer.close()
        self.depth_renderer.close()


def list_model_cameras(model):
    """Return fixed camera names from a MuJoCo model."""
    names = []
    for cam_id in range(model.ncam):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_id)
        if name is not None:
            names.append(name)
    return names


def validate_camera_names(model, camera_names):
    """Fail early with a useful message if a requested camera is missing."""
    missing = [
        name
        for name in camera_names
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name) == -1
    ]
    if not missing:
        return

    available = list_model_cameras(model)
    raise ValueError(
        f'Missing camera(s): {missing}. Available cameras in this XML: {available}'
    )


def validate_requested_cameras(object_name, camera_names, asset_dir):
    """Validate camera names before constructing renderers."""
    scene_xml = os.path.join(asset_dir, f'{object_name}.xml')
    model = mujoco.MjModel.from_xml_path(scene_xml)
    validate_camera_names(model, camera_names)


def make_run_dir(output_root, timestamp_format):
    """Create results/{timestamp}_mug_rack, adding a suffix only on collision."""
    timestamp = datetime.now().strftime(timestamp_format)
    base_dir = os.path.join(output_root, f'{timestamp}_mug_rack')
    run_dir = base_dir
    suffix = 1
    while os.path.exists(run_dir):
        run_dir = f'{base_dir}_{suffix:02d}'
        suffix += 1

    os.makedirs(os.path.join(run_dir, 'frames'), exist_ok=False)
    return run_dir


def get_opencv_camera_matrices(env, camera_name):
    """Return OpenCV K and world-to-OpenCV-camera extrinsic for a MuJoCo camera."""
    cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id == -1:
        raise ValueError(f'Camera "{camera_name}" not found')

    fovy_rad = np.deg2rad(env.model.cam_fovy[cam_id])
    fy = (env.height / 2.0) / np.tan(fovy_rad / 2.0)
    fx = fy
    cx = env.width / 2.0
    cy = env.height / 2.0
    intrinsic = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    cam_pos_world = env.data.cam_xpos[cam_id].copy()
    rot_world_from_gl_cam = env.data.cam_xmat[cam_id].reshape(3, 3).copy()

    # MuJoCo fixed cameras use OpenGL convention: +X right, +Y up, -Z forward.
    # Convert to OpenCV convention: +X right, +Y down, +Z forward.
    gl_to_cv = np.diag([1.0, -1.0, -1.0])
    rot_gl_cam_from_world = rot_world_from_gl_cam.T
    rot_cv_cam_from_world = gl_to_cv @ rot_gl_cam_from_world
    trans_cv_cam_from_world = gl_to_cv @ (-rot_gl_cam_from_world @ cam_pos_world)

    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3, :3] = rot_cv_cam_from_world
    extrinsic[:3, 3] = trans_cv_cam_from_world
    return intrinsic, extrinsic


def save_cameras_json(env, camera_names, output_path):
    """Write cameras.json in the requested OpenCV pinhole-camera format."""
    mujoco.mj_forward(env.model, env.data)
    views = []
    for camera_name in camera_names:
        intrinsic, extrinsic = get_opencv_camera_matrices(env, camera_name)
        views.append({
            'camera_name': camera_name,
            'intrinsic': intrinsic.tolist(),
            'extrinsic': extrinsic.tolist(),
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({'views': views}, f, indent=2)


def sanitize_depth(depth, depth_max):
    """Save only finite metric z-depths in the requested range."""
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0) & (depth <= depth_max)
    return np.where(valid, depth, 0.0).astype(np.float32)


def record_rgbd_frame(env, camera_names, frame_dir, depth_max):
    """Render and save one RGBD frame from every requested camera."""
    os.makedirs(frame_dir, exist_ok=False)
    for view_idx, camera_name in enumerate(camera_names):
        rgb, depth = env.render_rgbd(camera_name)
        rgb_path = os.path.join(frame_dir, f'rgb_{view_idx}.png')
        depth_path = os.path.join(frame_dir, f'depth_{view_idx}.npy')

        cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        np.save(depth_path, sanitize_depth(depth, depth_max))


def collect(args):
    """Run the mug rule trajectory and save multi-view RGBD frames."""
    if args.object not in MUG_OBJECTS:
        raise ValueError(
            f'--object must be a mug-rack object, got "{args.object}". '
            f'Available: {sorted(MUG_OBJECTS)}'
        )

    camera_names = tuple(args.cameras)
    if len(camera_names) != 3:
        raise ValueError(f'Exactly 3 cameras are required, got {len(camera_names)}')

    validate_requested_cameras(args.object, camera_names, args.asset_dir)
    env = MujocoEnv(
        args.object,
        height=args.height,
        width=args.width,
        cam_names=camera_names,
        asset_dir=args.asset_dir,
    )

    run_dir = make_run_dir(args.output_root, args.timestamp_format)
    frames_dir = os.path.join(run_dir, 'frames')
    save_cameras_json(env, camera_names, os.path.join(run_dir, 'cameras.json'))

    viewer_launched = False
    if args.viewer:
        env.launch_viewer()
        viewer_launched = True

    rule_frame_cap = RULE_DEMO_FRAME_CAPS[args.object]
    total_frames = min(args.max_frames, rule_frame_cap)
    planned_actions = build_mug_rule_trajectory(env, total_frames)
    dt = 1.0 / args.fps
    sim_steps_per_frame = max(1, int(dt / env.model.opt.timestep))

    print(f'Saving RGBD sequence to {run_dir}')
    print(f'Cameras: {camera_names}')
    print(f'Frames: {len(planned_actions)}, resolution: {args.height}x{args.width}')

    frame = 0
    try:
        while frame < len(planned_actions):
            if viewer_launched and not env.viewer_is_running():
                break

            t_start = time.time()
            action = planned_actions[frame]
            env.set_target(action['arm_pos'], action['arm_quat'], action['gripper_val'])
            env.step(n_substeps=sim_steps_per_frame)
            env.sync_viewer()

            frame_dir = os.path.join(frames_dir, f'{frame:06d}')
            record_rgbd_frame(env, camera_names, frame_dir, args.depth_max)

            frame += 1
            if frame % 20 == 0 or frame == len(planned_actions):
                print(
                    f'Frame {frame}/{len(planned_actions)} '
                    f'({action["phase"]}, gripper={action["gripper_val"]:.1f})'
                )

            if args.realtime:
                sleep_time = dt - (time.time() - t_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        print(f'\nInterrupted at frame {frame}. Partial sequence kept in {run_dir}')
    finally:
        env.close()

    print(f'Done. Saved {frame} frames to {run_dir}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Collect multi-view RGBD images for the rule-based mug-rack task.'
    )
    parser.add_argument('--object', type=str, default='mug_0',
                        help='Object XML stem to load from asset/{object}.xml')
    parser.add_argument('--asset_dir', type=str, default='asset',
                        help='Directory containing {object}.xml')
    parser.add_argument('--cameras', nargs=3, default=DEFAULT_CAMERA_NAMES,
                        help='Three XML camera names, default: cam_0 cam_1 cam_2')
    parser.add_argument('--output_root', type=str, default='results',
                        help='Root directory for {timestamp}_mug_rack output')
    parser.add_argument('--timestamp_format', type=str, default='%m%d%H%M',
                        help='datetime.strftime format used in the output folder name')
    parser.add_argument('--height', type=int, default=IMAGE_HEIGHT,
                        help='RGB/depth image height')
    parser.add_argument('--width', type=int, default=IMAGE_WIDTH,
                        help='RGB/depth image width')
    parser.add_argument('--fps', type=float, default=25.0,
                        help='Rule trajectory sampling rate')
    parser.add_argument('--max_frames', type=int, default=2000,
                        help='Maximum frames to save; mug rule demos are capped at 200')
    parser.add_argument('--depth_max', type=float, default=2.0,
                        help='Depth values above this meter value are saved as 0')
    parser.add_argument('--viewer', action='store_true',
                        help='Launch the MuJoCo passive viewer while recording')
    parser.add_argument('--realtime', action='store_true',
                        help='Sleep to maintain --fps while recording')
    return parser.parse_args()


if __name__ == '__main__':
    collect(parse_args())
