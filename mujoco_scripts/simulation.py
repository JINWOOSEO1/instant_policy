"""
MuJoCo simulation for Instant Policy demo collection.
Records RGBD images, EE poses, and gripper states during WebXR teleoperation.

Usage:
    python simulation.py --object mug
"""

import argparse
import os
import time

import cv2
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R

from mujoco_scripts.webxr_control import TeleopPolicy


# ─── Utility functions ───────────────────────────────────────────────────────

def scipy_quat_to_mujoco(quat_xyzw):
    """Convert scipy quaternion [x, y, z, w] to MuJoCo [w, x, y, z]."""
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


def mujoco_quat_to_mat(quat_wxyz):
    """Convert MuJoCo quaternion [w, x, y, z] to 3x3 rotation matrix."""
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat_wxyz)
    return mat.reshape(3, 3)


def get_viewer_cam_basis(viewer):
    """Compute camera-to-world rotation from the passive viewer's free camera.

    Returns a 3x3 proper rotation matrix (det=+1) whose columns are the
    camera's [forward, left, up] directions in world frame:
      - forward: into the screen (camera look direction)
      - left:    viewer's left
      - up:      viewer's up

    Phone deltas in (forward, left, up) convention are mapped to world frame
    via:  world_delta = cam_basis @ phone_delta
    This works for both position vectors and rotation vectors because the
    matrix is a proper rotation (det=+1).
    """
    az = np.deg2rad(viewer.cam.azimuth)
    el = np.deg2rad(viewer.cam.elevation)

    forward = np.array([
        np.cos(el) * np.cos(az),
        np.cos(el) * np.sin(az),
        np.sin(el),
    ])

    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)

    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    return np.column_stack([forward, -right, up])


def make_transform(rotation_mat, translation=None):
    """Create a 4x4 transform from a rotation matrix and optional translation."""
    T = np.eye(4)
    T[:3, :3] = rotation_mat
    if translation is not None:
        T[:3, 3] = translation
    return T


EE_FRAME_CALIB_PRESETS = {
    'identity': np.eye(4),
    'rx90': make_transform(R.from_euler('x', 90, degrees=True).as_matrix()),
    'rx-90': make_transform(R.from_euler('x', -90, degrees=True).as_matrix()),
    'rx180': make_transform(R.from_euler('x', 180, degrees=True).as_matrix()),
    'ry90': make_transform(R.from_euler('y', 90, degrees=True).as_matrix()),
    'ry-90': make_transform(R.from_euler('y', -90, degrees=True).as_matrix()),
    'ry180': make_transform(R.from_euler('y', 180, degrees=True).as_matrix()),
    'rz90': make_transform(R.from_euler('z', 90, degrees=True).as_matrix()),
    'rz-90': make_transform(R.from_euler('z', -90, degrees=True).as_matrix()),
    'rz180': make_transform(R.from_euler('z', 180, degrees=True).as_matrix()),
    'rx180_rz90': make_transform(R.from_euler('xz', [180, 90], degrees=True).as_matrix()),
    'rx180_rz-90': make_transform(R.from_euler('xz', [180, -90], degrees=True).as_matrix()),
    'ry180_rz90': make_transform(R.from_euler('yz', [180, 90], degrees=True).as_matrix()),
    'ry180_rz-90': make_transform(R.from_euler('yz', [180, -90], degrees=True).as_matrix()),
}


def parse_ee_frame_calib(spec):
    """Parse EE-frame calibration preset or `euler_deg:rx,ry,rz` string."""
    if spec in EE_FRAME_CALIB_PRESETS:
        return EE_FRAME_CALIB_PRESETS[spec].copy()

    if spec.startswith('euler_deg:'):
        angles = [float(v) for v in spec.split(':', 1)[1].split(',')]
        if len(angles) != 3:
            raise ValueError(
                f'Invalid EE frame calib `{spec}`. Expected euler_deg:rx,ry,rz.'
            )
        return make_transform(R.from_euler('xyz', angles, degrees=True).as_matrix())

    presets = ', '.join(sorted(EE_FRAME_CALIB_PRESETS))
    raise ValueError(
        f'Unknown EE frame calib `{spec}`. Use one of: {presets}, '
        'or euler_deg:rx,ry,rz'
    )


# ─── MujocoEnv ───────────────────────────────────────────────────────────────

class MujocoEnv:
    """MuJoCo environment for Panda robot manipulation tasks.

    Provides model loading, IK-based control, RGBD rendering, and
    observation/action interfaces. Independent of any teleop or policy logic.
    """

    def __init__(self, object_name, height=480, width=640, cam_names=None,
                 ee_frame_calib='identity'):
        self.object_name = object_name
        self.height = height
        self.width = width
        self.cam_names = cam_names or ['cam_left', 'cam_right', 'cam_front']
        self.ee_frame_calib_spec = ee_frame_calib
        self.T_ee_frame_calib = parse_ee_frame_calib(ee_frame_calib)
        self.T_ee_frame_calib_inv = np.linalg.inv(self.T_ee_frame_calib)

        # Load model
        scene_xml = f'asset/{object_name}.xml'
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data = mujoco.MjData(self.model)

        # End-effector references
        self.hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'hand')
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'gripper_tcp')

        # Joint limits for clamping IK solutions
        self.joint_range_low = self.model.jnt_range[:7, 0].copy()
        self.joint_range_high = self.model.jnt_range[:7, 1].copy()

        # Create renderers
        self.rgb_renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.depth_renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.depth_renderer.enable_depth_rendering()

        # IK targets (initialized in reset)
        self.target_pos = None
        self.target_quat = None

        # Viewer reference (set externally when using passive viewer)
        self.viewer = None

        # Reset to initial state
        self.reset()

    def _get_raw_ee_pose(self):
        """Get the physical TCP site pose in world frame before calibration."""
        T = np.eye(4)
        T[:3, :3] = self.data.site_xmat[self.ee_site_id].reshape(3, 3)
        T[:3, 3] = self.data.site_xpos[self.ee_site_id]
        return T

    def reset(self):
        """Reset environment to the scene_home keyframe."""
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, 'scene_home')
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        mujoco.mj_forward(self.model, self.data)

        # Initialize IK target to current TCP pose
        raw_ee_pose = self._get_raw_ee_pose()
        self.target_pos = raw_ee_pose[:3, 3].copy()
        self.target_quat = scipy_quat_to_mujoco(
            R.from_matrix(raw_ee_pose[:3, :3]).as_quat()
        )

    # ── Observation ──────────────────────────────────────────────────────

    def get_ee_pose(self):
        """Get the calibrated gripper pose in world frame."""
        return self._get_raw_ee_pose() @ self.T_ee_frame_calib

    def get_gripper_state(self):
        """Get gripper state: 1=open, 0=closed. Based on finger joint position."""
        finger_pos = self.data.qpos[7]  # finger_joint1
        return 1 if finger_pos > 0.02 else 0

    def get_camera_params(self, cam_name):
        """Get camera intrinsics and extrinsics."""
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)

        # Intrinsics from fovy
        fovy_rad = np.deg2rad(self.model.cam_fovy[cam_id])
        fy = (self.height / 2.0) / np.tan(fovy_rad / 2.0)
        fx = fy
        cx = self.width / 2.0
        cy = self.height / 2.0
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        # Extrinsics: world-to-camera transform
        cam_pos = self.data.cam_xpos[cam_id].copy()
        cam_mat = self.data.cam_xmat[cam_id].reshape(3, 3).copy()
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = cam_mat
        extrinsic[:3, 3] = cam_pos

        return intrinsic, extrinsic

    def render_rgb(self, cam_name):
        """Render RGB image from the specified camera."""
        self.rgb_renderer.update_scene(self.data, camera=cam_name)
        return self.rgb_renderer.render()

    def render_depth(self, cam_name):
        """Render depth image from the specified camera."""
        self.depth_renderer.update_scene(self.data, camera=cam_name)
        return self.depth_renderer.render()

    def render_rgbd(self, cam_name):
        """Render both RGB and depth from the specified camera."""
        return self.render_rgb(cam_name), self.render_depth(cam_name)

    def build_teleop_obs(self):
        """Build observation dict for TeleopPolicy.step()."""
        T_w_e = self.get_ee_pose()
        pos = T_w_e[:3, 3].copy()
        mat = T_w_e[:3, :3]
        quat_xyzw = R.from_matrix(mat).as_quat()  # scipy: [x, y, z, w]
        gripper_pos = np.array([self.data.qpos[7] / 0.04])

        obs = {
            'base_pose': np.zeros(3),
            'arm_pos': pos,
            'arm_quat': quat_xyzw,
            'gripper_pos': gripper_pos,
        }

        if self.viewer is not None:
            obs['cam_basis'] = get_viewer_cam_basis(self.viewer)

        return obs

    # ── Action ───────────────────────────────────────────────────────────

    def set_target(self, target_pos, target_quat_xyzw, gripper_val):
        """Set IK target and gripper command.

        Args:
            target_pos: desired calibrated EE position (3,)
            target_quat_xyzw: desired calibrated EE orientation as scipy quaternion [x,y,z,w]
            gripper_val: gripper command in [0, 255]
        """
        T_w_e = make_transform(
            R.from_quat(target_quat_xyzw).as_matrix(),
            target_pos,
        )
        T_w_raw = T_w_e @ self.T_ee_frame_calib_inv
        self.target_pos = T_w_raw[:3, 3].copy()
        self.target_quat = scipy_quat_to_mujoco(
            R.from_matrix(T_w_raw[:3, :3]).as_quat()
        )
        self.data.ctrl[7] = np.clip(gripper_val, 0, 255)

    def step(self, n_substeps=1):
        """Run IK + physics simulation for n_substeps.

        Uses the current target_pos and target_quat to compute joint targets.
        """
        for _ in range(n_substeps):
            mujoco.mj_forward(self.model, self.data)
            dq = self._ik_step(self.target_pos, self.target_quat)
            q_target = np.clip(
                self.data.qpos[:7] + dq,
                self.joint_range_low,
                self.joint_range_high,
            )
            self.data.ctrl[:7] = q_target
            mujoco.mj_step(self.model, self.data)

    def _ik_step(self, target_pos, target_quat_mj, step_size=0.5, damping=1e-4):
        """Damped least-squares IK: compute joint angle delta for one step."""
        current_pos = self.data.site_xpos[self.ee_site_id]
        current_mat = self.data.site_xmat[self.ee_site_id].reshape(3, 3)

        # Position error
        err_pos = target_pos - current_pos

        # Orientation error as rotation vector
        target_mat = mujoco_quat_to_mat(target_quat_mj)
        err_rot_mat = target_mat @ current_mat.T
        err_rot = R.from_matrix(err_rot_mat).as_rotvec()

        # 6D error
        err = np.concatenate([err_pos, err_rot])

        # Jacobian (3×nv for position, 3×nv for rotation)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
        J = np.vstack([jacp[:, :7], jacr[:, :7]])

        # Damped least squares
        JJT = J @ J.T + damping * np.eye(6)
        dq = J.T @ np.linalg.solve(JJT, err)

        return dq * step_size

    # ── Viewer ───────────────────────────────────────────────────────────

    def launch_viewer(self, azimuth=180.0, elevation=-25.0, distance=2.0, lookat=None):
        """Launch a passive MuJoCo viewer and store reference."""
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.cam.azimuth = azimuth
        self.viewer.cam.elevation = elevation
        self.viewer.cam.distance = distance
        self.viewer.cam.lookat[:] = lookat or [0, 0, 0.3]
        return self.viewer

    def sync_viewer(self):
        """Sync the passive viewer (call after stepping)."""
        if self.viewer is not None:
            self.viewer.sync()

    def viewer_is_running(self):
        """Check if the viewer window is still open."""
        return self.viewer is not None and self.viewer.is_running()

    def close(self):
        """Close the viewer."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    # ── Camera params I/O ────────────────────────────────────────────────

    def save_camera_params(self, output_dir):
        """Save intrinsics and extrinsics for all cameras to an npz file."""
        mujoco.mj_forward(self.model, self.data)
        cam_params = {}
        for i, cam_name in enumerate(self.cam_names):
            intrinsic, extrinsic = self.get_camera_params(cam_name)
            cam_params[f'cam{i}_intrinsic'] = intrinsic
            cam_params[f'cam{i}_extrinsic'] = extrinsic
        path = os.path.join(output_dir, 'camera_params.npz')
        np.savez(path, **cam_params)
        print(f'Camera parameters saved to {path}')


# ─── Demo collection (teleop) ────────────────────────────────────────────────

def collect_demo(args):
    """Collect a demonstration via WebXR teleoperation."""
    output_dir = f'results/{args.object}'
    rgbd_dir = os.path.join(output_dir, 'RGBD_images')
    ee_dir = os.path.join(output_dir, 'EE_pose')
    os.makedirs(rgbd_dir, exist_ok=True)
    os.makedirs(ee_dir, exist_ok=True)

    # Create environment
    env = MujocoEnv(args.object, ee_frame_calib=args.ee_frame_calib)
    print(f'Using EE frame calibration: {args.ee_frame_calib}')
    env.save_camera_params(output_dir)

    # Start WebXR teleop policy
    policy = TeleopPolicy()
    print('Waiting for WebXR connection. Open the server URL on your phone.')
    print('Press "Start Episode" on the phone to arm the session.')
    print('Recording begins only when you press "Start Tracking".')
    policy.reset()

    # Launch viewer
    env.launch_viewer()

    # Recording loop
    gripper_states = []
    recording_started = False
    frame = 0
    dt = 1.0 / args.fps
    sim_steps_per_frame = max(1, int(dt / env.model.opt.timestep))

    try:
        while frame < args.max_frames and env.viewer_is_running():
            t_start = time.time()

            # Get teleop action
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

                # Step simulation only while recording is active
                env.step(n_substeps=sim_steps_per_frame)
                env.sync_viewer()

                # Render and save images from both cameras
                rgb_imgs = []
                for i, cam_name in enumerate(env.cam_names):
                    rgb, depth = env.render_rgbd(cam_name)
                    cam_dir = os.path.join(rgbd_dir, f'cam{i}')
                    os.makedirs(cam_dir, exist_ok=True)
                    rgb_path = os.path.join(cam_dir, f'{frame:04d}.jpg')
                    cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                    depth_path = os.path.join(cam_dir, f'{frame:04d}_depth.npy')
                    np.save(depth_path, depth)
                    rgb_imgs.append(rgb)

                # Show camera views in an OpenCV window (side by side)
                tiled = np.hstack([cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in rgb_imgs])
                cv2.imshow('Camera Views', tiled)
                cv2.waitKey(1)

                # Save EE pose
                T_w_e = env.get_ee_pose()
                np.save(os.path.join(ee_dir, f'{frame:04d}.npy'), T_w_e)

                # Record gripper state
                grip = env.get_gripper_state()
                gripper_states.append(grip)

                frame += 1

                if frame % 50 == 0:
                    print(f'Frame {frame}/{args.max_frames}')

            # Maintain target FPS
            elapsed = time.time() - t_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print(f'\nRecording interrupted at frame {frame}.')

    # Save gripper states
    np.save(os.path.join(output_dir, 'gripper_state.npy'), np.array(gripper_states, dtype=np.int32))
    print(f'Saved {frame} frames to {output_dir}/')
    print(f'  RGBD images: {rgbd_dir}/')
    print(f'  EE poses: {ee_dir}/')
    print(f'  Gripper states: {output_dir}/gripper_state.npy')

    cv2.destroyAllWindows()
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MuJoCo demo collection for Instant Policy')
    parser.add_argument('--object', type=str, default='mug', help='Object name (loads asset/{object}.xml)')
    parser.add_argument('--fps', type=float, default=25.0, help='Recording frame rate')
    parser.add_argument('--max_frames', type=int, default=2000, help='Maximum number of frames to record')
    parser.add_argument(
        '--ee_frame_calib',
        type=str,
        default='identity',
        help='EE frame calibration preset or euler_deg:rx,ry,rz '
             f'(presets: {", ".join(sorted(EE_FRAME_CALIB_PRESETS))})',
    )
    args = parser.parse_args()
    collect_demo(args)
