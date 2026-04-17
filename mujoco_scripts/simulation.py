"""
MuJoCo simulation for Instant Policy demo collection.
Records RGBD images, EE poses, and gripper states during WebXR teleoperation.

Usage:
    python simulation.py --object mug
    python simulation.py --object box --rule
"""

import argparse
import os
import time

import cv2
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R

from mujoco_scripts.camera_utils import DEFAULT_CAMERA_NAMES
from mujoco_scripts.gen_seg_pcd import camera_pcd_to_world, depth_to_pointcloud
from mujoco_scripts.result_paths import (
    get_demo_dir,
    get_demo_pose_dir,
    get_demo_rgbd_dir,
    get_demo_root,
    get_object_root,
)
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


# ─── MujocoEnv ───────────────────────────────────────────────────────────────

class MujocoEnv:
    """MuJoCo environment for Panda robot manipulation tasks.

    Provides model loading, IK-based control, RGBD rendering, and
    observation/action interfaces. Independent of any teleop or policy logic.
    """

    def __init__(self, object_name, height=128, width=128, cam_names=None):
        self.object_name = object_name
        self.height = height
        self.width = width
        self.cam_names = list(cam_names or DEFAULT_CAMERA_NAMES)

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
        self.seg_renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.seg_renderer.enable_segmentation_rendering()

        # IK targets (initialized in reset)
        self.target_pos = None
        self.target_quat = None

        # Viewer reference (set externally when using passive viewer)
        self.viewer = None

        # Reset to initial state
        self.reset()

    def reset(self):
        """Reset environment to the scene_home keyframe."""
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, 'scene_home')
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        mujoco.mj_forward(self.model, self.data)

        # Initialize IK target to current TCP pose
        self.target_pos = self.data.site_xpos[self.ee_site_id].copy()
        ee_mat = self.data.site_xmat[self.ee_site_id].reshape(3, 3)
        self.target_quat = scipy_quat_to_mujoco(R.from_matrix(ee_mat).as_quat())
        self.gripper_val = float(self.data.ctrl[7])


    # ── Observation ──────────────────────────────────────────────────────

    def get_ee_pose(self):
        """Get the 4x4 SE3 pose of the gripper TCP in world frame."""
        T = np.eye(4)
        T[:3, :3] = self.data.site_xmat[self.ee_site_id].reshape(3, 3)
        T[:3, 3] = self.data.site_xpos[self.ee_site_id]
        return T

    def get_gripper_state(self):
        """Get gripper state: 1=open, 0=closed. Based on finger joint position."""
        finger_pos = self.data.qpos[7]  # finger_joint1
        return 1 if finger_pos > 0.02 else 0

    def get_body_position(self, body_name):
        """Get body position in world coordinates."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f'Body "{body_name}" not found in model')
        return self.data.xpos[body_id].copy()

    def get_geom_position(self, geom_name):
        """Get geom position in world coordinates."""
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id == -1:
            raise ValueError(f'Geom "{geom_name}" not found in model')
        return self.data.geom_xpos[geom_id].copy()

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

    def get_object_geom_ids(self, body_name):
        """Return the set of geom IDs belonging to *body_name* and all its descendants."""
        root_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        body_ids = {root_id}
        # Collect child bodies (MuJoCo stores a flat parent array)
        for bid in range(self.model.nbody):
            parent = bid
            while parent != 0:
                parent = self.model.body_parentid[parent]
                if parent == root_id:
                    body_ids.add(bid)
                    break
        geom_ids = set()
        for gid in range(self.model.ngeom):
            if self.model.geom_bodyid[gid] in body_ids:
                geom_ids.add(gid)
        return geom_ids

    def get_geom_ids_by_names(self, geom_names):
        """Return the set of geom IDs corresponding to the given geom name list."""
        geom_ids = set()
        for name in geom_names:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid == -1:
                raise ValueError(f'Geom "{name}" not found in model')
            geom_ids.add(gid)
        return geom_ids

    def render_seg_mask(self, cam_name, geom_ids):
        """Render a boolean segmentation mask for the given geom IDs.

        Args:
            cam_name: camera name string.
            geom_ids: set/list of MuJoCo geom IDs to include.

        Returns:
            (H, W) bool ndarray — True where the pixel belongs to one of *geom_ids*.
        """
        self.seg_renderer.update_scene(self.data, camera=cam_name)
        seg = self.seg_renderer.render()        # (H, W, 2): [geom_id, geom_type]
        seg_geom = seg[:, :, 0]
        mask = np.isin(seg_geom, list(geom_ids))
        return mask

    def masked_depth_to_world_pcd(self, depth, mask, intrinsic, extrinsic):
        """Convert one masked depth image into a world-frame pointcloud."""
        if mask is None:
            return None

        depth_masked = depth * mask.astype(np.float32)
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        pcd_cam = depth_to_pointcloud(depth_masked, fx, fy, cx, cy)
        if len(pcd_cam) == 0:
            return None
        return camera_pcd_to_world(pcd_cam, extrinsic)

    def get_segmented_pcd(self, mask_fn, cam_names=None, cam_params=None, return_stats=False):
        """Render RGBD views, apply per-camera masks, and merge world-frame PCDs.

        Args:
            mask_fn: callable(cam_name, rgb, depth) -> mask or None.
            cam_names: optional iterable of camera names.
            cam_params: optional dict {cam_name: (intrinsic, extrinsic)}.
            return_stats: when True, also return timing stats.

        Returns:
            Concatenated world-frame pointcloud, or None if every camera is empty.
            If return_stats=True, returns (pcd_w, stats).
        """
        cam_names = list(cam_names or self.cam_names)
        if cam_params is None:
            cam_params = {cam: self.get_camera_params(cam) for cam in cam_names}

        pcd_list = []
        stats = {
            'render': 0.0,
            'mask': 0.0,
            'pcd': 0.0,
        }

        for cam_name in cam_names:
            t0 = time.time()
            rgb, depth = self.render_rgbd(cam_name)
            stats['render'] += time.time() - t0

            t0 = time.time()
            mask = mask_fn(cam_name, rgb, depth)
            stats['mask'] += time.time() - t0
            if mask is None or mask.sum() == 0:
                continue

            intrinsic, extrinsic = cam_params[cam_name]
            t0 = time.time()
            pcd_world = self.masked_depth_to_world_pcd(depth, mask, intrinsic, extrinsic)
            stats['pcd'] += time.time() - t0
            if pcd_world is None:
                continue

            pcd_list.append(pcd_world)

        pcd_world = np.concatenate(pcd_list, axis=0) if pcd_list else None
        if return_stats:
            return pcd_world, stats
        return pcd_world

    def get_segmented_pcd_from_geom_ids(self, geom_ids, cam_names=None, cam_params=None, return_stats=False):
        """Convenience wrapper around get_segmented_pcd() for MuJoCo GT masks."""
        return self.get_segmented_pcd(
            lambda cam_name, _rgb, _depth: self.render_seg_mask(cam_name, geom_ids),
            cam_names=cam_names,
            cam_params=cam_params,
            return_stats=return_stats,
        )

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
            target_pos: desired EE position (3,)
            target_quat_xyzw: desired EE orientation as scipy quaternion [x,y,z,w]
            gripper_val: gripper command in [0, 255]
        """
        self.target_pos = target_pos.copy()
        self.target_quat = scipy_quat_to_mujoco(target_quat_xyzw)
        self.gripper_val = np.clip(gripper_val, 0, 255)

    def step_until_converged(self, joint_tol=1e-3, n_substeps=50,
                             max_ik_iters=100):
        """Run closed-loop IK until the EE reaches the target.

        Each IK iteration computes a joint delta, sets the PD target,
        then runs *n_substeps* physics steps to let the PD controller
        settle before re-evaluating.

        Returns the number of IK iterations executed.
        """
        for i in range(1, max_ik_iters + 1):
            mujoco.mj_forward(self.model, self.data)
            dq = self._ik_step(self.target_pos, self.target_quat)
            q_target = np.clip(
                self.data.qpos[:7] + dq,
                self.joint_range_low,
                self.joint_range_high,
            )
            self.data.ctrl[:7] = q_target
            self.data.ctrl[7] = self.gripper_val

            for _ in range(n_substeps):
                self.data.qfrc_applied[:7] = self.data.qfrc_bias[:7]
                mujoco.mj_step(self.model, self.data)

            if np.max(np.abs(dq)) < joint_tol:
                return i
        return max_ik_iters

    def step(self, n_substeps=1):
        """Run one IK update + n_substeps physics steps (no convergence check)."""
        mujoco.mj_forward(self.model, self.data)
        dq = self._ik_step(self.target_pos, self.target_quat)
        q_target = np.clip(
            self.data.qpos[:7] + dq,
            self.joint_range_low,
            self.joint_range_high,
        )
        self.data.ctrl[:7] = q_target
        self.data.ctrl[7] = self.gripper_val
        for _ in range(n_substeps):
            self.data.qfrc_applied[:7] = self.data.qfrc_bias[:7]
            mujoco.mj_step(self.model, self.data)

    def _ik_step(self, target_pos, target_quat_mj, step_size=0.5, damping=1e-4):
        """Damped least-squares IK: compute joint angle delta for one step."""
        current_pos = self.data.site_xpos[self.ee_site_id].copy()
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
        cam_params = {
            'camera_names': np.array(self.cam_names),
            'camera_dirs': np.array([f'cam{i}' for i in range(len(self.cam_names))]),
        }
        for i, cam_name in enumerate(self.cam_names):
            intrinsic, extrinsic = self.get_camera_params(cam_name)
            cam_params[f'cam{i}_intrinsic'] = intrinsic
            cam_params[f'cam{i}_extrinsic'] = extrinsic
        path = os.path.join(output_dir, 'camera_params.npz')
        np.savez(path, **cam_params)
        print(f'Camera parameters saved to {path}')


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


def record_frame(env, rgbd_dir, pose_dir, frame_idx, gripper_states):
    """Save images, EE pose, and gripper state for one frame."""
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


# ─── Demo collection (rule) ──────────────────────────────────────────────────

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


def collect_rule_demo(args):
    """Collect a demonstration via a built-in rule trajectory."""
    if args.object != 'box' and args.object != 'box_2':
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
            record_frame(env, rgbd_dir, pose_dir, frame, gripper_states)

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


# ─── Demo collection (teleop) ────────────────────────────────────────────────

def collect_teleop_demo(args):
    """Collect a demonstration via WebXR teleoperation."""
    object_root, demo_root, demo_dir, rgbd_dir, pose_dir = prepare_output_dirs(
        args.object,
        args.demo_index,
    )

    # Create environment
    env = MujocoEnv(args.object)
    env.save_camera_params(object_root)

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
                record_frame(env, rgbd_dir, pose_dir, frame, gripper_states)

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

    finalize_demo(demo_dir, rgbd_dir, pose_dir, frame, gripper_states, env)


def collect_demo(args):
    """Dispatch to either teleop or rule-based demo collection."""
    if args.rule:
        collect_rule_demo(args)
    else:
        collect_teleop_demo(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MuJoCo demo collection for Instant Policy')
    parser.add_argument('--object', type=str, default='mug', help='Object name (loads asset/{object}.xml)')
    parser.add_argument('--demo_index', type=int, default=0,
                        help='Demo index to save under results/{object}/demo/demo_{index}')
    parser.add_argument('--rule', action='store_true',
                        help='Generate a built-in rule-based demo instead of WebXR teleop')
    parser.add_argument('--fps', type=float, default=25.0, help='Recording frame rate')
    parser.add_argument('--max_frames', type=int, default=2000,
                        help='Maximum number of frames to record (rule mode is capped at 120)')
    args = parser.parse_args()
    collect_demo(args)
