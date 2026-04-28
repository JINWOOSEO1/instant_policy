"""
MuJoCo environment for Instant Policy.
Provides MujocoEnv class for Panda robot manipulation tasks.

Usage:
    from mujoco_scripts.simulation import MujocoEnv
"""

import os
import time

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R

from mujoco_scripts.camera_utils import (
    DEFAULT_CAMERA_NAMES,
    camera_pcd_to_world,
    depth_to_pointcloud,
)


SCENE_NOISE_STD = 0.02
SCENE_NOISE_CLIP = 0.03
MIN_OBJECT_CENTER_DISTANCE = 0.1
SCENE_MAX_RESAMPLES = 100


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
        self.target_object_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            'target_object',
        )
        self.target_object_base_pos = None
        if self.target_object_body_id != -1:
            self.target_object_base_pos = self.model.body_pos[self.target_object_body_id].copy()

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
        self.randomize_scene()

        # Initialize IK target to current TCP pose
        self.target_pos = self.data.site_xpos[self.ee_site_id].copy()
        ee_mat = self.data.site_xmat[self.ee_site_id].reshape(3, 3)
        self.target_quat = scipy_quat_to_mujoco(R.from_matrix(ee_mat).as_quat())
        self.gripper_val = float(self.data.ctrl[7])

    def randomize_scene(self):
        """Apply per-reset scene randomization while preserving XML orientations."""
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
            # source_noise = np.zeros(2)
            target_noise = np.clip(
                np.random.normal(loc=0.0, scale=SCENE_NOISE_STD, size=2),
                -SCENE_NOISE_CLIP,
                SCENE_NOISE_CLIP,
            )
            # target_noise = np.zeros(2)

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

    def step(self, n_substeps=1, converge=False, joint_tol=1e-3,
             max_ik_iters=100):
        """Run IK update(s) + physics steps.

        If *converge* is False (default): run one IK update + *n_substeps*
        physics steps — equivalent to the old ``step()``.

        If *converge* is True: repeat IK updates (each followed by
        *n_substeps* physics steps) until the joint delta is below
        *joint_tol* or *max_ik_iters* is reached — equivalent to the
        old ``step_until_converged()``.  Returns the number of IK
        iterations executed.
        """
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
            self.data.ctrl[7] = self.gripper_val

            for _ in range(n_substeps):
                self.data.qfrc_applied[:7] = self.data.qfrc_bias[:7]
                mujoco.mj_step(self.model, self.data)

            if converge and np.max(np.abs(dq)) < joint_tol:
                return i
        if converge:
            return max_ik_iters

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
