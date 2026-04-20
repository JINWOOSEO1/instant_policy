"""Rule-based trajectory builders for MuJoCo demo generation."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R


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


def build_box_rule_trajectory(env, total_frames):
    """Build a fixed pick-and-place plan for the current box task."""
    if total_frames < 8:
        raise ValueError('Rule-based box demo needs at least 8 frames')

    ee_pose = env.get_ee_pose()
    home_pos = ee_pose[:3, 3].copy()
    home_quat = R.from_matrix(ee_pose[:3, :3]).as_quat()

    small_box_center = env.get_geom_position('small_box_geom')
    target_box_bottom = env.get_geom_position('target_box_bottom')

    small_box_geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, 'small_box_geom')
    target_bottom_geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, 'target_box_bottom')
    box_half_height = env.model.geom_size[small_box_geom_id][2]
    target_bottom_half_height = env.model.geom_size[target_bottom_geom_id][2]

    pregrasp_pos = small_box_center + np.array([0.0, -0.03, 0.15])
    grasp_pos = small_box_center.copy()

    # Tuned for the current asset/box.xml geometry. XY scene randomization
    # preserves object heights, so a fixed transport height is sufficient.
    clearance_z = 0.47

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
    phase_frames = allocate_segment_frames(total_frames, [20, 10, 10, 15, 20, 10, 10, 5])

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


def build_mug_rule_trajectory(env, total_frames):
    """Build a rim-grasp and rack-hanging plan for mug tasks."""
    if total_frames < 8:
        raise ValueError('Rule-based mug demo needs at least 8 frames')

    ee_pose = env.get_ee_pose()
    home_pos = ee_pose[:3, 3].copy()
    home_quat = R.from_matrix(ee_pose[:3, :3]).as_quat()

    mug_center = env.get_body_position('source_object')
    target_branch = env.get_geom_position('rack_branch_top_2')

    # Grip the mug on the rim edge first, then carry it so the handle lines up
    # with the top rack branch before opening near the very end of the clip.
    grasp_pos = mug_center + np.array([0.01, -0.06, 0.13])
    pregrasp_pos = grasp_pos + np.array([0.0, 0.0, 0.10])

    lift_pos = np.array([
        grasp_pos[0],
        grasp_pos[1],
        max(home_pos[2] - 0.001, grasp_pos[2] + 0.25),
    ])

    # This target was tuned against the MuJoCo scene so that, while the rim is
    # still pinched, the mug handle is already wrapped around rack_branch_top_2.
    hang_pos = target_branch + np.array([-0.07, -0.133, 0.077])
    approach_branch_pos = hang_pos + np.array([0.0, -0.03, 0.05])

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
    phase_frames = allocate_segment_frames(total_frames, [30, 15, 10, 30, 40, 40, 15, 4, 3])

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
