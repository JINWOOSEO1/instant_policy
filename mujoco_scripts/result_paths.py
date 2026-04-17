"""Helpers for MuJoCo result layouts."""

import os


def get_object_root(object_name):
    return os.path.join('results', object_name)


def get_camera_params_path(object_name):
    return os.path.join(get_object_root(object_name), 'camera_params.npz')


def get_demo_root(object_name):
    return os.path.join(get_object_root(object_name), 'demo')


def get_demo_dir(object_name, demo_index):
    return os.path.join(get_demo_root(object_name), f'demo_{demo_index}')


def get_demo_rgbd_dir(object_name, demo_index=None):
    if demo_index is None:
        return os.path.join(get_demo_root(object_name), 'RGBD_images')
    return os.path.join(get_demo_dir(object_name, demo_index), 'RGBD_images')


def get_demo_mask_dir(object_name, demo_index=None):
    if demo_index is None:
        return os.path.join(get_demo_root(object_name), 'mask')
    return os.path.join(get_demo_dir(object_name, demo_index), 'mask')


def get_demo_seg_pcd_dir(object_name, demo_index=None):
    if demo_index is None:
        return os.path.join(get_demo_root(object_name), 'seg_pcd')
    return os.path.join(get_demo_dir(object_name, demo_index), 'seg_pcd')


def get_demo_pose_dir(object_name, demo_index=None):
    if demo_index is None:
        return os.path.join(get_demo_root(object_name), 'T_w_e')
    return os.path.join(get_demo_dir(object_name, demo_index), 'T_w_e')


def get_demo_gripper_state_path(object_name, demo_index=None):
    if demo_index is None:
        return os.path.join(get_demo_root(object_name), 'gripper_state.npy')
    return os.path.join(get_demo_dir(object_name, demo_index), 'gripper_state.npy')


def get_demo_file_path(object_name, demo_index):
    return os.path.join(get_demo_dir(object_name, demo_index), f'demo_{demo_index}.npy')


def _first_existing_path(candidates, predicate):
    for candidate in candidates:
        if predicate(candidate):
            return candidate
    return candidates[0]


def resolve_demo_pose_dir(object_name, demo_index):
    return _first_existing_path([
        get_demo_pose_dir(object_name, demo_index),
        os.path.join(get_demo_root(object_name), 'T_w_e', f'demo_{demo_index}'),
        get_demo_pose_dir(object_name),
        os.path.join(get_object_root(object_name), 'EE_pose'),
    ], os.path.isdir)


def resolve_demo_rgbd_dir(object_name, demo_index):
    return _first_existing_path([
        get_demo_rgbd_dir(object_name, demo_index),
        get_demo_rgbd_dir(object_name),
        os.path.join(get_object_root(object_name), 'RGBD_images'),
    ], os.path.isdir)


def resolve_demo_mask_dir(object_name, demo_index):
    return _first_existing_path([
        get_demo_mask_dir(object_name, demo_index),
        get_demo_mask_dir(object_name),
        os.path.join(get_object_root(object_name), 'mask'),
    ], os.path.isdir)


def resolve_demo_seg_pcd_dir(object_name, demo_index):
    return _first_existing_path([
        get_demo_seg_pcd_dir(object_name, demo_index),
        os.path.join(get_demo_root(object_name), 'seg_pcd', f'demo_{demo_index}'),
        get_demo_seg_pcd_dir(object_name),
        os.path.join(get_object_root(object_name), 'seg_pcd'),
    ], os.path.isdir)


def resolve_demo_gripper_state_path(object_name, demo_index):
    return _first_existing_path([
        get_demo_gripper_state_path(object_name, demo_index),
        os.path.join(get_demo_root(object_name), 'gripper_state', f'demo_{demo_index}.npy'),
        get_demo_gripper_state_path(object_name),
        os.path.join(get_object_root(object_name), 'gripper_state.npy'),
    ], os.path.exists)


def resolve_demo_file_path(object_name, demo_index):
    return _first_existing_path([
        get_demo_file_path(object_name, demo_index),
        os.path.join(get_demo_dir(object_name, demo_index), 'demo.npy'),
        os.path.join(get_demo_root(object_name), f'demo_{demo_index}.npy'),
        os.path.join(get_object_root(object_name), f'demo_{demo_index}.npy'),
    ], os.path.exists)


def get_live_root(object_name):
    return os.path.join(get_object_root(object_name), 'live')


def get_live_pose_dir(object_name):
    return os.path.join(get_live_root(object_name), 'T_w_e')


def get_live_gripper_state_path(object_name):
    return os.path.join(get_live_root(object_name), 'gripper_state.npy')


def get_live_step_path(object_name, step_index):
    return os.path.join(get_live_root(object_name), f'step_{step_index:03d}.npy')
