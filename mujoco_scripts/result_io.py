"""Shared demo/live result I/O helpers for MuJoCo scripts."""

import os

import numpy as np

from mujoco_scripts.result_paths import (
    get_live_gripper_state_path,
    get_live_pose_dir,
    get_live_root,
    get_live_step_path,
    resolve_demo_gripper_state_path,
    resolve_demo_pose_dir,
    resolve_demo_seg_pcd_dir,
    resolve_demo_file_path,
)


def load_raw_demo(object_name, demo_index):
    """Load one saved demo file and trim all sequences to a common length."""
    demo_path = resolve_demo_file_path(object_name, demo_index)
    if not os.path.exists(demo_path):
        raise FileNotFoundError(f'Demo file not found: {demo_path}')

    demo = np.load(demo_path, allow_pickle=True).item()
    min_len = min(len(demo['pcds']), len(demo['T_w_es']), len(demo['grips']))
    demo['pcds'] = list(demo['pcds'][:min_len])
    demo['T_w_es'] = list(demo['T_w_es'][:min_len])
    demo['grips'] = list(demo['grips'][:min_len])
    if 'gripper_commands' in demo:
        demo['gripper_commands'] = list(demo['gripper_commands'][:min_len])
    return demo, demo_path, min_len


def load_demo_from_results(object_name, demo_index):
    """Load one demo directly from saved result directories.

    Reads segmented pointclouds, EE poses, and gripper states from the canonical
    results layout under `results/{object}/demo/demo_{index}`.
    """
    seg_pcd_dir = resolve_demo_seg_pcd_dir(object_name, demo_index)
    pose_dir = resolve_demo_pose_dir(object_name, demo_index)
    gripper_state_path = resolve_demo_gripper_state_path(object_name, demo_index)

    if not os.path.isdir(seg_pcd_dir):
        raise FileNotFoundError(f"Segmented pointcloud dir not found: {seg_pcd_dir}")
    if not os.path.isdir(pose_dir):
        raise FileNotFoundError(f"EE pose dir not found: {pose_dir}")
    if not os.path.exists(gripper_state_path):
        raise FileNotFoundError(f"Gripper state file not found: {gripper_state_path}")

    pcd_files = sorted(
        file_name
        for file_name in os.listdir(seg_pcd_dir)
        if file_name.endswith(".npy")
    )
    pose_files = sorted(
        file_name
        for file_name in os.listdir(pose_dir)
        if file_name.endswith(".npy")
    )
    grips = list(np.load(gripper_state_path))

    pcds = [np.load(os.path.join(seg_pcd_dir, file_name)) for file_name in pcd_files]
    T_w_es = [np.load(os.path.join(pose_dir, file_name)) for file_name in pose_files]

    min_len = min(len(pcds), len(T_w_es), len(grips))
    demo = {
        "pcds": pcds[:min_len],
        "T_w_es": T_w_es[:min_len],
        "grips": grips[:min_len],
    }

    demo_path = (
        f"{seg_pcd_dir} + {pose_dir} + {gripper_state_path}"
    )
    return demo, demo_path, min_len


class LiveRolloutWriter:
    """Save deployment observations and executed commands into structured live logs."""

    def __init__(self, object_name):
        self.object_name = object_name
        self.enabled = True
        self.pose_counter = 0
        self.gripper_states = []

        os.makedirs(get_live_root(object_name), exist_ok=True)
        os.makedirs(get_live_pose_dir(object_name), exist_ok=True)

    def save_step(
        self,
        step_index,
        seg_pcd,
        T_w_e,
        gripper_state,
        actions,
        gripper_prediction,
        seg_pcd_full=None,
    ):
        """Save one model-input observation and its predicted outputs."""
        pcd_ee = seg_pcd_full if seg_pcd_full is not None else seg_pcd
        payload = {
            'step_index': step_index,
            'seg_pcd': seg_pcd.copy(),
            'pcd_ee': pcd_ee.copy(),
            'T_w_e': T_w_e.copy(),
            'grip': int(gripper_state),
            'actions': actions.copy(),
            'pred_grips': gripper_prediction.copy(),
        }
        np.save(get_live_step_path(self.object_name, step_index), payload, allow_pickle=True)

    def append_execution(self, T_w_e_next, gripper_state):
        """Append one executed EE pose and gripper command."""
        pose_path = os.path.join(get_live_pose_dir(self.object_name), f'{self.pose_counter:04d}.npy')
        np.save(pose_path, T_w_e_next)
        self.pose_counter += 1

        self.gripper_states.append(int(gripper_state))
        np.save(
            get_live_gripper_state_path(self.object_name),
            np.asarray(self.gripper_states, dtype=np.int32),
        )
