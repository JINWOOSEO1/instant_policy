"""
Save current results (seg_pcd, EE_pose, gripper_state) as a consolidated demo file.

Called by run_pipeline.sh after step 3 (gen_seg_pcd) for each demo iteration.

Usage:
    python mujoco_scripts/save_demo.py --object mug --demo_index 0
"""
import argparse
import os

import numpy as np

from mujoco_scripts.camera_utils import load_camera_entries
from mujoco_scripts.result_paths import (
    get_camera_params_path,
    get_demo_dir,
    get_demo_file_path,
    get_object_root,
    resolve_demo_gripper_state_path,
    resolve_demo_pose_dir,
    resolve_demo_rgbd_dir,
    resolve_demo_seg_pcd_dir,
)


def main():
    parser = argparse.ArgumentParser(description='Save consolidated demo file')
    parser.add_argument('--object', type=str, required=True)
    parser.add_argument('--demo_index', type=int, required=True)
    args = parser.parse_args()

    object_root = get_object_root(args.object)
    seg_pcd_dir = resolve_demo_seg_pcd_dir(args.object, args.demo_index)
    pose_dir = resolve_demo_pose_dir(args.object, args.demo_index)
    rgbd_dir = resolve_demo_rgbd_dir(args.object, args.demo_index)
    gripper_state_path = resolve_demo_gripper_state_path(args.object, args.demo_index)

    num_frames = len([f for f in os.listdir(seg_pcd_dir) if f.endswith('.npy')])
    pcds = [np.load(os.path.join(seg_pcd_dir, f'{i:04d}.npy')) for i in range(num_frames)]
    T_w_es = [np.load(os.path.join(pose_dir, f'{i:04d}.npy')) for i in range(num_frames)]
    grips = list(np.load(gripper_state_path))
    camera_entries = load_camera_entries(object_root, rgbd_dir=rgbd_dir)

    demo = {
        'pcds': pcds,
        'T_w_es': T_w_es,
        'grips': grips,
        'camera_names': [camera['name'] for camera in camera_entries],
        'camera_dirs': [camera['dir'] for camera in camera_entries],
    }

    cam_params_path = get_camera_params_path(args.object)
    if os.path.exists(cam_params_path):
        with np.load(cam_params_path, allow_pickle=True) as cam_params:
            demo['camera_params'] = {
                camera['dir']: {
                    'name': camera['name'],
                    'intrinsic': cam_params[f'cam{camera["index"]}_intrinsic'],
                    'extrinsic': cam_params[f'cam{camera["index"]}_extrinsic'],
                }
                for camera in camera_entries
                if f'cam{camera["index"]}_intrinsic' in cam_params.files
                and f'cam{camera["index"]}_extrinsic' in cam_params.files
            }

    out_path = get_demo_file_path(args.object, args.demo_index)
    os.makedirs(get_demo_dir(args.object, args.demo_index), exist_ok=True)
    np.save(out_path, demo, allow_pickle=True)
    print(f'Saved demo {args.demo_index} ({num_frames} frames) to {out_path}')


if __name__ == '__main__':
    main()
