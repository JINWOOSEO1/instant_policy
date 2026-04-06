"""
Save current results (seg_pcd, EE_pose, gripper_state) as a consolidated demo file.

Called by run_pipeline.sh after step 3 (gen_seg_pcd) for each demo iteration.

Usage:
    python mujoco_scripts/save_demo.py --object mug --demo_index 0
"""
import argparse
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Save consolidated demo file')
    parser.add_argument('--object', type=str, required=True)
    parser.add_argument('--demo_index', type=int, required=True)
    args = parser.parse_args()

    data_dir = f'results/{args.object}'
    seg_pcd_dir = os.path.join(data_dir, 'seg_pcd')
    ee_pose_dir = os.path.join(data_dir, 'EE_pose')

    num_frames = len([f for f in os.listdir(seg_pcd_dir) if f.endswith('.npy')])
    pcds = [np.load(os.path.join(seg_pcd_dir, f'{i:04d}.npy')) for i in range(num_frames)]
    T_w_es = [np.load(os.path.join(ee_pose_dir, f'{i:04d}.npy')) for i in range(num_frames)]
    grips = list(np.load(os.path.join(data_dir, 'gripper_state.npy')))

    demo = {'pcds': pcds, 'T_w_es': T_w_es, 'grips': grips}

    out_path = os.path.join(data_dir, f'demo_{args.demo_index}.npy')
    np.save(out_path, demo, allow_pickle=True)
    print(f'Saved demo {args.demo_index} ({num_frames} frames) to {out_path}')


if __name__ == '__main__':
    main()
