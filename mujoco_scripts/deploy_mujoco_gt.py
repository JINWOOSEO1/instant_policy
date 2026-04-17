'''
Deploy Instant Policy on MuJoCo simulation with ground-truth segmentation masks.

Instead of SAM2 tracking, uses MuJoCo's built-in segmentation rendering to
obtain pixel-perfect object masks directly from the simulator.

Usage:
    # Run from project root:
    python mujoco_scripts/deploy_mujoco_gt.py --object mug
'''
import os
import time

import numpy as np
import torch
import argparse

from instant_policy import sample_to_cond_demo, GraphDiffusion
from utils import transform_pcd, subsample_pcd, transform_to_pose

from mujoco_scripts.result_io import LiveRolloutWriter, load_raw_demo
from mujoco_scripts.result_paths import get_live_pose_dir, get_object_root
from mujoco_scripts.simulation import MujocoEnv

# ── Object → geom names mapping ────────────────────────────────────────────
# Each entry lists the MuJoCo geom names that make up the manipulation target.
OBJECT_GEOM_NAMES = {
    'mug': [
        # mug body
        'mug_bottom',
        'mug_wall_00', 'mug_wall_01', 'mug_wall_02', 'mug_wall_03',
        'mug_wall_04', 'mug_wall_05', 'mug_wall_06', 'mug_wall_07',
        'mug_wall_08', 'mug_wall_09', 'mug_wall_10', 'mug_wall_11',
        'mug_handle_top', 'mug_handle_mid', 'mug_handle_bot', 'mug_handle_low',
        # mug_rack body
        'rack_base', 'rack_post',
        'rack_branch_top_1', 'rack_branch_top_2',
        'rack_branch_mid_1', 'rack_branch_mid_2',
    ],
    'box': [
        # small_box body
        'small_box_geom',
        # target_box body
        'target_box_bottom',
        'target_box_wall_pos_x', 'target_box_wall_neg_x',
        'target_box_wall_pos_y', 'target_box_wall_neg_y',
    ]
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Deploy Instant Policy in MuJoCo with ground-truth segmentation'
    )
    parser.add_argument('--object', type=str, default='mug')
    parser.add_argument('--num_demos', type=int, default=1,
                        help='Number of demos to load (demo_0.npy, demo_1.npy, ...)')
    parser.add_argument('--execution_horizon', type=int, default=8,
                        help='Number of predicted actions to execute per inference step')
    args = parser.parse_args()

    ############################################################################
    # Rollout parameters
    num_demos           = args.num_demos
    num_traj_wp         = 10
    num_diffusion_iters = 4
    max_execution_steps = 100
    FPS                 = 10.0
    ############################################################################
    # Load Instant Policy model
    model_path = './checkpoints'
    device     = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = GraphDiffusion.load_from_checkpoint(
        f'{model_path}/model.pt',
        device=device,
        strict=True,
        map_location=device,
    )
    model.set_num_demos(num_demos)
    model.set_num_diffusion_steps(num_diffusion_iters)
    model.eval()

    ############################################################################
    # Load demonstration data
    data_dir = get_object_root(args.object)
    demos_processed = []
    for demo_idx in range(num_demos):
        demo, demo_path, min_len = load_raw_demo(args.object, demo_idx)

        demos_processed.append(sample_to_cond_demo(demo, num_traj_wp))
        print(f'Loaded demo {demo_idx} ({min_len} frames) from {demo_path}')

    full_sample = {
        'demos': demos_processed,
        'live':  {},
    }
    assert len(full_sample['demos'][0]['obs']) == num_traj_wp

    ############################################################################
    # Initialise MuJoCo environment
    env = MujocoEnv(args.object)
    env.launch_viewer()

    # Cache static camera intrinsics / extrinsics (cameras don't move)
    cam_names  = env.cam_names
    cam_params = {c: env.get_camera_params(c) for c in cam_names}

    dt = 1.0 / FPS

    ############################################################################
    # Ground-truth segmentation setup
    if args.object not in OBJECT_GEOM_NAMES:
        raise ValueError(
            f'Unknown object "{args.object}". '
            f'Available: {list(OBJECT_GEOM_NAMES.keys())}'
        )
    object_geom_ids = env.get_geom_ids_by_names(OBJECT_GEOM_NAMES[args.object])
    print(f'Object "{args.object}" has {len(object_geom_ids)} geom(s): {sorted(object_geom_ids)}')

    ############################################################################
    # Rollout loop
    live_writer = LiveRolloutWriter(args.object)
    ee_pose_deploy_dir = None
    if not live_writer.enabled:
        ee_pose_deploy_dir = get_live_pose_dir(args.object)
        os.makedirs(ee_pose_deploy_dir, exist_ok=True)
    ee_pose_counter = 0

    for k in range(max_execution_steps):
        if not env.viewer_is_running():
            print('Viewer closed — stopping.')
            break

        t_loop_start = time.time()

        # ── Observe current robot state ──────────────────────────────────────
        t0 = time.time()
        T_w_e = env.get_ee_pose()       # (4, 4) SE3, world frame
        grip  = env.get_gripper_state() # 0=closed, 1=open
        t_state = time.time() - t0

        pcd_w, pcd_stats = env.get_segmented_pcd(
            lambda cam_name, _rgb, _depth: env.render_seg_mask(cam_name, object_geom_ids),
            cam_names=cam_names,
            cam_params=cam_params,
            return_stats=True,
        )
        t_render_total = pcd_stats['render']
        t_seg_total = pcd_stats['mask']
        t_pcd_total = pcd_stats['pcd']

        if pcd_w is None:
            print(f'[step {k}] No valid pointcloud — skipping inference.')
            env.sync_viewer()
            continue

        # Transform from world frame to EE (gripper_tcp) frame
        pcd_ee = transform_pcd(pcd_w, np.linalg.inv(T_w_e))
        # ── Model inference ──────────────────────────────────────────────────
        t0 = time.time()
        live_obs = subsample_pcd(pcd_ee)
        full_sample['live'] = {
            'obs':    [live_obs],
            'grips':  [grip],
            'T_w_es': [T_w_e],
        }
        actions, pred_grips = model.predict_actions(full_sample)
        live_writer.save_step(
            k,
            live_obs,
            T_w_e,
            grip,
            actions,
            pred_grips,
            seg_pcd_full=pcd_ee,
        )
        print(f"Predicted gripps; {pred_grips.flatten()}")
        t_inference = time.time() - t0
        # actions:    (pred_horizon, 4, 4) relative EE transforms
        # pred_grips: (pred_horizon, 1)   -1=close, +1=open

        # ── Execute actions (IK + physics convergence loop) ────────────────
        t0 = time.time()
        actions_executed = 0

        for j in range(args.execution_horizon):
            T_w_e_next = T_w_e @ actions[j]
            pose_next = transform_to_pose(T_w_e_next)  # [x,y,z, qx,qy,qz,qw]
            grip_binary = int((pred_grips[j] + 1) / 2 > 0.5)

            if live_writer.enabled:
                live_writer.append_execution(T_w_e_next, grip_binary)
            else:
                np.save(os.path.join(ee_pose_deploy_dir, f'{ee_pose_counter:04d}.npy'), T_w_e_next)
                ee_pose_counter += 1

            grip_val = grip_binary * 255
            env.set_target(pose_next[:3], pose_next[3:], grip_val)
            n_ik_iters = env.step_until_converged(n_substeps=50, max_ik_iters=100)

            env.sync_viewer()
            actions_executed += 1

        t_execution = time.time() - t0

        # ── Maintain target FPS ──────────────────────────────────────────────
        elapsed = time.time() - t_loop_start
        sleep_time = max(0.0, dt - elapsed)
        time.sleep(sleep_time)

        # ── Timing log ───────────────────────────────────────────────────────
        print(f'[step {k:3d}] '
              f'total={elapsed*1000:6.1f}ms | '
              f'state={t_state*1000:5.1f}ms | '
              f'render={t_render_total*1000:5.1f}ms | '
              f'seg={t_seg_total*1000:6.1f}ms | '
              f'pcd={t_pcd_total*1000:5.1f}ms | '
              f'inference={t_inference*1000:6.1f}ms | '
              f'exec={t_execution*1000:5.1f}ms({actions_executed}acts) | '
              f'sleep={sleep_time*1000:5.1f}ms')

    env.close()
    print('Deployment finished.')
