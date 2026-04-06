'''
Deploy Instant Policy on MuJoCo simulation with real-time SAM2 video predictor tracking.

Usage:
    # Run from project root:
    python mujoco_scripts/deploy_mujoco.py --object mug
'''
import os
import tempfile
import time

import cv2
import numpy as np
import torch
import argparse

from instant_policy import sample_to_cond_demo, GraphDiffusion
from utils import transform_pcd, subsample_pcd, transform_to_pose

from mujoco_scripts.simulation import MujocoEnv
from mujoco_scripts.gen_seg_pcd import depth_to_pointcloud, camera_pcd_to_world
from mujoco_scripts.gen_mask import interactive_mask_selection

from sam2_repo.sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2_repo.sam2.sam2_image_predictor import SAM2ImagePredictor

# ── SAM2 performance optimizations (matches gen_mask.py) ─────────────────────
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

import torch._inductor.config as inductor_cfg
inductor_cfg.fx_graph_cache = True
if hasattr(inductor_cfg, "fx_graph_remote_cache"):
    inductor_cfg.fx_graph_remote_cache = False


class OnlineSAM2Tracker:
    """Wraps SAM2VideoPredictor for online per-frame tracking.

    Seeds on the first frame with keypoints, then tracks across subsequent
    frames using a growing memory bank — no keypoints needed after initialization.

    How it works:
      - inference_state["images"] is a (N, 3, H, W) CPU tensor that we extend
        each step by writing into a pre-allocated buffer.
      - SAM2's _get_image_feature() lazily computes backbone features on access,
        so extending the tensor is sufficient to add a new frame.
      - propagate_in_video(start_frame_idx=k, max_frame_num_to_track=1) processes
        only the new frame k, automatically attending to maskmem_features stored
        from all prior frames (the memory bank).
    """

    IMG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    IMG_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # Pre-allocate buffer in chunks to avoid O(N²) torch.cat copies
    _BUFFER_CHUNK = 64

    def __init__(self, video_predictor):
        self.predictor   = video_predictor
        self.image_size  = video_predictor.image_size
        self.inference_state = None
        self.frame_count = 0
        self.tmp_dir     = tempfile.mkdtemp()
        self._img_buffer = None   # pre-allocated (capacity, 3, H, W) CPU tensor
        self._img_capacity = 0

    def _preprocess(self, rgb: np.ndarray) -> torch.Tensor:
        """HxWx3 uint8 RGB → (3, image_size, image_size) float tensor (ImageNet-normalised)."""
        img = cv2.resize(rgb, (self.image_size, self.image_size))
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return (t - self.IMG_MEAN) / self.IMG_STD  # CPU tensor

    def _append_frame(self, img_t: torch.Tensor):
        """Append a (3, H, W) frame to the pre-allocated buffer and update inference_state."""
        idx = self.frame_count
        if idx >= self._img_capacity:
            # Grow buffer by chunk
            new_cap = self._img_capacity + self._BUFFER_CHUNK
            new_buf = torch.empty(new_cap, *img_t.shape, dtype=img_t.dtype)
            if self._img_buffer is not None:
                new_buf[:self._img_capacity] = self._img_buffer
            self._img_buffer = new_buf
            self._img_capacity = new_cap
        self._img_buffer[idx] = img_t
        # Point inference_state to a view of the valid portion (no copy)
        self.inference_state['images'] = self._img_buffer[:idx + 1]
        self.inference_state['num_frames'] = idx + 1

    @torch.inference_mode()
    def initialize(self, rgb: np.ndarray, points: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Initialize tracker with first-frame keypoint prompts.

        Args:
            rgb:    HxWx3 uint8 RGB image.
            points: (N, 2) float array of (x, y) pixel coords.
            labels: (N,) int array, 1=foreground / 0=background.

        Returns:
            Binary mask (H, W) bool for the first frame.
        """
        # init_state() requires a directory of JPEG frames
        cv2.imwrite(
            os.path.join(self.tmp_dir, '00000.jpg'),
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        )

        # offload_video_to_cpu=True keeps the image buffer on CPU so GPU memory
        # doesn't grow as we accumulate frames across the rollout
        self.inference_state = self.predictor.init_state(
            self.tmp_dir, offload_video_to_cpu=True
        )

        # Bootstrap pre-allocated buffer from init_state's loaded frame
        init_imgs = self.inference_state['images']  # (1, 3, H, W)
        self._img_buffer = torch.empty(
            self._BUFFER_CHUNK, *init_imgs.shape[1:], dtype=init_imgs.dtype
        )
        self._img_capacity = self._BUFFER_CHUNK
        self._img_buffer[0] = init_imgs[0]
        self.inference_state['images'] = self._img_buffer[:1]

        # Seed frame 0 with keypoints → stored in temp_output_dict
        self.predictor.add_new_points_or_box(
            self.inference_state,
            frame_idx=0,
            obj_id=1,
            points=points,
            labels=labels,
            normalize_coords=False,
        )

        # propagate_in_video preflight moves temp outputs to cond_frame_outputs,
        # building the initial memory bank entry for frame 0
        mask = None
        for _, _, masks in self.predictor.propagate_in_video(
            self.inference_state, max_frame_num_to_track=1
        ):
            mask = (masks[0, 0].cpu().numpy() > 0)

        self.frame_count = 1
        return mask

    @torch.inference_mode()
    def update(self, rgb: np.ndarray) -> np.ndarray:
        """Add a new frame and return its mask via memory-bank-guided tracking.

        Calls SAM2 internal methods directly, bypassing propagate_in_video()
        overhead (preflight, generator setup, processing-order calc per call).

        Args:
            rgb: HxWx3 uint8 RGB image (same camera as initialize()).

        Returns:
            Binary mask (H, W) bool, or None if propagation yields no result.
        """
        # Preprocess and append to pre-allocated buffer (no torch.cat copy)
        img_t = self._preprocess(rgb)
        frame_idx = self.frame_count
        self._append_frame(img_t)
        self.frame_count += 1

        # ── Direct _run_single_frame_inference (skip propagate_in_video overhead) ──
        obj_output_dict = self.inference_state["output_dict_per_obj"][0]

        current_out, pred_masks = self.predictor._run_single_frame_inference(
            inference_state=self.inference_state,
            output_dict=obj_output_dict,
            frame_idx=frame_idx,
            batch_size=1,
            is_init_cond_frame=False,
            point_inputs=None,
            mask_inputs=None,
            reverse=False,
            run_mem_encoder=True,
        )
        obj_output_dict["non_cond_frame_outputs"][frame_idx] = current_out
        self.inference_state["frames_tracked_per_obj"][0][frame_idx] = {
            "reverse": False
        }

        # Resize mask to original video resolution
        _, video_res_masks = self.predictor._get_orig_video_res_output(
            self.inference_state, pred_masks
        )
        mask = (video_res_masks[0, 0].cpu().numpy() > 0)

        # ── Prune old non-conditioning frame outputs ──
        # SAM2's memory bank only uses the most recent num_maskmem frames,
        # plus object pointers from max_obj_ptrs_in_encoder frames.
        # Keeping entries beyond that wastes memory and slows dict lookups.
        max_keep = max(
            getattr(self.predictor, 'num_maskmem', 7),
            getattr(self.predictor, 'max_obj_ptrs_in_encoder', 16),
        )
        non_cond = obj_output_dict["non_cond_frame_outputs"]
        if len(non_cond) > max_keep * 2:
            sorted_keys = sorted(non_cond.keys())
            cutoff = frame_idx - max_keep
            for old_key in sorted_keys:
                if old_key < cutoff:
                    del non_cond[old_key]
                else:
                    break

        return mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Deploy Instant Policy in MuJoCo with SAM2 video-predictor tracking'
    )
    parser.add_argument('--object', type=str, default='mug')
    parser.add_argument('--sam2_config', type=str,
                        default='configs/sam2.1/sam2.1_hiera_s.yaml',
                        help='SAM2 config path (relative to sam2 package)')
    parser.add_argument('--sam2_ckpt', type=str,
                        default='sam2_repo/checkpoints/sam2.1_hiera_small.pt',
                        help='SAM2 checkpoint path')
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
    data_dir = f'results/{args.object}'
    demos_processed = []
    for demo_idx in range(num_demos):
        demo_path = os.path.join(data_dir, f'demo_{demo_idx}.npy')
        demo = np.load(demo_path, allow_pickle=True).item()

        # Ensure all demo lists have the same length
        min_len = min(len(demo['pcds']), len(demo['T_w_es']), len(demo['grips']))
        demo['pcds']   = demo['pcds'][:min_len]
        demo['T_w_es'] = demo['T_w_es'][:min_len]
        demo['grips']  = demo['grips'][:min_len]

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

    # How many simulation substeps to take per action to approximate FPS
    dt = 1.0 / FPS
    sim_steps_per_action = max(1, int(dt / env.model.opt.timestep / args.execution_horizon))

    ############################################################################
    # Load SAM2
    print('Loading SAM2...')
    # Image predictor: used only for the interactive first-frame mask preview UI
    image_predictor = SAM2ImagePredictor(
        build_sam2(args.sam2_config, args.sam2_ckpt, device=device)
    )
    # Video predictor: used for memory-bank-based online tracking every step
    video_predictor = build_sam2_video_predictor(
        args.sam2_config, args.sam2_ckpt, device=device,
    )
    trackers = {c: OnlineSAM2Tracker(video_predictor) for c in cam_names}
    print('SAM2 loaded.')

    ############################################################################
    # Rollout loop
    points_per_cam  = {}
    labels_per_cam  = {}
    initialized_cam = {c: False for c in cam_names}

    for k in range(max_execution_steps):
        if not env.viewer_is_running():
            print('Viewer closed — stopping.')
            break

        t_loop_start = time.time()

        # ── Step 0 only: interactive keypoint selection ─────────────────────
        if k == 0:
            for cam in cam_names:
                rgb, _ = env.render_rgbd(cam)
                print(f'\n[{cam}] Select object keypoints.')
                pts, lbs = interactive_mask_selection(rgb, image_predictor, cam)
                points_per_cam[cam] = pts
                labels_per_cam[cam] = lbs
                print(f'[{cam}] {len(pts)} keypoint(s) confirmed.')

        # ── Observe current robot state ──────────────────────────────────────
        t0 = time.time()
        T_w_e = env.get_ee_pose()       # (4, 4) SE3, world frame
        grip  = env.get_gripper_state() # 0=closed, 1=open
        t_state = time.time() - t0

        pcd_list = []
        t_render_total = 0.0
        t_sam2_total = 0.0
        t_pcd_total = 0.0
        for cam in cam_names:
            t0 = time.time()
            rgb, depth = env.render_rgbd(cam)
            t_render_total += time.time() - t0

            intrinsic, extrinsic = cam_params[cam]
            fx, fy = intrinsic[0, 0], intrinsic[1, 1]
            cx, cy = intrinsic[0, 2], intrinsic[1, 2]

            t0 = time.time()
            if not initialized_cam[cam]:
                # First frame: seed video predictor with keypoints → builds memory bank
                mask = trackers[cam].initialize(
                    rgb, points_per_cam[cam], labels_per_cam[cam]
                )
                initialized_cam[cam] = True
            else:
                # Subsequent frames: memory-bank-guided tracking, no keypoints needed
                mask = trackers[cam].update(rgb)
            t_sam2_total += time.time() - t0

            if mask is None or mask.sum() == 0:
                continue  # empty mask → skip this camera, robot holds last command

            t0 = time.time()
            depth_masked = depth * mask.astype(np.float32)
            pcd_cam = depth_to_pointcloud(depth_masked, fx, fy, cx, cy)
            if len(pcd_cam) == 0:
                t_pcd_total += time.time() - t0
                continue
            pcd_list.append(camera_pcd_to_world(pcd_cam, extrinsic))
            t_pcd_total += time.time() - t0

        if not pcd_list:
            print(f'[step {k}] No valid pointcloud — skipping inference.')
            env.sync_viewer()
            continue

        pcd_w = np.concatenate(pcd_list, axis=0)

        # ── Model inference ──────────────────────────────────────────────────
        t0 = time.time()
        full_sample['live'] = {
            'obs':    [transform_pcd(subsample_pcd(pcd_w), np.linalg.inv(T_w_e))],
            'grips':  [grip],
            'T_w_es': [T_w_e],
        }
        actions, pred_grips = model.predict_actions(full_sample)
        print(f"Predicted gripps; {pred_grips.flatten()}")
        t_inference = time.time() - t0
        # actions:    (pred_horizon, 4, 4) relative EE transforms
        # pred_grips: (pred_horizon, 1)   -1=close, +1=open

        # ── Execute actions ──────────────────────────────────────────────────
        t0 = time.time()
        prev_grip_binary = grip
        actions_executed = 0
        for j in range(args.execution_horizon):
            T_w_e_next  = T_w_e @ actions[j]
            pose_7d     = transform_to_pose(T_w_e_next)  # [x,y,z, qx,qy,qz,qw]
            grip_binary = int((pred_grips[j] + 1) / 2 > 0.5)

            env.set_target(pose_7d[:3], pose_7d[3:], grip_binary * 255)
            env.step(n_substeps=sim_steps_per_action)
            env.sync_viewer()
            actions_executed += 1

            # Re-observe immediately when gripper state changes to avoid
            # open/close oscillation (CLAUDE.md performance tip)
            if grip_binary != prev_grip_binary:
                break
            prev_grip_binary = grip_binary
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
              f'sam2={t_sam2_total*1000:6.1f}ms | '
              f'pcd={t_pcd_total*1000:5.1f}ms | '
              f'inference={t_inference*1000:6.1f}ms | '
              f'exec={t_execution*1000:5.1f}ms({actions_executed}acts) | '
              f'sleep={sleep_time*1000:5.1f}ms')

    env.close()
    print('Deployment finished.')
