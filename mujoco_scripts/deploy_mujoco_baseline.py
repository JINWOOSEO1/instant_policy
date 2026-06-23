"""
Deploy Instant Policy on MuJoCo simulation.

By default this uses MuJoCo ground-truth segmentation masks, matching the old
`deploy_mujoco_gt.py` behavior. Pass `--sam2` to use interactive SAM2-based
mask initialization and online tracking instead.
"""

import argparse
import contextlib
import os
import tempfile
import time

import cv2
import numpy as np
import mujoco
import torch
from pathlib import Path

from instant_policy import GraphDiffusion, sample_to_cond_demo
from utils import subsample_pcd, transform_pcd, transform_to_pose

from mujoco_scripts.demo_generation import OBJECT_GEOM_NAMES, interactive_mask_selection
from mujoco_scripts.result_io import LiveRolloutWriter, load_demo_from_results
from mujoco_scripts.simulation import MujocoEnv


def _xyaxes_to_camera_matrix(xyaxes):
    axes = np.asarray(xyaxes, dtype=np.float64)
    if axes.shape != (6,):
        raise ValueError(f"--record-camera-xyaxes must have 6 values, got {axes.shape}")
    x_axis = axes[:3]
    y_axis = axes[3:]
    x_axis /= np.linalg.norm(x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_norm = np.linalg.norm(z_axis)
    if z_norm == 0.0:
        raise ValueError("--record-camera-xyaxes x/y axes must not be collinear")
    z_axis /= z_norm
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def _apply_record_camera_pose(env, camera_name, pos, xyaxes):
    cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id == -1:
        raise ValueError(f'Record camera "{camera_name}" not found')

    mujoco.mj_forward(env.model, env.data)

    camera_pos_world = env.data.cam_xpos[cam_id].copy()
    camera_mat_world = env.data.cam_xmat[cam_id].reshape(3, 3).copy()

    if pos is not None:
        camera_pos_world = np.asarray(pos, dtype=np.float64)
        if camera_pos_world.shape != (3,):
            raise ValueError(
                f"--record-camera-pos must have 3 values, got {camera_pos_world.shape}"
            )
    if xyaxes is not None:
        camera_mat_world = _xyaxes_to_camera_matrix(xyaxes)

    body_id = env.model.cam_bodyid[cam_id]
    body_pos_world = env.data.xpos[body_id].copy()
    body_mat_world = env.data.xmat[body_id].reshape(3, 3).copy()
    body_mat_inv = body_mat_world.T

    camera_pos_local = body_mat_inv @ (camera_pos_world - body_pos_world)
    camera_mat_local = body_mat_inv @ camera_mat_world
    camera_quat_local = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(camera_quat_local, camera_mat_local.reshape(-1))

    env.model.cam_pos[cam_id] = camera_pos_local
    env.model.cam_quat[cam_id] = camera_quat_local
    env.model.cam_pos0[cam_id] = camera_pos_world
    env.model.cam_mat0[cam_id] = camera_mat_world.reshape(-1)

    mujoco.mj_forward(env.model, env.data)


def setup_sam2_torch():
    """Apply the same torch settings used by the SAM2 demo pipeline."""
    if torch.cuda.is_available():
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    import torch._inductor.config as inductor_cfg

    inductor_cfg.fx_graph_cache = True
    if hasattr(inductor_cfg, "fx_graph_remote_cache"):
        inductor_cfg.fx_graph_remote_cache = False


class OnlineSAM2Tracker:
    """Track one camera stream online with SAM2VideoPredictor."""

    IMG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    IMG_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    _BUFFER_CHUNK = 64

    def __init__(self, video_predictor):
        self.predictor = video_predictor
        self.image_size = video_predictor.image_size
        self.inference_state = None
        self.frame_count = 0
        self.tmp_dir = tempfile.mkdtemp()
        self._img_buffer = None
        self._img_capacity = 0

    @contextlib.contextmanager
    def _suppress_tqdm_output(self):
        """Silence SAM2's internal tqdm progress bars during live deployment."""
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with contextlib.redirect_stderr(devnull):
                yield

    def _preprocess(self, rgb: np.ndarray) -> torch.Tensor:
        """Convert HxWx3 uint8 RGB into SAM2's normalized tensor format."""
        img = cv2.resize(rgb, (self.image_size, self.image_size))
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return (tensor - self.IMG_MEAN) / self.IMG_STD

    def _append_frame(self, img_t: torch.Tensor):
        """Append a frame into the pre-allocated image buffer."""
        idx = self.frame_count
        if idx >= self._img_capacity:
            new_cap = self._img_capacity + self._BUFFER_CHUNK
            new_buf = torch.empty(new_cap, *img_t.shape, dtype=img_t.dtype)
            if self._img_buffer is not None:
                new_buf[:self._img_capacity] = self._img_buffer
            self._img_buffer = new_buf
            self._img_capacity = new_cap

        self._img_buffer[idx] = img_t
        self.inference_state["images"] = self._img_buffer[:idx + 1]
        self.inference_state["num_frames"] = idx + 1

    @torch.inference_mode()
    def initialize(self, rgb: np.ndarray, init_mask: np.ndarray) -> np.ndarray:
        """Initialize the tracker on the first frame using an accepted mask."""
        cv2.imwrite(
            os.path.join(self.tmp_dir, "00000.jpg"),
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        )

        with self._suppress_tqdm_output():
            self.inference_state = self.predictor.init_state(
                self.tmp_dir,
                offload_video_to_cpu=True,
            )

        init_imgs = self.inference_state["images"]
        self._img_buffer = torch.empty(
            self._BUFFER_CHUNK,
            *init_imgs.shape[1:],
            dtype=init_imgs.dtype,
        )
        self._img_capacity = self._BUFFER_CHUNK
        self._img_buffer[0] = init_imgs[0]
        self.inference_state["images"] = self._img_buffer[:1]

        _, _, masks = self.predictor.add_new_mask(
            self.inference_state,
            frame_idx=0,
            obj_id=1,
            mask=init_mask.astype(np.uint8),
        )
        mask = masks[0, 0].cpu().numpy() > 0
        self.predictor.propagate_in_video_preflight(self.inference_state)

        self.frame_count = 1
        return mask

    @torch.inference_mode()
    def update(self, rgb: np.ndarray) -> np.ndarray:
        """Track the next frame with the stored SAM2 memory bank."""
        img_t = self._preprocess(rgb)
        frame_idx = self.frame_count
        self._append_frame(img_t)
        self.frame_count += 1

        mask = None
        with self._suppress_tqdm_output():
            for _, _, video_res_masks in self.predictor.propagate_in_video(
                self.inference_state,
                start_frame_idx=frame_idx,
                max_frame_num_to_track=1,
            ):
                mask = video_res_masks[0, 0].cpu().numpy() > 0
                break

        obj_output_dict = self.inference_state["output_dict_per_obj"][0]
        max_keep = max(
            getattr(self.predictor, "num_maskmem", 7),
            getattr(self.predictor, "max_obj_ptrs_in_encoder", 16),
        )
        non_cond = obj_output_dict["non_cond_frame_outputs"]
        if len(non_cond) > max_keep * 2:
            cutoff = frame_idx - max_keep
            for old_key in sorted(non_cond.keys()):
                if old_key < cutoff:
                    del non_cond[old_key]
                else:
                    break

        return mask


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Deploy Instant Policy in MuJoCo with GT masks or optional SAM2 masks"
    )
    parser.add_argument("--object", type=str, default="scene_standalone")
    parser.add_argument(
        "--sam2",
        action="store_true",
        help="Use interactive SAM2 masks instead of MuJoCo ground-truth masks",
    )
    parser.add_argument(
        "--sam2_config",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_s.yaml",
        help="SAM2 config path (used only with --sam2)",
    )
    parser.add_argument(
        "--sam2_ckpt",
        type=str,
        default="sam2_repo/checkpoints/sam2.1_hiera_small.pt",
        help="SAM2 checkpoint path (used only with --sam2)",
    )
    parser.add_argument(
        "--num_demos",
        type=int,
        default=1,
        help="Number of demos to load (demo_0.npy, demo_1.npy, ...)",
    )
    parser.add_argument(
        "--execution_horizon",
        type=int,
        default=8,
        help="Number of predicted actions to execute per inference step",
    )
    parser.add_argument("--object-x", type=float, default=None,
                        help="Override kettle x position on the table")
    parser.add_argument("--object-y", type=float, default=None,
                        help="Override kettle y position on the table")
    parser.add_argument("--headless", action="store_true",
                        help="Skip the passive viewer (offscreen rendering only)")
    parser.add_argument("--record-video", type=str, default=None,
                        help="Save rollout video to this MP4 path")
    parser.add_argument("--record-camera", type=str, default="cam_front",
                        help="Camera used for offscreen recording")
    parser.add_argument("--record-camera-pos", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"),
                        help="Override record camera position")
    parser.add_argument("--record-camera-xyaxes", type=float, nargs=6, default=None,
                        metavar=("X0", "X1", "X2", "Y0", "Y1", "Y2"),
                        help="Override record camera orientation as MuJoCo xyaxes")
    parser.add_argument("--record-fps", type=float, default=30.0,
                        help="Video frame rate")
    parser.add_argument("--record-width", type=int, default=640)
    parser.add_argument("--record-height", type=int, default=480)
    parser.add_argument("--max-duration", type=float, default=90.0,
                        help="Maximum video duration in seconds")
    args = parser.parse_args(argv)

    ############################################################################
    # Rollout parameters
    num_demos = args.num_demos
    num_traj_wp = 10
    num_diffusion_iters = 4
    max_execution_steps = 100
    FPS = 10.0
    ############################################################################
    # Load Instant Policy model
    model_path = "./checkpoints"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GraphDiffusion.load_from_checkpoint(
        f"{model_path}/model.pt",
        device=device,
        strict=True,
        map_location=device,
    )
    model.set_num_demos(num_demos)
    model.set_num_diffusion_steps(num_diffusion_iters)
    model.eval()

    ############################################################################
    # Load demonstration data
    demos_processed = []
    for demo_idx in range(num_demos):
        demo, demo_path, min_len = load_demo_from_results(args.object, demo_idx)
        demos_processed.append(sample_to_cond_demo(demo, num_traj_wp))
        print(f"Loaded demo {demo_idx} ({min_len} frames) from {demo_path}")

    full_sample = {
        "demos": demos_processed,
        "live": {},
    }
    assert len(full_sample["demos"][0]["obs"]) == num_traj_wp

    ############################################################################
    # Initialise MuJoCo environment
    env = MujocoEnv(
        args.object,
        height=480,
        width=640,
        cam_names=["cam_left", "cam_front", "cam_right"],
    )

    if args.object_x is not None or args.object_y is not None:
        jnt_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_JOINT, "kettle_root_jnt"
        )
        if jnt_id != -1:
            adr = env.model.jnt_qposadr[jnt_id]
            if args.object_x is not None:
                env.data.qpos[adr] = args.object_x
            if args.object_y is not None:
                env.data.qpos[adr + 1] = args.object_y
            mujoco.mj_forward(env.model, env.data)

    if args.record_camera_pos is not None or args.record_camera_xyaxes is not None:
        _apply_record_camera_pose(
            env,
            args.record_camera,
            args.record_camera_pos,
            args.record_camera_xyaxes,
        )

    if not args.headless:
        env.launch_viewer()

    record_renderer = None
    video_writer = None
    total_video_frames = 0
    if args.record_video:
        Path(args.record_video).parent.mkdir(parents=True, exist_ok=True)
        record_renderer = mujoco.Renderer(
            env.model, height=args.record_height, width=args.record_width
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(args.record_video), fourcc, args.record_fps,
            (args.record_width, args.record_height),
        )

    cam_names = env.cam_names
    cam_params = {cam_name: env.get_camera_params(cam_name) for cam_name in cam_names}
    dt = 1.0 / FPS

    ############################################################################
    # Segmentation setup
    if args.sam2:
        setup_sam2_torch()
        from sam2_repo.sam2.build_sam import build_sam2, build_sam2_video_predictor
        from sam2_repo.sam2.sam2_image_predictor import SAM2ImagePredictor

        print("Segmentation mode: SAM2")
        print("Loading SAM2...")
        image_predictor = SAM2ImagePredictor(
            build_sam2(args.sam2_config, args.sam2_ckpt, device=device)
        )
        video_predictor = build_sam2_video_predictor(
            args.sam2_config,
            args.sam2_ckpt,
            device=device,
        )
        trackers = {cam_name: OnlineSAM2Tracker(video_predictor) for cam_name in cam_names}
        init_masks_per_cam = {}
        initialized_cam = {cam_name: False for cam_name in cam_names}
        print("SAM2 loaded.")
    else:
        print("Segmentation mode: MuJoCo GT")
        if args.object not in OBJECT_GEOM_NAMES:
            raise ValueError(
                f'Unknown object "{args.object}". '
                f"Available: {sorted(OBJECT_GEOM_NAMES.keys())}"
            )
        object_geom_ids = env.get_geom_ids_by_names(
            OBJECT_GEOM_NAMES[args.object],
            strict=(args.object != "scene_standalone"),
        )
        print(
            f'Object "{args.object}" has {len(object_geom_ids)} geom(s): '
            f"{sorted(object_geom_ids)}"
        )

    ############################################################################
    # Rollout loop
    live_writer = LiveRolloutWriter(args.object)

    stop_recording = False
    rollout_start_time = time.monotonic()
    for k in range(max_execution_steps):
        if stop_recording:
            break
        if args.record_video and time.monotonic() - rollout_start_time >= args.max_duration:
            stop_recording = True
            break
        if not args.headless and not env.viewer_is_running():
            print("Viewer closed -- stopping.")
            break

        t_loop_start = time.time()

        if args.sam2 and k == 0:
            for cam_name in cam_names:
                rgb, _ = env.render_rgbd(cam_name)
                print(f"\n[{cam_name}] Select object keypoints.")
                points, labels, init_mask = interactive_mask_selection(
                    rgb,
                    image_predictor,
                    cam_name,
                    return_mask=True,
                )
                init_masks_per_cam[cam_name] = init_mask
                print(
                    f"[{cam_name}] {len(points)} keypoint(s) confirmed. "
                    f"Accepted mask area={int(init_mask.sum())} px."
                )

        # ── Observe current robot state ──────────────────────────────────────
        t0 = time.time()
        T_w_e = env.get_ee_pose()
        grip = env.get_gripper_state()
        t_state = time.time() - t0

        if args.sam2:
            def mask_fn(cam_name, rgb, _depth):
                if not initialized_cam[cam_name]:
                    mask = trackers[cam_name].initialize(
                        rgb,
                        init_masks_per_cam[cam_name],
                    )
                    initialized_cam[cam_name] = True
                    return mask
                return trackers[cam_name].update(rgb)
        else:
            def mask_fn(cam_name, _rgb, _depth):
                return env.render_seg_mask(cam_name, object_geom_ids)

        pcd_w, pcd_stats = env.get_segmented_pcd(
            mask_fn,
            cam_names=cam_names,
            cam_params=cam_params,
            return_stats=True,
        )
        t_render_total = pcd_stats["render"]
        t_seg_total = pcd_stats["mask"]
        t_pcd_total = pcd_stats["pcd"]

        if pcd_w is None:
            print(f"[step {k}] No valid pointcloud -- skipping inference.")
            if not args.headless:
                env.sync_viewer()
            continue

        pcd_ee = transform_pcd(pcd_w, np.linalg.inv(T_w_e))

        # ── Model inference ──────────────────────────────────────────────────
        t0 = time.time()
        live_obs = subsample_pcd(pcd_ee)
        full_sample["live"] = {
            "obs": [live_obs],
            "grips": [grip],
            "T_w_es": [T_w_e],
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
        print(f"Predicted grips: {pred_grips.flatten()}")
        t_inference = time.time() - t0

        # ── Execute actions (IK + physics convergence loop) ────────────────
        t0 = time.time()
        actions_executed = 0

        for j in range(args.execution_horizon):
            if args.record_video and time.monotonic() - rollout_start_time >= args.max_duration:
                stop_recording = True
                break

            T_w_e_next = T_w_e @ actions[j]
            pose_next = transform_to_pose(T_w_e_next)
            grip_binary = int((pred_grips[j] + 1) / 2 > 0.5)

            live_writer.append_execution(T_w_e_next, grip_binary)

            grip_val = grip_binary * 255
            env.set_target(pose_next[:3], pose_next[3:], grip_val)
            env.step(n_substeps=20, converge=True, max_ik_iters=300)

            if not args.headless:
                env.sync_viewer()
            if video_writer is not None:
                record_renderer.update_scene(env.data, camera=args.record_camera)
                frame_rgb = record_renderer.render()
                video_writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                total_video_frames += 1
                if time.monotonic() - rollout_start_time >= args.max_duration:
                    stop_recording = True
                    break
            actions_executed += 1

        t_execution = time.time() - t0
        if stop_recording:
            wall_sec = time.monotonic() - rollout_start_time
            encoded_sec = total_video_frames / args.record_fps
            print(
                "Reached max wall-clock duration "
                f"({wall_sec:.1f}s / {args.max_duration:.1f}s, "
                f"encoded video {encoded_sec:.1f}s) -- stopping process."
            )
            break

        # ── Maintain target FPS ──────────────────────────────────────────────
        elapsed = time.time() - t_loop_start
        sleep_time = max(0.0, dt - elapsed)
        time.sleep(sleep_time)

        # ── Timing log ───────────────────────────────────────────────────────
        print(
            f"[step {k:3d}] "
            f"total={elapsed*1000:6.1f}ms | "
            f"state={t_state*1000:5.1f}ms | "
            f"render={t_render_total*1000:5.1f}ms | "
            f"seg={t_seg_total*1000:6.1f}ms | "
            f"pcd={t_pcd_total*1000:5.1f}ms | "
            f"inference={t_inference*1000:6.1f}ms | "
            f"exec={t_execution*1000:5.1f}ms({actions_executed}acts) | "
            f"sleep={sleep_time*1000:5.1f}ms"
        )

    if video_writer is not None:
        video_writer.release()
        wall_sec = time.monotonic() - rollout_start_time
        print(
            f"Video saved: {args.record_video} "
            f"({total_video_frames} frames, "
            f"{total_video_frames / args.record_fps:.1f}s encoded, "
            f"{wall_sec:.1f}s wall-clock)"
        )
    env.close()
    print("Deployment finished.")


if __name__ == "__main__":
    main()
