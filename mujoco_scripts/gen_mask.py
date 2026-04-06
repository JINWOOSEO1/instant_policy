"""
Generate segmentation masks using SAM2.
1. Interactive keypoint selection on first frame (OpenCV)
2. SAM2 image predictor for initial mask preview (closed loop)
3. SAM2 video predictor for tracking through all frames

Usage:
    python gen_mask.py --object mug --sam2_config sam2.1/configs/sam2.1/sam2.1_hiera_s.yaml --sam2_ckpt sam2_repo/checkpoints/sam2.1_hiera_small.pt
"""

import argparse
import importlib
import os
import sys
import glob

import cv2
import numpy as np
import torch
from PIL import Image

from sam2_repo.sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2_repo.sam2.sam2_image_predictor import SAM2ImagePredictor

# Enable performance optimizations for Ampere+ GPUs (RTX 30xx, 40xx, A100, etc.)
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Cache torch.compile autotune results to disk so that
# cam1 (and subsequent runs) skip the ~90s autotuning overhead.
import torch._inductor.config as inductor_cfg
inductor_cfg.fx_graph_cache = True        # reuse compiled graphs across init_state calls
if hasattr(inductor_cfg, "fx_graph_remote_cache"):
    inductor_cfg.fx_graph_remote_cache = False # local-only cache


def sam2_cuda_extension_available():
    """Return whether the optional SAM2 CUDA extension is importable."""
    try:
        importlib.import_module("sam2._C")
        return True
    except Exception:
        return False


class KeypointSelector:
    """Interactive OpenCV keypoint selection."""

    def __init__(self, window_name='Select Keypoints'):
        self.window_name = window_name
        self.points = []
        self.labels = []  # 1 = foreground, 0 = background
        self.image = None
        self.display = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Left click = positive point (foreground)
            self.points.append([x, y])
            self.labels.append(1)
            self._draw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click = negative point (background)
            self.points.append([x, y])
            self.labels.append(0)
            self._draw()

    def _draw(self):
        self.display = self.image.copy()
        for (x, y), label in zip(self.points, self.labels):
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            cv2.circle(self.display, (x, y), 5, color, -1)
            cv2.circle(self.display, (x, y), 7, color, 2)
        cv2.imshow(self.window_name, self.display)

    def select(self, image_rgb):
        """Show image, let user click points. Press 't' to confirm, 'r' to reset."""
        self.image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        self.display = self.image.copy()
        self.points = []
        self.labels = []

        cv2.imshow(self.window_name, self.display)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print(f"  Left-click: foreground point | Right-click: background point")
        print(f"  Press 't' to confirm | Press 'r' to reset points")

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('t') and len(self.points) > 0:
                break
            elif key == ord('r'):
                self.points = []
                self.labels = []
                self.display = self.image.copy()
                cv2.imshow(self.window_name, self.display)
                print("  Points reset. Click again.")

        return np.array(self.points), np.array(self.labels)


def show_mask_overlay(image_rgb, mask, window_name='Mask Preview'):
    """Show mask overlaid on image."""
    mask = mask.astype(bool)
    overlay = image_rgb.copy()
    overlay[mask] = overlay[mask] * 0.5 + np.array([0, 255, 0]) * 0.5
    display = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imshow(window_name, display)


def interactive_mask_selection(image_rgb, predictor, cam_label):
    """Closed-loop mask selection: click points, preview mask, accept or retry."""
    selector = KeypointSelector(window_name=f'{cam_label} - Select Keypoints')

    while True:
        points, labels = selector.select(image_rgb)

        # Predict mask with SAM2 image predictor
        predictor.set_image(image_rgb)
        masks, scores, logits = predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )

        # Show best mask
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        show_mask_overlay(image_rgb, best_mask, window_name=f'{cam_label} - Mask Preview')

        print(f"  Mask score: {scores[best_idx]:.3f}")
        print(f"  Press 't' to accept | Press 'r' to retry with new points")

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('t'):
                cv2.destroyAllWindows()
                return points, labels
            elif key == ord('r'):
                cv2.destroyWindow(f'{cam_label} - Mask Preview')
                break
            elif key == ord('q'):
                print("  Exiting.")
                sys.exit(0)


def track_masks_with_video_predictor(video_predictor, jpg_dir, points, labels, num_frames):
    """Track masks through video using SAM2VideoPredictor."""
    state = video_predictor.init_state(video_path=jpg_dir)

    # Add prompts on frame 0
    _, _, _ = video_predictor.add_new_points_or_box(
        state,
        frame_idx=0,
        obj_id=1,
        points=points,
        labels=labels,
    )

    # Propagate through video with inference_mode for speed
    masks = {}
    with torch.inference_mode():
        for frame_idx, obj_ids, mask_logits in video_predictor.propagate_in_video(state):
            mask = (mask_logits[0, 0] > 0).cpu().numpy()  # Binary mask for obj_id=1
            masks[frame_idx] = mask

    return masks


def main():
    parser = argparse.ArgumentParser(description='Generate segmentation masks using SAM2')
    parser.add_argument('--object', type=str, default='mug', help='Object name')
    parser.add_argument('--sam2_config', type=str, default='configs/sam2.1/sam2.1_hiera_s.yaml',
                        help='SAM2 config file (relative to sam2 package)')
    parser.add_argument('--sam2_ckpt', type=str, default='sam2_repo/checkpoints/sam2.1_hiera_small.pt',
                        help='SAM2 checkpoint path')
    args = parser.parse_args()

    data_dir = f'results/{args.object}'
    rgbd_dir = os.path.join(data_dir, 'RGBD_images')
    mask_dir = os.path.join(data_dir, 'mask')
    os.makedirs(mask_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    has_sam2_cuda_ext = sam2_cuda_extension_available()
    apply_postprocessing = has_sam2_cuda_ext

    if not has_sam2_cuda_ext:
        print(
            "SAM2 CUDA extension `sam2._C` is not available. "
            "Disabling SAM2 post-processing to avoid runtime warnings during VOS autotuning/inference."
        )
        print(
            "Mask generation and tracking will still run; only small-hole/sprinkle cleanup is skipped."
        )

    # Build SAM2 image predictor for interactive mask preview
    print('Loading SAM2 image predictor...')
    sam2_model = build_sam2(
        args.sam2_config,
        args.sam2_ckpt,
        device=device,
        apply_postprocessing=apply_postprocessing,
    )
    image_predictor = SAM2ImagePredictor(sam2_model)

    # Process each camera
    cam_points = {}
    cam_labels = {}

    for cam_idx in range(3):
        cam_label = f'cam{cam_idx}'
        print(f'\n--- {cam_label} ---')

        # Load first frame
        cam_dir = os.path.join(rgbd_dir, f'cam{cam_idx}')
        first_frame_path = os.path.join(cam_dir, '0000.jpg')
        if not os.path.exists(first_frame_path):
            print(f'  First frame not found: {first_frame_path}')
            continue

        image_rgb = np.array(Image.open(first_frame_path))

        # Interactive keypoint selection + mask preview (closed loop)
        points, labels = interactive_mask_selection(image_rgb, image_predictor, cam_label)
        cam_points[cam_idx] = points
        cam_labels[cam_idx] = labels
        print(f'  Selected {len(points)} points')

    # Build SAM2 video predictor for tracking (vos_optimized enables torch.compile)
    print('\nLoading SAM2 video predictor...')
    video_predictor = build_sam2_video_predictor(
        args.sam2_config,
        args.sam2_ckpt,
        device=device,
        apply_postprocessing=apply_postprocessing,
        vos_optimized=False,
    )

    for cam_idx in range(3):
        if cam_idx not in cam_points:
            continue

        cam_label = f'cam{cam_idx}'
        print(f'\n--- Tracking {cam_label} ---')

        # Use camera subdirectory directly (already contains JPEGs)
        cam_dir = os.path.join(rgbd_dir, f'cam{cam_idx}')
        jpg_files = sorted(glob.glob(os.path.join(cam_dir, '*.jpg')))
        num_frames = len(jpg_files)

        if num_frames == 0:
            print(f'  No frames found for {cam_label}')
            continue

        # Track masks
        print(f'  Tracking {num_frames} frames...')
        masks = track_masks_with_video_predictor(
            video_predictor, cam_dir,
            cam_points[cam_idx], cam_labels[cam_idx],
            num_frames,
        )

        # Save masks
        for frame_idx, mask in masks.items():
            mask_path = os.path.join(mask_dir, f'{cam_label}_{frame_idx:04d}_mask.npy')
            np.save(mask_path, mask)

        print(f'  Saved {len(masks)} masks to {mask_dir}/')

    print('\nDone!')


if __name__ == '__main__':
    main()
