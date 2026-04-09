"""
Generate segmented pointclouds from masks and depth images.
Combines masks with depth data, deprojects to 3D, transforms to world frame,
and merges pointclouds from both cameras.

Usage:
    python gen_seg_pcd.py --object mug
"""

import argparse
import os
import glob

import numpy as np
from utils import transform_pcd, downsample_pcd


def depth_to_pointcloud(depth, fx, fy, cx, cy):
    """Deproject depth image to camera-frame pointcloud."""
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    # Filter invalid points
    valid = (points[:, 2] > 0) & (points[:, 2] < 2.0)
    return points[valid]


def camera_pcd_to_world(pcd_cam, cam_extrinsic):
    """Transform pointcloud from camera frame to world frame.

    MuJoCo camera convention: OpenGL style where -Z points into the scene.
    Depth deprojection gives +Z into the scene.
    Apply flip to convert from deprojection convention to MuJoCo camera frame.
    """
    # Flip Y and Z to go from OpenCV-like deprojection to OpenGL camera frame
    flip = np.diag([1.0, -1.0, -1.0])
    pcd_gl = (flip @ pcd_cam.T).T

    # cam_extrinsic is the camera's world pose: T_world_cam
    # cam_extrinsic[:3, :3] = rotation, cam_extrinsic[:3, 3] = position
    pcd_world = transform_pcd(pcd_gl, cam_extrinsic)
    return pcd_world


def main():
    parser = argparse.ArgumentParser(description='Generate segmented pointclouds')
    parser.add_argument('--object', type=str, default='mug', help='Object name')
    parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel size for downsampling')
    args = parser.parse_args()

    data_dir = f'results/{args.object}'
    rgbd_dir = os.path.join(data_dir, 'RGBD_images')
    mask_dir = os.path.join(data_dir, 'mask')
    pcd_dir = os.path.join(data_dir, 'seg_pcd')
    os.makedirs(pcd_dir, exist_ok=True)

    # Load camera parameters
    cam_params_path = os.path.join(data_dir, 'camera_params.npz')
    if not os.path.exists(cam_params_path):
        print(f'Camera parameters not found: {cam_params_path}')
        return
    cam_params = np.load(cam_params_path)

    # Load EE poses for world-to-EE-frame transformation
    ee_pose_dir = os.path.join(data_dir, 'EE_pose')
    if not os.path.exists(ee_pose_dir):
        print(f'EE pose directory not found: {ee_pose_dir}')
        return

    # Determine number of frames from depth files
    depth_files = sorted(glob.glob(os.path.join(rgbd_dir, 'cam0', '*_depth.npy')))
    num_frames = len(depth_files)
    if num_frames == 0:
        print('No depth files found.')
        return

    print(f'Processing {num_frames} frames...')

    for frame_idx in range(num_frames):
        cam_pcds = []

        for cam_idx in range(3):
            # Load mask
            mask_path = os.path.join(mask_dir, f'cam{cam_idx}_{frame_idx:04d}_mask.npy')
            if not os.path.exists(mask_path):
                continue
            mask = np.load(mask_path)

            # Load depth
            depth_path = os.path.join(rgbd_dir, f'cam{cam_idx}', f'{frame_idx:04d}_depth.npy')
            if not os.path.exists(depth_path):
                continue
            depth = np.load(depth_path)

            # Apply mask to depth
            depth_masked = depth * mask.astype(np.float32)

            # Get camera intrinsics
            intrinsic = cam_params[f'cam{cam_idx}_intrinsic']
            fx, fy = intrinsic[0, 0], intrinsic[1, 1]
            cx, cy = intrinsic[0, 2], intrinsic[1, 2]

            # Deproject to camera-frame pointcloud
            pcd_cam = depth_to_pointcloud(depth_masked, fx, fy, cx, cy)

            if len(pcd_cam) == 0:
                continue

            # Transform to world frame
            extrinsic = cam_params[f'cam{cam_idx}_extrinsic']
            pcd_world = camera_pcd_to_world(pcd_cam, extrinsic)
            cam_pcds.append(pcd_world)

        if len(cam_pcds) == 0:
            print(f'  Frame {frame_idx}: no valid pointcloud')
            pcd_merged = np.zeros((0, 3), dtype=np.float32)
        else:
            # Merge pointclouds from all cameras
            pcd_merged = np.concatenate(cam_pcds, axis=0)

            # Downsample
            if args.voxel_size > 0 and len(pcd_merged) > 0:
                pcd_merged = downsample_pcd(pcd_merged, voxel_size=args.voxel_size)

        # Transform from world frame to EE (gripper_tcp) frame
        ee_pose_path = os.path.join(ee_pose_dir, f'{frame_idx:04d}.npy')
        if not os.path.exists(ee_pose_path):
            print(f'  Frame {frame_idx}: EE pose not found, skipping')
            continue
        T_w_e = np.load(ee_pose_path)  # (4, 4) world-to-EE pose
        if len(pcd_merged) > 0:
            pcd_merged = transform_pcd(pcd_merged, np.linalg.inv(T_w_e))

        # Save per-frame pointcloud (in EE frame)
        np.save(os.path.join(pcd_dir, f'{frame_idx:04d}.npy'), pcd_merged.astype(np.float32))

        if frame_idx % 50 == 0:
            print(f'  Frame {frame_idx}/{num_frames}: {len(pcd_merged)} points')

    print(f'\nDone! Saved {num_frames} segmented pointclouds to {pcd_dir}/')


if __name__ == '__main__':
    main()
