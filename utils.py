import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as Rot


def downsample_pcd(pcd_np, voxel_size=0.01):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np)
    pcd_new = pcd.voxel_down_sample(voxel_size)
    return np.asarray(pcd_new.points)


def pose_to_transform(pose):
    T = np.eye(4)
    T[:3, 3] = pose[:3]
    T[:3, :3] = Rot.from_quat(pose[3:]).as_matrix()
    return T


def transform_to_pose(T):
    pose = np.zeros(7)
    pose[:3] = T[:3, 3]
    pose[3:] = Rot.from_matrix(T[:3, :3]).as_quat(canonical=True)
    return pose


def transform_pcd(pcd, T):
    return np.matmul(T[:3, :3], pcd.T).T + T[:3, 3]


def subsample_pcd(sample, num_points=2048):
    sample = downsample_pcd(sample)
    sample_filtered, _ = remove_statistical_outliers(sample, nb_neighbors=20, std_ratio=2.0)
    if len(sample_filtered) == 0:
        raise ValueError('Cannot subsample an empty point cloud')
    replace = len(sample_filtered) < num_points
    rand_idx = np.random.choice(len(sample_filtered), num_points, replace=replace)
    return sample_filtered[rand_idx]


def remove_statistical_outliers(point_cloud, nb_neighbors=20, std_ratio=2.0):
    # Create a PointCloud object from the NumPy array
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # Perform statistical outlier removal
    [filtered_pcd, inlier_indices] = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)

    # Convert the filtered PointCloud back to a NumPy array
    filtered_point_cloud = np.asarray(filtered_pcd.points)

    return filtered_point_cloud, inlier_indices
