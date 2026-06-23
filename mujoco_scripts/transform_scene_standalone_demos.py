"""
One-shot script: re-express scene_standalone demo data in robot-base frame.

Old world frame: MuJoCo world origin (floor level, table centre).
New world frame: robot base (panda_base body).

Transform T_base_w:
  R = R_z(-pi/2) = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
  t = [0.35, 0, -0.42]   (= -R @ robot_base_pos_old = -R @ [0, -0.35, 0.42])

Applied to:
  - seg_pcd/*.npy  (N,3) world-frame pointclouds
  - T_w_e/*.npy    (4,4) world-frame EE poses
"""

import os
import glob
import numpy as np

# ── Transform parameters ──────────────────────────────────────────────────────
# Panda base in old world frame: pos=(0, -0.35, 0.42), euler=(0, 0, pi/2)
R_base_w = np.array([
    [ 0,  1,  0],
    [-1,  0,  0],
    [ 0,  0,  1],
], dtype=np.float64)
t_base_w = np.array([0.35, 0.0, -0.42], dtype=np.float64)

T_base_w = np.eye(4, dtype=np.float64)
T_base_w[:3, :3] = R_base_w
T_base_w[:3,  3] = t_base_w


def transform_pcd(pcd: np.ndarray) -> np.ndarray:
    """Transform (N,3) pointcloud from old world frame to robot-base frame."""
    return (R_base_w @ pcd.T).T + t_base_w


def transform_T_w_e(T: np.ndarray) -> np.ndarray:
    """Transform 4x4 SE3 pose from old world frame to robot-base frame."""
    return T_base_w @ T


def process_demo(demo_dir: str):
    seg_pcd_dir = os.path.join(demo_dir, "seg_pcd")
    T_w_e_dir   = os.path.join(demo_dir, "T_w_e")

    pcd_files  = sorted(glob.glob(os.path.join(seg_pcd_dir, "*.npy")))
    pose_files = sorted(glob.glob(os.path.join(T_w_e_dir,   "*.npy")))

    print(f"  seg_pcd: {len(pcd_files)} files")
    for path in pcd_files:
        pcd = np.load(path)
        np.save(path, transform_pcd(pcd).astype(pcd.dtype))

    print(f"  T_w_e:   {len(pose_files)} files")
    for path in pose_files:
        T = np.load(path)
        np.save(path, transform_T_w_e(T).astype(T.dtype))


def main():
    results_root = os.path.join(
        os.path.dirname(__file__), "..", "results", "scene_standalone", "demo"
    )
    results_root = os.path.normpath(results_root)

    demo_dirs = sorted(
        d for d in glob.glob(os.path.join(results_root, "demo_*"))
        if os.path.isdir(d)
    )

    if not demo_dirs:
        print(f"No demo directories found under {results_root}")
        return

    print(f"Found {len(demo_dirs)} demo(s) under {results_root}")
    for demo_dir in demo_dirs:
        print(f"\nProcessing {os.path.basename(demo_dir)} ...")
        process_demo(demo_dir)

    print("\nDone. All demo data re-expressed in robot-base frame.")


if __name__ == "__main__":
    main()
