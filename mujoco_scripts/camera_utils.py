"""Shared camera metadata helpers for MuJoCo data collection scripts."""

import os
import re

import numpy as np


DEFAULT_CAMERA_NAMES = (
    'cam_left',
    'cam_right',
    'cam_front',
    'cam_wrist',
)


def camera_entries_from_names(camera_names):
    """Build ordered camera metadata entries from a list of camera names."""
    return [
        {
            'index': idx,
            'dir': f'cam{idx}',
            'name': cam_name,
        }
        for idx, cam_name in enumerate(camera_names)
    ]


def _camera_indices_from_npz_keys(cam_params):
    indices = []
    for key in cam_params.files:
        match = re.fullmatch(r'cam(\d+)_intrinsic', key)
        if match:
            indices.append(int(match.group(1)))
    return sorted(indices)


def _camera_index_from_dir(cam_dir):
    match = re.fullmatch(r'cam(\d+)', cam_dir)
    if match is None:
        raise ValueError(f'Invalid camera directory name: {cam_dir}')
    return int(match.group(1))


def load_camera_entries(data_dir, rgbd_dir=None):
    """Load ordered camera metadata from saved params or RGBD directories."""
    cam_params_path = os.path.join(data_dir, 'camera_params.npz')
    if os.path.exists(cam_params_path):
        with np.load(cam_params_path, allow_pickle=True) as cam_params:
            indices = _camera_indices_from_npz_keys(cam_params)
            if not indices:
                return []

            if 'camera_names' in cam_params.files:
                camera_names = [str(name) for name in cam_params['camera_names'].tolist()]
            else:
                camera_names = [
                    DEFAULT_CAMERA_NAMES[idx] if idx < len(DEFAULT_CAMERA_NAMES) else f'camera_{idx}'
                    for idx in indices
                ]

            if 'camera_dirs' in cam_params.files:
                camera_dirs = [str(cam_dir) for cam_dir in cam_params['camera_dirs'].tolist()]
            else:
                camera_dirs = [f'cam{idx}' for idx in indices]

        return [
            {
                'index': _camera_index_from_dir(cam_dir),
                'dir': cam_dir,
                'name': cam_name,
            }
            for cam_dir, cam_name in zip(camera_dirs, camera_names)
        ]

    if rgbd_dir is None:
        rgbd_dir = os.path.join(data_dir, 'RGBD_images')
    if not os.path.isdir(rgbd_dir):
        return camera_entries_from_names(DEFAULT_CAMERA_NAMES)

    camera_entries = []
    for cam_dir in os.listdir(rgbd_dir):
        full_path = os.path.join(rgbd_dir, cam_dir)
        if not os.path.isdir(full_path) or re.fullmatch(r'cam\d+', cam_dir) is None:
            continue

        cam_idx = _camera_index_from_dir(cam_dir)
        cam_name = DEFAULT_CAMERA_NAMES[cam_idx] if cam_idx < len(DEFAULT_CAMERA_NAMES) else cam_dir
        camera_entries.append({
            'index': cam_idx,
            'dir': cam_dir,
            'name': cam_name,
        })

    camera_entries.sort(key=lambda camera: camera['index'])
    return camera_entries
