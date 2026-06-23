#!/usr/bin/env python3
"""Batch runner: deploy_mujoco_baseline at random kettle positions, recording videos.

Each run samples:
  x ~ Uniform[x_min, x_max]
  y ~ Uniform[y_min, y_max]

Then calls deploy_mujoco_baseline with --object-x, --object-y, --headless,
and --record-video. Videos are saved as:

  results/video/{run_idx}_scene_standalone_x{x}_y{y}.mp4

A CSV manifest is written alongside the videos. Run with --dry-run first
to preview the commands without executing them.
"""

from __future__ import annotations

import argparse
import csv
import random
import subprocess
import sys
import time
from pathlib import Path


OBJECT = "scene_standalone"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEPLOY_MODULE = "mujoco_scripts.deploy_mujoco_baseline"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "video"
DEFAULT_RECORD_CAMERA = "cam_front"
DEFAULT_RECORD_CAMERA_POS = (1.296, -0.584, 0.749)
DEFAULT_RECORD_CAMERA_XYAXES = (0.511, 0.859, -0.000, -0.383, 0.228, 0.895)


def _position_token(value: float, decimals: int) -> str:
    rounded = round(float(value), decimals)
    if rounded == 0.0:
        rounded = 0.0
    return f"{rounded:.{decimals}f}"


def _video_path(
    output_dir: Path,
    run_idx: int,
    x: float,
    y: float,
    decimals: int,
) -> Path:
    return output_dir / (
        f"{run_idx:03d}_{OBJECT}_x{_position_token(x, decimals)}_"
        f"y{_position_token(y, decimals)}.mp4"
    )


def _sample_unused_position(
    rng: random.Random,
    *,
    run_idx: int,
    output_dir: Path,
    decimals: int,
    overwrite: bool,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> tuple[float, float, Path]:
    for _ in range(1000):
        x = rng.uniform(x_min, x_max)
        y = rng.uniform(y_min, y_max)
        path = _video_path(output_dir, run_idx, x, y, decimals)
        if overwrite or not path.exists():
            return x, y, path
    raise RuntimeError(
        "Could not sample a unique video filename after 1000 tries. "
        "Use --overwrite or increase --decimals."
    )


def _append_manifest(
    manifest_path: Path,
    *,
    run_idx: int,
    x: float,
    y: float,
    video_path: Path,
    returncode: int,
    elapsed_sec: float,
) -> None:
    exists = manifest_path.exists()
    with manifest_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow([
                "run", "object", "object_x", "object_y",
                "video_path", "returncode", "elapsed_sec",
            ])
        writer.writerow([
            run_idx,
            OBJECT,
            f"{x:.8f}",
            f"{y:.8f}",
            str(video_path),
            returncode,
            f"{elapsed_sec:.3f}",
        ])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-n", "--runs", type=int, required=True,
                        help="Number of random runs to execute")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible positions")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Directory for MP4 files and manifest CSV")
    parser.add_argument("--x-min", type=float, default=0.25,
                        help="Minimum kettle x position (default: 0.25)")
    parser.add_argument("--x-max", type=float, default=0.35,
                        help="Maximum kettle x position (default:  0.35)")
    parser.add_argument("--y-min", type=float, default=-0.15,
                        help="Minimum kettle y position (default: -0.15)")
    parser.add_argument("--y-max", type=float, default=0.15,
                        help="Maximum kettle y position (default:  0.15)")
    parser.add_argument("--decimals", type=int, default=4,
                        help="Decimal places used in the video filename")
    parser.add_argument("--num_demos", type=int, default=1,
                        help="Number of demos to load (passed to deploy script)")
    parser.add_argument("--execution_horizon", type=int, default=8,
                        help="Execution horizon (passed to deploy script)")
    parser.add_argument("--record-camera", type=str, default=DEFAULT_RECORD_CAMERA,
                        help="Camera name for offscreen recording")
    parser.add_argument("--record-camera-pos", type=float, nargs=3,
                        default=DEFAULT_RECORD_CAMERA_POS,
                        metavar=("X", "Y", "Z"),
                        help="World-frame camera position for offscreen recording")
    parser.add_argument("--record-camera-xyaxes", type=float, nargs=6,
                        default=DEFAULT_RECORD_CAMERA_XYAXES,
                        metavar=("X0", "X1", "X2", "Y0", "Y1", "Y2"),
                        help="World-frame camera orientation for offscreen recording")
    parser.add_argument("--record-fps", type=float, default=30.0,
                        help="Video frame rate")
    parser.add_argument("--record-width", type=int, default=640)
    parser.add_argument("--record-height", type=int, default=480)
    parser.add_argument("--max-duration", type=float, default=100.0,
                        help="Maximum video duration per run in seconds")
    parser.add_argument("--timeout-grace", type=float, default=120.0,
                        help="Extra seconds before killing a child run that exceeds max-duration")
    parser.add_argument("--python", default=sys.executable,
                        help="Python executable used for child runs")
    parser.add_argument("--no-sam2", action="store_true",
                        help="Use MuJoCo GT masks instead of SAM2 (default: SAM2)")
    parser.add_argument("--with-viewer", action="store_true",
                        help="Show passive viewer during each run (default: headless)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing MP4 files")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Keep running even if a child run fails")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing them")
    args, extra_args = parser.parse_known_args()

    if args.runs <= 0:
        parser.error("--runs must be positive")
    if args.x_min > args.x_max:
        parser.error("--x-min must be <= --x-max")
    if args.y_min > args.y_max:
        parser.error("--y-min must be <= --y-max")
    if args.decimals < 0:
        parser.error("--decimals must be >= 0")
    if args.max_duration <= 0:
        parser.error("--max-duration must be positive")
    if args.timeout_grace < 0:
        parser.error("--timeout-grace must be non-negative")

    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"batch_{OBJECT}_runs.csv"

    print(f"[Batch] object={OBJECT}")
    print(f"[Batch] x in [{args.x_min}, {args.x_max}], y in [{args.y_min}, {args.y_max}]")
    print(f"[Batch] max_duration={args.max_duration}s  record_fps={args.record_fps}")
    print(f"[Batch] videos      -> {output_dir}")
    print(f"[Batch] manifest    -> {manifest_path}")
    if extra_args:
        print(f"[Batch] extra args: {' '.join(extra_args)}")

    for run_idx in range(1, args.runs + 1):
        x, y, video_path = _sample_unused_position(
            rng,
            run_idx=run_idx,
            output_dir=output_dir,
            decimals=args.decimals,
            overwrite=args.overwrite,
            x_min=args.x_min,
            x_max=args.x_max,
            y_min=args.y_min,
            y_max=args.y_max,
        )

        cmd = [
            args.python, "-m", DEPLOY_MODULE,
            "--object", OBJECT,
            "--object-x", f"{x:.8f}",
            "--object-y", f"{y:.8f}",
            "--record-video", str(video_path),
            "--record-camera", args.record_camera,
            "--record-camera-pos", *(f"{value:g}" for value in args.record_camera_pos),
            "--record-camera-xyaxes", *(f"{value:g}" for value in args.record_camera_xyaxes),
            "--record-fps", f"{args.record_fps:g}",
            "--record-width", str(args.record_width),
            "--record-height", str(args.record_height),
            "--max-duration", f"{args.max_duration:g}",
            "--num_demos", str(args.num_demos),
            "--execution_horizon", str(args.execution_horizon),
        ]
        if not args.with_viewer:
            cmd.append("--headless")
        if not args.no_sam2:
            cmd.append("--sam2")
        cmd.extend(extra_args)

        print(
            f"\n[Batch] run {run_idx}/{args.runs}: "
            f"x={x:.4f}, y={y:.4f} -> {video_path.name}"
        )
        print("[Batch] command:", " ".join(cmd))

        if args.dry_run:
            continue

        start = time.monotonic()
        timeout = args.max_duration + args.timeout_grace
        timed_out = False
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT))
        try:
            returncode = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            print(
                f"[Batch] run {run_idx} exceeded hard timeout "
                f"({timeout:.1f}s); terminating child process..."
            )
            proc.terminate()
            try:
                returncode = proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                print(f"[Batch] run {run_idx} did not terminate; killing child process.")
                proc.kill()
                returncode = proc.wait()
        elapsed = time.monotonic() - start

        _append_manifest(
            manifest_path,
            run_idx=run_idx,
            x=x,
            y=y,
            video_path=video_path,
            returncode=returncode,
            elapsed_sec=elapsed,
        )

        if returncode != 0:
            print(
                f"[Batch] run {run_idx} failed with return code {returncode}"
            )
            if timed_out:
                print(f"[Batch] run {run_idx} timed out after {elapsed:.1f}s")
            if not args.continue_on_error:
                return returncode

    print("\n[Batch] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
