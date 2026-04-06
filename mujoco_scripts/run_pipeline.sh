#!/bin/bash
#
# Full MuJoCo Instant Policy pipeline:
#   1. Collect demo via WebXR teleoperation
#   2. Generate segmentation masks (SAM2)
#   3. Generate segmented pointclouds
#   (Steps 1-3 repeat for each demo)
#   4. Deploy Instant Policy (using all collected demos)
#
# Usage:
#   bash mujoco_scripts/run_pipeline.sh [OBJECT] [--num-demos 1] [--skip-to STEP]
#   bash mujoco_scripts/run_pipeline.sh [--object OBJECT] [--num-demos 1] [--skip-to STEP]
#
# Options:
#   OBJECT                Object name (default: mug), loads asset/{NAME}.xml
#   --object NAME         Object name override (same as positional OBJECT)
#   --num-demos N         Number of demos to collect (default: 1)
#   --fps FPS             Recording frame rate (default: 10)
#   --max-frames N        Max frames to record (default: 1000)
#   --sam2-config PATH    SAM2 config (default: configs/sam2.1/sam2.1_hiera_s.yaml)
#   --sam2-ckpt PATH      SAM2 checkpoint (default: sam2_repo/checkpoints/sam2.1_hiera_small.pt)
#   --voxel-size SIZE     Voxel size for downsampling (default: 0.005)
#   --execution-horizon N Actions per inference step (default: 8)
#   --skip-to STEP        Skip to step: collect, mask, pcd, deploy (default: collect)

set -euo pipefail

# ─── Project root ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Ensure both project root and mujoco_scripts are importable
export PYTHONPATH="${PROJECT_ROOT}:${SCRIPT_DIR}:${PYTHONPATH:-}"

# ─── Defaults ────────────────────────────────────────────────────────────────
OBJECT="mug"
NUM_DEMOS=1
FPS=25
MAX_FRAMES=2000
SAM2_CONFIG="configs/sam2.1/sam2.1_hiera_s.yaml"
SAM2_CKPT="sam2_repo/checkpoints/sam2.1_hiera_small.pt"
VOXEL_SIZE=0.005
EXECUTION_HORIZON=8
SKIP_TO="collect"
POSITIONAL_OBJECT_SET=0

# ─── Parse arguments ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --object)            OBJECT="$2";             shift 2 ;;
        --num-demos)         NUM_DEMOS="$2";          shift 2 ;;
        --fps)               FPS="$2";                shift 2 ;;
        --max-frames)        MAX_FRAMES="$2";         shift 2 ;;
        --sam2-config)       SAM2_CONFIG="$2";        shift 2 ;;
        --sam2-ckpt)         SAM2_CKPT="$2";          shift 2 ;;
        --voxel-size)        VOXEL_SIZE="$2";         shift 2 ;;
        --execution-horizon) EXECUTION_HORIZON="$2";  shift 2 ;;
        --skip-to)           SKIP_TO="$2";            shift 2 ;;
        -h|--help)
            sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
            exit 0 ;;
        -*)
            echo "Unknown option: $1"
            exit 1 ;;
        *)
            if [[ $POSITIONAL_OBJECT_SET -eq 1 ]]; then
                echo "Unexpected argument: $1"
                exit 1
            fi
            OBJECT="$1"
            POSITIONAL_OBJECT_SET=1
            shift ;;
    esac
done

# ─── Step ordering ───────────────────────────────────────────────────────────
STEPS=(collect mask pcd deploy)

start_idx=0
for i in "${!STEPS[@]}"; do
    if [[ "${STEPS[$i]}" == "$SKIP_TO" ]]; then
        start_idx=$i
        break
    fi
done

run_step() {
    local step_name="$1"
    for i in "${!STEPS[@]}"; do
        if [[ "${STEPS[$i]}" == "$step_name" && $i -ge $start_idx ]]; then
            return 0
        fi
    done
    return 1
}

echo "============================================================"
echo "  Instant Policy MuJoCo Pipeline"
echo "  Object: $OBJECT"
echo "  Number of demos: $NUM_DEMOS"
echo "  Starting from: $SKIP_TO"
echo "============================================================"
echo ""

RESULT_DIR="results/${OBJECT}"

# ─── Clean previous results (including old demo files) ────────────────────────
echo "  Cleaning previous results in ${RESULT_DIR} ..."
rm -f "${RESULT_DIR}"/demo_*.npy
for subdir in EE_pose mask RGBD_images/cam0 RGBD_images/cam1 RGBD_images/cam2 seg_pcd; do
    if [[ -d "${RESULT_DIR}/${subdir}" ]]; then
        rm -rf "${RESULT_DIR}/${subdir:?}"/*
    fi
done
echo "  Done."
echo ""

# ─── Demo collection loop (Steps 1-3 repeated per demo) ──────────────────────
for demo_idx in $(seq 0 $((NUM_DEMOS - 1))); do
    echo "************************************************************"
    echo "  Demo $((demo_idx + 1)) / $NUM_DEMOS"
    echo "************************************************************"
    echo ""

    # ─── Step 1: Collect demo ────────────────────────────────────────────
    if run_step "collect"; then
        echo "──────────────────────────────────────────────────────────"
        echo "  Step 1/4: Collect demo (WebXR teleoperation)"
        echo "──────────────────────────────────────────────────────────"
        python mujoco_scripts/simulation.py \
            --object "$OBJECT" \
            --fps "$FPS" \
            --max_frames "$MAX_FRAMES"
        echo ""
        echo "  Demo collection complete."
        echo ""
    fi

    # ─── Step 2: Generate masks ──────────────────────────────────────────
    if run_step "mask"; then
        echo "──────────────────────────────────────────────────────────"
        echo "  Step 2/4: Generate segmentation masks (SAM2)"
        echo "──────────────────────────────────────────────────────────"
        python mujoco_scripts/gen_mask.py \
            --object "$OBJECT" \
            --sam2_config "$SAM2_CONFIG" \
            --sam2_ckpt "$SAM2_CKPT"
        echo ""
        echo "  Mask generation complete."
        echo ""
    fi

    # ─── Step 3: Generate segmented pointclouds ──────────────────────────
    if run_step "pcd"; then
        echo "──────────────────────────────────────────────────────────"
        echo "  Step 3/4: Generate segmented pointclouds"
        echo "──────────────────────────────────────────────────────────"
        python mujoco_scripts/gen_seg_pcd.py \
            --object "$OBJECT" \
            --voxel_size "$VOXEL_SIZE"
        echo ""
        echo "  Pointcloud generation complete."
        echo ""
    fi

    # ─── Save consolidated demo file ─────────────────────────────────────
    echo "──────────────────────────────────────────────────────────"
    echo "  Saving demo ${demo_idx} ..."
    echo "──────────────────────────────────────────────────────────"
    python mujoco_scripts/save_demo.py \
        --object "$OBJECT" \
        --demo_index "$demo_idx"
    echo ""

    # ─── Clean intermediate files for next demo ──────────────────────────
    if [[ $demo_idx -lt $((NUM_DEMOS - 1)) ]]; then
        echo "  Cleaning intermediate files for next demo ..."
        for subdir in EE_pose mask RGBD_images/cam0 RGBD_images/cam1 RGBD_images/cam2 seg_pcd; do
            if [[ -d "${RESULT_DIR}/${subdir}" ]]; then
                rm -rf "${RESULT_DIR}/${subdir:?}"/*
            fi
        done
        rm -f "${RESULT_DIR}/gripper_state.npy"
        echo "  Done."
        echo ""
    fi
done

# ─── Step 4: Deploy ─────────────────────────────────────────────────────────
if run_step "deploy"; then
    echo "──────────────────────────────────────────────────────────"
    echo "  Step 4/4: Deploy Instant Policy"
    echo "──────────────────────────────────────────────────────────"
    python mujoco_scripts/deploy_mujoco.py \
        --object "$OBJECT" \
        --num_demos "$NUM_DEMOS" \
        --sam2_config "$SAM2_CONFIG" \
        --sam2_ckpt "$SAM2_CKPT" \
        --execution_horizon "$EXECUTION_HORIZON"
    echo ""
    echo "  Deployment complete."
    echo ""
fi

echo "============================================================"
echo "  Pipeline finished."
echo "============================================================"
