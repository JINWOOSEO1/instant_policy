#!/bin/bash
#
# Full MuJoCo Instant Policy pipeline:
#   1. Collect demo + generate masks (demo_generation.py)
#   2. Generate segmented pointclouds
#   (Steps 1-2 repeat for each demo)
#   3. Deploy Instant Policy (loading demos directly from results/)
#
# Usage:
#   bash mujoco_scripts/run_pipeline.sh [OBJECT] [--num-demos 1] [--skip-to STEP]
#   bash mujoco_scripts/run_pipeline.sh [--object OBJECT] [--num-demos 1] [--skip-to STEP]
#
# Options:
#   OBJECT                Object name (default: mug), loads asset/{NAME}.xml
#   --object NAME         Object name override (same as positional OBJECT)
#   --num-demos N         Number of demos to collect (default: 1)
#   --fps FPS             Recording frame rate (default: 25)
#   --max-frames N        Max frames to record (default: 2000)
#   --teleop              Use teleoperation for demo collection (default: rule-based)
#   --sam2                Use SAM2 for mask generation instead of MuJoCo GT seg_renderer
#   --sam2-config PATH    SAM2 config (default: configs/sam2.1/sam2.1_hiera_s.yaml)
#   --sam2-ckpt PATH      SAM2 checkpoint (default: sam2_repo/checkpoints/sam2.1_hiera_small.pt)
#   --voxel-size SIZE     Voxel size for downsampling (default: 0.005)
#   --execution-horizon N Actions per inference step (default: 8)
#   --skip-to STEP        Skip to step: collect, pcd, deploy (default: collect)

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
USE_TELEOP=0
USE_SAM2=0
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
        --teleop)            USE_TELEOP=1;            shift ;;
        --sam2)              USE_SAM2=1;              shift ;;
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
STEPS=(collect pcd deploy)

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
echo "  Demo mode: $([ $USE_TELEOP -eq 1 ] && echo 'Teleop' || echo 'Rule-based')"
echo "  Mask mode: $([ $USE_SAM2 -eq 1 ] && echo 'SAM2' || echo 'GT (object-mapped geoms)')"
echo "  Starting from: $SKIP_TO"
echo "============================================================"
echo ""

RESULT_DIR="results/${OBJECT}"
DEMO_DIR="${RESULT_DIR}/demo"
DEMO_POSE_SUBDIR="T_w_e"
LIVE_DIR="${RESULT_DIR}/live"
LIVE_POSE_DIR="${LIVE_DIR}/T_w_e"
COLLECT_ARGS=(--object "$OBJECT" --fps "$FPS" --max_frames "$MAX_FRAMES")

if [[ $USE_TELEOP -eq 1 ]]; then
    COLLECT_ARGS+=(--teleop)
fi

if [[ $USE_SAM2 -eq 1 ]]; then
    COLLECT_ARGS+=(--sam2 --sam2_config "$SAM2_CONFIG" --sam2_ckpt "$SAM2_CKPT")
fi

DEPLOY_ARGS=(
    --object "$OBJECT"
    --num_demos "$NUM_DEMOS"
    --execution_horizon "$EXECUTION_HORIZON"
)

if [[ $USE_SAM2 -eq 1 ]]; then
    DEPLOY_ARGS+=(--sam2 --sam2_config "$SAM2_CONFIG" --sam2_ckpt "$SAM2_CKPT")
fi

# ─── Clean previous results (including old demo files) ────────────────────────
echo "  Cleaning previous results in ${RESULT_DIR} ..."
rm -rf "${DEMO_DIR}"/demo_*
for subdir in "${DEMO_POSE_SUBDIR}" mask RGBD_images seg_pcd gripper_state; do
    if [[ -d "${DEMO_DIR}/${subdir}" ]]; then
        rm -rf "${DEMO_DIR}/${subdir:?}"
    fi
done
rm -f "${DEMO_DIR}/gripper_state.npy"

rm -f "${LIVE_DIR}"/step_*.npy
if [[ -d "${LIVE_POSE_DIR}" ]]; then
    rm -rf "${LIVE_POSE_DIR:?}"/*
fi
rm -f "${LIVE_DIR}/gripper_state.npy"
echo "  Done."
echo ""

# ─── Demo collection loop (Steps 1-3 repeated per demo) ──────────────────────
for demo_idx in $(seq 0 $((NUM_DEMOS - 1))); do
    echo "************************************************************"
    echo "  Demo $((demo_idx + 1)) / $NUM_DEMOS"
    echo "************************************************************"
    echo ""

    # ─── Step 1: Collect demo + generate masks ──────────────────────────
    if run_step "collect"; then
        echo "──────────────────────────────────────────────────────────"
        echo "  Step 1/3: Collect demo + generate masks"
        echo "──────────────────────────────────────────────────────────"
        python mujoco_scripts/demo_generation.py "${COLLECT_ARGS[@]}" --demo_index "$demo_idx"
        echo ""
        echo "  Demo collection complete."
        echo ""
    fi

    # ─── Step 2: Generate segmented pointclouds ──────────────────────────
    if run_step "pcd"; then
        echo "──────────────────────────────────────────────────────────"
        echo "  Step 2/3: Generate segmented pointclouds"
        echo "──────────────────────────────────────────────────────────"
        python mujoco_scripts/gen_seg_pcd.py \
            --object "$OBJECT" \
            --demo_index "$demo_idx" \
            --voxel_size "$VOXEL_SIZE"
        echo ""
        echo "  Pointcloud generation complete."
        echo ""
    fi

done

# ─── Step 3: Deploy ─────────────────────────────────────────────────────────
if run_step "deploy"; then
    echo "──────────────────────────────────────────────────────────"
    echo "  Step 3/3: Deploy Instant Policy"
    echo "──────────────────────────────────────────────────────────"
    python mujoco_scripts/deploy_mujoco.py "${DEPLOY_ARGS[@]}"
    echo ""
    echo "  Deployment complete."
    echo ""
fi

echo "============================================================"
echo "  Pipeline finished."
echo "============================================================"
