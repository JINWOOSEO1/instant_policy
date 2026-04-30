# Instant Policy

Deployment and simulation utilities for **Instant Policy: In-Context Imitation Learning via Graph Diffusion**.

Original project page: <https://www.robot-learning.uk/instant-policy>

<p align="center">
  <img src="./media/rollout_roll.gif" alt="Instant Policy rollout" width="700"/>
</p>

## Repository Layout

```text
.
|-- deploy.py                  # Robot deployment template
|-- deploy_sim.py              # RLBench evaluation entry point
|-- instant_policy.so          # Prebuilt Instant Policy extension
|-- instant_policy.pyi         # Type hints for the extension
|-- sim_utils.py               # RLBench helpers
|-- utils.py                   # Shared point-cloud / pose utilities
|-- asset/                     # MuJoCo scene XMLs and task assets
|-- media/                     # README media
`-- mujoco_scripts/            # MuJoCo collection, segmentation, and deployment
```

Large local dependencies and generated outputs are intentionally ignored:

```text
checkpoints/  results/  PyRep/  RLBench/  sam2_repo/  asset/panda.xml  asset/panda_assets/
```

## Setup

Create the conda environment and install the extra PyG wheel:

```bash
conda env create -f environment.yml
conda activate ip_env
pip install pyg-lib -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

Download the pretrained weights:

```bash
./download_weights.sh
```

The scripts expect weights at `checkpoints/model.pt`.

## RLBench Evaluation

Install RLBench and PyRep following the upstream RLBench instructions, then keep those checkouts at the repository root as `RLBench/` and `PyRep/`.

Run an evaluation:

```bash
python deploy_sim.py \
  --task_name plate_out \
  --num_demos 2 \
  --num_rollouts 10
```

Try other supported task names from `sim_utils.py`, such as `open_box` or `toilet_seat_down`.

## MuJoCo Pipeline

MuJoCo assets under `asset/*.xml` include task scenes. The Panda robot XML and meshes are local dependencies and should be placed at:

```text
asset/panda.xml
asset/panda_assets/
```

Run the full MuJoCo pipeline:

```bash
bash mujoco_scripts/run_pipeline.sh --object mug_0 --num-demos 1
```

Available object scenes include `box`, `mug_0`, `mug_1`, `mug_2`, `mug_3`, `mug_3branch`, and `mug_4`.

Useful options:

- `--sam2`: use SAM2 for mask generation and deployment-time segmentation. Without this flag, MuJoCo ground-truth masks are used.
- `--teleop`: collect demos with phone WebXR teleoperation. Without this flag, rule-based demo collection is used.
- `--num-demos`: number of demos to collect and use for deployment.
- `--skip-to collect|pcd|deploy`: resume from a later stage.

The pipeline runs:

```text
demo_generation.py -> gen_seg_pcd.py -> deploy_mujoco.py
```

Outputs are written under `results/{object}/`, which is ignored by git.

### Optional SAM2

For SAM2 segmentation, clone SAM2 locally into `sam2_repo/` and install its dependencies:

```bash
git clone https://github.com/facebookresearch/sam2.git sam2_repo
pip install -e sam2_repo
```

The default checkpoint path is:

```text
sam2_repo/checkpoints/sam2.1_hiera_small.pt
```

## Deploy On Your Robot

`deploy.py` shows the expected integration points for a real robot:

- Collect demonstrations as `{"pcds": [], "T_w_es": [], "grips": []}`.
- Convert each demo with `sample_to_cond_demo(demo, 10)`.
- Capture the current end-effector pose, segmented point cloud, and gripper state.
- Query `model.predict_actions(...)`.
- Send the predicted relative end-effector transforms and gripper commands to your controller.

In our experiments, two depth cameras placed around the scene worked well.

## Frame Convention

Demo point clouds saved in `results/{object}/demo/.../seg_pcd/` are in the world frame. Deployment-time inference transforms point clouds into the end-effector frame before querying the model. Keep this distinction in mind when debugging saved demos, live observations, or `step_*.npy` files.

## Citation

If you find Instant Policy useful, please cite:

```bibtex
@inproceedings{vosylius2024instant,
  title={Instant Policy: In-Context Imitation Learning via Graph Diffusion},
  author={Vosylius, Vitalis and Johns, Edward},
  booktitle={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2025}
}
```
