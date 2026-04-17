# Instant Policy

Code for deploying a pre-trained model on Linux machines from the paper: "Instant Policy: In-Context Imitation Learning via Graph Diffusion", 
[Project Webpage](https://www.robot-learning.uk/instant-policy)

<p align="center">
<img src="./media/rollout_roll.gif" alt="drawing" width="700"/>
</p>

## Setup

**Clone this repo**

```
git clone https://github.com/vv19/instant_policy.git
cd instant_policy
```

**Create conda environment**

```
conda env create -f environment.yml
conda activate ip_env
pip install pyg-lib -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
pip install -e .
```

For simulation deployment, install RLbench by following the instructions in https://github.com/stepjam/RLBench.

## Quick Start

### Try our model for RLBench tasks.

Download pre-trained weights.

```
./download_weights.sh
```

Run inference.

```
python deploy_sim.py \
 --task_name='plate_out' \
 --num_demos=2 \
 --num_rollouts=10
```

Try it out with different tasks, e.g. `open_box` or `toilet_seat_down`! More in `sim_utils.py`.

## Deploy on Your Robot

Every robot (and its user) uses different controllers and gets observations in different ways. In `deploy.py`, we provide examples of how to use Instant Policy for deployment on any robotic manipulator using parallel-jaw gripper.
- Collect demonstration in a form of `demo = {'pcds': [], 'T_w_es': [], 'grips': []}`, where `pcds` is a list of numpy arrays prepresenting the segmented point clouds expressed in the world frame, `T_w_es` is a list of numpy arrays (4, 4) representing poses of the gripper in the world frame and `grips` is a list of integers (0 for closed, 1 for open).
- Process the demonstrations using `sample_to_cond_demo(demo, num_traj_wp=10)`.
- Capture current observations of T_w_e, pcd_w, and grip.
- Query the model to obtain end-effector actions `actions, grips = model.predict_actions(...)`
- Plug in your controller and you are all set!

For more please see `deploy.py`.

Note that in our experiments, we found that two depth cameras placed around the scene worked well.

## MuJoCo Pipeline

The MuJoCo workflow is centered around `mujoco_scripts/run_pipeline.sh`.

Example:

```bash
bash mujoco_scripts/run_pipeline.sh --num-demos 1 --object box --sam2
```

Useful options:

- `--sam2`: use SAM2 for mask generation and deployment-time segmentation. If omitted, MuJoCo ground-truth masks are used.
- `--teleop`: collect demos with phone WebXR teleoperation. If omitted, the default is rule-based demo collection.
- `--num-demos`: number of demos to collect and use for deployment.
- `--object`: object / scene type to run.

The pipeline runs in this order:

```text
demo_generation.py -> gen_seg_pcd.py -> deploy_mujoco.py
```

### Results Layout

Outputs are organized under `results/`, with a separate folder for each object:

```text
results/
└── {object}/
    ├── demo/
    │   └── demo_{i}/
    │       ├── RGBD_images/
    │       ├── mask/
    │       ├── seg_pcd/
    │       ├── T_w_e/
    │       └── gripper_state.npy
    └── live/
        ├── step_000.npy
        ├── step_001.npy
        ├── ...
        ├── T_w_e/
        └── gripper_state.npy
```

`live/step_*.npy` stores the current observation used for predicting the next 8 actions, together with the prediction outputs. In particular, each step file contains the current `T_w_e`, gripper state, segmented pointcloud, and the predicted actions / gripper commands.

### Frame Convention Warning

Be careful with pointcloud frames:

- Demo pointclouds saved in `results/{object}/demo/.../seg_pcd/` are in the world frame.
- Deployment-time inference uses pointclouds transformed into the end-effector frame.

This distinction is important when debugging demos, live observations, or saved `step_*.npy` files.

## Notes on Observed Performance

To reach the best performance when deploying the current implementation of Instant Policy, there is a number of things to consider:

- Objects of interest should be well segmented.
- Tasks should follow Markovian assumption (there is no history of observations).
- Demonstrations should be short and consistent, without a lot of task irrelevant motions.
- Inference parameters (e.g. number of demonstrations and number of diffusion timesteps) can greatly influence the performance.
- Model uses segmented point clouds expressed in the end-effector frame -- when an object is grasped, there needs to be at least one more object in the observation to ground the motion.
- To avoid gripper states (open/close) oscillating, execute actions from the prediction horizon until gripper state changes and query the model again.

# Citing

If you find our paper interesting or useful in your work, please cite our paper:

```
@inproceedings{vosylius2024instant,
  title={Instant Policy: In-Context Imitation Learning via Graph Diffusion},
  author={Vosylius, Vitalis and Johns, Edward},
  booktitle = {Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2025}
}
```
