import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.spawn_boundary import BoundingBox
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *
from tqdm import tqdm, trange

from instant_policy import sample_to_cond_demo
from utils import pose_to_transform, subsample_pcd, transform_pcd, transform_to_pose


# Some examples of RLBench tasks
TASK_NAMES = {
    'lift_lid': TakeLidOffSaucepan,
    'phone_on_base': PhoneOnBase,
    'open_box': OpenBox,
    'slide_block': SlideBlockToTarget,
    'close_box': CloseBox,
    'basketball': BasketballInHoop,
    'buzz': BeatTheBuzz,
    'close_microwave': CloseMicrowave,
    'plate_out': TakePlateOffColoredDishRack,
    'toilet_seat_down': ToiletSeatDown,
    'toilet_seat_up': ToiletSeatUp,
    'toilet_roll_off': TakeToiletRollOffStand,
    'open_microwave': OpenMicrowave,
    'lamp_on': LampOn,
    'umbrella_out': TakeUmbrellaOutOfUmbrellaStand,
    'push_button': PushButton,
    'put_rubbish': PutRubbishInBin,
}


def override_bounds(pos, rot, env):
    if pos is not None:
        BoundingBox.within_boundary = lambda x, y, z: True  # Where we are going, we don't need boundaries
        env._scene._workspace_boundary._boundaries[0]._get_position_within_boundary = lambda x, y: pos
    env._scene.task.base_rotation_bounds = lambda: ((0.0, 0.0, rot - 0.0001), (0.0, 0.0, rot + 0.0001))


def rl_bench_demo_to_sample(demo):
    sample = {'pcds': [], 'T_w_es': [], 'grips': []}

    for k, obs in enumerate(demo):
        pcd = get_point_cloud(obs)
        sample['pcds'].append(pcd)
        sample['T_w_es'].append(pose_to_transform(obs.gripper_pose))
        sample['grips'].append(obs.gripper_open)

    return sample


def get_point_cloud(obs, camera_names=('front', 'left_shoulder', 'right_shoulder')):
    pcds = []
    for camera_name in camera_names:
        ordered_pcd = getattr(obs, f'{camera_name}_point_cloud')
        mask = getattr(obs, f'{camera_name}_mask')
        masked_pcd = ordered_pcd[mask > 60]  # Hack to get segmentations easily.
        pcds.append(masked_pcd)

    return np.concatenate(pcds, axis=0)


def create_sim_env(task_name, headless=False, restrict_rot=True):
    obs_config = ObservationConfig()
    obs_config.set_all(True)
    action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaIK(),
        gripper_action_mode=Discrete()
    )
    env = Environment(action_mode,
                      './',
                      obs_config=obs_config,
                      headless=headless)
    env.launch()
    task = env.get_task(TASK_NAMES[task_name])

    def temp(position, euler=None, quaternion=None, ignore_collisions=False, trials=300, max_configs=1,
             distance_threshold=0.65, max_time_ms=10, trials_per_goal=1, algorithm=None, relative_to=None):
        return env._robot.arm.get_linear_path(position, euler, quaternion, ignore_collisions=ignore_collisions,
                                              relative_to=relative_to)

    env._robot.arm.get_path = temp
    env._scene._start_arm_joint_pos = np.array([6.74760377e-05, -1.91104114e-02, -3.62065766e-05, -1.64271665e+00,
                                                -1.14094291e-07, 1.55336857e+00, 7.85427451e-01])

    rot_bounds = env._scene.task.base_rotation_bounds()
    mean_rot = (rot_bounds[0][2] + rot_bounds[1][2]) / 2
    if restrict_rot:
        env._scene.task.base_rotation_bounds = lambda: ((0.0, 0.0, max(rot_bounds[0][2], mean_rot - np.pi / 3)),
                                                        (0.0, 0.0, min(rot_bounds[1][2], mean_rot + np.pi / 3)))
    return env, task


def rollout_model(model, num_demos, task_name='phone_on_base', max_execution_steps=30,
                  execution_horizon=8, num_rollouts=2, headless=False, num_traj_wp=10, restrict_rot=True):
    ####################################################################################################################
    env, task = create_sim_env(task_name, headless=headless, restrict_rot=restrict_rot)
    ####################################################################################################################
    full_sample = {
        'demos': [dict()] * num_demos,
        'live': dict(),
    }
    for i in tqdm(range(num_demos), desc='Collecting demos', total=num_demos, leave=False):
        done = False
        while not done:
            try:
                demos = task.get_demos(1, live_demos=True, max_attempts=1000)  # -> List[List[Observation]]
                sample = rl_bench_demo_to_sample(demos[0])
                full_sample['demos'][i] = sample_to_cond_demo(sample, num_traj_wp)
                assert len(full_sample['demos'][i]['obs']) == num_traj_wp
                done = True
            except Exception:
                continue
    ####################################################################################################################
    successes = []
    pbar = trange(num_rollouts, desc=f'Evaluating model, SR: 0/{num_rollouts}', leave=False)
    for i in pbar:
        done = False
        while not done:
            try:
                task.reset()
                done = True
            except Exception:
                continue

        env_action = np.zeros(8)
        # number of steps in rollouts.
        success = 0
        for k in range(max_execution_steps):
            curr_obs = task.get_observation()
            T_w_e = pose_to_transform(curr_obs.gripper_pose)
            full_sample['live']['obs'] = [transform_pcd(subsample_pcd(get_point_cloud(curr_obs)),
                                                        np.linalg.inv(T_w_e))]
            full_sample['live']['grips'] = [curr_obs.gripper_open]
            full_sample['live']['T_w_es'] = [T_w_e]

            actions, grips = model.predict_actions(full_sample)

            for j in range(execution_horizon):
                env_action[:7] = transform_to_pose(T_w_e @ actions[j])
                env_action[7] = int((grips[j] + 1) / 2 > 0.5)
                try:
                    curr_obs, reward, terminate = task.step(env_action)
                    success = int(terminate and reward > 0.)
                except Exception:
                    terminate = True
                if terminate:
                    break

            else:
                continue
            break
        successes.append(success)
        pbar.set_description(f'Evaluating model, SR: {sum(successes)}/{len(successes)}')
        pbar.refresh()
    pbar.close()
    env.shutdown()
    return sum(successes) / len(successes)
    ####################################################################################################################
