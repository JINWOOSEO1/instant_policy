import argparse

import torch

from instant_policy import GraphDiffusion
from sim_utils import rollout_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default='plate_out')
    parser.add_argument('--num_demos', type=int, default=2)
    parser.add_argument('--num_rollouts', type=int, default=10)
    parser.add_argument('--restrict_rot', type=int, default=1)
    args = parser.parse_args()

    restrict_rot = bool(args.restrict_rot)
    task_name = args.task_name
    num_demos = args.num_demos
    num_rollouts = args.num_rollouts
    ####################################################################################################################
    model_path = './checkpoints'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = GraphDiffusion.load_from_checkpoint(f'{model_path}/model.pt',
                                                device=device,
                                                strict=True,
                                                map_location=device)

    model.set_num_demos(num_demos)
    model.set_num_diffusion_steps(4)
    model.eval()
    ####################################################################################################################
    sr = rollout_model(model, num_demos, task_name, num_rollouts=num_rollouts, execution_horizon=8,
                       num_traj_wp=10, restrict_rot=restrict_rot)
    print('Success rate:', sr)
