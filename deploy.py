"""Example integration points for deploying Instant Policy on a real robot."""

import numpy as np
import torch

from instant_policy import GraphDiffusion, sample_to_cond_demo
from utils import subsample_pcd, transform_pcd


if __name__ == '__main__':
    ####################################################################################################################
    # Define rollout parameters. 
    num_demos = 2
    num_traj_wp = 10
    num_diffusion_iters = 4
    compile_models = False
    max_execution_steps = 100
    ####################################################################################################################
    # Initialise and load the model.
    model_path = './checkpoints'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = GraphDiffusion.load_from_checkpoint(f'{model_path}/model.pt', 
                                                device=device,
                                                strict=True,
                                                map_location=device)

    model.set_num_demos(num_demos)
    model.set_num_diffusion_steps(num_diffusion_iters)
    model.eval()
    ####################################################################################################################
    # Process demonstrations.
    # TODO: Collect or load demonstrations in a form of {'pcds': [], 'T_w_es': [], 'grips': []}
    demos = []    
    full_sample = {
        'demos': [dict()] * num_demos,
        'live': dict(),
    }
    for i, demo in enumerate(demos):
        full_sample['demos'][i] = sample_to_cond_demo(demo, num_traj_wp)
        assert len(full_sample['demos'][i]['obs']) == num_traj_wp
    ####################################################################################################################
    # Rollout the model.
    for k in range(max_execution_steps):
        T_w_e = None  # TODO: end-effector pose in the world frame, [4, 4].
        pcd_w = None  # TODO: segmented point cloud observation in the world frame, [N, 3].
        grip = None  # TODO: whether the gripper is closed or opened, [0, 1]
        full_sample['live']['obs'] = [transform_pcd(subsample_pcd(pcd_w), np.linalg.inv(T_w_e))]
        full_sample['live']['grips'] = [grip]
        full_sample['live']['T_w_es'] = [T_w_e]
        
        # Inference on the model.
        actions, grips = model.predict_actions(full_sample)
        
        # TODO: Use whatever controller you have to execute all or part of the predicted actions.
        # TODO: actions: [Pred_horizon, 4, 4] are relative transforms of the end-effector.
        # TODO: To get next pose of the end-effector in the world frame you use T_w_e @ actions[j].
        # TODO: grips: [Pred_horizon, 1] are open and close commands: -1 is close, 1 is open.
