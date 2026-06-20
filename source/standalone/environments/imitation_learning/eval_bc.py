# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to run a trained Behavior Cloning policy to pick and lift the suture needle.


.. code-block:: bash

    ${IsaacLab_PATH}/isaaclab.sh -p source/standalone/environments/imitation_learning/eval_bc.py

    ~/IsaacLab/isaaclab.sh -p source/standalone/environments/imitation_learning/eval_bc.py \
  --task Isaac-Lift-Needle-PSM-IK-Abs-v0 \
  --checkpoint source/standalone/environments/imitation_learning/policies/bc_lift_n_50_policy_30.zip
"""

"""Launch Omniverse Toolkit first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play policy trained using imitation for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
# parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Pytorch model checkpoint to load.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import gymnasium as gym
import torch

from pathlib import Path
from stable_baselines3.common.policies import ActorCriticPolicy
from isaaclab_tasks.utils import parse_env_cfg
import orbit.surgical.tasks  # noqa: F401


def main():
    """Run a trained policy from robomimic with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)
    # we want to have the terms in the observations returned as a concatenated tensor
    # rather than a dictionary
    env_cfg.observations.policy.concatenate_terms = True

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    IL_DIR = Path(__file__).resolve().parent
    checkpoint = args_cli.checkpoint or IL_DIR / "policies" / "bc_lift_n_50_policy_30.zip"
    traj_path = IL_DIR / "data" / "lift_n_trajs_50" / "lift_n_1_success_ep4.pt"

    # acquire device
    # device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    # restore policy
    # policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=args_cli.checkpoint, device=device, verbose=True)
    policy = ActorCriticPolicy.load(checkpoint, device=args_cli.device)
    policy.eval()

    # reset environment
    obs_dict, _ = env.reset()
    # only cares about policy observations
    obs = obs_dict["policy"].to(args_cli.device)


    episode_id = 1
    episode_step = 0
    
    step_cnt = 0
    success_cnt = 0
    timeout_cnt = 0
    num_episodes = 50 # set number of episodes
    success_cnt = 0
    success_steps = []
    episode_reward = 0.0
    total_rewards = []
    episode_lengths = []
   
    
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute actions
            # actions = policy._predict(obs, deterministic=True)
            traj = torch.load(traj_path)
            for step in traj:
                actions = step["action"].to(env.device)
                
            if actions.ndim == 1:
                actions = actions.unsqueeze(0)
            actions = actions.to(env.device)
            
            
            # apply actions
            # obs_dict = env.step(actions)[0]
            obs_dict, rewards, terminated, truncated, info = env.step(actions)
            # only cares about policy observations
            obs = obs_dict["policy"].to(args_cli.device)
    

            episode_reward += rewards.mean().item()
            episode_step += 1

            success = info["log"]["Episode_Termination/object_lifted"]
            timeout = info["log"]["Episode_Termination/time_out"]


            # print("obs", obs.shape, obs[0, :10])
            # print("actions", actions.shape, actions[0])
            print(
                "step:", step["step"],
                "action shape:", actions.shape,
                "action:", actions[0],
                "terminated:", terminated,
                "truncated:", truncated,
            )
            
            if success:
                success_cnt += 1
                success_steps.append(episode_step)
                episode_lengths.append(episode_step)
            
            if timeout:
                timeout_cnt += 1

            if (terminated | truncated).any():
                episode_id += 1
                total_rewards.append(episode_reward)
                episode_lengths.append(episode_step)
                
                # reset
                episode_reward = 0.0
                episode_step = 0

                obs_dict, _ = env.reset()
                obs = obs_dict["policy"].to(args_cli.device)
    
    # print summary
    print("-" * 50)
    print(f"{'Metric':<25} | {'Value':<15}")
    print("-" * 50)
    print(f"{'Task':<25} | {args_cli.task:<15}")
    print(f"{'Checkpoint':<25} | {Path(checkpoint).name:<15}")
    print(f"{'Episodes':<25} | {episode_id:<15}")
    print(f"{'Success Episodes':<25} | {success_cnt:<15}")
    print(f"{'Success Rate':<25} | {success_cnt / episode_id * 100:.1f}%")
    print(f"{'Timeout Episodes':<25} | {timeout_cnt:<15}")
    print(f"{'Timeout Rate':<25} | {timeout_cnt / episode_id * 100:.1f}%")
    print(f"{'Mean Reward':<25} | {sum(total_rewards) / len(total_rewards):.3f}")
    print(f"{'Mean Episode Length':<25} | {sum(episode_lengths) / len(episode_lengths):.1f}")
    if success_steps:
        print(f"{'Mean Success Step':<25} | {sum(success_steps) / len(success_steps):.1f}")
    else:
        print(f"{'Mean Success Step':<25} | N/A")
    print("-" * 50)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
