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
    checkpoint = args_cli.checkpoint or IL_DIR / "policies" / "bc_lift_n_policy_2.zip"

    # acquire device
    # device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    # restore policy
    # policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=args_cli.checkpoint, device=device, verbose=True)
    policy = ActorCriticPolicy.load(checkpoint, device=args_cli.device)
    policy.eval()

    # reset environment
    obs_dict, _ = env.reset()
    # robomimic only cares about policy observations
    obs = obs_dict["policy"].to(args_cli.device)

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute actions
            # actions = policy(obs)
            actions = policy._predict(obs, deterministic=True)
            # actions = torch.from_numpy(actions).to(device=device).view(1, env.action_space.shape[1])
            # apply actions
            # obs_dict = env.step(actions)[0]
            obs_dict, rewards, terminated, truncated, info = env.step(actions)
            # robomimic only cares about policy observations
            obs = obs_dict["policy"].to(args_cli.device)

            if (terminated | truncated).any():
                obs_dict, _ = env.reset()
                obs = obs_dict["policy"].to(args_cli.device)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
