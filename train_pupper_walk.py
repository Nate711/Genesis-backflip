import argparse
import os
import pathlib
import pickle

import wandb
from reward_wrapper import Go2
from rsl_rl.runners import OnPolicyRunner

import genesis as gs
import re


def get_train_cfg(args):

    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": args.exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": args.max_iterations,
            "num_steps_per_env": 24,
            "policy_class_name": "ActorCritic",
            "record_interval": 50,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": 100,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "urdf_path": "/home/nathankau/pupperv3-monorepo/ros2_ws/src/pupper_v3_description/description/mujoco_xml/pupper_v3_complete.position.full_collision.xml",
        # "urdf_path": "urdf/go2/urdf/go2.urdf",
        "links_to_keep": [
            # "FL_foot",
            # "FR_foot",
            # "RL_foot",
            # "RR_foot",
        ],  # not used for MJCF????
        "num_actions": 12,
        "num_dofs": 12,
        # joint/link names
        "default_joint_angles": {  # [rad]
            "leg_front_r_1": 0.26,
            "leg_front_r_2": 0.0,
            "leg_front_r_3": -0.52,
            "leg_front_l_1": -0.26,
            "leg_front_l_2": 0.0,
            "leg_front_l_3": 0.52,
            "leg_back_r_1": 0.26,
            "leg_back_r_2": 0.0,
            "leg_back_r_3": -0.52,
            "leg_back_l_1": -0.26,
            "leg_back_l_2": 0.0,
            "leg_back_l_3": 0.52,
        },
        "dof_names": [
            "leg_front_r_1",
            "leg_front_r_2",
            "leg_front_r_3",
            "leg_front_l_1",
            "leg_front_l_2",
            "leg_front_l_3",
            "leg_back_r_1",
            "leg_back_r_2",
            "leg_back_r_3",
            "leg_back_l_1",
            "leg_back_l_2",
            "leg_back_l_3",
        ],
        "termination_contact_link_names": ["base_link"],
        "penalized_contact_link_names": ["base_link", "_2", "_1"],
        "feet_link_names": ["_3"],
        "base_link_name": ["base_link"],
        # PD
        "PD_stiffness": {"leg_": 5.0},
        "PD_damping": {"leg_": 0.25},
        "use_implicit_controller": True,
        # termination
        "termination_if_roll_greater_than": 0.4,
        "termination_if_pitch_greater_than": 0.4,
        "termination_if_height_lower_than": 0.0,
        # base pose
        "base_init_pos": [
            12.0,
            12.0,
            0.24,
        ],  # use 12 if using terrain. 0.42m is high right?
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        # random push
        "push_interval_s": -1,
        "max_push_vel_xy": 1.0,
        # time (second)
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "command_type": "ang_vel_yaw",  # 'ang_vel_yaw' or 'heading'
        "action_scale": 0.25,
        "action_latency": 0.02,
        "clip_actions": 100.0,
        "send_timeouts": True,
        "control_freq": 50,
        "decimation": 2,
        "feet_geom_offset": 1,
        "use_terrain": True,
        "terrain_cfg": {
            "vertical_scale": 0.005,
            "horizontal_scale": 0.25,
            "n_subterrains": (2, 2),
            "subterrain_size": (12, 12),
            "subterrain_types": [
                ["flat_terrain", "random_uniform_terrain"],
                ["pyramid_sloped_terrain", "discrete_obstacles_terrain"],
            ],
        },
        # domain randomization
        "randomize_friction": True,
        "friction_range": [0.2, 1.5],
        "randomize_base_mass": True,
        "added_mass_range": [-0.2, 0.2],
        "randomize_com_displacement": True,
        "com_displacement_range": [-0.01, 0.01],
        "randomize_motor_strength": False,
        "motor_strength_range": [0.9, 1.1],
        "randomize_motor_offset": True,
        "motor_offset_range": [-0.02, 0.02],
        "randomize_kp_scale": True,
        "kp_scale_range": [0.8, 1.2],
        "randomize_kd_scale": True,
        "kd_scale_range": [0.8, 1.2],
        # coupling
        "coupling": False,
    }
    obs_cfg = {
        "num_obs": 9 + 3 * env_cfg["num_dofs"],
        "num_history_obs": 1,
        "obs_noise": {
            "ang_vel": 0.1,
            "gravity": 0.02,
            "dof_pos": 0.01,
            "dof_vel": 0.5,
        },
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
        "num_priv_obs": 12 + 4 * env_cfg["num_dofs"],
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "soft_dof_pos_limit": 0.9,
        "base_height_target": 0.135,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.5,
            "lin_vel_z": -2.0,
            "ang_vel_xy": -0.05,
            "orientation": -10.0,
            "base_height": -5.0,
            # "torques": -0.0002,
            "collision": -1.0,
            "dof_vel": -0.0,
            "dof_acc": -2.5e-7,
            "feet_air_time": 1.0,
            "collision": -1.0,
            "action_rate": -0.01,
            "termination": -200.0,
        },
    }
    command_cfg = {
        "num_commands": 4,
        "lin_vel_x_range": [-0.75, 0.75],
        "lin_vel_y_range": [-0.5, 0.5],
        "ang_vel_range": [-1.0, 1.0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def create_log_dir(logs_dir: pathlib.Path, exp_name: str):
    logs_dir = pathlib.Path(logs_dir)
    # Intelligently create log dir
    log_dir = logs_dir / pathlib.Path(exp_name)

    exp_number = 0
    while log_dir.exists():
        exp_number += 1
        log_dir = logs_dir / pathlib.Path(f"{exp_name}-{exp_number}")

    gs.logger.info(f"New log directory is {log_dir}")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--exp_name", type=str, required=True, default="PupperV3Walk"
    )
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    parser.add_argument("-B", "--num_envs", type=int, default=10000)
    parser.add_argument("--max_iterations", type=int, default=1000)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("-o", "--offline", action="store_true", default=False)

    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--ckpt", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if args.debug:
        args.vis = True
        args.offline = True
        args.num_envs = 1

    gs.init(
        backend=gs.cpu if args.cpu else gs.gpu,
        logging_level="warning",
    )

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args)

    log_dir = create_log_dir(pathlib.Path("logs"), args.exp_name)
    wandb.init(
        project="genesis",
        name=args.exp_name,
        dir=log_dir,
        mode="offline" if args.offline else "online",
        config={
            "num_envs": args.num_envs,
            "train_cfg": train_cfg,
            "env_cfg": env_cfg,
            "obs_cfg": obs_cfg,
            "reward_cfg": reward_cfg,
            "command_cfg": command_cfg,
        },
    )

    env = Go2(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.vis,
        eval=args.eval,
        debug=args.debug,
        device=args.device,
    )

    gs.logger.warning(f"SUBSTEPS: {env.scene.substeps}")

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=args.device)

    if args.resume is not None:
        resume_dir = f"logs/{args.resume}"
        resume_path = os.path.join(resume_dir, f"model_{args.ckpt}.pt")
        gs.logger.warning(f"==> resume training from  {resume_path}")
        runner.load(resume_path)

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    # BUG: runner.learn calls get_observations and it returns zeros at the first step
    runner.learn(
        num_learning_iterations=args.max_iterations, init_at_random_ep_len=True
    )


if __name__ == "__main__":
    main()


"""
# training
python train_backflip.py -e EXP_NAME

# evaluation
python eval_backflip.py -e EXP_NAME --ckpt NUM_CKPT
"""
