"""Calibrate the deployment base_lin_vel formula against the IsaacSim ground truth.

Loads a trained D1H policy, plays it, and regresses the TRUE body-frame base velocity
(root_lin_vel_b) against [wheel odometry, gyro]:

    v_x_true ~= a * (mean_wheel_dq * radius) + b * gyro_y
    v_y_true ~= c * gyro_x

Expected (rigid-body kinematics, lever arm = base height h ~ 0.40 m):
    a ~ 1            (wheel rolling -> forward speed)
    b ~ +h ~ +0.40   (pitch forward -> +x velocity)
    c ~ -h ~ -0.40   (roll -> y velocity, sign from convention)

Use the printed a/b/c directly in the deployment FSMState_RLInstinctLab.cpp.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Calibrate D1H base_lin_vel deployment formula.")
parser.add_argument("--task", type=str, default="Instinct-Locomotion-Flat-D1H-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=1500)
parser.add_argument("--disable_fabric", action="store_true", default=False)
cli_args.add_instinct_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# defaults for the run to calibrate (cli_args provides --load_run / --checkpoint)
if args_cli.load_run is None:
    args_cli.load_run = "20260821_223509"
if args_cli.checkpoint is None:
    args_cli.checkpoint = "model_11400.pt"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import torch

from instinct_rl.runners import OnPolicyRunner

from isaaclab.utils.io import load_yaml
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from instinctlab.utils.wrappers.instinct_rl import InstinctRlOnPolicyRunnerCfg, InstinctRlVecEnvWrapper


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: InstinctRlOnPolicyRunnerCfg = cli_args.parse_instinct_rl_cfg(args_cli.task, args_cli)
    agent_cfg.load_run = args_cli.load_run

    log_root_path = os.path.abspath(os.path.join("logs", "instinct_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    agent_cfg_dict = load_yaml(os.path.join(log_dir, "params", "agent.yaml"))

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = InstinctRlVecEnvWrapper(env)

    ppo_runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    robot = env.unwrapped.scene["robot"]
    wheel_names = ["FL_foot_joint", "FR_foot_joint"]
    wheel_ids = [robot.joint_names.index(n) for n in wheel_names]
    radius = 0.087

    obs, _ = env.get_observations()
    rows = []
    print(f"[INFO] Collecting {args_cli.steps} steps (forced fwd/bwd for signal variance) ...")
    for t in range(args_cli.steps):
        # force velocity commands so v_x / odom / gyro_y get real variance (the policy
        # balancing in place keeps all signals near zero and the regression is noise)
        cmd = env.unwrapped.command_manager.get_command("base_velocity")
        frac = t / args_cli.steps
        if frac < 0.45:
            cmd[0, 0] = 1.0
        elif frac < 0.90:
            cmd[0, 0] = -0.8
        else:
            cmd[0, 0] = 0.0
        cmd[0, 1] = 0.0
        cmd[0, 2] = 0.0
        action = policy(obs)
        obs, _, _, _ = env.step(action)
        if t < 50:
            continue  # skip spawn/getup transient
        v_b = robot.data.root_lin_vel_b[0].cpu().numpy()      # true base velocity, body frame (m/s)
        w_b = robot.data.root_ang_vel_b[0].cpu().numpy()      # body angular velocity = gyro (rad/s)
        dq = robot.data.joint_vel[0, wheel_ids].cpu().numpy()  # wheel joint velocities (rad/s)
        odom = float(np.mean(dq)) * radius                     # wheel odometry x (m/s)
        rows.append([v_b[0], v_b[1], odom, w_b[0], w_b[1], w_b[2]])

    A = np.array(rows)  # [vx, vy, odom, wx, wy, wz]
    n = len(A)
    print(f"[INFO] collected {n} samples")

    # fit v_x = a*odom + b*wy
    Xx = np.column_stack([A[:, 2], A[:, 4]])
    coef_x, res_x, *_ = np.linalg.lstsq(Xx, A[:, 0], rcond=None)
    a, b = coef_x
    pred_x = Xx @ coef_x
    r2_x = 1 - np.sum((A[:, 0] - pred_x) ** 2) / np.sum((A[:, 0] - A[:, 0].mean()) ** 2)

    # fit v_y = c*wx  (and optionally wz, but wx dominates on flat)
    Xy = A[:, 3:4]
    coef_y, *_ = np.linalg.lstsq(Xy, A[:, 1], rcond=None)
    c = coef_y[0]
    pred_y = Xy @ coef_y
    r2_y = 1 - np.sum((A[:, 1] - pred_y) ** 2) / np.sum((A[:, 1] - A[:, 1].mean()) ** 2)

    # simple correlations for sign sanity
    corr_x_odom = np.corrcoef(A[:, 0], A[:, 2])[0, 1]
    corr_x_wy = np.corrcoef(A[:, 0], A[:, 4])[0, 1]
    corr_y_wx = np.corrcoef(A[:, 1], A[:, 3])[0, 1]

    print("\n===== CALIBRATION RESULT =====")
    print(f"v_x_true = {a:+.3f} * (mean_wheel_dq*{radius}) {b:+.3f} * gyro_y   (R^2={r2_x:.3f})")
    print(f"  expect a~1, b~+0.40 (lever arm h). corr(vx,odom)={corr_x_odom:+.2f}, corr(vx,wy)={corr_x_wy:+.2f}")
    print(f"v_y_true = {c:+.3f} * gyro_x                     (R^2={r2_y:.3f})")
    print(f"  expect c~-0.40 (=-h). corr(vy,wx)={corr_y_wx:+.2f}")
    print("\nDeployment formula (FSMState_RLInstinctLab.cpp):")
    print(f"  lin_vel[0] = {a:+.3f} * mean_wheel_speed*0.087 + ({b:+.3f}) * gyro[1]")
    print(f"  lin_vel[1] = ({c:+.3f}) * gyro[0]")
    print(f"  lin_vel[2] = 0")
    print("\nIf R^2 is low, the estimate misses structure (e.g. yaw term -wz*ry) —")
    print("include gyro_z in the x regression to check: v_x = a*odom + b*wy + d*wz")

    simulation_app.close()


if __name__ == "__main__":
    main()
