"""D1H-specific reward terms (bipedal wheel-legged).

Ported from DDT_Lab: source/ddt_lab/ddt_lab/tasks/manager_based/locomotion/mdp/rewards.py
Only the functions referenced by the D1H env config are kept.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _base_height_sigma(env: ManagerBasedRLEnv, target_height: float = 0.35) -> torch.Tensor:
    """Height gate that scales a reward by how tall the robot is.

    Uses a sharp sigmoid around ``target_height`` (0.35 m): rewards stay ~1 when
    standing at/above 0.35 m and vanish quickly below it, forcing the policy to
    keep the frame up to earn the (tracking) reward.
    """
    asset: RigidObject = env.scene["robot"]
    z = asset.data.root_pos_w[:, 2]
    return torch.sigmoid((z - target_height) / 0.02)


def track_lin_vel_x_split_exp(
    env: ManagerBasedRLEnv, std_fwd: float, std_bwd: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of the x-axis velocity, with separate forward/backward kernels.

    Deploy feedback: backward tracking is worse than forward (the robot drifts on
    reverse). Splitting the exponential kernel lets the two directions use
    different bandwidths: ``std_bwd`` tighter than ``std_fwd`` gives reverse motion
    a sharper gradient near the command, so the policy learns reverse as precisely
    as forward instead of one shared kernel averaging both.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    vel_x = asset.data.root_lin_vel_b[:, 0]
    err = torch.square(cmd_x - vel_x)
    std = torch.where(cmd_x >= 0.0, std_fwd, std_bwd)
    reward = torch.exp(-err / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    reward *= _base_height_sigma(env)
    return reward


def track_lin_vel_x_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of the x-axis (fore-aft) linear velocity command.

    Scaled by a base-height gate so the policy must stand tall to earn it.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 0] - asset.data.root_lin_vel_b[:, 0]
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    reward *= _base_height_sigma(env)
    return reward


def track_lin_vel_y_lateral_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward lateral (y) velocity tracking, gated on the y command.

    - When a lateral command is active (|cmd_y| > threshold): reward y-velocity
      tracking only when NOT both feet are planted (at least one foot lifted),
      so the feet must step/lift sideways instead of just tilting the body with
      both wheels down.
    - When NO lateral command is given: still penalize y drift by tracking the
      commanded y = 0, so the robot must not creep sideways while not strafing.
    """
    asset = env.scene[asset_cfg.name]
    contact_sensor = env.scene[sensor_cfg.name]
    contact = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 1.0
    n_feet_in_contact = torch.sum(contact.float(), dim=1)
    # at least one foot airborne (not both planted) — allows single-leg support or
    # alternating lift, but prevents pure body-tilt cheating with both wheels down
    single_support = n_feet_in_contact < 2

    cmd = env.command_manager.get_command(command_name)
    cmd_y = cmd[:, 1]
    y_error = torch.square(cmd_y - asset.data.root_lin_vel_b[:, 1])
    reward = torch.exp(-y_error / std**2)

    # smooth gate: 1 when a lateral command is active, 0 when near-zero
    y_cmd_gate = torch.sigmoid((torch.abs(cmd_y) - 0.05) / 0.01)
    # when a lateral command is active, require single-leg support (alternating
    # lift); when no lateral command, keep tracking y=0 (no single-support req)
    reward = reward * (y_cmd_gate * single_support.float() + (1.0 - y_cmd_gate))
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    reward *= _base_height_sigma(env)
    return reward


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel.

    Scaled by a base-height gate so the policy must stand tall to earn it.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    reward *= _base_height_sigma(env)
    return reward


def vel_smoothness(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward smooth (jerk-free) base linear velocity.

    Deploy feedback: speed tracking is discontinuous — the robot jerks between
    speed steps. Penalise the magnitude of the current frame's change in x/y
    linear velocity (acceleration proxy). When the command is ~constant, smooth
    means near-zero velocity change; when the command ramps, small changes keep
    the motion continuous instead of step-like.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    # current and previous frame base linear velocity (world frame, so compare directly)
    vel_now = asset.data.root_lin_vel_w[:, :2]
    if not hasattr(env, "_prev_base_vel"):
        env._prev_base_vel = vel_now.clone()
    dvel = torch.norm(vel_now - env._prev_base_vel, dim=1)
    env._prev_base_vel = vel_now.clone()
    reward = torch.exp(-dvel / 0.5)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def lateral_lift_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    lift_threshold: float = 0.15,
    max_lift_time: float = 0.5,
) -> torch.Tensor:
    """Reward sustained airborne feet while a lateral (y) command is active.

    Side-stepping needs a genuine lift so the foot can move sideways with stride.
    Reward grows with how long a foot stays airborne (capped at ``max_lift_time``),
    which encourages longer, larger strides instead of quick taps.
    """
    contact_sensor = env.scene[sensor_cfg.name]
    contact = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 1.0
    air_time = getattr(contact_sensor.data, "air_time", None)
    if air_time is None:
        air_time = contact_sensor.data.current_air_time
    air_time = air_time[:, sensor_cfg.body_ids]

    airborne = (~contact).float()
    lift_t = torch.clamp(air_time, 0.0, max_lift_time) * airborne

    cmd = env.command_manager.get_command(command_name)
    cmd_y = cmd[:, 1]
    lateral_gate = torch.sigmoid((torch.abs(cmd_y) - lift_threshold) / 0.05)
    reward = torch.sum(lift_t, dim=1)
    return lateral_gate * reward


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    reward = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def wheel_action_rate(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the rate of change of the wheel actions (wheel velocity jitter).

    The wheels are velocity-controlled (JointVelocityActionCfg, action indices
    3 and 7 in D1HActionsCfg order), so the action directly sets wheel speed.
    Penalizing the wheel action difference prevents the policy from swinging
    the wheels back-and-forth, which oscillates the trunk pitch.
    """
    action = env.action_manager.action
    prev = env.action_manager.prev_action
    wheel_ids = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else [3, 7]
    reward = torch.sum(torch.square(action[:, wheel_ids] - prev[:, wheel_ids]), dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def wheel_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the actual wheel joint acceleration (abrupt wheel speed jumps).

    The wheels are velocity-controlled; abrupt speed jumps (±20 rad/s) shook the
    torso. Penalizing the measured wheel joint acceleration (joint_acc) directly
    smooths the wheel speed profile.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_ids = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else [3, 7]
    reward = torch.sum(torch.square(asset.data.joint_acc[:, wheel_ids]), dim=1)
    reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward joint_power"""
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return reward


def joint_pos_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    running_reward = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1
    )
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        stand_still_scale * running_reward,
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_mirror(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    mirror_joints: list[list[str]],
    command_name: str | None = None,
    lateral_release: float = 0.0,
) -> torch.Tensor:
    """Penalize the difference between mirrored (left-right) joint pairs.

    Gated on the command so the penalty applies only when it is safe to demand
    leg symmetry: standing still, or driving forward/backward/turning (x or z
    velocity), where both legs should move together. While a lateral (y) command
    is active the penalty is switched off (``lateral_release``), because strafing
    REQUIRES the legs to move asymmetrically — one lifts and swings sideways
    while the other supports. Run 20260810_150316 raised this term to -30 with
    no gate and lateral tracking collapsed (err_y 0.92, y command ignored).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    for joint_pair in env.joint_mirror_joints_cache:
        diff = torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    if command_name is not None:
        cmd_y = env.command_manager.get_command(command_name)[:, 1]
        # 1 while a lateral command is active, 0 when it is quiet
        y_gate = torch.sigmoid((torch.abs(cmd_y) - 0.05) / 0.01)
        # full strength when there is no lateral command (standing / x / z motion),
        # `lateral_release` while strafing (default 0 -> penalty fully off)
        reward = reward * (1.0 - y_gate * (1.0 - lateral_release))
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward being upright (projected gravity z close to -1).

    Keep the SQUARED form. It is deliberately not ddt_rl_isaacgym's linear
    ``1 - g_z``: the squared form pays +3.0 for standing vs lying down (4.0 at
    g=-1 against 1.0 at g=0) where the linear form pays only +1.0. Run
    20260810_142451 switched to linear and the policy immediately chose to lie
    down (mean tilt 82.4 deg, track_x collapsed 12.31 -> 0.017) because the
    standing incentive no longer covered the effort, especially with tracking
    weights halved. Small-tilt posture is handled by flat_orientation_l2 (-15),
    which discriminates near-upright far better than either upward form.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        adjusted_target_height = target_height
    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_lin_vel_b[:, 2])
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def ang_vel_x_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize x-axis (roll) base angular velocity using L2 squared kernel.

    NO command gate. Run 20260810_155717 gated this term on the lateral command and
    the policy stopped learning to stand at all (upward collapsed 3.6 -> 1.9, i.e.
    lying down): during early training lateral commands are active most of the time,
    so the gate released the roll damping exactly when the policy needed it to learn
    to balance. The penalty is on roll ANGULAR VELOCITY, a dynamic quantity — it
    damps continuous rolling/shake, not the brief lean used during a strafe step,
    so it does not block side-stepping.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_ang_vel_b[:, 0])
    reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def ang_vel_y_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize y-axis (pitch) base angular velocity using L2 squared kernel.

    Split out of ang_vel_xy_l2 so the pitch rate (the dominant torso-shake axis
    for a wheel-legged biped) can be weighted independently of the roll rate.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_ang_vel_b[:, 1])
    reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    reward = torch.sum(is_contact, dim=1).float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def default_joint_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint position deviation from the default pose (sum of squares).

    Includes the upright filter (``clamp(-grav_z, 0, 0.7) / 0.7``) so the penalty fades when
    the robot is not roughly upright.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    q_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.square(q - q_default), dim=1)
    reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def reward_feet_air_time(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=[".*_foot"]),
    command_name: str = "base_velocity",
    min_air_t: float = 0.5,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward airborne feet (taken steps) when a velocity command is active."""
    contact_sensor = env.scene[sensor_cfg.name]
    contact = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > force_threshold
    # v2.x compat: use current_air_time (v2.3) instead of air_time (v3.0+)
    air_time = getattr(contact_sensor.data, "air_time", None)
    if air_time is None:
        air_time = contact_sensor.data.current_air_time
    air_time = air_time[:, sensor_cfg.body_ids]

    landed = contact & (air_time > min_air_t)
    rew_airTime = torch.sum(landed.float(), dim=-1)

    cmd = env.command_manager.get_command(command_name)
    moving_mask = torch.norm(cmd[:, :2], dim=1) > 0.1
    rew_airTime *= moving_mask

    return rew_airTime


def reward_collision_head(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["base_link"]),
    threshold: float = 10.0,
) -> torch.Tensor:
    """Penalize contacts on the head/base link above a force threshold."""
    contact_sensor = env.scene[sensor_cfg.name]
    forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    return torch.sum((forces > threshold).float(), dim=-1)


def reward_dof_thigh_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*_thigh_joint"]),
) -> torch.Tensor:
    """Penalize thigh joint velocities."""
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=-1)


def leg_activity(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lean_gain: float = 0.10,
    vel_penalty_gain: float = 0.0,
) -> torch.Tensor:
    """Reward the feet being under the center of mass (legs follow wheel roll).

    Control architecture: the wheels high-frequency hold the torso level AND
    produce the fore/aft translation; the legs LOW-FREQUENCY move the wheels to
    keep them under the COM (the wheel is the inverted-pendulum pivot — the COM
    must stay above/behind the contact point, so as the robot rolls forward the
    legs swing the wheels forward to stay under the COM).

    Reward is maximized when the MEAN foot x-offset (mean over BOTH feet) matches
    the COM x-offset from the body, i.e. the feet sit directly under the center of
    mass:
        feet_x_body ≈ com_x_body
    Using the MEAN offset keeps the two legs moving together in x: a staggered
    stance (one foot ahead, one behind) cancels in the mean and earns nothing.

    The command still gates the reward (no leg positioning demanded while
    standing still), but the TARGET is physical (COM) not heuristic, so the legs
    automatically follow however far the wheels have rolled.

    ``vel_penalty_gain`` > 0 subtracts a penalty proportional to the leg joint
    velocities, so fast swinging erodes the reward — the policy learns to move
    the legs slowly to settle them, not to jitter them at high frequency.
    """
    asset = env.scene[asset_cfg.name]
    foot_body_names = ["FL_foot", "FR_foot"]
    foot_ids = [asset.body_names.index(name) for name in foot_body_names]
    feet_pos_w = asset.data.body_pos_w[:, foot_ids, :]
    base_pos_w = asset.data.root_pos_w.unsqueeze(1)
    base_quat_w = asset.data.root_quat_w
    rel_w = feet_pos_w - base_pos_w
    n_envs, n_feet = rel_w.shape[:2]
    base_quat_w_expanded = base_quat_w.unsqueeze(1).expand(-1, n_feet, -1).reshape(-1, 4)
    rel_w_flat = rel_w.reshape(-1, 3)
    rel_b = quat_apply_inverse(base_quat_w_expanded, rel_w_flat).reshape(n_envs, n_feet, 3)
    feet_x_body = torch.mean(rel_b[:, :, 0], dim=1)

    # COM offset from body in body frame (x component)
    com_pos_w = asset.data.root_com_pose_w[:, :3]
    com_rel_w = com_pos_w - base_pos_w.squeeze(1)
    com_x_body = quat_apply_inverse(base_quat_w, com_rel_w)[:, 0]

    # target: feet directly under the COM
    err = torch.abs(feet_x_body - com_x_body)

    # ALWAYS active, including standing still (no command gate): keeping the feet
    # under the COM is the basic balance requirement, so the legs must position
    # themselves to hold the wheels under the COM whether moving or stationary.
    # Deploy feedback: with the moving gate, standing still left the legs passive
    # and the torso wobbled fore/aft by pitching. Removing the gate makes the legs
    # participate in static balance too.
    # sigma 0.02: tight reward, feet must sit within ±2cm of the COM projection.
    reward = torch.exp(-err / 0.02)

    if vel_penalty_gain > 0.0:
        leg_ids = asset_cfg.joint_ids
        if leg_ids is None:
            # default: all leg joints (hip/thigh/calf), keep wheels out
            leg_ids = [
                i for i, name in enumerate(asset.joint_names) if "_foot_joint" not in name
            ]
        vel = asset.data.joint_vel[:, leg_ids]
        reward = reward - vel_penalty_gain * torch.mean(torch.square(vel), dim=1)
    return reward


def pitch_target_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base pitch deviation from level (0 rad).

    LQR balance channel: the trunk stays pitch-flat and the wheels compensate
    any lean (inverted-pendulum balance). pitch = asin(g_x) of projected
    gravity in the body frame.
    """
    asset = env.scene[asset_cfg.name]
    g = asset.data.projected_gravity_b
    pitch = torch.asin(torch.clamp(g[:, 0], -1.0, 1.0))
    return torch.square(pitch)


def pitch_rate_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base pitch angular velocity squared (damping of the balance channel)."""
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b[:, 1])


def pitch_command_alignment(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize pitch that tilts WITH the velocity command direction.

    The robot tilts its trunk to help accelerate/decelerate: nose-up (pitch>0)
    when moving forward, tail-up (pitch<0) when reversing. This reward keeps
    the trunk flat regardless of command by penalizing pitch aligned with the
    command sign, i.e. pitch*cmd > 0.
    """
    asset = env.scene[asset_cfg.name]
    g = asset.data.projected_gravity_b
    pitch = torch.asin(torch.clamp(g[:, 0], -1.0, 1.0))
    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    aligned = pitch * torch.sign(cmd_x)
    return torch.square(torch.clamp(aligned, min=0.0))


def reward_body_pos_to_feet_x(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sigma: float,
) -> torch.Tensor:
    """Penalize body-x offset from the mid-point between the two feet."""
    asset = env.scene[asset_cfg.name]
    foot_body_names = ["FL_foot", "FR_foot"]
    foot_ids = [asset.body_names.index(name) for name in foot_body_names]

    feet_pos_w = asset.data.body_pos_w[:, foot_ids, :]
    base_pos_w = asset.data.root_pos_w.unsqueeze(1)
    base_quat_w = asset.data.root_quat_w

    rel_w = feet_pos_w - base_pos_w
    n_envs, n_feet = rel_w.shape[:2]
    base_quat_w_expanded = base_quat_w.unsqueeze(1).expand(-1, n_feet, -1).reshape(-1, 4)
    rel_w_flat = rel_w.reshape(-1, 3)
    rel_b = quat_apply_inverse(base_quat_w_expanded, rel_w_flat).reshape(n_envs, n_feet, 3)

    x_mean_abs = torch.abs(torch.mean(rel_b[:, :, 0], dim=1))
    reward = torch.exp(-x_mean_abs / sigma)
    return reward


def reward_body_feet_distance_x(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sigma: float,
) -> torch.Tensor:
    """Penalize x-distance between the two feet (forbids legs splitting fore/aft)."""
    asset = env.scene[asset_cfg.name]
    foot_body_names = ["FL_foot", "FR_foot"]
    foot_ids = [asset.body_names.index(name) for name in foot_body_names]

    feet_pos_w = asset.data.body_pos_w[:, foot_ids, :]
    base_quat_w = asset.data.root_quat_w

    foot_diff_w = feet_pos_w[:, 0, :] - feet_pos_w[:, 1, :]
    foot_diff_b = quat_apply_inverse(base_quat_w, foot_diff_w)

    x_err = torch.abs(foot_diff_b[:, 0]) / sigma
    reward = torch.square(x_err)
    return reward


def body_feet_vel_x(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the relative x-velocity difference between the two feet.

    Catches the "staggered gait" failure: the policy holds balance by swinging
    one leg forward while the other swings backward, so the two feet have
    opposite body-frame x velocities. A position penalty (reward_body_feet_distance_x)
    misses this because the stagger offset is time-varying and averages out;
    the opposite-signed velocities are caught instantly.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    foot_body_names = ["FL_foot", "FR_foot"]
    foot_ids = [asset.body_names.index(name) for name in foot_body_names]

    v_w = asset.data.body_lin_vel_w[:, foot_ids, :]  # (B, 2, 3) world frame
    base_quat_w = asset.data.root_quat_w
    n_envs, n_feet = v_w.shape[:2]
    base_quat_w_expanded = base_quat_w.unsqueeze(1).expand(-1, n_feet, -1).reshape(-1, 4)
    v_w_flat = v_w.reshape(-1, 3)
    v_b = quat_apply_inverse(base_quat_w_expanded, v_w_flat).reshape(n_envs, n_feet, 3)

    dx_vel = v_b[:, 0, 0] - v_b[:, 1, 0]
    reward = torch.square(dx_vel)
    reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def reward_body_feet_distance_y(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sigma: float,
    desired_feet_distance: float,
    command_name: str | None = None,
    lateral_threshold: float = 0.15,
) -> torch.Tensor:
    """Penalize deviation of the lateral foot distance from the desired stance width.

    Keeps the two wheels at ``desired_feet_distance`` (40 cm) apart while standing
    or driving forward/backward. When a lateral (y) command is active the penalty
    is switched OFF, because side-stepping needs the feet to spread/step sideways
    and pinning them at 40 cm would kill the gait.
    """
    asset = env.scene[asset_cfg.name]
    foot_body_names = ["FL_foot", "FR_foot"]
    foot_ids = [asset.body_names.index(name) for name in foot_body_names]

    feet_pos_w = asset.data.body_pos_w[:, foot_ids, :]
    base_quat_w = asset.data.root_quat_w

    foot_diff_w = feet_pos_w[:, 0, :] - feet_pos_w[:, 1, :]
    foot_diff_b = quat_apply_inverse(base_quat_w, foot_diff_w)

    y_abs = torch.abs(foot_diff_b[:, 1])
    y_err = torch.abs(y_abs - desired_feet_distance) / sigma
    reward = torch.square(y_err)
    if command_name is not None:
        cmd_y = env.command_manager.get_command(command_name)[:, 1]
        # 1 while a lateral command is active -> penalty off; 0 otherwise
        y_gate = torch.sigmoid((torch.abs(cmd_y) - lateral_threshold) / 0.02)
        reward = reward * (1.0 - y_gate)
    return reward


def reward_body_symmetry_y(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sigma: float,
) -> torch.Tensor:
    """Reward left/right feet having the same lateral (y) offset from the body."""
    asset = env.scene[asset_cfg.name]
    foot_body_names = ["FL_foot", "FR_foot"]
    foot_ids = [asset.body_names.index(name) for name in foot_body_names]

    feet_pos_w = asset.data.body_pos_w[:, foot_ids, :]
    base_pos_w = asset.data.root_pos_w.unsqueeze(1)
    base_quat_w = asset.data.root_quat_w

    rel_w = feet_pos_w - base_pos_w
    rel_b_0 = quat_apply_inverse(base_quat_w, rel_w[:, 0, :])
    rel_b_1 = quat_apply_inverse(base_quat_w, rel_w[:, 1, :])

    y1_abs = torch.abs(rel_b_0[:, 1])
    y2_abs = torch.abs(rel_b_1[:, 1])
    sym_err = torch.abs(y1_abs - y2_abs)

    reward = torch.exp(-sym_err / sigma)
    return reward


def reward_body_symmetry_z(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sigma: float,
) -> torch.Tensor:
    """Reward left/right feet having the same height (z) offset from the body."""
    asset = env.scene[asset_cfg.name]
    foot_body_names = ["FL_foot", "FR_foot"]
    foot_ids = [asset.body_names.index(name) for name in foot_body_names]

    feet_pos_w = asset.data.body_pos_w[:, foot_ids, :]
    base_pos_w = asset.data.root_pos_w.unsqueeze(1)
    base_quat_w = asset.data.root_quat_w

    rel_w = feet_pos_w - base_pos_w
    rel_b_0 = quat_apply_inverse(base_quat_w, rel_w[:, 0, :])
    rel_b_1 = quat_apply_inverse(base_quat_w, rel_w[:, 1, :])

    z1_abs = torch.abs(rel_b_0[:, 2])
    z2_abs = torch.abs(rel_b_1[:, 2])
    sym_err = torch.abs(z1_abs - z2_abs)

    reward = torch.exp(-sym_err / sigma)
    return reward


def lateral_leg_lift(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[".*_foot"]),
    lift_threshold: float = 0.1,
    lift_height: float = 0.02,
    lateral_gain: float = 0.5,
    min_y_shift: float = 0.03,
) -> torch.Tensor:
    """Reward lateral foot DISPLACEMENT (sideways stepping) when a y-command is given.

    Sideways movement of a wheeled biped needs the legs to step sideways (hip
    lateral swing), not just tilt the body. This rewards the foot's *lateral
    displacement* in the body frame (foot y-offset from the body) aligned with
    the commanded y direction, so the policy must actually move its feet
    sideways rather than just lift them.

    Reward is maximal when a foot is airborne (lifted) and displaced sideways
    in the command direction:
        reward = gate * sum over feet of (lifted * |foot_y_body - lateral_gain*cmd_y|)
    where foot_y_body is the foot's y-offset from the body COM. When the foot
    steps sideways (matching cmd_y), the offset grows and the reward increases;
    merely lifting in place (no lateral displacement) gives little reward.

    ``min_y_shift`` (default 0.03 m = 3 cm): only feet whose |y offset from the
    body| EXCEEDS this minimum earn reward. This enforces a real sideways stride —
    a foot that only twitches a couple mm sideways, or sits at the body center
    while "lifting", earns nothing. Prevents fake micro-steps from gaming the term.

    Uses the foot's position (not contact force) so the policy cannot game it by
    momentarily unloading the wheel.
    """
    asset = env.scene[asset_cfg.name]
    # foot position in body frame
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]  # (B, K, 3)
    base_pos_w = asset.data.root_pos_w.unsqueeze(1)  # (B, 1, 3)
    base_quat_w = asset.data.root_quat_w
    rel_w = foot_pos_w - base_pos_w
    n_envs, n_feet = rel_w.shape[:2]
    base_quat_w_expanded = base_quat_w.unsqueeze(1).expand(-1, n_feet, -1).reshape(-1, 4)
    rel_w_flat = rel_w.reshape(-1, 3)
    rel_b = quat_apply_inverse(base_quat_w_expanded, rel_w_flat).reshape(n_envs, n_feet, 3)
    foot_y_body = rel_b[:, :, 1]  # (B, K) lateral offset in body frame

    # airborne detection by position (foot z above terrain)
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    ground_z = env.scene.env_origins[:, 2].unsqueeze(1)
    airborne = (foot_z - ground_z > lift_height).float()  # (B, K)

    cmd = env.command_manager.get_command(command_name)
    cmd_y = cmd[:, 1]
    lateral_gate = torch.sigmoid((torch.abs(cmd_y) - lift_threshold) / 0.05)
    # target foot y-offset aligned with commanded direction
    target_y = lateral_gain * cmd_y  # (B,)
    # reward airborne feet that are displaced sideways toward the target
    lateral_match = torch.exp(-torch.square(foot_y_body - target_y.unsqueeze(1)) / 0.01)
    # minimum real stride: |y offset| must exceed min_y_shift to count
    real_shift = (torch.abs(foot_y_body) > min_y_shift).float()
    reward = torch.sum(airborne * real_shift * lateral_match, dim=1)
    return lateral_gate * reward


def standing_drift_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base linear velocity when the command is ~zero.

    Zero-drift while standing (the robot creeping, e.g. backward drift) comes
    from the legs pushing asymmetrically to hold balance. Penalize |v_xy| only
    while the command is near zero, gated by a smooth sigmoid so the gradient
    stays continuous at the command boundary.
    """
    asset = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    standing_mask = torch.sigmoid((command_threshold - cmd) / 0.01)
    reward = standing_mask * body_vel
    return reward


def no_step_forward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    command_threshold: float = 0.15,
) -> torch.Tensor:
    """Penalise feet in the air (stepping) when there is NO lateral command.

    Gates on the y command: when lin_y/ang_z is quiet (standing or pure forward),
    stepping is penalised so the wheels stay planted and the robot doesn't jitter
    by alternating steps. During lateral (y) commands the gate is off, so the
    alternating-step gait used for side-stepping is allowed.
    """
    sensor: ContactSensor = env.scene[sensor_cfg.name]

    in_contact = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1) > 1.0  # (B, K)
    in_air = ~in_contact  # (B, K)

    cmd = env.command_manager.get_command(command_name)
    lat_quiet = torch.norm(cmd[:, 1:3], dim=1) < command_threshold  # no lateral/rotation
    # penalise number of airborne feet when no lateral command is active
    return in_air.float().sum(dim=-1) * lat_quiet.float()
