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


def _base_height_sigma(env: ManagerBasedRLEnv, target_height: float = 0.40) -> torch.Tensor:
    """Height gate that scales a tracking reward by how tall the robot is.

    Saturating ramp up to ``target_height`` (0.40 m): tracking reward ramps from 0 at
    0.28 m to full at 0.40 m and stays 1.0 above. The bell (peak at 0.40) let the
    deterministic policy sink to ~0.35 m because its upward pull below the target was
    weak; the ramp makes crouching cost much more tracking reward, pushing the policy
    up to 0.40, while being flat above stops the overshoot the sigmoid gate caused
    (stood ~0.52 m).
    """
    asset: RigidObject = env.scene["robot"]
    z = asset.data.root_pos_w[:, 2]
    gate = (z - 0.28) / (target_height - 0.28)
    return torch.clamp(gate, 0.0, 1.0)


def collapsed_to_ground(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_threshold: float = 0.25,
    sustain_steps: int = 100,
) -> torch.Tensor:
    """Terminate when the base stays collapsed below ``height_threshold`` too long.

    Run 20260821_020020: with only time_out / terrain_out_of_bounds terminations a
    collapsed robot idles on the ground for the full 20 s episode collecting a
    comfortable net-positive reward (upward +3.78 dominates), so lying down costs
    nothing at the episode level. Fires when the base has been below
    ``height_threshold`` (well under the 0.40 target and the ~0.47 natural standing
    height) for ``sustain_steps`` steps (~1 s): transient dips during balance
    recovery do not spam resets, but a settled squat does.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    low = asset.data.root_pos_w[:, 2] < height_threshold
    if not hasattr(env, "_collapse_steps"):
        env._collapse_steps = torch.zeros_like(low, dtype=torch.float)
    env._collapse_steps[low] += 1.0
    env._collapse_steps[~low] = 0.0
    return env._collapse_steps >= sustain_steps


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


def ang_vel_x_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str | None = None,
    lateral_release: float = 0.5,
) -> torch.Tensor:
    """Penalize x-axis (roll) base angular velocity using L2 squared kernel.

    Gated on the lateral command: side-stepping REQUIRES the torso to lean into the
    roll axis (shifting the COM sideways), so roll velocity during a strafe is
    legitimate and should not be penalised as hard. With ``command_name`` set, the
    penalty applies at full strength when there is no lateral command (standing /
    driving forward) and is scaled to ``lateral_release`` while strafing.

    Note: keep ``lateral_release`` > 0 (NOT 0). Run 20260810_155717 set the gate to
    fully release roll during strafing and the policy stopped learning to stand
    (upward collapsed) because early training spends most time under lateral
    commands. A partial release (0.5) keeps enough roll damping to learn balance
    while allowing the strafe lean.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_ang_vel_b[:, 0])
    if command_name is not None:
        cmd_y = env.command_manager.get_command(command_name)[:, 1]
        y_gate = torch.sigmoid((torch.abs(cmd_y) - 0.05) / 0.02)
        reward = reward * (1.0 - y_gate * (1.0 - lateral_release))
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
    lateral_only: bool = False,
) -> torch.Tensor:
    """Reward airborne feet (taken steps) when a velocity command is active.

    Uses ``last_air_time`` (the duration of the most recent airborne phase, recorded
    on landing) rather than ``current_air_time``. For a WHEELED robot the wheel
    keeps tangential (x/y) contact force while rolling, so the sensor's internal
    ``is_contact`` (norm of all 3 force axes) stays True even when the leg is lifted
    and ``current_air_time`` is reset to 0 every frame — the reward would be 0 even
    though the leg did lift. ``last_air_time`` captures the air time at the moment
    the foot re-contacts, which is exactly the lift duration of the step.

    With ``lateral_only=True`` the reward is gated on the LATERAL (y) command only —
    standing/forward motion earns no air-time reward. This keeps the robot from
    idle-stepping in place while standing (the wheels should stay planted except
    when side-stepping).
    """
    contact_sensor = env.scene[sensor_cfg.name]
    air_time = getattr(contact_sensor.data, "last_air_time", None)
    if air_time is None:
        air_time = contact_sensor.data.current_air_time
    air_time = air_time[:, sensor_cfg.body_ids]

    # reward a foot that just landed after being airborne for at least min_air_t
    landed = (air_time > min_air_t).float()
    rew_airTime = torch.sum(landed, dim=-1)

    cmd = env.command_manager.get_command(command_name)
    if lateral_only:
        # gate on |cmd_y| only (side-stepping); standing/forward earns nothing
        lateral_mask = torch.abs(cmd[:, 1]) > 0.1
        rew_airTime *= lateral_mask.float()
    else:
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
    """Reward forward/backward leg swing when a velocity command is active.

    Encourages the legs to swing the feet ahead/behind the COM (pure
    fore-aft motion) to generate drive, rather than bobbing up/down (which
    changes the base height). Reward is maximized when the foot x-offset
    from the body matches the commanded lean direction:
        cmd>0 (forward)  -> feet swung backward (x offset matches -lean_gain*cmd)
        cmd<0 (reverse)  -> feet swung forward
    Gated on the command so standing still does not reward leg motion.

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
    x_mean = torch.mean(rel_b[:, :, 0], dim=1)

    cmd = env.command_manager.get_command(command_name)
    cmd_x = cmd[:, 0]
    desired = -lean_gain * cmd_x
    err = torch.abs(x_mean - desired)

    cmd_mag = torch.linalg.norm(cmd[:, :3], dim=1)
    moving_gate = torch.sigmoid((cmd_mag - command_threshold) / 0.05)
    reward = moving_gate * torch.exp(-err / 0.05)

    if vel_penalty_gain > 0.0:
        leg_ids = asset_cfg.joint_ids
        if leg_ids is None:
            leg_ids = [i for i, name in enumerate(asset.joint_names) if "_foot_joint" not in name]
        vel = asset.data.joint_vel[:, leg_ids]
        reward = reward - vel_penalty_gain * torch.mean(torch.square(vel), dim=1)
    return reward


def wheel_torque_balance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*_foot_joint"]),
) -> torch.Tensor:
    """Reward the wheel torque cancelling the gravity (overturning) torque.

    Core balance principle: the robot balances when the wheel torque equals the
    gravity torque about the wheel contact. For an inverted pendulum the gravity
    torque is
        tau_grav = m_total * g * (com_x_w - wheel_x_w)
    where (com_x_w - wheel_x_w) is the horizontal COM offset from the wheel
    contact. The wheel actuator produces its own torque (applied_torque). When
    they cancel the torso stays level.

    Reward = exp(-|tau_wheel - tau_grav| / sigma): directly teaches the policy
    to make the wheel output exactly the overturning torque — no reliance on the
    indirect speed-tracking signal. ``asset_cfg`` selects the wheel joints.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    device = asset.data.applied_torque.device
    wheel_ids = asset_cfg.joint_ids
    if wheel_ids is None:
        wheel_ids = [
            i for i, name in enumerate(asset.joint_names) if "_foot_joint" in name
        ]

    # total robot mass (sum over link default masses)
    m_total = asset.data.default_mass.sum(dim=-1).to(device)  # (B,)

    # horizontal COM offset from the wheel contact (world x)
    foot_body_names = ["FL_foot", "FR_foot"]
    foot_ids = [asset.body_names.index(name) for name in foot_body_names]
    wheel_x_w = asset.data.body_pos_w[:, foot_ids, 0].to(device)  # (B, 2)
    wheel_x_mean = torch.mean(wheel_x_w, dim=1)
    com_x_w = asset.data.root_com_pose_w[:, 0].to(device)
    d = com_x_w - wheel_x_mean  # (B,)

    g = torch.tensor(9.81, device=device, dtype=m_total.dtype)
    tau_grav = m_total * g * d  # (B,)

    # actual wheel torque (sum of both wheels)
    tau_wheel = asset.data.applied_torque[:, wheel_ids].sum(dim=-1).to(device)  # (B,)

    # exp reward (weak helper), NOT a hard linear penalty. Run 170753 showed that
    # -|dtau| at weight 3.0 dominated the budget (penalty ~-11), the policy could
    # not satisfy exact torque matching under P control, and standing collapsed
    # (upward 0.07, noise_std exploded to 6). exp keeps torque-matching as a soft
    # bonus that rewards improvement near the target without punishing the policy
    # into giving up when it cannot match exactly. Keep the weight small.
    # sigma 2.0 -> 4.0: 2.0 太紧(相对 12Nm 轮子 effort limit), 奖励贴地(实测~0.03),
    # 梯度太弱教不会轮子平衡。放宽到 4.0 让"接重心"信号可被挣到, 压住低频前后晃。
    reward = torch.exp(-torch.abs(tau_wheel - tau_grav) / 4.0)
    reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2].to(device), 0, 0.7) / 0.7
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
        y_gate = torch.sigmoid((torch.abs(cmd_y) - lateral_threshold) / 0.05)
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
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=[".*_foot"]),
    lift_threshold: float = 0.1,
    lift_height: float = 0.02,
    lateral_gain: float = 0.5,
) -> torch.Tensor:
    """Reward foot lift during lateral velocity commands.

    Detects lift via contact force: foot is airborne when net force z < 1N.
    Much more reliable than z-height on rough terrain.
    Gated on |cmd_y| so standing still does not reward leg motion.
    """
    contact_sensor = env.scene[sensor_cfg.name]
    net_fz = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]  # (B, K)
    airborne = (net_fz.abs() < 1.0).float()  # 1 where foot is off ground
    lift = torch.sum(airborne, dim=1)  # count of airborne feet
    cmd = env.command_manager.get_command(command_name)
    lateral_gate = torch.sigmoid((torch.abs(cmd[:, 1]) - lift_threshold) / 0.05)
    return lateral_gate * lateral_gain * lift


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
