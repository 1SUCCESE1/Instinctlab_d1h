"""D1H rough-ground env config (bipedal wheel-legged).

Ported from DDT_Lab: source/ddt_lab/ddt_lab/tasks/manager_based/locomotion/robots/d1h/rough_env_cfg.py
NP3O-specific parts (CostsCfg, policy history) are dropped; trained with standard instinct_rl PPO.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ViewerCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

import instinctlab.tasks.d1h.mdp as mdp  # isort: skip
from instinctlab.assets.d1h import D1H_CFG  # isort: skip
from instinctlab.envs import InstinctLabRLEnvCfg  # isort: skip

##
# Scene definition
##


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-2,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = D1H_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.10,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.5, 1.5), lin_vel_y=(-0.0, 0.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class D1HActionsCfg:
    # left leg: hip, thigh, calf
    fl_leg_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"],
        scale={
            "FL_hip_joint": 0.25,
            "FL_thigh_joint": 0.25,
            "FL_calf_joint": 0.25,
        },
        clip={".*": (-100.0, 100.0)},
        use_default_offset=True,
        preserve_order=True,
    )

    # left foot wheel: VELOCITY control — flat_19 (150447) config, the only version
    # that trained well under PPO (err_x 0.5, could stand and track). Velocity mode
    # gives the policy a direct speed target, which is what PPO can learn (unlike
    # pure torque control, which PPO could not learn to balance with).
    fl_foot_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["FL_foot_joint"],
        scale=5.0,
        clip={".*": (-100.0, 100.0)},
        use_default_offset=True,
        preserve_order=True,
    )

    # right leg: hip, thigh, calf
    fr_leg_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"],
        scale={
            "FR_hip_joint": 0.25,
            "FR_thigh_joint": 0.25,
            "FR_calf_joint": 0.25,
        },
        clip={".*": (-100.0, 100.0)},
        use_default_offset=True,
        preserve_order=True,
    )

    # right foot wheel: VELOCITY control — see FL_foot_vel comment.
    fr_foot_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["FR_foot_joint"],
        scale=5.0,
        clip={".*": (-100.0, 100.0)},
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), clip=(-100.0, 100.0), scale=0.25
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05), clip=(-100.0, 100.0), scale=1.0
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=(2.0, 2.0, 0.25),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_without_wheel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True),
                "wheel_asset_cfg": SceneEntityCfg("robot", joint_names=".*_foot_joint"),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0), scale=2.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100.0, 100.0), scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0), scale=1.0)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=(2.0, 2.0, 0.25),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_without_wheel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True),
                "wheel_asset_cfg": SceneEntityCfg("robot", joint_names=".*_foot_joint"),
            },
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)

        def __post_init__(self):
            pass

    @configclass
    class PrivCfg(ObsGroup):
        """Privileged physical parameters for the critic."""

        contact_state = ObsTerm(
            func=mdp.contact_state,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_foot"])},
            clip=(-1.0, 1.0),
            scale=1.0,
        )
        joint_kp_factor = ObsTerm(
            func=mdp.joint_kp_factor,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(0.0, 2.0),
            scale=1.0,
        )
        joint_kd_factor = ObsTerm(
            func=mdp.joint_kd_factor,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(0.0, 2.0),
            scale=1.0,
        )

    @configclass
    class ScannerCfg(ObsGroup):
        """Height-scan input for the critic / scan encoder."""

        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            scale=1.0,
        )

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    priv: PrivCfg = PrivCfg()
    scanner: ScannerCfg = ScannerCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 2.75),
            "dynamic_friction_range": (0.2, 2.75),
            "restitution_range": (0.0, 1.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-0.5, 2.0),
            "operation": "add",
            "recompute_inertia": True,
        },
    )

    add_base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.1, 0.1)},
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": (-10.0, 10.0),
            "torque_range": (-10.0, 10.0),
        },
    )

    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.2),
                "roll": (-3.14, 3.14),
                "pitch": (-3.14, 3.14),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (-0.5, 1.0),
            "velocity_range": (-0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (-1.0, 1.0)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP.

    Restored to match DDT_Lab dev branch rewards exactly.
    """

    # General
    is_terminated = RewTerm(func=mdp.is_terminated, weight=0.0)

    # -- task tracking. x uses split forward/backward kernels: deploy feedback says
    # flat_19 value: weight 8.0, simple track_lin_vel_x_exp with std sqrt(0.25)
    track_lin_vel_x_exp = RewTerm(
        func=mdp.track_lin_vel_x_exp,
        weight=8.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    # y 速度跟踪关闭: lin_vel_y 命令恒为 0, 从不出现真正的侧走。该奖励只是惩罚侧向
    # 漂移, 且部署端 165234 站立晃动严重。连同 lateral_lift_time/lateral_leg_lift 一起
    # 移除所有 y 轴奖励, 让站立/走行不承载侧向激励。
    track_lin_vel_y_lateral_exp = RewTerm(
        func=mdp.track_lin_vel_y_lateral_exp,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "std": 0.5,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_foot"]),
        },
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=4.0, params={"command_name": "base_velocity", "std": 0.5}
    )

    # velocity smoothness: penalise jerky base velocity (deploy feedback: speed
    # tracking discontinuous). exp(-|dv|/0.5), so a speed step costs real reward.
    vel_smoothness = RewTerm(func=mdp.vel_smoothness, weight=1.2)  # 0.8 -> 1.2 (2026-08-25): 压后退「滚-停-滚」的颠簸循环

    # NOTE: lateral_lift_time (air-time reward) removed — it rewarded "time lifted"
    # which conflicts with y-velocity tracking (an airborne foot produces no y
    # motion). Stride is driven by lateral_leg_lift (y-displacement) instead.

    # -- root penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.10)
    # split x/y: y (pitch rate) is the dominant torso-shake axis (fore-aft rocking).
    # weights kept low — heavy shake damping hurt tracking; ang_vel_y + pitch_rate_l2
    # both damp pitch rate, so keep each light.
    # ROLL is the dominant shake axis, not pitch: measured roll RMS was 1.08 rad/s
    # (~62 deg/s) vs pitch 0.45. All the earlier tuning targeted pitch terms and so
    # aimed at the wrong axis. ang_vel_x raised -0.10 -> -1.5 (share 0.36% -> ~5%).
    # NO command gate (see rewards.ang_vel_x_l2 docstring — gating it killed standing
    # learning in run 20260810_155717).
    # posture: 153024 baseline (ang_vel_x -1.5 / ang_vel_y -0.05 / flat -15) with the
    # single change the user requested — roll penalty -1.5 -> -2.0. The baseline was
    # the one that both learned to walk and kept shake controllable (roll 0.66); -2.0
    # is a mild tightening of the dominant shake axis without jumping to the -2.5
    # that (with tracking 15) froze walking.
    # NO lateral gate. Run 20260813_211857 gated roll on the lateral command
    # (release 0.5) and the policy could not stand (upward 2.6 vs 3.75 in flat_19):
    # early training spends most time under lateral commands, so the released roll
    # damping lets the robot fall. Keep full roll penalty always — strafe roll is
    # handled by the roll VELOCITY being a dynamic (not static lean) quantity.
    ang_vel_x_l2 = RewTerm(func=mdp.ang_vel_x_l2, weight=-2.0)
    # pitch 角速度惩罚: 与 pitch_rate_l2 同源(都是 root_ang_vel_b[:,1]², 仅差一个
    # upright 滤波), 二者叠加让 pitch 率惩罚难以单独调。关闭此条, 统一用
    # pitch_rate_l2(无滤波, 对站立更直接)控制 pitch 阻尼。
    ang_vel_y_l2 = RewTerm(func=mdp.ang_vel_y_l2, weight=0.0)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-15.0)
    # trunk level: penalize pitch (bum raised) to keep the torso horizontal.
    # REQUIRED — without pitch_target the robot cannot stand up.
    pitch_target_l2 = RewTerm(
        func=mdp.pitch_target_l2,
        # -30 回退 -20(2026-08-26): 部署反馈原地平衡抖动变严重。压 pitch 太狠 -> 策略
        # 更凶地纠轮子 -> 抖动(老坑)。-20 保持。后退问题不靠加强 pitch 惩罚解决。
        weight=-20.0,
    )
    # pitch rate: suppress torso pitch oscillation (-0.05 -> -0.3, 部署端pitch来回摆)
    pitch_rate_l2 = RewTerm(
        func=mdp.pitch_rate_l2,
        # -1.0 回退到 -0.7: 133242 部署端轮子前后调整幅度很大、晃动明显。压 pitch 速度
        # 过狠 -> 策略必须更凶地改轮速去抵消 pitch -> 轮子补偿更剧烈(注释里记过的坑)。
        # -0.7 是 105526 部署「好多了」的值。
        weight=-0.7,
    )
    # 直接奖励轮子输出重力平衡扭矩: tau_wheel ≈ m·g·(com_x - wheel_x)。轮足倒立摆
    # 的正确平衡原理——CoM 一偏, 轮子就滚过去接住重心。当前配置里 pitch 平衡全靠
    # pitch_target/pitch_rate/速度跟踪这些间接信号逼出来, 策略只能抖出平衡点
    # (部署端前后晃)。此条给轮子一个清晰的"接重心"梯度。
    # 必须用 exp 软形式 + 小权重(见 rewards.wheel_torque_balance docstring:
    # run 170753 用线性惩罚 weight 3.0 把预算打爆、站立崩了)。
    wheel_torque_balance = RewTerm(
        func=mdp.wheel_torque_balance,
        # 0.5 -> 0.8(2026-08-22): 停后晃动不减弱。raw 只有 ~0.18, 0.5 太弱, 轮子没在接重心。
        # 0.8 仍是 exp 软形式(不会像线性版 weight 3.0 崩)。
        weight=0.8,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_foot_joint"])},
    )
    pitch_command_alignment = RewTerm(
        func=mdp.pitch_command_alignment,
        weight=-4.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        # 2026-08-27: -7 -> -12。训练指标站 0.40 但确定性 play 站低(噪声掩盖均值)。
        # 加强 base_height 锚定, 让确定性策略也站得住 0.40。
        weight=-12.0,
        params={
            "target_height": 0.5,
        },
    )

    # -- joint penalties
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*(hip|thigh|calf)_joint"])},
    )
    joint_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        # flat_19 value: -0.01
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*(hip|thigh|calf)_joint"])},
    )
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        # restored to 153024's scope (.* all joints) and weight -2.5e-7
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*(hip|thigh|calf)_joint"])},
    )
    joint_vel_limits = RewTerm(
        func=mdp.joint_vel_limits,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_foot_joint"), "soft_ratio": 0.9},
    )
    joint_power = RewTerm(
        func=mdp.joint_power,
        weight=-2e-5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*(hip|thigh|calf)_joint"]),
        },
    )

    joint_mirror = RewTerm(
        func=mdp.joint_mirror,
        # restored to 153024's -30. Gated on the command (full when no lateral cmd,
        # off while strafing), so the strength is what kept legs together in x in the
        # baseline without killing the strafe gait.
        weight=-30.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            # gate on the command: full -30 when there is NO lateral (y) command —
            # i.e. standing still, driving forward/backward (x), or turning (z) —
            # where both legs should move together; fully off while strafing (y),
            # which needs the asymmetric alternating gait.
            "command_name": "base_velocity",
            "lateral_release": 0.5,
            "mirror_joints": [
                ["FL_hip_joint", "FR_hip_joint"],
                ["FL_thigh_joint", "FR_thigh_joint"],
                ["FL_calf_joint", "FR_calf_joint"],
            ],
        },
    )

    joint_pos_penalty = RewTerm(
        func=mdp.joint_pos_penalty,
        # -0.6 -> -0.3: -0.6 锁死了腿(部署端轮子剧烈前后滚平衡->pitch晃)。
        # 回退让腿能低频微调平衡, 腿位置准确性由 body_pos_to_feet_x/leg_activity 保证。
        weight=-0.2,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*(hip|thigh|calf)_joint"]),
            "stand_still_scale": 1.5,
            "velocity_threshold": 0.02,
            "command_threshold": 0.02,
        },
    )

    # -- Contact sensor
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["^(?!.*_foot).*"]), "threshold": 1.0},
    )
    contact_forces = RewTerm(
        func=mdp.contact_forces,
        weight=-0.0e-4,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_foot"]),
            "threshold": 400.0,
        },
    )

    # -- optional penalties
    upward = RewTerm(
        func=mdp.upward,
        weight=1.1,
    )

    # -- pose regularisation
    # Regression to run 20260808_131545: default_joint=-0.5 over all leg joints keeps
    # Strong default-pose anchoring (131545 value) combined with leg_activity=2.0:
    # legs stay near default (stable) but are actively driven to move when needed.
    default_joint_l2 = RewTerm(
        # 部署端反馈: 前进时腿频繁调整。加强默认位锚定, 让腿稳定在参考姿态附近。
        # -0.3 -> -0.6(仍弱于 -1.0 的"锁死", 保留一定自由度)。
        func=mdp.default_joint_l2,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*(hip|thigh|calf)_joint"])},
    )

    # 腾空时间奖励。DDT_Lab-dev 原版 weight=2.0, min_air_t=0.5。
    # 但轮足侧走是快速小步抬脚(~0.2s), 0.5s门槛太高导致奖励恒0。
    # 降min_air_t到0.1匹配轮足抬脚时间; 权重对齐DDT 2.0。
    # flat_19 value: disabled (weight 0)
    feet_air_time = RewTerm(
        func=mdp.reward_feet_air_time,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_foot"])},
    )

    # 头部碰撞惩罚
    collision_head = RewTerm(
        func=mdp.reward_collision_head,
        weight=-0.0,
    )

    # 大腿关节速度惩罚 (back to 0.0: -0.05 froze the thigh joints and stopped
    # walking learning; see joint_vel_l2 comment)
    dof_thigh_vel = RewTerm(
        func=mdp.reward_dof_thigh_vel,
        weight=-0.0,
    )

    # 机身不前后偏移两脚中点 (relaxed from 10.0/sigma0.01 so the legs can move;
    # base_height now enforces the standing height)
    body_pos_to_feet_x = RewTerm(
        func=mdp.reward_body_pos_to_feet_x,
        # flat_19 value: weight 2.0, sigma 0.05. weight 2.0 -> 1.0 (2026-08-26):
        # 后退策略用「轮滚+点头」而不用腿, 因为腿前摆(后退驱动)让机身偏离脚中点、
        # 被这项奖励惩罚。减半解除对后退腿前摆的压制。sigma 保持 0.05。
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sigma": 0.05,
        },
    )

    # 禁止两腿前后错开 (加强: -0.2 -> -1.0, DDT dev 值)
    # 之前 -0.2 太弱, 策略学会两腿交替前后叉腿站平衡(实测叉腿~4.7cm),
    # 部署端表现为左右脚不对称。-1.0 强制两腿 x 同步。
    body_feet_distance_x = RewTerm(
        func=mdp.reward_body_feet_distance_x,
        # 部署端反馈: 两腿前后错开(一前一后)。-3.0 -> -1.5 回退到 222130 能跑的值:
        # 翻倍版本是 232046 崩塌批次的一部分, 与趴地局部最优叠加后帮倒忙。
        weight=-1.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sigma": 0.1,
        },
    )

    # 两脚 x 方向速度同步: 抓"交替前后"失败模式。
    # 交替时两脚机体x速度相反, 速度差大; 位置惩罚抓不住(位置差随时间平均掉)。
    body_feet_vel_x = RewTerm(
        func=mdp.body_feet_vel_x,
        # 部署端反馈: 腿交替前后摆(两脚 x 速度相反)。-0.7 -> -1.0(2026-08-25):
        # 后退不连贯明显, 直接压两脚 x 速度差(交替步态)。
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # 固定两腿横向间距40cm (weight -0.8 -> -2.0 加强)。侧走时门控放开(交替步态
    # 需要腿张开), 站立/前进/后退时强制40cm。
    body_feet_distance_y = RewTerm(
        func=mdp.reward_body_feet_distance_y,
        # 234634 部署: 高速时腿左右分开。 -0.8 -> -1.5(2026-08-27) 压高速腿间距,
        # 低速/站立不变(-0.8 时就保持得好)。
        weight=-1.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sigma": 0.1,
            "desired_feet_distance": 0.4,
            "command_name": "base_velocity",
            "lateral_threshold": 0.05,
        },
    )

    # 左右腿横向对称 (0.5 -> 0.8: 部署端左右脚不对称, 加强对称。
    # 注意 weight=1.0 曾导致只学右侧走, 故取 0.8 平衡)
    body_symmetry_y = RewTerm(
        func=mdp.reward_body_symmetry_y,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sigma": 0.1,
        },
    )

    # 左右腿离地高度一致 (0.5 -> 0.8)
    body_symmetry_z = RewTerm(
        func=mdp.reward_body_symmetry_z,
        weight=0.8,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sigma": 0.1,
        },
    )

    # forward/backward leg swing for balance (COM offset drive, wheels balance)
    # 架构:「腿低频把轮子摆到位 + 轮子高频保pitch + 轮速辅助」。
    leg_activity = RewTerm(
        func=mdp.leg_activity,
        # 部署端反馈(2700轮导出): 前进时腿一前一后频繁调整。但上一轮把权重压到 0.5
        # + vel_penalty 0.05 后, 整个项变成净负值(实测 leg_activity≈-0.07), 前进驱动
        # 没了, 机器人学会"站着不动"(track_x 全程 0.01)。回退到 weight 1.5 + vel_penalty
        # 0.02。2026-08-25: 部署端腿调整仍多、后退不连贯, vel_penalty 0.02 -> 0.04
        # (weight 1.5 下仍为正, 保留驱动)进一步压腿速、平滑后退。weight 1.5 -> 2.0:
        # 后退必须靠腿驱动, 加强腿摆动奖励让「腿后退」比「轮滚+点头」更划算。
        # 2026-08-27 回退: weight 2.0/vel_penalty 0.06 把 track 从 6.78 压到 5.87、
        # 腿更活跃。回 1.5/0.04(232605 基线, 各项最好)。
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot"),
            "lean_gain": 0.07,
            "vel_penalty_gain": 0.04,
        },
    )

    # 侧走腾空奖励关闭: lin_vel_y 命令恒为 0, 侧走从不出现。该奖励在站立时仍有
    # ~5% 漏奖励, 与 165234 部署端站立晃动相关。连同 y 速度跟踪一起移除。
    lateral_lift_time = RewTerm(
        func=mdp.lateral_lift_time,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_foot"]),
            "lift_threshold": 0.15,
            "max_lift_time": 0.5,
        },
    )

    # 侧走抬脚奖励关闭: 同上。站立时 gate≈0.12 仍漏抬脚奖励, 会鼓励站立时抬脚/抖腿。
    lateral_leg_lift = RewTerm(
        func=mdp.lateral_leg_lift,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_foot"]),
            "lift_threshold": 0.1,
            "lift_height": 0.02,
            "lateral_gain": 0.5,
        },
    )

    # penalise stepping when no lateral command (standing / forward): keeps wheels
    # planted so the robot doesn't jitter by alternating steps. Side-stepping (y cmd)
    # turns this gate off, allowing the alternating-step gait.
    no_step_forward = RewTerm(
        func=mdp.no_step_forward,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_foot"]),
            "command_threshold": 0.15,
        },
    )

    # zero-drift: penalize base velocity when command is ~zero
    standing_drift_l2 = RewTerm(
        func=mdp.standing_drift_l2,
        # -5 -> -8(2026-08-22): 部署反馈 cmd=0 时向前 0.2 零飘压不住。0.2 漂移在 -5 只罚 -1.0/步。
        weight=-8.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # limb/torso jitter suppression (softened from -0.08 to -0.05)
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.02,
    )
    # NOTE: wheel_action_rate (-0.3) 已移除 — 133242/170412 部署「非常差」的共同嫌疑。
    # 把轮速命令磨钝后, 实机需要更激进的轮子响应, 反而晃、追不上。105526(无此项)部署
    # 「好多了」。轮子抖动交给 pitch_rate / wheel_acc 的弱惩罚压, 不再硬磨命令。

    # wheel acceleration penalty. -1e-6 -> -1e-7: 控制架构里轮子承担高频pitch平衡,
    # 需要能快速加减速。放松加速度惩罚让轮子响应更快(平衡优先于平滑)。
    wheel_acc_l2 = RewTerm(
        func=mdp.wheel_acc_l2,
        # 150447 baseline: -1e-7 (回退, -3e-7压住轮子反而加剧前后晃动)
        weight=-1e-7,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_ids=[3, 7]),
        },
    )



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )
    # 趴地终止: 020020 只有 time_out/出界, 机器人趴在地上也能混满 20 秒(净奖励还是正的),
    # 躺着完全不扣分 -> 学不会站立。base z < 0.25m 持续 >= 100 步(~1s) 才终止,
    # 平衡恢复中的瞬时下探不会刷重置, 蹲坐趴地才会。
    collapsed_to_ground = DoneTerm(
        func=mdp.collapsed_to_ground,
        params={"asset_cfg": SceneEntityCfg("robot"), "height_threshold": 0.25, "sustain_steps": 100},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class D1hMonitorCfg:
    pass


##
# Environment configuration
##


@configclass
class D1hEnvCfg(InstinctLabRLEnvCfg):
    """Configuration for the D1H locomotion velocity-tracking environment."""

    # Scene settings
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: D1HActionsCfg = D1HActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    monitors: D1hMonitorCfg = D1hMonitorCfg()
    viewer: ViewerCfg = ViewerCfg(
        eye=(2.0, 2.0, 0.5), lookat=(0.0, 0.0, 0.0), origin_type="asset_root", asset_name="robot"
    )

    # fmt: off
    joint_names = [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FL_foot_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FR_foot_joint",
    ]
    wheel_joint_names = [
        "FR_foot_joint", "FL_foot_joint",
    ]
    # fmt: on

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.0025
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # domain randomisation events (reset near standing height + pose perturbation)
        # NOTE: reset_root_state_uniform ADDS the z range to the default root z (0.40),
        #   so z here is a DELTA around the default. Old (0.0, 0.2) gave z = 0.40~0.60
        #   which lets the robot drop from a crouched pose. New delta keeps the robot
        #   near standing height and adds roll/pitch perturbation to learn recovery.
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.10, -0.02),
                "roll": (-0.15, 0.15),
                "pitch": (-0.15, 0.15),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }
        self.disable_zero_weight_rewards()

    def disable_zero_weight_rewards(self):
        """If the weight of rewards is 0, set rewards to None."""
        for attr in dir(self.rewards):
            if not attr.startswith("__"):
                reward_attr = getattr(self.rewards, attr)
                if not callable(reward_attr) and reward_attr.weight == 0:
                    setattr(self.rewards, attr, None)


@configclass
class D1hEnvCfg_PLAY(D1hEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
