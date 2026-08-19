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
    track_lin_vel_y_lateral_exp = RewTerm(
        func=mdp.track_lin_vel_y_lateral_exp,
        weight=12.0,
        params={
            "command_name": "base_velocity",
            "std": 0.5,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_foot"]),
        },
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=4.0, params={"command_name": "base_velocity", "std": 0.5}
    )

    # NOTE: wheel_torque_balance and vel_smoothness removed — not present in the
    # flat_19 (150447) config. flat_19 is the version that trained well under PPO.

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
    # pitch 角速度惩罚 -0.05 -> -0.2: 压住pitch来回摆(部署端反馈)。控制架构里
    # pitch平衡由轮子承担, 躯干自身不应高频摆动。
    ang_vel_y_l2 = RewTerm(func=mdp.ang_vel_y_l2, weight=-0.2)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-15.0)
    # trunk level: penalize pitch (bum raised) to keep the torso horizontal.
    # REQUIRED — without pitch_target the robot cannot stand up.
    pitch_target_l2 = RewTerm(
        func=mdp.pitch_target_l2,
        weight=-20.0,
    )
    # pitch rate: suppress torso pitch oscillation (-0.05 -> -0.3, 部署端pitch来回摆)
    pitch_rate_l2 = RewTerm(
        func=mdp.pitch_rate_l2,
        # 150447 baseline: -0.3 (回退, -0.5压住pitch反而让轮子补偿更剧烈)
        weight=-0.3,
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
        weight=-5.0,
        params={
            "target_height": 0.35,
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
            "lateral_release": 0.0,
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
        weight=-0.3,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*(hip|thigh|calf)_joint"]),
            "stand_still_scale": 1,
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
        # -1.0 -> -0.5: 默认位是「参考姿态」不是「锁定」。再放松一档, 让腿能
        # 更自由地大幅离开默认位去定位轮子(部署端腿太死, 靠pitch调姿态)。
        func=mdp.default_joint_l2,
        weight=-0.5,
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
        # flat_19 value: weight 2.0, sigma 0.05
        weight=2.0,
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
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sigma": 0.1,
        },
    )

    # 两脚 x 方向速度同步: 抓"交替前后"失败模式。
    # 交替时两脚机体x速度相反, 速度差大; 位置惩罚抓不住(位置差随时间平均掉)。
    body_feet_vel_x = RewTerm(
        func=mdp.body_feet_vel_x,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # 固定两腿横向间距40cm (weight -0.8 -> -2.0 加强)。侧走时门控放开(交替步态
    # 需要腿张开), 站立/前进/后退时强制40cm。
    body_feet_distance_y = RewTerm(
        func=mdp.reward_body_feet_distance_y,
        # 150447 baseline: -0.8 (回退, -2.0约束太强)。腿距保留30cm
        weight=-0.8,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sigma": 0.1,
            "desired_feet_distance": 0.4,
        },
    )

    # 左右腿横向对称 (0.5 -> 0.8: 部署端左右脚不对称, 加强对称。
    # 注意 weight=1.0 曾导致只学右侧走, 故取 0.8 平衡)
    body_symmetry_y = RewTerm(
        func=mdp.reward_body_symmetry_y,
        weight=0.8,
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
    # 2.0 -> 1.5: 用户要求不再提高, 位置精准度靠joint_pos_pen/body_pos_to_feet加强。
    leg_activity = RewTerm(
        func=mdp.leg_activity,
        # flat_19 value: 2.0
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot"),
            "lean_gain": 0.07,
            "vel_penalty_gain": 0.003,
        },
    )

    # lateral stepping: reward genuine sideways foot DISPLACEMENT (y-shift driven).
    # flat_19 value: 1.5, disabled — constant bias provides no learning gradient
    lateral_leg_lift = RewTerm(
        func=mdp.lateral_leg_lift,
        weight=1.5,
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
        # flat_19 value: -5.0
        weight=-5.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # limb/torso jitter suppression (softened from -0.08 to -0.05)
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.05,
    )

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
