"""
Customized D1H (bipedal wheel-legged) asset for Isaac Sim.

Ported from DDT_Lab: source/ddt_lab/ddt_lab/assets/ddt_robot.py (DDT_D1H_CFG).
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

__file_dir__ = os.path.dirname(os.path.realpath(__file__))

D1H_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=os.path.join(__file_dir__, "resources/d1h_description/urdf/robot.urdf"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.40),
        joint_pos={
            ".*L_hip_joint": 0.0,
            ".*R_hip_joint": 0.0,
            ".*_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
            ".*_foot_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[".*(hip|thigh|calf)_joint"],
            effort_limit=60.0,
            velocity_limit=20.0,
            stiffness=60.0,
            damping=2.0,
            friction=0.15,
            armature=0.0535,
            min_delay=0,  # physics time steps (min: 2.0*0=0.0ms)
            max_delay=4,  # physics time steps (max: 2.0*4=8.0ms)
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_foot_joint"],
            effort_limit_sim=12.0,
            velocity_limit_sim=50.0,
            # P torque control, matching ddt_rl_isaacgym: stiffness 10 N·m/rad,
            # damping 0.5 N·m·s/rad. The policy outputs a wheel position target
            # (JointPositionActionCfg); torque = 10*(q_target-q) - 0.5*w. This is the
            # mature wheel-legged control that lets the wheels balance passively via
            # damping while the policy supplies drive — replacing the velocity mode
            # (stiffness 0) that fought balance and capped tracking at err_x~0.48.
            stiffness=10.0,
            damping=0.5,
            friction=0.02,
        ),
    },
)
