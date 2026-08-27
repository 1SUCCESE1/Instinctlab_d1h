"""TPPO agent config for the D1H locomotion tasks (distillation from DDT teacher).

Uses instinct_rl's TPPO with OnnxTeacher (DDT flat.onnx) as the teacher.
The student (InstinctLab PPO) learns from the teacher's actions via distillation loss.
"""

from isaaclab.utils import configclass

from instinctlab.utils.wrappers.instinct_rl import (
    InstinctRlActorCriticCfg,
    InstinctRlNormalizerCfg,
    InstinctRlOnPolicyRunnerCfg,
    InstinctRlPpoAlgorithmCfg,
)


@configclass
class D1hPolicyCfg(InstinctRlActorCriticCfg):
    init_noise_std = 1.0
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = "elu"


@configclass
class D1hAlgorithmCfg(InstinctRlPpoAlgorithmCfg):
    class_name = "TPPO"

    # PPO base params
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.01
    num_learning_epochs = 5
    num_mini_batches = 4
    learning_rate = 1e-3
    schedule = "adaptive"
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.0

    # TPPO-specific: distillation from DDT teacher
    # teacher_logdir, teacher_policy_class_name, teacher_policy, etc. are set
    # by train_distill.py (the teacher is OnnxTeacher, not a torch checkpoint)
    teacher_act_prob = "exp"          # schedule: decays from 1.0 to 0.0 over training
    update_times_scale = 500          # teacher act prob decays to ~0 over 500 iterations
    distillation_loss_coef = 1.0      # weight of distillation loss vs PPO loss
    distill_target = "mse_sum"        # MSE sum over action dims
    label_action_with_critic_obs = False   # use actor obs, not critic obs for teacher labels
    action_labels_from_sample = False


@configclass
class D1hNormalizersCfg:
    policy: InstinctRlNormalizerCfg = InstinctRlNormalizerCfg()
    critic: InstinctRlNormalizerCfg = InstinctRlNormalizerCfg()


@configclass
class D1hTPPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    policy: D1hPolicyCfg = D1hPolicyCfg()
    algorithm: D1hAlgorithmCfg = D1hAlgorithmCfg()
    normalizers: D1hNormalizersCfg = D1hNormalizersCfg()

    num_steps_per_env = 24
    max_iterations = 25000
    save_interval = 100
    log_interval = 10
    experiment_name = "d1h_locomotion_flat"
