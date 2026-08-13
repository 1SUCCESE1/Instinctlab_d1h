"""PPO agent config for the D1H locomotion tasks.

Hyperparameters follow DDT_Lab's D1 rsl_rl PPO config; the config format is
Instinct-RL's (same as the InstinctLab locomotion tasks).
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
    class_name = "PPO"
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


@configclass
class D1hNormalizersCfg:
    policy: InstinctRlNormalizerCfg = InstinctRlNormalizerCfg()
    critic: InstinctRlNormalizerCfg = InstinctRlNormalizerCfg()


@configclass
class D1hFlatPPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    policy: D1hPolicyCfg = D1hPolicyCfg()
    algorithm: D1hAlgorithmCfg = D1hAlgorithmCfg()
    normalizers: D1hNormalizersCfg = D1hNormalizersCfg()

    num_steps_per_env = 24
    # 15000 -> 25000: 命令范围扩到±3m/s后, 高速段需要更长训练收敛
    max_iterations = 25000
    save_interval = 100
    log_interval = 10
    experiment_name = "d1h_locomotion_flat"

    load_run = None

    def __post_init__(self):
        super().__post_init__()  # type: ignore
        self.resume = self.load_run is not None
        self.run_name = ""


@configclass
class D1hRoughPPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    policy: D1hPolicyCfg = D1hPolicyCfg()
    algorithm: D1hAlgorithmCfg = D1hAlgorithmCfg()
    normalizers: D1hNormalizersCfg = D1hNormalizersCfg()

    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    log_interval = 10
    experiment_name = "d1h_locomotion_rough"

    load_run = None

    def __post_init__(self):
        super().__post_init__()  # type: ignore
        self.resume = self.load_run is not None
        self.run_name = ""
