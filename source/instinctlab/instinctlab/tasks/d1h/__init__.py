import gymnasium as gym

from .config import agents, d1h_env_cfg, d1h_flat_env_cfg

##
# Register Gym environments. Each task is wired to standard Instinct-RL PPO.
##

gym.register(
    id="Instinct-Locomotion-Flat-D1H-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": d1h_flat_env_cfg.D1hFlatEnvCfg,
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_ppo_cfg:D1hFlatPPORunnerCfg",
    },
)

gym.register(
    id="Instinct-Locomotion-Flat-D1H-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": d1h_flat_env_cfg.D1hFlatEnvCfg_PLAY,
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_ppo_cfg:D1hFlatPPORunnerCfg",
    },
)

gym.register(
    id="Instinct-Locomotion-Rough-D1H-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": d1h_env_cfg.D1hEnvCfg,
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_ppo_cfg:D1hRoughPPORunnerCfg",
    },
)

gym.register(
    id="Instinct-Locomotion-Rough-D1H-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": d1h_env_cfg.D1hEnvCfg_PLAY,
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_ppo_cfg:D1hRoughPPORunnerCfg",
    },
)
