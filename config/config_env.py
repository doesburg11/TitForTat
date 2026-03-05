"""Central runtime/environment settings for scripts/tune_eval_rllib.py.

Edit this file to control tune/eval behavior without CLI flags.
"""

config_env = {
    # Paths
    "ppo_config": "config/config_ppo.py",
    "checkpoint_dir": "checkpoints/sequential_pd_ppo",
    "from_checkpoint": None,
    "metrics_out": None,
    # Evaluation
    "eval_episodes": 100,
    # Environment horizon
    "n_sequential_games": 100,
    # Reproducibility
    "seed": None,
}

# Sweep-only settings for scripts/sweep_n_sequential_pd.py.
config_sweep_n_sequential_pd = {
    "n_sequential_games_values": [
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        45,
        50,
        55,
        60,
        65,
        70,
        75,
        80,
        85,
        90,
        95,
        100,
    ],
    "output_dir": "checkpoints/sweep_n_sequential_pd",
    "python_executable": None,
    "num_seeds": 20,
    "seed_start": 0,
    "ci_level": 0.95,
    "hypothesis_test_alpha": 0.05,
    "hypothesis_test_bootstrap_samples": 20000,
    "hypothesis_test_bootstrap_seed": 0,
    "hypothesis_test_correction": "holm",
}

# Sweep-only settings for scripts/stability_sweep.py.
config_stability_sweep = {
    "num_seeds": 5,
    "seed_start": 0,
    "output_dir": "checkpoints/stability_sweep",
    "python_executable": None,
    "ppo_config": "config/config_ppo.py",
    "eval_episodes": 100,
    "n_sequential_games": 50,
    "max_reward_cv": 0.15,
    "max_cooperation_std": 0.10,
    "max_rounds_cv": 0.10,
    "max_player_reward_gap": 1.0,
    "run_defection_gain_check": True,
    "defection_gain_episodes": 50,
    "defection_gain_tol": 1e-9,
}

# Settings for scripts/check_defection_gain.py.
config_defection_gain_check = {
    "checkpoint": "latest",
    "checkpoint_root": "checkpoints/sequential_pd_ppo",
    "n_sequential_games": 100,
    "episodes": 100,
    "seed": None,
    "output_json": None,
    "gain_tol": 1e-9,
}
