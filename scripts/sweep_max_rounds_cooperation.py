#!/usr/bin/env python3
"""Sweep max_rounds values and plot cooperation rates for both players."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


ENV_CONFIG_ENVVAR = "SEQUENTIAL_PD_ENV_CONFIG"
DEFAULT_ENV_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config_env.py"
DEFAULT_ROUNDS = [ 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100 ]
TARGET_EPISODES_PER_UPDATE = 64
MINIBATCH_DIVISOR = 8
MIN_MINIBATCH_SIZE = 128
MIN_TRAIN_BATCH_SIZE = 1024
DEFAULT_SWEEP_CONFIG = {
    "num_seeds": 1,
    "seed_start": 0,
    "ci_level": 0.95,
}
SWEEP_CONFIG_VAR = "config_sweep_max_rounds"


def _resolve_existing_file(path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path.cwd() / candidate).resolve()


def _load_config_env(path: str) -> tuple[Dict, str, Dict]:
    resolved_path = _resolve_existing_file(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Environment config file not found: {resolved_path}")

    module_name = f"_sequential_pd_env_config_sweep_{abs(hash(str(resolved_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, resolved_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from: {resolved_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "config_env"):
        raise ValueError(f"Environment config file must define `config_env`: {resolved_path}")
    config_env = getattr(module, "config_env")
    if not isinstance(config_env, dict):
        raise TypeError(
            f"`config_env` must be a dict, got {type(config_env).__name__} in {resolved_path}"
        )
    raw_sweep_config = getattr(module, SWEEP_CONFIG_VAR, {})
    if raw_sweep_config is None:
        raw_sweep_config = {}
    if not isinstance(raw_sweep_config, dict):
        raise TypeError(
            f"`{SWEEP_CONFIG_VAR}` must be a dict, got "
            f"{type(raw_sweep_config).__name__} in {resolved_path}"
        )
    unknown_sweep_keys = sorted(set(raw_sweep_config.keys()) - set(DEFAULT_SWEEP_CONFIG.keys()))
    if unknown_sweep_keys:
        raise ValueError(f"Unknown keys in `{SWEEP_CONFIG_VAR}`: {unknown_sweep_keys}")
    sweep_config = dict(DEFAULT_SWEEP_CONFIG)
    sweep_config.update(raw_sweep_config)
    return dict(config_env), str(resolved_path), sweep_config


def _load_config_ppo(path: str) -> tuple[Dict, str]:
    resolved_path = _resolve_existing_file(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"PPO config file not found: {resolved_path}")

    module_name = f"_sequential_pd_ppo_config_sweep_{abs(hash(str(resolved_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, resolved_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from: {resolved_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "config_ppo"):
        raise ValueError(f"PPO config file must define `config_ppo`: {resolved_path}")
    config_ppo = getattr(module, "config_ppo")
    if not isinstance(config_ppo, dict):
        raise TypeError(
            f"`config_ppo` must be a dict, got {type(config_ppo).__name__} in {resolved_path}"
        )
    return dict(config_ppo), str(resolved_path)


def _write_config_env(path: Path, config_env: Dict) -> None:
    path.write_text(f"config_env = {repr(config_env)}\n", encoding="utf-8")


def _write_config_ppo(path: Path, config_ppo: Dict) -> None:
    path.write_text(f"config_ppo = {repr(config_ppo)}\n", encoding="utf-8")


def _parse_rounds(values: str) -> List[int]:
    rounds = []
    for token in values.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f"max_rounds values must be > 0, got {value}")
        rounds.append(value)
    if not rounds:
        raise ValueError("No rounds provided.")
    return rounds


def _python_executable(cli_value: str | None) -> str:
    if cli_value:
        return cli_value
    project_python = Path(".conda/bin/python").resolve()
    if project_python.exists():
        return str(project_python)
    return sys.executable


def _ensure_matplotlib_available() -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install with: python -m pip install matplotlib"
        ) from exc


def _plot_results(results: List[Dict], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rounds = [item["max_rounds"] for item in results]
    coop_p1_mean = [
        float(item["cooperation_player_1_mean"])
        if item["cooperation_player_1_mean"] is not None
        else float("nan")
        for item in results
    ]
    coop_p2_mean = [
        float(item["cooperation_player_2_mean"])
        if item["cooperation_player_2_mean"] is not None
        else float("nan")
        for item in results
    ]
    coop_p1_low = [
        float(item["cooperation_player_1_ci_low"])
        if item["cooperation_player_1_ci_low"] is not None
        else float("nan")
        for item in results
    ]
    coop_p1_high = [
        float(item["cooperation_player_1_ci_high"])
        if item["cooperation_player_1_ci_high"] is not None
        else float("nan")
        for item in results
    ]
    coop_p2_low = [
        float(item["cooperation_player_2_ci_low"])
        if item["cooperation_player_2_ci_low"] is not None
        else float("nan")
        for item in results
    ]
    coop_p2_high = [
        float(item["cooperation_player_2_ci_high"])
        if item["cooperation_player_2_ci_high"] is not None
        else float("nan")
        for item in results
    ]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    line_p1, = ax.plot(rounds, coop_p1_mean, marker="o", linewidth=2.4, label="Player 1 mean")
    line_p2, = ax.plot(rounds, coop_p2_mean, marker="s", linewidth=2.4, label="Player 2 mean")
    ax.fill_between(
        rounds,
        coop_p1_low,
        coop_p1_high,
        color=line_p1.get_color(),
        alpha=0.18,
        label="Player 1 confidence band",
    )
    ax.fill_between(
        rounds,
        coop_p2_low,
        coop_p2_high,
        color=line_p2.get_color(),
        alpha=0.18,
        label="Player 2 confidence band",
    )
    ax.set_title("Sequential Iterated Prisoner's Dilemma")
    ax.set_xlabel("number of repeated prisoner's dilemma games")
    ax.set_ylabel("cooperation rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(rounds)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _round_down_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        return value
    return max(multiple, (value // multiple) * multiple)


def _scaled_ppo_batch_settings(max_rounds: int) -> Dict[str, int]:
    # Simultaneous-action env emits roughly 1 transition per game round.
    approx_episode_steps = int(max_rounds)
    train_batch_size = max(
        MIN_TRAIN_BATCH_SIZE, TARGET_EPISODES_PER_UPDATE * approx_episode_steps
    )
    raw_minibatch = max(MIN_MINIBATCH_SIZE, train_batch_size // MINIBATCH_DIVISOR)
    minibatch_size = min(train_batch_size, _round_down_to_multiple(raw_minibatch, 32))
    num_epochs = 10 if train_batch_size >= 8192 else 15
    return {
        "train_batch_size_per_learner": int(train_batch_size),
        "minibatch_size": int(minibatch_size),
        "num_epochs": int(num_epochs),
    }


def _timestamp_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _mean_confidence_interval(values: List[float], ci_level: float) -> Dict[str, float | None]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "sem": None,
            "ci_low": None,
            "ci_high": None,
            "ci_half_width": None,
        }
    mean_value = float(statistics.fmean(values))
    if len(values) == 1:
        return {
            "n": 1,
            "mean": mean_value,
            "std": 0.0,
            "sem": 0.0,
            "ci_low": mean_value,
            "ci_high": mean_value,
            "ci_half_width": 0.0,
        }

    std_value = float(statistics.stdev(values))
    sem_value = std_value / math.sqrt(len(values))
    z_value = statistics.NormalDist().inv_cdf(0.5 + (ci_level / 2.0))
    half_width = z_value * sem_value
    return {
        "n": len(values),
        "mean": mean_value,
        "std": std_value,
        "sem": sem_value,
        "ci_low": mean_value - half_width,
        "ci_high": mean_value + half_width,
        "ci_half_width": half_width,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rounds",
        type=str,
        default=",".join(str(x) for x in DEFAULT_ROUNDS),
        help="Comma-separated max_rounds values to sweep.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/max_rounds_cooperation_sweep",
        help="Directory for per-round runs, summary JSON, and plot.",
    )
    parser.add_argument(
        "--env-config",
        type=str,
        default=None,
        help=(
            "Base config_env file path. Defaults to SEQUENTIAL_PD_ENV_CONFIG when set, "
            "else config/config_env.py."
        ),
    )
    parser.add_argument(
        "--python-executable",
        type=str,
        default=None,
        help="Python executable used to launch scripts/tune_eval_rllib.py.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    _ensure_matplotlib_available()
    rounds = _parse_rounds(args.rounds)
    run_timestamp = _timestamp_token()

    if args.env_config is not None:
        env_config_path = args.env_config
    elif os.environ.get(ENV_CONFIG_ENVVAR):
        env_config_path = os.environ[ENV_CONFIG_ENVVAR]
    else:
        env_config_path = str(DEFAULT_ENV_CONFIG_PATH)

    base_env_config, resolved_env_config_path, sweep_config = _load_config_env(env_config_path)
    if "ppo_config" not in base_env_config:
        raise ValueError("config_env must define `ppo_config` for sweep runs.")
    base_ppo_config, resolved_ppo_config_path = _load_config_ppo(str(base_env_config["ppo_config"]))
    num_seeds = int(sweep_config["num_seeds"])
    seed_start = int(sweep_config["seed_start"])
    ci_level = float(sweep_config["ci_level"])
    if num_seeds <= 0:
        raise ValueError("config_sweep_max_rounds.num_seeds must be > 0")
    if not (0.0 < ci_level < 1.0):
        raise ValueError("config_sweep_max_rounds.ci_level must be in (0, 1)")
    seeds = list(range(seed_start, seed_start + num_seeds))
    python_exe = _python_executable(args.python_executable)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_episodes = int(base_env_config.get("eval_episodes", 0))
    if eval_episodes <= 0:
        # Cooperation rates require actual evaluation episodes.
        base_env_config["eval_episodes"] = 1

    results = []
    for max_rounds in rounds:
        run_dir = output_dir / f"max_rounds_{max_rounds}"
        run_dir.mkdir(parents=True, exist_ok=True)
        per_seed = []
        coop_values_p1: List[float] = []
        coop_values_p2: List[float] = []
        scaled_settings = _scaled_ppo_batch_settings(max_rounds)

        for seed in seeds:
            seed_dir = run_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_dir = seed_dir / f"checkpoint_{run_timestamp}"
            metrics_path = seed_dir / f"metrics_{run_timestamp}.json"
            env_config_run_path = seed_dir / f"config_env_{run_timestamp}.py"
            ppo_config_run_path = seed_dir / f"config_ppo_{run_timestamp}.py"

            config_env = dict(base_env_config)
            config_ppo = dict(base_ppo_config)
            config_ppo.update(scaled_settings)
            _write_config_ppo(ppo_config_run_path, config_ppo)

            config_env["ppo_config"] = str(ppo_config_run_path)
            config_env["max_rounds"] = int(max_rounds)
            config_env["min_rounds"] = min(int(config_env["min_rounds"]), int(max_rounds))
            config_env["seed"] = int(seed)
            config_env["checkpoint_dir"] = str(checkpoint_dir)
            config_env["metrics_out"] = str(metrics_path)
            config_env["from_checkpoint"] = None

            _write_config_env(env_config_run_path, config_env)

            env = os.environ.copy()
            env[ENV_CONFIG_ENVVAR] = str(env_config_run_path)
            cmd = [python_exe, "scripts/tune_eval_rllib.py"]

            print(f"[sweep] running max_rounds={max_rounds} seed={seed}")
            subprocess.run(cmd, check=True, env=env)

            run_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            eval_summary = run_metrics.get("eval_summary") or {}
            cooperation_rate = eval_summary.get("cooperation_rate") or {}
            coop_p1 = cooperation_rate.get("player_1")
            coop_p2 = cooperation_rate.get("player_2")
            if coop_p1 is not None:
                coop_values_p1.append(float(coop_p1))
            if coop_p2 is not None:
                coop_values_p2.append(float(coop_p2))

            per_seed.append(
                {
                    "seed": seed,
                    "cooperation_player_1": coop_p1,
                    "cooperation_player_2": coop_p2,
                    "metrics_path": str(metrics_path),
                    "checkpoint_path": run_metrics.get("checkpoint_path"),
                    "config_env_path": str(env_config_run_path),
                    "config_ppo_path": str(ppo_config_run_path),
                }
            )

        stats_p1 = _mean_confidence_interval(coop_values_p1, ci_level)
        stats_p2 = _mean_confidence_interval(coop_values_p2, ci_level)
        results.append(
            {
                "max_rounds": max_rounds,
                "num_seeds": len(seeds),
                "cooperation_player_1_mean": stats_p1["mean"],
                "cooperation_player_1_ci_low": stats_p1["ci_low"],
                "cooperation_player_1_ci_high": stats_p1["ci_high"],
                "cooperation_player_1_ci_half_width": stats_p1["ci_half_width"],
                "cooperation_player_1_std": stats_p1["std"],
                "cooperation_player_2_mean": stats_p2["mean"],
                "cooperation_player_2_ci_low": stats_p2["ci_low"],
                "cooperation_player_2_ci_high": stats_p2["ci_high"],
                "cooperation_player_2_ci_half_width": stats_p2["ci_half_width"],
                "cooperation_player_2_std": stats_p2["std"],
                "train_batch_size_per_learner": scaled_settings["train_batch_size_per_learner"],
                "minibatch_size": scaled_settings["minibatch_size"],
                "num_epochs": scaled_settings["num_epochs"],
                "per_seed": per_seed,
            }
        )

    plot_path = output_dir / f"cooperation_vs_max_rounds_{run_timestamp}.png"
    _plot_results(results, plot_path)

    summary = {
        "rounds": rounds,
        "base_env_config_path": resolved_env_config_path,
        "base_ppo_config_path": resolved_ppo_config_path,
        "run_timestamp": run_timestamp,
        "sweep_config": sweep_config,
        "seeds": seeds,
        "ci_level": ci_level,
        "batch_scaling": {
            "target_episodes_per_update": TARGET_EPISODES_PER_UPDATE,
            "minibatch_divisor": MINIBATCH_DIVISOR,
            "min_train_batch_size": MIN_TRAIN_BATCH_SIZE,
            "min_minibatch_size": MIN_MINIBATCH_SIZE,
        },
        "output_dir": str(output_dir),
        "plot_path": str(plot_path),
        "results": results,
    }
    summary_path = output_dir / f"summary_{run_timestamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[sweep] wrote summary: {summary_path}")
    print(f"[sweep] wrote plot:    {plot_path}")


if __name__ == "__main__":
    main()
