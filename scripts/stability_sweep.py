#!/usr/bin/env python3
"""Run a multi-seed tuning sweep and report stability metrics."""

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
from typing import Dict, Iterable, List, Optional


ENV_CONFIG_ENVVAR = "SEQUENTIAL_PD_ENV_CONFIG"
TARGET_EPISODES_PER_UPDATE = 64
MINIBATCH_DIVISOR = 8
MIN_MINIBATCH_SIZE = 128
MIN_TRAIN_BATCH_SIZE = 1024


def _finite(values: Iterable[Optional[float]]) -> List[float]:
    cleaned = []
    for value in values:
        if value is None:
            continue
        numeric = float(value)
        if math.isfinite(numeric):
            cleaned.append(numeric)
    return cleaned


def _summary(values: Iterable[Optional[float]]) -> Dict[str, Optional[float]]:
    xs = _finite(values)
    if not xs:
        return {"count": 0, "mean": None, "std": None, "cv": None, "min": None, "max": None}
    mean_value = statistics.fmean(xs)
    std_value = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    cv_value = None
    if abs(mean_value) > 1e-12:
        cv_value = std_value / abs(mean_value)
    return {
        "count": len(xs),
        "mean": mean_value,
        "std": std_value,
        "cv": cv_value,
        "min": min(xs),
        "max": max(xs),
    }


def _nested(source: Dict, *keys, default=None):
    value = source
    for key in keys:
        if not isinstance(value, dict):
            return default
        if key not in value:
            return default
        value = value[key]
    return value


def _build_tune_command(
    args,
) -> List[str]:
    return [
        args.python_executable,
        "scripts/tune_eval_rllib.py",
    ]


def _write_env_config(path: Path, config_env: Dict) -> None:
    path.write_text(f"config_env = {repr(config_env)}\n", encoding="utf-8")


def _write_ppo_config(path: Path, config_ppo: Dict) -> None:
    path.write_text(f"config_ppo = {repr(config_ppo)}\n", encoding="utf-8")


def _resolve_existing_file(path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path.cwd() / candidate).resolve()


def _load_ppo_config(path: str) -> tuple[Dict, str]:
    resolved_path = _resolve_existing_file(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"PPO config file not found: {resolved_path}")

    module_name = f"_sequential_pd_stability_ppo_config_{abs(hash(str(resolved_path)))}"
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


def _bool_check(name: str, value: Optional[float], threshold: float) -> Dict:
    passed = value is not None and value <= threshold
    return {
        "name": name,
        "value": value,
        "threshold": threshold,
        "passed": passed,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-seeds", type=int, default=5, help="How many random seeds to run.")
    parser.add_argument("--seed-start", type=int, default=0, help="First seed in the sweep.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/stability_sweep",
        help="Directory for per-seed metrics and summary JSON.",
    )
    parser.add_argument(
        "--python-executable",
        type=str,
        default=None,
        help=(
            "Python interpreter used to launch scripts/tune_eval_rllib.py. "
            "Defaults to ./.conda/bin/python when available, else current interpreter."
        ),
    )
    parser.add_argument(
        "--ppo-config",
        type=str,
        default="config/config_ppo.py",
        help="Path to base `config_ppo` file. Batch settings are auto-scaled per run.",
    )

    # Tuning args mirrored from tune_eval_rllib.py.
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--max-rounds", type=int, default=50)
    parser.add_argument("--min-rounds", type=int, default=1)
    parser.add_argument(
        "--horizon-mode",
        type=str,
        default="fixed",
        choices=["fixed", "random_revealed", "random_continuation"],
    )
    parser.add_argument("--continuation-prob", type=float, default=0.95)

    # Stability thresholds.
    parser.add_argument(
        "--max-reward-cv",
        type=float,
        default=0.15,
        help="Maximum allowed coefficient of variation of mean episode reward across seeds.",
    )
    parser.add_argument(
        "--max-cooperation-std",
        type=float,
        default=0.10,
        help="Maximum allowed std of cooperation rate across seeds.",
    )
    parser.add_argument(
        "--max-rounds-cv",
        type=float,
        default=0.10,
        help="Maximum allowed coefficient of variation of mean rounds per episode across seeds.",
    )
    parser.add_argument(
        "--max-player-reward-gap",
        type=float,
        default=1.0,
        help="Maximum allowed mean absolute reward gap between players.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_seeds <= 0:
        raise ValueError("num-seeds must be > 0")
    if args.max_rounds <= 0:
        raise ValueError("max-rounds must be > 0")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.python_executable is None:
        project_python = Path(".conda/bin/python").resolve()
        if project_python.exists():
            args.python_executable = str(project_python)
        else:
            args.python_executable = sys.executable

    per_seed = []
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    run_timestamp = _timestamp_token()
    base_ppo_config, resolved_ppo_config_path = _load_ppo_config(args.ppo_config)
    scaled_settings = _scaled_ppo_batch_settings(args.max_rounds)

    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = seed_dir / f"checkpoint_{run_timestamp}"
        metrics_out = seed_dir / f"metrics_{run_timestamp}.json"
        env_config_path = seed_dir / f"config_env_{run_timestamp}.py"
        ppo_config_path = seed_dir / f"config_ppo_{run_timestamp}.py"

        config_ppo = dict(base_ppo_config)
        config_ppo.update(scaled_settings)
        _write_ppo_config(ppo_config_path, config_ppo)

        config_env = {
            "ppo_config": str(ppo_config_path),
            "eval_episodes": args.eval_episodes,
            "max_rounds": args.max_rounds,
            "min_rounds": args.min_rounds,
            "horizon_mode": args.horizon_mode,
            "continuation_prob": args.continuation_prob,
            "seed": seed,
            "checkpoint_dir": str(checkpoint_dir),
            "from_checkpoint": None,
            "metrics_out": str(metrics_out),
        }
        _write_env_config(env_config_path, config_env)

        cmd = _build_tune_command(args)
        env = os.environ.copy()
        env[ENV_CONFIG_ENVVAR] = str(env_config_path)
        print(f"[sweep] running seed={seed}")
        subprocess.run(cmd, check=True, env=env)

        run_metrics = json.loads(metrics_out.read_text(encoding="utf-8"))
        eval_summary = run_metrics.get("eval_summary") or {}
        tune_history = run_metrics.get("tune_history") or []
        final_tune_reward = None
        if tune_history:
            final_tune_reward = tune_history[-1].get("episode_reward_mean")

        reward_p1 = _nested(eval_summary, "mean_episode_reward", "player_1")
        reward_p2 = _nested(eval_summary, "mean_episode_reward", "player_2")
        coop_p1 = _nested(eval_summary, "cooperation_rate", "player_1")
        coop_p2 = _nested(eval_summary, "cooperation_rate", "player_2")
        mean_rounds = eval_summary.get("mean_rounds_per_episode")
        reward_gap = None
        if reward_p1 is not None and reward_p2 is not None:
            reward_gap = abs(float(reward_p1) - float(reward_p2))

        per_seed.append(
            {
                "seed": seed,
                "metrics_path": str(metrics_out),
                "checkpoint_path": run_metrics.get("checkpoint_path"),
                "eval_summary": eval_summary,
                "final_tune_reward_mean": final_tune_reward,
                "reward_gap": reward_gap,
                "reward_player_1": reward_p1,
                "reward_player_2": reward_p2,
                "cooperation_player_1": coop_p1,
                "cooperation_player_2": coop_p2,
                "mean_rounds_per_episode": mean_rounds,
                "train_batch_size_per_learner": scaled_settings["train_batch_size_per_learner"],
                "minibatch_size": scaled_settings["minibatch_size"],
                "num_epochs": scaled_settings["num_epochs"],
                "ppo_config_path": str(ppo_config_path),
            }
        )

    reward_p1_summary = _summary(run["reward_player_1"] for run in per_seed)
    reward_p2_summary = _summary(run["reward_player_2"] for run in per_seed)
    coop_p1_summary = _summary(run["cooperation_player_1"] for run in per_seed)
    coop_p2_summary = _summary(run["cooperation_player_2"] for run in per_seed)
    rounds_summary = _summary(run["mean_rounds_per_episode"] for run in per_seed)
    reward_gap_summary = _summary(run["reward_gap"] for run in per_seed)
    final_tune_reward_summary = _summary(run["final_tune_reward_mean"] for run in per_seed)

    checks = [
        _bool_check("player_1_reward_cv", reward_p1_summary["cv"], args.max_reward_cv),
        _bool_check("player_2_reward_cv", reward_p2_summary["cv"], args.max_reward_cv),
        _bool_check(
            "player_1_cooperation_std",
            coop_p1_summary["std"],
            args.max_cooperation_std,
        ),
        _bool_check(
            "player_2_cooperation_std",
            coop_p2_summary["std"],
            args.max_cooperation_std,
        ),
        _bool_check("mean_rounds_cv", rounds_summary["cv"], args.max_rounds_cv),
        _bool_check(
            "mean_reward_gap",
            reward_gap_summary["mean"],
            args.max_player_reward_gap,
        ),
    ]
    stable = all(check["passed"] for check in checks)

    summary = {
        "config": vars(args),
        "base_ppo_config_path": resolved_ppo_config_path,
        "run_timestamp": run_timestamp,
        "batch_scaling": {
            "target_episodes_per_update": TARGET_EPISODES_PER_UPDATE,
            "minibatch_divisor": MINIBATCH_DIVISOR,
            "min_train_batch_size": MIN_TRAIN_BATCH_SIZE,
            "min_minibatch_size": MIN_MINIBATCH_SIZE,
        },
        "effective_ppo_batch_settings": scaled_settings,
        "seeds": seeds,
        "stable": stable,
        "checks": checks,
        "aggregate": {
            "reward_player_1": reward_p1_summary,
            "reward_player_2": reward_p2_summary,
            "cooperation_player_1": coop_p1_summary,
            "cooperation_player_2": coop_p2_summary,
            "mean_rounds_per_episode": rounds_summary,
            "reward_gap_abs": reward_gap_summary,
            "final_tune_reward_mean": final_tune_reward_summary,
        },
        "per_seed": per_seed,
    }

    summary_path = output_dir / f"summary_{run_timestamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[sweep] stability checks:")
    for check in checks:
        status = "PASS" if check["passed"] else "FAIL"
        print(
            f"  - {status} {check['name']}: "
            f"value={check['value']} threshold={check['threshold']}"
        )
    print(f"[sweep] verdict: {'STABLE' if stable else 'UNSTABLE'}")
    print(f"[sweep] summary written to: {summary_path}")


if __name__ == "__main__":
    main()
