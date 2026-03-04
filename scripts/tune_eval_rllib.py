#!/usr/bin/env python3
"""Tune and evaluate two independent RLlib policies on a MARL PD environment."""

from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import ray
from envs.prisoners_dilemma_env import (
    AGENT_IDS,
    COOPERATE,
    ENV_NAME,
    SequentialPrisonersDilemmaEnv,
)
from ray import tune
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.core import Columns
from ray.rllib.utils.framework import try_import_torch
from ray.tune.registry import register_env

torch, _ = try_import_torch()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_CONFIG_PATH = PROJECT_ROOT / "config" / "config_env.py"


def env_creator(env_config):
    return SequentialPrisonersDilemmaEnv(env_config)


def policy_mapping_fn(agent_id, *args, **kwargs):
    return f"policy_{agent_id}"


def _resolve_project_file(path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def load_ppo_config(path: str) -> tuple[Dict[str, Any], str]:
    resolved_path = _resolve_project_file(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"PPO config file not found: {resolved_path}")

    module_name = f"_sequential_pd_ppo_config_{abs(hash(str(resolved_path)))}"
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


def load_env_config(path: str) -> tuple[Dict[str, Any], str]:
    resolved_path = _resolve_project_file(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Environment config file not found: {resolved_path}")

    module_name = f"_sequential_pd_env_config_{abs(hash(str(resolved_path)))}"
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
    return dict(config_env), str(resolved_path)


def resolve_run_config(config_env_path: Optional[str] = None) -> tuple[SimpleNamespace, str]:
    if config_env_path is None:
        env_config_path = DEFAULT_ENV_CONFIG_PATH.resolve()
    else:
        env_config_path = _resolve_project_file(config_env_path)
    raw_config_env, resolved_path = load_env_config(str(env_config_path))

    config_env = dict(raw_config_env)

    try:
        config_env["ppo_config"] = str(config_env["ppo_config"])
        config_env["eval_episodes"] = int(config_env["eval_episodes"])
        config_env["n_sequential_games"] = int(config_env["n_sequential_games"])
        config_env["checkpoint_dir"] = str(config_env["checkpoint_dir"])
    except KeyError as exc:
        missing_key = exc.args[0]
        raise ValueError(f"Missing required key in `config_env`: {missing_key!r}") from exc

    if config_env["eval_episodes"] < 0:
        raise ValueError("eval_episodes must be >= 0")
    if config_env["n_sequential_games"] <= 0:
        raise ValueError("n_sequential_games must be > 0")

    seed = config_env.get("seed")
    if seed is not None:
        config_env["seed"] = int(seed)

    from_checkpoint = config_env.get("from_checkpoint")
    if from_checkpoint in ("", None):
        config_env["from_checkpoint"] = None
    else:
        config_env["from_checkpoint"] = str(from_checkpoint)

    metrics_out = config_env.get("metrics_out")
    if metrics_out in ("", None):
        config_env["metrics_out"] = None
    else:
        config_env["metrics_out"] = str(metrics_out)

    return SimpleNamespace(**config_env), resolved_path


def _parse_rollout_fragment_length(value):
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "auto":
            return "auto"
        if normalized.isdigit() or (normalized.startswith("-") and normalized[1:].isdigit()):
            return int(normalized)
    if isinstance(value, (int, float)):
        numeric = int(value)
        return numeric
    raise ValueError(
        "rollout_fragment_length must be an integer or 'auto', "
        f"got value={value!r} type={type(value).__name__}"
    )


def resolve_ppo_config(
    ppo_config_path: str,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], int, str]:
    raw_config_ppo, resolved_path = load_ppo_config(ppo_config_path)
    config_ppo = dict(raw_config_ppo)
    if "tune_iters" not in config_ppo:
        raise ValueError("`config_ppo` must define `tune_iters` (int > 0).")
    tune_iters = int(config_ppo.pop("tune_iters"))
    if tune_iters <= 0:
        raise ValueError(f"`tune_iters` must be > 0, got {tune_iters}.")
    if "rollout_fragment_length" in config_ppo:
        config_ppo["rollout_fragment_length"] = _parse_rollout_fragment_length(
            config_ppo["rollout_fragment_length"]
        )

    required_ppo_keys = (
        "num_learners",
        "num_gpus_per_learner",
        "num_env_runners",
        "num_envs_per_env_runner",
        "rollout_fragment_length",
        "sample_timeout_s",
        "num_cpus_per_env_runner",
        "num_cpus_for_main_process",
    )
    missing_ppo_keys = sorted(key for key in required_ppo_keys if key not in config_ppo)
    if missing_ppo_keys:
        raise ValueError(f"Missing required keys in `config_ppo`: {missing_ppo_keys}")

    learner_config = {
        "num_learners": config_ppo.pop("num_learners"),
        "num_gpus_per_learner": config_ppo.pop("num_gpus_per_learner"),
    }
    env_runner_config = {
        "num_env_runners": config_ppo.pop("num_env_runners"),
        "num_envs_per_env_runner": config_ppo.pop("num_envs_per_env_runner"),
        "rollout_fragment_length": config_ppo.pop("rollout_fragment_length"),
        "sample_timeout_s": config_ppo.pop("sample_timeout_s"),
        "num_cpus_per_env_runner": config_ppo.pop("num_cpus_per_env_runner"),
    }
    resource_config = {
        "num_cpus_for_main_process": config_ppo.pop("num_cpus_for_main_process"),
    }
    ppo_training_config = dict(config_ppo)

    return (
        ppo_training_config,
        learner_config,
        env_runner_config,
        resource_config,
        tune_iters,
        resolved_path,
    )


def _requested_resources(
    learner_config: Dict[str, Any],
    env_runner_config: Dict[str, Any],
    resource_config: Dict[str, Any],
) -> Dict[str, float]:
    num_learners = float(learner_config["num_learners"])
    num_gpus_per_learner = float(learner_config["num_gpus_per_learner"])
    num_env_runners = float(env_runner_config["num_env_runners"])
    num_cpus_per_env_runner = float(env_runner_config["num_cpus_per_env_runner"])
    num_cpus_for_main_process = float(resource_config["num_cpus_for_main_process"])

    return {
        "cpus": num_cpus_for_main_process + (num_env_runners * num_cpus_per_env_runner),
        "gpus": num_learners * num_gpus_per_learner,
    }


def validate_schedulable_resources(
    learner_config: Dict[str, Any],
    env_runner_config: Dict[str, Any],
    resource_config: Dict[str, Any],
    ppo_config_path: str,
) -> None:
    requested = _requested_resources(learner_config, env_runner_config, resource_config)
    cluster = ray.cluster_resources()
    available_cpus = float(cluster.get("CPU", 0.0))
    available_gpus = float(cluster.get("GPU", 0.0))

    print(
        "[resources] requested "
        f"cpus={requested['cpus']:.2f} gpus={requested['gpus']:.2f} "
        f"(num_learners={learner_config['num_learners']}, "
        f"num_gpus_per_learner={learner_config['num_gpus_per_learner']}, "
        f"num_env_runners={env_runner_config['num_env_runners']})"
    )
    print(f"[resources] cluster   cpus={available_cpus:.2f} gpus={available_gpus:.2f}")

    if requested["gpus"] > available_gpus + 1e-9:
        raise ValueError(
            "Unschedulable GPU request: requested "
            f"{requested['gpus']:.2f} GPUs but cluster has {available_gpus:.2f}. "
            "Reduce `num_learners` or `num_gpus_per_learner` in "
            f"{ppo_config_path}."
        )
    if requested["cpus"] > available_cpus + 1e-9:
        raise ValueError(
            "Unschedulable CPU request: requested "
            f"{requested['cpus']:.2f} CPUs but cluster has {available_cpus:.2f}. "
            "Reduce `num_env_runners`, `num_cpus_per_env_runner`, or "
            "`num_cpus_for_main_process` in "
            f"{ppo_config_path}."
        )


def build_ppo_config(
    args,
    ppo_training_config: Dict[str, Any],
    learner_config: Dict[str, Any],
    env_runner_config: Dict[str, Any],
    resource_config: Dict[str, Any],
) -> PPOConfig:
    env_config = {
        "n_sequential_games": args.n_sequential_games,
    }
    register_env(ENV_NAME, env_creator)

    policies = {f"policy_{agent_id}" for agent_id in AGENT_IDS}
    config = PPOConfig()
    if hasattr(config, "api_stack"):
        config = config.api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
    config = (
        config.environment(ENV_NAME, env_config=env_config)
        .framework("torch")
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=list(policies),
        )
        .training(**ppo_training_config)
    )
    if hasattr(config, "learners"):
        config = config.learners(
            num_learners=learner_config["num_learners"],
            num_gpus_per_learner=learner_config["num_gpus_per_learner"],
        )
    if hasattr(config, "resources"):
        resource_kwargs = {"num_cpus_for_main_process": resource_config["num_cpus_for_main_process"]}
        if not hasattr(config, "learners"):
            resource_kwargs["num_gpus"] = learner_config["num_gpus_per_learner"]
        config = config.resources(**resource_kwargs)
    if args.seed is not None and hasattr(config, "debugging"):
        config = config.debugging(seed=args.seed)

    # RLlib rollout worker APIs changed across versions; keep compatibility.
    if hasattr(config, "env_runners"):
        env_runner_kwargs = {
            "num_env_runners": env_runner_config["num_env_runners"],
            "num_envs_per_env_runner": env_runner_config["num_envs_per_env_runner"],
            "rollout_fragment_length": env_runner_config["rollout_fragment_length"],
            "sample_timeout_s": env_runner_config["sample_timeout_s"],
            "num_cpus_per_env_runner": env_runner_config["num_cpus_per_env_runner"],
        }
        config = config.env_runners(**env_runner_kwargs)
    elif hasattr(config, "rollouts"):
        rollout_kwargs = {
            "num_rollout_workers": env_runner_config["num_env_runners"],
            "num_envs_per_worker": env_runner_config["num_envs_per_env_runner"],
            "rollout_fragment_length": env_runner_config["rollout_fragment_length"],
            "num_cpus_per_worker": env_runner_config["num_cpus_per_env_runner"],
        }
        config = config.rollouts(**rollout_kwargs)

    return config


def extract_reward_mean(train_result: Dict) -> float:
    reward = train_result.get("episode_reward_mean")
    if reward is not None and not _is_missing(reward):
        return float(reward)
    reward = train_result.get("env_runners/episode_return_mean")
    if reward is not None and not _is_missing(reward):
        return float(reward)
    env_runner_metrics = train_result.get("env_runners", {})
    if "episode_return_mean" in env_runner_metrics:
        return float(env_runner_metrics["episode_return_mean"])
    return float("nan")


def extract_timesteps_total(train_result: Dict):
    candidate_keys = (
        "timesteps_total",
        "num_env_steps_sampled_lifetime",
        "num_env_steps_sampled",
        "num_agent_steps_sampled_lifetime",
        "num_agent_steps_sampled",
        "env_runners/num_env_steps_sampled_lifetime",
        "env_runners/num_env_steps_sampled",
    )
    for key in candidate_keys:
        if key in train_result and not _is_missing(train_result[key]):
            return train_result[key]

    env_runner_metrics = train_result.get("env_runners", {})
    for key in ("num_env_steps_sampled_lifetime", "num_env_steps_sampled"):
        if key in env_runner_metrics and not _is_missing(env_runner_metrics[key]):
            return env_runner_metrics[key]

    counters = train_result.get("counters", {})
    for key in ("num_env_steps_sampled", "num_env_steps_sampled_lifetime"):
        if key in counters and not _is_missing(counters[key]):
            return counters[key]

    for key, value in train_result.items():
        if key.endswith("/num_env_steps_sampled_lifetime") and not _is_missing(value):
            return value

    return "n/a"


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False


def _to_numpy(value):
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    if hasattr(value, "numpy") and callable(value.numpy):
        return value.numpy()
    return np.asarray(value)


def _extract_first_action(action_batch) -> int:
    action_np = _to_numpy(action_batch)
    return int(np.asarray(action_np).reshape(-1)[0])


def _resolve_checkpoint_path(path: str) -> str:
    if "://" in path:
        return path
    return str(_resolve_project_file(path))


def compute_eval_action(algo: Algorithm, policy_id: str, observation) -> int:
    module = algo.get_module(policy_id)
    if module is not None:
        obs_batch = np.expand_dims(np.asarray(observation, dtype=np.float32), axis=0)
        batch = {Columns.OBS: obs_batch}

        if getattr(module, "framework", None) == "torch" and torch is not None:
            batch[Columns.OBS] = torch.from_numpy(obs_batch)
            with torch.no_grad():
                module_out = module.forward_inference(batch)
        else:
            module_out = module.forward_inference(batch)

        if Columns.ACTIONS in module_out:
            return _extract_first_action(module_out[Columns.ACTIONS])
        if Columns.ACTIONS_FOR_ENV in module_out:
            return _extract_first_action(module_out[Columns.ACTIONS_FOR_ENV])
        if Columns.ACTION_DIST_INPUTS in module_out:
            action_dist_cls = module.get_inference_action_dist_cls()
            if action_dist_cls is None:
                raise ValueError(f"Inference distribution missing for module={policy_id!r}")
            action_dist = action_dist_cls.from_logits(module_out[Columns.ACTION_DIST_INPUTS])
            return _extract_first_action(action_dist.to_deterministic().sample())
        raise ValueError(
            f"Module output for {policy_id!r} missing action keys: "
            f"{list(module_out.keys())}"
        )

    # Backward-compat fallback if an old checkpoint restores policies only.
    policy = algo.get_policy(policy_id)
    if policy is None:
        raise ValueError(f"Policy not found for policy_id={policy_id!r}")

    actions, _state_out, _extra = policy.compute_actions([observation], explore=False)
    return int(actions[0])


def evaluate(algo: Algorithm, episodes: int, env_config: Dict) -> Dict:
    env = SequentialPrisonersDilemmaEnv(env_config)

    total_rewards = {agent: 0.0 for agent in AGENT_IDS}
    action_counts = {agent: 0 for agent in AGENT_IDS}
    cooperation_counts = {agent: 0 for agent in AGENT_IDS}
    rounds_per_episode = []

    for _ in range(episodes):
        obs, _infos = env.reset()
        terminated = {"__all__": False}
        truncated = {"__all__": False}
        episode_rewards = {agent: 0.0 for agent in AGENT_IDS}

        while not (terminated["__all__"] or truncated["__all__"]):
            actions = {}
            for agent_id, agent_obs in obs.items():
                policy_id = policy_mapping_fn(agent_id)
                action = compute_eval_action(algo, policy_id, agent_obs)
                actions[agent_id] = action
                action_counts[agent_id] += 1
                if action == COOPERATE:
                    cooperation_counts[agent_id] += 1

            obs, rewards, terminated, truncated, _infos = env.step(actions)
            for agent_id in AGENT_IDS:
                episode_rewards[agent_id] += float(rewards.get(agent_id, 0.0))

        rounds_per_episode.append(env.rounds_completed)
        for agent_id in AGENT_IDS:
            total_rewards[agent_id] += episode_rewards[agent_id]

    summary = {
        "episodes": episodes,
        "mean_episode_reward": {
            agent_id: total_rewards[agent_id] / max(episodes, 1) for agent_id in AGENT_IDS
        },
        "cooperation_rate": {
            agent_id: cooperation_counts[agent_id] / max(action_counts[agent_id], 1)
            for agent_id in AGENT_IDS
        },
        "mean_rounds_per_episode": sum(rounds_per_episode) / max(episodes, 1),
    }
    return summary


def checkpoint_to_path(checkpoint_obj) -> str:
    if isinstance(checkpoint_obj, str):
        return checkpoint_obj
    if hasattr(checkpoint_obj, "path"):
        return str(checkpoint_obj.path)
    if hasattr(checkpoint_obj, "checkpoint") and hasattr(checkpoint_obj.checkpoint, "path"):
        return str(checkpoint_obj.checkpoint.path)
    return str(checkpoint_obj)


def _collect_tune_history_from_dataframe(metrics_dataframe) -> List[Dict[str, Any]]:
    if metrics_dataframe is None or len(metrics_dataframe) == 0:
        return []

    tune_history = []
    for _idx, row in metrics_dataframe.iterrows():
        row_dict = row.to_dict()
        iteration = row_dict.get("training_iteration")
        if _is_missing(iteration):
            continue
        reward_mean = extract_reward_mean(row_dict)
        timesteps_total = extract_timesteps_total(row_dict)
        tune_history.append(
            {
                "iter": int(iteration),
                "episode_reward_mean": reward_mean,
                "timesteps_total": timesteps_total,
            }
        )
    return tune_history


def tune_with_tuner(args, ppo_config: PPOConfig, tune_iters: int) -> tuple[List[Dict[str, Any]], str]:
    tune_output_dir = Path(_resolve_checkpoint_path(args.checkpoint_dir))
    run_config = tune.RunConfig(
        name=tune_output_dir.name,
        storage_path=str(tune_output_dir.parent),
        stop={"training_iteration": tune_iters},
        checkpoint_config=tune.CheckpointConfig(
            checkpoint_at_end=True,
            num_to_keep=1,
        ),
    )
    tuner = tune.Tuner(
        PPO,
        run_config=run_config,
        param_space=ppo_config.to_dict(),
    )
    result_grid = tuner.fit()
    results = list(result_grid)
    if not results:
        raise RuntimeError("Tune produced no trial results.")

    result = results[0]
    tune_history = _collect_tune_history_from_dataframe(result.metrics_dataframe)
    if not tune_history:
        final_reward_mean = extract_reward_mean(result.metrics)
        final_timesteps = extract_timesteps_total(result.metrics)
        final_iter = int(result.metrics.get("training_iteration", tune_iters))
        tune_history = [
            {
                "iter": final_iter,
                "episode_reward_mean": final_reward_mean,
                "timesteps_total": final_timesteps,
            }
        ]

    checkpoint = result.checkpoint
    if checkpoint is None and result.best_checkpoints:
        checkpoint = result.best_checkpoints[0][0]
    if checkpoint is None:
        raise RuntimeError(
            "Tune completed but did not produce a checkpoint. "
            "Please ensure checkpointing is enabled."
        )
    checkpoint_path = checkpoint_to_path(checkpoint)

    for item in tune_history:
        i = item["iter"]
        if i == 1 or i == tune_iters or i % 10 == 0:
            print(
                f"[tune] iter={i} reward_mean={item['episode_reward_mean']:.3f} "
                f"timesteps_total={item['timesteps_total']}"
            )
    print(f"[tune] checkpoint saved to: {checkpoint_path}")
    return tune_history, checkpoint_path


def write_metrics_json(
    output_path: str,
    args,
    tune_history: List[Dict[str, Any]],
    checkpoint_path: Optional[str],
    eval_summary: Optional[Dict[str, Any]],
) -> str:
    resolved_output_path = Path(output_path).expanduser().resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": vars(args),
        "tune_history": tune_history,
        "checkpoint_path": checkpoint_path,
        "eval_summary": eval_summary,
    }
    resolved_output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(resolved_output_path)


def main(config_env_path: Optional[str] = None):
    run_config, resolved_env_config_path = resolve_run_config(config_env_path=config_env_path)
    print(f"[config] loaded env config: {resolved_env_config_path}")

    if run_config.seed is not None:
        random.seed(run_config.seed)
        np.random.seed(run_config.seed)
        if torch is not None and hasattr(torch, "manual_seed"):
            torch.manual_seed(run_config.seed)
    env_config = {
        "n_sequential_games": run_config.n_sequential_games,
    }

    ray.init(ignore_reinit_error=True, include_dashboard=False)
    register_env(ENV_NAME, env_creator)

    tune_history = []
    checkpoint_path = None
    eval_summary = None
    algo: Optional[Algorithm] = None
    tune_iters: Optional[int] = None

    if not run_config.from_checkpoint:
        (
            ppo_training_config,
            ppo_learner_config,
            ppo_env_runner_config,
            ppo_resource_config,
            tune_iters,
            resolved_ppo_config_path,
        ) = resolve_ppo_config(run_config.ppo_config)
        run_config.ppo_config = resolved_ppo_config_path
        run_config.tune_iters = tune_iters
        run_config.ppo_training_config = ppo_training_config
        run_config.ppo_learner_config = ppo_learner_config
        run_config.ppo_env_runner_config = ppo_env_runner_config
        run_config.ppo_resource_config = ppo_resource_config
        print(f"[config] loaded PPO config: {resolved_ppo_config_path}")
        validate_schedulable_resources(
            learner_config=ppo_learner_config,
            env_runner_config=ppo_env_runner_config,
            resource_config=ppo_resource_config,
            ppo_config_path=resolved_ppo_config_path,
        )
    else:
        run_config.ppo_config = _resolve_project_file(run_config.ppo_config).as_posix()
        run_config.ppo_training_config = None
        run_config.ppo_learner_config = None
        run_config.ppo_env_runner_config = None
        run_config.ppo_resource_config = None

    try:
        if run_config.from_checkpoint:
            checkpoint_path = _resolve_checkpoint_path(run_config.from_checkpoint)
        else:
            ppo_config = build_ppo_config(
                run_config,
                ppo_training_config=ppo_training_config,
                learner_config=ppo_learner_config,
                env_runner_config=ppo_env_runner_config,
                resource_config=ppo_resource_config,
            )
            tune_history, checkpoint_path = tune_with_tuner(
                run_config,
                ppo_config,
                tune_iters=tune_iters,
            )

        if checkpoint_path is None:
            raise RuntimeError("No checkpoint available for evaluation.")
        algo = Algorithm.from_checkpoint(checkpoint_path)

        if run_config.eval_episodes > 0:
            eval_summary = evaluate(algo, run_config.eval_episodes, env_config)
            print("[eval] summary:")
            print(json.dumps(eval_summary, indent=2))

        if run_config.metrics_out:
            output_path = write_metrics_json(
                output_path=run_config.metrics_out,
                args=run_config,
                tune_history=tune_history,
                checkpoint_path=checkpoint_path,
                eval_summary=eval_summary,
            )
            print(f"[metrics] wrote: {output_path}")
    finally:
        if algo is not None:
            algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
