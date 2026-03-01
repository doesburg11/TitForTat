#!/usr/bin/env python3
"""Train and evaluate two independent RLlib policies on a MARL PD environment."""

from __future__ import annotations

import argparse
import json
from typing import Dict

import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from envs.prisoners_dilemma_env import (
    AGENT_IDS,
    COOPERATE,
    ENV_NAME,
    SequentialPrisonersDilemmaEnv,
)


def env_creator(env_config):
    return SequentialPrisonersDilemmaEnv(env_config)


def policy_mapping_fn(agent_id, *args, **kwargs):
    return f"policy_{agent_id}"


def build_algorithm(args) -> Algorithm:
    env_config = {
        "max_rounds": args.max_rounds,
        "terminate_on_defection": args.terminate_on_defection,
    }
    register_env(ENV_NAME, env_creator)

    tmp_env = SequentialPrisonersDilemmaEnv(env_config)
    policies = {
        "policy_player_1": (None, tmp_env.observation_space, tmp_env.action_space, {}),
        "policy_player_2": (None, tmp_env.observation_space, tmp_env.action_space, {}),
    }

    config = (
        PPOConfig()
        .environment(ENV_NAME, env_config=env_config)
        .framework(args.framework)
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=list(policies.keys()),
        )
        .resources(num_gpus=args.num_gpus)
        .training(lr=args.lr)
    )

    # RLlib changed rollout APIs across versions; keep compatibility.
    if hasattr(config, "env_runners"):
        config = config.env_runners(num_env_runners=args.num_workers)
    else:
        config = config.rollouts(num_rollout_workers=args.num_workers)

    if args.train_batch_size is not None:
        config = config.training(train_batch_size=args.train_batch_size)

    return config.build()


def extract_reward_mean(train_result: Dict) -> float:
    if "episode_reward_mean" in train_result:
        return float(train_result["episode_reward_mean"])
    env_runner_metrics = train_result.get("env_runners", {})
    if "episode_return_mean" in env_runner_metrics:
        return float(env_runner_metrics["episode_return_mean"])
    return float("nan")


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
                action = int(
                    algo.compute_single_action(
                        observation=agent_obs,
                        policy_id=policy_id,
                        explore=False,
                    )
                )
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-iters", type=int, default=50, help="PPO train iterations.")
    parser.add_argument(
        "--eval-episodes", type=int, default=20, help="Episodes for post-train evaluation."
    )
    parser.add_argument("--max-rounds", type=int, default=50, help="Max rounds per episode.")
    parser.add_argument(
        "--terminate-on-defection",
        dest="terminate_on_defection",
        action="store_true",
        default=True,
        help="End after the round in which any player defects.",
    )
    parser.add_argument(
        "--no-terminate-on-defection",
        dest="terminate_on_defection",
        action="store_false",
        help="Keep playing to max_rounds even after defection.",
    )
    parser.add_argument("--framework", type=str, default="torch", choices=["torch", "tf2"])
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0, help="RLlib rollout workers.")
    parser.add_argument("--num-gpus", type=float, default=0.0)
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/sequential_pd_ppo",
        help="Directory where checkpoints are written.",
    )
    parser.add_argument(
        "--from-checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path. If set, skip training and only evaluate.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    env_config = {
        "max_rounds": args.max_rounds,
        "terminate_on_defection": args.terminate_on_defection,
    }

    ray.init(ignore_reinit_error=True, include_dashboard=False)
    register_env(ENV_NAME, env_creator)

    if args.from_checkpoint:
        algo = Algorithm.from_checkpoint(args.from_checkpoint)
    else:
        algo = build_algorithm(args)
        for i in range(1, args.train_iters + 1):
            result = algo.train()
            if i == 1 or i == args.train_iters or i % 10 == 0:
                reward_mean = extract_reward_mean(result)
                print(
                    f"[train] iter={i} reward_mean={reward_mean:.3f} "
                    f"timesteps_total={result.get('timesteps_total', 'n/a')}"
                )

        checkpoint_path = checkpoint_to_path(algo.save(args.checkpoint_dir))
        print(f"[train] checkpoint saved to: {checkpoint_path}")

    if args.eval_episodes > 0:
        eval_summary = evaluate(algo, args.eval_episodes, env_config)
        print("[eval] summary:")
        print(json.dumps(eval_summary, indent=2))

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
