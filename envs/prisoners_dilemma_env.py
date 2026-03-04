"""Simultaneous-action two-player Prisoner's Dilemma environment for RLlib."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from gymnasium.spaces import Box, Discrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv

COOPERATE = 0
DEFECT = 1
ACTION_NAMES = {
    COOPERATE: "cooperate",
    DEFECT: "defect",
}
AGENT_IDS = ("player_1", "player_2")
ENV_NAME = "sequential_prisoners_dilemma"

PAYOFF_MATRIX: Dict[Tuple[int, int], Tuple[float, float]] = {
    (COOPERATE, COOPERATE): (3.0, 3.0),
    (COOPERATE, DEFECT): (0.0, 5.0),
    (DEFECT, COOPERATE): (5.0, 0.0),
    (DEFECT, DEFECT): (1.0, 1.0),
}


class SequentialPrisonersDilemmaEnv(MultiAgentEnv):
    """Two-agent repeated Prisoner's Dilemma with simultaneous actions.

    Rules:
    - Both players choose actions each round.
    - A payoff is assigned from the joint action in that round.
    - Horizon mode can be one of:
      - `fixed`: always run exactly `max_rounds`.
      - `random_revealed`: sample episode horizon in [min_rounds, max_rounds].
      - `random_continuation`: after each round (after `min_rounds`), continue with
        probability `continuation_prob`; stop otherwise.
    """

    def __init__(self, config=None):
        super().__init__()
        config = config or {}

        self.max_rounds = int(config.get("max_rounds", 50))
        if self.max_rounds <= 0:
            raise ValueError("max_rounds must be > 0")
        self.min_rounds = int(config.get("min_rounds", self.max_rounds))
        if self.min_rounds <= 0:
            raise ValueError("min_rounds must be > 0")
        if self.min_rounds > self.max_rounds:
            raise ValueError("min_rounds must be <= max_rounds")
        self.horizon_mode = str(config.get("horizon_mode", "fixed"))
        valid_horizon_modes = {"fixed", "random_revealed", "random_continuation"}
        if self.horizon_mode not in valid_horizon_modes:
            raise ValueError(
                f"horizon_mode must be one of {sorted(valid_horizon_modes)}; "
                f"got {self.horizon_mode!r}"
            )
        self.continuation_prob = float(config.get("continuation_prob", 0.95))
        if not 0.0 <= self.continuation_prob <= 1.0:
            raise ValueError("continuation_prob must be in [0.0, 1.0]")

        # Observation = [last_own_action, last_opponent_action, round_progress]
        # last actions use -1.0 before first complete round.
        shared_observation_space = Box(
            low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        shared_action_space = Discrete(2)  # 0=cooperate, 1=defect

        # RLlib's new API stack: per-agent space dicts.
        self.observation_spaces = {
            agent_id: shared_observation_space for agent_id in AGENT_IDS
        }
        self.action_spaces = {agent_id: shared_action_space for agent_id in AGENT_IDS}

        # Keep shared fields for compatibility with old stack code paths.
        self.observation_space = shared_observation_space
        self.action_space = shared_action_space

        self.possible_agents = list(AGENT_IDS)
        self.agents = list(AGENT_IDS)

        self._last_joint_actions = (-1, -1)
        self.rounds_completed = 0
        self._episode_done = False
        self._episode_horizon = self.max_rounds

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self._last_joint_actions = (-1, -1)
        self.rounds_completed = 0
        self._episode_done = False
        self._episode_horizon = self._sample_episode_horizon()

        obs = {agent_id: self._build_obs(agent_id) for agent_id in AGENT_IDS}
        infos = {
            agent_id: {
                "round": 1,
                "episode_horizon": self._episode_horizon,
                "horizon_mode": self.horizon_mode,
            }
            for agent_id in AGENT_IDS
        }
        return obs, infos

    def step(self, action_dict):
        if self._episode_done:
            raise RuntimeError("step() called after episode is done; call reset().")
        missing_agents = [agent_id for agent_id in AGENT_IDS if agent_id not in action_dict]
        if missing_agents:
            raise ValueError(f"Missing actions for agents: {missing_agents}")

        player_1_action = int(action_dict[AGENT_IDS[0]])
        player_2_action = int(action_dict[AGENT_IDS[1]])
        if player_1_action not in (COOPERATE, DEFECT):
            raise ValueError(
                f"Invalid action for {AGENT_IDS[0]}: {player_1_action}; "
                "expected 0 (cooperate) or 1 (defect)."
            )
        if player_2_action not in (COOPERATE, DEFECT):
            raise ValueError(
                f"Invalid action for {AGENT_IDS[1]}: {player_2_action}; "
                "expected 0 (cooperate) or 1 (defect)."
            )

        reward_player_1, reward_player_2 = PAYOFF_MATRIX[(player_1_action, player_2_action)]

        self.rounds_completed += 1
        self._last_joint_actions = (player_1_action, player_2_action)

        terminated_all = self._should_terminate_episode()
        truncated_all = False

        self._episode_done = terminated_all or truncated_all
        # RLlib's new API stack expects final observations for ended agents,
        # especially for truncation value-bootstrapping.
        obs = {agent_id: self._build_obs(agent_id) for agent_id in AGENT_IDS}
        next_round = self.rounds_completed if self._episode_done else self.rounds_completed + 1
        infos = {
            agent_id: {
                "round": next_round,
                "episode_horizon": self._episode_horizon,
                "horizon_mode": self.horizon_mode,
            }
            for agent_id in AGENT_IDS
        }

        rewards = {
            AGENT_IDS[0]: reward_player_1,
            AGENT_IDS[1]: reward_player_2,
        }

        terminateds = {"__all__": terminated_all}
        truncateds = {"__all__": truncated_all}
        if self._episode_done:
            terminateds.update({AGENT_IDS[0]: terminated_all, AGENT_IDS[1]: terminated_all})
            truncateds.update({AGENT_IDS[0]: truncated_all, AGENT_IDS[1]: truncated_all})

        return obs, rewards, terminateds, truncateds, infos

    def _build_obs(self, agent_id: str):
        if agent_id == AGENT_IDS[0]:
            own_last, opp_last = self._last_joint_actions
        else:
            opp_last, own_last = self._last_joint_actions
        round_progress = float(self.rounds_completed) / float(self._episode_horizon)
        return np.array([float(own_last), float(opp_last), round_progress], dtype=np.float32)

    def _sample_episode_horizon(self) -> int:
        if self.horizon_mode == "fixed":
            return self.max_rounds
        if self.horizon_mode == "random_revealed":
            # np.random.randint upper bound is exclusive.
            return int(np.random.randint(self.min_rounds, self.max_rounds + 1))
        # random_continuation keeps max_rounds as a hard cap, but the actual
        # stop round is stochastic and not known in advance.
        return self.max_rounds

    def _should_terminate_episode(self) -> bool:
        if self.rounds_completed >= self.max_rounds:
            return True
        if self.horizon_mode in ("fixed", "random_revealed"):
            return self.rounds_completed >= self._episode_horizon
        # random_continuation
        if self.rounds_completed < self.min_rounds:
            return False
        return np.random.random() >= self.continuation_prob
