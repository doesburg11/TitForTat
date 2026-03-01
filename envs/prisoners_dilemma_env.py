"""Turn-based two-player Prisoner's Dilemma environment for RLlib."""

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
    """Two-agent turn-based repeated Prisoner's Dilemma.

    Rules:
    - Player 1 acts first, then Player 2 acts.
    - A payoff is assigned after both actions in the round are known.
    - If `terminate_on_defection` is True, the episode ends after the round
      where at least one player defects.
    """

    def __init__(self, config=None):
        super().__init__()
        config = config or {}

        self.max_rounds = int(config.get("max_rounds", 50))
        if self.max_rounds <= 0:
            raise ValueError("max_rounds must be > 0")
        self.terminate_on_defection = bool(config.get("terminate_on_defection", True))

        # Observation = [last_own_action, last_opponent_action, round_progress]
        # last actions use -1.0 before first complete round.
        self.observation_space = Box(
            low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = Discrete(2)  # 0=cooperate, 1=defect

        self.possible_agents = list(AGENT_IDS)
        self.agents = list(AGENT_IDS)

        self._next_player = AGENT_IDS[0]
        self._pending_action_player_1 = None
        self._last_joint_actions = (-1, -1)
        self.rounds_completed = 0
        self._episode_done = False

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self._next_player = AGENT_IDS[0]
        self._pending_action_player_1 = None
        self._last_joint_actions = (-1, -1)
        self.rounds_completed = 0
        self._episode_done = False

        obs = {self._next_player: self._build_obs(self._next_player)}
        infos = {self._next_player: {"round": 1}}
        return obs, infos

    def step(self, action_dict):
        if self._episode_done:
            raise RuntimeError("step() called after episode is done; call reset().")
        if self._next_player not in action_dict:
            raise ValueError(f"Expected action for active agent {self._next_player}.")

        active_agent = self._next_player
        action = int(action_dict[active_agent])
        if action not in (COOPERATE, DEFECT):
            raise ValueError(f"Invalid action {action}; expected 0 (cooperate) or 1 (defect).")

        # Phase 1: Player 1 acts, then we hand the turn to Player 2.
        if active_agent == AGENT_IDS[0]:
            self._pending_action_player_1 = action
            self._next_player = AGENT_IDS[1]

            obs = {self._next_player: self._build_obs(self._next_player)}
            rewards = {AGENT_IDS[0]: 0.0, AGENT_IDS[1]: 0.0}
            terminateds = {"__all__": False}
            truncateds = {"__all__": False}
            infos = {self._next_player: {"round": self.rounds_completed + 1}}
            return obs, rewards, terminateds, truncateds, infos

        # Phase 2: Player 2 acts, round is complete, compute payoff.
        player_1_action = int(self._pending_action_player_1)
        player_2_action = action
        reward_player_1, reward_player_2 = PAYOFF_MATRIX[(player_1_action, player_2_action)]

        self.rounds_completed += 1
        self._last_joint_actions = (player_1_action, player_2_action)
        self._pending_action_player_1 = None

        defection_happened = (player_1_action == DEFECT) or (player_2_action == DEFECT)
        terminated_all = self.terminate_on_defection and defection_happened
        truncated_all = (self.rounds_completed >= self.max_rounds) and not terminated_all

        self._episode_done = terminated_all or truncated_all
        self._next_player = AGENT_IDS[0]

        if self._episode_done:
            obs = {}
        else:
            obs = {self._next_player: self._build_obs(self._next_player)}

        rewards = {
            AGENT_IDS[0]: reward_player_1,
            AGENT_IDS[1]: reward_player_2,
        }

        terminateds = {"__all__": terminated_all}
        truncateds = {"__all__": truncated_all}
        if self._episode_done:
            terminateds.update({AGENT_IDS[0]: terminated_all, AGENT_IDS[1]: terminated_all})
            truncateds.update({AGENT_IDS[0]: truncated_all, AGENT_IDS[1]: truncated_all})

        infos = {
            AGENT_IDS[0]: {
                "round": self.rounds_completed,
                "action": ACTION_NAMES[player_1_action],
                "reward": reward_player_1,
            },
            AGENT_IDS[1]: {
                "round": self.rounds_completed,
                "action": ACTION_NAMES[player_2_action],
                "reward": reward_player_2,
            },
        }
        return obs, rewards, terminateds, truncateds, infos

    def _build_obs(self, agent_id: str):
        if agent_id == AGENT_IDS[0]:
            own_last, opp_last = self._last_joint_actions
        else:
            opp_last, own_last = self._last_joint_actions
        round_progress = float(self.rounds_completed) / float(self.max_rounds)
        return np.array([float(own_last), float(opp_last), round_progress], dtype=np.float32)
