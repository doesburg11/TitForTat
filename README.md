# TitForTat

## Overview

This project demonstrates a Tit-for-Tat-inspired repeated Prisoner's Dilemma setup with two independent learning agents using RLlib 2.54.0.

## Environment and MARL Setup

- Environment class: `envs/prisoners_dilemma_env.py`
- Agent IDs: `player_1`, `player_2`
- Action space: `0=cooperate`, `1=defect`
- Reward matrix:
  - `(C, C) -> (3, 3)`
  - `(C, D) -> (0, 5)`
  - `(D, C) -> (5, 0)`
  - `(D, D) -> (1, 1)`
- Turn order is sequential each round: Player 1 then Player 2
- Two independent RLlib policies are trained:
  - `policy_player_1` for `player_1`
  - `policy_player_2` for `player_2`

## Strategy Used Here

This project uses a strict sequential cooperation rule:

1. Start by cooperating.
2. In each round, Player 1 acts first, then Player 2 acts.
3. Keep playing new rounds while both players cooperate.
4. If a defection happens during a round, finish that round, then end the game.

<div align="center">
  <img src="assets/prisoners_dilemma_matrix.svg" alt="Prisoner's Dilemma payoff matrix" width="400" />
  <p><strong>Display 1: The reward after each round.</strong></p>
</div>

Example: if Player 1 defects, Player 2 still gets a final move in that same round. After Player 2's move, the interaction terminates and no new round starts.

This is stricter than classic Tit-for-Tat. In standard Tit-for-Tat, the game usually continues across many rounds with reciprocal responses. Here, any defection triggers termination after the current sequential round is completed.

## Historical Background (Rapoport / Axelrod)

- Anatol Rapoport is closely associated with Tit-for-Tat in repeated Prisoner's Dilemma research, including work with Albert Chammah in the 1960s.
- In 1980 and 1981, political scientist Robert Axelrod ran computer tournaments for iterated Prisoner's Dilemma strategies.
- Rapoport submitted Tit-for-Tat, and it ranked first in both tournaments.
- Axelrod's 1984 book *The Evolution of Cooperation* made these results widely known and influential.

## Training and Evaluation (RLlib 2.54.0)

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Train with two independent policies and evaluate:

```bash
python scripts/train_eval_rllib.py --train-iters 50 --eval-episodes 20
```

Evaluate only from a saved checkpoint:

```bash
python scripts/train_eval_rllib.py --from-checkpoint checkpoints/sequential_pd_ppo/checkpoint_000050 --eval-episodes 50
```

Useful options:

```bash
# Keep playing until max_rounds even after defection
python scripts/train_eval_rllib.py --no-terminate-on-defection

# Adjust episode length
python scripts/train_eval_rllib.py --max-rounds 100
```
