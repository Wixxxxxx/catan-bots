---
name: train-rl-agent
description: Scaffold a reproducible reinforcement-learning training run for a Catan agent following this repo's standards — a typed config, action-masked PPO (MaskablePPO) on a vectorized env, checkpointing, logging, and evaluation against baselines with confidence intervals. Use when setting up training or a new experiment.
---

# Train RL Agent

This skill creates a reproducible training + evaluation run that conforms to
[CLAUDE.md](../../../CLAUDE.md).

## When to use

- Setting up training for the Catan env for the first time.
- Launching a new experiment (new algorithm, hyperparameters, or reward).

## Prerequisites

Dependencies via `uv` (the env must already exist with an action mask):

```bash
uv add stable-baselines3 sb3-contrib tensorboard
uv add --dev pytest
```

## Steps

1. **Define a typed config** (algorithm, hyperparameters, seeds, total steps,
   paths) — never hardcode hyperparameters in the loop.
2. **Establish baselines** (random + heuristic) before training; every result is
   reported relative to them.
3. **Wrap the env** with an action-mask adapter and `VecEnv` for parallelism;
   keep a separate eval env.
4. **Train** with checkpointing + TensorBoard logging; run a short smoke train
   first.
5. **Evaluate** over many seeded games and report win rate ± confidence interval.
6. **Persist** config, seed, git SHA, checkpoint, and metrics together.
7. **Format & lint**: `uv run ruff format . && uv run ruff check .`.

## Layout

```text
src/catan_bots/training/
├── config.py       # TrainingConfig dataclass
├── trainer.py      # Trainer class — orchestrates a run
├── evaluator.py    # Evaluator class — win rate vs. baselines with CIs
└── baselines.py    # RandomAgent, HeuristicAgent
```

## Config + trainer skeleton

```python
"""Reproducible MaskablePPO training for Catan."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainingConfig:
    """Hyperparameters and paths for a single training run.

    Attributes:
        total_timesteps: Total environment steps to train for.
        seed: Master seed for reproducibility.
        n_envs: Number of parallel environments.
        learning_rate: Optimizer learning rate.
        checkpoint_dir: Where checkpoints and metrics are written.
    """

    total_timesteps: int = 1_000_000
    seed: int = 0
    n_envs: int = 8
    learning_rate: float = 3e-4
    checkpoint_dir: Path = field(default_factory=lambda: Path("runs"))


class Trainer:
    """Orchestrate a reproducible RL training run for Catan."""

    def __init__(self, config: TrainingConfig) -> None:
        """Store config and prepare output directories.

        Args:
            config: The training configuration for this run.
        """
        self.config = config

    def train(self) -> Path:
        """Run training to completion and checkpoint the policy.

        Returns:
            Path to the saved final checkpoint.
        """
        raise NotImplementedError
```

```python
def mask_fn(env):
    """Return the current action mask for MaskablePPO.

    Args:
        env: The wrapped Catan environment exposing ``action_mask()``.

    Returns:
        A boolean array of legal actions for the current state.
    """
    return env.action_mask()


# from sb3_contrib import MaskablePPO
# from sb3_contrib.common.wrappers import ActionMasker
# env = ActionMasker(CatanEnv(), mask_fn)
# model = MaskablePPO("MultiInputPolicy", env, seed=config.seed,
#                     tensorboard_log=str(config.checkpoint_dir), verbose=1)
# model.learn(total_timesteps=config.total_timesteps)
```

## Evaluation rule

Report results with uncertainty, never a single number:

```text
MaskablePPO vs heuristic: 0.62 win rate (95% CI [0.58, 0.66], n=600 games)
```

Randomize or balance seating to control for first-player advantage.

## Done when

- A run produces a checkpoint that beats the heuristic baseline by a
  statistically meaningful margin.
- Config, seed, git SHA, metrics, and checkpoint are saved together and
  reproducible.
