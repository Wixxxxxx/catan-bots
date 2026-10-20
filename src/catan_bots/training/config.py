"""Every setting of a self-play training run, saved alongside its results."""

import dataclasses
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from catanatron.game import TURNS_LIMIT


def default_workers() -> int:
    """Choose a worker count that leaves the machine responsive.

    Returns:
        CPU cores minus two, and at least one.
    """
    return max(1, (os.cpu_count() or 2) - 2)


@dataclass
class TrainingConfig:
    """Hyperparameters, league settings and paths for one training run.

    One network plays every learner seat (self-play). Each seat's moves are
    kept as their own trajectory, so credit and the end-of-game reward reach
    the seat that earned them.

    Attributes:
        run_name: Prefix of the run's directory name.
        output_dir: Directory runs are written under.
        seed: Master seed; every worker and game seed derives from it.
        total_steps: Learner decisions to train on before stopping.
        num_workers: Processes playing games in parallel.
        steps_per_iteration: Learner decisions collected per PPO update.
        num_players: Seats per game, 2 to 4.
        trading: Whether domestic trading is on.
        turn_limit: Turns after which a game is truncated.
        hidden_sizes: Widths of the network's hidden layers.
        learning_rate: Adam learning rate.
        gamma: Discount per own decision.
        gae_lambda: GAE smoothing.
        clip_coef: PPO ratio clipping.
        entropy_coef: Weight of the entropy bonus.
        value_coef: Weight of the value loss.
        max_grad_norm: Gradient-norm clipping.
        update_epochs: Passes over each batch per update.
        minibatch_size: Samples per gradient step.
        vp_shaping: Potential-based shaping weight on own victory points; 0
            disables it. Potential-based shaping cannot change which policy
            is optimal.
        self_play_fraction: Share of games with the current policy in every
            seat; the rest replace one to three seats with opponents.
        baseline_seat_fraction: Share of replaced seats given to baseline
            bots rather than past snapshots.
        baselines: Baseline bots that may fill replaced seats.
        snapshot_every: Iterations between league snapshots.
        max_snapshots: Snapshots kept; the oldest are dropped.
        checkpoint_every: Iterations between checkpoints.
        eval_every: Iterations between evaluations; 0 disables them.
        eval_games: Games per opponent per evaluation.
        eval_opponents: Bots the policy is evaluated against.
    """

    run_name: str = "selfplay"
    output_dir: str = "runs"
    seed: int = 0
    total_steps: int = 20_000_000
    num_workers: int = field(default_factory=default_workers)
    steps_per_iteration: int = 32_768
    num_players: int = 4
    trading: bool = True
    turn_limit: int = TURNS_LIMIT
    hidden_sizes: tuple[int, ...] = (256, 256)
    learning_rate: float = 2.5e-4
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 4096
    vp_shaping: float = 0.0
    self_play_fraction: float = 0.8
    baseline_seat_fraction: float = 0.25
    baselines: tuple[str, ...] = ("greedy_trader", "greedy_settler")
    snapshot_every: int = 10
    max_snapshots: int = 20
    checkpoint_every: int = 5
    eval_every: int = 20
    eval_games: int = 200
    eval_opponents: tuple[str, ...] = ("random", "greedy_settler", "greedy_trader")

    def __post_init__(self) -> None:
        """Normalise list values to tuples and reject impossible settings.

        Raises:
            ValueError: If a setting is out of range.
        """
        for item in dataclasses.fields(self):
            value = getattr(self, item.name)
            if isinstance(value, list):
                setattr(self, item.name, tuple(value))
        self._validate()

    def _validate(self) -> None:
        """Check every setting is in range.

        Raises:
            ValueError: Naming the first setting that is out of range.
        """
        if not 2 <= self.num_players <= 4:
            raise ValueError("num_players must be 2-4.")
        for name in ("self_play_fraction", "baseline_seat_fraction"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")
        if self.minibatch_size > self.steps_per_iteration:
            raise ValueError("minibatch_size cannot exceed steps_per_iteration.")
        if self.num_workers < 1 or self.total_steps < 1:
            raise ValueError("num_workers and total_steps must be positive.")

    def to_dict(self) -> dict[str, Any]:
        """Convert the configuration to plain JSON-ready values.

        Returns:
            Setting name to value, with tuples as lists.
        """
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TrainingConfig":
        """Build a configuration from plain values, rejecting unknown keys.

        Args:
            values: Setting name to value.

        Returns:
            The configuration.

        Raises:
            ValueError: If a key is not a setting.
        """
        known = {item.name for item in dataclasses.fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown settings: {sorted(unknown)}")
        return cls(**values)

    def save(self, path: Path) -> None:
        """Write the configuration as JSON.

        Args:
            path: File to write.
        """
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "TrainingConfig":
        """Read a configuration written by `save`, or by hand.

        Args:
            path: JSON file to read.

        Returns:
            The configuration.
        """
        return cls.from_dict(json.loads(path.read_text()))

    def with_overrides(self, overrides: dict[str, Any]) -> "TrainingConfig":
        """Return a copy with some settings replaced.

        Args:
            overrides: Setting name to new value.

        Returns:
            The updated configuration, validated.
        """
        return self.from_dict({**self.to_dict(), **overrides})
