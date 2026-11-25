"""Every setting of a self-play training run, saved alongside its results."""

import dataclasses
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from catanatron.game import TURNS_LIMIT

ARCHITECTURES = ("mlp", "graph")
UPDATE_DEVICES = ("cpu", "mps", "cuda", "auto")


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
        architecture: The policy network: `"mlp"` (one flat vector through
            fully connected layers) or `"graph"` (message passing over the
            board's nodes, edges and tiles).
        hidden_sizes: Widths of the MLP's hidden layers; unused by `"graph"`.
        graph_width: Width of every row vector in the graph network.
        graph_rounds: Rounds of message passing in the graph network.
        learning_rate: Adam learning rate at the start of the run.
        final_learning_rate_fraction: Share of `learning_rate` reached at
            `total_steps`, decaying linearly; 1.0 keeps it constant.
        target_kl: Stop an update's remaining minibatches once the measured
            KL divergence from the policy that collected the batch exceeds
            1.5 times this; 0 disables the limit.
        gamma: Discount per own decision.
        gae_lambda: GAE smoothing.
        clip_coef: PPO ratio clipping.
        entropy_coef: Weight of the entropy bonus.
        value_coef: Weight of the value loss.
        max_grad_norm: Gradient-norm clipping.
        update_epochs: Passes over each batch per update.
        minibatch_size: Target samples per gradient step. Each batch is split
            into equal chunks of at least this size; at least 2.
        update_device: Where PPO updates run: `"cpu"`, `"mps"` (Apple
            GPU), `"cuda"`, or `"auto"` for the first available of `"cuda"`,
            `"mps"` and `"cpu"`. Games are always played on CPU workers; the
            network is moved to this device for each update and back.
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
        eval_references: Fixed policy files (league snapshots or checkpoints,
            usually from an earlier run) the policy is evaluated against
            every time. They never change, so they show progress after the
            baselines are beaten.
        eval_oldest_snapshot: Also evaluate against the oldest snapshot still
            in the league, which shows whether the policy beats its own
            recent past.
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
    architecture: str = "mlp"
    hidden_sizes: tuple[int, ...] = (256, 256)
    graph_width: int = 64
    graph_rounds: int = 3
    learning_rate: float = 2.5e-4
    final_learning_rate_fraction: float = 1.0
    target_kl: float = 0.0
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 4096
    update_device: str = "cpu"
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
    eval_references: tuple[str, ...] = ()
    eval_oldest_snapshot: bool = False

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
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {ARCHITECTURES}.")
        if self.update_device not in UPDATE_DEVICES:
            raise ValueError(f"update_device must be one of {UPDATE_DEVICES}.")
        if self.graph_width < 1 or self.graph_rounds < 1:
            raise ValueError("graph_width and graph_rounds must be positive.")
        for name in (
            "self_play_fraction",
            "baseline_seat_fraction",
            "final_learning_rate_fraction",
        ):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")
        if self.target_kl < 0:
            raise ValueError("target_kl cannot be negative.")
        if self.minibatch_size < 2:
            raise ValueError(
                "minibatch_size must be at least 2 to normalise advantages."
            )
        if self.minibatch_size > self.steps_per_iteration:
            raise ValueError("minibatch_size cannot exceed steps_per_iteration.")
        if self.num_workers < 1 or self.total_steps < 1:
            raise ValueError("num_workers and total_steps must be positive.")

    def learning_rate_at(self, step: int) -> float:
        """Compute the scheduled learning rate after `step` learner decisions.

        The rate falls linearly from `learning_rate` to
        `learning_rate * final_learning_rate_fraction` at `total_steps`, and
        stays there beyond it. Resuming with a larger `total_steps` stretches
        the schedule, so the rate rises to where the longer run would be.

        Args:
            step: Learner decisions trained on so far.

        Returns:
            The learning rate for the next update.
        """
        progress = min(step / self.total_steps, 1.0)
        decay = progress * (1.0 - self.final_learning_rate_fraction)
        return self.learning_rate * (1.0 - decay)

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
