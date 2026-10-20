"""Who sits at the table: the current policy, past snapshots, or baselines.

Pure self-play in a four-player game can drift into strategies that only beat
copies of the latest policy. Mixing in past versions of itself and fixed
baseline bots keeps the learner honest — the same idea as OpenAI Five playing
most games against itself and some against past versions.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from catan_bots.training.agents import Agent, PolicyAgent, make_baseline
from catan_bots.training.policy import ActorCritic

LEARNER = "learner"
SNAPSHOT = "snapshot"
BASELINE = "baseline"


@dataclass(frozen=True)
class SeatPlan:
    """Who plays one seat in one game.

    Attributes:
        kind: `"learner"`, `"snapshot"` or `"baseline"`.
        name: Snapshot file path or baseline name; empty for the learner.
    """

    kind: str
    name: str = ""

    @property
    def is_learner(self) -> bool:
        """Whether this seat is played, and learned from, by the current policy."""
        return self.kind == LEARNER


class League:
    """Past snapshots of the policy, kept on disk.

    Attributes:
        directory: Where snapshot files live.
        max_snapshots: Snapshots kept; adding beyond this drops the oldest.
    """

    def __init__(self, directory: Path, max_snapshots: int) -> None:
        """Open a league directory, picking up snapshots already in it.

        Args:
            directory: Snapshot directory; created if missing.
            max_snapshots: Snapshots to keep.
        """
        self.directory = directory
        self.max_snapshots = max_snapshots
        directory.mkdir(parents=True, exist_ok=True)

    @property
    def snapshots(self) -> list[Path]:
        """Snapshot files, oldest first."""
        return sorted(self.directory.glob("snapshot_*.pt"))

    def add(self, policy: ActorCritic, iteration: int) -> Path:
        """Save the policy as a new snapshot, dropping the oldest beyond the limit.

        Args:
            policy: The current policy.
            iteration: Training iteration, used in the file name.

        Returns:
            The new snapshot's path.
        """
        path = self.directory / f"snapshot_{iteration:07d}.pt"
        torch.save(policy.snapshot(), path)
        for stale in self.snapshots[: -self.max_snapshots]:
            stale.unlink()
        return path


class LineupSampler:
    """Draw who sits in each seat for a game.

    With probability `self_play_fraction` every seat is the learner.
    Otherwise one to `num_players - 1` seats are replaced: each by a
    baseline with probability `baseline_seat_fraction` (always, while there
    are no snapshots yet), else by a past snapshot. At least one seat is
    always the learner.

    Attributes:
        num_players: Seats per game.
        self_play_fraction: Share of games with the learner in every seat.
        baseline_seat_fraction: Share of replaced seats given to baselines.
        baselines: Baseline names to draw from.
        snapshots: Snapshot paths to draw from.
    """

    def __init__(
        self,
        num_players: int,
        self_play_fraction: float,
        baseline_seat_fraction: float,
        baselines: tuple[str, ...],
        snapshots: list[str],
        rng: np.random.Generator,
    ) -> None:
        """Configure the sampler.

        Args:
            num_players: Seats per game.
            self_play_fraction: Share of games with the learner everywhere.
            baseline_seat_fraction: Share of replaced seats given to baselines.
            baselines: Baseline names to draw from.
            snapshots: Snapshot paths to draw from.
            rng: Generator for the draws.
        """
        self.num_players = num_players
        self.self_play_fraction = self_play_fraction
        self.baseline_seat_fraction = baseline_seat_fraction
        self.baselines = baselines
        self.snapshots = snapshots
        self.rng = rng

    def sample(self) -> list[SeatPlan]:
        """Draw one game's seating.

        Returns:
            A plan per seat, indexed like the environment's agents.
        """
        lineup = [SeatPlan(LEARNER)] * self.num_players
        no_opponents = not self.baselines and not self.snapshots
        if no_opponents or self.rng.random() < self.self_play_fraction:
            return lineup
        replaced = int(self.rng.integers(1, self.num_players))
        for seat in self.rng.choice(self.num_players, size=replaced, replace=False):
            lineup[seat] = self._opponent()
        return lineup

    def _opponent(self) -> SeatPlan:
        """Draw one opponent for a replaced seat.

        Returns:
            A baseline or snapshot plan.
        """
        use_baseline = not self.snapshots or (
            self.baselines and self.rng.random() < self.baseline_seat_fraction
        )
        if use_baseline:
            return SeatPlan(BASELINE, str(self.rng.choice(self.baselines)))
        return SeatPlan(SNAPSHOT, str(self.rng.choice(self.snapshots)))


class OpponentFactory:
    """Turn seat plans into agents, loading each snapshot from disk once.

    Attributes:
        max_cached: Snapshot policies kept in memory.
    """

    def __init__(self, max_cached: int = 32) -> None:
        """Create an empty factory.

        Args:
            max_cached: Snapshot policies kept in memory.
        """
        self.max_cached = max_cached
        self._policies: dict[str, ActorCritic] = {}

    def build(self, plan: SeatPlan) -> Agent:
        """Create the agent for a non-learner seat.

        Args:
            plan: A snapshot or baseline plan.

        Returns:
            A fresh agent for one game.

        Raises:
            ValueError: If asked to build the learner, which the caller plays.
        """
        if plan.kind == SNAPSHOT:
            return PolicyAgent(self._policy(plan.name))
        if plan.kind == BASELINE:
            return make_baseline(plan.name)
        raise ValueError("The learner seat is played by the caller.")

    def _policy(self, path: str) -> ActorCritic:
        """Load a snapshot, caching it and evicting the oldest beyond the limit.

        Args:
            path: Snapshot file.

        Returns:
            The snapshot's network.
        """
        if path not in self._policies:
            if len(self._policies) >= self.max_cached:
                self._policies.pop(next(iter(self._policies)))
            snapshot = torch.load(path, map_location="cpu", weights_only=True)
            self._policies[path] = ActorCritic.from_snapshot(snapshot)
        return self._policies[path]
