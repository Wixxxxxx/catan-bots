"""Measure a policy against fixed opponents, with confidence intervals.

Self-play cannot measure itself: a policy playing copies of itself wins about
1 in `num_players` games however strong it is. Progress is measured here
instead, against fixed baselines that do not change: the policy plays one
seat, the baseline plays every other, and the policy's seat rotates.
"""

import math
from dataclasses import dataclass
from multiprocessing.pool import Pool
from typing import Any

from catan_bots.training import workers


def wilson_interval(wins: int, games: int, z: float = 1.96) -> tuple[float, float]:
    """Compute the Wilson score interval for a win rate.

    More accurate than the normal approximation for small samples and for
    rates near 0 or 1.

    Args:
        wins: Games won.
        games: Games played.
        z: Normal quantile; 1.96 gives a 95% interval.

    Returns:
        Lower and upper bounds; `(0.0, 1.0)` with no games.
    """
    if games == 0:
        return 0.0, 1.0
    rate = wins / games
    denominator = 1 + z * z / games
    centre = (rate + z * z / (2 * games)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / games + z * z / (4 * games * games))
    return max(0.0, centre - margin / denominator), min(
        1.0, centre + margin / denominator
    )


@dataclass(frozen=True)
class EvaluationResult:
    """A policy's record against one baseline.

    Attributes:
        opponent: Baseline name.
        games: Games played.
        wins: Games the policy won.
        truncated: Games cut off without a winner.
        chance: The win rate expected by chance: 1 / players.
    """

    opponent: str
    games: int
    wins: int
    truncated: int
    chance: float

    @property
    def win_rate(self) -> float:
        """Share of games won."""
        return self.wins / self.games if self.games else 0.0

    @property
    def interval(self) -> tuple[float, float]:
        """95% Wilson confidence interval on the win rate."""
        return wilson_interval(self.wins, self.games)

    @property
    def beats_chance(self) -> bool:
        """Whether the whole confidence interval lies above chance."""
        return self.interval[0] > self.chance

    def summary(self) -> str:
        """Describe the result in one line.

        Returns:
            For example `"vs greedy_trader: 0.312 (95% CI [0.268, 0.360],
            n=400; chance 0.250)"`.
        """
        low, high = self.interval
        return (
            f"vs {self.opponent}: {self.win_rate:.3f} (95% CI [{low:.3f}, {high:.3f}], "
            f"n={self.games}; chance {self.chance:.3f})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to plain values for logging.

        Returns:
            The result's fields plus win rate and interval.
        """
        low, high = self.interval
        return {
            "opponent": self.opponent,
            "games": self.games,
            "wins": self.wins,
            "truncated": self.truncated,
            "win_rate": self.win_rate,
            "ci_low": low,
            "ci_high": high,
            "chance": self.chance,
        }


class Evaluator:
    """Spread evaluation games over a worker pool.

    Attributes:
        pool: Worker pool started by `workers.make_pool`.
        num_workers: Workers in the pool.
        num_players: Seats per game.
    """

    def __init__(self, pool: Pool, num_workers: int, num_players: int) -> None:
        """Bind to a pool.

        Args:
            pool: Worker pool.
            num_workers: Workers in the pool.
            num_players: Seats per game.
        """
        self.pool = pool
        self.num_workers = num_workers
        self.num_players = num_players

    def evaluate(
        self,
        weights: dict[str, Any],
        opponent: str,
        games: int,
        seed: int,
        deterministic: bool = False,
    ) -> EvaluationResult:
        """Play the policy against one baseline.

        Args:
            weights: Policy snapshot.
            opponent: Baseline filling every other seat.
            games: Games to play in total.
            seed: Seed for game seeds and sampling.
            deterministic: Play the most likely action instead of sampling.

        Returns:
            The policy's record.
        """
        tasks = self._split(weights, opponent, games, seed, deterministic)
        tallies = self.pool.map(workers.evaluate, tasks)
        return EvaluationResult(
            opponent=opponent,
            games=sum(t.games for t in tallies),
            wins=sum(t.wins for t in tallies),
            truncated=sum(t.truncated for t in tallies),
            chance=1 / self.num_players,
        )

    def _split(
        self,
        weights: dict[str, Any],
        opponent: str,
        games: int,
        seed: int,
        deterministic: bool,
    ) -> list[workers.EvaluateTask]:
        """Divide games between workers, keeping seat rotation continuous.

        Args:
            weights: Policy snapshot.
            opponent: Baseline name.
            games: Games in total.
            seed: Base seed; each task gets its own.
            deterministic: Play the most likely action instead of sampling.

        Returns:
            One task per worker that has games to play.
        """
        share, extra = divmod(games, self.num_workers)
        tasks, first = [], 0
        for index in range(self.num_workers):
            count = share + (index < extra)
            if count:
                tasks.append(
                    workers.EvaluateTask(
                        weights,
                        opponent,
                        count,
                        first,
                        seed * 1009 + index,
                        deterministic,
                    )
                )
            first += count
        return tasks
