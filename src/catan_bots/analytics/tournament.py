"""Repeated head-to-head matches used to compare bots."""

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass

from catanatron import Color, Game

from catan_bots.games import GameFactory

DRAW_LABEL = "draw"


@dataclass(frozen=True)
class TournamentResult:
    """Outcome of a fixed number of games between one roster of bots.

    Attributes:
        win_counts: Games won per colour; `None` counts games that ended
            without a winner.
        num_games: Total games played.
        turns_played: Completed turns per game, in play order.
    """

    win_counts: dict[Color | None, int]
    num_games: int
    turns_played: tuple[int, ...]

    def win_rate(self, color: Color | None) -> float:
        """Compute the share of games a colour won.

        Args:
            color: Colour to score, or `None` for games with no winner.

        Returns:
            Wins divided by games played, or 0.0 when no games were played.
        """
        if self.num_games == 0:
            return 0.0
        return self.win_counts.get(color, 0) / self.num_games

    @property
    def mean_turns(self) -> float:
        """Average number of completed turns per game."""
        if not self.turns_played:
            return 0.0
        return sum(self.turns_played) / len(self.turns_played)

    def standings(self) -> list[tuple[str, int, float]]:
        """Rank the results from most to fewest wins.

        Returns:
            `(label, wins, win_rate)` rows, best result first; the label is the
            colour's name, or `"draw"` for games with no winner.
        """
        rows = [
            (color.value if color else DRAW_LABEL, wins, self.win_rate(color))
            for color, wins in self.win_counts.items()
        ]
        return sorted(rows, key=lambda row: -row[1])

    def as_text(self) -> str:
        """Render the standings as a fixed-width text table.

        Returns:
            One line per colour, plus a line for games without a winner.
        """
        lines = [f"Win rates over {self.num_games} games:"]
        for label, wins, rate in self.standings():
            lines.append(f"  {label:12s}: {wins:3d}/{self.num_games}  ({rate:.0%})")
        return "\n".join(lines)


class Tournament:
    """Play a roster of bots against each other over many seeded games.

    Attributes:
        factory: Factory building each game of the tournament.
        num_games: Number of games a `run()` plays.
        first_seed: Seed of the first game; game `i` uses `first_seed + i`.
    """

    def __init__(
        self,
        factory: GameFactory,
        num_games: int = 20,
        first_seed: int = 0,
    ) -> None:
        """Configure the tournament.

        Args:
            factory: Factory building each game.
            num_games: Number of games to play per run.
            first_seed: Seed of the first game, incremented by one per game so
                the whole tournament is reproducible.

        Raises:
            ValueError: If `num_games` is not positive.
        """
        if num_games <= 0:
            raise ValueError(f"num_games must be positive, got {num_games}.")
        self.factory = factory
        self.num_games = num_games
        self.first_seed = first_seed

    def run(self) -> TournamentResult:
        """Play every game and tally the winners.

        Returns:
            The aggregated `TournamentResult`.
        """
        win_counts: Counter[Color | None] = Counter(
            {player.color: 0 for player in self.factory.players}
        )
        win_counts[None] = 0
        turns: list[int] = []
        for game in self.play_games():
            win_counts[game.winning_color()] += 1
            turns.append(game.state.num_turns)
        return TournamentResult(
            win_counts=dict(win_counts),
            num_games=self.num_games,
            turns_played=tuple(turns),
        )

    def play_games(self) -> Iterator[Game]:
        """Play each game in turn, yielding it once finished.

        Yields:
            Each finished `Game`, in seed order, so callers can collect their
            own per-game statistics.
        """
        for offset in range(self.num_games):
            yield self.factory.play(seed=self.first_seed + offset)
