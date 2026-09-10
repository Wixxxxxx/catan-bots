"""`GameAccumulator` hooks that collect data alongside a running game."""

from collections import Counter

from catanatron import Action, Game, GameAccumulator


class ActionTypeCounter(GameAccumulator):
    """Count how many times each `ActionType` occurs in a single game.

    Attributes:
        counts: Number of actions seen per `ActionType` name. Reset at the
            start of every game the accumulator is attached to.
    """

    def __init__(self) -> None:
        """Start with an empty tally."""
        super().__init__()
        self.counts: Counter[str] = Counter()

    def before(self, game: Game) -> None:
        """Clear the tally so the accumulator can be reused across games.

        Args:
            game: The game about to start; unused.
        """
        self.counts = Counter()

    def step(self, game_before_action: Game, action: Action) -> None:
        """Record one action.

        Args:
            game_before_action: Game state right before the action; unused.
            action: The action that was taken.
        """
        self.counts[action.action_type.name] += 1

    def ranked_counts(self) -> list[tuple[str, int]]:
        """List the tallied action types from most to least frequent.

        Returns:
            `(action_type_name, count)` pairs in descending count order.
        """
        return self.counts.most_common()

    @property
    def total_actions(self) -> int:
        """Total number of actions counted in the current game."""
        return sum(self.counts.values())
