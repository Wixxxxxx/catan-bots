"""Play games to completion while resolving discards through a policy."""

from collections.abc import Sequence

from catanatron import Action, Color, Game, GameAccumulator
from catanatron.game import TURNS_LIMIT
from catanatron.models.enums import ActionPrompt, ActionType

from catan_bots.rules.discard import DiscardPolicy, DiscardPolicyRegistry


class GameRunner:
    """Run games in which discarding on a seven is a real decision.

    Catanatron offers only `Action(color, DISCARD, None)` as legal and then
    discards a random half, so a discard choice cannot travel through
    `Player.decide`. This runner owns the play loop instead: on a discard
    prompt it asks the seat's `DiscardPolicy` and executes the explicit
    selection, bypassing the engine's action validation (which would reject any
    discard other than the `None` placeholder). Every other ply is delegated
    untouched to `Game.play_tick`.

    Attributes:
        discards: Registry resolving each seat's discard policy.
        turn_limit: Completed turns after which a game is truncated, matching
            catanatron's own safety valve.
    """

    def __init__(
        self,
        discards: DiscardPolicyRegistry | DiscardPolicy,
        turn_limit: int = TURNS_LIMIT,
    ) -> None:
        """Configure the runner.

        Args:
            discards: Either a registry of per-seat policies, or a single
                policy applied to every seat.
            turn_limit: Completed turns after which to truncate a game.
        """
        self.discards = (
            discards
            if isinstance(discards, DiscardPolicyRegistry)
            else DiscardPolicyRegistry(discards)
        )
        self.turn_limit = turn_limit

    def play(self, game: Game, accumulators: Sequence[GameAccumulator] = ()) -> Game:
        """Run a game to completion or truncation.

        Mirrors `Game.play`'s accumulator contract: `before` on a snapshot of
        the starting position, `step` per action, `after` on a snapshot of the
        final position.

        Args:
            game: Game to run. Advanced in place.
            accumulators: `GameAccumulator` hooks to notify.

        Returns:
            The same game, now finished or truncated.
        """
        for accumulator in accumulators:
            accumulator.before(game.copy())
        while game.winning_color() is None and game.state.num_turns < self.turn_limit:
            self.tick(game, accumulators)
        for accumulator in accumulators:
            accumulator.after(game.copy())
        return game

    def tick(self, game: Game, accumulators: Sequence[GameAccumulator] = ()) -> Action:
        """Advance the game by one ply.

        Args:
            game: Game to advance in place.
            accumulators: `GameAccumulator` hooks to notify.

        Returns:
            The resolved action that was executed.
        """
        if game.state.current_prompt is ActionPrompt.DISCARD:
            return self._execute_discard(game, accumulators)
        return game.play_tick(accumulators=list(accumulators))

    def _execute_discard(
        self, game: Game, accumulators: Sequence[GameAccumulator]
    ) -> Action:
        """Resolve the current discard prompt through the seat's policy.

        Args:
            game: Game sitting on a discard prompt; advanced in place.
            accumulators: `GameAccumulator` hooks to notify.

        Returns:
            The executed `DISCARD` action, carrying the discarded cards as its
            value rather than catanatron's `None` placeholder.
        """
        color = game.state.current_color()
        selection = self.discards.policy_for(color).resolve(game, color)
        action = Action(color, ActionType.DISCARD, selection)
        self._notify_step(game, action, accumulators)
        return game.execute(action, validate_action=False)

    @staticmethod
    def _notify_step(
        game: Game, action: Action, accumulators: Sequence[GameAccumulator]
    ) -> None:
        """Send one pre-action snapshot to every accumulator.

        Args:
            game: Game state right before the action is applied.
            action: Action about to be executed.
            accumulators: `GameAccumulator` hooks to notify.
        """
        if not accumulators:
            return
        snapshot = game.copy()
        for accumulator in accumulators:
            accumulator.step(snapshot, action)

    def __repr__(self) -> str:
        """Return a debug representation naming the default discard policy."""
        return f"{type(self).__name__}({type(self.discards.default).__name__})"


def seat_registry(
    policies: dict[Color, DiscardPolicy], default: DiscardPolicy
) -> DiscardPolicyRegistry:
    """Build a registry that gives named seats their own discard policy.

    Args:
        policies: Policy per colour, for comparing rules inside one game.
        default: Policy for colours not named in `policies`.

    Returns:
        A `DiscardPolicyRegistry` wrapping those choices.
    """
    return DiscardPolicyRegistry(default, policies)
