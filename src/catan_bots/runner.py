"""Play games under the official rules catanatron leaves unenforced."""

from collections.abc import Sequence

from catanatron import Action, Color, Game, GameAccumulator
from catanatron.game import TURNS_LIMIT
from catanatron.models.enums import ActionPrompt, ActionType

from catan_bots.rules.dev_cards import DevCardTimingRule
from catan_bots.rules.discard import DiscardPolicy, DiscardPolicyRegistry


class GameRunner:
    """Own the play loop so rules catanatron skips can be enforced.

    Two rules are layered on top of the engine:

    - **Discard on a seven.** Catanatron offers only
      `Action(color, DISCARD, None)` and discards a random half, so a discard
      choice cannot travel through `Player.decide`. With a policy configured,
      the runner asks the seat's `DiscardPolicy` and executes the explicit
      selection, bypassing the engine's validation (which would reject any
      discard other than the `None` placeholder).
    - **Development-card timing.** Before every decision the playable actions
      are filtered so a card cannot be played the turn it was bought. Bots and
      the engine's own validation both read the filtered list.

    Every ply is otherwise delegated untouched to `Game.play_tick`.

    Attributes:
        discards: Registry resolving each seat's discard policy, or `None` to
            keep catanatron's uniformly random discard.
        dev_card_timing: The timing rule, or `None` when disabled.
        turn_limit: Completed turns after which a game is truncated, matching
            catanatron's own safety valve.
    """

    def __init__(
        self,
        discards: DiscardPolicyRegistry | DiscardPolicy | None = None,
        enforce_dev_card_timing: bool = True,
        turn_limit: int = TURNS_LIMIT,
    ) -> None:
        """Configure the runner.

        Args:
            discards: A registry of per-seat policies, a single policy applied
                to every seat, or `None` to keep catanatron's random discard.
            enforce_dev_card_timing: Whether to forbid playing a development
                card the turn it was bought. Defaults to True (official rules);
                False reproduces the stock engine, for ablations.
            turn_limit: Completed turns after which to truncate a game.
        """
        self.discards = self._as_registry(discards)
        self.dev_card_timing = DevCardTimingRule() if enforce_dev_card_timing else None
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
        if (
            game.state.current_prompt is ActionPrompt.DISCARD
            and self.discards is not None
        ):
            return self._execute_discard(game, accumulators)
        self._enforce_dev_card_timing(game)
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

    def _enforce_dev_card_timing(self, game: Game) -> None:
        """Replace the playable actions with those the timing rule allows.

        Reassigns `state.playable_actions` rather than mutating it, because
        `State.copy` shares that list by reference.

        Args:
            game: Game about to ask its current player for a decision.
        """
        if self.dev_card_timing is None:
            return
        game.state.playable_actions = self.dev_card_timing.legal_actions(game.state)

    @staticmethod
    def _as_registry(
        discards: DiscardPolicyRegistry | DiscardPolicy | None,
    ) -> DiscardPolicyRegistry | None:
        """Normalise the discard configuration to a registry.

        Args:
            discards: A registry, a single policy, or `None`.

        Returns:
            The registry, a registry wrapping the single policy, or `None`.
        """
        if discards is None or isinstance(discards, DiscardPolicyRegistry):
            return discards
        return DiscardPolicyRegistry(discards)

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
        """Return a debug representation naming the active rules."""
        discard = type(self.discards.default).__name__ if self.discards else "engine"
        timing = self.dev_card_timing is not None
        return f"{type(self).__name__}(discard={discard}, dev_card_timing={timing})"


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
