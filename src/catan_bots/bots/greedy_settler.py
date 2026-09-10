"""A heuristic bot that prioritises expansion over everything else."""

import random
from typing import ClassVar

from catanatron import Action, ActionType, Color, Game
from catanatron.models.player import Player


class GreedySettlerBot(Player):
    """Prioritise city/settlement builds; fall back to random otherwise.

    The bot scans `PRIORITY` in order and plays a uniformly random action from
    the first action type that is currently legal. If none of the prioritised
    types are available it plays a uniformly random legal action.

    Attributes:
        PRIORITY: Action types tried in order, most desirable first.
    """

    PRIORITY: ClassVar[tuple[ActionType, ...]] = (
        ActionType.BUILD_CITY,
        ActionType.BUILD_SETTLEMENT,
        ActionType.BUY_DEVELOPMENT_CARD,
        ActionType.BUILD_ROAD,
    )

    def __init__(
        self, color: Color, is_bot: bool = True, seed: int | None = None
    ) -> None:
        """Seat the bot and give it its own random number generator.

        Args:
            color: Colour this bot plays.
            is_bot: Whether catanatron should treat this player as a bot.
            seed: Seed for the bot's private RNG, for reproducible tie-breaks.
                Defaults to `None` (non-deterministic tie-breaks).
        """
        super().__init__(color, is_bot)
        self._seed = seed
        self._rng = random.Random(seed)

    def decide(self, game: Game, playable_actions: list[Action]) -> Action:
        """Choose the action to play this ply.

        Args:
            game: Read-only view of the current game state.
            playable_actions: Legal actions for this bot right now.

        Returns:
            One action drawn from `playable_actions`.

        Raises:
            ValueError: If `playable_actions` is empty.
        """
        if not playable_actions:
            raise ValueError("Cannot decide with no playable actions.")
        candidates = self._preferred_actions(playable_actions)
        return self._rng.choice(candidates)

    def reset_state(self) -> None:
        """Restore the bot's RNG to its seeded starting point between games."""
        self._rng = random.Random(self._seed)

    def _preferred_actions(self, playable_actions: list[Action]) -> list[Action]:
        """Select the legal actions of the highest available priority type.

        Args:
            playable_actions: Legal actions for this bot right now.

        Returns:
            All actions of the first `PRIORITY` type that has any legal action,
            or every legal action when no prioritised type is available.
        """
        for action_type in self.PRIORITY:
            matches = [a for a in playable_actions if a.action_type == action_type]
            if matches:
                return matches
        return list(playable_actions)
