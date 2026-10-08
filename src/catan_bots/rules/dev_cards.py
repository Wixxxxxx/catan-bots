"""Development-card timing: a card cannot be played the turn it was bought.

Official rule: a development card may not be played in the turn it was bought.
The one exception — revealing victory-point cards to reach 10 points — needs no
handling, because catanatron never "plays" VP cards: they count towards
`ACTUAL_VICTORY_POINTS` the moment they are bought.

Catanatron's `player_can_play_dev` checks only the one-card-per-turn flag and
that a copy is in hand, so a card bought this turn is immediately playable.
Measured over 300 games of `WeightedRandomPlayer` bots, about half of all
development-card plays (55% in 2-player, 50% in 4-player) were these illegal
same-turn plays — mostly knights, then monopoly and year of plenty.

Which cards were bought this turn is derived from the action log rather than
tracked separately: catanatron logs each purchase as
`Action(color, BUY_DEVELOPMENT_CARD, card_drawn)`, and `State.copy` copies the
log, so the rule holds on any state, including copies used for lookahead.
"""

from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import ClassVar

from catanatron import Action, ActionType, Color
from catanatron.state import State
from catanatron.state_functions import player_key


class DevCardTimingRule:
    """Remove plays of development cards that were bought this turn.

    The restriction is per copy, not per card type: a player holding an older
    knight who buys a second knight may still play one knight this turn.

    Attributes:
        PLAY_ACTIONS: The card each play action spends.
    """

    PLAY_ACTIONS: ClassVar[dict[ActionType, str]] = {
        ActionType.PLAY_KNIGHT_CARD: "KNIGHT",
        ActionType.PLAY_YEAR_OF_PLENTY: "YEAR_OF_PLENTY",
        ActionType.PLAY_MONOPOLY: "MONOPOLY",
        ActionType.PLAY_ROAD_BUILDING: "ROAD_BUILDING",
    }

    def legal_actions(self, state: State) -> list[Action]:
        """Filter the current playable actions down to those this rule allows.

        Returns a new list and never mutates `state.playable_actions`, which
        `State.copy` shares by reference between a state and its copies.

        Args:
            state: Game state whose current decider's actions are filtered.

        Returns:
            The playable actions minus plays of cards bought this turn.
        """
        color = state.current_color()
        bought = self.cards_bought_this_turn(state, color)
        if not bought:
            return list(state.playable_actions)
        return [
            action
            for action in state.playable_actions
            if self._is_allowed(state, color, action, bought)
        ]

    @staticmethod
    def cards_bought_this_turn(state: State, color: Color) -> Counter[str]:
        """Count the development cards a player has bought in the current turn.

        Scans the action log backwards to the most recent `END_TURN`.

        Args:
            state: Game state whose action log is read.
            color: Colour of the player whose purchases are counted.

        Returns:
            Number of copies bought this turn, per development-card name.
        """
        bought: Counter[str] = Counter()
        for action in reversed(state.actions):
            if action.action_type is ActionType.END_TURN:
                break
            if (
                action.action_type is ActionType.BUY_DEVELOPMENT_CARD
                and action.color is color
            ):
                bought[action.value] += 1
        return bought

    @staticmethod
    def playable_copies(
        state: State, color: Color, card: str, bought: Counter[str]
    ) -> int:
        """Count the copies of one card a player may play right now.

        Args:
            state: Game state to read.
            color: Colour of the player.
            card: Development-card name, e.g. `"KNIGHT"`.
            bought: Cards that player bought this turn.

        Returns:
            Copies held that were bought before this turn; never negative.
        """
        held = state.player_state[f"{player_key(state, color)}_{card}_IN_HAND"]
        return max(0, held - bought[card])

    @classmethod
    def count_violations(cls, actions: Iterable[Action]) -> int:
        """Audit an action log for plays of cards bought the same turn.

        Cards only enter a hand by purchase and leave it by being played, so
        the copies a player may play are those bought in earlier turns and not
        yet spent. Useful for auditing recorded or training games.

        Args:
            actions: A game's action log, oldest first.

        Returns:
            Number of plays of a card with no copy bought before that turn.
        """
        owned: defaultdict[Color, Counter[str]] = defaultdict(Counter)
        bought: defaultdict[Color, Counter[str]] = defaultdict(Counter)
        violations = 0
        for action in actions:
            kind, color = action.action_type, action.color
            if kind is ActionType.BUY_DEVELOPMENT_CARD:
                bought[color][action.value] += 1
            elif kind in cls.PLAY_ACTIONS:
                card = cls.PLAY_ACTIONS[kind]
                if owned[color][card] > 0:
                    owned[color][card] -= 1
                else:
                    violations += 1
                    bought[color][card] -= 1
            elif kind is ActionType.END_TURN:
                owned[color].update(bought[color])
                bought[color].clear()
        return violations

    def _is_allowed(
        self, state: State, color: Color, action: Action, bought: Counter[str]
    ) -> bool:
        """Judge whether one playable action survives the timing rule.

        Args:
            state: Game state to read.
            color: Colour of the deciding player.
            action: Candidate action.
            bought: Cards that player bought this turn.

        Returns:
            False only for a play of a card with no copy bought before this turn.
        """
        card = self.PLAY_ACTIONS.get(action.action_type)
        if card is None:
            return True
        return self.playable_copies(state, color, card, bought) > 0
