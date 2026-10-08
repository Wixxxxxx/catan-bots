"""A heuristic bot that trades towards its nearest build."""

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import ClassVar

from catanatron import Color, Game
from catanatron.models.enums import RESOURCES
from catanatron.state import State
from catanatron.state_functions import player_key

from catan_bots.bots.greedy_settler import GreedySettlerBot
from catan_bots.bots.trading import TradingPlayer
from catan_bots.rules.builds import nearest_target
from catan_bots.rules.discard import hand_counts
from catan_bots.rules.trading import TradeNegotiation, TradeOffer


class GreedyTraderBot(GreedySettlerBot, TradingPlayer):
    """`GreedySettlerBot`'s build priorities, plus trading towards a build.

    Trading uses only this bot's own hand and public information:

    - **Propose** one card it holds beyond its nearest build's cost for one
      card that build is missing.
    - **Accept** an offer if it brings the bot's nearest build closer, unless
      the proposer is within two points of winning.
    - **Settle** with the accepting seat showing the fewest victory points,
      so trades do not feed the leader.

    Attributes:
        LEADER_MARGIN: Refuse offers from a seat this close to the VP target.
    """

    LEADER_MARGIN: ClassVar[int] = 2

    def propose_trade(
        self, game: Game, offers: Sequence[TradeOffer]
    ) -> TradeOffer | None:
        """Offer a spare card for a card the nearest build is missing.

        Args:
            game: The game; read-only.
            offers: Offers this bot may make now.

        Returns:
            A one-for-one offer from `offers`, or `None` if nothing helps.
        """
        state = game.state
        hand = hand_counts(state, self.color)
        target = nearest_target(state, self.color, hand)
        if target is None or target.missing_cards(hand) == 0:
            return None
        spare = target.surplus(hand)
        if not spare:
            return None
        give = max(spare, key=lambda r: (spare[r], -RESOURCES.index(r)))
        for want in (r for r in RESOURCES if target.cost.get(r, 0) > hand[r]):
            offer = TradeOffer.of({give: 1}, {want: 1})
            if offer in offers:
                return offer
        return None

    def respond_to_trade(self, game: Game, negotiation: TradeNegotiation) -> bool:
        """Accept if the trade brings a build closer and the proposer is not near winning.

        Args:
            game: The game; read-only.
            negotiation: The open offer.

        Returns:
            True to accept.
        """
        state = game.state
        if self._near_winning(game, negotiation.proposer):
            return False
        hand = hand_counts(state, self.color)
        after = Counter(hand)
        after.subtract(negotiation.offer.want_counts)
        after.update(negotiation.offer.give_counts)
        return self._distance(state, after) < self._distance(state, hand)

    def choose_trade_partner(
        self, game: Game, negotiation: TradeNegotiation
    ) -> Color | None:
        """Trade with the acceptor showing the fewest victory points.

        Args:
            game: The game; read-only.
            negotiation: The offer, with every answer in.

        Returns:
            The trailing acceptor, ties going to the earlier seat, or `None`.
        """
        acceptors = negotiation.acceptors
        if not acceptors:
            return None
        return min(acceptors, key=lambda c: self._visible_points(game.state, c))

    def _distance(self, state: State, hand: Mapping[str, int]) -> int:
        """Count the cards a hand lacks for this bot's nearest build.

        Args:
            state: Game state, used to judge which builds are reachable.
            hand: Count per resource in the hand.

        Returns:
            Missing cards for the nearest reachable build; a large number if
            nothing is reachable, so no trade looks helpful.
        """
        target = nearest_target(state, self.color, hand)
        return 99 if target is None else target.missing_cards(hand)

    def _near_winning(self, game: Game, color: Color) -> bool:
        """Judge whether a seat is close enough to winning to refuse it.

        Args:
            game: The game, read for the VP target.
            color: The seat to judge.

        Returns:
            True if its visible points are within `LEADER_MARGIN` of winning.
        """
        return (
            self._visible_points(game.state, color)
            >= game.vps_to_win - self.LEADER_MARGIN
        )

    @staticmethod
    def _visible_points(state: State, color: Color) -> int:
        """Read a seat's publicly visible victory points.

        Args:
            state: Game state to read.
            color: The seat.

        Returns:
            Visible victory points, excluding hidden VP cards.
        """
        return state.player_state[f"{player_key(state, color)}_VICTORY_POINTS"]
