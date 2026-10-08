"""The interface a bot implements to take part in domestic trading."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from catanatron import Color, Game

from catan_bots.rules.trading import TradeNegotiation, TradeOffer


class TradingPlayer(ABC):
    """A bot that proposes, answers and settles domestic trades.

    Mix into a catanatron `Player`. `GameRunner` calls these hooks around the
    bot's normal decisions; bots that do not implement this interface never
    propose, and decline every offer made to them.
    """

    @abstractmethod
    def propose_trade(
        self, game: Game, offers: Sequence[TradeOffer]
    ) -> TradeOffer | None:
        """Choose an offer to make, if any.

        Args:
            game: The game; read-only.
            offers: Offers this bot may make now, excluding any it already
                made this turn.

        Returns:
            One of `offers`, or `None` to stop trading for now.
        """

    @abstractmethod
    def respond_to_trade(self, game: Game, negotiation: TradeNegotiation) -> bool:
        """Decide whether to accept the offer on the table.

        Only called when this bot can pay for the offer.

        Args:
            game: The game; read-only.
            negotiation: The open offer, including earlier answers.

        Returns:
            True to accept.
        """

    @abstractmethod
    def choose_trade_partner(
        self, game: Game, negotiation: TradeNegotiation
    ) -> Color | None:
        """Pick which accepting seat to trade with, after an offer this bot made.

        Args:
            game: The game; read-only.
            negotiation: The offer, with every answer in.

        Returns:
            One of `negotiation.acceptors`, or `None` to walk away.
        """
