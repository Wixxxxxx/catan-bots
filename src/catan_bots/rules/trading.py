"""Player-to-player trading: the half of Catan catanatron does not model.

Catanatron implements only maritime trade with the bank. `TradeProtocol` adds
domestic trading on top of the engine, following the official rules:

- Only the player whose turn it is may trade, and only after rolling — not
  during the opening, a discard, a robber move or free road placement.
- Players other than the active player never trade with each other.
- Both sides give at least one card, and no resource appears on both sides:
  no gifts, and no trading a resource for itself.

One offer at a time works its way round the table:

1. The active player proposes an offer (`propose`).
2. Every other seat, in play order, accepts or rejects it (`respond`). A seat
   that cannot pay is declined automatically — to the table that looks
   exactly like a refusal, so it leaks nothing.
3. If anyone accepted, the active player picks one of them (`confirm`) or
   walks away (`cancel`). If nobody accepted, the offer closes by itself.

Agreed limits, all adjustable through `TradingRules`: offers of up to two
cards per side, at most three offers per turn. Counter-offers are a planned
second phase; a counter-offer slots in as an offer back to the active player.

Cards move between hands directly; the bank is untouched, so the total of
each resource in play is conserved. Trades are not engine actions and do not
appear in `state.actions`; the protocol keeps its own `history`.
"""

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from itertools import combinations_with_replacement

from catanatron import Color
from catanatron.models.actions import generate_playable_actions
from catanatron.models.decks import freqdeck_from_listdeck
from catanatron.models.enums import RESOURCES, ActionPrompt
from catanatron.state import State
from catanatron.state_functions import (
    player_deck_subtract,
    player_freqdeck_add,
    player_key,
)

from catan_bots.rules.discard import hand_counts, listdeck_from_counts


@dataclass(frozen=True)
class TradeOffer:
    """Cards the active player gives in exchange for cards it wants.

    Both sides are canonical listdecks (resource names repeated by count, in
    `RESOURCES` order), so equal offers compare equal. Build offers with `of`.

    Attributes:
        give: Cards the proposer hands over.
        want: Cards the proposer receives.
    """

    give: tuple[str, ...]
    want: tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject offers the official rules forbid.

        Raises:
            ValueError: If a side is empty, not canonical, names an unknown
                resource, or a resource appears on both sides.
        """
        for side in (self.give, self.want):
            if not side:
                raise ValueError("Both sides of a trade need at least one card.")
            if set(side) - set(RESOURCES):
                raise ValueError(f"Unknown resources in {side}.")
            if side != listdeck_from_counts(Counter(side)):
                raise ValueError(f"{side} is not in canonical order; use of().")
        if set(self.give) & set(self.want):
            raise ValueError("A resource cannot be traded for itself.")

    @classmethod
    def of(cls, give: Mapping[str, int], want: Mapping[str, int]) -> "TradeOffer":
        """Build an offer from per-resource counts.

        Args:
            give: Cards the proposer hands over, per resource.
            want: Cards the proposer receives, per resource.

        Returns:
            The canonical offer.
        """
        return cls(listdeck_from_counts(give), listdeck_from_counts(want))

    @property
    def give_counts(self) -> dict[str, int]:
        """Cards the proposer hands over, per resource."""
        return dict(Counter(self.give))

    @property
    def want_counts(self) -> dict[str, int]:
        """Cards the proposer receives, per resource."""
        return dict(Counter(self.want))


def enumerate_offers(max_cards_per_side: int) -> list[TradeOffer]:
    """List every legal offer up to a size limit, in a deterministic order.

    Args:
        max_cards_per_side: Most cards on either side of an offer.

    Returns:
        All offers with 1 to `max_cards_per_side` cards per side and no
        resource on both sides: 20 at one card, 230 at two, 1,170 at three.
    """
    sides = [
        side
        for size in range(1, max_cards_per_side + 1)
        for side in combinations_with_replacement(RESOURCES, size)
    ]
    return [
        TradeOffer(give, want)
        for give in sides
        for want in sides
        if not set(give) & set(want)
    ]


@dataclass(frozen=True)
class TradingRules:
    """Limits on domestic trading.

    Attributes:
        max_cards_per_side: Most cards on either side of an offer.
        max_offers_per_turn: Most offers the active player may make in a turn,
            whether or not they complete. Stops endless offer loops.
    """

    max_cards_per_side: int = 2
    max_offers_per_turn: int = 3


DEFAULT_TRADING_RULES = TradingRules()
"""The agreed defaults: offers of up to two cards per side, three per turn."""


@dataclass(frozen=True)
class TradeRecord:
    """A closed offer, completed or not.

    Attributes:
        turn: Turn number in which the offer was made.
        proposer: Colour of the active player who made it.
        offer: The offer.
        responses: Each other seat's answer: True for accept.
        partner: Colour the trade was made with, or `None` if it closed
            without a trade.
    """

    turn: int
    proposer: Color
    offer: TradeOffer
    responses: Mapping[Color, bool]
    partner: Color | None


@dataclass
class TradeNegotiation:
    """One offer working its way round the table.

    Attributes:
        turn: Turn number in which the offer was made.
        proposer: Colour of the active player.
        offer: The offer.
        responders: Every other seat, in play order after the proposer.
        responses: Answers so far: True for accept.
    """

    turn: int
    proposer: Color
    offer: TradeOffer
    responders: tuple[Color, ...]
    responses: dict[Color, bool] = field(default_factory=dict)

    @property
    def awaiting(self) -> Color | None:
        """The next seat that must answer, or `None` once all have."""
        return next((c for c in self.responders if c not in self.responses), None)

    @property
    def acceptors(self) -> tuple[Color, ...]:
        """Seats that accepted, in play order."""
        return tuple(c for c in self.responders if self.responses.get(c))


class TradeProtocol:
    """Run domestic trades on top of one catanatron game.

    Holds the open negotiation and the history of closed offers, so create a
    fresh protocol (or call `reset`) for every game.

    Attributes:
        rules: Limits on offers.
        catalog: Every offer the limits allow, in a fixed order.
        negotiation: The offer currently going round the table, if any.
        history: Every closed offer, oldest first.
    """

    def __init__(self, rules: TradingRules | None = None) -> None:
        """Create a protocol with no trades yet.

        Args:
            rules: Limits on offers. Defaults to the agreed `TradingRules()`.
        """
        self.rules = rules or TradingRules()
        self.catalog: tuple[TradeOffer, ...] = tuple(
            enumerate_offers(self.rules.max_cards_per_side)
        )
        self.negotiation: TradeNegotiation | None = None
        self.history: list[TradeRecord] = []

    def reset(self) -> None:
        """Forget the open negotiation and the history, for a new game."""
        self.negotiation = None
        self.history = []

    def offers_made_this_turn(self, state: State) -> int:
        """Count the offers made so far in the current turn.

        Args:
            state: Game state, read for the turn number.

        Returns:
            Closed and open offers made this turn.
        """
        closed = sum(1 for record in self.history if record.turn == state.num_turns)
        return closed + (self.negotiation is not None)

    def can_propose(self, state: State, color: Color) -> bool:
        """Judge whether a player may make an offer right now.

        Args:
            state: Game state to read.
            color: Colour of the would-be proposer.

        Returns:
            True only for the active player, after rolling, in the main phase,
            with no offer open and the per-turn limit not reached.
        """
        if self.negotiation is not None or state.is_initial_build_phase:
            return False
        if state.current_prompt is not ActionPrompt.PLAY_TURN or state.is_road_building:
            return False
        if state.current_color() is not color:
            return False
        if not state.player_state[f"{player_key(state, color)}_HAS_ROLLED"]:
            return False
        return self.offers_made_this_turn(state) < self.rules.max_offers_per_turn

    def legal_offers(self, state: State, color: Color) -> list[TradeOffer]:
        """List the offers a player may make right now.

        Args:
            state: Game state to read.
            color: Colour of the would-be proposer.

        Returns:
            Catalog offers whose cards the player holds; empty if it may not
            propose at all.
        """
        if not self.can_propose(state, color):
            return []
        hand = hand_counts(state, color)
        return [offer for offer in self.catalog if self._holds(hand, offer.give_counts)]

    def can_pay(self, state: State, color: Color, offer: TradeOffer) -> bool:
        """Judge whether a responder holds the cards an offer asks for.

        Args:
            state: Game state to read.
            color: Colour of the responder.
            offer: The offer.

        Returns:
            True if the responder could hand over the offer's `want` cards.
        """
        return self._holds(hand_counts(state, color), offer.want_counts)

    def decider(self) -> Color | None:
        """Name the seat that must act in the open negotiation.

        Returns:
            The next responder, then the proposer once there are acceptors to
            choose between; `None` with no negotiation open.
        """
        if self.negotiation is None:
            return None
        return self.negotiation.awaiting or self.negotiation.proposer

    def propose(self, state: State, color: Color, offer: TradeOffer) -> None:
        """Open an offer, declining for seats that cannot pay.

        If no seat can pay, the offer closes at once without a trade and is
        added to `history`.

        Args:
            state: Game state.
            color: Colour of the proposer.
            offer: The offer.

        Raises:
            ValueError: If the player may not make this offer now.
        """
        if offer not in self.legal_offers(state, color):
            raise ValueError(f"{color} may not offer {offer} now.")
        order = state.colors
        start = order.index(color)
        responders = tuple(
            order[(start + i) % len(order)] for i in range(1, len(order))
        )
        self.negotiation = TradeNegotiation(state.num_turns, color, offer, responders)
        self._advance(state)

    def respond(self, state: State, color: Color, accept: bool) -> None:
        """Record a responder's answer, declining for later seats that cannot pay.

        If every seat has now answered and none accepted, the offer closes
        without a trade and is added to `history`.

        Args:
            state: Game state.
            color: Colour of the responder.
            accept: True to accept the offer.

        Raises:
            ValueError: If it is not this seat's turn to answer, or it accepts
                without holding the cards asked for.
        """
        negotiation = self.negotiation
        if negotiation is None or negotiation.awaiting is not color:
            raise ValueError(f"{color} is not due to answer an offer.")
        if accept and not self.can_pay(state, color, negotiation.offer):
            raise ValueError(f"{color} cannot pay for {negotiation.offer}.")
        negotiation.responses[color] = accept
        self._advance(state)

    def confirm(self, state: State, partner: Color) -> TradeRecord:
        """Complete the open offer with one of the seats that accepted.

        Args:
            state: Game state; both hands are updated in place.
            partner: Colour of the accepting seat to trade with.

        Returns:
            The record of the completed trade.

        Raises:
            ValueError: If no offer awaits confirmation, or `partner` did not
                accept.
        """
        negotiation = self._awaiting_confirmation()
        if partner not in negotiation.acceptors:
            raise ValueError(f"{partner} did not accept this offer.")
        self._exchange(state, negotiation.proposer, partner, negotiation.offer)
        return self._close(partner)

    def cancel(self) -> TradeRecord:
        """Walk away from the open offer after seeing who accepted.

        Returns:
            The record of the offer, closed without a trade.

        Raises:
            ValueError: If no offer awaits confirmation.
        """
        self._awaiting_confirmation()
        return self._close(None)

    def _awaiting_confirmation(self) -> TradeNegotiation:
        """Return the negotiation if every responder has answered.

        Returns:
            The open negotiation.

        Raises:
            ValueError: If no offer is open or answers are still due.
        """
        negotiation = self.negotiation
        if negotiation is None or negotiation.awaiting is not None:
            raise ValueError("No offer is waiting for the proposer's decision.")
        return negotiation

    def _advance(self, state: State) -> None:
        """Decline for seats that cannot pay, and close an offer nobody took.

        Args:
            state: Game state, read for responders' hands.
        """
        negotiation = self.negotiation
        while negotiation.awaiting is not None and not self.can_pay(
            state, negotiation.awaiting, negotiation.offer
        ):
            negotiation.responses[negotiation.awaiting] = False
        if negotiation.awaiting is None and not negotiation.acceptors:
            self._close(None)

    def _close(self, partner: Color | None) -> TradeRecord:
        """Move the open negotiation into the history.

        Args:
            partner: Colour traded with, or `None` for no trade.

        Returns:
            The new history record.
        """
        negotiation = self.negotiation
        record = TradeRecord(
            turn=negotiation.turn,
            proposer=negotiation.proposer,
            offer=negotiation.offer,
            responses=dict(negotiation.responses),
            partner=partner,
        )
        self.history.append(record)
        self.negotiation = None
        return record

    @staticmethod
    def _exchange(
        state: State, proposer: Color, partner: Color, offer: TradeOffer
    ) -> None:
        """Swap an offer's cards between two hands and refresh legal actions.

        Args:
            state: Game state; edited in place.
            proposer: Colour giving `offer.give` and receiving `offer.want`.
            partner: Colour giving `offer.want` and receiving `offer.give`.
            offer: The agreed offer.
        """
        give = freqdeck_from_listdeck(offer.give)
        want = freqdeck_from_listdeck(offer.want)
        player_deck_subtract(state, proposer, give)
        player_freqdeck_add(state, partner, give)
        player_deck_subtract(state, partner, want)
        player_freqdeck_add(state, proposer, want)
        state.playable_actions = generate_playable_actions(state)

    @staticmethod
    def _holds(hand: Mapping[str, int], cards: Mapping[str, int]) -> bool:
        """Check that a hand covers some cards.

        Args:
            hand: Count per resource in the hand.
            cards: Count per resource required.

        Returns:
            True if the hand holds at least that many of every resource.
        """
        return all(hand.get(r, 0) >= n for r, n in cards.items())


def describe(offers: Iterable[TradeOffer]) -> list[str]:
    """Render offers as short human-readable strings, for logs and notebooks.

    Args:
        offers: Offers to render.

    Returns:
        One `"2 WOOD -> 1 ORE"`-style line per offer.
    """
    return [
        f"{_describe_side(o.give_counts)} -> {_describe_side(o.want_counts)}"
        for o in offers
    ]


def _describe_side(counts: Mapping[str, int]) -> str:
    """Render one side of an offer.

    Args:
        counts: Cards per resource on that side.

    Returns:
        A string such as `"2 WOOD + 1 BRICK"`.
    """
    return " + ".join(f"{n} {r}" for r, n in counts.items())
