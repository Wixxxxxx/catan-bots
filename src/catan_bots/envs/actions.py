"""A fixed, colour-free index for every action a Catan seat can take.

Catanatron describes legality as a variable-length list of `Action` objects
that name colours. A learning agent needs the opposite: one fixed
`Discrete(n)` space shared by every seat, plus a mask. `ActionTable` provides
that mapping. Keys are canonical and relative to the acting seat — a robbery
names its victim by seat offset, not colour — so one policy can sit in any
chair.

Discarding on a seven is exposed as five synthetic "discard one card of this
resource" actions, chosen repeatedly until half the hand is gone. That keeps
the branching factor at five while still letting the agent pick any of the
legal discard selections.

Domestic trading, which catanatron does not model, adds synthetic slots too:
one per offer in the trading catalog, accept, reject, cancel, and "trade with
the seat N places after the proposer" for each possible partner.
"""

from collections.abc import Hashable, Iterable, Sequence
from itertools import combinations_with_replacement

import numpy as np
from catanatron import Action, ActionType, Color
from catanatron.models.enums import RESOURCES
from catanatron.state import State

from catan_bots.envs.topology import BoardTopology
from catan_bots.observation import ObservationBuilder
from catan_bots.rules.trading import TradeOffer

DISCARD_CARD = "DISCARD_CARD"
PROPOSE_TRADE = "PROPOSE_TRADE"
ACCEPT_TRADE = "ACCEPT_TRADE"
REJECT_TRADE = "REJECT_TRADE"
CANCEL_TRADE = "CANCEL_TRADE"
CONFIRM_TRADE = "CONFIRM_TRADE"
MARITIME_RATES = (2, 3, 4)

ActionKey = tuple[Hashable, ...]

_SIMPLE_ACTIONS = (
    ActionType.ROLL,
    ActionType.END_TURN,
    ActionType.BUY_DEVELOPMENT_CARD,
    ActionType.PLAY_KNIGHT_CARD,
    ActionType.PLAY_ROAD_BUILDING,
)


class ActionTable:
    """Map between engine actions and indices of a fixed discrete space.

    Layout, in index order: five simple actions (roll, end turn, buy card,
    play knight, play road building), five discard picks, a settlement and a
    city per node, a road per edge, a robber move per tile and victim seat
    offset (0 meaning no victim), every year-of-plenty draw, a monopoly per
    resource, a maritime trade per give resource, rate and get resource, a
    proposal per trade offer, accept, reject, cancel, and a confirmation per
    partner seat offset.

    Attributes:
        topology: Board geometry the table was laid out for.
        max_players: Largest table size the robber and partner slots allow.
        trade_offers: Offers given a proposal slot, in slot order.
        keys: Canonical key at each index.
    """

    def __init__(
        self,
        topology: BoardTopology,
        max_players: int = 4,
        trade_offers: Sequence[TradeOffer] = (),
    ) -> None:
        """Lay out the table for one board geometry.

        Args:
            topology: Board geometry whose nodes, edges and tiles are indexed.
            max_players: Most seats a game can have; sets the robber-victim
                and trade-partner slots so one table serves every size.
            trade_offers: Offers to give proposal slots. Kept even when
                trading is switched off, so spaces match across settings.
        """
        self.topology = topology
        self.max_players = max_players
        self.trade_offers = tuple(trade_offers)
        self.keys: tuple[ActionKey, ...] = tuple(self._enumerate_keys())
        self._index: dict[ActionKey, int] = {k: i for i, k in enumerate(self.keys)}

    @property
    def size(self) -> int:
        """Number of actions in the space."""
        return len(self.keys)

    def key_of(self, state: State, action: Action) -> ActionKey:
        """Compute the canonical, colour-free key of an engine action.

        Args:
            state: Game state, used to express a robbery victim as a seat
                offset from the actor.
            action: A playable engine action.

        Returns:
            The action's key.

        Raises:
            ValueError: If the action type has no slot in the table.
        """
        kind, value = action.action_type, action.value
        if kind in _SIMPLE_ACTIONS:
            return (kind,)
        if kind in (ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY):
            return (kind, value)
        if kind is ActionType.BUILD_ROAD:
            return (kind, (min(value), max(value)))
        if kind is ActionType.MOVE_ROBBER:
            return (kind, value[0], self._victim_offset(state, action.color, value[1]))
        if kind is ActionType.PLAY_YEAR_OF_PLENTY:
            return (kind, tuple(sorted(value, key=RESOURCES.index)))
        if kind is ActionType.PLAY_MONOPOLY:
            return (kind, value)
        if kind is ActionType.MARITIME_TRADE:
            offered = [card for card in value[:4] if card is not None]
            return (kind, offered[0], len(offered), value[4])
        raise ValueError(f"No slot for action type {kind}.")

    def index_of(self, state: State, action: Action) -> int:
        """Look up the index of an engine action.

        Args:
            state: Game state the action is legal in.
            action: A playable engine action.

        Returns:
            Its index in the discrete space.
        """
        return self._index[self.key_of(state, action)]

    def legal_by_index(
        self, state: State, actions: Iterable[Action]
    ) -> dict[int, Action]:
        """Index the legal engine actions of one decision.

        Args:
            state: Game state the actions are legal in.
            actions: The legal engine actions.

        Returns:
            The engine action behind each legal index.

        Raises:
            ValueError: If two different legal actions share a slot, which
                would make the mask ambiguous.
        """
        legal: dict[int, Action] = {}
        for action in actions:
            index = self.index_of(state, action)
            if index in legal and legal[index] != action:
                raise ValueError(f"{action} and {legal[index]} share slot {index}.")
            legal[index] = action
        return legal

    def index(self, key: ActionKey) -> int:
        """Look up the index of a canonical key.

        Args:
            key: A key present in the table.

        Returns:
            Its index.
        """
        return self._index[key]

    def propose_index(self, offer: TradeOffer) -> int:
        """Look up the index of proposing a trade offer.

        Args:
            offer: An offer from the trading catalog.

        Returns:
            The proposal's index.
        """
        return self._index[(PROPOSE_TRADE, offer.give, offer.want)]

    def confirm_index(self, partner_offset: int) -> int:
        """Look up the index of trading with an accepting seat.

        Args:
            partner_offset: The partner's seat offset from the proposer.

        Returns:
            The confirmation's index.
        """
        return self._index[(CONFIRM_TRADE, partner_offset)]

    def discard_index(self, resource: str) -> int:
        """Look up the index of discarding one card of a resource.

        Args:
            resource: Resource name.

        Returns:
            The discard pick's index.
        """
        return self._index[(DISCARD_CARD, resource)]

    def discard_resource(self, index: int) -> str | None:
        """Read which resource a discard-pick index discards.

        Args:
            index: An index in the space.

        Returns:
            The resource, or `None` if the index is not a discard pick.
        """
        key = self.keys[index]
        return key[1] if key[0] == DISCARD_CARD else None

    def mask(self, legal_indices: Iterable[int]) -> np.ndarray:
        """Build the action mask for a set of legal indices.

        Args:
            legal_indices: Indices that are legal right now.

        Returns:
            An `int8` array of the space's size, 1 where legal.
        """
        mask = np.zeros(self.size, dtype=np.int8)
        mask[list(legal_indices)] = 1
        return mask

    @staticmethod
    def _victim_offset(state: State, actor: Color, victim: Color | None) -> int:
        """Express a robbery victim as a seat offset from the robber.

        Args:
            state: Game state holding the seating order.
            actor: Colour moving the robber.
            victim: Colour robbed, or `None` for no robbery.

        Returns:
            0 for no victim, otherwise the victim's seat offset (1 or more).
        """
        if victim is None:
            return 0
        return ObservationBuilder.seat_offset(state, actor, victim)

    def _enumerate_keys(self) -> list[ActionKey]:
        """List every key in table order.

        Returns:
            Keys for all action slots, in index order.
        """
        topology = self.topology
        keys: list[ActionKey] = [(kind,) for kind in _SIMPLE_ACTIONS]
        keys += [(DISCARD_CARD, resource) for resource in RESOURCES]
        keys += [(ActionType.BUILD_SETTLEMENT, node) for node in topology.nodes]
        keys += [(ActionType.BUILD_CITY, node) for node in topology.nodes]
        keys += [(ActionType.BUILD_ROAD, edge) for edge in topology.edges]
        keys += [
            (ActionType.MOVE_ROBBER, tile, victim)
            for tile in topology.tiles
            for victim in range(self.max_players)
        ]
        keys += [
            (ActionType.PLAY_YEAR_OF_PLENTY, draw)
            for draw in self._year_of_plenty_draws()
        ]
        keys += [(ActionType.PLAY_MONOPOLY, resource) for resource in RESOURCES]
        keys += [
            (ActionType.MARITIME_TRADE, give, rate, get)
            for give in RESOURCES
            for rate in MARITIME_RATES
            for get in RESOURCES
            if get != give
        ]
        keys += [(PROPOSE_TRADE, o.give, o.want) for o in self.trade_offers]
        keys += [(ACCEPT_TRADE,), (REJECT_TRADE,), (CANCEL_TRADE,)]
        keys += [(CONFIRM_TRADE, offset) for offset in range(1, self.max_players)]
        return keys

    @staticmethod
    def _year_of_plenty_draws() -> Sequence[tuple[str, ...]]:
        """List every year-of-plenty draw catanatron can offer.

        Two cards normally; one when the bank is too short for two.

        Returns:
            Draws as resource tuples in `RESOURCES` order.
        """
        pairs = list(combinations_with_replacement(RESOURCES, 2))
        singles = [(resource,) for resource in RESOURCES]
        return pairs + singles
