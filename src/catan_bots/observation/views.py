"""What one seat is allowed to know: the redacted views of a Catan position.

Catanatron's `State` is omniscient. Every seat's exact hand, every unplayed
development card, the remaining development deck *in draw order*, and hidden
victory-point cards are all plain fields. Training policies on it produces
agents that peek at opponents' hands and the deck — strong in self-play,
meaningless against real hidden-information play.

These dataclasses are the contract for what a seat may see:

Public — every seat sees it for every seat:
    The board (tiles, numbers, ports, robber, buildings, roads), each seat's
    visible victory points, resource-card and development-card *counts*,
    development cards played by type, pieces left in supply, longest-road
    length, award holders and port access, the bank, the deck *size*, and the
    turn and phase.

Private — only the perspective seat sees it, and only for itself:
    Its resource hand by type, its unplayed development cards by type, which
    of those it may play this turn, and its actual victory points (including
    VP cards).

Hidden from every seat:
    Opponents' hands by type, opponents' unplayed development cards by type,
    opponents' hidden VP cards, and the development deck's order.

Ownership on the board is recorded as a *seat offset* relative to the
perspective (0 = self, 1 = next seat to play, ...), not as a colour. That makes
the view symmetric: a single shared policy sees its own pieces the same way
from every chair.

The views are snapshots and carry no action history. Much of the hidden
information is recoverable from public history by card counting, which is
fair game; if a history is ever added, the resource stolen by `MOVE_ROBBER`
must be redacted for every seat other than the thief and the victim.
"""

from dataclasses import dataclass

from catanatron import Color

from catan_bots.analytics.board_stats import TileSummary

Edge = tuple[int, int]


@dataclass(frozen=True)
class PublicSeatView:
    """Everything about one seat that is visible to the whole table.

    Attributes:
        color: The seat's colour.
        seat_offset: Seats after the perspective in play order; 0 is the
            perspective seat itself.
        visible_victory_points: Victory points shown on the table, excluding
            hidden VP development cards.
        resource_card_count: Number of resource cards held, without types.
        development_card_count: Number of unplayed development cards held,
            without types.
        played_development_cards: Cards played so far, per type; played
            cards are face up.
        roads_available: Roads left in this seat's supply.
        settlements_available: Settlements left in this seat's supply.
        cities_available: Cities left in this seat's supply.
        longest_road_length: Length of this seat's longest road.
        has_longest_road: Whether this seat holds the Longest Road card.
        has_largest_army: Whether this seat holds the Largest Army card.
        port_resources: Ports this seat can trade through: resource names for
            2:1 ports, `None` for the generic 3:1 port.
    """

    color: Color
    seat_offset: int
    visible_victory_points: int
    resource_card_count: int
    development_card_count: int
    played_development_cards: dict[str, int]
    roads_available: int
    settlements_available: int
    cities_available: int
    longest_road_length: int
    has_longest_road: bool
    has_largest_army: bool
    port_resources: frozenset[str | None]


@dataclass(frozen=True)
class PrivateHandView:
    """The perspective seat's own private knowledge.

    Attributes:
        resources: Resource cards held, per type.
        development_cards: Unplayed development cards held, per type,
            including victory-point cards.
        playable_development_cards: Copies of each playable card type that
            are eligible to be played this turn — none once a card has been
            played this turn, and never a copy bought this turn. Whether it
            is this seat's decision right now is in the phase and action mask.
        actual_victory_points: True victory points, including hidden VP cards.
        has_played_development_card_this_turn: Whether a development card was
            already played this turn.
    """

    resources: dict[str, int]
    development_cards: dict[str, int]
    playable_development_cards: dict[str, int]
    actual_victory_points: int
    has_played_development_card_this_turn: bool


@dataclass(frozen=True)
class BoardView:
    """The physical board, with ownership relative to the perspective seat.

    Attributes:
        tiles: Every land tile, ordered by cube coordinate.
        robber_coordinate: Cube coordinate of the tile the robber is on.
        buildings: Node id to `(seat_offset, building_type)`.
        roads: Canonical edge `(low_node, high_node)` to seat offset.
        ports: Node id to the port it touches: a resource name for a 2:1 port,
            `None` for a 3:1 port. Nodes without a port are absent.
    """

    tiles: tuple[TileSummary, ...]
    robber_coordinate: tuple[int, int, int]
    buildings: dict[int, tuple[int, str]]
    roads: dict[Edge, int]
    ports: dict[int, str | None]


@dataclass(frozen=True)
class PhaseView:
    """Where the game is in its turn structure.

    Attributes:
        prompt: Name of the current `ActionPrompt`, e.g. `"PLAY_TURN"`.
        turn_number: Completed turns so far.
        deciding_seat_offset: Seat offset of the player deciding now. Differs
            from `turn_seat_offset` while players discard on a seven.
        turn_seat_offset: Seat offset of the player whose turn it is.
        has_rolled: Whether the player whose turn it is has rolled.
        is_initial_build_phase: Whether opening placements are under way.
        is_road_building: Whether free roads from Road Building are pending.
        free_roads_available: Free roads left to place.
    """

    prompt: str
    turn_number: int
    deciding_seat_offset: int
    turn_seat_offset: int
    has_rolled: bool
    is_initial_build_phase: bool
    is_road_building: bool
    free_roads_available: int


@dataclass(frozen=True)
class Observation:
    """One seat's complete, redacted view of a position.

    Attributes:
        perspective: Colour of the seat this view belongs to.
        num_players: Number of seats in the game.
        hand: The perspective seat's private knowledge.
        seats: Public view of every seat, indexed by seat offset, so
            `seats[0]` is the perspective seat.
        board: The board, with ownership as seat offsets.
        bank_resources: Resource cards left in the bank, per type.
        development_deck_size: Development cards left to draw.
        phase: Turn and phase information.
    """

    perspective: Color
    num_players: int
    hand: PrivateHandView
    seats: tuple[PublicSeatView, ...]
    board: BoardView
    bank_resources: dict[str, int]
    development_deck_size: int
    phase: PhaseView

    @property
    def opponents(self) -> tuple[PublicSeatView, ...]:
        """Public views of every other seat, in play order after this one."""
        return self.seats[1:]
