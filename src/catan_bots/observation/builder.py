"""Build a seat's redacted `Observation` from catanatron's omniscient `State`."""

from typing import ClassVar

from catanatron import Color
from catanatron.models.enums import DEVELOPMENT_CARDS, RESOURCES
from catanatron.state import State
from catanatron.state_functions import (
    player_key,
    player_num_dev_cards,
    player_num_resource_cards,
)

from catan_bots.analytics.board_stats import BoardInspector
from catan_bots.observation.views import (
    BoardView,
    Observation,
    PhaseView,
    PrivateHandView,
    PublicSeatView,
    TradeView,
)
from catan_bots.rules.dev_cards import DevCardTimingRule
from catan_bots.rules.discard import hand_counts
from catan_bots.rules.trading import TradeProtocol


class ObservationBuilder:
    """Turn a full game state into what one seat is allowed to know.

    This is the only component that reads `State` on a policy's behalf.
    Anything downstream — a vector encoder, a network, a bot — that consumes
    only `Observation` cannot leak hidden information by construction.

    Attributes:
        PLAYABLE_CARDS: Development cards that are played, as opposed to VP
            cards, which score automatically.
        dev_card_timing: Rule used to tell which held cards are eligible to
            be played this turn.
    """

    PLAYABLE_CARDS: ClassVar[tuple[str, ...]] = (
        "KNIGHT",
        "YEAR_OF_PLENTY",
        "MONOPOLY",
        "ROAD_BUILDING",
    )

    def __init__(self, dev_card_timing: DevCardTimingRule | None = None) -> None:
        """Create a builder.

        Args:
            dev_card_timing: Rule deciding which held cards are eligible to be
                played. Defaults to the official same-turn restriction.
        """
        self.dev_card_timing = dev_card_timing or DevCardTimingRule()

    def build(
        self,
        state: State,
        perspective: Color,
        trading: TradeProtocol | None = None,
    ) -> Observation:
        """Build one seat's redacted view of the current position.

        Args:
            state: Full game state. It is only read.
            perspective: Colour of the seat whose view is built.
            trading: The game's trading protocol, or `None` when trading is
                off, in which case the observation carries no trade view.

        Returns:
            The seat's `Observation`.

        Raises:
            ValueError: If `perspective` is not seated in this game.
        """
        if perspective not in state.colors:
            raise ValueError(f"{perspective} is not seated in this game.")
        seats_in_play_order = sorted(
            state.colors, key=lambda c: self.seat_offset(state, perspective, c)
        )
        return Observation(
            perspective=perspective,
            num_players=len(state.colors),
            hand=self._private_hand(state, perspective),
            seats=tuple(
                self._public_seat(state, perspective, color)
                for color in seats_in_play_order
            ),
            board=self._board(state, perspective),
            bank_resources=dict(zip(RESOURCES, state.resource_freqdeck)),
            development_deck_size=len(state.development_listdeck),
            phase=self._phase(state, perspective),
            trade=self._trade(state, perspective, trading) if trading else None,
        )

    @staticmethod
    def seat_offset(state: State, perspective: Color, color: Color) -> int:
        """Count how many seats after the perspective a colour plays.

        Args:
            state: Game state holding the seating order.
            perspective: The reference seat.
            color: The seat to locate.

        Returns:
            0 for the perspective itself, 1 for the next seat to play, and so on.
        """
        order = state.colors
        return (order.index(color) - order.index(perspective)) % len(order)

    def _public_seat(
        self, state: State, perspective: Color, color: Color
    ) -> PublicSeatView:
        """Read the table-visible facts about one seat.

        Args:
            state: Game state to read.
            perspective: The seat the observation is built for.
            color: The seat being described.

        Returns:
            That seat's `PublicSeatView`.
        """
        key = player_key(state, color)
        fields = state.player_state
        return PublicSeatView(
            color=color,
            seat_offset=self.seat_offset(state, perspective, color),
            visible_victory_points=fields[f"{key}_VICTORY_POINTS"],
            resource_card_count=player_num_resource_cards(state, color),
            development_card_count=player_num_dev_cards(state, color),
            played_development_cards={
                card: fields[f"{key}_PLAYED_{card}"] for card in self.PLAYABLE_CARDS
            },
            roads_available=fields[f"{key}_ROADS_AVAILABLE"],
            settlements_available=fields[f"{key}_SETTLEMENTS_AVAILABLE"],
            cities_available=fields[f"{key}_CITIES_AVAILABLE"],
            longest_road_length=fields[f"{key}_LONGEST_ROAD_LENGTH"],
            has_longest_road=bool(fields[f"{key}_HAS_ROAD"]),
            has_largest_army=bool(fields[f"{key}_HAS_ARMY"]),
            port_resources=frozenset(state.board.get_player_port_resources(color)),
        )

    def _private_hand(self, state: State, perspective: Color) -> PrivateHandView:
        """Read the perspective seat's private knowledge of its own hand.

        Args:
            state: Game state to read.
            perspective: The seat the observation is built for.

        Returns:
            That seat's `PrivateHandView`.
        """
        key = player_key(state, perspective)
        fields = state.player_state
        return PrivateHandView(
            resources=hand_counts(state, perspective),
            development_cards={
                card: fields[f"{key}_{card}_IN_HAND"] for card in DEVELOPMENT_CARDS
            },
            playable_development_cards=self._eligible_cards(state, perspective),
            actual_victory_points=fields[f"{key}_ACTUAL_VICTORY_POINTS"],
            has_played_development_card_this_turn=bool(
                fields[f"{key}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"]
            ),
        )

    def _eligible_cards(self, state: State, perspective: Color) -> dict[str, int]:
        """Count the held copies of each card eligible to be played this turn.

        Args:
            state: Game state to read.
            perspective: The seat whose cards are counted.

        Returns:
            Eligible copies per playable card type: zero for all once a card
            has been played this turn, and excluding copies bought this turn.
        """
        key = player_key(state, perspective)
        if state.player_state[f"{key}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"]:
            return dict.fromkeys(self.PLAYABLE_CARDS, 0)
        bought = self.dev_card_timing.cards_bought_this_turn(state, perspective)
        return {
            card: self.dev_card_timing.playable_copies(state, perspective, card, bought)
            for card in self.PLAYABLE_CARDS
        }

    def _board(self, state: State, perspective: Color) -> BoardView:
        """Read the board, recording ownership as seat offsets.

        Args:
            state: Game state to read.
            perspective: The seat the observation is built for.

        Returns:
            The `BoardView`.
        """
        board = state.board
        return BoardView(
            tiles=tuple(BoardInspector(board.map).tile_summaries()),
            robber_coordinate=board.robber_coordinate,
            buildings={
                node: (self.seat_offset(state, perspective, color), building)
                for node, (color, building) in board.buildings.items()
            },
            roads={
                (min(edge), max(edge)): self.seat_offset(state, perspective, color)
                for edge, color in board.roads.items()
            },
            ports={
                node: resource
                for resource, nodes in board.map.port_nodes.items()
                for node in nodes
            },
        )

    def _trade(
        self, state: State, perspective: Color, trading: TradeProtocol
    ) -> TradeView:
        """Read the trade offer on the table, with seats as offsets.

        Args:
            state: Game state, read for seating and the turn number.
            perspective: The seat the observation is built for.
            trading: The game's trading protocol.

        Returns:
            The `TradeView`; closed and empty when no offer is open.
        """
        made = trading.offers_made_this_turn(state)
        limit = trading.rules.max_offers_per_turn
        negotiation = trading.negotiation
        if negotiation is None:
            return TradeView(False, None, {}, {}, {}, None, made, limit)

        awaiting = negotiation.awaiting
        return TradeView(
            is_open=True,
            proposer_seat_offset=self.seat_offset(
                state, perspective, negotiation.proposer
            ),
            give=negotiation.offer.give_counts,
            want=negotiation.offer.want_counts,
            responses={
                self.seat_offset(state, perspective, color): accepted
                for color, accepted in negotiation.responses.items()
            },
            awaiting_seat_offset=(
                None
                if awaiting is None
                else self.seat_offset(state, perspective, awaiting)
            ),
            offers_made_this_turn=made,
            max_offers_per_turn=limit,
        )

    def _phase(self, state: State, perspective: Color) -> PhaseView:
        """Read the turn and phase, with seats as offsets.

        Args:
            state: Game state to read.
            perspective: The seat the observation is built for.

        Returns:
            The `PhaseView`.
        """
        turn_color = state.colors[state.current_turn_index]
        turn_key = player_key(state, turn_color)
        return PhaseView(
            prompt=state.current_prompt.name,
            turn_number=state.num_turns,
            deciding_seat_offset=self.seat_offset(
                state, perspective, state.current_color()
            ),
            turn_seat_offset=self.seat_offset(state, perspective, turn_color),
            has_rolled=bool(state.player_state[f"{turn_key}_HAS_ROLLED"]),
            is_initial_build_phase=bool(state.is_initial_build_phase),
            is_road_building=bool(state.is_road_building),
            free_roads_available=state.free_roads_available,
        )
