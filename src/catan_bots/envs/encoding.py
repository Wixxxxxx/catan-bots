"""Flatten a redacted `Observation` into a fixed-length vector for a network.

The encoder's only input is an `Observation`, never the engine `State`, so it
cannot leak hidden information: that guarantee is enforced upstream by
`ObservationBuilder` and its leak-invariance tests.

Every feature is scaled into [0, 1]. Counts are divided by a cap and clipped,
so the rare value beyond its cap saturates rather than leaving the space.
Seats are padded to four, so one network serves 2-, 3- and 4-player games.
"""

from collections.abc import Sequence

import numpy as np
from catanatron.game import TURNS_LIMIT
from catanatron.models.enums import RESOURCES, ActionPrompt
from gymnasium import spaces

from catan_bots.envs.topology import BoardTopology
from catan_bots.observation import Observation, PublicSeatView
from catan_bots.rules.discard import SequentialDiscard

MAX_PLAYERS = 4
DEVELOPMENT_CARDS = (
    "KNIGHT",
    "YEAR_OF_PLENTY",
    "MONOPOLY",
    "ROAD_BUILDING",
    "VICTORY_POINT",
)
PLAYABLE_CARDS = DEVELOPMENT_CARDS[:4]
PORTS: tuple[str | None, ...] = (None, *RESOURCES)
TILE_RESOURCES: tuple[str, ...] = (*RESOURCES, "DESERT")
DICE_NUMBERS = tuple(n for n in range(2, 13) if n != 7)
PROMPTS = tuple(prompt.name for prompt in ActionPrompt)

HAND_FEATURES = 5 + 5 + 4 + 1 + 1
SEAT_FEATURES = 1 + 1 + 1 + 1 + 4 + 3 + 1 + 1 + 1 + len(PORTS)
TILE_FEATURES = len(TILE_RESOURCES) + len(DICE_NUMBERS) + 1
NODE_FEATURES = MAX_PLAYERS + 2 + len(PORTS)
EDGE_FEATURES = MAX_PLAYERS
BANK_FEATURES = 5 + 1
PHASE_FEATURES = len(PROMPTS) + MAX_PLAYERS + MAX_PLAYERS + 5
DISCARD_FEATURES = 1 + 5


def _scaled(value: float, cap: float) -> float:
    """Scale a count into [0, 1] by a cap, saturating at the cap.

    Args:
        value: Count to scale.
        cap: Count that maps to 1.

    Returns:
        `value / cap`, clipped to [0, 1].
    """
    return min(max(value / cap, 0.0), 1.0)


def _one_hot(options: Sequence[object], value: object) -> list[float]:
    """Encode a value as a one-hot vector over options.

    Args:
        options: The possible values, in slot order.
        value: The value to mark; marks nothing if absent from `options`.

    Returns:
        A list with 1.0 in the value's slot and 0.0 elsewhere.
    """
    return [1.0 if option == value else 0.0 for option in options]


class ObservationEncoder:
    """Turn observations into fixed-length `float32` vectors.

    Layout, in order: own hand (16), four seat blocks (20 each), tiles
    (17 each), nodes (12 each), edges (4 each), bank and deck (6), phase (18)
    and discard progress (6) — 1,385 features on the base map. Seats, node owners and edge owners are by seat
    offset, so slot 0 is always the observing seat.

    Attributes:
        topology: Board geometry fixing the tile, node and edge slot order.
        size: Length of every encoded vector.
    """

    def __init__(self, topology: BoardTopology) -> None:
        """Lay out the encoding for one board geometry.

        Args:
            topology: Board geometry whose tiles, nodes and edges are encoded.
        """
        self.topology = topology
        self.size = (
            HAND_FEATURES
            + MAX_PLAYERS * SEAT_FEATURES
            + len(topology.tiles) * TILE_FEATURES
            + len(topology.nodes) * NODE_FEATURES
            + len(topology.edges) * EDGE_FEATURES
            + BANK_FEATURES
            + PHASE_FEATURES
            + DISCARD_FEATURES
        )

    def space(self) -> spaces.Box:
        """Describe the encoded vectors as a Gymnasium space.

        Returns:
            A `Box` of shape `(size,)` in [0, 1].
        """
        return spaces.Box(0.0, 1.0, shape=(self.size,), dtype=np.float32)

    def encode(
        self, observation: Observation, discard: SequentialDiscard | None = None
    ) -> np.ndarray:
        """Encode one seat's observation.

        Args:
            observation: The seat's redacted view.
            discard: The seat's discard in progress, if it is mid-way through
                choosing cards to discard.

        Returns:
            A `float32` vector of length `size`.

        Raises:
            ValueError: If the layout produced the wrong number of features.
        """
        features = (
            self._hand(observation)
            + self._seats(observation)
            + self._tiles(observation)
            + self._nodes(observation)
            + self._edges(observation)
            + self._bank(observation)
            + self._phase(observation)
            + self._discard(discard)
        )
        if len(features) != self.size:
            raise ValueError(f"Encoded {len(features)} features, expected {self.size}.")
        return np.asarray(features, dtype=np.float32)

    @staticmethod
    def _hand(observation: Observation) -> list[float]:
        """Encode the observing seat's private hand.

        Args:
            observation: The seat's view.

        Returns:
            16 features: resources, development cards, eligible plays,
            actual victory points and the played-this-turn flag.
        """
        hand = observation.hand
        return (
            [_scaled(hand.resources[r], 19) for r in RESOURCES]
            + [_scaled(hand.development_cards[c], 14) for c in DEVELOPMENT_CARDS]
            + [_scaled(hand.playable_development_cards[c], 14) for c in PLAYABLE_CARDS]
            + [_scaled(hand.actual_victory_points, 12)]
            + [float(hand.has_played_development_card_this_turn)]
        )

    def _seats(self, observation: Observation) -> list[float]:
        """Encode every seat's public view, padded to four seats.

        Args:
            observation: The seat's view.

        Returns:
            `MAX_PLAYERS * 20` features; absent seats are all zero.
        """
        features: list[float] = []
        for offset in range(MAX_PLAYERS):
            if offset < len(observation.seats):
                features += self._seat(observation.seats[offset])
            else:
                features += [0.0] * SEAT_FEATURES
        return features

    @staticmethod
    def _seat(seat: PublicSeatView) -> list[float]:
        """Encode one seat's public view.

        Args:
            seat: The seat's public view.

        Returns:
            20 features, beginning with a presence flag.
        """
        return (
            [1.0]
            + [_scaled(seat.visible_victory_points, 12)]
            + [_scaled(seat.resource_card_count, 20)]
            + [_scaled(seat.development_card_count, 25)]
            + [_scaled(seat.played_development_cards[c], 14) for c in PLAYABLE_CARDS]
            + [_scaled(seat.roads_available, 15)]
            + [_scaled(seat.settlements_available, 5)]
            + [_scaled(seat.cities_available, 4)]
            + [_scaled(seat.longest_road_length, 15)]
            + [float(seat.has_longest_road), float(seat.has_largest_army)]
            + [1.0 if port in seat.port_resources else 0.0 for port in PORTS]
        )

    def _tiles(self, observation: Observation) -> list[float]:
        """Encode every land tile in topology order.

        Args:
            observation: The seat's view.

        Returns:
            17 features per tile: resource, dice number and robber.
        """
        by_coordinate = {tile.coordinate: tile for tile in observation.board.tiles}
        features: list[float] = []
        for coordinate in self.topology.tiles:
            tile = by_coordinate[coordinate]
            features += _one_hot(TILE_RESOURCES, tile.resource)
            features += _one_hot(DICE_NUMBERS, tile.number)
            features += [float(coordinate == observation.board.robber_coordinate)]
        return features

    def _nodes(self, observation: Observation) -> list[float]:
        """Encode every node's building and port in topology order.

        Args:
            observation: The seat's view.

        Returns:
            12 features per node: owner seat offset, building type and port.
        """
        board = observation.board
        features: list[float] = []
        for node in self.topology.nodes:
            owner, building = board.buildings.get(node, (None, None))
            features += _one_hot(range(MAX_PLAYERS), owner)
            features += [float(building == "SETTLEMENT"), float(building == "CITY")]
            features += (
                _one_hot(PORTS, board.ports[node])
                if node in board.ports
                else [0.0] * len(PORTS)
            )
        return features

    def _edges(self, observation: Observation) -> list[float]:
        """Encode every edge's road owner in topology order.

        Args:
            observation: The seat's view.

        Returns:
            4 features per edge: the owner's seat offset, or all zero.
        """
        roads = observation.board.roads
        features: list[float] = []
        for edge in self.topology.edges:
            features += _one_hot(range(MAX_PLAYERS), roads.get(edge))
        return features

    @staticmethod
    def _bank(observation: Observation) -> list[float]:
        """Encode the bank and the development deck's size.

        Args:
            observation: The seat's view.

        Returns:
            6 features.
        """
        return [_scaled(observation.bank_resources[r], 19) for r in RESOURCES] + [
            _scaled(observation.development_deck_size, 25)
        ]

    @staticmethod
    def _phase(observation: Observation) -> list[float]:
        """Encode the prompt, whose decision and turn it is, and turn flags.

        Args:
            observation: The seat's view.

        Returns:
            18 features.
        """
        phase = observation.phase
        return (
            _one_hot(PROMPTS, phase.prompt)
            + _one_hot(range(MAX_PLAYERS), phase.deciding_seat_offset)
            + _one_hot(range(MAX_PLAYERS), phase.turn_seat_offset)
            + [float(phase.has_rolled), float(phase.is_initial_build_phase)]
            + [float(phase.is_road_building), _scaled(phase.free_roads_available, 2)]
            + [_scaled(phase.turn_number, TURNS_LIMIT)]
        )

    @staticmethod
    def _discard(discard: SequentialDiscard | None) -> list[float]:
        """Encode a discard in progress.

        Args:
            discard: The observing seat's discard in progress, if any.

        Returns:
            6 features: cards left to pick, then cards picked per resource.
        """
        if discard is None:
            return [0.0] * DISCARD_FEATURES
        return [_scaled(discard.remaining, 20)] + [
            _scaled(discard.chosen[r], 19) for r in RESOURCES
        ]
