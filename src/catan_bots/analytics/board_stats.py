"""Static analysis of a Catan map: tile layout and node production quality."""

from dataclasses import dataclass

from catanatron import Game
from catanatron.models.map import CatanMap

Coordinate = tuple[int, int, int]


@dataclass(frozen=True)
class TileSummary:
    """One land tile of the map.

    Attributes:
        coordinate: Cube coordinate `(x, y, z)` with `x + y + z == 0`.
        resource: Resource produced, or `"DESERT"` for the desert tile.
        number: Dice number on the tile, or `None` for the desert.
    """

    coordinate: Coordinate
    resource: str
    number: int | None


@dataclass(frozen=True)
class NodeProduction:
    """Expected per-roll resource yield of one board node.

    Attributes:
        node_id: Identifier of the node on the board.
        total: Sum of expected yields across all resources.
        by_resource: Expected yield per resource name.
    """

    node_id: int
    total: float
    by_resource: dict[str, float]


class BoardInspector:
    """Answer questions about a map's tiles and settlement-spot quality.

    Attributes:
        catan_map: The map being inspected.
    """

    def __init__(self, catan_map: CatanMap) -> None:
        """Bind the inspector to a map.

        Args:
            catan_map: Map to inspect.
        """
        self.catan_map = catan_map

    @classmethod
    def from_game(cls, game: Game) -> "BoardInspector":
        """Build an inspector for the map a game is played on.

        Args:
            game: Game whose board map should be inspected.

        Returns:
            An inspector bound to `game.state.board.map`.
        """
        return cls(game.state.board.map)

    def tile_summaries(self) -> list[TileSummary]:
        """Describe every land tile, ordered by cube coordinate.

        Returns:
            One `TileSummary` per land tile.
        """
        return [
            TileSummary(
                coordinate=coordinate,
                resource=tile.resource or "DESERT",
                number=tile.number,
            )
            for coordinate, tile in sorted(self.catan_map.land_tiles.items())
        ]

    def node_productions(self) -> list[NodeProduction]:
        """Rank every node by total expected resource production.

        Returns:
            One `NodeProduction` per node, best producer first.
        """
        productions = [
            NodeProduction(
                node_id=node_id,
                total=sum(counter.values()),
                by_resource=dict(counter),
            )
            for node_id, counter in self.catan_map.node_production.items()
        ]
        return sorted(productions, key=lambda p: -p.total)

    def top_producing_nodes(self, limit: int = 5) -> list[NodeProduction]:
        """Return the best settlement spots by expected production.

        Args:
            limit: Maximum number of nodes to return.

        Returns:
            Up to `limit` `NodeProduction` entries, best producer first.
        """
        return self.node_productions()[:limit]
