"""The fixed geometry of a Catan map: tile, node and edge orderings."""

from dataclasses import dataclass

from catanatron.models.map import CatanMap

Coordinate = tuple[int, int, int]
Edge = tuple[int, int]


@dataclass(frozen=True)
class BoardTopology:
    """Canonical orderings of a map's tiles, nodes and edges.

    Catanatron shuffles resources, numbers and ports between games but keeps
    the geometry of a template fixed, so these orderings index the same
    places in every game on that template. Both the action table and the
    observation encoder lay out their slots in this order.

    Attributes:
        tiles: Land-tile cube coordinates, sorted.
        nodes: Land node ids, sorted.
        edges: Land edges as `(low_node, high_node)`, sorted.
    """

    tiles: tuple[Coordinate, ...]
    nodes: tuple[int, ...]
    edges: tuple[Edge, ...]

    @classmethod
    def from_map(cls, catan_map: CatanMap) -> "BoardTopology":
        """Read the canonical orderings from a map.

        Args:
            catan_map: Any map built from the template of interest.

        Returns:
            The map's `BoardTopology`.
        """
        edges = {
            (min(edge), max(edge))
            for tile in catan_map.land_tiles.values()
            for edge in tile.edges.values()
        }
        return cls(
            tiles=tuple(sorted(catan_map.land_tiles)),
            nodes=tuple(sorted(catan_map.land_nodes)),
            edges=tuple(sorted(edges)),
        )

    def matches(self, catan_map: CatanMap) -> bool:
        """Check that a map shares this topology.

        Args:
            catan_map: Map to compare.

        Returns:
            True if its tiles, nodes and edges are the same as these.
        """
        return BoardTopology.from_map(catan_map) == self
