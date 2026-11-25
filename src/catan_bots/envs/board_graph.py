"""Neighbour tables for a board's tiles, nodes and edges, by slot index.

A graph network reads the board as three sets of rows (nodes, edges and land
tiles) in `BoardTopology` slot order, the same order the encoder writes them
in. To pass messages it needs to know which rows touch which. `BoardGraph`
holds that as small fixed-width integer tables: row `i` of `node_tiles` lists
the tile slots touching node slot `i`, padded with `PAD` where the node has
fewer than the table's width (a coastal node touches one or two tiles, not
three).
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from catanatron.models.map import CatanMap

from catan_bots.envs.topology import BoardTopology

PAD = -1
NODE_TILE_WIDTH = 3
NODE_NODE_WIDTH = 3
NODE_EDGE_WIDTH = 3
TILE_NODE_WIDTH = 6

Table = tuple[tuple[int, ...], ...]


def _padded(rows: list[list[int]], width: int) -> Table:
    """Pad every row of a ragged table to a fixed width with `PAD`.

    Args:
        rows: Neighbour slot indices per row, each sorted.
        width: Width to pad to.

    Returns:
        The table as tuples of exactly `width` entries.

    Raises:
        ValueError: If a row is wider than `width`.
    """
    for row in rows:
        if len(row) > width:
            raise ValueError(f"Row {row} is wider than {width}.")
    return tuple(tuple(row) + (PAD,) * (width - len(row)) for row in rows)


@dataclass(frozen=True)
class BoardGraph:
    """Which tiles, nodes and edges touch, as padded slot-index tables.

    Attributes:
        node_tiles: Per node, the land tiles it is a corner of (1-3).
        node_nodes: Per node, the nodes one edge away (2-3).
        node_edges: Per node, the edges it is an end of (2-3).
        edge_nodes: Per edge, its two end nodes.
        tile_nodes: Per tile, its six corner nodes.
    """

    node_tiles: Table
    node_nodes: Table
    node_edges: Table
    edge_nodes: Table
    tile_nodes: Table

    @property
    def num_nodes(self) -> int:
        """Number of node slots."""
        return len(self.node_tiles)

    @property
    def num_edges(self) -> int:
        """Number of edge slots."""
        return len(self.edge_nodes)

    @property
    def num_tiles(self) -> int:
        """Number of land-tile slots."""
        return len(self.tile_nodes)

    @classmethod
    def from_map(cls, catan_map: CatanMap) -> "BoardGraph":
        """Build the tables for a map's geometry, in `BoardTopology` order.

        Args:
            catan_map: Any map built from the template of interest.

        Returns:
            The map's neighbour tables.
        """
        topology = BoardTopology.from_map(catan_map)
        node_slot = {node: i for i, node in enumerate(topology.nodes)}
        tile_nodes = [
            sorted({node_slot[n] for n in catan_map.land_tiles[c].nodes.values()})
            for c in topology.tiles
        ]
        node_tiles: dict[int, list[int]] = defaultdict(list)
        for tile, corners in enumerate(tile_nodes):
            for node in corners:
                node_tiles[node].append(tile)
        edge_nodes = [[node_slot[a], node_slot[b]] for a, b in topology.edges]
        node_nodes: dict[int, list[int]] = defaultdict(list)
        node_edges: dict[int, list[int]] = defaultdict(list)
        for edge, (a, b) in enumerate(edge_nodes):
            node_nodes[a].append(b)
            node_nodes[b].append(a)
            node_edges[a].append(edge)
            node_edges[b].append(edge)
        nodes = range(len(topology.nodes))
        return cls(
            node_tiles=_padded([sorted(node_tiles[n]) for n in nodes], NODE_TILE_WIDTH),
            node_nodes=_padded([sorted(node_nodes[n]) for n in nodes], NODE_NODE_WIDTH),
            node_edges=_padded([sorted(node_edges[n]) for n in nodes], NODE_EDGE_WIDTH),
            edge_nodes=_padded(edge_nodes, 2),
            tile_nodes=_padded(tile_nodes, TILE_NODE_WIDTH),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the tables to plain lists, for saving inside a snapshot.

        Returns:
            Table name to a list of integer lists.
        """
        return {
            "node_tiles": [list(row) for row in self.node_tiles],
            "node_nodes": [list(row) for row in self.node_nodes],
            "node_edges": [list(row) for row in self.node_edges],
            "edge_nodes": [list(row) for row in self.edge_nodes],
            "tile_nodes": [list(row) for row in self.tile_nodes],
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "BoardGraph":
        """Rebuild tables written by `to_dict`.

        Args:
            values: Table name to a list of integer lists.

        Returns:
            The tables.
        """
        return cls(
            **{name: tuple(tuple(row) for row in rows) for name, rows in values.items()}
        )
