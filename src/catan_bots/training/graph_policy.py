"""A graph network policy that reads the board as connected nodes, edges and tiles.

The observation is the same flat vector the MLP reads. `GraphLayout` says
where the board blocks sit in it, so the network can view them as rows: 54
nodes x 12 features, 72 edges x 4 and 19 land tiles x 17. Everything else
(hand, seats, bank, phase, discard, trade) is the "global" block.

The forward pass:

1. **Embed.** One small layer per row kind, shared by every row of that
   kind, turns each row into a `width`-wide vector. The global block gets
   its own embedding, `g`.
2. **Message passing, `rounds` times.** Each row reads its neighbours'
   vectors through the `BoardGraph` tables (a node reads its tiles,
   neighbouring nodes and edges; an edge reads its two ends; a tile reads
   its six corners), plus `g`. Neighbour vectors are summed and divided by
   the table's width (a multiply by a fixed averaging matrix), so a node
   touching three tiles gets a bigger message than a coastal node touching
   one, yet the scale stays bounded. A shared
   update per row kind combines a row with its messages, and is added back
   to the row (residual) then layer-normalised. `g` is updated from the
   pooled rows. Every row of a kind uses the same weights; each round has
   its own.
3. **Heads.** Settlement and city scores come from each node's vector, road
   scores from each edge's, robber scores from each tile's (one per victim
   offset). The remaining actions (rolling, cards, trades...) and the value
   come from a pooled summary of the whole board plus `g`. The pieces are
   put back into action-table order, so the output is the same 587 logits
   the MLP produces, and masking and PPO are unchanged.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from catanatron import ActionType
from torch import nn
from torch.nn import functional

from catan_bots.envs import CatanAECEnv
from catan_bots.envs.actions import ActionTable
from catan_bots.envs.board_graph import PAD, BoardGraph
from catan_bots.envs.encoding import ObservationEncoder
from catan_bots.training.policy import ActorCritic, orthogonal_layer

BOARD_BLOCKS = ("tiles", "nodes", "edges")
Span = tuple[int, int]


@dataclass(frozen=True)
class GraphLayout:
    """Where the board sits in the observation and which piece scores each action.

    Attributes:
        observation_size: Length of the encoded observation.
        num_actions: Size of the action space.
        node_span: Start and stop of the node block in the observation.
        edge_span: Start and stop of the edge block.
        tile_span: Start and stop of the tile block.
        global_spans: Every other block, as start and stop, merged where
            adjacent.
        robber_victims: Robber slots per tile (one per victim seat offset).
        action_sources: For each action index, its position in the network's
            concatenated scores: node settlement scores, node city scores,
            edge road scores, tile robber scores, then the pooled "other"
            scores.
        graph: Neighbour tables in the encoder's slot order.
    """

    observation_size: int
    num_actions: int
    node_span: Span
    edge_span: Span
    tile_span: Span
    global_spans: tuple[Span, ...]
    robber_victims: int
    action_sources: tuple[int, ...]
    graph: BoardGraph

    @property
    def node_features(self) -> int:
        """Features per node row."""
        return (self.node_span[1] - self.node_span[0]) // self.graph.num_nodes

    @property
    def edge_features(self) -> int:
        """Features per edge row."""
        return (self.edge_span[1] - self.edge_span[0]) // self.graph.num_edges

    @property
    def tile_features(self) -> int:
        """Features per tile row."""
        return (self.tile_span[1] - self.tile_span[0]) // self.graph.num_tiles

    @property
    def global_features(self) -> int:
        """Features in the global block."""
        return sum(stop - start for start, stop in self.global_spans)

    @property
    def board_actions(self) -> int:
        """Actions scored from a single node, edge or tile."""
        graph = self.graph
        return (
            2 * graph.num_nodes
            + graph.num_edges
            + graph.num_tiles * self.robber_victims
        )

    @property
    def other_actions(self) -> int:
        """Actions scored from the pooled summary."""
        return self.num_actions - self.board_actions

    @classmethod
    def from_env(cls, env: CatanAECEnv) -> "GraphLayout":
        """Read the layout from an environment's encoder, action table and board.

        Args:
            env: The environment the policy will play.

        Returns:
            The layout.
        """
        return cls.build(env.encoder, env.table, env.board_graph)

    @classmethod
    def build(
        cls, encoder: ObservationEncoder, table: ActionTable, graph: BoardGraph
    ) -> "GraphLayout":
        """Combine an encoder's blocks, an action table and neighbour tables.

        Args:
            encoder: The observation encoder.
            table: The action table.
            graph: Neighbour tables for the same board.

        Returns:
            The layout.

        Raises:
            ValueError: If the three disagree about the board.
        """
        if encoder.topology != table.topology:
            raise ValueError("Encoder and action table use different boards.")
        topology = encoder.topology
        sizes = (len(topology.nodes), len(topology.edges), len(topology.tiles))
        if sizes != (graph.num_nodes, graph.num_edges, graph.num_tiles):
            raise ValueError("Neighbour tables do not match the board.")
        blocks = encoder.blocks
        return cls(
            observation_size=encoder.size,
            num_actions=table.size,
            node_span=_span(blocks["nodes"]),
            edge_span=_span(blocks["edges"]),
            tile_span=_span(blocks["tiles"]),
            global_spans=_merged(
                [_span(s) for name, s in blocks.items() if name not in BOARD_BLOCKS]
            ),
            robber_victims=table.max_players,
            action_sources=_action_sources(table),
            graph=graph,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the layout to plain values, for saving inside a snapshot.

        Returns:
            Field name to plain value.
        """
        return {
            "observation_size": self.observation_size,
            "num_actions": self.num_actions,
            "node_span": list(self.node_span),
            "edge_span": list(self.edge_span),
            "tile_span": list(self.tile_span),
            "global_spans": [list(span) for span in self.global_spans],
            "robber_victims": self.robber_victims,
            "action_sources": list(self.action_sources),
            "graph": self.graph.to_dict(),
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "GraphLayout":
        """Rebuild a layout written by `to_dict`.

        Args:
            values: Field name to plain value.

        Returns:
            The layout.
        """
        return cls(
            observation_size=values["observation_size"],
            num_actions=values["num_actions"],
            node_span=tuple(values["node_span"]),
            edge_span=tuple(values["edge_span"]),
            tile_span=tuple(values["tile_span"]),
            global_spans=tuple(tuple(span) for span in values["global_spans"]),
            robber_victims=values["robber_victims"],
            action_sources=tuple(values["action_sources"]),
            graph=BoardGraph.from_dict(values["graph"]),
        )


def _span(block: slice) -> Span:
    """Convert a block slice to a start and stop.

    Args:
        block: A contiguous slice.

    Returns:
        `(start, stop)`.
    """
    return block.start, block.stop


def _merged(spans: list[Span]) -> tuple[Span, ...]:
    """Merge spans that touch, keeping their order.

    Args:
        spans: Sorted, non-overlapping spans.

    Returns:
        The fewest spans covering the same positions.
    """
    merged: list[Span] = []
    for start, stop in spans:
        if merged and merged[-1][1] == start:
            merged[-1] = (merged[-1][0], stop)
        else:
            merged.append((start, stop))
    return tuple(merged)


def _action_sources(table: ActionTable) -> tuple[int, ...]:
    """Find, for every action index, which network score it reads.

    Args:
        table: The action table.

    Returns:
        Positions in the concatenated scores, one per action index; a
        permutation of `range(table.size)`.
    """
    topology = table.topology
    node_slot = {node: i for i, node in enumerate(topology.nodes)}
    edge_slot = {edge: i for i, edge in enumerate(topology.edges)}
    tile_slot = {tile: i for i, tile in enumerate(topology.tiles)}
    nodes, edges = len(topology.nodes), len(topology.edges)
    robber_start = 2 * nodes + edges
    other = robber_start + len(topology.tiles) * table.max_players
    sources = []
    for key in table.keys:
        kind = key[0]
        if kind is ActionType.BUILD_SETTLEMENT:
            sources.append(node_slot[key[1]])
        elif kind is ActionType.BUILD_CITY:
            sources.append(nodes + node_slot[key[1]])
        elif kind is ActionType.BUILD_ROAD:
            sources.append(2 * nodes + edge_slot[key[1]])
        elif kind is ActionType.MOVE_ROBBER:
            sources.append(
                robber_start + tile_slot[key[1]] * table.max_players + key[2]
            )
        else:
            sources.append(other)
            other += 1
    return tuple(sources)


def neighbour_matrix(table: tuple[tuple[int, ...], ...], sources: int) -> torch.Tensor:
    """Turn a padded neighbour table into a dense averaging matrix.

    Entry `[k, r]` is `1 / D` (`D` the table's width) when source row `r`
    neighbours row `k`, else 0. Multiplying it by a batch of source rows sums
    each row's neighbours divided by `D`, so a node touching three tiles gets
    a bigger message than a coastal node touching one, while the scale stays
    bounded. A dense matrix multiply is much faster to differentiate than
    gathering rows by index, and the board is small enough that the matrices
    are tiny (at most 72 x 54).

    Args:
        table: Neighbour indices per row, padded with `PAD`.
        sources: Number of source rows.

    Returns:
        Shape `(len(table), sources)`.
    """
    matrix = torch.zeros(len(table), sources)
    for row, neighbours in enumerate(table):
        for source in neighbours:
            if source != PAD:
                matrix[row, source] = 1.0 / len(neighbours)
    return matrix


def _projection(width: int) -> nn.Linear:
    """Create a bias-free, orthogonally initialised square layer.

    Args:
        width: Input and output width.

    Returns:
        The layer.
    """
    layer = nn.Linear(width, width, bias=False)
    nn.init.orthogonal_(layer.weight, np.sqrt(2))
    return layer


class RowUpdate(nn.Module):
    """One shared residual update for every row of one kind.

    `row <- LayerNorm(row + W_out ReLU(W_in [row, message_1, ...] + W_g g))`,
    the same weights for every row. The global vector `g` is the same for
    every row, so it is projected once per position and broadcast rather
    than copied into every row first.
    """

    def __init__(self, pieces: int, width: int, reads_context: bool = True) -> None:
        """Build the update.

        Args:
            pieces: Row-shaped inputs side by side: the row plus its messages.
            width: Row width.
            reads_context: Whether the global vector is added in.
        """
        super().__init__()
        self.combine = orthogonal_layer(pieces * width, width, gain=np.sqrt(2))
        self.context_in = _projection(width) if reads_context else None
        self.output = orthogonal_layer(width, width, gain=1.0)
        self.norm = nn.LayerNorm(width)

    def forward(
        self,
        rows: torch.Tensor,
        messages: list[torch.Tensor],
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Update every row from itself, its messages and the global vector.

        Args:
            rows: Row vectors, shape `(B, R, W)` (or `(B, W)` for the global
                vector's own update).
            messages: Same shape as `rows`.
            context: The global vector `(B, W)`, if this update reads it.

        Returns:
            The updated rows, same shape as `rows`.
        """
        hidden = self.combine(torch.cat([rows, *messages], dim=-1))
        if self.context_in is not None:
            hidden = hidden + self.context_in(context).unsqueeze(-2)
        return self.norm(rows + self.output(functional.relu(hidden)))


class MessageRound(nn.Module):
    """One round of message passing: every node, edge, tile and the global vector updated once.

    All four updates read the previous round's vectors, so the order inside
    a round does not matter.
    """

    def __init__(self, width: int) -> None:
        """Build the round's four shared updates.

        Args:
            width: Row width.
        """
        super().__init__()
        self.nodes = RowUpdate(4, width)
        self.edges = RowUpdate(2, width)
        self.tiles = RowUpdate(2, width)
        self.context = RowUpdate(4, width, reads_context=False)

    def forward(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        adjacency: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pass one round of messages.

        Args:
            state: Node `(B, N, W)`, edge `(B, E, W)` and tile `(B, T, W)`
                rows and the global vector `(B, W)`.
            adjacency: Averaging matrices from `neighbour_matrix`, by name.

        Returns:
            The updated `state`.
        """
        nodes, edges, tiles, context = state
        new_nodes = self.nodes(
            nodes,
            [
                adjacency["node_tiles"] @ tiles,
                adjacency["node_nodes"] @ nodes,
                adjacency["node_edges"] @ edges,
            ],
            context,
        )
        new_edges = self.edges(edges, [adjacency["edge_nodes"] @ nodes], context)
        new_tiles = self.tiles(tiles, [adjacency["tile_nodes"] @ nodes], context)
        new_context = self.context(
            context, [nodes.mean(1), edges.mean(1), tiles.mean(1)]
        )
        return new_nodes, new_edges, new_tiles, new_context


class GraphActorCritic(ActorCritic):
    """Policy and value from message passing over the board's nodes, edges and tiles.

    Attributes:
        layout: Where the board sits in the observation and action space.
        width: Width of every row vector.
        rounds: Rounds of message passing.
    """

    architecture = "graph"

    def __init__(self, layout: GraphLayout, width: int = 64, rounds: int = 3) -> None:
        """Build the network.

        Args:
            layout: Where the board sits in the observation and action space.
            width: Width of every row vector.
            rounds: Rounds of message passing.
        """
        super().__init__(layout.observation_size, layout.num_actions)
        self.layout = layout
        self.width = width
        self.rounds = rounds
        self._register_tables(layout)
        gain = np.sqrt(2)
        self.embed_nodes = _embedding(layout.node_features, width)
        self.embed_edges = _embedding(layout.edge_features, width)
        self.embed_tiles = _embedding(layout.tile_features, width)
        self.embed_context = _embedding(layout.global_features, width)
        self.message_rounds = nn.ModuleList(MessageRound(width) for _ in range(rounds))
        self.settlement_head = orthogonal_layer(width, 1, gain=0.01)
        self.city_head = orthogonal_layer(width, 1, gain=0.01)
        self.road_head = orthogonal_layer(width, 1, gain=0.01)
        self.robber_head = orthogonal_layer(width, layout.robber_victims, gain=0.01)
        self.summary = nn.Sequential(
            orthogonal_layer(5 * width, 4 * width, gain=gain), nn.ReLU()
        )
        self.other_head = orthogonal_layer(4 * width, layout.other_actions, gain=0.01)
        self.value_head = orthogonal_layer(4 * width, 1, gain=1.0)

    def _register_tables(self, layout: GraphLayout) -> None:
        """Store the averaging matrices and action order as non-saved buffers.

        They are rebuilt from the layout, so they are left out of the saved
        weights, and they move with the network between devices.

        Args:
            layout: The layout holding the neighbour tables.
        """
        graph = layout.graph
        matrices = {
            "node_tiles": (graph.node_tiles, graph.num_tiles),
            "node_nodes": (graph.node_nodes, graph.num_nodes),
            "node_edges": (graph.node_edges, graph.num_edges),
            "edge_nodes": (graph.edge_nodes, graph.num_nodes),
            "tile_nodes": (graph.tile_nodes, graph.num_nodes),
        }
        for name, (table, sources) in matrices.items():
            self.register_buffer(
                f"adjacency_{name}", neighbour_matrix(table, sources), persistent=False
            )
        self.register_buffer(
            "action_sources",
            torch.tensor(layout.action_sources, dtype=torch.long),
            persistent=False,
        )

    def _adjacency(self) -> dict[str, torch.Tensor]:
        """Collect the averaging matrices by name.

        Returns:
            Table name to matrix.
        """
        return {
            name: getattr(self, f"adjacency_{name}")
            for name in (
                "node_tiles",
                "node_nodes",
                "node_edges",
                "edge_nodes",
                "tile_nodes",
            )
        }

    def _rows(self, observations: torch.Tensor, span: Span, count: int) -> torch.Tensor:
        """View one board block of the observation as rows.

        Args:
            observations: Batch of encoded observations, shape `(B, obs)`.
            span: The block's start and stop.
            count: Rows in the block.

        Returns:
            Shape `(B, count, features)`.
        """
        block = observations[:, span[0] : span[1]]
        return block.reshape(observations.shape[0], count, -1)

    def _global(self, observations: torch.Tensor) -> torch.Tensor:
        """Gather the non-board blocks of the observation.

        Args:
            observations: Batch of encoded observations.

        Returns:
            Shape `(B, global_features)`.
        """
        parts = [
            observations[:, start:stop] for start, stop in self.layout.global_spans
        ]
        return torch.cat(parts, dim=-1)

    def encode(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed the board and run every round of message passing.

        Args:
            observations: Batch of encoded observations, shape `(B, obs)`.

        Returns:
            Final node `(B, N, W)`, edge `(B, E, W)` and tile `(B, T, W)`
            rows and the global vector `(B, W)`.
        """
        layout, graph = self.layout, self.layout.graph
        state = (
            self.embed_nodes(
                self._rows(observations, layout.node_span, graph.num_nodes)
            ),
            self.embed_edges(
                self._rows(observations, layout.edge_span, graph.num_edges)
            ),
            self.embed_tiles(
                self._rows(observations, layout.tile_span, graph.num_tiles)
            ),
            self.embed_context(self._global(observations)),
        )
        adjacency = self._adjacency()
        for message_round in self.message_rounds:
            state = message_round(state, adjacency)
        return state

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Score actions and estimate value.

        Args:
            observations: Batch of encoded observations, shape `(B, obs)`.

        Returns:
            Unmasked action logits `(B, actions)` in action-table order, and
            values `(B,)`.
        """
        nodes, edges, tiles, context = self.encode(observations)
        summary = self.summary(
            torch.cat(
                [context, nodes.mean(1), nodes.amax(1), edges.mean(1), tiles.mean(1)],
                dim=-1,
            )
        )
        scores = torch.cat(
            [
                self.settlement_head(nodes).squeeze(-1),
                self.city_head(nodes).squeeze(-1),
                self.road_head(edges).squeeze(-1),
                self.robber_head(tiles).flatten(1),
                self.other_head(summary),
            ],
            dim=1,
        )
        logits = scores.index_select(1, self.action_sources)
        return logits, self.value_head(summary).squeeze(-1)

    def settings(self) -> dict[str, Any]:
        """List the settings that rebuild this network.

        Returns:
            Width, rounds and the layout as plain values.
        """
        return {
            "graph_width": self.width,
            "graph_rounds": self.rounds,
            "layout": self.layout.to_dict(),
        }

    @classmethod
    def from_settings(
        cls, observation_size: int, num_actions: int, settings: dict[str, Any]
    ) -> "GraphActorCritic":
        """Build an untrained network from saved settings.

        Args:
            observation_size: Length of the encoded observation.
            num_actions: Size of the action space.
            settings: Holds `graph_width`, `graph_rounds` and `layout`.

        Returns:
            The network, with fresh weights.

        Raises:
            ValueError: If the saved layout disagrees with the sizes.
        """
        layout = GraphLayout.from_dict(settings["layout"])
        if (layout.observation_size, layout.num_actions) != (
            observation_size,
            num_actions,
        ):
            raise ValueError("Saved graph layout does not match the saved sizes.")
        return cls(layout, settings["graph_width"], settings["graph_rounds"])


def _embedding(features: int, width: int) -> nn.Sequential:
    """Build a shared row embedding.

    Args:
        features: Input features per row.
        width: Output width.

    Returns:
        A linear layer followed by ReLU.
    """
    return nn.Sequential(orthogonal_layer(features, width, gain=np.sqrt(2)), nn.ReLU())
