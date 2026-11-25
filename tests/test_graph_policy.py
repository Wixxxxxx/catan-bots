"""Tests for the board's neighbour tables and the graph-network policy."""

from collections import Counter
from pathlib import Path

import pytest
import torch
from catanatron import ActionType

from catan_bots.envs import BoardGraph, CatanAECEnv
from catan_bots.envs.board_graph import PAD
from catan_bots.training.architectures import build_policy, policy_from_snapshot
from catan_bots.training.config import TrainingConfig
from catan_bots.training.graph_policy import GraphActorCritic, GraphLayout
from catan_bots.training.league import League, load_snapshot
from catan_bots.training.policy import MlpActorCritic


@pytest.fixture(name="env", scope="module")
def fixture_env() -> CatanAECEnv:
    """Provide a 4-player environment.

    Returns:
        The environment.
    """
    return CatanAECEnv(num_players=4)


@pytest.fixture(name="graph", scope="module")
def fixture_graph(env: CatanAECEnv) -> BoardGraph:
    """Provide the base map's neighbour tables.

    Args:
        env: The environment.

    Returns:
        The tables.
    """
    return env.board_graph


def make_policy(env: CatanAECEnv, rounds: int = 2) -> GraphActorCritic:
    """Build a small, seeded graph network.

    Args:
        env: The environment.
        rounds: Rounds of message passing.

    Returns:
        The network, in evaluation mode.
    """
    torch.manual_seed(0)
    config = TrainingConfig(architecture="graph", graph_width=16, graph_rounds=rounds)
    return build_policy(config, env)


def real_observations(env: CatanAECEnv, count: int) -> torch.Tensor:
    """Encode observations from a game in progress, after random moves.

    Args:
        env: The environment.
        count: Observations to collect.

    Returns:
        Shape `(count, observation_size)`.
    """
    env.reset(seed=3)
    rows = []
    generator = torch.Generator().manual_seed(0)
    while len(rows) < count:
        observation = env.observe(env.agent_selection)
        rows.append(torch.from_numpy(observation["observation"]))
        legal = torch.from_numpy(observation["action_mask"]).nonzero().flatten()
        env.step(int(legal[torch.randint(len(legal), (1,), generator=generator)]))
    return torch.stack(rows)


def neighbours(table: tuple[tuple[int, ...], ...], row: int) -> set[int]:
    """List a row's neighbours, without padding.

    Args:
        table: A padded neighbour table.
        row: Row index.

    Returns:
        The neighbour indices.
    """
    return {i for i in table[row] if i != PAD}


def test_board_graph_has_the_base_map_shape(graph: BoardGraph) -> None:
    """54 nodes, 72 edges and 19 tiles, with the base map's degrees."""
    assert (graph.num_nodes, graph.num_edges, graph.num_tiles) == (54, 72, 19)
    tiles_per_node = Counter(len(neighbours(graph.node_tiles, n)) for n in range(54))
    assert tiles_per_node == {3: 24, 2: 12, 1: 18}
    nodes_per_node = Counter(len(neighbours(graph.node_nodes, n)) for n in range(54))
    assert nodes_per_node == {3: 36, 2: 18}
    assert all(len(neighbours(graph.tile_nodes, t)) == 6 for t in range(19))


def test_board_graph_tables_agree_with_each_other(graph: BoardGraph) -> None:
    """Every adjacency is listed from both sides."""
    for node in range(graph.num_nodes):
        for other in neighbours(graph.node_nodes, node):
            assert node in neighbours(graph.node_nodes, other)
        for edge in neighbours(graph.node_edges, node):
            assert node in graph.edge_nodes[edge]
        for tile in neighbours(graph.node_tiles, node):
            assert node in graph.tile_nodes[tile]
    for edge, (a, b) in enumerate(graph.edge_nodes):
        assert b in neighbours(graph.node_nodes, a)
        assert edge in neighbours(graph.node_edges, a)


def test_board_graph_round_trips_through_plain_values(graph: BoardGraph) -> None:
    """The tables survive conversion to lists and back."""
    assert BoardGraph.from_dict(graph.to_dict()) == graph


def test_layout_splits_the_observation_into_board_rows(env: CatanAECEnv) -> None:
    """Nodes, edges and tiles carry their encoder widths; the rest is global."""
    layout = GraphLayout.from_env(env)
    assert (layout.node_features, layout.edge_features, layout.tile_features) == (
        12,
        4,
        17,
    )
    board = sum(
        stop - start
        for start, stop in (layout.node_span, layout.edge_span, layout.tile_span)
    )
    assert board + layout.global_features == env.encoder.size
    assert layout.board_actions == 54 * 2 + 72 + 19 * 4
    assert GraphLayout.from_dict(layout.to_dict()) == layout


def test_every_action_reads_exactly_one_score(env: CatanAECEnv) -> None:
    """The action order is a permutation of the network's scores."""
    sources = GraphLayout.from_env(env).action_sources
    assert sorted(sources) == list(range(env.table.size))


def test_node_rows_match_the_encoded_nodes(env: CatanAECEnv) -> None:
    """Row k of the node view is node slot k's 12 encoded features."""
    policy = make_policy(env)
    observations = real_observations(env, 40)
    rows = policy._rows(observations, policy.layout.node_span, 54)
    start = env.encoder.blocks["nodes"].start
    for k in (0, 17, 53):
        expected = observations[:, start + 12 * k : start + 12 * (k + 1)]
        torch.testing.assert_close(rows[:, k], expected)


def test_board_actions_are_scored_by_their_own_row(env: CatanAECEnv) -> None:
    """Settlement, city, road and robber logits come from the matching row."""
    policy = make_policy(env)
    table, topology = env.table, env.table.topology
    observations = real_observations(env, 8)
    with torch.no_grad():
        logits, _ = policy(observations)
        nodes, edges, tiles, _ = policy.encode(observations)
        settlement = policy.settlement_head(nodes).squeeze(-1)
        city = policy.city_head(nodes).squeeze(-1)
        road = policy.road_head(edges).squeeze(-1)
        robber = policy.robber_head(tiles)
    for slot, node in enumerate(topology.nodes):
        settle_index = table.index((ActionType.BUILD_SETTLEMENT, node))
        city_index = table.index((ActionType.BUILD_CITY, node))
        torch.testing.assert_close(logits[:, settle_index], settlement[:, slot])
        torch.testing.assert_close(logits[:, city_index], city[:, slot])
    for slot, edge in enumerate(topology.edges):
        index = table.index((ActionType.BUILD_ROAD, edge))
        torch.testing.assert_close(logits[:, index], road[:, slot])
    for slot, tile in enumerate(topology.tiles):
        for victim in range(table.max_players):
            index = table.index((ActionType.MOVE_ROBBER, tile, victim))
            torch.testing.assert_close(logits[:, index], robber[:, slot, victim])


def test_one_round_of_messages_reaches_only_direct_neighbours(
    env: CatanAECEnv, graph: BoardGraph
) -> None:
    """Changing one node moves its own and its neighbours' settlement scores only."""
    policy = make_policy(env, rounds=1)
    table, topology = env.table, env.table.topology
    settle = [table.index((ActionType.BUILD_SETTLEMENT, n)) for n in topology.nodes]
    observations = real_observations(env, 1)
    node = 20
    changed = observations.clone()
    start = env.encoder.blocks["nodes"].start + 12 * node
    changed[0, start : start + 4] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    changed[0, start + 4] = 1.0
    with torch.no_grad():
        before = policy(observations)[0][0, settle]
        after = policy(changed)[0][0, settle]
    moved = {slot for slot in range(54) if not torch.isclose(before[slot], after[slot])}
    assert moved == {node} | neighbours(graph.node_nodes, node)


def test_graph_snapshot_round_trips_through_a_file(
    env: CatanAECEnv, tmp_path: Path
) -> None:
    """A saved graph snapshot reloads safely and computes identical outputs."""
    policy = make_policy(env)
    path = League(tmp_path, max_snapshots=1).add(policy, 1)
    snapshot = load_snapshot(path)
    assert snapshot["architecture"] == "graph"
    assert not any("adjacency" in name for name in snapshot["weights"]), (
        "tables are rebuilt, not saved"
    )
    rebuilt = policy_from_snapshot(snapshot)
    assert isinstance(rebuilt, GraphActorCritic)
    observations = real_observations(env, 6)
    for original, copy in zip(policy(observations), rebuilt(observations)):
        torch.testing.assert_close(original, copy)


def test_snapshots_from_before_architectures_load_as_mlps(env: CatanAECEnv) -> None:
    """An old snapshot without an architecture name is an MLP."""
    snapshot = MlpActorCritic(env.encoder.size, env.table.size, (32,)).snapshot()
    del snapshot["architecture"]
    assert isinstance(policy_from_snapshot(snapshot), MlpActorCritic)


def test_unknown_architecture_is_rejected(env: CatanAECEnv) -> None:
    """A snapshot naming an unknown network fails clearly."""
    snapshot = make_policy(env).snapshot()
    snapshot["architecture"] = "transformer"
    with pytest.raises(ValueError, match="transformer"):
        policy_from_snapshot(snapshot)


def test_graph_network_trains_on_its_gradients(env: CatanAECEnv) -> None:
    """Every weight receives a gradient from a policy-and-value loss."""
    policy = make_policy(env).train()
    observations = real_observations(env, 16)
    masks = torch.ones(16, env.table.size, dtype=torch.bool)
    actions = torch.randint(env.table.size, (16,))
    log_probs, entropy, values = policy.evaluate_actions(observations, masks, actions)
    (-(log_probs.mean()) + values.pow(2).mean() - entropy.mean()).backward()
    missing = [name for name, p in policy.named_parameters() if p.grad is None]
    assert not missing
