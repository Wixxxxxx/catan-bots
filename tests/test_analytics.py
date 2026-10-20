"""Tests for the analytics package: accumulators, board stats, and reports."""

import pytest
from catanatron import Color, Game
from catanatron.models.enums import RESOURCES
from catanatron.models.player import RandomPlayer

from catan_bots.analytics import (
    ActionTypeCounter,
    BoardInspector,
    GameReport,
    Tournament,
)
from catan_bots.games import GameFactory


@pytest.fixture(name="factory")
def fixture_factory() -> GameFactory:
    """Provide a two-bot factory used across the analytics tests.

    Returns:
        A `GameFactory` seating two random bots.
    """
    return GameFactory([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)])


@pytest.fixture(name="finished_game")
def fixture_finished_game(factory: GameFactory) -> Game:
    """Provide one played-out game.

    Args:
        factory: Factory building the game.

    Returns:
        A finished `Game`.
    """
    return factory.play(seed=42)


def test_counter_totals_match_action_log(factory: GameFactory) -> None:
    """The accumulator counts exactly the actions the game logged."""
    counter = ActionTypeCounter()
    game = factory.play(seed=5, accumulators=[counter])
    assert counter.total_actions == len(game.state.actions)


def test_counter_resets_between_games(factory: GameFactory) -> None:
    """Reusing an accumulator does not carry counts across games."""
    counter = ActionTypeCounter()
    factory.play(seed=5, accumulators=[counter])
    first_total = counter.total_actions
    factory.play(seed=5, accumulators=[counter])
    assert counter.total_actions == first_total


def test_ranked_counts_are_descending(factory: GameFactory) -> None:
    """Ranked counts come back ordered from most to least frequent."""
    counter = ActionTypeCounter()
    factory.play(seed=5, accumulators=[counter])
    counts = [count for _, count in counter.ranked_counts()]
    assert counts == sorted(counts, reverse=True)


def test_inspector_lists_every_land_tile(finished_game: Game) -> None:
    """The inspector summarises all 19 land tiles of the base map."""
    inspector = BoardInspector.from_game(finished_game)
    summaries = inspector.tile_summaries()
    assert len(summaries) == 19
    assert sum(1 for tile in summaries if tile.resource == "DESERT") == 1


def test_top_producing_nodes_are_ranked(finished_game: Game) -> None:
    """Top nodes are returned best-first and respect the requested limit."""
    inspector = BoardInspector.from_game(finished_game)
    top = inspector.top_producing_nodes(limit=3)
    assert len(top) == 3
    assert [node.total for node in top] == sorted(
        (node.total for node in top), reverse=True
    )


def test_report_snapshots_every_player(finished_game: Game) -> None:
    """The report holds one snapshot per seated player, keyed by colour."""
    report = GameReport(finished_game)
    assert {s.color for s in report.snapshots} == {Color.RED, Color.BLUE}
    assert report.snapshot_for(Color.RED).color is Color.RED


def test_report_snapshot_fields_are_consistent(finished_game: Game) -> None:
    """Snapshot totals agree with their per-resource breakdown."""
    snapshot = GameReport(finished_game).snapshot_for(Color.RED)
    assert set(snapshot.resources) == set(RESOURCES)
    assert snapshot.total_resources == sum(snapshot.resources.values())
    assert snapshot.victory_points >= snapshot.visible_victory_points


def test_report_is_a_frozen_view(factory: GameFactory) -> None:
    """A report keeps the turn count it was taken at as the game advances."""
    game = factory.create(seed=8)
    game.play_tick()
    report = GameReport(game)
    captured_actions = len(report.action_log)
    game.play_tick()
    assert len(report.action_log) == captured_actions


def test_report_renders_tables(finished_game: Game) -> None:
    """Rendered summaries mention the winner and every seated colour."""
    report = GameReport(finished_game)
    table = report.player_table()
    assert all(color.value in table for color in (Color.RED, Color.BLUE))
    assert "Winner" in report.summary()
    assert len(report.recent_actions(limit=4)) == 4


def test_unclaimed_largest_army_reports_zero(factory: GameFactory) -> None:
    """A game with no knights played reports an army size of 0."""
    game = factory.create(seed=8)
    report = GameReport(game)
    assert report.largest_army_holder is None
    assert report.largest_army_size == 0


def test_tournament_tally_covers_every_game(factory: GameFactory) -> None:
    """Wins and draws sum to the number of games played."""
    result = Tournament(factory, num_games=3, first_seed=0).run()
    assert result.num_games == 3
    assert sum(result.win_counts.values()) == 3
    assert len(result.turns_played) == 3


def test_tournament_is_reproducible(factory: GameFactory) -> None:
    """Two runs of the same tournament produce the same standings."""
    first = Tournament(factory, num_games=2, first_seed=0).run()
    second = Tournament(factory, num_games=2, first_seed=0).run()
    assert first.win_counts == second.win_counts
    assert first.turns_played == second.turns_played


def test_win_rates_sum_to_one(factory: GameFactory) -> None:
    """Win rates across colours and draws add up to 1."""
    result = Tournament(factory, num_games=2, first_seed=10).run()
    total = sum(result.win_rate(color) for color in result.win_counts)
    assert total == pytest.approx(1.0)
    assert result.mean_turns > 0
    assert "Win rates over 2 games" in result.as_text()


def test_non_positive_game_count_is_rejected(factory: GameFactory) -> None:
    """A tournament of zero games raises `ValueError`."""
    with pytest.raises(ValueError):
        Tournament(factory, num_games=0)
