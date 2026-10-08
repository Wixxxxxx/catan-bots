"""Tests for redacted per-seat observations.

The core tests are leak-invariance checks: changing something hidden in the
full state must leave every other seat's observation unchanged, while the
owner's own view does change — so a test cannot pass by observing nothing.
"""

import dataclasses
import random
from collections.abc import Mapping

import pytest
from catanatron import Action, ActionType, Color, Game
from catanatron.models.enums import ActionPrompt, RESOURCES
from catanatron.models.player import RandomPlayer
from catanatron.state_functions import (
    player_deck_replenish,
    player_key,
    player_num_dev_cards,
    player_num_resource_cards,
)

from catan_bots.analytics.board_stats import TileSummary
from catan_bots.games import GameFactory
from catan_bots.observation import Observation, ObservationBuilder

ROSTER = (Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE)
BUILDER = ObservationBuilder()


@pytest.fixture(name="game")
def fixture_game() -> Game:
    """Provide a four-player game well into play.

    Returns:
        A `Game` past the opening, with buildings and roads on the board.
    """
    game = GameFactory([RandomPlayer(c) for c in ROSTER]).create(seed=5)
    for _ in range(120):
        game.play_tick()
    return game


def observe_all(game: Game) -> dict[Color, Observation]:
    """Build every seat's observation of a position.

    Args:
        game: Game to observe.

    Returns:
        Observation per seated colour.
    """
    return {color: BUILDER.build(game.state, color) for color in game.state.colors}


def set_cards(game: Game, color: Color, cards: Mapping[str, int]) -> None:
    """Overwrite some of a player's card counts in place.

    Args:
        game: Game whose state is edited.
        color: Player to edit.
        cards: Count per player-state card name, e.g. `{"WOOD": 2}`.
    """
    key = player_key(game.state, color)
    for card, count in cards.items():
        game.state.player_state[f"{key}_{card}_IN_HAND"] = count


def assert_only_owner_sees(before: Game, after: Game, owner: Color) -> None:
    """Assert a change is visible to its owner and invisible to everyone else.

    Args:
        before: The original position.
        after: The same position with one hidden fact changed.
        owner: The only seat entitled to see the change.
    """
    seen_before, seen_after = observe_all(before), observe_all(after)
    assert seen_before[owner] != seen_after[owner], "test would pass vacuously"
    for color in before.state.colors:
        if color is not owner:
            assert seen_before[color] == seen_after[color], f"{color} sees it"


def test_deck_order_is_hidden_from_everyone(game: Game) -> None:
    """Reordering the development deck changes no seat's observation."""
    shuffled = game.copy()
    random.Random(0).shuffle(shuffled.state.development_listdeck)
    assert shuffled.state.development_listdeck != game.state.development_listdeck
    assert observe_all(game) == observe_all(shuffled)


def test_opponent_hand_types_are_hidden(game: Game) -> None:
    """Changing which resources a seat holds, at equal count, is private."""
    owner = Color.ORANGE
    before, after = game.copy(), game.copy()
    set_cards(before, owner, {"WOOD": 2, "BRICK": 0, "SHEEP": 0, "WHEAT": 0, "ORE": 1})
    set_cards(after, owner, {"WOOD": 0, "BRICK": 2, "SHEEP": 1, "WHEAT": 0, "ORE": 0})
    assert_only_owner_sees(before, after, owner)


def test_opponent_development_card_types_are_hidden(game: Game) -> None:
    """Changing which development card a seat holds, at equal count, is private."""
    owner = Color.WHITE
    before, after = game.copy(), game.copy()
    set_cards(before, owner, {"KNIGHT": 1, "MONOPOLY": 0})
    set_cards(after, owner, {"KNIGHT": 0, "MONOPOLY": 1})
    assert_only_owner_sees(before, after, owner)


def test_hidden_victory_points_are_hidden(game: Game) -> None:
    """A held VP card raises the owner's true score and nothing else visible."""
    owner = Color.RED
    key = player_key(game.state, owner)
    before, after = game.copy(), game.copy()
    set_cards(before, owner, {"KNIGHT": 1, "VICTORY_POINT": 0})
    set_cards(after, owner, {"KNIGHT": 0, "VICTORY_POINT": 1})
    after.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] += 1
    assert_only_owner_sees(before, after, owner)


def test_public_change_is_seen_by_everyone(game: Game) -> None:
    """Control: a public change, moving the robber, reaches every seat."""
    moved = game.copy()
    current = moved.state.board.robber_coordinate
    target = next(c for c in moved.state.board.map.land_tiles if c != current)
    moved.state.board.robber_coordinate = target
    before, after = observe_all(game), observe_all(moved)
    assert all(before[c] != after[c] for c in game.state.colors)


def test_own_hand_is_exact(game: Game) -> None:
    """A seat sees its own resources and development cards by type."""
    set_cards(game, Color.BLUE, {"ORE": 3, "WHEAT": 2, "KNIGHT": 1})
    hand = BUILDER.build(game.state, Color.BLUE).hand
    assert hand.resources["ORE"] == 3 and hand.resources["WHEAT"] == 2
    assert hand.development_cards["KNIGHT"] == 1


def test_opponent_counts_match_the_engine(game: Game) -> None:
    """Card counts shown for each seat equal the engine's true totals."""
    observation = BUILDER.build(game.state, Color.RED)
    for seat in observation.seats:
        assert seat.resource_card_count == player_num_resource_cards(
            game.state, seat.color
        )
        assert seat.development_card_count == player_num_dev_cards(
            game.state, seat.color
        )


def test_seats_are_in_play_order_from_the_perspective(game: Game) -> None:
    """Seat 0 is the perspective; the rest follow the seating rotation."""
    order = game.state.colors
    for perspective in order:
        observation = BUILDER.build(game.state, perspective)
        start = order.index(perspective)
        expected = [order[(start + i) % len(order)] for i in range(len(order))]
        assert [s.color for s in observation.seats] == expected
        assert [s.seat_offset for s in observation.seats] == list(range(len(order)))
        assert observation.opponents == observation.seats[1:]


def test_board_ownership_is_relative(game: Game) -> None:
    """A seat's own buildings are offset 0 in its view, k in others'."""
    node, (owner, _) = next(iter(game.state.board.buildings.items()))
    for perspective in game.state.colors:
        offset, _ = BUILDER.build(game.state, perspective).board.buildings[node]
        assert offset == ObservationBuilder.seat_offset(game.state, perspective, owner)
    assert BUILDER.build(game.state, owner).board.buildings[node][0] == 0


def test_roads_are_deduplicated(game: Game) -> None:
    """Each road appears once, though the engine stores both directions."""
    roads = BUILDER.build(game.state, Color.RED).board.roads
    assert len(roads) * 2 == len(game.state.board.roads)
    assert all(low < high for low, high in roads)


def test_board_is_complete(game: Game) -> None:
    """The board view has every land tile and both nodes of every port."""
    board = BUILDER.build(game.state, Color.RED).board
    assert len(board.tiles) == 19
    assert len(board.ports) == 18


def test_card_bought_this_turn_is_not_eligible() -> None:
    """A knight bought this turn is held but not eligible to be played."""
    game = GameFactory([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)]).create(
        seed=4
    )
    while game.state.is_initial_build_phase:
        game.play_tick()
    color = game.state.current_color()
    game.execute(Action(color, ActionType.ROLL, (2, 3)), validate_action=False)
    for resource in ("SHEEP", "WHEAT", "ORE"):
        player_deck_replenish(game.state, color, resource, 1)
    game.execute(
        Action(color, ActionType.BUY_DEVELOPMENT_CARD, "KNIGHT"), validate_action=False
    )
    hand = BUILDER.build(game.state, color).hand
    assert hand.development_cards["KNIGHT"] == 1
    assert hand.playable_development_cards["KNIGHT"] == 0


def test_no_card_is_eligible_after_playing_one(game: Game) -> None:
    """Once a card has been played this turn, nothing else is eligible."""
    set_cards(game, Color.RED, {"KNIGHT": 2, "MONOPOLY": 1})
    key = player_key(game.state, Color.RED)
    game.state.player_state[f"{key}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"] = True
    hand = BUILDER.build(game.state, Color.RED).hand
    assert not any(hand.playable_development_cards.values())


def test_phase_tracks_out_of_turn_discards() -> None:
    """While another seat discards, deciding and turn seats differ."""
    roster = [RandomPlayer(c) for c in ROSTER]
    for seed in range(1, 40):
        game = GameFactory(roster).create(seed=seed)
        while game.winning_color() is None and game.state.num_turns < 300:
            state = game.state
            turn_color = state.colors[state.current_turn_index]
            if (
                state.current_prompt is ActionPrompt.DISCARD
                and state.current_color() is not turn_color
            ):
                phase = BUILDER.build(state, turn_color).phase
                assert phase.prompt == "DISCARD"
                assert phase.turn_seat_offset == 0
                assert phase.deciding_seat_offset != 0
                return
            game.play_tick()
    pytest.fail("No out-of-turn discard found in 39 games.")


def test_unseated_perspective_is_rejected() -> None:
    """Observing from a colour that is not in the game raises `ValueError`."""
    game = GameFactory([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)]).create(
        seed=1
    )
    with pytest.raises(ValueError):
        BUILDER.build(game.state, Color.ORANGE)


def test_observation_holds_no_engine_objects(game: Game) -> None:
    """Structural guard: an observation contains only plain values.

    Fails if anyone later threads a `State`, `Board`, list (such as the
    development deck) or other live engine object into a view.
    """
    allowed_leaves = (int, float, str, bool, type(None), Color)

    def check(value: object, path: str) -> None:
        if dataclasses.is_dataclass(value):
            assert isinstance(value, (TileSummary,)) or type(
                value
            ).__module__.startswith("catan_bots.observation"), (
                f"{path}: unexpected dataclass {type(value).__name__}"
            )
            for field in dataclasses.fields(value):
                check(getattr(value, field.name), f"{path}.{field.name}")
        elif isinstance(value, dict):
            for k, v in value.items():
                check(k, f"{path}[key]")
                check(v, f"{path}[{k!r}]")
        elif isinstance(value, (tuple, frozenset)):
            for i, item in enumerate(value):
                check(item, f"{path}[{i}]")
        else:
            assert isinstance(value, allowed_leaves), (
                f"{path}: {type(value).__name__} is not a plain value"
            )

    for color in game.state.colors:
        check(BUILDER.build(game.state, color), "observation")


def test_bank_and_deck_size_are_public(game: Game) -> None:
    """The bank and the deck's size are shown; resources are all present."""
    observation = BUILDER.build(game.state, Color.RED)
    assert set(observation.bank_resources) == set(RESOURCES)
    assert observation.development_deck_size == len(game.state.development_listdeck)
