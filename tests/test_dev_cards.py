"""Tests for the development-card timing rule."""

from collections import Counter

import pytest
from catanatron import Action, ActionType, Color, Game
from catanatron.models.player import RandomPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.state_functions import player_deck_replenish, player_key

from catan_bots.games import GameFactory
from catan_bots.rules.dev_cards import DevCardTimingRule

ROSTER = (Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE)
PLAYS = DevCardTimingRule.PLAY_ACTIONS


def post_roll_game(seed: int = 4) -> tuple[Game, Color]:
    """Advance a two-player game to a post-roll turn with dev-card money.

    Args:
        seed: Game seed.

    Returns:
        The game and the colour whose turn it is, holding enough sheep, wheat
        and ore to buy several development cards.
    """
    game = GameFactory([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)]).create(
        seed=seed
    )
    while game.state.is_initial_build_phase:
        game.play_tick()
    color = game.state.current_color()
    game.execute(Action(color, ActionType.ROLL, (2, 3)), validate_action=False)
    for resource in ("SHEEP", "WHEAT", "ORE"):
        player_deck_replenish(game.state, color, resource, 5)
    return game, color


def buy(game: Game, color: Color, card: str) -> None:
    """Buy one specific development card for a player.

    Args:
        game: Game to act on.
        color: Buyer's colour.
        card: Development-card name to draw from the deck.
    """
    game.execute(
        Action(color, ActionType.BUY_DEVELOPMENT_CARD, card), validate_action=False
    )


def play_types(actions: list[Action]) -> set[ActionType]:
    """Collect the development-card play types among some actions.

    Args:
        actions: Actions to inspect.

    Returns:
        The play action types present.
    """
    return {a.action_type for a in actions if a.action_type in PLAYS}


def timing_violations(game: Game) -> int:
    """Replay a finished game's log and count same-turn development-card plays.

    Cards only enter a hand by purchase and leave it by being played, so the
    copies a player may play are those bought in earlier turns and not yet
    spent.

    Args:
        game: A played game.

    Returns:
        Number of plays of a card that had no copy bought before that turn.
    """
    owned_before_turn: dict[Color, Counter[str]] = {c: Counter() for c in ROSTER}
    bought_this_turn: dict[Color, Counter[str]] = {c: Counter() for c in ROSTER}
    violations = 0
    for action in game.state.actions:
        kind, color = action.action_type, action.color
        if kind is ActionType.BUY_DEVELOPMENT_CARD:
            bought_this_turn[color][action.value] += 1
        elif kind in PLAYS:
            card = PLAYS[kind]
            if owned_before_turn[color][card] > 0:
                owned_before_turn[color][card] -= 1
            else:
                violations += 1
                bought_this_turn[color][card] -= 1
        elif kind is ActionType.END_TURN:
            owned_before_turn[color].update(bought_this_turn[color])
            bought_this_turn[color].clear()
    return violations


def test_card_bought_this_turn_cannot_be_played() -> None:
    """A knight bought this turn is offered by the engine but removed."""
    game, color = post_roll_game()
    buy(game, color, "KNIGHT")
    assert ActionType.PLAY_KNIGHT_CARD in play_types(game.state.playable_actions)
    legal = DevCardTimingRule().legal_actions(game.state)
    assert ActionType.PLAY_KNIGHT_CARD not in play_types(legal)


@pytest.mark.parametrize("card", ["KNIGHT", "YEAR_OF_PLENTY", "MONOPOLY"])
def test_every_playable_card_type_is_restricted(card: str) -> None:
    """Each playable card type is blocked the turn it is bought."""
    game, color = post_roll_game()
    buy(game, color, card)
    legal = DevCardTimingRule().legal_actions(game.state)
    blocked = {kind for kind, name in PLAYS.items() if name == card}
    assert not play_types(legal) & blocked


def test_older_copy_remains_playable() -> None:
    """Buying a second knight does not block the one held from before."""
    game, color = post_roll_game()
    game.state.player_state[f"{player_key(game.state, color)}_KNIGHT_IN_HAND"] = 1
    buy(game, color, "KNIGHT")
    legal = DevCardTimingRule().legal_actions(game.state)
    assert ActionType.PLAY_KNIGHT_CARD in play_types(legal)


def test_purchases_are_counted_per_turn_and_colour() -> None:
    """Only the current player's purchases this turn are counted."""
    game, color = post_roll_game()
    buy(game, color, "KNIGHT")
    buy(game, color, "MONOPOLY")
    rule = DevCardTimingRule()
    assert rule.cards_bought_this_turn(game.state, color) == Counter(
        {"KNIGHT": 1, "MONOPOLY": 1}
    )
    other = next(c for c in game.state.colors if c is not color)
    assert not rule.cards_bought_this_turn(game.state, other)


def test_card_becomes_playable_next_turn() -> None:
    """A knight bought last turn may be played before rolling this turn."""
    game, color = post_roll_game()
    buy(game, color, "KNIGHT")
    game.execute(Action(color, ActionType.END_TURN, None))
    opponent = game.state.current_color()
    game.execute(Action(opponent, ActionType.ROLL, (2, 3)), validate_action=False)
    game.execute(Action(opponent, ActionType.END_TURN, None))
    assert game.state.current_color() is color
    legal = DevCardTimingRule().legal_actions(game.state)
    assert ActionType.PLAY_KNIGHT_CARD in play_types(legal)


def test_victory_point_card_still_scores_immediately() -> None:
    """The official exception holds: a VP card bought counts at once."""
    game, color = post_roll_game()
    key = player_key(game.state, color)
    before = game.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"]
    buy(game, color, "VICTORY_POINT")
    assert game.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] == before + 1


def test_rule_never_mutates_the_shared_action_list() -> None:
    """Filtering returns a new list; copies share the original by reference."""
    game, color = post_roll_game()
    buy(game, color, "KNIGHT")
    original = game.state.playable_actions
    snapshot = list(original)
    legal = DevCardTimingRule().legal_actions(game.state)
    assert legal is not original
    assert original == snapshot


def test_rule_keeps_every_non_play_action() -> None:
    """Only development-card plays are ever removed."""
    game, color = post_roll_game()
    buy(game, color, "KNIGHT")
    legal = set(DevCardTimingRule().legal_actions(game.state))
    removed = set(game.state.playable_actions) - legal
    assert all(a.action_type in PLAYS for a in removed)


def test_factory_games_never_play_a_card_the_turn_it_was_bought() -> None:
    """Full games under the default factory contain no timing violations."""
    roster = [WeightedRandomPlayer(c) for c in ROSTER]
    factory = GameFactory(roster)
    for seed in range(1, 21):
        assert timing_violations(factory.play(seed=seed)) == 0


def test_stock_engine_does_violate_the_rule() -> None:
    """With the rule disabled, the same games contain same-turn plays.

    Guards the verifier itself: it must be able to detect a violation.
    """
    roster = [WeightedRandomPlayer(c) for c in ROSTER]
    factory = GameFactory(roster, enforce_dev_card_timing=False)
    total = sum(timing_violations(factory.play(seed=s)) for s in range(1, 21))
    assert total > 0
