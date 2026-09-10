"""Tests for the `GreedySettlerBot` heuristic bot."""

import pytest
from catanatron import Action, ActionType, Color
from catanatron.models.player import RandomPlayer

from catan_bots.bots import GreedySettlerBot
from catan_bots.games import GameFactory


def make_action(action_type: ActionType, value: object = None) -> Action:
    """Build a red action of a given type.

    Args:
        action_type: Type of the action.
        value: Action payload. Defaults to `None`.

    Returns:
        The constructed `Action`.
    """
    return Action(Color.RED, action_type, value)


def test_prefers_city_over_settlement() -> None:
    """A city build outranks every other legal action."""
    bot = GreedySettlerBot(Color.RED, seed=0)
    actions = [
        make_action(ActionType.END_TURN),
        make_action(ActionType.BUILD_SETTLEMENT, 3),
        make_action(ActionType.BUILD_CITY, 5),
    ]
    assert bot.decide(game=None, playable_actions=actions).action_type is (
        ActionType.BUILD_CITY
    )


def test_falls_back_to_priority_order() -> None:
    """With no city available the bot takes the next priority type."""
    bot = GreedySettlerBot(Color.RED, seed=0)
    actions = [
        make_action(ActionType.END_TURN),
        make_action(ActionType.BUILD_ROAD, (0, 1)),
        make_action(ActionType.BUY_DEVELOPMENT_CARD),
    ]
    assert bot.decide(game=None, playable_actions=actions).action_type is (
        ActionType.BUY_DEVELOPMENT_CARD
    )


def test_falls_back_to_any_legal_action() -> None:
    """With no prioritised type legal the bot still returns a legal action."""
    bot = GreedySettlerBot(Color.RED, seed=0)
    actions = [make_action(ActionType.END_TURN), make_action(ActionType.ROLL)]
    assert bot.decide(game=None, playable_actions=actions) in actions


def test_seeded_bots_break_ties_identically() -> None:
    """Two bots with the same seed pick the same action among equals."""
    actions = [make_action(ActionType.BUILD_ROAD, edge) for edge in [(0, 1), (1, 2)]]
    first = GreedySettlerBot(Color.RED, seed=11)
    second = GreedySettlerBot(Color.RED, seed=11)
    picks = [
        (first.decide(None, actions), second.decide(None, actions)) for _ in range(5)
    ]
    assert all(a == b for a, b in picks)


def test_reset_state_restores_the_rng() -> None:
    """Resetting a seeded bot replays the same tie-break sequence."""
    actions = [make_action(ActionType.BUILD_ROAD, edge) for edge in [(0, 1), (1, 2)]]
    bot = GreedySettlerBot(Color.RED, seed=11)
    before = [bot.decide(None, actions) for _ in range(4)]
    bot.reset_state()
    after = [bot.decide(None, actions) for _ in range(4)]
    assert before == after


def test_empty_action_list_is_rejected() -> None:
    """Deciding with no legal actions raises `ValueError`."""
    with pytest.raises(ValueError):
        GreedySettlerBot(Color.RED).decide(game=None, playable_actions=[])


def test_bot_can_finish_a_real_game() -> None:
    """The bot plays a full catanatron game without erroring."""
    factory = GameFactory(
        [GreedySettlerBot(Color.RED, seed=1), RandomPlayer(Color.BLUE)]
    )
    game = factory.play(seed=42)
    assert game.state.actions
