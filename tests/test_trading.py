"""Tests for the domestic trading protocol."""

from collections import Counter
from collections.abc import Mapping

import pytest
from catanatron import Action, ActionType, Color, Game
from catanatron.models.enums import RESOURCES
from catanatron.models.player import RandomPlayer
from catanatron.state_functions import player_deck_to_array, player_key

from catan_bots.games import GameFactory
from catan_bots.rules.trading import (
    TradeOffer,
    TradeProtocol,
    TradingRules,
    describe,
    enumerate_offers,
)

ROSTER = (Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE)
WOOD_FOR_ORE = TradeOffer.of({"WOOD": 1}, {"ORE": 1})


def set_hand(game: Game, color: Color, hand: Mapping[str, int]) -> None:
    """Overwrite a player's resource hand in place.

    Args:
        game: Game whose state is edited.
        color: Player to edit.
        hand: Count per resource; omitted resources become 0.
    """
    key = player_key(game.state, color)
    for resource in RESOURCES:
        game.state.player_state[f"{key}_{resource}_IN_HAND"] = hand.get(resource, 0)


def post_roll_game(seed: int = 4) -> tuple[Game, Color]:
    """Advance a four-player game to the active player's post-roll phase.

    Args:
        seed: Game seed.

    Returns:
        The game and the active player's colour.
    """
    game = GameFactory([RandomPlayer(c) for c in ROSTER]).create(seed=seed)
    while game.state.is_initial_build_phase:
        game.play_tick()
    color = game.state.current_color()
    game.execute(Action(color, ActionType.ROLL, (2, 3)), validate_action=False)
    return game, color


def seat_after(game: Game, color: Color, offset: int) -> Color:
    """Find the seat a number of places after another in play order.

    Args:
        game: Game holding the seating order.
        color: Reference seat.
        offset: Places to move.

    Returns:
        The colour at that seat.
    """
    order = game.state.colors
    return order[(order.index(color) + offset) % len(order)]


def resources_in_play(game: Game) -> Counter[str]:
    """Count every resource card in hands and the bank.

    Args:
        game: Game to audit.

    Returns:
        Count per resource.
    """
    totals: Counter[str] = Counter(dict(zip(RESOURCES, game.state.resource_freqdeck)))
    for color in game.state.colors:
        totals.update(player_deck_to_array(game.state, color))
    return totals


@pytest.fixture(name="table")
def fixture_table() -> tuple[Game, Color, TradeProtocol]:
    """Provide a post-roll game where every seat can trade wood or ore.

    Returns:
        The game, the active colour and a fresh protocol.
    """
    game, active = post_roll_game()
    for color in game.state.colors:
        set_hand(game, color, {"WOOD": 2, "ORE": 2, "SHEEP": 1})
    return game, active, TradeProtocol()


@pytest.mark.parametrize(("cap", "expected"), [(1, 20), (2, 230), (3, 1170)])
def test_catalog_sizes(cap: int, expected: int) -> None:
    """The offer catalog matches the agreed sizes."""
    offers = enumerate_offers(cap)
    assert len(offers) == expected == len(set(offers))


@pytest.mark.parametrize(
    ("give", "want"),
    [
        ({}, {"ORE": 1}),
        ({"WOOD": 1}, {}),
        ({"WOOD": 1}, {"WOOD": 1}),
        ({"GOLD": 1}, {"ORE": 1}),
    ],
)
def test_forbidden_offers_are_rejected(
    give: Mapping[str, int], want: Mapping[str, int]
) -> None:
    """Gifts, like-for-like trades and unknown resources are refused."""
    with pytest.raises(ValueError):
        TradeOffer.of(give, want)


def test_offers_must_be_canonical() -> None:
    """Out-of-order sides are refused so equal offers always compare equal."""
    with pytest.raises(ValueError):
        TradeOffer(("ORE", "WOOD"), ("SHEEP",))
    assert TradeOffer.of({"ORE": 1, "WOOD": 1}, {"SHEEP": 1}).give == ("WOOD", "ORE")


def test_describe_renders_offers() -> None:
    """Offers render as readable one-liners."""
    assert describe([TradeOffer.of({"WOOD": 2}, {"ORE": 1})]) == ["2 WOOD -> 1 ORE"]


def test_only_active_player_after_rolling_may_propose() -> None:
    """Proposing needs the active seat, post-roll, in the main phase."""
    game = GameFactory([RandomPlayer(c) for c in ROSTER]).create(seed=4)
    protocol = TradeProtocol()
    assert not protocol.can_propose(game.state, game.state.current_color())
    while game.state.is_initial_build_phase:
        game.play_tick()
    active = game.state.current_color()
    assert not protocol.can_propose(game.state, active), "before rolling"
    game.execute(Action(active, ActionType.ROLL, (2, 3)), validate_action=False)
    assert protocol.can_propose(game.state, active)
    assert not protocol.can_propose(game.state, seat_after(game, active, 1))


def test_no_trading_while_placing_free_roads(table: tuple) -> None:
    """Road building pauses trading."""
    game, active, protocol = table
    game.state.is_road_building = True
    assert not protocol.can_propose(game.state, active)


def test_legal_offers_need_the_cards(table: tuple) -> None:
    """Only offers whose give side the proposer holds are legal."""
    game, active, protocol = table
    set_hand(game, active, {"WOOD": 1})
    offers = protocol.legal_offers(game.state, active)
    assert offers and all(o.give == ("WOOD",) for o in offers)


def test_responders_answer_in_play_order(table: tuple) -> None:
    """Every other seat answers, starting with the next one to play."""
    game, active, protocol = table
    protocol.propose(game.state, active, WOOD_FOR_ORE)
    for offset in (1, 2, 3):
        responder = seat_after(game, active, offset)
        assert protocol.decider() is responder
        protocol.respond(game.state, responder, accept=offset != 2)
    assert protocol.decider() is active
    assert protocol.negotiation.acceptors == (
        seat_after(game, active, 1),
        seat_after(game, active, 3),
    )


def test_seats_that_cannot_pay_are_declined(table: tuple) -> None:
    """A responder without the wanted cards is skipped as a refusal."""
    game, active, protocol = table
    broke = seat_after(game, active, 1)
    set_hand(game, broke, {"WOOD": 3})
    protocol.propose(game.state, active, WOOD_FOR_ORE)
    assert protocol.negotiation.responses[broke] is False
    assert protocol.decider() is seat_after(game, active, 2)


def test_accepting_without_the_cards_is_refused(table: tuple) -> None:
    """A seat cannot accept an offer it cannot pay for."""
    game, active, protocol = table
    protocol.propose(game.state, active, WOOD_FOR_ORE)
    responder = protocol.decider()
    set_hand(game, responder, {"WOOD": 3})
    with pytest.raises(ValueError):
        protocol.respond(game.state, responder, accept=True)


def test_answering_out_of_turn_is_refused(table: tuple) -> None:
    """Only the seat due to answer may answer."""
    game, active, protocol = table
    protocol.propose(game.state, active, WOOD_FOR_ORE)
    with pytest.raises(ValueError):
        protocol.respond(game.state, seat_after(game, active, 2), accept=True)


def test_confirm_swaps_exactly_the_offered_cards(table: tuple) -> None:
    """A completed trade moves exactly the offer between the two hands."""
    game, active, protocol = table
    partner = seat_after(game, active, 1)
    before = resources_in_play(game)
    protocol.propose(game.state, active, WOOD_FOR_ORE)
    protocol.respond(game.state, partner, accept=True)
    for offset in (2, 3):
        protocol.respond(game.state, seat_after(game, active, offset), accept=False)
    record = protocol.confirm(game.state, partner)
    assert Counter(player_deck_to_array(game.state, active)) == Counter(
        {"WOOD": 1, "ORE": 3, "SHEEP": 1}
    )
    assert Counter(player_deck_to_array(game.state, partner)) == Counter(
        {"WOOD": 3, "ORE": 1, "SHEEP": 1}
    )
    assert resources_in_play(game) == before
    assert record.partner is partner and protocol.negotiation is None


def test_trade_refreshes_legal_actions(table: tuple) -> None:
    """A trade that completes a city's cost makes the city buildable."""
    game, active, protocol = table
    set_hand(game, active, {"WHEAT": 2, "ORE": 2, "WOOD": 1})
    partner = seat_after(game, active, 1)
    protocol.propose(game.state, active, WOOD_FOR_ORE)
    protocol.respond(game.state, partner, accept=True)
    for offset in (2, 3):
        protocol.respond(game.state, seat_after(game, active, offset), accept=False)
    assert ActionType.BUILD_CITY not in {
        a.action_type for a in game.state.playable_actions
    }
    protocol.confirm(game.state, partner)
    assert ActionType.BUILD_CITY in {a.action_type for a in game.state.playable_actions}


def test_confirming_a_non_acceptor_is_refused(table: tuple) -> None:
    """The proposer may only trade with a seat that accepted."""
    game, active, protocol = table
    protocol.propose(game.state, active, WOOD_FOR_ORE)
    for offset, accept in ((1, True), (2, False), (3, False)):
        protocol.respond(game.state, seat_after(game, active, offset), accept=accept)
    with pytest.raises(ValueError):
        protocol.confirm(game.state, seat_after(game, active, 2))


def test_cancel_closes_without_trading(table: tuple) -> None:
    """The proposer may walk away after seeing who accepted."""
    game, active, protocol = table
    hands = {c: Counter(player_deck_to_array(game.state, c)) for c in game.state.colors}
    protocol.propose(game.state, active, WOOD_FOR_ORE)
    for offset in (1, 2, 3):
        protocol.respond(game.state, seat_after(game, active, offset), accept=True)
    record = protocol.cancel()
    assert record.partner is None and protocol.negotiation is None
    assert hands == {
        c: Counter(player_deck_to_array(game.state, c)) for c in game.state.colors
    }


def test_offer_nobody_accepts_closes_itself(table: tuple) -> None:
    """With no acceptors the offer closes without asking the proposer."""
    game, active, protocol = table
    protocol.propose(game.state, active, WOOD_FOR_ORE)
    for offset in (1, 2, 3):
        protocol.respond(game.state, seat_after(game, active, offset), accept=False)
    assert protocol.negotiation is None and protocol.decider() is None
    assert protocol.history[-1].partner is None


def test_cannot_confirm_before_everyone_answers(table: tuple) -> None:
    """Confirming or cancelling with answers still due is refused."""
    game, active, protocol = table
    protocol.propose(game.state, active, WOOD_FOR_ORE)
    with pytest.raises(ValueError):
        protocol.cancel()


def test_offers_per_turn_are_capped(table: tuple) -> None:
    """After the per-turn limit the active player may not propose again."""
    game, active, _ = table
    protocol = TradeProtocol(TradingRules(max_offers_per_turn=2))
    for _ in range(2):
        protocol.propose(game.state, active, WOOD_FOR_ORE)
        for offset in (1, 2, 3):
            protocol.respond(game.state, seat_after(game, active, offset), accept=False)
    assert protocol.offers_made_this_turn(game.state) == 2
    assert not protocol.can_propose(game.state, active)
    game.execute(Action(active, ActionType.END_TURN, None))
    assert protocol.offers_made_this_turn(game.state) == 0


def test_cannot_open_two_offers_at_once(table: tuple) -> None:
    """A new offer waits until the open one closes."""
    game, active, protocol = table
    protocol.propose(game.state, active, WOOD_FOR_ORE)
    with pytest.raises(ValueError):
        protocol.propose(game.state, active, WOOD_FOR_ORE)


def test_reset_clears_everything(table: tuple) -> None:
    """Resetting forgets the open offer and the history."""
    game, active, protocol = table
    protocol.propose(game.state, active, WOOD_FOR_ORE)
    protocol.reset()
    assert protocol.negotiation is None and protocol.history == []
