"""Tests for trading bots and trading inside `GameRunner`."""

from collections import Counter
from collections.abc import Mapping

import pytest
from catanatron import Action, ActionType, Color, Game
from catanatron.models.enums import RESOURCES
from catanatron.models.player import RandomPlayer
from catanatron.state_functions import player_deck_to_array, player_key

from catan_bots.bots import GreedySettlerBot, GreedyTraderBot, TradingPlayer
from catan_bots.games import GameFactory
from catan_bots.rules.dev_cards import DevCardTimingRule
from catan_bots.rules.trading import (
    TradeNegotiation,
    TradeOffer,
    TradeProtocol,
    TradingRules,
)
from catan_bots.runner import GameRunner

ROSTER = (Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE)


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


def set_points(game: Game, color: Color, points: int) -> None:
    """Overwrite a player's visible victory points.

    Args:
        game: Game whose state is edited.
        color: Player to edit.
        points: Visible points to set.
    """
    game.state.player_state[f"{player_key(game.state, color)}_VICTORY_POINTS"] = points


@pytest.fixture(name="game")
def fixture_game() -> Game:
    """Provide a four-player game just past the opening, after a roll.

    Returns:
        The game; every seat owns two settlements.
    """
    game = GameFactory([RandomPlayer(c) for c in ROSTER]).create(seed=4)
    while game.state.is_initial_build_phase:
        game.play_tick()
    active = game.state.current_color()
    game.execute(Action(active, ActionType.ROLL, (2, 3)), validate_action=False)
    return game


def negotiation_from(
    game: Game, proposer: Color, offer: TradeOffer
) -> TradeNegotiation:
    """Build an open negotiation without running the protocol.

    Args:
        game: Game holding the seating order.
        proposer: Colour making the offer.
        offer: The offer.

    Returns:
        A negotiation with no answers yet.
    """
    order = game.state.colors
    start = order.index(proposer)
    responders = tuple(order[(start + i) % len(order)] for i in range(1, len(order)))
    return TradeNegotiation(game.state.num_turns, proposer, offer, responders)


def test_proposes_spare_card_for_missing_card(game: Game) -> None:
    """With a city short one ore, the bot offers a spare card for ore."""
    bot = GreedyTraderBot(Color.RED)
    set_hand(game, Color.RED, {"WHEAT": 2, "ORE": 2, "WOOD": 3})
    offers = TradeProtocol().catalog
    assert bot.propose_trade(game, offers) == TradeOffer.of({"WOOD": 1}, {"ORE": 1})


def test_proposes_nothing_when_a_build_is_affordable(game: Game) -> None:
    """A bot that can already build has no reason to trade."""
    bot = GreedyTraderBot(Color.RED)
    set_hand(game, Color.RED, {"WHEAT": 2, "ORE": 3, "WOOD": 3})
    assert bot.propose_trade(game, TradeProtocol().catalog) is None


def test_only_proposes_offers_it_is_allowed(game: Game) -> None:
    """If its preferred offer is not on the list, the bot proposes nothing."""
    bot = GreedyTraderBot(Color.RED)
    set_hand(game, Color.RED, {"WHEAT": 2, "ORE": 2, "WOOD": 3})
    assert bot.propose_trade(game, []) is None


def test_accepts_a_trade_that_brings_a_build_closer(game: Game) -> None:
    """Giving a spare wood for a needed ore is accepted."""
    bot = GreedyTraderBot(Color.RED)
    set_hand(game, Color.RED, {"WHEAT": 2, "ORE": 2, "WOOD": 3})
    proposer = next(c for c in game.state.colors if c is not Color.RED)
    offer = TradeOffer.of({"ORE": 1}, {"WOOD": 1})
    assert bot.respond_to_trade(game, negotiation_from(game, proposer, offer))


def test_rejects_a_trade_that_does_not_help(game: Game) -> None:
    """Giving away a needed ore for a spare wood is refused."""
    bot = GreedyTraderBot(Color.RED)
    set_hand(game, Color.RED, {"WHEAT": 2, "ORE": 2, "WOOD": 3})
    proposer = next(c for c in game.state.colors if c is not Color.RED)
    offer = TradeOffer.of({"WOOD": 1}, {"ORE": 1})
    assert not bot.respond_to_trade(game, negotiation_from(game, proposer, offer))


def test_refuses_to_feed_a_seat_close_to_winning(game: Game) -> None:
    """A helpful trade is still refused from a seat two points from winning."""
    bot = GreedyTraderBot(Color.RED)
    set_hand(game, Color.RED, {"WHEAT": 2, "ORE": 2, "WOOD": 3})
    proposer = next(c for c in game.state.colors if c is not Color.RED)
    set_points(game, proposer, game.vps_to_win - 2)
    offer = TradeOffer.of({"ORE": 1}, {"WOOD": 1})
    assert not bot.respond_to_trade(game, negotiation_from(game, proposer, offer))


def test_settles_with_the_trailing_acceptor(game: Game) -> None:
    """Among acceptors, the bot trades with the one showing fewest points."""
    bot = GreedyTraderBot(Color.RED)
    others = [c for c in game.state.colors if c is not Color.RED]
    for points, color in zip((5, 2, 4), others):
        set_points(game, color, points)
    negotiation = negotiation_from(
        game, Color.RED, TradeOffer.of({"WOOD": 1}, {"ORE": 1})
    )
    negotiation.responses.update(dict.fromkeys(others, True))
    assert bot.choose_trade_partner(game, negotiation) is others[1]


class RecordingTrader(GreedyTraderBot):
    """Trading bot that records when it is asked to answer.

    Attributes:
        asked: Offers this bot was asked to answer.
    """

    def __init__(self, color: Color, seed: int | None = None) -> None:
        """Seat the bot with an empty record.

        Args:
            color: Colour this bot plays.
            seed: Seed for the bot's tie-break RNG.
        """
        super().__init__(color, seed=seed)
        self.asked: list[TradeOffer] = []

    def respond_to_trade(self, game: Game, negotiation: TradeNegotiation) -> bool:
        """Check the bot can pay, record the offer, then answer as usual.

        Args:
            game: The game.
            negotiation: The open offer.

        Returns:
            The usual answer.
        """
        hand = Counter(player_deck_to_array(game.state, self.color))
        assert all(hand[r] >= n for r, n in negotiation.offer.want_counts.items())
        self.asked.append(negotiation.offer)
        return super().respond_to_trade(game, negotiation)


def trader_roster() -> list:
    """Seat two recording traders and two non-trading bots.

    Returns:
        Players for red, blue, orange and white.
    """
    return [
        RecordingTrader(Color.RED, seed=1),
        GreedySettlerBot(Color.BLUE, seed=2),
        RecordingTrader(Color.ORANGE, seed=3),
        GreedySettlerBot(Color.WHITE, seed=4),
    ]


def play_with_trades(seed: int, rules: TradingRules | None = TradingRules()) -> tuple:
    """Play one game of the trader roster through a runner.

    Args:
        seed: Game seed.
        rules: Trading limits, or `None` to switch trading off.

    Returns:
        The finished game and the runner, whose protocol holds the history.
    """
    factory = GameFactory(trader_roster(), trading=rules)
    runner = factory.runner()
    return runner.play(factory.create(seed=seed)), runner


def test_traders_trade_only_with_each_other() -> None:
    """Non-trading bots decline everything, so trades pair the traders."""
    traders = {Color.RED, Color.ORANGE}
    completed = 0
    for seed in range(1, 4):
        _, runner = play_with_trades(seed)
        for record in runner.trades.history:
            assert record.proposer in traders
            if record.partner is not None:
                completed += 1
                assert record.partner in traders
            for color, accepted in record.responses.items():
                if color not in traders:
                    assert not accepted
    assert completed > 0


def test_responders_are_only_asked_offers_they_can_pay() -> None:
    """A trading bot is never asked about an offer it could not pay for."""
    game, _ = play_with_trades(1)
    asked = sum(
        len(p.asked) for p in game.state.players if isinstance(p, RecordingTrader)
    )
    assert asked > 0


def test_offers_respect_the_per_turn_limit_and_never_repeat() -> None:
    """At most three offers per turn, and never the same offer twice."""
    for seed in range(1, 4):
        _, runner = play_with_trades(seed)
        per_turn = Counter(record.turn for record in runner.trades.history)
        assert max(per_turn.values()) <= 3
        made = [(record.turn, record.offer) for record in runner.trades.history]
        assert len(made) == len(set(made))


def test_resources_are_conserved_with_trading() -> None:
    """Hands plus bank hold 19 of each resource on every ply."""
    factory = GameFactory(trader_roster())
    runner = factory.runner()
    game = factory.create(seed=2)
    runner.trades.reset()
    while game.winning_color() is None and game.state.num_turns < 400:
        runner.tick(game)
        totals = Counter(dict(zip(RESOURCES, game.state.resource_freqdeck)))
        for color in game.state.colors:
            totals.update(player_deck_to_array(game.state, color))
        assert all(totals[r] == 19 for r in RESOURCES)


def test_trades_never_reopen_same_turn_dev_card_plays() -> None:
    """Refreshing legal actions after a trade keeps the timing rule in force."""
    for seed in range(1, 6):
        game, _ = play_with_trades(seed)
        assert DevCardTimingRule.count_violations(game.state.actions) == 0


def test_trading_off_means_no_offers() -> None:
    """With trading switched off the runner holds no protocol."""
    game, runner = play_with_trades(1, rules=None)
    assert runner.trades is None
    assert game.state.actions


def test_history_resets_between_games() -> None:
    """Each game starts with an empty trade history."""
    factory = GameFactory(trader_roster())
    runner = factory.runner()
    runner.play(factory.create(seed=1))
    first = len(runner.trades.history)
    runner.play(factory.create(seed=1))
    assert len(runner.trades.history) == first > 0


def test_trading_player_is_an_interface() -> None:
    """The interface cannot be used without implementing its hooks."""
    with pytest.raises(TypeError):
        TradingPlayer()


def test_runner_reports_its_rules() -> None:
    """The runner's representation names every active rule."""
    assert "trading=True" in repr(GameRunner())
    assert "trading=False" in repr(GameRunner(trading=None))


def test_traders_sequence_is_reproducible() -> None:
    """The same seed replays the same offers."""
    _, first = play_with_trades(3)
    _, second = play_with_trades(3)
    assert [r.offer for r in first.trades.history] == [
        r.offer for r in second.trades.history
    ]
