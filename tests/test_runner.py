"""Tests for the game runner that resolves discards through a policy."""

from collections import Counter
from collections.abc import Mapping, Sequence

import pytest
from catanatron import Color, Game
from catanatron.models.enums import RESOURCES, ActionType
from catanatron.models.player import RandomPlayer
from catanatron.state_functions import player_deck_to_array

from catan_bots.analytics import ActionTypeCounter
from catan_bots.games import GameFactory
from catan_bots.rules.discard import (
    BuildPlanDiscard,
    DiscardPolicy,
    DiscardPolicyRegistry,
    UniformRandomDiscard,
)
from catan_bots.runner import GameRunner, seat_registry

ROSTER = (Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE)


class CountingDiscard(DiscardPolicy):
    """Wrap a policy and count how often it is consulted.

    Attributes:
        calls: Number of discards this policy has resolved.
    """

    def __init__(self, inner: DiscardPolicy) -> None:
        """Wrap another policy.

        Args:
            inner: Policy that makes the actual choice.
        """
        self.inner = inner
        self.calls = 0

    def choose(
        self,
        game: Game,
        color: Color,
        hand: Mapping[str, int],
        num_to_discard: int,
    ) -> Sequence[str]:
        """Count the call and delegate.

        Args:
            game: Read-only view of the game.
            color: Colour of the discarding player.
            hand: Count per resource name in that player's hand.
            num_to_discard: Number of cards that must be discarded.

        Returns:
            Whatever the wrapped policy chooses.
        """
        self.calls += 1
        return self.inner.choose(game, color, hand, num_to_discard)


def total_cards_in_play(game: Game) -> Counter[str]:
    """Count every resource card held by players plus the bank.

    Args:
        game: Game to audit.

    Returns:
        Count per resource name across all hands and the bank.
    """
    totals: Counter[str] = Counter()
    for color in game.state.colors:
        totals.update(player_deck_to_array(game.state, color))
    totals.update(dict(zip(RESOURCES, game.state.resource_freqdeck)))
    return totals


@pytest.fixture(name="roster")
def fixture_roster() -> list[RandomPlayer]:
    """Provide four random bots.

    Returns:
        One `RandomPlayer` per colour.
    """
    return [RandomPlayer(color) for color in ROSTER]


def test_runner_plays_a_full_game(roster: list[RandomPlayer]) -> None:
    """A run finishes and records actions."""
    game = GameFactory(roster).create(seed=9)
    finished = GameRunner(BuildPlanDiscard()).play(game)
    assert finished.winning_color() in set(ROSTER) | {None}
    assert finished.state.actions


def test_policy_discards_are_logged_with_values(roster: list[RandomPlayer]) -> None:
    """Discard actions carry the chosen cards, not catanatron's `None`."""
    game = GameRunner(BuildPlanDiscard()).play(GameFactory(roster).create(seed=9))
    discards = [a for a in game.state.actions if a.action_type is ActionType.DISCARD]
    assert discards
    # A discard is only prompted above the 7-card limit, so at least 8 // 2.
    assert all(len(a.value) >= 4 for a in discards)
    assert all(set(a.value) <= set(RESOURCES) for a in discards)


def test_resources_are_conserved_every_ply(roster: list[RandomPlayer]) -> None:
    """Hands plus bank always hold 19 of each resource.

    Catanatron applies an explicit discard without validating it, so a wrong
    selection would silently create or destroy cards.
    """
    game = GameFactory(roster).create(seed=9)
    runner = GameRunner(BuildPlanDiscard())
    while game.winning_color() is None and game.state.num_turns < 400:
        runner.tick(game)
        assert all(total_cards_in_play(game)[r] == 19 for r in RESOURCES)


def test_policy_is_consulted_for_every_discard(roster: list[RandomPlayer]) -> None:
    """Each logged discard corresponds to one policy call."""
    policy = CountingDiscard(BuildPlanDiscard())
    game = GameRunner(policy).play(GameFactory(roster).create(seed=9))
    discards = [a for a in game.state.actions if a.action_type is ActionType.DISCARD]
    assert policy.calls == len(discards) > 0


def test_per_seat_policies_are_routed(roster: list[RandomPlayer]) -> None:
    """Seats with their own policy are served by it."""
    red = CountingDiscard(BuildPlanDiscard())
    others = CountingDiscard(UniformRandomDiscard(seed=0))
    registry = seat_registry({Color.RED: red}, default=others)
    game = GameRunner(registry).play(GameFactory(roster).create(seed=9))
    discards = [a for a in game.state.actions if a.action_type is ActionType.DISCARD]
    red_discards = sum(1 for a in discards if a.color is Color.RED)
    assert red.calls == red_discards
    assert others.calls == len(discards) - red_discards


def test_accumulators_see_the_whole_game(roster: list[RandomPlayer]) -> None:
    """Accumulator hooks fire for policy discards as well as normal plies."""
    counter = ActionTypeCounter()
    game = GameRunner(BuildPlanDiscard()).play(
        GameFactory(roster).create(seed=9), [counter]
    )
    assert counter.total_actions == len(game.state.actions)
    assert counter.counts["DISCARD"] > 0


def test_turn_limit_truncates(roster: list[RandomPlayer]) -> None:
    """A tight turn limit truncates the game without a winner."""
    game = GameRunner(BuildPlanDiscard(), turn_limit=5).play(
        GameFactory(roster).create(seed=9)
    )
    assert game.state.num_turns <= 5
    assert game.winning_color() is None


def test_factory_without_policy_keeps_engine_default(
    roster: list[RandomPlayer],
) -> None:
    """Default factories leave catanatron's random discard in place."""
    factory = GameFactory(roster)
    assert factory.discard_policy is None
    game = factory.play(seed=9)
    discards = [a for a in game.state.actions if a.action_type is ActionType.DISCARD]
    assert discards
    assert all(len(a.value) >= 4 for a in discards)
    assert all(set(a.value) <= set(RESOURCES) for a in discards)


def test_factory_with_policy_is_reproducible(roster: list[RandomPlayer]) -> None:
    """A seeded factory with a deterministic policy replays the same game."""
    factory = GameFactory(roster, discard_policy=BuildPlanDiscard())
    first = factory.play(seed=9)
    second = factory.play(seed=9)
    assert first.winning_color() == second.winning_color()
    assert first.state.num_turns == second.state.num_turns


def test_runner_accepts_a_registry_or_a_policy() -> None:
    """Both a bare policy and a registry configure the runner."""
    policy = BuildPlanDiscard()
    assert GameRunner(policy).discards.policy_for(Color.RED) is policy
    registry = DiscardPolicyRegistry(policy)
    assert GameRunner(registry).discards is registry
