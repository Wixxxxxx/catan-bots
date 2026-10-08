"""Tests for the discard-on-seven rules and policies."""

from collections.abc import Mapping, Sequence

import pytest
from catanatron import Color, Game
from catanatron.models.enums import RESOURCES
from catanatron.models.player import RandomPlayer

from catan_bots.games import GameFactory
from catan_bots.rules.discard import (
    BuildPlanDiscard,
    DiscardPolicy,
    DiscardPolicyRegistry,
    UniformRandomDiscard,
    discard_count,
    enumerate_discards,
    hand_counts,
    listdeck_from_counts,
    production_rate,
)


def set_hand(game: Game, color: Color, hand: Mapping[str, int]) -> None:
    """Overwrite a player's resource hand in place.

    Args:
        game: Game whose state is edited.
        color: Colour of the player whose hand is replaced.
        hand: Target count per resource name; omitted resources become 0.
    """
    key = f"P{game.state.color_to_index[color]}"
    for resource in RESOURCES:
        game.state.player_state[f"{key}_{resource}_IN_HAND"] = hand.get(resource, 0)


@pytest.fixture(name="game")
def fixture_game() -> Game:
    """Provide a four-player game just past the initial build phase.

    Returns:
        A `Game` in which every seat owns two settlements.
    """
    factory = GameFactory(
        [RandomPlayer(c) for c in (Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE)]
    )
    game = factory.create(seed=42)
    while game.state.is_initial_build_phase:
        game.play_tick()
    return game


class StubDiscard(DiscardPolicy):
    """Policy returning a fixed selection, to exercise validation."""

    def __init__(self, selection: Sequence[str]) -> None:
        """Store the selection to return.

        Args:
            selection: Resource names this policy always "chooses".
        """
        self.selection = selection

    def choose(
        self,
        game: Game,
        color: Color,
        hand: Mapping[str, int],
        num_to_discard: int,
    ) -> Sequence[str]:
        """Return the stored selection regardless of the position.

        Args:
            game: Unused.
            color: Unused.
            hand: Unused.
            num_to_discard: Unused.

        Returns:
            The stored selection.
        """
        return self.selection


@pytest.mark.parametrize(
    ("hand_size", "expected"), [(8, 4), (9, 4), (7, 3), (0, 0), (22, 11)]
)
def test_discard_count_is_half_rounded_down(hand_size: int, expected: int) -> None:
    """The discard count is half the hand, rounded down."""
    assert discard_count(hand_size) == expected


def test_listdeck_is_canonical() -> None:
    """Counts expand to resource order, so equal hands give equal tuples."""
    assert listdeck_from_counts({"ORE": 2, "WOOD": 1}) == ("WOOD", "ORE", "ORE")
    assert listdeck_from_counts({"WOOD": -1}) == ()


def test_enumerate_respects_hand_limits() -> None:
    """No enumerated selection takes more of a resource than is held."""
    hand = {"WOOD": 2, "ORE": 1}
    options = enumerate_discards(hand, 2)
    assert set(options) == {
        ("WOOD", "WOOD"),
        ("WOOD", "ORE"),
    }


def test_enumerate_counts_multisets_not_cards() -> None:
    """A uniform 8-card hand has one option per resource, not 70."""
    options = enumerate_discards({r: 8 for r in RESOURCES}, 1)
    assert len(options) == 5
    assert len(enumerate_discards({r: 8 for r in RESOURCES}, 4)) == 70


def test_enumerate_is_deterministic_and_unique() -> None:
    """Enumeration returns distinct options in a stable order."""
    hand = {"WOOD": 3, "BRICK": 2, "ORE": 1}
    first = enumerate_discards(hand, 3)
    assert first == enumerate_discards(hand, 3)
    assert len(first) == len(set(first))


def test_enumerate_zero_discard_is_single_empty_option() -> None:
    """Discarding nothing is exactly one selection."""
    assert enumerate_discards({"WOOD": 3}, 0) == [()]


@pytest.mark.parametrize("num_to_discard", [-1, 5])
def test_enumerate_rejects_impossible_counts(num_to_discard: int) -> None:
    """Negative or oversized discard counts raise `ValueError`."""
    with pytest.raises(ValueError):
        enumerate_discards({"WOOD": 4}, num_to_discard)


def test_hand_counts_covers_every_resource(game: Game) -> None:
    """A read hand always carries all five resource keys."""
    set_hand(game, Color.RED, {"ORE": 3})
    hand = hand_counts(game.state, Color.RED)
    assert set(hand) == set(RESOURCES)
    assert hand["ORE"] == 3 and hand["WOOD"] == 0


def test_resolve_rejects_wrong_card_count(game: Game) -> None:
    """A policy discarding too few cards is caught before the engine sees it."""
    set_hand(game, Color.RED, {"WOOD": 8})
    with pytest.raises(ValueError, match="must drop 4 cards"):
        StubDiscard(["WOOD"]).resolve(game, Color.RED)


def test_resolve_rejects_cards_not_held(game: Game) -> None:
    """A policy discarding cards the player lacks raises `ValueError`."""
    set_hand(game, Color.RED, {"WOOD": 8})
    with pytest.raises(ValueError, match="hand holds"):
        StubDiscard(["ORE"] * 4).resolve(game, Color.RED)


def test_resolve_rejects_unknown_resources(game: Game) -> None:
    """A policy naming a non-resource raises `ValueError`."""
    set_hand(game, Color.RED, {"WOOD": 8})
    with pytest.raises(ValueError, match="Unknown resources"):
        StubDiscard(["GOLD"] * 4).resolve(game, Color.RED)


def test_uniform_random_discards_only_held_cards(game: Game) -> None:
    """The random baseline discards the right number of cards it holds."""
    set_hand(game, Color.RED, {"WOOD": 5, "ORE": 4})
    selection = UniformRandomDiscard(seed=1).resolve(game, Color.RED)
    assert len(selection) == 4
    assert set(selection) <= {"WOOD", "ORE"}


def test_uniform_random_is_seed_reproducible(game: Game) -> None:
    """Two equally seeded random policies make the same choice."""
    set_hand(game, Color.RED, {"WOOD": 5, "ORE": 4})
    first = UniformRandomDiscard(seed=7).resolve(game, Color.RED)
    second = UniformRandomDiscard(seed=7).resolve(game, Color.RED)
    assert first == second


def test_uniform_random_reset_replays_choices(game: Game) -> None:
    """Resetting the random policy replays its sequence of choices."""
    set_hand(game, Color.RED, {"WOOD": 5, "ORE": 4})
    policy = UniformRandomDiscard(seed=3)
    before = [policy.resolve(game, Color.RED) for _ in range(3)]
    policy.reset()
    assert before == [policy.resolve(game, Color.RED) for _ in range(3)]


def test_build_plan_keeps_a_fundable_city(game: Game) -> None:
    """With a city affordable, the surplus is discarded and the city kept."""
    set_hand(game, Color.RED, {"WHEAT": 2, "ORE": 3, "WOOD": 5})
    selection = BuildPlanDiscard().resolve(game, Color.RED)
    assert selection == ("WOOD",) * 5


def test_build_plan_keeps_progress_when_nothing_is_affordable(game: Game) -> None:
    """Short of a full build, it keeps the cards that progress the best one."""
    set_hand(game, Color.RED, {"WHEAT": 2, "ORE": 2, "WOOD": 4})
    selection = BuildPlanDiscard().resolve(game, Color.RED)
    assert selection == ("WOOD",) * 4


def test_build_plan_discards_exactly_half(game: Game) -> None:
    """Whatever the hand, the policy discards exactly the required count."""
    for hand in ({"WOOD": 9}, {"SHEEP": 5, "BRICK": 6}, {r: 3 for r in RESOURCES}):
        set_hand(game, Color.RED, hand)
        expected = discard_count(sum(hand.values()))
        assert len(BuildPlanDiscard().resolve(game, Color.RED)) == expected


def test_build_plan_is_deterministic(game: Game) -> None:
    """The same position always produces the same discard."""
    set_hand(game, Color.RED, {"WOOD": 4, "BRICK": 3, "SHEEP": 2})
    policy = BuildPlanDiscard()
    assert policy.resolve(game, Color.RED) == policy.resolve(game, Color.RED)


def test_build_plan_keeps_scarcest_production(game: Game) -> None:
    """Surplus beyond any build target keeps what the player produces least."""
    rates = {
        resource: production_rate(game.state, Color.RED, resource)
        for resource in RESOURCES
    }
    scarcest = min(RESOURCES, key=lambda r: (rates[r], RESOURCES.index(r)))
    abundant = max(RESOURCES, key=lambda r: (rates[r], -RESOURCES.index(r)))
    if rates[scarcest] == rates[abundant]:
        pytest.skip("This seed produces every resource at the same rate.")
    set_hand(game, Color.RED, {scarcest: 4, abundant: 4})
    selection = BuildPlanDiscard().resolve(game, Color.RED)
    assert selection.count(abundant) >= selection.count(scarcest)


def test_registry_routes_per_colour() -> None:
    """A registry hands each colour its override and others the default."""
    default = UniformRandomDiscard(seed=0)
    override = BuildPlanDiscard()
    registry = DiscardPolicyRegistry(default, {Color.RED: override})
    assert registry.policy_for(Color.RED) is override
    assert registry.policy_for(Color.BLUE) is default
