"""The discard-on-seven decision: how to enumerate it and how to make it.

Catanatron declines to model this decision. `discard_possibilities` returns a
single `Action(color, DISCARD, None)` regardless of hand, and `apply_action`
then discards a uniformly random half (`state.py`, with a TODO explaining the
decision tree would otherwise explode). It only explodes if same-resource cards
are treated as distinguishable: enumerating individual cards gives thousands of
combinations, while enumerating *resource multisets* gives a median of 11 and
never more than ~100 in practice.

`apply_action` does honour an explicit `action.value` — a listdeck of resource
names — and performs no validation on it, so a wrong selection silently
corrupts hands and the bank. Every policy here therefore goes through
`DiscardPolicy.resolve`, which validates before the engine ever sees it.

One engine caveat these policies cannot fix: catanatron decides *who* must
discard against `state.discard_limit` when the seven is rolled, but chains to
the next discarder against a hardcoded `> 7`. The two agree only at the default
limit of 7, so a custom `discard_limit` makes the engine skip discarders.
"""

import random
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import combinations_with_replacement

from catanatron import Color, Game
from catanatron.models.decks import (
    CITY_COST_FREQDECK,
    DEVELOPMENT_CARD_COST_FREQDECK,
    ROAD_COST_FREQDECK,
    SETTLEMENT_COST_FREQDECK,
)
from catanatron.models.enums import CITY, RESOURCES, SETTLEMENT
from catanatron.state import State
from catanatron.state_functions import get_player_buildings, player_deck_to_array

BuildCost = Mapping[str, int]


def _freqdeck_to_cost(freqdeck: Sequence[int]) -> dict[str, int]:
    """Convert a catanatron cost freqdeck into a resource-keyed cost.

    Args:
        freqdeck: Counts in `RESOURCES` order, as used by catanatron's decks.

    Returns:
        Mapping of resource name to required count, omitting zero entries.
    """
    return {r: n for r, n in zip(RESOURCES, freqdeck) if n}


CITY_COST = _freqdeck_to_cost(CITY_COST_FREQDECK)
SETTLEMENT_COST = _freqdeck_to_cost(SETTLEMENT_COST_FREQDECK)
DEVELOPMENT_CARD_COST = _freqdeck_to_cost(DEVELOPMENT_CARD_COST_FREQDECK)
ROAD_COST = _freqdeck_to_cost(ROAD_COST_FREQDECK)


def discard_count(hand_size: int) -> int:
    """Compute how many cards a player must discard.

    Official rule: half the hand, rounded down. Matches catanatron's
    `len(hand) // 2`.

    Args:
        hand_size: Number of resource cards held when the seven was rolled.

    Returns:
        The number of cards that must be discarded.
    """
    return hand_size // 2


def hand_counts(state: State, color: Color) -> dict[str, int]:
    """Read one player's resource hand as per-resource counts.

    Args:
        state: Game state to read.
        color: Colour of the player whose hand is read.

    Returns:
        Count per resource name, including zero entries for every resource.
    """
    held = Counter(player_deck_to_array(state, color))
    return {resource: held.get(resource, 0) for resource in RESOURCES}


def production_rate(state: State, color: Color, resource: str) -> float:
    """Compute a player's expected per-roll production of one resource.

    Measures how easily a discarded card can be replaced: a resource the
    player's own settlements and cities rarely produce is worth holding.

    Args:
        state: Game state to read.
        color: Colour of the player whose production is measured.
        resource: Resource name to measure.

    Returns:
        Expected cards per dice roll, counting cities double.
    """
    production = state.board.map.node_production
    rate = 0.0
    for node_id in get_player_buildings(state, color, SETTLEMENT):
        rate += production.get(node_id, {}).get(resource, 0.0)
    for node_id in get_player_buildings(state, color, CITY):
        rate += 2 * production.get(node_id, {}).get(resource, 0.0)
    return rate


def listdeck_from_counts(counts: Mapping[str, int]) -> tuple[str, ...]:
    """Expand per-resource counts into a canonical listdeck.

    Args:
        counts: Count per resource name; negative counts are treated as zero.

    Returns:
        Resource names repeated by count, ordered by `RESOURCES`, so the same
        counts always produce the same tuple.
    """
    cards: list[str] = []
    for resource in RESOURCES:
        cards.extend([resource] * max(0, counts.get(resource, 0)))
    return tuple(cards)


def enumerate_discards(
    hand: Mapping[str, int], num_to_discard: int
) -> list[tuple[str, ...]]:
    """Enumerate every distinct discard selection available from a hand.

    Same-resource cards are interchangeable, so selections are multisets rather
    than per-card combinations. This is what makes the discard decision small
    enough to expose to a learning agent.

    Args:
        hand: Count per resource name in the player's hand.
        num_to_discard: Number of cards that must be discarded.

    Returns:
        Canonical listdecks, one per distinct selection, in a deterministic
        order. A `num_to_discard` of 0 yields a single empty selection.

    Raises:
        ValueError: If `num_to_discard` is negative or exceeds the hand size.
    """
    hand_size = sum(hand.get(resource, 0) for resource in RESOURCES)
    if not 0 <= num_to_discard <= hand_size:
        raise ValueError(f"Cannot discard {num_to_discard} of {hand_size} cards.")
    options: list[tuple[str, ...]] = []
    for combination in combinations_with_replacement(RESOURCES, num_to_discard):
        selection = Counter(combination)
        if all(count <= hand.get(r, 0) for r, count in selection.items()):
            options.append(listdeck_from_counts(selection))
    return options


class DiscardPolicy(ABC):
    """Decide which cards a player throws away when a seven is rolled.

    Subclasses implement `choose`; callers use `resolve`, which validates the
    chosen cards before they reach catanatron's unvalidated `apply_action`.
    """

    def resolve(self, game: Game, color: Color) -> tuple[str, ...]:
        """Choose and validate the cards a player discards.

        Args:
            game: Game whose current state holds the player's hand.
            color: Colour of the discarding player.

        Returns:
            A canonical listdeck of exactly the required number of cards, each
            one held by the player.

        Raises:
            ValueError: If the policy chose the wrong number of cards, chose
                cards the player does not hold, or named an unknown resource.
        """
        hand = hand_counts(game.state, color)
        num_to_discard = discard_count(sum(hand.values()))
        selection = tuple(self.choose(game, color, hand, num_to_discard))
        self._validate(selection, hand, num_to_discard)
        return selection

    @abstractmethod
    def choose(
        self,
        game: Game,
        color: Color,
        hand: Mapping[str, int],
        num_to_discard: int,
    ) -> Sequence[str]:
        """Select the cards to discard.

        Args:
            game: Read-only view of the game. Do not mutate it.
            color: Colour of the discarding player.
            hand: Count per resource name in that player's hand.
            num_to_discard: Number of cards that must be discarded.

        Returns:
            The resource names to discard, one entry per card.
        """

    @staticmethod
    def _validate(
        selection: Sequence[str], hand: Mapping[str, int], num_to_discard: int
    ) -> None:
        """Check a selection against the hand and the required discard count.

        Args:
            selection: Resource names the policy chose to discard.
            hand: Count per resource name in the player's hand.
            num_to_discard: Number of cards that must be discarded.

        Raises:
            ValueError: If the count is wrong, a resource is unknown, or the
                selection takes more of a resource than the player holds.
        """
        if len(selection) != num_to_discard:
            raise ValueError(
                f"Discard must drop {num_to_discard} cards, got {len(selection)}."
            )
        chosen = Counter(selection)
        unknown = set(chosen) - set(RESOURCES)
        if unknown:
            raise ValueError(f"Unknown resources in discard: {sorted(unknown)}.")
        for resource, count in chosen.items():
            if count > hand.get(resource, 0):
                raise ValueError(
                    f"Discard takes {count} {resource} but hand holds "
                    f"{hand.get(resource, 0)}."
                )


class UniformRandomDiscard(DiscardPolicy):
    """Discard a uniformly random half of the hand.

    Reproduces catanatron's built-in behaviour, but from a private seeded RNG
    instead of the global `random` module. Kept as the ablation baseline any
    smarter policy has to beat.

    Attributes:
        rng: The policy's private random number generator.
    """

    def __init__(self, seed: int | None = None) -> None:
        """Create the policy with its own random number generator.

        Args:
            seed: Seed for the private RNG. Defaults to `None`, i.e.
                non-deterministic.
        """
        self._seed = seed
        self.rng = random.Random(seed)

    def choose(
        self,
        game: Game,
        color: Color,
        hand: Mapping[str, int],
        num_to_discard: int,
    ) -> Sequence[str]:
        """Sample cards to discard uniformly at random, without replacement.

        Args:
            game: Unused.
            color: Unused.
            hand: Count per resource name in the player's hand.
            num_to_discard: Number of cards that must be discarded.

        Returns:
            Randomly chosen resource names, one entry per discarded card.
        """
        return self.rng.sample(listdeck_from_counts(hand), k=num_to_discard)

    def reset(self) -> None:
        """Restore the RNG to its seeded starting point."""
        self.rng = random.Random(self._seed)


class BuildPlanDiscard(DiscardPolicy):
    """Keep the cards that fund the most valuable reachable build.

    The rule, in order:

    1. Walk the build targets this player could still use (city, settlement,
       development card, road) in value order, reserving the cards that fully
       fund each one. Reservations accumulate, so a card is never counted
       towards two builds.
    2. Spend any remaining keep budget on partial progress towards the best
       target that could not be funded.
    3. Fill whatever budget is left with the cards that are hardest to
       replace — the ones this player's own settlements and cities produce
       least often.

    Everything not reserved is discarded. The rule is deterministic: ties break
    on catanatron's `RESOURCES` order, so the same position always discards the
    same cards.

    Attributes:
        TARGETS: Build costs in descending strategic value.
    """

    TARGETS: tuple[tuple[str, BuildCost], ...] = (
        (CITY, CITY_COST),
        (SETTLEMENT, SETTLEMENT_COST),
        ("DEVELOPMENT_CARD", DEVELOPMENT_CARD_COST),
        ("ROAD", ROAD_COST),
    )

    def choose(
        self,
        game: Game,
        color: Color,
        hand: Mapping[str, int],
        num_to_discard: int,
    ) -> Sequence[str]:
        """Discard whatever the build plan does not reserve.

        Args:
            game: Read-only view of the game, used to judge which builds are
                still reachable and which resources are hard to replace.
            color: Colour of the discarding player.
            hand: Count per resource name in the player's hand.
            num_to_discard: Number of cards that must be discarded.

        Returns:
            The unreserved cards, as a canonical listdeck.
        """
        keep_budget = sum(hand.values()) - num_to_discard
        reserved = self._reserve(game.state, color, hand, keep_budget)
        surplus = Counter(hand)
        surplus.subtract(reserved)
        return listdeck_from_counts(surplus)

    def _reserve(
        self, state: State, color: Color, hand: Mapping[str, int], keep_budget: int
    ) -> Counter[str]:
        """Choose which cards to keep, up to the keep budget.

        Args:
            state: Game state used to judge targets and production.
            color: Colour of the discarding player.
            hand: Count per resource name in the player's hand.
            keep_budget: Number of cards the player may keep.

        Returns:
            Count per resource name to keep, totalling `keep_budget`.
        """
        targets = self._reachable_targets(state, color)
        reserved = self._reserve_funded_targets(hand, targets, keep_budget)
        self._add_partial_progress(hand, targets, reserved, keep_budget)
        self._add_hardest_to_replace(state, color, hand, reserved, keep_budget)
        return reserved

    def _reachable_targets(
        self, state: State, color: Color
    ) -> list[tuple[str, BuildCost]]:
        """List the build targets this player could still spend resources on.

        A target the player can never build — no pieces left in supply, no
        settlement to upgrade, an empty development deck — is worth no cards.

        Args:
            state: Game state to read.
            color: Colour of the discarding player.

        Returns:
            Costs of the reachable targets, in descending strategic value.
        """
        return [
            (name, cost)
            for name, cost in self.TARGETS
            if self._is_reachable(state, color, name)
        ]

    @staticmethod
    def _is_reachable(state: State, color: Color, target: str) -> bool:
        """Judge whether one build target is still available to a player.

        Args:
            state: Game state to read.
            color: Colour of the discarding player.
            target: Target name, as used in `TARGETS`.

        Returns:
            True if the player could still build it later this game.
        """
        key = f"P{state.color_to_index[color]}"
        if target == CITY:
            return bool(state.player_state[f"{key}_CITIES_AVAILABLE"]) and bool(
                get_player_buildings(state, color, SETTLEMENT)
            )
        if target == SETTLEMENT:
            return bool(state.player_state[f"{key}_SETTLEMENTS_AVAILABLE"])
        if target == "ROAD":
            return bool(state.player_state[f"{key}_ROADS_AVAILABLE"])
        return bool(state.development_listdeck)

    @staticmethod
    def _reserve_funded_targets(
        hand: Mapping[str, int],
        targets: Sequence[tuple[str, BuildCost]],
        keep_budget: int,
    ) -> Counter[str]:
        """Reserve the cards for every target the hand can fully fund.

        Args:
            hand: Count per resource name in the player's hand.
            targets: Reachable targets in descending strategic value.
            keep_budget: Number of cards the player may keep.

        Returns:
            Count per resource name reserved for fully funded targets.
        """
        reserved: Counter[str] = Counter()
        for _, cost in targets:
            affordable = all(
                reserved[resource] + count <= hand.get(resource, 0)
                for resource, count in cost.items()
            )
            fits = sum(reserved.values()) + sum(cost.values()) <= keep_budget
            if affordable and fits:
                reserved.update(cost)
        return reserved

    @staticmethod
    def _add_partial_progress(
        hand: Mapping[str, int],
        targets: Sequence[tuple[str, BuildCost]],
        reserved: Counter[str],
        keep_budget: int,
    ) -> None:
        """Reserve cards towards the best target that could not be funded.

        Mutates `reserved` in place, stopping as soon as the keep budget is
        exhausted.

        Args:
            hand: Count per resource name in the player's hand.
            targets: Reachable targets in descending strategic value.
            reserved: Reservations so far; extended in place.
            keep_budget: Number of cards the player may keep.
        """
        for _, cost in targets:
            for resource, count in cost.items():
                while (
                    reserved[resource] < count
                    and reserved[resource] < hand.get(resource, 0)
                    and sum(reserved.values()) < keep_budget
                ):
                    reserved[resource] += 1
            if sum(reserved.values()) >= keep_budget:
                return

    @staticmethod
    def _add_hardest_to_replace(
        state: State,
        color: Color,
        hand: Mapping[str, int],
        reserved: Counter[str],
        keep_budget: int,
    ) -> None:
        """Fill the remaining keep budget with the scarcest cards.

        Scarcity is this player's own expected production: a resource their
        settlements and cities rarely produce is the one worth holding.

        Args:
            state: Game state used to compute production rates.
            color: Colour of the discarding player.
            hand: Count per resource name in the player's hand.
            reserved: Reservations so far; extended in place.
            keep_budget: Number of cards the player may keep.
        """
        rates = {
            resource: production_rate(state, color, resource) for resource in RESOURCES
        }
        order = sorted(RESOURCES, key=lambda r: (rates[r], RESOURCES.index(r)))
        for resource in order:
            while (
                reserved[resource] < hand.get(resource, 0)
                and sum(reserved.values()) < keep_budget
            ):
                reserved[resource] += 1


class SequentialDiscard(DiscardPolicy):
    """A discard assembled one card at a time by an outside decider.

    Lets a learning agent make the full discard decision with only five
    options per step — "discard one card of this resource" — repeated until
    half the hand is chosen. Once complete it resolves like any other policy,
    so the result is validated before it reaches the engine.

    Attributes:
        color: Colour of the discarding player.
        hand: The player's hand when the discard began.
        required: Number of cards that must be discarded.
        chosen: Cards picked so far, per resource.
    """

    def __init__(self, color: Color, hand: Mapping[str, int]) -> None:
        """Begin a discard for one player.

        Args:
            color: Colour of the discarding player.
            hand: Count per resource name in that player's hand.
        """
        self.color = color
        self.hand = {resource: hand.get(resource, 0) for resource in RESOURCES}
        self.required = discard_count(sum(self.hand.values()))
        self.chosen: Counter[str] = Counter()

    @property
    def remaining(self) -> int:
        """Cards still to be picked."""
        return self.required - sum(self.chosen.values())

    @property
    def is_complete(self) -> bool:
        """Whether every required card has been picked."""
        return self.remaining == 0

    def choosable(self) -> list[str]:
        """List the resources that may be picked next.

        Returns:
            Resources with an unpicked card left in hand, in `RESOURCES`
            order; empty once the discard is complete.
        """
        if self.is_complete:
            return []
        return [r for r in RESOURCES if self.hand[r] - self.chosen[r] > 0]

    def choose_card(self, resource: str) -> None:
        """Pick one card of a resource to discard.

        Args:
            resource: Resource to discard one card of.

        Raises:
            ValueError: If that resource cannot be picked now.
        """
        if resource not in self.choosable():
            raise ValueError(f"Cannot discard {resource} now.")
        self.chosen[resource] += 1

    def choose(
        self,
        game: Game,
        color: Color,
        hand: Mapping[str, int],
        num_to_discard: int,
    ) -> Sequence[str]:
        """Return the cards picked so far.

        Args:
            game: Unused.
            color: Unused.
            hand: Unused.
            num_to_discard: Unused; `resolve` checks the count.

        Returns:
            The picked cards, as a canonical listdeck.
        """
        return listdeck_from_counts(self.chosen)


class DiscardPolicyRegistry:
    """Decide which discard policy governs each seat.

    Attributes:
        default: Policy used for any colour without an override.
        overrides: Per-colour policies, for comparing rules within one game.
    """

    def __init__(
        self,
        default: DiscardPolicy,
        overrides: Mapping[Color, DiscardPolicy] | None = None,
    ) -> None:
        """Store the default policy and any per-colour overrides.

        Args:
            default: Policy for colours with no override.
            overrides: Policy per colour. Defaults to no overrides.
        """
        self.default = default
        self.overrides: dict[Color, DiscardPolicy] = dict(overrides or {})

    def policy_for(self, color: Color) -> DiscardPolicy:
        """Look up the policy that resolves one seat's discards.

        Args:
            color: Colour of the discarding player.

        Returns:
            That colour's override, or the default policy.
        """
        return self.overrides.get(color, self.default)
