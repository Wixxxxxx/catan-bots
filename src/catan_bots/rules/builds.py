"""What a player can build, what it costs, and how far a hand is from it.

Shared by the discard policies, which keep cards towards a build, and by the
trading bots, which trade towards one.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from catanatron import Color
from catanatron.models.decks import (
    CITY_COST_FREQDECK,
    DEVELOPMENT_CARD_COST_FREQDECK,
    ROAD_COST_FREQDECK,
    SETTLEMENT_COST_FREQDECK,
)
from catanatron.models.enums import CITY, RESOURCES, SETTLEMENT
from catanatron.state import State
from catanatron.state_functions import get_player_buildings, player_key


def _freqdeck_to_cost(freqdeck: Sequence[int]) -> dict[str, int]:
    """Convert a catanatron cost freqdeck into a resource-keyed cost.

    Args:
        freqdeck: Counts in `RESOURCES` order, as used by catanatron's decks.

    Returns:
        Mapping of resource name to required count, omitting zero entries.
    """
    return {r: n for r, n in zip(RESOURCES, freqdeck) if n}


@dataclass(frozen=True)
class BuildTarget:
    """Something a player can spend resources on.

    Attributes:
        name: Target name: `"CITY"`, `"SETTLEMENT"`, `"DEVELOPMENT_CARD"` or
            `"ROAD"`.
        cost: Resource cards it costs, per resource.
    """

    name: str
    cost: Mapping[str, int]

    def missing_cards(self, hand: Mapping[str, int]) -> int:
        """Count the cards a hand still lacks to pay for this target.

        Args:
            hand: Count per resource name in the hand.

        Returns:
            Total cards short across all resources; 0 if affordable.
        """
        return sum(max(0, n - hand.get(r, 0)) for r, n in self.cost.items())

    def surplus(self, hand: Mapping[str, int]) -> dict[str, int]:
        """Count the cards a hand holds beyond this target's cost.

        Args:
            hand: Count per resource name in the hand.

        Returns:
            Spare cards per resource, omitting resources with none spare.
        """
        spare = {r: hand.get(r, 0) - self.cost.get(r, 0) for r in RESOURCES}
        return {r: n for r, n in spare.items() if n > 0}


CITY_TARGET = BuildTarget(CITY, _freqdeck_to_cost(CITY_COST_FREQDECK))
SETTLEMENT_TARGET = BuildTarget(SETTLEMENT, _freqdeck_to_cost(SETTLEMENT_COST_FREQDECK))
DEVELOPMENT_CARD_TARGET = BuildTarget(
    "DEVELOPMENT_CARD", _freqdeck_to_cost(DEVELOPMENT_CARD_COST_FREQDECK)
)
ROAD_TARGET = BuildTarget("ROAD", _freqdeck_to_cost(ROAD_COST_FREQDECK))

BUILD_TARGETS: tuple[BuildTarget, ...] = (
    CITY_TARGET,
    SETTLEMENT_TARGET,
    DEVELOPMENT_CARD_TARGET,
    ROAD_TARGET,
)
"""Build targets in descending strategic value."""


def is_reachable(state: State, color: Color, target: BuildTarget) -> bool:
    """Judge whether a player could still build a target later this game.

    A target is unreachable with no pieces left in supply, no settlement to
    upgrade (for a city), or an empty development deck.

    Args:
        state: Game state to read.
        color: Colour of the player.
        target: The build target.

    Returns:
        True if the target is still available to the player.
    """
    key = player_key(state, color)
    if target.name == CITY:
        return bool(state.player_state[f"{key}_CITIES_AVAILABLE"]) and bool(
            get_player_buildings(state, color, SETTLEMENT)
        )
    if target.name == SETTLEMENT:
        return bool(state.player_state[f"{key}_SETTLEMENTS_AVAILABLE"])
    if target.name == "ROAD":
        return bool(state.player_state[f"{key}_ROADS_AVAILABLE"])
    return bool(state.development_listdeck)


def reachable_targets(state: State, color: Color) -> list[BuildTarget]:
    """List the build targets a player could still spend resources on.

    Args:
        state: Game state to read.
        color: Colour of the player.

    Returns:
        Reachable targets, in descending strategic value.
    """
    return [target for target in BUILD_TARGETS if is_reachable(state, color, target)]


def nearest_target(
    state: State, color: Color, hand: Mapping[str, int]
) -> BuildTarget | None:
    """Find the reachable build a hand is closest to paying for.

    Args:
        state: Game state to read.
        color: Colour of the player.
        hand: Count per resource name in the player's hand.

    Returns:
        The reachable target with the fewest missing cards, ties going to the
        more valuable target, or `None` if nothing is reachable.
    """
    targets = reachable_targets(state, color)
    if not targets:
        return None
    return min(targets, key=lambda t: (t.missing_cards(hand), BUILD_TARGETS.index(t)))
