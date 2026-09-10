"""Human-readable snapshots of a game's state and outcome."""

from dataclasses import dataclass

from catanatron import Color, Game
from catanatron.models.enums import CITY, KNIGHT, RESOURCES, ROAD, SETTLEMENT
from catanatron.state import State
from catanatron.state_functions import (
    get_actual_victory_points,
    get_largest_army,
    get_longest_road_color,
    get_longest_road_length,
    get_played_dev_cards,
    get_player_buildings,
    get_visible_victory_points,
    player_key,
)

Coordinate = tuple[int, int, int]


@dataclass(frozen=True)
class PlayerSnapshot:
    """Everything worth reporting about one player at one point in time.

    Attributes:
        color: The player's colour.
        key: The `"P{index}"` prefix this player occupies in `player_state`.
        victory_points: Actual victory points, including hidden VP cards.
        visible_victory_points: Victory points visible to opponents.
        resources: Cards in hand per resource name.
        settlements: Node ids holding one of the player's settlements.
        cities: Node ids holding one of the player's cities.
        roads: Number of road edges the player has built.
        longest_road_length: Longest unbroken road segment, in edges.
        knights_played: Number of knight development cards played.
        has_longest_road: Whether the player holds the Longest Road card.
        has_largest_army: Whether the player holds the Largest Army card.
    """

    color: Color
    key: str
    victory_points: int
    visible_victory_points: int
    resources: dict[str, int]
    settlements: list[int]
    cities: list[int]
    roads: int
    longest_road_length: int
    knights_played: int
    has_longest_road: bool
    has_largest_army: bool

    @property
    def total_resources(self) -> int:
        """Total number of resource cards held."""
        return sum(self.resources.values())


class GameReport:
    """Summarise a game's board, players, and action log.

    The report reads state eagerly on construction, so it stays valid after the
    underlying game advances.

    Attributes:
        snapshots: One `PlayerSnapshot` per seated player, in seating order.
        winner: Winning colour, or `None` if the game is unfinished or hit
            catanatron's turn limit.
        num_turns: Completed turns at the time the report was taken.
        bank: Resource cards left in the bank, per resource name.
        development_cards_remaining: Undrawn development cards in the bank.
        robber_coordinate: Cube coordinate the robber sits on.
        longest_road_holder: Colour holding Longest Road, or `None`.
        largest_army_holder: Colour holding Largest Army, or `None`.
        largest_army_size: Knights played by the Largest Army holder, or 0
            while the card is unclaimed.
        action_log: Every action taken so far, oldest first.
    """

    def __init__(self, game: Game) -> None:
        """Capture a report of a game's current state.

        Args:
            game: Game to read. It is not modified.
        """
        state: State = game.state
        largest_army_holder, largest_army_size = get_largest_army(state)

        self.winner: Color | None = game.winning_color()
        self.num_turns: int = state.num_turns
        self.bank: dict[str, int] = dict(zip(RESOURCES, state.resource_freqdeck))
        self.development_cards_remaining: int = len(state.development_listdeck)
        self.robber_coordinate: Coordinate = state.board.robber_coordinate
        self.longest_road_holder: Color | None = get_longest_road_color(state)
        self.largest_army_holder: Color | None = largest_army_holder
        self.largest_army_size: int = largest_army_size or 0
        self.action_log = list(state.actions)
        self.snapshots: list[PlayerSnapshot] = [
            self._snapshot_player(state, color) for color in state.colors
        ]

    def snapshot_for(self, color: Color) -> PlayerSnapshot:
        """Look up one player's snapshot by colour.

        Args:
            color: Colour of the player to look up.

        Returns:
            That player's `PlayerSnapshot`.

        Raises:
            KeyError: If no player of that colour is in the game.
        """
        for snapshot in self.snapshots:
            if snapshot.color is color:
                return snapshot
        raise KeyError(f"No player with color {color} in this game.")

    def recent_actions(self, limit: int = 10) -> list[str]:
        """Format the tail of the action log.

        Args:
            limit: Maximum number of actions to format, counting back from the
                most recent one.

        Returns:
            One `"COLOR  ACTION_TYPE  value"` line per action, oldest first.
        """
        return [
            f"{action.color.value:8s}  {action.action_type.name:25s}  {action.value}"
            for action in self.action_log[-limit:]
        ]

    def player_table(self) -> str:
        """Render the per-player standings as a fixed-width text table.

        Returns:
            A table with one header row, a rule, and one row per player.
        """
        columns = ("Color", "VP", *RESOURCES, "Road", "Army")
        header = f"{columns[0]:10s}" + "".join(f"{name:>7s}" for name in columns[1:])
        rows = [header, "-" * len(header)]
        for snapshot in self.snapshots:
            cells = [f"{snapshot.color.value:10s}", f"{snapshot.victory_points:>7d}"]
            cells += [f"{snapshot.resources[r]:>7d}" for r in RESOURCES]
            cells += [
                f"{str(snapshot.has_longest_road):>7s}",
                f"{str(snapshot.has_largest_army):>7s}",
            ]
            rows.append("".join(cells))
        return "\n".join(rows)

    def summary(self) -> str:
        """Render the headline outcome of the game.

        Returns:
            A few lines naming the winner, turn count, and award holders.
        """
        winner = self.winner.value if self.winner else "none (unfinished)"
        return "\n".join(
            [
                f"Winner       : {winner}",
                f"Turns        : {self.num_turns}",
                f"Actions      : {len(self.action_log)}",
                f"Longest Road : {self._holder_label(self.longest_road_holder)}",
                f"Largest Army : {self._holder_label(self.largest_army_holder)}"
                f" ({self.largest_army_size} knights)",
            ]
        )

    @staticmethod
    def _holder_label(color: Color | None) -> str:
        """Name an award holder for display.

        Args:
            color: Colour holding the award, or `None` if unclaimed.

        Returns:
            The colour's name, or `"unclaimed"`.
        """
        return color.value if color else "unclaimed"

    @staticmethod
    def _snapshot_player(state: State, color: Color) -> PlayerSnapshot:
        """Read one player's state into an immutable snapshot.

        Args:
            state: Game state to read from.
            color: Colour of the player to snapshot.

        Returns:
            The player's `PlayerSnapshot`.
        """
        key = player_key(state, color)
        return PlayerSnapshot(
            color=color,
            key=key,
            victory_points=get_actual_victory_points(state, color),
            visible_victory_points=get_visible_victory_points(state, color),
            resources={r: state.player_state[f"{key}_{r}_IN_HAND"] for r in RESOURCES},
            settlements=list(get_player_buildings(state, color, SETTLEMENT)),
            cities=list(get_player_buildings(state, color, CITY)),
            roads=len(get_player_buildings(state, color, ROAD)),
            longest_road_length=get_longest_road_length(state, color),
            knights_played=get_played_dev_cards(state, color, KNIGHT),
            has_longest_road=bool(state.player_state[f"{key}_HAS_ROAD"]),
            has_largest_army=bool(state.player_state[f"{key}_HAS_ARMY"]),
        )
