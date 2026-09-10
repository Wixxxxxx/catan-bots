"""Tests for the `GameFactory` game builder."""

import pytest
from catanatron import Color
from catanatron.models.player import Player, RandomPlayer

from catan_bots.games import GameFactory


class CountingPlayer(RandomPlayer):
    """Random player that records how often it has been reset."""

    def __init__(self, color: Color) -> None:
        """Seat the player with a zeroed reset counter.

        Args:
            color: Colour this player plays.
        """
        super().__init__(color)
        self.resets = 0

    def reset_state(self) -> None:
        """Count one reset."""
        self.resets += 1


@pytest.fixture(name="roster")
def fixture_roster() -> list[Player]:
    """Provide a two-player roster of random bots.

    Returns:
        A red and a blue `RandomPlayer`.
    """
    return [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)]


def test_create_seats_every_player(roster: list[Player]) -> None:
    """A created game seats exactly the factory's roster."""
    game = GameFactory(roster).create(seed=1)
    assert set(game.state.colors) == {Color.RED, Color.BLUE}


def test_same_seed_reproduces_game(roster: list[Player]) -> None:
    """Two games from the same factory and seed play out identically."""
    factory = GameFactory(roster)
    first = factory.play(seed=7)
    second = factory.play(seed=7)
    assert first.winning_color() == second.winning_color()
    assert first.state.num_turns == second.state.num_turns


def test_create_resets_players() -> None:
    """Every created game gets players whose state has just been reset."""
    players = [CountingPlayer(Color.RED), CountingPlayer(Color.BLUE)]
    factory = GameFactory(players)
    factory.create(seed=1)
    factory.create(seed=2)
    assert [player.resets for player in players] == [2, 2]


def test_play_runs_game_to_completion(roster: list[Player]) -> None:
    """`play` returns a finished game with a populated action log."""
    game = GameFactory(roster).play(seed=3)
    assert game.winning_color() in {Color.RED, Color.BLUE, None}
    assert game.state.actions


def test_vps_to_win_is_forwarded(roster: list[Player]) -> None:
    """The configured victory-point target reaches the created game."""
    game = GameFactory(roster, vps_to_win=5).create(seed=1)
    assert game.vps_to_win == 5


@pytest.mark.parametrize("size", [0, 5])
def test_invalid_roster_size_is_rejected(size: int) -> None:
    """Rosters outside the 1-4 player range raise `ValueError`."""
    colors = list(Color) * 2
    with pytest.raises(ValueError):
        GameFactory([RandomPlayer(colors[i]) for i in range(size)])
