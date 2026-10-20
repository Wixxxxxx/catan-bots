"""Reproducible construction of `catanatron` games from a fixed roster."""

import random
from collections.abc import Sequence

from catanatron import Game
from catanatron.models.map import CatanMap
from catanatron.models.player import Player

from catan_bots.rules.discard import DiscardPolicy, DiscardPolicyRegistry
from catan_bots.rules.trading import DEFAULT_TRADING_RULES, TradingRules
from catan_bots.runner import GameRunner


class GameFactory:
    """Create fresh, reproducible games for one roster of players.

    `catanatron.Game` never resets the player objects it is handed, so reusing
    the same roster across games leaks state from stateful bots. This factory
    owns that lifecycle: every created game gets players whose `reset_state()`
    hook has just been called.

    Attributes:
        players: The roster handed to every game this factory creates.
        vps_to_win: Victory points required to win a created game.
        catan_map: Optional fixed map; `None` builds the standard base map.
        discard_policy: Policy resolving discard-on-seven, or `None` to keep
            catanatron's built-in uniformly random discard.
        enforce_dev_card_timing: Whether played games forbid playing a
            development card the turn it was bought (the official rule).
        trading: Limits on domestic trading in played games, or `None` for
            none.
    """

    def __init__(
        self,
        players: Sequence[Player],
        vps_to_win: int = 10,
        catan_map: CatanMap | None = None,
        discard_policy: DiscardPolicy | DiscardPolicyRegistry | None = None,
        enforce_dev_card_timing: bool = True,
        trading: TradingRules | None = DEFAULT_TRADING_RULES,
    ) -> None:
        """Store the roster and rules used for every created game.

        Args:
            players: Players to seat, at most four. Seating order is
                randomised by `Game` itself.
            vps_to_win: Victory points required to win. Defaults to 10.
            catan_map: Map to play on. Defaults to the standard base map.
            discard_policy: Policy (or per-seat registry) deciding which cards
                a player discards on a seven. Defaults to `None`, which leaves
                catanatron's uniformly random discard in place so parity with
                the stock engine stays the baseline.
            enforce_dev_card_timing: Whether to forbid playing a development
                card the turn it was bought. Defaults to True, the official
                rule; False reproduces the stock engine, for ablations.
            trading: Limits on domestic trading, or `None` to switch it off.
                Only bots implementing `TradingPlayer` make or accept offers.

        Raises:
            ValueError: If the roster is empty or holds more than four players.
        """
        if not 1 <= len(players) <= 4:
            raise ValueError(f"Expected 1-4 players, got {len(players)}.")
        self.players: tuple[Player, ...] = tuple(players)
        self.vps_to_win = vps_to_win
        self.catan_map = catan_map
        self.discard_policy = discard_policy
        self.enforce_dev_card_timing = enforce_dev_card_timing
        self.trading = trading

    def create(self, seed: int | None = None) -> Game:
        """Build one un-played game with freshly reset players.

        The game is not bound to this factory's rules: run it with `play`, or
        through a `GameRunner`, rather than `Game.play`, which applies none of
        them.

        Args:
            seed: Random seed for the game. The same seed with the same roster
                reproduces the same game. Defaults to a random seed.

        Returns:
            A `Game` positioned at the start of the initial build phase.
        """
        self._reset_players()
        self._seed_global_rng(seed)
        return Game(
            list(self.players),
            seed=seed,
            vps_to_win=self.vps_to_win,
            catan_map=self.catan_map,
        )

    def play(
        self, seed: int | None = None, accumulators: Sequence | None = None
    ) -> Game:
        """Build a game and run it to completion.

        Always runs through a `GameRunner`, which applies this factory's
        discard policy, development-card timing rule and trading limits.

        Args:
            seed: Random seed for the game. Defaults to a random seed.
            accumulators: `GameAccumulator` hooks to attach to the run.

        Returns:
            The finished `Game`; `game.winning_color()` is `None` if the run hit
            catanatron's turn limit without a winner.
        """
        game = self.create(seed)
        return self.runner().play(game, list(accumulators or []))

    def runner(self) -> GameRunner:
        """Build the runner that applies this factory's rules.

        Returns:
            A `GameRunner` configured with this factory's discard policy,
            development-card timing setting and trading limits.
        """
        return GameRunner(
            discards=self.discard_policy,
            enforce_dev_card_timing=self.enforce_dev_card_timing,
            trading=self.trading,
        )

    @staticmethod
    def _seed_global_rng(seed: int | None) -> None:
        """Make a seeded game reproducible, including for `seed=0`.

        `Game.__init__` resolves its seed with `seed or random.randrange(...)`,
        so the falsy seed 0 silently becomes a random one. Seeding the global
        RNG that `Game` draws from first makes every integer seed — 0
        included — reproduce the same game.

        Args:
            seed: Seed requested by the caller; `None` leaves the global RNG
                untouched so the game is genuinely random.
        """
        if seed is not None:
            random.seed(seed)

    def _reset_players(self) -> None:
        """Call the `reset_state()` hook on every player in the roster."""
        for player in self.players:
            player.reset_state()

    def __repr__(self) -> str:
        """Return a debug representation naming the seated players."""
        roster = ", ".join(repr(player) for player in self.players)
        return f"{type(self).__name__}([{roster}], vps_to_win={self.vps_to_win})"
