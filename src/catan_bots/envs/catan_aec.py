"""A PettingZoo AEC environment for 2-, 3- and 4-player Catan."""

from typing import Any, ClassVar

import numpy as np
from catanatron import Action, ActionType, Color, Game
from catanatron.game import TURNS_LIMIT
from catanatron.models.enums import ActionPrompt
from catanatron.models.map import BASE_MAP_TEMPLATE, CatanMap
from catanatron.models.player import Player
from gymnasium import logger, spaces
from pettingzoo import AECEnv

from catan_bots.analytics.reports import GameReport
from catan_bots.envs.actions import ActionTable
from catan_bots.envs.encoding import MAX_PLAYERS, ObservationEncoder
from catan_bots.envs.random_stream import RandomStream
from catan_bots.envs.topology import BoardTopology
from catan_bots.games import GameFactory
from catan_bots.observation import ObservationBuilder
from catan_bots.rules.dev_cards import DevCardTimingRule
from catan_bots.rules.discard import (
    DiscardPolicy,
    DiscardPolicyRegistry,
    SequentialDiscard,
    hand_counts,
)

SEAT_COLORS = (Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE)


class ExternalPlayer(Player):
    """Seat placeholder for a player whose decisions arrive through the env.

    Catanatron needs a `Player` per seat, but in the environment every
    decision is an agent's `step`. Being asked to decide therefore means
    something bypassed the environment, and raises.
    """

    def decide(self, game: Game, playable_actions: list[Action]) -> Action:
        """Refuse to decide; decisions come from `CatanAECEnv.step`.

        Args:
            game: Unused.
            playable_actions: Unused.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("Environment seats decide through CatanAECEnv.step.")


class CatanAECEnv(AECEnv):
    """Catan as a PettingZoo agent-environment-cycle game.

    **Agents.** One per seat: `"player_0"` to `"player_3"` (the first
    `num_players` of them), playing red, blue, orange and white respectively;
    `color_of` maps an agent to its colour. The number is an identity, not a
    seat position: catanatron randomises the seating order every game. Exactly one agent decides at a time, and the
    decider is always the engine's `state.current_color()` — not a fixed
    rotation, because players over the limit discard out of turn on a seven.

    **Observation.** A dict: `"observation"` is the deciding seat's redacted
    view (see `catan_bots.observation`) encoded as a `float32` vector in
    [0, 1]; `"action_mask"` is an `int8` vector over the action space, all
    zero for any agent not deciding now. No observation contains another
    seat's hand by type, unplayed development cards by type, hidden VP cards
    or the deck's order. Seats are encoded relative to the observer and
    padded to four, so one policy serves every seat and player count.

    **Actions.** `Discrete(n)` over `ActionTable` (351 on the base map),
    always used with the mask. Robbery victims are seat offsets. Playing a
    development card the turn it was bought is never legal when
    `enforce_dev_card_timing` is set.

    **Discarding on a seven.** With no `discard_policy`, the discarding agent
    decides: it is offered five "discard one card of this resource" actions
    and chooses one per step until half its hand is chosen. With a policy,
    discards resolve automatically and agents never see them.

    **Reward.** Zero-sum and terminal only: +1 to the winner and
    `-1 / (num_players - 1)` to every other seat. All-zero on truncation.

    **Termination and truncation.** Terminated when a seat reaches
    `vps_to_win`. Truncated, with zero reward, when `turn_limit` turns pass
    without a winner — a cut-off rather than an outcome, so a learner should
    bootstrap from its value estimate there instead of treating it as a draw.
    That way an agent that is behind gains nothing by stalling.

    **Randomness.** Each environment draws from its own `RandomStream`, so
    environments stepped alternately in one process each reproduce from their
    seeds.

    Attributes:
        num_players: Seats in each game.
        turn_limit: Turns after which a game is truncated.
        table: The action index.
        encoder: The observation encoder.
        game: The game in progress, or `None` before the first reset.
    """

    metadata: ClassVar[dict[str, Any]] = {
        "name": "catan_v0",
        "render_modes": ["ansi", "human"],
        "is_parallelizable": False,
    }

    def __init__(
        self,
        num_players: int = 4,
        discard_policy: DiscardPolicy | DiscardPolicyRegistry | None = None,
        enforce_dev_card_timing: bool = True,
        vps_to_win: int = 10,
        turn_limit: int = TURNS_LIMIT,
        render_mode: str | None = None,
    ) -> None:
        """Configure the environment.

        Args:
            num_players: Seats per game, 2 to 4. Defaults to 4.
            discard_policy: Policy that discards on agents' behalf, or `None`
                (the default) to make discarding an agent decision.
            enforce_dev_card_timing: Whether a development card may not be
                played the turn it was bought. Defaults to True (official).
            vps_to_win: Victory points needed to win.
            turn_limit: Turns after which a game is truncated.
            render_mode: `"ansi"`, `"human"` or `None`.

        Raises:
            ValueError: If `num_players` or `render_mode` is unsupported.
        """
        super().__init__()
        if not 2 <= num_players <= MAX_PLAYERS:
            raise ValueError(f"num_players must be 2-{MAX_PLAYERS}, got {num_players}.")
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(f"Unsupported render_mode {render_mode!r}.")
        self.num_players = num_players
        self.turn_limit = turn_limit
        self.render_mode = render_mode
        colors = SEAT_COLORS[:num_players]
        self.possible_agents = [f"player_{index}" for index in range(num_players)]
        self._color_of = dict(zip(self.possible_agents, colors))
        self._agent_of = {color: agent for agent, color in self._color_of.items()}
        self._discards = self._as_registry(discard_policy)
        self._dev_card_timing = DevCardTimingRule() if enforce_dev_card_timing else None
        self._random = RandomStream()
        self._factory = GameFactory(
            [ExternalPlayer(color) for color in colors], vps_to_win=vps_to_win
        )
        with self._random.active():
            topology = BoardTopology.from_map(CatanMap.from_template(BASE_MAP_TEMPLATE))
        self.table = ActionTable(topology, MAX_PLAYERS)
        self.encoder = ObservationEncoder(topology)
        self._builder = ObservationBuilder()
        self._action_space = spaces.Discrete(self.table.size)
        self._observation_space = spaces.Dict(
            {
                "observation": self.encoder.space(),
                "action_mask": spaces.Box(
                    0, 1, shape=(self.table.size,), dtype=np.int8
                ),
            }
        )
        self.game: Game | None = None
        self._legal_actions: dict[int, Action] = {}
        self._legal_indices: frozenset[int] = frozenset()
        self._pending_discard: SequentialDiscard | None = None

    def observation_space(self, agent: str) -> spaces.Dict:
        """Return the observation space, identical for every agent.

        Args:
            agent: Agent name.

        Returns:
            A `Dict` space with `"observation"` and `"action_mask"` entries.
        """
        return self._observation_space

    def action_space(self, agent: str) -> spaces.Discrete:
        """Return the action space, identical for every agent.

        Args:
            agent: Agent name.

        Returns:
            `Discrete` over the action table.
        """
        return self._action_space

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        """Start a new game.

        Args:
            seed: Game seed. The same seed and the same actions reproduce the
                same game. Defaults to this environment's own stream.
            options: Unused; accepted for API compatibility.
        """
        self.agents = list(self.possible_agents)
        self.rewards = dict.fromkeys(self.agents, 0.0)
        self._cumulative_rewards = dict.fromkeys(self.agents, 0.0)
        self.terminations = dict.fromkeys(self.agents, False)
        self.truncations = dict.fromkeys(self.agents, False)
        self.infos = {agent: {} for agent in self.agents}
        self._skip_agent_selection = None
        self._pending_discard = None
        with self._random.active():
            self.game = self._factory.create(seed)
        self._auto_resolve_discards()
        self._load_decision()

    def step(self, action: int | None) -> None:
        """Apply the deciding agent's action and advance to the next decision.

        Args:
            action: A legal index for the deciding agent, or `None` for an
                agent whose game has ended.

        Raises:
            ValueError: If the action is not legal for the deciding agent.
        """
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return
        self._cumulative_rewards[agent] = 0.0
        self._clear_rewards()
        self._apply(int(action))
        self._auto_resolve_discards()
        self._settle_outcome()
        self._accumulate_rewards()
        self._load_decision()

    def observe(self, agent: str) -> dict[str, np.ndarray]:
        """Build one agent's observation.

        Args:
            agent: Agent name.

        Returns:
            The encoded redacted view and the agent's action mask.
        """
        color = self._color_of[agent]
        observation = self._builder.build(self.game.state, color)
        discard = self._pending_discard
        own_discard = (
            discard if discard is not None and discard.color is color else None
        )
        deciding = agent == self.agent_selection and not self._is_over()
        return {
            "observation": self.encoder.encode(observation, own_discard),
            "action_mask": self.table.mask(self._legal_indices if deciding else ()),
        }

    def render(self) -> str | None:
        """Render the standings.

        Returns:
            The standings as text in `"ansi"` mode; `None` otherwise (they are
            printed in `"human"` mode).
        """
        if self.render_mode is None:
            logger.warn("render() called without a render_mode.")
            return None
        report = GameReport(self.game)
        text = f"{report.summary()}\n\n{report.player_table()}"
        if self.render_mode == "human":
            print(text)
            return None
        return text

    def close(self) -> None:
        """Release resources; the environment holds none."""

    def color_of(self, agent: str) -> Color:
        """Look up the colour an agent plays.

        Args:
            agent: Agent name.

        Returns:
            The agent's colour.
        """
        return self._color_of[agent]

    def _apply(self, index: int) -> None:
        """Carry out one legal action index.

        Args:
            index: Index chosen by the deciding agent.

        Raises:
            ValueError: If the index is not legal right now.
        """
        if index not in self._legal_indices:
            raise ValueError(f"Action {index} is not legal for {self.agent_selection}.")
        resource = self.table.discard_resource(index)
        if resource is not None:
            self._pick_discard(resource)
            return
        with self._random.active():
            self.game.execute(self._legal_actions[index])

    def _pick_discard(self, resource: str) -> None:
        """Add one card to the deciding agent's discard, executing it when full.

        Args:
            resource: Resource of the card picked.
        """
        discard = self._pending_discard
        discard.choose_card(resource)
        if discard.is_complete:
            self._execute_discard(discard, discard.color)

    def _execute_discard(self, policy: DiscardPolicy, color: Color) -> None:
        """Resolve and execute one player's discard.

        Args:
            policy: Policy choosing the cards; validated by `resolve`.
            color: Colour of the discarding player.
        """
        selection = policy.resolve(self.game, color)
        action = Action(color, ActionType.DISCARD, selection)
        with self._random.active():
            self.game.execute(action, validate_action=False)
        self._pending_discard = None

    def _auto_resolve_discards(self) -> None:
        """Resolve pending discards through the discard policy, if one is set."""
        while (
            self._discards is not None
            and not self._is_over()
            and self.game.state.current_prompt is ActionPrompt.DISCARD
        ):
            color = self.game.state.current_color()
            self._execute_discard(self._discards.policy_for(color), color)

    def _settle_outcome(self) -> None:
        """Mark termination or truncation and assign terminal rewards."""
        winner = self.game.winning_color()
        if winner is not None:
            for agent in self.agents:
                self.terminations[agent] = True
                self.rewards[agent] = self._terminal_reward(agent, winner)
        elif self.game.state.num_turns >= self.turn_limit:
            for agent in self.agents:
                self.truncations[agent] = True

    def _terminal_reward(self, agent: str, winner: Color) -> float:
        """Compute one agent's zero-sum terminal reward.

        Args:
            agent: Agent name.
            winner: Winning colour.

        Returns:
            +1 for the winner, `-1 / (num_players - 1)` for everyone else.
        """
        if self._color_of[agent] is winner:
            return 1.0
        return -1.0 / (self.num_players - 1)

    def _load_decision(self) -> None:
        """Point `agent_selection` at the decider and cache its legal actions."""
        self._legal_actions, self._legal_indices = {}, frozenset()
        if self._is_over():
            return
        state = self.game.state
        color = state.current_color()
        self.agent_selection = self._agent_of[color]
        if state.current_prompt is ActionPrompt.DISCARD:
            self._load_discard_picks(color)
            return
        self._legal_actions = self.table.legal_by_index(state, self._engine_legal())
        self._legal_indices = frozenset(self._legal_actions)

    def _load_discard_picks(self, color: Color) -> None:
        """Offer the discarding agent its card-by-card discard choices.

        Args:
            color: Colour of the discarding player.
        """
        if self._pending_discard is None or self._pending_discard.color is not color:
            self._pending_discard = SequentialDiscard(
                color, hand_counts(self.game.state, color)
            )
        self._legal_indices = frozenset(
            self.table.discard_index(r) for r in self._pending_discard.choosable()
        )

    def _engine_legal(self) -> list[Action]:
        """List the engine actions legal under the configured rules.

        Returns:
            The playable actions, minus same-turn development-card plays when
            the timing rule is enforced.
        """
        state = self.game.state
        if self._dev_card_timing is None:
            return list(state.playable_actions)
        return self._dev_card_timing.legal_actions(state)

    def _is_over(self) -> bool:
        """Check whether the game has been won or truncated.

        Returns:
            True once there is a winner or the turn limit is reached.
        """
        state = self.game.state
        return (
            self.game.winning_color() is not None or state.num_turns >= self.turn_limit
        )

    @staticmethod
    def _as_registry(
        discards: DiscardPolicy | DiscardPolicyRegistry | None,
    ) -> DiscardPolicyRegistry | None:
        """Normalise the discard configuration to a registry.

        Args:
            discards: A registry, a single policy, or `None`.

        Returns:
            The registry, a registry wrapping the single policy, or `None`.
        """
        if discards is None or isinstance(discards, DiscardPolicyRegistry):
            return discards
        return DiscardPolicyRegistry(discards)
