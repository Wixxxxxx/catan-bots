"""Things that can sit in an environment seat and choose actions.

`PolicyAgent` plays a network, `RandomAgent` plays uniformly at random, and
`BotAgent` lets any catanatron bot — including trading bots — play inside the
environment, so baselines can fill seats during training and evaluation
under exactly the same rules as the learner.
"""

from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable

import numpy as np
import torch
from catanatron import Color
from catanatron.models.enums import RESOURCES, ActionPrompt
from catanatron.models.player import Player
from catanatron.players.weighted_random import WeightedRandomPlayer

from catan_bots.bots import GreedySettlerBot, GreedyTraderBot, TradingPlayer
from catan_bots.envs import CatanAECEnv
from catan_bots.envs.actions import ACCEPT_TRADE, CANCEL_TRADE, REJECT_TRADE
from catan_bots.rules.discard import BuildPlanDiscard
from catan_bots.rules.trading import TradeOffer
from catan_bots.training.policy import ActorCritic

Observation = dict[str, np.ndarray]


class Agent(ABC):
    """Chooses an action index for the environment's deciding seat."""

    @abstractmethod
    def act(self, env: CatanAECEnv, agent: str, observation: Observation) -> int:
        """Choose a legal action for the deciding agent.

        Args:
            env: The environment, positioned at `agent`'s decision.
            agent: The deciding agent's name.
            observation: That agent's observation, including its mask.

        Returns:
            A legal action index.
        """

    def reset(self) -> None:
        """Forget per-game state before a new game. Stateless by default."""


class RandomAgent(Agent):
    """Choose uniformly among legal actions."""

    def __init__(self, seed: int | None = None) -> None:
        """Create the agent with its own generator.

        Args:
            seed: Generator seed.
        """
        self.rng = np.random.default_rng(seed)

    def act(self, env: CatanAECEnv, agent: str, observation: Observation) -> int:
        """Pick a legal action uniformly at random.

        Args:
            env: Unused.
            agent: Unused.
            observation: Holds the action mask.

        Returns:
            A legal action index.
        """
        return int(self.rng.choice(np.flatnonzero(observation["action_mask"])))


class PolicyAgent(Agent):
    """Play a trained network.

    Attributes:
        policy: The network.
        deterministic: Take the most likely action instead of sampling.
    """

    def __init__(self, policy: ActorCritic, deterministic: bool = False) -> None:
        """Wrap a network.

        Args:
            policy: The network; switched to evaluation mode.
            deterministic: Take the most likely action instead of sampling.
        """
        self.policy = policy.eval()
        self.deterministic = deterministic

    def act(self, env: CatanAECEnv, agent: str, observation: Observation) -> int:
        """Let the network choose among legal actions.

        Args:
            env: Unused.
            agent: Unused.
            observation: The encoded observation and mask.

        Returns:
            The chosen action index.
        """
        obs = torch.from_numpy(observation["observation"]).unsqueeze(0)
        mask = torch.from_numpy(observation["action_mask"].astype(bool)).unsqueeze(0)
        action, _, _ = self.policy.act(obs, mask, self.deterministic)
        return int(action.item())


class BotAgent(Agent):
    """Let a catanatron bot play an environment seat.

    Engine decisions go through the bot's `decide`. Trading bots also make,
    answer and settle offers through their `TradingPlayer` hooks; other bots
    reject every offer. Discards follow `BuildPlanDiscard`, one card per
    step, since catanatron bots have no discard hook.

    Attributes:
        bot_factory: Builds the bot for a colour.
    """

    def __init__(self, bot_factory: Callable[[Color], Player]) -> None:
        """Wrap a bot type.

        Args:
            bot_factory: Called with a colour to build that seat's bot.
        """
        self.bot_factory = bot_factory
        self._bots: dict[Color, Player] = {}
        self._discard_policy = BuildPlanDiscard()

    def reset(self) -> None:
        """Drop the bots so each game starts with fresh ones."""
        self._bots = {}

    def act(self, env: CatanAECEnv, agent: str, observation: Observation) -> int:
        """Translate the bot's decision into an action index.

        Args:
            env: The environment, at this seat's decision.
            agent: The deciding agent's name.
            observation: Unused; bots read the game directly.

        Returns:
            A legal action index.
        """
        color = env.color_of(agent)
        bot = self._bot(color)
        trades = env.trades
        if trades is not None and trades.negotiation is not None:
            return self._negotiate(env, bot, color)
        if env.game.state.current_prompt is ActionPrompt.DISCARD:
            return self._discard_pick(env, color)
        return self._play(env, bot, color)

    def _bot(self, color: Color) -> Player:
        """Return this seat's bot, building it on first use.

        Args:
            color: The seat's colour.

        Returns:
            The bot.
        """
        if color not in self._bots:
            self._bots[color] = self.bot_factory(color)
        return self._bots[color]

    def _negotiate(self, env: CatanAECEnv, bot: Player, color: Color) -> int:
        """Answer the open offer, or settle it if this seat made it.

        Args:
            env: The environment.
            bot: This seat's bot.
            color: This seat's colour.

        Returns:
            Accept, reject, confirm-with-partner or cancel.
        """
        table = env.table
        negotiation = env.trades.negotiation
        trader = isinstance(bot, TradingPlayer)
        if negotiation.awaiting is color:
            accept_index = table.index((ACCEPT_TRADE,))
            accepts = trader and bot.respond_to_trade(env.game, negotiation)
            if accepts and accept_index in env.legal_indices:
                return accept_index
            return table.index((REJECT_TRADE,))
        partner = bot.choose_trade_partner(env.game, negotiation) if trader else None
        if partner not in negotiation.acceptors:
            return table.index((CANCEL_TRADE,))
        order = env.game.state.colors
        offset = (order.index(partner) - order.index(color)) % len(order)
        return table.confirm_index(offset)

    def _discard_pick(self, env: CatanAECEnv, color: Color) -> int:
        """Pick the next card of a `BuildPlanDiscard` selection.

        The hand does not change until the discard completes, so the plan is
        the same at every pick.

        Args:
            env: The environment, mid-discard.
            color: This seat's colour.

        Returns:
            The discard-pick index for the next planned card.
        """
        planned = Counter(self._discard_policy.resolve(env.game, color))
        planned.subtract(env.pending_discard.chosen)
        resource = next(r for r in RESOURCES if planned[r] > 0)
        return env.table.discard_index(resource)

    def _play(self, env: CatanAECEnv, bot: Player, color: Color) -> int:
        """Make an offer if the bot wants one, otherwise its engine move.

        Args:
            env: The environment.
            bot: This seat's bot.
            color: This seat's colour.

        Returns:
            A proposal index or an engine-action index.
        """
        if isinstance(bot, TradingPlayer):
            offer = bot.propose_trade(env.game, self._fresh_offers(env, color))
            if offer is not None:
                return env.table.propose_index(offer)
        legal = env.legal_engine_actions()
        chosen = bot.decide(env.game, list(legal.values()))
        return next(index for index, action in legal.items() if action == chosen)

    @staticmethod
    def _fresh_offers(env: CatanAECEnv, color: Color) -> list[TradeOffer]:
        """List legal offers this seat has not already made this turn.

        Args:
            env: The environment.
            color: This seat's colour.

        Returns:
            Offers the bot may choose from.
        """
        trades = env.trades
        if trades is None:
            return []
        turn = env.game.state.num_turns
        made = {record.offer for record in trades.history if record.turn == turn}
        return [o for o in trades.legal_offers(env.game.state, color) if o not in made]


BASELINES: dict[str, Callable[[], Agent]] = {
    "random": RandomAgent,
    "weighted_random": lambda: BotAgent(WeightedRandomPlayer),
    "greedy_settler": lambda: BotAgent(GreedySettlerBot),
    "greedy_trader": lambda: BotAgent(GreedyTraderBot),
}
"""Baseline opponents by name, weakest first."""


def make_baseline(name: str) -> Agent:
    """Create a baseline opponent by name.

    Args:
        name: A key of `BASELINES`.

    Returns:
        A fresh agent.

    Raises:
        ValueError: If the name is unknown.
    """
    if name not in BASELINES:
        raise ValueError(f"Unknown baseline {name!r}; choose from {sorted(BASELINES)}.")
    return BASELINES[name]()
