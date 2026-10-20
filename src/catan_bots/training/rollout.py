"""Play games and record each learner seat's moves as its own trajectory.

This is the bookkeeping that makes four-seat self-play correct. Every seat
the current policy plays gets its own trajectory, so:

- Advantages chain each move to that seat's *own* next decision, never to
  another player's.
- Each seat's end-of-game reward (+1 for the winner, -1/(n-1) for each
  loser) lands on that seat's last move. Losers learn that losing is bad.
- A truncated game (turn limit) is a cut-off, not an outcome: the last move
  bootstraps from the value of the position instead of a reward.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from catanatron.state_functions import player_key

from catan_bots.envs import CatanAECEnv
from catan_bots.training.league import LineupSampler, OpponentFactory, SeatPlan
from catan_bots.training.policy import ActorCritic

Observation = dict[str, np.ndarray]


@dataclass
class SeatTrajectory:
    """One learner seat's decisions in one game.

    Attributes:
        observations: Encoded observation at each decision.
        masks: Legality mask at each decision.
        actions: Action taken at each decision.
        log_probs: Log-probability of each action when taken.
        values: Value estimate at each decision.
        rewards: Reward credited to each decision.
        bootstrap_value: Value of the position after the last decision: 0
            for a finished game, the value estimate for a truncated one.
        potential: Shaping potential at the latest decision.
    """

    observations: list[np.ndarray] = field(default_factory=list)
    masks: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    bootstrap_value: float = 0.0
    potential: float = 0.0

    def __len__(self) -> int:
        """Number of decisions recorded."""
        return len(self.actions)

    def record(
        self,
        observation: np.ndarray,
        mask: np.ndarray,
        action: int,
        log_prob: float,
        value: float,
    ) -> None:
        """Append one decision with no reward yet.

        Args:
            observation: Encoded observation.
            mask: Boolean legality mask.
            action: Action taken.
            log_prob: Its log-probability.
            value: Value estimate.
        """
        self.observations.append(observation)
        self.masks.append(mask)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(0.0)

    def credit(self, reward: float) -> None:
        """Add reward to the latest decision; ignored before the first one.

        Args:
            reward: Reward earned since that decision.
        """
        if self.rewards:
            self.rewards[-1] += reward


def compute_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    bootstrap_value: float,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute generalised advantage estimates for one trajectory.

    Args:
        rewards: Reward per decision.
        values: Value estimate per decision.
        bootstrap_value: Value after the last decision.
        gamma: Discount per decision.
        lam: GAE smoothing.

    Returns:
        Advantages and returns (advantage plus value), as `float32` arrays.
    """
    count = len(rewards)
    advantages = np.zeros(count, dtype=np.float32)
    running = 0.0
    for t in reversed(range(count)):
        next_value = bootstrap_value if t == count - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
        running = delta + gamma * lam * running
        advantages[t] = running
    returns = advantages + np.asarray(values, dtype=np.float32)
    return advantages, returns


@dataclass
class RolloutBatch:
    """Learner decisions ready for a PPO update.

    Attributes:
        observations: `(N, obs)` float32.
        masks: `(N, actions)` bool.
        actions: `(N,)` int64.
        log_probs: `(N,)` float32.
        values: `(N,)` float32.
        advantages: `(N,)` float32.
        returns: `(N,)` float32.
    """

    observations: np.ndarray
    masks: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray

    def __len__(self) -> int:
        """Number of decisions in the batch."""
        return len(self.actions)

    @classmethod
    def from_trajectories(
        cls, trajectories: Sequence[SeatTrajectory], gamma: float, lam: float
    ) -> "RolloutBatch":
        """Flatten trajectories, computing advantages per trajectory.

        Args:
            trajectories: Non-empty seat trajectories.
            gamma: Discount per decision.
            lam: GAE smoothing.

        Returns:
            The batch.
        """
        gae = [
            compute_gae(t.rewards, t.values, t.bootstrap_value, gamma, lam)
            for t in trajectories
        ]
        return cls(
            observations=np.stack([o for t in trajectories for o in t.observations]),
            masks=np.stack([m for t in trajectories for m in t.masks]),
            actions=np.array([a for t in trajectories for a in t.actions], np.int64),
            log_probs=np.array(
                [p for t in trajectories for p in t.log_probs], np.float32
            ),
            values=np.array([v for t in trajectories for v in t.values], np.float32),
            advantages=np.concatenate([adv for adv, _ in gae]),
            returns=np.concatenate([ret for _, ret in gae]),
        )

    @classmethod
    def concatenate(cls, batches: Sequence["RolloutBatch"]) -> "RolloutBatch":
        """Join batches from several workers.

        Args:
            batches: Batches to join.

        Returns:
            One batch holding them all, in order.
        """
        return cls(
            observations=np.concatenate([b.observations for b in batches]),
            masks=np.concatenate([b.masks for b in batches]),
            actions=np.concatenate([b.actions for b in batches]),
            log_probs=np.concatenate([b.log_probs for b in batches]),
            values=np.concatenate([b.values for b in batches]),
            advantages=np.concatenate([b.advantages for b in batches]),
            returns=np.concatenate([b.returns for b in batches]),
        )


@dataclass
class GameSummary:
    """What happened in one training game.

    Attributes:
        learner_seats: Seats played by the current policy.
        num_players: Seats in the game.
        winner_kind: `"learner"`, `"snapshot"`, `"baseline"`, or `None` when
            truncated.
        turns: Completed turns.
        decisions: Decisions by every seat.
        learner_decisions: Decisions by learner seats.
        offers: Trade offers made.
        trades: Trades completed.
    """

    learner_seats: int
    num_players: int
    winner_kind: str | None
    turns: int
    decisions: int
    learner_decisions: int
    offers: int
    trades: int

    @property
    def mixed(self) -> bool:
        """Whether any seat was an opponent rather than the learner."""
        return self.learner_seats < self.num_players


@dataclass
class RolloutStats:
    """Totals over a batch of training games.

    Attributes:
        games: Games played.
        decisions: Decisions by every seat.
        learner_decisions: Decisions recorded for training.
        turns: Completed turns, summed.
        truncated: Games cut off by the turn limit.
        offers: Trade offers made.
        trades: Trades completed.
        mixed_games: Games with at least one opponent seat.
        mixed_learner_wins: Of those, games a learner seat won.
        mixed_learner_expected: Of those, the learner wins expected by
            chance (learner seats / players, summed).
    """

    games: int = 0
    decisions: int = 0
    learner_decisions: int = 0
    turns: int = 0
    truncated: int = 0
    offers: int = 0
    trades: int = 0
    mixed_games: int = 0
    mixed_learner_wins: int = 0
    mixed_learner_expected: float = 0.0

    def add(self, game: GameSummary) -> None:
        """Fold one game into the totals.

        Args:
            game: The game's summary.
        """
        self.games += 1
        self.decisions += game.decisions
        self.learner_decisions += game.learner_decisions
        self.turns += game.turns
        self.truncated += game.winner_kind is None
        self.offers += game.offers
        self.trades += game.trades
        if game.mixed:
            self.mixed_games += 1
            self.mixed_learner_wins += game.winner_kind == "learner"
            self.mixed_learner_expected += game.learner_seats / game.num_players

    def merge(self, other: "RolloutStats") -> None:
        """Fold another worker's totals into these.

        Args:
            other: Totals to add.
        """
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))


class RolloutCollector:
    """Play games with the current policy and record its seats' decisions.

    Attributes:
        env: The environment games are played in.
        policy: The current policy, playing every learner seat.
        opponents: Builds agents for snapshot and baseline seats.
        gamma: Discount per decision.
        gae_lambda: GAE smoothing.
        vp_shaping: Potential-based shaping weight on own victory points.
    """

    def __init__(
        self,
        env: CatanAECEnv,
        policy: ActorCritic,
        opponents: OpponentFactory,
        gamma: float,
        gae_lambda: float,
        vp_shaping: float = 0.0,
    ) -> None:
        """Configure the collector.

        Args:
            env: The environment.
            policy: The current policy.
            opponents: Builds non-learner seats.
            gamma: Discount per decision.
            gae_lambda: GAE smoothing.
            vp_shaping: Shaping weight; 0 disables shaping.
        """
        self.env = env
        self.policy = policy
        self.opponents = opponents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.vp_shaping = vp_shaping

    def collect(
        self, min_steps: int, sampler: LineupSampler, rng: np.random.Generator
    ) -> tuple[RolloutBatch, RolloutStats]:
        """Play whole games until at least `min_steps` learner decisions.

        Args:
            min_steps: Learner decisions to gather; the last game is always
                finished, so slightly more may be returned.
            sampler: Draws each game's seating.
            rng: Draws each game's seed.

        Returns:
            The batch and the games' totals.
        """
        trajectories: list[SeatTrajectory] = []
        stats = RolloutStats()
        while stats.learner_decisions < min_steps:
            seats, summary = self.play(sampler.sample(), int(rng.integers(1 << 31)))
            trajectories += [t for t in seats if len(t)]
            stats.add(summary)
        batch = RolloutBatch.from_trajectories(
            trajectories, self.gamma, self.gae_lambda
        )
        return batch, stats

    def play(
        self, lineup: Sequence[SeatPlan], seed: int
    ) -> tuple[list[SeatTrajectory], GameSummary]:
        """Play one game.

        Args:
            lineup: Who sits in each seat, indexed like the agents.
            seed: Game seed.

        Returns:
            Each learner seat's trajectory, and the game's summary.
        """
        env = self.env
        env.reset(seed=seed)
        plans = dict(zip(env.possible_agents, lineup))
        others = {
            a: self.opponents.build(p) for a, p in plans.items() if not p.is_learner
        }
        seats = {a: SeatTrajectory() for a, p in plans.items() if p.is_learner}
        decisions = 0
        for agent in env.agent_iter():
            observation, reward, terminated, truncated, _ = env.last()
            trajectory = seats.get(agent)
            if trajectory is not None:
                trajectory.credit(reward)
            if terminated or truncated:
                if trajectory is not None:
                    self._finish(trajectory, agent, observation, terminated)
                env.step(None)
                continue
            decisions += 1
            if trajectory is None:
                env.step(others[agent].act(env, agent, observation))
            else:
                self._shape(trajectory, agent)
                env.step(self._decide(trajectory, observation))
        return list(seats.values()), self._summary(plans, seats, decisions)

    def _decide(self, trajectory: SeatTrajectory, observation: Observation) -> int:
        """Let the policy choose for a learner seat, recording the decision.

        Args:
            trajectory: The seat's trajectory.
            observation: The seat's observation and mask.

        Returns:
            The chosen action index.
        """
        obs = observation["observation"]
        mask = observation["action_mask"].astype(bool)
        action, log_prob, value = self.policy.act(
            torch.from_numpy(obs).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0)
        )
        chosen = int(action.item())
        trajectory.record(
            obs, mask, chosen, float(log_prob.item()), float(value.item())
        )
        return chosen

    def _finish(
        self,
        trajectory: SeatTrajectory,
        agent: str,
        observation: Observation,
        terminated: bool,
    ) -> None:
        """Close a seat's trajectory when its game ends.

        Args:
            trajectory: The seat's trajectory.
            agent: The seat's agent name.
            observation: The final observation.
            terminated: True if someone won; False if truncated.
        """
        if terminated:
            trajectory.credit(-trajectory.potential)
            trajectory.bootstrap_value = 0.0
            return
        self._shape(trajectory, agent)
        obs = torch.from_numpy(observation["observation"]).unsqueeze(0)
        with torch.no_grad():
            _, value = self.policy(obs)
        trajectory.bootstrap_value = float(value.item())

    def _shape(self, trajectory: SeatTrajectory, agent: str) -> None:
        """Credit potential-based shaping for progress since the last decision.

        Adds `gamma * phi(now) - phi(previous)`, where phi is the shaping
        weight times the seat's own victory points over ten. Shaping of this
        form cannot change which policy is optimal.

        Args:
            trajectory: The seat's trajectory.
            agent: The seat's agent name.
        """
        if not self.vp_shaping:
            return
        state = self.env.game.state
        key = player_key(state, self.env.color_of(agent))
        potential = (
            self.vp_shaping * state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] / 10
        )
        trajectory.credit(self.gamma * potential - trajectory.potential)
        trajectory.potential = potential

    def _summary(
        self,
        plans: dict[str, SeatPlan],
        seats: dict[str, SeatTrajectory],
        decisions: int,
    ) -> GameSummary:
        """Summarise the game just played.

        Args:
            plans: Each agent's seat plan.
            seats: Learner trajectories by agent.
            decisions: Decisions by every seat.

        Returns:
            The game's summary.
        """
        env = self.env
        winner = env.game.winning_color()
        winner_kind = None
        if winner is not None:
            winner_agent = next(a for a in plans if env.color_of(a) is winner)
            winner_kind = plans[winner_agent].kind
        history = env.trades.history if env.trades is not None else []
        return GameSummary(
            learner_seats=len(seats),
            num_players=len(plans),
            winner_kind=winner_kind,
            turns=env.game.state.num_turns,
            decisions=decisions,
            learner_decisions=sum(len(t) for t in seats.values()),
            offers=len(history),
            trades=sum(record.partner is not None for record in history),
        )
