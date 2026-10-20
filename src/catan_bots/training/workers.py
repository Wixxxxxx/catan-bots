"""Worker processes that play games in parallel for training and evaluation.

Each worker process builds its own environment and policy once, at start-up,
then serves tasks: collect training games with the weights it is sent, or
play evaluation games against a baseline. Workers use the "spawn" start
method on every platform — the macOS default, and safe with PyTorch — and
one PyTorch thread each, so N workers use N cores.

The per-process state lives in a module-level variable set by
`init_worker`; that is the standard multiprocessing pattern for state that
must outlive a single task.
"""

import multiprocessing
from dataclasses import dataclass
from multiprocessing.pool import Pool
from typing import Any

import numpy as np
import torch

from catan_bots.envs import CatanAECEnv
from catan_bots.rules.trading import DEFAULT_TRADING_RULES
from catan_bots.training.agents import PolicyAgent, make_baseline
from catan_bots.training.config import TrainingConfig
from catan_bots.training.league import LineupSampler, OpponentFactory
from catan_bots.training.policy import ActorCritic
from catan_bots.training.rollout import RolloutBatch, RolloutCollector, RolloutStats


@dataclass(frozen=True)
class CollectTask:
    """Ask a worker for training games.

    Attributes:
        weights: Current policy snapshot.
        min_steps: Learner decisions to gather.
        seed: Seed for game seeds, seating and sampling.
        snapshots: League snapshot paths opponents may be drawn from.
    """

    weights: dict[str, Any]
    min_steps: int
    seed: int
    snapshots: tuple[str, ...]


@dataclass(frozen=True)
class EvaluateTask:
    """Ask a worker for evaluation games against one baseline.

    Attributes:
        weights: Policy snapshot to evaluate.
        opponent: Baseline filling every other seat.
        games: Games to play.
        first_game: Index of the first game, so seats rotate across tasks.
        seed: Seed for game seeds and sampling.
        deterministic: Play the most likely action instead of sampling.
    """

    weights: dict[str, Any]
    opponent: str
    games: int
    first_game: int
    seed: int
    deterministic: bool = False


@dataclass(frozen=True)
class EvaluateTally:
    """Results of one evaluation task.

    Attributes:
        games: Games played.
        wins: Games the policy won.
        truncated: Games cut off without a winner.
    """

    games: int
    wins: int
    truncated: int


class WorkerState:
    """Everything a worker process builds once and reuses.

    Attributes:
        config: The run's configuration.
        env: The worker's environment.
        policy: A policy whose weights are replaced for every task.
        collector: Plays and records training games.
    """

    def __init__(self, config: TrainingConfig) -> None:
        """Build the worker's environment and policy.

        Args:
            config: The run's configuration.
        """
        self.config = config
        self.env = make_env(config)
        self.policy = ActorCritic(
            self.env.encoder.size, self.env.table.size, config.hidden_sizes
        ).eval()
        self.collector = RolloutCollector(
            self.env,
            self.policy,
            OpponentFactory(max_cached=config.max_snapshots),
            config.gamma,
            config.gae_lambda,
            config.vp_shaping,
        )

    def load(self, weights: dict[str, Any]) -> None:
        """Load the weights sent with a task.

        Args:
            weights: A policy snapshot.
        """
        self.policy.load_state_dict(weights["weights"])
        self.policy.eval()


_STATE: WorkerState | None = None


def make_env(config: TrainingConfig) -> CatanAECEnv:
    """Build the environment a run trains and evaluates in.

    Args:
        config: The run's configuration.

    Returns:
        The environment.
    """
    return CatanAECEnv(
        num_players=config.num_players,
        trading=DEFAULT_TRADING_RULES if config.trading else None,
        turn_limit=config.turn_limit,
    )


def init_worker(config_values: dict[str, Any]) -> None:
    """Set up a freshly started worker process.

    Args:
        config_values: The run's configuration as plain values.
    """
    global _STATE
    torch.set_num_threads(1)
    _STATE = WorkerState(TrainingConfig.from_dict(config_values))


def make_pool(config: TrainingConfig, processes: int) -> Pool:
    """Start a pool of worker processes.

    Args:
        config: The run's configuration.
        processes: Number of workers.

    Returns:
        The pool; close it with a `with` block or `terminate()`.
    """
    context = multiprocessing.get_context("spawn")
    return context.Pool(
        processes, initializer=init_worker, initargs=(config.to_dict(),)
    )


def collect(task: CollectTask) -> tuple[RolloutBatch, RolloutStats]:
    """Play training games in this worker.

    Args:
        task: What to collect.

    Returns:
        The worker's batch and game totals.
    """
    state = _STATE
    state.load(task.weights)
    torch.manual_seed(task.seed)
    rng = np.random.default_rng(task.seed)
    config = state.config
    sampler = LineupSampler(
        config.num_players,
        config.self_play_fraction,
        config.baseline_seat_fraction,
        config.baselines,
        list(task.snapshots),
        rng,
    )
    return state.collector.collect(task.min_steps, sampler, rng)


def evaluate(task: EvaluateTask) -> EvaluateTally:
    """Play evaluation games: the policy in one seat, a baseline in the rest.

    The policy's seat rotates with the game index; catanatron also shuffles
    the seating order every game, so no seat position is favoured.

    Args:
        task: What to evaluate.

    Returns:
        Wins and truncations over the task's games.
    """
    state = _STATE
    state.load(task.weights)
    torch.manual_seed(task.seed)
    rng = np.random.default_rng(task.seed)
    env = state.env
    wins = truncated = 0
    for game in range(task.first_game, task.first_game + task.games):
        policy_agent = env.possible_agents[game % len(env.possible_agents)]
        winner = _play_evaluation_game(env, state.policy, task, policy_agent, rng)
        wins += winner == policy_agent
        truncated += winner is None
    return EvaluateTally(task.games, wins, truncated)


def _play_evaluation_game(
    env: CatanAECEnv,
    policy: ActorCritic,
    task: EvaluateTask,
    policy_agent: str,
    rng: np.random.Generator,
) -> str | None:
    """Play one evaluation game.

    Args:
        env: The worker's environment.
        policy: The policy under evaluation.
        task: The evaluation task.
        policy_agent: The seat the policy plays.
        rng: Draws the game seed.

    Returns:
        The winning agent's name, or `None` if the game was truncated.
    """
    env.reset(seed=int(rng.integers(1 << 31)))
    seats = {
        agent: PolicyAgent(policy, task.deterministic)
        if agent == policy_agent
        else make_baseline(task.opponent)
        for agent in env.possible_agents
    }
    for agent in env.agent_iter():
        observation, _, terminated, truncated, _ = env.last()
        if terminated or truncated:
            env.step(None)
            continue
        env.step(seats[agent].act(env, agent, observation))
    winner = env.game.winning_color()
    if winner is None:
        return None
    return next(a for a in env.possible_agents if env.color_of(a) is winner)
