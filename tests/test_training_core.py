"""Tests for the training building blocks: config, policy, rollouts, league, PPO."""

from pathlib import Path

import numpy as np
import pytest
import torch

from catan_bots.envs import CatanAECEnv
from catan_bots.training.agents import (
    BASELINES,
    PolicyAgent,
    RandomAgent,
    make_baseline,
)
from catan_bots.training.config import TrainingConfig
from catan_bots.training.evaluator import EvaluationResult, wilson_interval
from catan_bots.training.league import (
    BASELINE,
    LEARNER,
    SNAPSHOT,
    League,
    LineupSampler,
    OpponentFactory,
    SeatPlan,
)
from catan_bots.training.policy import ActorCritic
from catan_bots.training.ppo import PPOUpdater
from catan_bots.training.rollout import (
    RolloutBatch,
    RolloutCollector,
    SeatTrajectory,
    compute_gae,
)

SELF_PLAY = [SeatPlan(LEARNER)] * 4


@pytest.fixture(name="env")
def fixture_env() -> CatanAECEnv:
    """Provide a 4-player environment with a short turn limit.

    Returns:
        The environment.
    """
    return CatanAECEnv(num_players=4, turn_limit=400)


@pytest.fixture(name="policy")
def fixture_policy(env: CatanAECEnv) -> ActorCritic:
    """Provide an untrained policy sized for the environment.

    Args:
        env: The environment.

    Returns:
        The policy.
    """
    torch.manual_seed(0)
    return ActorCritic(env.encoder.size, env.table.size, (64,)).eval()


def collector(
    env: CatanAECEnv, policy: ActorCritic, **kwargs: float
) -> RolloutCollector:
    """Build a collector with default discounting.

    Args:
        env: The environment.
        policy: The policy.
        **kwargs: Overrides for gamma, gae_lambda or vp_shaping.

    Returns:
        The collector.
    """
    settings = {"gamma": 0.999, "gae_lambda": 0.95, "vp_shaping": 0.0, **kwargs}
    return RolloutCollector(env, policy, OpponentFactory(), **settings)


def test_config_round_trips_through_json(tmp_path: Path) -> None:
    """Saving and loading preserves every setting, tuples included."""
    config = TrainingConfig(num_workers=3, hidden_sizes=(32, 16), baselines=("random",))
    config.save(tmp_path / "config.json")
    assert TrainingConfig.load(tmp_path / "config.json") == config


@pytest.mark.parametrize(
    "overrides",
    [{"num_players": 5}, {"self_play_fraction": 1.5}, {"minibatch_size": 10**9}],
)
def test_config_rejects_impossible_settings(overrides: dict) -> None:
    """Out-of-range settings raise `ValueError`."""
    with pytest.raises(ValueError):
        TrainingConfig().with_overrides(overrides)


def test_config_rejects_unknown_settings() -> None:
    """A misspelt setting is an error, not silently ignored."""
    with pytest.raises(ValueError, match="Unknown settings"):
        TrainingConfig.from_dict({"learning_rat": 0.1})


def test_policy_never_chooses_masked_actions(policy: ActorCritic) -> None:
    """Sampled and deterministic actions are always legal."""
    observations = torch.randn(256, policy.observation_size)
    masks = torch.rand(256, policy.num_actions) < 0.05
    masks[:, 0] = True
    for deterministic in (False, True):
        actions, _, _ = policy.act(observations, masks, deterministic)
        assert masks[torch.arange(256), actions].all()


def test_evaluate_actions_matches_act(policy: ActorCritic) -> None:
    """Re-scoring taken actions reproduces their log-probabilities."""
    observations = torch.randn(32, policy.observation_size)
    masks = torch.ones(32, policy.num_actions, dtype=torch.bool)
    actions, log_probs, values = policy.act(observations, masks)
    new_log_probs, entropy, new_values = policy.evaluate_actions(
        observations, masks, actions
    )
    torch.testing.assert_close(new_log_probs, log_probs)
    torch.testing.assert_close(new_values, values)
    assert (entropy > 0).all()


def test_snapshot_round_trip(policy: ActorCritic) -> None:
    """A rebuilt snapshot computes exactly the same outputs."""
    rebuilt = ActorCritic.from_snapshot(policy.snapshot())
    observations = torch.randn(4, policy.observation_size)
    for original, copy in zip(policy(observations), rebuilt(observations)):
        torch.testing.assert_close(original, copy)


def test_gae_on_a_hand_computed_example() -> None:
    """GAE matches a worked example with terminal reward."""
    advantages, returns = compute_gae([0.0, 1.0], [0.5, 0.2], 0.0, gamma=0.9, lam=0.8)
    delta_1 = 1.0 - 0.2
    delta_0 = 0.9 * 0.2 - 0.5
    np.testing.assert_allclose(
        advantages, [delta_0 + 0.9 * 0.8 * delta_1, delta_1], rtol=1e-6
    )
    np.testing.assert_allclose(returns, advantages + np.array([0.5, 0.2]), rtol=1e-6)


def test_gae_bootstraps_a_truncated_trajectory() -> None:
    """A cut-off trajectory's last move looks ahead to the bootstrap value."""
    advantages, _ = compute_gae([0.0], [0.4], 0.7, gamma=0.9, lam=0.95)
    assert advantages[0] == pytest.approx(0.9 * 0.7 - 0.4)


def test_each_seat_gets_its_own_final_reward(
    env: CatanAECEnv, policy: ActorCritic
) -> None:
    """The winner's last move gets +1, each loser's -1/3, and nothing else is rewarded."""
    seats, summary = collector(env, policy).play(SELF_PLAY, seed=11)
    assert summary.winner_kind == LEARNER
    finals = sorted(t.rewards[-1] for t in seats)
    assert finals == pytest.approx([-1 / 3, -1 / 3, -1 / 3, 1.0])
    assert all(r == 0 for t in seats for r in t.rewards[:-1])
    assert all(t.bootstrap_value == 0 for t in seats)


def test_opponent_seats_are_not_recorded(env: CatanAECEnv, policy: ActorCritic) -> None:
    """Only learner seats produce trajectories."""
    lineup = [SeatPlan(LEARNER), *[SeatPlan(BASELINE, "greedy_settler")] * 3]
    seats, summary = collector(env, policy).play(lineup, seed=3)
    assert len(seats) == 1 and summary.learner_seats == 1
    assert summary.learner_decisions == len(seats[0]) < summary.decisions


def test_truncated_games_bootstrap_instead_of_rewarding(policy: ActorCritic) -> None:
    """At the turn limit, no seat is rewarded and each bootstraps its value."""
    env = CatanAECEnv(num_players=4, turn_limit=12)
    seats, summary = collector(env, policy).play(SELF_PLAY, seed=5)
    assert summary.winner_kind is None
    assert all(r == 0 for t in seats for r in t.rewards)
    assert all(t.bootstrap_value != 0 for t in seats)


def test_shaping_cancels_out_over_a_game(env: CatanAECEnv, policy: ActorCritic) -> None:
    """With gamma = 1, shaping telescopes away: total reward equals the outcome.

    Both games reseed PyTorch first so the policy samples identical moves and
    the games are the same; only the shaping differs.
    """
    torch.manual_seed(7)
    plain, _ = collector(env, policy, gamma=1.0).play(SELF_PLAY, seed=11)
    torch.manual_seed(7)
    shaped, _ = collector(env, policy, gamma=1.0, vp_shaping=1.0).play(
        SELF_PLAY, seed=11
    )
    assert [len(t) for t in plain] == [len(t) for t in shaped], "games diverged"
    for unshaped, with_shaping in zip(plain, shaped):
        assert sum(with_shaping.rewards) == pytest.approx(
            sum(unshaped.rewards), abs=1e-9
        )
        assert any(r != 0 for r in with_shaping.rewards[:-1])


def test_batch_shapes_and_legality(env: CatanAECEnv, policy: ActorCritic) -> None:
    """A collected batch is consistently shaped and every action was legal."""
    sampler = LineupSampler(4, 1.0, 0.0, (), [], np.random.default_rng(0))
    batch, stats = collector(env, policy).collect(
        500, sampler, np.random.default_rng(1)
    )
    assert len(batch) == stats.learner_decisions >= 500
    assert batch.observations.shape == (len(batch), env.encoder.size)
    assert batch.masks[np.arange(len(batch)), batch.actions].all()
    joined = RolloutBatch.concatenate([batch, batch])
    assert len(joined) == 2 * len(batch)


def test_trajectory_credit_before_first_move_is_ignored() -> None:
    """Crediting an empty trajectory is a no-op."""
    trajectory = SeatTrajectory()
    trajectory.credit(5.0)
    assert len(trajectory) == 0 and trajectory.rewards == []


def test_lineups_always_keep_a_learner() -> None:
    """Every sampled lineup has at least one learner seat."""
    sampler = LineupSampler(
        4, 0.0, 0.5, ("random",), ["a.pt"], np.random.default_rng(0)
    )
    for _ in range(500):
        lineup = sampler.sample()
        assert any(plan.is_learner for plan in lineup)
        assert any(not plan.is_learner for plan in lineup)


def test_pure_self_play_setting() -> None:
    """With self_play_fraction = 1, every seat is the learner."""
    sampler = LineupSampler(
        4, 1.0, 0.5, ("random",), ["a.pt"], np.random.default_rng(0)
    )
    assert all(sampler.sample() == SELF_PLAY for _ in range(50))


def test_baselines_fill_seats_until_snapshots_exist() -> None:
    """With no snapshots yet, replaced seats are baselines."""
    sampler = LineupSampler(4, 0.0, 0.0, ("random",), [], np.random.default_rng(0))
    kinds = {plan.kind for _ in range(100) for plan in sampler.sample()}
    assert kinds == {LEARNER, BASELINE}


def test_league_keeps_only_the_newest_snapshots(
    tmp_path: Path, policy: ActorCritic
) -> None:
    """Adding beyond the limit drops the oldest snapshot files."""
    league = League(tmp_path / "league", max_snapshots=2)
    for iteration in (1, 2, 3):
        league.add(policy, iteration)
    assert [p.name for p in league.snapshots] == [
        "snapshot_0000002.pt",
        "snapshot_0000003.pt",
    ]


def test_opponent_factory_builds_each_kind(tmp_path: Path, policy: ActorCritic) -> None:
    """Snapshots become policy agents and names become baselines."""
    path = League(tmp_path, 5).add(policy, 1)
    factory = OpponentFactory()
    assert isinstance(factory.build(SeatPlan(SNAPSHOT, str(path))), PolicyAgent)
    assert isinstance(factory.build(SeatPlan(BASELINE, "random")), RandomAgent)
    with pytest.raises(ValueError):
        factory.build(SeatPlan(LEARNER))


@pytest.mark.parametrize("name", sorted(BASELINES))
def test_every_baseline_plays_legal_moves(name: str) -> None:
    """Each baseline completes a game in the environment using only legal moves."""
    env = CatanAECEnv(num_players=4, turn_limit=300)
    seats = {agent: make_baseline(name) for agent in env.possible_agents}
    env.reset(seed=2)
    for agent in env.agent_iter():
        observation, _, terminated, truncated, _ = env.last()
        if terminated or truncated:
            env.step(None)
            continue
        action = seats[agent].act(env, agent, observation)
        assert observation["action_mask"][action]
        env.step(action)


def test_unknown_baseline_is_rejected() -> None:
    """Asking for a baseline that does not exist raises `ValueError`."""
    with pytest.raises(ValueError):
        make_baseline("alpha_zero")


def test_ppo_update_changes_the_policy(env: CatanAECEnv, policy: ActorCritic) -> None:
    """One update moves the weights and reports finite diagnostics."""
    sampler = LineupSampler(4, 1.0, 0.0, (), [], np.random.default_rng(0))
    batch, _ = collector(env, policy).collect(300, sampler, np.random.default_rng(1))
    config = TrainingConfig(num_workers=1, minibatch_size=128, steps_per_iteration=300)
    before = [p.detach().clone() for p in policy.parameters()]
    metrics = PPOUpdater(policy, config).update(batch, np.random.default_rng(2))
    assert any(not torch.equal(a, b) for a, b in zip(before, policy.parameters()))
    assert all(np.isfinite(v) for v in metrics.values())


def test_wilson_interval_known_values() -> None:
    """The interval matches hand-computed values and handles no games."""
    low, high = wilson_interval(5, 10)
    assert (low, high) == pytest.approx((0.2366, 0.7634), abs=1e-4)
    assert wilson_interval(0, 0) == (0.0, 1.0)
    assert wilson_interval(0, 20)[0] == 0.0


def test_evaluation_result_reports_against_chance() -> None:
    """Results compare the interval with the chance win rate."""
    strong = EvaluationResult("greedy_trader", 400, 200, 0, 0.25)
    weak = EvaluationResult("greedy_trader", 400, 100, 0, 0.25)
    assert strong.beats_chance and not weak.beats_chance
    assert "vs greedy_trader: 0.500" in strong.summary()
    assert strong.to_dict()["ci_low"] > 0.25
