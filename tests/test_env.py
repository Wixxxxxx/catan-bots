"""Tests for the PettingZoo Catan environment."""

from collections import Counter
from collections.abc import Callable

import numpy as np
import pytest
from catanatron import ActionType, Color
from catanatron.models.enums import RESOURCES, ActionPrompt
from catanatron.state_functions import player_deck_to_array, player_key
from pettingzoo.test import api_test, seed_test

from catan_bots.envs import CatanAECEnv, env, raw_env
from catan_bots.envs.actions import DISCARD_CARD
from catan_bots.envs.catan_aec import ExternalPlayer
from catan_bots.envs.encoding import SEAT_FEATURES
from catan_bots.rules.dev_cards import DevCardTimingRule
from catan_bots.rules.discard import BuildPlanDiscard

StepHook = Callable[[CatanAECEnv, str, dict], None]


def play_random(
    environment: CatanAECEnv,
    seed: int,
    rng: np.random.Generator,
    before_step: StepHook | None = None,
    max_steps: int = 20_000,
) -> dict[str, float]:
    """Play one game with masked uniformly random agents.

    Args:
        environment: Unwrapped environment to play in.
        seed: Game seed.
        rng: Generator choosing among legal actions.
        before_step: Optional hook called with the environment, the deciding
            agent and its observation before each live step.
        max_steps: Safety cap on steps.

    Returns:
        Each agent's reward as reported by `last()` when its game ended.
    """
    environment.reset(seed=seed)
    final: dict[str, float] = {}
    for steps, agent in enumerate(environment.agent_iter()):
        observation, reward, terminated, truncated, _ = environment.last()
        if terminated or truncated:
            final[agent] = reward
            environment.step(None)
            continue
        if before_step is not None:
            before_step(environment, agent, observation)
        legal = np.flatnonzero(observation["action_mask"])
        environment.step(int(rng.choice(legal)))
        assert steps < max_steps, "game did not finish"
    return final


def resources_in_play(environment: CatanAECEnv) -> Counter[str]:
    """Count every resource card in hands and the bank.

    Args:
        environment: Environment with a game in progress.

    Returns:
        Count per resource; 19 each when nothing is created or destroyed.
    """
    state = environment.game.state
    totals: Counter[str] = Counter(dict(zip(RESOURCES, state.resource_freqdeck)))
    for color in state.colors:
        totals.update(player_deck_to_array(state, color))
    return totals


# Masked environments return a dict observation holding the action mask,
# PettingZoo's own convention (its chess environment does the same); the API
# test warns about any non-Box observation regardless.
@pytest.mark.filterwarnings("ignore:Observation space for each agent:UserWarning")
@pytest.mark.filterwarnings("ignore:Observation is not a NumPy array:UserWarning")
@pytest.mark.parametrize("num_players", [2, 3, 4])
def test_pettingzoo_api_conformance(num_players: int) -> None:
    """PettingZoo's own API test passes at every supported size."""
    api_test(env(num_players=num_players), num_cycles=1000)


def test_pettingzoo_seed_determinism() -> None:
    """PettingZoo's seed test: equal seeds give equal trajectories."""
    seed_test(lambda: env(num_players=4), num_cycles=300)


@pytest.mark.parametrize("num_players", [2, 3, 4])
def test_player_count_setting(num_players: int) -> None:
    """The setting seats the requested number of agents."""
    environment = raw_env(num_players=num_players)
    environment.reset(seed=1)
    assert environment.agents == [f"player_{i}" for i in range(num_players)]
    assert len(environment.game.state.colors) == num_players


@pytest.mark.parametrize("num_players", [1, 5])
def test_unsupported_player_counts_are_rejected(num_players: int) -> None:
    """Fewer than two or more than four players raise `ValueError`."""
    with pytest.raises(ValueError):
        raw_env(num_players=num_players)


def test_spaces_are_shared_across_player_counts() -> None:
    """2- and 4-player games use identical spaces, so one policy serves both."""
    two, four = raw_env(num_players=2), raw_env(num_players=4)
    assert two.action_space("player_0") == four.action_space("player_0")
    assert two.observation_space("player_0") == four.observation_space("player_0")
    assert four.table.size == 351 and four.encoder.size == 1385


def test_absent_seats_encode_as_zero() -> None:
    """In a 2-player game, seat blocks 2 and 3 are padding."""
    environment = raw_env(num_players=2)
    environment.reset(seed=3)
    vector = environment.observe("player_0")["observation"]
    seats_start = 16
    padding = vector[seats_start + 2 * SEAT_FEATURES : seats_start + 4 * SEAT_FEATURES]
    assert not padding.any()
    assert vector[seats_start] == 1.0 and vector[seats_start + SEAT_FEATURES] == 1.0


def test_action_table_keys_are_unique() -> None:
    """Every slot has a distinct key."""
    table = raw_env().table
    assert len(set(table.keys)) == table.size


def test_every_legal_action_has_its_own_slot() -> None:
    """Across full games, legal actions map one-to-one onto mask slots."""
    rng = np.random.default_rng(0)

    def check(environment: CatanAECEnv, agent: str, observation: dict) -> None:
        state = environment.game.state
        if state.current_prompt is ActionPrompt.DISCARD:
            return
        legal = DevCardTimingRule().legal_actions(state)
        assert observation["action_mask"].sum() == len(set(legal))

    for seed in range(3):
        play_random(raw_env(num_players=4), seed, rng, before_step=check)


def test_only_the_decider_has_legal_actions() -> None:
    """Every agent but the decider sees an all-zero mask."""
    environment = raw_env(num_players=4)
    environment.reset(seed=2)
    for agent in environment.agents:
        mask = environment.observe(agent)["action_mask"]
        assert mask.any() == (agent == environment.agent_selection)


def test_illegal_action_is_rejected() -> None:
    """Stepping with a masked-out action raises `ValueError`."""
    environment = raw_env(num_players=4)
    environment.reset(seed=2)
    mask = environment.observe(environment.agent_selection)["action_mask"]
    with pytest.raises(ValueError):
        environment.step(int(np.flatnonzero(mask == 0)[0]))


def test_terminal_rewards_are_zero_sum() -> None:
    """The winner gets +1 and the others share -1 equally."""
    rng = np.random.default_rng(1)
    for seed in range(3):
        environment = raw_env(num_players=4)
        final = play_random(environment, seed, rng)
        winner = environment.game.winning_color()
        assert winner is not None
        for agent, reward in final.items():
            expected = 1.0 if environment.color_of(agent) is winner else -1 / 3
            assert reward == pytest.approx(expected)
        assert sum(final.values()) == pytest.approx(0.0)


def test_turn_limit_truncates_with_zero_reward() -> None:
    """Hitting the turn limit truncates every agent with no reward."""
    environment = raw_env(num_players=4, turn_limit=5)
    final = play_random(environment, 0, np.random.default_rng(0))
    assert environment.game.winning_color() is None
    assert set(final.values()) == {0.0}
    assert environment.game.state.num_turns == 5


def test_resources_are_conserved_through_whole_games() -> None:
    """Hands plus bank stay at 19 of each resource, including discards."""
    rng = np.random.default_rng(2)

    def check(environment: CatanAECEnv, agent: str, observation: dict) -> None:
        assert all(n == 19 for n in resources_in_play(environment).values())

    for seed in range(3):
        play_random(raw_env(num_players=4), seed, rng, before_step=check)


def test_agents_choose_their_own_discards() -> None:
    """With no policy, a discarding agent picks held cards one at a time."""
    rng = np.random.default_rng(3)
    seen: list[bool] = []

    def check(environment: CatanAECEnv, agent: str, observation: dict) -> None:
        state = environment.game.state
        if state.current_prompt is not ActionPrompt.DISCARD:
            return
        color = environment.color_of(agent)
        assert state.current_color() is color
        picks = [
            environment.table.keys[i]
            for i in np.flatnonzero(observation["action_mask"])
        ]
        assert picks and all(key[0] == DISCARD_CARD for key in picks)
        key = player_key(state, color)
        progress = observation["observation"][-6:]
        assert progress[0] > 0, "cards remain to be picked"
        chosen = dict(zip(RESOURCES, np.rint(progress[1:] * 19).astype(int)))
        for _, resource in picks:
            assert state.player_state[f"{key}_{resource}_IN_HAND"] > chosen[resource]
        seen.append(True)

    for seed in range(4):
        play_random(raw_env(num_players=4), seed, rng, before_step=check)
    assert seen


def test_discard_halves_the_hand() -> None:
    """After an agent's last pick, its hand has shrunk by exactly half."""
    environment = raw_env(num_players=4)
    rng = np.random.default_rng(4)
    for seed in range(10):
        environment.reset(seed=seed)
        for agent in environment.agent_iter():
            observation, _, terminated, truncated, _ = environment.last()
            if terminated or truncated:
                environment.step(None)
                continue
            state = environment.game.state
            if state.current_prompt is ActionPrompt.DISCARD:
                color = environment.color_of(agent)
                before = len(player_deck_to_array(state, color))
                while (
                    environment.game.state.current_prompt is ActionPrompt.DISCARD
                    and (environment.agent_selection == agent)
                ):
                    mask = environment.observe(agent)["action_mask"]
                    environment.step(int(rng.choice(np.flatnonzero(mask))))
                after = len(player_deck_to_array(environment.game.state, color))
                assert after == before - before // 2
                return
            environment.step(
                int(rng.choice(np.flatnonzero(observation["action_mask"])))
            )
    pytest.fail("No discard occurred in 10 games.")


def test_discard_policy_hides_discards_from_agents() -> None:
    """With a policy, no agent is ever offered a discard pick."""
    rng = np.random.default_rng(5)

    def check(environment: CatanAECEnv, agent: str, observation: dict) -> None:
        assert environment.game.state.current_prompt is not ActionPrompt.DISCARD

    environment = raw_env(num_players=4, discard_policy=BuildPlanDiscard())
    for seed in range(3):
        play_random(environment, seed, rng, before_step=check)
    discards = [
        a for a in environment.game.state.actions if a.action_type.name == "DISCARD"
    ]
    assert discards


def test_environment_games_respect_dev_card_timing() -> None:
    """No environment game ever plays a card the turn it was bought."""
    rng = np.random.default_rng(6)
    for seed in range(5):
        environment = raw_env(num_players=4)
        play_random(environment, seed, rng)
        assert DevCardTimingRule.count_violations(environment.game.state.actions) == 0


def test_observation_vector_hides_opponent_hand_types() -> None:
    """Defence in depth: the encoded vector, not just the view, is invariant."""
    environment = raw_env(num_players=4)
    environment.reset(seed=7)
    observer = environment.agent_selection
    opponent = next(a for a in environment.agents if a != observer)
    state = environment.game.state
    key = player_key(state, environment.color_of(opponent))
    fields = state.player_state
    total = sum(fields[f"{key}_{r}_IN_HAND"] for r in RESOURCES)
    fields.update({f"{key}_{r}_IN_HAND": 0 for r in RESOURCES})
    fields[f"{key}_WOOD_IN_HAND"] = total + 2
    before = environment.observe(observer)["observation"]
    fields[f"{key}_WOOD_IN_HAND"] = total
    fields[f"{key}_ORE_IN_HAND"] = 2
    after = environment.observe(observer)["observation"]
    np.testing.assert_array_equal(before, after)


def test_interleaved_environments_each_reproduce() -> None:
    """Environments stepped alternately in one process match solo runs.

    Catanatron uses the global RNG, so without per-environment streams two
    interleaved games would perturb each other's dice.
    """

    def trajectory(environment: CatanAECEnv, steps: int) -> list[int]:
        rng = np.random.default_rng(0)
        taken = []
        for _ in range(steps):
            mask = environment.observe(environment.agent_selection)["action_mask"]
            action = int(rng.choice(np.flatnonzero(mask)))
            environment.step(action)
            taken.append(action)
        return taken

    solo = raw_env(num_players=4)
    solo.reset(seed=11)
    expected = trajectory(solo, 400)

    first, second = raw_env(num_players=4), raw_env(num_players=4)
    first.reset(seed=11)
    second.reset(seed=99)
    rng_a, rng_b = np.random.default_rng(0), np.random.default_rng(1)
    interleaved = []
    for _ in range(400):
        mask = first.observe(first.agent_selection)["action_mask"]
        action = int(rng_a.choice(np.flatnonzero(mask)))
        first.step(action)
        interleaved.append(action)
        other = second.observe(second.agent_selection)["action_mask"]
        second.step(int(rng_b.choice(np.flatnonzero(other))))
    assert interleaved == expected


def test_robbery_victims_are_seat_offsets() -> None:
    """Robber slots name victims by seat offset, never by colour."""
    table = raw_env().table
    robber_keys = [k for k in table.keys if k[0] is ActionType.MOVE_ROBBER]
    assert {key[2] for key in robber_keys} == {0, 1, 2, 3}
    assert not any(isinstance(part, Color) for key in table.keys for part in key)


def test_render_ansi_reports_standings() -> None:
    """ANSI rendering returns the standings as text."""
    environment = raw_env(num_players=4, render_mode="ansi")
    play_random(environment, 0, np.random.default_rng(0))
    text = environment.render()
    assert "Winner" in text and "VP" in text


def test_seat_placeholder_refuses_to_decide() -> None:
    """Engine seats never decide on their own."""
    with pytest.raises(RuntimeError):
        ExternalPlayer(Color.RED).decide(None, [])
