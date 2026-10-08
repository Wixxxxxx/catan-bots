"""Tests for domestic trading inside the PettingZoo environment."""

from collections import Counter

import numpy as np
import pytest
from catanatron.models.enums import RESOURCES
from catanatron.state_functions import player_deck_to_array, player_key

from catan_bots.envs import CatanAECEnv, raw_env
from catan_bots.envs.actions import (
    ACCEPT_TRADE,
    CANCEL_TRADE,
    CONFIRM_TRADE,
    PROPOSE_TRADE,
    REJECT_TRADE,
)
from catan_bots.rules.trading import TradeOffer

TRADE_KINDS = {PROPOSE_TRADE, ACCEPT_TRADE, REJECT_TRADE, CANCEL_TRADE, CONFIRM_TRADE}
WOOD_FOR_ORE = TradeOffer.of({"WOOD": 1}, {"ORE": 1})


def legal_keys(environment: CatanAECEnv) -> set[tuple]:
    """Collect the keys legal for the deciding agent.

    Args:
        environment: Environment mid-game.

    Returns:
        The deciding agent's legal action keys.
    """
    mask = environment.observe(environment.agent_selection)["action_mask"]
    return {environment.table.keys[i] for i in np.flatnonzero(mask)}


def to_proposal_point(seed: int = 3) -> CatanAECEnv:
    """Play random moves until the decider may propose, then even the hands.

    Every seat is given two wood and two ore, so a wood-for-ore offer is
    always affordable and every responder can pay.

    Args:
        seed: Game seed.

    Returns:
        An environment whose decider may propose right now.
    """
    environment = raw_env(num_players=4)
    environment.reset(seed=seed)
    rng = np.random.default_rng(seed)
    while not any(k[0] == PROPOSE_TRADE for k in legal_keys(environment)):
        mask = environment.observe(environment.agent_selection)["action_mask"]
        environment.step(int(rng.choice(np.flatnonzero(mask))))
    state = environment.game.state
    for color in state.colors:
        key = player_key(state, color)
        for resource in RESOURCES:
            state.player_state[f"{key}_{resource}_IN_HAND"] = 0
        state.player_state[f"{key}_WOOD_IN_HAND"] = 2
        state.player_state[f"{key}_ORE_IN_HAND"] = 2
    environment._load_decision()
    return environment


def step_key(environment: CatanAECEnv, key: tuple) -> None:
    """Take the action with a given key for the deciding agent.

    Args:
        environment: Environment mid-game.
        key: Canonical action key; must be legal now.
    """
    environment.step(environment.table.index(key))


def propose_wood_for_ore(environment: CatanAECEnv) -> str:
    """Make the decider propose one wood for one ore.

    Args:
        environment: Environment at a proposal point.

    Returns:
        The proposing agent.
    """
    proposer = environment.agent_selection
    step_key(environment, (PROPOSE_TRADE, WOOD_FOR_ORE.give, WOOD_FOR_ORE.want))
    return proposer


def test_responders_are_asked_in_turn() -> None:
    """After a proposal, each other seat is asked to accept or reject."""
    environment = to_proposal_point()
    proposer = propose_wood_for_ore(environment)
    asked = []
    for _ in range(3):
        assert environment.agent_selection != proposer
        assert legal_keys(environment) == {(ACCEPT_TRADE,), (REJECT_TRADE,)}
        asked.append(environment.agent_selection)
        step_key(environment, (ACCEPT_TRADE,))
    assert len(set(asked)) == 3
    assert environment.agent_selection == proposer


def test_proposer_chooses_among_acceptors() -> None:
    """The proposer may trade with any acceptor, or cancel."""
    environment = to_proposal_point()
    proposer = propose_wood_for_ore(environment)
    for answer in (ACCEPT_TRADE, REJECT_TRADE, ACCEPT_TRADE):
        step_key(environment, (answer,))
    assert environment.agent_selection == proposer
    assert legal_keys(environment) == {
        (CONFIRM_TRADE, 1),
        (CONFIRM_TRADE, 3),
        (CANCEL_TRADE,),
    }


def test_confirmed_trade_moves_the_cards() -> None:
    """Confirming swaps the offer's cards and returns control to the engine."""
    environment = to_proposal_point()
    proposer = propose_wood_for_ore(environment)
    for _ in range(3):
        step_key(environment, (ACCEPT_TRADE,))
    step_key(environment, (CONFIRM_TRADE, 1))
    state = environment.game.state
    order = state.colors
    proposer_color = environment.color_of(proposer)
    partner = order[(order.index(proposer_color) + 1) % len(order)]
    assert Counter(player_deck_to_array(state, proposer_color)) == {"WOOD": 1, "ORE": 3}
    assert Counter(player_deck_to_array(state, partner)) == {"WOOD": 3, "ORE": 1}
    assert environment.agent_selection == proposer
    assert environment._trading.negotiation is None


def test_unanimous_rejection_returns_to_the_proposer() -> None:
    """An offer nobody takes closes itself; the proposer plays on."""
    environment = to_proposal_point()
    proposer = propose_wood_for_ore(environment)
    for _ in range(3):
        step_key(environment, (REJECT_TRADE,))
    assert environment.agent_selection == proposer
    assert not any(
        k[0] in {CONFIRM_TRADE, CANCEL_TRADE} for k in legal_keys(environment)
    )


def test_offers_are_capped_per_turn_in_the_environment() -> None:
    """After three offers no proposal is legal until the next turn."""
    environment = to_proposal_point()
    for _ in range(3):
        propose_wood_for_ore(environment)
        for _ in range(3):
            step_key(environment, (REJECT_TRADE,))
    assert not any(k[0] == PROPOSE_TRADE for k in legal_keys(environment))


def test_open_offer_appears_in_every_observation() -> None:
    """Every seat's encoded trade block shows the open offer."""
    environment = to_proposal_point()
    propose_wood_for_ore(environment)
    trade = environment.encoder.blocks["trade"]
    for agent in environment.agents:
        block = environment.observe(agent)["observation"][trade]
        assert block[0] == 1.0, f"{agent} cannot see the open offer"


def test_only_the_trade_decider_may_act() -> None:
    """While an offer is open, every agent but the decider is masked out."""
    environment = to_proposal_point()
    propose_wood_for_ore(environment)
    for agent in environment.agents:
        mask = environment.observe(agent)["action_mask"]
        assert mask.any() == (agent == environment.agent_selection)


def test_trading_off_never_offers_a_trade_slot() -> None:
    """With trading off, no trade action is ever legal; spaces are unchanged."""
    environment = raw_env(num_players=4, trading=None)
    assert environment.table.size == raw_env(num_players=4).table.size
    environment.reset(seed=0)
    rng = np.random.default_rng(0)
    for agent in environment.agent_iter():
        observation, _, terminated, truncated, _ = environment.last()
        if terminated or truncated:
            environment.step(None)
            continue
        keys = {
            environment.table.keys[i]
            for i in np.flatnonzero(observation["action_mask"])
        }
        assert not any(k[0] in TRADE_KINDS for k in keys)
        environment.step(int(rng.choice(np.flatnonzero(observation["action_mask"]))))


def test_trade_block_is_zero_with_trading_off() -> None:
    """The trade features stay zero when trading is switched off."""
    environment = raw_env(num_players=4, trading=None)
    environment.reset(seed=0)
    vector = environment.observe(environment.agent_selection)["observation"]
    assert not vector[environment.encoder.blocks["trade"]].any()


@pytest.mark.parametrize("num_players", [2, 3])
def test_trading_works_at_smaller_tables(num_players: int) -> None:
    """Fewer seats mean fewer responders and partner slots."""
    environment = raw_env(num_players=num_players)
    environment.reset(seed=5)
    rng = np.random.default_rng(5)
    while not any(k[0] == PROPOSE_TRADE for k in legal_keys(environment)):
        mask = environment.observe(environment.agent_selection)["action_mask"]
        environment.step(int(rng.choice(np.flatnonzero(mask))))
    proposal = next(k for k in legal_keys(environment) if k[0] == PROPOSE_TRADE)
    step_key(environment, proposal)
    responses = 0
    while (
        environment._trading.negotiation and environment._trading.negotiation.awaiting
    ):
        step_key(environment, (REJECT_TRADE,))
        responses += 1
    assert responses <= num_players - 1
