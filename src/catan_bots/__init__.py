"""Reinforcement-learning bots and analytics for the game of Catan.

The package holds the production code: bots, game construction, and analytics.
Notebooks in [notebooks/](../../notebooks) are for research only and call into
these modules rather than redefining logic.
"""

from catan_bots.analytics import (
    ActionTypeCounter,
    BoardInspector,
    GameReport,
    NodeProduction,
    PlayerSnapshot,
    TileSummary,
    Tournament,
    TournamentResult,
)
from catan_bots.bots import GreedySettlerBot
from catan_bots.games import GameFactory
from catan_bots.rules import (
    BuildPlanDiscard,
    DiscardPolicy,
    DiscardPolicyRegistry,
    UniformRandomDiscard,
    enumerate_discards,
)
from catan_bots.runner import GameRunner

__all__ = [
    "ActionTypeCounter",
    "BoardInspector",
    "BuildPlanDiscard",
    "DiscardPolicy",
    "DiscardPolicyRegistry",
    "GameFactory",
    "GameRunner",
    "GameReport",
    "GreedySettlerBot",
    "NodeProduction",
    "PlayerSnapshot",
    "TileSummary",
    "Tournament",
    "TournamentResult",
    "UniformRandomDiscard",
    "enumerate_discards",
]
