"""Rules this repo models more faithfully than catanatron does by default."""

from catan_bots.rules.builds import (
    BUILD_TARGETS,
    BuildTarget,
    nearest_target,
    reachable_targets,
)
from catan_bots.rules.dev_cards import DevCardTimingRule
from catan_bots.rules.discard import (
    BuildPlanDiscard,
    DiscardPolicy,
    DiscardPolicyRegistry,
    SequentialDiscard,
    UniformRandomDiscard,
    discard_count,
    enumerate_discards,
    hand_counts,
    listdeck_from_counts,
    production_rate,
)
from catan_bots.rules.trading import (
    DEFAULT_TRADING_RULES,
    TradeNegotiation,
    TradeOffer,
    TradeProtocol,
    TradeRecord,
    TradingRules,
    enumerate_offers,
)

__all__ = [
    "BUILD_TARGETS",
    "DEFAULT_TRADING_RULES",
    "BuildTarget",
    "TradeNegotiation",
    "TradeOffer",
    "TradeProtocol",
    "TradeRecord",
    "TradingRules",
    "enumerate_offers",
    "nearest_target",
    "reachable_targets",
    "BuildPlanDiscard",
    "DevCardTimingRule",
    "DiscardPolicy",
    "DiscardPolicyRegistry",
    "SequentialDiscard",
    "UniformRandomDiscard",
    "discard_count",
    "enumerate_discards",
    "hand_counts",
    "listdeck_from_counts",
    "production_rate",
]
