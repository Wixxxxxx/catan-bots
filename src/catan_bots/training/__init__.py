"""Self-play training: one shared policy learning in every seat.

Run from the command line with `python -m catan_bots.training`; see
`__main__` for the commands.
"""

from catan_bots.training.architectures import build_policy, policy_from_snapshot
from catan_bots.training.config import TrainingConfig
from catan_bots.training.evaluator import EvaluationResult, Evaluator, wilson_interval
from catan_bots.training.graph_policy import GraphActorCritic, GraphLayout
from catan_bots.training.policy import ActorCritic, MlpActorCritic
from catan_bots.training.trainer import Trainer

__all__ = [
    "ActorCritic",
    "EvaluationResult",
    "Evaluator",
    "GraphActorCritic",
    "GraphLayout",
    "MlpActorCritic",
    "Trainer",
    "TrainingConfig",
    "build_policy",
    "policy_from_snapshot",
    "wilson_interval",
]
