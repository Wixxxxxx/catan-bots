"""Self-play training: one shared policy learning in every seat.

Run from the command line with `python -m catan_bots.training`; see
`__main__` for the commands.
"""

from catan_bots.training.config import TrainingConfig
from catan_bots.training.evaluator import EvaluationResult, Evaluator, wilson_interval
from catan_bots.training.policy import ActorCritic
from catan_bots.training.trainer import Trainer

__all__ = [
    "ActorCritic",
    "EvaluationResult",
    "Evaluator",
    "Trainer",
    "TrainingConfig",
    "wilson_interval",
]
