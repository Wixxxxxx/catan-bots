"""Multi-agent Catan environments.

Use `env()` for the wrapped environment PettingZoo tooling expects, or
`raw_env()` for direct access to `CatanAECEnv`.
"""

from pettingzoo import AECEnv
from pettingzoo.utils import wrappers

from catan_bots.envs.actions import ActionTable
from catan_bots.envs.catan_aec import CatanAECEnv
from catan_bots.envs.encoding import ObservationEncoder
from catan_bots.envs.random_stream import RandomStream
from catan_bots.envs.topology import BoardTopology


def raw_env(**kwargs: object) -> CatanAECEnv:
    """Create an unwrapped Catan environment.

    Args:
        **kwargs: Passed to `CatanAECEnv`.

    Returns:
        The environment.
    """
    return CatanAECEnv(**kwargs)


def env(**kwargs: object) -> AECEnv:
    """Create a Catan environment with PettingZoo's standard safety wrappers.

    The wrappers reject out-of-range actions and enforce calling `reset`
    before anything else.

    Args:
        **kwargs: Passed to `CatanAECEnv`.

    Returns:
        The wrapped environment.
    """
    environment = raw_env(**kwargs)
    environment = wrappers.AssertOutOfBoundsWrapper(environment)
    return wrappers.OrderEnforcingWrapper(environment)


__all__ = [
    "ActionTable",
    "BoardTopology",
    "CatanAECEnv",
    "ObservationEncoder",
    "RandomStream",
    "env",
    "raw_env",
]
