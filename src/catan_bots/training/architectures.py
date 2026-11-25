"""Build the configured policy network, or rebuild one from a snapshot.

A snapshot names its architecture, so a run can load opponents and
evaluation references trained with a different network than its own.
Snapshots written before architectures were named are MLPs.
"""

from typing import Any

from catan_bots.envs import CatanAECEnv
from catan_bots.training.config import TrainingConfig
from catan_bots.training.graph_policy import GraphActorCritic, GraphLayout
from catan_bots.training.policy import ActorCritic, MlpActorCritic

DEFAULT_ARCHITECTURE = MlpActorCritic.architecture
NETWORKS: dict[str, type[ActorCritic]] = {
    MlpActorCritic.architecture: MlpActorCritic,
    GraphActorCritic.architecture: GraphActorCritic,
}
_SNAPSHOT_KEYS = ("architecture", "observation_size", "num_actions", "weights")


def build_policy(config: TrainingConfig, env: CatanAECEnv) -> ActorCritic:
    """Build an untrained network of the configured architecture.

    Args:
        config: The run's configuration.
        env: The environment the policy plays, fixing the observation and
            action layout.

    Returns:
        The network, in evaluation mode.
    """
    if config.architecture == GraphActorCritic.architecture:
        policy: ActorCritic = GraphActorCritic(
            GraphLayout.from_env(env), config.graph_width, config.graph_rounds
        )
    else:
        policy = MlpActorCritic(env.encoder.size, env.table.size, config.hidden_sizes)
    return policy.eval()


def policy_from_snapshot(snapshot: dict[str, Any]) -> ActorCritic:
    """Rebuild a network, with its weights, from `ActorCritic.snapshot` output.

    Args:
        snapshot: Architecture, settings and weights.

    Returns:
        The network, in evaluation mode.

    Raises:
        ValueError: If the snapshot names an unknown architecture.
    """
    name = snapshot.get("architecture", DEFAULT_ARCHITECTURE)
    if name not in NETWORKS:
        raise ValueError(f"Unknown architecture {name!r} in snapshot.")
    settings = {k: v for k, v in snapshot.items() if k not in _SNAPSHOT_KEYS}
    policy = NETWORKS[name].from_settings(
        snapshot["observation_size"], snapshot["num_actions"], settings
    )
    policy.load_state_dict(snapshot["weights"])
    return policy.eval()
