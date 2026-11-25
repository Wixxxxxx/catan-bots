"""Policy-and-value networks shared by every learner seat.

`ActorCritic` is the interface the rest of training uses: masking, sampling,
scoring taken actions and packaging weights are the same for every network.
Each architecture only says how an observation becomes action logits and a
value (`forward`) and which settings rebuild it (`settings`). This module
holds the interface and the original fully connected network, `MlpActorCritic`;
the graph network lives in `graph_policy`, and `architectures` picks between
them.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

MASKED_LOGIT = -1e9


def orthogonal_layer(inputs: int, outputs: int, gain: float) -> nn.Linear:
    """Create a linear layer with orthogonal weights and zero bias.

    Args:
        inputs: Input width.
        outputs: Output width.
        gain: Orthogonal initialisation gain; small for policy heads so the
            first policy is close to uniform.

    Returns:
        The initialised layer.
    """
    layer = nn.Linear(inputs, outputs)
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class ActorCritic(nn.Module, ABC):
    """A network that scores every action and estimates the deciding seat's value.

    The policy scores every action in the fixed action space; illegal
    actions are masked to a probability of exactly zero. The value estimates
    the deciding seat's expected final reward.

    Attributes:
        architecture: Name saved in snapshots so the right class rebuilds them.
        observation_size: Length of the encoded observation.
        num_actions: Size of the action space.
    """

    architecture: ClassVar[str]

    def __init__(self, observation_size: int, num_actions: int) -> None:
        """Record the sizes every architecture shares.

        Args:
            observation_size: Length of the encoded observation.
            num_actions: Size of the action space.
        """
        super().__init__()
        self.observation_size = observation_size
        self.num_actions = num_actions

    @abstractmethod
    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Score actions and estimate value.

        Args:
            observations: Batch of encoded observations, shape `(B, obs)`.

        Returns:
            Unmasked action logits `(B, actions)` and values `(B,)`.
        """

    @abstractmethod
    def settings(self) -> dict[str, Any]:
        """List the constructor settings, beyond the two sizes, that rebuild this network.

        Returns:
            Plain values (numbers, strings, lists, dicts) safe to save.
        """

    @classmethod
    @abstractmethod
    def from_settings(
        cls, observation_size: int, num_actions: int, settings: dict[str, Any]
    ) -> "ActorCritic":
        """Build an untrained network from saved settings.

        Args:
            observation_size: Length of the encoded observation.
            num_actions: Size of the action space.
            settings: What `settings` returned, read from a snapshot.

        Returns:
            The network, with fresh weights.
        """

    def distribution(
        self, observations: torch.Tensor, masks: torch.Tensor
    ) -> tuple[Categorical, torch.Tensor]:
        """Build the masked action distribution.

        Args:
            observations: Batch of encoded observations.
            masks: Boolean legality masks, shape `(B, actions)`.

        Returns:
            The categorical distribution over legal actions, and values.
        """
        logits, values = self(observations)
        logits = logits.masked_fill(~masks, MASKED_LOGIT)
        return Categorical(logits=logits), values

    @torch.no_grad()
    def act(
        self,
        observations: torch.Tensor,
        masks: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Choose actions without tracking gradients.

        Args:
            observations: Batch of encoded observations.
            masks: Boolean legality masks.
            deterministic: Take the most likely action instead of sampling.

        Returns:
            Actions, their log-probabilities, and value estimates.
        """
        dist, values = self.distribution(observations, masks)
        actions = dist.probs.argmax(-1) if deterministic else dist.sample()
        return actions, dist.log_prob(actions), values

    def evaluate_actions(
        self, observations: torch.Tensor, masks: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score previously taken actions, with gradients, for a PPO update.

        Args:
            observations: Batch of encoded observations.
            masks: Boolean legality masks.
            actions: Actions that were taken.

        Returns:
            Log-probabilities of `actions`, distribution entropies, values.
        """
        dist, values = self.distribution(observations, masks)
        return dist.log_prob(actions), dist.entropy(), values

    def snapshot(self) -> dict[str, Any]:
        """Package the architecture, its settings and the weights for saving or sending.

        Returns:
            A dict `architectures.policy_from_snapshot` can rebuild the
            network from.
        """
        return {
            "architecture": self.architecture,
            "observation_size": self.observation_size,
            "num_actions": self.num_actions,
            **self.settings(),
            "weights": {k: v.detach().cpu() for k, v in self.state_dict().items()},
        }


class MlpActorCritic(ActorCritic):
    """A fully connected torso with a policy head and a value head.

    The whole observation is one flat vector; every input number has its own
    weights into the first hidden layer.

    Attributes:
        hidden_sizes: Widths of the hidden layers.
    """

    architecture = "mlp"

    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        hidden_sizes: Sequence[int] = (256, 256),
    ) -> None:
        """Build the network.

        Args:
            observation_size: Length of the encoded observation.
            num_actions: Size of the action space.
            hidden_sizes: Widths of the hidden layers.
        """
        super().__init__(observation_size, num_actions)
        self.hidden_sizes = tuple(hidden_sizes)
        layers: list[nn.Module] = []
        width = observation_size
        for size in self.hidden_sizes:
            layers += [orthogonal_layer(width, size, gain=np.sqrt(2)), nn.Tanh()]
            width = size
        self.torso = nn.Sequential(*layers)
        self.policy_head = orthogonal_layer(width, num_actions, gain=0.01)
        self.value_head = orthogonal_layer(width, 1, gain=1.0)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Score actions and estimate value.

        Args:
            observations: Batch of encoded observations, shape `(B, obs)`.

        Returns:
            Unmasked action logits `(B, actions)` and values `(B,)`.
        """
        hidden = self.torso(observations)
        return self.policy_head(hidden), self.value_head(hidden).squeeze(-1)

    def settings(self) -> dict[str, Any]:
        """List the settings that rebuild this network.

        Returns:
            The hidden layer widths.
        """
        return {"hidden_sizes": list(self.hidden_sizes)}

    @classmethod
    def from_settings(
        cls, observation_size: int, num_actions: int, settings: dict[str, Any]
    ) -> "MlpActorCritic":
        """Build an untrained network from saved settings.

        Args:
            observation_size: Length of the encoded observation.
            num_actions: Size of the action space.
            settings: Holds `hidden_sizes`.

        Returns:
            The network, with fresh weights.
        """
        return cls(observation_size, num_actions, settings["hidden_sizes"])
