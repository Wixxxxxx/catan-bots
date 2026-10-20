"""The policy-and-value network shared by every learner seat."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

MASKED_LOGIT = -1e9


def _layer(inputs: int, outputs: int, gain: float) -> nn.Linear:
    """Create a linear layer with orthogonal weights and zero bias.

    Args:
        inputs: Input width.
        outputs: Output width.
        gain: Orthogonal initialisation gain.

    Returns:
        The initialised layer.
    """
    layer = nn.Linear(inputs, outputs)
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class ActorCritic(nn.Module):
    """A shared torso with a masked policy head and a value head.

    The policy head scores every action in the fixed action space; illegal
    actions are masked to a probability of exactly zero. The value head
    estimates the deciding seat's expected final reward.

    Attributes:
        observation_size: Length of the encoded observation.
        num_actions: Size of the action space.
        hidden_sizes: Widths of the hidden layers.
    """

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
        super().__init__()
        self.observation_size = observation_size
        self.num_actions = num_actions
        self.hidden_sizes = tuple(hidden_sizes)
        layers: list[nn.Module] = []
        width = observation_size
        for size in self.hidden_sizes:
            layers += [_layer(width, size, gain=np.sqrt(2)), nn.Tanh()]
            width = size
        self.torso = nn.Sequential(*layers)
        self.policy_head = _layer(width, num_actions, gain=0.01)
        self.value_head = _layer(width, 1, gain=1.0)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Score actions and estimate value.

        Args:
            observations: Batch of encoded observations, shape `(B, obs)`.

        Returns:
            Unmasked action logits `(B, actions)` and values `(B,)`.
        """
        hidden = self.torso(observations)
        return self.policy_head(hidden), self.value_head(hidden).squeeze(-1)

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
        """Package the architecture and weights for saving or sending.

        Returns:
            A dict `from_snapshot` can rebuild the network from.
        """
        return {
            "observation_size": self.observation_size,
            "num_actions": self.num_actions,
            "hidden_sizes": list(self.hidden_sizes),
            "weights": {k: v.detach().cpu() for k, v in self.state_dict().items()},
        }

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, Any]) -> "ActorCritic":
        """Rebuild a network from `snapshot` output.

        Args:
            snapshot: Architecture and weights.

        Returns:
            The network, in evaluation mode.
        """
        policy = cls(
            snapshot["observation_size"],
            snapshot["num_actions"],
            snapshot["hidden_sizes"],
        )
        policy.load_state_dict(snapshot["weights"])
        return policy.eval()
