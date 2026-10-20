"""Proximal Policy Optimization updates for the shared policy."""

import numpy as np
import torch
from torch import nn

from catan_bots.training.config import TrainingConfig
from catan_bots.training.policy import ActorCritic
from catan_bots.training.rollout import RolloutBatch


class PPOUpdater:
    """Improve the policy from a batch of its own decisions.

    Standard clipped PPO: the probability ratio between the new and old
    policy is clipped so one update cannot move the policy too far, plus a
    value-function loss and an entropy bonus that keeps exploration alive.
    Advantages are normalised per minibatch.

    Attributes:
        policy: The network being trained.
        config: Hyperparameters.
        optimizer: Adam over the network's parameters.
    """

    def __init__(self, policy: ActorCritic, config: TrainingConfig) -> None:
        """Prepare the optimiser.

        Args:
            policy: The network to train.
            config: Hyperparameters.
        """
        self.policy = policy
        self.config = config
        self.optimizer = torch.optim.Adam(
            policy.parameters(), lr=config.learning_rate, eps=1e-5
        )

    def update(self, batch: RolloutBatch, rng: np.random.Generator) -> dict[str, float]:
        """Run the configured epochs of minibatch updates over a batch.

        Args:
            batch: The policy's recent decisions with advantages.
            rng: Shuffles minibatches.

        Returns:
            Mean policy loss, value loss, entropy, approximate KL divergence
            and clip fraction over all minibatches.
        """
        tensors = self._tensors(batch)
        totals = dict.fromkeys(
            ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"), 0.0
        )
        steps = 0
        self.policy.train()
        for _ in range(self.config.update_epochs):
            for indices in self._minibatches(len(batch), rng):
                for name, value in self._step(tensors, indices).items():
                    totals[name] += value
                steps += 1
        self.policy.eval()
        return {name: value / max(steps, 1) for name, value in totals.items()}

    @staticmethod
    def _tensors(batch: RolloutBatch) -> dict[str, torch.Tensor]:
        """Convert a batch's arrays to tensors.

        Args:
            batch: The batch.

        Returns:
            Tensor per field.
        """
        return {
            "observations": torch.from_numpy(batch.observations),
            "masks": torch.from_numpy(batch.masks),
            "actions": torch.from_numpy(batch.actions),
            "log_probs": torch.from_numpy(batch.log_probs),
            "advantages": torch.from_numpy(batch.advantages),
            "returns": torch.from_numpy(batch.returns),
        }

    def _minibatches(self, size: int, rng: np.random.Generator) -> list[np.ndarray]:
        """Shuffle sample indices into minibatches.

        Args:
            size: Samples in the batch.
            rng: Shuffles the order.

        Returns:
            Index arrays, each at most `minibatch_size` long.
        """
        order = rng.permutation(size)
        step = self.config.minibatch_size
        return [order[start : start + step] for start in range(0, size, step)]

    def _step(
        self, tensors: dict[str, torch.Tensor], indices: np.ndarray
    ) -> dict[str, float]:
        """Take one gradient step on one minibatch.

        Args:
            tensors: The whole batch as tensors.
            indices: The minibatch's sample indices.

        Returns:
            This minibatch's losses and diagnostics.
        """
        index = torch.from_numpy(indices)
        mb = {name: tensor[index] for name, tensor in tensors.items()}
        log_probs, entropy, values = self.policy.evaluate_actions(
            mb["observations"], mb["masks"], mb["actions"]
        )
        advantages = mb["advantages"]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        log_ratio = log_probs - mb["log_probs"]
        ratio = log_ratio.exp()
        clip = self.config.clip_coef
        policy_loss = -torch.min(
            ratio * advantages, ratio.clamp(1 - clip, 1 + clip) * advantages
        ).mean()
        value_loss = 0.5 * (values - mb["returns"]).pow(2).mean()
        entropy_mean = entropy.mean()
        loss = (
            policy_loss
            + self.config.value_coef * value_loss
            - self.config.entropy_coef * entropy_mean
        )
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        with torch.no_grad():
            approx_kl = ((ratio - 1) - log_ratio).mean()
            clip_fraction = ((ratio - 1).abs() > clip).float().mean()
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy_mean.item(),
            "approx_kl": approx_kl.item(),
            "clip_fraction": clip_fraction.item(),
        }
