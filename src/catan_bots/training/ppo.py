"""Proximal Policy Optimization updates for the shared policy."""

import logging
from typing import Any

import numpy as np
import torch
from torch import nn

from catan_bots.training.config import TrainingConfig
from catan_bots.training.policy import ActorCritic
from catan_bots.training.rollout import RolloutBatch

LOGGER = logging.getLogger(__name__)


class PPOUpdater:
    """Improve the policy from a batch of its own decisions.

    Standard clipped PPO: the probability ratio between the new and old
    policy is clipped so one update cannot move the policy too far, plus a
    value-function loss and an entropy bonus that keeps exploration alive.
    Advantages are normalised per minibatch.

    Clipping only stops the objective rewarding further moves; several epochs
    of minibatch steps can still carry the policy well away from the one that
    collected the batch. With `config.target_kl` set, each minibatch first
    measures the KL divergence from that policy and the update stops, before
    taking the step, once it exceeds 1.5 times the target.

    Updates run on `config.update_device`. The network and the optimiser's
    state move there for the update and the network moves back to the CPU
    afterwards, because rollout workers and snapshots read it from the CPU.

    Attributes:
        policy: The network being trained.
        config: Hyperparameters.
        device: Where updates run.
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
        self.device = resolve_device(config.update_device)
        self.optimizer = torch.optim.Adam(
            policy.parameters(), lr=config.learning_rate, eps=1e-5
        )

    @property
    def learning_rate(self) -> float:
        """The optimiser's current learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    def set_learning_rate(self, learning_rate: float) -> None:
        """Change the optimiser's learning rate for the next update.

        Args:
            learning_rate: The new rate.
        """
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

    def optimizer_state(self) -> dict[str, Any]:
        """Copy the optimiser's state to the CPU, for a checkpoint any machine can read.

        Returns:
            The optimiser's `state_dict`, with every tensor on the CPU.
        """
        return _to_cpu(self.optimizer.state_dict())

    def update(self, batch: RolloutBatch, rng: np.random.Generator) -> dict[str, float]:
        """Run up to the configured epochs of minibatch updates over a batch.

        Args:
            batch: The policy's recent decisions with advantages.
            rng: Shuffles minibatches.

        Returns:
            Mean policy loss, value loss, entropy, approximate KL divergence
            and clip fraction over the minibatch steps taken, plus the
            learning rate, the number of steps taken, `kl_stopped` (1.0 if
            the KL limit ended the update early, else 0.0) and
            `skipped_minibatches` (minibatches whose loss was not finite,
            which are never applied).
        """
        self._move_to(self.device)
        tensors = self._tensors(batch)
        totals = dict.fromkeys(
            ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"), 0.0
        )
        steps, skipped, stopped = 0, 0, False
        self.policy.train()
        for _ in range(self.config.update_epochs):
            for indices in self._minibatches(len(batch), rng):
                loss, diagnostics = self._loss(tensors, indices)
                if not torch.isfinite(loss):
                    skipped += 1
                    LOGGER.warning(
                        "Skipped a minibatch of %d with a non-finite loss.",
                        len(indices),
                    )
                    continue
                if self._kl_exceeded(diagnostics["approx_kl"]):
                    stopped = True
                    break
                self._apply(loss)
                for name, value in diagnostics.items():
                    totals[name] += value
                steps += 1
            if stopped:
                break
        self._move_to(torch.device("cpu"))
        self.policy.eval()
        metrics = {name: value / max(steps, 1) for name, value in totals.items()}
        return {
            **metrics,
            "learning_rate": self.learning_rate,
            "minibatch_steps": float(steps),
            "kl_stopped": float(stopped),
            "skipped_minibatches": float(skipped),
        }

    def _move_to(self, device: torch.device) -> None:
        """Move the network, and the optimiser state it will need, to a device.

        Adam's step counters stay on the CPU, where PyTorch expects them.

        Args:
            device: Target device.
        """
        self.policy.to(device)
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if key != "step" and torch.is_tensor(value):
                    state[key] = value.to(device)

    def _kl_exceeded(self, approx_kl: float) -> bool:
        """Check a minibatch's KL divergence against the configured limit.

        Args:
            approx_kl: Approximate KL divergence from the collecting policy.

        Returns:
            True if a limit is set and `approx_kl` is above 1.5 times it.
        """
        target = self.config.target_kl
        return target > 0 and approx_kl > 1.5 * target

    def _tensors(self, batch: RolloutBatch) -> dict[str, torch.Tensor]:
        """Convert a batch's arrays to tensors on the update device.

        Args:
            batch: The batch.

        Returns:
            Tensor per field.
        """
        arrays = {
            "observations": batch.observations,
            "masks": batch.masks,
            "actions": batch.actions,
            "log_probs": batch.log_probs,
            "advantages": batch.advantages,
            "returns": batch.returns,
        }
        return {
            name: torch.from_numpy(array).to(self.device)
            for name, array in arrays.items()
        }

    def _minibatches(self, size: int, rng: np.random.Generator) -> list[np.ndarray]:
        """Shuffle sample indices into minibatches of nearly equal size.

        Batches are rarely a multiple of `minibatch_size`, because workers
        finish the games they are in. Cutting fixed-size chunks would leave a
        remainder that can be a single sample, whose advantage standard
        deviation is NaN and which turns every weight NaN. Instead the batch
        is split into `size // minibatch_size` chunks (at least one) whose
        sizes differ by at most one, so every chunk holds at least
        `minibatch_size` samples (or the whole batch, if it is smaller).

        Args:
            size: Samples in the batch.
            rng: Shuffles the order.

        Returns:
            Index arrays covering every sample exactly once.
        """
        order = rng.permutation(size)
        chunks = max(1, size // self.config.minibatch_size)
        return np.array_split(order, chunks)

    def _loss(
        self, tensors: dict[str, torch.Tensor], indices: np.ndarray
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute one minibatch's loss with the current weights.

        Args:
            tensors: The whole batch as tensors.
            indices: The minibatch's sample indices.

        Returns:
            The differentiable total loss, and the minibatch's losses and
            diagnostics, measured before any step is taken on it.
        """
        index = torch.from_numpy(indices).to(self.device)
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
        with torch.no_grad():
            approx_kl = ((ratio - 1) - log_ratio).mean()
            clip_fraction = ((ratio - 1).abs() > clip).float().mean()
        return loss, {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy_mean.item(),
            "approx_kl": approx_kl.item(),
            "clip_fraction": clip_fraction.item(),
        }

    def _apply(self, loss: torch.Tensor) -> None:
        """Take one clipped gradient step on a minibatch loss.

        Args:
            loss: The loss returned by `_loss`.
        """
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.optimizer.step()


def resolve_device(name: str) -> torch.device:
    """Turn an `update_device` setting into a device, checking it exists.

    Args:
        name: `"cpu"`, `"mps"`, `"cuda"` or `"auto"`.

    Returns:
        The device. `"auto"` picks CUDA, then MPS, then the CPU.

    Raises:
        ValueError: If the named accelerator is not available here.
    """
    available = {
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
        "cpu": True,
    }
    if name == "auto":
        return torch.device(next(d for d in ("cuda", "mps", "cpu") if available[d]))
    if not available.get(name, False):
        raise ValueError(f"update_device {name!r} is not available on this machine.")
    return torch.device(name)


def _to_cpu(value: Any) -> Any:
    """Copy every tensor inside nested dicts and lists to the CPU.

    Args:
        value: A tensor, or a dict or list that may hold tensors.

    Returns:
        The same structure with CPU tensors.
    """
    if torch.is_tensor(value):
        return value.cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    return value
