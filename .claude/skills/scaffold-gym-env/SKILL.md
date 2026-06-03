---
name: scaffold-gym-env
description: Scaffold a new Gymnasium (OpenAI Gym successor) environment for Catan following this repo's standards — a typed Env subclass with documented observation/action spaces, action masking, seeded reset, the 5-tuple step contract, registration, and a validity smoke test. Use when creating a new environment or a new env variant from scratch.
---

# Scaffold Gym Environment

This skill creates a new Gymnasium environment that conforms to
[CLAUDE.md](../../../CLAUDE.md) and is ready for RL training.

## When to use

- Starting a brand-new Catan environment.
- Adding a new variant (e.g. a simplified board) as a separate env class.

## Prerequisites

Ensure dependencies exist (managed with `uv`):

```bash
uv add gymnasium numpy
uv add --dev pytest
```

## Steps

1. **Decide the contract first.** Write down, in the class docstring, the
   observation space, action space, action-mask scheme, reward definition, and
   termination/truncation conditions before writing logic.
2. **Create the package layout:**

   ```text
   src/catan_bots/envs/
   ├── __init__.py        # registers the env id
   └── catan_env.py       # the Env subclass
   ```

3. **Implement the `Env` subclass** with atomic helper methods. Use the template
   below as a starting point.
4. **Register** the env id in `envs/__init__.py`.
5. **Validate**: run `check_env` and a random-rollout smoke test.
6. **Format & lint**: `uv run ruff format . && uv run ruff check .`.

## Template

```python
"""Catan reinforcement-learning environment (Gymnasium API)."""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CatanEnv(gym.Env):
    """A Gymnasium environment for the game of Catan.

    Observation space, action space, action mask, reward, and termination are
    documented here as the env's contract. Update this docstring whenever the
    contract changes.

    Attributes:
        observation_space: Encoding of board, resources, and turn state.
        action_space: Discrete set of game actions (use the mask for legality).
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, render_mode: str | None = None) -> None:
        """Initialize spaces and configuration.

        Args:
            render_mode: One of ``metadata["render_modes"]`` or ``None``.
        """
        super().__init__()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(self._num_actions())
        self.observation_space = self._build_observation_space()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """Start a new game and return the initial observation.

        Args:
            seed: Seed routed to ``self.np_random`` for determinism.
            options: Optional reset configuration.

        Returns:
            A ``(observation, info)`` tuple. ``info`` includes the action mask.
        """
        super().reset(seed=seed)
        self._init_state()
        return self._build_observation(), self._build_info()

    def step(
        self, action: int
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Apply one action and advance the game.

        Args:
            action: Index into ``action_space``; must be legal per the mask.

        Returns:
            ``(observation, reward, terminated, truncated, info)``.
        """
        self._apply_action(action)
        reward = self._compute_reward()
        terminated = self._is_terminal()
        return self._build_observation(), reward, terminated, False, self._build_info()

    def action_mask(self) -> np.ndarray:
        """Return a boolean mask of currently legal actions.

        Returns:
            A 1-D boolean array aligned with ``action_space``.
        """
        raise NotImplementedError

    # --- atomic helpers (each does one thing) ---
    def _num_actions(self) -> int: ...
    def _build_observation_space(self) -> spaces.Space: ...
    def _init_state(self) -> None: ...
    def _apply_action(self, action: int) -> None: ...
    def _compute_reward(self) -> float: ...
    def _is_terminal(self) -> bool: ...
    def _build_observation(self) -> Any: ...
    def _build_info(self) -> dict[str, Any]:
        """Return diagnostics including the action mask."""
        return {"action_mask": self.action_mask()}
```

## Validation snippet

```bash
uv run python - <<'PY'
from gymnasium.utils.env_checker import check_env
from catan_bots.envs.catan_env import CatanEnv

env = CatanEnv()
check_env(env)
obs, info = env.reset(seed=0)
for _ in range(50):
    mask = info["action_mask"]
    action = int(mask.nonzero()[0][0])  # first legal action
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
print("env OK")
PY
```

## Done when

- `check_env` passes and the smoke rollout runs without error.
- Spaces, reward, and mask are fully documented in docstrings.
