# catan-bots

Reinforcement-learning bots for the game of Catan: a Gymnasium environment, RL
training pipelines, gameplay analytics, and dashboards.

See [CLAUDE.md](CLAUDE.md) for coding standards and project layout.

## Project log

### 2026-06-16 — Scaffolding complete, implementation not started

**Status:** Pre-implementation. One commit ("Initial commit"). No src code yet.

**What's done:**

- `pyproject.toml` wired with `catanatron>=3.2.1` and `gymnasium>=1.3.0`.
- Five specialized subagents configured in `.claude/agents/`: gym-environment,
  rl-expert, catan-expert, game-data-scientist, dashboard-engineer.
- Two ready-to-run skills in `.claude/skills/`: `scaffold-gym-env` (full env
  template) and `train-rl-agent` (MaskablePPO + evaluation scaffold).
- `src/catan_bots/playground.ipynb` — thorough `catanatron` API walkthrough
  covering `Game`, `State`, `Player`, `Action`, `GameAccumulator`, and all
  built-in bots. The catanatron API is well understood.

**What's missing (everything):**

- `src/catan_bots/envs/` — Gymnasium env does not exist yet.
- No training pipeline, no baselines, no tests.
- `catan.py` and `main.py` are stubs.

**Blockers / decisions needed:**

- Game scope: full 4-player Catan or simplified 2-player for the first env
  iteration? Recommendation: 2-player first — smaller action space, faster
  training feedback loop.

**Next steps** (ordered, concrete):

1. Decide 2-player vs. 4-player scope for v1 env (30 seconds of thought).
2. Ask Claude to run the `scaffold-gym-env` skill — it generates the full
   `CatanEnv` skeleton with correct spaces, action mask, and 5-tuple step.
3. Implement `_init_state` / `reset` using `catanatron.Game(players, seed=...)`.
4. Implement `step` wrapping `game.execute(action)` and map legal actions to
   the action space index.
5. Run `check_env` + smoke rollout (snippet is in the scaffold skill).
6. Add MaskablePPO deps (`uv add stable-baselines3 sb3-contrib tensorboard`)
   and run the `train-rl-agent` skill for the first smoke training run.
