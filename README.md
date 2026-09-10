# catan-bots

Reinforcement-learning bots for the game of Catan: a Gymnasium environment, RL
training pipelines, gameplay analytics, and dashboards.

See [CLAUDE.md](CLAUDE.md) for coding standards and project layout.

## Getting started

```bash
uv sync                 # install deps and the catan_bots package itself
uv run pytest           # run the test suite
uv run ruff format .    # format
uv run ruff check .     # lint
```

Production code lives in [src/catan_bots/](src/catan_bots/);
[notebooks/playground.ipynb](notebooks/playground.ipynb) is research only and
imports those modules.

## Project log

### 2026-09-10 — Production code extracted into modules

**Status:** The package is importable and tested; the notebook is research-only.

**What changed:**

- `src/catan_bots/` is now a real installed package (`uv sync` installs it), so
  `from catan_bots import ...` works from notebooks, scripts and tests.
- Everything that used to be defined inside the notebook now lives in modules:
  - [src/catan_bots/games.py](src/catan_bots/games.py) — `GameFactory`, which
    resets players between games and makes any integer seed reproducible
    (plain `catanatron.Game` silently treats `seed=0` as "no seed").
  - [src/catan_bots/bots/greedy_settler.py](src/catan_bots/bots/greedy_settler.py)
    — `GreedySettlerBot`, with a private seeded RNG for reproducible tie-breaks.
  - [src/catan_bots/analytics/](src/catan_bots/analytics/) — `GameReport` and
    `PlayerSnapshot`, `BoardInspector`, `Tournament`/`TournamentResult`, and the
    `ActionTypeCounter` accumulator.
- [notebooks/playground.ipynb](notebooks/playground.ipynb) (moved out of `src/`)
  now imports those modules and only demonstrates and explores them.
- 28 tests in [tests/](tests/) cover the new modules; the empty `catan.py` stub
  is gone.

**What's missing:**

- `src/catan_bots/envs/` — Gymnasium env does not exist yet.
- No training pipeline, no baselines beyond catanatron's built-ins.
- `main.py` is still a stub.

**Next steps:**

1. Decide 2-player vs. 4-player scope for v1 env.
2. Run the `scaffold-gym-env` skill to generate the `CatanEnv` skeleton, reusing
   `GameFactory` for construction and `GameReport` for episode diagnostics.
3. Run the `train-rl-agent` skill for a first MaskablePPO smoke run, and
   evaluate it with `Tournament` against the built-in bots.

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
