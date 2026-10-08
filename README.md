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

### 2026-10-08 — Official dev-card timing and redacted observations

**Status:** Roadmap approved. Step 1 (redacted observations) and the
development-card timing fix are done. Next: the PettingZoo AEC environment.

**Approved roadmap:**

1. Redacted per-seat observations — done.
2. PettingZoo AEC environment driven by `state.current_color()`.
3. Strong non-RL baselines (value function, depth-2 alpha-beta).
4. Self-play training: shared-policy MaskablePPO, then league play.
5. Player-to-player trading — required to replicate the real game.
6. Rules fidelity: dev-card timing (done); `TURNS_LIMIT` handled as
   truncation, not a loss, inside the environment.

**Development-card timing.** Official rules forbid playing a development card
the turn it was bought. Catanatron only enforces one card per turn, and over
300 games about **half of all development-card plays** were these illegal
same-turn plays (55% in 2-player, 50% in 4-player).
[src/catan_bots/rules/dev_cards.py](src/catan_bots/rules/dev_cards.py) derives
the cards bought this turn from the action log and strips their plays before
every decision. It is on by default in `GameRunner` and `GameFactory`, which
now always plays through the runner; `enforce_dev_card_timing=False`
reproduces the stock engine for ablations.

**Redacted observations.** [src/catan_bots/observation/](src/catan_bots/observation/)
defines exactly what a seat may know. Opponents' hands and development cards
appear only as counts; the deck's order and hidden VP cards are never exposed;
board ownership is recorded as seat offsets so one shared policy sees the
board the same way from every chair. Leak-invariance tests change each hidden
fact and assert no other seat's view moves; four deliberately injected leaks
were each caught by the matching test.


### 2026-10-08 — Discard-on-seven is now a modelled decision

**Status:** Discard rules implemented and measured. Masked observations,
PettingZoo AEC and trading are still unimplemented (analysis only).

**Why:** catanatron does not model this decision. `discard_possibilities`
returns one placeholder action and the engine throws away a uniformly random
half of the hand, with a TODO explaining that enumerating the choice would
explode the decision tree. That explosion only happens if same-resource cards
are treated as distinguishable. Measured over 729 real discard events,
enumerating resource *multisets* gives a median of 11 options and never more
than 100, against a mean of 1,773 per-card combinations.

**What changed:**

- [src/catan_bots/rules/discard.py](src/catan_bots/rules/discard.py) —
  `enumerate_discards` (the full multiset choice, for exposing discard to a
  learning agent), a `DiscardPolicy` ABC whose `resolve` validates selections
  before the engine sees them, `UniformRandomDiscard` (seeded engine-parity
  baseline) and `BuildPlanDiscard` (the better rule).
- [src/catan_bots/runner.py](src/catan_bots/runner.py) — `GameRunner` owns the
  play loop and resolves discard prompts through a policy, since a discard
  choice cannot travel through `Player.decide`.
- `GameFactory` takes an optional `discard_policy`; with none, catanatron's
  random discard stays in place so engine parity remains the baseline.

**Measured effect** (same bot in all seats, only the discard rule differing):

- Win rate over 3000 seat-balanced games: **52.6%**, 95% CI [50.8%, 54.4%].
- Paired on 1413 identical positions, the kept hand can afford a build
  **33.5%** of the time vs **25.7%** for random (McNemar z = 6.0).

**Engine caveat found:** catanatron picks discarders against
`state.discard_limit` but chains to the next discarder against a hardcoded
`> 7`, so any non-default `discard_limit` makes it skip discarders.

**Next steps:**

1. Redacted per-seat observations (the hidden-information leak).
2. A PettingZoo AEC wrapper driven by `state.current_color()`.
3. A minimal domestic-trade protocol.


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
