# catan-bots

Reinforcement-learning bots for the game of Catan: a PettingZoo multi-agent
environment, self-play PPO training, and gameplay analytics.

See [CLAUDE.md](CLAUDE.md) for coding standards and project layout, and
[PROJECT_STATUS.md](PROJECT_STATUS.md) for the current status, latest metrics
and next steps. The [project log](#project-log) keeps the history; dashboards
are not built yet.

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

## Training

Self-play PPO: one shared network plays every learner seat, each seat's moves
are kept as their own trajectory, and past snapshots plus baseline bots fill
some seats. Everything runs from one command line:

```bash
uv run python -m catan_bots.training train --config configs/selfplay_4p_m5pro.json
uv run python -m catan_bots.training resume runs/<run-directory>
uv run python -m catan_bots.training evaluate runs/<run-directory>/checkpoints/latest.pt
uv run python -m catan_bots.training benchmark
```

Change any setting without editing the file with `--set`, for example
`--set total_steps=50000000 num_workers=8`. A run writes everything to
`runs/<run_name>-<timestamp>/`: its config, the git commit it ran on,
checkpoints, league snapshots, `metrics.jsonl`, `eval.jsonl`, TensorBoard
logs and `train.log`, a copy of the console output including any warning or
error that stops the run. Watch it with `uv run tensorboard --logdir runs`.

There is one configuration per machine:

| Config | Machine | Workers | Budget |
|---|---|---|---|
| `configs/selfplay_4p_m5pro.json` | 18-core M5 Pro laptop, 24 GB | 14 | 1B decisions, about 10 hours |
| `configs/selfplay_4p_m5pro_graph.json` | The same, graph network, updates on the GPU | 14 | 1B decisions; see [Policy networks](#policy-networks) for speed |
| `configs/selfplay_4p.json` | 10-core Mac mini, 16 GB | 8 | 20M decisions |
| `configs/smoke.json` | Any; checks the pipeline in seconds | 2 | 8k decisions |

### Policy networks

`architecture` picks the network; everything else (masking, sampling, PPO,
the league, evaluation) is shared. Both read the same 1,413-number
observation and output the same 587 action scores.

- **`"mlp"`** (default):
  [MlpActorCritic](src/catan_bots/training/policy.py). The observation is one
  flat vector through two 256-wide layers; 579k parameters. Every input number
  has its own weights, so what it learns about one intersection does not
  carry over to another.
- **`"graph"`**:
  [GraphActorCritic](src/catan_bots/training/graph_policy.py). The board
  blocks are read as rows (54 intersections, 72 road slots, 19 hexes) and
  passed through `graph_rounds` rounds of message passing over the board's
  adjacency ([board_graph.py](src/catan_bots/envs/board_graph.py)). One set of
  weights is shared by every row of a kind, so a lesson learned at one
  intersection applies to all of them. Settlement and city scores come from
  each intersection's vector, road scores from each road slot's, robber
  scores from each hex's, and everything else from a pooled summary.
  417k parameters at `graph_width` 64 and 3 rounds.

Snapshots record their architecture, so a graph run can evaluate against, or
draw league opponents from, MLP checkpoints and the other way round. Old
snapshots without the field load as MLPs.

The graph network costs more per decision. Measured on the M5 Pro:

| | MLP | Graph (64 wide, 3 rounds) |
|---|---|---|
| One move, one CPU thread | 0.06 ms | 0.42 ms |
| Collecting, 14 workers | 37.9k decisions/s | 10.1k decisions/s |
| One PPO update (65k decisions × 4 epochs), CPU | about 1 s | about 37 s |
| The same update on the Apple GPU (`update_device`) | about 0.4 s | about 19 s |
| Training including updates | about 27k decisions/s | about 3.3k decisions/s |

The update is limited by memory traffic over 54 to 72 rows per decision, not
by CPU threads (6 and 18 threads take the same time), so the graph config
runs updates on the GPU with `"update_device": "mps"`. Games are always
played on CPU workers; the network moves to the GPU for each update and back.
Checkpoints always store CPU tensors, so a run resumes on any machine.

### Who sits at the table

[src/catan_bots/training/league.py](src/catan_bots/training/league.py) draws the
seating at the start of each game; seats do not change during a game. The
numbers below are from `configs/selfplay_4p_m5pro.json`.

- **Snapshots.** Every 50 iterations the policy is saved to the run's `league/`
  directory. The newest 50 are kept, covering roughly the last 200M decisions.
  These are separate from the resume checkpoints, written every 25 iterations.
  Both are saved under a temporary name and renamed into place, so an
  interrupted save never leaves a corrupt file.
- **Lineups.** 80% of games put the current policy in every seat. The rest
  replace one to three seats, always leaving at least one learner.
- **Replaced seats.** Each goes to a baseline bot with 25% probability, otherwise
  to a snapshot picked uniformly. Until the first snapshot exists, every replaced
  seat goes to a baseline.
- **Learning.** Only learner seats produce training data; snapshots and baselines
  are fixed opponents.

### Evaluation

Every `eval_every` iterations (100, about every 5 minutes on the M5 Pro) and
once at the end, training pauses and the current policy plays `eval_games`
games in one seat against three copies of each opponent; chance is 25%. Each
result goes to `eval.jsonl` and TensorBoard as `eval/win_rate_vs_<label>`.

- **Baselines** (`eval_opponents`): random, `GreedySettler`, `GreedyTrader`.
  These stop being informative once the policy beats them.
- **Its own past** (`eval_oldest_snapshot`, label `league_oldest`): the oldest
  snapshot still in the league. Well above 25% means it still beats where it
  was a few hours earlier; a slide towards 25% means it has plateaued, and below
  25% means it has forgotten something.
- **Fixed references** (`eval_references`, labels `reference_0`, ...): policy
  files that never change, usually the final checkpoint of an earlier run. They
  give a bar that stays put across runs. A missing file stops the run before it
  starts.

`evaluate --opponents` also accepts policy files, so two checkpoints can be
compared directly:

```bash
uv run python -m catan_bots.training evaluate new/latest.pt --opponents old/latest.pt
```

### Learning-rate schedule and KL limit

- **`final_learning_rate_fraction`** (0.1 on the M5 Pro): the learning rate falls
  linearly from `learning_rate` to this share of it at `total_steps`. Large steps
  help early; late in a run they mostly add noise to a policy that is already
  good. 1.0 keeps the rate constant.
- **`target_kl`** (0.02): KL divergence measures how far the updated policy's
  action probabilities have moved from the policy that played the games. PPO's
  clipping stops rewarding big moves but does not prevent them across several
  epochs. Before each minibatch step the trainer measures this divergence and,
  if it exceeds 1.5 × `target_kl`, ends the update for that batch. It is a
  safety net against a single bad update wrecking the policy, and it should
  rarely fire; `train/kl_stopped` records when it does. 0 disables it.

### On an M5 Pro laptop

`configs/selfplay_4p_m5pro.json` uses 14 workers. Measured with `benchmark`,
14 and 16 workers give nearly the same throughput (43.8k vs 44.9k decisions/s
collecting), so 14 leaves cores for the PPO update and the laptop itself.
Training including updates runs at about 27k decisions/s.

Keep the laptop plugged in with the lid open, and keep it awake:

```bash
caffeinate -is uv run python -m catan_bots.training train --config configs/selfplay_4p_m5pro.json
```

### On a Mac mini

`configs/selfplay_4p.json` is sized for a 10-core Apple Silicon Mac mini with
16 GB of RAM: 8 worker processes, leaving two cores for the learner and the
system. Each worker needs a few hundred MB. It predates the learning-rate
schedule, KL limit and extra evaluations, which are off unless set.

1. Install [uv](https://docs.astral.sh/uv/) (`brew install uv`), then clone
   the repository and run `uv sync` in it.
2. Measure the machine once: `uv run python -m catan_bots.training benchmark`.
   It prints decisions per second for one worker and for all of them.
3. Start training, keeping the Mac awake for the whole run:

   ```bash
   caffeinate -is uv run python -m catan_bots.training train --config configs/selfplay_4p.json
   ```

4. If the run stops for any reason, continue it from its last checkpoint with
   `resume`; nothing before that checkpoint is lost. Pass
   `--set total_steps=...` to train past the original budget.

## Project log

### 2026-11-24 — First medium run stopped by a one-sample minibatch

**Status:** The 200M-decision MLP run stopped at iteration 98 (7.4M
decisions, 5 minutes in) with NaN weights. The cause is fixed and the run can
resume from its iteration-75 checkpoint.

**Cause:** Each iteration collects at least 65,536 decisions, but workers
finish the games they are in, so the batch size varies. The update cut the
shuffled batch into chunks of 8,192 and the remainder became a last, smaller
chunk. In iteration 98 the batch was one more than a multiple of 8,192. The
one-sample chunk's advantage standard deviation was NaN, which made the loss,
the gradient step and every weight NaN; the next minibatch failed with
`ValueError: Expected parameter logits ... invalid values`. The chance is
about 1 in 8,192 per iteration, about 30% over a 3,000-iteration run.

**What changed:**

- **Equal minibatches.** Batches are split into `size // minibatch_size`
  chunks whose sizes differ by at most one, so none is smaller than
  `minibatch_size`. `minibatch_size` must be at least 2.
- **Non-finite guard.** A minibatch whose loss is not finite is skipped, not
  applied, and counted in `train/skipped_minibatches`.
- **`train.log`.** Console output, warnings and the traceback of any error
  that stops training are written to the run directory.
- **Clean resume.** Resuming drops `metrics.jsonl` and `eval.jsonl` lines
  logged after the checkpoint, since those iterations are trained again.

### 2026-11-24 — Graph-network policy

**Status:** A second policy network, `architecture: "graph"`, is in place and
tested end to end, alongside the MLP. It trains about 8× slower per decision
on the M5 Pro, and in its first 1.6M decisions it learned more slowly than the
MLP did. Next: the user picks the network for the medium run. Current status
now lives in [PROJECT_STATUS.md](PROJECT_STATUS.md), which every session
updates.

**What changed:**

- **Graph network** ([graph_policy.py](src/catan_bots/training/graph_policy.py),
  [board_graph.py](src/catan_bots/envs/board_graph.py)): message passing over
  the board's intersections, road slots and hexes, with shared weights per
  kind and per-row action heads. See [Policy networks](#policy-networks).
- **One interface, two networks.** `ActorCritic` is now an abstract base;
  `MlpActorCritic` is the old network unchanged. `build_policy` builds the
  configured one and `policy_from_snapshot` rebuilds any snapshot, so runs
  can mix architectures in the league and in evaluation.
- **`update_device`.** PPO updates can run on the Apple GPU; games stay on CPU
  workers. Checkpoints store CPU tensors.
- **Benchmark flag.** `benchmark --set architecture=graph` measures either
  network.
- **Status log.** [PROJECT_STATUS.md](PROJECT_STATUS.md) holds the current
  status, metrics, assumptions and next steps. [CLAUDE.md](CLAUDE.md) requires
  every session to read it first and update it, and a `Stop` hook in
  [.claude/settings.json](.claude/settings.json) blocks the end of a turn while
  changed files are newer than it.

**Check run** (`selfplay_4p_m5pro_graph.json`, 1.65M decisions, 20
iterations, 9 minutes, learning rate decayed over those 1.65M). Compared with
the MLP check run of 2026-11-11 at the same number of decisions, 200 games
each:

| Network, decisions | vs random | vs `GreedySettler` | vs `GreedyTrader` | vs oldest snapshot |
|---|---|---|---|---|
| MLP, 1.63M | 87.0% [81.6, 91.0] | 45.5% [38.7, 52.4] | 18.0% [13.3, 23.9] | 73.5% [67.0, 79.1] |
| Graph, 1.65M | 77.0% [70.7, 82.3] | 29.5% [23.6, 36.2] | 11.5% [7.8, 16.7] | 71.5% [64.9, 77.3] |

- **Not ahead yet.** The graph network trails on every baseline at this
  point. One seed and a short run settle nothing, and the two runs' learning
  rates decayed over different budgets, but the hoped-for sample efficiency
  did not show in the first 1.6M decisions.
- **Throughput.** 6.5 s collecting and 18.7 s updating on the GPU per
  65k-decision iteration, about 3.3k decisions/s against the MLP's 27k. The
  update dominates.
- **KL.** The largest KL was 0.005 and the limit never fired, so as with the
  MLP the learning rate has room to rise.

### 2026-11-11 — Ready for a long run on the M5 Pro

**Status:** The trainer is prepared for runs of hundreds of millions of
decisions. Next: a medium run of about 2 hours to see where the policy
plateaus, then the 1B-decision run with that run's final checkpoint as a fixed
reference.

**What changed:**

- **Atomic snapshots.** League snapshots are written to a temporary file and
  renamed, like checkpoints. Leftover partial files are deleted when a league
  is opened. Before, a crash mid-save left a corrupt snapshot that a resumed
  run would load and fail on.
- **M5 Pro config** (`configs/selfplay_4p_m5pro.json`): 14 workers, 64k
  decisions per iteration, and snapshot, checkpoint and evaluation intervals
  scaled for a 1B-decision budget.
- **Evaluation that does not saturate.** Evaluations can include fixed
  reference policies and the oldest league snapshot (see
  [Evaluation](#evaluation)).
- **Learning-rate decay and a KL limit** (see
  [Learning-rate schedule and KL limit](#learning-rate-schedule-and-kl-limit)).
- **Dependencies.** Removed `stable-baselines3` and `sb3-contrib`, which nothing
  imported. The masked PPO is this repo's own: illegal actions' logits are set
  to -1e9 in [policy.py](src/catan_bots/training/policy.py) before sampling.

**Check run** (3M decisions, 38 iterations, 2.5 minutes, learning rate decayed
over those 3M steps). Evaluation as in [Evaluation](#evaluation), 200 games
each:

| Iteration | vs random | vs `GreedySettler` | vs `GreedyTrader` | vs oldest snapshot |
|---|---|---|---|---|
| 20 | 87.0% [81.6, 91.0] | 45.5% [38.7, 52.4] | 18.0% [13.3, 23.9] | 73.5% [67.0, 79.1] |
| 38 | 99.0% [96.4, 99.7] | 54.0% [47.1, 60.8] | 38.0% [31.6, 44.9] | 90.0% [85.1, 93.4] |

At 3M decisions the policy already beats a table of `GreedyTrader` bots
(interval above 25%), which the 919k-decision run did not. Other observations:

- **Throughput.** About 27k decisions/s including updates: 2.1 s collecting
  and 1.1 s updating per iteration. An evaluation round of four opponents takes
  about 16 s.
- **KL.** The largest measured KL was 0.004, far below the 0.03 stop point, so
  the limit never fired. Such small moves suggest the learning rate has room to
  rise; worth testing in the medium run.
- **Trading fell** from about 115 to 24 completed trades per game while win
  rates rose, but offers only fell from about 600 to 470 per game. Most offers
  now go unaccepted, and each is a decision the policy is trained on. Whether
  it is learning to trade well or just learning that trades fail is worth
  checking.

### 2026-11-08 — Status review and documentation sync

**Status:** Documentation only; no code changed. The environment, trading,
redacted observations and self-play trainer are in place. The only recorded
training is the 14-minute check below (about 919k decisions of a planned 20M),
which has not clearly beaten a table of `GreedyTrader` bots. Older entries in
this log are kept as written, so their references to Gymnasium and MaskablePPO
describe plans at the time.

**Roadmap** (the numbering approved on 2026-10-08):

| Step | Item | State |
|---|---|---|
| 1 | Redacted per-seat observations | Done |
| 2 | PettingZoo AEC environment | Done |
| 3 | Strong non-RL baselines (value function, depth-2 alpha-beta) | Not started |
| 4 | Self-play training with league play | Built; long run not recorded |
| 5 | Player-to-player trading | First phase done; counter-offers planned |
| 6 | Rules fidelity (dev-card timing, turn-limit truncation) | Done |

Step 4 changed from the plan: training uses this repo's own PPO rather than
MaskablePPO, because each seat needs its own trajectory.

**Known gaps:**

- **Baselines can see hidden information.** `BotAgent` hands bots the full game.
  The greedy bots were not found reading opponents' hands, but a search-based
  baseline (step 3) must be restricted or it will play with knowledge the
  learner lacks.
- **Encoded-vector test coverage.** Only opponents' hand types are tested on the
  encoded vector. Development cards, VP cards and deck order are tested on the
  redacted view the encoder is built from.
- **No memory.** The policy sees snapshots with no action history, so it cannot
  count cards as a human can. This limits strength but leaks nothing.
- **Snapshot window.** The 20 kept snapshots cover the last 200 iterations of
  about 610 in a full run, so the oldest opponents leave the pool.
- **Unused dependencies.** `stable-baselines3` and `sb3-contrib` are not imported
  by the package, tests or notebook.
- **Stub.** `main.py` only prints a greeting; the real entry point is
  `python -m catan_bots.training`.

**Next:** run the full 20M-step training on the Mac mini, aiming to beat a table
of `GreedyTrader` bots, then build the stronger baselines (step 3).

### 2026-10-20 — Self-play trainer, and a first run that learns

**Status:** Training works end to end and learns. Next: long runs on the Mac
mini, aiming to beat a table of `GreedyTrader` bots, then stronger baselines.

**Compute:** start on the 10-core Mac mini (16 GB), which costs nothing extra.
Training is CPU-bound, not GPU-bound, and scales linearly with cores. If runs
outgrow it, the fixed-monthly option is a rented multi-core server.

**Speed:** profiling showed 30% of each step went on finding legal trade
offers and 30% on building and encoding observations. Grouping offers by give
side, sparse observation encoding (3.4x faster) and caching static board parts
(1.5x faster) took a policy-driven process from 481 to 728 steps/s. Each
rewrite was checked against the previous version on 44,679 live game states.
Ruff is now a pinned dev dependency, so every machine lints the same way.

**Trainer** ([src/catan_bots/training/](src/catan_bots/training/)): one shared
network plays every learner seat. Each seat's moves form their own trajectory,
so credit and each seat's end-of-game reward reach the right player — unlike
PettingZoo's SB3 bridge, which merges all seats into one stream. 80% of games
are pure self-play; the rest put past snapshots or baseline bots in one to
three seats. Checkpoints are atomic and runs resume.

**First learning check** (this repo's 4-core dev machine, 14 minutes of wall
clock including evaluations, about 16 seconds per iteration, VP shaping on).
Evaluation puts the policy in one seat against three copies of a baseline;
chance is 25%.

| Iteration | Decisions | vs random | vs `GreedySettler` |
|---|---|---|---|
| 15 | 320k | 48.3% [36.2, 60.7] | 28.3% [18.5, 40.8] |
| 30 | 622k | 78.3% [66.4, 86.9] | 36.7% [25.6, 49.3] |
| 45 | 919k | 98.3% [91.1, 99.7] | 51.7% [39.3, 63.8] |

Iteration 45 over 200 games per opponent: `GreedySettler` 55.5% [48.6, 62.2],
`WeightedRandom` 45.5% [38.7, 52.4], `GreedyTrader` 29.5% [23.6, 36.2]. A
table of three trading bots, which trade with each other, is not yet beaten.

### 2026-10-08 — Player-to-player trading

**Status:** Step 5 done, ahead of baselines and training as agreed, so no
model is trained on a game without trading. Next: strong baselines (step 3).

**Rules** ([src/catan_bots/rules/trading.py](src/catan_bots/rules/trading.py)):

- Only the active player, after rolling, proposes. Every other seat answers
  in play order, then the proposer trades with one acceptor or cancels.
- No gifts and no like-for-like trades. Offers and answers are public.
- Agreed limits: up to two cards per side (230 offers), three offers per
  turn. Counter-offers are a planned second phase.

**Environment:** 587 actions (236 trade slots) and 1,413 features (a 28-feature
trade block). On by default; `trading=None` switches it off without changing
the spaces.

**Bots:** a `TradingPlayer` interface and `GreedyTraderBot`, which trades
towards its nearest build, refuses seats close to winning, and settles with
the trailing acceptor. Bots without the interface decline every offer.

**Measured effect** (2,000 seat-balanced games: two `GreedyTraderBot`s against
two `GreedySettlerBot`s, identical build logic, only trading differing):
traders won **62.0%** of games, 95% CI [59.9%, 64.1%], against 50% expected.
That came from only 5.8 completed trades per game (121.8 offers). Because
non-traders refuse everything, the two traders effectively formed a trading
partnership against the other two — the coalition dynamic that makes
4-player trading strategically interesting.


### 2026-10-08 — Multi-agent environment

**Status:** Step 2 done. The environment supports 2, 3 and 4 players behind
`num_players` (default 4, the focus). Next: player-to-player trading, ahead of
baselines and training so no model is trained on a game without it.

**What it is:** [src/catan_bots/envs/](src/catan_bots/envs/), a PettingZoo AEC
environment that passes PettingZoo's own `api_test` at every size and its
`seed_test`.

- **Turn structure.** The decider is always the engine's
  `state.current_color()`, so out-of-turn discards on a seven are handled.
- **Actions.** A fixed `Discrete(351)` table with a mask, shared by every seat
  and player count. Robbery victims are seat offsets, never colours.
- **Observations.** 1,385 features in [0, 1], encoded only from the redacted
  `Observation`, so the vector cannot leak hidden information. Seats are
  relative to the observer and padded to four.
- **Discarding.** With no policy, the agent discards card by card — five
  options per step, any legal selection reachable. With a policy, discards
  resolve automatically.
- **Rules.** Development-card timing is enforced.
- **Reward.** Zero-sum and terminal: +1 to the winner, -1/(n-1) to the rest.
  Hitting `turn_limit` is a truncation with zero reward, not an outcome, so
  stalling earns a losing agent nothing.
- **Randomness.** Each environment owns a `RandomStream`, fixing the
  global-RNG interleaving problem: environments stepped alternately in one
  process now reproduce exactly from their seeds.

**Throughput:** ~1,800 steps/s for 4 players with masked random agents, single
core. Encoding is plain Python and is the obvious optimisation target once
training shows where time goes.


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
