# Project status

The current state of catan-bots for the next session: read this first, and
update it before finishing any prompt that changes something (see
[CLAUDE.md](CLAUDE.md#project-status-log)).

**Last updated:** 2026-11-24, by Claude, after adding the graph network and
its check run.

## Status

- **Branch `prep/long-run`, not yet committed.** It holds the long-run
  preparation (atomic snapshots, M5 Pro config, LR decay, KL limit, reference
  and oldest-snapshot evaluation, removed SB3) and the graph-network policy.
  The user commits and opens the PR themselves.
- **Ready for the medium run** (about 200M decisions) with either network.
  Which network to use is the user's decision; see
  [Open questions](#open-questions).
- 259 tests pass and ruff is clean, as of 2026-11-24.

## Current performance

No long run has been recorded. Every figure below comes from a short check
run on the M5 Pro. Win rates are from 200 games in one seat against three
copies of the opponent (chance is 25%), with 95% Wilson intervals.

**MLP check run** (2026-11-11, 3M decisions, 38 iterations, learning rate
decayed over the 3M):

| Iteration (decisions) | vs random | vs GreedySettler | vs GreedyTrader | vs oldest snapshot |
|---|---|---|---|---|
| 20 (1.6M) | 87.0% [81.6, 91.0] | 45.5% [38.7, 52.4] | 18.0% [13.3, 23.9] | 73.5% [67.0, 79.1] |
| 38 (3.0M) | 99.0% [96.4, 99.7] | 54.0% [47.1, 60.8] | 38.0% [31.6, 44.9] | 90.0% [85.1, 93.4] |

**Graph check run** (2026-11-24, `selfplay_4p_m5pro_graph.json`, 1.65M
decisions, 20 iterations, learning rate decayed over the 1.65M):

| Iteration (decisions) | vs random | vs GreedySettler | vs GreedyTrader | vs oldest snapshot |
|---|---|---|---|---|
| 20 (1.65M) | 77.0% [70.7, 82.3] | 29.5% [23.6, 36.2] | 11.5% [7.8, 16.7] | 71.5% [64.9, 77.3] |

At the same number of decisions the graph network trails the MLP on every
baseline. That's one seed over a short run, so it settles nothing.

**Throughput on the M5 Pro** (measured 2026-11-24):

| | MLP | Graph (64 wide, 3 rounds) |
|---|---|---|
| Collecting, 14 workers | 37.9k decisions/s | 10.1k decisions/s |
| PPO update per 65k-decision iteration, CPU | about 1 s | about 37 s |
| The same on the Apple GPU (`update_device: mps`) | about 0.4 s | 18.7 s in the check run |
| Training including updates | about 27k decisions/s (2026-11-11 run) | about 3.3k decisions/s (2026-11-24 run) |

## What has been done

- **Environment:** a PettingZoo AEC environment with 1,413 observation
  features and 587 actions. It covers trading, sequential discards,
  official dev-card timing and turn-limit truncation. Observations are
  redacted per seat, and the hidden-information tests pass.
- **Training:**
  - self-play PPO written in this repo, with masked logits, per-seat
    trajectories and potential-based VP shaping;
  - a league of past snapshots and baseline bots;
  - atomic checkpoints and snapshots, and resume;
  - linear learning-rate decay and a KL early stop;
  - evaluation against baselines, fixed reference files and the oldest
    league snapshot.
- **Policy networks:** `architecture` selects one of two.
  - `"mlp"`: `MlpActorCritic`, two 256-wide layers.
  - `"graph"`: `GraphActorCritic`, message passing over a board graph of 54
    intersections, 72 road slots and 19 hexes. Its neighbor tables live in
    `envs/board_graph.py`, and its per-row heads are mapped back to
    action-table order.
  - Snapshots record their architecture, so either kind loads in a run of
    the other.
  - `update_device` runs PPO updates on the Apple GPU.
- **Configs:**
  - `selfplay_4p_m5pro.json`: MLP, 14 workers, 1B-decision budget.
  - `selfplay_4p_m5pro_graph.json`: the graph network, with updates on MPS.
  - `selfplay_4p.json`: the Mac mini config.
  - `smoke.json`: a seconds-long pipeline check.
- **Process:** this file, the CLAUDE.md rule that keeps it current, and a
  Stop hook (`.claude/hooks/check_status_log.sh`) that enforces it.

## Assumptions in force

- **One board.** The base map's geometry is fixed. The graph tables and the
  action table assume it; only resources, numbers and ports are shuffled
  between games.
- **No memory.** The policy sees the current observation only, with no move
  history or card counting.
- **Messages scaled by width.** Graph messages are neighbor sums divided by
  the table width (3 or 6), not means, so a node touching three tiles gets a
  bigger message than a coastal node touching one.
- **Shared network.** Actor and critic share the network in both
  architectures (no `shared_torso` option yet).
- **MPS is not bit-identical to the CPU.** It matches within 1e-3 after one
  update, as tested. Runs that update on MPS are reproducible only
  approximately.
- **Baselines see the whole game.** The greedy bots get the full game
  object, but no leak was found. A search baseline would have to be
  restricted to the redacted view.

## Open questions

- **Which network for the medium run?** The MLP trains about 8× faster per
  decision on the M5 Pro: 200M decisions take about 2 hours instead of about
  17. The graph network was meant to need fewer decisions, but at 1.65M it
  was behind. A fair test is a longer graph run (for example 20M decisions,
  about 1.7 hours), compared with the MLP at equal wall-clock time and at
  equal decisions.
- **The learning rate.** The largest KL was 0.004 in the MLP check run and
  0.005 in the graph check run, so the learning rate may be too low for both.
  Worth testing in the medium run.
- **Trading.** Completed trades fell from about 115 to 24 per game while
  offers stayed around 470. It's unclear whether the policy is learning to
  trade well or learning that offers fail.

## Next steps

1. **Commit and PR.** The user commits `prep/long-run` and opens the PR.
2. **Medium run.** About 200M decisions with the chosen config, under
   `caffeinate -is`. Record where it plateaus.
3. **1B-decision run.** Use the medium run's final checkpoint as an
   `eval_references` entry.
4. **Graph network speed.** The GPU update is 75% of each iteration. Options:
   - a smaller `graph_width`: 32 wide with 2 rounds measured about 5 s per
     update on synthetic data;
   - overlapping collection with the update;
   - batched inference across several games per worker.
5. **Memory features.** Card counting from public events, such as
   resources each opponent must hold and dev cards remaining.
6. **More logging.** Explained variance, and a `shared_torso` experiment.
7. **Stronger baselines.** A value-function bot and depth-2 search,
   restricted to the redacted view.
8. **Clean up `main.py`.** It is still a stub that prints a greeting.
