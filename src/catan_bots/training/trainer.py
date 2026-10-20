"""Run, checkpoint and resume a self-play training run.

A run lives in one directory:

    runs/<run_name>-<timestamp>/
    ├── config.json          every setting, reloadable
    ├── metadata.json        git commit, machine, versions, start time
    ├── checkpoints/latest.pt
    ├── league/snapshot_*.pt past policies used as opponents
    ├── metrics.jsonl        one line per iteration
    ├── eval.jsonl           one line per evaluation
    └── tensorboard/         the same metrics for TensorBoard

Checkpoints are written atomically (to a temporary file, then renamed), so a
run interrupted at any moment — a sleeping laptop, a reclaimed cloud machine
— resumes from its last checkpoint.
"""

import json
import logging
import math
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from catan_bots.training import workers
from catan_bots.training.config import TrainingConfig
from catan_bots.training.evaluator import EvaluationResult, Evaluator
from catan_bots.training.league import League
from catan_bots.training.policy import ActorCritic
from catan_bots.training.ppo import PPOUpdater
from catan_bots.training.rollout import RolloutBatch, RolloutStats

LOGGER = logging.getLogger(__name__)
CHECKPOINT_NAME = "latest.pt"


class Trainer:
    """Own one training run: its policy, optimiser, league and files.

    Create with `Trainer.create` for a new run or `Trainer.resume` to
    continue one, then call `train`.

    Attributes:
        config: The run's configuration.
        run_dir: The run's directory.
        policy: The policy being trained.
        updater: PPO updates for the policy.
        league: Past snapshots used as opponents.
        iteration: Updates completed.
        global_step: Learner decisions trained on.
    """

    def __init__(self, config: TrainingConfig, run_dir: Path) -> None:
        """Build the run's policy and optimiser; use `create` or `resume`.

        Args:
            config: The run's configuration.
            run_dir: The run's directory, which must exist.
        """
        self.config = config
        self.run_dir = run_dir
        env = workers.make_env(config)
        torch.manual_seed(config.seed)
        self.policy = ActorCritic(
            env.encoder.size, env.table.size, config.hidden_sizes
        ).eval()
        self.updater = PPOUpdater(self.policy, config)
        self.league = League(run_dir / "league", config.max_snapshots)
        self.rng = np.random.default_rng(config.seed)
        self.iteration = 0
        self.global_step = 0
        self._evaluated_at = -1

    @classmethod
    def create(cls, config: TrainingConfig) -> "Trainer":
        """Start a new run in a fresh, timestamped directory.

        Args:
            config: The run's configuration.

        Returns:
            The trainer, ready to `train`.
        """
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        run_dir = Path(config.output_dir) / f"{config.run_name}-{stamp}"
        (run_dir / "checkpoints").mkdir(parents=True)
        config.save(run_dir / "config.json")
        (run_dir / "metadata.json").write_text(
            json.dumps(run_metadata(config), indent=2) + "\n"
        )
        return cls(config, run_dir)

    @classmethod
    def resume(
        cls, run_dir: Path, overrides: dict[str, Any] | None = None
    ) -> "Trainer":
        """Continue a run from its latest checkpoint.

        Args:
            run_dir: The run's directory.
            overrides: Settings to change, such as `total_steps` to train
                longer or `num_workers` on a different machine. Saved back to
                `config.json`.

        Returns:
            The trainer, restored and ready to `train`.

        Raises:
            FileNotFoundError: If the run has no checkpoint yet.
        """
        config = TrainingConfig.load(run_dir / "config.json")
        if overrides:
            config = config.with_overrides(overrides)
            config.save(run_dir / "config.json")
        trainer = cls(config, run_dir)
        trainer._restore(
            torch.load(run_dir / "checkpoints" / CHECKPOINT_NAME, weights_only=False)
        )
        return trainer

    @property
    def checkpoint_path(self) -> Path:
        """Where the latest checkpoint is written."""
        return self.run_dir / "checkpoints" / CHECKPOINT_NAME

    def train(self) -> Path:
        """Train until `total_steps`, checkpointing and evaluating on schedule.

        Returns:
            The final checkpoint's path.
        """
        config = self.config
        writer = SummaryWriter(
            self.run_dir / "tensorboard", purge_step=self.iteration + 1
        )
        LOGGER.info("Run directory: %s", self.run_dir)
        with workers.make_pool(config, config.num_workers) as pool:
            evaluator = Evaluator(pool, config.num_workers, config.num_players)
            while self.global_step < config.total_steps:
                self._iterate(pool, writer)
                self._on_schedule(evaluator, writer)
            self.save_checkpoint()
            if config.eval_every and self._evaluated_at != self.iteration:
                self._evaluate(evaluator, writer)
        writer.close()
        LOGGER.info("Finished at step %s: %s", self.global_step, self.checkpoint_path)
        return self.checkpoint_path

    def _iterate(self, pool: Any, writer: SummaryWriter) -> None:
        """Collect one batch of games and update the policy once.

        Args:
            pool: Worker pool.
            writer: TensorBoard writer.
        """
        started = time.perf_counter()
        self.iteration += 1
        batch, stats = self._collect(pool)
        collected = time.perf_counter()
        metrics = self.updater.update(batch, self.rng)
        self.global_step += len(batch)
        timing = {
            "collect_seconds": collected - started,
            "update_seconds": time.perf_counter() - collected,
        }
        self._log(writer, stats, metrics, timing)

    def _on_schedule(self, evaluator: Evaluator, writer: SummaryWriter) -> None:
        """Snapshot, checkpoint and evaluate when their intervals come round.

        Args:
            evaluator: Evaluator bound to the worker pool.
            writer: TensorBoard writer.
        """
        config, iteration = self.config, self.iteration
        if iteration % config.snapshot_every == 0:
            self.league.add(self.policy, iteration)
        if iteration % config.checkpoint_every == 0:
            self.save_checkpoint()
        if config.eval_every and iteration % config.eval_every == 0:
            self._evaluate(evaluator, writer)

    def _collect(self, pool: Any) -> tuple[RolloutBatch, RolloutStats]:
        """Gather a batch from every worker in parallel.

        Args:
            pool: Worker pool.

        Returns:
            The combined batch and game totals.
        """
        config = self.config
        weights = self.policy.snapshot()
        snapshots = tuple(str(path) for path in self.league.snapshots)
        min_steps = math.ceil(config.steps_per_iteration / config.num_workers)
        tasks = [
            workers.CollectTask(weights, min_steps, self._seed(index), snapshots)
            for index in range(config.num_workers)
        ]
        results = pool.map(workers.collect, tasks)
        stats = RolloutStats()
        for _, worker_stats in results:
            stats.merge(worker_stats)
        return RolloutBatch.concatenate([batch for batch, _ in results]), stats

    def _seed(self, index: int) -> int:
        """Derive a distinct, reproducible seed for one worker task.

        Args:
            index: Worker index.

        Returns:
            A seed unique to this run, iteration and worker.
        """
        sequence = np.random.SeedSequence([self.config.seed, self.iteration, index])
        return int(sequence.generate_state(1)[0])

    def _evaluate(
        self, evaluator: Evaluator, writer: SummaryWriter
    ) -> list[EvaluationResult]:
        """Evaluate the policy against every configured baseline.

        Args:
            evaluator: Evaluator bound to the worker pool.
            writer: TensorBoard writer.

        Returns:
            One result per baseline.
        """
        self._evaluated_at = self.iteration
        weights = self.policy.snapshot()
        results = []
        for opponent in self.config.eval_opponents:
            result = evaluator.evaluate(
                weights, opponent, self.config.eval_games, self._seed(10_000)
            )
            results.append(result)
            LOGGER.info("[eval %s] %s", self.iteration, result.summary())
            writer.add_scalar(
                f"eval/win_rate_vs_{opponent}", result.win_rate, self.global_step
            )
            self._append_jsonl(
                "eval.jsonl",
                {
                    "iteration": self.iteration,
                    "step": self.global_step,
                    **result.to_dict(),
                },
            )
        return results

    def _log(
        self,
        writer: SummaryWriter,
        stats: RolloutStats,
        metrics: dict[str, float],
        timing: dict[str, float],
    ) -> None:
        """Record one iteration to TensorBoard, `metrics.jsonl` and the console.

        Args:
            writer: TensorBoard writer.
            stats: Game totals for the iteration.
            metrics: PPO losses and diagnostics.
            timing: Seconds spent collecting and updating.
        """
        seconds = timing["collect_seconds"] + timing["update_seconds"]
        record = {
            "iteration": self.iteration,
            "step": self.global_step,
            "games": stats.games,
            "mean_turns": stats.turns / max(stats.games, 1),
            "truncated_games": stats.truncated,
            "offers_per_game": stats.offers / max(stats.games, 1),
            "trades_per_game": stats.trades / max(stats.games, 1),
            "mixed_win_ratio": stats.mixed_learner_wins
            / max(stats.mixed_learner_expected, 1e-9),
            "decisions_per_second": stats.decisions / max(seconds, 1e-9),
            **metrics,
            **timing,
        }
        for name, value in record.items():
            if name not in ("iteration", "step"):
                writer.add_scalar(f"train/{name}", value, self.global_step)
        self._append_jsonl("metrics.jsonl", record)
        LOGGER.info(
            "iter %d | step %s | %d games, %.0f turns avg | %.0f decisions/s | "
            "win ratio vs opponents %.2f | trades/game %.1f | entropy %.2f | kl %.4f",
            self.iteration,
            f"{self.global_step:,}",
            stats.games,
            record["mean_turns"],
            record["decisions_per_second"],
            record["mixed_win_ratio"],
            record["trades_per_game"],
            metrics["entropy"],
            metrics["approx_kl"],
        )

    def _append_jsonl(self, name: str, record: dict[str, Any]) -> None:
        """Append one JSON record to a log file in the run directory.

        Args:
            name: Log file name.
            record: The record.
        """
        with (self.run_dir / name).open("a") as handle:
            handle.write(json.dumps(record) + "\n")

    def save_checkpoint(self) -> Path:
        """Write the full training state atomically.

        Returns:
            The checkpoint's path.
        """
        state = {
            "policy": self.policy.snapshot(),
            "optimizer": self.updater.optimizer.state_dict(),
            "iteration": self.iteration,
            "global_step": self.global_step,
            "rng": self.rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
        }
        temporary = self.checkpoint_path.with_suffix(".tmp")
        torch.save(state, temporary)
        os.replace(temporary, self.checkpoint_path)
        return self.checkpoint_path

    def _restore(self, state: dict[str, Any]) -> None:
        """Load training state written by `save_checkpoint`.

        Args:
            state: The checkpoint's contents.
        """
        self.policy.load_state_dict(state["policy"]["weights"])
        self.updater.optimizer.load_state_dict(state["optimizer"])
        self.iteration = state["iteration"]
        self.global_step = state["global_step"]
        self.rng.bit_generator.state = state["rng"]
        torch.set_rng_state(state["torch_rng"])


def run_metadata(config: TrainingConfig) -> dict[str, Any]:
    """Describe the code and machine a run started on, for reproducibility.

    Args:
        config: The run's configuration.

    Returns:
        Git commit and dirty flag, versions, platform, cores and start time.
    """
    return {
        "started": datetime.now(UTC).isoformat(),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_dirty": bool(_git(["status", "--porcelain"])),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "seed": config.seed,
    }


def _git(arguments: list[str]) -> str:
    """Run a git command, returning its output or `"unknown"` on failure.

    Args:
        arguments: Arguments after `git`.

    Returns:
        The stripped output, or `"unknown"`.
    """
    try:
        result = subprocess.run(
            ["git", *arguments], capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip()
