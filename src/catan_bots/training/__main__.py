"""Command line for self-play training.

Examples:
    uv run python -m catan_bots.training train --config configs/selfplay_4p.json
    uv run python -m catan_bots.training resume runs/selfplay-20261020-120000
    uv run python -m catan_bots.training evaluate runs/<run>/checkpoints/latest.pt
    uv run python -m catan_bots.training benchmark
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import torch

from catan_bots.training import workers
from catan_bots.training.agents import BASELINES
from catan_bots.training.config import TrainingConfig
from catan_bots.training.evaluator import Evaluator
from catan_bots.training.policy import ActorCritic
from catan_bots.training.trainer import Trainer


def parse_overrides(items: list[str]) -> dict[str, Any]:
    """Parse `key=value` settings, reading values as JSON where possible.

    Args:
        items: Strings such as `total_steps=5000000` or `baselines=["random"]`.

    Returns:
        Setting name to value; values that are not JSON stay strings.

    Raises:
        ValueError: If an item has no `=`.
    """
    overrides: dict[str, Any] = {}
    for item in items:
        key, separator, raw = item.partition("=")
        if not separator:
            raise ValueError(f"Expected key=value, got {item!r}.")
        try:
            overrides[key] = json.loads(raw)
        except json.JSONDecodeError:
            overrides[key] = raw
    return overrides


def run_config_for(checkpoint: Path) -> TrainingConfig:
    """Find the configuration of the run a checkpoint or snapshot belongs to.

    Args:
        checkpoint: A `latest.pt` or league snapshot inside a run directory.

    Returns:
        The run's configuration, or the defaults if none is found.
    """
    for directory in checkpoint.resolve().parents:
        candidate = directory / "config.json"
        if candidate.exists():
            return TrainingConfig.load(candidate)
    return TrainingConfig()


def load_weights(checkpoint: Path) -> dict[str, Any]:
    """Load policy weights from a checkpoint or a league snapshot.

    Args:
        checkpoint: The file.

    Returns:
        A policy snapshot.
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return state.get("policy", state)


def command_train(args: argparse.Namespace) -> int:
    """Start a new run.

    Args:
        args: Parsed arguments.

    Returns:
        Exit status.
    """
    config = TrainingConfig.load(args.config) if args.config else TrainingConfig()
    config = config.with_overrides(parse_overrides(args.set))
    Trainer.create(config).train()
    return 0


def command_resume(args: argparse.Namespace) -> int:
    """Continue a run from its latest checkpoint.

    Args:
        args: Parsed arguments.

    Returns:
        Exit status.
    """
    Trainer.resume(args.run_dir, parse_overrides(args.set)).train()
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    """Evaluate a checkpoint against baselines and print the results.

    Args:
        args: Parsed arguments.

    Returns:
        Exit status.
    """
    config = run_config_for(args.checkpoint)
    weights = load_weights(args.checkpoint)
    worker_count = args.workers or config.num_workers
    with workers.make_pool(config, worker_count) as pool:
        evaluator = Evaluator(pool, worker_count, config.num_players)
        for opponent in args.opponents:
            result = evaluator.evaluate(
                weights, opponent, args.games, args.seed, args.deterministic
            )
            print(result.summary(), flush=True)
    return 0


def command_benchmark(args: argparse.Namespace) -> int:
    """Measure game throughput with one worker and with many.

    Uses an untrained policy in every seat, so it measures the machine, not
    the policy.

    Args:
        args: Parsed arguments.

    Returns:
        Exit status.
    """
    config = TrainingConfig(self_play_fraction=1.0, num_workers=args.workers)
    env = workers.make_env(config)
    weights = ActorCritic(
        env.encoder.size, env.table.size, config.hidden_sizes
    ).snapshot()
    for count in sorted({1, args.workers}):
        rate = _measure(config, weights, count, args.steps)
        print(
            f"{count:2d} worker(s): {rate:,.0f} decisions/s "
            f"(~{rate * 3600 / 1e6:.1f}M per hour, {rate / count:,.0f} per worker)",
            flush=True,
        )
    return 0


def _measure(
    config: TrainingConfig, weights: dict[str, Any], count: int, steps: int
) -> float:
    """Time game collection on a warmed-up pool.

    Args:
        config: Run configuration.
        weights: Policy snapshot to play.
        count: Workers.
        steps: Learner decisions each worker collects.

    Returns:
        Decisions per second across all workers.
    """
    with workers.make_pool(config, count) as pool:
        warmup = [workers.CollectTask(weights, 64, seed, ()) for seed in range(count)]
        pool.map(workers.collect, warmup)
        tasks = [
            workers.CollectTask(weights, steps, 100 + seed, ()) for seed in range(count)
        ]
        started = time.perf_counter()
        results = pool.map(workers.collect, tasks)
        elapsed = time.perf_counter() - started
    return sum(stats.decisions for _, stats in results) / elapsed


def build_parser() -> argparse.ArgumentParser:
    """Define the command line.

    Returns:
        The argument parser.
    """
    parser = argparse.ArgumentParser(prog="python -m catan_bots.training")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="start a new run")
    train.add_argument("--config", type=Path, help="JSON configuration file")
    train.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    train.set_defaults(handler=command_train)

    resume = commands.add_parser("resume", help="continue a run")
    resume.add_argument("run_dir", type=Path)
    resume.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    resume.set_defaults(handler=command_resume)

    evaluate = commands.add_parser("evaluate", help="evaluate a checkpoint")
    evaluate.add_argument("checkpoint", type=Path)
    evaluate.add_argument("--opponents", nargs="+", default=list(BASELINES))
    evaluate.add_argument("--games", type=int, default=400)
    evaluate.add_argument("--workers", type=int, default=0)
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--deterministic", action="store_true")
    evaluate.set_defaults(handler=command_evaluate)

    benchmark = commands.add_parser("benchmark", help="measure this machine")
    benchmark.add_argument("--workers", type=int, default=TrainingConfig().num_workers)
    benchmark.add_argument("--steps", type=int, default=3000)
    benchmark.set_defaults(handler=command_benchmark)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command line.

    Args:
        argv: Arguments; defaults to the process's own.

    Returns:
        Exit status.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
    )
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
