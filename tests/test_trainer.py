"""End-to-end tests for the trainer and its command line."""

import json
from pathlib import Path

import pytest
import torch

from catan_bots.training.__main__ import load_weights, parse_overrides, run_config_for
from catan_bots.training.config import TrainingConfig
from catan_bots.training.trainer import Trainer


def tiny_config(output_dir: Path, **overrides: object) -> TrainingConfig:
    """Build a configuration small enough to train in seconds.

    Args:
        output_dir: Where the run is written.
        **overrides: Settings to change.

    Returns:
        The configuration.
    """
    values = {
        "run_name": "test",
        "output_dir": str(output_dir),
        "total_steps": 600,
        "num_workers": 1,
        "steps_per_iteration": 300,
        "minibatch_size": 150,
        "update_epochs": 1,
        "hidden_sizes": (32,),
        "turn_limit": 80,
        "snapshot_every": 1,
        "checkpoint_every": 1,
        "eval_every": 1,
        "eval_games": 2,
        "eval_opponents": ("random",),
        **overrides,
    }
    return TrainingConfig(**values)


@pytest.fixture(name="finished_run", scope="module")
def fixture_finished_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train one tiny run to completion, shared by the tests below.

    Args:
        tmp_path_factory: Pytest's temporary directory factory.

    Returns:
        The run directory.
    """
    trainer = Trainer.create(tiny_config(tmp_path_factory.mktemp("runs")))
    trainer.train()
    return trainer.run_dir


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSON-lines log.

    Args:
        path: The log file.

    Returns:
        One dict per line.
    """
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_run_writes_every_artifact(finished_run: Path) -> None:
    """A run leaves its config, metadata, checkpoint, league and logs."""
    for name in ("config.json", "metadata.json", "metrics.jsonl", "eval.jsonl"):
        assert (finished_run / name).exists(), name
    assert (finished_run / "checkpoints" / "latest.pt").exists()
    assert list((finished_run / "league").glob("snapshot_*.pt"))
    assert list((finished_run / "tensorboard").iterdir())
    assert not list((finished_run / "checkpoints").glob("*.tmp"))


def test_metadata_records_code_and_machine(finished_run: Path) -> None:
    """Metadata names the git commit, versions and machine."""
    metadata = json.loads((finished_run / "metadata.json").read_text())
    assert {"git_commit", "git_dirty", "torch", "platform", "cpu_count", "seed"} <= set(
        metadata
    )


def test_training_reaches_the_step_budget(finished_run: Path) -> None:
    """Iterations are logged in order and stop once the budget is met."""
    metrics = read_jsonl(finished_run / "metrics.jsonl")
    assert [m["iteration"] for m in metrics] == list(range(1, len(metrics) + 1))
    assert metrics[-1]["step"] >= 600
    if len(metrics) > 1:
        assert metrics[-2]["step"] < 600, "trained past the budget"


def test_each_iteration_is_evaluated_once(finished_run: Path) -> None:
    """No iteration is evaluated twice, even at the end of training."""
    evaluations = read_jsonl(finished_run / "eval.jsonl")
    iterations = [e["iteration"] for e in evaluations]
    assert len(iterations) == len(set(iterations))
    assert all(
        0 <= e["ci_low"] <= e["win_rate"] <= e["ci_high"] <= 1 for e in evaluations
    )


def test_resume_continues_where_training_stopped(finished_run: Path) -> None:
    """Resuming with a larger budget continues the iteration count."""
    before = read_jsonl(finished_run / "metrics.jsonl")
    trainer = Trainer.resume(finished_run, {"total_steps": before[-1]["step"] + 300})
    assert trainer.iteration == before[-1]["iteration"]
    trainer.train()
    after = read_jsonl(finished_run / "metrics.jsonl")
    assert [m["iteration"] for m in after] == list(range(1, len(after) + 1))
    assert len(after) > len(before)
    saved = TrainingConfig.load(finished_run / "config.json")
    assert saved.total_steps == before[-1]["step"] + 300


def test_checkpoint_restores_the_exact_policy(finished_run: Path) -> None:
    """A resumed trainer holds the checkpointed weights."""
    state = torch.load(finished_run / "checkpoints" / "latest.pt", weights_only=False)
    trainer = Trainer.resume(finished_run)
    for name, tensor in trainer.policy.state_dict().items():
        torch.testing.assert_close(tensor, state["policy"]["weights"][name])


def test_resume_needs_a_checkpoint(tmp_path: Path) -> None:
    """A run with no checkpoint cannot be resumed."""
    trainer = Trainer.create(tiny_config(tmp_path))
    with pytest.raises(FileNotFoundError):
        Trainer.resume(trainer.run_dir)


def test_parse_overrides_reads_json_values() -> None:
    """Numbers, lists and booleans parse as JSON; anything else stays a string."""
    parsed = parse_overrides(
        ["total_steps=500", 'baselines=["random"]', "trading=false", "run_name=x"]
    )
    assert parsed == {
        "total_steps": 500,
        "baselines": ["random"],
        "trading": False,
        "run_name": "x",
    }
    with pytest.raises(ValueError):
        parse_overrides(["no-equals-sign"])


def test_checkpoints_and_snapshots_both_load(finished_run: Path) -> None:
    """Evaluation accepts a full checkpoint or a league snapshot."""
    snapshot = next((finished_run / "league").glob("snapshot_*.pt"))
    for path in (finished_run / "checkpoints" / "latest.pt", snapshot):
        assert set(load_weights(path)) >= {"weights", "hidden_sizes"}
        assert run_config_for(path).hidden_sizes == (32,)
