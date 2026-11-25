"""End-to-end tests for the trainer and its command line."""

import json
import shutil
from pathlib import Path

import pytest
import torch

from catan_bots.training.__main__ import load_weights, parse_overrides, run_config_for
from catan_bots.training.config import TrainingConfig
from catan_bots.training.graph_policy import GraphActorCritic
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
    for name in (
        "config.json",
        "metadata.json",
        "metrics.jsonl",
        "eval.jsonl",
        "train.log",
    ):
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


def test_resume_drops_records_logged_after_the_checkpoint(
    finished_run: Path, tmp_path: Path
) -> None:
    """Iterations logged after the last checkpoint are removed on resume,
    because they will be trained again."""
    copy = tmp_path / "run"
    shutil.copytree(finished_run, copy)
    checkpoint = torch.load(copy / "checkpoints" / "latest.pt", weights_only=False)
    later = checkpoint["iteration"] + 1
    for name in ("metrics.jsonl", "eval.jsonl"):
        with (copy / name).open("a") as handle:
            handle.write(json.dumps({"iteration": later, "step": 0}) + "\n")
    Trainer.resume(copy)
    for name in ("metrics.jsonl", "eval.jsonl"):
        iterations = [r["iteration"] for r in read_jsonl(copy / name)]
        assert max(iterations) == checkpoint["iteration"], name


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


def test_run_evaluates_against_references_and_its_past(
    finished_run: Path, tmp_path: Path
) -> None:
    """References and the oldest snapshot are evaluated under stable labels."""
    reference = finished_run / "checkpoints" / "latest.pt"
    config = tiny_config(
        tmp_path,
        eval_references=(str(reference),),
        eval_oldest_snapshot=True,
        final_learning_rate_fraction=0.1,
        target_kl=0.02,
    )
    trainer = Trainer.create(config)
    trainer.train()
    evaluations = read_jsonl(trainer.run_dir / "eval.jsonl")
    labels = {e["opponent"] for e in evaluations}
    assert {"random", "reference_0", "league_oldest"} <= labels
    assert all(
        e["spec"] == str(reference)
        for e in evaluations
        if e["opponent"] == "reference_0"
    )
    metrics = read_jsonl(trainer.run_dir / "metrics.jsonl")
    steps_before = [0] + [m["step"] for m in metrics[:-1]]
    for record, step in zip(metrics, steps_before):
        assert record["learning_rate"] == pytest.approx(config.learning_rate_at(step))


def test_missing_reference_fails_before_creating_a_run(tmp_path: Path) -> None:
    """A mistyped reference path stops the run before anything is written."""
    config = tiny_config(tmp_path / "runs", eval_references=("missing.pt",))
    with pytest.raises(FileNotFoundError, match="missing.pt"):
        Trainer.create(config)
    assert not (tmp_path / "runs").exists()


def test_graph_run_trains_resumes_and_plays_mlp_references(
    finished_run: Path, tmp_path: Path
) -> None:
    """A graph-network run trains (on the GPU if there is one), resumes and keeps
    training, and evaluates against an MLP checkpoint."""
    reference = finished_run / "checkpoints" / "latest.pt"
    config = tiny_config(
        tmp_path,
        architecture="graph",
        graph_width=16,
        graph_rounds=1,
        update_device="auto",
        eval_references=(str(reference),),
        eval_oldest_snapshot=True,
    )
    trainer = Trainer.create(config)
    trainer.train()
    assert isinstance(trainer.policy, GraphActorCritic)
    assert load_weights(trainer.checkpoint_path)["architecture"] == "graph"
    snapshot = next((trainer.run_dir / "league").glob("snapshot_*.pt"))
    assert load_weights(snapshot)["architecture"] == "graph"
    labels = {e["opponent"] for e in read_jsonl(trainer.run_dir / "eval.jsonl")}
    assert {"reference_0", "league_oldest"} <= labels
    resumed = Trainer.resume(
        trainer.run_dir, {"total_steps": trainer.global_step + 300}
    )
    assert isinstance(resumed.policy, GraphActorCritic)
    for name, tensor in resumed.policy.state_dict().items():
        torch.testing.assert_close(tensor, trainer.policy.state_dict()[name])
    resumed.train()
    assert resumed.iteration > trainer.iteration


def test_checkpoints_and_snapshots_both_load(finished_run: Path) -> None:
    """Evaluation accepts a full checkpoint or a league snapshot."""
    snapshot = next((finished_run / "league").glob("snapshot_*.pt"))
    for path in (finished_run / "checkpoints" / "latest.pt", snapshot):
        assert set(load_weights(path)) >= {"weights", "hidden_sizes"}
        assert run_config_for(path).hidden_sizes == (32,)
