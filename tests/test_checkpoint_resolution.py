from __future__ import annotations

from pathlib import Path

import pytest
from open_provence.trainer import ResolvedCheckpoint, resolve_resume_checkpoint_path


def _make_checkpoint(dir_path: Path) -> None:
    dir_path.mkdir(parents=True)
    (dir_path / "trainer_state.json").write_text("{}", encoding="utf-8")


def test_resolve_explicit_checkpoint_returns_parent(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint-0500"
    _make_checkpoint(checkpoint_dir)

    resolved = resolve_resume_checkpoint_path(checkpoint_dir)

    assert isinstance(resolved, ResolvedCheckpoint)
    assert resolved.checkpoint_dir == checkpoint_dir.resolve()
    assert resolved.run_dir == tmp_path.resolve()
    assert resolved.steps == 500


def test_resolve_parent_directory_picks_latest_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    older = run_dir / "checkpoint-0100"
    newest = run_dir / "checkpoint-0500"
    _make_checkpoint(older)
    _make_checkpoint(newest)

    resolved = resolve_resume_checkpoint_path(run_dir)

    assert resolved.checkpoint_dir == newest.resolve()
    assert resolved.run_dir == run_dir.resolve()
    assert resolved.steps == 500


def test_resolve_parent_directory_without_checkpoints_errors(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(ValueError):
        resolve_resume_checkpoint_path(run_dir)


def test_resolve_missing_path_errors(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        resolve_resume_checkpoint_path(missing)
