import shutil
from pathlib import Path

import pytask
import pytest
from pytask import ExitCode

from embedding_space_exploration import config
from embedding_space_exploration.config import ROOT


def test_pytask_build(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Copy project files to temp directory
    shutil.copytree(ROOT / "documents", tmp_path / "documents")
    shutil.copytree(ROOT / "src", tmp_path / "src")
    shutil.copy(ROOT / "myst.yml", tmp_path / "myst.yml")

    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "BLD", tmp_path / "bld")
    monkeypatch.setattr(config, "DOCUMENTS", tmp_path / "documents")
    monkeypatch.setattr(config, "SRC", tmp_path / "src" / "embedding_space_exploration")

    session = pytask.build(
        config=ROOT / "pyproject.toml",
        force=True,
    )
    assert session.exit_code == ExitCode.OK
