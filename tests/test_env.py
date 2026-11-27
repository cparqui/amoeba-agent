"""Loading local settings from .env."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from click.testing import CliRunner
from dotenv import dotenv_values

from experiments.common.env import load_env
from experiments.common.llama_server import free_port
from experiments.translation.run import cli

from tests.test_pipeline import _config, fake_bin  # noqa: F401  (fixture)

pytestmark = pytest.mark.dotenv
KEYS = ("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD")


@pytest.fixture(autouse=True)
def _restore_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """load_dotenv writes os.environ directly; make monkeypatch restore these keys."""
    for k in KEYS:
        monkeypatch.setenv(k, "placeholder")
        monkeypatch.delenv(k)


def _dead_uri() -> str:
    return f"http://127.0.0.1:{free_port()}"


def test_dotenv_in_working_dir_is_loaded(fake_bin: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    uri = _dead_uri()
    (tmp_path / ".env").write_text(f"MLFLOW_TRACKING_URI={uri}\nMLFLOW_TRACKING_USERNAME=admin\n")
    monkeypatch.chdir(tmp_path)
    res = CliRunner().invoke(cli, ["pipeline", "-c", str(_config(tmp_path))])
    # Without the .env this would be "no MLflow URI is set"; with it, the URI is tried.
    assert "not reachable" in res.output and uri in res.output
    assert f"settings: {tmp_path / '.env'}" in res.output
    assert os.environ["MLFLOW_TRACKING_USERNAME"] == "admin"


def test_shell_env_beats_dotenv(fake_bin: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    shell, file = _dead_uri(), _dead_uri()
    (tmp_path / ".env").write_text(f"MLFLOW_TRACKING_URI={file}\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", shell)
    res = CliRunner().invoke(cli, ["pipeline", "-c", str(_config(tmp_path))])
    assert shell in res.output and file not in res.output


def test_explicit_env_file(fake_bin: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    uri = _dead_uri()
    other = tmp_path / "workstation.env"
    other.write_text(f"MLFLOW_TRACKING_URI={uri}\n")
    monkeypatch.chdir(tmp_path)
    res = CliRunner().invoke(cli, ["--env-file", str(other), "pipeline", "-c", str(_config(tmp_path))])
    assert uri in res.output


def test_no_dotenv_is_fine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    empty = tmp_path / "a" / "b"
    empty.mkdir(parents=True)
    monkeypatch.chdir(empty)
    monkeypatch.setattr("experiments.common.env.find_dotenv", lambda usecwd=True: "")
    assert load_env() is None


def test_env_example_lists_tracking_keys() -> None:
    example = Path(__file__).resolve().parents[1] / ".env.example"
    values = dotenv_values(example)
    assert set(KEYS) <= set(values)
    assert not values["MLFLOW_TRACKING_PASSWORD"]  # no secret in the template
