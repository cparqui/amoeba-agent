"""Run provenance and optional MLflow tracking.

Provenance (git commit, host, Python, packages) is always written into the run dir
as `provenance.json`, so a run is reproducible even without a tracking server.

MLflow is used only when a tracking URI is configured (config `tracking.uri` or
$MLFLOW_TRACKING_URI). The client is `mlflow-skinny` (the `experiments` dependency
group); artifacts go wherever the server is configured to put them, so a server
started with `--artifacts-destination /some/disk/path` needs no S3 at all.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel, Field


class TrackingError(RuntimeError):
    pass


class TrackingConfig(BaseModel):
    """Tracking is on by default and required: a run with no URI fails before it
    starts, unless tracking is turned off explicitly (`enabled: false` or
    `--no-track`). Untracked runs can be uploaded later with the `track` command."""

    uri: str | None = Field(default=None, description="MLflow tracking URI; default $MLFLOW_TRACKING_URI.")
    experiment: str | None = Field(default=None, description="MLflow experiment name; default set by the experiment.")
    enabled: bool = True

    def resolved_uri(self) -> str | None:
        return self.uri or os.environ.get("MLFLOW_TRACKING_URI")


def check_ready(cfg: TrackingConfig, timeout: float = 5.0) -> str | None:
    """Fail fast, before any compute: return the URI to log to, or None if disabled.

    For an HTTP server this hits its /health endpoint, so a server that is down
    fails in seconds rather than after MLflow's own multi-minute retry loop.
    """
    if not cfg.enabled:
        return None
    uri = cfg.resolved_uri()
    if not uri:
        raise TrackingError(
            "tracking is on but no MLflow URI is set. Set tracking.uri in the config or "
            "MLFLOW_TRACKING_URI (e.g. http://workstation:5000), or pass --no-track "
            "and upload the run later with the `track` command."
        )
    if uri.startswith(("http://", "https://")):
        import httpx  # noqa: PLC0415

        try:
            r = httpx.get(f"{uri.rstrip('/')}/health", timeout=timeout)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise TrackingError(
                f"MLflow server at {uri} is not reachable ({exc}). Start it, or pass --no-track "
                "and upload the run later with the `track` command."
            ) from exc
    return uri


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def provenance(packages: tuple[str, ...] = ("amoeba-agent", "httpx", "pydantic")) -> dict[str, Any]:
    versions = {}
    for p in packages:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            versions[p] = None
    status = _git("status", "--porcelain")
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "argv": sys.argv,
        "packages": versions,
    }


def flatten(d: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """Nested dict -> {'a.b': 'value'} with values as short strings, for MLflow params."""
    out: dict[str, str] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, f"{key}."))
        else:
            s = json.dumps(v) if isinstance(v, (list, tuple)) else str(v)
            out[key] = s[:6000]  # MLflow param value limit
    return out


class Tracker:
    """Thin wrapper so experiment code never branches on whether MLflow is on."""

    def __init__(self, mlflow: Any | None) -> None:
        self._mlflow = mlflow

    @property
    def active(self) -> bool:
        return self._mlflow is not None

    @property
    def run_id(self) -> str | None:
        run = self._mlflow.active_run() if self._mlflow else None
        return run.info.run_id if run else None

    def params(self, d: dict[str, Any]) -> None:
        if self._mlflow:
            self._mlflow.log_params(flatten(d))

    def tags(self, d: dict[str, Any]) -> None:
        if self._mlflow:
            self._mlflow.set_tags(flatten(d))

    def metrics(self, d: dict[str, Any], step: int | None = None) -> None:
        if self._mlflow:
            clean = {k: float(v) for k, v in d.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
            if clean:
                self._mlflow.log_metrics(clean, step=step)

    def dict(self, d: dict[str, Any], name: str) -> None:
        if self._mlflow:
            self._mlflow.log_dict(d, name)

    def artifacts(self, path: Path, artifact_path: str | None = None) -> None:
        if self._mlflow:
            if path.is_dir():
                self._mlflow.log_artifacts(str(path), artifact_path)
            elif path.exists():
                self._mlflow.log_artifact(str(path), artifact_path)

    @contextmanager
    def child(self, name: str) -> Iterator[Tracker]:
        """A nested run, e.g. one per condition, for side-by-side comparison."""
        if not self._mlflow:
            yield self
            return
        with self._mlflow.start_run(run_name=name, nested=True):
            yield self


@contextmanager
def tracking(cfg: TrackingConfig, default_experiment: str, run_name: str) -> Iterator[Tracker]:
    """An MLflow run, or a no-op tracker when tracking is disabled."""
    uri = check_ready(cfg)
    if not uri:
        yield Tracker(None)
        return
    # MLflow's default (7 retries, backoff 2) takes ~4 minutes to give up on a
    # dead server; the run dir is kept either way, so fail sooner.
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "3")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    try:
        import mlflow  # noqa: PLC0415  (optional dependency)
    except ImportError as exc:
        raise TrackingError(
            "tracking is configured but mlflow is not installed; run `uv sync --all-packages --group experiments`"
        ) from exc
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(cfg.experiment or default_experiment)
    with mlflow.start_run(run_name=run_name):
        yield Tracker(mlflow)
