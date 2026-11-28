"""Run provenance and live MLflow tracking.

Provenance (git commit, host, Python, packages) is always written into the run dir
as `provenance.json`, so a run is reproducible even without a tracking server.

Tracking is live: the MLflow run opens when the experiment starts, metrics are
logged as each piece of work finishes, and system metrics (CPU, memory, disk,
network, GPU) stream from a background thread. The run dir is the source of truth.
An experiment describes what *should* be in MLflow as a `Snapshot` computed from
its run dir, and `LiveRun.sync()` logs whatever is missing. A ledger in the run
dir (`mlflow.json`) records run ids and the highest step logged per metric. So
catching up after an outage, `--resume` and the `track` command all reuse the
same MLflow run without duplicating points.

What happens when MLflow misbehaves:

    no URI set                     error before anything starts (or pass --no-track)
    401 / bad credentials          one warning, tracking disabled for this run
    403 Invalid Host header        one warning, tracking disabled (fix --allowed-hosts)
    403 permission denied          one warning, tracking disabled
    unreachable / 5xx / timeout    one warning, then quiet retries with backoff, up to
                                   retry_max_attempts; one line when it comes back,
                                   and one if it gives up. A final attempt is always
                                   made at the end of the run.

A run that ends untracked or partly tracked prints the `track` command that
uploads it. Nothing is ever lost: the run dir holds everything.
"""

from __future__ import annotations

import json
import logging
import math
import os
import platform
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Protocol

import httpx
from pydantic import BaseModel, Field

LEDGER = "mlflow.json"
INVALID_HOST = "Invalid Host header"

# MLflow prints an "agent hint" on import and a "View run ... at:" line for every
# run it closes (one per condition here). One link to the parent run is printed
# instead, at start and end.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "true")


class TrackingError(RuntimeError):
    pass


class TrackingConfig(BaseModel):
    """Tracking is on by default and required: a run with no URI fails before it
    starts, unless tracking is turned off explicitly (`enabled: false` or
    `--no-track`). Untracked runs can be uploaded later with the `track` command."""

    uri: str | None = Field(default=None, description="MLflow tracking URI; default $MLFLOW_TRACKING_URI.")
    experiment: str | None = Field(default=None, description="MLflow experiment name; default set by the experiment.")
    enabled: bool = True
    system_metrics: bool = Field(default=True, description="Log CPU/memory/disk/network/GPU usage under system/.")
    system_metrics_interval_s: float = Field(default=10.0, description="Seconds between system-metric samples.")
    retry_interval_s: float = Field(default=30.0, description="First wait before retrying an unreachable server.")
    retry_max_interval_s: float = Field(default=600.0, description="Longest wait between retries (backoff doubles).")
    retry_max_attempts: int = Field(default=20, description="Retries per run before giving up on live tracking.")
    request_timeout_s: float = Field(default=30.0, description="Timeout for each MLflow request.")

    def resolved_uri(self) -> str | None:
        return self.uri or os.environ.get("MLFLOW_TRACKING_URI")


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Probing the server
# --------------------------------------------------------------------------- #


class Status(str, Enum):
    OK = "ok"
    UNAVAILABLE = "unavailable"  # down, refusing, timing out, 5xx: worth retrying
    AUTH = "auth"  # 401: missing or wrong credentials
    HOST = "host"  # 403 Invalid Host header: server's --allowed-hosts
    FORBIDDEN = "forbidden"  # 403: user lacks permission
    ERROR = "error"  # anything else (wrong URL, proxy page, ...)

    @property
    def retryable(self) -> bool:
        return self in (Status.OK, Status.UNAVAILABLE)


def auth_headers() -> dict[str, str]:
    """The same credentials the MLflow client uses, from the environment (or .env)."""
    if token := os.environ.get("MLFLOW_TRACKING_TOKEN"):
        return {"Authorization": f"Bearer {token}"}
    user, pw = os.environ.get("MLFLOW_TRACKING_USERNAME"), os.environ.get("MLFLOW_TRACKING_PASSWORD")
    if user and pw:
        import base64  # noqa: PLC0415

        return {"Authorization": "Basic " + base64.b64encode(f"{user}:{pw}".encode()).decode()}
    return {}


def probe(uri: str, experiment: str = "Default", timeout: float = 5.0) -> tuple[Status, str]:
    """Classify a tracking server with one authenticated read request.

    Uses a real API endpoint rather than /health, because /health skips both the
    Host-header check and authentication and would report a server that rejects
    every logging call as healthy.
    """
    if not uri.startswith(("http://", "https://")):
        return Status.OK, "local store"
    url = f"{uri.rstrip('/')}/api/2.0/mlflow/experiments/get-by-name"
    verify = os.environ.get("MLFLOW_TRACKING_INSECURE_TLS", "").lower() not in ("1", "true")
    try:
        r = httpx.get(url, params={"experiment_name": experiment}, headers=auth_headers(), timeout=timeout, verify=verify)
    except httpx.HTTPError as exc:
        return Status.UNAVAILABLE, f"{type(exc).__name__}: {exc}"
    body = r.text[:200].strip()
    if r.status_code == 200 or (r.status_code == 404 and "RESOURCE_DOES_NOT_EXIST" in r.text):
        return Status.OK, "ok"
    if r.status_code == 401:
        has = "credentials were rejected" if auth_headers() else "no credentials were given"
        return Status.AUTH, f"401 Unauthorized ({has})"
    if r.status_code == 403:
        if INVALID_HOST in r.text:
            return Status.HOST, f"403 {INVALID_HOST}"
        return Status.FORBIDDEN, f"403 Forbidden: {body}"
    if r.status_code >= 500:
        return Status.UNAVAILABLE, f"HTTP {r.status_code}: {body}"
    return Status.ERROR, f"HTTP {r.status_code}: {body}"


ADVICE = {
    Status.AUTH: "check MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD (e.g. in .env)",
    Status.HOST: "add this host *with its port* to the server's --allowed-hosts, e.g. localhost:*",
    Status.FORBIDDEN: "give this user write (EDIT) permission on the experiment, or use an admin login",
    Status.ERROR: "check that the URI points at an MLflow server",
}


def classify_exception(exc: BaseException, uri: str, experiment: str) -> tuple[Status, str]:
    """Turn a failed MLflow client call into a Status, probing when the error is ambiguous."""
    code = None
    get_code = getattr(exc, "get_http_status_code", None)
    if callable(get_code):
        try:
            code = get_code()
        except Exception:  # noqa: BLE001
            code = None
    text = str(exc)
    if code == 401:
        return Status.AUTH, "401 Unauthorized"
    if code == 403 or "PERMISSION_DENIED" in text:
        if INVALID_HOST in text:
            return Status.HOST, f"403 {INVALID_HOST}"
        user = os.environ.get("MLFLOW_TRACKING_USERNAME") or "this user"
        return Status.FORBIDDEN, f"403 Forbidden: {user} may read but not write to experiment {experiment!r}"
    status, detail = probe(uri, experiment)
    if status is Status.OK:
        # The server answers reads but this call failed: treat as transient.
        return Status.UNAVAILABLE, text[:200]
    return status, detail


# --------------------------------------------------------------------------- #
# What an experiment wants logged
# --------------------------------------------------------------------------- #

Series = dict[str, dict[int, float]]  # metric key -> step -> value


@dataclass
class RunSpec:
    params: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, Any] = field(default_factory=dict)
    series: Series = field(default_factory=dict)


@dataclass
class Snapshot:
    """Desired MLflow state, computed from the run dir. Logging is the diff."""

    parent: RunSpec
    children: dict[str, RunSpec] = field(default_factory=dict)


def add_point(series: Series, key: str, step: int, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return
    if math.isnan(value) or math.isinf(value):
        return
    series.setdefault(key, {})[int(step)] = float(value)


# --------------------------------------------------------------------------- #
# MLflow backend
# --------------------------------------------------------------------------- #


class Backend(Protocol):
    def experiment_id(self, name: str) -> str: ...
    def create_run(self, experiment_id: str, name: str, tags: dict[str, str], parent: str | None) -> str: ...
    def log(self, run_id: str, metrics: list[tuple[str, float, int]], params: dict[str, str], tags: dict[str, str]) -> None: ...
    def set_status(self, run_id: str, status: str) -> None: ...
    def artifacts(self, run_id: str, path: Path) -> None: ...
    def start_system_metrics(self, run_id: str, interval: float, resume: bool) -> Any: ...
    def stop_system_metrics(self, handle: Any) -> None: ...


class MlflowBackend:
    """MLflow client calls, with MLflow's own retries off: LiveRun does the retrying."""

    def __init__(self, uri: str, timeout: float) -> None:
        os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "0"
        os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = str(int(timeout))
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        # The system-metrics thread logs its own warning when the server drops;
        # LiveRun already reports outages once, so keep MLflow's copy quiet.
        logging.getLogger("mlflow.system_metrics").setLevel(logging.CRITICAL)
        logging.getLogger("mlflow.utils.autologging_utils").setLevel(logging.CRITICAL)
        try:
            from mlflow.tracking import MlflowClient  # noqa: PLC0415
        except ImportError as exc:
            raise TrackingError("mlflow is not installed; run `uv sync --all-packages`") from exc
        self.uri = uri
        self.client = MlflowClient(tracking_uri=uri)

    def experiment_id(self, name: str) -> str:
        exp = self.client.get_experiment_by_name(name)
        return exp.experiment_id if exp else self.client.create_experiment(name)

    def create_run(self, experiment_id: str, name: str, tags: dict[str, str], parent: str | None) -> str:
        if parent:
            tags = {**tags, "mlflow.parentRunId": parent}
        return self.client.create_run(experiment_id, tags=tags, run_name=name).info.run_id

    def log(self, run_id: str, metrics: list[tuple[str, float, int]], params: dict[str, str], tags: dict[str, str]) -> None:
        from mlflow.entities import Metric, Param, RunTag  # noqa: PLC0415

        now = int(time.time() * 1000)
        ms = [Metric(k, v, now, s) for k, v, s in metrics]
        ps = [Param(k, v) for k, v in params.items()]
        ts = [RunTag(k, v) for k, v in tags.items()]
        for i in range(0, max(len(ms), 1), 900):  # log_batch takes at most 1000 metrics
            self.client.log_batch(run_id, metrics=ms[i : i + 900], params=ps if i == 0 else [], tags=ts if i == 0 else [])

    def set_status(self, run_id: str, status: str) -> None:
        if status == "RUNNING":
            self.client.update_run(run_id, status="RUNNING")
        else:
            self.client.set_terminated(run_id, status=status)

    def artifacts(self, run_id: str, path: Path) -> None:
        self.client.log_artifacts(run_id, str(path))

    def start_system_metrics(self, run_id: str, interval: float, resume: bool) -> Any:
        try:
            from mlflow.system_metrics.system_metrics_monitor import SystemMetricsMonitor  # noqa: PLC0415

            mon = SystemMetricsMonitor(run_id, sampling_interval=interval, resume_logging=resume, tracking_uri=self.uri)
        except Exception:  # noqa: BLE001  (psutil missing, etc.: system metrics are optional)
            return None
        mon.start()
        return mon

    def stop_system_metrics(self, handle: Any) -> None:
        if handle is not None:
            try:
                handle.finish()
            except Exception:  # noqa: BLE001
                pass


# --------------------------------------------------------------------------- #
# Live run
# --------------------------------------------------------------------------- #


class Mode(str, Enum):
    OFF = "off"  # --no-track
    ONLINE = "online"
    OFFLINE = "offline"  # unreachable; retrying
    GAVE_UP = "gave_up"  # retries exhausted; one last try at the end
    DISABLED = "disabled"  # credentials / host / permission problem


def _warn(msg: str) -> None:
    print(f"warning: {msg}", file=sys.stderr, flush=True)


def _info(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class LiveRun:
    """One experiment run in MLflow, kept in step with its run dir.

    Call `start()` once, `sync()` whenever new results land (cheap when nothing
    changed, and quiet while offline), and `finish()` at the end. Thread-safe.
    """

    def __init__(
        self,
        cfg: TrackingConfig,
        default_experiment: str,
        run_dir: Path,
        snapshot: Callable[[], Snapshot],
        *,
        system_metrics: bool | None = None,
        backend_factory: Callable[[str, float], Backend] | None = None,
        probe_fn: Callable[[str, str], tuple[Status, str]] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cfg = cfg
        self.experiment = cfg.experiment or default_experiment
        self.run_dir = Path(run_dir)
        self.snapshot = snapshot
        self.system_metrics = cfg.system_metrics if system_metrics is None else system_metrics
        self._backend_factory = backend_factory or (lambda uri, timeout: MlflowBackend(uri, timeout))
        self._probe = probe_fn or (lambda uri, exp: probe(uri, exp))
        self._clock = clock
        self._lock = threading.RLock()
        self._backend: Backend | None = None
        self._sysmon: Any = None
        self.uri = cfg.resolved_uri()
        self.mode = Mode.OFF
        self.reason = ""
        self.attempts = 0
        self._interval = cfg.retry_interval_s
        self._next_at = 0.0
        self.ledger = self._load_ledger()

    # ---- ledger ---------------------------------------------------------- #

    def _load_ledger(self) -> dict[str, Any]:
        p = self.run_dir / LEDGER
        if p.exists():
            return json.loads(p.read_text())
        return {"uri": None, "experiment_id": None, "run_id": None, "children": {}, "pushed": {}, "meta": [], "finished": False}

    def _save_ledger(self) -> None:
        tmp = self.run_dir / f".{LEDGER}.tmp"
        tmp.write_text(json.dumps(self.ledger, indent=2))
        tmp.replace(self.run_dir / LEDGER)

    @property
    def run_id(self) -> str | None:
        return self.ledger.get("run_id")

    @property
    def run_url(self) -> str | None:
        """Browser link to the parent run, for an HTTP tracking server."""
        led = self.ledger
        if not (self.uri and self.uri.startswith(("http://", "https://")) and led.get("run_id")):
            return None
        return f"{self.uri.rstrip('/')}/#/experiments/{led['experiment_id']}/runs/{led['run_id']}"

    @property
    def tracking(self) -> bool:
        """True unless tracking is off or disabled for this run."""
        return self.mode not in (Mode.OFF, Mode.DISABLED)

    # ---- lifecycle ------------------------------------------------------- #

    def start(self) -> None:
        """Check the server and open (or reopen) the MLflow run. Never raises for
        server problems; raises TrackingError only when no URI is configured."""
        if not self.cfg.enabled:
            self.mode = Mode.OFF
            return
        if not self.uri:
            raise TrackingError(
                "tracking is on but no MLflow URI is set. Set MLFLOW_TRACKING_URI (e.g. in .env) or "
                "tracking.uri in the config, or pass --no-track and upload the run later with `track`."
            )
        if self.ledger.get("uri") and self.ledger["uri"] != self.uri:
            _warn(f"run dir was tracked at {self.ledger['uri']}; starting a new MLflow run at {self.uri}")
            self.ledger = {**self._load_ledger(), "uri": None, "experiment_id": None, "run_id": None,
                           "children": {}, "pushed": {}, "meta": [], "finished": False}
        with self._lock:
            status, detail = self._probe(self.uri, self.experiment)
            if status is Status.OK:
                self.mode = Mode.ONLINE
                self._push()
                if self.mode is Mode.ONLINE:
                    _info(f"tracking live to MLflow at {self.uri} (run {self.run_id})"
                          + (f"\n  view at {self.run_url}" if self.run_url else ""))
            elif status is Status.UNAVAILABLE:
                self._go_offline(detail)
            else:
                self._disable(status, detail)

    def sync(self, *, force: bool = False) -> None:
        """Log anything new. While offline this only probes when a retry is due."""
        with self._lock:
            if self.mode in (Mode.OFF, Mode.DISABLED):
                return
            if self.mode in (Mode.OFFLINE, Mode.GAVE_UP):
                if self.mode is Mode.GAVE_UP and not force:
                    return
                if not force and self._clock() < self._next_at:
                    return
                status, detail = self._probe(self.uri, self.experiment)  # type: ignore[arg-type]
                if status is Status.UNAVAILABLE:
                    self._retry_failed(detail, final=force)
                    return
                if status is not Status.OK:
                    self._disable(status, detail)
                    return
                _info(f"MLflow at {self.uri} is reachable again after {self.attempts} retr"
                      f"{'y' if self.attempts == 1 else 'ies'}; catching up")
                self.mode = Mode.ONLINE
                self.attempts, self._interval = 0, self.cfg.retry_interval_s
            self._push()

    def finish(self, status: str = "FINISHED") -> bool:
        """Final sync, artifacts, close the run. Returns True if MLflow has everything."""
        with self._lock:
            if self.mode is Mode.OFF:
                return False
            self.sync(force=True)
            if self.mode is not Mode.ONLINE:
                return False
            try:
                self._stop_sysmon()
                self._backend.artifacts(self.run_id, self.run_dir)  # type: ignore[union-attr]
                for rid in self.ledger["children"].values():
                    self._backend.set_status(rid, status)  # type: ignore[union-attr]
                self._backend.set_status(self.run_id, status)  # type: ignore[union-attr]
            except Exception as exc:  # noqa: BLE001
                self._handle_failure(exc)
                return False
            self.ledger["finished"] = status == "FINISHED"
            self._save_ledger()
            return True

    def abort(self) -> None:
        """Mark the MLflow run failed (best effort) when the experiment crashes."""
        with self._lock:
            if self.mode is Mode.ONLINE and self.run_id:
                try:
                    self._stop_sysmon()
                    self._backend.set_status(self.run_id, "FAILED")  # type: ignore[union-attr]
                except Exception:  # noqa: BLE001
                    pass

    # ---- internals ------------------------------------------------------- #

    def _push(self) -> None:
        # A local problem building the snapshot must never look like a server outage.
        try:
            snap = self.snapshot()
        except Exception as exc:  # noqa: BLE001
            if not getattr(self, "_snapshot_warned", False):
                _warn(f"could not read results for tracking ({type(exc).__name__}: {exc}); will retry next sync")
                self._snapshot_warned = True
            return
        try:
            self._push_unchecked(snap)
        except Exception as exc:  # noqa: BLE001  (classified below; never kills the run)
            self._handle_failure(exc)

    def _push_unchecked(self, snap: Snapshot) -> None:
        if self._backend is None:
            self._backend = self._backend_factory(self.uri, self.cfg.request_timeout_s)  # type: ignore[arg-type]
        b, led = self._backend, self.ledger
        reopened = False
        if not led["run_id"]:
            led["uri"] = self.uri
            led["experiment_id"] = b.experiment_id(self.experiment)
            led["run_id"] = b.create_run(led["experiment_id"], self.run_dir.name, {}, None)
            self._save_ledger()
        elif not getattr(self, "_reopened", False):
            b.set_status(led["run_id"], "RUNNING")  # resuming a run dir: reopen the same MLflow run
            reopened = True
        self._reopened = True
        self._push_run("", led["run_id"], snap.parent)
        for name, spec in snap.children.items():
            rid = led["children"].get(name)
            if not rid:
                rid = b.create_run(led["experiment_id"], name, {}, led["run_id"])
                led["children"][name] = rid
                self._save_ledger()
            self._push_run(name, rid, spec)
        if self.system_metrics and self._sysmon is None:
            self._sysmon = b.start_system_metrics(led["run_id"], self.cfg.system_metrics_interval_s, resume=reopened or bool(led["pushed"]))

    def _push_run(self, key: str, run_id: str, spec: RunSpec) -> None:
        led = self.ledger
        pushed = led["pushed"].setdefault(key or "__parent__", {})
        meta_key = key or "__parent__"
        params = flatten(spec.params) if meta_key not in led["meta"] else {}
        tags = flatten(spec.tags) if meta_key not in led["meta"] else {}
        points = [
            (k, v, s)
            for k, steps in spec.series.items()
            for s, v in sorted(steps.items())
            if s > pushed.get(k, -1)
        ]
        if not points and not params and not tags:
            return
        self._backend.log(run_id, points, params, tags)  # type: ignore[union-attr]
        for k, _, s in points:
            pushed[k] = max(pushed.get(k, -1), s)
        if meta_key not in led["meta"]:
            led["meta"].append(meta_key)
        self._save_ledger()

    def _handle_failure(self, exc: BaseException) -> None:
        status, detail = classify_exception(exc, self.uri, self.experiment)  # type: ignore[arg-type]
        if status is Status.UNAVAILABLE:
            if self.mode is Mode.ONLINE:
                self._go_offline(detail)
            else:
                self._retry_failed(detail)
        else:
            self._disable(status, detail)

    def _go_offline(self, detail: str) -> None:
        self._stop_sysmon()
        self.mode = Mode.OFFLINE
        self.attempts, self._interval = 0, self.cfg.retry_interval_s
        self._next_at = self._clock() + self._interval
        self.reason = detail
        _warn(
            f"MLflow at {self.uri} is not reachable ({detail}). The run continues and results are "
            f"saved locally; retrying quietly (first in {self._interval:.0f}s, up to "
            f"{self.cfg.retry_max_attempts} attempts)."
        )

    def _retry_failed(self, detail: str, final: bool = False) -> None:
        self.attempts += 1
        self.reason = detail
        if final:
            return  # the end-of-run attempt doesn't count against the budget or print
        if self.attempts >= self.cfg.retry_max_attempts:
            self.mode = Mode.GAVE_UP
            _warn(f"giving up on live tracking after {self.attempts} retries ({detail}); "
                  "will try once more at the end of the run")
            return
        self._interval = min(self._interval * 2, self.cfg.retry_max_interval_s)
        self._next_at = self._clock() + self._interval

    def _disable(self, status: Status, detail: str) -> None:
        self._stop_sysmon()
        self.mode = Mode.DISABLED
        self.reason = detail
        _warn(f"MLflow at {self.uri} rejected this client: {detail}. Tracking is disabled for this run; "
              f"{ADVICE.get(status, 'check the server')}, then upload the run with `track`.")

    def _stop_sysmon(self) -> None:
        if self._sysmon is not None and self._backend is not None:
            self._backend.stop_system_metrics(self._sysmon)
        self._sysmon = None
