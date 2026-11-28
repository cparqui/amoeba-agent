"""Live MLflow tracking: logging as results land, outages, bad credentials, backfill."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Iterator

import pytest
from click.testing import CliRunner

from experiments.common.llama_server import free_port
from experiments.common.tracking import (
    LEDGER,
    LiveRun,
    Mode,
    RunSpec,
    Snapshot,
    Status,
    TrackingConfig,
    add_point,
    probe,
)
from experiments.translation.run import cli
from tests.test_pipeline import _config, fake_bin  # noqa: F401  (fixture)

# --------------------------------------------------------------------------- #
# A fake MLflow server that answers every API call with one fixed response
# --------------------------------------------------------------------------- #


@pytest.fixture
def fake_mlflow() -> Iterator:
    servers: list[HTTPServer] = []

    def make(status: int, body: str = "") -> str:
        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _reply(self):
                data = body.encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_GET = do_POST = _reply

        srv = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return f"http://127.0.0.1:{srv.server_port}"

    yield make
    for srv in servers:
        srv.shutdown()


def test_probe_classifies_responses(fake_mlflow) -> None:
    assert probe(fake_mlflow(200, "{}"))[0] is Status.OK
    assert probe(fake_mlflow(404, '{"error_code": "RESOURCE_DOES_NOT_EXIST"}'))[0] is Status.OK
    assert probe(fake_mlflow(401, "You are not authenticated."))[0] is Status.AUTH
    assert probe(fake_mlflow(403, "Invalid Host header - possible DNS rebinding attack detected"))[0] is Status.HOST
    assert probe(fake_mlflow(403, '{"error_code": "PERMISSION_DENIED"}'))[0] is Status.FORBIDDEN
    assert probe(fake_mlflow(503, "down"))[0] is Status.UNAVAILABLE
    assert probe(f"http://127.0.0.1:{free_port()}")[0] is Status.UNAVAILABLE
    assert probe(fake_mlflow(404, "<html>not mlflow</html>"))[0] is Status.ERROR
    assert probe("file:///tmp/mlruns")[0] is Status.OK


def test_probe_sends_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str | None] = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            seen.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "admin")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "pw")
    try:
        assert probe(f"http://127.0.0.1:{srv.server_port}")[0] is Status.OK
    finally:
        srv.shutdown()
    assert seen == ["Basic YWRtaW46cHc="]  # base64("admin:pw")


# --------------------------------------------------------------------------- #
# LiveRun state machine, with a fake backend, probe and clock
# --------------------------------------------------------------------------- #


class Down(Exception):
    """Stands in for a connection error from the MLflow client."""


class FakeBackend:
    def __init__(self) -> None:
        self.down = False
        self.points: dict[tuple[str, str], list[int]] = {}
        self.runs: dict[str, dict] = {}
        self.sysmon_starts = 0

    def _check(self) -> None:
        if self.down:
            raise Down("connection refused")

    def experiment_id(self, name: str) -> str:
        self._check()
        return "1"

    def create_run(self, experiment_id, name, tags, parent):
        self._check()
        rid = f"run{len(self.runs)}"
        self.runs[rid] = {"name": name, "parent": parent, "status": "RUNNING"}
        return rid

    def log(self, run_id, metrics, params, tags):
        self._check()
        for k, _, s in metrics:
            self.points.setdefault((run_id, k), []).append(s)

    def set_status(self, run_id, status):
        self._check()
        self.runs[run_id]["status"] = status

    def artifacts(self, run_id, path):
        self._check()

    def start_system_metrics(self, run_id, interval, resume):
        self.sysmon_starts += 1
        return object()

    def stop_system_metrics(self, handle):
        pass


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _make(tmp_path: Path, **cfg):
    backend, clock = FakeBackend(), Clock()
    probes: list[str] = []
    state = {"steps": 1, "server": Status.OK}

    def snap() -> Snapshot:
        child = RunSpec(params={"model": "m"})
        for k in range(1, state["steps"] + 1):
            add_point(child.series, "parse_rate", k, 1.0)
        return Snapshot(parent=RunSpec(params={"x": 1}), children={"m/sexpr/evolving/free": child})

    def fake_probe(uri, exp):
        probes.append(uri)
        return state["server"], "fake"

    live = LiveRun(
        TrackingConfig(uri="http://mlflow:5000", retry_interval_s=30, retry_max_interval_s=120, **cfg),
        "exp", tmp_path, snap,
        backend_factory=lambda uri, timeout: backend, probe_fn=fake_probe, clock=clock,
    )
    return live, backend, clock, probes, state


def test_outage_retries_quietly_and_catches_up(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    live, backend, clock, probes, state = _make(tmp_path)
    live.start()
    assert live.mode is Mode.ONLINE and backend.sysmon_starts == 1

    # The server drops while logging step 2: one warning, then offline.
    backend.down, state["server"], state["steps"] = True, Status.UNAVAILABLE, 2
    live.sync()
    assert live.mode is Mode.OFFLINE
    err = capsys.readouterr().err
    assert err.count("not reachable") == 1 and "retrying quietly" in err

    # Syncs before the retry is due don't even probe.
    n = len(probes)
    for _ in range(10):
        live.sync()
    assert len(probes) == n

    # Due retries that fail are silent and back off (30s -> 60s -> 120s cap).
    state["steps"] = 3
    for wait in (30, 60, 120):
        clock.t += wait
        live.sync()
    assert live.attempts == 3 and capsys.readouterr().err == ""

    # The server comes back: one line, then everything missing is logged once.
    backend.down, state["server"] = False, Status.OK
    clock.t += 120
    live.sync()
    assert live.mode is Mode.ONLINE
    assert "reachable again after 3 retries" in capsys.readouterr().err
    child = live.ledger["children"]["m/sexpr/evolving/free"]
    assert backend.points[(child, "parse_rate")] == [1, 2, 3]  # no gaps, no duplicates
    assert backend.sysmon_starts == 2  # restarted after the outage

    assert live.finish()
    assert backend.runs[live.run_id]["status"] == "FINISHED"
    assert json.loads((tmp_path / LEDGER).read_text())["finished"] is True


def test_retry_limit_then_final_attempt(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    live, backend, clock, probes, state = _make(tmp_path, retry_max_attempts=2)
    backend.down, state["server"] = True, Status.UNAVAILABLE
    live.start()
    assert live.mode is Mode.OFFLINE
    for _ in range(5):
        clock.t += 1000
        live.sync()
    assert live.mode is Mode.GAVE_UP
    err = capsys.readouterr().err
    assert err.count("giving up") == 1 and err.count("not reachable") == 1
    n = len(probes)
    clock.t += 10_000
    live.sync()
    assert len(probes) == n  # no more retries once given up

    # finish() still makes one last attempt, and succeeds if the server is back.
    backend.down, state["server"] = False, Status.OK
    assert live.finish()


def test_bad_credentials_disable_tracking(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    live, backend, clock, probes, state = _make(tmp_path)
    state["server"] = Status.AUTH
    live.start()
    assert live.mode is Mode.DISABLED and not backend.runs
    err = capsys.readouterr().err
    assert "Tracking is disabled for this run" in err and "MLFLOW_TRACKING_USERNAME" in err
    n = len(probes)
    clock.t += 10_000
    live.sync()
    assert len(probes) == n and not live.finish()  # never retried


# --------------------------------------------------------------------------- #
# Through the CLI
# --------------------------------------------------------------------------- #


def _mlflow_runs(uri: str, experiment: str):
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=[experiment], output_format="list")
    parents = [r for r in runs if "mlflow.parentRunId" not in r.data.tags]
    children = {r.info.run_name: r for r in runs if "mlflow.parentRunId" in r.data.tags}
    return MlflowClient(uri), parents, children


def test_pipeline_logs_live(fake_bin: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    uri = (tmp_path / "mlruns").as_uri()
    tracking = {"uri": uri, "experiment": "live", "system_metrics_interval_s": 0.1}
    res = CliRunner().invoke(cli, ["pipeline", "-c", str(_config(tmp_path, tracking=tracking))], catch_exceptions=False)
    assert res.exit_code == 0, res.output
    assert "tracking live to MLflow" in res.output

    client, parents, children = _mlflow_runs(uri, "live")
    assert len(parents) == 1
    parent = parents[0]
    assert parent.info.status == "FINISHED"
    assert parent.data.metrics["progress.fraction"] == 1.0
    assert {"translate_s.small", "translate_s.tiny"} <= set(parent.data.metrics)
    assert any(k.startswith("system/") for k in parent.data.metrics), parent.data.metrics.keys()

    # Metrics were logged once per finished repeat (steps 1 and 2), fidelity at the end.
    run = children["small/sexpr/evolving/free"]
    steps = [m.step for m in client.get_metric_history(run.info.run_id, "parse_rate")]
    assert steps == [1, 2]
    assert [m.step for m in client.get_metric_history(run.info.run_id, "fidelity")] == [2]
    assert run.info.status == "FINISHED" and "raw-input" in children

    run_dir = next((tmp_path / "runs").iterdir())
    led = json.loads((run_dir / LEDGER).read_text())
    assert led["finished"] and led["run_id"] == parent.info.run_id


def test_resume_continues_the_same_mlflow_run(fake_bin: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    uri = (tmp_path / "mlruns").as_uri()
    cfg = str(_config(tmp_path, tracking={"uri": uri, "experiment": "resume", "system_metrics": False}))
    runner = CliRunner()
    assert runner.invoke(cli, ["pipeline", "-c", cfg], catch_exceptions=False).exit_code == 0
    run_dir = next((tmp_path / "runs").iterdir())
    res = runner.invoke(cli, ["pipeline", "-c", cfg, "--resume", str(run_dir)], catch_exceptions=False)
    assert res.exit_code == 0, res.output
    client, parents, children = _mlflow_runs(uri, "resume")
    assert len(parents) == 1  # same MLflow run, not a second one
    steps = [m.step for m in client.get_metric_history(children["small/sexpr/evolving/free"].info.run_id, "parse_rate")]
    assert steps == [1, 2]  # nothing logged twice


def test_dead_server_run_continues(fake_bin: Path, tmp_path: Path) -> None:  # noqa: F811
    uri = f"http://127.0.0.1:{free_port()}"
    res = CliRunner().invoke(cli, ["pipeline", "-c", str(_config(tmp_path, tracking={"uri": uri}))])
    assert res.exit_code == 0, res.output
    assert res.output.count("not reachable") == 1  # warned once, not per condition
    assert "MLflow does not have the complete run" in res.output and "track" in res.output
    run_dir = next((tmp_path / "runs").iterdir())
    assert (run_dir / "summary.md").exists()


def test_growing_jsonl_skips_half_written_line(tmp_path: Path) -> None:
    from experiments.translation.run import load_jsonl

    f = tmp_path / "r.jsonl"
    f.write_text('{"a": 1}\n{"a": 2}\n{"a": 3')
    assert load_jsonl(f, growing=True) == [{"a": 1}, {"a": 2}]
    with pytest.raises(json.JSONDecodeError):
        load_jsonl(f)


def test_snapshot_error_is_not_an_outage(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    live, backend, clock, probes, state = _make(tmp_path)
    live.start()

    def broken() -> Snapshot:
        raise ValueError("half-written line")

    live.snapshot = broken
    live.sync()
    live.sync()
    assert live.mode is Mode.ONLINE
    assert capsys.readouterr().err.count("could not read results") == 1


def test_parallel_pipeline_logs_live(fake_bin: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    uri = (tmp_path / "mlruns").as_uri()
    cfg = _config(tmp_path, tracking={"uri": uri, "experiment": "par", "system_metrics": False})
    res = CliRunner().invoke(cli, ["pipeline", "-c", str(cfg), "--parallel"], catch_exceptions=False)
    assert res.exit_code == 0, res.output
    assert "warning" not in res.output
    client, parents, children = _mlflow_runs(uri, "par")
    assert parents[0].info.status == "FINISHED" and parents[0].data.metrics["progress.fraction"] == 1.0
    for name in ("small/sexpr/evolving/free", "tiny/sexpr/evolving/constrained"):
        assert [m.step for m in client.get_metric_history(children[name].info.run_id, "parse_rate")] == [1, 2]


@pytest.mark.parametrize(
    "status,body,advice",
    [
        (401, "You are not authenticated.", "MLFLOW_TRACKING_USERNAME"),
        (403, "Invalid Host header - possible DNS rebinding attack detected", "--allowed-hosts"),
    ],
)
def test_rejected_client_disables_tracking(fake_bin: Path, tmp_path: Path, fake_mlflow, status, body, advice) -> None:  # noqa: F811
    uri = fake_mlflow(status, body)
    res = CliRunner().invoke(cli, ["pipeline", "-c", str(_config(tmp_path, tracking={"uri": uri}))])
    assert res.exit_code == 0, res.output
    assert "Tracking is disabled for this run" in res.output and advice in res.output
    assert (next((tmp_path / "runs").iterdir()) / "summary.md").exists()


def test_tracking_required_by_default(fake_bin: Path, tmp_path: Path) -> None:  # noqa: F811
    res = CliRunner().invoke(cli, ["pipeline", "-c", str(_config(tmp_path))])
    assert res.exit_code != 0 and "no MLflow URI is set" in res.output
    assert not (tmp_path / "runs").exists() or not any((tmp_path / "runs").iterdir())
    assert not (tmp_path / "requests.jsonl").exists()


def test_preflight_lists_every_problem(fake_bin: Path, tmp_path: Path) -> None:  # noqa: F811
    cfg = _config(tmp_path, tracking={"enabled": False})
    import yaml

    data = yaml.safe_load(Path(cfg).read_text())
    data["data"] = str(tmp_path / "missing.jsonl")
    data["endpoints"][1]["serve"]["model_path"] = str(tmp_path / "nope.gguf")
    data["endpoints"].append({"name": "hosted", "base_url": "https://api.example/v1", "backend": "openai",
                              "api_key_env": "AMOEBA_TEST_UNSET_KEY"})
    data["models"] = ["small", "tiny", "hosted"]
    Path(cfg).write_text(yaml.safe_dump(data))
    res = CliRunner().invoke(cli, ["pipeline", "-c", cfg])
    assert res.exit_code != 0
    for msg in ("missing.jsonl does not exist", "nope.gguf does not exist", "$AMOEBA_TEST_UNSET_KEY"):
        assert msg in res.output


def test_no_track_then_backfill(fake_bin: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    runner = CliRunner()
    res = runner.invoke(cli, ["pipeline", "-c", str(_config(tmp_path)), "--no-track"], catch_exceptions=False)
    assert res.exit_code == 0 and "upload later with" in res.output
    run_dir = next((tmp_path / "runs").iterdir())

    uri = (tmp_path / "mlruns").as_uri()
    res = runner.invoke(cli, ["track", str(run_dir), "--uri", uri], catch_exceptions=False)
    assert res.exit_code == 0, res.output
    assert json.loads((run_dir / LEDGER).read_text())["finished"]

    again = runner.invoke(cli, ["track", str(run_dir), "--uri", uri])
    assert again.exit_code != 0 and "already logged" in again.output

    _, parents, children = _mlflow_runs(uri, "e0.0-translation")
    assert len(parents) == 1
    assert parents[0].data.params["parallel"] == "False" and "translate_s.small" in parents[0].data.metrics
    assert "small/sexpr/evolving/free" in children
