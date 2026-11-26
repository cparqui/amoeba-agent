"""Server lifecycle and the one-command pipeline, against a fake llama-server."""

from __future__ import annotations

import json
import os
import stat
import sys
import textwrap
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from experiments.common.llama_server import ServerError, ServeSpec, llama_server
from experiments.translation.backends import Endpoint
from experiments.translation.run import cli

FAKE = textwrap.dedent(
    f"""\
    #!{sys.executable}
    # Minimal stand-in for llama-server: --version, /health, /props, /v1/chat/completions.
    import json, os, sys, time
    from http.server import BaseHTTPRequestHandler, HTTPServer

    argv = sys.argv[1:]
    if "--version" in argv:
        print("version: 9999 (fake)", file=sys.stderr)
        sys.exit(0)
    if os.environ.get("FAKE_CRASH"):
        print("error: failed to load model", file=sys.stderr)
        sys.exit(3)
    port = int(argv[argv.index("--port") + 1])
    model = argv[argv.index("-hf") + 1] if "-hf" in argv else argv[argv.index("-m") + 1]
    started = time.monotonic()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                # 503 while "loading", like the real server
                ok = time.monotonic() - started > 0.3
                self._json(200 if ok else 503, {{"status": "ok" if ok else "loading"}})
            elif self.path == "/props":
                self._json(200, {{"model_path": model, "chat_template": "x" * 100}})
            else:
                self._json(404, {{}})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with open(os.environ["FAKE_REQUESTS"], "a") as fh:
                fh.write(json.dumps({{"model": model, "seed": req.get("seed")}}) + "\\n")
            self._json(200, {{
                "choices": [{{"message": {{"content": "(on cup table)"}}}}],
                "usage": {{"prompt_tokens": 10, "completion_tokens": 4}},
            }})

    print(f"listening on {{port}}", flush=True)
    HTTPServer(("127.0.0.1", port), H).serve_forever()
    """
)


@pytest.fixture
def fake_bin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "fake-llama-server"
    path.write_text(FAKE)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("LLAMA_SERVER", str(path))
    monkeypatch.setenv("FAKE_REQUESTS", str(tmp_path / "requests.jsonl"))
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    return path


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_server_lifecycle(fake_bin: Path, tmp_path: Path) -> None:
    log = tmp_path / "srv.log"
    with llama_server(ServeSpec(hf="org/model:Q4", startup_timeout=20), log) as srv:
        assert srv.base_url.endswith(f":{srv.port}/v1")
        assert "fake" in srv.info["version"]
        assert srv.info["props"] == {"model_path": "org/model:Q4"}  # template dropped
        pid = srv.pid
        assert _alive(pid)
    assert not _alive(pid)
    assert "-hf org/model:Q4" in log.read_text()


def test_server_crash_reports_log(fake_bin: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_CRASH", "1")
    with pytest.raises(ServerError, match="failed to load model"):
        with llama_server(ServeSpec(hf="x", startup_timeout=20), tmp_path / "srv.log"):
            pass


def test_missing_binary() -> None:
    with pytest.raises(ServerError, match="not found"):
        with llama_server(ServeSpec(hf="x", bin="/nonexistent/llama-server")):
            pass


def test_endpoint_needs_url_or_serve() -> None:
    with pytest.raises(ValueError):
        Endpoint(name="a")
    with pytest.raises(ValueError):
        ServeSpec(hf="a", model_path="b")
    assert Endpoint(name="a", serve={"hf": "x"}).managed


def _config(tmp_path: Path, **extra) -> Path:
    data = tmp_path / "obs.jsonl"
    data.write_text(
        "\n".join(
            json.dumps(o)
            for o in [
                {"id": "a", "content": "The cup is on the table.", "group": "cup", "qa": [{"q": "Where is the cup?", "a": ["table"]}]},
                {"id": "b", "content": "On the table sits a cup.", "group": "cup", "qa": [{"q": "What is on the table?", "a": ["cup"]}]},
            ]
        )
        + "\n"
    )
    cfg = {
        "endpoints": [
            {"name": "small", "serve": {"hf": "org/small:Q8", "startup_timeout": 20}},
            {"name": "tiny", "serve": {"model_path": "/models/tiny.gguf", "startup_timeout": 20}},
            {"name": "judge", "serve": {"hf": "org/judge:Q4", "startup_timeout": 20}},
        ],
        "judge": "judge",
        "data": str(data),
        "out_dir": str(tmp_path / "runs"),
        "syntaxes": ["sexpr"],
        "lexicon_modes": ["evolving"],
        "decoding": ["free", "constrained"],
        "repeats": 2,
        "seed": 100,
        **extra,
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_pipeline_end_to_end(fake_bin: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")  # test-only local store
    cfg = _config(tmp_path, tracking={"uri": (tmp_path / "mlruns").as_uri(), "experiment": "test-e0.0"})
    res = CliRunner().invoke(cli, ["pipeline", "-c", str(cfg)], catch_exceptions=False)
    assert res.exit_code == 0, res.output

    run_dir = next((tmp_path / "runs").iterdir())
    for f in ("results.jsonl", "judge.jsonl", "summary.md", "summary.csv", "provenance.json", "servers.json"):
        assert (run_dir / f).exists(), f
    # 2 models x 2 decodings x 2 repeats x 2 observations
    assert len((run_dir / "results.jsonl").read_text().splitlines()) == 16
    servers = json.loads((run_dir / "servers.json").read_text())
    assert set(servers) == {"small", "tiny", "judge:judge"}
    assert (run_dir / "servers" / "small.log").exists()

    # Servers ran one at a time, in order, and the seed varies per repeat.
    reqs = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    order = [r["model"] for r in reqs]
    first_tiny = order.index("/models/tiny.gguf")
    assert set(order[:first_tiny]) == {"org/small:Q8"}
    assert {r["seed"] for r in reqs if r["seed"] is not None} == {100, 101}

    # MLflow: one parent run plus a child per summary row.
    import mlflow

    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    runs = mlflow.search_runs(experiment_names=["test-e0.0"], output_format="list")
    parents = [r for r in runs if "mlflow.parentRunId" not in r.data.tags]
    children = [r for r in runs if "mlflow.parentRunId" in r.data.tags]
    assert len(parents) == 1 and parents[0].data.tags["experiment"] == "E0.0"
    names = {r.info.run_name for r in children}
    assert "small/sexpr/evolving/free" in names and "raw-input" in names
    child = next(r for r in children if r.info.run_name == "small/sexpr/evolving/free")
    assert child.data.metrics["parse_rate"] == 1.0
    assert child.data.params["model"] == "small"


def test_resume_skips_finished_models(fake_bin: Path, tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    runner = CliRunner()
    assert runner.invoke(cli, ["run", "-c", str(cfg)], catch_exceptions=False).exit_code == 0
    run_dir = next((tmp_path / "runs").iterdir())
    (tmp_path / "requests.jsonl").unlink()
    res = runner.invoke(cli, ["run", "-c", str(cfg), "--resume", str(run_dir)], catch_exceptions=False)
    assert "all conditions done" in res.output
    assert not (tmp_path / "requests.jsonl").exists()  # no server started, no calls made
