"""Start, health-check and stop a `llama-server` for one model.

    spec = ServeSpec(hf="ggml-org/gemma-3-1b-it-GGUF:Q4_K_M", args=["-c", "4096"])
    with llama_server(spec, log_path=run_dir / "servers" / "gemma.log") as srv:
        srv.base_url   # http://127.0.0.1:<free port>/v1
        srv.info       # build, version, /props: recorded as run provenance

The server owns its port (picked free at start), its log file and its process
group, so an experiment can loop over models without anything started by hand.
Shutdown always runs, including on Ctrl-C or an exception in the body.
"""

from __future__ import annotations

import os
import shlex
import signal
import socket
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import httpx
from pydantic import BaseModel, Field, model_validator

DEFAULT_BIN = "llama-server"


class ServeSpec(BaseModel):
    """How to launch one model. Exactly one of `hf` or `model_path`."""

    hf: str | None = Field(default=None, description="Hugging Face repo[:quant], passed as -hf.")
    model_path: str | None = Field(default=None, description="Local GGUF file, passed as -m.")
    args: list[str] = Field(default_factory=list, description="Extra llama-server flags, e.g. ['-c', '4096', '-ngl', '99'].")
    bin: str | None = Field(default=None, description="llama-server binary; default $LLAMA_SERVER or 'llama-server'.")
    host: str = "127.0.0.1"
    port: int | None = Field(default=None, description="Fixed port; default a free one.")
    startup_timeout: float = Field(default=900.0, description="Seconds to wait for /health (covers first-run downloads).")
    env: dict[str, str] = Field(default_factory=dict, description="Extra environment, e.g. CUDA_VISIBLE_DEVICES.")

    @model_validator(mode="after")
    def _one_model(self) -> ServeSpec:
        if bool(self.hf) == bool(self.model_path):
            raise ValueError("serve needs exactly one of 'hf' or 'model_path'")
        return self

    def binary(self) -> str:
        return self.bin or os.environ.get("LLAMA_SERVER") or DEFAULT_BIN

    def command(self, port: int) -> list[str]:
        model = ["-hf", self.hf] if self.hf else ["-m", str(self.model_path)]
        return [self.binary(), *model, "--host", self.host, "--port", str(port), *self.args]


@dataclass
class RunningServer:
    spec: ServeSpec
    port: int
    pid: int
    log_path: Path | None
    command: list[str]
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def root_url(self) -> str:
        return f"http://{self.spec.host}:{self.port}"

    @property
    def base_url(self) -> str:
        return f"{self.root_url}/v1"


class ServerError(RuntimeError):
    pass


def free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def binary_version(binary: str) -> str | None:
    """`llama-server --version` output (it prints to stderr), or None."""
    try:
        p = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    out = (p.stdout + p.stderr).strip()
    return out or None


def _tail(path: Path | None, n: int = 30) -> str:
    if not path or not path.exists():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-n:])


def _wait_healthy(proc: subprocess.Popen, url: str, timeout: float, log_path: Path | None) -> None:
    """Poll /health until 200. llama-server answers 503 while the model loads."""
    deadline = time.monotonic() + timeout
    with httpx.Client(timeout=5.0) as http:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise ServerError(f"llama-server exited with code {proc.returncode}\n{_tail(log_path)}")
            try:
                if http.get(f"{url}/health").status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
    raise ServerError(f"llama-server not healthy after {timeout:.0f}s\n{_tail(log_path)}")


def _props(url: str) -> dict[str, Any]:
    """Best-effort /props: model path, context size, build info on recent builds."""
    try:
        r = httpx.get(f"{url}/props", timeout=10.0)
        if r.status_code == 200:
            data = r.json()
            # Drop the (long) chat template; keep identifying fields.
            data.pop("chat_template", None)
            data.pop("chat_template_tool_use", None)
            return data
    except (httpx.HTTPError, ValueError):
        pass
    return {}


def _stop(proc: subprocess.Popen, grace: float = 20.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=10)


@contextmanager
def llama_server(spec: ServeSpec, log_path: Path | None = None) -> Iterator[RunningServer]:
    """Run one llama-server for the duration of the block."""
    port = spec.port or free_port(spec.host)
    cmd = spec.command(port)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("ab") if log_path else subprocess.DEVNULL
    if log_path:
        log.write(f"$ {shlex.join(cmd)}\n".encode())  # type: ignore[union-attr]
        log.flush()  # type: ignore[union-attr]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env={**os.environ, **spec.env},
            start_new_session=True,  # own process group, so shutdown reaches children too
        )
    except FileNotFoundError as exc:
        if log_path:
            log.close()  # type: ignore[union-attr]
        raise ServerError(
            f"{spec.binary()!r} not found; install llama.cpp or set LLAMA_SERVER / serve.bin"
        ) from exc

    srv = RunningServer(spec=spec, port=port, pid=proc.pid, log_path=log_path, command=cmd)
    try:
        started = time.monotonic()
        _wait_healthy(proc, srv.root_url, spec.startup_timeout, log_path)
        srv.info = {
            "command": shlex.join(cmd),
            "version": binary_version(spec.binary()),
            "startup_s": round(time.monotonic() - started, 2),
            "props": _props(srv.root_url),
        }
        yield srv
    finally:
        _stop(proc)
        if log_path:
            log.close()  # type: ignore[union-attr]
