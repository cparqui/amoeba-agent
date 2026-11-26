"""E0.0 runner: translate observations under every condition, judge, score.

One command does everything, starting and stopping a llama-server per model:

    uv run python -m experiments.translation pipeline -c experiments/translation/config.yaml

The stages can also be run on their own:

    uv run python -m experiments.translation run   -c experiments/translation/config.yaml
    uv run python -m experiments.translation judge runs/e0.0/<stamp> -c ...
    uv run python -m experiments.translation score runs/e0.0/<stamp>

Each stage is a plain function (config and run dir in, files in the run dir out),
so it can become a pipeline-framework step (e.g. ZenML `@step`) without changes.
"""

from __future__ import annotations

import asyncio
import csv
import json
import re
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Iterator

import click
import httpx
import yaml
from pydantic import BaseModel, Field

from ..common.llama_server import ServerError, llama_server
from ..common.tracking import TrackingConfig, provenance, tracking
from . import metrics
from .backends import Client, Endpoint
from .prompts import JUDGE_SYSTEM, JUDGE_USER, perceive_system, perceive_user
from .syntax import SYNTAXES, ParseError, facts, has_variable, render, strip_output, symbols

HERE = Path(__file__).parent
LEXICON_MODES = ("empty", "seeded", "evolving", "seeded_evolving")
DECODING = ("free", "constrained")


class Config(BaseModel):
    endpoints: list[Endpoint]
    models: list[str] | None = Field(default=None, description="Endpoint names to test; default all.")
    judge: str | None = None
    data: str = str(HERE / "data" / "observations.jsonl")
    out_dir: str = "runs/e0.0"
    syntaxes: list[str] = ["sexpr", "prolog", "json"]
    lexicon_modes: list[str] = ["empty", "seeded", "evolving"]
    decoding: list[str] = ["free", "constrained"]
    repeats: int = 3
    temperature: float = 0.7
    max_tokens: int = 300
    lexicon_cap: int = 150
    seed_lexicon: list[str] = []
    seed: int | None = Field(default=None, description="Sampling seed; repeat r uses seed + r. None = unseeded.")
    judge_repeats: list[int] = Field(default=[0], description="Which repeats the judge answers questions for.")
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)

    @classmethod
    def load(cls, path: str | Path) -> Config:
        cfg = cls.model_validate(yaml.safe_load(Path(path).read_text()))
        for s in cfg.syntaxes:
            if s not in SYNTAXES:
                raise click.BadParameter(f"unknown syntax {s!r}; choose from {list(SYNTAXES)}")
        for m in cfg.lexicon_modes:
            if m not in LEXICON_MODES:
                raise click.BadParameter(f"unknown lexicon mode {m!r}; choose from {LEXICON_MODES}")
        for n in [*(cfg.models or []), *([cfg.judge] if cfg.judge else [])]:
            cfg.endpoint(n)
        return cfg

    def model_names(self) -> list[str]:
        return self.models or [ep.name for ep in self.endpoints if ep.name != self.judge]

    def endpoint(self, name: str) -> Endpoint:
        for ep in self.endpoints:
            if ep.name == name:
                return ep
        raise click.BadParameter(f"no endpoint named {name!r}")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


# --------------------------------------------------------------------------- #
# Translation
# --------------------------------------------------------------------------- #


def translate_record(syntax: str, text: str) -> dict[str, Any]:
    """Parse one model output into the fields every record carries."""
    cleaned = strip_output(text)
    try:
        exprs = SYNTAXES[syntax].parse(cleaned)
    except ParseError as exc:
        return {"ok": False, "error": f"parse: {exc}", "rendered": [], "facts": [], "preds": [], "ents": [], "has_var": False}
    preds, ents = symbols(exprs)
    return {
        "ok": True,
        "error": None,
        "rendered": [render(e) for e in exprs],
        "facts": sorted(json.dumps(f) for f in facts(exprs)),
        "preds": sorted(preds),
        "ents": sorted(ents),
        "has_var": has_variable(exprs),
    }


async def run_condition(
    client: Client,
    cfg: Config,
    observations: list[dict[str, Any]],
    syntax: str,
    lexicon_mode: str,
    decoding: str,
    repeat: int,
    sink: Any,
) -> None:
    syn = SYNTAXES[syntax]
    system = perceive_system(syn)
    seed = list(dict.fromkeys(s.lower() for s in cfg.seed_lexicon))
    known: list[str] = list(seed) if lexicon_mode.startswith("seeded") else []
    grammar = syn.grammar if decoding == "constrained" else None

    for idx, obs in enumerate(observations):
        if lexicon_mode == "empty":
            shown: list[str] = []
        elif lexicon_mode == "seeded":
            shown = seed
        else:
            shown = known[-cfg.lexicon_cap :]
        before = set(known)
        rec: dict[str, Any] = {
            "model": client.ep.name, "syntax": syntax, "lexicon": lexicon_mode, "decoding": decoding,
            "repeat": repeat, "idx": idx, "obs_id": obs["id"], "group": obs.get("group"),
            "novel": bool(obs.get("novel")), "output": None,
            "latency_s": None, "prompt_tokens": 0, "completion_tokens": 0,
        }
        try:
            c = await client.complete(
                system, perceive_user(obs["content"], shown),
                grammar=grammar, temperature=cfg.temperature, max_tokens=cfg.max_tokens,
                seed=None if cfg.seed is None else cfg.seed + repeat,
            )
            rec.update(output=c.text, latency_s=c.latency_s,
                       prompt_tokens=c.prompt_tokens, completion_tokens=c.completion_tokens)
            rec.update(translate_record(syntax, c.text))
        except httpx.HTTPError as exc:
            rec.update(ok=False, error=f"http: {exc}", rendered=[], facts=[], preds=[], ents=[], has_var=False)

        used = [*rec["preds"], *rec["ents"]]
        rec["reused"] = sorted(s for s in used if s in before)
        rec["new"] = sorted(s for s in set(used) if s not in before)
        for s in used:
            if s not in known:
                known.append(s)
        rec["lexicon_after"] = len(known)
        sink(rec)


async def run_endpoint(
    ep: Endpoint, cfg: Config, observations: list[dict[str, Any]], done: set, sink: Any
) -> None:
    client = Client(ep)
    try:
        for decoding in cfg.decoding:
            if decoding == "constrained" and not ep.supports_grammar:
                click.echo(f"[{ep.name}] backend {ep.backend} takes no grammar; skipping constrained")
                continue
            for syntax in cfg.syntaxes:
                for mode in cfg.lexicon_modes:
                    for r in range(cfg.repeats):
                        if (ep.name, syntax, mode, decoding, r) in done:
                            continue
                        t0 = time.perf_counter()
                        await run_condition(client, cfg, observations, syntax, mode, decoding, r, sink)
                        click.echo(f"[{ep.name}] {syntax}/{mode}/{decoding} #{r} {time.perf_counter() - t0:.0f}s")
    finally:
        await client.aclose()




# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #

CONDITION_FIELDS = ("model", "syntax", "lexicon", "decoding", "repeat")


@contextmanager
def serving(ep: Endpoint, run_dir: Path | None) -> Iterator[tuple[Endpoint, dict[str, Any]]]:
    """Yield a reachable endpoint, starting a llama-server first if `ep.serve` is set."""
    if not ep.serve:
        yield ep, {}
        return
    log = run_dir / "servers" / f"{ep.name}.log" if run_dir else None
    click.echo(f"[{ep.name}] starting llama-server" + (f" (log: {log})" if log else ""))
    with llama_server(ep.serve, log) as srv:
        click.echo(f"[{ep.name}] ready on port {srv.port} after {srv.info['startup_s']}s")
        yield ep.model_copy(update={"base_url": srv.base_url}), srv.info
    click.echo(f"[{ep.name}] server stopped")


def _record_server(run_dir: Path, name: str, info: dict[str, Any]) -> None:
    if not info:
        return
    path = run_dir / "servers.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[name] = info
    path.write_text(json.dumps(data, indent=2))


def new_run_dir(cfg: Config, config_path: str | Path, resume: str | Path | None) -> Path:
    out = Path(resume) if resume else Path(cfg.out_dir) / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(Path(config_path).read_text())
    (out / "provenance.json").write_text(json.dumps(provenance(), indent=2))
    return out


def _conditions(cfg: Config, ep: Endpoint) -> list[tuple]:
    """Every (model, syntax, lexicon, decoding, repeat) this endpoint will run."""
    return [
        (ep.name, syntax, mode, decoding, r)
        for decoding in cfg.decoding
        if decoding != "constrained" or ep.supports_grammar
        for syntax in cfg.syntaxes
        for mode in cfg.lexicon_modes
        for r in range(cfg.repeats)
    ]


def translate_stage(cfg: Config, run_dir: Path, *, parallel: bool = False) -> dict[str, float]:
    """Translate every observation under every condition into results.jsonl.

    Models run one at a time by default. Each gets its own server, started and
    stopped around it, so latency is measured without other models competing for
    the same hardware. `parallel` starts all servers at once (faster, noisier).

    Resume works per whole condition-repeat: partial ones are dropped and rerun,
    because an evolving lexicon depends on every earlier step. Returns seconds
    spent translating per model.
    """
    observations = load_jsonl(cfg.data)
    results = run_dir / "results.jsonl"
    existing = load_jsonl(results)
    counts: dict[tuple, int] = {}
    for r in existing:
        k = tuple(r[f] for f in CONDITION_FIELDS)
        counts[k] = counts.get(k, 0) + 1
    done = {k for k, n in counts.items() if n == len(observations)}
    kept = [r for r in existing if tuple(r[f] for f in CONDITION_FIELDS) in done]
    results.write_text("".join(json.dumps(r) + "\n" for r in kept))

    endpoints = [cfg.endpoint(n) for n in cfg.model_names()]
    todo = [ep for ep in endpoints if any(k not in done for k in _conditions(cfg, ep))]
    for ep in endpoints:
        if ep not in todo:
            click.echo(f"[{ep.name}] all conditions done; skipping")

    timings: dict[str, float] = {}
    with results.open("a") as fh:

        def sink(rec: dict[str, Any]) -> None:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()

        if parallel:
            with ExitStack() as stack:
                live = []
                for ep in todo:
                    lep, info = stack.enter_context(serving(ep, run_dir))
                    _record_server(run_dir, ep.name, info)
                    live.append(lep)

                async def main() -> None:
                    await asyncio.gather(*(run_endpoint(ep, cfg, observations, done, sink) for ep in live))

                t0 = time.perf_counter()
                asyncio.run(main())
                timings["all"] = time.perf_counter() - t0
        else:
            for ep in todo:
                with serving(ep, run_dir) as (lep, info):
                    _record_server(run_dir, ep.name, info)
                    t0 = time.perf_counter()
                    asyncio.run(run_endpoint(lep, cfg, observations, done, sink))
                    timings[ep.name] = time.perf_counter() - t0
    click.echo(f"wrote {results}")
    return timings


def judge_stage(
    cfg: Config, run_dir: Path, *, judge_name: str | None = None, repeats: list[int] | None = None
) -> Path:
    """Fidelity: answer questions from the translations alone, and from the raw input."""
    name = judge_name or cfg.judge
    if not name:
        raise click.UsageError("set judge in the config or pass --judge")
    want = set(cfg.judge_repeats if repeats is None else repeats)
    observations = {o["id"]: o for o in load_jsonl(cfg.data)}
    records = [r for r in load_jsonl(run_dir / "results.jsonl") if r["repeat"] in want]
    out = run_dir / "judge.jsonl"

    async def main(ep: Endpoint) -> list[dict[str, Any]]:
        client = Client(ep)
        rows: list[dict[str, Any]] = []

        async def ask(info: str, q: dict[str, Any], base: dict[str, Any]) -> None:
            c = await client.complete(
                JUDGE_SYSTEM, JUDGE_USER.format(info=info, question=q["q"]), temperature=0.0, max_tokens=40
            )
            ans = strip_output(c.text)
            rows.append({**base, "question": q["q"], "answer": ans, "correct": _correct(ans, q["a"])})

        try:
            for o in observations.values():
                for q in o["qa"]:
                    await ask(o["content"], q, {"source": "raw", "obs_id": o["id"]})
            for r in records:
                info = "\n".join(r["rendered"]) or "(nothing)"
                for q in observations[r["obs_id"]]["qa"]:
                    base = {k: r[k] for k in (*metrics.CONDITION_KEYS, "repeat")}
                    await ask(info, q, {"source": "expr", "obs_id": r["obs_id"], **base})
        finally:
            await client.aclose()
        return rows

    with serving(cfg.endpoint(name), run_dir) as (ep, info):
        _record_server(run_dir, f"judge:{name}", info)
        rows = asyncio.run(main(ep))
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    click.echo(f"wrote {out}")
    return out


def score_stage(run_dir: Path) -> list[dict[str, Any]]:
    """Summarize a run into summary.csv and summary.md."""
    rows = metrics.summarize(load_jsonl(run_dir / "results.jsonl"), load_jsonl(run_dir / "judge.jsonl"))
    with (run_dir / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=metrics.COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (run_dir / "summary.md").write_text(metrics.markdown_table(rows) + "\n")
    return rows


def _correct(answer: str, accepted: list[str]) -> bool:
    norm = lambda s: re.sub(r"[^a-z0-9.,()+:]+", " ", s.lower()).strip()  # noqa: E731
    a = norm(answer)
    return any(norm(x) and norm(x) in a for x in accepted)


def condition_name(row: dict[str, Any]) -> str:
    if row.get("model") == "(raw input)":
        return "raw-input"
    return "/".join(str(row.get(k)) for k in metrics.CONDITION_KEYS)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


@click.group()
def cli() -> None:
    """E0.0: can off-the-shelf models translate observations into the internal language?"""


@cli.command()
@click.option("-c", "--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--resume", type=click.Path(exists=True, file_okay=False), help="Continue an earlier run dir.")
@click.option("--parallel", is_flag=True, help="Run all models at once (noisier latency).")
@click.option("--judge/--no-judge", "do_judge", default=True, show_default=True, help="Run the fidelity judge.")
@click.option("--track/--no-track", default=True, show_default=True, help="Log to MLflow when a tracking URI is set.")
def pipeline(config_path: str, resume: str | None, parallel: bool, do_judge: bool, track: bool) -> None:
    """Run the whole experiment: serve each model, translate, judge, score, track."""
    cfg = Config.load(config_path)
    if not track:
        cfg.tracking.enabled = False
    run_dir = new_run_dir(cfg, config_path, resume)
    click.echo(f"run dir: {run_dir}")
    prov = json.loads((run_dir / "provenance.json").read_text())

    with tracking(cfg.tracking, "e0.0-translation", run_dir.name) as tr:
        if tr.active:
            click.echo(f"tracking to MLflow at {cfg.tracking.resolved_uri()}")
        tr.tags({
            "experiment": "E0.0", "run_dir": str(run_dir), "git_commit": prov["git_commit"],
            "git_dirty": prov["git_dirty"], "host": prov["host"],
        })
        tr.params({
            "models": cfg.model_names(), "judge": cfg.judge, "syntaxes": cfg.syntaxes,
            "lexicon_modes": cfg.lexicon_modes, "decoding": cfg.decoding, "repeats": cfg.repeats,
            "temperature": cfg.temperature, "max_tokens": cfg.max_tokens, "lexicon_cap": cfg.lexicon_cap,
            "seed": cfg.seed, "judge_repeats": cfg.judge_repeats, "parallel": parallel,
            "data": Path(cfg.data).name,
        })
        try:
            timings = translate_stage(cfg, run_dir, parallel=parallel)
            tr.metrics({f"translate_s.{k}": v for k, v in timings.items()})
            if do_judge and cfg.judge:
                judge_stage(cfg, run_dir)
            rows = score_stage(run_dir)
            for row in rows:
                with tr.child(condition_name(row)) as c:
                    c.params({k: row.get(k) for k in metrics.CONDITION_KEYS})
                    c.metrics(row)
        finally:
            tr.artifacts(run_dir)
    click.echo((run_dir / "summary.md").read_text())


@cli.command()
@click.option("-c", "--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--resume", type=click.Path(exists=True, file_okay=False), help="Continue an earlier run dir.")
@click.option("--parallel", is_flag=True, help="Run all models at once (noisier latency).")
def run(config_path: str, resume: str | None, parallel: bool) -> None:
    """Translate every observation under every condition (no judge, no tracking)."""
    cfg = Config.load(config_path)
    translate_stage(cfg, new_run_dir(cfg, config_path, resume), parallel=parallel)


@cli.command()
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
@click.option("-c", "--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--judge", "judge_name", help="Endpoint to answer questions; default config.judge.")
@click.option("--repeat", "repeats", multiple=True, type=int, help="Repeats to judge; default config.judge_repeats.")
def judge(run_dir: str, config_path: str, judge_name: str | None, repeats: tuple[int, ...]) -> None:
    """Fidelity: answer questions from the translations alone, and from the raw input."""
    judge_stage(Config.load(config_path), Path(run_dir), judge_name=judge_name, repeats=list(repeats) or None)


@cli.command()
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
def score(run_dir: str) -> None:
    """Summarize a run into summary.csv and summary.md."""
    click.echo(metrics.markdown_table(score_stage(Path(run_dir))))


@cli.command()
@click.option("-c", "--config", "config_path", required=True, type=click.Path(exists=True))
def check(config_path: str) -> None:
    """Start (if managed) and ping every endpoint with one translation."""
    cfg = Config.load(config_path)

    async def ping(ep: Endpoint) -> None:
        client = Client(ep)
        try:
            syn = SYNTAXES["sexpr"]
            g = syn.grammar if ep.supports_grammar else None
            c = await client.complete(
                perceive_system(syn), perceive_user("The cup is on the table.", []), grammar=g, max_tokens=60
            )
            click.echo(f"[ok]   {ep.name}: {strip_output(c.text)!r} ({c.latency_s:.2f}s)")
        finally:
            await client.aclose()

    for ep in cfg.endpoints:
        try:
            with serving(ep, None) as (live, _):
                asyncio.run(ping(live))
        except (ServerError, httpx.HTTPError, ValueError) as exc:
            click.echo(f"[fail] {ep.name}: {exc}")
