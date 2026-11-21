"""E0.0 runner: translate observations under every condition, judge, score.

    uv run python -m experiments.translation run   -c experiments/translation/config.yaml
    uv run python -m experiments.translation judge runs/e0.0/<stamp> -c ...
    uv run python -m experiments.translation score runs/e0.0/<stamp>
"""

from __future__ import annotations

import asyncio
import csv
import json
import re
import time
from pathlib import Path
from typing import Any

import click
import httpx
import yaml
from pydantic import BaseModel, Field

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

    @classmethod
    def load(cls, path: str | Path) -> Config:
        cfg = cls.model_validate(yaml.safe_load(Path(path).read_text()))
        for s in cfg.syntaxes:
            if s not in SYNTAXES:
                raise click.BadParameter(f"unknown syntax {s!r}; choose from {list(SYNTAXES)}")
        for m in cfg.lexicon_modes:
            if m not in LEXICON_MODES:
                raise click.BadParameter(f"unknown lexicon mode {m!r}; choose from {LEXICON_MODES}")
        return cfg

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
# CLI
# --------------------------------------------------------------------------- #


@click.group()
def cli() -> None:
    """E0.0: can off-the-shelf models translate observations into the internal language?"""


@cli.command()
@click.option("-c", "--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--resume", type=click.Path(exists=True, file_okay=False), help="Continue an earlier run dir.")
def run(config_path: str, resume: str | None) -> None:
    """Translate every observation under every condition."""
    cfg = Config.load(config_path)
    observations = load_jsonl(cfg.data)
    out = Path(resume) if resume else Path(cfg.out_dir) / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(Path(config_path).read_text())
    results = out / "results.jsonl"

    # Resume works per whole condition-repeat: partial ones are dropped and rerun,
    # because an evolving lexicon depends on every earlier step.
    existing = load_jsonl(results)
    counts: dict[tuple, int] = {}
    for r in existing:
        k = (r["model"], r["syntax"], r["lexicon"], r["decoding"], r["repeat"])
        counts[k] = counts.get(k, 0) + 1
    complete = {k for k, n in counts.items() if n == len(observations)}
    kept = [r for r in existing if (r["model"], r["syntax"], r["lexicon"], r["decoding"], r["repeat"]) in complete]
    results.write_text("".join(json.dumps(r) + "\n" for r in kept))
    done = complete

    fh = results.open("a")

    def sink(rec: dict[str, Any]) -> None:
        fh.write(json.dumps(rec) + "\n")
        fh.flush()

    names = cfg.models or [ep.name for ep in cfg.endpoints if ep.name != cfg.judge]
    endpoints = [cfg.endpoint(n) for n in names]

    async def main() -> None:
        # Endpoints run in parallel; conditions on one endpoint run in sequence.
        await asyncio.gather(*(run_endpoint(ep, cfg, observations, done, sink) for ep in endpoints))

    try:
        asyncio.run(main())
    finally:
        fh.close()
    click.echo(f"wrote {results}")


def _correct(answer: str, accepted: list[str]) -> bool:
    norm = lambda s: re.sub(r"[^a-z0-9.,()+:]+", " ", s.lower()).strip()  # noqa: E731
    a = norm(answer)
    return any(norm(x) and norm(x) in a for x in accepted)


@cli.command()
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
@click.option("-c", "--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--judge", "judge_name", help="Endpoint to answer questions; default config.judge.")
@click.option("--repeat", default=0, show_default=True, help="Which repeat to judge.")
def judge(run_dir: str, config_path: str, judge_name: str | None, repeat: int) -> None:
    """Fidelity: answer questions from the translations alone, and from the raw input."""
    cfg = Config.load(config_path)
    name = judge_name or cfg.judge
    if not name:
        raise click.UsageError("set judge in the config or pass --judge")
    observations = {o["id"]: o for o in load_jsonl(cfg.data)}
    records = [r for r in load_jsonl(Path(run_dir) / "results.jsonl") if r["repeat"] == repeat]
    out = Path(run_dir) / "judge.jsonl"

    async def main() -> list[dict[str, Any]]:
        client = Client(cfg.endpoint(name))
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
                    base = {k: r[k] for k in metrics.CONDITION_KEYS}
                    await ask(info, q, {"source": "expr", "obs_id": r["obs_id"], **base})
        finally:
            await client.aclose()
        return rows

    rows = asyncio.run(main())
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    click.echo(f"wrote {out}")


@cli.command()
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
def score(run_dir: str) -> None:
    """Summarize a run into summary.csv and summary.md."""
    d = Path(run_dir)
    rows = metrics.summarize(load_jsonl(d / "results.jsonl"), load_jsonl(d / "judge.jsonl"))
    with (d / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=metrics.COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    table = metrics.markdown_table(rows)
    (d / "summary.md").write_text(table + "\n")
    click.echo(table)


@cli.command()
@click.option("-c", "--config", "config_path", required=True, type=click.Path(exists=True))
def check(config_path: str) -> None:
    """Ping every endpoint with one translation."""
    cfg = Config.load(config_path)

    async def main() -> None:
        for ep in cfg.endpoints:
            client = Client(ep)
            try:
                syn = SYNTAXES["sexpr"]
                g = syn.grammar if ep.supports_grammar else None
                c = await client.complete(perceive_system(syn), perceive_user("The cup is on the table.", []), grammar=g, max_tokens=60)
                click.echo(f"[ok]   {ep.name}: {strip_output(c.text)!r} ({c.latency_s:.2f}s)")
            except Exception as exc:  # noqa: BLE001
                click.echo(f"[fail] {ep.name}: {exc}")
            finally:
                await client.aclose()

    asyncio.run(main())
