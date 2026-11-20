"""Scoring for E0.0. Pure functions over the JSONL records written by `run`.

All metrics are per condition (model x syntax x lexicon mode x decoding), pooled
over repeats. A failed parse counts as an empty translation.
"""

from __future__ import annotations

import itertools
import statistics
from collections import defaultdict
from typing import Any, Iterable

CONDITION_KEYS = ("model", "syntax", "lexicon", "decoding")


def jaccard(a: set[str] | frozenset[str], b: set[str] | frozenset[str]) -> float:
    """Set overlap; two empty sets score 0 (no evidence of agreement)."""
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def mean_pairwise(sets: list[set[str]]) -> float | None:
    pairs = list(itertools.combinations(sets, 2))
    return statistics.fmean(jaccard(a, b) for a, b in pairs) if pairs else None


def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def _mean(xs: Iterable[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return statistics.fmean(vals) if vals else None


def cond_of(rec: dict[str, Any]) -> tuple[str, ...]:
    return tuple(rec[k] for k in CONDITION_KEYS)


def summarize(
    records: list[dict[str, Any]], judged: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    by_cond: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_cond[cond_of(r)].append(r)

    fidelity: dict[tuple[str, ...], list[bool]] = defaultdict(list)
    for j in judged or []:
        if j.get("source") == "expr":
            fidelity[cond_of(j)].append(bool(j["correct"]))

    rows = []
    for cond, recs in sorted(by_cond.items()):
        repeats: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for r in recs:
            repeats[r["repeat"]].append(r)
        for rs in repeats.values():
            rs.sort(key=lambda r: r["idx"])

        # Paraphrase consistency, within each repeat
        fact_cons, pred_cons = [], []
        for rs in repeats.values():
            groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for r in rs:
                if r.get("group"):
                    groups[r["group"]].append(r)
            for members in groups.values():
                fact_cons.append(mean_pairwise([set(m["facts"]) for m in members]))
                pred_cons.append(mean_pairwise([set(m["preds"]) for m in members]))

        # Stability: the same observation across repeats
        by_obs: dict[str, list[set[str]]] = defaultdict(list)
        for r in recs:
            by_obs[r["obs_id"]].append(set(r["facts"]))
        stability = _mean(mean_pairwise(v) for v in by_obs.values())

        # Lexicon behaviour
        reused = sum(len(r["reused"]) for r in recs if r["ok"])
        used = sum(len(r["preds"]) + len(r["ents"]) for r in recs if r["ok"])
        final_sizes, convergence = [], []
        for rs in repeats.values():
            final_sizes.append(rs[-1]["lexicon_after"])
            third = max(1, len(rs) // 3)
            first = sum(len(r["new"]) for r in rs[:third])
            last = sum(len(r["new"]) for r in rs[-third:])
            convergence.append(last / first if first else None)

        novel = [r["has_var"] for r in recs if r["novel"]]
        lat = [r["latency_s"] for r in recs if r.get("latency_s") is not None]
        fid = fidelity.get(cond, [])

        rows.append(
            {
                **dict(zip(CONDITION_KEYS, cond)),
                "n": len(recs),
                "parse_rate": sum(r["ok"] for r in recs) / len(recs),
                "consistency": _mean(fact_cons),
                "pred_consistency": _mean(pred_cons),
                "stability": stability,
                "reuse_rate": reused / used if used else None,
                "lexicon_final": _mean(final_sizes),
                "convergence": _mean(convergence),
                "novel_var_rate": sum(novel) / len(novel) if novel else None,
                "fidelity": sum(fid) / len(fid) if fid else None,
                "latency_p50_s": _pct(lat, 0.5),
                "latency_p95_s": _pct(lat, 0.95),
                "completion_tokens": _mean(r["completion_tokens"] for r in recs),
            }
        )

    raw = [bool(j["correct"]) for j in judged or [] if j.get("source") == "raw"]
    if raw:
        rows.append({"model": "(raw input)", "fidelity": sum(raw) / len(raw)})
    return rows


COLUMNS = (
    *CONDITION_KEYS,
    "n",
    "parse_rate",
    "consistency",
    "pred_consistency",
    "stability",
    "reuse_rate",
    "lexicon_final",
    "convergence",
    "novel_var_rate",
    "fidelity",
    "latency_p50_s",
    "latency_p95_s",
    "completion_tokens",
)


def fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(c)) for c in COLUMNS) + " |")
    return "\n".join(lines)
