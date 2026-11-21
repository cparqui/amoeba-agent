"""Tests for experiment E0.0 (no model servers needed)."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from experiments.translation import metrics
from experiments.translation.backends import Client, Endpoint, build_body
from experiments.translation.run import Config, load_jsonl, run_condition, translate_record
from experiments.translation.syntax import (
    SYNTAXES,
    ParseError,
    facts,
    has_variable,
    parse_json,
    parse_prolog,
    parse_sexpr,
    strip_output,
    symbols,
)

SAME = {
    "sexpr": "(on cup table)\n(at_time 7 (touches ?x1 self))\n(not (lit lamp))",
    "prolog": "on(cup, table).\nat_time(7, touches(X1, self)).\nnot(lit(lamp)).",
    "json": '[["on","cup","table"],["at_time",7,["touches","?x1","self"]],["not",["lit","lamp"]]]',
}


def test_syntaxes_agree() -> None:
    results = {name: facts(SYNTAXES[name].parse(text)) for name, text in SAME.items()}
    assert results["sexpr"] == results["prolog"] == results["json"]
    assert len(results["sexpr"]) == 3


def test_examples_parse() -> None:
    for syn in SYNTAXES.values():
        for _, translation in syn.examples:
            assert syn.parse(translation)


def test_nesting_order_is_ignored() -> None:
    a = parse_sexpr("(at_time 1 (conf 0.8 (on cup table)))")
    b = parse_sexpr("(conf 0.8 (at_time 1 (on cup table)))")
    assert facts(a) == facts(b)
    assert facts(a) != facts(parse_sexpr("(on cup table)"))


def test_and_flattens_and_symbols_normalize() -> None:
    a = parse_sexpr("(and (On Cup table) (has-property cup red))")
    b = parse_sexpr("(on cup table)\n(has_property cup red)")
    assert facts(a) == facts(b)
    preds, ents = symbols(a)
    assert preds == {"on", "has_property"} and ents == {"cup", "table", "red"}


def test_variables_anonymized() -> None:
    assert facts(parse_sexpr("(touches ?x1 self)")) == facts(parse_sexpr("(touches ?blob self)"))
    assert has_variable(parse_prolog("touches(X, self)."))
    assert not has_variable(parse_json('[["on","cup","table"]]'))


@pytest.mark.parametrize(
    "parse,text",
    [
        (parse_sexpr, "(on cup table"),
        (parse_sexpr, "(on)"),
        (parse_sexpr, "on cup table"),
        (parse_prolog, "on(cup, table)"),
        (parse_prolog, "cup."),
        (parse_json, '["on","cup"]'),
        (parse_json, "not json"),
    ],
)
def test_bad_input_raises(parse, text) -> None:
    with pytest.raises(ParseError):
        parse(text)


def test_strip_output() -> None:
    assert strip_output("<think>hmm</think>\n```lisp\n(on cup table)\n```") == "(on cup table)"


def test_grammar_placement() -> None:
    kw = dict(grammar="root ::= x", temperature=0.0, max_tokens=10)
    assert build_body(Endpoint(name="a", base_url="x"), "s", "u", **kw)["grammar"] == "root ::= x"
    vb = build_body(Endpoint(name="b", base_url="x", backend="vllm"), "s", "u", **kw)
    assert vb["structured_outputs"]["grammar"] == "root ::= x"
    ob = build_body(Endpoint(name="c", base_url="x", backend="vllm", grammar_field="guided_grammar"), "s", "u", **kw)
    assert ob["guided_grammar"] == "root ::= x"
    with pytest.raises(ValueError):
        build_body(Endpoint(name="d", base_url="x", backend="openai"), "s", "u", **kw)


def test_translate_record() -> None:
    ok = translate_record("sexpr", "(on cup table)")
    assert ok["ok"] and ok["preds"] == ["on"]
    bad = translate_record("sexpr", "the cup is on the table")
    assert not bad["ok"] and bad["facts"] == []


def _fake_server(replies: list[str]) -> httpx.MockTransport:
    it = iter(replies)
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": next(it)}}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 8},
            },
        )

    t = httpx.MockTransport(handler)
    t.seen = seen  # type: ignore[attr-defined]
    return t


def test_run_condition_and_score() -> None:
    observations = [
        {"id": "a", "content": "The cup is on the table.", "group": "cup", "novel": False},
        {"id": "b", "content": "On the table sits a cup.", "group": "cup", "novel": False},
        {"id": "c", "content": "Something unknown touched you.", "group": None, "novel": True},
    ]
    transport = _fake_server(["(on cup table)", "(on cup table)", "(touches ?x self)"])
    ep = Endpoint(name="fake", base_url="http://fake/v1")
    cfg = Config(endpoints=[ep], temperature=0.0)
    records: list[dict] = []

    async def go() -> None:
        client = Client(ep, transport=transport)
        await run_condition(client, cfg, observations, "sexpr", "evolving", "constrained", 0, records.append)
        await client.aclose()

    asyncio.run(go())

    # The evolving lexicon reaches the second prompt, and the grammar is sent.
    second_prompt = transport.seen[1]["messages"][1]["content"]  # type: ignore[attr-defined]
    assert "on, cup, table" in second_prompt
    assert "grammar" in transport.seen[0]  # type: ignore[attr-defined]
    assert records[1]["reused"] == ["cup", "on", "table"] and records[1]["new"] == []

    row = metrics.summarize(records)[0]
    assert row["parse_rate"] == 1.0
    assert row["consistency"] == 1.0
    assert row["novel_var_rate"] == 1.0
    assert row["lexicon_final"] == 5  # on, cup, table, touches, self
    assert "| fake |" in metrics.markdown_table([row])


def test_shipped_data_is_valid() -> None:
    from experiments.translation.run import HERE

    obs = load_jsonl(HERE / "data" / "observations.jsonl")
    assert len({o["id"] for o in obs}) == len(obs) >= 30
    assert all(o["qa"] and all(q["a"] for q in o["qa"]) for o in obs)
    groups = [o["group"] for o in obs if o["group"]]
    assert all(groups.count(g) >= 3 for g in set(groups))
    Config.load(HERE / "config.example.yaml")
