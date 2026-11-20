from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from amoeba.agent import Action, Agent, Genome, Intent, ModelUpdate, Params, Percept, RelationOp, Route, Store, Usage


class FakeLLM:
    """Returns canned outputs per schema; records every call."""

    def __init__(self, outputs: dict[type, Callable[[str], Any] | Any] | None = None) -> None:
        self.outputs = outputs or {}
        self.calls: list[dict[str, Any]] = []

    async def complete(self, *, route: Route, system: str, user: str, schema: type, params: Params):
        self.calls.append({"schema": schema.__name__, "user": user, "route": route})
        out = self.outputs.get(schema)
        if callable(out) and not isinstance(out, type):
            out = out(user)
        if out is None:
            out = schema()
        return out, Usage(prompt_tokens=len(user) // 4, completion_tokens=10)


def default_outputs() -> dict[type, Any]:
    return {
        Percept: Percept(
            entities=[{"mention": "food", "entity_id": None}, {"mention": "self", "entity_id": "self"}],
            facts=["food is to the right"],
            salient="food to the right",
        ),
        ModelUpdate: ModelUpdate(
            ops=[
                RelationOp(op="ASSERT", s="food", p="is to the right of", o="self"),
                RelationOp(op="ASSERT", s="self", p="has energy", o="10", o_is_literal=True),
            ]
        ),
        Intent: Intent(goal="reach the food", rationale="energy is finite"),
        Action: Action(name="move_right"),
    }


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM(default_outputs())


@pytest.fixture
def agent(fake_llm: FakeLLM) -> Agent:
    return Agent(Genome.load(), Store(":memory:"), llm=fake_llm, cell_id="test")
