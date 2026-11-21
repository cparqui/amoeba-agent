import asyncio

from fastapi.testclient import TestClient

from amoeba.agent import STAGES, Action, ActionSpec, Agent, Observation
from amoeba.client import Client
from amoeba.server import create_app

AFFORDANCES = [ActionSpec(name="move_left"), ActionSpec(name="move_right"), ActionSpec(name="eat")]


def observe() -> Observation:
    return Observation(text="You smell food to your right. Energy: 10.", affordances=AFFORDANCES)


def test_full_step(agent: Agent, fake_llm) -> None:
    result = asyncio.run(agent.step(observe()))
    assert result.step == 1
    assert result.action == Action(name="move_right") and not result.action_fallback
    assert result.surprise is None
    assert result.ops_applied == 2
    assert [c["schema"] for c in fake_llm.calls] == ["Percept", "ModelUpdate", "Intent", "Action"]
    assert set(STAGES) | {"retrieve", "update", "total"} <= set(result.timings_ms)
    assert set(result.usage) == set(STAGES)
    assert result.context.relations_total == 2 and result.context.relations_in_context == 2
    assert agent.store.resolve("food") == "food"
    assert "move_right" in agent._last_context and "food" in agent._last_context
    # the decide prompt saw the updated model
    decide_prompt = fake_llm.calls[2]["user"]
    assert "food is_to_the_right_of self" in decide_prompt


def test_known_entities_reach_perceive(agent: Agent, fake_llm) -> None:
    asyncio.run(agent.step(observe()))
    asyncio.run(agent.step(observe()))
    assert "food" in fake_llm.calls[4]["user"]  # second perceive call lists known entities
    assert agent.state()["step"] == 2


def test_invalid_action_falls_back(agent: Agent, fake_llm) -> None:
    fake_llm.outputs[Action] = Action(name="fly")
    result = asyncio.run(agent.step(observe()))
    assert result.action.name == "move_left" and result.action_fallback


def test_no_affordances_is_noop(agent: Agent) -> None:
    result = asyncio.run(agent.step(Observation(text="Darkness.")))
    assert result.action.name == "noop" and result.action_fallback


def test_server_and_client(agent: Agent) -> None:
    with TestClient(create_app(agent)) as http:
        client = Client(http=http)
        out = client.step("You smell food to your right.", [a.model_dump() for a in AFFORDANCES])
        assert out["action"]["name"] == "move_right" and out["surprise"] is None
        assert client.state()["step"] == 1
        model = client.model("food")
        assert model["entity"] == "food" and model["relations"]
        assert client.model()["relations"]
        assert http.get("/model", params={"entity": "nobody"}).status_code == 404
