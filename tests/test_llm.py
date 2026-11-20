import asyncio
import json

import httpx
import pytest

from amoeba.agent import Intent, LLMError, OpenAICompatLLM, Params, Route


def reply(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}], "usage": {"prompt_tokens": 7, "completion_tokens": 3}}


def make(replies: list[str], seen: list[dict]) -> OpenAICompatLLM:
    queue = list(replies)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=reply(queue.pop(0)))

    return OpenAICompatLLM("http://llm/v1", transport=httpx.MockTransport(handler))


def call(llm: OpenAICompatLLM):
    return asyncio.run(llm.complete(route=Route(model="m"), system="sys", user="u", schema=Intent, params=Params()))


def test_constrained_request_and_retry_on_invalid_json() -> None:
    seen: list[dict] = []
    intent, usage = call(make(["not json", '{"goal": "eat", "rationale": "hungry"}'], seen))
    assert intent == Intent(goal="eat", rationale="hungry")
    assert usage.prompt_tokens == 14 and usage.completion_tokens == 6
    assert seen[0]["response_format"]["type"] == "json_schema"
    assert seen[0]["response_format"]["json_schema"]["schema"]["title"] == "Intent"
    assert len(seen[1]["messages"]) == 4  # retry carries the error back


def test_gives_up_after_one_retry() -> None:
    with pytest.raises(LLMError):
        call(make(["nope", "still nope"], []))
