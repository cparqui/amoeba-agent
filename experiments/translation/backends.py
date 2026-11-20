"""Model endpoints for the experiment.

Everything speaks the OpenAI-compatible `/chat/completions` API, which llama.cpp's
`llama-server`, vLLM, Ollama, LM Studio, MLX servers and hosted APIs all serve.
The only backend-specific part is how a grammar constraint is passed, so that is
configured per endpoint rather than hard-coded:

    llama_cpp  -> {"grammar": "<gbnf>"}                     (default local backend)
    vllm       -> {"structured_outputs": {"grammar": ...}}  (set grammar_field for older versions)
    openai     -> no grammar support; constrained conditions are skipped
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx
from pydantic import BaseModel, Field

GRAMMAR_FIELDS: dict[str, str | None] = {
    "llama_cpp": "grammar",
    "vllm": "structured_outputs.grammar",
    "openai": None,
}


class Endpoint(BaseModel):
    """One model behind one OpenAI-compatible server."""

    name: str
    base_url: str
    model: str = "default"
    backend: str = "llama_cpp"
    api_key_env: str | None = None
    grammar_field: str | None = Field(
        default=None, description="Dotted body path for the grammar; overrides the backend default."
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Merged into every request body, e.g. chat_template_kwargs to disable thinking.",
    )
    timeout: float = 180.0

    @property
    def grammar_path(self) -> str | None:
        return self.grammar_field or GRAMMAR_FIELDS.get(self.backend)

    @property
    def supports_grammar(self) -> bool:
        return self.grammar_path is not None


class Completion(BaseModel):
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0


def _set_path(body: dict[str, Any], dotted: str, value: Any) -> None:
    *parents, leaf = dotted.split(".")
    node = body
    for key in parents:
        node = node.setdefault(key, {})
    node[leaf] = value


def build_body(
    ep: Endpoint,
    system: str,
    user: str,
    *,
    grammar: str | None,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": ep.model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        **ep.extra,
    }
    if grammar is not None:
        if not ep.grammar_path:
            raise ValueError(f"endpoint {ep.name} ({ep.backend}) cannot take a grammar")
        _set_path(body, ep.grammar_path, grammar)
    return body


class Client:
    """Thin async client; one per endpoint."""

    def __init__(self, ep: Endpoint, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.ep = ep
        key = os.environ.get(ep.api_key_env) if ep.api_key_env else None
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self._http = httpx.AsyncClient(
            base_url=ep.base_url.rstrip("/"), headers=headers, timeout=ep.timeout, transport=transport
        )

    async def complete(
        self,
        system: str,
        user: str,
        *,
        grammar: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 400,
    ) -> Completion:
        body = build_body(
            self.ep, system, user, grammar=grammar, temperature=temperature, max_tokens=max_tokens
        )
        start = time.perf_counter()
        resp = await self._http.post("/chat/completions", json=body)
        resp.raise_for_status()
        latency = time.perf_counter() - start
        data = resp.json()
        usage = data.get("usage") or {}
        return Completion(
            text=data["choices"][0]["message"].get("content") or "",
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            latency_s=latency,
        )

    async def aclose(self) -> None:
        await self._http.aclose()
