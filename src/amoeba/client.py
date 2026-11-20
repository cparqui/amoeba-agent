"""Thin HTTP client for a running Amoeba cell."""

from __future__ import annotations

from typing import Any, Self

import httpx


class Client:
    """Talks to one cell's /step, /state and /model endpoints."""

    def __init__(self, base_url: str = "http://127.0.0.1:8700", http: httpx.Client | None = None) -> None:
        self.base_url = base_url
        self._http = http or httpx.Client(base_url=base_url, timeout=300.0)

    def step(self, text: str, affordances: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Send an observation and return the cell's response."""
        return self._json(self._http.post("/step", json={"text": text, "affordances": affordances or []}))

    def state(self) -> dict[str, Any]:
        """Fetch the cell's state."""
        return self._json(self._http.get("/state"))

    def model(self, entity: str | None = None, hops: int = 1, limit: int = 50) -> dict[str, Any]:
        """Fetch part of the cell's world model."""
        params: dict[str, Any] = {"hops": hops, "limit": limit}
        if entity is not None:
            params["entity"] = entity
        return self._json(self._http.get("/model", params=params))

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @staticmethod
    def _json(resp: httpx.Response) -> dict[str, Any]:
        resp.raise_for_status()
        return resp.json()
