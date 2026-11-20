"""Thin HTTP client for a running Amoeba cell."""

from __future__ import annotations

from typing import Any

import httpx


class Client:
    """Talks to one cell's /step, /state and /model endpoints."""

    def __init__(self, base_url: str = "http://127.0.0.1:8700") -> None:
        self.base_url = base_url
        self._http = httpx.Client(base_url=base_url)

    def step(self, text: str) -> dict[str, Any]:
        """Send an observation and return the cell's response."""
        raise NotImplementedError

    def state(self) -> dict[str, Any]:
        """Fetch the cell's state."""
        raise NotImplementedError

    def model(self, entity: str | None = None, hops: int = 1) -> dict[str, Any]:
        """Fetch part of the cell's world model."""
        raise NotImplementedError
