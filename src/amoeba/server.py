"""HTTP embodiment: a FastAPI process hosting one Amoeba cell."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from amoeba.agent import Agent, Observation


def create_app(agent: Agent | None = None) -> FastAPI:
    """Build the FastAPI app for a single cell."""
    app = FastAPI(title="amoeba")

    @app.post("/step")
    async def step(observation: Observation) -> None:
        """Push an observation into the cell and get its action."""
        raise HTTPException(status_code=501, detail="not implemented")

    @app.get("/state")
    async def state() -> None:
        """Report the cell's identity and running stats."""
        raise HTTPException(status_code=501, detail="not implemented")

    @app.get("/model")
    async def model(entity: str | None = None, hops: int = 1) -> None:
        """Read part of the cell's world model."""
        raise HTTPException(status_code=501, detail="not implemented")

    return app
