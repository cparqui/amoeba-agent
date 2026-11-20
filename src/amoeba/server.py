"""HTTP embodiment: a FastAPI process hosting one Amoeba cell."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

from amoeba.agent import Agent, LLMError, Observation, StepResult


def create_app(
    agent: Agent | None = None,
    genome_path: str | Path | None = None,
    db_path: str | Path | None = None,
    cell_id: str | None = None,
) -> FastAPI:
    """Build the FastAPI app for a single cell.

    Without an `agent`, one is built from the given paths, falling back to the
    AMOEBA_GENOME, AMOEBA_DB and AMOEBA_CELL_ID environment variables.
    """
    if agent is None:
        agent = Agent.create(
            genome_path=genome_path or os.environ.get("AMOEBA_GENOME") or None,
            db_path=db_path or os.environ.get("AMOEBA_DB") or ":memory:",
            cell_id=cell_id or os.environ.get("AMOEBA_CELL_ID") or "cell",
        )
    app = FastAPI(title=f"amoeba:{agent.cell_id}")
    app.state.agent = agent

    @app.post("/step")
    async def step(observation: Observation) -> StepResult:
        """Push an observation into the cell and get its action."""
        try:
            return await agent.step(observation)
        except LLMError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.get("/state")
    async def state() -> dict[str, Any]:
        """Report the cell's identity and running stats."""
        return agent.state()

    @app.get("/model")
    async def model(entity: str | None = None, hops: int = 1, limit: int = 50) -> dict[str, Any]:
        """Read part of the cell's world model: one entity's neighborhood, or the strongest relations."""
        if entity is None:
            relations = agent.store.strongest(limit)
        else:
            entity_id = agent.store.resolve(entity)
            if entity_id is None:
                raise HTTPException(status_code=404, detail=f"unknown entity: {entity}")
            relations = agent.store.neighborhood([entity_id], hops, limit)
            entity = entity_id
        return {"entity": entity, "hops": hops, "relations": [r.model_dump() for r in relations]}

    return app
