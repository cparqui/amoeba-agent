"""The Amoeba cell: one file holding the PMDA loop, its world model and its genome.

This module has no web dependencies. `server.py` wraps it for HTTP; environments
may also import `Agent` directly and run it in-process.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Interface types (environment <-> cell)
# --------------------------------------------------------------------------- #


class ActionSpec(BaseModel):
    """An action the environment currently affords the cell."""

    name: str
    description: str = ""
    args_schema: dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    """A sensation pushed into the cell by its environment."""

    text: str
    affordances: list[ActionSpec] = Field(default_factory=list)


class Action(BaseModel):
    """The cell's response to an observation."""

    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class SurpriseReport(BaseModel):
    """How well the cell predicted the observation it just received."""

    bits_with_model: float | None = None
    bits_without_model: float | None = None
    model_value: float | None = None
    excess: float | None = None


class StepResult(BaseModel):
    """Everything one PMDA step returns."""

    step: int
    action: Action
    surprise: SurpriseReport


# --------------------------------------------------------------------------- #
# World model (soma)
# --------------------------------------------------------------------------- #


class Entity(BaseModel):
    """A node in the world model: identity only, meaning lives in relations."""

    id: str
    aliases: list[str] = Field(default_factory=list)


class Relation(BaseModel):
    """A subject-predicate-object relation; its id makes it referenceable."""

    id: str
    s: str
    p: str
    o: str
    o_is_literal: bool = False
    weight: float = 1.0
    confidence: float = 0.5


class Store:
    """Per-cell relation store (SQLite)."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def neighborhood(self, entity_ids: list[str], hops: int = 1) -> list[Relation]:
        """Return relations within `hops` of the given entities."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Genome (heritable)
# --------------------------------------------------------------------------- #


class Program(BaseModel):
    """One expression program: prompts and settings a cell can express."""

    prompts: dict[str, str] = Field(default_factory=dict)
    models: dict[str, str] = Field(default_factory=dict)
    params: dict[str, float] = Field(default_factory=dict)


class Genome(BaseModel):
    """Heritable configuration of a cell."""

    programs: dict[str, Program] = Field(default_factory=lambda: {"default": Program()})


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


class Scorer(Protocol):
    """Scores the negative log-likelihood of a target given a context."""

    async def nll(self, context: str, target: str) -> float: ...


# --------------------------------------------------------------------------- #
# The cell
# --------------------------------------------------------------------------- #


class Agent:
    """A single Amoeba cell running the perceive-model-decide-act loop."""

    def __init__(self, genome: Genome, store: Store, scorer: Scorer | None = None) -> None:
        self.genome = genome
        self.store = store
        self.scorer = scorer
        self.program = "default"
        self.steps = 0

    async def step(self, observation: Observation) -> StepResult:
        """Run one full PMDA cycle on an observation."""
        raise NotImplementedError

    async def perceive(self, observation: Observation) -> Any:
        """Turn a raw observation into a percept."""
        raise NotImplementedError

    async def model(self, percept: Any, surprise: SurpriseReport) -> Any:
        """Update the world model from a percept, gated by surprise."""
        raise NotImplementedError

    async def decide(self, percept: Any) -> Any:
        """Choose an intent from the percept and the world model."""
        raise NotImplementedError

    async def act(self, intent: Any, affordances: list[ActionSpec]) -> Action:
        """Turn an intent into a concrete action."""
        raise NotImplementedError
