"""The Amoeba cell: one file holding the PMDA loop, its world model and its genome.

This module has no web-framework dependencies. `server.py` wraps it for HTTP;
environments may also import `Agent` directly and run it in-process.

v0 scope: perceive -> model -> decide -> act with a relational world model.
Surprise scoring is deferred: `StepResult.surprise` is always None for now, and
the cell keeps the rendered decision context of its last step as the hook the
scorer will use later.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sqlite3
import time
from importlib import resources
from pathlib import Path
from string import Template
from typing import Any, Literal, Protocol, TypeVar

import httpx
import yaml
from pydantic import BaseModel, Field, ValidationError

STAGES = ("perceive", "model", "decide", "act")

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
    """How well the cell predicted the observation it just received (deferred)."""

    bits_with_model: float | None = None
    bits_without_model: float | None = None
    model_value: float | None = None
    excess: float | None = None


class Usage(BaseModel):
    """Token usage of one model call, when the server reports it."""

    prompt_tokens: int = 0
    completion_tokens: int = 0


class ContextStats(BaseModel):
    """Size of what the cell put in front of the model, and of its soma."""

    relations_in_context: int = 0
    context_chars: int = 0
    entities_total: int = 0
    relations_total: int = 0
    db_bytes: int = 0


class StepResult(BaseModel):
    """Everything one PMDA step returns."""

    step: int
    action: Action
    action_fallback: bool = False
    surprise: SurpriseReport | None = None
    ops_applied: int = 0
    timings_ms: dict[str, float] = Field(default_factory=dict)
    usage: dict[str, Usage] = Field(default_factory=dict)
    context: ContextStats = Field(default_factory=ContextStats)


# --------------------------------------------------------------------------- #
# Stage outputs
# --------------------------------------------------------------------------- #


class Mention(BaseModel):
    """Something the observation refers to, resolved to an entity id if known."""

    mention: str
    entity_id: str | None = None


class Percept(BaseModel):
    """Perceive's structured reading of an observation."""

    entities: list[Mention] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)
    salient: str = ""


class RelationOp(BaseModel):
    """One edit to the world model."""

    op: Literal["ASSERT", "REINFORCE", "WEAKEN", "RETRACT"]
    s: str
    p: str
    o: str
    o_is_literal: bool = False


class ModelUpdate(BaseModel):
    """Model's output: the edits to apply this step."""

    ops: list[RelationOp] = Field(default_factory=list)


class Intent(BaseModel):
    """Decide's output: what the cell is trying to do and why."""

    goal: str
    rationale: str = ""


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
    evidence: int = 1
    credit: float = 0.0
    first_seen: int = 0
    last_seen: int = 0
    provenance: int = 0

    def render(self) -> str:
        """Compact one-line form used in prompts."""
        o = json.dumps(self.o) if self.o_is_literal else self.o
        return f"[{self.id}] {self.s} {self.p} {o} (w={self.weight:.2f}, c={self.confidence:.2f})"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS entity (
  id TEXT PRIMARY KEY,
  aliases TEXT NOT NULL DEFAULT '[]',
  first_seen INTEGER NOT NULL DEFAULT 0,
  last_seen INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS relation (
  id TEXT PRIMARY KEY,
  s TEXT NOT NULL REFERENCES entity(id),
  p TEXT NOT NULL,
  o TEXT NOT NULL,
  o_is_literal INTEGER NOT NULL DEFAULT 0,
  weight REAL NOT NULL DEFAULT 1.0,
  confidence REAL NOT NULL DEFAULT 0.5,
  evidence INTEGER NOT NULL DEFAULT 1,
  credit REAL NOT NULL DEFAULT 0.0,
  first_seen INTEGER NOT NULL DEFAULT 0,
  last_seen INTEGER NOT NULL DEFAULT 0,
  provenance INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS rel_s ON relation(s);
CREATE INDEX IF NOT EXISTS rel_o ON relation(o);
CREATE UNIQUE INDEX IF NOT EXISTS rel_spo ON relation(s, p, o, o_is_literal);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

# Update rule constants (v0; candidates for genome parameters later).
REINFORCE_WEIGHT = 0.25
REINFORCE_CONFIDENCE = 0.2  # fraction of the gap to 1.0
WEAKEN_WEIGHT = 0.25
WEAKEN_CONFIDENCE = 0.8  # multiplier


def _norm(text: str) -> str:
    """Normalize a mention for alias matching: case- and punctuation-insensitive."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _slug(text: str) -> str:
    """Turn a mention or predicate into an id-safe token."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "entity"


class Store:
    """Per-cell relation store (SQLite). One file per cell; ':memory:' allowed."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.upsert_entity("self", ["self", "me", "I"], step=0)
        self.conn.commit()

    # ----- entities -------------------------------------------------------- #

    def upsert_entity(self, entity_id: str, aliases: list[str] | tuple[str, ...] = (), step: int = 0) -> str:
        """Create an entity or merge new aliases into it; returns its id."""
        row = self.conn.execute("SELECT aliases FROM entity WHERE id = ?", (entity_id,)).fetchone()
        if row is None:
            names = list(dict.fromkeys([entity_id, *aliases]))
            self.conn.execute(
                "INSERT INTO entity (id, aliases, first_seen, last_seen) VALUES (?, ?, ?, ?)",
                (entity_id, json.dumps(names), step, step),
            )
        else:
            names = list(dict.fromkeys([*json.loads(row["aliases"]), *aliases]))
            self.conn.execute(
                "UPDATE entity SET aliases = ?, last_seen = MAX(last_seen, ?) WHERE id = ?",
                (json.dumps(names), step, entity_id),
            )
        return entity_id

    def resolve(self, mention: str) -> str | None:
        """Map a mention to an existing entity id via its aliases, or None."""
        key = _norm(mention)
        if not key:
            return None
        for row in self.conn.execute("SELECT id, aliases FROM entity"):
            if _norm(row["id"]) == key or any(_norm(a) == key for a in json.loads(row["aliases"])):
                return row["id"]
        return None

    def ensure(self, mention: str, step: int) -> str:
        """Resolve a mention, creating a new entity if it is unknown."""
        entity_id = self.resolve(mention) or _slug(mention)
        return self.upsert_entity(entity_id, [mention], step)

    def entity_names(self, limit: int = 50) -> list[str]:
        """Most recently seen entities (excluding relation ids), as 'id (aliases)'."""
        rows = self.conn.execute(
            "SELECT id, aliases FROM entity WHERE id NOT IN (SELECT id FROM relation) "
            "ORDER BY last_seen DESC, id LIMIT ?",
            (limit,),
        ).fetchall()
        out = []
        for row in rows:
            others = [a for a in json.loads(row["aliases"]) if a != row["id"]]
            out.append(f"{row['id']} ({', '.join(others)})" if others else row["id"])
        return out

    # ----- relations ------------------------------------------------------- #

    def _next_relation_id(self) -> str:
        row = self.conn.execute("SELECT value FROM meta WHERE key = 'next_relation'").fetchone()
        n = int(row["value"]) if row else 1
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES ('next_relation', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(n + 1),),
        )
        return f"r{n}"

    def _find(self, s: str, p: str, o: str, literal: bool) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM relation WHERE s = ? AND p = ? AND o = ? AND o_is_literal = ?",
            (s, p, o, int(literal)),
        ).fetchone()

    def _delete(self, relation_id: str) -> None:
        self.conn.execute("DELETE FROM relation WHERE id = ?", (relation_id,))
        self.conn.execute("DELETE FROM entity WHERE id = ?", (relation_id,))

    def apply(self, ops: list[RelationOp], step: int) -> int:
        """Apply relation ops; returns how many changed the store."""
        changed = 0
        for op in ops:
            s = self.ensure(op.s, step)
            p = _slug(op.p)
            o = op.o if op.o_is_literal else self.ensure(op.o, step)
            row = self._find(s, p, o, op.o_is_literal)
            if op.op in ("ASSERT", "REINFORCE"):
                if row is None:
                    rid = self._next_relation_id()
                    self.conn.execute(
                        "INSERT INTO relation (id, s, p, o, o_is_literal, first_seen, last_seen, provenance) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (rid, s, p, o, int(op.o_is_literal), step, step, step),
                    )
                    self.upsert_entity(rid, step=step)
                else:
                    self.conn.execute(
                        "UPDATE relation SET weight = weight + ?, evidence = evidence + 1, "
                        "confidence = confidence + (1.0 - confidence) * ?, last_seen = ? WHERE id = ?",
                        (REINFORCE_WEIGHT, REINFORCE_CONFIDENCE, step, row["id"]),
                    )
                changed += 1
            elif row is not None and op.op == "WEAKEN":
                weight = row["weight"] - WEAKEN_WEIGHT
                if weight <= 0:
                    self._delete(row["id"])
                else:
                    self.conn.execute(
                        "UPDATE relation SET weight = ?, confidence = confidence * ?, last_seen = ? WHERE id = ?",
                        (weight, WEAKEN_CONFIDENCE, step, row["id"]),
                    )
                changed += 1
            elif row is not None and op.op == "RETRACT":
                self._delete(row["id"])
                changed += 1
        self.conn.commit()
        return changed

    def neighborhood(self, entity_ids: list[str], hops: int = 1, limit: int | None = None) -> list[Relation]:
        """Relations within `hops` of the given entities, strongest first.

        hops=1 returns relations touching the seed entities; hops=2 also those
        touching their neighbors. Relations about included relations (reified,
        e.g. `r3 holds_when dark`) are always added.
        """
        if not entity_ids:
            return []
        hops = max(1, hops)
        rows = self.conn.execute(
            """
            WITH RECURSIVE frontier(id, depth) AS (
              SELECT value, 0 FROM json_each(?)
              UNION
              SELECT CASE WHEN r.s = f.id THEN r.o ELSE r.s END, f.depth + 1
              FROM relation r JOIN frontier f ON r.s = f.id OR (r.o = f.id AND r.o_is_literal = 0)
              WHERE f.depth < ? AND NOT (r.s = f.id AND r.o_is_literal = 1)
            ),
            nodes AS (SELECT DISTINCT id FROM frontier)
            SELECT DISTINCT r.* FROM relation r
            WHERE r.s IN nodes OR (r.o_is_literal = 0 AND r.o IN nodes)
            ORDER BY r.weight * r.confidence DESC, r.last_seen DESC, r.id
            """,
            (json.dumps(entity_ids), hops - 1),
        ).fetchall()
        rels = [self._row(r) for r in rows]
        seen = {r.id for r in rels}
        if seen:
            marks = ",".join("?" * len(seen))
            meta = self.conn.execute(f"SELECT * FROM relation WHERE s IN ({marks})", tuple(seen)).fetchall()
            rels += [self._row(r) for r in meta if r["id"] not in seen]
        return rels[:limit] if limit else rels

    def about(self, entity: str, hops: int = 1) -> list[Relation]:
        """Everything the cell knows about one entity (by id or alias)."""
        entity_id = self.resolve(entity)
        return self.neighborhood([entity_id], hops) if entity_id else []

    def strongest(self, limit: int = 50) -> list[Relation]:
        """The strongest relations overall."""
        rows = self.conn.execute(
            "SELECT * FROM relation ORDER BY weight * confidence DESC, last_seen DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row(r) for r in rows]

    def stats(self) -> dict[str, int]:
        """Size of the soma: counts and bytes on disk."""
        relations = self.conn.execute("SELECT COUNT(*) FROM relation").fetchone()[0]
        entities = self.conn.execute("SELECT COUNT(*) FROM entity").fetchone()[0] - relations
        pages = self.conn.execute("PRAGMA page_count").fetchone()[0]
        page_size = self.conn.execute("PRAGMA page_size").fetchone()[0]
        return {"entities": entities, "relations": relations, "db_bytes": pages * page_size}

    @staticmethod
    def _row(row: sqlite3.Row) -> Relation:
        data = dict(row)
        data["o_is_literal"] = bool(data["o_is_literal"])
        return Relation(**data)


# --------------------------------------------------------------------------- #
# Genome (heritable)
# --------------------------------------------------------------------------- #


class StagePrompt(BaseModel):
    """System and user templates for one stage (`$name` placeholders)."""

    system: str
    user: str


class Route(BaseModel):
    """Which model serves a stage, and where."""

    model: str = "default"
    base_url: str | None = None  # None: use Genome.base_url


class Params(BaseModel):
    """Numeric settings of a program."""

    hops: int = 1
    max_relations_in_context: int = 40
    max_ops_per_step: int = 8
    known_entities_in_prompt: int = 50
    temperature: float = 0.2
    max_tokens: int = 512


class Program(BaseModel):
    """One expression program: prompts, model routing and settings a cell can express."""

    prompts: dict[str, StagePrompt] = Field(default_factory=dict)
    models: dict[str, Route] = Field(default_factory=dict)
    params: Params = Field(default_factory=Params)

    def route(self, stage: str) -> Route:
        return self.models.get(stage, Route())


class Genome(BaseModel):
    """Heritable configuration of a cell."""

    name: str = "default"
    base_url: str = "http://127.0.0.1:8080/v1"
    programs: dict[str, Program] = Field(default_factory=lambda: {"default": Program()})

    @classmethod
    def load(cls, path: str | Path | None = None) -> Genome:
        """Load a genome from YAML; with no path, the packaged default."""
        if path is None:
            text = resources.files("amoeba").joinpath("genome.default.yaml").read_text()
        else:
            text = Path(path).read_text()
        return cls.model_validate(yaml.safe_load(text))

    def digest(self) -> str:
        """Short content hash, for lineage tracking."""
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Model calls
# --------------------------------------------------------------------------- #

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """A model call failed or never produced valid output."""


class LLM(Protocol):
    """Anything that can answer a stage prompt with a validated object."""

    async def complete(
        self, *, route: Route, system: str, user: str, schema: type[T], params: Params
    ) -> tuple[T, Usage]: ...


class OpenAICompatLLM:
    """Client for OpenAI-compatible chat servers (llama.cpp, vLLM, Ollama).

    Uses JSON-schema constrained decoding via `response_format` and validates
    the reply with pydantic, retrying once with the validation error.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout, transport=transport
        )

    async def complete(
        self, *, route: Route, system: str, user: str, schema: type[T], params: Params
    ) -> tuple[T, Usage]:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        usage = Usage()
        error: Exception | None = None
        for _ in range(2):
            payload = {
                "model": route.model,
                "messages": messages,
                "temperature": params.temperature,
                "max_tokens": params.max_tokens,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
                },
            }
            try:
                resp = await self._http.post("/chat/completions", json=payload)
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                raise LLMError(f"model call failed: {exc}") from exc
            data = resp.json()
            u = data.get("usage") or {}
            usage.prompt_tokens += int(u.get("prompt_tokens") or 0)
            usage.completion_tokens += int(u.get("completion_tokens") or 0)
            content = data["choices"][0]["message"].get("content") or ""
            try:
                return schema.model_validate_json(content), usage
            except ValidationError as exc:
                error = exc
                messages += [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": f"That was not valid: {exc}. Reply with JSON matching the schema only."},
                ]
        raise LLMError(f"no valid {schema.__name__} after retry: {error}")

    async def aclose(self) -> None:
        await self._http.aclose()


class Router:
    """An LLM that dispatches each stage to the server named by its route."""

    def __init__(self, default_base_url: str) -> None:
        self.default_base_url = default_base_url
        self._clients: dict[str, OpenAICompatLLM] = {}

    async def complete(
        self, *, route: Route, system: str, user: str, schema: type[T], params: Params
    ) -> tuple[T, Usage]:
        url = route.base_url or self.default_base_url
        if url not in self._clients:
            self._clients[url] = OpenAICompatLLM(url)
        return await self._clients[url].complete(route=route, system=system, user=user, schema=schema, params=params)


# --------------------------------------------------------------------------- #
# Scoring (deferred: surprise is not computed in v0)
# --------------------------------------------------------------------------- #


class Scorer(Protocol):
    """Scores the negative log-likelihood of a target given a context."""

    async def nll(self, context: str, target: str) -> float: ...


# --------------------------------------------------------------------------- #
# The cell
# --------------------------------------------------------------------------- #


def _render_relations(relations: list[Relation]) -> str:
    return "\n".join(r.render() for r in relations) or "(none yet)"


def _render_affordances(affordances: list[ActionSpec]) -> str:
    lines = []
    for a in affordances:
        args = f" args={json.dumps(a.args_schema)}" if a.args_schema else ""
        lines.append(f"- {a.name}: {a.description}{args}".rstrip(": "))
    return "\n".join(lines) or "(no actions available)"


def _render_percept(percept: Percept, entity_ids: list[str]) -> str:
    lines = [f"salient: {percept.salient}"] if percept.salient else []
    lines += [f"entities: {', '.join(entity_ids)}"] if entity_ids else []
    lines += [f"- {f}" for f in percept.facts]
    return "\n".join(lines) or "(empty)"


class Agent:
    """A single Amoeba cell running the perceive-model-decide-act loop."""

    def __init__(
        self,
        genome: Genome,
        store: Store,
        llm: LLM | None = None,
        scorer: Scorer | None = None,
        cell_id: str = "cell",
    ) -> None:
        self.genome = genome
        self.store = store
        self.llm: LLM = llm or Router(genome.base_url)
        self.scorer = scorer
        self.cell_id = cell_id
        self.program = "default"
        self.steps = 0
        self.last_result: StepResult | None = None
        self._last_context: str | None = None  # hook for surprise scoring
        self._lock = asyncio.Lock()

    @classmethod
    def create(
        cls,
        genome_path: str | Path | None = None,
        db_path: str | Path = ":memory:",
        cell_id: str = "cell",
        llm: LLM | None = None,
    ) -> Agent:
        """Build a cell from a genome file and a database path."""
        return cls(Genome.load(genome_path), Store(db_path), llm=llm, cell_id=cell_id)

    @property
    def expressed(self) -> Program:
        return self.genome.programs[self.program]

    # ----- the loop -------------------------------------------------------- #

    async def step(self, observation: Observation) -> StepResult:
        """Run one full PMDA cycle on an observation."""
        async with self._lock:
            self.steps += 1
            n = self.steps
            params = self.expressed.params
            timings: dict[str, float] = {}
            usage: dict[str, Usage] = {}
            t_step = time.perf_counter()

            t = time.perf_counter()
            percept, usage["perceive"] = await self.perceive(observation)
            focus = self._resolve(percept, n)
            timings["perceive"] = _ms(t)

            t = time.perf_counter()
            before = self.store.neighborhood(focus, params.hops, params.max_relations_in_context)
            timings["retrieve"] = _ms(t)

            t = time.perf_counter()
            ops, usage["model"] = await self.model(percept, focus, before)
            timings["model"] = _ms(t)

            t = time.perf_counter()
            applied = self.store.apply(ops, n)
            after = self.store.neighborhood(focus, params.hops, params.max_relations_in_context)
            timings["update"] = _ms(t)

            t = time.perf_counter()
            (intent, context), usage["decide"] = await self.decide(percept, focus, after, observation.affordances)
            timings["decide"] = _ms(t)

            t = time.perf_counter()
            (action, fallback), usage["act"] = await self.act(intent, observation.affordances)
            timings["act"] = _ms(t)
            timings["total"] = _ms(t_step)

            self._last_context = f"{context}\n\nACTION: {action.model_dump_json()}"
            stats = self.store.stats()
            result = StepResult(
                step=n,
                action=action,
                action_fallback=fallback,
                surprise=None,
                ops_applied=applied,
                timings_ms=timings,
                usage=usage,
                context=ContextStats(
                    relations_in_context=len(after),
                    context_chars=len(context),
                    entities_total=stats["entities"],
                    relations_total=stats["relations"],
                    db_bytes=stats["db_bytes"],
                ),
            )
            self.last_result = result
            return result

    async def _call(self, stage: str, schema: type[T], **values: Any) -> tuple[T, Usage, str]:
        program = self.expressed
        prompt = program.prompts[stage]
        user = Template(prompt.user).safe_substitute(values)
        system = Template(prompt.system).safe_substitute(values)
        out, usage = await self.llm.complete(
            route=program.route(stage), system=system, user=user, schema=schema, params=program.params
        )
        return out, usage, user

    def _resolve(self, percept: Percept, step: int) -> list[str]:
        """Map percept mentions to entity ids (creating new ones), plus `self`."""
        ids: list[str] = []
        for m in percept.entities:
            known = self.store.resolve(m.entity_id) if m.entity_id else None
            entity_id = self.store.upsert_entity(known, [m.mention], step) if known else self.store.ensure(m.mention, step)
            m.entity_id = entity_id
            ids.append(entity_id)
        ids.append(self.store.upsert_entity("self", step=step))
        self.store.conn.commit()
        return list(dict.fromkeys(ids))

    # ----- stages ---------------------------------------------------------- #

    async def perceive(self, observation: Observation) -> tuple[Percept, Usage]:
        """Turn a raw observation into a percept."""
        known = self.store.entity_names(self.expressed.params.known_entities_in_prompt)
        percept, usage, _ = await self._call(
            "perceive", Percept, observation=observation.text, known_entities=", ".join(known) or "(none yet)"
        )
        return percept, usage

    async def model(self, percept: Percept, focus: list[str], relations: list[Relation]) -> tuple[list[RelationOp], Usage]:
        """Propose world-model edits from a percept and the relevant relations."""
        params = self.expressed.params
        update, usage, _ = await self._call(
            "model",
            ModelUpdate,
            percept=_render_percept(percept, focus),
            relations=_render_relations(relations),
            max_ops=params.max_ops_per_step,
        )
        return update.ops[: params.max_ops_per_step], usage

    async def decide(
        self, percept: Percept, focus: list[str], relations: list[Relation], affordances: list[ActionSpec]
    ) -> tuple[tuple[Intent, str], Usage]:
        """Choose an intent from the percept and the updated world model."""
        intent, usage, context = await self._call(
            "decide",
            Intent,
            percept=_render_percept(percept, focus),
            relations=_render_relations(relations),
            affordances=_render_affordances(affordances),
        )
        return (intent, context), usage

    async def act(self, intent: Intent, affordances: list[ActionSpec]) -> tuple[tuple[Action, bool], Usage]:
        """Turn an intent into one afforded action (falls back if invalid)."""
        if not affordances:
            return (Action(name="noop"), True), Usage()
        action, usage, _ = await self._call(
            "act", Action, intent=f"{intent.goal} ({intent.rationale})", affordances=_render_affordances(affordances)
        )
        spec = next((a for a in affordances if a.name == action.name), None)
        required = set(spec.args_schema.get("required", [])) if spec else set()
        if spec is None or not required <= set(action.args):
            return (Action(name=affordances[0].name), True), usage
        return (action, False), usage

    # ----- introspection --------------------------------------------------- #

    def state(self) -> dict[str, Any]:
        """Identity and running stats of the cell."""
        return {
            "cell_id": self.cell_id,
            "step": self.steps,
            "program": self.program,
            "genome": {"name": self.genome.name, "digest": self.genome.digest()},
            "store": self.store.stats(),
            "last": self.last_result.model_dump() if self.last_result else None,
        }


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)
