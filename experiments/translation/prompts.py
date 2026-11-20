"""Perceiver and judge prompts."""

from __future__ import annotations

from .syntax import Syntax

PERCEIVE_SYSTEM = """\
You are the perception stage of a small agent. Translate each observation into the \
agent's internal language: relations between things, and nothing else.

Rules:
- Each relation has a predicate and its arguments, usually (predicate subject object).
- The only built-in operators are and, or, not. Everything else is a predicate.
- To qualify a statement (when it happened, how sure, who said it), wrap it in a \
predicate, e.g. at_time over a statement.
- The agent itself is the symbol self.
- Reuse symbols from the lexicon whenever one fits. Coin a new lowercase snake_case \
symbol only when none does.
- If something cannot be identified, use a variable instead of guessing a name.
- Output only the translation, no commentary.

Syntax: {notes}

Examples:
{examples}"""

PERCEIVE_USER = """\
Lexicon (symbols you already use):
{lexicon}

Observation:
{observation}

Translation:"""

JUDGE_SYSTEM = """\
Answer the question using only the information given. Reply with a few words. \
If the information does not say, reply "unknown"."""

JUDGE_USER = """\
Information:
{info}

Question: {question}
Answer:"""


def perceive_system(syntax: Syntax) -> str:
    examples = "\n\n".join(f"Observation: {o}\nTranslation:\n{t}" for o, t in syntax.examples)
    return PERCEIVE_SYSTEM.format(notes=syntax.notes, examples=examples)


def perceive_user(observation: str, lexicon: list[str]) -> str:
    return PERCEIVE_USER.format(
        lexicon=", ".join(lexicon) if lexicon else "(empty)", observation=observation
    )
