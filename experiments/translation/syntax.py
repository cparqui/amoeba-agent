"""Candidate surface syntaxes for the internal language, and one canonical form.

Every syntax parses into the same term tree (`Expr`, `Sym`, `Var`, `Lit`). Metrics
compare *facts*, a canonical form that is insensitive to how modifiers nest:

    (at_time t1 (conf 0.8 (on cup table)))
    (conf 0.8 (at_time t1 (on cup table)))

both become one fact: relation `on(cup, table)` under the context
{at_time(t1), conf(0.8)}. Only `and`, `or` and `not` are built in; everything
else, including time and confidence, is an ordinary (learned) predicate.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Union

BUILTINS = frozenset({"and", "or", "not"})


# --------------------------------------------------------------------------- #
# Terms
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Sym:
    name: str


@dataclass(frozen=True)
class Var:
    name: str


@dataclass(frozen=True)
class Lit:
    value: str  # numbers and quoted strings, kept as text


@dataclass(frozen=True)
class Expr:
    head: str
    args: tuple["Term", ...]


Term = Union[Sym, Var, Lit, Expr]


class ParseError(ValueError):
    pass


def norm_symbol(name: str) -> str:
    """Case- and separator-insensitive symbol identity."""
    return re.sub(r"[\s\-]+", "_", name.strip().lower())


def render(term: Term) -> str:
    """Render any term as an s-expression (the log format)."""
    if isinstance(term, Expr):
        return "(" + " ".join([term.head, *(render(a) for a in term.args)]) + ")"
    if isinstance(term, Var):
        return "?" + term.name
    if isinstance(term, Lit):
        return term.value
    return term.name


# --------------------------------------------------------------------------- #
# Parsers
# --------------------------------------------------------------------------- #

_NUM = re.compile(r"-?\d+(\.\d+)?$")


def _atom(tok: str) -> Term:
    if tok.startswith("?"):
        if len(tok) < 2:
            raise ParseError("empty variable")
        return Var(tok[1:].lower())
    if tok.startswith('"') or _NUM.match(tok):
        return Lit(tok)
    return Sym(tok)


def parse_sexpr(text: str) -> list[Expr]:
    tokens = re.findall(r'\(|\)|"[^"]*"|[^\s()]+', text)
    pos = 0

    def expr() -> Expr:
        nonlocal pos
        pos += 1  # "("
        if pos >= len(tokens) or tokens[pos] in "()":
            raise ParseError("expression needs a symbol head")
        head = tokens[pos]
        if head.startswith(("?", '"')) or _NUM.match(head):
            raise ParseError(f"bad head {head!r}")
        pos += 1
        args: list[Term] = []
        while pos < len(tokens) and tokens[pos] != ")":
            args.append(expr() if tokens[pos] == "(" else _atom(tokens[pos]))
            if not isinstance(args[-1], Expr):
                pos += 1
        if pos >= len(tokens):
            raise ParseError("unbalanced parentheses")
        pos += 1  # ")"
        if not args:
            raise ParseError(f"({head}) has no arguments")
        return Expr(head, tuple(args))

    out: list[Expr] = []
    while pos < len(tokens):
        if tokens[pos] != "(":
            raise ParseError(f"stray token {tokens[pos]!r}")
        out.append(expr())
    if not out:
        raise ParseError("no expressions")
    return out


def parse_prolog(text: str) -> list[Expr]:
    tokens = re.findall(r"-?\d+(?:\.\d+)?|[A-Za-z_][A-Za-z0-9_\-]*|\"[^\"]*\"|'[^']*'|[(),.]", text)
    pos = 0

    def term() -> Term:
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        if tok in "(),.":
            raise ParseError(f"unexpected {tok!r}")
        if tok[0].isupper() or tok[0] == "_":
            return Var(tok.lstrip("_").lower() or "_")
        if tok[0] in "\"'" or _NUM.match(tok):
            return Lit('"' + tok[1:-1] + '"' if tok[0] == "'" else tok)
        if pos < len(tokens) and tokens[pos] == "(":
            pos += 1
            args = [term()]
            while tokens[pos] == ",":
                pos += 1
                args.append(term())
            if tokens[pos] != ")":
                raise ParseError("expected )")
            pos += 1
            return Expr(tok, tuple(args))
        return Sym(tok)

    out: list[Expr] = []
    try:
        while pos < len(tokens):
            t = term()
            if not isinstance(t, Expr):
                raise ParseError(f"clause must be a compound term, got {render(t)}")
            if pos >= len(tokens) or tokens[pos] != ".":
                raise ParseError("clause must end with '.'")
            pos += 1
            out.append(t)
    except IndexError as exc:
        raise ParseError("truncated input") from exc
    if not out:
        raise ParseError("no clauses")
    return out


def parse_json(text: str) -> list[Expr]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ParseError(str(exc)) from exc

    def term(x: object) -> Term:
        if isinstance(x, list):
            if len(x) < 2 or not isinstance(x[0], str):
                raise ParseError(f"expression needs a head and arguments: {x!r}")
            return Expr(x[0], tuple(term(a) for a in x[1:]))
        if isinstance(x, bool) or x is None:
            return Lit(json.dumps(x))
        if isinstance(x, (int, float)):
            return Lit(str(x))
        if isinstance(x, str):
            if x.startswith("?"):
                return Var(x[1:].lower())
            return Lit(json.dumps(x)) if " " in x.strip() else Sym(x)
        raise ParseError(f"unsupported value {x!r}")

    if not isinstance(data, list) or not data:
        raise ParseError("top level must be a non-empty array of expressions")
    out = [term(x) for x in data]
    if not all(isinstance(t, Expr) for t in out):
        raise ParseError("every top-level item must be an expression")
    return out  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Syntax definitions: parser, GBNF grammar, prompt notes, examples
# --------------------------------------------------------------------------- #

_GBNF_SEXPR = r"""
root  ::= expr ("\n" expr){0,15} "\n"?
expr  ::= "(" sym (" " term)+ ")"
term  ::= expr | sym | var | num | str
sym   ::= [a-z] [a-z0-9_]{0,31}
var   ::= "?" [a-z0-9_]{1,16}
num   ::= "-"? [0-9]{1,9} ("." [0-9]{1,6})?
str   ::= "\"" [^"\n]{0,60} "\""
"""

_GBNF_PROLOG = r"""
root  ::= clause ("\n" clause){0,15} "\n"?
clause ::= cmp "."
cmp   ::= sym "(" arg (", " arg){0,7} ")"
arg   ::= cmp | sym | var | num | str
sym   ::= [a-z] [a-z0-9_]{0,31}
var   ::= [A-Z] [A-Za-z0-9_]{0,15}
num   ::= "-"? [0-9]{1,9} ("." [0-9]{1,6})?
str   ::= "\"" [^"\n]{0,60} "\""
"""

_GBNF_JSON = r"""
root  ::= "[" ws expr ("," ws expr){0,15} ws "]"
expr  ::= "[" ws sym ("," ws item){1,8} ws "]"
item  ::= expr | sym | var | num | str
sym   ::= "\"" [a-z] [a-z0-9_]{0,31} "\""
var   ::= "\"?" [a-z0-9_]{1,16} "\""
num   ::= "-"? [0-9]{1,9} ("." [0-9]{1,6})?
str   ::= "\"" [^"\n]{0,60} "\""
ws    ::= [ \n]{0,2}
"""


@dataclass(frozen=True)
class Syntax:
    name: str
    parse: Callable[[str], list[Expr]]
    grammar: str
    notes: str
    examples: tuple[tuple[str, str], ...]  # (observation, translation)


SYNTAXES: dict[str, Syntax] = {
    "sexpr": Syntax(
        name="sexpr",
        parse=parse_sexpr,
        grammar=_GBNF_SEXPR.strip(),
        notes=(
            "Write one s-expression per line: (predicate subject object). "
            "Variables start with '?', e.g. ?x1. Numbers are bare; free text goes in double quotes."
        ),
        examples=(
            (
                "The kettle is on the stove and it is hot.",
                "(on kettle stove)\n(has_property kettle hot)",
            ),
            (
                "t=7 an unknown object touched you; the lamp is not lit.",
                "(at_time 7 (touches ?x1 self))\n(not (lit lamp))",
            ),
        ),
    ),
    "prolog": Syntax(
        name="prolog",
        parse=parse_prolog,
        grammar=_GBNF_PROLOG.strip(),
        notes=(
            "Write one Prolog-style term per line, ending with a period: predicate(subject, object). "
            "Variables start with an uppercase letter, e.g. X1. Numbers are bare; free text goes in double quotes."
        ),
        examples=(
            (
                "The kettle is on the stove and it is hot.",
                "on(kettle, stove).\nhas_property(kettle, hot).",
            ),
            (
                "t=7 an unknown object touched you; the lamp is not lit.",
                "at_time(7, touches(X1, self)).\nnot(lit(lamp)).",
            ),
        ),
    ),
    "json": Syntax(
        name="json",
        parse=parse_json,
        grammar=_GBNF_JSON.strip(),
        notes=(
            'Write one JSON array of expressions. Each expression is an array: ["predicate", "subject", "object"]. '
            'Variables are strings starting with "?", e.g. "?x1". Numbers are bare.'
        ),
        examples=(
            (
                "The kettle is on the stove and it is hot.",
                '[["on", "kettle", "stove"], ["has_property", "kettle", "hot"]]',
            ),
            (
                "t=7 an unknown object touched you; the lamp is not lit.",
                '[["at_time", 7, ["touches", "?x1", "self"]], ["not", ["lit", "lamp"]]]',
            ),
        ),
    ),
}


# --------------------------------------------------------------------------- #
# Canonical form
# --------------------------------------------------------------------------- #

Fact = tuple[tuple[tuple[str, ...], ...], str, tuple[str, ...]]


def _key(term: Term, anonymize: bool) -> str:
    if isinstance(term, Var):
        return "?" if anonymize else "?" + term.name
    if isinstance(term, Lit):
        return term.value.lower()
    if isinstance(term, Sym):
        return norm_symbol(term.name)
    return render(term)


def facts(exprs: list[Expr], anonymize_vars: bool = True) -> frozenset[Fact]:
    """Flatten nested statements into (context, predicate, args) facts.

    The context is the *sorted* tuple of wrappers (modifiers, `not`, `or`) around
    a relation, so the order in which a model nests modifiers does not matter.
    Variables are anonymized by default, so ?x1 and ?blob compare equal.
    """
    out: set[Fact] = set()

    def walk(e: Expr, ctx: tuple[tuple[str, ...], ...]) -> None:
        head = norm_symbol(e.head)
        subs = [a for a in e.args if isinstance(a, Expr)]
        if head == "and":
            for s in subs:
                walk(s, ctx)
            return
        if head in ("not", "or"):
            for s in subs:
                walk(s, ctx + ((head,),))
            return
        if subs:  # a modifier over one or more statements
            wrapper = (head, *(_key(a, anonymize_vars) for a in e.args if not isinstance(a, Expr)))
            for s in subs:
                walk(s, ctx + (wrapper,))
            return
        out.add((tuple(sorted(ctx)), head, tuple(_key(a, anonymize_vars) for a in e.args)))

    for e in exprs:
        walk(e, ())
    return frozenset(out)


def symbols(exprs: list[Expr]) -> tuple[set[str], set[str]]:
    """(predicates, entity constants) used, excluding built-ins."""
    preds: set[str] = set()
    ents: set[str] = set()

    def walk(t: Term) -> None:
        if isinstance(t, Expr):
            h = norm_symbol(t.head)
            if h not in BUILTINS:
                preds.add(h)
            for a in t.args:
                walk(a)
        elif isinstance(t, Sym):
            ents.add(norm_symbol(t.name))

    for e in exprs:
        walk(e)
    return preds, ents


def has_variable(exprs: list[Expr]) -> bool:
    def walk(t: Term) -> bool:
        if isinstance(t, Var):
            return True
        return isinstance(t, Expr) and any(walk(a) for a in t.args)

    return any(walk(e) for e in exprs)


def strip_output(text: str) -> str:
    """Remove reasoning blocks and code fences some models wrap around output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^```[a-zA-Z]*\n?|```\s*$", "", text.strip(), flags=re.MULTILINE)
    return text.strip()
