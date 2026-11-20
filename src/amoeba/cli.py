"""The `amoeba` command: launch a cell and talk to it."""

from __future__ import annotations

import json
from typing import Any

import click
import httpx

URL = click.option("--url", default="http://127.0.0.1:8700", show_default=True, help="Running cell.")


def _print(data: Any) -> None:
    click.echo(json.dumps(data, indent=2))


def _call(fn: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except httpx.HTTPStatusError as exc:
        raise click.ClickException(f"{exc.response.status_code}: {exc.response.text}") from exc
    except httpx.HTTPError as exc:
        raise click.ClickException(f"cannot reach cell: {exc}") from exc


@click.group()
@click.version_option(package_name="amoeba-agent")
def main() -> None:
    """Launch and interact with Amoeba cells."""


@main.command()
@click.option("--genome", type=click.Path(exists=True, dir_okay=False), help="Genome YAML (default: packaged).")
@click.option("--db", default=":memory:", show_default=True, help="World-model database for this cell.")
@click.option("--cell-id", default="cell", show_default=True)
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8700, show_default=True)
def run(genome: str | None, db: str, cell_id: str, host: str, port: int) -> None:
    """Launch a cell as a local API process (the process is its body)."""
    import uvicorn

    from amoeba.server import create_app

    uvicorn.run(create_app(genome_path=genome, db_path=db, cell_id=cell_id), host=host, port=port)


@main.command()
@click.argument("text")
@click.option(
    "--affordance", "-a", "affordances", multiple=True, metavar="NAME[:DESCRIPTION]", help="An afforded action (repeatable)."
)
@URL
def step(text: str, affordances: tuple[str, ...], url: str) -> None:
    """Send one observation to a running cell."""
    from amoeba.client import Client

    specs = []
    for a in affordances:
        name, _, desc = a.partition(":")
        specs.append({"name": name.strip(), "description": desc.strip()})
    with Client(url) as client:
        _print(_call(client.step, text, specs))


@main.command()
@URL
def state(url: str) -> None:
    """Show a running cell's state."""
    from amoeba.client import Client

    with Client(url) as client:
        _print(_call(client.state))


@main.command()
@click.argument("entity", required=False)
@click.option("--hops", default=1, show_default=True)
@click.option("--limit", default=50, show_default=True)
@URL
def model(entity: str | None, hops: int, limit: int, url: str) -> None:
    """Show part of a running cell's world model."""
    from amoeba.client import Client

    with Client(url) as client:
        _print(_call(client.model, entity, hops, limit))
