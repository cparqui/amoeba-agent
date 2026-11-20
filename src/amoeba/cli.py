"""The `amoeba` command: launch a cell and talk to it."""

from __future__ import annotations

import click


@click.group()
@click.version_option(package_name="amoeba-agent")
def main() -> None:
    """Launch and interact with Amoeba cells."""


@main.command()
@click.option("--genome", type=click.Path(), help="Genome YAML file.")
@click.option("--db", type=click.Path(), help="World-model database for this cell.")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8700, show_default=True)
def run(genome: str | None, db: str | None, host: str, port: int) -> None:
    """Launch a cell as a local API process."""
    import uvicorn

    from amoeba.server import create_app

    uvicorn.run(create_app(), host=host, port=port)


@main.command()
@click.argument("text")
@click.option("--url", default="http://127.0.0.1:8700", show_default=True)
def step(text: str, url: str) -> None:
    """Send one observation to a running cell."""
    raise click.ClickException("not implemented")


@main.command()
@click.option("--url", default="http://127.0.0.1:8700", show_default=True)
def state(url: str) -> None:
    """Show a running cell's state."""
    raise click.ClickException("not implemented")


@main.command()
@click.argument("entity", required=False)
@click.option("--hops", default=1, show_default=True)
@click.option("--url", default="http://127.0.0.1:8700", show_default=True)
def model(entity: str | None, hops: int, url: str) -> None:
    """Show part of a running cell's world model."""
    raise click.ClickException("not implemented")
