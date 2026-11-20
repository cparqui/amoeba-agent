from click.testing import CliRunner

import amoeba
from amoeba.cli import main
from amoeba.server import create_app


def test_package_imports() -> None:
    assert amoeba.Agent is not None


def test_cli_help() -> None:
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    for command in ("run", "step", "state", "model"):
        assert command in result.output


def test_app_builds() -> None:
    paths = {route.path for route in create_app().routes}
    assert {"/step", "/state", "/model"} <= paths
