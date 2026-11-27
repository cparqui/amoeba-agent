"""Load local settings and secrets from a `.env` file.

Experiments call `load_env()` at startup, so `uv run python -m experiments.<name> ...`
picks up the repo's `.env` (git-ignored; see `.env.example`) with no extra flags.

Variables already set in the shell win over the file, so a one-off
`MLFLOW_TRACKING_URI=... uv run ...` still works. Values are never written to
run dirs or MLflow: only `config.yaml` is copied, so keep secrets out of it.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import find_dotenv, load_dotenv


def load_env(path: str | Path | None = None) -> Path | None:
    """Load `path`, or the nearest `.env` from the working directory upward.

    Returns the file loaded, or None if there was none.
    """
    found = Path(path) if path else (Path(p) if (p := find_dotenv(usecwd=True)) else None)
    if not found or not found.is_file():
        return None
    load_dotenv(found, override=False)
    return found
