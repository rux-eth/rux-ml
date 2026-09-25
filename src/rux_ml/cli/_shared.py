"""Shared helpers for the CLI subpackage.

Owns:
- ``GlobalOptions`` — the typed payload the root callback stuffs into the Typer
  ``Context.obj`` so subcommands can retrieve global flags without re-parsing.
- ``parse_set_overrides`` — translates ``--set key=value`` repeatable flags into
  the ``overrides`` mapping accepted by ``RuxMLConfig.from_layers`` (per D17).
- ``not_implemented`` — uniform body for v0 no-op subcommands, naming the PR
  that will land the real implementation.
- ``refuse_oracle_source`` — the PR-040 oracle-quarantine preflight, mapped to
  ``typer.BadParameter`` (exit 2).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

from rux_ml.data import OracleQuarantineError, check_oracle_quarantine

if TYPE_CHECKING:
    from rux_ml.config import RuxMLConfig


@dataclass
class GlobalOptions:
    """Parsed values from the root ``@app.callback()``.

    Stashed in ``typer.Context.obj`` so every subcommand can retrieve it via
    ``ctx.ensure_object(GlobalOptions)``.
    """

    config: Path
    problem: str | None
    study: str | None
    overrides: dict[str, Any] = field(default_factory=dict)
    verbose: bool = False
    dry_run: bool = False


def parse_set_overrides(raw: list[str]) -> dict[str, Any]:
    """Parse ``['training.learning_rate=0.05', 'tuning.n_trials=100']`` into a dict.

    Values are JSON-parsed when possible (numbers, bools, lists, objects) and
    fall back to the raw string otherwise. The resulting dict is passed straight
    through to ``RuxMLConfig.from_layers(..., overrides=...)``; dot-paths into
    the schema are validated downstream (``extra="forbid"`` on every model
    surfaces typos as ``ValidationError``).
    """
    out: dict[str, Any] = {}
    for item in raw:
        if "=" not in item:
            msg = f"--set expects 'key=value', got {item!r}"
            raise typer.BadParameter(msg)
        key, _, value_str = item.partition("=")
        key = key.strip()
        if not key:
            msg = f"--set key must be non-empty, got {item!r}"
            raise typer.BadParameter(msg)
        try:
            value: Any = json.loads(value_str)
        except json.JSONDecodeError:
            value = value_str
        out[key] = value
    return out


def not_implemented(verb: str, pr: str) -> None:
    """Uniform no-op body for v0 subcommands."""
    typer.secho(
        f"{verb}: not yet implemented (lands in {pr})",
        fg=typer.colors.YELLOW,
        err=True,
    )


def get_options(ctx: typer.Context) -> GlobalOptions:
    """Retrieve the ``GlobalOptions`` payload, raising if the callback didn't run."""
    if not isinstance(ctx.obj, GlobalOptions):
        typer.echo(
            "internal error: GlobalOptions missing from Typer context",
            err=True,
        )
        sys.exit(2)
    return ctx.obj


def refuse_oracle_source(cfg: RuxMLConfig) -> None:
    """Run the oracle quarantine on ``cfg.data.source_path`` before any trial exists (PR-040).

    Verbs that build a training set call this first so a refused source exits 2
    with the named error, instead of surfacing mid-run (a FAIL trial row, or a
    subprocess sweep that logs each child's failure and exits 0). No-op when
    ``source_path`` is unset — the verb's own "source_path required" check owns
    that error.
    """
    if cfg.data.source_path is None:
        return
    try:
        check_oracle_quarantine(cfg.data.source_path, cfg.data.oracle)
    except OracleQuarantineError as exc:
        raise typer.BadParameter(str(exc)) from exc
