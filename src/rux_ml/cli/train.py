"""``rux-ml train`` — single baseline training (real body lands in PR-006)."""

from __future__ import annotations

import typer

from rux_ml.cli._shared import get_options, not_implemented


def run_command(ctx: typer.Context) -> None:
    """Run a single baseline training; recorded as a 1-trial Optuna study (per D7)."""
    opts = get_options(ctx)
    not_implemented(
        f"train (config={opts.config}, problem={opts.problem}, study={opts.study})",
        "PR-006",
    )
