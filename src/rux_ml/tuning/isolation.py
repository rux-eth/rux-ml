"""Parent-side subprocess dispatcher (per PR-008 + D10/D16).

The parent doesn't use ``study.optimize`` for the subprocess path. It loops
``n_trials`` times calling :func:`run_subprocess_trial`, which spawns a fresh
``python -m rux_ml._internal.trial_runner`` child via ``subprocess.run`` with
spawn semantics (per ``docs/CONSTRAINTS.md`` "CUDA + fork is forbidden").
Each child runs its own ``study.optimize(..., n_trials=1)``; SQLite coordinates
state across trials.

The pattern is grounded in D10's cited Optuna distributed tutorial ("each
worker = one trial") adapted to sequential single-GPU execution per D6.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import optuna
from optuna.trial import TrialState

if TYPE_CHECKING:
    from rux_ml.config import RuxMLConfig

logger = logging.getLogger(__name__)


def _write_overrides_json(overrides: dict[str, object]) -> Path:
    """Serialise the parent's ``--set`` overrides to a tmp JSON for the child.

    Returns the path; caller is responsible for cleanup (we use a NamedTemp
    that persists until explicit unlink so the path survives the child run).
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="rux-ml-overrides-", delete=False
    ) as fd:
        json.dump(overrides, fd)
    return Path(fd.name)


def _child_argv(
    cfg_path: Path,
    problem: str | None,
    study_layer: str | None,
    study_name: str,
    overrides_json: Path | None,
) -> list[str]:
    """Build the child command-line argv."""
    argv: list[str] = [
        sys.executable,
        "-m",
        "rux_ml._internal.trial_runner",
        "--config",
        str(cfg_path),
        "--study-name",
        study_name,
    ]
    if problem is not None:
        argv.extend(["--problem", problem])
    if study_layer is not None:
        argv.extend(["--study", study_layer])
    if overrides_json is not None:
        argv.extend(["--overrides-json", str(overrides_json)])
    return argv


def _mark_trial_failed(study_name: str, storage_url: str) -> None:
    """Mark any RUNNING trial in the study as FAIL (used after timeout/kill)."""
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
    except KeyError:
        return  # study doesn't exist; nothing to mark
    for trial in study.trials:
        if trial.state == TrialState.RUNNING:
            # Reaching across Optuna's public surface — ``Study._storage`` is the
            # only documented way to set a trial's state externally as of 4.8.
            study._storage.set_trial_state_values(  # pyright: ignore[reportPrivateUsage]
                trial._trial_id,  # pyright: ignore[reportPrivateUsage]
                TrialState.FAIL,
            )


def run_subprocess_trial(
    cfg: RuxMLConfig,
    cfg_path: Path,
    problem: str | None,
    study_layer: str | None,
    study_name: str,
    overrides: dict[str, object],
) -> int:
    """Run one Optuna trial in an isolated subprocess; return the child's exit code.

    Honors ``cfg.tuning.trial_timeout_s`` — on timeout, the child is killed and
    any RUNNING trial in the study is marked ``FAIL`` so subsequent trials
    don't see a stale running trial.

    Args:
        cfg: Loaded ``RuxMLConfig`` (the parent's view of config; the child
            re-loads from the same TOML stack via ``--config`` + ``--problem``
            + ``--study`` + ``--overrides-json``).
        cfg_path: Base TOML path passed via ``--config``.
        problem: Problem-layer name (``opts.problem``) — passed unchanged to child.
        study_layer: Study-layer name (``opts.study``) — config overlay, NOT
            the Optuna study identifier.
        study_name: Optuna study identifier (the SQLite row key).
        overrides: The parent's ``--set`` overrides dict; serialised to a tmp
            JSON the child reads.

    Returns the child's exit code (0 on success, non-zero on failure).
    """
    overrides_json: Path | None = None
    if overrides:
        overrides_json = _write_overrides_json(overrides)
    try:
        argv = _child_argv(cfg_path, problem, study_layer, study_name, overrides_json)
        logger.info("spawning child: %s", " ".join(argv))
        try:
            result = subprocess.run(
                argv,
                check=False,
                timeout=cfg.tuning.trial_timeout_s,
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                "child trial exceeded trial_timeout_s=%s; killing and marking trial FAIL",
                cfg.tuning.trial_timeout_s,
            )
            _mark_trial_failed(study_name, cfg.runs.storage_url)
            return 124  # canonical timeout exit code (per GNU `timeout`)
        return result.returncode
    finally:
        if overrides_json is not None:
            overrides_json.unlink(missing_ok=True)
