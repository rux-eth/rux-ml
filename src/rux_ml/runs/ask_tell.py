"""``one_off_run`` context manager — wrap a single training as a 1-trial study.

Per D7 / PR-009: both sweep and one-off paths share the same Optuna storage
and the same provenance schema. ``one_off_run`` is the helper that makes the
1-trial baseline path indistinguishable in shape from a sweep trial.

Usage::

    with one_off_run(cfg, problem=opts.problem, study=opts.study) as run:
        attrs = TrialAttrs.from_cfg(cfg, data_hashes, metric=cfg.training.metric)
        attrs.record(run.trial)
        score = fit_and_score(...)
        run.tell(score)
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import optuna

from rux_ml.runs.provenance import ensure_storage_parent, study_name

if TYPE_CHECKING:
    from collections.abc import Generator

    from rux_ml.config import RuxMLConfig


@dataclass
class OneOffRun:
    """Yielded by :func:`one_off_run`. The caller calls ``run.tell(score)`` to
    complete the trial; if the caller exits the ``with`` block without telling,
    the trial is marked ``FAIL`` so storage doesn't keep a ``RUNNING`` trial."""

    trial: optuna.Trial
    study: optuna.Study
    _told: bool = field(default=False)

    def tell(self, score: float) -> None:
        """Complete the trial with the given score; subsequent ``with`` exit is a no-op."""
        self.study.tell(self.trial, score)
        self._told = True


@contextmanager
def one_off_run(
    cfg: RuxMLConfig,
    *,
    problem: str | None,
    study: str | None,
    direction: str = "maximize",
) -> Generator[OneOffRun]:
    """Create-or-load a 1-trial Optuna study; yield a :class:`OneOffRun`.

    Args:
        cfg: Resolved ``RuxMLConfig``; ``cfg.runs.storage_url`` selects storage.
        problem: ``opts.problem`` (CLI ``--problem`` overlay name) — used for
            study-name template substitution; **not** the Optuna study name itself.
        study: ``opts.study`` (CLI ``--study`` overlay name) — same role.
        direction: ``"maximize"`` / ``"minimize"`` per the training metric.
            Callers typically pass ``optuna_direction(cfg.training.metric)``.

    On exception from the ``with`` block, the trial is marked ``FAIL`` and the
    exception re-raises. On clean exit without ``.tell(score)``, the trial is
    silently marked ``FAIL`` (storage hygiene — no lingering ``RUNNING`` trials).
    """
    ensure_storage_parent(cfg.runs.storage_url)
    name = study_name(cfg, problem=problem, study=study)
    study_obj = optuna.create_study(
        study_name=name,
        storage=cfg.runs.storage_url,
        direction=direction,
        load_if_exists=True,
    )
    trial = study_obj.ask()
    run = OneOffRun(trial=trial, study=study_obj)
    try:
        yield run
    except BaseException:
        if not run._told:  # pyright: ignore[reportPrivateUsage]
            study_obj.tell(trial, state=optuna.trial.TrialState.FAIL)
        raise
    if not run._told:  # pyright: ignore[reportPrivateUsage]
        # Caller exited cleanly without telling; mark FAIL so storage isn't left RUNNING.
        study_obj.tell(trial, state=optuna.trial.TrialState.FAIL)
