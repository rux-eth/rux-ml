"""``rux-ml solve`` — single one-shot solver run (per PR-020 / Q-Shape: B partial mirror).

Mirrors :func:`rux_ml.cli.train.run_command` in shape — wraps a one-off
solve as a 1-trial Optuna study so the same SQLite store holds Trainer
trials and Solver trials. Per PR-020 Q-Shape research findings (Optuna +
MLflow ecosystem precedent): the solving layer reuses the cross-cutting
substrate (config / study / provenance) without forcing the impedance-
mismatched layers (cfg.data / cfg.cv / fittable-model registry).

Solver-trial provenance differs from Trainer-trial provenance:
- ``data_hash`` / ``data_bytes_hash`` / ``data_logical_hash`` are
  placeholder strings (``"none:solver-trial"``) — solver runs have no
  tabular data input. The real provenance lives in ``solving_cfg_hash``
  (covering ``cfg.solving.problem_module`` + solver / verbose / opts).
- ``solver_status``, ``objective_value``, ``solver_iter_count``,
  ``solve_time_s`` are populated from the :class:`SolverResult`.
- ``metric`` is hardcoded to ``"objective_value"`` (the solver's
  optimized quantity is what gets recorded as the trial score).

PR-020 deliberately defers solver-internal HPO to a follow-up PR; this
command is one-shot only (no sweep machinery).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

import typer

from rux_ml._internal.env import get_versions, pin_threads
from rux_ml._internal.git import git_sha
from rux_ml._internal.seeds import make_seed_bag
from rux_ml.cli._shared import get_options
from rux_ml.config import RuxMLConfig
from rux_ml.config.root import cfg_hash, layer_cfg_hash
from rux_ml.runs import TrialAttrs, one_off_run
from rux_ml.solving import make_solver

if TYPE_CHECKING:
    from rux_ml.solving.result import SolverResult


_SOLVER_TRIAL_METRIC = "objective_value"
_SOLVER_DATA_PLACEHOLDER = "none:solver-trial"


def _load_problem(problem_module: str) -> Any:
    """Import ``problem_module`` and call its ``build_problem()`` function.

    The module is required to export a ``build_problem()`` callable that
    takes no arguments and returns the family-specific problem type
    (for the CVXPY family: a ``cvxpy.Problem``). The CLI imports the
    module fresh on each invocation (no caching) so users can iterate
    on problem definitions without restarting.
    """
    try:
        module = importlib.import_module(problem_module)
    except ImportError as exc:
        msg = (
            f"cfg.solving.problem_module={problem_module!r} could not be imported; "
            f"ensure the module is on sys.path (e.g., in the workbench repo or a "
            f"location in PYTHONPATH). Original error: {exc}"
        )
        raise typer.BadParameter(msg) from exc

    builder = getattr(module, "build_problem", None)
    if builder is None or not callable(builder):
        msg = (
            f"cfg.solving.problem_module={problem_module!r} does not export a "
            f"callable ``build_problem()`` function. Define ``def build_problem(): ..."
            f"`` in the module."
        )
        raise typer.BadParameter(msg)

    return builder()


def _record_solver_attrs(
    cfg: RuxMLConfig,
    *,
    result: SolverResult,
    peak_rss_mb: float,
    trial: Any,
) -> None:
    """Construct + record ``TrialAttrs`` for a solver trial.

    Per PR-020 Q-Provenance (option i): single ``TrialAttrs`` schema with
    Optional Trainer-side AND Optional Solver-side fields. Trainer fields
    (``training_cfg_hash``, ``xgboost_version``, etc) stay populated from
    ``cfg.training``'s defaults so the schema validates; the meaningful
    solver-trial provenance lives in ``solving_cfg_hash`` +
    ``solver_status`` / ``objective_value`` / ``solver_iter_count`` /
    ``solve_time_s``.
    """
    bag = make_seed_bag(master_entropy=cfg.tuning.entropy, trial_number=trial.number)
    versions = get_versions(cfg.memory)

    attrs = TrialAttrs(
        # Per-layer config hashes (all 9 layers including the new ``solving``).
        data_cfg_hash=layer_cfg_hash(cfg, "data"),
        features_cfg_hash=layer_cfg_hash(cfg, "features"),
        training_cfg_hash=layer_cfg_hash(cfg, "training"),
        tuning_cfg_hash=layer_cfg_hash(cfg, "tuning"),
        runs_cfg_hash=layer_cfg_hash(cfg, "runs"),
        registry_cfg_hash=layer_cfg_hash(cfg, "registry"),
        memory_cfg_hash=layer_cfg_hash(cfg, "memory"),
        cv_cfg_hash=layer_cfg_hash(cfg, "cv"),
        solving_cfg_hash=layer_cfg_hash(cfg, "solving"),
        root_cfg_hash=cfg_hash(cfg),
        # Code + data provenance. Solver trials use placeholder data
        # hashes (no Parquet input); the real solver-side provenance is
        # in ``solving_cfg_hash``.
        git_sha=git_sha(),
        data_hash=_SOLVER_DATA_PLACEHOLDER,
        data_bytes_hash=_SOLVER_DATA_PLACEHOLDER,
        data_logical_hash=_SOLVER_DATA_PLACEHOLDER,
        # Trial-level metric identity.
        metric=_SOLVER_TRIAL_METRIC,
        # PR-013 required env block (filled from versions + bag).
        entropy_hex=bag.entropy_hex,
        image_digest=versions.image_digest,
        xgboost_version=versions.xgboost_version,
        cuda_runtime_version=versions.cuda_runtime_version,
        omp_threads=versions.omp_threads,
        peak_rss_mb=peak_rss_mb,
        gpu_model=versions.gpu_model,
        driver_version=versions.driver_version,
        # PR-020 solver-runtime fields.
        solver_status=result.solver_status,
        objective_value=result.objective_value,
        solver_iter_count=result.iter_count,
        solve_time_s=result.solve_time_s,
    )
    attrs.record(trial)


def run_command(ctx: typer.Context) -> None:
    """Run a single one-shot solver; recorded as a 1-trial Optuna study.

    Mirror of ``rux-ml train`` for solver-shaped runs. The optimized
    quantity (``SolverResult.objective_value``) is the trial's score;
    the study direction is fixed at ``"minimize"`` because cvxpy
    canonicalizes to a minimization problem regardless of user-side
    Maximize/Minimize sense.

    Raises:
        typer.BadParameter: when ``cfg.solving`` is None (the verb
            requires a solving config) or when ``problem_module`` fails
            to import / lacks ``build_problem()``.
    """
    opts = get_options(ctx)
    cfg = RuxMLConfig.from_layers(
        opts.config,
        problem=opts.problem,
        study=opts.study,
        overrides=opts.overrides,
    )

    if cfg.solving is None:
        msg = (
            "rux-ml solve requires cfg.solving to be set (via "
            "configs/problems/<problem>.toml [solving] block or "
            "--set solving.kind=cvxpy --set solving.problem_module=...)"
        )
        raise typer.BadParameter(msg)

    # PR-011: pin OMP/BLAS/POLARS env vars (no-op for cvxpy / clarabel
    # but kept for thread-count consistency with Trainer trials).
    pin_threads(cfg.memory)

    problem = _load_problem(cfg.solving.problem_module)  # pyright: ignore[reportAttributeAccessIssue]
    solver = make_solver(cfg.solving)

    with one_off_run(
        cfg,
        problem=opts.problem,
        study=opts.study,
        direction="minimize",
    ) as run:
        result = solver.solve(problem)

        # Solver-trial provenance recorded post-solve. Peak RSS is not
        # measured via Watchdog here — solver runs are typically
        # negligible-memory at workbench scale; the watchdog adds
        # overhead disproportionate to the use case. Future PR can
        # add an opt-in flag if needed. ``peak_rss_mb=0.0`` is a
        # placeholder consistent with the "solver trials are tiny" model.
        _record_solver_attrs(
            cfg,
            result=result,
            peak_rss_mb=0.0,
            trial=run.trial,
        )
        run.tell(result.objective_value)

    typer.echo(f"status:    {result.solver_status}")
    typer.echo(f"  obj_value:    {result.objective_value:.6g}")
    typer.echo(f"  solve_time_s: {result.solve_time_s:.4f}")
    typer.echo(f"  iter_count:   {result.iter_count}")
    typer.echo(f"  study:        {run.study.study_name}")
    typer.echo(f"  trial:        {run.trial.number}")
    typer.echo(f"  storage:      {cfg.runs.storage_url}")
