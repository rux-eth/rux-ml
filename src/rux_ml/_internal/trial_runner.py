"""Subprocess-per-trial child entry point (per PR-008 + D10/D16).

Invoked by the parent ``rux-ml tune`` CLI via:

    python -m rux_ml._internal.trial_runner \\
        --config configs/base.toml \\
        [--problem <name>] [--study <name>] \\
        --study-name <optuna-study-identifier> \\
        [--overrides-json /tmp/overrides.json]

The child runs **exactly one trial** (`study.optimize(build_objective(cfg),
n_trials=1)`) and exits. Storage (SQLite) coordinates state across trials per
D10/D16. The parent never calls ``study.optimize`` — it loops ``n_trials``
times spawning fresh children, which is the canonical Optuna distributed
pattern (each worker = one independent trial) adapted to sequential
single-GPU execution.

**Critical**: top-level imports stay stdlib-only so the child can pin BLAS /
OpenMP / Polars thread env vars (from ``cfg.memory``) **before** importing
numpy / polars / sklearn / xgboost. Those libraries read the env vars at
import-time to size their thread pools; setting them late is a no-op.
"""

# pyright: reportMissingTypeArgument=false

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rux_ml.config import RuxMLConfig


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="rux_ml._internal.trial_runner",
        description="Run a single Optuna trial in an isolated subprocess (PR-008).",
    )
    parser.add_argument("--config", type=Path, required=True, help="Base TOML config path.")
    parser.add_argument(
        "--problem", type=str, default=None, help="Optional configs/problems/<name>.toml overlay."
    )
    parser.add_argument(
        "--study", type=str, default=None, help="Optional configs/studies/<name>.toml overlay."
    )
    parser.add_argument(
        "--study-name",
        type=str,
        required=True,
        help="Optuna study identifier (the SQLite row key, not the config layer name).",
    )
    parser.add_argument(
        "--overrides-json",
        type=Path,
        default=None,
        help="Path to a JSON file with a dict[str, Any] of dot-path overrides.",
    )
    return parser.parse_args(argv)


def _load_overrides(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    raw: object = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        msg = f"overrides JSON at {path} must contain a dict, got {type(raw).__name__}"
        raise TypeError(msg)
    return dict(raw)  # type: ignore[arg-type]




def main(argv: list[str] | None = None) -> int:
    """Child entry point — run one trial and exit.

    Returns the process exit code so the parent can distinguish success (0)
    from failure (non-zero). Hard failures (config load, OOM) propagate as
    non-zero exits; pruning / objective TrialPruned exceptions are absorbed
    by Optuna and the trial is recorded as PRUNED in storage.
    """
    args = _parse_args(argv)
    overrides = _load_overrides(args.overrides_json)

    # Config import is light (pydantic + tomllib only); safe before env pinning.
    from rux_ml.config import RuxMLConfig  # noqa: PLC0415

    cfg: RuxMLConfig = RuxMLConfig.from_layers(
        args.config,
        problem=args.problem,
        study=args.study,
        overrides=overrides,
    )

    # CRITICAL: pin BLAS/OpenMP/Polars env vars BEFORE importing numpy/polars/sklearn/xgboost.
    from rux_ml._internal.env import pin_threads  # noqa: PLC0415

    pin_threads(cfg.memory)

    # Lazy-import the rest of the stack now that env is pinned.
    # ``seeds`` triggers numpy via its function bodies (top-level imports stay
    # stdlib + pydantic) but we import lazily for symmetry with the rest.
    from rux_ml._internal.seeds import make_seed_bag  # noqa: PLC0415
    from rux_ml.training import optuna_direction  # noqa: PLC0415
    from rux_ml.tuning import (  # noqa: PLC0415
        build_objective,
        create_or_load,
        make_pruner,
        make_sampler,
    )

    # PR-013: study-level bag (trial_number=0 as the sentinel for study-creation
    # seeding) supplies the sampler seed. The objective derives per-trial bags
    # internally using the same master entropy + actual trial.number.
    study_bag = make_seed_bag(master_entropy=cfg.tuning.entropy, trial_number=0)

    study = create_or_load(
        name=args.study_name,
        storage=cfg.runs.storage_url,
        sampler=make_sampler(cfg.tuning, seed=study_bag.sampler_seed),
        pruner=make_pruner(cfg.tuning),
        direction=optuna_direction(cfg.training.metric),
        load_if_exists=True,
    )
    objective = build_objective(cfg)
    study.optimize(objective, n_trials=1)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
