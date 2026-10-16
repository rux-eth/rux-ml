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

**Thread pools (PR-063)**: the thread env must be in place **before** the
pools that size at load. libgomp and both OpenBLAS copies (numpy's, scipy's)
read it in load-time constructors; loading ``RuxMLConfig`` already loads
numpy / scipy / sklearn / xgboost / polars / pyarrow / pandas (the
``rux_ml.training`` package inits, since PR-017), so this module's own
:func:`~rux_ml._internal.env.pin_threads` comes too late for them (desktop:
libgomp 32 and 2 x 31 OpenBLAS workers against a pin of 24 / 1). The parent
therefore passes the four values in the child's env at spawn
(``tuning/isolation.py``); here the child records the env it started with,
warns (does not refuse — tests call :func:`main` in-process) when it
disagrees with ``cfg.memory``, and still pins: Polars (first use) and
LightGBM's own libgomp (loaded lazily) size after the pin and honour it.
Top-level imports stay stdlib-only so the snapshot sees the start env.
"""

# pyright: reportMissingTypeArgument=false

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

    from rux_ml.config import MemoryConfig, RuxMLConfig

logger = logging.getLogger(__name__)


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


def _warn_on_thread_env_mismatch(
    memory: MemoryConfig, start_env: Mapping[str, str | None]
) -> list[str]:
    """Warn (not refuse) when the env this process started with is not ``memory``'s.

    Returns the mismatched keys. A mismatch means libgomp / OpenBLAS sized
    their pools from the start env, not from ``cfg.memory`` (PR-063): the
    child was not spawned by ``tuning/isolation.py`` (e.g. run by hand).
    """
    from rux_ml._internal.env import thread_env  # noqa: PLC0415

    want = thread_env(memory)
    mismatched = [key for key, value in want.items() if start_env.get(key) != value]
    if mismatched:
        logger.warning(
            "thread env at start %s != cfg.memory %s (mismatched: %s): libgomp / OpenBLAS "
            "sized their pools from the start env and the pin below cannot reach them "
            "(PR-063); spawn the child through rux_ml.tuning.isolation.run_subprocess_trial",
            {key: start_env.get(key) for key in want},
            want,
            mismatched,
        )
    return mismatched


def main(argv: list[str] | None = None) -> int:
    """Child entry point — run one trial and exit.

    Returns the process exit code so the parent can distinguish success (0)
    from failure (non-zero). Hard failures (config load, OOM) propagate as
    non-zero exits; pruning / objective TrialPruned exceptions are absorbed
    by Optuna and the trial is recorded as PRUNED in storage.
    """
    args = _parse_args(argv)
    overrides = _load_overrides(args.overrides_json)

    # PR-063: the thread env this process STARTED with, read before the heavy import.
    from rux_ml._internal.env import THREAD_ENV_KEYS  # noqa: PLC0415 — stdlib-only module

    start_env = {key: os.environ.get(key) for key in THREAD_ENV_KEYS}

    # NOT light: rux_ml.config -> config/training.py -> the rux_ml.training package
    # inits load numpy / scipy / sklearn / xgboost / polars / pyarrow / pandas
    # (since PR-017 a98349f; PR-063 Phase 1), and libgomp / OpenBLAS size their
    # pools as they load. Their values come from the spawn env (tuning/isolation.py).
    from rux_ml.config import RuxMLConfig  # noqa: PLC0415

    cfg: RuxMLConfig = RuxMLConfig.from_layers(
        args.config,
        problem=args.problem,
        study=args.study,
        overrides=overrides,
    )

    _warn_on_thread_env_mismatch(cfg.memory, start_env)

    # Still pin (idempotent): Polars sizes on first use and LightGBM maps its own
    # libgomp lazily, both after this line (PR-063 Q1, desktop).
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
