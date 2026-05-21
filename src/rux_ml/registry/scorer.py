"""Holdout-fold scoring for promoted bundles (per PR-032).

Three public surfaces:

- :class:`_BoosterTrainerShim` — wraps a raw ``xgb.Booster`` exposing the
  ``Trainer`` Protocol (``predict`` + ``predict_proba``) so the metric
  registry's ``compute_score`` can be called on bundle-loaded boosters
  the same way it's called on sklearn-wrapped trainers. ``fit`` raises
  ``NotImplementedError`` — the shim is score-only. Same shape PR-033's
  native-``xgb.train`` adapter will use.

- :class:`HoldoutScoreReceipt` — Pydantic-validated JSON receipt schema
  for ``rux-ml registry score`` outputs. Body shape (``metrics`` dict +
  ``artifacts`` dict) mirrors MLflow ``EvaluationResult.metrics`` /
  ``artifacts_metadata.json`` convention. Provenance wrapper
  (``bundle_version``, ``manifest_git_sha``, ``promoted_from``,
  ``holdout``, ``scored_at*``) is rux-ml-specific (no cited tool bundles
  inline; defensible per AWS ML Lens BP03 for no-tracker-server
  environments).

- :func:`score_bundle_on_holdout` — orchestrates load-bundle →
  reconstruct splits → wrap booster → ``compute_score`` → write receipt
  JSON + predictions parquet under ``output_dir``.

Schema convention citations (Phase 3 research, 2026-05-21):
- Kedro ``tracking.MetricsDataSet`` @ starter tag ``0.19.14``
- MLflow ``mlflow.models.evaluate.EvaluationResult`` @ master
- AWS Well-Architected ML Lens BP03 (self-contained receipts when no
  tracker server exists)

This module is **not** exported from ``rux_ml.registry``'s public
``__init__.py`` (PR-010 sub-decision B1 — strict inference-deps
separation). CLI ``score`` imports it directly via
``from rux_ml.registry.scorer import score_bundle_on_holdout``.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import polars as pl
import xgboost as xgb
from pydantic import BaseModel, ConfigDict

from rux_ml._internal.seeds import make_seed_bag_from_hex
from rux_ml.data import load_parquet, make_splits, materialize
from rux_ml.registry.bundle import load_bundle
from rux_ml.registry.champion import read_champion
from rux_ml.registry.paths import champion_path, version_dir
from rux_ml.runs import load_run
from rux_ml.training import compute_score

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from rux_ml.config import RuxMLConfig


# ---------------------------------------------------------------------------
# Trainer-Protocol-compatible Booster shim
# ---------------------------------------------------------------------------


class _BoosterTrainerShim:
    """Wraps an :class:`xgb.Booster` with sklearn-style fit/predict/predict_proba.

    ``predict`` returns the booster's raw predictions; for binary
    classification, ``predict_proba`` returns the ``[1-p, p]`` 2D shape
    sklearn consumers (e.g. :func:`compute_score`'s ``_auc`` helper) expect.

    ``fit`` raises :class:`NotImplementedError` — the shim is score-only.
    The bundle's training stack lives in promote-time re-fit; this shim
    serves only the inference + scoring path.

    Construction uses ``xgb.DMatrix(x, enable_categorical=True)`` per the
    existing in-repo precedent at ``tests/golden/test_xgb_baseline.py:381``
    so models trained with native categorical handling round-trip
    correctly.
    """

    def __init__(self, booster: xgb.Booster) -> None:
        self._booster = booster

    def fit(self, *args: Any, **kwargs: Any) -> _BoosterTrainerShim:
        # *args / **kwargs match Trainer Protocol signature parity even though unused.
        del args, kwargs
        msg = "_BoosterTrainerShim is score-only; cannot fit (use registry/promote.py for re-fits)"
        raise NotImplementedError(msg)

    def _dmatrix(self, x: Any) -> xgb.DMatrix:
        return xgb.DMatrix(x, enable_categorical=True)

    def predict(self, x: Any) -> NDArray[Any]:
        raw = self._booster.predict(self._dmatrix(x))  # pyright: ignore[reportUnknownMemberType]
        return np.asarray(raw)

    def predict_proba(self, x: Any) -> NDArray[Any]:
        """Return ``[1-p, p]`` 2D shape (binary classification) compatible with sklearn.

        For binary XGBoost, ``booster.predict`` returns a 1D array of
        positive-class probabilities. The metric registry's ``_auc`` /
        ``_logloss`` helpers expect 2D sklearn-style proba — this method
        reshapes.
        """
        p = np.asarray(self._booster.predict(self._dmatrix(x)))  # pyright: ignore[reportUnknownMemberType]
        if p.ndim == 1:
            return np.column_stack([1.0 - p, p])
        return p


# ---------------------------------------------------------------------------
# Receipt schema (Pydantic-validated)
# ---------------------------------------------------------------------------


class _Artifact(BaseModel):
    """One entry in ``HoldoutScoreReceipt.artifacts`` (MLflow shape)."""

    model_config = ConfigDict(extra="forbid")

    path: str
    content_type: str


class _PromotedFrom(BaseModel):
    """Origin of the bundle (mirrors :class:`registry.manifest.PromotedFrom`)."""

    model_config = ConfigDict(extra="forbid")

    study: str
    trial_number: int
    metric_value: float


class _Holdout(BaseModel):
    """Description of the held-out test fold the bundle was scored against."""

    model_config = ConfigDict(extra="forbid")

    n_rows: int
    time_range: list[str] | None  # ``None`` when ``data.time_column`` is unset
    split_kind: str
    split_ratios: dict[str, float]


class HoldoutScoreReceipt(BaseModel):
    """Pydantic-validated receipt for ``rux-ml registry score`` output.

    Body shape (``metrics`` dict + ``artifacts`` dict) follows MLflow
    ``EvaluationResult`` + ``artifacts_metadata.json`` convention.
    Provenance wrapper is rux-ml-specific (no cited tool bundles inline;
    defensible per AWS ML Lens BP03 for no-tracker-server environments).
    """

    model_config = ConfigDict(extra="forbid")

    bundle_version: str
    bundle_dir: str
    manifest_git_sha: str
    promoted_from: _PromotedFrom
    holdout: _Holdout
    metrics: dict[str, float]
    artifacts: dict[str, _Artifact]
    scored_at: str
    scored_at_git_sha: str


# ---------------------------------------------------------------------------
# Scorer orchestration
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    """Return current HEAD SHA, or ``"unknown"`` if not in a git repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _reconstruct_split_seed(cfg: RuxMLConfig, study: str, trial_number: int) -> int:
    """Reconstruct the trial's ``bag.split_seed`` for random-split reproducibility.

    For ``time_ordered`` split, ``temporal_train_val_test_split`` ignores
    the seed (deterministic per PR-024) — any int is fine; we return ``0``.

    For ``random``, we re-open the originating trial via
    :func:`runs.load_run` to recover ``attrs.entropy_hex``, then derive
    the same ``split_seed`` ``make_splits`` would have seen at trial
    time via :func:`make_seed_bag_from_hex`.
    """
    if cfg.data.split_kind == "time_ordered":
        return 0
    run = load_run(cfg.runs.storage_url, study, trial_number)
    if run.attrs is None:
        msg = (
            f"trial #{trial_number} in study {study!r} has invalid provenance "
            f"(missing TrialAttrs); cannot reconstruct split_seed for random-split path"
        )
        raise ValueError(msg)
    bag = make_seed_bag_from_hex(run.attrs.entropy_hex)
    return bag.split_seed


def score_bundle_on_holdout(
    cfg: RuxMLConfig,
    *,
    problem: str,
    version: str | None = None,
    output_dir: Path = Path("receipts"),
) -> HoldoutScoreReceipt:
    """Score a promoted bundle on its truly-held-out ``splits["test"]`` fold.

    Loads the bundle (default: champion), reconstructs the test fold via
    :func:`make_splits` with the trial's ``split_seed``, wraps the raw
    :class:`xgb.Booster` in a :class:`_BoosterTrainerShim`, calls
    :func:`compute_score`, and persists a JSON receipt + Polars
    predictions parquet under ``output_dir``. Returns the in-memory
    receipt (already written to disk).

    Args:
        cfg: Resolved ``RuxMLConfig``. Must have ``data.source_path`` +
            ``data.target_column`` set.
        problem: Registry problem name.
        version: Bundle version-id; defaults to the current champion.
        output_dir: Where to write the receipt + preds parquet. Created
            if missing. Default ``Path("receipts")``.

    Raises:
        FileNotFoundError: if the champion pointer or bundle is missing.
        ValueError: if the holdout test fold is empty or the random-split
            trial's provenance can't be reconstructed.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve version (default = champion).
    if version is None:
        champ = read_champion(champion_path(cfg.registry.root, problem))
        version = cast("str", champ["version"])

    # Load bundle (skops Pipeline + UBJ Booster + Pydantic manifest).
    bundle_dir = version_dir(cfg.registry.root, problem, version)
    pipeline, booster, manifest = load_bundle(bundle_dir)

    # Reconstruct splits + carve the holdout test fold.
    if cfg.data.source_path is None or cfg.data.target_column is None:
        msg = "registry score requires data.source_path and data.target_column"
        raise ValueError(msg)

    df = materialize(load_parquet(cfg.data.source_path))
    split_seed = _reconstruct_split_seed(
        cfg, manifest.promoted_from.study, manifest.promoted_from.trial_number
    )
    splits = make_splits(cfg, df, seed=split_seed)
    test = splits["test"]

    if test.height == 0:
        msg = f"holdout test fold is empty for problem {problem!r} (n_rows=0)"
        raise ValueError(msg)

    # Transform features through the bundled pipeline + score via Booster shim.
    target_col = cfg.data.target_column
    x_test = test.drop(target_col)
    y_test = test[target_col].to_numpy()
    x_test_t = cast("pl.DataFrame", pipeline.transform(x_test))  # pyright: ignore[reportUnknownMemberType]
    shim = _BoosterTrainerShim(booster)
    x_test_pd = x_test_t.to_pandas()
    metric_value = compute_score(cfg.training.metric, shim, x_test_pd, y_test)

    # Predictions parquet — sibling to the JSON receipt.
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    preds_filename = f"holdout_preds_{problem}_{stamp}.parquet"
    preds_path = output_dir / preds_filename
    y_pred = shim.predict(x_test_pd)
    preds_cols: dict[str, Any] = {}
    if cfg.data.time_column is not None and cfg.data.time_column in test.columns:
        preds_cols[cfg.data.time_column] = test[cfg.data.time_column]
    preds_cols["y_true"] = y_test
    preds_cols["y_pred"] = y_pred
    pl.DataFrame(preds_cols).write_parquet(preds_path)

    # Holdout time-range (None when no time_column).
    time_range: list[str] | None = None
    if cfg.data.time_column is not None and cfg.data.time_column in test.columns:
        tc = test[cfg.data.time_column]
        time_range = [str(tc.min()), str(tc.max())]

    # Build + validate the receipt; atomic JSON write.
    receipt = HoldoutScoreReceipt(
        bundle_version=version,
        bundle_dir=str(bundle_dir),
        manifest_git_sha=manifest.git_sha,
        promoted_from=_PromotedFrom(
            study=manifest.promoted_from.study,
            trial_number=manifest.promoted_from.trial_number,
            metric_value=manifest.promoted_from.metric_value,
        ),
        holdout=_Holdout(
            n_rows=test.height,
            time_range=time_range,
            split_kind=cfg.data.split_kind,
            split_ratios=dict(cfg.data.split_ratios),
        ),
        metrics={cfg.training.metric: float(metric_value)},
        artifacts={
            "predictions": _Artifact(
                path=str(preds_path),
                content_type="application/vnd.apache.parquet",
            ),
        },
        scored_at=_now_iso(),
        scored_at_git_sha=_git_sha(),
    )

    receipt_filename = f"holdout_score_{problem}_{stamp}.json"
    receipt_path = output_dir / receipt_filename
    tmp = receipt_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(receipt.model_dump(mode="json"), indent=2, sort_keys=True) + "\n")
    os.replace(tmp, receipt_path)

    return receipt
