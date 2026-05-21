"""Functional objective for the Optuna HPO loop.

Two public functions:

- :func:`walk_search_space` — translates ``cfg.search_space`` (the PR-002
  ``SearchSpec`` tagged union) into ``trial.suggest_*`` calls and returns a
  flat dot-path dict of overrides (e.g. ``{"training.learning_rate": 0.05}``).
- :func:`build_objective` — closes over ``base_cfg`` and returns the
  ``Callable[[Trial], float]`` that ``study.optimize`` consumes. The objective
  is **K-fold CV-mean** per the PR-007 Tier-2 research:
  1. ``walk_search_space`` for overrides -> deep-merged into ``base_cfg`` ->
     ``RuxMLConfig.model_validate`` -> fresh ``trial_cfg``.
  2. Per-trial ``user_attrs`` recorded via ``TrialAttrs.from_cfg(...).record(trial)``.
  3. Data loaded once; Splitter built via PR-015's ``make_splitter`` with
     ``seed=trial_cfg.tuning.entropy`` (PR-013 will spawn a proper ``cv_seed``).
  4. ExtMem x Splitter compat checked once at the top -- incompatible pairings
     raise ``NotImplementedError`` (the PR-015-deferred C1 gate executing here).
  5. For each fold: features pipeline fit + transform, trainer fit with
     XGBoost-internal ``early_stopping_rounds`` per fold, score -> ``trial.report``
     for WilcoxonPruner consumption, ``trial.should_prune`` check.
  6. Returns ``statistics.fmean(fold_scores)`` to ``study.tell``.

**No** ``XGBoostPruningCallback`` is wired inside the CV loop (Optuna #3203
incompatibility -- duplicate ``step=0,1,...`` reports across folds break
iteration-level pruners). The integration callback is reserved for a future
single-fit objective regime.
"""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING, Any, cast

import optuna
import polars as pl
from xgboost import ExtMemQuantileDMatrix

from rux_ml._internal.env import EnvironmentVersions, get_versions
from rux_ml._internal.memory import MemoryPressureError, Watchdog
from rux_ml._internal.seeds import SeedBag, make_seed_bag
from rux_ml.config import (
    CatSpec,
    FloatSpec,
    GroupKFoldCV,
    IntSpec,
    RuxMLConfig,
)
from rux_ml.data import (
    load_parquet,
    make_splits,
    make_splitter,
    materialize,
)
from rux_ml.features import cardinalities_from, make_features
from rux_ml.runs import TrialAttrs, data_hashes
from rux_ml.training import (
    compute_score,
    estimate_x_bytes,
    make_trainer,
    select_ingest,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from rux_ml.config import SearchSpec


# ---------- Search space walker ----------


def walk_search_space(
    search_space: dict[str, SearchSpec], trial: optuna.Trial
) -> dict[str, Any]:
    """Translate each ``SearchSpec`` entry into a ``trial.suggest_*`` call.

    Keys are dot-paths into the ``RuxMLConfig`` schema (e.g.
    ``"training.learning_rate"``). The returned dict is later deep-merged with
    the base config via ``_apply_overrides``.
    """
    overrides: dict[str, Any] = {}
    for name, spec in search_space.items():
        if isinstance(spec, FloatSpec):
            overrides[name] = trial.suggest_float(name, spec.low, spec.high, log=spec.log)
        elif isinstance(spec, IntSpec):
            overrides[name] = trial.suggest_int(name, spec.low, spec.high, log=spec.log)
        else:
            # CatSpec is the only remaining variant in the tagged union; the
            # discriminator narrows here at the type level.
            assert isinstance(spec, CatSpec)
            overrides[name] = trial.suggest_categorical(name, spec.choices)
    return overrides


# ---------- Trial config derivation ----------


def _unflatten(flat: dict[str, Any]) -> dict[str, Any]:
    """Convert ``{'a.b': 1}`` to ``{'a': {'b': 1}}`` for deep-merge."""
    nested: dict[str, Any] = {}
    for key, value in flat.items():
        parts = key.split(".")
        cursor: dict[str, Any] = nested
        for part in parts[:-1]:
            sub: Any = cursor.setdefault(part, {})
            if not isinstance(sub, dict):
                msg = f"override path conflict at {part!r} in {key!r}"
                raise ValueError(msg)
            cursor = cast("dict[str, Any]", sub)
        cursor[parts[-1]] = value
    return nested


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge -- overlay wins on scalar collision."""
    out: dict[str, Any] = dict(base)
    for key, value in overlay.items():
        existing = out.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            out[key] = _deep_merge(
                cast("dict[str, Any]", existing), cast("dict[str, Any]", value)
            )
        else:
            out[key] = value
    return out


def _apply_overrides(base_cfg: RuxMLConfig, overrides: dict[str, Any]) -> RuxMLConfig:
    """Build a fresh ``RuxMLConfig`` with ``overrides`` deep-merged in."""
    base_dict = base_cfg.model_dump()
    merged = _deep_merge(base_dict, _unflatten(overrides))
    return RuxMLConfig.model_validate(merged)


# ---------- Objective ----------


def _require(cfg: RuxMLConfig) -> tuple[Any, str]:
    """Validate that the data layer has the inputs the objective needs."""
    if cfg.data.source_path is None or cfg.data.target_column is None:
        msg = (
            "rux-ml tune requires data.source_path and data.target_column to be set "
            "(via configs/problems/<problem>.toml)"
        )
        raise ValueError(msg)
    return cfg.data.source_path, cfg.data.target_column


def _strip_target(df: pl.DataFrame, target_col: str) -> tuple[pl.DataFrame, pl.Series]:
    return df.drop(target_col), df[target_col]


def _carve_substrate(
    base_cfg: RuxMLConfig, df_full: pl.DataFrame, target_col: str
) -> tuple[pl.DataFrame, pl.Series]:
    """Carve the HPO CV substrate (``splits["train"] + splits["val"]``) per PR-031.

    The held-out test fold (``splits["test"]``, ~15% of source data by default
    ``data.split_ratios``) is excluded from HPO. Carve is done once with the
    study-level bag's ``split_seed`` (``trial_number=0`` sentinel — same
    precedent as the sampler seed at ``cli/tune.py:73``) so all trials in a
    study share the identical substrate; deterministic per study identity for
    both ``random`` and ``time_ordered`` ``data.split_kind``.

    Convention citation (Phase 3 research): AutoGluon ``tabular-essentials``
    tutorial at SHA ``f8c428cbbef3bc319ff3f7710f5900e65637f4c4`` (``fit(train_data)
    → evaluate(test_data)``); sklearn user guide §3.1; Optuna issue #2184
    (in-objective splitting acknowledged as anti-pattern — move the carve
    outside the closure).
    """
    study_bag = make_seed_bag(master_entropy=base_cfg.tuning.entropy, trial_number=0)
    splits = make_splits(base_cfg, df_full, seed=study_bag.split_seed)
    df_substrate = pl.concat([splits["train"], splits["val"]])
    return _strip_target(df_substrate, target_col)


def _check_extmem_compat(
    x_full: pl.DataFrame, splitter: Any, cfg: RuxMLConfig
) -> None:
    """Raise ``NotImplementedError`` for the incompatible ExtMem-Splitter pairing.

    Implements the PR-015 sub-decision C1 gate at its execution site. The
    materialised-fallback path is deferred to a follow-up PR.
    """
    dmatrix_cls = select_ingest(estimate_x_bytes(x_full), cfg.data)
    if dmatrix_cls is ExtMemQuantileDMatrix and not splitter.extmem_compatible:
        msg = (
            f"ExtMemQuantileDMatrix ingest path is active (X exceeds "
            f"data.gpu_in_memory_x_gb_max={cfg.data.gpu_in_memory_x_gb_max} GB) but the "
            f"selected Splitter ({type(splitter).__name__}) is not extmem-compatible. "
            f"Either switch cv.kind to 'time_series' (the only ExtMem-compatible Splitter at "
            f"v0), or wait for the materialised-fallback follow-up PR. See PR-015 sub-decision "
            f"C1 and prs/PR-015-cv-strategy.md."
        )
        raise NotImplementedError(msg)


def _fold_scores(
    cfg: RuxMLConfig,
    x_full: pl.DataFrame,
    y_full: pl.Series,
    splitter: Any,
    groups: Any,
    trial: optuna.Trial,
    bag: SeedBag,
) -> list[float]:
    """Iterate folds, fit per-fold, score, report, and respect pruning.

    ``bag.xgb_seed`` (PR-013) plumbs into ``make_trainer`` so every fold's
    XGBoost RNG is pinned. ``bag.cv_seed`` already lives in the ``splitter``
    constructed upstream.
    """
    scores: list[float] = []
    for fold_idx, (train_idx, test_idx) in enumerate(
        splitter.split(x_full, y_full, groups=groups)
    ):
        x_tr = x_full[train_idx.tolist()]
        x_te = x_full[test_idx.tolist()]
        y_tr = y_full[train_idx.tolist()]
        y_te = y_full[test_idx.tolist()]

        # Features pipeline fit per fold for leakage hygiene (cardinalities re-computed on train).
        cards = cardinalities_from(x_tr, cfg.features.spec.categorical_columns)
        pipeline = make_features(cfg.features, cardinalities=cards if cards else None)
        pipeline.fit(x_tr, y_tr.to_numpy())  # pyright: ignore[reportUnknownMemberType]
        # sklearn Pipeline.transform stubs return Unknown; PR-005's terminal `to_polars`
        # step guarantees a Polars DataFrame at the boundary.
        x_tr_t = cast("pl.DataFrame", pipeline.transform(x_tr))  # pyright: ignore[reportUnknownMemberType]
        x_te_t = cast("pl.DataFrame", pipeline.transform(x_te))  # pyright: ignore[reportUnknownMemberType]

        # Trainer fit per fold. XGBoost-internal early_stopping_rounds runs against the
        # held-out fold val (= test fold here). No XGBoostPruningCallback (Optuna #3203).
        trainer = make_trainer(cfg.training, seed=bag.xgb_seed)
        x_tr_pd = x_tr_t.to_pandas()
        x_te_pd = x_te_t.to_pandas()
        trainer.fit(
            x_tr_pd,
            y_tr.to_numpy(),
            eval_set=[(x_te_pd, y_te.to_numpy())],
            verbose=False,
        )
        fold_score = compute_score(cfg.training.metric, trainer, x_te_pd, y_te.to_numpy())
        scores.append(fold_score)

        # Feed WilcoxonPruner: report per-fold scores; check if the trial should prune.
        trial.report(fold_score, step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned
    return scores


def _record_attrs(
    trial: optuna.Trial,
    trial_cfg: RuxMLConfig,
    hashes: dict[str, str],
    peak_rss_mb: float,
    bag: SeedBag,
    versions: EnvironmentVersions,
) -> None:
    """Record the per-trial provenance triple including PR-013's seed + version block."""
    TrialAttrs.from_cfg(
        trial_cfg,
        hashes,
        metric=trial_cfg.training.metric,
        peak_rss_mb=peak_rss_mb,
        bag=bag,
        versions=versions,
    ).record(trial)


def build_objective(base_cfg: RuxMLConfig) -> Callable[[optuna.Trial], float]:
    """Return the closure ``study.optimize`` consumes.

    The closure is K-fold CV-mean: each trial does ``cfg.cv.n_splits`` fits and
    returns ``statistics.fmean(fold_scores)``. Per-fold scores are reported to
    Optuna so WilcoxonPruner / MedianPruner can paired-test against running
    trials and prune dominated ones early.
    """
    source_path, target_col = _require(base_cfg)

    # Data is loaded once outside the closure so every trial shares the same in-memory copy.
    # (Per D6 sequential trials, the closure is only ever called serially.)
    df_full = materialize(load_parquet(source_path))
    # PR-031: HPO CV substrate is splits["train"] + splits["val"]; the test fold
    # is truly held out from HP search. See _carve_substrate for the convention
    # cite + study-level seed contract.
    x_substrate, y_substrate = _carve_substrate(base_cfg, df_full, target_col)
    hashes = data_hashes(source_path)

    # PR-013: capture environment versions once per subprocess. Each subprocess
    # runs n_trials=1 (per PR-008), so this is effectively per-trial — but
    # placing it outside the closure makes the cost explicit: one xgboost import
    # + one pair of nvidia-smi invocations per subprocess.
    versions = get_versions(base_cfg.memory)

    def objective(trial: optuna.Trial) -> float:
        overrides = walk_search_space(base_cfg.search_space, trial)
        trial_cfg = _apply_overrides(base_cfg, overrides)

        # PR-013: derive a per-trial SeedBag from the study master + trial.number.
        # ``trial.number`` is stable per study; identical (master, number) pairs
        # reproduce the same bag.
        bag = make_seed_bag(
            master_entropy=trial_cfg.tuning.entropy,
            trial_number=trial.number,
        )

        # Build the Splitter per PR-015 using the per-trial cv seed (was the master
        # entropy in PR-007; PR-013 spawns a distinct cv_seed per trial).
        splitter = make_splitter(trial_cfg.cv, seed=bag.cv_seed)
        groups = None
        if isinstance(trial_cfg.cv, GroupKFoldCV):
            groups_column = trial_cfg.cv.groups_column
            if groups_column not in x_substrate.columns:
                msg = (
                    f"GroupKFoldCV.groups_column={groups_column!r} not found in input "
                    f"DataFrame columns: {x_substrate.columns}"
                )
                raise ValueError(msg)
            groups = x_substrate[groups_column].to_numpy()

        # ExtMem-incompatible Splitter gate (PR-015 sub-decision C1).
        _check_extmem_compat(x_substrate, splitter, trial_cfg)

        # Watchdog wraps the per-trial fit work (PR-011). Observational +
        # post-fit-check: a tripped watchdog converts to optuna.TrialPruned.
        # Attrs are recorded in a `finally` so peak_rss_mb lands on pruned
        # AND completed trials.
        with Watchdog(
            threshold_gb=trial_cfg.memory.watchdog_threshold_gb,
            sample_hz=trial_cfg.memory.watchdog_sample_hz,
        ) as wd:
            try:
                scores = _fold_scores(
                    trial_cfg, x_substrate, y_substrate, splitter, groups, trial, bag
                )
            except MemoryPressureError:
                raise optuna.TrialPruned from None
            finally:
                _record_attrs(trial, trial_cfg, hashes, wd.peak_mb, bag, versions)

        if wd.tripped:
            raise optuna.TrialPruned
        return statistics.fmean(scores)

    return objective
