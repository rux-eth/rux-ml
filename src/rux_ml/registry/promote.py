"""Registry promotion (per D8 + PR-010 sub-decision A1 — re-fit at promote time).

``promote()`` is the entry point. The flow:

1. Load the originating trial via ``runs.load_run``.
2. Validate ``TrialAttrs.from_trial(frozen)`` (PR-009 explicit hand-off) —
   raises ``pydantic.ValidationError`` if any required-now provenance field
   is missing; promote refuses.
3. Apply ``trial.params`` overrides to ``base_cfg`` → fresh ``trial_cfg``.
4. Re-fit the features pipeline + trainer on the trial's data (final-fit
   reproduction; no CV folds, no per-fold reporting).
5. Compose ``ModelManifest`` from trial attrs + library versions +
   feature_list_hash.
6. Save bundle (``pipeline.skops`` + ``model.ubj`` + ``manifest.json``) to
   ``registry/<problem>/<version>/``.
7. Atomically rewrite ``registry/<problem>/champion.json``.

This module is NOT exported from ``rux_ml.registry``'s public ``__init__.py``
(sub-decision B1 — strict inference-deps separation). CLI promote imports
it directly via ``from rux_ml.registry.promote import promote``.
"""

from __future__ import annotations

import importlib.metadata as _metadata
from typing import TYPE_CHECKING, Any, cast

from rux_ml._internal.hashing import sha256_canonical
from rux_ml._internal.seeds import SeedBag, make_seed_bag_from_hex
from rux_ml.config import RuxMLConfig
from rux_ml.data import load_parquet, make_splits, materialize
from rux_ml.features import cardinalities_from, make_features
from rux_ml.registry.bundle import save_bundle
from rux_ml.registry.champion import write_champion
from rux_ml.registry.manifest import (
    LibraryVersions,
    ModelManifest,
    PromotedFrom,
)
from rux_ml.registry.paths import champion_path, format_version_id, version_dir
from rux_ml.runs import TrialAttrs, load_run
from rux_ml.runs.provenance import data_hashes
from rux_ml.training import make_trainer

if TYPE_CHECKING:
    import polars as pl
    import xgboost as xgb
    from sklearn.pipeline import Pipeline


def _unflatten(flat: dict[str, Any]) -> dict[str, Any]:
    """Convert ``{'a.b': 1}`` to ``{'a': {'b': 1}}``."""
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


def _apply_trial_params(base_cfg: RuxMLConfig, params: dict[str, Any]) -> RuxMLConfig:
    """Rebuild ``RuxMLConfig`` with the trial's search-space draws merged in.

    Mirrors ``tuning.objective._apply_overrides`` but kept local so the
    registry module doesn't reach into ``tuning/`` privates.
    """
    base_dict = base_cfg.model_dump()
    merged = _deep_merge(base_dict, _unflatten(params))
    return RuxMLConfig.model_validate(merged)


def _refit(cfg: RuxMLConfig, *, bag: SeedBag) -> tuple[Pipeline, xgb.Booster]:
    """Final-fit reproduction of the trial's pipeline + booster.

    Uses the train fold for fitting and the val fold for XGBoost-internal
    early stopping (matches ``cli/train.py``'s 1-trial fit shape). The test
    fold is unused — held out for future golden-regression evaluation (PR-014).

    ``bag`` (PR-013) carries the originating trial's ``split_seed`` +
    ``xgb_seed`` reconstructed from its ``entropy_hex``. Without this, the
    promotion re-fit would use different seeds than the trial's evaluation,
    so the registered bundle would not match the metrics recorded in
    ``user_attrs`` — silent reproducibility breakage.
    """
    if cfg.data.source_path is None or cfg.data.target_column is None:
        msg = "promote requires data.source_path and data.target_column"
        raise ValueError(msg)

    df = materialize(load_parquet(cfg.data.source_path))
    splits = make_splits(cfg, df, seed=bag.split_seed)
    x_train = splits["train"].drop(cfg.data.target_column)
    y_train = splits["train"][cfg.data.target_column]
    x_val = splits["val"].drop(cfg.data.target_column)
    y_val = splits["val"][cfg.data.target_column]

    cards = cardinalities_from(x_train, cfg.features.spec.categorical_columns)
    pipeline = make_features(cfg.features, cardinalities=cards if cards else None)
    pipeline.fit(x_train, y_train.to_numpy())  # pyright: ignore[reportUnknownMemberType]
    x_train_t = cast("pl.DataFrame", pipeline.transform(x_train))  # pyright: ignore[reportUnknownMemberType]
    x_val_t = cast("pl.DataFrame", pipeline.transform(x_val))  # pyright: ignore[reportUnknownMemberType]

    trainer = make_trainer(cfg.training, seed=bag.xgb_seed)
    trainer.fit(
        x_train_t.to_pandas(),
        y_train.to_numpy(),
        eval_set=[(x_val_t.to_pandas(), y_val.to_numpy())],
        verbose=False,
    )
    booster = cast(
        "xgb.Booster",
        trainer.get_booster(),  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
    )
    return pipeline, booster


def _library_versions() -> LibraryVersions:
    """Snapshot installed library versions at promote time."""
    return LibraryVersions(
        xgboost=_metadata.version("xgboost"),
        skops=_metadata.version("skops"),
        scikit_learn=_metadata.version("scikit-learn"),
        polars=_metadata.version("polars"),
        numpy=_metadata.version("numpy"),
        rux_ml=_metadata.version("rux-ml"),
    )


def _feature_list_hash(cfg: RuxMLConfig) -> str:
    """SHA-256 of the sorted feature-column list (numeric + categorical)."""
    columns = sorted(
        [*cfg.features.spec.numeric_columns, *cfg.features.spec.categorical_columns]
    )
    return sha256_canonical(columns)


def _build_manifest(
    cfg: RuxMLConfig,
    attrs: TrialAttrs,
    *,
    problem: str,
    version: str,
    study_name: str,
    trial_number: int,
    metric_value: float,
) -> ModelManifest:
    from rux_ml.registry.champion import now_iso  # noqa: PLC0415 — same-package helper

    if cfg.data.source_path is None:
        msg = "promote requires data.source_path"
        raise ValueError(msg)
    hashes = data_hashes(cfg.data.source_path)
    return ModelManifest(
        version=version,
        problem=problem,
        promoted_from=PromotedFrom(
            study=study_name,
            trial_number=trial_number,
            metric_value=metric_value,
        ),
        metric_name=attrs.metric,
        metric_value=metric_value,
        data_cfg_hash=attrs.data_cfg_hash,
        features_cfg_hash=attrs.features_cfg_hash,
        training_cfg_hash=attrs.training_cfg_hash,
        tuning_cfg_hash=attrs.tuning_cfg_hash,
        runs_cfg_hash=attrs.runs_cfg_hash,
        registry_cfg_hash=attrs.registry_cfg_hash,
        memory_cfg_hash=attrs.memory_cfg_hash,
        cv_cfg_hash=attrs.cv_cfg_hash,
        root_cfg_hash=attrs.root_cfg_hash,
        git_sha=attrs.git_sha,
        data_hash=hashes["data_hash"],
        data_bytes_hash=hashes["data_bytes_hash"],
        data_logical_hash=hashes["data_logical_hash"],
        library_versions=_library_versions(),
        feature_list_hash=_feature_list_hash(cfg),
        created_at=now_iso(),
    )


def promote(
    base_cfg: RuxMLConfig,
    *,
    problem: str,
    study_name: str,
    trial_number: int,
) -> str:
    """Promote a trial to a new registry version.

    Args:
        base_cfg: Resolved ``RuxMLConfig`` (loaded by the caller from the same
            TOML stack the trial was run with).
        problem: Problem identifier — the per-problem subtree in the registry.
        study_name: Optuna study identifier the trial belongs to.
        trial_number: ``trial.number`` within the study (PR-007 / PR-009 convention).

    Returns the newly-assigned ``version_id``.

    Raises:
        pydantic.ValidationError: if the trial's ``user_attrs`` fail
            ``TrialAttrs.from_trial`` validation (PR-009 promotion-gate).
        KeyError: if the study or trial doesn't exist in storage.
    """
    # 1. Load the trial + 2. validate provenance.
    run = load_run(base_cfg.runs.storage_url, study_name, trial_number)
    if run.attrs is None:
        # ``load_run`` swallows ValidationError → attrs=None; re-raise loudly here.
        # Re-invoke the strict path so we get a typed error chain for promote.
        import optuna  # noqa: PLC0415

        study = optuna.load_study(study_name=study_name, storage=base_cfg.runs.storage_url)
        frozen = next(t for t in study.trials if t.number == trial_number)
        TrialAttrs.from_trial(frozen)  # raises pydantic.ValidationError

    attrs = run.attrs
    assert attrs is not None  # narrowed above

    # 3. Apply trial.params to base_cfg.
    trial_cfg = _apply_trial_params(base_cfg, run.params)

    # 4. Re-fit pipeline + booster — PR-013 reconstructs the original trial's
    # SeedBag from its recorded ``entropy_hex`` so the promoted bundle uses
    # the SAME split + xgb_seed the trial reported metrics for.
    bag = make_seed_bag_from_hex(attrs.entropy_hex)
    pipeline, booster = _refit(trial_cfg, bag=bag)

    # 5. Compose manifest.
    metric_value = run.value if run.value is not None else 0.0
    version = format_version_id(attrs.root_cfg_hash, fmt=trial_cfg.registry.version_format)
    manifest = _build_manifest(
        trial_cfg,
        attrs,
        problem=problem,
        version=version,
        study_name=study_name,
        trial_number=trial_number,
        metric_value=metric_value,
    )

    # 6. Save bundle.
    dest = version_dir(trial_cfg.registry.root, problem, version)
    save_bundle(pipeline, booster, manifest, dest)

    # 7. Atomically rewrite champion.json.
    write_champion(
        champion_path(trial_cfg.registry.root, problem),
        version=version,
        promoted_from_study=study_name,
        promoted_from_trial_number=trial_number,
        metric_value=metric_value,
    )
    return version


def rollback(
    base_cfg: RuxMLConfig, *, problem: str, version: str
) -> None:
    """Atomically point ``champion.json`` at a prior ``version``.

    Raises:
        FileNotFoundError: if the target ``<version>/`` directory or its
            ``manifest.json`` doesn't exist.
    """
    target_dir = version_dir(base_cfg.registry.root, problem, version)
    manifest_path = target_dir / "manifest.json"
    if not manifest_path.exists():
        msg = (
            f"can't rollback {problem!r} to {version!r}: no manifest at {manifest_path}. "
            f"List existing versions via `rux-ml registry list`."
        )
        raise FileNotFoundError(msg)
    from rux_ml.registry.manifest import read as read_manifest  # noqa: PLC0415

    manifest = read_manifest(manifest_path)
    write_champion(
        champion_path(base_cfg.registry.root, problem),
        version=version,
        promoted_from_study=manifest.promoted_from.study,
        promoted_from_trial_number=manifest.promoted_from.trial_number,
        metric_value=manifest.metric_value,
    )
