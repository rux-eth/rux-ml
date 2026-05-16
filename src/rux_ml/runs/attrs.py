"""Per-trial ``user_attrs`` schema (per PR-009 + PR-013).

Formalizes the provenance triple from ``docs/CONSTRAINTS.md`` into a Pydantic
model so:
- Writing is atomic and type-checked: ``TrialAttrs.from_cfg(...).record(trial)``
- Reading validates required fields: ``TrialAttrs.from_trial(frozen_trial)``
- PR-010's registry-promotion code can safely assume the schema validates,
  rejecting any trial that fails validation per `CONSTRAINTS.md`'s
  reproducibility rule.

**PR-013 tightening**: ``entropy_hex``, ``image_digest``, ``xgboost_version``,
``cuda_runtime_version``, and ``omp_threads`` flipped from Optional → required.
``gpu_model`` and ``driver_version`` stay Optional (best-effort from
``nvidia-smi``; absent on CPU-only dev hosts where the field is genuinely
``None`` rather than "unknown"). ``best_iteration`` stays Optional —
populated only when XGBoost early stopping fires.

``model_config = extra="ignore"`` on read: trial user_attrs may contain
unrelated keys (e.g., Optuna-internal or user-set debugging attrs); we
silently drop them rather than rejecting valid trials.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from rux_ml._internal.git import git_sha
from rux_ml.config import cfg_hash, layer_cfg_hash

if TYPE_CHECKING:
    import optuna

    from rux_ml._internal.env import EnvironmentVersions
    from rux_ml._internal.seeds import SeedBag
    from rux_ml.config import RuxMLConfig


class TrialAttrs(BaseModel):
    """Per-trial provenance schema written to Optuna ``user_attrs``."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    # Per-layer config hashes (8 layers per PR-007 + PR-015 — the codebase
    # writes more than the 5 enumerated in CONSTRAINTS.md "Per-Trial
    # Provenance Triple"; that doc is the documented minimum).
    data_cfg_hash: str
    features_cfg_hash: str
    training_cfg_hash: str
    tuning_cfg_hash: str
    runs_cfg_hash: str
    registry_cfg_hash: str
    memory_cfg_hash: str
    cv_cfg_hash: str

    # Root config hash (full config, elided per D17).
    root_cfg_hash: str

    # Code version.
    git_sha: str

    # Data provenance (composite + components per D9).
    data_hash: str
    data_bytes_hash: str
    data_logical_hash: str

    # Trial-level metric identity.
    metric: str

    # PR-013: tightened from Optional to required. Every trial records the
    # per-trial entropy fingerprint so PR-010's promote can re-derive the
    # exact same seed bag via `make_seed_bag_from_hex`. Promotion to the
    # registry rejects trials missing any of these.
    entropy_hex: str
    image_digest: str
    xgboost_version: str
    cuda_runtime_version: str
    omp_threads: int

    # PR-011 tightened from Optional → required: every trial records the peak RSS
    # the watchdog observed during the per-trial body.
    peak_rss_mb: float

    # Optional: filled in conditionally (today) or by later PRs.
    best_iteration: int | None = None        # PR-006 writes when early stopping fires

    # PR-013: GPU-only — populated when `nvidia-smi` is available. CPU dev
    # hosts (e.g., Mac) genuinely don't have these so they stay Optional.
    gpu_model: str | None = None
    driver_version: str | None = None

    @classmethod
    def from_cfg(
        cls,
        cfg: RuxMLConfig,
        data_hashes: dict[str, str],
        *,
        metric: str,
        peak_rss_mb: float,
        bag: SeedBag,
        versions: EnvironmentVersions,
        best_iteration: int | None = None,
    ) -> TrialAttrs:
        """Construct a ``TrialAttrs`` from a resolved config + data hashes +
        the per-trial :class:`SeedBag` + the :class:`EnvironmentVersions`
        block.

        Used by both the 1-trial baseline path (``cli/train.py``) and the
        sweep objective (``tuning/objective.py``). ``peak_rss_mb`` is sourced
        from the PR-011 ``Watchdog`` wrapping the trial body; ``bag`` and
        ``versions`` come from PR-013's ``make_seed_bag`` and ``get_versions``.
        """
        return cls(
            data_cfg_hash=layer_cfg_hash(cfg, "data"),
            features_cfg_hash=layer_cfg_hash(cfg, "features"),
            training_cfg_hash=layer_cfg_hash(cfg, "training"),
            tuning_cfg_hash=layer_cfg_hash(cfg, "tuning"),
            runs_cfg_hash=layer_cfg_hash(cfg, "runs"),
            registry_cfg_hash=layer_cfg_hash(cfg, "registry"),
            memory_cfg_hash=layer_cfg_hash(cfg, "memory"),
            cv_cfg_hash=layer_cfg_hash(cfg, "cv"),
            root_cfg_hash=cfg_hash(cfg),
            git_sha=git_sha(),
            data_hash=data_hashes["data_hash"],
            data_bytes_hash=data_hashes["data_bytes_hash"],
            data_logical_hash=data_hashes["data_logical_hash"],
            metric=metric,
            entropy_hex=bag.entropy_hex,
            image_digest=versions.image_digest,
            xgboost_version=versions.xgboost_version,
            cuda_runtime_version=versions.cuda_runtime_version,
            omp_threads=versions.omp_threads,
            peak_rss_mb=peak_rss_mb,
            best_iteration=best_iteration,
            gpu_model=versions.gpu_model,
            driver_version=versions.driver_version,
        )

    def record(self, trial: optuna.Trial) -> None:
        """Write every non-None field to ``trial.user_attrs`` atomically.

        ``exclude_none=True`` so Optional fields that aren't populated
        (``gpu_model`` / ``driver_version`` on CPU-only hosts, ``best_iteration``
        when early stopping didn't fire) don't pollute the trial with ``None``
        literals.
        """
        for key, value in self.model_dump(exclude_none=True).items():
            trial.set_user_attr(key, value)

    @classmethod
    def from_trial(cls, frozen_trial: optuna.trial.FrozenTrial) -> TrialAttrs:
        """Read + validate from a completed Optuna trial.

        Raises ``pydantic.ValidationError`` if any required-now field is
        missing — PR-010's promotion code relies on this to reject trials
        whose provenance is incomplete.
        """
        return cls.model_validate(frozen_trial.user_attrs)
