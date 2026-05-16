"""Sampler factory (per PR-007 Tier-2 research, sub-decisions B1 + C1).

Locked-in defaults:
- ``"tpe"`` (default) — ``TPESampler(multivariate, group, n_startup_trials,
  constant_liar, seed)``. Routes for any XGBoost search space that includes
  categoricals / conditionals (which they all do — `booster`, `grow_policy`,
  conditional regularisation params).
- ``"gp"`` — ``GPSampler(seed)``. Optuna 4.x native (~5x faster than the
  deprecated ``BoTorchSampler`` per the Optuna 3.6 release notes). Best for
  purely-numerical sub-studies up to AutoSampler's hard-coded 250-trial bound.
- ``"hebo"`` — opt-in escalation via ``optunahub.load_module("samplers/hebo")``.
  Raises a clear ``ImportError`` pointing at ``pip install optunahub hebo`` if
  the optional deps aren't installed (sub-decision C1).

``BoTorchSampler`` is intentionally not on the menu — Optuna 3.6 deprecated it
for single-objective HPO; no cited tabular-GBM advantage justifies the slower
compute + extra dep cost.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optuna.samplers import BaseSampler, GPSampler, TPESampler

if TYPE_CHECKING:
    from rux_ml.config import TuningConfig

_HEBO_INSTALL_HINT = (
    "HEBO sampler requires the optional 'optunahub' + 'hebo' deps. Install via:\n"
    "    pip install optunahub hebo\n"
    "or switch `tuning.sampler` to 'tpe' (default) or 'gp'."
)


def _make_tpe(cfg: TuningConfig, *, seed: int | None) -> TPESampler:
    return TPESampler(
        multivariate=cfg.multivariate,
        group=cfg.group,
        n_startup_trials=cfg.n_startup_trials,
        constant_liar=cfg.constant_liar,
        seed=seed,
    )


def _make_gp(seed: int | None) -> GPSampler:
    return GPSampler(seed=seed)


def _make_hebo(seed: int | None) -> BaseSampler:
    # `optunahub` and `hebo` are not in default runtime deps (sub-decision C1).
    # Both imports are deliberate lazy-with-fallback; basedpyright pragmas suppress
    # the missing-import warning when the optional packages aren't installed.
    try:
        import optunahub  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise ImportError(_HEBO_INSTALL_HINT) from exc
    try:
        # optunahub.load_module fetches the HEBO sampler package from the registry.
        module = optunahub.load_module("samplers/hebo")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    except Exception as exc:
        raise ImportError(_HEBO_INSTALL_HINT) from exc
    return module.HEBOSampler(seed=seed)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportReturnType]


def make_sampler(cfg: TuningConfig, *, seed: int | None = None) -> BaseSampler:
    """Return the configured Optuna sampler.

    ``seed`` flows into each sampler's ``random_state`` / ``seed`` argument for
    reproducibility. PR-013's ``SeedSequence(entropy).spawn()`` will pipe a
    derived ``sampler_seed`` here unchanged.
    """
    kind = cfg.sampler
    if kind == "tpe":
        return _make_tpe(cfg, seed=seed)
    if kind == "gp":
        return _make_gp(seed)
    if kind == "hebo":
        return _make_hebo(seed)
    msg = f"unknown sampler kind {kind!r}"  # pragma: no cover — Literal type narrows this
    raise ValueError(msg)
