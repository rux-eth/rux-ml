# PR-007: Optuna basics

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PR** (research-backed at design time).

## Research findings

_To be populated by `PROCEDURE-pr-research.md`._

---

## Scope

Wire up Optuna for HPO per D6 + D16 — sequential trials, in-process, SQLite storage, search space declared in TOML, pruning callback, with the `tune` CLI verb group functional.

- `src/rux_ml/tuning/study.py` — `create_or_load(name, storage, sampler_cfg, pruner_cfg, load_if_exists=True) -> Study`
- `src/rux_ml/tuning/samplers.py` — sampler factory:
  - default `TPESampler(multivariate=True, group=True, n_startup_trials=20)`
  - alternates `GPSampler`, `BoTorchSampler` (HEBO via `BoTorchSampler` integration) configurable per `[tuning]`
- `src/rux_ml/tuning/pruners.py` — pruner factory:
  - default `HyperbandPruner`
  - `WilcoxonPruner` available for k-fold CV objectives (experimental)
- `src/rux_ml/tuning/objective.py`:
  - `walk_search_space(search_space, trial) -> dict[str, Any]` — translates `cfg.search_space` (D16 SearchSpec) into `trial.suggest_*` calls
  - `build_objective(base_cfg) -> Callable[[Trial], float]` returning a closure that:
    1. computes overrides via `walk_search_space`
    2. builds `trial_cfg = base_cfg.model_copy(update=overrides, deep=True)`
    3. records the per-trial `user_attrs` set (config hashes, data hash, git_sha — entropy lands in PR-013)
    4. invokes `make_data` / `make_features` / `make_trainer` / `fit` with `XGBoostPruningCallback(trial, "validation_0-<metric>")`
    5. returns the val metric
- CLI bodies (`src/rux_ml/cli/tune.py`):
  - `tune start <study-name> --n-trials N` → `create_or_load(...)` + `study.optimize(objective, n_trials=N)` (in-process for now)
  - `tune resume <study-name> --n-trials N` → same call (idempotent `load_if_exists=True`)
  - `tune status <study-name>` → print `n_trials_completed`, current best, current best `user_attrs`
  - `tune retry-trial <study-name> <trial-id>` → `study.add_trial(...)` with prior params (or `RetryFailedTrialCallback` registration)
- `configs/studies/example.toml` — illustrative search-space TOML
- Tests:
  - Unit: `walk_search_space` translates each `SearchSpec` variant correctly
  - Integration: tune a dummy XGBoost on synthetic data for 5 trials, assert the study persists in SQLite, best_trial reachable
  - GPU-gated integration: same, with `device="cuda"`
  - Pruning callback: a contrived objective that should be pruned actually does get pruned

NOT in scope: subprocess-per-trial isolation (PR-008), runs query/compare CLI (PR-009), registry promote (PR-010), `peak_rss_mb` (PR-011), `entropy_hex` (PR-013).

## Dependencies

PR-006.

## Architecture section implemented

`docs/ARCHITECTURE.md` → "Tuning" component row, "Decision Rules: Optuna sampler / pruner", "Key Abstractions: `RuxMLConfig` and `SearchSpec`".

## Verification criteria

- [ ] `tune start <name> --n-trials 5` runs to completion against synthetic data
- [ ] `tune resume <name> --n-trials 3` adds 3 more trials (8 total) without losing the original 5
- [ ] `tune status <name>` reports trial counts + best value + best `user_attrs`
- [ ] SQLite DB at `studies/studies.db` is created and queryable via `optuna.load_study`
- [ ] `XGBoostPruningCallback` actually prunes when configured to
- [ ] Search space declared in TOML drives `trial.suggest_*` calls (assert via inspecting trial params)
- [ ] Per-trial `user_attrs` schema present (subset of D17 — full set in PR-009 + PR-013)

## Research backing

Tier 1:

- D6: [Optuna efficient optimization docs](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html), [qfournier 2025](https://qfournier.github.io/blog/2025/xgboost/), [Optuna FAQ on storage](https://optuna.readthedocs.io/en/stable/faq.html)
- D16: [Optuna pruning tutorial](https://optuna.readthedocs.io/en/v2.0.0/tutorial/pruning.html), [Hydra Optuna Sweeper precedent](https://hydra.cc/docs/plugins/optuna_sweeper/), [Pydantic frozen `model_copy`](https://github.com/pydantic/pydantic/discussions/4250)

State assessment must verify Optuna 4.x sampler/pruner/callback APIs and `XGBoostPruningCallback` import path are unchanged.

## Notes

- Run trials in-process here; subprocess isolation is its own PR (PR-008) so we can debug Optuna-specific issues separately from the subprocess machinery.
- Per D6, `WilcoxonPruner` is "experimental" — expose it but mark in TOML comments.
