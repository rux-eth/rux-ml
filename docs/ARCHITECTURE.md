# Architecture

`rux-ml` is a personal ML research workbench for the full lifecycle of tabular gradient-boosted models — primarily XGBoost, with a contract that admits LightGBM, CatBoost, and other sklearn-compatible models later. It is operated via a single CLI (`rux-ml`) and a Python package (`rux_ml`); there is no web UI and no long-running server.

This document describes the architecture as designed across the 17 decisions logged in `docs/0.0/DESIGN-log.md` plus v0.1 amendments in `docs/0.1/DESIGN-log.md`. Hard rules live in `docs/CONSTRAINTS.md`; soft patterns in `docs/CONVENTIONS.md`. <!-- rewrite-doc-refs:skip-line -->

---

## System Overview

The workbench supports the end-to-end loop **data → features → training → tuning → run logging → registry promotion**, with reproducibility (seed, config, data, code, environment) recorded per trial. Hyperparameter sweeps (Optuna) and one-off baseline trainings are first-class peers — both are recorded as Optuna trials in a shared SQLite study, with artifacts content-addressed via `optuna.artifacts`. Promoted models are bundled to a filesystem registry (`pipeline.skops` + `model.ubj` + Pydantic-validated manifest) and selected via an atomically-rewritten `champion.json` per problem.

The host is a single Linux desktop with an RTX 4090 (24 GB VRAM), i9-13900K (24 threads), and 36 GB DDR5 RAM. **System RAM is the binding constraint.** XGBoost runs GPU-first via `device="cuda"`; data ingestion auto-selects between `QuantileDMatrix` (when X fits comfortably in VRAM) and `ExtMemQuantileDMatrix` with host-RAM caching (when larger). HPO trials run **sequentially** with `n_jobs=1` because XGBoost-internal GPU parallelism saturates the 4090; concurrent trials cause OOM.

---

## Components

The Python package is organized into layered subpackages. Dependencies flow downward only (see `docs/CONVENTIONS.md`).

| Layer | Module | Responsibility |
|---|---|---|
| **CLI** | `src/rux_ml/cli/` | Typer-backed entry points — `data`, `train`, `tune`, `runs`, `registry` verb groups |
| **Config** | `src/rux_ml/config/` | Pydantic-settings models per layer, composed into `RuxMLConfig`; loads from layered TOML + env + CLI overrides |
| **Data** | `src/rux_ml/data/` | Polars/Parquet loaders, train/val/test splits, content-addressed dataset versioning (composite hash + manifest + CAS), XGBoost `DataIter` for `ExtMemQuantileDMatrix`, `Splitter` Protocol + concrete strategies (per PR-015) |
| **Features** | `src/rux_ml/features/` | sklearn `Pipeline`+`ColumnTransformer` orchestrator; Polars-expression stateless transforms wrapped in `FunctionTransformer`; `category_encoders` `NestedCVWrapper` for high-card categoricals |
| **Training** | `src/rux_ml/training/` | `Trainer` `typing.Protocol` (sklearn API: `fit`/`predict`/`predict_proba`/`best_iteration_`); model factory (`XGBClassifier`, etc.); metric registry |
| **Tuning** | `src/rux_ml/tuning/` | Optuna study orchestration; `objective(trial, base_cfg)`; `SearchSpec`-walker; samplers (TPE default) and pruners (Hyperband default); subprocess-per-trial spawn |
| **Runs** | `src/rux_ml/runs/` | Optuna-as-experiment-log read API; canonical `user_attrs` schema (Pydantic-validated `TrialAttrs` per PR-009); `one_off_run` context manager wrapping `study.ask`/`study.tell`; query helpers (`list_runs`, `load_run`, `compare_runs` returning Polars / Pydantic objects) |
| **Registry** | `src/rux_ml/registry/` | Model bundle write/read (`pipeline.skops` + `model.ubj` + Pydantic-validated `manifest.json`); promotion logic with atomic `champion.json` rewrite via tmp + `os.replace`; thin `load_model(problem, version="champion")` API with strict inference-deps separation (no transitive training-stack imports — verified by subprocess test) |
| **Internal** | `src/rux_ml/_internal/` | Logging, hashing helpers (`xxhash`/`blake3`), env (OMP/BLAS pinning, `WORKBENCH_HOME` resolution), `SeedSequence` + `.spawn()`, `psutil` memory watchdog raising `MemoryPressureError` |

`src/rux_ml/_internal/trial_runner.py` is the entry point invoked by `subprocess.run` per trial — see Data Flow below.

The repository skeleton is in `docs/CONVENTIONS.md`; the Python package internal structure is laid out in detail in D15 of `docs/0.0/DESIGN-log.md`. <!-- rewrite-doc-refs:skip-line -->

---

## Data Flow

End-to-end shape from a CLI invocation through to a promoted registry artifact. The parent process never initializes CUDA — that's the child's job — so the spawn boundary is also the CUDA-init boundary.

```
CLI: rux-ml tune start \
     --problem churn_v1 --study churn_xgb_wide --n-trials 50

  PARENT PROCESS (no CUDA init)
  cli.tune.start():
    ├─ load RuxMLConfig via pydantic-settings:
    │     base.toml → problems/churn_v1.toml → studies/churn_xgb_wide.toml
    │     overlaid by env vars (RUXML_*) and CLI dot-path overrides
    ├─ resolve WORKBENCH_HOME, ensure studies/ + registry/ + data/ exist
    ├─ tuning.study.create_or_load(
    │     name="churn_xgb_wide_<study_id>",
    │     storage="sqlite:///studies/studies.db",
    │     load_if_exists=True)
    └─ for i in range(n_trials=50):           # parent doesn't use study.optimize;
         subprocess.run([                       # see PR-008 sub-decision A1
           sys.executable, "-m", "rux_ml._internal.trial_runner",
           "--config", toml_path,
           "--problem", problem,                # passes through CLI --problem
           "--study", study_layer,              # passes through CLI --study (config overlay)
           "--study-name", optuna_study_name,
           "--overrides-json", tmp_json_path,   # CLI --set overrides serialised here
         ], check=False, timeout=cfg.tuning.trial_timeout_s)
         # Child runs its own study.optimize(..., n_trials=1) and writes to storage.
         # SQLite coordinates state across children (D6 sequential trials).
         ▼
  CHILD PROCESS (fresh interpreter — spawn semantics; CUDA init OK)
  rux_ml._internal.trial_runner.main():
    ├─ argparse (stdlib only — no heavy imports yet)
    ├─ overrides = json.loads(overrides_json)
    ├─ base_cfg = RuxMLConfig.from_layers(...)  (light: pydantic + tomllib)
    ├─ pin env vars from cfg.memory BEFORE numpy/polars/sklearn/xgboost import:
    │     OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS, POLARS_MAX_THREADS
    ├─ lazy-import: rux_ml.tuning + rux_ml.training + heavy libs
    ├─ study = tuning.create_or_load(name, storage, sampler, pruner, direction, load_if_exists=True)
    ├─ study.optimize(build_objective(cfg), n_trials=1)
    │   # build_objective (PR-007) runs K-fold CV-mean per cfg.cv:
    │   #   walk_search_space → overrides → trial_cfg = RuxMLConfig.model_validate(deep-merged)
    │   #   wrap per-trial body in Watchdog (PR-011): 1Hz psutil RSS sampler + post-fit check
    │   #   record 8+ layer user_attrs via TrialAttrs.from_cfg().record() (includes peak_rss_mb)
    │   #   make_splitter(cfg.cv, seed=cfg.tuning.entropy); ExtMem-compat gate
    │   #   for fold in folds: fit features + trainer, score, trial.report(score, fold_idx)
    │   #   if trial.should_prune(): raise optuna.TrialPruned
    │   #   return statistics.fmean(fold_scores) → study.tell internally
    └─ exit cleanly (psutil trip in PR-011 → MemoryPressureError → optuna.TrialPruned)

After study completes — promotion is an explicit step:

  CLI: rux-ml registry promote \
       --problem churn_v1 --study churn_xgb_wide_<study_id> --trial <best>

  cli.registry.promote() (per PR-010 sub-decision A1 — re-fit at promote):
    ├─ load trial from study storage via runs.load_run
    ├─ validate TrialAttrs.from_trial(frozen)
    │     raises pydantic.ValidationError if provenance is incomplete →
    │     promotion REFUSED (CONSTRAINTS.md reproducibility rule)
    ├─ apply trial.params overrides to base RuxMLConfig → trial_cfg
    ├─ re-fit final (pipeline, booster) on train+val (no CV folds, no
    │     per-fold reporting — just a clean final fit)
    ├─ compose ModelManifest from TrialAttrs + library versions +
    │     feature_list_hash
    ├─ write registry/<problem>/<version>/{pipeline.skops, model.ubj, manifest.json}
    │     where <version> = v_<YYYY>_<MM>_<DD>_<short_hash>
    └─ atomic rewrite registry/<problem>/champion.json (tmp + os.replace)
```

For a one-off (non-sweep) baseline training, `cli.train.run()` follows the same path but wraps the call as a single trial via `study.ask()` + `study.tell()` — sweep and one-off runs share the same store and the same provenance schema.

---

## Decision Rules (codified in the workbench)

These are runtime branches the workbench must auto-select, based on configuration and observed inputs.

### CV strategy by data shape (per PR-023 D1/D2)

The Splitter chosen for a problem is driven by the **shape of `(time, group)` in the input data**, not by the trainer family. The decision tree:

1. **No temporal ordering** → `KFoldCV` (default) or `StratifiedKFoldCV` (imbalanced classification) or `GroupKFoldCV` (entity-leakage-prone). Standard sklearn semantics.
2. **Single-asset time series** (1 row per timestamp; ordered) → `TimeSeriesSplitCV` with `gap = label_horizon` (in row units), OR `CombinatorialPurgedCV` for overlapping-label / variable-horizon problems with `embargo_pct ∈ [0.005, 0.02]` per AFML §7.4.2. Here row-count and time-unit semantics coincide.
3. **Stacked panel** (many assets per timestamp; per-asset forward labels) → **panel-aware** path is required. Two routes:
   - Simple walk-forward — `TimeSeriesSplitCV(time_column=…, embargo_time="<duration>")`. The splitter translates the duration into a row-count `gap` by inspecting the actual timestamp distribution. Use when problem geometry is "one train window, one test window per fold."
   - Combinatorial purged CV — `PanelCombinatorialPurgedCV(time_column=…, asset_column=…, target_horizon_bars=<h>, embargo_pct=<p>)`. Folds over **unique sorted timestamps**; skfolio CPCV runs on the timestamp axis; per-asset purge is timestamp-atomic (drop a timestamp from train → drop every asset's row at that timestamp). Use when problem geometry is "multiple combinatorial backtest paths needed for HP ranking."

**Foot-gun**: `TimeSeriesSplitCV.gap` (row-count) on a K-row-per-timestamp panel produces ≈ `gap / K` timestamps of per-asset embargo. The motivating bug: `gap=24` on a 1084-asset hourly panel ≈ 0.022 timestamps of separation, not 24 hours. PR-023's D2 polymorphic `embargo_time` and D1 `PanelCombinatorialPurgedCV` fix this; setting `gap` (only) on a panel still works at the row level but is rarely what the user wanted.

The structural test `tests/config/test_base_toml_discriminator_hygiene.py` walks every variant of `CVConfig` and asserts no variant-specific knob sits at the base-table position — panel-aware fields (`time_column`, `asset_column`, `embargo_time`, `target_horizon_bars`, `embargo_pct`) all default to `None` / `0` / `0.0` so the rule is honored without manual upkeep.

### CatBoost ingest + categorical handling (per PR-019)

CatBoost's ``fit(DataFrame, y)`` accepts pandas/polars DataFrames directly; no ``Pool`` construction is required (Q-Wrap §9 PROVEN at v1.2.10). ``Pool`` is the optional ``DMatrix``/``Dataset`` analog and is exposed by ``src/rux_ml/training/catboost/ingest.py::build_pool`` as a utility for advanced users (e.g., explicit ``baseline=``, ``weights=``, ``timestamp=`` knobs), but the factory does not call it.

**Categorical handling (Q-Cat)**: CatBoost does NOT auto-detect pandas Categorical dtype — opposite of LightGBM. It actively errors on category-dtype columns not listed in ``cat_features=``. The ``_CatBoostTrainerShim`` extracts categorical column names from the input DataFrame at fit time via ``select_dtypes(include="category")`` and threads them through to the underlying estimator. The workbench's ``_ColumnRouter`` (PR-005) is unchanged — its low-card-passthrough output (pandas Categorical) is exactly what the shim looks for; high-card target-encoded floats pass through as numeric.

CatBoost's native categorical algorithm is **Ordered Target Statistics** with smoothing (Prokhorenkova et al. 2018) — different from LightGBM's Fisher partitioning and XGBoost's partition-based split. For the workbench's existing high-card target-encoding path (NestedCVWrapper), Pargent et al. 2022 groups CatBoost ordered TS and K-fold target encoding both under "regularized target encoding" and finds them comparable; whether CatBoost ordered TS strictly beats NestedCV target encoding for catboost-only studies is an open empirical question (deferred to post-v0 benchmark).

**Threading (Q-Parallel)**: CatBoost uses Intel TBB, NOT OpenMP — the workbench's ``OMP_NUM_THREADS`` env-var pinning (from PR-011) is **invisible to CatBoost**. The factory reads ``OMP_NUM_THREADS`` from the trial subprocess's environment and passes it explicitly as ``thread_count=`` to the CatBoost estimator. Same intent, different transport. CONVENTIONS.md documents this.

**GPU (Q-GPU)**: CatBoost ships prebuilt CUDA-enabled PyPI wheels via ``uv add catboost`` (no extras, no source build, no container delta). RTX 4090 (CC 8.9) is field-confirmed. ``cfg.device == "cuda"`` activates ``task_type="GPU"`` + ``devices="0"`` (single-GPU pin). CatBoost-GPU is ~2.6–4.6× slower than XGBoost-GPU but materially faster than CatBoost-CPU at workbench scale. GPU bit-exact determinism is NOT achievable (CatBoost issue #546); CPU bit-exact requires the recipe documented in CONVENTIONS.md.

### LightGBM ingest path (per PR-018 Q-Ingest)

LightGBM's ``Dataset`` always quantizes input features to uint8 histograms at construction (`max_bin=255` default), giving ~8x memory compression vs raw float arrays. **No tier-switch decision rule is needed** — unlike XGBoost (`QuantileDMatrix` vs `ExtMemQuantileDMatrix` per D3), LightGBM's binned dataset fits in host RAM at workbench scale (36 GB host; ~50 MB for a 1Mx50 dataset post-binning). The out-of-core path (`two_round=True` for memory-mapped files; `Dataset(data=[Sequence(...), ...])` for chunked readers) is file-based, not host-RAM-tier — deferred to a future PR if needed.

`src/rux_ml/training/lightgbm/ingest.py::build_dataset` is the single helper; it constructs a `lightgbm.Dataset` from pandas DataFrame + label, with `categorical_feature="auto"` for pandas-Categorical auto-detection (see LightGBM categorical handling below).

### LightGBM categorical handling (per PR-018 Q-Cat)

LightGBM's native categorical splitter is **Fisher (1958) optimal partitioning over a sorted gradient histogram** — different algorithm from XGBoost's partition-based split, but the workbench feeds both the same pipeline output. Low-cardinality categoricals (per `categorical_low_card_threshold`) reach LightGBM as pandas Categorical dtype, and LightGBM's `_data_from_pandas` auto-detects them under the default `categorical_feature="auto"`. High-cardinality categoricals reach LightGBM as `NestedCVWrapper`-target-encoded floats — which **LightGBM's own docs recommend** ("treat high-card as numeric"; corroborated by Pargent et al. 2021: regularized target encoding outperforms native LightGBM handling on high-card).

No `_ColumnRouter` changes were needed — the same sentinel (renamed `PASSTHROUGH_TO_XGB_CATEGORICAL` → `PASSTHROUGH_NATIVE_CATEGORICAL` in PR-018) serves both families.

### XGBoost ingest path (per D3)

```
estimate_X_bytes(data) →
  if X_bytes ≲ 18 GB (fits in 24 GB VRAM with headroom):
    QuantileDMatrix on GPU (device="cuda", tree_method="hist")
  else:
    ExtMemQuantileDMatrix on GPU with cache_host_ratio
    tuned to host RAM (CUDA async / RMM pool required)

CPU hist only as fallback for features unsupported on GPU.
```

The threshold is configurable in `[data]` TOML (default `gpu_in_memory_x_gb_max = 18`); the host-RAM cache ratio is configurable in `[memory]`.

### Categorical encoding (per D4)

```
for each categorical column:
  if cardinality(col) ≤ features.categorical_low_card_threshold:
    pass through as Polars Categorical → XGBoost enable_categorical=True
  else:
    category_encoders TargetEncoder via NestedCVWrapper
    (or hash encoding if cardinality is truly massive — configurable)
```

The threshold is a TOML knob in `[features]` with **no default value** (BEST-GUESS — to be tuned on first dataset).

### Memory-pressure response (per D10, locked-in by PR-011)

```
Watchdog wraps each trial body (rux_ml._internal.memory.Watchdog):
  background thread samples psutil.Process.memory_info().rss at
  memory.watchdog_sample_hz (default 1 Hz); tracks peak; sets `tripped` flag
  if RSS > memory.watchdog_threshold_gb (default 28 GB, BEST-GUESS).

Observational + post-fit-check (not preemptive):
  XGBoost training is a long-running C call — Python signals from a background
  thread aren't reliable inside C extensions. The trial body checks
  wd.tripped after the fit returns:
    if wd.tripped: raise optuna.TrialPruned (caught by objective wrapper)
  Hard OOMs (process > OS limit) are handled by PR-008's subprocess
  isolation: OS kill → non-zero exit → parent marks trial FAIL.

Recorded on every trial: TrialAttrs.peak_rss_mb (required field — PR-011).

NOT used: dynamic Polars batch shrinking, mid-trial DMatrix swap.
The static pre-trial path (D3 ingest selector) is the only adaptive layer.
```

### Trial isolation (per D10/D16)

```
Default: subprocess-per-trial via subprocess.run of
         python -m rux_ml._internal.trial_runner

Override: tuning.trial_isolation = "in_process" (TOML)
          (faster startup, but Python may not reclaim RAM between trials —
          documented Optuna OOM cure is the subprocess path)
```

### Optuna sampler / pruner (per D6, locked-in by PR-007 Tier-2 research)

```
objective:
  K-fold CV-mean (per cfg.cv via PR-015's make_splitter)             # default; per-fold scores reported via trial.report() for WilcoxonPruner
  Single fit per trial                                               # not exposed at v0; reserved for a future regime

sampler:
  TPESampler(multivariate=True, group=True, constant_liar=True,      # default — XGBoost search spaces always include categoricals/conditionals which route to TPE per Optuna AutoSampler convention
             n_startup_trials=20)
  GPSampler(...)                                                     # for purely-numerical sub-studies up to AutoSampler's hard-coded 250-trial GP→post-GP boundary
  HEBO (via optunahub, opt-in)                                       # not in default deps; sampler factory raises clear ImportError pointing at `pip install optunahub hebo`
  BoTorchSampler                                                     # REMOVED — deprecated in Optuna 3.6 (~5x slower than GPSampler; no cited tabular-GBM advantage)

pruner:
  WilcoxonPruner                                                     # default — Optuna 3.6+ design intent for "k-fold cross-validation score of a machine learning model"
  MedianPruner                                                       # conservative alternate (Optuna's own xgboost_cv_integration.py uses it; production-grade)
  HyperbandPruner, SuccessiveHalvingPruner                           # preserved in the literal for hypothetical single-fit objectives
  NopPruner ("none")                                                 # disables pruning
```

**Important**: `XGBoostPruningCallback` is **NOT** wired inside the CV loop (Optuna #3203 — duplicate `step=0,1,…` reports per fold break iteration-level pruners). Per-fold XGBoost-internal `early_stopping_rounds` handles within-fold convergence by using the held-out test fold as its `eval_set`; Optuna pruning operates only at fold granularity via `trial.report(fold_score, fold_idx)`.

**This eval_set placement is a research-acknowledged pragmatic deviation, not a textbook-clean pattern.** Per PR-022 Phase 3 research (2026-05-18), it matches the library-blessed default of `xgboost.cv()` / `lightgbm.cv()` / `catboost.cv()` but produces an optimism bias in CV-reported metrics that compounds across HPO trials. Workbench CV scores should be treated as point estimates for HP ranking, not as unbiased generalization estimates. Alternative patterns (Position B: inner val carved from train fold, sklearn's `HistGradientBoosting*` default; Position C: no early stopping in CV with post-CV retrain, XGBoost's own recommendation) are explicitly available; switching to either is deferred to a future Tier-2 study per memory `project_cv_strategy_tier2`. See `docs/CONVENTIONS.md` "HPO objective shape" for the full reframe + cited sources.

**TPE + Wilcoxon caveat**: Optuna's WilcoxonPruner docs note "TPESampler currently cannot utilize the information of pruned trials effectively" under Wilcoxon. Real tradeoff documented; mitigation is `cfg.tuning.pruner = "median"` for users who want the TPE feedback loop to learn from pruned trials.

All choices are TOML knobs in `[tuning]`.

---

## Key Abstractions

### `Solver` Protocol + solving layer (per PR-020 / docs/0.1/DESIGN-log.md Q1)

The Solver layer is the parallel-Protocol surface to the Trainer layer. Per `docs/0.1/DESIGN-log.md` Q1: two parallel `typing.Protocol`s with no shared parent — `Trainer.fit(X, y) → predict(X)` and `Solver.solve(problem) → SolverResult` have fundamentally different data flow; the workbench unifies the **factory/registry** (string-name dispatch via `make_trainer` / `make_solver`) but NOT the runtime contract.

**Per PR-020 Q-Shape research: B-partial-mirror.** The solving layer reuses cross-cutting infrastructure (config-layer discriminated union, registry dict, factory dispatcher, Optuna study substrate, per-trial provenance triple, conformance test pattern) but BYPASSES semantics-mismatched layers — solver runs have no `cfg.data` (problems are matrices, not Parquet), no `cfg.cv` (single solves have no folds), no fittable-model bundle (solver output is `x_star`, not a reusable estimator).

```python
# src/rux_ml/solving/protocol.py
class Solver(Protocol):
    def solve(self, problem: Any) -> SolverResult: ...
```

`problem` is `Any` because future Solver families take different types — cvxpy uses `cvxpy.Problem`; OSQP-direct uses raw matrix tuples; pyomo-direct uses `ConcreteModel`.

**Per PR-020 Q-First-Solver: CVXPY ships as the first family.** Zero dep addition (cvxpy + clarabel were already transitive via skfolio; promoted to direct deps in PR-020). One `cfg.solving.solver=` switch unlocks Clarabel / OSQP / SCS / ECOS / HiGHS through cvxpy's DCP modeling layer. Default `solver="CLARABEL"` for per-trial provenance stability (Clarabel ranks #3 in `qpsolvers/free_for_all_qpbenchmark`; modern Rust solver). Backend-specific tolerance/iter knobs flow through `cfg.solving.solver_opts: dict[str, Any]` (escape hatch mirroring `XGBoostTraining.model_kwargs`).

```python
# src/rux_ml/solving/__init__.py
SOLVER_FAMILIES: dict[str, Callable[..., Solver]] = {
    "cvxpy": make_cvxpy_solver,
}
```

**`RuxMLConfig.solving: SolvingConfig | None = None`** — Optional field. Trainer-shaped studies leave it None; solver-shaped studies set it. Per-trial provenance: `TrialAttrs.solving_cfg_hash` is populated only when `cfg.solving is not None`. Solver-runtime fields (`solver_status`, `objective_value`, `solver_iter_count`, `solve_time_s`) are populated post-solve by `rux-ml solve`.

**CLI verb**: `rux-ml solve` is a single command (analog of `rux-ml train`) — not a typer-group. Solver-internal HPO (tuning Clarabel's `max_iter`, OSQP's `rho`/`alpha`, etc) is deferred to a follow-up Tier-2 PR; v0.1's solve is one-shot only. The user defines `build_problem() -> cvxpy.Problem` in a Python module pointed at by `cfg.solving.problem_module: str`; the CLI imports + calls + records.

**Hashing limitation** (documented in `docs/CONVENTIONS.md`): `solving_cfg_hash` covers the `problem_module` *path* but NOT the module's source content. Users either keep problem modules inside the workbench's git (`git_sha` captures edits) or accept the gap.

### `Trainer` Protocol + family registry (per D5; refactored in PR-017)

The unifying contract across model families is the sklearn estimator API expressed as a `typing.Protocol`. Zero runtime cost; full compile-time substitutability across `XGBClassifier`, `XGBRegressor`, `LGBMClassifier`, `CatBoostClassifier`, sklearn estimators, and any future custom model.

The Protocol covers only the **universal subset** (`fit` + `predict`) so both classifiers and regressors satisfy it. Classification-only (`predict_proba`) and conditionally-available (`best_iteration_` after early-stopping fit) attributes are accessed defensively at call sites — the metric registry casts to a classifier surface for AUC / logloss; the CLI uses `getattr(..., None)` for `best_iteration`.

The Protocol is intentionally **not** `@runtime_checkable`. Per PEP 544 + CPython 3.12 typing docs, `isinstance(x, MyProtocol)` only checks method *names*, not signatures — a factory returning `lambda X, y: None` would pass `isinstance` and fail at real `.fit()` calls. The behavioral conformance test (`tests/training/test_registry_conformance.py`) is the runtime gate instead: it parametrizes over `TRAINER_FAMILIES.keys()` and asserts each factory's output passes a tiny end-to-end fit-predict smoke. Anchored on sklearn `parametrize_with_checks` + Optuna `pytest_samplers.py`.

```python
# src/rux_ml/training/protocol.py
from typing import Protocol, Any
from numpy.typing import ArrayLike

class Trainer(Protocol):
    def fit(self, X: Any, y: ArrayLike, **kwargs: Any) -> "Trainer": ...
    def predict(self, X: Any) -> ArrayLike: ...
```

**Per-family layout** (per PR-017 / `docs/0.1/DESIGN-log.md` Q2): each Trainer family lives as a subpackage under `src/rux_ml/training/<family>/` with three files — `__init__.py` (public API), `factory.py` (the `make_<family>_trainer` function), `config.py` (the family's Pydantic schema variant of `TrainingConfig`). XGBoost is the first family at `src/rux_ml/training/xgboost/`; LightGBM and CatBoost subpackages land in PR-018 / PR-019.

**B-explicit registry** (per Q4): families are registered in a plain dict in `src/rux_ml/training/__init__.py`:

```python
# src/rux_ml/training/__init__.py
TRAINER_FAMILIES: dict[str, Callable[..., Trainer]] = {
    "xgboost": make_xgboost_trainer,
}
```

Hand-maintained — no decorator-driven registration. Pattern anchored on HuggingFace transformers' `MODEL_MAPPING_NAMES`; entry-points (a la `pytest11`) ruled out because they're for third-party discovery, which the workbench has no need for. The top-level `make_trainer(cfg)` dispatches against this dict:

```python
def make_trainer(cfg: TrainingConfig, *, seed: int | None = None) -> Trainer:
    family = cfg.kind  # discriminator on each variant
    factory = TRAINER_FAMILIES[family]
    return factory(cfg, seed=seed)
```

**Discriminated-union `TrainingConfig`** (per Q-A1 + Q-MK research findings): the per-family config schema is a discriminated union over per-variant Pydantic models, all inheriting from a shared `TrainingBase` for family-agnostic fields (`device`, `metric`, `early_stopping_rounds`). Variants declare `kind: Literal["<family>"]` for the discriminator + family-specific fields. `model_kwargs: dict[str, Any]` is retained only on variants whose upstream `__init__` accepts `**kwargs` (XGBoost, LightGBM); CatBoost variants will OMIT `model_kwargs` because `CatBoostClassifier.__init__` rejects unknown kwargs.

```python
# src/rux_ml/config/training.py
TrainingConfig = Annotated[
    XGBoostTraining,  # | LightGBMTraining | CatBoostTraining (PR-018 / PR-019)
    Field(discriminator="kind"),
]
```

The native `xgb.train()` API is reserved for the ~5 % of cases where the sklearn wrapper falls short — chiefly `QuantileDMatrix(..., ref=train_dmat)` and `ExtMemQuantileDMatrix` paths. This branch is configurable on the `XGBoostTraining` variant (`use_native: bool`).

### `RuxMLConfig` and `SearchSpec` (per D2 / D17 / D16)

Root configuration is a Pydantic-settings `BaseSettings` composed of per-layer models, with a tagged-union `SearchSpec` for HPO declarations:

```python
# src/rux_ml/config/root.py
class RuxMLConfig(BaseSettings):
    data: DataConfig
    features: FeaturesConfig
    training: TrainingConfig
    tuning: TuningConfig
    runs: RunsConfig
    registry: RegistryConfig
    memory: MemoryConfig
    search_space: dict[str, SearchSpec] = {}

# src/rux_ml/config/tuning.py
class FloatSpec(BaseModel):
    type: Literal["float"] = "float"
    low: float; high: float; log: bool = False
class IntSpec(BaseModel):
    type: Literal["int"] = "int"
    low: int; high: int; log: bool = False
class CatSpec(BaseModel):
    type: Literal["categorical"] = "categorical"
    choices: list[str | int | float]
SearchSpec = Annotated[Union[FloatSpec, IntSpec, CatSpec],
                       Field(discriminator="type")]
```

The objective walks `cfg.search_space` and translates each entry to a `trial.suggest_*` call.

### CV Strategy — `Splitter` Protocol (per PR-015)

Repeated cross-validation is expressed through a `typing.Protocol` over the **universal subset of the sklearn splitter API**: `split(X, y, *, groups)` yielding `(train_idx, test_idx)` row-index pairs (sklearn convention). Polars-in (matches PR-005), numpy-index-out — repeated CV stays in indices so PR-007's objective owns the materialisation policy.

```python
# src/rux_ml/data/cv.py
from typing import Protocol, ClassVar
from collections.abc import Iterator
import polars as pl
import numpy as np
from numpy.typing import NDArray

class Splitter(Protocol):
    extmem_compatible: ClassVar[bool]
    def split(
        self,
        X: pl.DataFrame,
        y: pl.Series | None = None,
        *,
        groups: NDArray[np.int_] | None = None,
    ) -> Iterator[tuple[NDArray[np.int_], NDArray[np.int_]]]: ...
    def get_n_splits(self) -> int: ...
```

Concrete strategies (each constructible from its `CVConfig` variant via `make_splitter(cfg, *, seed=…)`):

- `KFoldSplitter` / `StratifiedKFoldSplitter` / `TimeSeriesSplitter` / `GroupKFoldSplitter` — wrap sklearn `KFold` / `StratifiedKFold` / `TimeSeriesSplit(gap, max_train_size)` / `GroupKFold`. `TimeSeriesSplitter` additionally accepts `time_column` + polymorphic `embargo_time: int | str` (PR-023 D2) — duration strings parsed via `pandas.Timedelta` and translated to a row-count gap from the input's timestamp distribution.
- `CombinatorialPurgedSplitter` — wraps `skfolio.model_selection.CombinatorialPurgedCV` (BSD-3); flattens skfolio's `(train, list[test_path])` yield into the sklearn `(train, test)` shape (per-path decomposition out of scope at v0). Surfaces ergonomic `target_horizon_bars` + `embargo_pct` knobs that convert internally to skfolio's row-count `purged_size` / `embargo_size` (PR-023 D3, AFML Snippet 7.3).
- `PanelCombinatorialPurgedSplitter` (PR-023 D1) — wraps the same skfolio CPCV but runs it over the **unique sorted timestamps** read from `time_column`, then maps each timestamp-level fold back to row indices via a row-to-timestamp-index map. Per-asset purge is timestamp-atomic (drop a timestamp from train → drop every asset's row at that timestamp). `asset_column` is required and validated at split time.

**Per-strategy leakage guarantees** (Q2.b research output, PR-023 panel additions):

| Strategy | Guarantees | Does NOT guarantee |
|---|---|---|
| `KFold` (shuffled) | Each row in exactly one test fold | Group separation; class balance; temporal ordering |
| `StratifiedKFold` | Class proportions per fold | Group separation; temporal ordering |
| `GroupKFold` | Each group in exactly one test fold | Class balance; equal fold sizes; temporal ordering |
| `TimeSeriesSplit` | Train precedes test; `gap` excludes adjacent. With `embargo_time="<dur>"` + `time_column`, separation honored in **time units** (panel-aware). | Group separation; variable-horizon label purging; class balance |
| `CombinatorialPurgedCV` | Two-sided label-overlap purge + one-sided post-test embargo in **row units** (AFML §7.4.2). With ergonomic `target_horizon_bars` + `embargo_pct`, equivalent row-count derivation per AFML Snippet 7.3. | Class balance; group separation; panel-atomic timestamp purge |
| `PanelCombinatorialPurgedCV` | Folds over unique sorted timestamps; per-asset purge is timestamp-atomic (`target_horizon_bars` + `embargo_pct` in timestamp units); combinatorial paths preserved via post-hoc row-index decomposition | Class balance |

**Groups column-to-array convention**: `GroupKFoldCV` carries `groups_column: str`; the caller resolves it via `df[col].to_numpy()` before calling `splitter.split(..., groups=arr)`. Splitter holds no DataFrame state.

**ExtMem compatibility**: only `TimeSeriesSplitter` is `extmem_compatible` at v0 (file-level path requires partition-aligned strategies); incompatible pairings raise `NotImplementedError` at training time — materialised fallback deferred to a follow-up PR.

Per-strategy config lives in `src/rux_ml/config/cv.py` as a tagged-union `CVConfig` (one Pydantic sub-model per strategy, discriminated by `kind`). `RuxMLConfig.cv` adds the 8th per-layer config hash (`cv_cfg_hash`) to the per-trial provenance triple (`docs/CONSTRAINTS.md`).

The one-shot `train_val_test_split(df, *, ratios, seed) → dict[str, pl.DataFrame]` from PR-004 is intentionally **separate** from the Splitter Protocol — different shapes for different memory regimes (one-shot returns DataFrames; repeated returns indices to avoid K-fold materialisation).

### Model bundle + manifest (per D8)

The registry stores models as a two-file bundle plus a Pydantic-validated manifest:

```
registry/<problem>/<version>/
  pipeline.skops      # sklearn FE Pipeline via skops.io
  model.ubj           # XGBoost Booster via save_model
  manifest.json       # Pydantic-validated lineage
registry/<problem>/champion.json  # atomically rewritten on promotion
```

Manifest schema includes the full provenance triple (config hashes, data hash, git SHA, env/version pins, the originating Optuna study + trial number, the metric value, and `feature_list_hash`).

The loader is intentionally thin and depends only on `xgboost`, `skops`, `sklearn`, and `polars`:

```python
def load_model(problem: str, version: str = "champion") -> tuple[Pipeline, xgb.Booster]:
    ...
```

Inference code never imports the training stack.

---

## Storage

The workbench uses three on-disk stores plus the layered config files. All runtime stores live under `WORKBENCH_HOME` (default = repo root) and are gitignored.

| Store | Path | Format | Purpose |
|---|---|---|---|
| **Optuna study** | `studies/studies.db` | SQLite (`RDBStorage`) | Per-trial params, intermediate metrics, `user_attrs` (the run log) |
| **Optuna artifacts** | `studies/artifacts/...` | `optuna.artifacts.FileSystemArtifactStore` | Per-trial model bundles, plots, prediction CSVs |
| **Model registry** | `registry/<problem>/<version>/...` | filesystem | Promoted bundles + manifest + atomic `champion.json` |
| **Data CAS** | `data/cas/...` + `data/manifests/...` | filesystem (hardlinks + JSON) | Versioned dataset snapshots; manifests index by composite `data_hash`. **Hardlinks require source and `data/cas/` to share a filesystem; cross-device errors (`OSError(errno.EXDEV)`) fall back to `shutil.copy2` with a warning** |
| **Logs** | `logs/...` | structured JSONL | Per-run logs (trial-correlated) |
| **Config** | `configs/{base.toml, problems/<n>.toml, studies/<n>.toml}` | TOML | Layered configuration |

Optuna SQLite is the **single source of truth** for trial history. The registry is a curated, promoted slice — promotion is explicit, never implicit-by-being-the-best. Artifacts are content-addressed by Optuna in `studies/artifacts/`; promotion copies the relevant pair into `registry/<problem>/<version>/` so the registry is independently loadable.

There is no migration system because there is no shared schema beyond Optuna's own and the manifest schema. Schema evolution for the manifest is handled via the Pydantic model in `src/rux_ml/registry/manifest.py` — backward-compatibility is the loader's responsibility, validated at load time.

---

## Configuration Architecture (per D17)

Three-tier layered TOML composition, loaded by pydantic-settings:

```
configs/
├── base.toml                       # always loaded; project-wide defaults
├── problems/
│   ├── churn_v1.toml               # dataset paths, target column, eval metric
│   └── fraud_v1.toml
└── studies/
    ├── churn_xgb_wide.toml         # search space, n_trials, sampler, pruner
    ├── churn_xgb_narrow.toml
    └── fraud_xgb_baseline.toml
```

Override precedence (highest → lowest):

1. CLI dot-path overrides via repeatable `--set` (`--set training.learning_rate=0.01`)
2. Env vars (`RUXML_TRAINING__LEARNING_RATE=0.01`)
3. `.env` file (dev only; gitignored)
4. Study TOML
5. Problem TOML
6. `base.toml`
7. Pydantic field defaults

`TomlConfigSettingsSource(deep_merge=True)` ensures nested tables overlay correctly. Lists replace; they do not concatenate. `extra="forbid"` is set on every Pydantic model so typos raise validation errors.

`config_hash` semantics: per-layer hashes (`data_cfg_hash`, `features_cfg_hash`, `training_cfg_hash`, `tuning_cfg_hash`) plus `root_cfg_hash`. All five are stored in Optuna `user_attrs`. Per-layer hashes enable cache invalidation per concern; the root hash is run identity. Elided fields (paths, timestamps, runtime-only fields) live in a constant `_HASH_ELIDED_FIELDS` in `src/rux_ml/config/root.py`.

The data hash (`data_hash`, computed in `src/rux_ml/data/versioning.py`) is **content of the dataset**, distinct from `data_cfg_hash` (the config layer).

---

## Reproducibility Architecture (per D9)

Each trial's reproducibility footprint:

1. **Code:** `git_sha` recorded; `uv.lock` + `Cargo.lock` committed
2. **Environment:** Container pinned by `@sha256:...` digest; `image_digest` recorded
3. **Python deps:** `uv.lock` (universal cross-platform lock)
4. **Rust deps:** `Cargo.lock` (when crate exists)
5. **Data:** composite `data_hash` = SHA-256 over sorted per-partition byte hashes + `logical_hash` (sorted canonical column projection); both stored
6. **Config:** per-layer + root `*_cfg_hash` (D17)
7. **Seeds (per PR-013):** `make_seed_bag(master_entropy=cfg.tuning.entropy, trial_number=trial.number)` derives a per-trial `SeedBag` via `SeedSequence(entropy=master, spawn_key=(trial_number,))` → `.generate_state(4)` for a stable per-trial entropy → fresh `SeedSequence` → `.spawn(4)` → four `int32`-safe child seeds (`split_seed`, `cv_seed`, `sampler_seed`, `xgb_seed`). The per-trial `entropy_hex` is recorded; `make_seed_bag_from_hex` reconstructs an identical bag without needing the master or trial number — the contract PR-010's `registry/promote.py` relies on for deterministic re-fit at promotion. `OMP_NUM_THREADS` pinned via PR-011.
8. **Hardware (per PR-013):** required `xgboost_version` (`xgboost.__version__`) + `cuda_runtime_version` (`xgboost.build_info()["CUDA_VERSION"]`); optional `gpu_model` + `driver_version` (from `nvidia-smi`, GPU-only)

XGBoost GPU `hist` is **near-deterministic, not bit-exact across hardware**. Multi-GPU is explicitly non-deterministic. We accept this as a documented limitation; version-logging lets us attribute drift if it appears.

Container digest pinning is non-negotiable per `docs/CONSTRAINTS.md`.

---

## Memory & Parallelism Architecture (per D10)

| Layer | Mechanism | Default |
|---|---|---|
| OOM hard cap | Container `--memory=32g` (cgroup v2 `memory.max`) | 32 GB |
| OOM soft cap | Container `--memory-reservation=28g` | 28 GB |
| In-process watchdog | `psutil` RSS sampler at 1 Hz; `MemoryPressureError → optuna.TrialPruned` | 28 GB threshold (BEST-GUESS) |
| Trial isolation | `subprocess.run` of `python -m rux_ml._internal.trial_runner` (spawn semantics) | enabled |
| Thread allocation | `OMP_NUM_THREADS=24`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `POLARS_MAX_THREADS=24`, XGBoost `nthread=24` | per-library all-or-one |
| Skipped | `RLIMIT_AS` (unreliable on Linux) | — |
| On-demand profiling | `memray attach --aggregate` | manual |
| ExtMem host-RAM cache | `MemoryConfig.cache_host_ratio` (`null` = XGBoost auto-estimate) — consumed by the D3 ingest path when X exceeds `gpu_in_memory_x_gb_max` | `null` |

Sequential trials (D6) plus 24-thread CPU means each library may use the whole CPU when active. BLAS env vars are pinned to 1 to suppress nested oversubscription that `threadpoolctl` cannot reliably reach across distinct OpenMP runtimes (libgomp vs libiomp).

---

## Testing Strategy (per D12)

| Layer | What | Where |
|---|---|---|
| Unit | Per-transformer; per-function; tiny synthetic data | `tests/<layer>/test_*.py` |
| Common contract | All custom transformers pass round-trip + idempotent + correct-output-type | `tests/common/test_estimator_contract.py` |
| Integration | Small end-to-end fit/predict through FE → train → score | `tests/integration/` |
| Golden regression | `np.testing.assert_allclose(preds, golden, atol, rtol)` + AUC/RMSE bands; tiny fixed-seed dataset | `tests/golden/`; marker: `@pytest.mark.golden` |
| Property-based | `hypothesis.extra.numpy`/`pandas` for shape/dtype/idempotence/inverse-transform | within unit tests; selective use only |
| GPU-gated | `@pytest.mark.gpu` (skipped on CPU-only machines) | within unit/integration tests |
| CLI | `typer.testing.CliRunner` invocations on tiny fixtures | `tests/cli/` |
| Rust core (when crate exists) | `cargo test` per crate | `crates/<crate>/tests/` |
| PyO3 boundary (when crate exists) | pytest exercises Python-side API | `tests/<binding>/` |

`pyproject.toml` registers markers (`gpu`, `slow`, `golden`, `integration`) and sets `addopts = "-m 'not gpu and not slow and not golden'"` so default `pytest` is fast.

Test data fixtures live in `tests/conftest.py` (shared) or per-test directory. Golden snapshot files live under `tests/golden/fixtures/`.

---

## Rust+PyO3 Boundary (per D13)

**No custom Rust at v0.** `crates/.gitkeep` reserves the directory; no `Cargo.toml` at the root yet. The first Rust PR materializes the workspace + first crate together, only when a profile shows a Python hot path consuming >5 % of a real workbench task.

The most likely first crate is a `pyo3-polars` expression plugin for a row-dependent transform Polars can't vectorize cleanly (custom rolling stat, custom target-encoder variant, exotic categorical hash). `xxhash`/`blake3` Python bindings cover hashing without needing custom Rust; `pyarrow.compute` covers most other kernel needs.

Optional-accelerator import convention: `try: from rux_ml._kernels import fn ; except ImportError: from .pure_py import fn`. Pure-Python reference implementations live alongside every Rust function, so the package remains importable without `maturin develop`.

---

## What's intentionally NOT in the architecture

- **Online model serving** — out of scope. The registry has a loader; serving builds on it externally if needed.
- **Distributed / multi-GPU training** — single-machine, single-GPU only.
- **Time-series-specific tooling** — tabular GBM focus.
- **AutoML automation** — AutoGluon/PyCaret/pytorch-tabular all rejected (D5; FE control is non-negotiable).
- **Web dashboards** — non-negotiable per `docs/CONSTRAINTS.md`.
- **Feature store** (Feast / Hopsworks) — overkill for solo (D4).
- **Notebooks-first workflow** — CLI + scripts only.
- **Raw source data in repo** — paths in TOML; CAS holds versioned snapshots only.

These can be revisited via `PROCEDURE-design-planning.md` when the constraints that excluded them change.

---

## Cross-Decision Consistency Notes

Two interactions worth flagging explicitly:

1. **SQLite storage (D6) + subprocess-per-trial (D10) + parent-child coordination via shared SQLite (D16)** — works because writes are serialized in time (parent waits for `subprocess.run` exit before the next trial). Optuna's "no SQLite for parallel" warning targets concurrent writers; we are sequential.
2. **`data_hash` (D9, dataset content) vs `data_cfg_hash` (D17, data-layer config hash)** — two distinct hashes sharing a confusing root name. `data_hash` is content; `*_cfg_hash` are config. Both are stored in `user_attrs` per `docs/CONSTRAINTS.md`. `config_hash` (D7) is an alias for `root_cfg_hash` for backward-compatible queries.

---

## Pointers

- **Constraints (hard rules):** `docs/CONSTRAINTS.md`
- **Conventions (soft patterns):** `docs/CONVENTIONS.md`
- **Decision research trail:** `docs/0.0/DESIGN-log.md` (v0: D1 through D17) + `docs/0.1/DESIGN-log.md` (v0.1: Q1 through Q6 + deferred A1–A6) <!-- rewrite-doc-refs:skip-line -->
- **PR plan:** `docs/0.1/ROADMAP.md` (v0 roadmap frozen at `docs/0.0/ROADMAP.md`) <!-- rewrite-doc-refs:skip-line -->
- **Per-PR research status:** `docs/0.1/RESEARCH-BACKLOG.md` (v0 backlog frozen at `docs/0.0/RESEARCH-BACKLOG.md`) <!-- rewrite-doc-refs:skip-line -->
- **Procedures:** `PROCEDURE-design-planning.md`, `PROCEDURE-pr-research.md`, `PROCEDURE-code-audit.md`
