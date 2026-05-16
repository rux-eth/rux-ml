# Architecture

`rux-ml` is a personal ML research workbench for the full lifecycle of tabular gradient-boosted models — primarily XGBoost, with a contract that admits LightGBM, CatBoost, and other sklearn-compatible models later. It is operated via a single CLI (`rux-ml`) and a Python package (`rux_ml`); there is no web UI and no long-running server.

This document describes the architecture as designed across the 17 decisions logged in `docs/DESIGN-log.md`. Hard rules live in `docs/CONSTRAINTS.md`; soft patterns in `docs/CONVENTIONS.md`.

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
| **Registry** | `src/rux_ml/registry/` | Model bundle write/read (`pipeline.skops` + `model.ubj` + `manifest.json`); promotion logic with atomic `champion.json` rewrite; thin `load_model(problem, version="champion")` API |
| **Internal** | `src/rux_ml/_internal/` | Logging, hashing helpers (`xxhash`/`blake3`), env (OMP/BLAS pinning, `WORKBENCH_HOME` resolution), `SeedSequence` + `.spawn()`, `psutil` memory watchdog raising `MemoryPressureError` |

`src/rux_ml/_internal/trial_runner.py` is the entry point invoked by `subprocess.run` per trial — see Data Flow below.

The repository skeleton is in `docs/CONVENTIONS.md`; the Python package internal structure is laid out in detail in D15 of `docs/DESIGN-log.md`.

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
    ├─ start psutil watchdog (PR-011 — deferred)
    ├─ study = tuning.create_or_load(name, storage, sampler, pruner, direction, load_if_exists=True)
    ├─ study.optimize(build_objective(cfg), n_trials=1)
    │   # build_objective (PR-007) runs K-fold CV-mean per cfg.cv:
    │   #   walk_search_space → overrides → trial_cfg = RuxMLConfig.model_validate(deep-merged)
    │   #   record 8-layer user_attrs via runs.provenance.build_user_attrs
    │   #   make_splitter(cfg.cv, seed=cfg.tuning.entropy); ExtMem-compat gate
    │   #   for fold in folds: fit features + trainer, score, trial.report(score, fold_idx)
    │   #   if trial.should_prune(): raise optuna.TrialPruned
    │   #   return statistics.fmean(fold_scores) → study.tell internally
    └─ exit cleanly (psutil trip in PR-011 → MemoryPressureError → optuna.TrialPruned)

After study completes — promotion is an explicit step:

  CLI: rux-ml registry promote \
       --problem churn_v1 --study churn_xgb_wide_<study_id> --trial <best>

  cli.registry.promote():
    ├─ load trial from study storage
    ├─ download artifacts via optuna.artifacts.download_artifact()
    ├─ compose manifest from trial user_attrs + library versions
    ├─ write registry/<problem>/<version>/{pipeline.skops, model.ubj, manifest.json}
    │     where <version> = v_<YYYY>_<MM>_<DD>_<short_hash>
    └─ atomic rewrite registry/<problem>/champion.json
```

For a one-off (non-sweep) baseline training, `cli.train.run()` follows the same path but wraps the call as a single trial via `study.ask()` + `study.tell()` — sweep and one-off runs share the same store and the same provenance schema.

---

## Decision Rules (codified in the workbench)

These are runtime branches the workbench must auto-select, based on configuration and observed inputs.

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

### Memory-pressure response (per D10)

```
psutil watchdog at memory.watchdog_threshold_gb (default 28 GB) sampling at 1 Hz:
  if RSS > threshold:
    raise MemoryPressureError → optuna.TrialPruned + diagnostics log

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

**Important**: `XGBoostPruningCallback` is **NOT** wired inside the CV loop (Optuna #3203 — duplicate `step=0,1,…` reports per fold break iteration-level pruners). Per-fold XGBoost-internal `early_stopping_rounds` handles within-fold pruning; Optuna pruning operates only at fold granularity via `trial.report(fold_score, fold_idx)`.

**TPE + Wilcoxon caveat**: Optuna's WilcoxonPruner docs note "TPESampler currently cannot utilize the information of pruned trials effectively" under Wilcoxon. Real tradeoff documented; mitigation is `cfg.tuning.pruner = "median"` for users who want the TPE feedback loop to learn from pruned trials.

All choices are TOML knobs in `[tuning]`.

---

## Key Abstractions

### `Trainer` Protocol (per D5)

The unifying contract across model families is the sklearn estimator API expressed as a `typing.Protocol`. Zero runtime cost; full compile-time substitutability across `XGBClassifier`, `XGBRegressor`, `LGBMClassifier`, `CatBoostClassifier`, sklearn estimators, and any future custom model.

The Protocol covers only the **universal subset** (`fit` + `predict`) so both classifiers and regressors satisfy it. Classification-only (`predict_proba`) and conditionally-available (`best_iteration_` after early-stopping fit) attributes are accessed defensively at call sites — the metric registry casts to a classifier surface for AUC / logloss; the CLI uses `getattr(..., None)` for `best_iteration`.

```python
# src/rux_ml/training/protocol.py
from typing import Protocol, Any
from numpy.typing import ArrayLike

class Trainer(Protocol):
    def fit(self, X: Any, y: ArrayLike, **kwargs: Any) -> "Trainer": ...
    def predict(self, X: Any) -> ArrayLike: ...
```

The model factory (`src/rux_ml/training/factory.py`) returns the configured concrete trainer:

```python
def make_trainer(cfg: TrainingConfig) -> Trainer:
    if cfg.kind == "xgboost":
        return XGBClassifier(**cfg.model_kwargs)
    # ... LightGBM / CatBoost / sklearn cases as added
```

The native `xgb.train()` API is reserved for the ~5 % of cases where the sklearn wrapper falls short — chiefly `QuantileDMatrix(..., ref=train_dmat)` and `ExtMemQuantileDMatrix` paths. This branch is configurable in `[training]`.

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

- `KFoldSplitter` / `StratifiedKFoldSplitter` / `TimeSeriesSplitter` / `GroupKFoldSplitter` — wrap sklearn `KFold` / `StratifiedKFold` / `TimeSeriesSplit(gap, max_train_size)` / `GroupKFold`
- `CombinatorialPurgedSplitter` — wraps `skfolio.model_selection.CombinatorialPurgedCV` (BSD-3); flattens skfolio's `(train, list[test_path])` yield into the sklearn `(train, test)` shape (per-path decomposition out of scope at v0)

**Per-strategy leakage guarantees** (Q2.b research output):

| Strategy | Guarantees | Does NOT guarantee |
|---|---|---|
| `KFold` (shuffled) | Each row in exactly one test fold | Group separation; class balance; temporal ordering |
| `StratifiedKFold` | Class proportions per fold | Group separation; temporal ordering |
| `GroupKFold` | Each group in exactly one test fold | Class balance; equal fold sizes; temporal ordering |
| `TimeSeriesSplit` | Train precedes test; `gap` excludes adjacent | Group separation; variable-horizon label purging; class balance |
| `CombinatorialPurgedCV` | Two-sided label-overlap purge + one-sided post-test embargo (AFML §7.4.2) | Class balance; group separation |

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

1. CLI dot-path overrides (`--training.learning_rate=0.01`)
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
7. **Seeds:** `SeedSequence(entropy)` per run; `entropy_hex` persisted; `.spawn()` derives per-component seeds (`split_seed`, `cv_seed`, `sampler_seed`, `xgb_seed`); `OMP_NUM_THREADS` pinned
8. **Hardware:** `gpu_model`, `driver_version`, `cuda_runtime_version`, `xgboost_version`

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
- **Decision research trail:** `docs/DESIGN-log.md` (D1 through D17)
- **PR plan:** `docs/ROADMAP.md`
- **Per-PR research status:** `docs/RESEARCH-BACKLOG.md`
- **Procedures:** `PROCEDURE-design-planning.md`, `PROCEDURE-pr-research.md`, `PROCEDURE-code-audit.md`
