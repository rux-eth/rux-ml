# rux-ml

A personal ML research workbench for the full lifecycle of tabular gradient-boosted models — primarily **XGBoost**, with a contract that admits LightGBM, CatBoost, and other sklearn-compatible models later.

Operated via a single CLI (`rux-ml`) and a Python package (`rux_ml`). No web UI, no long-running server. Single-user, single-machine.

```
data → features → CV strategy → training → tuning → run logging → registry promotion
                                                          │
                              per-trial provenance triple ─┘
```

Every trial — sweep or one-off — records a reproducibility footprint (config hashes, data hash, git SHA, seed entropy, container digest, library/CUDA/driver versions, peak RSS) so any registered model can be re-fit to the exact predictions the trial reported.

---

## Table of contents

- [What this is built for](#what-this-is-built-for)
- [Install](#install)
- [Quick start](#quick-start)
- [Repo layout](#repo-layout)
- [Configuration](#configuration)
- [The lifecycle](#the-lifecycle)
  - [1. Data ingest + versioning](#1-data-ingest--versioning)
  - [2. Features](#2-features)
  - [3. CV strategy (overfit testing)](#3-cv-strategy-overfit-testing)
  - [4. Baseline training](#4-baseline-training)
  - [5. Hyperparameter tuning](#5-hyperparameter-tuning)
  - [6. Run logging + inspection](#6-run-logging--inspection)
  - [7. Registry promotion + model export](#7-registry-promotion--model-export)
  - [8. Loading a model for inference](#8-loading-a-model-for-inference)
- [Reproducibility contract](#reproducibility-contract)
- [Memory & threading](#memory--threading)
- [Containerized runtime](#containerized-runtime)
- [Testing](#testing)
- [Development](#development)
- [CLI reference](#cli-reference)
- [Doc map](#doc-map)

---

## What this is built for

The workbench targets the bare-metal box described in `docs/ARCHITECTURE.md`:

| Resource | Value | Implication |
|---|---|---|
| GPU | RTX 4090 (24 GB VRAM) | XGBoost runs `device="cuda"` by default; sequential trials saturate it |
| CPU | i9-13900K (24 threads) | `OMP_NUM_THREADS=24` per trial; BLAS pinned to 1 to avoid oversubscription |
| RAM | 36 GB DDR5 | **Binding constraint.** 28 GB watchdog threshold matches the 32 GB container cap |
| OS | Ubuntu 22.04 (Linux) | macOS works for CPU dev; GPU paths require Linux + NVIDIA driver |

Non-negotiables (full list in `docs/CONSTRAINTS.md`):

- **No web UI / no long-running server** — CLI + Python imports only
- **Single-GPU, sequential trials** (`n_jobs=1`)
- **CUDA + `fork` is forbidden** — subprocess-per-trial uses spawn semantics
- **Tolerance-based golden tests only** — never exact-hash regression tests against GPU output
- **Zero hardcoded parameters** — all configurable values live in TOML
- **Container base images digest-pinned** by `@sha256:...`, not by tag
- **Per-trial provenance triple required** for any model that gets promoted

---

## Install

The project uses [`uv`](https://docs.astral.sh/uv/) for Python dependency management and [`maturin`](https://www.maturin.rs/) for future Rust+PyO3 components (no Rust code at v0).

```bash
git clone git@github.com:rux-eth/rux-ml.git
cd rux-ml
uv sync                       # creates .venv with Python 3.12 + all deps
uv run rux-ml --help          # verify the CLI loads
```

For the containerized runtime (Linux + NVIDIA Container Toolkit required for GPU), see [Containerized runtime](#containerized-runtime).

---

## Quick start

End-to-end on a tiny dataset, top to bottom:

```bash
# 1. Tell rux-ml where your dataset lives + what to train on.
cat > configs/problems/demo.toml <<'EOF'
[data]
source_path = "/abs/path/to/your.parquet"
target_column = "y"
split_ratios = { train = 0.7, val = 0.15, test = 0.15 }

[features]
categorical_low_card_threshold = 10
[features.spec]
numeric_columns = ["x1", "x2", "x3"]
categorical_columns = ["cat"]

[training]
metric = "auc"   # routes to XGBClassifier; "rmse" / "mae" route to XGBRegressor
EOF

# 2. Sanity-check: hash the dataset, snapshot it into the CAS.
uv run rux-ml --problem demo data hash "$DATA_PATH"
uv run rux-ml --problem demo data version demo "$DATA_PATH"

# 3. Baseline training (single 1-trial study).
uv run rux-ml --problem demo train

# 4. Hyperparameter sweep (requires configs/studies/<name>.toml — see below).
cat > configs/studies/demo_wide.toml <<'EOF'
[search_space."training.learning_rate"]
type = "float"
low  = 0.01
high = 0.3
log  = true

[search_space."training.max_depth"]
type = "int"
low  = 3
high = 8

[tuning]
n_trials = 30
entropy  = 42   # pin the master seed for reproducibility
EOF

uv run rux-ml --problem demo --study demo_wide tune start

# 5. Inspect the runs.
uv run rux-ml --problem demo runs list
uv run rux-ml --problem demo runs show <trial_number> --study demo_<study_id>

# 6. Promote the best trial.
uv run rux-ml --problem demo registry promote \
    --problem demo --study demo_<study_id> --trial <best>

# 7. Load the promoted model in Python.
python - <<'PY'
from rux_ml.registry import load_model
pipeline, booster = load_model("demo")  # version="champion" by default
PY
```

Every step beyond `data hash` writes to either `studies/`, `registry/`, or `data/cas/` — all gitignored, all under `WORKBENCH_HOME` (defaults to the repo root).

---

## Repo layout

```
rux-ml/
├── src/rux_ml/                  Python package — layered, dependencies flow downward
│   ├── cli/                     Typer CLI: data / train / tune / runs / registry
│   ├── config/                  Pydantic-settings models per layer + RuxMLConfig root
│   ├── data/                    Parquet loaders, splits, CAS versioning, Splitter Protocol
│   ├── features/                sklearn Pipeline + ColumnTransformer + target-encoder shim
│   ├── training/                Trainer Protocol + factory (make_trainer) + metric registry
│   ├── tuning/                  Optuna study, objective, samplers, pruners, subprocess dispatch
│   ├── runs/                    TrialAttrs schema + ask/tell wrapper + query helpers
│   ├── registry/                Two-file bundle, manifest, champion.json, load_model
│   └── _internal/               Hashing, env pinning, memory watchdog, seeds, git_sha
├── configs/
│   ├── base.toml                Project-wide defaults (always loaded)
│   ├── problems/<name>.toml     Per-dataset: source_path, target, feature spec, metric
│   └── studies/<name>.toml      Per-sweep: search_space, n_trials, sampler/pruner overrides
├── tests/
│   ├── <layer>/                 Unit tests mirroring src/ layout
│   ├── cli/                     Typer CliRunner end-to-end tests
│   ├── container/               Static + gated Docker smoke tests
│   ├── integration/             CPU bit-exact + GPU near-determinism contracts
│   └── golden/                  Tolerance-based regression infrastructure + fixtures
├── docs/                        Living architecture / constraints / conventions / design log
├── prs/                         Per-PR specs with research findings (one file per PR)
├── crates/.gitkeep              Reserved for future Rust crates (D13 trigger)
├── Dockerfile                   Single-stage GPU-enabled image; digest-pinned bases
├── docker-compose.yml           Compose service with GPU passthrough + 32 GB memory cap
├── Makefile                     install / lint / format / type / test / docker-* / regenerate-golden
├── pyproject.toml               Python deps, ruff/basedpyright/pytest config
└── uv.lock                      Cross-platform dep lock (committed)

# Runtime-generated, gitignored:
├── studies/                     Optuna SQLite + per-trial artifacts
├── registry/<problem>/          Promoted model bundles + champion.json
├── data/cas/, data/manifests/   Content-addressed dataset snapshots
└── logs/                        Structured JSONL logs (trial-correlated)
```

The CLI dependency rule is: `cli/` imports any layer, but no layer imports from `cli/`. Cross-layer calls go through factory functions (`make_features(cfg)`, `make_trainer(cfg)`, `make_splitter(cfg.cv, seed=...)`) — never deep-imports of internal modules.

---

## Configuration

Three-tier layered TOML loaded by pydantic-settings. Precedence (highest → lowest):

1. **CLI overrides** — `--set training.learning_rate=0.01 --set tuning.n_trials=100` (repeatable; values JSON-parsed when possible)
2. **Env vars** — `RUXML_TRAINING__LEARNING_RATE=0.01` (note the `__` for nesting)
3. **`.env` file** — local dev, gitignored
4. **Study TOML** — `configs/studies/<name>.toml` (selected via `--study <name>`)
5. **Problem TOML** — `configs/problems/<name>.toml` (selected via `--problem <name>`)
6. **`configs/base.toml`** — always loaded
7. **Pydantic field defaults** in `src/rux_ml/config/<layer>.py`

`extra="forbid"` is set on every Pydantic model so typos surface as validation errors, not silently ignored fields. `TomlConfigSettingsSource(deep_merge=True)` overlays nested tables correctly; lists replace (don't concatenate).

Every layer carries a `<layer>_cfg_hash` (8 hashes total — data, features, training, tuning, runs, registry, memory, cv) plus a `root_cfg_hash`. All 9 are written to `user_attrs` per trial and used by the registry to reject promotions whose provenance is incomplete.

Per-layer config schemas live in `src/rux_ml/config/<layer>.py`; the root composition is `RuxMLConfig` in `src/rux_ml/config/root.py`.

---

## The lifecycle

### 1. Data ingest + versioning

Raw datasets live **outside** the repo. The workbench's content-addressed store (`data/cas/`) is a versioned cache, not a source-of-truth.

```bash
# Compute a composite hash without snapshotting (cheap, read-only).
uv run rux-ml --problem demo data hash /abs/path/to/your.parquet
# {
#   "data_hash":         "<bytes_hash>|<logical_hash>",   ← composite
#   "data_bytes_hash":   "11f180...",                     ← SHA-256 over sorted partition bytes
#   "data_logical_hash": "98f8c6..."                      ← SHA-256 over canonical column projection
# }

# Snapshot into the CAS (hardlinks where possible, fallback to copy on cross-device).
uv run rux-ml --problem demo data version demo /abs/path/to/your.parquet

# List versioned datasets.
uv run rux-ml --problem demo data list
uv run rux-ml --problem demo data list --name demo
```

**Where it lives**:
- `src/rux_ml/data/loaders.py` — Polars Parquet loaders + lazy/streaming
- `src/rux_ml/data/versioning.py` — composite `data_hash`, manifest read/write, CAS hardlinks
- `src/rux_ml/data/splits.py` — `train_val_test_split(df, *, ratios, seed)` (one-shot, returns Polars frames)
- `src/rux_ml/data/data_iter.py` — XGBoost `DataIter` for `ExtMemQuantileDMatrix` (out-of-VRAM path)

The composite `data_hash` is **dataset content**; the per-layer `data_cfg_hash` is **the data-layer config** (paths, target column, split ratios). Both end up in `user_attrs`.

### 2. Features

The features layer is a sklearn `Pipeline` orchestrating Polars-expression stateless transforms (wrapped in `FunctionTransformer`) + a stateful `ColumnTransformer` for categorical encoding.

**Categorical encoding decision rule** (per D4):

```
for each categorical column:
  if cardinality(col) <= features.categorical_low_card_threshold:
    Pass through as Polars Categorical → XGBoost enable_categorical=True
  else:
    category_encoders TargetEncoder via NestedCVWrapper
```

`features.categorical_low_card_threshold` has **no default** — set it per problem. Threshold ~10 is a typical starting point for low-card columns; raise it if your dataset has wider Categoricals (zip codes, product IDs) that XGBoost's native handling tolerates.

**Where it lives**:
- `src/rux_ml/features/pipeline.py` — `make_features(cfg, *, cardinalities=...)` factory
- `src/rux_ml/features/encoders.py` — high-card target encoder via `category_encoders.NestedCVWrapper`
- `src/rux_ml/features/cardinalities.py` — `cardinalities_from(df, cols)` helper for the decision rule

### 3. CV strategy (overfit testing)

Repeated CV (used by the Optuna objective) is expressed through a `Splitter` `typing.Protocol` over the **universal subset of the sklearn splitter API**. Five concrete strategies ship at v0, each with its own Pydantic config:

| Strategy | Use case | Leakage guarantees | ExtMem? |
|---|---|---|---|
| `KFoldCV` | IID tabular, balanced target | Each row in exactly one test fold | no |
| `StratifiedKFoldCV` | IID tabular, imbalanced classification target | Class proportions per fold | no |
| `TimeSeriesSplitCV` | Time-indexed (fixed horizon labels) | Train precedes test; `gap` excludes adjacent | **yes** |
| `GroupKFoldCV` | Grouped data (entity ID, session ID) | Each group in exactly one test fold | no |
| `CombinatorialPurgedCV` | Time-indexed with variable / overlapping horizons (financial labels per AFML §7.4.2) | Two-sided label-overlap purge + one-sided post-test embargo | no |

Pick the strategy in your problem config:

```toml
# configs/problems/demo.toml
[cv]
kind     = "stratified_kfold"   # ∈ kfold / stratified_kfold / time_series / group_kfold / cpcv
n_splits = 5
shuffle  = true
```

Time-series + CPCV variants have additional knobs (`gap`, `max_train_size`, `embargo_size`, `purged_size`, `n_test_folds`); see `src/rux_ml/config/cv.py` for the tagged-union schema.

**One-shot vs repeated CV**:
- **One-shot** — `train_val_test_split(df, *, ratios, seed)` for `rux-ml train`. Returns 3 materialized Polars frames. Does NOT consume `cfg.cv`.
- **Repeated** — `Splitter` Protocol via `make_splitter(cfg.cv, seed=bag.cv_seed)`. Returns row-index pairs. Used by the Optuna objective (K-fold CV-mean).

**ExtMem compatibility**: only `TimeSeriesSplitCV` works with `ExtMemQuantileDMatrix` (the out-of-VRAM ingest path). Pairing any other Splitter with ExtMem raises `NotImplementedError` at training time — materialized fallback is a follow-up PR.

**Where it lives**:
- `src/rux_ml/data/cv.py` — Splitter Protocol + 5 concrete strategies + `make_splitter` factory
- `src/rux_ml/config/cv.py` — tagged-union `CVConfig` (one variant per strategy)

### 4. Baseline training

`rux-ml train` runs a single in-process fit + predict on the configured problem. It's recorded as a **1-trial Optuna study** so sweep and one-off runs share the same storage and the same `TrialAttrs` provenance schema.

```bash
uv run rux-ml --problem demo train
# score (auc): 0.901234
#   study:    demo_<study_id>
#   trial:    0
#   storage:  sqlite:///studies/studies.db
#   peak_rss_mb: 234.5
#   entropy_hex: 4323988864d84a1fdff6ed7da001c6a4
#   best_iter: 19
```

**XGBoost ingest path decision rule** (per D3):

```
estimate_X_bytes(data) →
  if X_bytes <= data.gpu_in_memory_x_gb_max (default 18 GB):
    QuantileDMatrix on GPU (device="cuda", tree_method="hist")
  else:
    ExtMemQuantileDMatrix on GPU + host-RAM cache (cache_host_ratio)
```

The threshold is a TOML knob in `[data]`; the host-RAM cache ratio is `[memory].cache_host_ratio` (`null` = XGBoost auto-estimate).

**Where it lives**:
- `src/rux_ml/cli/train.py` — the `run_command` entry point
- `src/rux_ml/training/factory.py` — `make_trainer(cfg, *, seed=None)` returns `XGBClassifier` / `XGBRegressor` (task derived from `cfg.training.metric`)
- `src/rux_ml/training/ingest.py` — `select_ingest(x_bytes, cfg.data)` + `estimate_x_bytes(df)` for the decision rule
- `src/rux_ml/training/metrics.py` — metric registry: AUC / logloss → classifier; RMSE / MAE → regressor

### 5. Hyperparameter tuning

Optuna sweeps with **K-fold CV-mean objective + WilcoxonPruner** by default (per PR-007 Tier-2 research):

```bash
uv run rux-ml --problem demo --study demo_wide tune start
uv run rux-ml --problem demo --study demo_wide tune start --n-trials 100
uv run rux-ml --problem demo tune resume <study_name> --n-trials 50
uv run rux-ml --problem demo tune status <study_name>
uv run rux-ml --problem demo tune retry-trial <study_name> <trial_id>
```

The study config (`configs/studies/<name>.toml`) declares the search space as a tagged-union `SearchSpec`:

```toml
[search_space."training.learning_rate"]
type = "float"
low  = 0.01
high = 0.3
log  = true

[search_space."training.max_depth"]
type = "int"
low  = 3
high = 8

[search_space."training.subsample"]
type = "float"
low  = 0.6
high = 1.0

[search_space."tuning.sampler"]
type    = "categorical"
choices = ["tpe", "gp"]

[tuning]
n_trials          = 100
sampler           = "tpe"            # tpe (default) / gp / hebo
pruner            = "wilcoxon"       # wilcoxon (default) / median / hyperband / successive_halving / none
trial_isolation   = "subprocess"     # subprocess (default) / in_process
trial_timeout_s   = 600              # null disables; subprocess-only
entropy           = 42                # master seed; null auto-pins per run
```

**Objective shape**: each trial calls `make_splitter(cfg.cv, seed=bag.cv_seed)` and runs `cfg.cv.n_splits` fits. Per-fold scores are reported via `trial.report(fold_score, step=fold_idx)` for WilcoxonPruner. The trial returns `statistics.fmean(fold_scores)`.

**Sampler choice**:

| Sampler | When to use |
|---|---|
| `tpe` (default) | XGBoost search spaces with categoricals / conditionals; default per Optuna AutoSampler |
| `gp` | Purely-numerical sub-studies up to 250 trials |
| `hebo` | Opt-in via `optunahub`; raises clear `ImportError` if optional deps missing |

**Pruner choice**: `wilcoxon` is the K-fold CV-mean canonical (per Optuna 3.6+ docs). Use `median` instead if you want TPE to learn from pruned trials (TPE+Wilcoxon has a known interaction documented in Optuna's docs).

**Subprocess-per-trial isolation** is the default: each trial spawns a fresh interpreter via `python -m rux_ml._internal.trial_runner`, pins BLAS/OpenMP/Polars threads BEFORE the heavy imports, runs the trial, exits. This is the CUDA + fork prohibition in action and Optuna's documented OOM cure (the parent never re-uses Python memory between trials).

**Where it lives**:
- `src/rux_ml/cli/tune.py` — `start` / `resume` / `status` / `retry-trial` verbs
- `src/rux_ml/tuning/objective.py` — `build_objective(base_cfg)` returning the closure; K-fold CV body with per-fold pipeline+trainer fits
- `src/rux_ml/tuning/samplers.py` — `make_sampler(cfg.tuning, *, seed)`
- `src/rux_ml/tuning/pruners.py` — `make_pruner(cfg.tuning)`
- `src/rux_ml/tuning/isolation.py` — `run_subprocess_trial` parent-side dispatcher
- `src/rux_ml/_internal/trial_runner.py` — child entry point (strict module-load order; pins env BEFORE heavy imports)

### 6. Run logging + inspection

Optuna SQLite is the single source of truth for trial history. Each trial records the full per-trial provenance triple (8 cfg hashes + data hashes + git SHA + entropy + image digest + library versions + peak RSS) via a Pydantic-validated `TrialAttrs` schema.

```bash
uv run rux-ml --problem demo runs list                    # all trials across studies
uv run rux-ml --problem demo runs list --study <name>     # one study
uv run rux-ml --problem demo runs list --problem demo     # all studies whose name starts with "demo_"
uv run rux-ml --problem demo runs show <trial_number> --study <name>
uv run rux-ml --problem demo runs compare <t1> <t2> [<t3> ...] --study <name>
```

Query the runs from Python:

```python
from rux_ml.runs import list_runs, load_run, compare_runs

df = list_runs("sqlite:///studies/studies.db")               # → pl.DataFrame
run = load_run("sqlite:///studies/studies.db", "demo_xyz", 0) # → Run namedtuple
diff = compare_runs("sqlite:///studies/studies.db", "demo_xyz", [0, 1, 2])
```

**Where it lives**:
- `src/rux_ml/cli/runs.py` — `list` / `show` / `compare` verbs
- `src/rux_ml/runs/attrs.py` — `TrialAttrs` Pydantic schema (required fields gate promotion)
- `src/rux_ml/runs/ask_tell.py` — `one_off_run(cfg, ...)` context manager (shared between `train` and `tune`)
- `src/rux_ml/runs/query.py` — `list_runs` / `load_run` / `compare_runs` (Polars + Pydantic returns)

### 7. Registry promotion + model export

The registry is a **per-problem filesystem store** of promoted bundles, selected via an atomically-rewritten `champion.json`.

```bash
# Promote a specific trial to a new registry version.
uv run rux-ml registry promote --problem demo --study demo_xyz --trial 42
# promoted: demo@v_2026_05_17_a8f3c2
#   source:  study=demo_xyz trial=42
#   champion: registry/demo/champion.json

# List all problems with their current champion + recent versions.
uv run rux-ml registry list

# Roll back to a prior version (atomic champion.json rewrite).
uv run rux-ml registry rollback --problem demo --to v_2026_05_14_111aaa
```

**Bundle layout**:

```
registry/<problem>/
├── champion.json                 atomic pointer to current production version
└── <version>/                    e.g. v_2026_05_17_a8f3c2
    ├── pipeline.skops            sklearn FE Pipeline (skops.io)
    ├── model.ubj                 XGBoost Booster (xgb.save_model)
    └── manifest.json             Pydantic-validated lineage: cfg hashes, data hash,
                                  git SHA, library versions, originating study+trial,
                                  metric value, feature_list_hash
```

**The promotion flow** (re-fit at promote, per PR-010 sub-decision A1):

1. Load the originating trial via `runs.load_run`.
2. Validate `TrialAttrs.from_trial(frozen)` — **raises `pydantic.ValidationError` and refuses promotion** if any required provenance field is missing.
3. Apply `trial.params` to `base_cfg` → `trial_cfg`.
4. **Reconstruct the trial's `SeedBag` from `attrs.entropy_hex`** via `make_seed_bag_from_hex` — the re-fit uses the EXACT same `split_seed` and `xgb_seed` the trial reported metrics for.
5. Re-fit `(pipeline, booster)` on train+val (no CV folds, no per-fold reporting).
6. Compose `ModelManifest` from `TrialAttrs` + library versions + `feature_list_hash`.
7. Save the bundle to `registry/<problem>/<version>/`.
8. Atomic rewrite of `registry/<problem>/champion.json` via tmp + `os.replace`.

**Why the re-fit step exists**: trial bodies are CV (K folds × subset training); promotion needs a single final fit on the full train+val split that's loadable as one Booster. Without the seed reconstruction (step 4), the re-fit would silently produce a different model than the trial's reported metrics — defeating reproducibility.

**Where it lives**:
- `src/rux_ml/cli/registry.py` — `promote` / `list` / `rollback` verbs
- `src/rux_ml/registry/promote.py` — promotion + rollback logic (NOT exported from `rux_ml.registry`'s public `__init__.py` — keeps the inference surface free of training-stack deps)
- `src/rux_ml/registry/bundle.py` — `save_bundle` / `load_bundle`
- `src/rux_ml/registry/champion.py` — atomic champion writes via tmp + `os.replace`
- `src/rux_ml/registry/manifest.py` — Pydantic `ModelManifest` schema

### 8. Loading a model for inference

The loader is **inference-only**: importing `rux_ml.registry` does NOT transitively pull in the training stack (Optuna, sklearn pipelines, XGBoost training code paths). This is enforced by a subprocess test in `tests/registry/test_promote.py::test_registry_inference_deps_separation`.

```python
from rux_ml.registry import load_model

# Default: load the current champion.
pipeline, booster = load_model("demo")

# Or load an explicit version.
pipeline, booster = load_model("demo", version="v_2026_05_17_a8f3c2")

# End-to-end predict on raw input (run the FE pipeline yourself):
import xgboost as xgb
x_transformed = pipeline.transform(raw_df)         # → Polars DataFrame
dmatrix = xgb.DMatrix(x_transformed.to_pandas(), enable_categorical=True)
preds = booster.predict(dmatrix)
```

The loader signature is `load_model(problem, *, version="champion", registry_root=Path("registry"))`. Inference clients depend only on `xgboost`, `skops`, `scikit-learn`, and `polars`.

---

## Reproducibility contract

Every trial — sweep or one-off — has a **per-trial provenance triple** in its `user_attrs` (canonical fields documented in `docs/CONSTRAINTS.md`):

| Field | Source | Notes |
|---|---|---|
| `data_hash` (+ `data_bytes_hash` + `data_logical_hash`) | `rux_ml.data.versioning` | Composite SHA-256 over sorted partition bytes + canonical column projection |
| `data_cfg_hash`, `features_cfg_hash`, `training_cfg_hash`, `tuning_cfg_hash`, `runs_cfg_hash`, `registry_cfg_hash`, `memory_cfg_hash`, `cv_cfg_hash` | per-layer config models (paths/timestamps elided) | 8 hashes |
| `root_cfg_hash` | full config hash (elided fields excluded) | Run identity |
| `git_sha` | `git rev-parse HEAD` | Code version; `uv.lock` + future `Cargo.lock` committed |
| `entropy_hex` | `make_seed_bag(...)` per-trial | 128-bit per-trial entropy; reconstructs the full `SeedBag` via `make_seed_bag_from_hex` |
| `image_digest` | `.docker-image-digest` (or `"unknown"` outside the container) | Set by `make docker-build` |
| `xgboost_version`, `cuda_runtime_version`, `omp_threads` | `xgboost.__version__`, `xgboost.build_info()["CUDA_VERSION"]`, `cfg.memory.omp_threads` | All required |
| `gpu_model`, `driver_version` | `nvidia-smi --query-gpu` | Optional (CPU-only hosts skip) |
| `peak_rss_mb` | `psutil` watchdog | Required |

**SeedBag derivation** (per PR-013): `SeedSequence(entropy=master, spawn_key=(trial.number,))` → 128-bit per-trial entropy → fresh `SeedSequence` → `.spawn(4)` → four `int32`-safe child seeds for `(split, cv, sampler, xgb)`.

**Determinism contracts**:

| Mode | Conditions | Contract |
|---|---|---|
| CPU bit-exact | `device="cpu"` + `tree_method="hist"` + `OMP_NUM_THREADS=1` + pinned `xgb_seed` | `np.testing.assert_array_equal` matches across runs |
| GPU near-determinism | `device="cuda"` + pinned `xgb_seed` | `np.testing.assert_allclose(atol=1e-5, rtol=1e-4)` matches across runs on the same hardware |

GPU is NOT bit-exact across hardware (`docs/CONSTRAINTS.md` Tolerance-Based Golden Tests Only). The golden regression infrastructure in `tests/golden/` uses tolerance bands intentionally so library upgrades don't trigger spurious failures.

---

## Memory & threading

| Mechanism | Where | Default |
|---|---|---|
| OOM hard cap | Container `mem_limit: 32g` (cgroup v2 `memory.max`) | 32 GB |
| OOM soft cap | Container `mem_reservation: 28g` | 28 GB |
| In-process watchdog | `rux_ml._internal.memory.Watchdog` (1 Hz `psutil` RSS sampler) | 28 GB threshold |
| Trial isolation | `subprocess.run` of `python -m rux_ml._internal.trial_runner` | enabled |
| Thread allocation | `OMP_NUM_THREADS=24` / `OPENBLAS=MKL=1` / `POLARS_MAX_THREADS=24` | per-library all-or-one |
| ExtMem host-RAM cache | `MemoryConfig.cache_host_ratio` | `null` = XGBoost auto-estimate |

`pin_threads(cfg.memory)` is the canonical surface — every CLI verb calls it. The subprocess child calls it **before** importing numpy/polars/sklearn/xgboost because those libraries read the env at import-time to size their thread pools (setting them later is a no-op against the existing pool).

`threadpoolctl` is intentionally NOT used — it can't reliably reach across distinct OpenMP runtimes (libgomp vs libiomp).

Sequential trials (D6) plus 24-thread CPU means each library may use the whole CPU when active. BLAS env vars are pinned to 1 to suppress nested oversubscription.

---

## Containerized runtime

The container ships the workbench with a digest-pinned CUDA base + uv-managed Python + tini PID 1 + non-root `rux` user.

```bash
make docker-build        # build rux-ml:local; captures digest to .docker-image-digest
make docker-run ARGS="--problem demo train"
make docker-shell        # bash shell in the container
```

Compose service: `services.rux-ml` with GPU passthrough (`deploy.resources.reservations.devices` — driver: nvidia, count: 1, capabilities: [gpu]) + `mem_limit: 32g` + `mem_reservation: 28g`. Host-bind mounts for `./configs`, `./studies`, `./registry`, `./data`, `./logs` so trials persist outside the container.

**Pinning**:
- Base: `nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:5645fec...e9749`
- uv: `ghcr.io/astral-sh/uv:0.11.14@sha256:1025398...85d97`
- Both digests are resolved from the Docker Hub / GHCR manifest endpoints, not from rolling tags.

The container is required for GPU work on Linux; CPU dev work (Mac or otherwise) doesn't need it. See `docs/CONVENTIONS.md` "Container conventions" for the full set of fixed decisions.

---

## Testing

Default `uv run pytest` excludes the slow / GPU / golden / Docker suites — runs fast on any machine.

```bash
make test               # default (fast)
make test-gpu           # @pytest.mark.gpu — requires CUDA + workbench
make test-golden        # @pytest.mark.golden — tolerance-based regression
make regenerate-golden  # rewrite tests/golden/fixtures/golden_v1/ — MANUAL ONLY (never CI)

# Container smoke tests (gated by RUXML_RUN_DOCKER_TESTS=1 + docker binary):
RUXML_RUN_DOCKER_TESTS=1 uv run pytest -m docker
```

**Registered markers** (`pyproject.toml`):

| Marker | What | Default behavior |
|---|---|---|
| `gpu` | Requires CUDA | Auto-skipped on CPU-only hosts |
| `slow` | > ~5s | Excluded from default |
| `golden` | Tolerance-based regression test | Excluded from default |
| `docker` | Requires Docker daemon + NVIDIA Container Toolkit | Excluded from default |
| `integration` | Multi-component | Included in default |

**Golden test policy**: if `tests/golden/test_xgb_baseline.py` fails, follow the 3-step investigation procedure in `docs/CONVENTIONS.md` "Regenerating golden fixtures" BEFORE running `make regenerate-golden`. Auto-regen masks real regressions.

---

## Development

```bash
make lint     # uv run ruff check .
make format   # uv run ruff format .
make type     # uv run basedpyright src/ tests/
make test     # uv run pytest (default markers)
```

**Code style**:

- Ruff for linting + formatting (config in `pyproject.toml`)
- Basedpyright (strict defaults) for type checking
- Imports: external → internal → relative, ruff-sorted
- `extra="forbid"` on every Pydantic model — typos surface as validation errors

**Module dependency rules** (`docs/CONVENTIONS.md`):

```
            cli/
              │
              ▼
   ┌──── tuning/ ───┐
   │       │        │
   │       ▼        │
   ▼   training/    ▼
runs/      │     registry/
   │       ▼
   │   features/
   │       │
   │       ▼
   └──── data/ ──── config/  (config is leaf upward)
                     │
                     ▼
                _internal/
```

- `cli/` may import any layer; no layer imports `cli/`.
- `_internal/` is importable from everywhere (hashing, env pinning, seeds, memory watchdog).
- Cross-layer calls go through factory functions (`make_features`, `make_trainer`, `make_splitter`).
- `config/root.py` is the only file that imports ALL per-layer configs (prevents cycles).

**Public API** (`src/rux_ml/__init__.py`):

```python
__all__ = ["RuxMLConfig", "Trainer", "load_run", "load_model", "list_runs", "list_registry"]
```

Everything else requires qualified imports (`from rux_ml.tuning import ...`).

**Per-PR procedure** — see `PROCEDURE-pr-research.md`. Every PR runs a state assessment before implementation; non-trivial decisions cite reputable sources. `docs/0.2/ROADMAP.md` is the PR index; `docs/0.2/RESEARCH-BACKLOG.md` tracks per-PR research status.

---

## CLI reference

All verbs accept the global options `--config <path>` (default `configs/base.toml`), `--problem <name>` (overlays `configs/problems/<name>.toml`), `--study <name>` (overlays `configs/studies/<name>.toml`), and `--set <key>=<value>` (repeatable; dot-path overrides applied last). Plus `--verbose` / `-v`, `--dry-run`, `--version`. Values to `--set` are JSON-parsed when possible (numbers, bools, lists, objects); typos in the dot-path surface as `ValidationError` because every Pydantic model has `extra="forbid"`.

```
rux-ml data hash <path>                          Composite data_hash for a Parquet path
rux-ml data version <name> <path>                Snapshot into the CAS + write a manifest
rux-ml data list [--name <n>]                    List versioned datasets

rux-ml train                                     Single baseline; recorded as a 1-trial study

rux-ml tune start [<study_name>] [--n-trials N]  Create or load a study; run N trials
rux-ml tune resume <study_name> [--n-trials N]   Add N more trials to an existing study
rux-ml tune status <study_name>                  Progress + current best
rux-ml tune retry-trial <study_name> <trial_id>  Re-enqueue a failed trial

rux-ml runs list [--study <n>] [--problem <n>]   List trials (Polars table)
rux-ml runs show <trial_number> --study <n>      Params + metric + full provenance for one trial
rux-ml runs compare <t1> <t2> [...]  --study <n> Side-by-side diff across trials

rux-ml registry promote --problem <p> --study <s> --trial <n>   Promote a trial
rux-ml registry list                              All problems + current champion + recent versions
rux-ml registry rollback --problem <p> --to <v>   Atomic champion.json rewrite to a prior version
```

`rux-ml --help` and `rux-ml <verb> --help` list everything Typer exposes; this table is a quick reference, not a substitute.

---

## Doc map

| When you need to know... | Read |
|---|---|
| What the architecture is + how data flows | `docs/ARCHITECTURE.md` |
| Hard rules — what's non-negotiable | `docs/CONSTRAINTS.md` |
| Soft patterns — directory conventions, naming, CV defaults | `docs/CONVENTIONS.md` |
| v0 design rationale (D1–D17) with research trail | `docs/0.0/DESIGN-log.md` <!-- rewrite-doc-refs:skip-line --> |
| Active design conversation log (current version) | `docs/0.2/DESIGN-log.md` |
| PR index with phases + dependencies | `docs/0.2/ROADMAP.md` |
| Per-PR research status + drift watch | `docs/0.2/RESEARCH-BACKLOG.md` |
| How to run a design session | `PROCEDURE-design-planning.md` |
| Mandatory pre-PR research procedure | `PROCEDURE-pr-research.md` |
| Post-design alignment checks | `PROCEDURE-code-audit.md` |
| One PR's spec + research findings | `prs/PR-NNN-*.md` |

The DESIGN-log is the place to start for "why did we choose X over Y?" — every architectural decision is logged with citations.

---

## License

MIT
