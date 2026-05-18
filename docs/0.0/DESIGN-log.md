# Design Log

Design decisions and rationale from planning sessions. Append new sessions below.

---

## Session: 2026-05-14 → 2026-05-15 — Initial Design (D1–D17)

### Context

First design session for `rux-ml`. The project began as a fresh checkout of the `vibe-rails` template (`CLAUDE.md` carried the `<!-- STATUS: uninitialized -->` marker). The user's stated goal was a "structured machine learning training and tuning environment" — refined through Phase 1 questioning into a personal XGBoost-first ML research workbench with primary focus on optimization (time + memory), reuse of battle-tested libraries, conventional standardized pipelines, reproducibility, and flexibility for future model families.

Run on a single Linux desktop (RTX 4090 24 GB VRAM, i9-13900K 24 threads, 36 GB DDR5 RAM — system RAM is the binding constraint) accessed via Tailscale SSH. Solo user; no web UI; no shared infra.

Phase 1 established framing and a 60-day research staleness threshold. Phase 2 made seventeen decisions (D1–D17), each backed by parallel/sequential research rounds via subagents with WebSearch+WebFetch. Phase 3 converged twice — first after D13, then after D17 once the user pointed out that "structure & architecture" had been under-decided. Phase 4 (this doc set) records the converged design.

### User priorities (in order)

1. **Optimization** (time + memory; RAM-bound at 36 GB)
2. **Reuse over reinvent** — battle-tested libraries; custom code (Python or Rust) only when no mature library covers the need
3. **Convention over novelty**
4. **Reproducibility** — every run traceable
5. **Flexibility** — XGBoost today, other model families later

### Decisions

Each decision was preceded by a focused research round (parallel where independent, sequential where dependent), with findings labeled PROVEN / CONVENTION / BEST-GUESS and citations recorded in conversation. The user approved every lean. Below are the locked decisions; full research summaries are linked from the citations.

**D1 — Project skeleton, deps, container, layout (PROVEN)**

- Python dependency manager: **`uv`** ([uv adoption 2026](https://aleyan.com/blog/2026-why-arent-we-uv-yet/), [PyTorch wheel-variants blog](https://pytorch.org/blog/pytorch-wheel-variants/))
- PyO3 build tool: **`maturin`** ([PyO3 docs](https://pyo3.rs/v0.28.0/building-and-distribution.html))
- Workspace pattern: **Cargo workspace at root with flat `crates/`** (polars/ruff pattern, deferred per D13)
- Container base: **`nvidia/cuda:12.4.1-devel-ubuntu22.04`** ([XGBoost CI](https://xgboost.readthedocs.io/en/stable/contrib/ci.html))
- Dev loop: `uv run maturin develop --uv` (when crate exists)

**D2 — Configuration (PROVEN)**

- **Pydantic v2 + pydantic-settings + TOML on disk** ([pydantic-settings docs](https://docs.pydantic.dev/latest/concepts/pydantic_settings/))
- Optuna driven directly in Python — **not** via `hydra-optuna-sweeper` (last PyPI release 4 years ago, [PyPI hydra-optuna-sweeper](https://pypi.org/project/hydra-optuna-sweeper/))
- Hydra's release stagnation (last stable Feb 2023) flagged as conflict; pydantic-settings chosen for active maintenance

**D3 — Data layer (PROVEN)**

- **Polars (lazy + streaming)** as primary DataFrame ([Polars benchmarks](https://pola.rs/posts/benchmarks/))
- **Parquet + zstd** as canonical on-disk format; **Arrow IPC** for hot intermediate caches
- **GPU-first** (user override of CPU-first lean — explicitly approved): `QuantileDMatrix` (`device="cuda"`) when X ≲ 18 GB VRAM, `ExtMemQuantileDMatrix` with `cache_host_ratio` to 36 GB host RAM when larger ([XGBoost ExtMem tutorial](https://xgboost.readthedocs.io/en/stable/tutorials/external_memory.html), [NVIDIA Polars+XGBoost blog](https://developer.nvidia.com/blog/training-xgboost-models-with-gpu-accelerated-polars-dataframes/))
- DuckDB explicitly rejected as primary DataFrame; cuDF rejected (24 GB VRAM cap)

**D4 — Feature engineering (PROVEN)**

- **Hybrid: Polars expressions (stateless) + sklearn `Pipeline` + `ColumnTransformer` (stateful)**; `FunctionTransformer` wraps the Polars block ([sklearn `set_output("polars")`](https://scikit-learn.org/stable/whats_new/v1.4.html))
- **XGBoost `enable_categorical=True`** eliminates manual encoding for low/med-card categoricals ([XGBoost categorical tutorial](https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html))
- **`category_encoders` `NestedCVWrapper`** for very-high-cardinality columns
- Cardinality threshold: TOML knob, **no default** (BEST-GUESS — tune on first dataset)
- AutoGluon / PyCaret / pytorch-tabular rejected (scope-stealing or stalled)

**D5 — Training abstraction (PROVEN)**

- **sklearn estimator API as the contract** + **`typing.Protocol`** for typing; no custom `Trainer` class ([XGBoost sklearn estimator interface](https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html))
- Native `xgb.train()` reserved for the ~5 % cases: `QuantileDMatrix(ref=...)` and `ExtMemQuantileDMatrix`
- Optuna pruning works through the sklearn API via `XGBoostPruningCallback`

**D6 — Hyperparameter tuning (PROVEN)**

- **Optuna 4.x** + **`TPESampler(multivariate=True, group=True, n_startup_trials=20)`** + **`HyperbandPruner`** + `XGBoostPruningCallback` ([Optuna efficient optimization docs](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html))
- **`n_jobs=1` sequential trials** — XGBoost GPU saturates the 4090; concurrent trials cause OOM ([qfournier 2025 blog](https://qfournier.github.io/blog/2025/xgboost/), [XGBoost #6225](https://github.com/dmlc/xgboost/issues/6225))
- **SQLite `RDBStorage`** ([Optuna FAQ](https://optuna.readthedocs.io/en/stable/faq.html))
- HEBO/GP/Wilcoxon-pruner as TOML alternatives, not defaults (BEST-GUESS trial-budget thresholds)
- `scikit-optimize` ruled out (archived)

**D7 — Experiment tracking (PROVEN)**

- **Optuna as the entire experiment log** ([Optuna artifacts tutorial](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/012_artifact_tutorial.html))
- One-off (non-sweep) runs wrapped as 1-trial studies via `study.ask()` + `study.tell()`
- **`optuna.artifacts.FileSystemArtifactStore`** for files
- Environment + reproducibility triple in `user_attrs`
- MLflow deferred — no need for a second DB at v0

**D8 — Model registry (PROVEN)**

- **Filesystem registry**: `registry/<problem>/<version>/{pipeline.skops, model.ubj, manifest.json}` + atomic `champion.json` rewrite ([XGBoost saving_model](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html), [skops persistence](https://skops.readthedocs.io/en/stable/persistence.html), [MLflow alias RFC #10336](https://github.com/mlflow/mlflow/issues/10336))
- **Two-file split** (`pipeline.skops` + `model.ubj`) — combined pickles forbidden
- Version-string format `v_<YYYY>_<MM>_<DD>_<short_hash>` (BEST-GUESS — no canonical convention found)
- Promotion is explicit, never implicit by `study.best_trial`

**D9 — Reproducibility & data versioning (PROVEN)**

- **DIY content-addressed manifest** + composite `data_hash` (bytes_hash + logical_hash) — DVC's MD5-on-bytes shown to be layout-fragile ([Arrow #40202](https://github.com/apache/arrow/issues/40202))
- **`SeedSequence(entropy)` + `.spawn()`** for per-component seeds ([NumPy docs](https://numpy.org/doc/stable/reference/random/bit_generators/generated/numpy.random.SeedSequence.html))
- Commit `uv.lock` + `Cargo.lock`; pin container by **`@sha256:...` digest** ([Docker digests](https://docs.docker.com/dhi/core-concepts/digests/))
- Accept GPU "near-deterministic, not bit-exact" ([XGBoost #8820](https://github.com/dmlc/xgboost/issues/8820), [#5458](https://github.com/dmlc/xgboost/issues/5458))
- Per-trial provenance triple + version logging
- DVC considered but rejected — collaboration features unneeded; MD5-bytes fragile for Parquet
- No canonical "hash a Parquet directory" recipe in industry — conflict flagged

**D10 — Memory & parallelism strategy (PROVEN)**

- Docker `--memory=32g` (cgroup v2 `memory.max`) + `--memory-reservation=28g` ([Netdata cgroups v2](https://www.netdata.cloud/academy/diagnosing-linux-cgroups/))
- `psutil` watchdog at 28 GB (BEST-GUESS threshold) raising `MemoryPressureError → optuna.TrialPruned`
- Skip `resource.setrlimit(RLIMIT_AS)` — unreliable on Linux ([numpy #26551](https://github.com/numpy/numpy/issues/26551), [pynisher #16](https://github.com/automl/pynisher/issues/16))
- Threads: `OMP_NUM_THREADS=24`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `POLARS_MAX_THREADS=24`, XGBoost `nthread=24`
- **Subprocess-per-trial via `ProcessPoolExecutor`/`subprocess.run`** — documented Optuna OOM cure ([Optuna #1178](https://github.com/optuna/optuna/issues/1178))
- `memray attach --aggregate` for on-demand profiling
- "Auto-degrade mid-trial" NOT used (not battle-tested); static pre-trial path only

**D11 — CLI / entry-point design (PROVEN)**

- **Typer + pydantic-settings v2 (`TomlConfigSettingsSource`)** + dot-path CLI overrides
- Single `rux-ml` binary with **nested subcommands**: `data`, `train`, `tune {start, resume, status, retry-trial}`, `runs {list, show, compare}`, `registry {promote, list, rollback}` ([vLLM CLI](https://docs.vllm.ai/en/latest/cli/), [Optuna CLI](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/004_cli.html))
- Resume via `rux-ml tune resume <name> --n-trials N` as sugar over `create_study(load_if_exists=True) + optimize(n_trials=N)`
- `pydantic-settings v2 CliApp` flagged as less mature; Typer+pydantic-settings chosen for battle-testing
- cyclopts considered, deferred (too new)

**D12 — Testing strategy (PROVEN)**

- **pytest + pytest-xdist**; sklearn-style layout (`tests/` mirroring `src/`)
- Registered markers: `gpu`, `slow`, `golden`, `integration` ([pytest good practices](https://docs.pytest.org/en/latest/explanation/goodpractices.html))
- **Tolerance-based golden regression tests** (`np.testing.assert_allclose` + AUC/RMSE bands) — **never exact hashes** per `docs/CONSTRAINTS.md`
- `hypothesis` selectively for transformer invariants (shape/dtype/idempotence)
- `cargo test` + pytest via Makefile (when Rust exists) — no cross-language runner

**D13 — Rust+PyO3 boundary (PROVEN)**

- **No custom Rust at v0** — pydantic-core trajectory ([Pydantic V2 Plan](https://docs.pydantic.dev/2.0/blog/pydantic-v2/)), not polars-from-day-1
- `crates/.gitkeep` reserves the directory; no `Cargo.toml` at root yet (Cargo requires ≥1 member)
- **Trigger:** profiled Python hot path >5 % of a real workbench task (BEST-GUESS threshold)
- First crate likely **`pyo3-polars` expression plugin** for a documented row-dependent transform ([Polars expression plugins](https://docs.pola.rs/user-guide/plugins/expr_plugins/))
- **try/except ImportError with pure-Python fallback** when Rust accelerators land (Flask/ujson convention)
- Use `xxhash`/`blake3` Python bindings before reaching for custom Rust for hashing
- Polars docs aggressively push plugins — counter-evidence from pydantic-core's patient trajectory cited as resolution

**D14 — Repository layout (PROVEN)**

- **`src/rux_ml/` layout** ([PyPA src layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)); top-level `tests/` ([pytest good practices](https://docs.pytest.org/en/latest/explanation/goodpractices.html))
- Runtime-generated dirs at repo root, gitignored: `studies/`, `registry/`, `data/`, `logs/`; `WORKBENCH_HOME` env override
- Closest known precedent: **lightning-hydra-template** ([ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template))
- Inventions (`studies/`, `registry/`, `data/cas/`, `data/manifests/`) mapped to analogs in `docs/CONVENTIONS.md`
- Commit `uv.lock` + `Cargo.lock` ([Cargo FAQ](https://doc.rust-lang.org/cargo/faq.html))
- Option A picked: source data lives outside the repo; no `data/raw/`

**D15 — Python package internal structure (PROVEN)**

- **Layered subpackages** per concern ([PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning/tree/master/src/lightning/pytorch), [Optuna](https://github.com/optuna/optuna/tree/master/optuna), [vLLM](https://github.com/vllm-project/vllm/tree/main/vllm)): `config/`, `data/`, `features/`, `training/`, `tuning/`, `runs/`, `registry/`, `cli/`, `_internal/`
- **`config/` subpackage** with one module per layer + `root.py` composing them (vLLM pattern; [vllm/config/__init__.py](https://github.com/vllm-project/vllm/blob/main/vllm/config/__init__.py)) — prevents import cycles
- **`cli/` subpackage** with one module per verb group ([Typer add_typer](https://typer.tiangolo.com/tutorial/subcommands/add-typer/))
- **Curated public API with explicit `__all__`** (~6 symbols); qualified imports for the rest ([sklearn `__init__.py`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/__init__.py))
- Dependency rule: layers depend downward only; `cli` imports any layer, layers never import `cli`

**D16 — Architectural pattern & data flow (PROVEN)**

- **Hybrid functional + OO**: Optuna-functional objective wrapping sklearn-class trainers ([Optuna pruning tutorial](https://optuna.readthedocs.io/en/v2.0.0/tutorial/pruning.html))
- **Factory functions per layer + explicit CLI wiring**; no DI container ([Better Stack DI guide](https://betterstack.com/community/guides/scaling-python/python-dependency-injection/))
- **Subprocess-per-trial via `subprocess.run` of `python -m rux_ml._internal.trial_runner`** with **spawn semantics** — CUDA + fork is broken ([PyTorch #40403](https://github.com/pytorch/pytorch/issues/40403), [vLLM #8893](https://github.com/vllm-project/vllm/issues/8893))
- **TOML path + JSON override dict** passed to child (not pickled config) — avoids Pydantic pickle edge cases
- **Shared SQLite storage** as parent-child coordination channel ([Optuna distributed tutorial](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html))
- **`search_space: dict[str, SearchSpec]` in TOML** + `base_cfg.model_copy(update=overrides, deep=True)` for trial config ([Hydra Optuna Sweeper](https://hydra.cc/docs/plugins/optuna_sweeper/), [Pydantic frozen `model_copy`](https://github.com/pydantic/pydantic/discussions/4250))

**D17 — Configuration architecture (PROVEN)**

- **Three-tier layered TOML**: `configs/{base.toml, problems/<n>.toml, studies/<n>.toml}` ([pydantic-settings docs](https://pydantic.dev/docs/validation/latest/concepts/pydantic_settings/), [Kedro configuration](https://docs.kedro.org/en/stable/configure/configuration_basics/), [Lightning-Hydra-template](https://github.com/ashleve/lightning-hydra-template))
- `TomlConfigSettingsSource(deep_merge=True)` ([pydantic-settings configuration files](https://deepwiki.com/pydantic/pydantic-settings/3.2-configuration-files))
- **Precedence**: CLI > env (`RUXML_*` with `__` for nesting) > `.env` > study > problem > base > defaults
- **Per-layer hashes** (`data_cfg_hash`, `features_cfg_hash`, `training_cfg_hash`, `tuning_cfg_hash`) + **`root_cfg_hash`**, with documented field elision ([DVC repro](https://dvc.org/doc/command-reference/repro), [MLflow datasets](https://mlflow.org/docs/latest/python_api/mlflow.data.html))
- `extra="forbid"` on every Pydantic model so typos raise validation errors
- Env prefix `RUXML_` (BEST-GUESS — no canonical "project name uppercased" convention beyond intuition)

### BEST-GUESS items the user acknowledged

Per `docs/CONSTRAINTS.md` and the project's research-backed-decisions rule, each is explicitly flagged as not research-derived. Each is configurable in TOML where possible; revisit if it hurts.

| Item | Decision | Owner / mitigation |
|---|---|---|
| 60-day research staleness threshold | Phase 1 framing | User-acknowledged; configurable in `docs/CONSTRAINTS.md` |
| CUDA point release `12.4.1` specifically | D1 | Matches XGBoost CI but my pick within the range |
| Categorical-cardinality cutoff | D4 | TOML knob, **no default** — tune on first dataset |
| HEBO/GP trial-budget threshold ("<50 trials") | D6 | TOML knob; default TPE always |
| Version-string format `v_<YYYY>_<MM>_<DD>_<short_hash>` | D8 | Easy to change; documented in `CONVENTIONS.md` |
| Memory watchdog threshold (28 GB) | D10 | TOML knob; derived as `36 - 4 OS - 4 Docker` |
| Profile trigger ">5 %" for first Rust crate | D13 | TOML knob; documented in ARCHITECTURE.md |
| Option A (source data outside repo) | D14 | Path in TOML; can shift to in-repo `data/raw/` later |
| `configs/` plural vs `conf/` singular | D14 | My pick from plurality-of-templates |
| Exact filenames within subpackages | D15 | Matches conventions but no canonical naming |
| Three layers vs four (`base/env/problem/study`) | D17 | My pick; Kedro uses two, Hydra uses experiment-as-overlay |
| Env var prefix `RUXML_` | D17 | Common-sense "project name uppercased" |
| Hash elision list | D17 | Documented per-field constant; revisit if drift |

### Conflicts in evidence flagged during research

These remain logged for future researchers running `PROCEDURE-pr-research.md`:

- **DuckDB vs Polars** on Parquet scans — both within ~30 %; query shape matters more than engine. Chose Polars on broader fit.
- **Hydra "industry standard" vs Hydra release stagnation** (last stable Feb 2023). Chose Pydantic-settings.
- **MLflow file backend deprecated to KTLO** while older blog posts still recommend it. We don't use MLflow at v0.
- **threadpoolctl fails across distinct OpenMP runtimes** (libgomp vs libiomp). Chose env-var pinning over `threadpoolctl`.
- **No canonical Parquet-directory hashing recipe** — DVC uses MD5 of bytes (fragile), Iceberg uses snapshot id (heavy), Xet uses chunks. Composite hash chosen.
- **Polars plugin docs aggressively push custom plugins** vs pydantic-core's "wait 5 years for measured demand." Chose patient trajectory.
- **CCDS `data/raw|interim|processed|external` convention** vs our `data/cas/` + `data/manifests/`. Different model; documented in `CONVENTIONS.md`.

### Cross-decision interactions noted

- **SQLite (D6) + subprocess-per-trial (D10) + shared storage coordination (D16)** — works because writes are sequential in time; parent waits for `subprocess.run` to exit. Optuna's "no SQLite for parallel writers" warning targets concurrent writers; we are sequential.
- **`data_hash` (D9, dataset content) vs `data_cfg_hash` (D17, data-layer config hash)** — distinct hashes; both stored in `user_attrs`. `config_hash` (D7) aliases `root_cfg_hash` for backward-compat queries.

### Changes to docs

- **`CLAUDE.md`** — rewritten from the `vibe-rails` bootstrap template to project-specific guidance (Phase 4 of `PROCEDURE-design-planning.md`).
- **`docs/ARCHITECTURE.md`** — fully populated with the 17-decision architecture.
- **`docs/CONSTRAINTS.md`** — domain constraints added; 60-day staleness threshold recorded.
- **`docs/CONVENTIONS.md`** — populated with directory, naming, dependency, public API, test, logging, config, and Rust+PyO3 conventions.
- **`docs/ROADMAP.md`** — populated with the phased PR plan (PR-001 through PR-014).
- **`docs/RESEARCH-BACKLOG.md`** — every PR indexed with Tier-1 design-research status and staleness threshold.
- **`prs/PR-001-*.md` through `prs/PR-014-*.md`** — one file per PR copied from `prs/PR-TEMPLATE.md`.

### Procedure compliance notes

- Phase 1 (Idea) — completed with explicit user confirmation of framing + staleness threshold.
- Phase 2 (Decisions) — completed across 17 decisions; every lean tied to a labeled, cited finding; BEST-GUESS items explicitly flagged and user-acknowledged (an additional `feedback-research-discipline` memory was saved after the user pointed out that intuition had crept into early leans).
- Phase 3 (Convergence) — completed twice. First convergence (D1–D13) was reopened when the user noted that project-structure decisions had been undershot, leading to D14–D17.
- Phase 4 (Docs) — this commit. No code yet.
- Phase 5 (Implementation) — pending. PR-001 cannot begin until `PROCEDURE-pr-research.md` Phase 1 (State Assessment) runs against it, per `docs/CONSTRAINTS.md`.

---
