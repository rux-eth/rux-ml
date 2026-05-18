# PR-007: Optuna basics

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-2 PR** (reclassified 2026-05-16 from Tier-1). D6 + D16 cover the high-level architecture (sequential trials, SQLite storage, TPE+Hyperband defaults), but two design surfaces are unresearched:
1. **CV integration depth in the objective** — PR-015 (Tier-2) was inserted between PR-006 and PR-007 *after* the original design session; how the objective consumes the Splitter (single-fit vs CV-mean vs nested-CV; pruning-with-CV interaction) was not covered by D6/D16.
2. **Sampler/pruner budget thresholds** — D6 itself labels HEBO/GP/Wilcoxon as "BEST-GUESS trial-budget thresholds." When TPE saturates, when GP/BoTorch are worth the dep, which pruner pairs with XGBoost early stopping — none cited.

Tier-2 status inherited via the project pattern: a PR that consumes a Tier-2 abstraction (here, the PR-015 Splitter) inherits Tier-2 status because the integration design is itself unresearched.

## Research findings

### State Assessment (2026-05-16, LOCAL-ONLY)

**Current state of the codebase**:

- `dev` at `9703b25` (PR-015 merged + roadmap fix). Working tree clean.
- `src/rux_ml/tuning/` does not exist. `src/rux_ml/cli/tune.py` has 4 stub subcommands (`tune start/resume/status/retry-trial`) all calling `not_implemented(..., "PR-007")`.
- `TuningConfig` (PR-002) pins every field PR-007 consumes: `sampler ∈ {tpe, gp, botorch, hebo}`, `pruner ∈ {hyperband, median, successive_halving, wilcoxon, none}`, `n_trials=50`, `n_startup_trials=20`, `multivariate=True`, `group=True`, `trial_isolation` (deferred to PR-008), `entropy: int | None` (PR-013 will replace with SeedSequence-derived per-component seed).
- `SearchSpec` tagged union already in `config/tuning.py` (PR-002): `FloatSpec | IntSpec | CatSpec` discriminated on `type`. `RuxMLConfig.search_space: dict[str, SearchSpec]` already loads from TOML.
- **PR-006 surface** PR-007 inherits: `_build_user_attrs`, `_HASH_LAYERS` (with `cv` added by PR-015 — 8 layers), `_data_hashes`, `_ensure_storage_parent`, `_study_name`, `_fit_and_score` — all private inside `cli/train.py`.
- **PR-015 surface** PR-007 consumes: `make_splitter(cfg.cv, *, seed) -> Splitter`; `splitter.extmem_compatible: ClassVar[bool]`; the 5 concrete strategies. ExtMem-incompatible Splitter + ExtMem ingest pairing must raise `NotImplementedError` at trial-build time (per PR-015 sub-decision C1; PR-007's objective is the place that gate actually executes).
- Optuna 4.8.0 already installed (PR-006 added it).

**Local API verification (Optuna 4.x — installed library):**

| Item | Where | Status |
|---|---|---|
| `TPESampler`, `GPSampler`, `RandomSampler` | `optuna.samplers` | ✅ import OK |
| `HyperbandPruner`, `WilcoxonPruner`, `MedianPruner`, `SuccessiveHalvingPruner`, `NopPruner` | `optuna.pruners` | ✅ import OK |
| `RetryFailedTrialCallback` | `optuna.storages` | ✅ import OK |
| `XGBoostPruningCallback` | `optuna.integration` / `optuna_integration` | ❌ **needs `optuna-integration[xgboost]`** as runtime dep (Optuna 4.x split out integration into a separate package) |
| `BoTorchSampler` | `optuna.integration` | ❌ needs `optuna-integration[botorch]` (same pattern) |

**Stale assumptions in the PR-007 file (drafted 2026-05-14, pre-PR-006 + pre-PR-015):**

1. **`XGBoostPruningCallback` import path** — the spec's data-flow snippet imports from `optuna.integration` directly. Stale: Optuna 4.x ships it via the separate `optuna-integration[xgboost]` package.
2. **Pruning callback passed to `fit(..., callbacks=[...])`** — PR-006 surfaced this. XGBoost 3.x moved `callbacks` to constructor / `set_params`, not a fit kwarg. PR-007 wires pruning via `trainer.set_params(callbacks=[XGBoostPruningCallback(...)])` after `make_trainer`.
3. **Objective body says "single fit"** — the spec scope reads "invokes `make_data` / `make_features` / `make_trainer` / `fit`" (one fit per trial). The ROADMAP entry, the `project-cv-strategy-tier2` memory, and the planning commit `3b5c1a1` all say PR-007 "Consumes the Splitter from PR-015". **This is the spec inconsistency Tier-2 research must resolve.**
4. **BoTorch / HEBO listed as alternates** — both require extra optional deps; D6 already labeled them BEST-GUESS. Tier-2 research must resolve "when is the extra dep worth it."

**New constraints learned from PR-001 → PR-015:**

1. **Provenance helpers are private to `cli/train.py` today** (PR-006). PR-007 needs the same `_build_user_attrs`, `_HASH_LAYERS`, `_data_hashes` for every trial in the sweep. Either share them (extract into a module) or duplicate.
2. **`cv_cfg_hash` is now an 8th layer** (PR-015). Reusing PR-006's helpers picks this up for free.
3. **`make_splitter(cfg.cv, *, seed=...)`** (PR-015) is the integration point; `splitter.extmem_compatible` gates the ExtMem error.
4. **`make_trainer` already returns a `Trainer` Protocol** (PR-006) that conforms structurally; PR-007 just calls `trainer.set_params(callbacks=...)` on the returned instance.
5. **`cfg.tuning.entropy: int | None`** is the single-int placeholder PR-013 will replace; PR-007 uses it directly as the sampler `seed=` arg (same hand-off pattern PR-006 used for the split seed).
6. **`configs/studies/` directory doesn't exist yet** but `RuxMLConfig.from_layers` handles its absence; PR-007 may create the dir and add a tiny example TOML.

### Research Questions (Phase 2, 2026-05-16)

Per `feedback_research_discipline` memory: every recommendation must be tied to a labeled cited finding. Tier-2 = full Phase 3 web research is non-negotiable per `PROCEDURE-pr-research.md`.

**Must-answer:**

| # | Question | Success criterion |
|---|---|---|
| **Q1.a** | For tabular-GBM (XGBoost) HPO with Optuna in 2026, what is the canonical objective shape — single fit on a train/val split, K-fold CV with mean aggregation, or nested CV? | Recommendation + ≥1 alternative + cited evidence (≥1 production HPO pipeline post-2024 + ≥1 sklearn/XGBoost-canonical reference); recommendation labeled `proven`/`convention`/`best-guess-given-constraints`. |
| **Q1.b** | If CV-mean is recommended: how do production HPO pipelines aggregate fold scores (mean / median / robust mean / Wilcoxon-aware)? | Named function + ≥1 cited production example + a clear "use X when…" rule. |
| **Q2.a** | How does pruning interact with K-fold CV when K folds are evaluated within one trial? Does HyperbandPruner / MedianPruner work at the fold level, or only at the iteration level within one fit? Does WilcoxonPruner replace them for CV objectives? | Per-pruner interaction note + cited Optuna doc / issue / canonical example for CV-pruning. |
| **Q3.a** | At what trial budget does TPE saturate vs GP / BoTorch? D6 marked this BEST-GUESS — resolve with 2024+ cited benchmarks on tabular HPO (Optuna paper, qfournier 2025, NeurIPS benchmark tracks, sklearn HPO bake-offs). | Cited budget thresholds (with the benchmark data shape they were measured on) + a "lean toward sampler X when trial budget is Y" rule. |
| **Q3.b** | When is BoTorch (or HEBO via BoTorchSampler) worth the `optuna-integration[botorch]` dep cost vs TPE for tabular-GBM HPO? | Cited production preference + decision rule + explicit "the dep is not worth installing unless…" line. |
| **Q4.a** | For XGBoost-with-early-stopping HPO, what's the current default pruner choice (Hyperband / SuccessiveHalving / Median / Wilcoxon)? D6 picked Hyperband as default but flagged Wilcoxon as experimental — confirm or amend with 2024+ practice. Address both regimes (single-fit AND CV-mean) independently. | Recommendation per regime + cited evidence (Optuna 4.x docs + ≥1 production HPO pipeline). |

**Dependencies:**
- Q1.b depends on Q1.a (only matters if CV chosen).
- Q2.a depends on Q1.a (CV-pruning only matters if CV chosen).
- Q3.a / Q3.b independent of Q1 / Q2.
- Q4.a touches both regimes; agent addresses each independently.

**Dispatch plan**: three parallel general-purpose web-research agents:
- **Agent A** — Q1.a + Q1.b + Q2.a (objective shape + CV-pruning interaction)
- **Agent B** — Q3.a + Q3.b (sampler budget thresholds — D6 BEST-GUESS resolution)
- **Agent C** — Q4.a (pruner choice for both objective regimes)

**Explicitly excluded from this round** (covered or deferred):
- Optuna `study.ask()` / `study.tell()` / `study.optimize()` API mechanics — already exercised in PR-006.
- Constructor-vs-`fit` placement of `XGBoostPruningCallback` — PR-006 already resolved (constructor side).
- Subprocess-per-trial isolation — PR-008.
- Run-logging schema / `peak_rss_mb` — PR-009 / PR-011.
- Entropy / SeedSequence wiring — PR-013.
- Provenance-helper extraction location (`runs/provenance.py` vs inline) — code-organization judgment call, not research.

### Findings (Phase 3, 2026-05-16)

Three parallel web-research agents ran 2026-05-16 against the question scope above. Per-question detail (options + sources + disconfirming-evidence-sought + recommendations) lives in this PR's design conversation; the cross-question summary is preserved here.

| Q | Recommendation | Status | Top source |
|---|---|---|---|
| Q1.a | **K-fold CV-mean** is the canonical objective shape (not single-fit; not nested CV) | **convention** | [NVIDIA Kaggle Grandmasters Playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/) + [Optuna `OptunaSearchCV` defaults](https://optuna-integration.readthedocs.io/en/stable/reference/generated/optuna_integration.OptunaSearchCV.html) |
| Q1.b | **Arithmetic mean** of fold scores as returned value; median only after measured outlier evidence | **convention** (mean) / **best-guess** (robust aggregates — single-source) | Optuna `OptunaSearchCV.mean_test_score` |
| Q2.a | **WilcoxonPruner at fold granularity** + XGBoost-internal `early_stopping_rounds` per fold. **Do NOT use `XGBoostPruningCallback` inside CV** — Optuna #3203 documents the duplicate-step warnings + no proper pruning when the callback is used per-fold | **proven** | [Optuna #3203](https://github.com/optuna/optuna/issues/3203) + [WilcoxonPruner tutorial](https://optuna.readthedocs.io/en/latest/tutorial/20_recipes/013_wilcoxon_pruner.html) |
| Q3.a | **TPESampler default** (`multivariate=True`, `constant_liar=True`); GPSampler available for purely-numerical sub-studies up to 250 trials. XGBoost search spaces always include categoricals/conditionals → routes to TPE per Optuna's AutoSampler logic. **AutoSampler hard-codes `_MAX_BUDGET_FOR_SINGLE_GP=250`** as the GP→post-GP boundary. | **convention** | [AutoSampler source](https://raw.githubusercontent.com/optuna/optunahub-registry/main/package/samplers/auto_sampler/_sampler.py) + [Optuna 4.8 samplers reference](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html) |
| Q3.b | **Skip `optuna-integration[botorch]` entirely** — Optuna 3.6 release blog explicitly motivates GPSampler with "~5× faster than BoTorchSampler"; no cited tabular-GBM advantage. **HEBO via `optunahub`** is the documented optional escalation. | **best-guess-given-constraints** (thin production-preference evidence for HEBO over TPE on tabular GBM) | [Optuna 3.6 release blog](https://medium.com/optuna/announcing-optuna-3-6-f5d7efeb5620) + [HEBO arXiv 2012.03826](https://arxiv.org/abs/2012.03826) + [Kégl HPO systematic study](https://balazskegl.medium.com/navigating-the-maze-of-hyperparameter-optimization-insights-from-a-systematic-study-6019675ea96c) |
| Q4.a (CV-mean regime) | **WilcoxonPruner default** (purpose-built for CV-mean; experimental v3.6+); **MedianPruner as conservative fallback** (Optuna's own `xgboost_cv_integration.py` example uses MedianPruner — production-grade cite) | **best-guess-given-constraints** (WilcoxonPruner API experimental) | [Optuna 3.6 release: "k-fold cross validation score of a machine learning model"](https://medium.com/optuna/announcing-optuna-3-6-f5d7efeb5620) + [xgboost_cv_integration.py](https://github.com/optuna/optuna-examples/blob/main/xgboost/xgboost_cv_integration.py) |
| Q4.a (single-fit regime, hypothetical) | HyperbandPruner + `XGBoostPruningCallback` per boosting round — **not the default regime** in PR-007 per Q1.a's CV-mean choice; preserved in the literal for future single-fit objective variants | **convention** | [Optuna 4.8 efficient_optimization_algorithms tutorial](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html) |

**Cross-agent convergence to flag:**

- **Agents A and C independently converged on WilcoxonPruner** for the CV-mean regime, both citing the same Optuna 3.6 language ("k-fold cross-validation score of a machine learning model"). Both flagged the XGBoostPruningCallback × CV-fold incompatibility ([Optuna #3203](https://github.com/optuna/optuna/issues/3203)). Strong signal in the highest-uncertainty area of this PR.
- **Agent B identified BoTorchSampler was deprecated** for single-objective HPO in Optuna 3.6 (Mar 2024) — this resolves D6's "BoTorch as alternate" ambiguity by removing it from the option table entirely.
- **Agent C surfaced a TPE+Wilcoxon caveat**: "TPESampler currently cannot utilize the information of pruned trials effectively" under WilcoxonPruner. Real tradeoff, but the alternative (non-TPE sampler) costs more on the categorical-routing front; accept and document.

### Synthesis (Phase 4, 2026-05-16)

**Outcome: Amend** — the PR-007 spec body was built on D6's TPE+Hyperband framing rooted in iterative-DL pruning intuition that doesn't transfer cleanly to the CV-mean objective shape. Research locks in a different design.

**User-approved sub-decisions (2026-05-16):**

- **A1** — WilcoxonPruner default + MedianPruner conservative alternate; both selectable via `cfg.tuning.pruner`. Accept the experimental-API risk.
- **B1** — Narrow `TuningConfig.sampler` literal to `["tpe", "gp", "hebo"]` (drop `botorch`); change `pruner` default to `"wilcoxon"`; add `constant_liar: bool = True` field.
- **C1** — HEBO is an opt-in escalation; the sampler factory raises a clear `ImportError` with a `pip install optunahub hebo` pointer when `cfg.tuning.sampler="hebo"` is selected without the optional deps installed. `optunahub` / `hebo` NOT added to default runtime deps.
- **D1** — Extract `_HASH_LAYERS`, `_build_user_attrs`, `_data_hashes`, `_ensure_storage_parent`, `_study_name` from `src/rux_ml/cli/train.py` into a new public `src/rux_ml/runs/provenance.py`. Both `cli/train.py` (PR-006) and `cli/tune.py` (PR-007) import from there. PR-009 (run logging) inherits the foundation.

**Substantive amendments to PR-007 spec from research:**

1. **Objective regime: single-fit → K-fold CV-mean.** Every trial calls `make_splitter(cfg.cv, seed=…)`, iterates `splitter.split(X, y, groups=…)`, fits + scores per fold, reports per-fold via `trial.report(fold_score, fold_idx)` to feed WilcoxonPruner, and returns `mean(fold_scores)` to `study.tell`.
2. **Pruner default: Hyperband → Wilcoxon** for the CV-mean regime. Hyperband stays in the literal for hypothetical future single-fit objectives.
3. **`XGBoostPruningCallback` removed from the objective entirely.** `early_stopping_rounds` runs per fold (XGBoost-internal, already wired by PR-006's `make_trainer`); nothing bridges back to Optuna. The objective is "K independent fits, per-fold report, mean return."
4. **`BoTorchSampler` rejected** (deprecated in Optuna 3.6); HEBO opt-in via `optunahub` (sub-decision C1).
5. **`TuningConfig` schema adjustments** per sub-decision B1.
6. **ExtMem × CV consequence** (inherited from PR-015 sub-decision C1): every CV-mean trial does K fits; for the ExtMem path only `TimeSeriesSplitter` is compatible. PR-007's objective checks `splitter.extmem_compatible` at trial-build time and raises `NotImplementedError` for incompatible pairings. **This is the place that gate actually executes** (PR-015 deferred it to here).

**Changes to `docs/ARCHITECTURE.md`** (same commit): rewrite "Decision Rules: Optuna sampler / pruner (per D6)" with the research-locked answer; D6's framing was based on iterative-DL/early-stopping intuition that didn't transfer to CV-mean.

**Changes to `docs/CONVENTIONS.md`** (same commit): add per-fold-reporting pattern + "do not use `XGBoostPruningCallback` inside CV" note.

**Changes to `docs/CONSTRAINTS.md`**: None (per-trial provenance triple unchanged).

**No new prerequisite PRs surfaced.** PR-008 (subprocess isolation) and PR-013 (seed management) remain correctly sequenced downstream.

### Gate Check (Phase 5, 2026-05-16)

- Premise still valid: ✓ (HPO via Optuna is right; only the specifics changed)
- No prerequisite PRs surfaced: ✓
- User approved locked-in spec: ✓ (2026-05-16; sub-decisions A1, B1, C1, D1)
- Implementation cleared: ✓ (2026-05-16)

---

## Scope (locked-in after Phase 3 research — 2026-05-16)

Wire up Optuna 4.x for HPO per D6 + D16 + the research-locked direction: **sequential trials, in-process, SQLite storage, K-fold CV-mean objective consuming PR-015's Splitter, WilcoxonPruner default with MedianPruner fallback, TPESampler default**, with the `tune` CLI verb group functional.

**Config schema changes (`src/rux_ml/config/tuning.py`):**
- Narrow `sampler: Literal["tpe", "gp", "botorch", "hebo"]` → `Literal["tpe", "gp", "hebo"]` (drop `botorch` — deprecated in Optuna 3.6 per Q3.b finding).
- Change `pruner` default from `"hyperband"` → `"wilcoxon"` (CV-mean regime per Q4.a).
- Add `constant_liar: bool = True` (TPESampler option per AutoSampler convention).
- Keep `pruner` literal at its current 5-way set (`hyperband` / `median` / `successive_halving` / `wilcoxon` / `none`) so users can swap MedianPruner in as the conservative alternate.

**Provenance helper extraction (sub-decision D1, `src/rux_ml/runs/provenance.py` — new module):**
- Extract `_HASH_LAYERS`, `_build_user_attrs(cfg, data_hashes)`, `_data_hashes(source_path)`, `_ensure_storage_parent(url)`, `_study_name(cfg, opts)` from PR-006's `src/rux_ml/cli/train.py` into a new public module.
- `src/rux_ml/cli/train.py` updated to import from `rux_ml.runs.provenance`.
- Both `cli/train.py` (1-trial) and `cli/tune.py` (sweep) call the same helpers; PR-009 inherits the foundation.

**Tuning package (`src/rux_ml/tuning/` — new):**
- `samplers.py` — `make_sampler(cfg: TuningConfig, *, seed: int | None) -> BaseSampler`:
  - `"tpe"` → `TPESampler(multivariate=cfg.multivariate, group=cfg.group, n_startup_trials=cfg.n_startup_trials, constant_liar=cfg.constant_liar, seed=seed)`
  - `"gp"` → `GPSampler(seed=seed)`
  - `"hebo"` → lazy import via `optunahub.load_module("samplers/hebo")` with clear `ImportError` ("install `optunahub` + `hebo`") on missing dep (sub-decision C1).
- `pruners.py` — `make_pruner(cfg: TuningConfig) -> BasePruner`:
  - `"wilcoxon"` (default) → `WilcoxonPruner(p_threshold=0.1, n_startup_steps=2)`
  - `"median"` → `MedianPruner(n_startup_trials=cfg.n_startup_trials)`
  - `"hyperband"` → `HyperbandPruner()` (preserved; pairs with future single-fit objectives)
  - `"successive_halving"` → `SuccessiveHalvingPruner()`
  - `"none"` → `NopPruner()`
- `study.py` — `create_or_load(name: str, storage: str, sampler: BaseSampler, pruner: BasePruner, direction: str, load_if_exists: bool = True) -> Study`. Thin wrapper around `optuna.create_study` ensuring the storage parent dir exists (via `runs/provenance.py`).
- `objective.py`:
  - `walk_search_space(search_space: dict[str, SearchSpec], trial: Trial) -> dict[str, Any]` — translates each `SearchSpec` variant (`FloatSpec` / `IntSpec` / `CatSpec`) into a `trial.suggest_*` call. Dot-path keys (`training.learning_rate`) unflatten via the existing `_unflatten_dot_paths` from `config/root.py`.
  - `build_objective(base_cfg: RuxMLConfig) -> Callable[[Trial], float]` — closure that:
    1. computes `overrides = walk_search_space(base_cfg.search_space, trial)`
    2. builds `trial_cfg = base_cfg.model_copy(update=overrides, deep=True)` (per D16 trial-config derivation)
    3. records the per-trial `user_attrs` via `provenance.build_user_attrs(trial_cfg, data_hashes)`
    4. loads data + features pipeline once (outside the fold loop; features re-fit per fold for leakage hygiene)
    5. **builds the Splitter via `make_splitter(trial_cfg.cv, seed=trial_cfg.tuning.entropy)`**; checks `splitter.extmem_compatible` against the data ingest path and raises `NotImplementedError` for incompatible pairings (per PR-015 sub-decision C1)
    6. iterates `(train_idx, test_idx)` from the Splitter, fits the features pipeline + trainer per fold (XGBoost-internal `early_stopping_rounds` runs inside each fold; no `XGBoostPruningCallback`)
    7. computes per-fold score, reports `trial.report(fold_score, fold_idx)` (feeds WilcoxonPruner), checks `trial.should_prune()` and raises `optuna.TrialPruned()` if so
    8. returns `mean(fold_scores)` (arithmetic mean per Q1.b)

**CLI bodies (`src/rux_ml/cli/tune.py`):**
- `tune start <study-name> --n-trials N` → load `RuxMLConfig` → build sampler/pruner → `create_or_load(...)` with `direction=optuna_direction(cfg.training.metric)` → `study.optimize(build_objective(base_cfg), n_trials=N)` (in-process; subprocess isolation lands in PR-008)
- `tune resume <study-name> --n-trials N` → same call (idempotent via `load_if_exists=True`)
- `tune status <study-name>` → print `n_trials_completed`, `study.best_value`, `study.best_trial.params`, `study.best_trial.user_attrs`
- `tune retry-trial <study-name> <trial-id>` → load failed trial; `study.add_trial(FixedTrial(prior_params))` with the same param dict; surface the resulting score

**Example study TOML (`configs/studies/example.toml` — new):**
- Illustrative `[search_space]` for a typical XGBoost sweep (learning_rate FloatSpec log, max_depth IntSpec, subsample/colsample_bytree FloatSpec, n_estimators IntSpec)
- `[tuning]` block showing sampler=tpe, pruner=wilcoxon, n_trials=50
- `[cv]` block showing the default `kind="kfold", n_splits=5`

**Runtime deps:**
- Add `optuna-integration[xgboost]` so `XGBoostPruningCallback` is available for the (non-default) single-fit regime. Note: NOT used in the CV-mean default path, but ships now so callers who pick a future single-fit pruner don't get a confusing missing-import error.

**Test coverage:**
- `tests/runs/test_provenance.py` — Round-trip the extracted helpers; verify hash-layer set matches PR-006's via the 8-layer expected set.
- `tests/tuning/test_samplers.py` — `make_sampler` returns the right class for each kind; `"hebo"` raises clear ImportError without `optunahub`; seed flows through.
- `tests/tuning/test_pruners.py` — `make_pruner` returns the right class for each kind; default is WilcoxonPruner; MedianPruner picks up `n_startup_trials` from cfg.
- `tests/tuning/test_objective.py` — `walk_search_space` translates each `SearchSpec` variant via a fake Trial; `build_objective` end-to-end on a tiny synthetic CV objective; ExtMem-incompatible pairing raises `NotImplementedError`.
- `tests/tuning/test_study.py` — `create_or_load` builds + persists to SQLite; idempotent reload.
- `tests/cli/test_tune_subcommands.py` — `rux-ml tune start <name> --n-trials 5` end-to-end on a synthetic Parquet, study persists in SQLite, best_trial reachable, `user_attrs` includes the full 8-layer provenance.
- GPU-gated (`@pytest.mark.gpu`) — same with `--set training.device=cuda`.
- Pruning sanity: a contrived objective whose later folds always score worst should actually trigger `WilcoxonPruner` and `optuna.TrialPruned` (test that pruning fires at all; we're not benchmarking statistical power).

**Stub-removal:**
- `tests/cli/test_subcommands.py` — drop the four `tune {start,resume,status,retry-trial}` rows from the "not yet implemented" parametrize list (same pattern as PR-006 did for `train`).

**Docs riding with this PR:**
- `docs/ARCHITECTURE.md` "Decision Rules: Optuna sampler / pruner" — rewrite to reflect the CV-mean default + WilcoxonPruner + skip-BoTorch + HEBO-opt-in research findings.
- `docs/CONVENTIONS.md` — new "HPO objective shape" subsection: K-fold CV-mean + per-fold reporting + "do not use `XGBoostPruningCallback` inside CV" note.
- `docs/0.0/RESEARCH-BACKLOG.md` — PR-007 row: `fully-researched 2026-05-16` + `implementation-cleared 2026-05-16`.
- `docs/0.0/ROADMAP.md` — flip PR-007 row `[ ]` → `[x]` **in this PR's commit** (per `feedback-roadmap-flip-in-pr` memory — don't miss this time).

**NOT in scope:**
- Subprocess-per-trial isolation (PR-008).
- Runs query/compare CLI bodies (PR-009).
- Registry promote (PR-010).
- `peak_rss_mb` watchdog (PR-011).
- `entropy_hex` / full SeedSequence spawn (PR-013) — PR-007 uses `cfg.tuning.entropy` directly as a single int per the PR-006 hand-off pattern.
- Multi-objective Optuna.
- Distributed HPO across machines.
- `BoTorchSampler` reinstatement.

## Dependencies

PR-006.

## Architecture section implemented

`docs/ARCHITECTURE.md` → "Tuning" component row, "Decision Rules: Optuna sampler / pruner", "Key Abstractions: `RuxMLConfig` and `SearchSpec`".

## Verification criteria (locked-in 2026-05-16)

- [ ] `TuningConfig.sampler` literal narrowed to `["tpe", "gp", "hebo"]`; `pruner` default flipped to `"wilcoxon"`; `constant_liar: bool = True` added (sub-decision B1)
- [ ] `src/rux_ml/runs/provenance.py` exists with public `HASH_LAYERS`, `build_user_attrs`, `data_hashes`, `ensure_storage_parent`, `study_name`; `cli/train.py` imports from it (sub-decision D1)
- [ ] `src/rux_ml/tuning/{samplers,pruners,study,objective}.py` exist with the locked signatures above
- [ ] `make_sampler(cfg, seed)` returns the right class for each kind; `"hebo"` raises `ImportError` with `pip install optunahub hebo` pointer when `optunahub` isn't installed (sub-decision C1)
- [ ] `make_pruner(cfg)` returns the right class; default is `WilcoxonPruner` (sub-decision A1); `MedianPruner` selectable via `cfg.tuning.pruner="median"`
- [ ] `walk_search_space` translates each `SearchSpec` variant (`FloatSpec` / `IntSpec` / `CatSpec`) to the right `trial.suggest_*` call
- [ ] `build_objective(base_cfg)` returns a closure that:
  - builds a Splitter via `make_splitter(trial_cfg.cv, seed=trial_cfg.tuning.entropy)`
  - raises `NotImplementedError` when `cfg.data` selects ExtMem AND `not splitter.extmem_compatible` (the PR-015-deferred gate executing here)
  - iterates folds, reports per-fold via `trial.report(fold_score, fold_idx)`, checks `trial.should_prune()`, returns `mean(fold_scores)`
  - records the 8-layer per-trial `user_attrs` set (via `runs/provenance.build_user_attrs`)
  - does **NOT** wire `XGBoostPruningCallback` (Optuna #3203 incompatibility with CV; XGBoost-internal `early_stopping_rounds` runs per fold)
- [ ] `rux-ml tune start <name> --n-trials 5` runs to completion against synthetic data; study persists in SQLite at `cfg.runs.storage_url`
- [ ] `rux-ml tune resume <name> --n-trials 3` adds 3 more trials (8 total) without losing the original 5 (idempotent `load_if_exists=True`)
- [ ] `rux-ml tune status <name>` reports trial counts + best value + best `user_attrs`
- [ ] `rux-ml tune retry-trial <name> <id>` re-runs a failed trial with its prior params via `study.add_trial(FixedTrial(...))`
- [ ] `configs/studies/example.toml` exists with a typical XGBoost search space + Wilcoxon pruner + `[cv] kind="kfold"` block
- [ ] `tests/cli/test_subcommands.py` `tune {start,resume,status,retry-trial}` rows removed from the stub list (matches PR-006's `train` removal pattern)
- [ ] WilcoxonPruner pruning actually fires on a contrived objective whose later folds always under-perform (test that pruning is wired correctly, not benchmarking statistical power)
- [ ] GPU-gated integration test (`@pytest.mark.gpu`) passes with `--set training.device=cuda`
- [ ] `docs/ARCHITECTURE.md` "Decision Rules: Optuna sampler / pruner" rewritten with the research-locked direction
- [ ] `docs/CONVENTIONS.md` gets the "HPO objective shape" subsection
- [ ] `docs/0.0/RESEARCH-BACKLOG.md` PR-007 row marked `fully-researched 2026-05-16` + `implementation-cleared 2026-05-16`
- [ ] `docs/0.0/ROADMAP.md` PR-007 row flipped `[ ]` → `[x]` **in this PR's commit**
- [ ] `uv run pytest`, `uv run ruff check .`, `uv run basedpyright src/ tests/` all green

## Research backing

Tier 1:

- D6: [Optuna efficient optimization docs](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html), [qfournier 2025](https://qfournier.github.io/blog/2025/xgboost/), [Optuna FAQ on storage](https://optuna.readthedocs.io/en/stable/faq.html)
- D16: [Optuna pruning tutorial](https://optuna.readthedocs.io/en/v2.0.0/tutorial/pruning.html), [Hydra Optuna Sweeper precedent](https://hydra.cc/docs/plugins/optuna_sweeper/), [Pydantic frozen `model_copy`](https://github.com/pydantic/pydantic/discussions/4250)

State assessment must verify Optuna 4.x sampler/pruner/callback APIs and `XGBoostPruningCallback` import path are unchanged.

## Notes

- Run trials in-process here; subprocess isolation is its own PR (PR-008) so we can debug Optuna-specific issues separately from the subprocess machinery.
- Per D6, `WilcoxonPruner` is "experimental" — expose it but mark in TOML comments.
