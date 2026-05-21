# Design Log — v0.3

Design decisions and rationale from planning sessions for the v0.3 cut. Append new sessions below.

The canonical record of v0.2 design (D1–D7 + A1–A5 deferred ambiguities) lives in [`docs/0.2/DESIGN-log.md`](../0.2/DESIGN-log.md). Decisions and constraints from v0.2 remain in force unless an entry in this log explicitly supersedes them. v0.1 design at [`docs/0.1/DESIGN-log.md`](../0.1/DESIGN-log.md). The frozen v0.0 record is at [`docs/0.0/DESIGN-log.md`](../0.0/DESIGN-log.md).

---

## Session: 2026-05-20 — v0.3 sprint scoping (PR-030 scaffolding)

### Context

The v0.3 sprint was scoped from a **two-pass phantom-implementation audit** run on 2026-05-20. The audit was prompted by an attempt to exercise `rux-ml`'s least-tested workbench surface (promotion + scoring on a held-out window for the crypto-h3 baseline). The exercise immediately surfaced that **no code path in the repo consumes `splits["test"]`** — the carved holdout fold has zero consumers in `src/` or `tests/`. The user paused before proceeding and asked for a thorough state assessment to surface other phantoms before scoping any single fix.

Two parallel exploration passes were run (Pass 1: unused symbols + stub returns + tests-of-existence; Pass 2: provenance + watchdog + whole-subsystem audits). Pass 2 produced two false positives (subprocess watchdog and pydantic.ValidationError handling — both verified clean via direct introspection: `objective.py:302` wraps the trial body in `Watchdog` and `issubclass(pydantic.ValidationError, ValueError) == True` so the existing `except ValueError` catches it). The verified phantom inventory after both passes:

1. **Holdout test fold** (`splits["test"]`) — carved by `make_splits`, read by zero code paths
2. **ExtMem out-of-core ingest path** — entire infrastructure (`ParquetDataIter`, `select_ingest`'s ExtMem branch, `cache_host_ratio`, `use_native`, `_check_extmem_compat` guard) wired in design + docs, never executed in production
   - **2b** (borderline): `select_ingest()` IS called in production at `cli/train.py:87-90` but its return value drives only a `typer.echo` log message, not behavior. Covered by phantom #2's fix.
3. **Optuna Artifacts Store** — `RunsConfig.artifacts_root` declared + path-elided, `FileSystemArtifactStore` never instantiated, never used. Per `docs/ARCHITECTURE.md:443-444` it should store "Per-trial model bundles, plots, prediction CSVs" — none happen.
4. **`rux-ml tune retry-trial`** — verb body at `cli/tune.py:206-228` exists with real logic (`study.enqueue_trial(prior.params)` + `study.optimize(n_trials=1)`), zero integration test; test-file docstring overstates coverage.

The v0.3 sprint's theme is **honesty cut** — complete the documented behaviors so `docs/ARCHITECTURE.md` and `docs/CONSTRAINTS.md`'s "No Phantom Implementations" rule are both honest.

### Decisions

**This session establishes the sprint structure (which PR addresses which phantom) but defers the architectural sub-decisions to a follow-up design session.** Each sub-decision below has a stated lean from the 2026-05-20 chat session, but is NOT locked until the v0.3 design session runs `PROCEDURE-design-planning.md` from Phase 1 to Phase 5.

#### S1 — Sprint structure

- **Decision**: 6 implementation PRs (PR-031 through PR-036) bundled under v0.3.0. PR-030 (this PR) is sprint scaffolding only; no code change.
- **Rationale**: phantom #1 splits into two semantically-paired PRs (HPO objective restriction + new CLI verb consuming `splits["test"]`) because they're independently testable but the second is only semantically meaningful after the first. Phantom #2 is one large PR. Phantom #3 is one PR. Phantom #4 is one small test-only PR. The version cut is its own PR per `docs/VERSIONING.md §2`.
- **Status**: **proven** (precedent: PR-022..PR-027 sprint structure for v0.2).

### Pending — to be locked in v0.3 design session

The following architectural sub-decisions are LEANS from 2026-05-20 chat, NOT locked. Each implementation PR's Phase 1 + Phase 2 will surface drift; the design session resolves before PR-031 implementation begins.

#### D1 — Holdout semantics (resolved by design session before PR-031 implementation)

- **Question**: Is `splits["test"]` "truly held out from HPO" or "post-HPO sanity check"?
- **Lean**: truly held out. `tuning/objective.py` is changed to build CV substrate from `make_splits(...)["train"] + ["val"]`, not `df_full`. Alternative (sanity-check-only) makes the "holdout" label misleading and undermines PR-032's value.
- **Acknowledged cost**: HPO sees 15% less data, may produce slightly different HPs. Existing study metrics (e.g., crypto-h3 baseline at RMSE 0.027101) will shift — this is acceptable for a correctness fix; bundle a baseline-receipt-refresh step into PR-031.
- **Open**: does this apply to `data.split_kind == "random"` path too (for consistency), or only `time_ordered`?

#### D2 — ExtMem activation pattern (resolved by design session before PR-033 implementation)

- **Question**: How does the trainer switch between sklearn-wrapper and native `xgb.train` API at runtime?
- **Lean**: native API requires an adapter exposing `Trainer` Protocol (`fit/predict`) over `xgb.train` + `Booster`. Switch happens when `cfg.training.use_native == True` OR when `select_ingest` returns `ExtMemQuantileDMatrix`.
- **Open**: 
  - `ParquetDataIter` chunking strategy from a single-file source (split into N batches by row count? Polars streaming mode?)
  - Default `cache_host_ratio` value (XGBoost auto-estimate vs explicit fraction)
  - Test strategy when no real workload triggers the 18 GB threshold (force-flag for tests; reduced threshold for synthetic-large dataset)
  - `compute_score` compatibility — does the metric registry need a Booster-aware path? `_rmse(trainer, ...)` calls `trainer.predict(x_eval)` which assumes sklearn-wrapper; ExtMem path needs equivalent.
  - Per-fold ExtMem rebuild cost in HPO (probably prohibitive — may want to disable ExtMem in CV objective and only honor it for `cli/train.py` baseline runs)

#### D3 — Artifacts store: what to upload (resolved by design session before PR-034 implementation)

- **Question**: Prediction CSVs + fold scores JSON (diagnostic only) OR per-trial bundles (replaces re-fit at promote)?
- **Lean**: **diagnostic only**. The alternative reverses D8/PR-010 sub-decision A1 ("re-fit at promote time") explicitly — a v0.3 reversal of a v0 architectural choice is much bigger work and out of scope for the honesty cut.
- **Open**: 
  - When to upload (in objective post-fit? in trial_runner post-optimize?)
  - Whether to record artifact IDs in `TrialAttrs` (new field) or query Optuna by trial number at retrieval time
  - Cleanup policy (do we ever delete old artifacts?)

#### D4 — retry-trial: test-only or behavior change (resolved by design session before PR-035 implementation)

- **Question**: Does the verb body need any behavior change, or just a test?
- **Lean**: **test-only**. The body at `cli/tune.py:206-228` looks correct on review: loads study, validates trial exists, enqueues with original params, runs one trial. The test contract is the only gap.

### Deferred / acknowledged ambiguities (to be populated by design session)

To be added during the v0.3 design session.

### v0.3 implementation plan

See [`ROADMAP.md`](ROADMAP.md) for the full PR list. The design session output (when it runs) replaces this section's deferred items with locked decisions.

### Process notes

- This session is **PR-030 scaffolding only** — pure docs + PR stubs creation. The user later opted (2026-05-20) to skip the standalone v0.3 design session and resolve D1–D4 per-PR via each PR's Phase 1 (option-1 plan). The "Pending" section above is therefore informational; each implementation PR's `## Research findings` records the actual resolution.
- Per-Phase Approval Gate (NON-NEGOTIABLE per `docs/CONSTRAINTS.md`) honored at every phase of every PR's research procedure.
- Phantom audit transcript (the 2026-05-20 conversation) is captured in chat history; PR-030's `## Research findings` section references it for traceability.

---

## Session: 2026-05-20 → 2026-05-21 — PR-031 holdout-substrate (D1 resolution + Phase 3 web research)

### Context

Per the option-1 plan (2026-05-20), each v0.3 implementation PR resolves its own architectural sub-decision via its Phase 1 state assessment. PR-031 owned D1 (holdout semantics — truly held out vs post-HPO sanity check).

### Decision

#### D1 — HPO substrate excludes the held-out test fold (truly held out)

- **Decision**: `tuning/objective.py:build_objective` builds its CV substrate from `splits["train"] + splits["val"]` (≈85% of source data by default `data.split_ratios`). The `splits["test"]` partition is never seen by HP search.
- **Carve mechanics**: one `_carve_substrate(base_cfg, df_full, target_col)` call outside the per-trial closure. Internally `make_splits(cfg, df, seed=study_bag.split_seed)` where `study_bag = make_seed_bag(master_entropy=base_cfg.tuning.entropy, trial_number=0)` — same `trial_number=0` sentinel as the sampler seed at `cli/tune.py:73`. The substrate is reproducible per study identity, so all trials in a study share identical substrate rows.
- **Applies to both split kinds**: `random` (deterministic via the study-level seed) and `time_ordered` (deterministic by definition per PR-024).
- **Rejected alternatives**:
  - **Option B — HPO CV on full df** (workbench's pre-PR-031 behavior). Cited only in Optuna's `*_simple.py` demos at `optuna-examples/xgboost/xgboost_simple.py` (file SHA `592117186ae62c8252f961c8e9a38fb181d8b80b`); Optuna issue #2184 explicitly flags this as anti-pattern.
  - **Option C — Nested CV** (outer loop estimates generalization, inner loop tunes HPs). Cited at sklearn `examples/model_selection/plot_nested_cross_validation_iris.py` SHA `15082123761afffc97bdc992316705b388ac0850`. Rejected: K × J fits multiply sequential single-GPU wall-time by 5×; conflicts with workbench compute budget and the existing `make_splits(train/val/test)` API contract.
- **Status**: **convention** (≥3 cited production systems use the substrate-shrink pattern: scikit-learn user guide §3.1; AutoGluon `tabular-essentials.html` at SHA `f8c428cbbef3bc319ff3f7710f5900e65637f4c4`; mlfinlab purged-CV workflow). Group D Synthesis-Verification Probe verified the exact 3-element combination (carve test fold → HPO on remainder → score on test) in the AutoGluon tutorial.
- **Disconfirming evidence search**: explicit search for `"why HPO should see the full dataset"`, `"holding out from hyperparameter tuning is wrong"`, `"nested CV vs train/val/test debate"` returned zero reputable defenders of Option B. Only legitimate critique of Option A is that Option C is even better when compute allows.
- **Risks accepted**: HPO sees ~15% less data than the full df; HPs may be slightly suboptimal in a small-data regime. Acceptable for workbench scope. Nested CV (Option C) deferred to a future Tier-2 study if HPO compounding bias becomes empirically measurable.
- **Research citation**: `prs/PR-031-hpo-honors-holdout-fold.md` Phase 3 findings.

### Sub-decisions resolved en route

- **D1.b — random-split consistency**: yes, exclude the test fold for `random` too. Same study-level seed makes the substrate deterministic per study identity.
- **Variable renames**: `x_full`/`y_full` → `x_substrate`/`y_substrate` throughout `tuning/objective.py` (the variable no longer holds the full df after the carve — naming honest).
- **Backward compatibility on `studies/studies.db`**: no schema migration needed. Substrate change is in-objective behavior, not persisted state. Old + new trials coexist; new trials are measured against a different substrate visible via the trial's `git_sha`.
- **Existing tests at `tests/tuning/test_objective.py`**: all 7 pass without modification (they assert on completion + intermediate-value count + hash differentiation, not exact row counts or metric values).

### Implementation outcomes

- New module-level helper `_carve_substrate(base_cfg, df_full, target_col)` in `tuning/objective.py`.
- `build_objective` calls `_carve_substrate` once outside the closure; per-trial path is unchanged.
- 2 new unit tests on `_carve_substrate`: `test_carve_substrate_excludes_test_fold_for_time_ordered` (time-ordered substrate is first 85 rows of 100) + `test_carve_substrate_random_is_deterministic_across_invocations` (study-level seed reproducibility).
- Docs updated in the same commit: `docs/ARCHITECTURE.md` + `docs/CONVENTIONS.md` HPO objective sections; `docs/0.3/ROADMAP.md` PR-031 row flipped `[ ]` → `[x]`; `docs/0.3/RESEARCH-BACKLOG.md` PR-031 row → `state-assessed` + `fully-researched` + `implementation-cleared 2026-05-21`; `CHANGELOG.md [Unreleased] ### Changed` loud entry.
- Empirical receipt at `prs/PR-031-baseline-receipt.json` pending workbench run.

### Process notes

- Phase 1 + Phase 2 + Phase 5 done locally; Phase 3 dispatched a `general-purpose` agent with WebSearch/WebFetch for the Q1+Q7 question pair.
- Per-Phase Approval Gate held at every phase boundary (5 user approvals).
- Group D MCP-Verification Round per the post-PR-028/29 procedure: Probe 1 verified at Phase 1 (identifiers exist in repo); Probe 2 verified at Phase 3 (AutoGluon citation locks the exact 3-element combination); Probe 3 N/A (no state-registration surface).

---

## Session: 2026-05-21 — PR-032 holdout-score CLI verb (Phase 3 web research + schema amendments)

### Context

Per the option-1 plan (2026-05-20), each v0.3 implementation PR resolves its own architectural sub-decisions via its Phase 1 state assessment. PR-032 owned the receipt-schema + CLI-verb-shape sub-decisions for the holdout-scoring CLI verb that consumes the test fold PR-031 made truly held out.

### Decisions

#### D2 — `rux-ml registry score` CLI verb shape + receipt schema

- **Decision**: new CLI verb `rux-ml registry score --problem <name> [--version <v>] [--output <dir>]` orchestrating load_bundle → reconstruct splits → `_BoosterTrainerShim` (Trainer Protocol adapter over raw `xgb.Booster`) → `compute_score` → write Pydantic-validated `HoldoutScoreReceipt` JSON + Polars predictions parquet to `receipts/`.
- **Carve mechanics**: For `time_ordered` split, deterministic — no entropy_hex needed (any seed value works; the scorer passes 0). For `random` split, re-open the originating trial via `runs.load_run(storage, study, trial_number)` to read `attrs.entropy_hex`, then derive the trial's `bag.split_seed` via `make_seed_bag_from_hex`. Same row identities as the bundle's training-time substrate — no silent leakage.
- **Receipt schema body**: `metrics: dict[str, float]` + `artifacts: dict[str, _Artifact]` (where `_Artifact` has `path` + `content_type`). Follows MLflow `EvaluationResult.metrics` + `artifacts_metadata.json` convention; Kedro `tracking.MetricsDataSet` precedent for the CLI-persisted flow.
- **Receipt provenance wrapper**: `bundle_version`, `bundle_dir`, `manifest_git_sha`, `promoted_from`, `holdout`, `scored_at`, `scored_at_git_sha`. rux-ml-specific; no production tool surveyed bundles provenance inline (MLflow / Kedro / ZenML push it to tracking server / MLMD / catalog versions). Defensible per AWS Well-Architected ML Lens BP03 for the no-tracker-server constraint.
- **Verb-naming note**: `score` differs from MLflow `predict` (predictions only) and MLflow `evaluate` (Python API only). rux-ml's `score` produces both in one CLI invocation. Documented in `docs/CONVENTIONS.md` to prevent confusion.
- **Rejected alternatives**:
  - **Python API only** (the majority pattern per Phase 3 — MLflow / AutoGluon / HF evaluate / ZenML / W&B all do this). Rejected because rux-ml is CLI-first; users would have to write a script per evaluation. Kedro's CLI+receipt pattern is the closer fit for the constraints.
  - **Booster-aware paths in `compute_score`** (extending metric registry with type-branched logic). Rejected — higher complexity; the Booster shim is reusable for PR-033's ExtMem adapter and keeps the metric registry homogeneous.
  - **Inline scoring in scorer module** (bypassing `compute_score`). Rejected — duplicates metric logic; risks divergence from HPO's scoring.
- **Status**: **convention** for the 4-element flow (CLI → versioned-bundle-load → score-on-holdout → JSON receipt) — cited at Kedro starter tag `0.19.14`; **convention** for the body shape (metrics dict + artifacts dict) — cited at MLflow + Kedro; **best-guess-given-constraints** for the provenance wrapper — defensible per AWS ML Lens BP03 but no inline-bundling precedent.
- **Disconfirming evidence search**: explicit search for "no standard / ad-hoc model evaluation receipts" and for score-on-holdout CLI verbs in MLflow / AutoGluon / W&B / Optuna. **Result**: outside Kedro's `tracking.MetricsDataSet`, the dominant convention is Python-API-only with persistence delegated to a tracker. No mainstream tool ships a `score-on-holdout` CLI verb that loads a registry bundle AND writes a typed receipt. MLModelScope paper confirms ad-hoc scripting is the dominant reality.
- **Research citation**: `prs/PR-032-registry-score-cli-verb.md` Phase 3 findings.

### Sub-decisions resolved en route

- **CLI verb signature**: `--problem` required; `--version` defaults to champion via `read_champion`; `--output` defaults to `Path("receipts")`. Matches existing `cli/registry.py` flag convention (`promote`, `list`, `rollback`).
- **Predictions parquet column layout**: `(time_column, y_true, y_pred)` when `data.time_column` is set; `(y_true, y_pred)` otherwise. Asset column NOT included by default (would inflate parquet for panels); opt-in via future `--include-asset` flag if needed.
- **Receipts directory**: `receipts/` tracked at repo root; `receipts/.gitignore` excludes `*.parquet` (large, regeneratable); JSON receipts tracked (single source of truth for the metric).
- **`promote.py:97` stale docstring comment** ("held out for future golden-regression evaluation (PR-014)") replaced with accurate cross-reference to `rux-ml registry score` + the PR-031 substrate-shrink contract.

### Implementation outcomes

- New module `src/rux_ml/registry/scorer.py`: `_BoosterTrainerShim`, `HoldoutScoreReceipt`, `score_bundle_on_holdout`. NOT exported from `rux_ml.registry`'s public `__init__.py` (preserves PR-010 sub-decision B1 — strict inference-deps separation; CLI imports directly).
- New CLI verb `cli/registry.py:score`.
- 9 new tests: 7 in `tests/registry/test_scorer.py` (Booster shim correctness + scorer end-to-end + champion default + random-split determinism + missing-champion error); 2 in `tests/cli/test_registry_subcommands.py` (CLI verb + missing-champion `BadParameter`).
- Docs updated in the same commit: `docs/ARCHITECTURE.md` Storage table gains "Holdout score receipts" row; `docs/CONVENTIONS.md` gains "Holdout score receipts (per PR-032)" subsection (schema + verb-naming note + reconstruction strategy + Booster shim contract); `docs/0.3/ROADMAP.md` PR-032 row flipped `[ ]` → `[x]`; `docs/0.3/RESEARCH-BACKLOG.md` PR-032 row → `state-assessed` + `fully-researched` + `implementation-cleared 2026-05-21`; `CHANGELOG.md [Unreleased]` gains `### Added` (verb + receipt) + `### Fixed` (stale comment cleanup) entries.

### Process notes

- Phase 1 + Phase 2 + Phase 5 done locally; Phase 3 dispatched a `general-purpose` agent with WebSearch/WebFetch for the Q1 + Q2 + Q-Group-D probe.
- Phase 4 Outcome Branch: **Amend → Apply** (3 schema/doc amendments). User approved all 3 inline; synthesis resumed.
- Per-Phase Approval Gate held at every phase boundary (6 user approvals: Phase 1 → Phase 2 → Phase 3 → Phase 4 outcome → Phase 4 amend → Phase 5).
- Group D MCP-Verification Round: Probe 1 verified at Phase 1 (all identifiers exist in repo); Probe 2 verified at Phase 3 (Kedro starter at tag `0.19.14` cites the exact 4-element combination — required-not-coincidental); Probe 3 N/A (receipts are filesystem outputs, no state-registration surface).

---

## Session: 2026-05-21 — PR-033 ExtMem path activation (D2 resolution + Phase 3 synthesis BGGC)

### Context

Per the option-1 plan (2026-05-20), each v0.3 implementation PR resolves its own architectural sub-decisions via its Phase 1 state assessment. PR-033 owned D2 (ExtMem activation pattern) — the largest scope of the v0.3 sprint. Phase 1 surfaced significant **scope reduction**: every additional consumer site (registry/promote, HPO objective) was reviewed and deferred. Phase 3 surfaced the most consequential BGGC finding of the sprint.

### Decisions

#### D2 — ExtMem activation via opt-in native adapter (scope-reduced)

- **Decision**: new `XGBoostNativeAdapter` class in `src/rux_ml/training/xgboost/native_adapter.py` is the `Trainer`-Protocol-compatible wrapper around `xgb.train`. The dispatch decision from `select_ingest(estimate_x_bytes(X), cfg.data)` is honored *inside the adapter* — `QuantileDMatrix` for in-VRAM frames; `ExtMemQuantileDMatrix` (host-RAM cached via `cache_host_ratio` when on GPU) for over-threshold frames, with chunks materialized by the new `single_source_iter` helper in `src/rux_ml/data/data_iter.py` (Polars `iter_slices` → N temp Parquets → `ParquetDataIter`).
- **Activation surface**: `cli/train.py::_fit_and_score` branches on `cfg.training.use_native` (read for the first time in production) — `True` routes through `XGBoostNativeAdapter`; `False` (default) stays on the existing sklearn-wrapper path via `make_trainer`. Honest stderr log: `ingest path: <DMatrixCls> (native API|sklearn wrapper)`.
- **Scope reduction** (locked at Phase 1):
  - `registry/promote.py` UNCHANGED — promote-time re-fit operates on the train+val substrate (~85% of source data), which fits in VRAM at workbench scale. Native-path promotion deferred to a follow-up once a real workload demands it.
  - `tuning/objective.py` UNCHANGED — per-fold ExtMem rebuild cost is prohibitive (multi-minute per fold × `cv.n_splits`); the `_check_extmem_compat` guard preserves the existing "HPO + ExtMem = `NotImplementedError`" semantic for free.
  - Auto-on-ExtMem-trigger DEFERRED — XGBoost 3.2's own tutorial warns ExtMem is slower than `QuantileDMatrix` when data fits in RAM; opt-in only via `use_native=True`.
- **Rejected alternatives**:
  - **Inheritance refactor unifying `_BoosterTrainerShim` (PR-032) and `XGBoostNativeAdapter` under a shared base** — rejected at Phase 4. The shim is score-only (bundle-load + inference); the adapter is fit+score. Different lifecycle, different ownership, different test surfaces. Parallel adapter classes (Amend candidate 3) keep each concern clean.
  - **`use_native` auto-flip when `select_ingest` returns `ExtMemQuantileDMatrix`** — rejected. XGBoost 3.2 tutorial's "ExtMem slower than QuantileDMatrix when data fits in RAM" warning means auto-on can degrade performance on RAM-resident workloads that *technically* exceed the 18 GB heuristic. Explicit opt-in matches the documented behavior.
  - **Test-mode `RUXML_FORCE_EXTMEM=1` env var** — rejected. The PR-005 cross-field validator pattern is cleaner: tests force the path by setting `gpu_in_memory_x_gb_max=1e-9` in a `DataConfig`. No new env-var surface; same code path.
- **Status**: **best-guess-given-constraints** for the 4-element synthesis combination (`xgb.train` + `ExtMemQuantileDMatrix` + `cache_host_ratio` + sklearn-Trainer-Protocol wrapper). Phase 3 Q-Group-D Probe 2 (Synthesis-Verification) returned **NO single cited working example** of all four ingredients together. Each ingredient is independently sourced (XGBoost 3.2 [`external_memory.html`](https://xgboost.readthedocs.io/en/stable/tutorials/external_memory.html) chunking pattern; [`external_memory.py` demo](https://github.com/dmlc/xgboost/blob/master/demo/guide-python/external_memory.py) verbatim `ParquetDataIter` shape; XGBoost 3.2 [`python_api.html`](https://xgboost.readthedocs.io/en/release_3.2.0/python/python_api.html) parameter mapping table; PR-032's `_BoosterTrainerShim` sklearn-adapter-over-booster precedent in-repo) but the synthesis is the workbench's own composition. **Mitigation**: `tests/training/test_native_adapter.py::test_native_adapter_predict_proba_matches_sklearn_wrapper_within_tol` asserts bit-exact (within `atol=1e-5`) numeric agreement between adapter and sklearn-wrapper on shared params + seed + data. Without this test the BGGC label would be unbounded.
- **Research citation**: `prs/PR-033-extmem-path-activation.md` Phase 3 findings.

### Sub-decisions resolved en route

- **`cache_host_ratio` GPU-gating**: XGBoost 3.2's CPU `ExtMemQuantileDMatrix` rejects the kwarg with `Check failed: detail::HostRatioIsAuto(config.cache_host_ratio)`. Adapter gates on `cfg.device == "cuda"`. Documented in `CONVENTIONS.md` parameter-mapping table; tested by `test_native_adapter_threads_cache_host_ratio_when_device_cuda` (monkey-patched DMatrix constructor — real GPU not reachable on CI / macOS).
- **`predict`/`predict_proba` input types**: pass-through to `xgb.DMatrix(x, enable_categorical=True)` mirrors PR-032's `_BoosterTrainerShim._dmatrix` — accepts pandas/polars/ndarray uniformly. The metric registry calls `predict_proba` with pandas (via `x_val_t.to_pandas()` in the CLI); promotion-time consumers may pass polars; both round-trip cleanly.
- **Native objective string**: `xgb.train` does not auto-detect task from data (unlike sklearn wrapper). Adapter maps `task_for_metric(cfg.metric)` → `binary:logistic` / `reg:squarederror`. Multi-class out of scope at v0.
- **Parameter renames (sklearn → native)**: `random_state` → `seed` (in `params` dict); `n_estimators` → `num_boost_round` (kwarg, not param); `early_stopping_rounds` stays as kwarg but only honored when `evals` non-empty. Codified in `CONVENTIONS.md` table.
- **TemporaryDirectory ownership**: `fit()` creates a per-call `tempfile.TemporaryDirectory(prefix="rux_ml_extmem_")` for chunked Parquets + XGBoost's ExtMem cache files. Cleaned on context exit (some cache-file warnings from XGBoost trying to re-remove already-cleaned files are non-fatal noise).

### Implementation outcomes

- New module `src/rux_ml/training/xgboost/native_adapter.py` (`XGBoostNativeAdapter` class) — exported from `rux_ml.training.xgboost` + `rux_ml.training`.
- New helper `single_source_iter` in `src/rux_ml/data/data_iter.py` — exported from `rux_ml.data`.
- `src/rux_ml/cli/train.py::_fit_and_score` branches on `cfg.training.use_native` for the first time; honest stderr log emits `(native API)` vs `(sklearn wrapper)`.
- 8 new tests in `tests/training/test_native_adapter.py` covering: fit/predict smoke; **mandatory BGGC mitigation** (`predict_proba_matches_sklearn_wrapper_within_tol` at `atol=1e-5`); `predict_proba` 2D shape; `select_ingest`-monkey-patch verification for both `QuantileDMatrix` and `ExtMemQuantileDMatrix` branches; `cache_host_ratio` GPU-gated threading; real ExtMem path on CPU smoke; `single_source_iter` chunk-count + iterator-yield contract + invalid-input guards.
- 1 new CLI integration test in `tests/cli/test_train_subcommand.py::test_train_use_native_smoke` — verifies the `--set training.use_native=true` end-to-end through `cli/train` produces a valid AUC + emits the "native API" log line.
- Docs updated in the same commit: `docs/ARCHITECTURE.md` (D3 ingest rule body + Memory & Parallelism `cache_host_ratio` row); `docs/CONVENTIONS.md` (new "XGBoost native API + ExtMem path (per PR-033)" subsection with honest activation warning + parameter-mapping table + BGGC label disclosure); `docs/0.3/ROADMAP.md` PR-033 row flipped `[ ]` → `[x]`; `docs/0.3/RESEARCH-BACKLOG.md` PR-033 row → `state-assessed` + `fully-researched` + `implementation-cleared 2026-05-21`; `CHANGELOG.md [Unreleased] ### Added` (loud entry).

### Process notes

- Phase 1 + Phase 2 + Phase 5 done locally; Phase 3 dispatched a `general-purpose` agent with WebSearch/WebFetch for the Q-Group-D Probe 2 (Synthesis-Verification) — surfaced the BGGC finding.
- Phase 4 Outcome Branch: **Amend → Apply** (4 amendments). User approved all 4 inline at session pause; implementation resumed on 2026-05-21.
- Per-Phase Approval Gate held at every phase boundary (6 user approvals: Phase 1 → Phase 2 → Phase 3 → Phase 4 outcome → Phase 4 amend → Phase 5). Streak preserved.
- Group D MCP-Verification Round: Probe 1 verified at Phase 1 (every identifier — `xgb.train`, `ExtMemQuantileDMatrix`, `cache_host_ratio`, `ParquetDataIter`, `select_ingest`, `cfg.training.use_native`, `cfg.memory.cache_host_ratio` — exists in pinned `xgboost` and in `dev` HEAD). Probe 2 returned **NO cite** for the 4-element synthesis → BGGC + mitigation test (the most consequential probe-2 finding of the sprint). Probe 3 N/A (no state-registration surface; `XGBoostNativeAdapter` is constructor-instantiated, not registered).
