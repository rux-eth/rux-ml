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
