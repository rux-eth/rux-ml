# PR-015: CV strategy — Splitter Protocol + library research + splits.py rewrite

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-2 PR** (research-pending). All five phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form. The scope, verification criteria, and per-strategy implementation details below are deliberately under-specified pending Phase 3 research.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

## Research findings

_To be populated by `PROCEDURE-pr-research.md`. Do not begin implementation until this section exists with completed findings from all required phases._

---

## Why this PR exists

User direction (2026-05-16, recorded in project memory `project-cv-strategy-tier2`):

> The actual CV/k-fold/walk-forward/etc. stuff needs its own Tier-2 PRs. We need to know what libraries/tools to use for them, how to prevent data leakage, and how to implement them so that they are optimized and work with parallelism.

The naive shuffled `train_val_test_split` introduced in PR-004 is the **tabular IID default** — it ignores time dependence and group leakage. Before PR-007 (Optuna `objective` design) can decide what metric the HPO loop is minimizing, the workbench needs a typed CV abstraction and at least one concrete strategy per family (random / time-series / group / combinatorial-purged).

PR-005's use of `category_encoders.NestedCVWrapper(StratifiedKFold(n_splits=5))` for target-encoding leakage prevention is **out of scope** for this PR — it's a library-provided narrow slice that doesn't compose with project-wide CV strategy. PR-015 may revisit or override that default if research surfaces a conflict.

## Scope (subject to revision after Phase 3 research)

**Splitter Protocol + abstraction (`src/rux_ml/data/cv.py` — new):**
- `Splitter` `typing.Protocol` defining `split(X, y=None, *, groups=None) -> Iterator[tuple[NDArray, NDArray]]` and (where applicable) `get_n_splits(X, y=None, *, groups=None) -> int`
- Concrete implementations per the research findings (likely set: `KFold`, `StratifiedKFold`, `TimeSeriesSplit`, `GroupKFold`, walk-forward, CPCV)
- Per-strategy leakage notes documenting what each guarantees and what it does NOT guarantee
- Per-strategy parallelism notes documenting interaction with D6's `n_jobs=1` and D10's subprocess-per-trial

**Rewrite of `src/rux_ml/data/splits.py`:**
- PR-004's `train_val_test_split` deprecated/replaced with a Splitter-driven default
- New API: `train_val_test_split(df, *, splitter, target_col=None, group_col=None, ...)` returning the same `{train, val, test}` shape
- Backward-compatible behavior when the default `Splitter` is `KFold`-style random (so existing tests continue to make sense, though they will be rewritten to use the new API explicitly)

**Configuration:**
- Either extend `DataConfig` with a `cv:` block or introduce a new `CVConfig` layer (decision in Phase 3)
- CV configuration must be hashable so the `data_cfg_hash` / `root_cfg_hash` correctly reflect CV-strategy changes per D17

**Test coverage:**
- Split determinism given the same seed (every concrete strategy)
- Per-strategy leakage tests (time-leakage for `TimeSeriesSplit`; group-leakage for `GroupKFold`; embargo verification for CPCV)
- Integration: `train_val_test_split` end-to-end against synthetic data for each strategy

**Documentation updates riding with this PR:**
- `docs/ARCHITECTURE.md` — add a CV strategy section under Key Abstractions or Data
- `docs/CONVENTIONS.md` — add CV strategy conventions (which strategy defaults for which data shape; how leakage is documented per-strategy)
- `docs/DESIGN-log.md` — append a "Session 2026-XX-XX — CV strategy" record with the research trail and decisions made
- `docs/CONSTRAINTS.md` — only if a hard rule emerges (e.g., "all time-indexed data must use a Splitter that honors temporal ordering")

NOT in scope:
- Replacing PR-005's `NestedCVWrapper(StratifiedKFold)` for target-encoder leakage prevention (separate concern; revisit only if the research surfaces a project-wide conflict)
- Optuna objective integration (lands in PR-007; PR-015 only provides the Splitter that PR-007 will consume)
- Multi-objective CV / nested CV for hyperparameter selection (defer to PR-007 design or a later PR)

## Dependencies

PR-004 (data layer — provides Polars/Parquet ingest and the existing `splits.py` to be rewritten) and PR-005 (features layer — the `cardinalities_from` pattern + categorical handling that the CV-strategy tests will likely interact with).

**Blocks** PR-007 (Optuna basics) — PR-007's `objective(trial, base_cfg)` consumes a Splitter from the resolved `RuxMLConfig`. ROADMAP updated accordingly.

## Architecture section implemented

`docs/ARCHITECTURE.md` → new "CV Strategy" section + updates to Data Flow showing how the Splitter participates in the train/val/test slicing and in the future HPO objective.

## Verification criteria (subject to revision after research)

- [ ] `Splitter` Protocol defined in `src/rux_ml/data/cv.py`
- [ ] At least these concrete strategies implemented and tested: `KFold`, `StratifiedKFold`, `TimeSeriesSplit`, `GroupKFold`
- [ ] Walk-forward implementation (library-provided or thin wrapper around sklearn)
- [ ] CPCV implementation (library port, OSS adaptation, or custom — Phase 3 decides)
- [ ] Per-strategy leakage tests pass (time-leakage, group-leakage, embargo-respecting)
- [ ] `train_val_test_split` rewritten to consume a Splitter; deterministic with seed; tests cover each strategy
- [ ] CV configuration is hashable; per-trial `data_cfg_hash` and `root_cfg_hash` change correctly when CV strategy changes
- [ ] Parallelism interaction with D6 (`n_jobs=1`) + D10 (subprocess-per-trial) is documented and tested where relevant
- [ ] `docs/ARCHITECTURE.md` and `docs/CONVENTIONS.md` updated
- [ ] `docs/DESIGN-log.md` appended with the research session record

## Research backing (Tier 2 — required research topics)

**Phase 3 (web research) MUST cover, at minimum:**

1. **Library choice per CV family:**
   - sklearn `KFold` / `StratifiedKFold` / `TimeSeriesSplit` / `GroupKFold` — versions, maintenance, known limitations
   - `mlxtend` `EvaluateClassifier` / time-series helpers — is it still maintained? worth the dep?
   - CPCV: any battle-tested OSS implementation (e.g., `mlfinlab` historically had one, before going commercial; check current OSS state) vs porting from Lopez de Prado's *Advances in Financial Machine Learning*
   - Walk-forward: sklearn `TimeSeriesSplit` is the obvious choice, but check for alternatives that handle expanding-vs-rolling windows cleanly

2. **Data-leakage prevention per strategy:**
   - `KFold`: random shuffle — only safe for truly IID data; how to detect non-IID via grouping/temporal hints
   - `StratifiedKFold`: class-balance preservation; what stratification means for regression
   - `TimeSeriesSplit`: prevents time-leakage; embargo windows (do we need them on top?)
   - `GroupKFold`: prevents group-leakage; how groups are specified (which column / how derived)
   - CPCV: handles label overlap + embargo; specifically for finance-style data with bar-aggregated labels

3. **Parallelism interactions:**
   - sklearn cross-validators are pure Python iterators — fine sequentially
   - Sequential trials per D6 = no per-fold parallelism inside a trial; but each trial in PR-007 evaluates one fold or all folds?
   - Subprocess-per-trial per D10 = fold parallelism inside the child is OK (no GPU contention if XGBoost is `device="cpu"` for those folds), but interacts with thread pinning from D10 (`OMP_NUM_THREADS=24`)

4. **API ergonomics:**
   - Does the Splitter take Polars or pandas/numpy? PR-005 established the Polars-out / pandas-internally convention
   - How does CV configuration compose with `cfg.search_space` for HPO (Phase 3 may surface that CV params themselves should be tunable)

**Reputable sources** (per `docs/CONSTRAINTS.md`):
- sklearn user guide on cross-validation (https://scikit-learn.org/stable/modules/cross_validation.html)
- Lopez de Prado *Advances in Financial Machine Learning* (CPCV chapter — published reference, not random blog)
- Any production system documentation (vLLM, Hugging Face, Optuna's own docs on CV)
- Recent (2024–2026) engineering blog posts from serious ML teams describing time-series ML pipelines

## Notes

- This PR is **broad scope** by user direction — covers CV abstraction + rewrites `splits.py` + researches all CV variants in one round. The alternative (narrow PR-015 + multiple follow-up PRs for each variant) was considered and rejected for slower coverage.
- The user prefers **library/tool reuse over custom implementations** per `user-profile` memory; CPCV especially is at risk of "no good OSS implementation exists" — Phase 3 must surface this honestly.
- After Phase 3, this PR file gets rewritten with the locked-in Splitter API, the chosen library/implementation per strategy, and concrete verification criteria. Treat the current scope/verification sections as the planning skeleton, not the final spec.
