# PR-015: CV strategy — Splitter Protocol + library research + splits.py rewrite

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-2 PR** (research-pending). All five phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form. The scope, verification criteria, and per-strategy implementation details below are deliberately under-specified pending Phase 3 research.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

## Research findings

### State Assessment (2026-05-16, LOCAL-ONLY)

**Current state of the codebase**:

- `dev` at `6961f83` (PR-006 merged + ROADMAP fix). Working tree clean.
- `src/rux_ml/data/splits.py` exposes a single function — `train_val_test_split(df, *, ratios, seed) -> dict[str, pl.DataFrame]` (Polars `df.sample(fraction=1.0, shuffle=True, seed=...)` + slice). No Splitter abstraction; no class hierarchy.
- **Active callers of the current split:** exactly one — `src/rux_ml/cli/train.py:97`, with `_DEFAULT_SPLIT_SEED = 0` as a PR-013 placeholder. `tests/data/test_splits.py` covers 7 cases (determinism, row preservation, ratio validation, empty frame, seed-difference) — PR-015 rewrites or extends these.
- **No `src/rux_ml/data/cv.py` exists yet.** No `Splitter` Protocol, no `CVConfig` layer model.
- **Library-internal CV already in place:** `category_encoders.NestedCVWrapper(TargetEncoder, cv=5, shuffle=True, random_state=0)` in `src/rux_ml/features/encoders.py`. Per `project-cv-strategy-tier2` memory + PR-005 CV-scope note, this is explicitly **out of scope** for PR-015 unless Phase 3 surfaces a project-wide conflict.
- **Library availability (local verification only):** `sklearn 1.8.0` installed; `KFold` / `StratifiedKFold` / `TimeSeriesSplit` / `GroupKFold` / `GroupShuffleSplit` all import; `split(X, y, groups)` + `get_n_splits(X, y, groups)` API surface confirmed. **`mlxtend` not installed; `mlfinlab` not installed.** Phase 3 verifies current maintenance state of each candidate.
- **Downstream blockers:** PR-007 (Optuna basics) consumes the Splitter from `RuxMLConfig`; PR-013 (Seed management) needs to know what `cv_seed` means before it can spawn one. ROADMAP already sequences both after PR-015.

**Assumptions in the PR-015 draft (2026-05-16)**:

- A `Splitter` `typing.Protocol` with `split(X, y=None, *, groups=None) -> Iterator[tuple[NDArray, NDArray]]` + `get_n_splits(...)` (sklearn-shape).
- Concrete strategies likely covering `{KFold, StratifiedKFold, TimeSeriesSplit, GroupKFold, walk-forward, CPCV}`.
- `train_val_test_split` rewrites to consume a `Splitter`; PR-005's `NestedCVWrapper(StratifiedKFold)` stays out of scope unless Phase 3 surfaces a conflict.
- CV config is hashable so `data_cfg_hash` / `root_cfg_hash` reflect strategy changes (per D17). Whether it lives on `DataConfig.cv` or a new `CVConfig` is deferred to Phase 3.
- Splitter input shape (Polars vs pandas/numpy) deferred to Phase 3.
- Single broad-scope PR over multiple narrow PRs (per the PR-015 `## Notes` section's recorded user direction).

**Stale assumptions**: None substantive — the PR file is a planning skeleton that explicitly defers specifics to Phase 3.

**New constraints learned from PR-001 → PR-006**:

1. **`cli/train.py` integration test is the regression guard.** `tests/cli/test_train_subcommand.py` runs `rux-ml train` end-to-end and depends on the current `train_val_test_split` signature; the rewrite must keep it green.
2. **`optuna>=4.0` now in runtime deps** (added by PR-006). Phase 3 should check whether any `optuna.integration` CV helper displaces a custom Splitter (e.g., `OptunaSearchCV`).
3. **PR-005's Polars-out / pandas-waist convention.** Splitter input is whatever the data layer produces (Polars `DataFrame`); Splitter output convention (numpy index arrays vs Polars partitions) needs to be chosen.
4. **All Pydantic configs are `extra="forbid"` `StrictModel`s** (PR-002). Any CV config block must be a typed schema, not a free-form dict.
5. **PR-013 will replace `_DEFAULT_SPLIT_SEED = 0`** with a `SeedSequence(entropy).spawn()`-derived `cv_seed`. PR-015's Splitter must accept a seed parameter (int or `numpy.random.Generator`) so PR-013 can plug a derived seed in unchanged.
6. **D6 (`n_jobs=1` sequential trials) + D10 (subprocess-per-trial spawn, `OMP_NUM_THREADS=24`, BLAS pinned to 1)** — per-fold parallelism inside a trial collides with both. Phase 3 must produce a parallelism interaction matrix.
7. **ARCHITECTURE.md / CONVENTIONS.md / CONSTRAINTS.md have no CV-strategy sections today.** PR-015 will add them. ARCHITECTURE.md "Reproducibility Architecture" point 7 already references `cv_seed` in the spawned seed set — that line stays valid; PR-015 makes its meaning concrete.

### Research Questions (Phase 2, 2026-05-16)

**Must-answer:**

| # | Question | Success criterion |
|---|---|---|
| **Q1.a** | Which library implements `KFold` / `StratifiedKFold` for IID-tabular use? | Named library + version + URL; ≥1 alternative cited; recommendation labeled `proven`/`convention`/`best-guess-given-constraints` with a 2-sentence rationale citing maintenance + adoption. |
| **Q1.b** | Walk-forward — sklearn `TimeSeriesSplit` vs alternatives; expanding-vs-rolling window selection. | Named winner + ≥1 alternative + a citable expanding-vs-rolling decision rule. |
| **Q1.c** | `GroupKFold` — sklearn vs alternatives; how groups are specified. | Library + ≥1 alternative; a concrete groups-specification pattern backed by ≥2 production examples. |
| **Q1.d** | **CPCV** (combinatorial purged CV with embargo) — is there a maintained OSS implementation as of 2026? `mlfinlab` (commercial post-2022?), `purged-kfold` repos, or port-from-paper? | A current-state library survey with ≥3 candidates checked (license, last release, adoption), or a "no good OSS exists" finding labeled `best-guess-given-constraints` with implementation cost estimate. |
| **Q2.a** | Time-leakage: does `TimeSeriesSplit` need embargo windows? Under what conditions? | A documented rule with ≥2 cited examples. |
| **Q2.b** | Group-leakage prevention semantics across strategies — what each guarantees vs does NOT. | Per-strategy table with explicit "does not guarantee X" lines; cited sources. |
| **Q2.c** | CPCV — what does the embargo parameter actually purge? Label-overlap handling? | A precise definition citable to López de Prado *Advances in Financial Machine Learning* + ≥1 cross-check from a recent (2024+) post or production system. |
| **Q3.a** | Per-fold parallelism inside a trial vs D6's `n_jobs=1` GPU saturation rule — what's safe? | Decision matrix: GPU vs CPU folds × {sequential, parallel} with cited XGBoost / Optuna issue references for OOM/contention cases. |
| **Q3.b** | Subprocess-per-trial (D10 spawn) + Splitter — pickle-able / cross-process-safe? | A "yes/no, and here's why" with cited Optuna+XGBoost subprocess case studies. |
| **Q3.c** | Does any candidate Splitter strategy conflict with `ExtMemQuantileDMatrix` (which iterates Parquet files, not a materialized X)? | Per-strategy compatibility note with citations to XGBoost ExtMem docs. |
| **Q4.a** | Splitter input shape — Polars (matches PR-005) or numpy/pandas (matches sklearn)? | Recommendation + ≥1 alternative + a cited example from a Polars-first production project using sklearn splitters. |
| **Q4.b** | Splitter output — numpy index arrays (sklearn convention) or Polars partition DataFrames (PR-004 convention)? | Recommendation + downstream consumer compatibility check (PR-007 objective + `rux-ml train`). |
| **Q4.c** | Config composition: extend `DataConfig` with `cv:` block, or introduce a new `CVConfig` layer? | Recommendation + cited precedent (≥2 Pydantic-settings-based ML projects) + impact on `*_cfg_hash` set in `docs/CONSTRAINTS.md`. |

**Dependencies**:
- Q4.a depends on Q1 (library choice constrains input shape).
- Q4.b depends on Q4.a.
- Q4.c is independent.
- Q2.* depends on Q1.* (per-strategy guarantees follow library choice).
- Q3.* are independent of Q1/Q2 (parallelism is structural).

**Dispatch plan**: three parallel general-purpose web-research agents:
- **Agent A** — Q1.a-c + Q2.a-b (sklearn-shaped strategies + IID/time/group leakage)
- **Agent B** — Q1.d + Q2.c (CPCV deep dive + OSS-or-port decision)
- **Agent C** — Q3.* + Q4.a-c (parallelism + ergonomics + config composition)

**Explicitly excluded from this round** (deferred):
- Replacing PR-005's `NestedCVWrapper(StratifiedKFold)` for target-encoding — separate concern; only revisit if Q2 surfaces a project-wide conflict.
- Multi-objective CV / nested CV for HPO — defer to PR-007 or later PR.
- Sample-weighting / class-imbalance Splitter variants — defer.
- Profile-driven Rust accelerator for CPCV — falls under D13's future trigger.

### Findings (Phase 3, 2026-05-16)

Three parallel web-research agents ran on 2026-05-16 against the question scope above. Per-question detail (options considered, sources, disconfirming evidence sought, recommendations with epistemic labels) lives in this PR's design conversation; the cross-question summary is preserved here.

| Q | Recommendation | Status | Top source |
|---|---|---|---|
| Q1.a | sklearn `KFold` / `StratifiedKFold` (1.8.0, Dec 2025) | **proven** | https://scikit-learn.org/stable/modules/cross_validation.html |
| Q1.b | sklearn `TimeSeriesSplit(gap, max_train_size)`; `mlxtend.GroupTimeSeriesSplit` documented as the rolling+group upgrade path | **proven** (sklearn) / **convention** (expand-vs-roll rule) | https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html |
| Q1.c | sklearn `GroupKFold` + `StratifiedGroupKFold` + `LeaveOneGroupOut`; TOML `groups_column: str` resolved to `np.ndarray` at the data-layer boundary | **proven** (lib) / **convention** (column→array pattern) | https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html |
| Q1.d | **`skfolio.model_selection.CombinatorialPurgedCV`** (BSD-3, v0.20.1 released 2026-04-21, ~2k stars). `mlfinlab` rejected (proprietary post-2022 — "all rights reserved" + £100/mo/user); `timeseriescv` rejected (last commit Feb 2022). | **proven** | https://github.com/skfolio/skfolio |
| Q2.a | Embargo required iff label horizon ≥1 bar OR features include rolling/lag stats OR residuals autocorrelated. `TimeSeriesSplit(gap=k)` for the simple case; skfolio CPCV for variable-horizon. | **proven** | López de Prado AFML Ch.7 §7.4.2 + Bergmeir-Hyndman 2018 |
| Q2.b | Per-strategy leakage guarantee/non-guarantee table (lands in ARCHITECTURE.md) | **proven** | https://scikit-learn.org/stable/modules/cross_validation.html |
| Q2.c | AFML canonical: one-sided post-test embargo (size = `h·N`, `h ∈ [0.005, 0.02]`) + two-sided label-overlap purge. `embargo_pct` is a TOML knob with **no hardcoded default** (per CONSTRAINTS.md "Zero Hardcoded Parameters"). | **proven** (definition) / **convention** (h range) | López de Prado AFML Ch.7 §7.4.2; cross-checks at skfolio CPCV docs + timeseriescv source |
| Q3.a | Sequential folds on GPU; **never** parallelise folds on a single GPU (OOM per XGBoost #5029, #6225, #11298) | **proven** | https://github.com/dmlc/xgboost/issues/5029 |
| Q3.b | Build Splitter in the child from `cfg.cv`; no cross-process pickling (matches Optuna's own parallelism model + the project's existing Trainer/Features-in-child pattern) | **convention** | https://optuna.readthedocs.io/en/stable/faq.html |
| Q3.c | Hybrid: file-level folds when Splitter allows (KFold-sequential, TimeSeriesSplit, GroupKFold-respecting-partitions, walk-forward); materialised fallback otherwise; config-load validation rejects incompatible combinations | **best-guess-given-constraints** | https://xgboost.readthedocs.io/en/stable/tutorials/external_memory.html |
| Q4.a | Splitter accepts `pl.DataFrame`; internal `.height` + `.to_numpy()` only where sklearn requires arrays (Stratified, Group, CPCV) | **best-guess-given-constraints** | https://github.com/machml/polars_splitters + https://github.com/scikit-learn/scikit-learn/issues/28341 |
| Q4.b | Yield `(np.ndarray, np.ndarray)` row-index pairs (sklearn convention); CV (repeated) and `train_val_test_split` (one-shot, DataFrame return per PR-004) intentionally have different shapes for memory reasons | **convention** | https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html |
| Q4.c | New top-level `CVConfig` (8th layer); `cv_cfg_hash` joins the per-trial provenance triple | **convention** | https://deepwiki.com/vllm-project/vllm/2-core-architecture + https://kedro-mlflow.readthedocs.io/en/0.12.0/source/08_API/kedro_mlflow.config.html |

**Cross-agent convergence to flag:** Agents A and B independently recommended `skfolio` for CPCV from disjoint question sets. Both independently rejected `mlfinlab` (licensing) and `timeseriescv` (staleness). The previous-generation OSS reference (`mlfinlab`) has moved out of bounds; the next-generation reference (`skfolio`) is in.

### Synthesis (Phase 4, 2026-05-16)

**Outcome: Confirm** — research fills in the spec's intentional placeholders without contradicting the premise. Four bounded user-facing sub-decisions resolved in the same session.

**User-approved sub-decisions (2026-05-16):**

- **A1** — Accept `skfolio>=0.20.1` as a runtime dep (status: proven; risk accepted: ~80 MB cvxpy+clarabel transitive deps).
- **B1** — Splitter accepts `pl.DataFrame`; internal `.height` + `.to_numpy()` only where sklearn delegation requires (status: best-guess-given-constraints; risk accepted: must avoid sklearn helpers that wrap `_check_array`).
- **C1** — File-level ExtMem-CV compatibility only in v0; incompatible Splitter+ExtMem pairings raise `NotImplementedError` at config-load time with a clear message (materialised fallback deferred to a follow-up PR when a real problem hits ExtMem-sized data).
- **D2** — `train_val_test_split(df, *, ratios, seed) → dict[str, pl.DataFrame]` signature stays exactly as PR-004 defined it; documented as a one-shot convenience separate from the new Splitter Protocol. PR-006's `cli/train.py` keeps working untouched.

**Changes to this PR from research:**

- **New file** `src/rux_ml/data/cv.py` — `Splitter` `typing.Protocol` + concrete strategies (`KFoldSplitter`, `StratifiedKFoldSplitter`, `TimeSeriesSplitter`, `GroupKFoldSplitter`, `CombinatorialPurgedSplitter` wrapping `skfolio.model_selection.CombinatorialPurgedCV`).
- **New file** `src/rux_ml/config/cv.py` — `CVConfig` as a tagged-union over Splitter kinds (per-kind config schema; `extra="forbid"` via `StrictModel`).
- **Existing** `src/rux_ml/config/root.py` — add `cv: CVConfig` field + `cv` entry in `_HASH_ELIDED_FIELDS`.
- **Existing** `src/rux_ml/cli/train.py` — append `"cv"` to `_HASH_LAYERS` so the per-layer-hash loop now records `cv_cfg_hash` in the trial's `user_attrs`. (Splitter integration into the training/objective flow lands in PR-007 per the original spec.)
- **Existing** `src/rux_ml/data/splits.py` — left as-is per sub-decision D2; the existing `train_val_test_split` is annotated with a docstring note pointing at the Splitter Protocol for repeated CV.
- **`pyproject.toml`** — adds `skfolio>=0.20.1` runtime dep.
- **`configs/base.toml`** — new `[cv]` section header with a default-pointer comment.

**Changes to `docs/ARCHITECTURE.md`** (same commit):
- New "CV Strategy" subsection under "Key Abstractions" — Splitter Protocol; per-strategy leakage table (Q2.b output); column-to-array groups pattern.
- "Components" table — 8th row for `src/rux_ml/data/cv.py` (or absorbed into the Data row with the Splitter mentioned).
- "Storage / Config" — note the 8th `*_cfg_hash` (`cv_cfg_hash`).

**Changes to `docs/CONSTRAINTS.md`** (same commit):
- "Per-Trial Provenance Triple" — add `cv_cfg_hash` to the required `user_attrs` list.

**Changes to `docs/CONVENTIONS.md`** (same commit):
- "Per-trial user_attr keys" — add `cv_cfg_hash` to the documented set.
- New subsection: "CV strategy conventions" — column-to-array groups pattern; one-shot (`train_val_test_split`) vs repeated (`Splitter`) distinction; per-strategy default-selection rules.

**No new prerequisite PRs surfaced.** PR-007 (Optuna basics) and PR-013 (seed management) remain correctly sequenced downstream; both will consume `cfg.cv` once PR-015 lands.

**Phase-3 / Phase-4 propagation summary table** (durable):

| Layer | What changes |
|---|---|
| Code (new) | `src/rux_ml/data/cv.py`, `src/rux_ml/config/cv.py` |
| Code (touched) | `src/rux_ml/config/root.py`, `src/rux_ml/config/__init__.py`, `src/rux_ml/data/__init__.py`, `src/rux_ml/cli/train.py`, `configs/base.toml`, `pyproject.toml`, `uv.lock` |
| Tests (new) | `tests/data/test_cv.py` (Protocol conformance, per-strategy, determinism), `tests/data/test_cv_leakage.py` (group + time + embargo), `tests/config/test_cv_config.py` |
| Tests (touched) | `tests/cli/test_train_subcommand.py` (assert `cv_cfg_hash` in `user_attrs`) |
| Docs (touched) | `docs/ARCHITECTURE.md`, `docs/CONSTRAINTS.md`, `docs/CONVENTIONS.md`, `docs/RESEARCH-BACKLOG.md` |

### Gate Check (Phase 5, 2026-05-16)

- Premise still valid: ✓ (research confirms the broad-scope-single-PR direction)
- No prerequisite PRs surfaced: ✓
- User approved locked-in spec: ✓ (2026-05-16; sub-decisions A1, B1, C1, D2)
- Implementation cleared: ✓ (2026-05-16)

---

## Why this PR exists

User direction (2026-05-16, recorded in project memory `project-cv-strategy-tier2`):

> The actual CV/k-fold/walk-forward/etc. stuff needs its own Tier-2 PRs. We need to know what libraries/tools to use for them, how to prevent data leakage, and how to implement them so that they are optimized and work with parallelism.

The naive shuffled `train_val_test_split` introduced in PR-004 is the **tabular IID default** — it ignores time dependence and group leakage. Before PR-007 (Optuna `objective` design) can decide what metric the HPO loop is minimizing, the workbench needs a typed CV abstraction and at least one concrete strategy per family (random / time-series / group / combinatorial-purged).

PR-005's use of `category_encoders.NestedCVWrapper(StratifiedKFold(n_splits=5))` for target-encoding leakage prevention is **out of scope** for this PR — it's a library-provided narrow slice that doesn't compose with project-wide CV strategy. PR-015 may revisit or override that default if research surfaces a conflict.

## Scope (locked-in after Phase 3 research — 2026-05-16)

**Splitter Protocol + abstraction (`src/rux_ml/data/cv.py` — new):**
- `Splitter` `typing.Protocol` defining `split(X: pl.DataFrame, y: pl.Series | None = None, *, groups: np.ndarray | None = None) -> Iterator[tuple[NDArray[np.int_], NDArray[np.int_]]]` and `get_n_splits(...)` (sklearn-shape with **Polars-in** per sub-decision B1).
- Concrete strategies (locked set):
  - `KFoldSplitter` — wraps `sklearn.model_selection.KFold`
  - `StratifiedKFoldSplitter` — wraps `sklearn.model_selection.StratifiedKFold`
  - `TimeSeriesSplitter` — wraps `sklearn.model_selection.TimeSeriesSplit(gap, max_train_size)` (walk-forward; `max_train_size=None` ⇒ expanding window, `max_train_size=int` ⇒ rolling window per Q1.b decision rule)
  - `GroupKFoldSplitter` — wraps `sklearn.model_selection.GroupKFold` (groups resolved from `cfg.cv.groups_column: str` via `df[col].to_numpy()` at the data-layer boundary per Q1.c convention)
  - `CombinatorialPurgedSplitter` — wraps `skfolio.model_selection.CombinatorialPurgedCV` (per sub-decision A1; AFML-canonical one-sided post-test embargo + two-sided label-overlap purge)
- Each Splitter is constructible from its `CVConfig` variant (Splitter is built inside the trial subprocess from `cfg.cv` per Q3.b — no cross-process pickling).
- Each Splitter accepts a `seed: int | np.random.Generator | None = None` parameter so PR-013's `cv_seed = SeedSequence(entropy).spawn()` can plug in unchanged.
- Output: `(np.ndarray, np.ndarray)` row-index pairs (sklearn convention per Q4.b).

**Configuration (`src/rux_ml/config/cv.py` — new):**
- `CVConfig` as a tagged union over Splitter kinds — one Pydantic sub-model per concrete strategy (`KFoldCV`, `StratifiedKFoldCV`, `TimeSeriesSplitCV`, `GroupKFoldCV`, `CombinatorialPurgedCV`), discriminated by a `kind` literal field.
- All sub-models inherit `StrictModel` (`extra="forbid"`).
- `embargo_pct` (and any other AFML-tunable knobs on CPCV) have **no hardcoded defaults** per `docs/CONSTRAINTS.md` "Zero Hardcoded Parameters" + Q2.c.
- `groups_column: str` on `GroupKFoldCV` references a column in the input DataFrame; resolved to `np.ndarray` at the data-layer boundary, not inside the Splitter.
- `src/rux_ml/config/root.py` adds `cv: CVConfig` field + `"cv"` entry in `_HASH_ELIDED_FIELDS` (no fields elided for v0).
- `src/rux_ml/cli/train.py`'s `_HASH_LAYERS` tuple appends `"cv"` so `_build_user_attrs` records `cv_cfg_hash` on every trial.

**ExtMem compatibility validation (per sub-decision C1):**
- A small static table maps each Splitter strategy to {ExtMem-compatible, requires-materialisation}.
- The Splitter Protocol exposes `extmem_compatible: ClassVar[bool]`.
- `cli/train.py` (and future PR-007's objective) check this at training time — when ExtMem is the active ingest path AND `splitter.extmem_compatible` is False, raise `NotImplementedError` with a clear message naming both the Splitter kind and the deferred-fallback follow-up. **No materialisation fallback in v0.**

**`src/rux_ml/data/splits.py` (per sub-decision D2):**
- Signature **untouched** — `train_val_test_split(df, *, ratios, seed) -> dict[str, pl.DataFrame]` stays as PR-004 defined it.
- Docstring updated to point at the new Splitter Protocol for repeated-CV use cases; the function is documented as a one-shot convenience that bypasses the Splitter Protocol.
- PR-006's `cli/train.py` call path remains untouched.

**Test coverage:**
- `tests/data/test_cv.py` — Splitter Protocol conformance per strategy; determinism under `seed`; sklearn delegation surface (KFold/StratifiedKFold/TimeSeriesSplit/GroupKFold); `extmem_compatible` flags.
- `tests/data/test_cv_leakage.py` — per-strategy leakage tests: time-leakage on `TimeSeriesSplit` (train indices precede test indices; `gap` honored), group-leakage on `GroupKFold` (no group ID spans train+test in any fold), embargo verification on CPCV (training observations within `embargo_pct·N` after each test fold are purged).
- `tests/config/test_cv_config.py` — tagged-union discriminator round-trip; `extra="forbid"` rejects unknown kinds; `cv_cfg_hash` integration with `layer_cfg_hash(cfg, "cv")`.
- `tests/cli/test_train_subcommand.py` — extend the existing `user_attrs` assertion to include `cv_cfg_hash` (the only PR-006 test that exercises the provenance triple).

**Documentation updates riding with this PR:**
- `docs/ARCHITECTURE.md` — new "CV Strategy" section under "Key Abstractions" (Splitter Protocol snippet; per-strategy leakage guarantee table; column-to-array groups pattern); Storage section gets an 8th `*_cfg_hash` row.
- `docs/CONSTRAINTS.md` — "Per-Trial Provenance Triple" gets `cv_cfg_hash` added.
- `docs/CONVENTIONS.md` — "Per-trial user_attr keys" adds `cv_cfg_hash`; new subsection on CV conventions (one-shot vs repeated; column-to-array groups; per-strategy default selection by data shape).
- `docs/RESEARCH-BACKLOG.md` — PR-015 row gets `fully-researched 2026-05-16` + `implementation-cleared 2026-05-16`.
- `docs/DESIGN-log.md` — defer; the PR file's own Research findings section is the durable record per `PROCEDURE-pr-research.md`.

**NOT in scope:**
- Replacing PR-005's `NestedCVWrapper(StratifiedKFold)` for target-encoder leakage prevention (Phase 3 surfaced no project-wide conflict).
- Optuna objective integration (PR-007 consumes the Splitter).
- Multi-objective CV / nested CV for hyperparameter selection (defer to PR-007 or later).
- ExtMem-CV materialised fallback (sub-decision C1 deferral; follow-up PR if/when needed).
- Sample-weighting / class-imbalance Splitter variants.
- `mlxtend.GroupTimeSeriesSplit` integration (mentioned as the documented upgrade path for true-rolling-with-groups; not added as a runtime dep at v0).

## Dependencies

PR-004 (data layer — provides Polars/Parquet ingest and the existing `splits.py` to be rewritten) and PR-005 (features layer — the `cardinalities_from` pattern + categorical handling that the CV-strategy tests will likely interact with).

**Blocks** PR-007 (Optuna basics) — PR-007's `objective(trial, base_cfg)` consumes a Splitter from the resolved `RuxMLConfig`. ROADMAP updated accordingly.

## Architecture section implemented

`docs/ARCHITECTURE.md` → new "CV Strategy" section + updates to Data Flow showing how the Splitter participates in the train/val/test slicing and in the future HPO objective.

## Verification criteria (locked-in 2026-05-16)

- [ ] `Splitter` `typing.Protocol` defined in `src/rux_ml/data/cv.py` with the locked signature (`pl.DataFrame` in, `(np.ndarray, np.ndarray)` index pairs out)
- [ ] Five concrete strategies implemented and tested: `KFoldSplitter`, `StratifiedKFoldSplitter`, `TimeSeriesSplitter`, `GroupKFoldSplitter`, `CombinatorialPurgedSplitter`
- [ ] `CombinatorialPurgedSplitter` wraps `skfolio.model_selection.CombinatorialPurgedCV` (per sub-decision A1)
- [ ] Each Splitter exposes `extmem_compatible: ClassVar[bool]`; `cli/train.py` raises `NotImplementedError` at training time when ExtMem is active and `extmem_compatible is False` (per sub-decision C1)
- [ ] Each Splitter is deterministic given the same seed (unit tests cover all 5 strategies)
- [ ] Per-strategy leakage tests pass: time-leakage on `TimeSeriesSplit` (train precedes test; `gap` honored); group-leakage on `GroupKFold` (no group ID spans train+test in any fold); embargo on CPCV (training observations within `embargo_pct·N` after each test fold are purged)
- [ ] `CVConfig` (tagged union over 5 sub-models, `extra="forbid"` via `StrictModel`) lives at `src/rux_ml/config/cv.py`; `RuxMLConfig.cv` field added; `_HASH_ELIDED_FIELDS` includes `"cv"`
- [ ] `cv_cfg_hash` is computable via `layer_cfg_hash(cfg, "cv")` and is recorded in Optuna `user_attrs` by `cli/train.py`'s `_build_user_attrs` (extended `_HASH_LAYERS` tuple)
- [ ] `tests/cli/test_train_subcommand.py::test_train_records_user_attrs_in_sqlite` asserts `cv_cfg_hash` present alongside the existing 7 layer hashes
- [ ] `train_val_test_split` signature **unchanged** (per sub-decision D2); docstring updated to point at the Splitter Protocol for repeated CV
- [ ] `skfolio>=0.20.1` added to runtime `dependencies` in `pyproject.toml`
- [ ] `configs/base.toml` gains a `[cv]` section header with comment-only content
- [ ] `docs/ARCHITECTURE.md` — new "CV Strategy" subsection (Splitter Protocol snippet + per-strategy leakage table + groups column-to-array pattern); Storage / Configuration table mentions the 8th `cv_cfg_hash`
- [ ] `docs/CONSTRAINTS.md` — `cv_cfg_hash` added to "Per-Trial Provenance Triple"
- [ ] `docs/CONVENTIONS.md` — `cv_cfg_hash` added to "Per-trial user_attr keys"; new "CV strategy conventions" subsection (one-shot vs repeated; column-to-array groups; per-strategy default selection)
- [ ] `docs/RESEARCH-BACKLOG.md` — PR-015 row gets `fully-researched 2026-05-16` + `implementation-cleared 2026-05-16`
- [ ] `uv run pytest`, `uv run ruff check .`, `uv run basedpyright src/ tests/` all green

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
