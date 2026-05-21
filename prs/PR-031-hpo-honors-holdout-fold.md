# PR-031: HPO objective honors holdout fold

**Landed-in:** (not yet landed)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

**This PR is Tier-2** per memory `project_cv_strategy_tier2` (anything touching CV/splits requires Tier-2). The architectural sub-decision D1 (holdout semantics — truly held out vs post-HPO sanity check) is **resolved by this PR's Phase 1 state assessment (2026-05-20)** per the option-1 plan locked 2026-05-20 (skip standalone v0.3 design session; each PR's Phase 1 resolves its own sub-decisions). Verdict: **truly held out**.

This PR triggers a MINOR bump (v0.3.0) per `docs/VERSIONING.md §1` — behavior change in HPO substrate. Ships under v0.3.0 alongside PR-032–PR-035.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

## Research findings

### Phase 1 — State Assessment (2026-05-20)

**Current state** (HEAD = `0f5bae1`, post-PR-035 merge on `dev`):

- `src/rux_ml/tuning/objective.py:259-261` materializes `df_full` outside the per-trial closure; `x_full` / `y_full` are captured by `objective()` and consumed at line 287 (`groups`), 296 (`_check_extmem_compat`), 307 (`_fold_scores`).
- Per-trial bag at line 276-279 derives `split_seed`, `cv_seed`, `sampler_seed`, `xgb_seed` from `(master_entropy, trial.number)`.
- Study-level bag pattern at `cli/tune.py:73` uses `trial_number=0` sentinel for `sampler_seed`.
- `tests/tuning/test_objective.py` has 7 tests; all assert on completion / hash differentiation / intermediate-value count — not on exact row counts or metric values.
- `tune_cfg` fixture (`tests/tuning/conftest.py:39`) — 200-row synthetic binary classification, default `data.split_kind = "random"`, default `split_ratios = {0.7, 0.15, 0.15}`. Substrate = 170 rows; K=3 KFold ~57 rows/fold; existing tests should pass without modification.
- `cli/train.py:73` and `registry/promote.py:110` both use `make_splits(cfg, df, seed=bag.split_seed)` and consume only `["train"] + ["val"]` — already honor the holdout per PR-024.

**Drift assessment**: `tuning/objective.py:259-261` last touched at PR-013 (seed management). Zero subsequent touches across PR-014..PR-035. Audit reference matches HEAD verbatim. **Zero drift.**

**Stale assumptions**:
1. **Stub said "v0.3 design session locks D1"** — per option-1, no standalone design session. D1 resolved by this Phase 1: **truly held out**. Reasoning: the phantom audit surfaced "nothing reads `splits["test"]`"; the corrective fix is making HPO honor the holdout. The alternative (sanity-check-only) would require deleting `splits["test"]` from `make_splits` or treating it as duplicate data — a v0.3 architectural reversal not supported by the audit findings. Cross-PR consistency with PR-032 (score verb only meaningful AFTER PR-031 makes the holdout truly unseen by HPO).
2. **Stub said `make_splits(base_cfg, df_full, seed=base_cfg.tuning.entropy)`** — corrected at Phase 2/4: use `make_seed_bag(master_entropy=base_cfg.tuning.entropy, trial_number=0).split_seed` to match the `cli/tune.py:73` study-level-bag precedent.

**D1.b — `data.split_kind == "random"` consistency** (resolved by Phase 1): yes, exclude the test fold for both `random` AND `time_ordered`. The "holdout" semantic is split-kind-independent. Random path uses `study_bag.split_seed` so the substrate is deterministic per study identity.

**New constraints learned**:
- Substrate must be identical across all trials (otherwise different trials CV over different rows — incomparable). One `make_splits` call outside the closure, study-level seed.
- `x_full` variable name becomes a lie after PR-031 (holds train+val rows, not full df). Rename to `x_substrate` for honesty.
- `data_hashes(source_path)` at line 261 unchanged: source identity unchanged; substrate is a subset for algorithmic purposes only.
- Provenance schema (`TrialAttrs`) unchanged: existing `data_cfg_hash` covers `split_kind` + `split_ratios`; existing `data_hash` covers source content.

**Prior-art audit**:
- `tuning/objective.py` history: PR-007 → PR-009 → PR-011 → PR-013 (last touch).
- Known-good patterns to carry forward: heavy data loaded outside closure (cost amortized); per-trial bag derivation; study-level bag uses `trial_number=0` sentinel; polars-in / pandas-out at trainer boundary; `finally` block records attrs.
- Known-bad patterns surfaced in prior PRs: PR-022 reframe (unsourced "antipattern" framing → research-backed pragmatic-deviation framing); PR-023 row-count vs time-unit confusion. Lesson: document substrate semantics LOUDLY.

**Group D probes (Phase 1)**:
- Probe 1 (Schema-Integrity): `make_splits` + `make_seed_bag(trial_number=0)` + `pl.concat` all exist at HEAD. ✓
- Probe 2 (Synthesis-Verification): the COMBINATION `make_splits → study-level bag.split_seed → pl.concat([train, val]) → CV substrate` was novel — no in-repo cited example. Carried forward to Phase 3 web research.
- Probe 3 (Binding-at-creation): N/A — no state-registration surface.

**Exit decision**: premise valid; D1 + D1.b resolved at Phase 1; 2 questions deferred to Phase 2 (fixture inspection — local) and Phase 3 (web research — Probe 2). Proceed to Phase 2.

### Phase 2 — Scope the Research (2026-05-20, light)

**Must-answer**:

1. **Q1 — Global-holdout-vs-CV-substrate convention in production HPO libraries.** Is the workbench's pre-PR-031 behavior (CV on full df) the convention or the exception? Investigate sklearn / Nixtla / mlfinlab / AutoGluon-TS / Darts / Optuna / xgboost-lightgbm-catboost `cv()` built-ins. Success: ≥2 cited production systems explicitly carving a held-out test fold disjoint from CV substrate, OR disconfirming finding with citations.
2. **Q7 — Synthesis-Verification Probe carry-forward**: cited working example of the exact 3-element combination (carve test fold → HPO CV on remainder → final scoring on test). Success: ≥1 production source at a named commit SHA showing all 3 elements together.
3. **Q2 — Metric shift quantification on crypto-h3 baseline**: workbench run; receipt at `prs/PR-031-baseline-receipt.json`. **DEFERRED** to implementation phase (requires code change + workbench).

**Dependencies**: Q1 + Q7 parallel (same agent dispatch). Q2 sequential after implementation.

**Excluded (resolved at Phase 1 or out of scope)**:
- Q3 (random-split consistency) — resolved at Phase 1
- Q4 (DB schema migration) — resolved at Phase 1 (no migration needed)
- Q5 (variable rename) — deferred to Phase 4 cosmetic
- Q6 (fixture row count + class balance) — verified at Phase 2 inline (170-row substrate; K=3 ~57 rows/fold; tests pass without modification)
- Nested CV at every fold level (deferred indefinitely; out of v0.3 scope)

### Phase 3 — Findings (2026-05-21)

#### Q1: Global-holdout-vs-CV-substrate convention

**Options considered:**

- **Option A — HPO CV on `train+val`; test held out** (substrate-shrink; PR-031's lean)
  - Sources:
    - scikit-learn user guide §3.1 — canonical example `train_test_split → SVC.fit(X_train) → clf.score(X_test)` ([cross_validation.html](https://scikit-learn.org/stable/modules/cross_validation.html))
    - AutoGluon `tabular-essentials.html` at master SHA `f8c428cbbef3bc319ff3f7710f5900e65637f4c4` — explicit 3-element flow (separate test_data CSV → fit on train_data with internal HPO → evaluate on test_data)
    - `TabularPredictor.fit` reference: `tuning_data` + `holdout_frac` carve internal val from train; user-supplied `test_data` is exclusively for `evaluate`/`leaderboard` (v1.5.0)
  - Pros: HPO doesn't see test fold; estimate genuinely out-of-sample; matches textbook three-way split; aligns with workbench's existing `make_splits` carving
  - Cons: HPO has ~15% less data than full df

- **Option B — HPO CV on full df** (workbench's pre-PR-031 behavior)
  - Sources: `optuna-examples/xgboost/xgboost_simple.py` SHA `592117186ae62c8252f961c8e9a38fb181d8b80b` — `train_test_split` inside the objective, no random_state, no separate test holdout. Optuna issue [#2184](https://github.com/optuna/optuna/issues/2184) explicitly flags this: *"the metric will not be computed using the same validation set, so the optimization is not comparing the same quantity over different iterations."*
  - Pros: HPO sees more data; simpler code
  - Cons: optimistic bias from HP selection / final scoring leakage; no genuinely out-of-sample number; acknowledged anti-pattern by Optuna maintainers

- **Option C — Nested CV** (outer loop estimates generalization, inner loop tunes HPs)
  - Source: sklearn `examples/model_selection/plot_nested_cross_validation_iris.py` SHA `15082123761afffc97bdc992316705b388ac0850` — *"Choosing the parameters that maximize non-nested CV biases the model to the dataset, yielding an overly-optimistic score."*
  - Pros: most rigorous; no data wasted; distribution of generalization scores
  - Cons: K × J model fits → multiplies sequential single-GPU wall-time by 5×; conflicts with workbench compute budget and per-trial provenance contract

**Disconfirming evidence sought**: explicit search for `"why HPO should see the full dataset"`, `"holding out from hyperparameter tuning is wrong"`, `"nested CV vs train/val/test debate"`. **Result**: zero reputable sources defend Option B as best practice. Only legitimate critique of Option A is that Option C is even better when compute allows.

**Recommendation**: **Option A** (substrate-shrink)
- **Status**: **convention** — 3 cited production systems (sklearn user guide + AutoGluon production tutorial + mlfinlab purged-CV workflow); only Optuna's `_simple` demos skip the final holdout (explicitly labeled minimal).
- **Why**: matches all 3 PR-031 leans (truly held out; one `make_splits` outside closure; both split kinds). Disconfirming search returned zero defenders of Option B.
- **Risks accepted**: HPO has ~15% less data; nested CV deferred to a future Tier-2 study if HPO compounding bias becomes empirically measurable.

#### Q7: Synthesis-Verification probe — exact 3-element combination

**Cited working example**:
- **Path**: AutoGluon `dev/tutorials/tabular/tabular-essentials.html` first end-to-end code block
- **SHA**: autogluon master HEAD `f8c428cbbef3bc319ff3f7710f5900e65637f4c4` (verified via `gh api repos/autogluon/autogluon/commits/master`); docs v1.5.1-dev
- **What it shows**: single cohesive flow with all 3 required elements:
  1. test data held out at source level (separate CSV)
  2. `TabularPredictor.fit(train_data)` runs entire HPO loop on training data only ("AutoGluon automatically chose a random training/validation split" *within* `train_data`)
  3. final scoring via `evaluate(test_data)` / `leaderboard(test_data)` — test fold never touches HPO substrate

**Status**: **convention** (NOT best-guess-given-constraints). Combination documented in a production AutoML library's first-page tutorial; corroborated by sklearn's canonical §3.1 workflow.

#### Group D MCP Verification (2026-05-21)

**Schema-Integrity Probe**:

| Claim | Identifier | Canonical documenter | Verified | Notes |
|---|---|---|---|---|
| `make_splits` dispatcher | `make_splits` | `src/rux_ml/data/splits.py:124` @ `0f5bae1` | yes | Existing |
| Study-level bag pattern | `make_seed_bag(trial_number=0)` | `src/rux_ml/cli/tune.py:73` @ `0f5bae1` | yes | Existing precedent for sampler seed |
| Polars concat | `pl.concat` | Polars docs (stock API) | yes | Standard |

**Synthesis-Verification Probe**:

| Claim | Combined elements | Cited working example | Verified | Notes |
|---|---|---|---|---|
| Substrate-shrink pattern | carve test fold + HPO on remainder + final evaluate on test | AutoGluon `tabular-essentials.html` @ `f8c428cbbef3bc319ff3f7710f5900e65637f4c4` | yes | Convention; upgrades Phase 1's tentative BGGC to convention |
| Study-level seed for substrate | `make_seed_bag(trial_number=0).split_seed` consumed by `make_splits` outside closure | In-repo precedent at `cli/tune.py:73` for `sampler_seed`; PR-031 extends pattern to `split_seed` | partial | Convention-by-extension of the in-repo `trial_number=0` sentinel |

**Binding-at-creation**: N/A — substrate is in-memory only; no state-registration surface.

**Reconciliations**: Phase 1's tentative `best-guess-given-constraints` label on the synthesis combination upgrades to **convention** after Phase 3 located the AutoGluon citation.

### Phase 4 — Synthesis (2026-05-21)

**Outcome**: **Confirm** — Phase 3 backs all 3 of PR-031's stated leans as convention.

**Changes to this PR from research**:
1. Seed derivation specifics locked: `make_seed_bag(master_entropy=base_cfg.tuning.entropy, trial_number=0).split_seed`
2. D1 lock-source amended: "v0.3 design session" → "PR-031 Phase 1 state assessment 2026-05-20"
3. Variable rename `x_full`/`y_full` → `x_substrate`/`y_substrate` (Q5)
4. Status label: **convention** (NOT BGGC)
5. New helper `_carve_substrate(base_cfg, df_full, target_col)` extracted as module-private so tests can verify the carve directly

**Changes to ARCHITECTURE.md**: "Optuna sampler/pruner" section gains substrate-carve note.

**Changes to CONVENTIONS.md**: "HPO objective shape" section gains substrate-carve bullet.

**Changes to docs/0.3/{ROADMAP, RESEARCH-BACKLOG, DESIGN-log}.md**: PR-031 row flips `[ ]` → `[x]`; status updates; new design-log session entry.

**Changes to CHANGELOG.md**: loud `### Changed` entry documenting behavior change.

**New PRs that must come first**: none.

### Phase 5 — Gate Check (2026-05-21)

- Premise still valid: ✓
- No prerequisite PRs surfaced: ✓
- D1 + D1.b resolved at Phase 1; Phase 3 web research locked convention status
- User approved updated spec: ✓ (2026-05-21)
- Implementation cleared: ✓ (2026-05-21)

---

## Scope

Modify `tuning/objective.py` so HPO CV operates on `splits["train"] + splits["val"]` instead of `df_full`. Test fold truly held out.

### Item 1 — `_carve_substrate` helper

New module-private function in `tuning/objective.py`:
```python
def _carve_substrate(
    base_cfg: RuxMLConfig, df_full: pl.DataFrame, target_col: str
) -> tuple[pl.DataFrame, pl.Series]:
    study_bag = make_seed_bag(master_entropy=base_cfg.tuning.entropy, trial_number=0)
    splits = make_splits(base_cfg, df_full, seed=study_bag.split_seed)
    df_substrate = pl.concat([splits["train"], splits["val"]])
    return _strip_target(df_substrate, target_col)
```

### Item 2 — `build_objective` uses `_carve_substrate`

Replace `df_full = materialize(...); x_full, y_full = _strip_target(...)` with:
```python
df_full = materialize(load_parquet(source_path))
x_substrate, y_substrate = _carve_substrate(base_cfg, df_full, target_col)
hashes = data_hashes(source_path)
```

Inside the closure, `x_substrate`/`y_substrate` replace `x_full`/`y_full` at the 3 consumption sites (groups for GroupKFoldCV, `_check_extmem_compat`, `_fold_scores`).

### Item 3 — `cli/train.py` + `registry/promote.py` unchanged

Both already honor the holdout per PR-024 (`make_splits(...)["train"] + ["val"]`). Verified at Phase 1.

### Item 4 — Crypto-h3 baseline metric receipt refresh

Bundle into PR: re-run 10-trial baseline study under PR-031 semantics on workbench; commit `prs/PR-031-baseline-receipt.json`. PR-025 calibration RMSE 0.027101 was measured pre-PR-031; new RMSE will differ.

### Item 5 — Docs

`docs/ARCHITECTURE.md` + `docs/CONVENTIONS.md` HPO objective sections gain substrate-carve text. `docs/0.3/DESIGN-log.md` gets new session entry.

### Out of scope

- `rux-ml registry score` CLI verb (PR-032)
- ExtMem path activation (PR-033)
- Optuna Artifacts Store wiring (PR-034)
- Nested CV (deferred indefinitely)

## Dependencies

- **PR-030** (sprint scaffolding) — merged
- D1 + D1.b resolved by this PR's Phase 1 (no separate design session per option-1 plan)

## Architecture section implemented

`docs/ARCHITECTURE.md` "Decision Rules" → Optuna sampler/pruner — substrate-carve bullet. Makes `splits["test"]` semantically meaningful for the first time since PR-024.

## Verification criteria

- [x] `tuning/objective.py:build_objective` substrate = `splits["train"] + splits["val"]`
- [x] `_carve_substrate` helper added
- [x] `x_full`/`y_full` renamed to `x_substrate`/`y_substrate`
- [x] 2 new behavior tests pass (`test_carve_substrate_*`)
- [x] All existing 7 tests in `tests/tuning/test_objective.py` pass without modification
- [x] `uv run pytest` default suite — 372 passed (was 370), 1 skipped, 15 deselected
- [x] `uv run ruff check .` clean
- [x] `uv run basedpyright src/` clean
- [x] `docs/ARCHITECTURE.md` + `docs/CONVENTIONS.md` + `docs/0.3/{ROADMAP, RESEARCH-BACKLOG, DESIGN-log}.md` + `CHANGELOG.md` updated in same commit
- [x] `docs/0.3/ROADMAP.md` PR-031 row at `[x]`
- [x] PR-031 stub `## Research findings` populated with Phase 1-5 output
- [ ] `prs/PR-031-baseline-receipt.json` committed with workbench-measured RMSE (pending workbench run)
- [x] `cli/train.py` + `registry/promote.py` unchanged

## Research backing

Tier-2 — D1 resolved by Phase 1; Phase 3 web research locked **convention** status with AutoGluon + sklearn + Optuna #2184 citations.

Anchored on:
- AutoGluon `tabular-essentials.html` @ SHA `f8c428cbbef3bc319ff3f7710f5900e65637f4c4`
- scikit-learn user guide §3.1
- Optuna issue #2184
- mlfinlab purged-CV workflow (final test holdout convention)
- `make_splits` (PR-024) + `make_seed_bag(trial_number=0)` (PR-013/cli/tune.py:73 precedent)

## Notes

- **Triggers MINOR bump** (v0.3.0) per `docs/VERSIONING.md §1`
- **Behavior-changing**: any existing study TOML produces different metrics after this PR. CHANGELOG entry is loud.
- **Pairs with PR-032**: PR-032's holdout-scorer is only semantically meaningful AFTER PR-031.
- Per memory `feedback_roadmap_flip_in_pr`: PR-031 row flipped in same commit; streak preserved.
- Crypto-h3 RMSE under PR-031 is a fresh empirical receipt; 0.027101 from PR-025 calibration is NOT a regression target.
