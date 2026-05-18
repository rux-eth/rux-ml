# PR-019: CatBoost Trainer family

**Landed-in:** v0.1.0 (pending v0.1.0 cut in PR-021)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PRs** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

**This PR is Tier-2** per `docs/0.1/RESEARCH-BACKLOG.md`. The registry pattern is research-backed by PR-017; CatBoost-specific instantiation is not. All 5 phases of `PROCEDURE-pr-research.md` must run.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

## Research findings

`PROCEDURE-pr-research.md` 5-phase Tier-2 procedure completed 2026-05-18.

### State Assessment (2026-05-18) — Phase 1

**Current state (post-PR-018)**: 2 Trainer families registered (`xgboost`, `lightgbm`). `[project.optional-dependencies]` table introduced by PR-018. `configs/search_spaces/lightgbm.toml` exists. `_ColumnRouter` sentinel renamed `PASSTHROUGH_TO_XGB_CATEGORICAL → PASSTHROUGH_NATIVE_CATEGORICAL` in PR-018. `PASSTHROUGH_NATIVE_CATEGORICAL` already family-agnostic per PR-018 design.

**CatBoost outlier surface** (flagged in PR-017 Q-MK; re-verified at v1.2.10): no `**kwargs` on `__init__` (117 explicit params); no sklearn inheritance; pandas required as hard dep. Implication: `CatBoostTraining` has no `model_kwargs` escape hatch.

**New constraints surfaced (Phase 1)**: per-family kwarg-name translation richer than LightGBM (CatBoost capitalizes metric names `AUC`/`Logloss`/`RMSE`/`MAE`; renames `n_estimators` → `iterations`, `max_depth` → `depth`, `random_state` → `random_seed`); `early_stopping_rounds` is a CatBoost constructor kwarg (simpler than LightGBM's fit-time callback); `cat_features` extraction needed (CatBoost does NOT auto-detect pandas Categorical, unlike LightGBM).

### Phase 3 — Research findings (6 parallel agents)

**Q-GPU — Default to GPU.** PROVEN. Material divergence from PR-018's LightGBM: CatBoost ships prebuilt CUDA-enabled PyPI wheels (`uv add catboost`, no extras, no source build, no container delta). RTX 4090 (CC 8.9) field-confirmed via [issue #2649](https://github.com/catboost/catboost/issues/2649). [szilard/GBM-perf](https://github.com/szilard/GBM-perf) V100 2024-06-06: CatBoost-GPU ~2.6-4.6× slower than XGBoost-GPU at 1M-10M rows but materially faster than CPU. Factory toggles `task_type="GPU"` + `devices="0"` when `cfg.device == "cuda"`. GPU bit-exact determinism NOT achievable (issue #546).

**Q-Cat — Pattern-A shim extracts `cat_features` at fit time.** PROVEN. CatBoost does NOT auto-detect pandas Categorical (opposite of LightGBM); raises error on category-dtype columns not in `cat_features=` ([issue #757](https://github.com/catboost/catboost/issues/757), open FR [#1386](https://github.com/catboost/catboost/issues/1386)). The `_CatBoostTrainerShim.fit()` extracts column names via `select_dtypes(include="category")` and passes through as `cat_features=`. `_ColumnRouter` unchanged. CatBoost's ordered Target Statistics ([Prokhorenkova et al. 2018](https://arxiv.org/abs/1706.09516)) is different from LightGBM's Fisher partitioning and XGBoost's partition-based split; vs `NestedCVWrapper` target encoding for high-card, Pargent et al. 2022 ([arxiv:2104.00629](https://arxiv.org/pdf/2104.00629)) groups both under "regularized target encoding" → comparable.

**Q-Ingest** — covered by Q-Wrap §9 (no separate research): CatBoost's `fit(DataFrame, y)` works natively without `Pool`. Thin `build_pool()` placeholder in `ingest.py` for symmetry with xgboost/lightgbm subpackages.

**Q-Parallel — CatBoost uses Intel TBB, NOT OpenMP. Factory passes `thread_count` explicitly.** PROVEN. Workbench's `OMP_NUM_THREADS` env-var pinning (PR-011) is **invisible to CatBoost** — TBB doesn't read OMP env vars by default. Factory reads `OMP_NUM_THREADS` from the trial subprocess env and passes as `thread_count=` to CatBoost. PR-011's `pin_threads()` is the transport; the factory translates. CONVENTIONS.md documents this. Bit-exact CPU determinism for CatBoost requires `thread_count=1` + `bootstrap_type='No'` + `rsm=1` + `random_strength=0` + `random_seed` + `has_time=True` + `boosting_type='Plain'` ([issue #1587](https://github.com/catboost/catboost/issues/1587)) — dedicated test deferred to follow-up Tier-2 PR.

**Q-Err — No translation layer needed.** PROVEN. `CatBoostError` is a flat `Exception` subclass (`_catboost.pyx@v1.2.10` L188-193). CPU OOM usually surfaces as OS SIGKILL (issues #968, #1814); GPU OOM (`TOutOfMemoryError`) is uncatchable from Python ([issue #2678](https://github.com/catboost/catboost/issues/2678), open). Subprocess-per-trial isolation (PR-008) already converts "subprocess died" → "trial failed" → study continues. A naive `CatBoostError → MemoryPressureError` would misclassify input-validation errors as OOM. **No `errors.py` module in PR-019 scope.**

**Q-HPO — 7-knob search space.** PROVEN with strong convergent evidence. `learning_rate`, `depth`, `l2_leaf_reg`, `random_strength`, `bagging_temperature`, `border_count`, `bootstrap_type` (categorical over `{Bayesian, Bernoulli, MVS}`). `iterations` + `grow_policy` FIXED (CatBoost docs say iterations should be large + use early-stopping; grow_policy is fixed at `SymmetricTree` per Optuna canonical examples). Anchored on [CatBoost parameter-tuning](https://catboost.ai/en/docs/concepts/parameter-tuning) + [Optuna catboost_simple.py](https://github.com/optuna/optuna-examples/blob/main/catboost/catboost_simple.py) + [CatBoost team Optuna tutorial](https://github.com/catboost/tutorials/blob/master/hyperparameters_tuning/hyperparameters_tuning_using_optuna_and_hyperopt.ipynb).

**Q-Wrap — 14-field `CatBoostTraining` schema; complete translation table.** PROVEN against `catboost/python-package/catboost/core.py @ v1.2.10` L5305-5425. No `**kwargs`. Key findings:
- Metric translation: `auc → AUC`, `logloss → Logloss`, `rmse → RMSE`, `mae → MAE` (case-sensitive).
- `loss_function` task-derived (`Logloss` classifier; `RMSE` regressor) because `AUC` is eval-only (Q-Wrap §4).
- `early_stopping_rounds` is a top-level constructor kwarg (L5396) — single kwarg shorthand for `od_type="Iter"` + `od_wait=N`.
- `random_state` → `random_seed` rename; `n_estimators` → `iterations`; `max_depth` → `depth`.
- Factory injects `verbose=False`, `allow_writing_files=False`.
- `fit(DataFrame, target)` works without `Pool` construction.
- **bootstrap_type ↔ randomness-kwarg interaction**: Bayesian accepts `bagging_temperature`, rejects `subsample`; Bernoulli/MVS/Poisson accept `subsample`, reject `bagging_temperature`. Factory conditionally includes only the active kwarg (CatBoost throws on mismatches).

### Phase 4 — Locked sub-decisions

| # | Decision |
|---|---|
| 1 | `device` inherits from `TrainingBase` (default `"cuda"`); factory toggles `task_type="GPU"` accordingly. Q-GPU confirmed GPU is the ergonomic default. |
| 2 | `thread_count` sourced from `OMP_NUM_THREADS` env var (set by PR-011's `pin_threads()` from `cfg.memory.omp_threads`). Reuse existing field, no `MemoryConfig` schema bloat. |
| 3 | CPU bit-exact determinism test for CatBoost deferred to follow-up Tier-2 PR (recipe documented in CONVENTIONS.md). |
| 4 | Pattern-A shim `_CatBoostTrainerShim` for `cat_features` auto-extraction. |
| 5 | Auto-derive `loss_function` from task; `eval_metric` from `_METRIC_TRANSLATE`. `loss_function` NOT exposed as a config field (`AUC` is eval-only trap). |
| 6 | 14 typed fields on `CatBoostTraining` (search-space dims + family-agnostic plumbing). CTR/text/imbalance knobs deferred. |

### Verification artifacts

- `make test` → **319 passed**, 0 failed, 15 deselected. Was 310 pre-PR-019; added 9 new tests (1 conformance-test entry + 8 in `test_catboost_smoke.py`).
- `uv run basedpyright src/` → **0 errors, 0 warnings, 0 notes**.
- `uv run ruff check .` → **All checks passed**.
- XGBoost + LightGBM integrations untouched per `One PR, One Thing`.

---

## Scope

Add CatBoost as the third Trainer family. Parallel in shape to PR-018 (LightGBM); but with CatBoost-specific research surface.

Concretely:

- **New subpackage** `src/rux_ml/training/catboost/`:
  - `__init__.py` — exports `make_catboost_trainer`
  - `factory.py` — wraps `CatBoostClassifier` / `CatBoostRegressor` with lazy `import catboost` + clear-error-on-missing pattern
  - `config.py` — `CatBoostConfig` Pydantic schema
  - `errors.py` (if research #3 surfaces a need) — `CatBoostError → MemoryPressureError` normalization layer
- **Registry entry** — add `"catboost": _make_catboost_trainer` to `TRAINER_FAMILIES`
- **PEP 631 extra** in `pyproject.toml`: `catboost = ["catboost>=1.2"]` (pin minimum per research)
- **Search-space TOML** — `configs/search_spaces/catboost.toml`
- **Categorical-handling decision** per research #1 — either bypass `_ColumnRouter` (CatBoost owns categorical) or coexist; document the decision in `docs/ARCHITECTURE.md`
- **Thread-pinning** per research #2 — extend `_internal/env.pin_threads` if CatBoost needs different env-var settings, or document that v0 pinning carries forward
- **Conformance test** — `tests/training/test_registry_conformance.py` automatically picks up the new family; assert it passes
- **End-to-end smoke test** — `tests/training/test_catboost_smoke.py` with a tiny fixed-seed dataset, fit-then-predict round trip
- **GPU smoke test** (gated by `@pytest.mark.gpu`) on RTX 4090
- **Amend `docs/ARCHITECTURE.md`** — CatBoost-specific decision rules (categorical handling, thread pinning if differs, error normalization)
- **CHANGELOG `[Unreleased]`** — user-facing entry for the new family

Out of scope: Solver layer (PR-020), v0.1.0 cut (PR-021), any changes to XGBoost or LightGBM integration.

## Dependencies

**PR-017** (Trainer registry refactor) — must be merged first.

**Not blocked by PR-018** — parallelizable.

## Architecture section implemented

`docs/ARCHITECTURE.md` "Components" — adds CatBoost to the configured trainer families table.

`docs/ARCHITECTURE.md` "Decision Rules" — adds CatBoost-specific decision rules per research #1 (categorical) and #2 (thread pinning if differs).

`docs/0.1/DESIGN-log.md` Q1–Q5 — second non-XGBoost family validating the pattern.

## Verification criteria

Populated after research. Initial sketch (refine in Phase 2-3):

- [ ] `src/rux_ml/training/catboost/` subpackage exists with required files
- [ ] `TRAINER_FAMILIES["catboost"]` resolves to the CatBoost factory
- [ ] `pyproject.toml` declares `[catboost]` extra with pinned minimum version
- [ ] `uv sync --extra catboost` works
- [ ] Running without `[catboost]` installed gives a clear `ImportError` on family instantiation
- [ ] `configs/search_spaces/catboost.toml` exists with research-anchored HPO ranges
- [ ] `tests/training/test_catboost_smoke.py` passes
- [ ] `tests/training/test_registry_conformance.py` passes for the new family
- [ ] GPU smoke test passes on the workbench's RTX 4090 (or documented as out-of-scope per research #4)
- [ ] `CatBoostError → MemoryPressureError` normalization is tested if research #3 surfaces the need
- [ ] `docs/ARCHITECTURE.md` carries CatBoost-specific decision rules
- [ ] `CHANGELOG.md` `[Unreleased]` has the CatBoost-family entry
- [ ] `docs/0.1/ROADMAP.md` PR-019 row flipped `[ ]` → `[x]` in this PR's commit
- [ ] `docs/0.1/RESEARCH-BACKLOG.md` PR-019 row marked `fully-researched YYYY-MM-DD` + `implementation-cleared YYYY-MM-DD`

## Research backing

Architectural pattern locked by `docs/0.1/DESIGN-log.md` Q1–Q5. CatBoost-specific decisions require Phase 2-3 web research per the five open questions above.

Reference precedents:

- [CatBoost categorical features doc](https://catboost.ai/en/docs/concepts/algorithm-main-stages_cat-features-processing)
- [CatBoost GPU tutorial](https://catboost.ai/en/docs/features/training-on-gpu)
- [CatBoost `core.py` source @ v1.2.7](https://github.com/catboost/catboost/blob/v1.2.7/catboost/python-package/catboost/core.py) — note: CatBoost doesn't inherit from sklearn (per the 2026-05-18 research findings)
- CatBoost paper: Prokhorenkova et al. 2018, "CatBoost: unbiased boosting with categorical features"

## Notes

- The **CatBoost outlier nature** flagged in the 2026-05-18 design session (sklearn never imported; pandas hard-required; duck-typed sklearn-compatibility) means PR-019 may need a slightly different factory shape than PR-017/018 — e.g., the `Trainer` Protocol conformance check needs CatBoost's `fit/predict` shape, which mimics sklearn but isn't inherited from `BaseEstimator`. The Protocol approach (vs ABC inheritance) handles this gracefully.
- The **categorical-handling decision (#1)** has implications for the `_ColumnRouter` v0 abstraction. If CatBoost is allowed to bypass `_ColumnRouter`, the categorical-encoding decision rule in `docs/ARCHITECTURE.md` needs a per-family override.
- Per `One PR, One Thing`, CatBoost-specific research belongs in this PR.
- **Parallelizable with PR-018** — both depend on PR-017 only.
