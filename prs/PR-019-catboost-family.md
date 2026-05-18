# PR-019: CatBoost Trainer family

**Landed-in:** (not yet landed)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PRs** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

**This PR is Tier-2** per `docs/0.1/RESEARCH-BACKLOG.md`. The registry pattern is research-backed by PR-017; CatBoost-specific instantiation is not. All 5 phases of `PROCEDURE-pr-research.md` must run.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

## Research findings

_To be populated by `PROCEDURE-pr-research.md`. Do not begin implementation until this section exists with completed findings from all required phases._

**Open research questions** (must be resolved before implementation):

1. **CatBoost native categorical handling vs `_ColumnRouter`.** CatBoost's claim to fame is native categorical support — order statistics + one-hot for low-cardinality, target statistics for high-cardinality. Does this obsolete the v0 `categorical_low_card_threshold` BEST-GUESS (PR-005) for catboost-only studies? Success criteria: documented decision on whether CatBoost bypasses `_ColumnRouter` entirely, integrates with a flag, or coexists. Cite CatBoost docs + Yandex paper.

2. **Symmetric tree parallelism characteristics.** CatBoost uses oblivious / symmetric trees — different parallelism strategy than XGBoost's level-wise or LightGBM's leaf-wise growth. Does this change the v0 thread-pinning strategy (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `POLARS_MAX_THREADS` from PR-011)? Success criteria: documented thread-pinning rules for CatBoost (may be same, may differ). Cite CatBoost docs + parallelism benchmarks.

3. **`CatBoostError` normalization into `MemoryPressureError`.** The v0 memory watchdog (PR-011) catches XGBoost OOM via `MemoryPressureError → optuna.TrialPruned`. Does CatBoost surface OOM differently? Does it need an explicit exception-translation layer in the factory? Success criteria: documented error-normalization path. Cite CatBoost source on memory failure modes.

4. **CatBoost-CUDA on consumer RTX 4090.** Like LightGBM, verify the official CUDA build works on consumer cards (compute capability 8.9). If it works, ensure the workbench's container CUDA 12.4.1 pin is compatible. Cite CatBoost GPU tutorial + GitHub issues.

5. **Canonical CatBoost HPO search-space.** What hyperparameters belong in `configs/search_spaces/catboost.toml`? At minimum: `learning_rate`, `depth`, `l2_leaf_reg`, `random_strength`, `bagging_temperature`, `border_count`. Success criteria: research-anchored ranges cited from production references.

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
