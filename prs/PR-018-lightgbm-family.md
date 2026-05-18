# PR-018: LightGBM Trainer family

**Landed-in:** v0.1.0

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PRs** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

**This PR is Tier-2** per `docs/0.1/RESEARCH-BACKLOG.md`. The registry pattern is research-backed by PR-017; LightGBM-specific instantiation is not. All 5 phases of `PROCEDURE-pr-research.md` must run.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

## Research findings

`PROCEDURE-pr-research.md` 5-phase Tier-2 procedure completed 2026-05-18.

### State Assessment (2026-05-18) — Phase 1

**Current state (post-PR-017):**
- Training layer is multi-family: `src/rux_ml/training/xgboost/` subpackage; `TRAINER_FAMILIES = {"xgboost": ...}`; `TrainingConfig = Annotated[XGBoostTraining, Field(discriminator="kind")]` (single-variant union).
- `configs/search_spaces/` directory doesn't yet exist; v0 SearchSpec is INLINE in study TOMLs.
- `pyproject.toml` has NO `[project.optional-dependencies]` table yet; PR-018 introduces it.
- `_ColumnRouter` (`src/rux_ml/features/pipeline.py:60`) uses `PASSTHROUGH_TO_XGB_CATEGORICAL` sentinel — Q-Cat may extend or bypass.

**New constraints surfaced (not in stub):**
- Per-family kwarg-name translation (XGBoost `eval_metric=` → LightGBM `metric=`; `device="cuda"` → `device="gpu"` or `"cuda"`; etc.).
- GPU build variants are non-trivial (OpenCL vs CUDA vs CPU; no prebuilt CUDA wheel on PyPI).
- `early_stopping_rounds` semantics differ between families.
- `use_native` flag symmetry decision.
- `_ColumnRouter` integration path (extend vs bypass).

### Research Questions — Phase 2

Five must-answer; Q-A3 doc-only. All independent → 5 parallel research agents.

| Q | Bundle | Status |
|---|---|---|
| Q-GPU | LightGBM-CUDA on RTX 4090 | PROVEN |
| Q-Cat | LightGBM categorical vs `_ColumnRouter` | PROVEN |
| Q-Ingest | Dataset memory profile | PROVEN |
| Q-Wrap | Sklearn-wrapper kwarg-translation | PROVEN |
| Q-HPO | Canonical HPO search-space defaults | PROVEN |

### Phase 3 — Research findings

Five parallel agents. All citations from primary sources at pinned versions/commits.

**Q-GPU — CPU-only ship; defer GPU.** PROVEN.
- LightGBM has two GPU build variants (OpenCL `device_type=gpu`; native CUDA `device_type=cuda`); neither has prebuilt CUDA wheel on PyPI.
- Performance ([szilard/GBM-perf benchmarks, V100 2024-06-06](https://github.com/szilard/GBM-perf)): LightGBM-GPU is **8-28x slower than XGBoost-GPU** at workbench-scale (100K-10M rows). Corroborated by user reports [#6697](https://github.com/microsoft/LightGBM/issues/6697) (A100), [#6531](https://github.com/microsoft/LightGBM/issues/6531) (RTX 4060).
- Install path via `uv add lightgbm`: CPU wheel only. CUDA build requires `pip install --no-binary lightgbm --config-settings=cmake.define.USE_CUDA=ON`; documented as flaky ([#6417](https://github.com/microsoft/LightGBM/issues/6417), [#5785](https://github.com/microsoft/LightGBM/issues/5785)).
- Decision: ship CPU-only; `make_lightgbm_trainer` raises `NotImplementedError("LightGBM GPU support deferred ...")` when `cfg.device == "cuda"`. Future PR widens path when prereqs (prebuilt wheel OR vetted source-build recipe; container delta; golden benchmarks showing GPU > CPU) are met.
- Sources: [LightGBM Installation Guide @ v4.5.0](https://lightgbm.readthedocs.io/en/v4.5.0/Installation-Guide.html), [GPU Tutorial @ v4.5.0](https://lightgbm.readthedocs.io/en/v4.5.0/GPU-Tutorial.html).

**Q-Cat — Near-zero `_ColumnRouter` change.** PROVEN.
- LightGBM's native splitter is Fisher (1958) optimal partitioning over a sorted gradient histogram — different algorithm from XGBoost's partition-based split.
- High-cardinality behavior: LightGBM's own docs at [Advanced-Topics @ v4.5.0](https://lightgbm.readthedocs.io/en/v4.5.0/Advanced-Topics.html#categorical-feature-support) recommend **"treat high-card as numeric"** — i.e., the workbench's existing `NestedCVWrapper` target encoding for high-card columns is **what LightGBM itself recommends**. Corroborated by Pargent et al. 2021 ([arxiv:2104.00629](https://arxiv.org/pdf/2104.00629)): regularized target encoding ≥ native LightGBM handling on high-card.
- Low-cardinality auto-detect: `_data_from_pandas @ v4.5.0` ([basic.py](https://github.com/microsoft/LightGBM/blob/v4.5.0/python-package/lightgbm/basic.py)) auto-detects pandas Categorical dtype columns under `categorical_feature="auto"` — exactly what the workbench's `_ColumnRouter` produces for low-card.
- **Migration cost: rename `PASSTHROUGH_TO_XGB_CATEGORICAL` → `PASSTHROUGH_NATIVE_CATEGORICAL`** (same sentinel serves both families). No `_ColumnRouter` logic changes.

**Q-Ingest — No tier-switch needed.** PROVEN.
- LightGBM's `Dataset` always quantizes to uint8 histograms at construction (`max_bin=255` default), giving ~8x compression vs raw float ([basic.py @ v4.5.0](https://github.com/microsoft/LightGBM/blob/v4.5.0/python-package/lightgbm/basic.py)).
- Out-of-core path is file-based (`two_round=True`; `Sequence` API), not host-RAM-tier. Single in-memory tier suffices for workbench-scale data (RAM-bound 36 GB host; 1Mx50 → ~50 MB binned).
- Thin `src/rux_ml/training/lightgbm/ingest.py::build_dataset(x, y, data_cfg)` helper for symmetry with `xgboost/ingest.py`. No `lightgbm_in_memory_x_gb_max` knob.

**Q-Wrap — Complete kwarg-translation table.** PROVEN.
- `LGBMModel.__init__` ([sklearn.py L485-507 @ v4.5.0](https://github.com/microsoft/LightGBM/blob/v4.5.0/python-package/lightgbm/sklearn.py#L485-L507)) accepts `**kwargs: Any` — confirms Q-MK precondition for retaining `model_kwargs` on `LightGBMTraining`.
- **`early_stopping_rounds` is a FIT-TIME CALLBACK in v4.5+**, not a constructor kwarg ([callback.py L452](https://github.com/microsoft/LightGBM/blob/v4.5.0/python-package/lightgbm/callback.py#L452)). Diverges from XGBoost. Pattern A (Phase 4 sub-decision): factory returns a thin shim whose `fit()` injects `lightgbm.early_stopping(N)` callback when `eval_set` present.
- **`logloss` → `binary_logloss`** translation (LightGBM rejects "logloss" as `metric=` value). `_METRIC_TRANSLATE = {"logloss": "binary_logloss"}` in factory.
- `random_state` alone is NOT bit-exact — must pair with `deterministic=True` for PR-013 CPU bit-exact contract.
- `subsample` requires `subsample_freq > 0` to fire (default `subsample_freq=1` in `LightGBMTraining`).
- `max_depth=-1` means "no limit" in LightGBM (XGBoost uses 0).
- Sklearn-wrapper aliases: `min_data_in_leaf` → `min_child_samples`; `feature_fraction` → `colsample_bytree`; `bagging_fraction` → `subsample`; `bagging_freq` → `subsample_freq`; `lambda_l1` → `reg_alpha`; `lambda_l2` → `reg_lambda`.

**Q-HPO — Research-anchored 9-knob search space.** PROVEN.
- 7-of-9 hyperparameters have convergent evidence (Optuna canonical example [`optuna-examples/lightgbm/lightgbm_simple.py`](https://github.com/optuna/optuna-examples/blob/main/lightgbm/lightgbm_simple.py) + Optuna stepwise [`LightGBMTuner @ _lightgbm_tuner/optimize.py`](https://github.com/optuna/optuna-integration/blob/main/optuna_integration/lightgbm/_lightgbm_tuner/optimize.py) + LightGBM tuning guide [Parameters-Tuning](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html)).
- `learning_rate` and `max_depth` have source disagreement — rux-ml leans documented inline.
- `boosting_type` FIXED to `"gbdt"`, not searched (Optuna canonical + stepwise tuner both fix it).
- Output: [`configs/search_spaces/lightgbm.toml`](../configs/search_spaces/lightgbm.toml).

### Phase 4 — Sub-decisions (user-approved)

| # | Decision |
|---|---|
| 1 | `LightGBMTraining` inherits `device: Literal["cuda", "cpu"]` from `TrainingBase` unchanged; factory raises `NotImplementedError` on `cfg.device == "cuda"`. (No type-narrow override → no `# pyright: ignore`.) |
| 2 | Keep `early_stopping_rounds` on `TrainingBase` (family-agnostic concept); LightGBM factory provides Pattern-A shim that injects `lightgbm.early_stopping(N)` callback at fit time. |
| 3 | Rename `PASSTHROUGH_TO_XGB_CATEGORICAL` → `PASSTHROUGH_NATIVE_CATEGORICAL`. |
| 4 | OMIT `use_native` on `LightGBMTraining` for v0.1. |
| 5 | `deterministic: bool = False` (matches LightGBM default; opt in for PR-013 bit-exact tests). |
| 6 | Add `tests/training/test_lightgbm_smoke.py` with categorical column to exercise Q-Cat integration. |

### Verification artifacts

- `make test` → **310 passed**, 0 failed, 15 deselected (gpu/slow/golden/docker — gated). Was 305 pre-PR-018; added 5 new tests (1 conformance-test entry for lightgbm + 4 in `test_lightgbm_smoke.py`).
- `uv run basedpyright src/` → **0 errors, 0 warnings, 0 notes**.
- `uv run ruff check .` → **All checks passed**.
- Existing v0 + PR-017 tests unchanged. The XGBoost integration is untouched per `One PR, One Thing`.

---

## Scope

Add LightGBM as the second Trainer family, exercising the registry pattern PR-017 established.

Concretely:

- **New subpackage** `src/rux_ml/training/lightgbm/`:
  - `__init__.py` — exports `make_lightgbm_trainer` (the family factory)
  - `factory.py` — wraps `LGBMClassifier` / `LGBMRegressor` with lazy `import lightgbm` + clear-error-on-missing pattern (per Q3)
  - `config.py` — `LightGBMConfig` Pydantic schema (per Q5 co-location)
  - `dataset.py` (if needed) — analog of XGBoost's DMatrix-construction helpers, per ingest-path research #3
- **Registry entry** — add `"lightgbm": _make_lightgbm_trainer` to `TRAINER_FAMILIES` in `src/rux_ml/training/__init__.py`
- **PEP 631 extra** in `pyproject.toml`: `lightgbm = ["lightgbm>=4.5"]` (pin minimum per research)
- **Search-space TOML** — `configs/search_spaces/lightgbm.toml` with HPO ranges per research #4
- **Categorical-handling integration** — per research #2 outcome, either:
  - Extend `_ColumnRouter` to admit a LightGBM-specific path, or
  - Document that LightGBM bypasses `_ColumnRouter` (with rationale)
- **Conformance test** — `tests/training/test_registry_conformance.py` (introduced by PR-017) automatically picks up the new family; assert it passes for LightGBM
- **End-to-end smoke test** — `tests/training/test_lightgbm_smoke.py` with a tiny fixed-seed dataset, fit-then-predict round trip, AUC tolerance band (per the v0 golden-test pattern from PR-014)
- **GPU smoke test** (gated by `@pytest.mark.gpu`) — train on the workbench's RTX 4090, asserts the GPU code path executes
- **Codify A3 in `docs/CONVENTIONS.md`** — extras-naming convention (this PR's commit)
- **Amend `docs/ARCHITECTURE.md`** — categorical-encoding decision rule per research #2; ingest-path decision rule per research #3
- **CHANGELOG `[Unreleased]`** — user-facing entry for the new family

Out of scope: CatBoost (PR-019), Solver layer (PR-020), v0.1.0 cut (PR-021), any changes to XGBoost integration.

## Dependencies

**PR-017** (Trainer registry refactor) — must be merged first. PR-018 relies on the `TRAINER_FAMILIES` dict + the subpackage-layout convention + the `Trainer` Protocol's confirmed shape.

## Architecture section implemented

`docs/ARCHITECTURE.md` "Components" — adds LightGBM to the configured trainer families table.

`docs/ARCHITECTURE.md` "Decision Rules" — adds LightGBM-specific categorical-encoding rule per research #2 and ingest-path rule per research #3.

`docs/CONVENTIONS.md` — codifies the extras-naming convention (per A3).

`docs/0.1/DESIGN-log.md` Q1–Q5 — first exercise of the multi-family pattern.

## Verification criteria

Populated after research. Initial sketch (refine in Phase 2-3):

- [ ] `src/rux_ml/training/lightgbm/` subpackage exists with at least `__init__.py`, `factory.py`, `config.py`
- [ ] `TRAINER_FAMILIES["lightgbm"]` resolves to the LightGBM factory
- [ ] `pyproject.toml` declares `[lightgbm]` extra with pinned minimum version
- [ ] `uv sync --extra lightgbm` works
- [ ] Running without `[lightgbm]` installed gives a clear `ImportError` on family instantiation (not at module-load time)
- [ ] `configs/search_spaces/lightgbm.toml` exists with research-anchored HPO ranges
- [ ] `tests/training/test_lightgbm_smoke.py` passes — fit-then-predict round trip
- [ ] `tests/training/test_registry_conformance.py` passes for the new family
- [ ] GPU smoke test passes on the workbench's RTX 4090 (or is documented as out-of-scope for this PR with a follow-up issue if RTX 4090 LightGBM-CUDA is found unavailable per research #1)
- [ ] `docs/CONVENTIONS.md` carries the extras-naming convention (A3)
- [ ] `docs/ARCHITECTURE.md` carries the LightGBM-specific decision rules
- [ ] `CHANGELOG.md` `[Unreleased]` has the LightGBM-family entry
- [ ] `docs/0.1/ROADMAP.md` PR-018 row flipped `[ ]` → `[x]` in this PR's commit
- [ ] `docs/0.1/RESEARCH-BACKLOG.md` PR-018 row marked `fully-researched YYYY-MM-DD` + `implementation-cleared YYYY-MM-DD`

## Research backing

Architectural pattern locked by `docs/0.1/DESIGN-log.md` Q1–Q5 (2026-05-18 design session). LightGBM-specific decisions require Phase 2-3 web research per the five open questions above.

Reference precedents:

- [LightGBM GPU tutorial](https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html)
- [LightGBM `Dataset` reference](https://lightgbm.readthedocs.io/en/latest/Python-API.html#data-structure-api)
- [LightGBM categorical features doc](https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html#categorical-feature-support)
- LightGBM-Optuna integration: [`optuna-integration` LightGBM module](https://github.com/optuna/optuna-integration/tree/main/optuna_integration/lightgbm)

## Notes

- The **GPU question (#1)** is the highest-risk open research. If LightGBM-CUDA doesn't work on consumer RTX 4090 cards, the workbench either accepts CPU-only LightGBM (significant performance regression vs XGBoost-CUDA) or falls back to OpenCL. Document the outcome clearly.
- The **categorical-handling decision (#2)** has architectural consequences: if LightGBM bypasses `_ColumnRouter`, the per-family ingest path becomes a real concern that PR-019 (CatBoost) will face independently.
- This PR is **parallelizable with PR-019** post-017. Both add a Trainer family to the same registry; they don't conflict.
- Per `One PR, One Thing`, LightGBM-specific research belongs in this PR. The pattern was settled in `docs/0.1/DESIGN-log.md`; this PR exercises and validates it.
