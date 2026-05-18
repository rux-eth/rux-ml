# PR-018: LightGBM Trainer family

**Landed-in:** (not yet landed)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PRs** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

**This PR is Tier-2** per `docs/0.1/RESEARCH-BACKLOG.md`. The registry pattern is research-backed by PR-017; LightGBM-specific instantiation is not. All 5 phases of `PROCEDURE-pr-research.md` must run.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

## Research findings

_To be populated by `PROCEDURE-pr-research.md`. Do not begin implementation until this section exists with completed findings from all required phases._

**Open research questions** (must be resolved before implementation):

1. **LightGBM-CUDA on consumer RTX 4090.** Does the official LightGBM CUDA build work on consumer cards (RTX 4090 = Ada Lovelace, compute capability 8.9)? Or is OpenCL required for consumer GPU support? Success criteria: a working `gpu_use_dp=true` or equivalent on the workbench's actual hardware, with documented GPU-build install path. Cite LightGBM official GPU tutorial + GitHub issues for consumer-card support.

2. **LightGBM categorical handling vs the v0 `_ColumnRouter`** (PR-005). LightGBM has native categorical support via `categorical_feature=` parameter that bypasses one-hot encoding. Does the v0 `_ColumnRouter` strategy (per the categorical-encoding decision rule in `docs/ARCHITECTURE.md`) need a LightGBM-specific override? Success criteria: documented decision on whether LightGBM bypasses or integrates with `_ColumnRouter`. Cite LightGBM docs + benchmarks.

3. **`Dataset` (LightGBM's `DMatrix`-equivalent) memory profile.** The workbench's RAM-bound 36 GB constraint shaped the v0 XGBoost ingest-path decision rule (per D3 — `QuantileDMatrix` vs `ExtMemQuantileDMatrix`). Does LightGBM's `Dataset` have analogous tiers? Success criteria: a LightGBM-specific ingest-path decision rule for `docs/ARCHITECTURE.md`. Cite LightGBM `Dataset` source + memory benchmarks.

4. **Canonical LightGBM HPO search-space.** What hyperparameters belong in `configs/search_spaces/lightgbm.toml`? At minimum: `learning_rate`, `num_leaves`, `max_depth`, `min_data_in_leaf`, `feature_fraction`, `bagging_fraction`, `lambda_l1`, `lambda_l2`. Success criteria: research-anchored ranges + log/linear scales cited from production references (Kaggle Grandmaster posts, LightGBM-Optuna integration examples, mature open-source repos).

5. **A3 — Extras-naming convention.** Codify into `docs/CONVENTIONS.md` in this PR's commit: per-family extras named after the family (`[lightgbm]`), not after the backend (`[scikit-learn-lightgbm]`). Document the rule for future families.

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
