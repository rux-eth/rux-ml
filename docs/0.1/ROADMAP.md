# Roadmap — v0.1

Ordered list of PRs for the v0.1 cut. Each PR is a single, reviewable change. Dependencies flow downward — no PR should be started before its predecessors are merged.

Full PR descriptions live in `prs/`. This file is the index.

The v0.0 roadmap (PR-001 through PR-016) is frozen at [`docs/0.0/ROADMAP.md`](../0.0/ROADMAP.md) — every row is `[x]`.

**Status legend:** `[ ]` pending | `[x]` merged | `[~]` in progress

---

## Phase I: Multi-family extensibility

| PR | Description | Status | Depends on |
|----|-------------|--------|------------|
| [PR-017](../../prs/PR-017-trainer-registry-refactor.md) | **Tier-2** Trainer registry refactor — move `xgboost` integration into `src/rux_ml/training/xgboost/` subpackage; add `TRAINER_FAMILIES: dict[str, Callable]` in `src/rux_ml/training/__init__.py`; Protocol-conformance test that iterates the registry; **scope reduction** per Q-Dep: xgboost stays hard, `pyproject.toml` UNTOUCHED; amend `docs/VERSIONING.md §1` for the family-removal-as-MINOR rule (A5); resolve `RuxMLConfig.training` shape (A1, discriminated union with `kind`); amend `docs/ARCHITECTURE.md` + `docs/CONVENTIONS.md` in the same commit | `[x]` | — |
| [PR-018](../../prs/PR-018-lightgbm-family.md) | **Tier-2** LightGBM Trainer family — `src/rux_ml/training/lightgbm/`; `[lightgbm]` PEP 631 extra (introduces `[project.optional-dependencies]` table); registry entry; Pydantic config schema co-located; search-space TOML at `configs/search_spaces/lightgbm.toml`; codify extras-naming convention in `CONVENTIONS.md` (A3); **scope reduction** per Q-GPU: ship CPU-only, defer GPU to follow-up PR (LightGBM-GPU is 8-28x slower than XGBoost-GPU on workbench-scale data per szilard/GBM-perf benchmarks; install path via uv is non-trivial). Q-Cat: rename `PASSTHROUGH_TO_XGB_CATEGORICAL` → `PASSTHROUGH_NATIVE_CATEGORICAL` (same sentinel serves both families). Pattern-A shim for fit-time early-stopping callback injection. | `[x]` | PR-017 |
| [PR-019](../../prs/PR-019-catboost-family.md) | **Tier-2** CatBoost Trainer family — `src/rux_ml/training/catboost/` with 14-field `CatBoostTraining` (no `**kwargs` per CatBoost outlier surface); `[catboost]` extra; registry entry; search-space TOML (7 search dims). **GPU is the default** (Q-GPU: CatBoost ships prebuilt CUDA wheels via uv; RTX 4090 field-confirmed). Pattern-A `_CatBoostTrainerShim` for `cat_features` auto-extraction (Q-Cat: CatBoost doesn't auto-detect pandas Categorical, opposite of LightGBM). Factory reads `OMP_NUM_THREADS` env and passes as `thread_count=` (Q-Parallel: CatBoost uses Intel TBB, NOT OpenMP). No `errors.py` translation layer (Q-Err: subprocess isolation handles). | `[x]` | PR-017 |
| [PR-020](../../prs/PR-020-solver-protocol.md) | **Tier-2** Solver Protocol + first Solver family (bundled to avoid phantom) — new `Solver` `typing.Protocol`; `src/rux_ml/solving/__init__.py` with `SOLVER_FAMILIES` dict; `make_solver(cfg)` factory; first Solver subpackage (TBD, requires Tier-2 research — OSQP / Clarabel / scipy.optimize / cvxpy-wrapped); `rux-ml solve` CLI verb (A4); `[<solver>]` extra; Protocol-conformance test for the solving registry | `[ ]` | PR-017 |
| [PR-021](../../prs/PR-021-v0-1-0-cut.md) | **Tier-2** v0.1.0 version cut — update `scripts/rewrite_doc_refs.py` PATH_REWRITES for `docs/0.0/* → docs/0.1/*` migration; resolve the versioned→versioned mapping wrinkle (A6 — opt-out list or smarter regex for historical citations); bump `pyproject.toml` `version` `0.0.1 → 0.1.0`; bump `src/rux_ml/__init__.py` `__version__`; rewrite `CHANGELOG.md` `[Unreleased]` → `[0.1.0] - YYYY-MM-DD`; tag `v0.1.0` at merge per `docs/VERSIONING.md §7` | `[ ]` | PR-017, PR-018, PR-019, PR-020 |

---

## Notes

- Every v0.1 PR is **Tier-2** per memory `feedback_tier_inheritance` and the per-family research surface — the architectural pattern was research-backed by the 2026-05-18 design session, but per-family integration details (LightGBM GPU on RTX 4090, CatBoost categorical handling, choice of first Solver family) are not. Each PR runs the full 5-phase `PROCEDURE-pr-research.md` before implementation.
- PRs 018 + 019 are **parallelizable post-017** — independent of each other. PR-020 can be drafted in parallel but blocks on its own Tier-2 research before implementation.
- PR-021 is the **version-cut ritual** per `docs/VERSIONING.md §2` workflow step 6. It lands last, after all v0.1 implementation PRs are merged on `dev`. The `v0.1.0` git tag lands at PR-021's merge commit.
- **Six ambiguities deferred from the design session** (A1–A6 in [`docs/0.1/DESIGN-log.md`](DESIGN-log.md) "Deferred research" section) resolve in specific PRs above — each PR's `## Research findings` section must address its assigned ambiguity before implementation begins.
- Phase I is intentionally small — the workbench's v0.0 surface is already capable for single-family research; v0.1 adds breadth, not depth. Each PR keeps One-PR-One-Thing discipline. CV strategy revisits per problem ([`feedback_cv_strategy_tier2`](MEMORY.md)) and the first Rust crate per D13's profile-driven trigger are post-v0.1 work.
- Per `docs/CONSTRAINTS.md` (Per-Phase Approval Gate, Research-Backed Decisions, No Phantom Implementations): each PR includes a real test exercising actual end-to-end behavior. PR-020's bundling of Protocol + first Solver family explicitly avoids the phantom-implementation trap that would result from shipping the Protocol without a real consumer.
