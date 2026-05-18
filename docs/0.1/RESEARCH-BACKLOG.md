# Research Backlog — v0.1

Index of v0.1 PRs with research status. Every PR runs `PROCEDURE-pr-research.md` before implementation — this document tracks the state of each PR's research and flags drift risk per the time-decay policy.

The v0.0 research backlog (PR-001 through PR-016) is frozen at [`docs/0.0/RESEARCH-BACKLOG.md`](../0.0/RESEARCH-BACKLOG.md) — all rows are `implementation-cleared`.

**Tiers:**
- **Tier 1** — Design-time research exists. Phase 1 (State Assessment) required before implementation; Phases 2-4 may be light if no drift found.
- **Tier 2** — Design-time research is partial or absent. Full 5-phase procedure required before the PR is written in final form.

**Status legend:**
- `design-research ✓` — research done at design time
- `design-research ~` — partial design research (pattern locked, per-instance details open)
- `design-research ✗` — no design-time research
- `state-assessed YYYY-MM-DD` — Phase 1 of `PROCEDURE-pr-research.md` completed
- `fully-researched YYYY-MM-DD` — all 5 phases of `PROCEDURE-pr-research.md` completed
- `implementation-cleared YYYY-MM-DD` — Phase 5 Gate Check passed

---

## Tier 1 — Implementation-Ready

**None.** All v0.1 PRs are Tier-2 — see below. The 2026-05-18 design session (`docs/0.1/DESIGN-log.md`) research-backed the **architectural pattern** (two Protocols, subpackage-per-family, optional extras, B-explicit registry, hybrid config, MINOR-with-deprecation), but per-family integration details are unresolved.

## Tier 2 — Research-Pending

| PR | Title | Design research | Required research topics |
|----|-------|-----------------|--------------------------|
| [PR-017](../../prs/PR-017-trainer-registry-refactor.md) | Trainer registry refactor | `design-research ~` (Q1–Q6 architectural pattern; per-instance details open) | (1) Pydantic discriminated-union vs registry-of-blocks for `RuxMLConfig.training` (A1) — type-safety, validation, error-message quality, migration cost from existing v0 single-family block. (2) `typing.Protocol` `runtime_checkable` signature-checking limits — what does the conformance test actually assert? Method names only, or duck-typed full-signature? (3) Manifest-schema migration — existing v0 `manifest.json` files may carry an implicit `family: "xgboost"` or no family field; what's the back-compat path? (4) A5 — codify `docs/VERSIONING.md §1` MINOR triggers update + family-removal-as-MINOR rule. |
| [PR-018](../../prs/PR-018-lightgbm-family.md) | LightGBM Trainer family | `design-research ~` | (1) LightGBM-CUDA on consumer RTX 4090 — does the official build work on consumer cards, or is OpenCL required? Performance comparison vs XGBoost-CUDA on representative tabular workloads. (2) LightGBM categorical handling — does the `_ColumnRouter` (PR-005) integrate cleanly, or does LightGBM need its own per-family categorical strategy? (3) `Dataset` memory profile vs XGBoost `DMatrix` — does the workbench's RAM-bound 36 GB constraint shift the ingest-path decision rule for LightGBM? (4) Canonical LightGBM HPO search-space — what hyperparameters belong in `configs/search_spaces/lightgbm.toml`? (5) A3 — codify extras-naming convention in `CONVENTIONS.md`. |
| [PR-019](../../prs/PR-019-catboost-family.md) | CatBoost Trainer family | `design-research ~` | (1) CatBoost native categorical handling vs the v0 `_ColumnRouter` + categorical-encoding decision rule — does CatBoost obsolete the `categorical_low_card_threshold` BEST-GUESS for catboost-only studies? (2) Symmetric tree structure parallelism — does CatBoost saturate the 4090 the way XGBoost does, or is the thread-pinning strategy different? (3) `CatBoostError` normalization into `MemoryPressureError` — does the v0 memory watchdog (PR-011) catch CatBoost OOM the same way? (4) Canonical CatBoost HPO search-space. |
| [PR-020](../../prs/PR-020-solver-protocol.md) | Solver Protocol + first Solver family | `design-research ~` (Q1 lean validated; per-instance open) | (1) First Solver family choice — OSQP / Clarabel / SCS / scipy.optimize / cvxpy-wrapped. Tradeoffs: install footprint, license, problem-class coverage (QP-only vs QP+SOCP+SDP), Python-side ergonomics. (2) Canonical `Solver` Protocol surface — `solve(problem) → result` is the lean from Q1, but what's the concrete `problem` type? cvxpy-style structured object? raw matrices? (3) A4 — `rux-ml solve` CLI verb design — what subcommands mirror `rux-ml train`? Does provenance triple extend to solver runs the same way? (4) Solver-side study integration — does Optuna's K-fold CV-mean objective (PR-007) translate to solver tuning, or is the HPO surface fundamentally different for constrained optimization? |
| [PR-021](../../prs/PR-021-v0-1-0-cut.md) | v0.1.0 version cut | `design-research ~` | (1) A6 — `scripts/rewrite_doc_refs.py` versioned→versioned mapping: opt-out list (explicit set of refs that should stay pinned to `docs/0.0/*`) vs smarter regex (e.g., heuristic for "see Dn in"). Test the rewrite carefully on the v0.0 → v0.1 cut. (2) CHANGELOG audit — what user-facing changes from PR-017 through PR-020 belong in `[0.1.0]`? (3) Post-merge tag command + verification (per `docs/VERSIONING.md §7`). |

---

## Drift Watch

Per the time-decay policy in `PROCEDURE-pr-research.md`, any PR marked `fully-researched` or `state-assessed` more than the project's staleness threshold before implementation begins must re-run Phase 1 (State Assessment).

**Project staleness threshold:** **60 days** (recorded in [`docs/CONSTRAINTS.md`](../CONSTRAINTS.md); BEST-GUESS, user-acknowledged).

**Currently watching:** the v0.1 design session completed 2026-05-18. Each PR's `fully-researched` date must complete by **2026-07-17** for direct use, or be revalidated. Tier-2 PRs do NOT inherit the design-time staleness window — each PR's `state-assessed` date starts the clock from whenever its full 5-phase research completes.

### Per-PR drift-risk notes

- **PR-017** — XGBoost API + Pydantic v2 release cadence within the staleness window. Specifically: any change to `xgboost.XGBClassifier` interface, sklearn-base ABC changes, or Pydantic v2 discriminator semantics. Phase 1 should check XGBoost release notes + `xgboost.__version__` vs the locked v0 version.
- **PR-018 + PR-019** — LightGBM and CatBoost release cycles. Both projects ship quarterly-ish; check for CUDA support changes, API deprecations, and any reported regressions on consumer RTX 4090. NVIDIA's driver cadence also matters for the container's CUDA 12.4.1 pin.
- **PR-020** — Solver-library landscape is fast-moving (Clarabel is younger than OSQP; cvxpy's solver list changes). Phase 1 should re-survey the QP solver landscape at implementation time.
- **PR-021** — `docs/VERSIONING.md` may have been amended by PR-017 (per A5); the rewrite script's mapping table must reflect those amendments. Cross-check before running.

---

## Design References

- [`docs/0.1/DESIGN-log.md`](DESIGN-log.md) — v0.1 design session (Q1–Q6 decisions + research trail + deferred A1–A6 ambiguities)
- [`docs/0.1/ROADMAP.md`](ROADMAP.md) — v0.1 PR plan
- [`docs/0.0/DESIGN-log.md`](../0.0/DESIGN-log.md) — v0.0 design history (D1–D17 + PR-015 CV + PR-016 migration); decisions and constraints remain in force unless v0.1 DESIGN-log explicitly supersedes
- [`docs/VERSIONING.md`](../VERSIONING.md) — versioning policy + bump rules + changelog format (flat, meta-rule)
- [`docs/CONSTRAINTS.md`](../CONSTRAINTS.md) — hard rules (carried forward from v0; nothing new in v0.1 yet)
