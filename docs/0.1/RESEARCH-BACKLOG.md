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
| [PR-017](../../prs/PR-017-trainer-registry-refactor.md) | Trainer registry refactor | `design-research ~` → **`fully-researched 2026-05-18`** + **`implementation-cleared 2026-05-18`**. Resolved: (1) Q-A1 + Q-A1b + Q-MK — discriminated union with `kind` discriminator + shared `TrainingBase` + per-variant typed params; `model_kwargs` retained only on variants whose upstream accepts `**kwargs` (XGBoost ✓, LightGBM ✓, CatBoost ✗). (2) Q-CT — NO `@runtime_checkable`; behavioral parametrize-over-registry test pattern anchored on sklearn `parametrize_with_checks` + Optuna `pytest_samplers.py` (PEP 544 + CPython 3.12 typing docs). (3) Q-Dep — xgboost stays HARD-required; PR-017 does NOT touch `pyproject.toml` (scope reduction); only siblings are optional extras (PR-018+). (4) Manifest schema — no migration needed (no explicit `family` field; implicit via `training_cfg_hash`). (5) A5 — codified `docs/VERSIONING.md §1` MINOR triggers update + family-removal-as-MINOR rule. Gate Check sub-decisions: `TrainingBase` in `training/base.py` (inlined model_config to avoid circular import via `rux_ml.config.__init__`); `kind` declared on each variant only; TOMLs require explicit `kind = "xgboost"` under `[training]` (Pydantic dispatch runs before defaults). |
| [PR-018](../../prs/PR-018-lightgbm-family.md) | LightGBM Trainer family | `design-research ~` → **`fully-researched 2026-05-18`** + **`implementation-cleared 2026-05-18`**. Resolved: (1) Q-GPU — ship CPU-only; LightGBM-GPU is 8-28x slower than XGBoost-GPU on workbench-scale data per szilard/GBM-perf benchmarks; `cfg.device == "cuda"` raises `NotImplementedError` with deferred-to-follow-up-PR message. (2) Q-Cat — near-zero `_ColumnRouter` change; rename `PASSTHROUGH_TO_XGB_CATEGORICAL` → `PASSTHROUGH_NATIVE_CATEGORICAL` (same sentinel serves both families); LightGBM's `categorical_feature="auto"` auto-detects pandas Categorical from the existing pipeline output; high-card target encoding is what LightGBM itself recommends per Pargent 2021. (3) Q-Ingest — no tier-switch needed; LightGBM `Dataset` always uint8-bins (~8x compression vs raw float). (4) Q-Wrap — complete kwarg-translation table with surprises: `early_stopping_rounds` is fit-time callback (Pattern-A shim handles); `logloss` → `binary_logloss` translation; `random_state` alone NOT bit-exact (needs `deterministic=True`); `subsample` requires `subsample_freq > 0`. (5) Q-HPO — 9-knob search-space TOML template anchored on Optuna canonical example + stepwise tuner. (6) A3 — extras-naming convention codified in CONVENTIONS.md: `[<family>]` named after family, not backend package. |
| [PR-019](../../prs/PR-019-catboost-family.md) | CatBoost Trainer family | `design-research ~` → **`fully-researched 2026-05-18`** + **`implementation-cleared 2026-05-18`**. Resolved: (1) Q-GPU — DEFAULT to GPU (opposite of LightGBM); CatBoost ships prebuilt CUDA wheels via uv; RTX 4090 field-confirmed; `task_type="GPU"` + `devices="0"` when cfg.device="cuda". GPU bit-exact NOT achievable (issue #546). (2) Q-Cat — `_ColumnRouter` unchanged; `_CatBoostTrainerShim` extracts `cat_features=[col_names]` from DataFrame via `select_dtypes(include="category")` at fit time (CatBoost does NOT auto-detect Categorical dtype, opposite of LightGBM). (3) Q-Parallel — CatBoost uses Intel TBB, NOT OpenMP; workbench's OMP_NUM_THREADS pinning is invisible; factory reads env and passes as `thread_count=` explicitly. CPU bit-exact recipe documented in CONVENTIONS.md; dedicated determinism test deferred to follow-up Tier-2 PR. (4) Q-Err — NO translation layer; `CatBoostError` is flat Exception, OOM usually SIGKILL'd, GPU OOM uncatchable; subprocess isolation handles. (5) Q-HPO — 7-knob search space (anchored on Optuna catboost_simple.py + CatBoost team's tutorial). iterations + grow_policy fixed. (6) Q-Wrap — 14-field `CatBoostTraining` schema (no `**kwargs`); `_METRIC_TRANSLATE = {"auc": "AUC", "logloss": "Logloss", "rmse": "RMSE", "mae": "MAE"}`; loss_function task-derived; early_stopping_rounds is top-level ctor kwarg; bootstrap_type ↔ randomness-kwarg conditional (Bayesian rejects `subsample`; Bernoulli/MVS/Poisson reject `bagging_temperature`). |
| [PR-020](../../prs/PR-020-solver-protocol.md) | Solver Protocol + first Solver family | `design-research ~` → **`fully-researched 2026-05-18`** + **`implementation-cleared 2026-05-18`**. Resolved: (1) Q-First-Solver — **CVXPY** (zero dep addition; covers LP/QP/QCQP/SOCP/SDP/MILP via one `solver=` switch; pin `CLARABEL` default for determinism; future PRs add clarabel-direct/OSQP-direct as backends if DCP overhead > 5%). (2) Q-Shape — **B-partial-mirror** (Optuna + MLflow precedent: reuse cross-cutting substrate, drop semantics-mismatched layers; skfolio's A-full-mirror only works because portfolio data has a natural reshape). (3) Q-Registry — skip for v0.1; Optuna study + user_attrs provenance is sufficient. (4) Q-HPO — defer solver-internal HPO; `rux-ml solve` is one-shot only. (5) Q-CLI — single command (analog of `rux-ml train`), not a typer-group. (6) Q-Provenance — single `TrialAttrs` with Optional solver fields (`solving_cfg_hash`, `solver_status`, `objective_value`, `solver_iter_count`, `solve_time_s`); Trainer trials work unchanged. Gate Check sub-decisions: `SolverResult` is a plain dataclass; `solver_opts` escape hatch handles per-backend setting-name divergence (Clarabel `tol_gap_abs` vs OSQP `eps_abs`); solver-trial data hashes use `"none:solver-trial"` placeholders. |
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
