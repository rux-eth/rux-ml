# PR-020: Solver Protocol + first Solver family

**Landed-in:** v0.1.0

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PRs** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

**This PR is Tier-2** per `docs/0.1/RESEARCH-BACKLOG.md`. The two-Protocols-no-shared-parent lean (Q1) is research-backed; Solver-specific instantiation is not. All 5 phases of `PROCEDURE-pr-research.md` must run.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

## Research findings

`PROCEDURE-pr-research.md` 5-phase Tier-2 procedure completed 2026-05-18.

### State Assessment (2026-05-18) — Phase 1

**Surprise**: `cvxpy 1.8.2` + `clarabel 0.11.1` are **already runtime-installed** via `skfolio>=0.20.1` (hard dep from PR-015). Promoted to direct deps in PR-020 rather than relying on transitive.

**Impedance mismatch surfaced** between Solver and the workbench's existing layers — Phase 1 elevated this to a top-level Q-Shape question (which the stub didn't anticipate):

| Workbench layer | Tabular shape | Solver shape | Mismatch |
|---|---|---|---|
| `cfg.data` | Polars/Parquet + `target_column` | matrices `(P, q, A, l, u)` or `cvxpy.Problem` | Yes |
| `cfg.cv` | K-fold / TimeSeriesSplit / CPCV | single solve, no folds | Yes |
| Registry bundle | `pipeline.skops` + `model.ubj` + manifest | `x_star` solution vector | Yes |
| Optuna study (PR-007 objective) | K-fold-CV-mean | single-trial solve | Yes |
| `TrialAttrs` schema | Trainer-specific required fields | needs solver-specific fields | Yes |

### Phase 3 — Research findings

**Q-First-Solver — CVXPY.** PROVEN.
- Zero dep addition (cvxpy + clarabel already transitive via skfolio per Phase 1 surprise).
- Covers LP/QP/QCQP/SOCP/SDP/MILP through one `solver=` switch.
- Pin `solver="CLARABEL"` in TOML for per-trial provenance stability. Clarabel ranks #3 in [qpsolvers/free_for_all_qpbenchmark](https://github.com/qpsolvers/free_for_all_qpbenchmark/blob/main/results/free_for_all.md); modern Rust solver; Apache-2.0.
- Source pins: [cvxpy v1.8.2](https://github.com/cvxpy/cvxpy), [clarabel v0.11.1](https://github.com/oxfordcontrol/Clarabel.rs).
- Trade-off: cvxpy adds DCP compilation overhead on every `solve()`; future PRs may register clarabel-direct / OSQP-direct as backends if profiled > 5% (analog of the v0 D13 "first Rust crate" threshold).

**Q-Shape — B (partial mirror).** CONVENTION.
- skfolio (already in workbench deps) uses A-full-mirror but only because portfolio data has a natural `X = returns matrix` reshape — QP matrices `(P, q, A, l, u)` lack that. (Source: [skfolio/optimization/_mean_risk.py](https://raw.githubusercontent.com/skfolio/skfolio/main/src/skfolio/optimization/convex/_mean_risk.py))
- Optuna + MLflow validate substrate-reuse without forcing fittable-model assumptions. (Sources: [Optuna first-tutorial](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/001_first.html), [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/))
- C-utility-module has no documented precedent in workbench-shaped projects.
- Decision: keep config-layer + registry-dict + factory + Optuna study + per-trial provenance; drop cfg.data / cfg.cv / Trainer-shaped registry for solver trials.

### Phase 4 — Locked sub-decisions

| # | Decision |
|---|---|
| 1 | **Q-Shape = B (partial mirror)** — keep cross-cutting plumbing, drop semantics-mismatched layers. |
| 2 | **Q-First-Solver = CVXPY** — pin `solver="CLARABEL"` default; `solver_opts: dict[str, Any]` escape hatch for backend-specific knobs. |
| 3 | **Q-Registry = skip for v0.1** — Optuna study + `user_attrs` provenance is sufficient; no solver bundle in PR-020. |
| 4 | **Q-HPO = defer** — `rux-ml solve` is one-shot only; solver-internal HPO is a follow-up Tier-2 PR. |
| 5 | **Q-CLI = single command** `rux-ml solve` (analog of `rux-ml train`; not a typer-group). |
| 6 | **Q-Provenance = single `TrialAttrs` with Optional solver fields** — `solving_cfg_hash` + `solver_status` + `objective_value` + `solver_iter_count` + `solve_time_s`, all Optional. Trainer trials work unchanged. |

### Phase 5 — Gate Check sub-decisions (locked during implementation)

- **`SolverResult` is a plain `@dataclass`** (not Pydantic) — runtime result type, not config.
- **`solving_cfg_hash` covers `problem_module` path but NOT the module's source content.** Documented limitation in `docs/CONVENTIONS.md`; users keep problem modules inside the workbench's git so `git_sha` covers their content.
- **Solver-trial data hashes use placeholder `"none:solver-trial"`** — solver runs have no Parquet input. `TrialAttrs` schema validates string presence, not content.
- **Backend setting names differ across cvxpy backends** (Clarabel uses `tol_gap_abs` / `tol_feas` / `max_iter`; OSQP uses `eps_abs` / `eps_rel`). Rather than fake unified abstraction across names, the workbench passes `solver_opts: dict[str, Any]` straight through (matches the Trainer-side `model_kwargs` escape hatch pattern).
- **`solver_opts` flows through `Problem.solve(solver=..., **solver_opts)`** — users consult the backend's docs for the right setting names.

### Verification artifacts

- `make test` → **324 passed**, 0 failed, 15 deselected. Was 319 pre-PR-020; added 5 new tests (2 in conformance + 3 in cvxpy smoke).
- `uv run basedpyright src/` → **0 errors, 0 warnings, 0 notes**.
- `uv run ruff check .` → **All checks passed**.
- `rux-ml solve --help` registers cleanly in the CLI; XGBoost / LightGBM / CatBoost integrations untouched per `One PR, One Thing`.

---

## Scope

Introduce the second Protocol surface (`Solver`) alongside the first concrete Solver family. Bundled deliberately to avoid the phantom-implementation trap that would result from shipping the Protocol with an empty registry.

Concretely:

- **New top-level layer** `src/rux_ml/solving/`:
  - `__init__.py` — exports `make_solver` + `SOLVER_FAMILIES: dict[str, Callable[[RuxMLConfig], Solver]]`
  - `protocol.py` — the `Solver` `typing.Protocol` per research #2
  - `<first-family>/` subpackage (TBD per research #1) — `__init__.py`, `factory.py`, `config.py`
- **Registry entry** — add `"<first-family>": _make_<first-family>_solver` to `SOLVER_FAMILIES`
- **`RuxMLConfig.solving` block** — analog of `cfg.training`, with a `family` selector (shape per A1 resolution in PR-017)
- **PEP 631 extra** in `pyproject.toml`: `<first-family> = [...]`
- **`rux-ml solve` CLI verb** per research #3 — wire as a Typer subcommand in `src/rux_ml/cli/`
- **Conformance test** — `tests/solving/test_registry_conformance.py` parallel to the training-side test
- **End-to-end smoke test** — `tests/solving/test_<first-family>_smoke.py` — construct a small QP problem, solve, verify result quality
- **Amend `docs/ARCHITECTURE.md`**:
  - New "Components" entry for the `solving/` layer
  - New "Key Abstractions" section for `Solver` Protocol
  - Solver-side data flow diagram (or text equivalent)
- **Amend `docs/CONVENTIONS.md`** per research #3 — `rux-ml solve` verb design + Solver subpackage layout (mirrors Trainer)
- **Optional: amend Reproducibility Architecture** if research #5 surfaces a need
- **CHANGELOG `[Unreleased]`** — user-facing entry for the new layer + first Solver family

Out of scope: additional Solver families (post-v0.1 PRs), Solver tuning via Optuna (deferred to a future PR if research #4 indicates need), MOSEK/Gurobi/CPLEX commercial solver support.

## Dependencies

**PR-017** (Trainer registry refactor) — must be merged first. PR-020 reuses the registry-dict + conformance-test pattern.

Optional: parallelizable with PR-018 / PR-019 in terms of code, but Phase 2-3 research can run in parallel with theirs.

## Architecture section implemented

`docs/ARCHITECTURE.md` — new top-level "Components" entry for `solving/`; new "Key Abstractions" section for `Solver` Protocol; data-flow diagram updated to show the Solver path.

`docs/CONVENTIONS.md` — `rux-ml solve` verb design + Solver subpackage layout.

`docs/0.1/DESIGN-log.md` Q1 (two-Protocol pattern) — first concrete instantiation of the `Solver` side.

## Verification criteria

Populated after research. Initial sketch (refine in Phase 2-3):

- [ ] `src/rux_ml/solving/` top-level layer exists with `__init__.py`, `protocol.py`, and `<first-family>/` subpackage
- [ ] `Solver` Protocol defined per research #2
- [ ] `SOLVER_FAMILIES["<first-family>"]` resolves to the family factory
- [ ] `RuxMLConfig.solving` block validates correctly
- [ ] `pyproject.toml` declares `[<first-family>]` extra with pinned minimum version
- [ ] `uv sync --extra <first-family>` works
- [ ] Running without `[<first-family>]` installed gives a clear `ImportError` on family instantiation
- [ ] `rux-ml solve --help` shows the new verb's subcommands
- [ ] `tests/solving/test_<first-family>_smoke.py` passes — solves a small QP problem
- [ ] `tests/solving/test_registry_conformance.py` passes
- [ ] `docs/ARCHITECTURE.md` describes the `Solver` Protocol + the solving layer
- [ ] `docs/CONVENTIONS.md` codifies the `rux-ml solve` verb design (A4)
- [ ] `CHANGELOG.md` `[Unreleased]` has the Solver-layer + first-family entries
- [ ] `docs/0.1/ROADMAP.md` PR-020 row flipped `[ ]` → `[x]` in this PR's commit
- [ ] `docs/0.1/RESEARCH-BACKLOG.md` PR-020 row marked `fully-researched YYYY-MM-DD` + `implementation-cleared YYYY-MM-DD`

## Research backing

Q1 from `docs/0.1/DESIGN-log.md` (two-Protocol pattern, no shared parent) is the architectural foundation:

- Optuna `BaseSampler` / `BasePruner` independent ABCs ([`v4.8.0`](https://github.com/optuna/optuna/blob/v4.8.0/optuna/samplers/_base.py))
- cvxpy / pyomo / scipy.optimize solver-side data flow fundamentally different from `fit(X,y) → predict` — cvxpy `SOLVER_MAP_QP` ([`v1.5.0`](https://github.com/cvxpy/cvxpy/blob/v1.5.0/cvxpy/reductions/solvers/defines.py))

Solver-specific decisions require Phase 2-3 web research per the five open questions above.

Reference precedents for the Protocol surface:

- cvxpy `Problem(obj, constraints).solve(solver='ECOS')` — most ergonomic, most opinionated
- pyomo `SolverFactory('glpk').solve(model)` — interfaces-only, external binaries
- scipy.optimize `minimize(fun, x0, method='SLSQP', constraints=())` — eager bundled solvers
- OSQP raw API: `osqp.OSQP().setup(P, q, A, l, u).solve()` — lowest-level, fastest

## Notes

- This PR is the **highest architectural-risk PR in v0.1** — it introduces a second top-level layer (`solving/`) that's currently zero-state in the codebase. The Phase 2-3 research must converge on a Solver Protocol surface that's stable enough to admit future families without rework.
- The **first-Solver-family choice (#1)** is consequential. OSQP is the smallest and fastest for QP-only; cvxpy is the most general but pulls many deps. Picking OSQP and adding cvxpy later is cheaper than the reverse.
- **Bundling Protocol + first family** in one PR is a deliberate violation of the strictest reading of `One PR, One Thing` to avoid `No Phantom Implementations` — shipping the Protocol with an empty registry would be a phantom (a module declared but never called from the main flow). This PR ships both together so the conformance test has at least one real entry.
- PR-020 is **parallelizable with PR-018 / PR-019** post-017 in terms of code, but its research surface is the largest of the v0.1 PRs — start the research early.
