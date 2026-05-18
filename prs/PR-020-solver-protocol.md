# PR-020: Solver Protocol + first Solver family

**Landed-in:** (not yet landed)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PRs** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

**This PR is Tier-2** per `docs/0.1/RESEARCH-BACKLOG.md`. The two-Protocols-no-shared-parent lean (Q1) is research-backed; Solver-specific instantiation is not. All 5 phases of `PROCEDURE-pr-research.md` must run.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

## Research findings

_To be populated by `PROCEDURE-pr-research.md`. Do not begin implementation until this section exists with completed findings from all required phases._

**Open research questions** (must be resolved before implementation):

1. **First Solver family choice.** Candidates: OSQP, Clarabel, SCS, scipy.optimize, cvxpy-wrapped (multi-backend). Tradeoffs: install footprint (some solvers ship as small C extensions, cvxpy pulls a lot), license (some commercial — MOSEK, Gurobi, CPLEX — out of scope for an open workbench), problem-class coverage (QP-only vs QP+SOCP+SDP), Python-side ergonomics. Success criteria: a single first Solver family chosen with cited tradeoff analysis. Anchor on cvxpy's solver list and the 2026-05-18 research (cvxpy ships OSQP/ECOS/CLARABEL/SCS as required deps, rest as extras).

2. **Canonical `Solver` Protocol surface.** The lean from Q1 is `solve(problem) → result`, but what's the concrete `problem` type?
   - Option A: cvxpy-style structured object (`Problem(Minimize(obj), constraints)`)
   - Option B: raw matrices (`P, q, A, l, u` OSQP-style)
   - Option C: a workbench-specific dataclass that wraps either
   Success criteria: a concrete `Solver.solve(...)` signature with rationale. Anchor on cvxpy / pyomo / scipy.optimize conventions.

3. **A4 — `rux-ml solve` CLI verb design.** What subcommands mirror `rux-ml train` / `rux-ml tune`? Options:
   - `rux-ml solve start --problem <name>` — analog of `rux-ml tune start`
   - `rux-ml solve status / resume / retry-trial` — analogs of tuning verbs
   - `rux-ml solve once --problem <name>` — analog of `rux-ml train` (single solve, no HPO)
   Success criteria: a CLI surface that's consistent with the v0 verb naming convention (`docs/CONVENTIONS.md` CLI verbs). Document the design in `docs/CONVENTIONS.md`.

4. **Solver-side study integration with Optuna.** Does the v0 K-fold CV-mean objective (PR-007) translate to solver tuning, or is the HPO surface fundamentally different for constrained optimization? Are there per-solver hyperparameters worth tuning (e.g., OSQP's `rho`, `alpha`)? Success criteria: documented decision on whether Solver families participate in `rux-ml tune` the same way Trainer families do, or whether solver HPO is a separate verb / out of scope for v0.1.

5. **Provenance triple extension.** The v0 provenance triple (`docs/ARCHITECTURE.md` "Reproducibility Architecture") includes `data_hash`, per-layer config hashes, `git_sha`, `entropy_hex`, `image_digest`, library/CUDA versions, `omp_threads`, `peak_rss_mb`. Does it carry forward unchanged for Solver runs, or does Solver runs need different provenance fields (e.g., `problem_hash` instead of `data_hash`)? Success criteria: documented provenance schema for Solver runs.

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
