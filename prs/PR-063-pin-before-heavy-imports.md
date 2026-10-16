# PR-063: The env pin runs before the heavy imports (xgboost / numpy load after `pin_threads`)

<!-- STUB 2026-10-16 — the follow-up of PR-062 (gate lean 3, approved by the operator's go 2026-10-15 ≈ 20:56 CDT and the build go 2026-10-16 ≈ 08:49 CDT). Doc only: `PROCEDURE-pr-research.md` has NOT run; nothing here is a finding of this PR's own research. The evidence quoted below is PR-062's (prs/PR-062-thread-budget-reaches-xgboost.md, Research findings). -->

**Landed-in:** (not yet landed)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PRs** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

**Proposed tier: Tier-2** (PR-062's path-tier checkpoint: "re-tier to Tier-2 if B (import order, BLAS) is folded in"; wide import churn, cycle risk). The tier is the operator's call at this PR's Phase 1 gate.

## Research findings

_To be populated by `PROCEDURE-pr-research.md`. Do not begin implementation until this section exists with completed findings from all required phases._

### Carried from PR-062 (inputs to Phase 1, not this PR's findings)

- **(proven, PR-062 probe 2, the desktop 2026-10-15)** The subprocess trial child loads xgboost and numpy before it pins: after `from rux_ml.config import RuxMLConfig` (`_internal/trial_runner.py:87`) both are in `sys.modules` and libgomp is mapped; after `pin_threads` (`:97-99`) `OMP_NUM_THREADS` = "24" but `omp_get_max_threads()` = 32. Control (the env set before the interpreter starts): 24. The chain: `config/__init__.py:23` → `config/training.py:27` → `rux_ml.training.base` → `training/__init__.py:35,47` → `training/xgboost/__init__.py:15` → `training/xgboost/factory.py:17`.
- **(proven, program PR-082 R3-a)** In-process `rux-ml train`: `rux_ml.cli` imports every verb module at load (`cli/__init__.py:26`); the config is known only after Typer parses `--config/--problem/--set`, inside `run_command` (`cli/train.py` pin after the cfg build). 24 set, 32 effective; a cpu M6 fit stalled at 16.81 s.
- **(proven)** The comment at `_internal/trial_runner.py:86` ("Config import is light (pydantic + tomllib only)") is false.
- **(proven)** One libgomp is mapped in the trial child: scikit-learn's vendored copy (`scikit_learn.libs/libgomp-*.so`); xgboost's pool is that library's. A ctypes `omp_get_max_threads()` read on it works on Linux (macOS uses another OpenMP runtime — not checked).
- **(best-guess)** `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS` are no-ops by the same mechanism (numpy loads before the pin; the trial_order process held 65 threads against 64 unset before any fit, the threads not attributed per library). `POLARS_MAX_THREADS` may still work if Polars sizes its pool lazily. Program PR-027 R4 saw promote's pre-build memory fall 7.1 → 5.6 GiB with `pin_threads` (unattributed).
- **(proven, PR-062 probe 1)** A stall reproduced through `rux-ml train` at nthread 0 (a 32-thread pool): `boost` 26.866 s against 0.070 – 0.074 s in the other three runs on the `[m9]` batch shape. The stall rate at 0 vs 24 vs 8 is unranked (counts far too small). PR-062's build-window comparison arm (12 vs 12 fits) adds counts; see its Build status.
- **PR-062 already makes the XGBoost / CatBoost thread count independent of the import order** (an explicit `threads` argument from `memory.omp_threads`). What remains here is every library that still relies on the env pin.

## Scope

<!-- Draft scope from PR-062's options B / B-lite; to be fixed by this PR's Phase 2 – 4. -->

1. **The import order** — so `pin_threads` runs before xgboost / numpy load, on: the trial child (`trial_runner`'s `rux_ml.config` chain), in-process `tune` (`tuning/objective.py:40` imports xgboost at module load) and the CLI verbs (a pre-parse of `--config/--problem/--set` before `rux_ml.cli` imports the verbs, or a re-exec). Option B (every module-level xgboost import lazy) vs B-lite (only the trial child's chain) is a Phase 2 / 4 decision.
2. **The false `trial_runner.py:86` comment.**
3. **The BLAS / MKL / Polars pins** — measured per library first (do they reach their pools today?).
4. **The stall rate** — whether a 32-thread pool stalls more than a 24- or 8-thread one (input to any change of the `memory.omp_threads` default, which moves every `root_cfg_hash` and is its own config PR).

### Not in scope

- The XGBoost / CatBoost thread transport (PR-062).
- Any config change (`memory.*` defaults) — moves `root_cfg_hash`.
- The predict side (`registry score`; PR-062 gate lean 6).

## Dependencies

PR-062 (the explicit transport) merged first is assumed; not a hard dependency.

## Architecture section implemented

`docs/ARCHITECTURE.md` §memory / thread allocation (D10: "`OMP_NUM_THREADS=24`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `POLARS_MAX_THREADS=24`"); `docs/CONVENTIONS.md` §Per-family threading model.

## Verification criteria

<!-- Populated after research. Candidates carried from PR-062: -->
- [ ] An effect test, not an env test (the PR-011 lesson): in a fresh subprocess, `omp_get_max_threads()` == `memory.omp_threads` after the pin, with xgboost not in `sys.modules` before it (Linux, ctypes on the mapped libgomp).
- [ ] The BLAS / MKL pins measured by effect per library.
- [ ] No `root_cfg_hash` / `training_cfg_hash` moves.

## Research backing

- PR-062 (`prs/PR-062-thread-budget-reaches-xgboost.md`): Research findings, probe 1 and probe 2; program `docs/0.5/evidence/ruxml-track/pr062-probes-20261015/`.
- Program `prs/PR-082-probe-runner-statistic-null-holm-freeze.md`: R3-a (:1205-1210), Q16 (:1462-1477).
- Program `prs/PR-027-*.md:799` (R4).

## Notes

- Do not resurrect `pr-056/promote-pin-threads` (an env pin after import; not adopted, program PR-027:799).
