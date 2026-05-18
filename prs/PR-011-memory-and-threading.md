# PR-011: Memory & threading

**Landed-in:** v0.0.1

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PR** (research-backed at design time).

## Research findings

### State Assessment (2026-05-16, LOCAL-ONLY)

**Current state of the codebase**:

- `dev` at `1cdf52f` (PR-010 merged). Working tree clean.
- `src/rux_ml/_internal/` has `__init__.py`, `git.py`, `hashing.py`, `trial_runner.py`. **No `env.py` / `memory.py` yet.**
- PR-008 already wrote `_pin_thread_env(memory)` in `trial_runner.py`. PR-011 extracts it into `_internal/env.py::pin_threads(cfg)` per spec; subprocess child + in-process CLI verb bodies both call it.
- `cli/train.py` (PR-006) and `cli/tune.py` in-process path do NOT currently pin threads — they inherit shell env. PR-011 fixes.
- `TrialAttrs.peak_rss_mb: float | None = None` (PR-009 Optional slot). PR-011 tightens to required.
- `MemoryConfig` (PR-002) has all knobs: `watchdog_threshold_gb=28.0`, `watchdog_sample_hz=1.0`, `omp_threads=24`, `openblas_threads=1`, `mkl_threads=1`, `polars_threads=24`.
- **`psutil` is NOT in `pyproject.toml` deps** — PR-011 adds it.

**Local API verification (already-installed libraries)**:

| Item | Status |
|---|---|
| `os.environ[...]=` thread-env pinning before heavy imports | **STILL CURRENT** (PR-008 exercises this) |
| `psutil.Process(pid).memory_info().rss` | **NEEDS INSTALL** |
| `threading.Thread` for background sampling | stdlib stable |
| `optuna.TrialPruned` raised from inside the `study.optimize` callback | **STILL CURRENT** (PR-007 exercises this) |

**Stale assumption in PR-011 file**:

The spec says `pin_threads` is "Called at the very top of `_internal/trial_runner.py`" — PR-008 already implemented this as `_pin_thread_env`. PR-011 just renames + relocates it to be importable by other callers.

**New constraints learned from PR-006 → PR-010**:

1. **Watchdog can't safely interrupt XGBoost mid-fit.** XGBoost training is a long-running C call; Python signal/interrupt machinery from a background thread isn't reliable inside C extensions. PR-011's watchdog is **observational + post-fit-check**:
   - Background thread samples RSS at 1 Hz; tracks `peak_mb`; sets a `tripped` flag on threshold cross.
   - Trial body checks `wd.tripped` after the fit returns; if set, raises `MemoryPressureError` → caught by the objective wrapper → re-raised as `optuna.TrialPruned`.
   - Hard OOM kills are handled by PR-008's subprocess isolation (OS kills child → non-zero exit code → parent marks trial FAIL via `_mark_trial_failed`).
2. **`TrialAttrs.peak_rss_mb` is Optional today.** Tightening to required touches ~6 test-fixture sites that build `TrialAttrs` (test_attrs, test_query, test_runs_subcommands, test_registry_subcommands, test_promote — both registry + runs sides).
3. **`build_objective` in `tuning/objective.py`** is the single sweep-side wrap point; wrap the watchdog around the per-trial loop body (after the closure's setup, before per-fold work).
4. **`cli/train.py` 1-trial path** has its own fit loop (not via `build_objective`); wrap separately.
5. **`threadpoolctl` cross-runtime limitation (D10)** — env-var pinning is the chosen mechanism; PR-011 doesn't import `threadpoolctl`.

### Synthesis (Phase 4, 2026-05-16)

**Outcome: Confirm** — D10 design research stands. Tier-1 classification stands (all sub-decisions are engineering judgment).

**User-approved sub-decisions (2026-05-16):**

- **A1** — Extract PR-008's `_pin_thread_env` to public `_internal/env.py::pin_threads(memory)`. `trial_runner.py` imports + calls.
- **B1** — Parent CLI verbs (`train`, `tune start/resume/retry-trial`) call `pin_threads(cfg.memory)` at body start. Subprocess children already self-pin.
- **C1** — Observational watchdog: background thread samples RSS, sets `tripped` flag on threshold cross; trial body's post-fit check raises `MemoryPressureError`. Caught by objective wrapper → `optuna.TrialPruned`. Hard OOMs are PR-008's responsibility.
- **D1** — `TrialAttrs.peak_rss_mb` tightened to required (`float`, no default). `TrialAttrs.from_cfg` accepts `peak_rss_mb` kw. ~6 test fixtures updated to pass a placeholder/real value.
- **E1** — Background thread inside the trial process (one thread per trial; negligible overhead).

**Changes from research**: none — D10's cited Optuna OOM #1178 + sklearn parallelism + numpy RLIMIT_AS #26551 all hold; psutil RSS still canonical.

**Changes to `docs/ARCHITECTURE.md`**: Memory & Parallelism table — confirm the row already references `psutil` watchdog + `cfg.memory.watchdog_threshold_gb`; mention `peak_rss_mb` is now required on `TrialAttrs`.

**Changes to `docs/CONSTRAINTS.md`**: None (the "Per-Trial Provenance Triple" already lists `peak_rss_mb`).

**No new prerequisite PRs surfaced.** PR-013 (seed management) builds on this PR's `peak_rss_mb` recording but doesn't reshape the watchdog.

### Gate Check (Phase 5, 2026-05-16)

- Premise still valid: ✓ (D10 design research stands)
- No prerequisite PRs surfaced: ✓
- User approved Tier-1 classification: ✓ (2026-05-16)
- User approved locked-in spec (A1, B1, C1, D1, E1): ✓ (2026-05-16)
- Implementation cleared: ✓ (2026-05-16)

---

## Scope

Implement memory protection and thread coordination per D10. After this PR, every trial runs under a memory watchdog with thread allocation pinned to avoid oversubscription.

- `src/rux_ml/_internal/env.py`:
  - `pin_threads(cfg: MemoryConfig) -> None` setting `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `POLARS_MAX_THREADS` per D10 defaults (24/1/1/24)
  - Called at the very top of `_internal/trial_runner.py` and at CLI startup for in-process mode
- `src/rux_ml/_internal/memory.py`:
  - `class MemoryPressureError(Exception)`
  - `class Watchdog`:
    - `start(threshold_gb)` spawns a 1 Hz `psutil.Process(os.getpid()).memory_info().rss` poll thread
    - On threshold trip → raise `MemoryPressureError` (signaled to main thread)
    - Tracks peak RSS for `peak_rss_mb` user_attr
- `src/rux_ml/tuning/objective.py` updates:
  - Wrap the trial body so `MemoryPressureError` is converted to `optuna.TrialPruned` with diagnostic logging
  - Record `peak_rss_mb` in `user_attrs` after every trial
- `src/rux_ml/_internal/trial_runner.py` updates:
  - Start watchdog at child entry; record peak before `study.tell`
- `configs/base.toml` `[memory]` section: `watchdog_threshold_gb = 28` (BEST-GUESS per D10), `watchdog_sample_hz = 1`, `omp_threads = 24`, `openblas_threads = 1`, `mkl_threads = 1`, `polars_threads = 24`
- `runs.attrs.TrialAttrs` updated to require `peak_rss_mb` (was Optional in PR-009)
- Tests:
  - Unit: `pin_threads` sets the correct env vars
  - Unit: `Watchdog` raises `MemoryPressureError` when synthetically tripped
  - Integration: a contrived objective that allocates a huge array → watchdog trips → trial marked `PRUNED` with `peak_rss_mb` recorded
  - Verify `threadpoolctl` is NOT relied on across libgomp/libiomp boundary (per D10 conflict)

NOT in scope: container memory caps (PR-012); cgroup-level enforcement is the container's job.

## Dependencies

PR-008.

## Architecture section implemented

`docs/ARCHITECTURE.md` → "Memory & Parallelism Architecture", "Decision Rules: Memory-pressure response".

## Verification criteria

- [ ] `pin_threads(cfg)` results in the documented env vars being set in the running process
- [ ] Watchdog samples at 1 Hz; threshold trip raises within 2 seconds in a test
- [ ] `MemoryPressureError` is converted to `optuna.TrialPruned` with logged diagnostic (peak RSS, threshold, time-of-trip)
- [ ] Every completed trial has `peak_rss_mb` recorded
- [ ] Updated `TrialAttrs` validation requires `peak_rss_mb`
- [ ] `threadpoolctl` is used only around sklearn preprocessing (NOT around XGBoost training) per D10 conflict resolution

## Research backing

Tier 1:

- D10: [Netdata cgroups v2](https://www.netdata.cloud/academy/diagnosing-linux-cgroups/), [sklearn parallelism](https://scikit-learn.org/stable/computing/parallelism.html), [Optuna OOM #1178](https://github.com/optuna/optuna/issues/1178), [numpy RLIMIT_AS #26551](https://github.com/numpy/numpy/issues/26551)

State assessment must verify:
- `psutil.Process.memory_info().rss` is still the canonical RSS source
- XGBoost still respects `nthread` env / arg; Polars still respects `POLARS_MAX_THREADS`
- The `threadpoolctl` cross-runtime limitation has not been resolved (if it has, we can simplify)

## Notes

- Per `docs/CONSTRAINTS.md`, `RLIMIT_AS` is forbidden — do NOT add `resource.setrlimit` even as a "belt-and-braces" measure. It causes more failures than it prevents.
- The watchdog runs in a thread, not a process — the cost is one thread per trial, negligible.
- BEST-GUESS threshold of 28 GB is documented in TOML comments alongside the field; revisit if memory pressure is hit too often (lower) or too rarely (raise).
