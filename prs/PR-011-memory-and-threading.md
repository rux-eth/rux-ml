# PR-011: Memory & threading

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PR** (research-backed at design time).

## Research findings

_To be populated by `PROCEDURE-pr-research.md`._

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
