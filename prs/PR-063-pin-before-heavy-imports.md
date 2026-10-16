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

**Gate (Phase 1), the rumpy operator's "go on all, 1-11" (≈ 14:05 CDT 2026-10-16, item 11 of the rumpy lead's 14:01 list):** **Tier-1 light path** (mirrors PR-062); **Q2** the parent passes the four pinned values in the trial child's env at `isolation.py:123` (the child keeps its own pin, warns — not refuses — on a mismatch; the `:86` comment and `:18-21` docstring rewritten); **Q3** no CLI restructure in PR-063 (a re-exec revisited only if Q1 shows a material BLAS pool); **Q4** `nthread = memory.omp_threads` folded into the native adapter's matrices, the model bytes re-checked on the desktop; **Q5** the stall rate moved out to its own backlog item. **Q1 — the Linux thread-pool measurement (≈ 15 – 20 min under the rumpy heavy lock) runs after the rumpy v0.5 probe's record** (no desktop rux-ml run before the probe's tag and windows), then the build. The merge into `dev` is its own word.

### State Assessment (2026-10-16)

<!-- Phase 1, 2026-10-16 ≈ 13:50 CDT. One agent on the Mac only (worktree `rux-ml-pr-063`, branch `pr-063-research` from `dev` 82b1511); no desktop, no ssh. Two Tier-A fetches of primary source (libgomp, OpenBLAS). The Mac probe is a scratch script (not committed); its arms and outputs are quoted below. Mac: macOS 27.0.1 arm64, 14 CPUs, Python 3.12.15, the worktree's own `.venv` built by `uv sync --frozen --offline` (gitignored). Lock: xgboost 3.2.0, numpy 2.4.5, polars 1.40.1, scikit-learn 1.8.0, scipy 1.17.1, pyarrow 24.0.0, lightgbm 4.6.0, catboost 1.2.10, threadpoolctl 3.6.0. -->

**Current state**:

- **(proven, code) Why the config import is heavy: a package `__init__`, not `config/` itself.** `trial_runner` imports `rux_ml.config` at `_internal/trial_runner.py:87`, before the pin at `:96-99`. `config/__init__.py:23` → `config/training.py:27-30` imports four light config classes (`rux_ml.training.base`, `.catboost.config`, `.lightgbm.config`, `.xgboost.config`). Importing any of them first runs the eager package inits: `training/__init__.py:30-47` (catboost `:31`, `make_trainer` `:35`, lightgbm `:36`, metrics `:40` → `sklearn.metrics` at `training/metrics.py:24`, xgboost `:47`), then `training/xgboost/__init__.py:15-21` (factory → `factory.py:17` xgboost; ingest → `ingest.py:30`; native adapter → `native_adapter.py:40-42` numpy / polars / xgboost) and `training/catboost/__init__.py:31-32` (factory → pandas `catboost/factory.py:43`).
- **(proven, Mac measurement) What is loaded at each point.** In fresh interpreters:
  - after `import rux_ml._internal.trial_runner` alone: none of the heavy modules (agrees with PR-062 probe 2 on Linux);
  - after `from rux_ml.config import RuxMLConfig`: xgboost, numpy, polars, sklearn, scipy, pyarrow, pandas;
  - `import rux_ml.config.memory` alone loads the same seven. Any `rux_ml.config.*` import pays for the package inits.
  - after `import rux_ml.cli` (the in-process CLI): the same seven plus optuna and typer.
  The import order is plain Python, so it is the same on Linux.
- **(proven, git) When the `trial_runner.py:86` comment became false.** At PR-008 (`5768b26`, 2026-05-16), `config/training.py` imported only `typing`, `pydantic` and `StrictModel`, so the comment was true. It has been false since PR-017 (`a98349f`, 2026-05-18): the discriminated union made `config/training.py` import `rux_ml.training.*` (`git log -S "from rux_ml.training"`). The docstring's promise at `trial_runner.py:18-21` ("top-level imports stay stdlib-only so the child can pin … before importing numpy / polars / sklearn / xgboost") has been broken since then too.
- **(proven, code) The subprocess path: the parent never pins, and it spawns the child with no `env=`.** `cli/tune.py:96-102` calls `pin_threads` only in the `in_process` branch. The subprocess branch (`:104-118`) calls `run_subprocess_trial`, and that calls `subprocess.run(argv, check=False, timeout=…)` at `tuning/isolation.py:123-127` (argv from `:46-69`). So the child inherits the parent's shell environment, usually unset. This is exactly probe 2's failing condition. Probe 2's *control* (`OMP_NUM_THREADS=24` alone, set before the interpreter starts, PR-062 file :187 → `omp_get_max_threads()` 24) is, for libgomp, the condition the parent could produce itself at spawn. *(checker: the control set OMP alone, not the four values; the OpenBLAS / Polars part of a four-value spawn env is unmeasured, Q1.)*
- **(proven, code) In-process pin sites.** `cli/train.py:717` (after the cfg build at `:709-714`; numpy / optuna / polars are already imported at `:22-24`, and `rux_ml.cli` has imported every verb at `cli/__init__.py:26`); `cli/tune.py:99` (`in_process` only); `cli/solve.py:170`. `cli/registry.py`, `cli/data.py` and `cli/runs.py` never call `pin_threads` (0 hits). Promote's XGBoost refit has the explicit transport (`registry/promote.py:246`).
- **(proven, primary source) libgomp reads `OMP_NUM_THREADS` when the library loads.** `libgomp/env.c` (gcc master): `static void __attribute__((constructor)) initialize_env (void)` parses `NUM_THREADS` (`parse_unsigned_long_list`) into `gomp_global_icv.nthreads_var`. On Linux, loading the library is the moment the env is read, so no pin after the load can reach that value. This is the mechanism behind PR-062's 24-set / 32-effective result.
- **(proven, primary source) OpenBLAS reads its env and starts its threads when it loads.** `driver/others/memory.c` (OpenMathLib develop): `void CONSTRUCTOR gotoblas_init(void)` (`CONSTRUCTOR` → `__attribute__((constructor(101)))`) calls `openblas_read_env()`, `blas_get_cpu_number()` and `blas_thread_init()`. numpy loads before the pin on every path, so on Linux `OPENBLAS_NUM_THREADS=1` very likely reaches nothing today. *Mechanism from upstream source. Not checked: the build flags of the OpenBLAS copies vendored in the numpy / scipy wheels, and the per-library thread counts on the desktop (Q1).*
- **(proven, primary source; checker 2026-10-16) OpenBLAS falls back to `OMP_NUM_THREADS`.** `driver/others/memory.c` (develop) `blas_get_cpu_number()`: `if (blas_goto_num > 0) blas_num_threads = blas_goto_num; else if (blas_omp_num > 0) blas_num_threads = blas_omp_num; else blas_num_threads = MAX_CPU_NUMBER;`, then capped at `get_num_procs()`. `blas_goto_num` is `OPENBLAS_NUM_THREADS`, then `GOTO_NUM_THREADS`; `blas_omp_num` is `OMP_NUM_THREADS`. So an explicit `OPENBLAS_NUM_THREADS` outranks the OMP value, and with neither set each copy sizes to the CPU count.
- **(best-guess, arithmetic; checker 2026-10-16) Most of the 65 pre-fit threads are very likely two OpenBLAS pools.** PR-062 probe 2 (file :202-203): before any fit, 64 threads unset vs **48** with `OMP_NUM_THREADS=24` alone. 64 − 48 = 16 = 2 × (32 − 24): two OpenBLAS copies (numpy's and scipy's wheels each vendor one; not listed on the desktop) sized by the OMP fallback above, each spawning `n − 1` workers (2 × 31 + the main thread ≈ 63 of 64). libgomp starts no threads before its first parallel region, so it cannot explain the drop. If this holds, every in-process verb carries ≈ 62 OpenBLAS workers today, and the trial child would drop to ≈ 2 with `OPENBLAS_NUM_THREADS=1` at spawn. Q1 confirms or refutes it per library.
- **(proven, lock) MKL is not in the environment.** `uv.lock` has 0 matches for `mkl`. `MKL_NUM_THREADS` pins a library that is not installed. Harmless, but `ARCHITECTURE.md:621` and `README.md:626` describe it as if it acts.
- **(proven, Mac measurement) Polars sizes its pool on first use, not on import.**
  - `import polars`, then set `POLARS_MAX_THREADS=3`: `pl.thread_pool_size()` = **3**.
  - Run one query first, then set it: **14** (the CPU count; a later env value is ignored).
  - In the trial child's own order (config import → the real `pin_threads` with `MemoryConfig(omp_threads=3, polars_threads=3, …)` → the post-pin imports `:104-111`): **3**. No Polars query runs during the imports.
  So `POLARS_MAX_THREADS` reaches the pool today on the trial-child path. *Linux: inferred, not measured. The same Python path and Polars' own pool code, so high confidence; Q1 reads it once.*
- **(proven, Mac measurement; NOT valid for Linux) On macOS a pin after import does reach OpenMP.** The trial-child order with the pin after the config import: threadpoolctl shows **two `libomp.dylib` copies, both at 3**. In a control with the env set at start, also 3. In a third arm, where threadpoolctl's `omp_get_max_threads` call initialises the runtime *before* the pin, both copies stay at **14** after it. LLVM libomp reads the env when its runtime first initialises (on first use); libgomp reads it at load (above). **On the Mac today's order works, so a Mac test cannot see the defect.**
- **(proven, PR-062 Build status) Where Linux stands after PR-062.** In the trial child's order on the desktop, `omp_get_max_threads()` = 32 on both the branch and `dev`. The process holds 65 threads before any fit (not measured per library; the arithmetic above attributes ≈ 62 to two OpenBLAS pools, best-guess). A default fit adds 26 threads on the branch (booster `nthread` 24) against 34 on `dev`. The `[m9]`-shape arm: 0 / 12 stalls at 24 (median 0.0718 s) vs 12 / 12 slow at 0 (median 27.24 s).
- **(proven, code) The explicit transport after PR-062.** XGBoost's estimator, booster and `fit_batches` matrices carry the budget: factory `n_jobs`, `native_adapter.py:131` booster `nthread`, `batches.py:232`, `:267`, `:286`. CatBoost's `thread_count` does too. LightGBM accepts the argument and ignores it (`lightgbm/factory.py:157-162`). It imports lazily, after the pin on the trial-child path, so whether its OpenMP honours the env depends on whether it maps a fresh libgomp or reuses the one already loaded at 32. That is unknown (Q1).

**Assumptions at PR draft time** (the stub, 2026-10-16):

- The fix is to make the pin run *before the heavy imports*: by import order (B: every module-level xgboost import lazy; B-lite: only the trial child's chain), or by an argv pre-parse / re-exec for the CLI.
- Whether the BLAS / MKL / Polars pins reach their pools is unknown and must be measured per library.
- PR-062 already makes the XGBoost / CatBoost thread count independent of the import order.
- The effect test reads `omp_get_max_threads()` through ctypes on the mapped libgomp (Linux).
- The stall rate at 0 / 24 / 8 is in scope (scope item 4).

**Stale assumptions** (where current state disagrees with PR assumptions):

1. **"Before the import" is the wrong event. What matters is when each library sizes its pool, and that differs by library and platform.** libgomp and OpenBLAS do it at load (constructors; primary source). Polars does it at first use (Mac-proven). LLVM libomp on macOS does it at first OpenMP use (Mac-proven). So the import order matters only for libgomp and OpenBLAS on Linux. **Polars is not broken on the trial-child path**: its pin already reaches the pool there.
2. **The trial child has a fix the stub does not list: the parent sets the env when it spawns.** The parent holds `cfg` (`isolation.py` `run_subprocess_trial(cfg=…)`) and spawns with no `env=` (`:123-127`). Passing the four pinned values in the child's environment gives libgomp probe 2's *control* condition (`OMP_NUM_THREADS=24` alone; measured 24), and gives each OpenBLAS copy its own value (`OPENBLAS_NUM_THREADS` outranks the OMP fallback; unmeasured, Q1). No import changes and no cycle risk. The stub only lists import reordering, a pre-parse or a re-exec.
3. **"PR-062 already makes XGBoost independent of the import order" is only partly true.** The native adapter builds its training and eval matrices with no `nthread`: `QuantileDMatrix` at `native_adapter.py:151-155`, `ExtMemQuantileDMatrix(data_iter, **kwargs)` at `:167-175` (kwargs = `enable_categorical` / `cache_host_ratio` only), eval `DMatrix` at `:207`. Building those matrices (the quantile sketch) still uses XGBoost's default thread count. That default is believed to be the OpenMP default, 32 on the desktop today (*best-guess from XGBoost's `nthread=None → -1` API; not measured*). This path is used by `rux-ml train` (`cli/train.py:448`). `batches.py` and the sklearn wrapper do pass `nthread` to their matrices. *(checker: the predict-side `DMatrix` at `native_adapter.py:236`, `:244` and `registry/scorer.py:97` carry no `nthread` either; the predict side is out of scope, stub §Not in scope. No other fit-side matrix construction exists in `src/`: `batches.py:238`, `:241`, `:285` carry it.)*
4. **Scope item 1's "in-process `tune` (`tuning/objective.py:40`)" is one site out of many.** On every in-process verb, `rux_ml.cli` loads the whole heavy stack before Typer parses (`cli/__init__.py:26`; Mac: seven heavy modules plus optuna and typer). Making `objective.py:40` lazy changes nothing there. Only (a) lazy verb modules plus lazy package inits plus a pre-parse, or (b) a re-exec, moves the in-process pin ahead of libgomp / OpenBLAS.
5. **The `trial_runner.py:86` comment is not just false: the docstring's design (`:18-21`) has been broken since PR-017.** Restoring it needs the four package inits to stop eager re-exports (PEP 562 `__getattr__`), or the training config classes to move out of `rux_ml.training`. That is wider than "only the trial child's chain". B-lite is therefore not small, and the spawn-env route (2) is.
6. **A Mac effect test cannot catch this defect.** On macOS today's order already gives the pinned value (two libomp copies at 3), so a Mac-green effect test would pass on `dev` too. The effect test must be Linux-only, and a Mac run of it proves nothing. Instrument: `threadpoolctl.threadpool_info()` (3.6.0 is in the lock as a scikit-learn dependency) reads libgomp, every OpenBLAS copy and LLVM libomp, per library, in one call, which is simpler than ctypes over `/proc/self/maps`. Using it to *measure* does not touch D10, which rejects threadpoolctl only as the *pinning mechanism* (`_internal/env.py:39-41`; `README.md:631`).
7. **`MKL_NUM_THREADS` pins nothing.** MKL is not installed (lock).
8. **Scope item 4 (the stall rate) no longer serves this PR.** Since PR-062, XGBoost fits run at an explicit 24 whatever the OpenMP default. Ranking 32 / 24 / 8 matters only to a change of the `memory.omp_threads` default, which the stub itself puts out of scope (it moves `root_cfg_hash`).

**New constraints** (learned from prior PRs or codebase evolution):

- **Linux is the only valid platform for an effect test** (stale 1, 6). The Mac can prove import order and Polars behaviour, nothing about libgomp or OpenBLAS.
- **No config change.** The values come from `cfg.memory`; `memory.*` defaults move `root_cfg_hash` (stub §Not in scope).
- **PR-011's tests check the environment, not the effect** (`tests/_internal/test_trial_runner.py:52-70`; `tests/_internal/test_env.py`, PR-062 finding). New tests read the pools.
- **Many in-process tests run the CLI through Typer's `CliRunner`.** A re-exec design cannot run under it and would need a guard. Re-exec cost on the Mac: `import rux_ml.cli` ≈ 2.00 s wall, `import rux_ml.config` ≈ 1.42 s (one run each, warm disk). Phase 3 would re-measure it on the desktop.
- **Do not resurrect `pr-056/promote-pin-threads`** (program PR-027:799). `cli/registry.py` has no pin, and promote's XGBoost refit already carries the explicit budget (`registry/promote.py:246`).
- **Watch for import cycles** if the package inits go lazy: `training/factory.py:55` already breaks one (`from rux_ml.training import TRAINER_FAMILIES  # noqa: PLC0415 — break import cycle`, on `dev` 82b1511; the PR-062 file's option B row).
- *(best-guess, Intel spec not fetched)* The desktop's i9-13900K has 24 cores and 32 logical CPUs (8 P-cores with HT + 16 E-cores). `omp_get_max_threads()` = 32 is the logical count, and the default `memory.omp_threads` = 24 is the core count. rux-ml `CLAUDE.md` and `README.md:50` say "24 threads". Reported only; no change proposed here.

**Proposed tier: Tier-1, light path** *(the operator rules at the gate)*. This applies only if Q2 – Q5 land on their leans: no import restructuring, just the spawn env, the native adapter's `nthread`, the comment and the docs. Phase 3 (light) would be the one desktop window below. This mirrors PR-062's approved precedent. **Re-tier to Tier-2** if the operator picks B / B-lite (lazy package inits) or a CLI re-exec: those are wide import churn with cycle risk. *Against Tier-1:* rux-ml `CLAUDE.md:52` says v0.1 PRs are Tier-2 (per `docs/0.1/RESEARCH-BACKLOG.md`). *Checker:* the OpenBLAS attribution above makes Q3's re-exec revisit likely to fire, which is a Tier-2 trigger.

**Must-answer at the gate** (few; each with a lean):

1. **What each thread pool holds on Linux today vs with the env set at start.** *(desktop)*
   - *Lean (prediction):* libgomp 32 today / 24 at start (proven, PR-062); each OpenBLAS copy at the CPU count today / 1 at start (mechanism proven; PR-062's pre-fit counts already fit two copies at 32, 64 vs 48 with `OMP_NUM_THREADS=24` alone = 2 × 8, best-guess attribution); Polars 24 in both (Mac-proven, Linux inferred); LightGBM reuses the libgomp already loaded at 32 (best-guess, low).
   - *Basis:* the two constructors above, and the Mac Polars arms.
   - *Confidence:* medium overall.
   - *How:* three fresh-interpreter arms (unset; today's order with the pin after the config import; the env at start). Each reads `threadpool_info()`, `pl.thread_pool_size()` and `/proc/self/task/*/comm` before any fit, plus whether `import lightgbm` maps a second libgomp. This also confirms or refutes the 65-thread attribution above.
2. **The trial child: how does the pin get ahead of libgomp / OpenBLAS?**
   - *Lean:* the parent passes the four pinned values in the child's environment at `isolation.py:123` (`env={**os.environ, …}` from `cfg.memory`). The child keeps its own idempotent `pin_threads` and logs a warning (not a refusal: in-process tests call `main`) when its starting env disagrees with `cfg.memory`. The `:86` comment and the `:18-21` docstring are rewritten to the true state.
   - *Basis:* probe 2's control (`OMP_NUM_THREADS=24` alone before start → 24) is this condition for libgomp, proven; for OpenBLAS the same load-time constructor applies (primary source), the effect unmeasured (Q1); no imports move.
   - *Alternatives:* B-lite (lazy package inits, so the child's own order holds again: wider, cycle risk); `threadpoolctl.threadpool_limits` after the import (reaches pools that are already built, but D10 rejects it, and `omp_set_num_threads` sets only the calling thread's value: best-guess, unfetched).
   - *Confidence:* medium-high.
3. **The in-process verbs (`train`, in-process `tune`, `solve`).**
   - *Lean:* no CLI restructure in PR-063. Make the remaining OpenMP use on the in-process fit path explicit (Q4). State in the docs that libgomp / OpenBLAS on in-process verbs run at the library default. Revisit a re-exec only if Q1 shows a material in-process BLAS pool.
   - *Basis:* PR-062 already bounds the in-process XGBoost / CatBoost fits; the CLI fix is the wide part; the smaller-PRs ruling.
   - *Alternatives:* (a) re-exec once after the cfg build when the env disagrees (≈ 2 s re-import on the Mac; not under `CliRunner`); (b) lazy verb modules + lazy package inits + an argv pre-parse (wide; the Tier-2 trigger).
   - *Confidence:* medium, depends on Q1.
   - *Checker's note:* the arithmetic above predicts ≈ 62 OpenBLAS workers on every in-process verb today, so Q1 will very likely meet this revisit condition. Whether 'material' means the thread count alone or a measured cost (BLAS work overlapping a fit) is for the operator to define before Q1 runs.
4. **The native adapter's matrices** (`native_adapter.py:151-155`, `:167-175`, `:207`).
   - *Lean:* fold the explicit `nthread = memory_cfg.omp_threads` into PR-063. It is the same root cause, a few lines, and moves no config hash. Re-check the bytes on the desktop in the same window: PR-062 found the model bytes equal at `nthread` 0 vs 24 where the matrices already carry it, but the native adapter's `QuantileDMatrix` path was not among its probe shapes.
   - *Alternative:* its own PR (smaller-PRs ruling).
   - *Confidence:* medium.
5. **Scope item 4 (the stall rate).**
   - *Lean:* move it out of PR-063 to its own research item (RESEARCH-BACKLOG row), the input to any `memory.omp_threads` default change.
   - *Basis:* stale 8.
   - *Confidence:* medium-high.

**What needs the Linux desktop, and how long** (one heavy-lock window, outside the v0.5 probe's windows):

- Q1 attribution: three arms of ≈ 5 s each.
- Q4 byte re-check: cpu, native adapter, `QuantileDMatrix` path, default vs explicit `nthread`, a probe-1 shape. A few minutes.
- The post-build effect test: `omp_get_max_threads` / `threadpool_info` in a child spawned by the parent equals `memory.omp_threads`, with OpenBLAS at `memory.openblas_threads`. About 1 min.
- Total ≈ **15 – 20 min wall** including the mkdir lock, `git pull --ff-only` of the branch and `uv sync --frozen`.
- If Q5 stays in scope: + ≈ 15 min (12 fits at each of 32 / 24 / 8 on the `[m9]` shape; a stalled fit ≈ 27 s).

**Phase 1 spend:** one agent, ≈ 22 tool calls. ≈ 0.15 – 0.2M tokens by the cost model (68k + 3k × 22 + the pack). $0. Two Tier-A web fetches (gcc `libgomp/env.c`; OpenBLAS `driver/others/memory.c`). No desktop, no ssh. Exit: the premise holds and is narrowed (libgomp + OpenBLAS on Linux; Polars fine; MKL absent), so there is no loop to design-planning.

**Checker pass** (2026-10-16, started 13:52 CDT; Mac only, ≤ 12 tool calls, one Tier-A fetch of OpenBLAS `driver/others/memory.c`). Every code cite re-read on `82b1511`: all hold except three corrected in place (`training/factory.py:47` → `:55`; the cfg build `:708-713` → `:709-714`; the `CLAUDE.md:52` wording). Re-run in fresh interpreters on the worktree `.venv`: `trial_runner` alone loads none of the heavy modules; `rux_ml.config.memory` loads the seven; `rux_ml.cli` adds optuna + typer; Polars 3 (set after import) / 14 (query first) / 3 (trial-child order); two `libomp.dylib` at 3 in the trial-child order and in the OMP-at-start control (the `omp_touch` arm was not re-run). `git`: `config/` and `rux_ml/__init__.py` import nothing outside pydantic / typing / `rux_ml.config` at `5768b26` and at `a98349f^`, so the `:86` comment held until PR-017. The spawn path is the only Python child spawn (`isolation.py:123`; the other `subprocess` calls are tools). Added: the OpenBLAS OMP fallback (primary source) and the 65-thread attribution (best-guess). Corrected: probe 2's control set `OMP_NUM_THREADS` alone, so the spawn-env route matches it for libgomp only. The libgomp `env.c` constructor was not re-fetched.

### Q1 measured (Linux, 2026-10-16)

<!-- The Q1 desktop window on the gate's go (item 11, ≈ 14:05 CDT). Harness `scripts/measure_pr063_threads.py` (commit `7b537b2`), run from a fresh clone `~/pr063-build` at `7b537b2` with the locked env (`uv sync --locked --all-extras`), under `~/rumpy-heavy.lock` (mkdir) 15:12:39 – 15:13:10 CDT (31 s). Receipt: `prs/PR-063-q1-threads-20261016.json` (3 runs × 3 arms, interleaved, all rc 0). Desktop: Ubuntu 22.04, kernel 6.8.0-138, 32 logical CPUs (affinity 32), Python 3.12.13; numpy 2.4.5, scipy 1.17.1, scikit-learn 1.8.0, xgboost 3.2.0, polars 1.40.1, pyarrow 24.0.0, lightgbm 4.6.0, threadpoolctl 3.6.0. Each arm is a fresh interpreter with `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS` / `POLARS_MAX_THREADS` / `GOTO_NUM_THREADS` stripped from its env; the pinned values (24 / 1 / 1 / 24) come from `configs/base.toml` `[memory]` through `pin_threads` itself. -->

**Arms.**
- `cli_today`: `import rux_ml.cli`, then `pin_threads` (the in-process verbs' order).
- `trial_child`: the order at `trial_runner.py:86-111` (config import → `pin_threads` → the post-pin imports), env unset at start. This is today's subprocess child.
- `env_at_start`: the same order with the four values in the env at start. This is the condition Q2's spawn env produces.

Each arm reads before any fit: `/proc/self/task` first, then `threadpool_info()`, then a 2 s idle window, then `pl.thread_pool_size()` (which builds Polars' pool), then `import lightgbm` and `threadpool_info()` again. An `import` audit hook records which import was in flight at each rise in the thread count.

**Result** (the counts are identical in all 3 runs of each arm; ranges are given where a figure varied):

| | `cli_today` | `trial_child` | `env_at_start` |
|---|---|---|---|
| **Threads before any fit** | **65** | **65** | **3** |
| main thread | 1 | 1 | 1 |
| numpy's OpenBLAS (`numpy.libs/libscipy_openblas64_`, 0.3.31.188.0, pthreads): pool / workers; rise at `numpy._core._multiarray_umath` | 32 / 31 | 32 / 31 | 1 / 0 |
| scipy's OpenBLAS (`scipy.libs/libscipy_openblas`, 0.3.30, pthreads): pool / workers; rise at `scipy.special._ufuncs` | 32 / 31 | 32 / 31 | 1 / 0 |
| libgomp (`scikit_learn.libs/libgomp-e985bcbb`, the only copy mapped): `omp_get_max_threads` / threads | 32 / 0 | 32 / 0 | 24 / 0 |
| `jemalloc_bg_thd` (one at `pyarrow.lib`, one at the Polars runtime import) | 2 | 2 | 2 |
| MKL mapped | no | no | no |
| Polars `thread_pool_size()` | 24 | 24 | 24 |
| Threads after Polars' pool is built (+24 `polars-N`, +3 `jemalloc_bg_thd`) | 92 | 92 | 30 |
| `import lightgbm`: a second libgomp, `/usr/lib/x86_64-linux-gnu/libgomp.so.1`, at | 24 | 24 | 24 |
| Threads after `import lightgbm` (no new threads) | 92 | 92 | 30 |
| CPU used by non-main threads by the pre-fit read | 4.88 – 4.96 s | 4.92 – 4.96 s | 0 |
| CPU used by non-main threads in the 2 s idle window | 0 | 0 | 0 |
| VmRSS before any fit | 279.7 – 295.1 MiB | 248.9 – 249.4 MiB | 248.5 – 248.8 MiB |
| VmSize before any fit | 5.26 – 5.28 GiB | 5.23 GiB | 2.81 GiB |
| Import wall to the pre-fit read | 1.28 – 1.94 s | 0.63 – 0.64 s | 0.56 – 0.57 s |

*Reading note:* the receipt's `import_rises` lists a 65 → 92 (or 3 → 30) rise labelled `rux_ml.tuning.study`. That rise is Polars' pool, built by the harness's own `pl.thread_pool_size()` call. The hook first saw it at `import lightgbm`'s first import event and labelled it with the last import started before that. The pre-fit count (65 / 3) was read before the call.

**Against Phase 1's predictions:**
- **Confirmed (proven):** libgomp at 32 today and 24 with the env at start.
- **Confirmed (proven):** each OpenBLAS copy sizes to the CPU count today and to 1 with the env at start. There are two copies, numpy's and scipy's, both pthreads builds, so each spawns `n − 1` workers when it loads.
- **Confirmed (proven) — the 65-thread attribution:** 1 main + 31 + 31 OpenBLAS workers + 2 `jemalloc_bg_thd` = 65. The best-guess label is retired.
- **Confirmed on Linux:** Polars is at 24 on both paths. Its pool is built on first use, which comes after the pin.
- **Refuted:** "LightGBM reuses the libgomp already loaded at 32." `import lightgbm` maps the system `libgomp.so.1` as a second copy. That copy reads the env when it loads, after the pin, so it is at 24 on every arm, and LightGBM's OpenMP honours the pin today on both paths. *This holds for the desktop's system libgomp; a container image would need its own check (not checked).*
- **Confirms the lock:** MKL is not mapped.
- **Not run:** an `OMP_NUM_THREADS`-alone arm. So PR-062's 48 is not reproduced here. The mechanism predicts 1 + 23 + 23 + 2 = 49. PR-062's unset figure was 64 against 65 here; the one-thread difference is not chased.

**Q2 (the spawn env) — the OpenBLAS part is now measured.** The `env_at_start` arm reaches every pool that sizes at load:
- libgomp 24;
- both OpenBLAS copies 1 (0 workers);
- Polars 24;
- LightGBM's libgomp 24.

The child holds 3 threads before any fit instead of 65. The child's own pin after the import reaches neither OpenBLAS copy nor libgomp: the `trial_child` arm equals `cli_today` on every pool.

**Q3's revisit condition (is the in-process BLAS pool material?)** — answered by measure; nothing built:
- **By thread count: yes.** Every in-process verb (`cli_today`) holds 62 OpenBLAS workers (2 × 31) before any fit, the same as today's trial child. Its `OPENBLAS_NUM_THREADS=1` reaches neither copy.
- **By measured cost at idle: none shown.**
  - The workers used 0 CPU in the 2 s idle window.
  - RSS equals the pinned child's (≈ 249 MiB, in the trial-child order); the extra is ≈ 2.4 GiB of virtual address space only.
  - At start-up the 62 workers used ≈ 4.9 s of CPU (≈ 80 ms each, their spin before sleeping). That ran on other cores: the import wall is 0.64 s vs 0.57 s.
- **During a fit: not measured.** The open cost is a numpy / scipy BLAS call during a fit waking 31 workers that spin next to XGBoost's 24-thread team on 32 logical CPUs. Which in-process fit paths make such calls, and what that costs, is not known from Q1.
- **So the condition fires on the count and not on any measured cost.** The checker's open point stands: whether "material" means the count (→ a re-exec, a Tier-2 PR of its own) or a measured overlap cost (→ measure first) is the operator's to rule. *Lean:* do not open a re-exec on the count alone. Instead, add a backlog item that measures a BLAS overlap on the in-process `train` path, beside Q5's stall-rate item. *Basis:* PR-062 already bounds the in-process XGBoost / CatBoost fits, and the pools sit idle at 0 CPU. *Confidence:* medium.

**Spend:** one agent, ≤ 26 tool calls; one desktop window of 31 s under the lock; $0.

### Build status (2026-10-16)

<!-- The build on the gate's go (item 11). The builder agent returned nothing (no build commit on the branch after the Q1 commits); the reviewer built it, tests first, at `303c3a5`. Desktop window under `~/rumpy-heavy.lock` (mkdir) 16:10:38 – 16:15:07 CDT, clone `~/pr063-build` detached at `303c3a5` (then `7b537b2` for the pre-fix arm), `uv sync --locked --all-extras`, the four thread vars and `RUMPY_HARNESS_DATA_ROOT` unset; outputs `~/pr063-build-job.log`, `~/pr063-bytes/{branch,prefix}.json`. -->

**Built (`303c3a5`).**
- **Q2, the spawn env.** `tuning/isolation.py` spawns the trial child with `env={**os.environ, **thread_env(cfg.memory)}`. `_internal/env.py` gains `THREAD_ENV_KEYS` and `thread_env(memory)`, the one source for `pin_threads` and the spawn env. The parent's own env is not touched.
- **Q2, the child.** `trial_runner.main` snapshots the four values it started with *before* the heavy `rux_ml.config` import, warns (`logging`, not a refusal) on a mismatch with `cfg.memory`, then still pins (Polars and LightGBM's libgomp size after the pin). The `:18-21` docstring and the `:86` comment are rewritten to the true state.
- **Q4.** `nthread = memory_cfg.omp_threads` (`_matrix_nthread()`) on every native-adapter matrix: `QuantileDMatrix`, `ExtMemQuantileDMatrix`, the eval `DMatrix`, and the `predict` / `predict_proba` `DMatrix`. A `model_kwargs` `nthread` still overrides the booster only (the gate's letter). `registry/scorer.py:97` is untouched (§Not in scope).
- **Not built:** Q3 (no CLI restructure, no re-exec) and Q5 (moved to `docs/0.1/RESEARCH-BACKLOG.md`, beside a Q3 overlap-cost item marked as a lean, not ruled).
- Docs: `ARCHITECTURE.md` thread-allocation row, `CHANGELOG.md` [Unreleased] Fixed, `env.py` module docstring. `README.md:626` / `:631` (MKL described as acting) not touched.

**Tests first.** 8 new tests; 7 failed on the Mac before the fix (3 adapter, 1 isolation, 3 trial-runner); the 8th, the Linux-only effect test (`threadpool_info` + `pl.thread_pool_size()` in a child run with the env the parent passes, after the child's `rux_ml.config` import: libgomp 5, OpenBLAS 2, Polars 4), is skipped on the Mac by design (Phase 1 stale 6).

**Results.**
- **Mac** (the worktree `.venv` lacks the catboost / lightgbm extras): the touched modules 31 passed / 1 skipped (the Linux test) / 2 deselected (catboost, lightgbm not installed); the full suite 789 passed / 5 skipped / 17 failed, every failure a catboost / lightgbm test (environmental). ruff, ruff format and basedpyright on the touched files clean.
- **Desktop (Linux):** the full suite **806 passed / 5 skipped / 0 failed** at `303c3a5` against **798 / 5 / 0** at `7b537b2` (pre-fix): +8 = the new tests, the skipped count unchanged, so the Linux effect test ran and passed. `basedpyright src/` 0 errors.
- **Revert spot-check (desktop):** with `env=child_env` removed, the effect test fails (`the child must be spawned with an explicit env`); restored, the tree clean.
- **Q4 model bytes (desktop, `scripts/measure_pr063_bytes.py`; 200,000 × 27 float32, 50 rounds, an eval set, seed 7, each fit twice):** `matrix_nthread_explicit` False (pre-fix, matrices at the OpenMP default 32) vs True (branch, 24). **The model `ubj` sha256 and the prediction sha256 are equal between pre-fix and branch on all four QuantileDMatrix arms** (cpu and cuda × regression and binary), and each self-reproduces.
- **Found, pre-existing, not fixed (out of scope):** the native adapter's **ExtMem path with an eval set fails on both versions**: `ValueError('data did not contain feature names, but the following fields are expected: feat__0, …')`. The ExtMem training matrix (from the Parquet iterator) carries no feature names, while the eval `DMatrix` (pandas) does. So the ExtMem bytes were not compared; the ExtMem matrix's `nthread` is covered by its unit test only. This is its own item (a defect on `dev`).

**Spend:** $0; one desktop window of 4 min 29 s.

**Independent review (2026-10-16 ≈ 16:25 CDT; Mac only, no desktop).** The build was written by the previous reviewer, so a second reviewer read `git diff origin/dev...HEAD` against the gate.
- **Checked, no defect:** the env is in the child from process start: `python -m rux_ml._internal.trial_runner` loads `rux_ml/__init__` (importlib.metadata only), the empty `_internal/__init__`, and stdlib top-level imports; `_internal/env.py` is stdlib-only (`MemoryConfig` under `TYPE_CHECKING`), so the snapshot comes before `rux_ml.config`. The parent's `cfg` and the child's come from the same layers (`--config` / `--problem` / `--study` / the `--set` overrides as JSON, the same inherited `RUXML_*` env and cwd), so `thread_env(cfg.memory)` is what the child pins to. The parent's `os.environ` is untouched (a new dict). The mismatch path only logs a warning. `src/` configures no logging handler, so in the real child it reaches stderr through logging's last-resort handler. `nthread` is on all five native-adapter matrices and nothing else in fitting changed; `model_kwargs` `nthread` still overrides the booster only (the gate's letter). No config or default changed; Q3 / Q5 not built.
- **Revert checks (each fix reverted alone, its test run, the file restored; tree clean after):** all 9 caught: `env=` removed; env passed without the thread values; the snapshot and check moved after the pin; the warning never fires; `nthread` removed from QuantileDMatrix / ExtMem / the eval DMatrix / predict / predict_proba.
- **Fixed, test first:** (1) the warning named only the mismatched keys; it now logs all four start and `cfg.memory` values plus the mismatched list (new test `test_the_child_warning_logs_all_four_values`, red before the fix). (2) Docs and comments that still described PR-008's pre-import pin as working: `docs/CONVENTIONS.md` §Subprocess-per-trial env pinning (steps 1 – 6 claimed `rux_ml.config` is light and the pin comes before any heavy import), §Per-family threading model ("The import-order fix is PR-063"; the XGBoost row "it is not"; the LightGBM row "best-guess, unmeasured" although Q1 measured 24), `README.md:629`, and the comments at `cli/tune.py:97`, `cli/train.py:715`, `_internal/seeds.py:26` ("children self-pin"). Wording only; no behaviour change, so the Linux effect test's coverage is unchanged.
- **Left as found (not defects of this build):** `README.md:626` describes MKL as acting (MKL is not installed; Phase 1 stale 7). `test_main_checks_the_env_it_started_with` would run `main` past its stub if the check call were deleted, and so would open the study named in `configs/base.toml`'s storage; it is a failure-mode-only side effect.
- **Mac results after the fixes:** the touched modules (`test_trial_runner`, `test_isolation`, `test_thread_budget`, `test_env`): 32 passed / 1 skipped (the Linux effect test) / 2 failed (`test_catboost_takes_the_same_argument` and `test_lightgbm_accepts_the_argument_and_ignores_it`: `ModuleNotFoundError`, because the extras are not in this venv). `ruff check --no-cache`, `ruff format --check` and `basedpyright` are clean on the touched files.

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
