# PR-008: Subprocess-per-trial isolation

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PR** (research-backed at design time). Note: this PR has hard CUDA + multiprocessing constraints — state assessment is especially important.

## Research findings

### State Assessment (2026-05-16, LOCAL-ONLY)

**Current state of the codebase**:

- `dev` at `f4ccc73` (PR-007 merged). Working tree clean.
- `src/rux_ml/_internal/` has `__init__.py`, `git.py`, `hashing.py`. **No `trial_runner.py` yet.**
- `src/rux_ml/tuning/` has `__init__.py`, `samplers.py`, `pruners.py`, `study.py`, `objective.py`. **No `isolation.py` yet.**
- `TuningConfig.trial_isolation: Literal["subprocess", "in_process"] = "subprocess"` pinned in PR-002 — default IS subprocess; PR-007's `cli/tune.py::start()` currently ignores it and always runs in-process. PR-008 honors it.
- `cli/tune.py::start()` (PR-007) calls `study.optimize(build_objective(cfg), n_trials=n_trials)` directly. PR-008 swaps in a `cfg.tuning.trial_isolation` branch.
- `MemoryConfig` (PR-002) carries the thread knobs (`omp_threads=24`, `openblas_threads=1`, `mkl_threads=1`, `polars_threads=24`) the child must export as env vars **before** any heavy library imports.
- `build_objective(cfg)` (PR-007) is fully usable inside the child unchanged — it loads data once, builds Splitter per-trial, runs CV-mean + WilcoxonPruner, records the 8-layer `user_attrs` (including PR-015's `cv_cfg_hash`).
- `runs/provenance.py` (PR-007) helpers work from any process loading `RuxMLConfig`.
- `docs/CONSTRAINTS.md` (D10/D16): "Any subprocess that uses CUDA must be created with `spawn` start method (or via `subprocess.run` of a sibling script)." `subprocess.run` is the locked path — automatically gives fresh-interpreter spawn semantics.

**Verification of D10 + D16 design-research assumptions (local-only)**:

| Item | Status | Where verified |
|---|---|---|
| `optuna.load_study(name, storage)` cross-process API | **STILL CURRENT** | exercised by PR-007's `tests/cli/test_tune_subcommands.py` |
| `subprocess.run(... check=False, timeout=...)` stdlib API | **STILL CURRENT** (stdlib) | — |
| XGBoost ≥ 3.x + CUDA forbids fork | **STILL CURRENT** | `CONSTRAINTS.md` locked + PR-006 confirmed XGBoost 3.2.0 installed |
| SQLite serialized-write multi-process (sequential trials) | **STILL CURRENT** | PR-006 + PR-007 tests round-trip studies through `sqlite:///` URLs successfully |

**Stale assumption in the PR-008 file / architecture diagram**:

The `docs/ARCHITECTURE.md` Data Flow diagram shows `study.optimize(_spawn_trial_dispatcher, n_trials=50)` in the parent with the dispatcher invoking `subprocess.run` and "parent reads final score from study after subprocess exits." This pattern is semantically broken for sequential subprocess dispatch: `study.optimize(func, n_trials=N)` calls `study.tell(trial, func_return)` for each trial. If the dispatcher returns the score the child just told, `study.optimize` will double-tell. If the dispatcher raises `TrialPruned` to suppress the tell, the parent-side trial is marked PRUNED while the child told a completed trial — confusing storage state.

The cleaner pattern (used by Optuna's distributed tutorial, which D10 cites): each child is its own independent worker that runs `study.optimize(build_objective(cfg), n_trials=1)` against shared SQLite storage. Parent doesn't call `study.optimize` at all — it loops `n_trials` times invoking `subprocess.run` of the child.

This is implementation-detail drift from the architecture diagram, not a design-decision drift. D10's `subprocess.run` + spawn + SQLite-coordination choice stands.

**New constraints learned from PR-006 / PR-007 / PR-015**:

1. **Env-var pinning ordering** — child must set `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS` / `POLARS_MAX_THREADS` from `MemoryConfig` **before** importing numpy / polars / sklearn / xgboost. `trial_runner.py`'s module-level imports stay minimal; thread-pinning happens after parsing args and loading `MemoryConfig`, **before** importing the rest.
2. **Override propagation** — parent CLI parses `--set key=value` flags into `opts.overrides: dict[str, Any]`. For subprocess dispatch, the parent serializes this dict to a tmp JSON file and passes `--overrides-json <path>` to the child (matches the architecture diagram).
3. **`build_objective` is the child's main work function** unchanged — the child does load `RuxMLConfig` → `create_or_load` study → `study.optimize(build_objective(cfg), n_trials=1)` → exit.
4. **PR-015's ExtMem×Splitter gate** is already inside `build_objective`; runs unchanged in the child.
5. **No timeout knob on `TuningConfig` today.** PR-008 adds `trial_timeout_s: int | None = None` (no default; opt-in) for the "kill hanging trials" verification criterion.
6. **No state drift from PR-007** on the in-process path — `cfg.tuning.trial_isolation = "in_process"` keeps PR-007's existing path working.

### Synthesis (Phase 4, 2026-05-16)

**Outcome: Confirm** — D10 + D16 design research stands; only the architecture-diagram drawing needs a minor correction. Tier-1 classification stands (sub-decisions below are implementation-within-design choices, not new research questions; A1 specifically aligns with the cited Optuna distributed tutorial pattern that D10 already references).

**User-approved sub-decisions (2026-05-16):**

- **A1** — Manual loop in parent: `cli/tune.py::start()` loops `n_trials` times calling `subprocess.run([... rux_ml._internal.trial_runner --n-trials 1 ...])`. Child runs its own `study.optimize(build_objective(cfg), n_trials=1)`. Storage coordinates state. Pattern grounded in D10's cited Optuna distributed tutorial ("each worker = one trial").
- **B1** — Override propagation via JSON file path (`--overrides-json <path>`). Matches the architecture diagram. Parent writes a tmp JSON; child reads and passes to `RuxMLConfig.from_layers(overrides=…)`.
- **C1** — Add `trial_timeout_s: int | None = None` to `TuningConfig`. `subprocess.run(..., timeout=...)`; on `TimeoutExpired` the parent kills the child and marks the trial via `study.tell(trial, state=optuna.trial.TrialState.FAIL)`.
- **D1** — Update `docs/ARCHITECTURE.md` Data Flow diagram in this PR's commit to reflect the manual-loop pattern.

**Changes to PR-008 spec from research**:

- Adopt the manual-loop dispatch pattern (A1) over the architecture diagram's `study.optimize(dispatcher, ...)` shape.
- Add `trial_timeout_s` to `TuningConfig`.
- Document the env-pinning ordering convention in `docs/CONVENTIONS.md`.
- Architecture diagram correction rides with this commit.

**Changes to `docs/ARCHITECTURE.md`** (same commit):
- Data Flow diagram — replace the parent's `study.optimize(_spawn_trial_dispatcher, n_trials=50)` line with a manual `for _ in range(N): subprocess.run([...])` loop. Child block remains as drawn (it runs its own `study.optimize(..., n_trials=1)`).

**Changes to `docs/CONVENTIONS.md`** (same commit):
- New "Subprocess-per-trial env pinning" subsection — the child must export thread env vars before any heavy import.

**Changes to `docs/CONSTRAINTS.md`**: None.

**No new prerequisite PRs surfaced.** PR-009 (run logging) and PR-011 (memory watchdog) remain correctly sequenced downstream.

### Gate Check (Phase 5, 2026-05-16)

- Premise still valid: ✓ (D10 + D16 stand; only the diagram drawing needs correction)
- No prerequisite PRs surfaced: ✓
- User approved Tier-1 classification: ✓ (2026-05-16)
- User approved locked-in spec (A1, B1, C1, D1): ✓ (2026-05-16)
- Implementation cleared: ✓ (2026-05-16)

---

## Scope

Add subprocess-per-trial isolation per D10 + D16. Each Optuna trial runs in a fresh subprocess (spawn semantics) to reclaim RAM deterministically and to keep CUDA initialization out of the parent.

- `src/rux_ml/_internal/trial_runner.py` — child entry point invoked as `python -m rux_ml._internal.trial_runner --study X --storage URL --config TOML --trial-id N --overrides-json PATH`:
  1. Pin OMP/BLAS env vars (per D10) — even though PR-011 owns the watchdog, the env pinning is needed by every child
  2. Reload `RuxMLConfig` from the TOML stack
  3. `optuna.load_study(name, storage)`; retrieve the trial by id
  4. Walk search space → overrides → `trial_cfg`
  5. Record `user_attrs` set
  6. Run the same training pipeline as PR-007's in-process objective
  7. `study.tell(trial, score)` (or raise `optuna.TrialPruned`)
- `src/rux_ml/tuning/isolation.py` — `_spawn_trial_dispatcher(trial)`:
  - serializes overrides to a temp JSON file
  - `subprocess.run([...], check=False)` of `python -m rux_ml._internal.trial_runner ...`
  - parent reads final score from the study after subprocess exits (avoids re-raising across the boundary)
  - timeouts + signal handling so a runaway trial can be killed
- `src/rux_ml/cli/tune.py` updates: `start` / `resume` now choose between in-process and subprocess dispatcher based on `cfg.tuning.trial_isolation` (TOML knob, default `"subprocess"`)
- Tests:
  - Unit: dispatcher serializes overrides correctly; subprocess invocation argv is correct
  - Integration: tune for 3 trials with `trial_isolation = "subprocess"`; assert each child's PID is distinct from the parent; assert results land in SQLite
  - **CUDA-spawn correctness test** (`@pytest.mark.gpu`): if the parent imported anything that initializes CUDA, the spawn boundary must still work — assert child can `import xgboost` and call `xgb.QuantileDMatrix(..., device="cuda")` without `Cannot re-initialize CUDA in forked subprocess`

NOT in scope: psutil watchdog inside the child (PR-011), peak_rss recording (PR-011), entropy/seeds (PR-013).

## Dependencies

PR-007.

## Architecture section implemented

`docs/ARCHITECTURE.md` → "Decision Rules: Trial isolation", "Data Flow" (the parent → child arrow), "Internal" component row (`_internal/trial_runner.py`).

## Verification criteria

- [ ] `python -m rux_ml._internal.trial_runner --help` works
- [ ] `tune start <name> --n-trials 3` with `trial_isolation = "subprocess"` runs 3 distinct subprocesses
- [ ] Child PID ≠ parent PID for every trial (process-list assertion)
- [ ] CUDA initializes successfully in the child even when parent has CUDA-touching libraries imported
- [ ] Override JSON is correctly read by the child and produces the expected `trial_cfg`
- [ ] SQLite parent-child coordination works: trials appear in storage, parent observes the final score
- [ ] Timeout / kill behavior: if the child hangs, parent kills it after a configurable timeout and marks the trial FAILED
- [ ] In-process mode (`trial_isolation = "in_process"`) still works and matches PR-007 behavior

## Research backing

Tier 1:

- D10: [Optuna OOM issue #1178](https://github.com/optuna/optuna/issues/1178), [Optuna distributed tutorial](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html)
- D16: [PyTorch CUDA fork issue #40403](https://github.com/pytorch/pytorch/issues/40403), [vLLM CUDA spawn #8893](https://github.com/vllm-project/vllm/issues/8893), [Optuna distributed tutorial](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html)

State assessment must verify:
- `optuna.load_study(storage=...)` from a child process is still the documented coordination mechanism (no breaking change to multi-process semantics)
- XGBoost ≥ 2.x continues to require `spawn` for CUDA across subprocesses
- SQLite locking has not regressed for serialized-write multi-process use

## Notes

- Per `docs/CONSTRAINTS.md`, `fork` is forbidden when CUDA is touched. Whether or not the parent ever calls XGBoost-GPU, defaulting to `subprocess.run` of a sibling script keeps us safe.
- The dispatcher pattern (`subprocess.run` + read final score from storage) is preferred over `ProcessPoolExecutor(mp_context="spawn")` per D16 — stronger isolation and easier log capture. If you choose `ProcessPoolExecutor`, you MUST pin `mp_context=mp.get_context("spawn")` explicitly.
