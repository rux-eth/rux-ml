# PR-008: Subprocess-per-trial isolation

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PR** (research-backed at design time). Note: this PR has hard CUDA + multiprocessing constraints — state assessment is especially important.

## Research findings

_To be populated by `PROCEDURE-pr-research.md`._

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
