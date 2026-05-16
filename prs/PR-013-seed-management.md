# PR-013: Seed management

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PR** (research-backed at design time).

## Research findings

### State Assessment (2026-05-16)

**Current state**:
- `TuningConfig.entropy: int | None = None` (`config/tuning.py:70`) is the only seed-input today; it flows in three places: `make_sampler(cfg.tuning, seed=cfg.tuning.entropy)` in `_internal/trial_runner.py:115` + `cli/tune.py:73`, and `make_splitter(cfg.cv, seed=cfg.tuning.entropy)` in `tuning/objective.py:256`.
- `make_trainer(cfg.training)` (`training/factory.py:48`) does **not** thread `random_state` through to XGBoost — XGBoost defaults to its internal nondeterministic init for every trial.
- Hardcoded `_DEFAULT_SPLIT_SEED = 0` constants in `cli/train.py:41` and `registry/promote.py:50` (both with explicit "PR-013 will replace" comments).
- `features/encoders.py:38` has `random_state: int = 0` default for `NestedCVWrapper` (target-encoder internal CV) — orthogonal to the {split, cv, sampler, xgb} bag.
- TrialAttrs Optional fields awaiting PR-013 (`runs/attrs.py:73-80`): `entropy_hex`, `image_digest`, `xgboost_version`, `cuda_runtime_version`, `gpu_model`, `driver_version`, `omp_threads`. PR-011 deliberately left `omp_threads` optional pending PR-013.
- PR-015's `make_splitter(cfg.cv, seed=...)` already accepts a seed and documents which strategies thread it (KFold/StratifiedKFold when `shuffle=True`) vs ignore it (TimeSeriesSplit, GroupKFold, CPCV).
- PR-007's `make_sampler(cfg.tuning, seed=...)` already accepts a seed.
- PR-008's strict module-load order (`stdlib top-level → cfg → pin_threads → heavy imports`) in `trial_runner.py` requires `_internal/seeds.py` to be lazy-imported after `pin_threads`, since `SeedSequence` lives in numpy.

**Assumptions at PR draft time**:
1. `numpy.random.SeedSequence.spawn` API unchanged.
2. XGBoost sklearn `random_state` kwarg still works.
3. `xgboost.config_context()` exposes CUDA runtime info.
4. `.docker-image-digest` file exists when in container; fallback "unknown" otherwise.

**Stale assumptions**:
- **Assumption 3**: XGBoost 3.x exposes build CUDA via `xgboost.build_info()["CUDA_VERSION"]` (confirmed in the PR-012 container smoke output on the workbench: `CUDA_VERSION: [12, 9]`). `config_context()` is for *configuring* XGBoost, not querying build info. Mechanical correction — use `build_info()` instead.
- All other assumptions hold.

**New constraints learned from prior PRs**:
- PR-015 settled `cv_seed` integration semantics in `data/cv.py:make_splitter`. No new research surface — mechanical plumbing.
- PR-007 settled `sampler_seed` integration semantics in `tuning/samplers.py:make_sampler`. Same.
- PR-009's `TrialAttrs.from_cfg(...).record(trial)` is the canonical write path. PR-013 must either add a `record_extras(trial, *, bag, versions)` method or expand `from_cfg` to take bag + version kwargs without breaking PR-009 callsites in `objective.py` + `cli/train.py`.
- PR-011 set the precedent: tighten a previously-Optional field to required in a later PR. PR-013 does the same for `entropy_hex`, `image_digest`, `xgboost_version`, `cuda_runtime_version`, `omp_threads`.
- PR-012 (PR #17, merge pending) produces `.docker-image-digest`. PR-013's `image_digest` capture treats missing file as the literal string `"unknown"` per spec line 43.

### Research Questions

**Must-answer**:

1. **Q1** — How should the SeedBag be derived per trial — fresh per-trial entropy, or spawned from a study-level master? Success: a pattern with one cited authoritative source.

**Dependencies**: none — Q1 is independent of the rest, which are local mechanical/preference choices.

**Explicitly excluded from this round** (nice-to-have):
- Adding a 5th `target_encoder_seed` slot to the bag (sub-decision H lean kept it out).
- Multi-GPU determinism (out of scope per D9 + CONSTRAINTS).
- Per-thread XGBoost RNG isolation (BLAS env vars + `nthread=1` for CPU bit-exact is sufficient).

### Findings

**Q1: SeedBag derivation strategy**

- *Options considered:*
  - **Option A: Per-trial spawn from study-level entropy.** `SeedSequence(entropy=cfg.tuning.entropy, spawn_key=(trial.number,))` per trial; `.spawn(4)` yields the four child streams for `(split, cv, sampler, xgb)`. Sampler seed at study level uses `cfg.tuning.entropy` directly.
    - Sources: [NumPy `SeedSequence`](https://numpy.org/doc/stable/reference/random/bit_generators/generated/numpy.random.SeedSequence.html), [scientific-python NumPy RNG best practices](https://blog.scientific-python.org/numpy/numpy-rng/), [NumPy parallel-RNG guide](https://numpy.org/doc/stable/reference/random/parallel.html)
    - Pros: full reproducibility across the whole study from one master integer; per-trial seeds distinct via `spawn_key`; matches numpy's documented pattern for independent streams (`spawn_key` is the canonical mechanism for tagging child streams with a stable per-context key — `trial.number` here).
    - Cons: requires both the master entropy AND the trial number to reproduce a single trial in isolation; the master can be `None` (auto) which then auto-pins on first use, recorded as `entropy_hex`.
  - **Option B: Per-trial fresh `SeedSequence()`** ignoring `cfg.tuning.entropy` for non-sampler use.
    - Cons: loses across-study reproducibility — re-running an entire study with the same config produces different results. Conflicts with `docs/CONSTRAINTS.md` reproducibility goals.
  - **Option C: Reuse `cfg.tuning.entropy` directly for all four slots every trial.**
    - Cons: every trial gets identical split/cv/xgb seeds — destroys per-trial randomness; defeats the purpose of HPO.

- *Disconfirming evidence sought:* searched for "SeedSequence anti-patterns" / "spawn_key concerns" — none found. The scientific-python blog and numpy docs explicitly recommend this pattern for "experiment seed + per-task spawn." Bonus: numpy's docs note that `spawn_key` is for stable child identification across non-deterministic ordering (which Optuna's trial dispatch can exhibit if pruning skips trial numbers — but with sequential subprocess-per-trial dispatch this isn't a current concern, just future-proofs).

- *Recommendation:* Option A
  - **Status**: PROVEN (numpy documentation + scientific-python blog)
  - **Why**: only pattern that gives both study-level reproducibility (one master entropy reproduces the whole study) AND per-trial independence.
  - **Risks accepted**: reproducing a single trial in isolation requires both `cfg.tuning.entropy` AND the trial number; recorded `entropy_hex` is the per-trial spawn-key-applied entropy, not the master — both should be present (master in cfg, trial-derived in user_attrs).

### Synthesis

**Outcome**: **Confirm** — Q1 findings support the PR-013 spec; sub-decisions B–H are local preference picks ranked against cited APIs (numpy SeedSequence, XGBoost sklearn estimator, PR-015/PR-007 locked integration surfaces). No phase-3 web research beyond Q1 needed. One mechanical Phase-1 correction (`config_context` → `build_info` for build metadata).

**Changes to this PR** from research:
- **Q1-A locked**: per-trial `SeedSequence(entropy=master_entropy, spawn_key=(trial.number,))` → `.spawn(4)` → `(split, cv, sampler, xgb)` ints via `bit_generator.random_raw()` masked to 64 bits or via `generate_state(1, dtype=np.uint32)` per the canonical pattern.
- **Sub-decision A1 locked** (SeedBag derivation): per-trial spawn from study-level entropy (`cfg.tuning.entropy`), recording the per-trial-derived `entropy_hex` in `user_attrs`. Master `cfg.tuning.entropy` stays in config.
- **Sub-decision B1 locked**: `entropy_hex: str` **required** on `TrialAttrs` (no default; promotion rejects trials missing it).
- **Sub-decision C1 locked**: `image_digest: str` **required** on `TrialAttrs`. Helper `read_image_digest()` returns the contents of `.docker-image-digest` or the literal string `"unknown"`.
- **Sub-decision D1 locked**: `xgboost_version: str` required (`xgboost.__version__`); `cuda_runtime_version: str` required (`f"{major}.{minor}"` from `xgboost.build_info()["CUDA_VERSION"]`); `gpu_model: str | None` optional (nvidia-smi-derived, GPU-only); `driver_version: str | None` optional (same).
- **Sub-decision E1 locked**: `omp_threads: int` required (always derivable from `cfg.memory.omp_threads`).
- **Sub-decision F1 locked**: `make_trainer(cfg, *, seed: int | None = None) -> Trainer`; `_xgb_kwargs` inserts `random_state=seed` when `seed is not None`.
- **Sub-decision G1 locked**: `registry/promote.py` reads source trial's `entropy_hex` from its `user_attrs`, reconstructs the `SeedBag` via the same derivation path (needs the trial's `entropy_hex` directly, since the master + trial.number derivation has already been applied), uses `bag.split_seed` + `bag.xgb_seed` for the promotion re-fit. Without this the promoted bundle would be trained on a different split than the trial reported. **`make_seed_bag` accepts the trial-level `entropy_hex` directly** as an alternative entry point.
- **Sub-decision H1 locked**: leave `features/encoders.py:38` `random_state=0` alone. Out of scope.
- **Mechanical correction**: `xgboost.config_context()` → `xgboost.build_info()["CUDA_VERSION"]` for cuda_runtime_version capture.

**Changes to ARCHITECTURE.md**:
- Reproducibility Architecture point 7 already mentions the SeedSequence pattern; refine wording to reference `_internal/seeds.py:make_seed_bag` + per-trial `spawn_key=(trial.number,)`.

**Changes to CONVENTIONS.md**:
- Add new "Seed management conventions (per PR-013)" subsection: SeedBag derivation pattern, master vs per-trial entropy, GPU near-determinism contract reaffirmed, promote re-fit reconstructs the original bag from `entropy_hex`, `nthread=1` + `tree_method=hist` + `device=cpu` for the CPU bit-exact test.

**Changes to CONSTRAINTS.md**:
- None new (the per-trial provenance triple already lists `entropy_hex`, `image_digest`, library/CUDA/driver versions, `omp_threads` as required). PR-013 makes the code match the constraint.

**Changes to ROADMAP.md**:
- `[ ]` → `[x]` flip on PR-013 row in the implementation commit (per [[feedback-roadmap-flip-in-pr]]).

**Changes to RESEARCH-BACKLOG.md**:
- PR-013 row: `state-assessed 2026-05-16` + `implementation-cleared 2026-05-16`.

**New PRs that must come first**: none.

**Research-backed details now locked in this PR**:
- SeedBag derivation: `SeedSequence(entropy=master, spawn_key=(trial.number,)).spawn(4)` → uint32 ints for each of split/cv/sampler/xgb.
- Required fields on `TrialAttrs`: `entropy_hex`, `image_digest`, `xgboost_version`, `cuda_runtime_version`, `omp_threads`.
- Optional GPU-only fields: `gpu_model`, `driver_version`.
- CPU bit-exact determinism contract: `device="cpu"` + `tree_method="hist"` + `OMP_NUM_THREADS=1` + pinned `xgb_seed`.
- GPU near-determinism contract: pinned `xgb_seed`, `atol=1e-5` (per D9, reaffirmed).

### Gate Check

- Premise still valid: ✓ (reproducibility-grade seed management is foundational for PR-014's golden tests and for the CONSTRAINTS provenance rule)
- No prerequisite PRs surfaced: ✓ (depends on PR-009 + PR-015, both merged; PR-012 desirable for `image_digest` capture but not strictly required per spec line 43)
- User approved updated spec: ✓ (2026-05-16)
- Implementation cleared: ✓ (2026-05-16)

---

## Scope

Implement reproducibility-grade seed management per D9. After this PR, every trial has a recorded master entropy with derived per-component seeds, and version provenance is fully captured.

- `src/rux_ml/_internal/seeds.py`:
  - `class SeedBag(BaseModel)` with fields: `entropy_hex`, `split_seed`, `cv_seed`, `sampler_seed`, `xgb_seed` (all int)
  - `make_seed_bag(entropy: int | None = None) -> SeedBag` — `SeedSequence(entropy)` + `.spawn(4)` to derive child streams; serializes `entropy` as hex
  - `seed_from_bag(bag, component) -> int` accessor used by callers
- `src/rux_ml/data/splits.py` updates: accept a `SeedBag` and use `bag.split_seed` (was a single seed in PR-004)
- `src/rux_ml/training/factory.py` updates: pass `bag.xgb_seed` as `random_state` to estimators
- `src/rux_ml/tuning/samplers.py` updates: pass `bag.sampler_seed` to the Optuna sampler
- `src/rux_ml/tuning/objective.py` and `_internal/trial_runner.py` updates:
  - Generate a `SeedBag` per trial (or accept one from `cfg.tuning.entropy` if user pinned it)
  - Record `entropy_hex` and the derived seeds in `user_attrs`
- Version provenance recorder in `_internal/env.py`:
  - `record_versions(trial)` writing `xgboost_version`, `cuda_runtime_version` (from `xgboost.config_context()` or `nvidia-smi --query-gpu`), `gpu_model`, `driver_version`, `image_digest` (from `.docker-image-digest` written by PR-012)
  - Called from the trial body once
- `runs.attrs.TrialAttrs` updates: every previously-Optional field listed in `docs/CONSTRAINTS.md` now required
- Tests:
  - Unit: same `entropy` → same `SeedBag`; different entropy → different bag
  - Determinism (CPU): pin entropy + run training twice → identical `predict_proba` (CPU `hist`, single thread)
  - Near-determinism (GPU, gated by `@pytest.mark.gpu`): pin entropy + run twice → predictions match within `atol=1e-5` (per D9 "near-deterministic, not bit-exact" framing)
  - Integration: every trial in a fresh `tune start` run records the full provenance set; `runs show` displays it

NOT in scope: golden regression tests (PR-014 builds on this PR's determinism contract).

## Dependencies

PR-009. (PR-012 desirable for `image_digest` capture but not strictly required — `image_digest` falls back to "unknown" if no `.docker-image-digest` file exists.)

## Architecture section implemented

`docs/ARCHITECTURE.md` → "Reproducibility Architecture" point 7 (seeds + version logging), "Decision Rules" (none new — this is a fill-in).

## Verification criteria

- [ ] `make_seed_bag()` is deterministic given the same `entropy`
- [ ] `entropy_hex` round-trips: writing then reading gives the same `SeedBag`
- [ ] CPU determinism test passes (bit-exact)
- [ ] GPU near-determinism test passes (within `atol=1e-5`)
- [ ] Every trial's `user_attrs` includes the full set required by `docs/CONSTRAINTS.md`
- [ ] `TrialAttrs` validation now requires every previously-Optional provenance field
- [ ] Promotion (PR-010) of a trial without complete provenance fails — re-run a tune to verify nothing regressed

## Research backing

Tier 1:

- D9: [NumPy SeedSequence](https://numpy.org/doc/stable/reference/random/bit_generators/generated/numpy.random.SeedSequence.html), [scientific-python NumPy RNG best practices](https://blog.scientific-python.org/numpy/numpy-rng/), [XGBoost #8820 GPU non-det](https://github.com/dmlc/xgboost/issues/8820), [XGBoost #5458](https://github.com/dmlc/xgboost/issues/5458)

State assessment must verify:
- `numpy.random.SeedSequence.spawn` API is unchanged
- XGBoost `random_state` still flows through the sklearn wrapper
- GPU determinism behavior has not regressed (a near-bit-exact change would be a major event worth flagging)

## Notes

- Per D9, multi-GPU training is explicitly non-deterministic — but we don't have multi-GPU, so this isn't a concern.
- If GPU near-determinism tests fail intermittently, do NOT loosen the tolerance silently. Investigate the change against XGBoost's release notes; flag a constraint update if the behavior has shifted.
- `entropy` is stored as a hex string in `user_attrs` so it's human-readable and serializable; convert to int at use time via `int(entropy_hex, 16)`.
