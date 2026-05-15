# PR-013: Seed management

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PR** (research-backed at design time).

## Research findings

_To be populated by `PROCEDURE-pr-research.md`._

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
