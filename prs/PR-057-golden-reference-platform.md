# PR-057: the golden fixture's reference platform is the desktop (linux-x86_64)

**Landed-in:** (not yet landed)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `

**Tier: Tier-1 proposed (operator's call).** R-14 track, program PR-027 **Q8 step 3** on the rux-capital program lead's instruction of 2026-09-29, carrying the operator's decision ("go on all", 2026-09-29): the desktop is the golden fixture's reference platform. The research is program PR-027 build step R4 (2026-09-28; desktop records `~/pr027-r4/q8/`). The merge is operator-gated.

## Research findings

### State Assessment (2026-09-29)

Baselines: rux-ml `dev` @ `6c9cc68` (PR-055 merged); xgboost 3.2.0, scikit-learn 1.8.0, numpy 2.4.5, polars 1.40.1, skops 0.14.0, equal to `golden_v1/manifest.json`; the default suite 730 passed / 3 skipped on the Mac.

**Current state (read):**
- `golden_v1` (PR-014) was generated on the Mac (darwin-arm64) on 2026-05-16. Its `manifest.json` records library versions, `entropy_hex`, the data and config hashes, and no platform.
- `test_golden_xgb_baseline_in_process` fits the pipeline on the committed `synthetic.parquet` (CPU `hist`, `n_jobs=1`, `OMP_NUM_THREADS=1`, `subsample` = `colsample_bytree` = 0.8) and compares `predict_proba[:, 1]` at `atol=1e-5, rtol=1e-4` plus the AUC band 0.005.
- `docs/CONVENTIONS.md` §Regenerating golden fixtures: diff the versions (1), run the CPU determinism contract (2), and only then `make regenerate-golden` with a reviewed `manifest.json` diff (3).

**Program PR-027 R4 (2026-09-28, the desktop at `6c9cc68`):** step 1: the library versions equal the manifest's. Step 2: `test_determinism_cpu.py` 2 passed, the GPU determinism test 1 passed. Even so, `test_golden_xgb_baseline_in_process` misses 30 of 30 predictions (max abs diff 0.1588), while `test_golden_load_model_matches_in_process` passes. The Mac passes all 7 golden tests at the same commit. The committed fixture is platform-dependent: the same bytes, config and library versions give different trees on linux-x86_64.

**Drift:** none. Steps 1 and 2 are clean, so step 3 (regenerate) applies; what is new is *where*. The operator's decision (2026-09-29) makes the desktop, the only training compute, the reference platform.

**Decisions the build forced:**
1. **The platform check comes before the fit and the skip names both platforms.** A manifest with no `reference_platform` fails: a fixture of unknown platform is never compared, never silently passed and never skipped.
2. **`test_golden_load_model_matches_in_process` is unchanged and runs everywhere.** It compares a fit with its own registry round trip, which does not depend on the platform.
3. **A regeneration off the reference platform is not refused in code.** The rule is documented, and such a regeneration shows as a `reference_platform` change in the reviewed `manifest.json` diff.

### The cross-platform probe (2026-09-29)

The lead's brief carried a best guess, "cross-architecture floating point, unproven". To test it, one script ran on the Mac (darwin-arm64) and on the desktop (linux-x86_64) at `eac1e09`. It fitted the golden config and pipeline on the committed `synthetic.parquet` with `OMP_NUM_THREADS=1`, under four sampling settings, and the predictions were compared as hex floats.

| setting | `subsample` | `colsample_bytree` | Mac vs desktop |
|---|---|---|---|
| golden | 0.8 | 0.8 | 30 / 30 beyond 1e-5, max abs 0.158781 |
| no sampling | 1.0 | 1.0 | bit-identical |
| rows only | 0.8 | 1.0 | 0 / 30 beyond 1e-5, max abs 5.96e-8 |
| columns only | 1.0 | 0.8 | 30 / 30 beyond 1e-5, max abs 0.158484 |

Both machines read the same data bytes (sha256 `b1aee1ab…`) and made the same train / val split (the CSV sha256 of each is equal).

- **Proven:** the miss follows `colsample_bytree`. The same seed draws different feature subsets on the two platforms, while floating point alone moves a prediction by at most about one float32 ULP.
- **Best guess, unproven:** the column draw goes through the C++ standard library, whose shuffle and distribution algorithms are implementation-defined (libc++ on macOS, libstdc++ on Linux). XGBoost's source was not read.

Either way the decision stands: a fixture is comparable only on the platform that generated it.

### Found on the desktop

`make regenerate-golden` died with `pytest: error: unrecognized arguments: --regenerate-golden` and regenerated nothing. The flag is registered by `tests/golden/conftest.py`, and pytest reads that hook before argument parsing only when `tests/golden` is on the command line. This was fixed test-first: the recipe now passes `tests/golden`, and `tests/golden/test_regenerate_target.py`, in the default suite and collect-only, failed on the dev Makefile with the same usage error.

### Stopped at the regeneration (2026-09-29)

`make regenerate-golden` at `1460b2b` on the desktop rewrote all four files, and `synthetic.parquet` changed: its sha256 is `733f7d7d…` against the committed `b1aee1ab…`, which the Mac still regenerates byte for byte today.

- **Where the data differs:** in the float columns only. `x1` differs in 27 / 200 rows (max abs 4.4e-16, 14 ULP), `x2` in 24 / 200 (8.9e-16, 32 ULP), `x3` in 22 / 200 (6.7e-16, 128 ULP). `cat` and `y` are identical.
- **Effect on the fit:** none. The desktop fit on the desktop data is bit-identical to the desktop fit on the committed data, so the data change adds nothing to the prediction difference.
- **Cause (best guess, unproven):** numpy's BLAS or libm under `make_classification`.

Manifest review:

- `library_versions` and `entropy_hex` are equal.
- `reference_platform` Linux / x86_64 is added.
- `cuda_runtime_version` goes from unknown to 12.9 (the desktop's xgboost is a CUDA build).
- The data hashes follow the parquet.
- `root_cfg_hash` goes from `cd75bc2c…` to `c5ba8632…`. This is code drift since 2026-05-16, not the platform: the Mac computes `c5ba8632…` today too.
- AUC goes from 0.9511 to 0.9422.

Per the brief the build stopped here for the lead's call. **Resolved (2026-09-29):** the lead approved accepting the desktop's data: the difference is ≤ 8.9e-16, it does not change the fit, and the desktop is the reference. The four files were committed from the desktop at `4e1de63`.

### Synthesis / Gate

**Outcome: Confirm** (the operator's decision as stated). **Gate:** track-autonomous under program PR-027's approved build; the operator's approval is pending at merge.

---

## Scope

- **`tests/golden/conftest.py`**: `current_platform()` and `require_reference_platform(manifest, current)`. The gate returns on the reference platform, SKIPS with both platforms named on any other, and fails on a manifest with no `reference_platform`.
- **`tests/golden/test_xgb_baseline.py`**: `_build_manifest` records `reference_platform`; `test_golden_xgb_baseline_in_process` checks the gate before the fit (compare path only).
- **`tests/golden/fixtures/golden_v1/`**: regenerated on the desktop (`make regenerate-golden`).
- **Tests (new):** 7 items in `tests/golden/test_helpers.py` (skip on a mismatch × 3 platforms, compare on a match, fail on a missing platform × 3 manifest shapes) and 1 in `tests/golden/test_xgb_baseline.py` (the manifest records the platform it ran on).
- Docs: `docs/CONVENTIONS.md` §Regenerating golden fixtures, `CHANGELOG.md`, `docs/0.3/RESEARCH-BACKLOG.md`.

## Dependencies

None (off `dev` `6c9cc68`).

## Architecture section implemented

None changed. The rule lives in `docs/CONVENTIONS.md` §Regenerating golden fixtures; `docs/CONSTRAINTS.md` Tolerance-Based Golden Tests Only is unchanged (no tolerance moves).

## Verification criteria

- [x] The gate's 7 unit tests failed first (ImportError: no `require_reference_platform`) and pass.
- [x] The manifest test failed first (`KeyError: 'reference_platform'`) and passes.
- [x] `tests/golden/test_regenerate_target.py` failed first (`unrecognized arguments: --regenerate-golden`) and passes.
- [x] Mac: the default suite is green.
- [x] `make regenerate-golden` on the desktop at `1460b2b` rewrote the four files. `synthetic.parquet` differs at ≤ 8.9e-16 on the float columns and was accepted by the lead (see Research findings). The manifest diff was reviewed and committed at `4e1de63`.
- [x] Desktop at `4e1de63`: `make test-golden` passed 15 of 15 (the 7 original tests and the 8 new ones).
- [x] Mac at `4e1de63`: the golden suite passed 14 and skipped 1. `test_golden_xgb_baseline_in_process` skipped with the reason "compared only on the fixture's reference platform Linux-x86_64; this host is Darwin-arm64". The default suite passed 731 with 3 skipped.

## Research backing

Program PR-027 build step R4 (Q8, 2026-09-28) and the operator's decision of 2026-09-29; the cross-platform probe above.

## Notes

- The Mac (darwin-arm64) no longer compares the in-process golden, so a training-stack regression that shows only in predictions is caught on the desktop. The registry round-trip golden, and every other test, still run on the Mac.
- A regeneration after a library upgrade follows the same three steps, on the desktop.
- **Known limitation (non-blocking):** a model fitted with `colsample_bytree` < 1 differs between darwin-arm64 and linux-x86_64 from the same seed, and the per-trial record (`TrialAttrs`) does not store the platform. Training runs only on the desktop, so v0.3 is unaffected.
