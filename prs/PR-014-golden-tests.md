# PR-014: Golden regression test infrastructure

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PR** (research-backed at design time).

## Research findings

### State Assessment (2026-05-16)

**Current state**:
- No `tests/golden/` directory — clean slate.
- No `make_data(cfg)` factory exists. The codebase uses `load_parquet` + `materialize` + `train_val_test_split` directly (`cli/train.py:_fit_and_score` is the canonical pattern).
- `rux_ml.registry.load_model(problem, *, version, registry_root)` exists and returns `(Pipeline, xgb.Booster)`.
- `pyproject.toml:100-106` already registers the `golden` marker and excludes it from default runs via `addopts = "-m 'not gpu and not slow and not golden and not docker'"`. ✓
- `Makefile` has `test-golden: uv run pytest -m golden`. Needs the new `regenerate-golden` target.
- No `pytest_addoption` anywhere in the codebase — `--regenerate-golden` flag is a fresh add via `tests/golden/conftest.py`.
- PR-013 contract on CPU: `device="cpu"` + `tree_method="hist"` + `OMP_NUM_THREADS=1` + pinned `xgb_seed` → bit-exact predictions (`tests/integration/test_determinism_cpu.py` asserts this with `assert_array_equal`).
- `TrialAttrs.entropy_hex` is required as of PR-013 — every recorded trial has a stable seed footprint that can be embedded in the golden `manifest.json`.

**Assumptions at PR draft time**:
1. `np.testing.assert_allclose(atol, rtol)` API unchanged.
2. Pinned XGBoost / sklearn / Polars versions still installable.
3. The full pipeline (load → split → features → trainer → fit → predict) is the right surface to golden-test.
4. Spec's `make_data → make_features → make_trainer → fit → predict_proba` description maps to existing surfaces.

**Stale assumptions**:
- **Assumption 4**: there is no `make_data` factory. Use `load_parquet` + `train_val_test_split` directly per the `cli/train.py` pattern. Mechanical correction.
- All other assumptions hold.

**New constraints learned from prior PRs**:
- **PR-013**: golden tests can lean on `make_seed_bag(master_entropy=<fixed>, trial_number=0)` and thread `bag.xgb_seed` into `make_trainer(cfg, seed=...)` + `bag.split_seed` into `train_val_test_split`. With CPU single-thread this is bit-exact, but the spec's `atol=1e-5 / rtol=1e-4` stays as the conservative across-XGBoost-minor-version envelope.
- **PR-010**: `load_model` is the inference surface; promoting a trained model into a tmp registry and round-tripping through `load_model` catches skops/ubj serialization regressions that an in-process-only test misses.
- **PR-012**: container exists but goldens don't need it — they run anywhere XGBoost CPU does.

### Research Questions

**Must-answer**:

1. **Q1** — Is there a current pytest-recommended pattern for `--regenerate-fixture`-style flags that's better than `pytest_addoption` + `request.config.getoption`? Success: confirm `pytest_addoption` is still the documented mechanism.

**Dependencies**: none — Q1 is independent.

**Explicitly excluded from this round** (nice-to-have):
- Golden tests for individual transformers (unit-test territory; spec line 39).
- CI integration (no CI at v0; spec line 39).
- Adding more golden surfaces beyond the single end-to-end one (spec line 70: "Be deliberate. Each additional golden surface increases the cost of a CUDA upgrade").

### Findings

**Q1: Pytest custom-flag pattern**

- *Options considered:*
  - **Option A: `pytest_addoption` + `request.config.getoption("--flag")`** — the documented pytest pattern from [pytest docs "How to write parametrized tests"](https://docs.pytest.org/en/stable/example/parametrize.html#deferring-the-setup-of-parametrized-resources) and the [conftest.py documentation](https://docs.pytest.org/en/stable/reference/fixtures.html#conftest-py-sharing-fixtures-across-multiple-files).
    - Pros: standard, no extra deps, documented.
    - Cons: requires the helper to read the option each call (cheap).
  - **Option B: Separate CLI module `python -m rux_ml.golden.regenerate`** — out-of-band regeneration.
    - Cons: duplicates the test logic; PR-014 spec line 33 explicitly says "Makefile target `make regenerate-golden` invoking `pytest -m golden --regenerate-golden`" — keeping it in pytest is the intended path.
  - **Option C: Env-var-based opt-in (`RUXML_REGEN_GOLDEN=1`)** — used by PR-012 for docker smoke gating.
    - Cons: env vars are coarser than CLI flags; the docker case used env because the gate is "do this surface at all" while regen is "do this surface differently."

- *Disconfirming evidence sought:* searched for "pytest_addoption deprecated" — none found; the pattern has been stable since pytest 3.x.

- *Recommendation:* Option A
  - **Status**: PROVEN (pytest official docs + multiple production OSS projects use this pattern, e.g., pytest-snapshot, syrupy, polars own snapshot tests)
  - **Why**: matches the spec's intent and uses pytest's documented extension mechanism.
  - **Risks accepted**: none material.

### Synthesis

**Outcome**: **Confirm** — Q1 findings support the spec; sub-decisions B–J are local choices that rank against single sources (PR-013's locked CPU determinism contract, PR-010's `load_model` API, `np.testing` docs, pytest docs). Mechanical correction: drop `make_data` references and use the existing `load_parquet`/`train_val_test_split` pattern.

**Changes to this PR** from research:
- **Sub-decision A1 locked**: reuse `cli/train.py:_fit_and_score` pattern. No new `make_data` factory.
- **Sub-decision B1 locked**: two-part golden — (1) in-process fit + predict against fixtures; (2) registry round-trip via promote → `load_model` → predict, matching Part 1 within tolerance. Catches both training-stack and serialization regressions.
- **Sub-decision C1 locked**: `tests/golden/fixtures/golden_v1/{synthetic.parquet, preds.npy, metric.json, manifest.json}`. Versioned dir name lets a future `golden_v2` coexist.
- **Sub-decision D1 locked**: `pytest_addoption` in `tests/golden/conftest.py` adds `--regenerate-golden`; tests read via `request.config.getoption(...)`.
- **Sub-decision E1 locked**: `atol=1e-5, rtol=1e-4` for predictions; `abs_tol=0.005` for the AUC metric band.
- **Sub-decision F1 locked**: dedicated `test_helpers.py` proves the tolerance assertions reject a deliberate perturbation (the spec's "test the test" verification criterion).
- **Sub-decision G1 locked**: golden fit uses CPU `device="cpu"` + `tree_method="hist"` + `n_jobs=1` + `OMP_NUM_THREADS=1` (autouse fixture, same shape as `test_determinism_cpu.py`). Portable everywhere.
- **Sub-decision H1 locked**: `manifest.json` records library versions (xgboost, sklearn, numpy, polars, skops), `cuda_runtime_version` from PR-013's `xgboost.build_info()`, master `entropy_hex`, `data_hash`, `root_cfg_hash`, `created_at` ISO timestamp.
- **Sub-decision I1 locked**: `golden` marker already registered; add `Makefile` target `regenerate-golden`.
- **Sub-decision J1 locked**: test docstring lists the 3-step investigation procedure before regen (check manifest diff → run CPU determinism test → only then `make regenerate-golden`).

**Changes to ARCHITECTURE.md**:
- None new (Testing Strategy table already lists the golden row).

**Changes to CONVENTIONS.md**:
- New "Regenerating golden fixtures (per PR-014)" subsection describing the policy.

**Changes to CONSTRAINTS.md**:
- None (Tolerance-Based Golden Tests Only rule already documented; PR-014 makes the code match).

**Changes to ROADMAP.md**:
- `[ ]` → `[x]` flip on PR-014 row in the implementation commit (per [[feedback-roadmap-flip-in-pr]]).

**Changes to RESEARCH-BACKLOG.md**:
- PR-014 row: `state-assessed 2026-05-16` + `implementation-cleared 2026-05-16`.

**New PRs that must come first**: none.

**Research-backed details now locked in this PR**:
- Fixture layout: `tests/golden/fixtures/golden_v1/`.
- Tolerances: `atol=1e-5, rtol=1e-4` (predictions); `abs_tol=0.005` (AUC band).
- Flag mechanism: `pytest_addoption` + `request.config.getoption("--regenerate-golden")`.
- Determinism floor: CPU `tree_method="hist"` + `OMP_NUM_THREADS=1` + pinned `xgb_seed`.
- Manifest schema: library versions + cuda_runtime_version + entropy_hex + data_hash + root_cfg_hash + ISO timestamp.

### Gate Check

- Premise still valid: ✓ (golden tests are the last v0 reproducibility surface; PR-013's determinism contract + PR-010's load_model make them tractable)
- No prerequisite PRs surfaced: ✓ (depends on PR-010 + PR-013, both merged)
- User approved updated spec: ✓ (2026-05-16)
- Implementation cleared: ✓ (2026-05-16)

---

## Scope

Lay down the golden-regression-test scaffolding per D12 + `docs/CONSTRAINTS.md` "Tolerance-Based Golden Tests Only", and ship one end-to-end golden test exercising the full pipeline.

- `tests/golden/conftest.py`:
  - Fixed-seed synthetic dataset fixture (small, deterministic)
  - Helper `assert_predictions_close(actual, golden, *, atol=1e-5, rtol=1e-4)` wrapping `np.testing.assert_allclose` with informative diff messages
  - Helper `assert_metric_within(metric_value, baseline, *, abs_tol=0.005)` for AUC/RMSE bands
- `tests/golden/fixtures/`:
  - `golden_v1/preds.npy` — golden prediction array (regenerated only via an explicit `--regenerate-golden` flag)
  - `golden_v1/metric.json` — `{"auc": <value>}`
  - `golden_v1/manifest.json` — what was used (data fingerprint, config hash, library versions)
- `tests/golden/test_xgb_baseline.py`:
  - Marker: `@pytest.mark.golden`
  - Runs `make_data` → `make_features` → `make_trainer` → `fit` → `predict_proba` end-to-end
  - Asserts predictions close + metric within tolerance bands
  - Documents in the test docstring that "this test will fail when CUDA / XGBoost / sklearn versions change — investigate before regenerating golden"
- `Makefile` target `make test-golden` invoking `pytest -m golden`
- `Makefile` target `make regenerate-golden` invoking `pytest -m golden --regenerate-golden` (writes new fixtures; intended to be manually reviewed and committed)
- `docs/CONVENTIONS.md` updated with a new section "Regenerating Golden Fixtures" describing the policy (NEVER auto-regenerate in CI; always manual + diffed against the committed manifest)
- Tests on the test infrastructure itself:
  - `assert_predictions_close` raises on actual divergence
  - `assert_metric_within` raises when the metric drifts beyond the tolerance band

NOT in scope: golden tests for individual transformers (those are unit-test territory); CI integration (no CI yet).

## Dependencies

PR-010 (registry; the golden test loads via `load_model` to exercise the inference path) and PR-013 (seed management; the test pins entropy explicitly).

## Architecture section implemented

`docs/ARCHITECTURE.md` → "Testing Strategy" (Golden regression row), `docs/CONSTRAINTS.md` "Tolerance-Based Golden Tests Only".

## Verification criteria

- [ ] `make test-golden` passes with the shipped fixture
- [ ] Tolerance assertions reject a deliberate perturbation (test the test)
- [ ] `--regenerate-golden` flag rewrites fixtures and prints a diff summary
- [ ] The test docstring explains the procedure for investigating a failure (don't just regenerate)
- [ ] No exact hash compared anywhere in the test (per `docs/CONSTRAINTS.md`)
- [ ] `pyproject.toml` `addopts` excludes `golden` from default `pytest` runs (already set in PR-001; verify still true)

## Research backing

Tier 1:

- D12: [pytest good practices](https://docs.pytest.org/en/latest/explanation/goodpractices.html), [Shaped golden tests in AI](https://www.shaped.ai/blog/golden-tests-in-ai), [Techment LLM regression](https://www.techment.com/blogs/llm-regression-testing/)

State assessment must verify:
- `np.testing.assert_allclose` behavior on `atol`/`rtol` is unchanged
- The pinned XGBoost / sklearn / Polars versions in the fixture manifest are still installable

## Notes

- This is the ONLY golden test we ship at v0 — adding more is fine, but each additional golden surface increases the cost of a CUDA upgrade. Be deliberate.
- Failures are debugging events, not "regenerate and move on" events. The Makefile target is intentionally separated and explicit.
- After this PR, the v0 milestone is complete: the workbench can ingest data, train and tune a model, log runs, promote to a registry, and verify regression — all with reproducibility provenance.
