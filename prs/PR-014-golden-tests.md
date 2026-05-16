# PR-014: Golden regression test infrastructure

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PR** (research-backed at design time).

## Research findings

_To be populated by `PROCEDURE-pr-research.md`._

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
