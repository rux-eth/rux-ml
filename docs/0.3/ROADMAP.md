# Roadmap — v0.3

Ordered list of PRs for the v0.3 cut. Each PR is a single, reviewable change. Dependencies flow downward — no PR should be started before its predecessors are merged.

Full PR descriptions live in `prs/`. This file is the index.

The v0.2 roadmap (PR-022 through PR-027 + PR-028/29 procedural) is frozen at [`docs/0.2/ROADMAP.md`](../0.2/ROADMAP.md) — every row is `[x]`. The v0.1 roadmap is at [`docs/0.1/ROADMAP.md`](../0.1/ROADMAP.md). The v0.0 roadmap is at [`docs/0.0/ROADMAP.md`](../0.0/ROADMAP.md).

**Status legend:** `[ ]` pending | `[x]` merged | `[~]` in progress

---

## Theme: Honesty cut — eliminate phantom implementations

The v0.3 sprint completes design promises that landed without working production paths. Surfaced by the 2026-05-20 phantom-implementation audit (4 confirmed phantoms across two audit passes):

1. **Holdout test fold** never consumed by any code (carved by `make_splits`, ignored by `cli/train.py`, `registry/promote.py`, and the HPO objective which operates on the full df).
2. **ExtMem out-of-core ingest path** — `ParquetDataIter`, `select_ingest`, `cache_host_ratio`, `use_native` all wired in design + docs, never executed in production (sklearn-wrapper path bypasses the entire `select_ingest` decision).
3. **Optuna Artifacts Store** — `RunsConfig.artifacts_root` declared, `FileSystemArtifactStore` never instantiated, never used; docs claim it stores per-trial bundles + plots + prediction CSVs (nothing happens).
4. **`rux-ml tune retry-trial`** verb body exists, zero integration test coverage; test file docstring overstates coverage.

Plus one borderline ("2b" in the audit): `select_ingest()` is called in production but its return value drives only a telemetry log message, not behavior — covered by phantom #2's fix.

| PR | Description | Status | Depends on |
|----|-------------|--------|------------|
| [PR-030](../../prs/PR-030-v0-3-sprint-scaffolding.md) | **(this PR)** v0.3 sprint scaffolding — creates `docs/0.3/` + 6 PR stubs + CHANGELOG `[Unreleased]` entry. No code change. | `[x]` | — |
| [PR-031](../../prs/PR-031-hpo-honors-holdout-fold.md) | **Tier-2** HPO objective honors holdout fold — `tuning/objective.py` builds CV substrate from `make_splits(...)["train"] + ["val"]`, not `df_full`. Test fold becomes truly held out from HP search. Behavior-changing: existing study metrics will shift. | `[ ]` | PR-030 + design session |
| [PR-032](../../prs/PR-032-registry-score-cli-verb.md) | **Tier-2** `rux-ml registry score` CLI verb + scorer module — loads bundle via `load_bundle`, reconstructs splits, scores `splits["test"]`, writes receipt JSON + preds parquet under `receipts/`. Removes stale "future golden-regression evaluation (PR-014)" comment from `promote.py:97`. | `[ ]` | PR-031 |
| [PR-033](../../prs/PR-033-extmem-path-activation.md) | **Tier-2** ExtMem path activation — native `xgb.train` adapter exposing `Trainer` Protocol; routes `select_ingest` decision into actual DMatrix construction; `ParquetDataIter` chunking; thread `cache_host_ratio`; honor `use_native`. Test-mode flag forces the path on small datasets. Covers phantom #2 + #2b. | `[ ]` | PR-030 + design session |
| [PR-034](../../prs/PR-034-artifacts-store-integration.md) | **Tier-2** Optuna Artifacts Store integration — initialize `FileSystemArtifactStore(artifacts_root)` per study; upload per-trial diagnostic artifacts (preds CSV + fold scores JSON) on trial close. **Diagnostic only**: promote still re-fits (D8/PR-010 sub-decision A1 preserved). | `[ ]` | PR-030 + design session |
| [PR-035](../../prs/PR-035-tune-retry-trial-test.md) | **Tier-1** `tune retry-trial` integration test — `test_tune_retry_trial_*` in `tests/cli/test_tune_subcommands.py`. Exercises enqueue → optimize → verify. No production code change. | `[ ]` | — |
| [PR-036](../../prs/PR-036-v0-3-0-cut.md) | **Tier-2** v0.3.0 version cut — `scripts/rewrite_doc_refs.py` for `docs/0.2/* → docs/0.3/*`; pyproject `0.2.0 → 0.3.0`; CHANGELOG `[Unreleased]` → `[0.3.0]` (bundles PR-028 + PR-029 + PR-031..PR-035); tag `v0.3.0`. Analog of PR-021 / PR-026. | `[ ]` | PR-031, PR-032, PR-033, PR-034, PR-035 |

---

## Design session prerequisite

Before PR-031 implementation begins, a v0.3 design session runs `PROCEDURE-design-planning.md` from Phase 1 to resolve architectural sub-decisions that the implementation PRs can't decide unilaterally. The session's output populates [`DESIGN-log.md`](DESIGN-log.md).

**Sub-decisions to resolve in the design session:**

- **D1 — Holdout semantics**: Is `splits["test"]` "truly held out from HPO" (PR-031 changes `tuning/objective.py` substrate to train+val only) or "post-HPO sanity check" (HPO sees full df; test fold is just a final receipt)? Lean: truly held out — anything else makes the "holdout" label misleading.
- **D2 — ExtMem activation pattern**: How does the trainer switch between sklearn-wrapper and native `xgb.train` API at runtime? `Trainer` Protocol expects `fit/predict`; native API returns a `Booster` directly — needs an adapter. Sub-questions: `ParquetDataIter` chunking strategy; default `cache_host_ratio`; test strategy when no real workload triggers the threshold.
- **D3 — Artifacts: what to upload**: Prediction CSVs + fold scores JSON (diagnostic only — preserves D8/PR-010 A1 re-fit-at-promote) OR per-trial bundles (replaces re-fit; reverses A1). Lean: diagnostic-only — the alternative reverses an explicit prior decision and is much bigger work.
- **D4 — retry-trial**: Test-only fix, or behavior change too? Lean: test-only — body looks correct on review.

---

## Notes

- **Theme bundling**: PR-031 + PR-032 together complete the holdout story (HPO loop excludes test; new CLI verb scores it). They're independently mergeable but semantically paired.
- **PR-031 is behavior-changing**: any existing study TOML (e.g., `crypto_breakout_h3_baseline`) will produce different metrics after PR-031 because CV operates on 85% of data, not 100%. The crypto-h3 baseline at RMSE 0.027101 will need a fresh receipt — easy to confuse with regression. Bundle a baseline-receipt-refresh step into PR-031.
- **PR-033 is the largest scope** (~6–10 hours estimated). Switching the trainer between sklearn and native APIs at runtime is real architectural work; could surface compatibility issues with the metric registry (`compute_score` assumes `trainer.predict_proba` etc.). May warrant splitting if Phase 1 surfaces drift.
- **Each implementation PR runs the full `PROCEDURE-pr-research.md`** (Phase 1 state assessment + prior-art audit + Phases 2-4 + Group D probes per the post-PR-028/29 procedure). Each commit flips its row in this ROADMAP per `feedback_roadmap_flip_in_pr`.
- **Per-Phase Approval Gate held** through PR-029 (20-consecutive-PR streak). v0.3 preserves the streak.
- **PR-036 is the version-cut ritual** per `docs/VERSIONING.md §2`. It lands last, after all v0.3 implementation PRs are merged on `dev`. The `v0.3.0` git tag lands at PR-036's merge commit.
- **No prerequisite real-dataset work**: unlike v0.2 (surfaced by the first real-dataset run), v0.3 is driven by static phantom audit. The crypto-h3 dataset will be re-used as the testbed for PR-031's behavior-change verification.
- **Re-using existing infrastructure**: every PR completes existing surfaces rather than adding new abstractions. The "reuse over reinvent" constraint applies — PR-033 reuses `ParquetDataIter` (PR-006) and `select_ingest` (PR-006/D3); PR-034 reuses Optuna's stock `FileSystemArtifactStore`; PR-031 reuses `make_splits` (PR-024).
