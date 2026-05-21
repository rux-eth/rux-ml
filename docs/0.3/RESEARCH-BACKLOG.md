# Research Backlog — v0.3

Index of v0.3 PRs with research status. Every PR runs `PROCEDURE-pr-research.md` before implementation — this document tracks the state of each PR's research and flags drift risk per the time-decay policy.

The v0.2 research backlog is frozen at [`docs/0.2/RESEARCH-BACKLOG.md`](../0.2/RESEARCH-BACKLOG.md) — all rows are `implementation-cleared`. v0.1 backlog at [`docs/0.1/RESEARCH-BACKLOG.md`](../0.1/RESEARCH-BACKLOG.md). v0.0 backlog at [`docs/0.0/RESEARCH-BACKLOG.md`](../0.0/RESEARCH-BACKLOG.md).

**Tiers:**
- **Tier 1** — Design-time research exists. Phase 1 (State Assessment) required before implementation; Phases 2-4 may be light if no drift found.
- **Tier 2** — Design-time research is partial or absent. Full 5-phase procedure required before the PR is written in final form.

**Status legend:**
- `design-research ✓` — research done at design time
- `design-research ~` — partial design research (pattern locked, per-instance details open)
- `design-research ✗` — no design-time research
- `state-assessed YYYY-MM-DD` — Phase 1 of `PROCEDURE-pr-research.md` completed
- `fully-researched YYYY-MM-DD` — all 5 phases of `PROCEDURE-pr-research.md` completed
- `implementation-cleared YYYY-MM-DD` — Phase 5 Gate Check passed

---

## Tier 1 — Implementation-Ready

| PR | Title | Design research | Required research topics |
|----|-------|-----------------|--------------------------|
| [PR-030](../../prs/PR-030-v0-3-sprint-scaffolding.md) | v0.3 sprint scaffolding | `design-research ✓` — no architecture decisions; pure docs + PR stubs creation; pattern inherited from PR-022/PR-026 sprint-bookkeeping precedent. | — |
| [PR-035](../../prs/PR-035-tune-retry-trial-test.md) | `tune retry-trial` integration test | `design-research ✓` + `state-assessed 2026-05-20` + `implementation-cleared 2026-05-20` — test-only PR; pattern inherited from existing `test_tune_subcommands.py` test functions (start/resume/status). Phase 1 confirmed `cli/tune.py:206-231` body intact (line range corrected from stub's 206-228); D4 verdict test-only; retry verb is state-agnostic (no failed-trial simulation needed); 3 tests per distinct code path. | — (all required research complete) |

## Tier 2 — Research-Pending

| PR | Title | Design research | Required research topics |
|----|-------|-----------------|--------------------------|
| [PR-031](../../prs/PR-031-hpo-honors-holdout-fold.md) | HPO objective honors holdout fold | `state-assessed 2026-05-20` + `fully-researched 2026-05-21` + `implementation-cleared 2026-05-21` — D1 holdout-semantics resolved at Phase 1 (truly held out); Phase 3 web research backed Option A (substrate-shrink) as **convention** with AutoGluon `tabular-essentials.html` at SHA `f8c428cbbef3bc319ff3f7710f5900e65637f4c4` + sklearn user guide §3.1 + Optuna issue #2184; disconfirming search returned zero defenders of Option B (full-df HPO); Group D Probe 2 verified the exact 3-element synthesis. | — (all required research complete; Q2 empirical receipt deferred to workbench run alongside implementation) |
| [PR-032](../../prs/PR-032-registry-score-cli-verb.md) | `rux-ml registry score` CLI verb + scorer module | `design-research ~` — pattern locked by existing `cli/registry.py` verbs (promote, list, rollback) + `load_bundle` API; scorer-module shape open. | (1) CLI verb signature (positional vs flag args; default version = champion); (2) Receipt JSON schema (mirrors `prs/PR-025-calibration-results.json` precedent? new shape?); (3) Predictions parquet column layout (timestamp + asset + y_true + y_pred? or wider with manifest fields?); (4) Whether `compute_score` works for raw booster + DMatrix (currently assumes Trainer wrapper); (5) Whether the test fold reconstruction needs the trial's `entropy_hex` (no for `time_ordered`, possibly for `random`). |
| [PR-033](../../prs/PR-033-extmem-path-activation.md) | ExtMem path activation | `design-research ~` — D3/PR-006 originally researched the dispatch decision; the trainer-wire-up shape is open. Locked by v0.3 design session sub-decision D2. | (1) Native `xgb.train` adapter design (wrapping Booster in Trainer-Protocol-compatible shim); (2) `ParquetDataIter` chunking strategy from single-file source (split into N batches by row count? Polars `streaming` mode?); (3) Default `cache_host_ratio` (XGBoost auto-estimate vs explicit fraction; survey what xgboost docs recommend); (4) `use_native` semantics — opt-in only, or auto-flip when ExtMem dispatch fires?; (5) Test-mode flag for forcing the path on small data; (6) `compute_score` compatibility — does the metric registry need a Booster-aware path?; (7) Per-fold ExtMem rebuild cost in HPO (probably prohibitive — flag if so). |
| [PR-034](../../prs/PR-034-artifacts-store-integration.md) | Optuna Artifacts Store integration | `design-research ~` — locked by v0.3 design session sub-decision D3 (lean: diagnostic-only; preserves D8/PR-010 A1). | (1) Optuna `FileSystemArtifactStore` API surface (current docs); (2) When to upload per trial (in objective post-fit, or in trial_runner post-optimize?); (3) Whether to record artifact IDs in `TrialAttrs` (new field?) or query Optuna by trial number; (4) Cleanup policy — do we ever delete old artifacts?; (5) Test strategy (round-trip upload + download with assertion). |
| [PR-036](../../prs/PR-036-v0-3-0-cut.md) | v0.3.0 version cut | `design-research ✓` — pattern locked by PR-021 (v0.1.0 cut) + PR-026 (v0.2.0 cut). | (1) Phase 1 verifies `scripts/rewrite_doc_refs.py` PATH_REWRITES still has correct entries for `docs/0.2/* → docs/0.3/*` (pattern was updated at PR-026 cut-time); (2) CHANGELOG `[Unreleased]` rolls into `[0.3.0]` with PR-028 + PR-029 (procedural) plus PR-031..PR-035 (feature) entries; (3) Version-tag sequence (no retroactive v0.2.x tags needed). |

---

## Drift Watch

Per the time-decay policy in `PROCEDURE-pr-research.md`, any PR marked `fully-researched` or `state-assessed` more than the project's staleness threshold before implementation begins must re-run Phase 1 (State Assessment).

**Project staleness threshold:** **60 days** (recorded in [`docs/CONSTRAINTS.md`](../CONSTRAINTS.md); BEST-GUESS, user-acknowledged).

**Currently watching:** the v0.3 sprint was scoped 2026-05-20 from a phantom audit. Each PR's `fully-researched` date must complete by **2026-07-19** for direct use, or be revalidated. The v0.2 design-session staleness clock (2026-05-19 + 60 days = 2026-07-18) is INDEPENDENT — PR-023 D1/D2 architectural decisions still in force as v0.3 starts.

### Per-PR drift-risk notes

- **PR-031** — `tuning/objective.py:259-261` was last touched at the v0.2 sprint (PR-023 / PR-025); re-verify at Phase 1 that `df_full = materialize(load_parquet(source_path))` is still the load pattern and that no subsequent PR shifted the substrate semantics.
- **PR-032** — `registry/promote.py:97` carries the stale "future golden-regression evaluation (PR-014)" comment — Phase 1 re-confirms this comment is still present in `dev` HEAD before scope-listing the comment removal.
- **PR-033** — `xgboost` version pinned in `uv.lock`; verify at Phase 1 that the `xgb.train` + `ExtMemQuantileDMatrix` + `ParquetDataIter` APIs at the locked version match the v0.3 implementation assumptions. ExtMem path is XGBoost-version-sensitive (the cache_host_ratio kwarg landed in xgboost 2.x; if uv.lock is on an older minor, recheck).
- **PR-034** — Optuna `FileSystemArtifactStore` API surface check; the artifact-store module was renamed once between Optuna 3.x versions. Phase 1 verifies the current install's actual import path.
- **PR-035** — Phase 1 verifies `cli/tune.py:206-228` retry body has not drifted since the audit on 2026-05-20.

### Untracked-but-watched ambiguities (from v0.3 design session, to be added)

To be populated after the v0.3 design session writes [`DESIGN-log.md`](DESIGN-log.md).

---

## Design References

- [`docs/0.3/DESIGN-log.md`](DESIGN-log.md) — v0.3 design session (placeholder until session runs)
- [`docs/0.3/ROADMAP.md`](ROADMAP.md) — v0.3 PR plan
- [`docs/0.2/DESIGN-log.md`](../0.2/DESIGN-log.md) — v0.2 design history (D1–D7); decisions remain in force unless v0.3 DESIGN-log explicitly supersedes
- [`docs/0.1/DESIGN-log.md`](../0.1/DESIGN-log.md) — v0.1 design history (Q1–Q6)
- [`docs/0.0/DESIGN-log.md`](../0.0/DESIGN-log.md) — v0.0 design history (D1–D17)
- [`docs/VERSIONING.md`](../VERSIONING.md) — versioning policy + bump rules + changelog format (flat, meta-rule)
- [`docs/CONSTRAINTS.md`](../CONSTRAINTS.md) — hard rules (the "No Phantom Implementations" constraint is what motivated the v0.3 sprint)
