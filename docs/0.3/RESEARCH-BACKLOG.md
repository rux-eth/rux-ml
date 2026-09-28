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
| [PR-032](../../prs/PR-032-registry-score-cli-verb.md) | `rux-ml registry score` CLI verb + scorer module | `state-assessed 2026-05-21` + `fully-researched 2026-05-21` + `implementation-cleared 2026-05-21` — Phase 3 web research located Kedro `tracking.MetricsDataSet` at starter tag `0.19.14` (4-element CLI → versioned-bundle-load → score-on-holdout → JSON receipt) + MLflow `EvaluationResult.metrics`/`artifacts_metadata.json` (receipt body shape). Phase 4 outcome Amend → Apply: 3 schema/doc amendments locked at user approval (metrics dict; artifacts dict; verb-naming note). Booster shim resolved Phase 1; entropy_hex reconstruction per split_kind resolved Phase 1. | — (all required research complete) |
| [PR-033](../../prs/PR-033-extmem-path-activation.md) | ExtMem path activation | `state-assessed 2026-05-21` + `fully-researched 2026-05-21` + `implementation-cleared 2026-05-21` — Phase 1 surfaced scope reduction (opt-in `use_native` only; `registry/promote.py` + HPO objective unchanged; `_check_extmem_compat` preserved). Phase 3 located each ingredient independently (XGBoost 3.2 `external_memory.html` + `external_memory.py` demo + `python_api.html`) but Group D Probe 2 returned **NO cite** for the 4-element synthesis (`xgb.train` + `ExtMemQuantileDMatrix` + `cache_host_ratio` + sklearn-Trainer-Protocol wrapper); status downgraded to `best-guess-given-constraints`. Phase 4 outcome Amend → Apply: 4 amendments locked at user approval (BGGC label; mandatory numeric-agreement test; parallel adapter classes vs inheritance refactor; CONVENTIONS honest activation warning). Mitigation: `test_native_adapter_predict_proba_matches_sklearn_wrapper_within_tol` (atol=1e-5) bounds the BGGC risk. | — (all required research complete) |
| [PR-034](../../prs/PR-034-artifacts-store-integration.md) | Optuna Artifacts Store integration | `state-assessed 2026-05-21` + `fully-researched 2026-05-22` + `implementation-cleared 2026-05-22` — Phase 1 surfaced disk-space ceiling (3.31 GB/trial × 100 trials = 331 GB) and per-PR-D3 resolution (option-1 plan, 2026-05-20). Phase 3 located each ingredient independently (Optuna 4.8 tag `v4.8.0` SHA `689c62d`; MLflow @ v2.22.4 commit `ee89741`; Kedro @ kedro-datasets-5.1.0 commit `ab64c20`; Optuna 4.8 tutorial + `optuna-examples` pytorch_checkpoint + dashboard/hitl) but Group D Probe 2 returned **NO cite** for the 4-element synthesis (`FileSystemArtifactStore(per-study path)` + `upload_artifact in-objective` + `flat metrics.json` + `get_all_artifact_meta retrieval`); status downgraded to `best-guess-given-constraints`. Phase 4 outcome Amend → Apply: 5 amendments locked at user approval (per-study sub-directory; split schema; no `TrialAttrs.artifact_ids` field; no auto-cleanup; keyword-only `upload_artifact`). Mitigation: `test_artifact_upload_roundtrip_via_get_all_artifact_meta` bounds the BGGC risk. | — (all required research complete) |
| [PR-036](../../prs/PR-036-v0-3-0-cut.md) | v0.3.0 version cut | `design-research ✓` — pattern locked by PR-021 (v0.1.0 cut) + PR-026 (v0.2.0 cut). | (1) Phase 1 verifies `scripts/rewrite_doc_refs.py` PATH_REWRITES still has correct entries for `docs/0.2/* → docs/0.3/*` (pattern was updated at PR-026 cut-time); (2) CHANGELOG `[Unreleased]` rolls into `[0.3.0]` with PR-028 + PR-029 (procedural) plus PR-031..PR-035 (feature) entries; (3) Version-tag sequence (no retroactive v0.2.x tags needed). |
| [PR-040](../../prs/PR-040-oracle-quarantine-training-set-refusal.md) | Oracle quarantine: the training-set builder refuses oracle-derived inputs (post-v0.3.0, standalone, no version bump) | `state-assessed 2026-09-24` + `fully-researched 2026-09-24` + `implementation-cleared 2026-09-24`. Tier-2 by operator ruling (the stub proposed Tier-1). Phase 1 found 15 stale assumptions: `snapshot()` is not on the training path; the refusal lives in `load_parquet` + `compute_data_hash` + a tune preflight. Phase 3 proved the mechanics at polars 1.40.1 / CPython 3.12 / pydantic 2.13.4. Tag exclusion enforced at read is **convention**; column-prefix refusal and the combination are **best-guess-given-constraints**. Phase 4 outcome: Amend → Apply (symlink-target ancestor check). D17 elision widened to output-neutral guard config. | — (all required research complete) |
| [PR-041](../../prs/PR-041-m9-problem-configs.md) | M9 problem configs: one label per problem, the others as diagnostics (R-14 track; program v0.3 C9) | `state-assessed 2026-09-26` + light Phases 2–4 (Tier-1 proposed; contract ruled by program D37 / D45 #3). Resumed from an uncommitted track; two defects fixed test-first (`brier` untranslated in LightGBM / CatBoost; nulls counted as values in diagnostics). Gate: track-autonomous, merge operator-gated. | — (Q1–Q5 to the program lead: training root, schema-file feature list, label names + markout null semantics, MAE vs objective) |
| [PR-042](../../prs/PR-042-symbol-holdout-split.md) | Symbol-holdout one-off split kind + the embargoed temporal split (R-14 track; program v0.3 D41, C9) | `state-assessed 2026-09-26` + light Phases 2–4 (Tier-1 proposed). Stable per-coin hash assignment (not the per-trial seed) so prefixes and targets share the holdout; `[m9] h_max_ms` fail-closed embargo floor. Gate: track-autonomous, merge operator-gated. | — (Q6: the program's h_max / fill-horizon grid; Q7: C9 says the time-block regime runs 'via the panel purged splitter' while its run schedule counts 36 one-off fits) |
| [PR-043](../../prs/PR-043-label-as-feature-refusal.md) | Label quarantine: the `y__`-as-feature refusal (R-14 track; program v0.3 D38 #3, C6 / C13; extends PR-040) | `state-assessed 2026-09-26` + light Phases 2–4 (Tier-1 proposed). Probe: before this PR a `y__` feature trained with exit 0. One check on the feature spec at every build, PR-040's error family and exit-2 mapping reused. Gate: track-autonomous, merge operator-gated. | — (Q9: the real-file C6 demonstration is the program's, after PR-024) |
| [PR-044](../../prs/PR-044-m9-honesty-metric.md) | M9 honesty metric, scoring rules and the fail-closed signed-key reader (R-14 track; program v0.3 D41, D45 #3, C9) | `state-assessed 2026-09-26` + light Phases 2–4 (Tier-1 proposed). The program's real `gates.yaml` verifies today and lacks the four keys → M9 training refuses until the operator signs them in (correct per C9). Over-deduction aggregation defined here (net relative bias), flagged. Gate: track-autonomous, merge operator-gated. | — (Q11–Q15 to the lead) |
| [PR-045](../../prs/PR-045-m9-leakage-tests.md) | Leakage audit + the three M9 regime overlays (R-14 track; program v0.3 D41, C9) | `state-assessed 2026-09-26` + light Phases 2–4 (Tier-1 proposed). Independent audit (binary search / set intersection) recorded per fit, enforced for `[m9]` regimes that promise separation. Gate: track-autonomous, merge operator-gated. | — (the real-set run is the program's, after PR-024) |
| [PR-046](../../prs/PR-046-m9-learning-curve.md) | The nested-prefix learning curve + D43 verdict (R-14 track; program v0.3 D43, C9) | `state-assessed 2026-09-26` + light Phases 2–4 (Tier-1 proposed). Prefixes of the train window with a fixed OOS set; the followed scalar is config (BEST-GUESS `score`). Gate: track-autonomous, merge operator-gated. | — (Q17, Q18 to the lead) |
| [PR-047](../../prs/PR-047-training-set-hash-bridge.md) | Training-set data-hash bridge: sidecar + equality test (R-14 track; program PR-024 A13, C6 / C11; covers PR-040 successor (ii) for the bridge) | `state-assessed 2026-09-26` + light Phases 2–4 (Tier-1 proposed; added by the lead mid-track on the operator's A13 'go'). Opened files read from Polars itself. Gate: track-autonomous, merge operator-gated. | — (Q19: the harness manifest schema; Q20: hive keys) |
| [PR-050](../../prs/PR-050-promote-refit-data-hash-equality.md) | Promote refuses a re-fit whose `data_hash` differs from the trial's (R-14 track; PR-040 successor (i); program D45 #6, C11) | `state-assessed 2026-09-27` + light Phases 2–4 (Tier-1 proposed; delegated by the program lead). Probe: before this PR a trial promoted by re-fitting on other data exited 0. Gate: track-autonomous, merge operator-gated. | — |

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
