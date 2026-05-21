# PR-032: `rux-ml registry score` CLI verb + holdout scorer

**Landed-in:** (not yet landed)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

**This PR is Tier-2** by inheritance from PR-031 (holdout semantics) per memory `feedback_tier_inheritance`. Architectural sub-decision D2 (verb shape + receipt schema + Booster shim approach) is **resolved by this PR's Phase 1 state assessment + Phase 3 web research (2026-05-21)** per the option-1 plan locked 2026-05-20. Verdict: CLI verb with Kedro `tracking.MetricsDataSet`-style 4-element flow + MLflow-style metrics/artifacts receipt body + rux-ml-specific provenance wrapper.

This PR triggers a MINOR bump (v0.3.0) per `docs/VERSIONING.md §1` — new CLI verb + new public scorer module + new file format.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

## Research findings

### Phase 1 — State Assessment (2026-05-21)

**Current state** (HEAD = `145e358`, post-PR-031 merge on `dev`):

- `training/metrics.py:112-114` — `compute_score(metric, trainer, x_eval, y_eval) -> float` takes a `Trainer`-typed object. `_rmse`/`_mae` call `trainer.predict`; `_auc`/`_logloss` call `_predict_proba(trainer, ...)`. **Loaded bundles have raw `xgb.Booster`, not `Trainer` — needs a shim.**
- `registry/bundle.py:55` — `load_bundle(dir) -> (Pipeline, xgb.Booster, ModelManifest)` returns the raw booster.
- `registry/promote.py:97` — stale comment confirmed: *"held out for future golden-regression evaluation (PR-014)."* The "future" never materialized.
- `registry/manifest.py` — `ModelManifest.promoted_from` has `study + trial_number + metric_value`; **no `entropy_hex`** on the manifest. For `time_ordered` splits this doesn't matter (deterministic); for `random` must re-open Optuna DB via `runs.load_run`.
- `cli/registry.py` — existing verbs (`promote`, `list`, `rollback`) use `--problem` / `--study` / `--trial` / `--to` flag convention; wrap errors as `typer.BadParameter`.
- `tests/golden/test_xgb_baseline.py:381-382` — in-repo precedent for `xgb.DMatrix(x, enable_categorical=True) + booster.predict(dmatrix)`. Will reuse for the shim.
- `receipts/` directory does NOT exist; `.gitignore` doesn't reference it. PR-032 creates both.

**Drift assessment**: `cli/registry.py` + `registry/*` last touched PR-010 (4fd07ba); `training/metrics.py` last touched PR-006 (7947ca6); PR-031 didn't touch any. **Zero drift since 2026-05-20 audit.**

**Stale assumptions from stub**:
1. "v0.3 design session locks D2" — per option-1, no separate design session. D2 resolved by this Phase 1 + Phase 3.
2. Stub Q4 (`compute_score` compatibility) — corrected at Phase 1: **Booster shim** is the right path. Reusable for PR-033's ExtMem adapter.
3. Stub Q5 (`entropy_hex` reconstruction) — clarified: `time_ordered` doesn't need it (deterministic); `random` requires `runs.load_run` → `attrs.entropy_hex` → `make_seed_bag_from_hex`.
4. Stub Q6 (Receipts directory) — locked at Phase 1: `receipts/` tracked; `.gitignore` excludes `*.parquet`.
5. Stub Q1 (CLI verb signature) — locked at Phase 1: matches existing `cli/registry.py` flag convention.

**New constraints**:
1. Booster shim must use `xgb.DMatrix(x, enable_categorical=True)` (matches existing golden test).
2. Random-split path requires an Optuna DB read at score time.
3. Manifest doesn't carry `entropy_hex` — for `random` split, scorer re-opens trial via `runs.load_run`.
4. Receipts directory creates from zero; no precedent to inherit (PR-031 receipt at `prs/PR-031-baseline-receipt.json` is per-PR paper trail, different shape).

**Prior-art audit**: `cli/registry.py` + `registry/*` created at PR-010; `training/metrics.py` created at PR-006; both stable through PR-031 era. Known-good patterns: `_load_cfg(ctx)` for config resolution; `read_champion + version_dir + load_bundle` for the round-trip; `xgb.DMatrix + booster.predict` for raw-booster scoring; `typer.BadParameter` for CLI errors; atomic JSON writes via tmp + `os.replace`.

**Group D probes (Phase 1)**:
- Probe 1: all identifiers exist at HEAD. ✓
- Probe 2: the combination "load_bundle + reconstruct splits + Booster shim + compute_score on `splits["test"]`" was novel — carried forward to Phase 3.
- Probe 3: N/A (no state-registration surface).

**Exit decision**: 5 stub sub-decisions resolved at Phase 1; 2 Phase 3 web-research questions; 1 Phase 4 Amend candidate likely (receipt schema may diverge from convention). Proceed to Phase 2.

### Phase 2 — Scope the Research (2026-05-21)

**Must-answer**:

1. **Q1 — Score-on-holdout CLI verb convention in production ML workbenches** (MLflow / W&B / AutoGluon / Kedro / ZenML / HF evaluate / Optuna). Verb signature + output format. Success: ≥2 cited production systems with verb signature + receipt format, OR explicit "ad-hoc / no standard" finding.
2. **Q2 — Receipt JSON schema convention**. Field names, types, nesting across cited systems. Recommend a schema for rux-ml.

**Dependencies**: Q1 + Q2 share a single Phase 3 dispatch.

**Explicitly excluded** (resolved at Phase 1 or trivially-locked):
- Q3 (predictions parquet layout): `(time_column, y_true, y_pred)` lean — trivial.
- Q4 (`compute_score` compatibility): Booster shim — resolved.
- Q5 (`entropy_hex` reconstruction): per-split-kind strategy — resolved.
- Q6 (Receipts directory): tracked; `.parquet` gitignored — resolved.
- CLI verb signature: matches existing convention — resolved.
- Nested CV / per-fold holdout: out of v0.3 scope.

**Phase 3 is mandatory** (not skippable per the documented exception): Q1 + Q2 lock a long-lived contract (the receipt schema will be consumed by future tooling); Phase 1 Probe 2 only found an in-repo "convention-by-extension" cite.

### Phase 3 — Findings (2026-05-21)

#### Q1: Score-on-holdout CLI verb convention

**Options considered:**

- **Option A — First-class CLI verb scoring saved model + persisting receipt**: Only **Kedro** at starter tag `0.19.14` fits. `kedro run` executes a node returning a metrics dict bound (via `outputs="metrics"`) to a `tracking.MetricsDataSet` catalog entry writing `data/09_tracking/metrics.json`. Bundle loaded from `data/06_models/regressor.pickle` (`versioned: true`).
- **Option B — Python API, no CLI verb**: Majority pattern. MLflow `mlflow.models.evaluate(...)` → `EvaluationResult.save(path)`; AutoGluon `TabularPredictor.evaluate(data) → dict`; HF `Evaluator.compute(...) → dict`; ZenML `@step`; W&B no score-on-holdout CLI.
- **Option C — Ad-hoc / no standard**: MLModelScope paper, AWS ML Lens BP03 documents this as the dominant reality.

**Disconfirming evidence sought**: zero reputable defenders of Option B for production grade workflows. Kedro is the only Option-A precedent.

**Recommendation**: **Option A** — `rux-ml registry score` is the right shape.
- **Status**: **convention** for the 4-element flow at Kedro `0.19.14`; **best-guess-given-constraints** for rux-ml-specific provenance wrapper.
- **Risks accepted**: verb-name `score` may confuse MLflow users (their `predict` = predictions only; `evaluate` = Python only). Documented in CONVENTIONS.md.

#### Q2: Receipt JSON schema

| System | Body | Provenance wrapper? |
|---|---|---|
| MLflow `EvaluationResult.save` | `metrics.json` = `{metric: scalar}` + `artifacts_metadata.json` = `{name: {uri, content_type}}` | NO (tracker server) |
| Kedro `tracking.MetricsDataSet` | flat `{r2_score, mae, max_error}` | NO (Kedro session ID + catalog versions) |
| AutoGluon | `dict[str, float]` (in-memory; no persist) | n/a |
| HF evaluate | dict (no persist) | n/a |
| ZenML | dict → artifact via MLMD | NO (MLMD) |

**Convergence**: body = `{metric: scalar}` (every persistor). **Divergence**: nobody bundles provenance inline; pushed to tracker / MLMD / catalog versions.

**Recommendation** (Phase 4 Amend candidates):
1. Flatten `score` → `metrics: {<name>: <float>, ...}` (MLflow + Kedro convention; allows multi-metric without schema migration).
2. Rename `preds_path` → `artifacts: {predictions: {path, content_type}}` (MLflow `artifacts_metadata.json` shape; future-extensible).

Provenance wrapper stays inline (`bundle_version`, `manifest_git_sha`, `promoted_from`, `holdout`, `scored_at*`) — labeled `best-guess-given-constraints`; defensible per AWS ML Lens BP03 for no-tracker-server environments.

#### Group D MCP Verification (2026-05-21)

| Probe | Verdict |
|---|---|
| Schema-Integrity | **Verified**: Kedro `tracking.MetricsDataSet` @ `0.19.14`; MLflow `EvaluationResult.metrics` @ master; in-repo `xgb.DMatrix(x, enable_categorical=True) + booster.predict` @ `tests/golden/test_xgb_baseline.py:381-382`; rux-ml `load_bundle`, `read_champion`, `make_splits` all at HEAD `145e358` |
| Synthesis-Verification | **Proven** for 4-element flow: Kedro starter `0.19.14` (`nodes.py:44-55` + `pipeline.py` `outputs="metrics"` + `catalog.yml` `tracking.MetricsDataSet → data/09_tracking/metrics.json`) — combination required (the receipt can't be produced without all 4 elements) |
| Binding-at-creation | N/A — receipts are filesystem outputs |

### Phase 4 — Synthesis (2026-05-21)

**Outcome**: **Amend → Apply** (3 amendments approved by user at Outcome Branch).

**Stub corrections applied in implementation commit**:
1. Receipt body: `score: {...}` → `metrics: {<name>: <float>}` dict (MLflow + Kedro convention)
2. Receipt artifacts: `preds_path: <str>` → `artifacts: {predictions: {path, content_type}}` (MLflow shape)
3. `docs/CONVENTIONS.md` adds verb-naming note (`score` ≠ MLflow `predict`/`evaluate`)

**Changes to docs/ARCHITECTURE.md**: Storage table gains "Holdout score receipts" row pointing at `receipts/holdout_score_<problem>_<date>.json` + `.parquet` shape.

**Changes to docs/CONVENTIONS.md**: new "Holdout score receipts (per PR-032)" subsection — receipt schema (metrics + artifacts dicts + provenance wrapper); verb-naming note; per-split-kind reconstruction strategy; Booster shim contract.

**Changes to docs/0.3/{ROADMAP, RESEARCH-BACKLOG, DESIGN-log}.md**: PR-032 row → `[x]`; status updates; new design-log session entry recording D2 + Phase 3 findings.

**Changes to CHANGELOG.md**: `[Unreleased] ### Added` (verb + receipt module) + `### Fixed` (stale `promote.py:97` comment cleanup).

**New PRs that must come first**: none.

### Phase 5 — Gate Check (2026-05-21)

- Premise still valid: ✓
- No prerequisite PRs surfaced: ✓
- D2 resolved at Phase 1 + Phase 3; 3 Phase 4 amendments locked at user approval
- User approved updated spec: ✓ (2026-05-21)
- Implementation cleared: ✓ (2026-05-21)

---

## Scope

New CLI verb + scorer module + receipts directory.

### Item 1 — `src/rux_ml/registry/scorer.py`

- `_BoosterTrainerShim`: wraps raw `xgb.Booster` exposing Trainer Protocol (`predict` + `predict_proba`; `fit` raises NotImplementedError). Uses `xgb.DMatrix(x, enable_categorical=True)` per in-repo precedent.
- `HoldoutScoreReceipt`: Pydantic-validated schema. Body shape: `metrics: dict[str, float]` + `artifacts: dict[str, _Artifact]`. Provenance wrapper: `bundle_version`, `bundle_dir`, `manifest_git_sha`, `promoted_from`, `holdout`, `scored_at`, `scored_at_git_sha`.
- `score_bundle_on_holdout(cfg, *, problem, version=None, output_dir=Path("receipts"))`: orchestrates the full flow.
- `_reconstruct_split_seed(cfg, study, trial_number)`: deterministic for `time_ordered`; `runs.load_run → entropy_hex → make_seed_bag_from_hex` for `random`.
- NOT exported from `rux_ml.registry`'s public `__init__.py` (preserves PR-010 sub-decision B1 — strict inference-deps separation).

### Item 2 — `src/rux_ml/cli/registry.py` `score` verb

Between `list` and `rollback` (read verb between read and write):

```
rux-ml registry score \
    --problem <name> \
    [--version <version>] \
    [--output <dir>]
```

### Item 3 — `src/rux_ml/registry/promote.py:97` stale comment cleanup

Old: `"held out for future golden-regression evaluation (PR-014)"`
New: `"consumed by rux-ml registry score (PR-032)"` + note about PR-031 holdout from HPO.

### Item 4 — `receipts/.gitignore`

```
*.parquet
!.gitignore
```

### Item 5 — Tests

`tests/registry/test_scorer.py` (7 tests):
- `test_booster_trainer_shim_predict_matches_direct_dmatrix`
- `test_booster_trainer_shim_predict_proba_returns_2d_for_binary`
- `test_booster_trainer_shim_fit_raises_not_implemented`
- `test_score_bundle_on_holdout_writes_receipt_and_parquet`
- `test_score_bundle_on_holdout_defaults_to_champion`
- `test_score_bundle_on_holdout_random_split_reconstructs_seed`
- `test_score_bundle_on_holdout_missing_champion_raises`

`tests/cli/test_registry_subcommands.py` (2 tests):
- `test_registry_score_writes_receipt_and_parquet`
- `test_registry_score_missing_champion_raises_bad_parameter`

### Item 6 — Docs propagation

- `docs/ARCHITECTURE.md` Storage table: new "Holdout score receipts" row
- `docs/CONVENTIONS.md`: new "Holdout score receipts (per PR-032)" subsection
- `docs/0.3/DESIGN-log.md`: new session entry
- `docs/0.3/ROADMAP.md`: PR-032 row flipped `[ ]` → `[x]`
- `docs/0.3/RESEARCH-BACKLOG.md`: PR-032 row updated
- `CHANGELOG.md`: `[Unreleased]` gains `### Added` + `### Fixed`

### Out of scope

- Multi-bundle scoring sweep (one bundle at a time)
- Per-asset / per-time-period breakdown (point-estimate score only at v0.3)
- ExtMem path activation (PR-033)
- Optuna Artifacts Store wiring (PR-034)

## Dependencies

- **PR-030** + **PR-031** merged
- D2 resolved by this PR's Phase 1 + Phase 3 (no separate design session per option-1)

## Architecture section implemented

`docs/ARCHITECTURE.md` Storage section (new "Holdout score receipts" row); `docs/CONVENTIONS.md` new subsection. Closes phantom #1b from the 2026-05-20 audit (test fold now has a consumer).

## Verification criteria

- [x] `_BoosterTrainerShim` in `src/rux_ml/registry/scorer.py` conforms to Trainer Protocol
- [x] `score_bundle_on_holdout` end-to-end test passes on synthetic bundle
- [x] `rux-ml registry score --problem <p>` writes receipt JSON + preds parquet under `receipts/`
- [x] Receipt JSON validates against `HoldoutScoreReceipt` Pydantic schema
- [x] `compute_score` returns sensible AUC on the synthetic bundle via the shim
- [x] `promote.py:97` stale comment replaced
- [x] `receipts/.gitignore` excludes `*.parquet`; JSON tracked
- [x] `docs/ARCHITECTURE.md` + `docs/CONVENTIONS.md` + `docs/0.3/{ROADMAP, RESEARCH-BACKLOG, DESIGN-log}.md` + `CHANGELOG.md` updated in same commit
- [x] `docs/0.3/ROADMAP.md` PR-032 row at `[x]`
- [x] PR-032 stub `## Research findings` populated with Phase 1-5 output
- [x] `uv run pytest` — 381 passed (was 372 after PR-031), 1 skipped, 15 deselected
- [x] `uv run ruff check .` clean
- [x] `uv run basedpyright src/` clean

## Research backing

Tier-2 — D2 resolved by Phase 1 + Phase 3 web research. Convention status backed by:
- Kedro `tracking.MetricsDataSet` @ starter tag `0.19.14`
- MLflow `mlflow.models.evaluate.EvaluationResult.save` @ master
- AWS Well-Architected ML Lens BP03 (defensible provenance-bundle in no-tracker-server environments)
- In-repo precedent: `tests/golden/test_xgb_baseline.py:381` (DMatrix + enable_categorical=True + booster.predict)

## Notes

- **Triggers MINOR bump** (v0.3.0) per `docs/VERSIONING.md §1`
- **Semantically paired with PR-031**: the score is only "held out" because PR-031 made the test fold unseen by HPO
- Per memory `feedback_roadmap_flip_in_pr`: PR-032 row flipped in same commit; streak preserved
- Booster shim is reusable for PR-033's native-API ExtMem adapter
- `_carve_substrate` (from PR-031) and `_reconstruct_split_seed` (this PR) are symmetric: substrate carving for HPO; test-fold reconstruction for scoring — both use `make_splits` so row identities agree
