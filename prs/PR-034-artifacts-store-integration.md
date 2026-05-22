# PR-034: Optuna Artifacts Store integration (diagnostic-only)

**Landed-in:** (Unreleased — will be bundled into v0.3.0 by PR-036)

## Before Implementation (NON-NEGOTIABLE)

This PR was implemented after `PROCEDURE-pr-research.md` completed all 5 phases on 2026-05-21 → 2026-05-22. Findings appended to `## Research findings` below.

**Tier-2 PR** — research-pending at scaffold time; the architectural sub-decision D3 (artifacts: what to upload) was resolved via this PR's own Phase 1 state assessment per the option-1 plan (2026-05-20). Lean confirmed: **diagnostic-only**. The alternative (per-trial bundles replace re-fit at promote) reverses D8/PR-010 sub-decision A1 and stays out of scope.

This PR triggers a MINOR bump (v0.3.0) per `docs/VERSIONING.md §1` — new per-trial artifact upload behavior.

## Research findings

### Phase 1 — State Assessment (2026-05-21)

**Current state** (at `dev` HEAD `b693870`, post-PR-033 merge):

- Optuna pinned at `4.8.0`; `from optuna.artifacts import FileSystemArtifactStore, upload_artifact, download_artifact` imports cleanly.
- `RunsConfig.artifacts_root` field exists with default `Path("studies/artifacts")`. Field is path-elided in `config/root.py:60` (excluded from config hashing).
- No `runs/artifacts.py` precursor — entirely unimplemented.
- `TrialAttrs` schema (`runs/attrs.py:40-154`) has NO `artifact_ids` field.
- `tuning/objective.py::_fold_scores` (post-PR-031 + PR-033) returns `list[float]`. Natural upload-injection point: after `_fold_scores` returns and before `return statistics.fmean(scores)` (line 344). Nests cleanly after `_record_attrs` (line 340).
- `cli/train.py::run_command` (post-PR-033 native adapter branch): natural upload-injection point post-`_fit_and_score` and pre-`run.tell(score)` (line 215+).
- `registry/promote.py` UNCHANGED since PR-010 (D8/A1 re-fit preserved).
- Test baseline: 390 passed + 1 skipped (= 391 collected).

**Assumptions at PR draft time** (re-verified):
1. Optuna `FileSystemArtifactStore` API ✓
2. `artifacts_root` field exists + path-elided ✓
3. No prior artifact-store code to refactor ✓
4. `TrialAttrs` schema stable ✓
5. Watchdog/Pruned exception flow preserved ✓
6. D8/PR-010 A1 honored ✓

**Stale assumptions**: none.

**New constraints** (surfaced by Phase 1):
- **Disk-space severity** — the stub's "per-fold predictions CSV" Notes warning is exact: crypto-h3 baseline = 20.7M rows × 10 folds × 2 floats × 8 bytes = **3.31 GB/trial**; a 100-trial HPO sweep = **331 GB**. The PR's "preds CSV" lean is infeasible at v0.3 scale. **Phase 4 must downscale.**

### Phase 2 — Research scope (5 must-answer questions, 2026-05-21)

| # | Question | Dependency |
|---|---|---|
| Q1 | Exact Optuna 4.8 `FileSystemArtifactStore` + `upload_artifact` + `download_artifact` workflow + on-disk layout | — |
| Q2 | Query-by-trial-number pattern (avoid `TrialAttrs.artifact_ids` field) | depends on Q1 |
| Q3 | When to upload (in-objective post-fit vs post-`study.optimize`) | independent |
| Q4 | Per-trial diagnostic JSON schema convention | independent |
| Q5 | Cleanup/retention policy at v0.3 | independent |

**Explicitly excluded** (deferred): per-trial bundle uploads (D8/A1 violation), plot/PNG generation (Medium tier), sub-sampled predictions CSV (future Tier-2 PR with sampling policy), auto-cleanup TTL, solver-trial artifacts.

### Phase 3 — Findings (2026-05-21 web research via 4 parallel agents)

**Q1 — Optuna 4.8 artifacts API.** Status: **proven**. Tag `v4.8.0`, SHA `689c62dbfc14f3476c51c14fda2e4f818ed2ee30`.

- `upload_artifact(*, artifact_store, file_path, study_or_trial, storage=None, mimetype=None, encoding=None) → str` — **keyword-only** (positional API deprecated at 4.0, removed at 6.0 via `convert_positional_args` decorator at `optuna/artifacts/_upload.py` L53-58).
- `download_artifact(*, artifact_store, file_path, artifact_id) → None` — addressed by `artifact_id` only; raises `FileExistsError` if `file_path` already exists.
- `FileSystemArtifactStore(base_path: str | Path)` — **flat** on-disk layout: one file per artifact at `<base_path>/<uuid4>`, no per-study/per-trial nesting. Original filename preserved only in `ArtifactMeta.filename`.
- `upload_artifact` returns the new `uuid4` `artifact_id` AND auto-persists the `ArtifactMeta` JSON into `trial.system_attrs` under prefix `"artifacts:"` (`_upload.py` L22, L104-109). **The caller does not need to record `artifact_id` themselves.**

**Q2 — Retrieval-by-trial-number.** Status: **proven**. Use `optuna.artifacts.get_all_artifact_meta(trial, *, storage=storage) → list[ArtifactMeta]` (`_list_artifact_meta.py` L16-18). Caveat from the docstring (L27-29): _"Optuna does not provide the API that stores the used artifact store information, so please manage the information in the user side."_ — `rux-ml` must record the artifact-store **base_path** (one entry per study), but NOT per-trial artifact_ids.

**Q3 — When to upload.** Status: **convention** (one-org caveat: 3 cited examples are all Optuna-maintained). All cited working examples put `upload_artifact` **inside the objective, post-fit, no try/except guard**:
- Optuna 4.8 tutorial `tutorial/20_recipes/012_artifact_tutorial.py` (canonical)
- `optuna-examples/pytorch/pytorch_checkpoint.py` @ HEAD `7dded62`
- `optuna-examples/dashboard/hitl/main.py` @ file SHA `e39085e`

Zero cited post-`study.optimize` uploads using `upload_artifact`. Disconfirming-evidence search returned no warnings against in-objective uploads.

**Q4 — Diagnostic JSON schema.** Status: **convention** for the top-level shape (`Dict[str, float]`); **BGGC** for the per-fold mixed-type sidecar (no exact precedent surveyed).

- MLflow `EvaluationResult` @ tag `v2.22.4` (commit `ee89741`) `mlflow/models/evaluation/base.py` L636-637: `json.dump(self.metrics, fp, cls=NumpyEncoder)`; `metrics: Dict[str, float]`.
- Kedro `MetricsDataset` @ tag `kedro-datasets-5.1.0` (commit `ab64c20`) `kedro_datasets/tracking/metrics_dataset.py` L52: `def save(self, data: dict[str, float])`.
- ZenML `RunMetadataRequest.values: Dict[str, MetadataType]` @ tag `0.85.0` (commit `cb407b2`) — flat top-level keys.
- W&B `Summary` (`wandb/sdk/wandb_summary.py` @ tag `v0.21.4`) and Neptune `Handler` (`src/neptune/handler.py` @ tag `1.9.1`) both admit hierarchical access but flat-key idiom is supported and dominant in metrics persistence.

**3 of 5 systems persist flat `Dict[str, float]`** — convention threshold met for the top-level shape. The `fold_<i>_<metric>` key prefix idiom and the `metric_mean`/`metric_std`/`n_folds` summary keys are unattested in the survey → BGGC for those naming choices. The mixed-type per-fold sidecar (`row_count`/`fit_seconds`/`peak_rss_mb`/`timestamp`) is novel to this workbench → BGGC.

**Drift flag (informational only — do not retroactively rewrite per `feedback_pr_spec_historicity`)**: `kedro-datasets` removed the `tracking/` module in 9.x. PR-032's "Kedro `tracking.MetricsDataSet` @ starter tag `0.19.14`" anchor pointed at the core `kedro` repo (correct), but the underlying dataset class has been deprecated upstream. PR-034 uses MLflow `EvaluationResult` @ v2.22.4 as the primary anchor.

**Q5 — Cleanup policy.** Status: **proven** (Optuna FAQ explicitly prescribes the lean) **+ convention** (MLflow + ZenML manual-only).

- Optuna 4.8 FAQ ("How can I delete all the artifacts…"): *"create a new directory or bucket for each study so that all the artifacts linked to a study can be entirely removed by deleting the directory or the bucket … it is hard to officially support the delete feature and they are not planning to support this feature in the future."*
- `optuna/artifacts/_filesystem.py` @ v4.8.0 — only methods: `open_reader`, `write`, `remove(artifact_id)`. No TTL, no GC, no scan/expire API.
- MLflow: manual `mlflow gc` CLI (lifecycle-policy FR open at `mlflow/mlflow#18300`).
- ZenML: manual `zenml artifact prune` (no built-in TTL).
- W&B: per-artifact TTL via SDK — but managed-server feature, not a filesystem-store contract.

**This raises Optuna FAQ's per-study-directory pattern to the load-bearing convention** — switches `make_artifact_store` from a single global path to `<artifacts_root>/<study_name>/` so cleanup is `rm -rf studies/artifacts/<study>/`.

### Group D — MCP Verification (2026-05-22)

**Probe 1 (Schema-Integrity)** — every recommended identifier verified in installed `optuna==4.8.0`:

| Identifier | Surface | Status |
|---|---|---|
| `FileSystemArtifactStore(base_path: str \| Path)` | `optuna.artifacts` | ✓ |
| `upload_artifact(*, artifact_store, file_path, study_or_trial, storage=None, mimetype=None, encoding=None) → str` | keyword-only confirmed | ✓ |
| `download_artifact(*, artifact_store, file_path, artifact_id) → None` | keyword-only confirmed | ✓ |
| `get_all_artifact_meta(trial, *, storage=None) → list[ArtifactMeta]` | confirmed | ✓ |
| `ArtifactMeta` fields: `artifact_id, filename, mimetype, encoding` | confirmed | ✓ |
| `ARTIFACTS_ATTR_PREFIX = "artifacts:"` (system_attrs) | confirmed | ✓ |

**Probe 2 (Synthesis-Verification)** — the 4-element combination (`FileSystemArtifactStore(per-study path) + upload_artifact in-objective + flat metrics.json + get_all_artifact_meta retrieval`): each ingredient independently cited but **NO single working example combines all 4 elements together**. Status: **`best-guess-given-constraints`** (same probe-2 outcome as PR-033 — synthesis is the workbench's own composition).

**Probe 3 (Binding-at-creation)** — empirically verified live with `optuna.storages.RDBStorage(sqlite://...)` + 1-trial study:
- After `upload_artifact(...)`: `trial.system_attrs["artifacts:<uuid4>"]` contains `{"artifact_id", "filename", "mimetype", "encoding"}` JSON.
- `get_all_artifact_meta(trial, storage=storage)` returns the populated list.
- On-disk layout: flat — `<base_path>/<uuid4>`, no per-trial subdir.

### Phase 4 — Synthesis (Outcome: Amend → Apply, 2026-05-22)

5 amendments to the Phase 1 working hypothesis, locked at user approval:

1. **Per-study sub-directory layout** (was: single global `artifacts_root`). `make_artifact_store(cfg, *, study_name)` constructs `FileSystemArtifactStore(str(cfg.runs.artifacts_root / study_name))`. Aligns with Optuna 4.8 FAQ.
2. **Split schema** (was: single `summary_stats.json`): `metrics.json` (flat `Dict[str, float]`, convention) + `fold_meta.json` (mixed-type per-fold list, BGGC).
3. **No `TrialAttrs.artifact_ids` field** — Optuna auto-persists in `system_attrs`; retrieval via `get_all_artifact_meta(trial, storage=storage)`. Strict provenance schema stays clean.
4. **No auto-cleanup at v0.3** — Optuna FAQ confirms maintainers refuse to officially support delete; the lean is upstream-prescribed. Document manual `rm -rf studies/artifacts/<study>/` in CONVENTIONS.
5. **All `upload_artifact` calls use keyword-only args** — positional API deprecated at 4.0, removed at 6.0.

### Phase 5 — Gate Check (2026-05-22)

- Premise still valid: ✓ (D3 lean survives all 4 questions + Group D)
- No prerequisite PRs surfaced: ✓ (PR-030 + PR-033 dependencies already met)
- User approved updated spec: ✓ (2026-05-22)
- Implementation cleared

**BGGC items + mitigation** (parallel to PR-033's pattern):
- 4-element synthesis combination → bounded by mandatory `test_artifact_upload_roundtrip_via_get_all_artifact_meta` (upload → `get_all_artifact_meta` → `download_artifact` → assert JSON equality)
- `fold_meta.json` schema (mixed-type per-fold) → bounded by `test_fold_meta_json_required_keys` (structural test asserts required keys per fold)

## Scope (final, post Phase 1 reduction)

Wire up `optuna.artifacts.FileSystemArtifactStore` so per-trial diagnostic artifacts are actually stored. Completes phantom #3 from the 2026-05-20 audit.

### Item 1 — `make_artifact_store` + `upload_diagnostics` helpers (NEW `src/rux_ml/runs/artifacts.py`)

```python
def make_artifact_store(cfg: RuxMLConfig, *, study_name: str) -> FileSystemArtifactStore: ...
def upload_diagnostics(
    trial: optuna.Trial, artifact_store: FileSystemArtifactStore, *,
    metrics: dict[str, float], fold_meta: list[dict[str, Any]],
    tmp_dir: Path,
) -> tuple[str, str]: ...  # (metrics_artifact_id, fold_meta_artifact_id)
```

### Item 2 — Per-trial upload in `tuning/objective.py`

After `_fold_scores` returns, build the flat metrics dict + per-fold meta list and call `upload_diagnostics(trial, ...)` inside the objective closure. `_fold_scores` extended to track per-fold timing + row count.

### Item 3 — Per-trial upload in `cli/train.py`

Symmetric upload for the one-fit baseline (`n_folds=1` semantic).

### Item 4 — `cli/solve.py` skipped

Solver trials produce `(x_star, objective_value)`, not predictions. No diagnostic upload; existing `solver_status` + `objective_value` provenance fields are sufficient. Document the exception in CONVENTIONS.

### Item 5 — `rux-ml runs show` augmentation

Augment `cli/runs.py:show` to call `get_all_artifact_meta(trial, storage=storage)` and emit a `Diagnostic artifacts:` section listing `(filename, artifact_id)` pairs.

### Item 6 — Docs update

- `docs/ARCHITECTURE.md` Storage table: per-trial diagnostic artifacts row updates from aspiration → active. Data Flow notes upload step.
- `docs/CONVENTIONS.md`: new "Per-trial diagnostic artifacts (per PR-034)" subsection — schema split, in-objective convention, per-study base_path layout, manual cleanup, solver exception, BGGC label.
- `docs/0.3/{ROADMAP, RESEARCH-BACKLOG, DESIGN-log}.md` updated.
- `CHANGELOG.md [Unreleased] ### Added` loud entry.

### Out of scope (deferred per Phase 1 + Phase 2)

- Per-trial bundle uploads (D8/A1 violation).
- Plot/PNG generation (Medium tier — not v0.3).
- Sub-sampled predictions CSV (future Tier-2 PR with sampling policy).
- Auto-cleanup TTL.
- Solver-trial artifacts.

## Dependencies

- **PR-030** (sprint scaffolding) — merged.
- **PR-033** (substrate-shrink already complete; no new dependency surfaced) — merged.

## Architecture section implemented

`docs/ARCHITECTURE.md` Storage table (per-trial diagnostic artifacts row) + Data Flow (upload step) + Components table (`runs/artifacts.py` module).

## Verification criteria

- [x] `FileSystemArtifactStore` instantiated in production (`tuning/objective.py` + `cli/train.py` paths).
- [x] Per-study sub-directory layout (`<artifacts_root>/<study_name>/`) per Optuna FAQ.
- [x] Per-trial `metrics.json` (flat `Dict[str, float]`) + `fold_meta.json` (per-fold list) uploaded.
- [x] All `upload_artifact` calls use keyword-only args.
- [x] **Mandatory BGGC mitigation test passes** — round-trip via `get_all_artifact_meta` + `download_artifact` returns identical JSON.
- [x] `cli/runs.py:show` emits "Diagnostic artifacts:" section.
- [x] `registry/promote.py` UNCHANGED (D8/A1 preservation).
- [x] `TrialAttrs` schema unchanged (no `artifact_ids` field).
- [x] `docs/ARCHITECTURE.md` Storage table updated.
- [x] `docs/0.3/ROADMAP.md` PR-034 row `[ ] → [x]`.
- [x] `CHANGELOG.md [Unreleased] ### Added` loud entry.

## Research backing

Tier-2 — D3 resolved via this PR's own Phase 1 state assessment (option-1 plan, 2026-05-20). Phase 3 web research anchored on:

- Optuna v4.8.0 tag SHA `689c62dbfc14f3476c51c14fda2e4f818ed2ee30` — `optuna/artifacts/__init__.py`, `_upload.py`, `_download.py`, `_filesystem.py`, `_list_artifact_meta.py`.
- Optuna 4.8.0 tutorial `tutorial/20_recipes/012_artifact_tutorial.py`.
- `optuna-examples` `pytorch/pytorch_checkpoint.py` @ HEAD `7dded62`; `dashboard/hitl/main.py` @ SHA `e39085e`.
- MLflow `EvaluationResult` @ tag `v2.22.4` (commit `ee89741`).
- Kedro `MetricsDataset` @ tag `kedro-datasets-5.1.0` (commit `ab64c20`).
- ZenML `RunMetadataRequest` @ tag `0.85.0` (commit `cb407b2`).
- W&B `Summary` @ tag `v0.21.4`; Neptune `Handler` @ tag `1.9.1`.
- Optuna 4.8.0 FAQ; MLflow `mlflow/mlflow#18300`; ZenML artifact-prune docs; W&B TTL docs.

**Status**: `best-guess-given-constraints` for the 4-element combination (`FileSystemArtifactStore(per-study path)` + `upload_artifact in-objective` + `flat metrics.json` + `get_all_artifact_meta retrieval`) — Group D Probe 2 returned NO single cited working example. Mitigation: round-trip test.

## Notes

- **Triggers MINOR bump** (v0.3.0) per `docs/VERSIONING.md §1`.
- **D8/PR-010 A1 preserved**: this PR does NOT change `registry/promote.py`'s re-fit behavior. Artifacts are diagnostic only.
- Per memory `feedback_roadmap_flip_in_pr`: PR-034's implementation commit flips `docs/0.3/ROADMAP.md` PR-034 row `[ ]` → `[x]` in the same commit.
- **Disk-space concern from Phase 1 (3.3 GB/trial)** resolved by Amendment 2 (summary stats only — no per-row CSV). Sub-sampled predictions deferred to future Tier-2 PR.
- **PR-032 anchor drift** (Kedro `tracking/` removed in 9.x) noted informationally only; PR-032 stays historical per `feedback_pr_spec_historicity`. PR-034 uses MLflow `EvaluationResult` as primary anchor.
