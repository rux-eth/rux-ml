# PR-010: Model registry

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PR** (research-backed at design time).

## Research findings

### State Assessment (2026-05-16, LOCAL-ONLY)

**Current state of the codebase**:

- `dev` at `d6d7748` (PR-009 merged). Working tree clean.
- `src/rux_ml/registry/` **does not exist**. `src/rux_ml/cli/registry.py` has 3 stub subcommands (`promote`, `list`, `rollback`) calling `not_implemented(..., "PR-010")`.
- `RegistryConfig` (PR-002): `root: Path = Path("registry")`, `version_format: str = "v_{date}_{short_hash}"`.
- `runs.TrialAttrs.from_trial(frozen)` (PR-009) — the explicit promotion-validation hand-off; raises `pydantic.ValidationError` on missing required-now fields.
- `runs.load_run(storage_url, study_name, trial_number)` (PR-009) — returns a `Run` rich object with `params` + `attrs: TrialAttrs | None`.
- **`skops` is NOT in `pyproject.toml` deps yet** — PR-010 must add it.
- **`optuna.artifacts` is NOT used anywhere** — PR-007's `build_objective` does NOT upload `pipeline.skops` / `model.ubj` artifacts. The architecture-diagram `optuna.artifacts.upload_artifact(...)` was design-intent but unimplemented. PR-010 resolves this by re-fitting at promotion (sub-decision A1).

**Local API verification (already-installed libraries)**:

| Item | Status |
|---|---|
| XGBoost 3.2.0 `Booster.save_model("…ubj")` / `load_model(...)` | **STILL CURRENT** (exercised in PR-006) |
| `optuna.artifacts.FileSystemArtifactStore` + `upload_artifact` / `download_artifact` (Optuna 4.8) | available but **not used today** |
| `skops.io.dump` / `load` | **NEEDS INSTALL** |

**Stale assumption in PR-010 file + architecture diagram**:

The spec step 3 says "Download artifacts from Optuna (`pipeline.skops`, `model.ubj`)". Neither PR-007's objective nor PR-008's trial_runner uploads artifacts — there's nothing in the artifact store to download at promotion. **Resolved**: promote re-fits at promotion time (sub-decision A1) and the architecture diagram is updated to drop the `optuna.artifacts.upload_artifact` line from the trial body.

**New constraints learned from PR-006 / PR-007 / PR-008 / PR-009**:

1. **`TrialAttrs.from_trial(frozen)` is the promotion-validation hand-off** (PR-009 explicit). PR-010's promote refuses to promote when validation raises.
2. **`runs.load_run(storage, study, trial_number)` is the trial-fetch surface** (PR-009).
3. **Trial-ID convention** is `trial.number` scoped to `--study NAME` (PR-007 / PR-009 precedent).
4. **Re-fit path**: `build_objective` is per-trial-with-CV; promote needs a different code path that does a single full-train fit on the data + materialises the (Pipeline, Booster) pair. Reuse `make_features` / `make_trainer` / `make_splitter`'s underlying logic without the CV loop.
5. **Inference-deps separation** (D8): `src/rux_ml/registry/__init__.py` exports `load_model` and imports only `xgboost` / `skops` / `sklearn` / `polars` + registry sub-modules. NO transitive `rux_ml.tuning` / `rux_ml.data.cv` / `rux_ml.training` imports (those are training-stack). The test enforces this via subprocess + blocked `sys.modules`.

### Synthesis (Phase 4, 2026-05-16)

**Outcome: Confirm** — D8 design research stands. Tier-1 classification stands (sub-decision A is an implementation-strategy choice within D8's design space; the rest are bounded engineering judgment).

**User-approved sub-decisions (2026-05-16):**

- **A1** — Re-fit at promotion time. Self-contained PR-010; promote requires training stack (already in workbench deps); `load_model` stays light. Architecture diagram's `optuna.artifacts.upload_artifact(...)` line is dropped — replaced with "promote re-fits final model from cfg + trial.params on full train+val data."
- **B1** — Strict separation: `registry/__init__.py` imports only `xgboost` / `skops` / `sklearn` / `polars` + registry sub-modules. Inference-deps test enforces.
- **C1** — `v_{YYYY}_{MM}_{DD}_{short_hash}` with `short_hash = root_cfg_hash[:6]`. Sortable by date; deterministic given the same config+data.
- **D1** — `champion.json` schema: `{"version", "promoted_at", "promoted_from": {"study", "trial_number", "metric_value"}}`. Atomic rewrite via tmp + `os.replace`.
- **E1** — Tmp file in same dir + `os.replace` (POSIX-atomic rename). Standard pattern; matches v0's single-machine sequential model.

**Changes from research**:

- `skops` added to runtime deps.
- Architecture-diagram drift: the trial-body `optuna.artifacts.upload_artifact(...)` line is removed. Promote re-fits. Updated in the same commit.
- `library_versions` block on the manifest is populated at promote time via `importlib.metadata.version("…")` so the bundle records exactly what produced it.

**Changes to `docs/ARCHITECTURE.md`** (same commit): Data Flow "After study completes — promotion is an explicit step" section refreshed to reflect re-fit-at-promote.

**Changes to `docs/CONSTRAINTS.md`**: None.

**Changes to `docs/CONVENTIONS.md`**: None.

**No new prerequisite PRs surfaced.** PR-011 (memory & threading) and PR-013 (seed management) remain correctly sequenced downstream; PR-014 (golden regression tests) builds on the registry as the loadable-bundle surface.

### Gate Check (Phase 5, 2026-05-16)

- Premise still valid: ✓ (D8 design research stands)
- No prerequisite PRs surfaced: ✓
- User approved Tier-1 classification: ✓ (2026-05-16)
- User approved locked-in spec (A1, B1, C1, D1, E1): ✓ (2026-05-16)
- Implementation cleared: ✓ (2026-05-16)

---

## Scope

Implement the filesystem model registry per D8 + `docs/CONSTRAINTS.md` "Two-File Model Bundle". After this PR, a tuning run's best trial can be promoted to a stable, name-addressable model loadable from a thin client.

- `src/rux_ml/registry/manifest.py`:
  - `class ModelManifest(BaseModel)` with all fields from D8 spec (version, problem, optuna_study, optuna_trial_number, metric, hashes, version pins, feature_list_hash, created_at)
  - `read(path) -> ModelManifest` (validates on load; raises on schema mismatch)
  - `write(path, manifest)` (atomic via tmp + rename)
- `src/rux_ml/registry/bundle.py`:
  - `save_bundle(pipeline, booster, manifest, dest_dir)` writing `pipeline.skops`, `model.ubj`, `manifest.json`
  - `load_bundle(dir_path) -> tuple[Pipeline, xgb.Booster, ModelManifest]`
- `src/rux_ml/registry/promote.py`:
  - `promote(problem, study_name, trial_id) -> str` (returns version_id):
    1. Load trial via `runs.load_run`
    2. Validate `TrialAttrs` (must have all required provenance per PR-009)
    3. Download artifacts from Optuna (`pipeline.skops`, `model.ubj`)
    4. Compose manifest from trial attrs + library versions
    5. Compute version_id = `v_<YYYY>_<MM>_<DD>_<short_hash>` per D8 BEST-GUESS format
    6. Write bundle to `registry/<problem>/<version>/`
    7. Atomically rewrite `registry/<problem>/champion.json`
- `src/rux_ml/registry/paths.py` — version-id formatter, problem-dir resolver, champion-pointer helpers
- `src/rux_ml/registry/__init__.py` — exports `load_model(problem, version="champion") -> tuple[Pipeline, xgb.Booster]` per D8
- CLI bodies (`src/rux_ml/cli/registry.py`):
  - `registry promote --problem X --study S --trial T`
  - `registry list` — all problems with current champion + recent versions
  - `registry rollback --problem X --to <version>` (atomically rewrites champion.json to a prior version)
- Tests:
  - Unit: manifest round-trip; bundle write/read; champion.json atomic rewrite (assert no partial-write race)
  - Integration: promote a trial from PR-007's tune run, then `load_model("synth_problem")` returns both objects, `predict` works
  - **Inference-deps test:** in a stripped-down env that imports only `xgboost`, `skops`, `sklearn`, `polars`, `load_model` still works (per D8 design intent — inference doesn't depend on the training stack)
  - Rollback: promote v1, promote v2, rollback to v1 — `champion.json` correctly points back

NOT in scope: registry retention/garbage collection (out of scope for v0; manifest enables it later).

## Dependencies

PR-009.

## Architecture section implemented

`docs/ARCHITECTURE.md` → "Registry" component row, "Storage" (Model registry row), "Key Abstractions: Model bundle + manifest", "Data Flow" (the registry promotion section).

## Verification criteria

- [ ] `registry promote --problem X --study S --trial T` writes a complete bundle and rewrites `champion.json`
- [ ] Promoted trial MUST have all required `TrialAttrs` — promotion of a trial missing provenance fields fails loudly
- [ ] `load_model("X")` returns the champion bundle as `(Pipeline, Booster)` — `predict` works on a held-out sample
- [ ] `load_model("X", version="v_2026_05_14_a8f3c2")` loads a specific version
- [ ] `registry rollback --problem X --to <version>` atomically updates `champion.json`
- [ ] `champion.json` is never partially written (concurrent-read safety verified by a fault-injection test)
- [ ] Inference-deps test passes: a thin importer can `load_model` without importing the workbench training stack
- [ ] Manifest schema mismatch on read raises a typed `ManifestSchemaError`

## Research backing

Tier 1:

- D8: [XGBoost saving_model](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html), [skops persistence](https://skops.readthedocs.io/en/stable/persistence.html), [MLflow alias RFC #10336](https://github.com/mlflow/mlflow/issues/10336), [sklearn model persistence](https://scikit-learn.org/stable/model_persistence.html)

State assessment must verify:
- XGBoost `Booster.save_model("...ubj")` and `Booster.load_model(...)` are unchanged
- `skops.io.dump` / `load` API for sklearn Pipelines is current
- No new sklearn version-incompatibility warnings between bundle write and read

## Notes

- Champion is a JSON file, not a symlink (per D8 — cross-platform footguns ruled out symlinks).
- Version-string format is BEST-GUESS — change here if it doesn't survive first real use; document the change in `docs/CONVENTIONS.md`.
- Per D8, retention/garbage collection is intentionally deferred. Add a separate PR if disk pressure becomes real.
