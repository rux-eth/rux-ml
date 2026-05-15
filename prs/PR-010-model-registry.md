# PR-010: Model registry

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PR** (research-backed at design time).

## Research findings

_To be populated by `PROCEDURE-pr-research.md`._

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
