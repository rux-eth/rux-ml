# PR-009: Run logging

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PR** (research-backed at design time).

## Research findings

### State Assessment (2026-05-16, LOCAL-ONLY)

**Current state of the codebase**:

- `dev` at `2cdaf97` (PR-008 merged). Working tree clean.
- `src/rux_ml/runs/` has `__init__.py` + `provenance.py` (PR-007 extracted: `HASH_LAYERS`, `build_user_attrs`, `data_hashes`, `ensure_storage_parent`, `study_name`). **No `attrs.py` / `ask_tell.py` / `query.py` yet.**
- `src/rux_ml/cli/runs.py` has 3 stub subcommands (`list`, `show`, `compare`) calling `not_implemented(..., "PR-009")`.
- `cli/train.py` (PR-006) calls `study.ask()` + `for k, v in build_user_attrs(...).items(): trial.set_user_attr(k, v)` + `trial.set_user_attr("metric", …)` + `study.tell(trial, score)`. PR-009 refactors this into `with one_off_run(...) as trial:`.
- `tuning/objective.py` (PR-007) does the same pattern inside `build_objective`'s closure. PR-009 refactors to use the canonical `TrialAttrs.record(trial)` path.
- `build_user_attrs(cfg, data_hashes_)` (PR-007) returns a `dict[str, str]` with **8 per-layer `*_cfg_hash` fields** (data/features/training/tuning/runs/registry/memory/cv) + `root_cfg_hash` + `git_sha` + `data_hash`/`data_bytes_hash`/`data_logical_hash`. 13 fields total; `metric` and `best_iteration` written separately.
- `docs/CONSTRAINTS.md` "Per-Trial Provenance Triple" lists the required user_attrs but enumerates 5 layer hashes (vs the 8 the codebase actually writes); the project already exceeds the documented minimum.

**Local API verification (Optuna 4.8 installed)**:

| Item | Status | Where verified |
|---|---|---|
| `study.ask()` + `study.tell(trial, value)` | **STILL CURRENT** | PR-006 + PR-007 exercise this |
| `trial.set_user_attr(k, v)` + `trial.user_attrs` | **STILL CURRENT** | round-tripped in PR-007 tests |
| `study.trials_dataframe()` includes `user_attrs_<key>` columns | **STILL CURRENT** (verified 2026-05-16) | `study.trials_dataframe().columns` shows `user_attrs_test_attr` |
| `optuna.load_study(name, storage).trials` | **STILL CURRENT** | `cli/tune.py::status` exercises this |
| `optuna.get_all_study_summaries(storage=...)` | **STILL CURRENT** | PR-006/007 tests exercise this |

**Stale assumptions**: None substantive. PR-009 spec is internally consistent with the PR-007 scaffolding it builds on.

**New constraints learned from PR-006 / PR-007 / PR-008**:

1. **`build_user_attrs` exists** (PR-007). PR-009 IS its formalization — replace with `TrialAttrs.from_cfg(...).record(trial)`. 2 callers to migrate (`cli/train.py` + `tuning/objective.py`).
2. **`metric` + `best_iteration`** are currently written separately by `cli/train.py` and `objective.py`. PR-009's TrialAttrs absorbs them so recording is atomic.
3. **PR-008's subprocess child writes user_attrs via the same `build_user_attrs`** (inside `build_objective`). PR-009's refactor must keep working through subprocess isolation — the new `TrialAttrs.record` is called from the same place.
4. **8 layer hashes today vs 5 listed in `CONSTRAINTS.md` "Per-Trial Provenance Triple"** — codebase already exceeds documented minimum. PR-009's `TrialAttrs` reflects actual reality (8 layers); the CONSTRAINTS.md text stays as documented minimum.
5. **Trial-ID semantics for the CLI** — PR-007's `tune retry-trial` already uses `trial.number` scoped to a study. PR-009's `runs show/compare` adopt the same convention with required `--study` flag.

### Synthesis (Phase 4, 2026-05-16)

**Outcome: Confirm** — D7 + D11 design research stands. Tier-1 classification stands (all sub-decisions are engineering judgment / spec-driven / project-convention / internal-precedent; none require external research). PR-007 already laid the scaffolding; PR-009 finishes the formalization.

**User-approved sub-decisions (2026-05-16):**

- **A1** — Replace `build_user_attrs` with `TrialAttrs.from_cfg(...).record(trial)`. Single canonical write path; refactor 2 callers.
- **B1** — Include `metric` (required) and `best_iteration` (Optional) on the `TrialAttrs` schema. Atomic per-trial record.
- **C1** — Implement `with one_off_run(study_name, cfg, …) as trial:` and refactor `cli/train.py` to use it. Drops the manual `study.ask` / `study.tell` from `train.py`.
- **D1** — Polars DataFrames for `list_runs` + `compare_runs`; Pydantic-validated `Run` rich object for `load_run`. Matches PR-005's Polars-first convention; pipeable per spec.
- **E1** — `trial.number` scoped to `--study NAME` (required for `runs show/compare`). Matches PR-007's `retry-trial` convention.

**Changes to PR-009 spec from research**:

- `TrialAttrs` schema reflects the actual 8-layer write set (vs the 5 enumerated in CONSTRAINTS.md). CONSTRAINTS.md text stays as documented minimum.
- Required-now fields = what PR-006/007/015 currently write: 8 layer hashes + root_cfg_hash + git_sha + data_hash + data_bytes_hash + data_logical_hash + metric. Optional fields = future-PR fields (entropy_hex / image_digest / library versions / peak_rss_mb / best_iteration).
- `build_user_attrs` removed; 2 callers migrate to `TrialAttrs.from_cfg(...).record(trial)`.
- `runs show/compare` require `--study NAME` (E1) — convention matches `tune retry-trial`.

**Changes to `docs/ARCHITECTURE.md`** (same commit): "Runs" component row updated to mention `TrialAttrs` schema validation.

**Changes to `docs/CONSTRAINTS.md`**: None (the "Per-Trial Provenance Triple" minimum stands; codebase reality exceeds it).

**Changes to `docs/CONVENTIONS.md`**: None — "Per-trial user_attr keys" subsection already lists the 8 hashes (PR-015 added `cv_cfg_hash`).

**No new prerequisite PRs surfaced.** PR-010 (registry promotion) consumes `TrialAttrs.read(trial)` — promotion logic can safely assume the schema validates.

### Gate Check (Phase 5, 2026-05-16)

- Premise still valid: ✓ (D7 + D11 design research stands)
- No prerequisite PRs surfaced: ✓
- User approved Tier-1 classification: ✓ (2026-05-16)
- User approved locked-in spec (A1, B1, C1, D1, E1): ✓ (2026-05-16)
- Implementation cleared: ✓ (2026-05-16)

---

## Scope

Codify the per-trial `user_attrs` schema, formalize the one-off-run-as-1-trial helper, and implement the `runs` CLI verb group per D7 + D11 + `docs/CONSTRAINTS.md`.

- `src/rux_ml/runs/attrs.py`:
  - `class TrialAttrs(BaseModel)` enumerating every required `user_attrs` field per `docs/CONSTRAINTS.md` ("Per-Trial Provenance Triple")
  - `record(trial, attrs: TrialAttrs)` writing all fields atomically
  - `read(trial) -> TrialAttrs` validating on read; raise on missing required fields
- `src/rux_ml/runs/ask_tell.py`:
  - `with one_off_run(study_name, base_cfg) as trial:` context manager that wraps a non-sweep training as a 1-trial study via `study.ask()` + `study.tell()`
  - Refactor `cli.train.run()` (PR-006) to use this helper so sweep + one-off paths share the same provenance recorder
- `src/rux_ml/runs/query.py`:
  - `list_runs(study_name=None, problem=None) -> pl.DataFrame` over `trials_dataframe()`
  - `load_run(trial_id) -> Run` rich object exposing params, metrics, attrs, artifact list
  - `compare_runs(*trial_ids) -> pl.DataFrame` side-by-side params + metrics
- CLI bodies (`src/rux_ml/cli/runs.py`):
  - `runs list [--study X] [--problem Y]`
  - `runs show <trial-id>` (formatted; includes provenance triple)
  - `runs compare <trial-id> <trial-id> ...`
- Tests:
  - Unit: `TrialAttrs` validation round-trips; missing required field on read raises
  - Integration: a tune run from PR-007/008 has all required `user_attrs`; `runs list` returns it; `runs show` formats it; `runs compare` produces a wide DataFrame
  - One-off: `rux-ml train` writes the same provenance set as a sweep trial would

NOT in scope: registry promotion (PR-010), peak_rss (PR-011), entropy_hex (PR-013) — but `TrialAttrs` schema includes those fields with `Optional` for now; PRs 011/013 fill them in.

## Dependencies

PR-008.

## Architecture section implemented

`docs/ARCHITECTURE.md` → "Runs" component row, "Storage" (Optuna study row), "Reproducibility Architecture" point 7 list.

## Verification criteria

- [ ] `TrialAttrs` model includes every key listed in `docs/CONSTRAINTS.md` "Per-Trial Provenance Triple"
- [ ] Reading a trial that's missing any *required-now* attr raises a typed error (later-PR fields are `Optional` until their PR lands)
- [ ] `runs list` shows trials across studies
- [ ] `runs show <id>` displays params + metrics + provenance triple in human-readable form
- [ ] `runs compare A B C` returns a Polars DataFrame with params/metrics side-by-side
- [ ] `with one_off_run(...)` produces a 1-trial study indistinguishable in shape from a sweep trial

## Research backing

Tier 1:

- D7: [Optuna artifacts tutorial](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/012_artifact_tutorial.html), [Optuna `ask`/`tell` API](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html), [Optuna user_attrs](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/003_attributes.html)
- D11: [Optuna CLI](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/004_cli.html)

State assessment must verify the `study.ask()` + `study.tell()` API is unchanged and that `trials_dataframe()` still includes `user_attrs` columns.

## Notes

- Pydantic-validated provenance schema means PR-010's promotion code can safely assume the manifest fields exist — promotion will REJECT a trial whose attrs don't validate, enforcing reproducibility per `docs/CONSTRAINTS.md`.
- `runs compare` output should be a Polars DataFrame, not a styled terminal table, so it's pipeable to other commands (`| jc`, `| fzf`, etc.).
