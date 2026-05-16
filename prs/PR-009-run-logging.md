# PR-009: Run logging

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PR** (research-backed at design time).

## Research findings

_To be populated by `PROCEDURE-pr-research.md`._

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
