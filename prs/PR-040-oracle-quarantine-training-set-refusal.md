# PR-040: Oracle quarantine — the training-set builder refuses oracle-derived inputs

**Landed-in:** (not yet landed)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PRs** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

**Proposed tier: Tier-1.** The behaviour is fully specified by an external criterion (below) and by a
reference implementation that already exists and is evidenced; nothing here is an open research
question. Phase 1 must still run — the 60-day staleness threshold in `docs/CONSTRAINTS.md` applies,
and this PR touches `data/versioning.py`, which PR-033/PR-034 moved around. **The tier is the
operator's call, not this file's.**

## Research findings

_To be populated by `PROCEDURE-pr-research.md`. Do not begin implementation until this section exists with completed findings from all required phases._

---

## Why this PR exists (context from outside this repo)

This is the **third and last owner** of an acceptance criterion in the `rux-capital/program` repo:
**ACCEPTANCE C13 — Oracle quarantine** (Stage-0/1 bake-off protocol, rule 1). C13 is verified at the
v0.2 cut against three owners, and **rux-ml's is the only one not built**:

| owner | obligation | state (2026-09-24) |
|---|---|---|
| `rumpy-harness` PR-018 | the `oracle__` namespace, the tagged store, the harness-side loader refusal | evidenced at close-out |
| `rumpy-harness` PR-019 | the promotable-flag refusal | demonstrated on real rows — `promote --trial 9` exited 2, naming three oracle-tagged trials |
| **this PR** | **the training-set-builder refusal** | **not started** |

**`rux-ml` currently has no concept of an oracle at all.** As of 2026-09-24 the only occurrence of
the string "oracle" anywhere in `src/`, `docs/`, `prs/` or `configs/` is `docs/0.2/DESIGN-log.md:71`,
an unrelated use of the phrase "correctness oracle" about splitter testing.

**Why it matters.** The harness builds perfect-foresight *oracle* labels to measure a ceiling — what
a strategy could earn if it knew the future. Those columns and stores exist so the v0.2 pilot can be
run at all. If any of it reaches a training set, the model learns the future and every metric
downstream of it is meaningless. The harness refuses to *serve* it; rux-ml must refuse to *ingest*
it. Two independent refusals, because one of them will eventually be wrong.

**This PR has no dependency on anything in flight in the program repo** and can run entirely in
parallel. It is the likeliest last thing standing at the v0.2 cut.

## The contract to mirror

The reference implementation is `rumpy_harness/oracle/store.py::production_load`:

```python
def production_load(path: Path, cfg: OracleConfig) -> pl.DataFrame:
    """The production/training-side reader (ACCEPTANCE C13's harness half): refuses a file under
    a tagged directory (any ancestor), and any table with a column in the oracle namespace."""
    for d in (path.parent, *path.parent.parents):
        if (d / cfg.tag_file).is_file():
            raise OracleQuarantineError(f"{path}: under the oracle store {d} — refused")
    df = pl.read_parquet(path)
    oracle_cols = [c for c in df.columns if c.startswith(cfg.namespace)]
    if oracle_cols:
        raise OracleQuarantineError(f"{path}: oracle columns {oracle_cols} — refused")
    return df
```

Two refusals, and the first one is the subtle half:

1. **Any path under a directory carrying the tag file, at _any_ ancestor level** — not only under a
   configured store root. A dataset copied, symlinked or snapshotted out of the store keeps its
   tagged parent somewhere above it, and checking only the configured root would miss it.
2. **Any table carrying a column in the oracle namespace.**

Harness values, to be read from config here rather than copied as literals
(`docs/CONSTRAINTS.md`: "All configurable values from config files"):

| thing | harness value | source |
|---|---|---|
| namespace prefix | `oracle__` | `config/harness.toml` `[oracle] namespace` |
| tag file name | `ORACLE.tag` | `config/harness.toml` `[oracle] tag_file` |

## Scope

**In scope.** The refusal, at the point a dataset enters the workbench, plus the tests that exercise
it end to end through the data layer.

Attachment points, from reading the repo on 2026-09-24 — Phase 1 confirms or replaces these:

- `src/rux_ml/data/versioning.py` — `snapshot()` / `write_manifest()`, the content-addressed ingest.
  Its `Manifest` already records `schema_` as field → dtype, so an `oracle__` column is visible at
  exactly the point the harness refuses one. **Most likely the right home**: everything entering the
  CAS passes through here, and a refusal recorded in the manifest is auditable afterwards.
- `src/rux_ml/data/loaders.py` — `load_parquet()` / `iter_parquet_files()`, the Polars scan path.
  Note `load_parquet` returns a **`LazyFrame`**: a column check that forces collection would change
  the layer's ingest semantics (D3 — Polars lazy is the default path). Phase 2 must decide whether
  the check reads the parquet **schema** without collecting, or sits at `materialize()`, or both.
- `src/rux_ml/config/data.py` — `DataConfig.source_path`, where the dataset path is named, and the
  natural home for the two config values above.

**Explicitly out of scope**, stated so the cut does not read them as omissions:

- No change to the harness side. Its half is built and evidenced; this PR mirrors it, it does not
  refactor it.
- No shared package between the repos. They are independently governed, and a common dependency
  would couple two release cycles to make two small functions identical. **Two independent
  implementations of one rule is the point** — a single shared one has a single point of failure.
- No retroactive scan of existing CAS contents. If that is wanted it is its own PR with its own
  decision about what to do with anything found.

## Dependencies

None inside rux-ml. `docs/0.3/ROADMAP.md` (PR-030…PR-036) is complete and PR-039 has landed; this
PR neither depends on nor blocks any of them.

Outside rux-ml it is a **cut blocker** for `rux-capital/program` v0.2 via ACCEPTANCE C13.

## Architecture section implemented

`docs/ARCHITECTURE.md` — the data layer's ingest path (D3 Polars lazy ingest, D9/D14 content-addressed
versioning). No new abstraction: this completes an existing surface, per the roadmap's
"reuse over reinvent" note.

## Verification criteria

Populated properly after research; these are the criteria the external one demands, so they are
recorded now rather than invented later.

- [ ] A parquet with a column named `oracle__<anything>` is **refused** at ingest, and the error
      names the offending columns.
- [ ] A parquet under a directory containing the tag file is **refused**, and the error names the
      tagged directory.
- [ ] The tag is honoured at **any ancestor depth**, not only at the configured root — a file nested
      several levels below a tagged directory is still refused.
- [ ] A clean dataset is unaffected: no change to the manifest, the hashes or the split behaviour of
      anything that does not carry oracle inputs. **This is the regression that matters** — a
      refusal that also changes clean-path hashes would invalidate every existing receipt.
- [ ] The namespace prefix and tag filename come from **config**, not literals in code.
- [ ] The refusal raises a **named, catchable** error type, not a bare `ValueError`.
- [ ] A test exercises the refusal **end to end through the data layer**, not only the helper
      (`docs/CONSTRAINTS.md`: no phantom implementations).
- [ ] `CHANGELOG.md` `[Unreleased]` entry per `docs/VERSIONING.md` §6.
- [ ] The `docs/0.x/ROADMAP.md` row flips in the same commit.

## Research backing

Tier-1 rationale: the behaviour is specified by ACCEPTANCE C13 and by an evidenced reference
implementation. Phase 1 must still establish:

1. Where ingest actually funnels today — `versioning.snapshot`, `loaders.load_parquet`, or both —
   after PR-033's `XGBoostNativeAdapter` / `single_source_iter` and PR-034's artifact-store work.
2. Whether a lazy `LazyFrame` path can be checked on **schema alone**, without forcing a collect and
   changing D3's ingest semantics.
3. Whether any existing dataset or manifest under `data/cas/` would newly refuse — i.e. whether this
   is inert on today's corpus or is itself a finding.

## Notes

**The operator's steps, if any, are none.** Nothing here touches signed files, spend, live hosts or
data deletes.

**Open at the time of writing (2026-09-24):** the tier (proposed Tier-1 above) is the operator's
call, and whether this opens a v0.4 sprint or lands standalone against `dev` — `docs/0.3/ROADMAP.md`
is complete, so there is no open version to flip a row in. PR-037 and PR-038 do not exist in `prs/`;
this file takes **PR-040** rather than reusing a gap, since a gap may be deliberate.
