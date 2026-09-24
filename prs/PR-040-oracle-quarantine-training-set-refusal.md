# PR-040: Oracle quarantine — the training-set builder refuses oracle-derived inputs

**Landed-in:** (not yet landed)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PRs** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

**Tier: Tier-2 (operator ruling 2026-09-24, after Phase 1; the stub proposed Tier-1).** Original stub text: The behaviour is fully specified by an external criterion (below) and by a
reference implementation that already exists and is evidenced; nothing here is an open research
question. Phase 1 must still run — the 60-day staleness threshold in `docs/CONSTRAINTS.md` applies,
and this PR touches `data/versioning.py`, which PR-033/PR-034 moved around. **The tier is the
operator's call, not this file's.**

## Research findings

_To be populated by `PROCEDURE-pr-research.md`. Do not begin implementation until this section exists with completed findings from all required phases._

### State Assessment (2026-09-24)

Baselines: rux-ml `dev` @ 679b096, harness @ b9922ad, program @ 022792b. Every probe was read-only; scripts are in `/private/tmp/claude-501/pr040-scratch/`. The rux-ml probes ran on polars 1.40.1 (`uv.lock:1111-1112`), and the key behaviours reproduce the same on the harness's 1.44.2. Workbench checks were read-only over SSH.

**Current state**:
- **Five training-set builders share one read.** Each calls `materialize(load_parquet(<cfg.data.source_path>))`:
  - `cli/train.py:84`
  - `tuning/objective.py:314`, reached from in-process tune, the subprocess child via `_internal/trial_runner.py:128`, and retry-trial at `cli/tune.py:242`
  - `registry/promote.py:112`, the re-fit
  - `registry/scorer.py:263`
  - `scripts/calibrate_pr025.py:321`. The stub and five of six readers missed this one; it sets no `data_hash` attr (`:409-411`).
- **`load_parquet` and `materialize` cannot carry the refusal as they are.** `load_parquet(path)` is just `pl.scan_parquet(path)` and takes no config (`data/loaders.py:11-18`). `materialize(lf)` never sees a path (`loaders.py:21-33`).
- **Hashing is a second, independent read path.** `compute_data_hash` runs its own `pl.scan_parquet` over the `iter_parquet_files` list (`data/versioning.py:72-73`; `loaders.py:36-49`, sorted `rglob('*.parquet')`). It never calls `load_parquet`. It is reached through `data_hashes` (`runs/provenance.py:74-88`) at:
  - `train.py:194`, before `one_off_run` (`:197`) and before the load at `:84`
  - `objective.py:319`, after the load at `:314`
  - `promote.py:174`, after the re-fit

  `scorer.py` and the calibration script never hash.
- **`snapshot()` is not on the training path.** Its only caller is `cli/data.py:51` (`rux-ml data version`). Nothing reads the CAS or data manifests back (`README.md:224-225`: "never on the training path").
- **PR-033/PR-034 added no new ingest.** `ParquetDataIter` and `single_source_iter` only re-read temp chunks written from the already-featurized frame (`training/xgboost/native_adapter.py:151-161`; `data/data_iter.py:97`). There is no CSV, IPC or NDJSON reader anywhere in `src/`.
- **How a refusal would surface:**
  - `rux-ml train`: `one_off_run` marks the trial FAIL on any exception (`runs/ask_tell.py:78-85`). A refusal inside `load_parquet` therefore leaves a FAIL row with no user_attrs, and the full-data hash has already run.
  - Default subprocess tune (`config/tuning.py:62`): the parent prints "child exited with code 1; continuing to next trial" n times, **exits 0**, and records 0 trial rows. This was probed; the loop is `cli/tune.py:104-120`.
  - Registry verbs: they catch `(FileNotFoundError, KeyError, ValueError)` and raise `typer.BadParameter`, which exits 2 (`cli/registry.py:44-47,117-122`). train and tune have no catch.
- **Accidental refusals today** (unnamed Polars errors, not quarantine):
  - A directory source holding a **non-empty** non-parquet file (such as a harness `ORACLE.tag`) raises `InvalidOperationError` at `collect_schema()`. A **zero-byte** tag is skipped silently. All 156 real harness tags are non-empty JSON.
  - Heterogeneous multi-file sources raise `SchemaError` or `ColumnNotFoundError`, because polars defaults to `missing_columns`/`extra_columns='raise'`.
  - A single file under a tagged directory loads cleanly.
  - `compute_data_hash` and `snapshot()` of a tagged directory **succeed**: they hardlink only `*.parquet` into a flat `data/cas/<aa>/<hash>__<name>` with no tag reachable (`versioning.py:129-131,147-158`).
- **Config:**
  - `DataConfig` defaults are Python literals (`config/data.py:17-22`), and `configs/base.toml` `[data]` only describes values in comments (`:10-12`). CONSTRAINTS names the Pydantic models as the single source of truth (`docs/CONSTRAINTS.md:109`).
  - Both levels are `extra="forbid"` (`config/_strict_model.py:14-18`; `config/root.py:110-115`).
  - `_HASH_ELIDED_FIELDS` is at `root.py:55-65`; `cfg_hash` and `layer_cfg_hash` at `root.py:223-252`.
- **No exception hierarchy.** The only custom errors are `MemoryPressureError(Exception)` (`_internal/memory.py:29`) and `ManifestSchemaError(ValueError)` (`registry/manifest.py:22`). CONVENTIONS has no section on errors.
- **Harness reference.** `production_load` is at `src/rumpy_harness/oracle/store.py:79-89`, one commit (700a556, 2026-09-16), and matches the stub verbatim. It:
  - walks `(path.parent, *path.parent.parents)` lexically, with no `resolve`, `absolute` or `expanduser`
  - tests the tag with `is_file()`
  - reads eagerly with `pl.read_parquet`
  - matches column names with case-sensitive `startswith`
  - raises `OracleQuarantineError(RuntimeError)` (`:23`)

  Its only callers are tests; `lockbox/readers.py:56` mentions it in a string. Values: `config/harness.toml:143-146`, `namespace = "oracle__"`, `tag_file = "ORACLE.tag"`. `OracleConfig` fields are required, with no defaults and no validation (`config.py:340-348`).

**Assumptions at PR draft time**:
- Tier-1: "fully specified by an external criterion … and a reference implementation; nothing here is an open research question" (`PR-040:15-19`).
- PR-033/PR-034 "moved `data/versioning.py` around" (`:18`).
- `snapshot()` is "most likely the right home", for four reasons (`:92-95`):
  - everything entering the CAS passes through it
  - `Manifest.schema_` shows an `oracle__` column "at exactly the point the harness refuses one"
  - a refusal "recorded in the manifest is auditable afterwards"
- The lazy question is "schema without collecting, or at `materialize()`, or both" (`:96-99`).
- `DataConfig` is the natural home for the two values (`:100-101`).
- "A dataset copied, symlinked or snapshotted out of the store keeps its tagged parent somewhere above it" (`:72-74`).
- Clean path: "no change to the manifest, the hashes"; a hash move "would invalidate every existing receipt" (`:137-139`).
- "The `docs/0.x/ROADMAP.md` row flips in the same commit" (`:145`). No dependencies; can run in parallel (`:115-116`).
- Research-backing item 3 assumes a corpus under `data/cas/` (`:156-157`).
- Reference at `rumpy_harness/oracle/store.py` (`:54`).
- The only "oracle" string in rux-ml is `docs/0.2/DESIGN-log.md:71`. **Confirmed.**

**Stale assumptions**:
1. **PR-033/PR-034 did not touch `versioning.py` or `loaders.py`.**
   - `versioning.py`: `git log --follow` shows 2be39b3 (PR-004) and 251b7c5 (PR-005, lint only).
   - `loaders.py`: 2be39b3 only.
   - a13abb3 (PR-033) touched only `data_iter.py` and `data/__init__.py`. aa1f616 (PR-034) touched no `data/` file.
2. **`snapshot()` guards no training set.** All five builders bypass it (see Current state). A snapshot-only refusal is not a C13 refusal.
3. **`Manifest.schema_` does not show every oracle column.** It comes from `collect_schema()` over an explicit file list (`versioning.py:72-76`), which:
   - reflects the first file only (probed)
   - turns off hive discovery, so an `oracle__k=v/` partition key reaches the training frame through `load_parquet(dir)` but shows in no footer, no `schema_` and no `data_hash`. Probed: `oracle__flag=` and `flag=` trees hash identically.
4. **"Recorded in the manifest" cannot work.** A refused dataset writes no manifest (`versioning.py:160-171`). `Manifest` is `extra="forbid"` (`:35`), so any new field changes the schema.
5. **"Copied/symlinked/snapshotted keeps its tagged parent" is false** for:
   - single-file copies
   - hardlinks, including rux-ml's own CAS snapshot (same inode)
   - file symlinks under a lexical walk
   - symlinks into a subdirectory *below* the tagged day directory

   It is true only for a copy or symlink of the whole tagged day directory, where `is_file()` follows the link. *Corrected from the reader claim that directory symlinks always evade.*
6. **`materialize()` cannot host the tag check.** It receives no path. A path is recoverable only by introspecting the plan (`explain()` truncates multi-file lists; `serialize(format='json')` is deprecated), so it is not a sound home. *Corrected from "impossible".*
7. **The lazy/collect framing is not the real issue.** The real questions are *which* schema and *in what order* (see Q2).
8. **`DataConfig` is not a hash-neutral home.** Un-elided fields there move `data_cfg_hash` and `root_cfg_hash` for every config, which conflicts with the stub's own "no change to … the hashes" criterion if that covers config hashes.
9. **"Invalidate every existing receipt" overstates the risk.**
   - Nothing recomputes and compares a recorded data or config hash (grep finds no equality checks in `src/`).
   - The only real receipt (on the workbench, untracked) carries no hash field; it embeds `root_cfg_hash[:6]` in `bundle_version`.
   - The real stranding risk is a *required* schema field on `TrialAttrs` or `ModelManifest`.
10. **"No dependency / ROADMAP row flips" is doubtful.** No roadmap is open (`docs/0.3` is complete). A new config knob is **MINOR** under `docs/VERSIONING.md:16`, and a required TOML field is **MAJOR** (`:15`). Neither is the PATCH class every post-cut standalone PR used (PR-028/029/037/038/039, "no row to flip").
11. **There is no `data/cas/` corpus** on either host (see Q3).
12. **The reference path is missing `src/`.** It is `src/rumpy_harness/oracle/store.py:79-89`.
13. **"D3" is ambiguous.** The stub's D3 (Polars lazy ingest) is `docs/0.0/DESIGN-log.md:43-48`. `docs/0.3/DESIGN-log.md:57` D3 is the Artifacts store. Cite it as "v0 D3".
14. **The Tier-1 rationale does not meet the repo's tier definition** (see Tier lean).
15. **Incidental:** `configs/problems/crypto_breakout_h3.toml:16` says "Workbench-absolute path", but the value at `:17` is relative.

**New constraints**:
- **Where the refusal lives.** It must live in `load_parquet`, or a helper `load_parquet` calls, so all five builders are covered, including `scripts/calibrate_pr025.py`. Entry-point preflights alone are not enough.
- **Which files the tag walk covers.** It must enumerate the file set Polars actually reads, not `iter_parquet_files`. Polars follows symlinked subdirectories; Python 3.12 `rglob` does not.
  - Probe (critic/p2.py): a keys-only oracle row reached the training frame through a symlinked subdirectory. The per-file walk over `iter_parquet_files` found no tag, lexically or resolved.
  - The alternative is to refuse directory sources that contain symlinks.
- **Path normalization.** Walk a normalized **copy** of each path (`expanduser` + `resolve`), and include the source directory itself.
  - Polars expands `~`; the literal walk visits `['~', '.']` (probed).
  - A relative path's walk stops at `.`, and the only problem config is relative.
  - Keep the scan target and the manifest `source_path` string unchanged.
- **Ordering.** Run the tag walk before any Polars `collect_schema`/`collect`; a non-empty tag inside a directory source makes `collect_schema` itself raise.
- **Column check.** Use the directory `LazyFrame.collect_schema()` (catches hive keys) plus per-file footers over the Polars file set (gives a named refusal on heterogeneous sources). Check the **raw** source schema before feature selection or target drop, because `target_column` is taken from the raw frame (`cli/train.py:62-63`).
- **Raise-only.** Do not change `iter_parquet_files`, the `_logical_hash` scan, scan options (`hive_partitioning`, `extra_columns`, `missing_columns`) or the scan target. Changing the scan target to a file list would also turn off hive partitioning in training. Leave the pre-existing gap between what is hashed and what is loaded to its own PR.
- **Test fixtures.** Write a **non-empty** tag (Polars skips zero-byte files). Assert the named error type, never "any exception": today's accidental Polars errors would pass an any-exception test (phantom risk).
- **Config fields.** No `Field(exclude=True)` on the knobs; see the Field(exclude=True) note under Hash-regression. Validate `namespace` `min_length=1` (empty refuses everything) and `tag_file` `min_length=1` with no path separator (empty silently disables the tag check, since `Path(d)/"" == d`). Land schema field and TOML key in one commit (`extra="forbid"`).
- **Schema compatibility.** Any new `TrialAttrs`, `ModelManifest` or data-`Manifest` field must be Optional (precedent: `solving_cfg_hash`, `runs/attrs.py:98`). A required field makes all 24 workbench trials unpromotable and the champion bundle unloadable (`registry/manifest.py:55-101`).
- **Regression test.** Add a pinned-value clean-path test, since none exists today. Pinning a data hash is allowed: CONSTRAINTS forbids exact hashes only for GPU-trained model outputs (`docs/CONSTRAINTS.md:105`).
- **Real-store demonstration.** It must run bare-metal on the workbench or with an added mount. `docker-compose.yml` mounts only `./configs ./studies ./registry ./data ./logs`, and the oracle store is a sibling directory.

**Prior-art (things prior PRs already learned)**
- **PR-004 (2be39b3):**
  - Hash over an explicit file list and never scan a parent directory, because the old `_logical_hash` mixed in unrelated files (`prs/PR-004-data-layer.md:99-101`). This is also what turned off hive keys in hashing.
  - `collect_schema()` was locked in as the zero-materialization schema read (`PR-004:34,49,88`). New finding: it is first-file-only on multi-file sources.
  - Hardlink with an EXDEV → `copy2` fallback (`versioning.py:107-126`).
- **PR-005 (251b7c5):** the last touch on `versioning.py`, lint only.
- **PR-008 (5768b26):** added the subprocess tune loop's continue-on-nonzero branch (`cli/tune.py:112-117`). It has no test.
- **PR-014 (d4966bb):** golden_v1 fixture. `manifest.json:5-8` pins data hashes that still reproduce at HEAD. Its `root_cfg_hash` (cd75bc2c…) already differs from HEAD (c5ba8632…), and nothing asserts it.
- **PR-020 (00c9784):** top-level `solving: … | None = None` moved `root_cfg_hash` for every config, silently.
- **PR-022 (9c2e7bc):** `[data]` is not a discriminated union, so new common keys are safe in `base.toml` (`prs/PR-024:28`).
- **PR-024 (20854bb):**
  - Refusals that depend only on config go in a `RuxMLConfig` `model_validator` (`root.py:117-148`).
  - Adding `DataConfig.split_kind`/`time_column` moved `data_cfg_hash` silently. Removing them from the HEAD dump reproduces a pre-PR-024 recorded `data_cfg_hash` exactly (b2842c7b…, local `studies.db`, git_sha cdbffd2).
- **PR-027 (a3123e4):** kept Tier-2 for a single schema decision (`PR-027:57-58`). Its bug shipped because every fixture used the happy dtype, so PR-040 fixtures need multi-file, heterogeneous, hive, symlinked, relative and deep-tag cases.
- **PR-031 (d2831aa):** `data_hashes` hashes the full source, and a metric move forced a baseline-receipt refresh.
- **PR-032 (63e0f91):** `receipts/.gitignore` says "JSON receipts ARE tracked".
- **PR-033 (a13abb3):** `single_source_iter` works on post-feature temp files, so it is not an ingest point. Keep scope tight.
- **PR-034 (aa1f616):** no data-layer change. `runs.artifacts_root` elision is the precedent for excluding an output-neutral field from hashing; the elision list is itself a D17 BEST-GUESS item (`docs/0.0/DESIGN-log.md:185`).
- **PR-037 (49a3ee2), PR-038 (b46427f):** no `prs/` files (confirmed). They are the "no row to flip" PATCH-class precedent.
- **PR-039 (4fb462c):** established the pattern of CLI flags with TOML counterparts as `T | None = None` (`CONVENTIONS.md:157-168`). Its CHANGELOG entry landed under the released `[0.3.0] ### Fixed` (`CHANGELOG.md:29-30`), not `[Unreleased]` as its commit message claims. Flagging for the operator, not fixing.
- **Harness 700a556 / 842e142:** the `production_load` reference, and the "tagged directory anywhere" test. That test writes a fresh non-empty tag in a tagged tree (`tests/test_oracle_store.py:240-253`); it never covers a single-file copy.

**Phase-1 answers to the three required questions**

*Q1: Where does ingest funnel?*
- **There is no single choke point. There are two shared read functions with different coverage:**
  - `load_parquet` covers all five builders but not `data hash`/`data version`.
  - `compute_data_hash` covers train (pre-trial), tune, promote, `data hash` and `snapshot` (before `_link_or_copy`, `versioning.py:148` vs `:153-157`), but not score or the calibration script.
- **Recommendation (inferred):**
  - The primary refusal goes in `load_parquet` or a helper it calls.
  - A secondary, raise-only check in `compute_data_hash` would also close CAS laundering and fire before train's hash and trial row without moving any hash (it only raises).
  - Not `materialize`, not `ParquetDataIter`, not `snapshot()` alone.
- Fail-fast for subprocess tune (a parent-side preflight) and train (before `data_hashes` and `one_off_run`) is a Phase-2 choice.
- Evidence: see Current state; critic CR1 grep over `src/` and `scripts/`.

*Q2: Can the lazy path be checked on schema alone without a collect (D3)?*
- **Yes.** Footer-only reads (`collect_schema`, `pl.read_parquet_schema`, `pq.read_schema`) return the schema of a file whose data pages are corrupted, while `collect()` fails on it (probed). They take 0.9–1.9 ms on the 1.41 GB `data/crypto_breakout_h3.parquet`.
- **D3 is not violated.** v0 D3 (`docs/0.0/DESIGN-log.md:43-48`) does not forbid collecting at ingest. Every builder collects immediately after the scan (Q2-C1), and train, tune and promote already do a full streaming collect in `_logical_hash` (`versioning.py:78-80`). The laziness concern is moot.
- **Conditions:**
  - (a) Run the tag walk **first**.
  - (b) Schema = directory LazyFrame `collect_schema()` ∪ per-file footers of the Polars file set.
- **The multi-file "first-file-only" gap is a naming gap, not a leak.** Polars raises by default, so an `oracle__` footer column cannot reach a training frame silently today. *Corrected from reader claims of a leak.*
- **The one silent column path is hive partition keys.**

*Q3: Would anything already under `data/cas/` newly refuse?*
- **No. PR-040 is inert on today's corpus.**
  - No rux-ml `data/cas` or `data/manifests` exists on the Mac (`find ~ -maxdepth 6`) or the workbench (`find / -xdev`). The workbench's `~/projects/rux-capital/maestro/data/manifests` is another project's format.
  - Neither host has `RUXML_*` env vars or a `.env`. Compose bind-mounts `./data`, and the only Docker volume is `openclaw_data`.
  - Parquets: Mac 2, workbench 3 (at 01c7816; `git diff 01c7816 679b096 -- src tests configs` is empty). All have 0 `oracle__` columns and no tagged ancestor up to `/`.
  - The configured source has only ever been `data/crypto_breakout_h3.parquet` (725fd7b). No study user_attr mentions "oracle".
- **Positive control:** the workbench `~/projects/rux-capital/oracle-store` has 156 day-dirs, each tagged (none at the store root), and 1404 parquets, all with `oracle__` columns and a tagged parent. It is absent on the Mac.
- **Caveat:** the store is reachable today via `--set data.source_path=…` with only the accidental directory refusal, and single-file paths are not refused at all.
- `pr025_calibration.db` (30 trials) carries no `data_hash`. Configs history implies crypto_breakout_h3 (best-guess, non-blocking).

**Hash-regression analysis**

| Value | Raise-only refusal | Adding the two knobs | Evidence |
|---|---|---|---|
| `data_hash` = bytes\|logical, `version_id` | **Cannot move.** Inputs are only `*.parquet` bytes and row content. `load_parquet` is not on the hash path. | Not config-dependent | `versioning.py:51-101,149-151`; `provenance.py:83-88` |
| data-`Manifest` JSON | Refused data writes no manifest. JSON already changes on every re-snapshot (`created_at`, `:169`). | Only if a field is added; a required field breaks `read_manifest` | `versioning.py:35,160-186` |
| `data_cfg_hash` | n/a | Moves if the knobs go in `DataConfig` un-elided (3c89c1f5… → dce525c0… on crypto_h3+hpo). Stable if listed in `_HASH_ELIDED_FIELDS["data"]`. Unaffected by a top-level layer. | probes `probe_field_add.py`, `v_cfg2.py` |
| `root_cfg_hash` | n/a | Moves for un-elided `[data]` fields. **Moves for any top-level `[oracle]` layer even if every field is elided** (the layer key stays in the dump, `root.py:229-236`). Stable only for `[data]` + elided. | same probes |
| Version-id suffix `root_cfg_hash[:6]` | n/a | New trials of an identical TOML get a new suffix. Re-promoting a pre-PR trial reuses its recorded hash. | `registry/paths.py:42`; `promote.py:252` |
| `feature_list_hash` | no | no | `promote.py:151-156` |

- **Existing records are frozen and never re-checked.**
  - 24 workbench trials, all with `data_hash` b2d36926…\|29a2c053….
  - Champion bundle `v_2026_05_22_5a3138`: `data_cfg_hash` 3c89c1f5… equals the HEAD value.
  - Untracked champion receipt: no hash fields; embeds 5a3138.

  A config-hash move breaks **comparability across the PR boundary** (in `runs compare`, and for joins by version-id). It does not break validity. No rux-capital file cites any rux-ml hash (grep of `b2d369261bd2|v_2026_05_22_5a3138|5a31382c` returns nothing).
- **`Field(exclude=True)` is a trap.** It keeps hashes still, but `objective.py:138-142` and `promote.py:81-89` rebuild configs through `model_dump()` → `model_validate`. A TOML or `--set` value falls back to the default, and promote's re-fit ingest runs on that rebuilt config (probed: `toml__` → `oracle__`).
- **Anchors for pinning:**
  - golden_v1 `11f18064…|98f8c686…` (committed; reproduces at HEAD)
  - real corpus `b2d36926…|29a2c053…` (reproduces at HEAD in ~1.5 s, 3.3 GB RSS)
- **Latent risk:** `pyproject.toml` allows `polars>=1.40`. A lock bump inside PR-040 could move `logical_hash`; the struct hash is stable 1.40.1 → 1.44.2 on golden, but other versions are unverified.
- **Design constraints that follow:**
  1. The refusal must be raise-only and must not change the scan.
  2. No change to hashing inputs, and no polars bump.
  3. Choose explicitly between "`[data]` + elided" (widens D17's "paths, timestamps, runtime-only" rule: `root.py:52`, `ARCHITECTURE.md:488`, `CONVENTIONS.md:156`; it would be the first non-path elision, needing a doc update) and "accept a one-time documented move" (PR-020/PR-024 precedent, with a CHANGELOG note).
  4. No top-level `[oracle]` layer if hash neutrality is wanted.
  5. No required provenance fields.
  6. Add pinned-value tests for data hashes and, if the elision option is chosen, for config hashes.

**Staleness check (60-day)**

Threshold per `docs/CONSTRAINTS.md:144-148`.
- **Past the threshold:**
  - v0 D3/D9/D14 design (2026-05-14→15, `docs/0.0/DESIGN-log.md:7`): 132–133 days
  - PR-004 and PR-005 data-layer code (2026-05-16): 131 days
  - PR-031/PR-033 (2026-05-21): 126 days
  - PR-034 (2026-05-22): 125 days
  - The v0.3 drift-watch deadline (2026-07-19, `docs/0.3/RESEARCH-BACKLOG.md:46`) passed 67 days ago.
- **Local drift is still zero:** no `src/` or `tests/` commit after 4fb462c (2026-05-31), and polars is still locked at 1.40.1 (unchanged since 2be39b3). Only `README.md`, one study TOML, `PR-039` (1 line) and the PR-040 stub changed.
- **Fresh:** the harness reference (700a556, 8 days old) and program C13 (@ 022792b; an uncommitted working-tree edit moves C13 from `:98-100` to `:99-101`).
- **Verdict:** the research is stale by age, and Phase 1 found more than 9 stale stub assumptions. So the time-decay policy (`PROCEDURE-pr-research.md:237`) requires re-running Phases 2-4; they cannot be light.
- **Premise check** (`PROCEDURE:44`), my assessment: the premise itself (refuse oracle inputs at the training-set builder) survives. The attachment point, verification criteria and version class change, so no loop to design-planning is forced by Phase 1 alone. VERSIONING §2 may force one independently.

**For the lead** (program repo, ACCEPTANCE C13)

**C13 verbatim** (`program docs/0.2/ACCEPTANCE.md:99-101` @ 022792b + uncommitted working-tree edit): "Every oracle-derived column lives in the `oracle__` namespace and every Stage-1 output in a store separate from production/training; a test shows the training-set builder and the promotion path refuse any `oracle__` column and any artifact tagged as produced under oracle inputs; the C9 record lists the oracle store path." — note it asks for **a test**, and names **the promotion path** as well as the builder.

1. **Owner record.** C13's §2b.4 note (`program docs/0.2/ACCEPTANCE.md:100` @ 022792b) says "a rux-ml PR"; record it as "rux-ml PR-040". `docs/0.2/ROADMAP.md` has no rux-ml row, although `program docs/VERSIONING.md:57` requires "rux-ml PR-NNN" references.
2. **Refusal sites to cite.** `load_parquet` covers all five rux-ml builders: train; tune (in-process, subprocess child, retry); registry promote re-fit; registry score; `scripts/calibrate_pr025.py`. `compute_data_hash`/`snapshot`, if adopted, covers CAS laundering. `snapshot()` alone covers no training path.
3. **Scope of rux-ml's half.**
   - It sees only the `ORACLE.tag` file and the `oracle__` namespace. It cannot see the harness lineage `runs.oracle_tagged` (`harness lineage/store.py:95`; `accounting/promote.py:359-378`).
   - Its "promotion path" refusal is a refusal at re-fit ingest only. Promote recomputes `data_hash` and never compares it to the trial's recorded value (`promote.py:171-197`), and `source_path` is elided from config hashes. So rux-ml cannot prove a trial was not *tuned* under oracle inputs (BAKEOFF rule 1, `program docs/0.1/BAKEOFF-PROTOCOL.md:18`).
   - All 24 promotable pre-PR workbench trials carry the clean crypto `data_hash`.
4. **Corpus inertness evidence.** No rux-ml CAS or data manifests on either host. No `RUXML_*` or `.env` overrides. One configured source ever. Every parquet has 0 `oracle__` columns and no tagged ancestor.
5. **Positive control.** The workbench oracle-store has 156 tagged day-dirs, all tags non-empty JSON, and 1404 parquets that would all trip both refusals. The store is absent on the Mac.
6. **Harness `production_load` gaps to record, not fix** (out of PR-040's scope):
   - relative paths stop at `.`, and `../x` gives a false positive when the cwd holds a tag
   - no `resolve()`, so a file symlink escapes
   - `~` is walked literally while polars expands it
   - a tag inside a directory passed as the path is not checked
   - keys-only single-file copies pass
   - empty `namespace` or `tag_file` are not validated (empty `tag_file` disables the tag half)
   - case-variant prefixes and nested struct fields pass
   - tables are written before the tag (`scripts/build_oracle_store.py:44-46`), so an interrupted build leaves an untagged day directory
   - `production_load` has **no non-test caller**, and PR-018's C13 half was evidenced by tests only
7. **Correct the overclaim** in `program docs/0.1/HANDOFF-20260924-B.md:21-23` (and the stub). Only a copy or symlink of the *whole tagged day directory* keeps the tag. Single-file copies, hardlinks, rux-ml CAS snapshots and file symlinks lose it; only the namespace check survives those.
8. **Record the intended divergence** as "same rule, not same code or coverage". rux-ml will be stricter: path normalization, a walk over Polars' actual file set, hive-key columns, and the directory itself. A parity check should not flag this.
9. **Value sync.** There is no shared package, so `oracle__` and `ORACLE.tag` are copied by value. All four harness configs agree today (`harness.toml:143-146`, `harness-pilot-bf.toml:156-159`, `variants/projected.toml:138-141`, `variants/rg65536.toml:138-141`). Consider a cut-time equality check.
10. **Correct `program prs/PR-008-rux-ml-assessment.md:15`.** The champion receipt is **untracked, not gitignored**: `git check-ignore` rc=1, and `receipts/.gitignore:2` says "JSON receipts ARE tracked". It embeds `root_cfg_hash[:6]` and no data hash.
11. **Evidence form.**
    - rux-ml has no CI (`.github/workflows` is absent), so no CI run id can be cited.
    - Default subprocess `tune start` exits 0 even when every child refuses.
    - Registry verbs exit 2 with a ValueError-based error and exit 1 (traceback; inferred) with a RuntimeError-based one.
    - A PR-019-style real-store demonstration must run bare-metal on the workbench.
    - Record which command and exit code count. Program VERSIONING §2b.5 treats a green suite as supporting evidence only, and the precedent is split (PR-018: tests; PR-019: a real demonstration).
12. **Timing (resolved by operator, 2026-09-24).** No version bump or cut: PR-040 lands standalone against `dev`. Nothing in rux-ml versioning gates C13's close.
13. **Pre-existing rux-ml provenance gap** (not C13, but it affects any receipt that cites `data_hash`): `data_hash` does not cover hive partition keys, files behind symlinked subdirectories, or `.pq` files that Polars loads. The hashed data can therefore differ from the trained data. This is latent today, because the only source is a single file.
14. **Citation fixes:** the reference is `src/rumpy_harness/oracle/store.py:79-89` @ b9922ad. Polars versions differ (rux-ml 1.40.1, harness 1.44.2), but every probed behaviour is identical.

**Tier lean**

**Tier-2 (lean; the operator decides).** Reasons:
1. **Repo definition.** Tier 1 = "Design-time research exists"; Tier 2 = "partial or absent" (`docs/0.3/RESEARCH-BACKLOG.md:7-9`). rux-ml has no design-time research on oracle quarantine: the only "oracle" hit across docs, configs, src, tests, README, CHANGELOG and the PROCEDUREs is the unrelated `docs/0.2/DESIGN-log.md:71`. The harness reference evidences the *rule*, not rux-ml's ingest.
2. **Time-decay.** Stale assumptions were found, so Phases 2-4 must re-run (`PROCEDURE-pr-research.md:237`).
3. **Tier-inheritance rule.** Reclassify when more than one sub-decision needs research rather than preference. Phase 1 surfaced at least eight:
   - (a) refusal sites, and file enumeration that matches Polars' symlink-following
   - (b) schema-check composition (hive keys, ordering)
   - (c) path normalization versus the reference
   - (d) config placement and hash elision (widening D17)
   - (e) default policy versus "no literals", which also sets the bump class
   - (f) error base class and exit behaviour
   - (g) fail-fast placement under subprocess tune and train ordering
   - (h) whether promote does a provenance check
4. **Precedent.** PR-027 stayed Tier-2 over a single schema decision (`prs/PR-027-int64-timestamp-time-unit.md:57-58`).

*Counter-case:* the Q6 reader leaned Tier-1 because no external web research is needed. That is not the repo's tier criterion. It is true, though, that most of Phase 3 can be local probing against the pinned polars 1.40.1, so a Tier-2 run should be cheap.

**Open decisions for the operator**
1. **Tier:** Tier-2 (lean) or Tier-1?
2. ~~**Bump class and process**~~ — **RESOLVED (operator, 2026-09-24): ignore versioning; no bump, no cut, no v0.4 scaffolding.** PR-040 lands standalone against `dev` with "no row to flip" (PR-037/038/039 precedent).
3. **Default policy under "never as literals in code":**
   - (i) a required field: fail-closed; breaks 25 `DataConfig(...)` and 38 `RuxMLConfig(...)` direct constructions, `default_factory=DataConfig` (`root.py:95`), and TOML-less test configs such as golden
   - (ii) `None` plus a raise at the point of use (D4 precedent, `config/features.py:25`; `features/encoders.py:61-67`)
   - (iii) a Pydantic default (rux-ml convention; conflicts with the instruction)
4. **Placement and hash policy:** flat `data.oracle_*` or a `data.oracle` sub-model, elided (hash-stable, first non-path elision, D17 doc update), or accept a documented `data_cfg_hash`/`root_cfg_hash` move? Rule out a top-level `[oracle]` layer if stability matters.
5. **Refusal sites:** `load_parquet` (required). Also, each optional:
   - `compute_data_hash`, which covers snapshot, `data hash`, and train before hashing and trial creation
   - a parent-side preflight in `cli/tune.py`, so the sweep exits non-zero
   - should `rux-ml data hash` refuse or warn?
6. **Directory sources:** write a file enumerator that matches Polars (following symlinked subdirectories), or refuse directory sources that contain symlinks? And glob `source_path`: expand it the way Polars does, or reject it (`iter_parquet_files` already raises on globs)?
7. **Path semantics:** `expanduser` + `resolve` + include the directory itself + nested tags (stricter than the harness), or a literal mirror with its gaps documented?
8. **Other reference gaps:** close them (case-variant prefix, nested struct fields, a directory named `ORACLE.tag`), or mirror them?
9. **Hive-key detection:** via the directory `LazyFrame.collect_schema()` (follows exactly what training sees) or by parsing `key=value` path segments (stable if scan options change)?
10. **Error type:** a `ValueError` subclass (becomes `BadParameter`, exit 2 on registry verbs) or a `RuntimeError` subclass (harness style, traceback)? Where it lives (`data/` plus `data/__init__.__all__`), and whether CONVENTIONS gains an errors section.
11. **Promote provenance:** add a promote-time check (e.g. refuse when the re-fit `data_hash` differs from the trial's recorded one), or leave rux-ml's promotion half as a refusal at re-fit ingest only?
12. **Scope confirmations:**
    - drop the "auditable in the manifest" criterion
    - defer the pre-existing hash-vs-load divergence (hive keys, symlinked subdirectories, `.pq`) to its own PR
    - restate "no change to the manifest" as field-level invariants
13. **Outside PR-040** (flag only): commit the untracked workbench champion receipt, as `receipts/.gitignore` intends? Fix PR-039's misfiled CHANGELOG entry?

**Phase 1 gate outcome (operator, 2026-09-24): all leans approved.** These are locked as operator decisions; Phase 3 tests them with evidence and flags any that the evidence contradicts (it does not quietly re-open them):

1. **Tier-2.** The full 5-phase procedure applies.
2. No version bump or cut; standalone against `dev`, no roadmap row. `[Unreleased]` CHANGELOG entry only.
3. **Missing values fail closed.** Oracle config is `None` by default, and the check raises at the point of use when it is absent (D4 precedent). The values live only in `configs/base.toml`.
4. **A `[data.oracle]` sub-model** (`namespace`, `tag_file`), **excluded from hashing** via `_HASH_ELIDED_FIELDS["data"]`, so `data_cfg_hash` and `root_cfg_hash` stay stable. D17 widens to output-neutral fields, with docs and a pinned config-hash test in the same commit. No top-level `[oracle]` section.
5. **One `check_oracle_quarantine(path, oracle_cfg)`**, called from `load_parquet`, `compute_data_hash` (so `data hash` and `data version` refuse too) and a parent-side preflight in `cli/tune.py`.
6. **Directory sources:** the tag walk follows symlinks, so it covers at least every file Polars reads. Glob sources are rejected.
7. **Path handling:** expand `~`, then check the parent directories of both the path as written and the resolved path, and refuse if either has a tag. Include the source directory itself and any tag inside a directory source.
8. **Close the reference gaps:** case-insensitive prefix, recursive check into struct and list field names, and `exists()` rather than `is_file()` for the tag.
9. **Column check:** `collect_schema()` of the same LazyFrame training uses (this catches hive keys), plus per-file footer reads so mixed-schema directories get a named refusal.
10. **`OracleQuarantineError(ValueError)`** in `rux_ml/data/`, exported. `train` and the tune preflight catch it and exit 2. No CONVENTIONS errors section.
11. **The promote-time `data_hash` comparison is deferred** to its own PR; the limitation is recorded in the "For the lead" list.
12. **Scope:** drop "auditable in the manifest"; defer the gap between what is hashed and what is loaded; restate "no change to the manifest" as field-level invariants.
13. The workbench receipt commit and the PR-039 CHANGELOG move are separate commits, outside PR-040.

### Research Questions (2026-09-24)

Pinned versions (`uv.lock`): polars 1.40.1, pyarrow 24.0.0, pydantic 2.13.4, pydantic-settings 2.14.1, typer 0.25.1, click 8.3.3. Python 3.12 (`.python-version`; Dockerfile `uv python install 3.12`). Every answer is cited at these versions, or labelled BGGC.

**Must-answer:**

1. **Which files does `pl.scan_parquet(<dir>)` read at polars 1.40.1, and which schema does it report?** Cover:
   - file selection: extensions (`.parquet`, `.pq`, extensionless?), hidden or `_`/`.`-prefixed files, non-parquet files (the non-empty-tag `InvalidOperationError` seen in Phase 1)
   - whether it follows symlinked subdirectories and symlinked files
   - `~` expansion; glob detection
   - hive-partition discovery defaults for directory sources
   - whether `collect_schema()` reflects the first file only
   - *Success:* file:line citations in the polars source at the `py-1.40.1` tag for path expansion, file filtering and schema inference, each confirmed by a local probe against the pinned wheel. This defines the minimum the tag walk and column check must cover (decisions 6 and 9).

2. **What is the cheapest schema read that doesn't collect, and does it expose everything that reaches the frame?** Compare `LazyFrame.collect_schema()`, `pl.read_parquet_schema` and `pyarrow.parquet.read_schema`:
   - cost per file (footer only?)
   - hive keys: present or absent
   - nested field names in `Struct`, `List(Struct)` and `Array(Struct)`, and how to walk them recursively
   - behaviour on a zero-byte or truncated file
   - *Success:* cited polars and pyarrow docs or source for each API's I/O behaviour, plus a probe table on the 1.41 GB real file and synthetic nested and hive fixtures. Picks the per-file footer API for decision 9 and the recursive traversal for decision 8.

3. **Path normalization and a symlink-following walk on Python 3.12 that is safe against cycles.** Cover:
   - `Path.expanduser`, and `Path.resolve(strict=False)` versus `strict=True`, on missing paths and symlink loops (`RuntimeError`/`OSError`)
   - `os.walk(followlinks=True)` cycle behaviour; the documented warning about infinite recursion
   - the `Path.rglob` symlink semantics that changed in 3.13 (`recurse_symlinks`), versus 3.12
   - Unicode case-folding: `str.casefold()` versus `lower()` for the prefix match
   - *Success:* Python 3.12 docs or CPython source citations, plus one cited pattern already working in a reputable OSS project for a walk that follows symlinks and tracks visited `(st_dev, st_ino)`. Otherwise label it BGGC. Depends on Q1 (the walk must cover at least what Polars reads).

4. **Does an optional nested sub-model survive rux-ml's config round-trips?** Specifically, can `DataConfig.oracle: OracleQuarantineConfig | None = None` under `extra="forbid"`:
   - survive TOML load, `deep_merge`, `--set data.oracle.namespace=…` and env `RUXML_DATA__ORACLE__…` overrides
   - survive the `model_dump()` → `model_validate` rebuilds in `objective.py:138-142` and `promote.py:81-89`, keeping the TOML values rather than falling back to `None`
   - stay hash-stable when elided as a whole key (`_elide` on `"oracle"`)
   - enforce `min_length=1` and a no-path-separator rule on `tag_file`
   - *Success:* pydantic 2.13 and pydantic-settings 2.14 docs or source citations for nested-model env delimiters and dump/validate round-tripping, plus probes. The probes must show hashes identical before and after the change on `crypto_breakout_h3` + hpo and golden_v1, and values surviving the objective and promote rebuilds.

5. **Is refusing at ingest by column-name namespace plus a directory tag an established way to prevent leakage, and what are the alternatives?**
   - Candidate alternatives: schema or metadata allow-lists, lineage tags carried in parquet key-value metadata, feature-store point-in-time joins, data-contract validation.
   - Actively look for cases where name-based quarantine is known to fail (renames, derived columns, joins).
   - *Success:* at least 2 independent cited systems per option (the convention bar), with pros and cons. Records whether decision 5/7/8's design is `convention` or BGGC, and names any evasion path the design must document for the lead. It does not re-open the approved design unless the evidence shows it fails C13.

6. **Exit-code and exception semantics.** Does click 8.3.3 / typer 0.25.1 map `BadParameter`/`UsageError` to exit 2, and what does an uncaught exception in a typer command exit with? Is `ValueError` the right base in Python's documented exception hierarchy for "valid type, refused value" (compared with `RuntimeError`/`PermissionError`)?
   - *Success:* click source file:line for the exit codes; Python docs for the exception semantics. Confirms decision 10 and the evidence form for the lead (which command and which exit code count as proof for C13).

**Dependencies:**
- Q1 must come before Q3, because the walk's coverage is defined by what Polars reads.
- Q2, Q4, Q5 and Q6 are independent and can run in parallel with Q1.

**Explicitly excluded from this round:**
- A promote-time comparison of the recorded `data_hash` (decision 11, its own PR).
- Aligning the hash path with the load path for hive keys, symlinked subdirectories and `.pq` files (a pre-existing provenance gap, its own PR).
- A retroactive scan of the data store, which is inert per Phase 1.
- Fixing the harness `production_load` gaps; those are recorded for the lead only.
- CI setup for rux-ml.
- Keeping the rux-ml and harness config values in sync (a cut-time equality check is the lead's call).

_Phase 2 complete 2026-09-24 — halted at the Per-Phase Approval Gate._

### Findings (2026-09-24)

Four capped research agents (polars; Python paths and exit codes; config; prior art). Probe scripts: `/private/tmp/claude-501/pr040-scratch/p3/`; driver's Group D probe: `.../gd/gd.py`.

**Q1+Q2: What polars reads from a directory source, and how to read the schema without a collect**

*Answers (all proven at 1.40.1; source cited at the `py-1.40.1` tag, confirmed by probes):*
- **File selection.** The walk reads every subdirectory and keeps every non-directory entry larger than 0 bytes, whatever its name or extension (`crates/polars-io/src/path_utils/mod.rs:477-479`, `md.len() > 0`). That includes hidden `.x` files, `_x` files and extensionless files. Zero-byte files, including a zero-byte `ORACLE.tag`, are skipped silently. Mixed extensions in one directory raise `InvalidOperationError` (`:512`); that is the "accidental refusal" from Phase 1. The scan parameter `hidden_file_prefix` defaults to `None` and is not exposed by `scan_parquet` at 1.40.1.
- **Symlinks.** Symlinked directories and files are both followed through `metadata()` (`:475-481`). There is no cycle guard; the driver probe of a cycle raised `OSError`.
- **`~` and globs.** `~` is expanded in Python (`polars/io/_utils.py:336-337`). A path is a glob if it contains `*`, `?` or `[` (`mod.rs:128-133`).
- **Hive keys.** A directory source has hive discovery on by default, so an `oracle__k=1/` path key shows up in `collect_schema()`. A single-file scan shows no hive keys.
- **Mixed-schema directories.** `collect_schema()` returns the first file's schema only. `collect()` then raises `SchemaError`, which is unnamed, so per-file reads are needed to produce a named refusal.
- **Schema reads are footer-only.** On the 1.41 GB real file: `pq.read_schema` 0.53 ms, `pl.read_parquet_schema` 0.95 ms, `scan_parquet().collect_schema()` 0.92 ms. **`pl.read_parquet_schema` is `scan_parquet(source).collect_schema()`** (`polars/io/parquet/functions.py:375`), which runs with glob=True, so it raises `ComputeError` on a legitimate `p[1].parquet`. Driver-reproduced.

*Options (per-file footer read):*
- **A: `pl.scan_parquet(f, glob=False).collect_schema()`.** Same parser and dtype model as training, so one traversal serves both reads. Safe for literal `[` in filenames. Uniform `ComputeError` on corrupt files. It is not an independent parser.
- **B: `pyarrow.parquet.read_schema`.** The fastest, and an independent parser. It needs a second traversal over pyarrow types, raises a different exception family (`ArrowInvalid`), and could see the file differently from polars.
- **C: `pl.read_parquet_schema`.** Rejected because of the glob=True trap.
- **D: private `polars.io._expand_paths`.** Rejected: it is private API and can drift.

*Disconfirming evidence sought:*
- Hidden-file filtering: none at 1.40.1. The newer docs (1.44.2) add `hidden_file_prefix`, which can only shrink the read set, so the walk stays a superset.
- Glob characters beyond `*?[`: none; `{}` is not a glob.
- GitHub issues on `_SUCCESS`/hidden files: no results.

*Recommendation:* Option A for per-file reads, plus `collect_schema()` of the training LazyFrame (which catches hive keys). Run a recursive traversal over `Struct.fields`, `List.inner` and `Array.inner`, matching with `casefold().startswith(prefix.casefold())`. **The tag walk must run before any `scan_parquet` call**, because polars raises on a non-empty tag first. **Status: proven.** Locks 6, 8 and 9 are confirmed.

**Q3: Path normalization and a cycle-safe walk that follows symlinks (Python 3.12)**

*Answers (proven; 3.12 docs and `pathlib.py` at 3.12.14, plus probes):*
- **`expanduser`.** It raises `RuntimeError` when the home directory can't be resolved.
- **`resolve()` on a symlink loop.** It raises `RuntimeError("Symlink loop from …")` for both strict modes (`pathlib.py:1237`). 3.13 changes this to `OSError` or no error at all, so the code must catch both.
- **`os.walk(followlinks=True)`.** It has no visited tracking; the docs warn it can recurse infinitely. Its default `onerror=None` **silently skips unreadable directories**: a tag inside a chmod-000 directory was missed, so the walk fails open.
- **`Path.rglob` on 3.12.** It does not follow symlinks and found 0 tags through a symlinked subdirectory. Unusable.
- **Tag presence.** **`Path.exists()` follows symlinks and swallows `ELOOP`** (`pathlib.py:45,852`), so a dangling or looping `ORACLE.tag` symlink reads as absent. `os.path.lexists` / `exists(follow_symlinks=False)` detects presence.
- **Case folding.** `casefold()` is the documented way to match without regard to case. It is identical to `lower()` for ASCII `oracle__`.
- **Both ancestor chains are needed.** The path as written and the resolved path each find tags the other misses.

*Options:*
- **A: `os.walk(followlinks=True, onerror=raise)`** with a visited set of `(st_dev, st_ino)` that prunes `dirnames`, a `lexists` tag check, and the ancestor chains of both `absolute()` and `resolve()`.
- **B: `rglob` / `Path.walk` without a visited set.** It misses symlinked subdirectories on 3.12 and has no cycle guard.
- **C: `os.walk` with the default `onerror`.** It fails open on permission errors.

*Cited working example of the exact combination:* setuptools 82.0.1 `_distutils/filelist.py:328-353` (`os.walk(path, followlinks=True)` + `(st_dev, st_ino)` + `del dirs[:]`, "Ref bpo-44497"). A second system, Rust `walkdir` `follow_links` loop detection, is cited from its docs only.

*Recommendation:* Option A. **Status: proven.**

**Q4: Does the `data.oracle` sub-model survive the config round-trips, and do the hashes stay put?**

*Answers (proven; probes at pydantic 2.13.4 / pydantic-settings 2.14.1):*
- **Loading and overrides.** `[data.oracle]` loads through the real `from_layers` (base → `crypto_breakout_h3` → hpo). It survives `--set data.oracle.namespace=…` (`cli/_shared.py:40-64`) and `RUXML_DATA__ORACLE__NAMESPACE` (`root.py:111-112`, `env_prefix="RUXML_"`, `env_nested_delimiter="__"`). Setting the env var with no TOML table fails validation, because `tag_file` is required.
- **Rebuilds.** The `objective.py:138-142` / `promote.py:81-89` dump → validate rebuild keeps the values.
- **Hash stability. Without the exclusion, adding the field moves `data_cfg_hash` and `root_cfg_hash` even when the value is `None`**, because the dump gains `"oracle": null`. With `"oracle"` in `_HASH_ELIDED_FIELDS["data"]`, all cases match HEAD:
  - hpo: data `3c89c1f5…`, root `06f13e9c…`
  - golden: data `2d7467a5…`, root `c5ba8632…`

  **The exclusion is required, not optional.** Driver-reproduced independently (`gd.py` case 5).
- **`_elide`.** It drops the whole key (`root.py:69-70`).
- **Validators.** `min_length=1` plus the path-separator check reject `""`, `a/b` and `a\\b`.
- **Stored artifacts.** None persists a DataConfig or RuxMLConfig snapshot: trial attrs, `ModelManifest`, data manifest and scorer receipt are hashes only. The subprocess child rebuilds the config from TOML (`tuning/isolation.py:33-42`).

*Options:*
- **A (locked):** Optional, elided.
- **B:** Optional, not elided. Every hash moves.
- **C:** Always present through `default_factory`. It needs values in code, which conflicts with lock 3.

*Disconfirming evidence sought:* pydantic-settings issue #420 (`Optional` nested + `extra=forbid` + `.env`, a 2.5.0 regression, closed). It doesn't apply here: no `env_file` is configured. Pydantic documents no guarantee that dump → validate round-trips; the probe is the evidence.

*Recommendation:* Option A. Update the elision comment at `root.py:52-53`. Pin both hash pairs in a test. **Status: proven.**

**Q5: Is namespace-plus-tag refusal an established leakage guard?**

*Options (each ≥2 cited systems):*
- **A: Tag-driven exclusion, enforced at read time, with the tag inherited down a hierarchy.**
  - Sources: AWS Lake Formation LF-Tags ("Tables inherit LF-Tags from databases…"); Databricks Unity Catalog ABAC governed tags; Hadoop `FileInputFormat.hiddenFileFilter` (a reserved `_`/`.` prefix at read; cited at trunk, no pinned SHA); Kaufman et al. KDD'11 §4.1 "legitimacy tags".
  - This is a **convention**.
- **B: A declared allow-list or schema environment.** TFDV `not_in_environment('SERVING')`; pandera `strict=True`. Stronger against renames, but it needs a per-problem feature declaration that rux-ml doesn't have.
- **C: A lineage marker in parquet key-value metadata.** Probe: polars drops it on rewrite.
- **D: Storage-level separation.** Already true on the harness side: a separate store.
- **E: Point-in-time / legitimacy timestamps.**

*Disconfirming evidence:*
- **No cited system refuses a column because of a reserved name prefix**, and none uses the exact prefix-or-tag combination.
- Kaufman et al. §5: a leak "may [be] partly plug[ged] … but not … seal[ed] completely". Kapoor & Narayanan (Patterns 2023): the remedy is a written argument that leakage is absent.
- Probed evasions: renaming `oracle__y`; a derived column under a clean name; metadata dropped on rewrite.

*Recommendation:* Keep the locked design. It meets C13's syntactic text ("refuse any `oracle__` column and any artifact tagged …"). **Status: the tag half is convention; the column-prefix half and the combination are best-guess-given-constraints.** It is one layer of defense in depth, not a complete leakage defense. Its evasion paths go to the lead.

**Q6: Exit codes and exception base**

*Answers (proven; installed source, plus a probe):*
- click: `ClickException.exit_code = 1` (`click/exceptions.py:30`), `UsageError.exit_code = 2` (`:65`), `BadParameter(UsageError)` (`:95`); `sys.exit(e.exit_code)` in `core.py:1449-1453`.
- **An uncaught non-click exception in a typer command exits 1**, whether pretty exceptions are on or off. It exits 2 only if caught and re-raised as `typer.BadParameter`.
- The rux-ml registry verbs already do that for `ValueError` (`cli/registry.py:46-47,121-122`).
- Python docs: `ValueError` means "right type but an inappropriate value".

*Options:*
- **`ValueError` base:** registry verbs already exit 2 with it.
- **`RuntimeError` base (the harness style):** exits 1 with a traceback unless caught.

*Recommendation:* `OracleQuarantineError(ValueError)`. **The check must wrap `RuntimeError` (from `expanduser`/`resolve`) and `OSError` (from the walk or `lexists`) into `OracleQuarantineError`**, so errors in the check fail closed with exit 2 rather than escaping as exit 1. **Status: proven.**

### Group D: MCP Verification (2026-09-24)

**Schema-Integrity Probe:**

| Claim | Identifier | Canonical documenter | Verified? | Notes |
|---|---|---|---|---|
| Q1 | zero-byte skip `md.len() > 0`; `has_glob` `*?[`; mixed-ext raise | `pola-rs/polars` `crates/polars-io/src/path_utils/mod.rs` @ `py-1.40.1` :128-133, :477-479, :512 | yes | fetched raw at tag |
| Q2 | `read_parquet_schema` = `scan_parquet(source).collect_schema()` | venv `polars/io/parquet/functions.py:375` (Version 1.40.1) | yes | |
| Q3 | `_IGNORED_ERRNOS` incl. `ELOOP`; `exists(*, follow_symlinks=True)`; `RuntimeError("Symlink loop…")` | uv Python 3.12.14 `pathlib.py:45,852,1237` | yes | |
| Q3 | `_UniqueDirs` `(st_dev, st_ino)`, `del dirs[:]`, `followlinks=True` | venv `setuptools/_distutils/filelist.py:328,335,350,353` | yes | |
| Q4 | `env_prefix="RUXML_"`, `env_nested_delimiter="__"`, `nested_model_default_partial_update=True` | `src/rux_ml/config/root.py:110-114` @ 679b096 | yes | the third setting was not named by the agent; env-only use still needs the TOML table (tag_file required) |
| Q4 | `_elide` key filter; `_HASH_ELIDED_FIELDS["data"]` | `root.py:55-65,69-70` | yes | |
| Q6 | `exit_code = 1` / `= 2`; `BadParameter(UsageError)`; `sys.exit(e.exit_code)` | venv `click/exceptions.py:30,65,95`; `click/core.py:1449-1453` | yes | |
| Q6 | registry `ValueError` → `BadParameter` | `src/rux_ml/cli/registry.py:46-47,121-122` | yes | |

**Synthesis-Verification Probe** (driver-written `gd.py`, independent of the agents' scripts; two agents both wrote to `p3/probe.py`, so the agent probe file is not a reliable record):

| Claim | Combined elements | Cited working example | Verified? | Notes |
|---|---|---|---|---|
| Walk | expanduser + ancestor chains of `absolute()` and `resolve()` + `os.walk(followlinks, onerror=raise)` + `(st_dev, st_ino)` prune + `lexists` | setuptools `filelist.py:328-353` for the walk core; `gd.py` cases 1-3 for the full combination | yes | a zero-byte tag in the directory source was refused while polars read it fine; a tagged dir behind a symlinked subdir, next to a symlink cycle, was refused while polars raised `OSError`; a file symlink pointing into a tagged dir was refused through the resolved chain |
| Columns | directory `collect_schema()` + per-file `scan_parquet(f, glob=False).collect_schema()` + recursive casefold walk | `gd.py` case 4 | yes | found hive `oracle__k` and nested `s.Oracle__z`; `read_parquet_schema` raised `ComputeError` on `p[1].parquet` as claimed |
| Hash | `oracle: OQ \| None = None` on the declared field type + `"oracle"` elided | `gd.py` case 5 | yes | unelided: both hashes move for `None` and for set; elided: both equal HEAD. Driver's first probe was invalid (a subclass instance serialized under the parent type drops `oracle`) and was corrected. **Implementation note:** the field must be declared on `DataConfig` itself |

**Binding-at-creation:** not applicable. There is no registration; config is read per call.

**Reconciliations:**
- The column-prefix half is downgraded to BGGC (Q5), and the gap is recorded.
- Lock 8 is refined: `lexists`, not `exists()`.
- Lock 6 is refined: `onerror` must re-raise.
- Lock 10 is refined: wrap `RuntimeError`/`OSError`, and catch explicitly for exit 2.
- The polars per-file API is fixed as `scan_parquet(f, glob=False)`, not `read_parquet_schema`.
- **Open gap, the Amend candidate:** ancestors of the *realpath of symlinked subdirectories and files found during the walk* are not covered by lock 7, which resolves only the source path. A symlink pointing *below* a tagged day dir would pass (Phase 1 stale-assumption 5). The cheap fix: check the resolved ancestor chain for every symlink the walk meets.

_Phase 3 complete 2026-09-24 — halted at the Per-Phase Approval Gate._

### Synthesis (2026-09-24)

**Outcome**: **Amend → Apply.** The findings confirm the 13 locked decisions and refine five of them. One gap was amended with operator approval on 2026-09-24: symlink-target ancestors are now checked.

**Changes to this PR** from research:
- The attachment point moves from `snapshot()` to `load_parquet` + `compute_data_hash` + a tune preflight (Phase 1).
- The per-file schema API is locked to `scan_parquet(f, glob=False).collect_schema()`, not `pl.read_parquet_schema`, which has the glob trap (Q2).
- Lock 8 refined: the tag check uses `lexists`, not `exists()` (Q3).
- Lock 6 refined: the walk runs with `onerror=raise`, plus a `(st_dev, st_ino)` visited set (Q3).
- Lock 10 refined: `RuntimeError`/`OSError` from the check are wrapped into `OracleQuarantineError`. Exit 2 needs an explicit catch, because an uncaught error exits 1 (Q6).
- Lock 7 amended: every symlink the walk meets also has its resolved target's ancestors checked.
- Hash elision is **required**, not optional (Q4). The field must be declared on `DataConfig` itself (Group D).
- `rux-ml data hash` gains a config load.
- Scope, verification criteria, research backing and notes are rewritten below. Removed or replaced stub specifics:
  - `snapshot()` as home
  - "recorded in the manifest"
  - "copied/symlinked keeps its tagged parent"
  - "a bare `ValueError`" (it is now a named `ValueError` subclass)
  - the roadmap-row flip

**Changes to ARCHITECTURE.md** (drafted now; they ride with the implementation commit, per the PR-033 a13abb3 precedent):
- New decision rule, "Oracle quarantine at ingest (per PR-040; program ACCEPTANCE C13)".
- The D17 elision sentence (Configuration Architecture) now admits output-neutral guard config.

**Changes to CONSTRAINTS.md**:
- New domain constraint, "Oracle Quarantine at Ingest (NON-NEGOTIABLE)".

**Other doc changes**:
- `docs/CONVENTIONS.md` elision paragraph.
- `docs/0.3/DESIGN-log.md` PR-040 session.
- `docs/0.3/RESEARCH-BACKLOG.md` PR-040 row.
- At implementation: `README.md` data-layer text, `configs/README.md`, the `root.py:52-53` comment, and the `CHANGELOG.md` `[Unreleased]` entry.

**New PRs that must come first**: none. Two follow-up PRs are deferred and are not prerequisites: the promote-time `data_hash` comparison, and aligning the hash path with the load path.

**Research-backed details now locked in this PR**: see `## Scope`, "The check" and "Call sites", and `## Verification criteria`.

**For the lead: Phase 3/4 additions** (to the Phase 1 list above, which stands with item 12 resolved):
15. **Epistemic status of rux-ml's half.**
    - Tag exclusion enforced at read is **convention**, per AWS Lake Formation LF-Tag inheritance, Databricks Unity Catalog governed tags, Hadoop `hiddenFileFilter` and Kaufman et al. KDD'11 legitimacy tags.
    - Refusing a column by name prefix, and the prefix-or-tag combination, is **best-guess-given-constraints**: no cited system does exactly this. It meets C13 because C13's text is syntactic.
16. **Known evasions of both halves** (the harness has the same ones). The C13 record should list them as accepted limits; the second, independent refusal and a written argument that leakage is absent cover what syntax cannot (Kapoor & Narayanan 2023).
    - (a) rename `oracle__y` to a clean name (probed)
    - (b) a derived or aggregated column under a clean name (probed)
    - (c) a join that pulls oracle-derived values into a clean table outside the tagged tree
    - (d) a single file, or a rewrite, copied out of the tagged tree, when its columns have been renamed
    - (e) oracle values carried as data (map keys, string payloads)
    - (f) Unicode lookalike prefixes (not probed)
    - (g) ingest paths that bypass `load_parquet`, which CONSTRAINTS now forbids
    - (h) a study-level override of `[data.oracle]` values. It is not visible in any config hash (elided by design); the value is visible only in the TOML.
17. **Exit codes after PR-040.** Every refused rux-ml verb exits 2 with `OracleQuarantineError`: `train`, `tune start/resume`, `registry promote/score`, `data hash/version`. That supersedes the Phase 1 item 11 note that `tune` exits 0 and uncaught errors exit 1.
18. **Evidence form offered for C13:**
    - (i) the `tests/data/test_quarantine.py`-style refusal suite plus the CLI end-to-end tests at the implementation SHA ("a test shows")
    - (ii) the bare-metal workbench demonstration: `rux-ml train --set data.source_path=<oracle-store day file | day dir>`, exit 2, named error. This is the PR-019-style real-row evidence.
    - The lead decides which counts.
19. **polars drift note.** Newer polars (1.44.2 docs) adds a `hidden_file_prefix` scan option. It can only shrink the read set, so rux-ml's walk stays a superset. No action needed.

_Phase 4 complete 2026-09-24. Halted at the Per-Phase Approval Gate before Phase 5 (Gate Check)._

### Implementation record (2026-09-24)

Branch `pr-040/oracle-quarantine-training-set-refusal` off `dev` @ 679b096. Tests were written first. The 3 new files (`tests/data/test_quarantine.py`, `tests/config/test_oracle_config.py`, `tests/cli/test_oracle_quarantine_cli.py`) failed at collection with `ImportError` (RED), then passed after the implementation (GREEN).

- **Tests.** 60 new tests; 454 passed on the full default suite.
- **Mutation check.** With the `resume` and `retry-trial` preflights removed, their two end-to-end tests fail; restored, they pass.
- **Pinned config hashes.** Computed at 679b096 before any source change: hpo data `3c89c1f5…`, root `06f13e9c…`; minimal data `2d7467a5…`, root `83c3090f…`. All still equal after PR-040, with `[data.oracle]` set, overridden or unset.
- **Pinned data hash.** The golden_v1 `data_hash` is read from the committed `manifest.json` and still reproduces.
- **Deviation from the approved spec (disclosed).** The tag check uses `Path.exists(follow_symlinks=False)`, not `os.path.lexists`. `lexists` returns `False` for an existing tag in a directory without search permission (verified at 3.12.14), which fails open. `Path.exists(follow_symlinks=False)` raises `PermissionError`, which the check wraps into a refusal. Test: `test_unsearchable_ancestor_fails_closed`. Docs updated to match.
- **`registry score` side effect (pre-existing, not changed).** `scorer.py:247` creates `output_dir` before loading data, so a refused score leaves an empty `receipts/` directory. The test asserts that directory is empty.
- **Formatting.** 25 files were already unformatted at HEAD under ruff 0.15.13. The first `ruff format` pass reformatted 13 unrelated files; those were reverted to HEAD, and the 11 touched-and-unformatted files were rebuilt from HEAD with only PR-040's edits.
- **Not done here.** The real-store demonstration (bare-metal on the workbench). It needs the branch on the workbench via `git push`/`git pull`, which awaits operator approval.

### Gate Check

- Premise still valid: ✓ (refusing oracle inputs at the training-set builder survives; only the attachment point moved)
- No prerequisite PRs surfaced: ✓ (two deferred follow-ups are not prerequisites)
- User approved updated spec: ✓ (2026-09-24)
- Implementation cleared

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

**In scope.** The refusal wherever a dataset enters the workbench, plus the tests that exercise it
end to end through the data layer and the CLI. *(Rewritten at Phase 4, 2026-09-24. The stub's
attachment points, which pointed at `snapshot()` as "most likely the right home", were stale; see
Phase 1.)*

**The check.** A new module `src/rux_ml/data/quarantine.py` exports `OracleQuarantineError(ValueError)`
and `check_oracle_quarantine(path, oracle_cfg)`; both go into `rux_ml.data.__all__`. The check is
**raise-only**: it never changes the scan target, the scan options or any hash input. In order:

1. **Fail closed on missing config.** If `oracle_cfg is None`, it refuses, and the error names `[data.oracle]` in `configs/base.toml`.
2. **Reject glob sources**, meaning any source containing `*`, `?` or `[` (polars' own rule, `path_utils/mod.rs:128-133`).
3. **`expanduser`, then check two ancestor chains:** `[p.absolute(), *parents]` and `[p.resolve(), *parents]`. It refuses if `(d / tag_file).exists(follow_symlinks=False)` for any `d` in either chain. *(Implementation note: the approved spec said `os.path.lexists`; it swallows `PermissionError` and reports a tag in an unsearchable directory as absent, which fails open. `Path.exists(follow_symlinks=False)` also does not follow symlinks, and raises on `EACCES`, which the check wraps into a refusal. Test: `test_unsearchable_ancestor_fails_closed`.)*
4. **Walk directory sources.** For a directory source, run `os.walk(followlinks=True, onerror=<re-raise>)` with a visited set of `(st_dev, st_ino)` that prunes `dirnames`, the pattern at setuptools `_distutils/filelist.py:328-353`. It refuses on a tag in any visited directory. **For every symlink the walk meets (directory or file), it also checks the ancestor chain of the symlink's resolved target** (the Phase 3 amendment, approved 2026-09-24).
5. **Check column names.** It checks every file larger than 0 bytes with `pl.scan_parquet(f, glob=False).collect_schema()`, then `pl.scan_parquet(path).collect_schema()` on the source itself, which includes hive partition keys. A top-level name, or a nested `Struct`/`List`/`Array` field name, is refused if `name.casefold().startswith(namespace.casefold())`. The error names every offending column path.
6. **Wrap errors from the check itself.** `RuntimeError` (from `expanduser`/`resolve` on 3.12) and `OSError` (from the walk or `lexists`) are re-raised as `OracleQuarantineError`, so the check fails closed with a named error. A polars parse error on a non-parquet file is a pre-existing condition and propagates unchanged.

The tag steps (1-4) run before any polars call, because polars raises on a non-empty tag file first.

**Call sites.**
- **`load_parquet(path, *, oracle)`** in `data/loaders.py`. The keyword is required and has no default. This covers all five training-set builders:
  - `cli/train.py:84`
  - `tuning/objective.py:314`, which reaches in-process tune, the subprocess child and retry-trial
  - `registry/promote.py:112`
  - `registry/scorer.py:263`
  - `scripts/calibrate_pr025.py:321`
- **`compute_data_hash(path, *, oracle)`** and **`data_hashes(source_path, *, oracle)`**. This covers `train`'s pre-trial hash (`cli/train.py:194`, before `one_off_run`, so no FAIL row is written), `rux-ml data hash` and `data version`. `data version` refuses before `_link_or_copy`, so nothing is written to the CAS. **`data hash` gains a config load** (`cli/data.py:38` currently loads none).
- **Tune parent preflight.** `cli/tune.py` `start`/`resume`/`retry-trial` run the check before opening a study, dispatching a child or enqueueing a trial. `train`, the tune preflight and `data hash`/`data version` map the error to `typer.BadParameter`, which exits 2. The registry verbs already map `ValueError` to exit 2 (`cli/registry.py:46-47,121-122`).

**Config.**
- `OracleQuarantineConfig(StrictModel)` in `config/data.py`, with `namespace: str = Field(min_length=1)` and `tag_file: str = Field(min_length=1)`. A validator rejects `/`, `\`, `os.sep` and `os.altsep` in `tag_file`.
- `DataConfig.oracle: OracleQuarantineConfig | None = None`, declared on `DataConfig` itself (a subclass field is dropped from the dump; Group D).
- The values live **only** in `configs/base.toml` as `[data.oracle]`: `namespace = "oracle__"`, `tag_file = "ORACLE.tag"`, mirrored from harness `config/harness.toml:143-146`.
- `"oracle"` is added to `_HASH_ELIDED_FIELDS["data"]`, and the comment at `root.py:52-53` is updated. This widens D17's elision rule to output-neutral guard config.
- `configs/README.md` documents the table.

**Docs riding with the code** (drafted at Phase 4; they stay uncommitted until the implementation commit):
- `docs/ARCHITECTURE.md`: new decision rule "Oracle quarantine at ingest"; the D17 elision wording.
- `docs/CONSTRAINTS.md`: new "Oracle Quarantine at Ingest" domain constraint.
- `docs/CONVENTIONS.md`: the elision wording.
- `docs/0.3/DESIGN-log.md`: PR-040 session.
- `docs/0.3/RESEARCH-BACKLOG.md`: PR-040 row.
- At implementation: `README.md` (data-layer text) and the `CHANGELOG.md` `[Unreleased]` entry.

**Explicitly out of scope**, stated so the cut does not read them as omissions:

- No change to the harness side. Its half is built and evidenced; this PR mirrors the rule, it does not
  refactor the harness. Harness gaps are recorded for the lead only.
- No shared package between the repos. They are independently governed, and a common dependency
  would couple two release cycles to make two small functions identical. **Two independent
  implementations of one rule is the point**: a single shared one has a single point of failure.
  rux-ml is deliberately **stricter** than `production_load`: path normalization, symlink following, hive keys,
  nested fields, case-insensitive matching, `lexists`.
- No retroactive scan of existing CAS contents. Phase 1 found none exist on either host.
- No promote-time comparison of the re-fit `data_hash` against the trial's recorded one (decision 11; its own PR).
- No alignment of what is hashed with what is loaded (hive keys, symlinked subdirectories, `.pq`
  files). This gap existed before PR-040; it gets its own PR.
- No version bump, no cut, no roadmap row (operator ruling 2026-09-24).
- No change to `iter_parquet_files`, the `_logical_hash` scan or any hash input.

## Dependencies

None inside rux-ml. `docs/0.3/ROADMAP.md` (PR-030…PR-036) is complete and PR-039 has landed; this
PR neither depends on nor blocks any of them.

Outside rux-ml it is a **cut blocker** for `rux-capital/program` v0.2 via ACCEPTANCE C13.

## Architecture section implemented

`docs/ARCHITECTURE.md`, in three places:
- the data layer's ingest path (v0 D3 Polars lazy ingest, D9/D14 content-addressed versioning), which gains the new decision rule "Oracle quarantine at ingest"
- Configuration Architecture (D17), whose elision rule widens to output-neutral guard config
- `docs/CONSTRAINTS.md`, which gains a new domain constraint

This is a new function and one error type, not a new abstraction.

## Verification criteria

Locked at Phase 4 (2026-09-24). Every refusal test asserts `OracleQuarantineError` specifically,
never a generic `Exception`, because today's accidental polars errors would pass a generic assertion.
Tag fixtures are **non-empty** unless the case is specifically about zero-byte tags.

*Refusal (unit, through `check_oracle_quarantine`, and again through `load_parquet`):*
- [x] Top-level `oracle__x` column refused; the error names it.
- [x] Nested `Struct` / `List(Struct)` / `Array(Struct)` field `…oracle__x` refused; the error names the path.
- [x] Case-variant prefix (`Oracle__x`, `ORACLE__x`) refused.
- [x] Hive partition key `oracle__k=1/` on a directory source refused.
- [x] Heterogeneous directory where only a later file carries `oracle__y` refused by name, not by polars `SchemaError`.
- [x] Tag in the source file's own directory refused; the error names the tagged directory.
- [x] Tag several ancestor levels up refused (any depth).
- [x] Tag inside a directory source (nested subdirectory) refused.
- [x] Zero-byte tag refused, even though polars would read the directory cleanly.
- [x] Symlinked subdirectory leading into a tagged directory refused.
- [x] File symlink pointing into a tagged directory refused (resolved chain).
- [x] Symlink pointing **below** a tagged directory refused (symlink-target ancestors, the Phase 3 amendment).
- [x] Dangling-symlink tag refused (symlinks not followed). A tag in an unsearchable ancestor refuses (fails closed).
- [x] A symlink cycle terminates. An unreadable subdirectory refuses (fails closed, no silent skip).
- [x] Glob source refused. `oracle=None` refused.
- [x] Relative and `~` source paths are checked against their real ancestors.

*End to end (CLI, through the real config loader):*
- [x] `rux-ml train` on an oracle source exits 2, and no trial row is created.
- [x] `rux-ml tune start` and `tune resume` (default subprocess isolation) exit 2 before dispatching any child; 0 trials are recorded. `tune retry-trial` exits 2 and enqueues nothing.
- [x] `rux-ml registry promote` (re-fit ingest) and `rux-ml registry score` exit 2.
- [x] `rux-ml data hash` and `rux-ml data version` exit 2; nothing is written under `cas_root` or `manifests_root`.

*Clean-path regression (the regression that matters):*
- [x] Pinned golden_v1 `data_hash` is unchanged (`tests/golden/fixtures/golden_v1/manifest.json:5-8`, `11f18064…|98f8c686…`).
- [x] Pinned `data_cfg_hash` / `root_cfg_hash` are unchanged, for crypto_breakout_h3+hpo (`3c89c1f5…` / `06f13e9c…`) and for the golden config (`2d7467a5…` / `c5ba8632…`), with `[data.oracle]` both set and unset.
- [x] A clean dataset loads to an identical frame, and `snapshot()` writes an identical manifest apart from `created_at`.
- [x] Config values survive the `objective.py`/`promote.py` dump → validate rebuild, `--set data.oracle.namespace=…` and `RUXML_DATA__ORACLE__NAMESPACE`.
- [x] Validators reject an empty `namespace`, an empty `tag_file` and a `tag_file` containing a separator.

*Process:*
- [x] The namespace prefix and tag filename appear **only** in `configs/base.toml` (plus test fixtures), with no literals in `src/`.
- [x] Full default suite green (`uv run pytest`), plus `ruff check`, `ruff format --check` and `basedpyright src/`. *(At implementation: 454 passed. The 13 failures are catboost/lightgbm `ModuleNotFoundError`, identical at HEAD 679b096 (394 passed + the same 13), because those are optional extras not installed on the Mac. Golden: 7 passed, fixtures unchanged. `ruff check` is clean. `ruff format --check` passes on every file PR-040 created or that was format-clean at HEAD; 25 files were already unformatted at HEAD under ruff 0.15.13, and PR-040 leaves that drift alone. basedpyright: 0 errors outside the 16 pre-existing catboost/lightgbm unresolved imports.)*
- [ ] **Real-store demonstration** (evidence for the lead): bare-metal on the workbench, `rux-ml train --set data.source_path=<oracle-store day file>` and `--set data.source_path=<oracle-store day dir>` both exit 2 with the named error. The command, exit code and SHA are recorded in this file.
- [x] `CHANGELOG.md` `[Unreleased]` entry per `docs/VERSIONING.md` §6.
- ~~The `docs/0.x/ROADMAP.md` row flips in the same commit.~~ Dropped: no open roadmap (operator ruling 2026-09-24).

## Research backing

Tier-2 (operator ruling after Phase 1). See `## Research findings`: State Assessment, Research
Questions, Findings (Q1-Q6), Group D and Synthesis, all dated 2026-09-24.

## Notes

**The operator's steps, if any, are none.** Nothing here touches signed files, spend, live hosts or
data deletes. The real-store demonstration only reads the oracle store.

**Resolved 2026-09-24:**
- Tier-2.
- Standalone against `dev`: no version bump, no roadmap row.
- PR-037 and PR-038 exist in git history (49a3ee2, b46427f) without `prs/` files. This file keeps **PR-040**.
