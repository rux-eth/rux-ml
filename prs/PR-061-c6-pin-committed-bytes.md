# PR-061: Per-file data-hash digests agree Mac = desktop on the same bytes (the c6 pin over committed bytes)

<!-- Written 2026-10-15 from the program lead's approved draft, rux-capital program `docs/0.5/evidence/ruxml-track/PR-061-digests-draft-20261015.md`, through its Phase 5 Gate Check (build cleared by the operator's "go" ≈ 14:25 CDT). -->
<!-- Bare `:NNN` cites are this file's own lines (kept line-for-line with the draft). `pr061-p3/`, `docs/0.5/…`, `docs/0.3/ACCEPTANCE.md`, `PR-081` – `PR-084` and `c1-ruxml-20261001.json` are rux-capital program paths. -->
<!-- Build notes (the test-first runs, the checks, the Mac half of the C10 record) are appended under §Build notes at the end. -->

**Landed-in:** (not yet landed)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PRs** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

**Tier: Tier-2, light (proposed; the lead's call — gate decision 1).** This PR is a track of the rux-capital program lead. It carries program v0.5 ACCEPTANCE **C10** (`docs/0.5/ACCEPTANCE.md:104-106`): the per-file data-hash digests of one table agree Mac = desktop, recorded as `docs/0.5/evidence/ruxml-digests-<date>.json`. It is the item the v0.3 C1 waiver deferred (`docs/0.3/ACCEPTANCE.md:382-385`, carried at :360). The merge belongs to the program lead and must land on `dev` before the `prereg-v05-probe` tag (program PR-082:262).

## Research findings

### State Assessment (2026-10-15)

Baseline: rux-ml `dev` @ `09b9bd5` as read by the two Phase 1 lenses (18 tool calls each). On 2026-10-01 the desktop all-extras suite there gave **2 failed / 755 passed / 3 skipped**. The 2 failures are this PR's: `tests/cli/test_data_hash_per_file_c6.py::test_the_c6_subtree_digest_is_unchanged_on_every_path[fill]` and `[walk]` (program `docs/0.3/evidence/close-out/c1-ruxml-20261001.json:53-59`). The Mac suite is green (PR-057:15, 730 passed / 3 skipped).

**Current state**:
- **How `data_hash` is built** (proven). `data_hash = bytes_hash|logical_hash` (`src/rux_ml/data/versioning.py:56-59`).
  - `bytes_hash` = sha256 of the canonical JSON of the sorted per-file xxh3_64 digests (`_internal/hashing.py:26-44`). Only file bytes enter it; no path and no file order.
  - `logical_hash` hashes each row as `pl.struct(sorted column names).hash(seed=0)`, one file at a time, scanned with the schema of `files[0]` of a Path-sorted list (PR-053). The u64s from every file are sorted together and sha256'd as raw native-endian bytes (`versioning.py:67-118`; :91-93 schema / cols, :101-103 hash, :114-115 sort + sha256 of the memoryview; `loaders.py:46-59`). No float-to-text step, no path and no file order enter it.
  - The bridge sidecar's per-file entries are plain sha256 of each file's bytes, keyed by the path relative to the subtree (`bridge.py:78-79, :97-102, :123-136`). Fields that vary by machine (`source_path`, `computed_at`, `rux_ml_git_sha`, `polars_version`) are sidecar metadata and are not hashed. The sidecar records `polars_version` (`:134`) but **no platform or libm**.
- **Where the hash is used** (proven). `train` records it in trial `user_attrs` (`cli/train.py:574`). `promote` re-computes it and refuses on a mismatch with the trial's value or the harness set manifest's `ruxml_sidecars.<t>.data_hash`, then writes `ModelManifest.data_hash` (`registry/promote.py:311-325, :369-388, :430`; `registry/manifest.py:62-70, :153`). `rux-ml data hash` prints the composite; `data version` makes a snapshot `version_id = sha256(bytes|logical)[:16]`; `data bridge` writes the sidecar and exits 2 on a mismatch (`cli/data.py:33-44, :47-65, :91-135`; `versioning.py:226-228`; `bridge.py:152-204`). `solve` uses a placeholder (`cli/solve.py:11, :115`).
- **The harness never computes it** (proven; harness rule A13). It only copies `data_hash` and `polars_version` from rux-ml's sidecar into `ruxml_sidecars` (`rumpy_harness/training/materialize.py:35-37, :797-799, :822-834`). No harness change is needed.
- **Same bytes give the same digest on both machines** (proven). The committed golden `tests/golden/fixtures/golden_v1/synthetic.parquet` (sha256 `733f7d7d…`, committed from the desktop) hashes on the Mac today to logical `64a492714827bcd8…` and bytes `20ae5203066c6335…`, equal to the desktop-written manifest (`golden_v1/manifest.json:5-7`, reference_platform Linux/x86_64). It covers String / Float64 / Int64.
- **Polars' struct hash agrees Mac = desktop over every dtype c6 carries, and more** (proven from the run's failure list, not per-test). `PINNED_EDGE` (`three_files 1cf2a974…`, `tests/data/test_logical_hash_per_file.py:39-60, :150-167`) covers Int8, UInt64, Float64 with NaN / -0.0 / ±inf / null, Float32, nullable Boolean, nullable unicode String, Categorical, Enum, Date and tz Datetime. The desktop run's only failures were the two c6 pins, so `PINNED_EDGE` passed there. This is inferred from the aggregate count (2 failed, only c6 named), not read from the junit XML at the desktop's `~/pr029-ruxml/*.xml`.
- **What fails on the desktop is the pin, not the per-file path** (proven at `48ffd4e`; same test names at `09b9bd5`). The record at `48ffd4e` reads "got `c216d68087f8…3af01fd5aa44a`, pinned `185d2b9cf368…e472aced2d4d8`" for fill and "got `9f6ced5a9560…1a95a04dd5b9d`, pinned `81f710717d19…0adc9629bd4e7`" for walk (`c1-ruxml-20261001.json:40-41`). The per-file vs whole-source equalities (`test_data_hash_per_file_c6.py:33-37`) run before the pin (`:39`), so they passed. At `09b9bd5` the record names the same two tests without values (`:56-59`, "the same two c6 digest-pin failures"). Caveat: the "got / pinned" text may be the track agent's transcription; the junit XML would make it direct. The Mac reproduces both pins today (fill `185d2b9cf3685f35`, walk `81f710717d194c77`).
- **What the failing test pins** (proven). Only the `logical_hash` of `c6_set`'s `fill/` and `walk/` subtrees, gated on the Polars version (1.40.1) and never on the platform (`test_data_hash_per_file_c6.py:21-26, :38-39`). The pins were computed on the Mac at `9e3f0c3` (pre-PR-053 code; PR-053:21, :62). No `bytes_hash` and no per-file sha256 is pinned in `tests/cli`.
- **`c6_set` is generated, not committed** (proven). It is drawn at test time from `np.random.default_rng(0)` and written by Polars `write_parquet` (`tests/cli/conftest.py:118-131`). `_c6_day` (`:42-112`) draws `rng.gamma` at :53 (`feat__spread_bp`, and through it `y__walk_bp` / `y__half_spread_bp`) and :92 (`y__queue_ahead_at_ack`), `rng.beta` at :79 (fill), and `rng.normal` at :64 / :81. The passing `PINNED_EDGE` frame uses `rng.normal` only, at n = 120 (`test_logical_hash_per_file.py:41-43`).
- **The likely cause is the fixture's data** (best-guess, strong; ~~mechanism proven by precedent~~). ~~numpy's gamma and beta call the platform libm on most draws, macOS libm vs glibc. PR-057 proved this mechanism for the golden fixture: data generated on the desktop differed from the Mac's in float columns by up to 128 ULP (PR-057:55-57).~~ Program PR-081 Q2 found `log` differs between Apple libm and glibc (406 of 200,000 near-1 ratios by 1 ULP; `prs/PR-081-…:241-242`). ~~One flipped accept/reject step shifts the RNG stream, so every later draw differs.~~ **Corrected at Phase 4 (Q1a, Q4, Group D D2; *proven*):** the fixture's data is the cause, but the mechanism is FMA contraction. numpy 2.4.5's arm64 wheel (clang 15.0.0; FMA is in base AArch64) fuses `random_standard_gamma`'s `V = 1.0 + c*X`, and the x86_64 wheel (gcc 14.2.1, baseline X86_V2) does not; beta(2,2) inherits the difference. It is not libm: libm enters gamma only in the accept/reject `log`, which decided none of the differing draws. It is not a stream shift: the PCG64 state is equal after all 210 calls. PR-057 did not prove a libm mechanism; it labels its golden-data cause "best guess, unproven" (PR-057:59). PR-081 Q2's `log` finding holds for PR-081's own table, not for c6. Mac per-file sha256 prefixes recorded for the comparison: fill `4a9f5b21d46eff2f` / `0d0fcbc79af6315b` / `c49c9b30a357e15c`; walk `4cfddc6b59dce6d3` / `3c0d2e3a23b59ba8` / `48bc780cb309f6cf`. ~~The counter-case is not excluded until Q1 runs.~~ Excluded once Q1 and Q2 ran.
- **Polars' own documentation** (proven): `Expr.hash` is not guaranteed stable across Polars versions (Mac probe of `pl.Expr.hash.__doc__`). `pyproject.toml:18` sets only a floor `polars>=1.40`; `uv.lock:1111-1112` pins 1.40.1; the desktop ran `uv run --frozen` with 1.40.1 (`c1-ruxml-20261001.json:14-17`).
- **`c6_set` is shared** (proven). `conftest.py` plus 11 test files use it: `test_data_hash_per_file_c6`, `test_label_quarantine_cli`, `test_m9_a7_honesty_forms`, `test_m9_batch_train_cli`, `test_m9_batches`, `test_m9_bridge_cli`, `test_m9_oos_export`, `test_m9_promote_batches`, `test_m9_promote_cli`, `test_m9_promote_manifest_refit`, `test_m9_train_cli`. PR-060's files are among them (PR-060 draft:52-53, :123).
- **Prior art**: PR-053 (the per-file logical hash and the c6 pin), PR-057 (a platform-dependent golden check resolved by naming the desktop as the reference platform, checked before the fit, no silent skip; PR-057:1, :22, :27-29). rux-ml `CONSTRAINTS.md:105` forbids exact-hash tests only for GPU-trained model outputs, so an exact data-hash pin is legitimate.

**Assumptions at PR draft time**:
- "A machine-dependent data hash would break cross-machine reproducibility"; the pinned digests differ "under the same Polars 1.40.1 (platform-dependent; cause not investigated)" (program `docs/0.3/ACCEPTANCE.md:382-385`).
- The rux-ml track row names "the machine-independent per-file data-hash digests" as work to do, i.e. a code fix to the digest (program `docs/0.5/ROADMAP.md:31`).
- One rux-ml row covers the digests and the early-stopping item, sized ≈ 0.3–0.4M together (ROADMAP:31).
- C10's "the per-file digests of one table agree Mac = desktop" (ACCEPTANCE:104-106) is unambiguous.

**Stale assumptions** (where current state disagrees):
- **The hash is very likely already machine-independent.** Same bytes give the same digest (golden), and every c6 dtype hashes the same on both machines (`PINNED_EDGE`). The evidence points to machine-dependent fixture data. The PR may be test-only, with no change to the digest code. Not proven until Q1.
- **"Per-file" in the test name comes from PR-053's one-file-at-a-time logical hash**, not from a cross-machine property (PR-053:62; `test_data_hash_per_file_c6.py:1-3`).
- **The early-stopping item has left this row.** It is PR-060's lean (3) (PR-060 draft:18, :123). ROADMAP:31's size is for both items and must be re-stated for this PR alone.
- **C10's wording is ambiguous in a way that decides whether it can be met.** If "one table" means the same file bytes hashed on both machines, the per-file sha256 and `bytes_hash` agree by construction and only `logical_hash` is a real test. If it means the table rebuilt on each machine, PR-081 Q2 makes C10 unsatisfiable by design (log-based features differ by ULP; PR-081:241-245, :129). The same wording sits in program PR-083:56-57 and PR-084:34-35.
- *(Cosmetic, not this PR's unless the lead adds it)*: rux-ml `README.md:256, :595` describes `logical_hash` as "SHA-256 over the canonical column projection"; the code is a sorted u64 row-hash digest (`versioning.py:70-72`). `CLAUDE.md:48, :52` still says "No code yet" and points at `docs/0.1`.

**New constraints**:
- **Must not move** (proven by code): the `bytes_hash` and `logical_hash` definitions, the composite format `bytes|logical`, the sidecar keys `data_hash` / `polars_version` / `files[{path, sha256}]`, and the existing pins (`PINNED_EDGE`, the golden manifest, and `PINNED_C6` if the Mac bytes are committed) (`versioning.py:56-118, :171-177`; `bridge.py:124-136`; `test_logical_hash_per_file.py:158-161`; `test_data_hash_per_file_c6.py:23-26`). A definition change would move every recorded value: trial `user_attrs`, `ModelManifest.data_hash`, the harness set manifests' `ruxml_sidecars.<t>.data_hash` (promote and `check_bridge` would refuse), snapshot `version_id`s, the golden manifest and both pinned sets (`promote.py:320-325, :384-388`; `bridge.py:191-203`; `versioning.py:226-228`).
- **`_c6_day` / `c6_set` stay untouched.** Rewriting them changes the data under 11 test files, including PR-060's, and breaks the "files disjoint" basis of PR-060's decision 1. The fix keeps to the pin test or its own fixture.
- **The harness stays out** (A13; `materialize.py:35-37`). The fix and the C10 evidence run in rux-ml's environment.
- **Tests first.** The new pin over committed bytes must fail before the fixture is committed (user global rule).
- **Git is the only sync route** between Mac and desktop (program CLAUDE.md §Environment; never rsync / scp). Hashing one set of bytes on both machines therefore means a committed file set.
- **rux-ml Phase 3 requires a web search for every must-answer question** (`PROCEDURE-pr-research.md:80`). The Polars hash-stability question cannot be closed from code reading alone.
- **Merge order.** Both rux-ml PRs land on `dev` before PR-082's tag, because the freeze records rux-ml's commit and any later change is a new trial (PR-082:262; PR-060 draft:97, :109); and before PR-083's probe run (PR-083:56-57).
- **rux-ml has no hosted CI.** Verification is local on the Mac, then on the desktop diffed against `dev` (PR-060 draft:96).

**Downstream contracts**:

| Consumer | Contract | Satisfiable |
|---|---|---|
| Program ACCEPTANCE C10 | The per-file data-hash digests of one table agree Mac = desktop; the PR is on `dev`; the record `docs/0.5/evidence/ruxml-digests-<date>.json`. | **yes, under the same-bytes reading** (gate decision 2): already shown for the golden table; a committed c6 table hashed with `rux-ml data hash` on both machines closes it. **No** under the rebuild reading (PR-081 Q2). |
| Program PR-083 | "Every cell's provenance must reproduce Mac = desktop" before the probe run. | **partly.** `data_hash` reproduces on identical bytes. The table build cannot (PR-081 forces desktop-only), and fitted-model outputs cannot with `colsample_bytree < 1` (PR-057:33-47, :112). PR-083's wording should say "hashed on both machines". |
| Program PR-084 | The trained model file's sha256 equal on Mac and desktop, with C10's digests as an input. | **not through this PR**, and likely not as written (PR-057:22). C10 supplies only the table's `data_hash`. |
| rux-ml promote (re-fit + harness manifest compare) | The re-fit `data_hash` equals the trial's and the set manifest's. | **yes**, if the digest definition is untouched (the lean). |
| rux-ml `check_bridge` / harness `ruxml_sidecars` | Files' (path, sha256) and `data_hash` equal the harness set manifest's. | **yes.** sha256 is machine-independent; `data_hash` is unchanged under the lean. |
| Harness A13 | rux-ml computes `data_hash`; the harness only copies it. | **yes.** No harness change. |
| Program PR-081 probe set | Set hash = sorted (relpath, sha256) plus manifest content fields; kept separate from rux-ml's `data_hash`. | **yes.** rux-ml's per-file sha256 equals the set's per-file sha256 for the same bytes; `bytes_hash` is a different digest (xxh3) of the same files. |
| rux-ml PR-060 (route, in parallel) | ~~Files disjoint~~ No shared code or test file; the three docs files (CHANGELOG `[Unreleased]`, `docs/ARCHITECTURE.md`, `docs/0.3/RESEARCH-BACKLOG.md`) take append-only lines from both, and whichever PR merges second rebases its lines *(corrected at Phase 4: PR-060 draft :95, :156, :194 edit the same three docs files)*; both on `dev` before the tag; PR-060's suite diffed against `dev`'s 2 c6 failures unless this PR lands first. | **yes**, if this PR stays out of `_c6_day` (constraint above). |
| The v0.3 C1 waiver | "A rux-ml investigation" of why the pinned digests differ. | **yes.** The Q1 spike names the cause at no spend. |

**Must-answer, carried to Phase 2** (from the lenses):
- **Q1 (decisive).** Do the desktop's generated c6 parquet bytes and values differ from the Mac's, and in which columns? One read-only desktop run at `dev` in a scratch dir under the mkdir heavy lock: build c6 with the fixture loop (`conftest.py:122-130`), print each file's sha256 and the first differing value per column against the Mac's. Fold in a `-rA` run of the one test at `09b9bd5` to confirm which assertion fails there. ~~Columns from gamma / beta confirm the libm mechanism.~~ Gamma / beta columns were found, but the mechanism is FMA contraction, not libm (Q1a; corrected at Phase 4).
- **Q2.** Does the Mac-generated c6, committed on a scratch branch and hashed on the desktop, give `185d2b9c…` / `81f71071…`? This is the C10 evidence itself.
- **Q3 (web, mandatory).** Is Polars 1.40.1's `struct(...).hash(seed=0)` bit-stable between linux-x86_64 and darwin-arm64? Polars docs and issues on platform stability and the hasher used, plus Q2 as the empirical check.
- **Q4 (web).** numpy's gamma / beta dependence on the ~~platform libm~~ platform (answered: arithmetic contraction in the wheel's build; libm enters only in comparisons and the normal tail — Q4; corrected at Phase 4), from numpy's documentation or source, to back the mechanism Q1 observes.

### Path-tier checkpoint

**Tier-2, light** (proposed). Drift was found, and it narrows the premise: the work is probably a test-fixture fix plus a cross-machine proof, not a hash redesign. That does not reverse v0.5's design or C10 (C10 stays satisfiable under the same-bytes reading), so there is no loop to design-planning. Tier-2 because the cause is still a best-guess: it has not been run (PR-TEMPLATE:18; the user's rule to assume there is a bug; PR-060 draft decision 1). "Light" because Phase 3 is two desktop spikes (Q1, Q2) and two Tier-A web checks (Q3, Q4), with no harness research run. If Q1 shows identical bytes on both machines, the hash differs on c6 data, the fix would move every stored `data_hash`, and the PR becomes a full Tier-2 and returns to the lead before any build.

### Gate leans (one decision each)

1. **Tier.** *Lean: Tier-2, light* — run the Q1 / Q2 spikes and the two web checks before the PR is written in final form. *Alternative:* Tier-1 — accept the fixture-data cause from the evidence above and go straight to the build (lens 1's lean if Q1 confirms; the faster path, but it builds on an untested cause).
2. **What C10's "one table" means, and which table.** *Lean: the same file bytes hashed on both machines, on a committed copy of c6's `fill/` and `walk/` subtrees.* It needs no other PR and can be moved by git. *Alternatives:* (a) the PR-081 desktop-built probe table hashed on both machines — closer to the probe, but getting its bytes onto the Mac has no route yet (git only, never rsync); (b) the table rebuilt on each machine — unsatisfiable by design (PR-081 Q2). The wording of C10, PR-083:56-57 and PR-084:34-35 is the lead's.
3. **The shape of the fix.** *Lean: no change to the digest code.* A dedicated committed fixture for the pin test; the test pins each committed file's sha256 (to catch data drift) and the logical digest; `_c6_day` and `c6_set` stay as they are. *Alternatives:* (a) a per-platform pin (two values per subtree — admits a machine-dependent hash instead of testing for one); (b) a named reference platform as in PR-057 (fits a model whose output truly differs; the hash need not); (c) rewrite `_c6_day` with draws exact on every platform (rejected: moves data under 11 test files including PR-060's).
4. **Which bytes are committed.** *Lean: the current Mac bytes,* so `PINNED_C6` does not move and no recorded value changes; committed bytes make the generating platform irrelevant. *Alternative:* desktop bytes, following PR-057's reference-platform convention; `PINNED_C6` then changes to `c216d680…` / `9f6ced5a…`.
5. **Where platform and libm are recorded.** *Lean: in the C10 record only* (the evidence JSON names machine, platform, libm, Polars / numpy versions and rux-ml commit beside each digest). No sidecar change. *Alternative:* the sidecar gains a platform field (`bridge.py:134`) — a code change to a file the harness reads; PR-081 Q2 asks for platform / libm in its own manifest, not rux-ml's.
6. **The Polars-version dependence.** *Lean: out of scope, recorded as a follow-up in Notes.* *Alternatives:* an exact `polars` pin in `pyproject.toml`, or `check_bridge` / promote comparing `polars_version` so a version bump is refused by name instead of as a bare `data_hash` mismatch.
7. **An explicit little-endian cast in the logical digest.** *Lean (lens 1): yes* — `astype('<u8', copy=False)` at `versioning.py:115, :144`, a no-op on both little-endian machines, with a test that `PINNED_EDGE` and `PINNED_C6` still reproduce. It makes the digest well-defined on any host. *Alternative:* defer as non-blocking (both machines are little-endian; one fewer change before the tag).
8. **Number and merge order.** *Lean: PR-061.* Land it before PR-060 if it is ready first, so PR-060's desktop suite starts clean; otherwise PR-060's gate diffs against `dev`'s 2 c6 failures. Both before the tag. *Alternative:* fold into PR-060 (ROADMAP:31 as written; against the smaller-PRs ruling and PR-060's decision 1). The operator orders the merges.

### Gate Check (Phase 1 → 2 – 3, 2026-10-15)

- **APPROVED 2026-10-15 ≈ 12:40 CDT, the program operator's "go"** on the lead's go-list of ≈ 12:00 CDT: Tier-2 light; leans 1 – 8 as
  stated above (Tier-2 light; C10's "one table" = the same bytes hashed on both machines, on a committed c6 copy; no digest-code change; the
  Mac bytes committed; platform / libm in the C10 record only; the Polars-version dependence a follow-up; the little-endian cast yes; PR-061,
  merge order by the operator). Phases 2 – 3 run now (Q1 / Q2 one desktop spike under the mkdir lock, Q3 / Q4 Tier-A web), halt at Phase 4.
  If Q1 finds identical bytes on both machines, the PR becomes full Tier-2 and returns to the lead before any build. The program texts
  (C10, ROADMAP:31, PR-083:56-57, PR-084:34-35) are re-worded at this PR's Phase 4 synthesis, not now. Estimate ≈ 0.4M, $0.

### Research Questions (Phase 2, 2026-10-15)

Ranked from the State Assessment's must-answer list. No question was added unless an answer raised one; those are marked **(new — raised by Qn)**.

**Must-answer:**
1. **Q1 (decisive).** Do the desktop-generated c6 parquet bytes and values differ from the Mac's, in which columns, and which assertion fails at `09b9bd5`?
   - *Why:* it separates machine-dependent fixture data (a test-only fix) from a machine-dependent hash (a digest change, full Tier-2).
   - *Success criterion:* per-file sha256 on both machines; the first differing value per column; the failing assertion from `-rA` at `09b9bd5`. Identical bytes ⇒ full Tier-2, back to the lead.
   - *Tier / how answered:* a read-only desktop spike in scratch under the mkdir heavy lock. `c6spike.py` calls `tests.cli.conftest._c6_day` with the `default_rng(0)` loop of `conftest.py:122-130` on both machines, then a column compare and `pytest -rA` of the one test. The web leg runs through Q4.
   - **Q1a (new — raised by Q1 / Q4).** Is the value difference libm with a flipped accept/reject shifting the stream (the draft's :39), or something else? *Success:* the RNG state compared after every call, and each differing call reproduced bit-exactly by a named arithmetic model on each machine (`mech.py`, Mac + desktop).
   - **Q1b (new — raised by Q1a).** What causes the two 1-ULP normal-tail differences? Non-blocking.
2. **Q2.** Do the Mac-generated c6 bytes, hashed on the desktop, give `185d2b9c…` / `81f71071…`?
   - *Why:* this is C10's evidence under the same-bytes reading (lean 2), and it tests lean 4.
   - *Success criterion:* `logical_hash` = `PINNED_C6` by every route (CLI, `compute_data_hash`, per-file, whole-source), with `bytes_hash` and per-file sha256 equal.
   - *Tier / how answered:* the same desktop spike. The web leg runs through Q3.
3. **Q3 (web, Tier A).** Is Polars 1.40.1's `struct(...).hash(seed=0)` bit-stable between linux-x86_64 and darwin-arm64?
   - *Why:* it explains Q2 rather than only observing it, and decides whether lean 3 rests on code or on chance.
   - *Success criterion:* the hasher and its seeding named at the pinned tag, the platform branches listed, any documented guarantee quoted, and the issue search recorded.
   - *How answered:* Polars source at `py-1.40.1`, foldhash 0.2.0, the xxHash README, a Mac probe, `gh search issues` ×5, and a web search.
   - **Q3a (new — raised by Q3).** Which float bit patterns can `logical_hash` not see?
4. **Q4 (web, Tier A).** Where do numpy 2.4.5's gamma / beta / normal depend on the platform?
   - *Why:* it backs Q1's mechanism and names the cause that the CHANGELOG (Scope 5) and the C10 record (lean 5) carry.
   - *Success criterion:* the libm and arithmetic sites in `distributions.c` at `v2.4.5`, numpy's own compatibility statement, and a Mac replica.
   - *How answered:* numpy source and docs, GitHub search ×4, one secondary arXiv paper, and the Mac `probe.py` at N = 20,000.

**Dependencies:** Q1 and Q2 share one desktop spike under one lock but give independent results. Q4 interprets Q1, Q3 explains Q2, and Q1a depends on Q1 and Q4.

**Explicitly excluded from this round:**
- The ARCHITECTURE section (:137, "locate it at Phase 2"): not located; carried to Phase 4.
- The program texts: Phase 4, per the Gate Check.
- `PINNED_EDGE` per test from the junit XML: :35 stays inferred; it is the build's criterion :144.

### Findings (Phase 3, 2026-10-15)

*Evidence kept (the lead, 2026-10-15):* `docs/0.5/evidence/ruxml-track/pr061-p3/` — `desktop-run.log` (the desktop spike's log: the cmp, hash, rewrite and pytest blocks), `mac-hash.json`, the scripts `c6spike.py` / `mech.py` / `remote_run.sh` / `probe.py`. The parquet / npz data and the downloaded sources stay in the session scratch (regenerable). The lead re-read the log's "hash Mac bytes on desktop" block: fill `185d2b9c…` = `PINNED_C6`. The Mac bytes went to the desktop as data over ssh stdin — inside the standing rule (git or ssh stdin; never scp / rsync), no ruling needed.

**Sources.** Two Phase 3 agents ran 15 tool calls each: the desktop spike (Q1 / Q2) and Tier A (Q3 / Q4). Scratch is only under `/private/tmp/claude-501/-Users-maxrux-projects-rux-capital-program/6e62687f-560b-472e-8a0c-25642e3c1e90/scratchpad/pr061-p3/` (`desktop/`, `q3q4/`). The writer re-read the digest values below in `desktop/desktop-run.log` (:15-89) and `desktop/mac-hash.json`; they equal the agents' reports. By the cost model the spend is ≈ 2 × (68k + 3k × 15) + this writer ≈ 0.33M, against the gate's ≈ 0.4M. The actual figure is the lead's to report.

**Q1 — the desktop's c6 differs from the Mac's in values only.** *Proven.*
- **Scope of the difference.** All 6 files differ (fill and walk, 2026-02-16 → 02-18). Schema and row counts are equal: 6,912 rows per subtree. What differs is float values in gamma- / beta-fed columns, plus one normal-tail value. Rows and ULP are per day out of 2,304:
  - **fill:**
    - `feat__spread_bp` (gamma, `conftest.py:53`): 243-255 rows, ≤ 9 ULP.
    - `y__queue_ahead_at_ack` (gamma, :92): 198-217 rows, ≤ 7 ULP.
    - `y__fill_frac`, `_h60000`, `_h600000`, `_h3600000` (beta, :79): 223-256 rows, up to 576 ULP, from cancellation in `clip(beta − 0.2)`. The beta draw itself differs by ≤ 9 ULP.
  - **walk:**
    - `feat__spread_bp`: 261-269 rows, ≤ 7 ULP.
    - `y__half_spread_bp`: 237-241 rows, ≤ 7 ULP.
    - `y__walk_bp`: 166-179 rows, ≤ 16 ULP.
  - **one normal:** fill 2026-02-18 `feat__trade_n_300s` row 2236 is Mac `-3.738052221422189` vs desktop `-3.7380522214221896`, 1 ULP, a ziggurat-tail draw (|x| > 3.654).
  - Every other column is equal in every file.
- **First differing values:**
  - fill 02-16 `feat__spread_bp` row 12: `0.9942677228750104` / `0.9942677228750106`.
  - `y__fill_frac` row 14: `0.25691330525732753` / `0.25691330525732736`.
  - `y__queue_ahead_at_ack` row 0: `454.0598130064817` / `454.0598130064815`.
  - walk 02-16 `y__walk_bp` row 55: `0.5112038279597803` / `0.5112038279597804`.
- **Per-file sha256 prefixes.** These close :171's "not measured".
  - Desktop: fill `f535a77c` / `906c50ec` / `968ca5d7`; walk `cdc14eb7` / `ee67c7c8` / `a448bfe4`.
  - Mac: fill `4a9f5b21` / `0d0fcbc7` / `c49c9b30`; walk `4cfddc6b` / `3c0d2e3a` / `48bc780c`. These equal :39.
- **Q1c: the failing assertion at `09b9bd5`.** *Proven.* This makes :36 direct rather than a transcription.
  - Desktop `pytest -rA`: 2 failed in 0.85 s, both at `test_data_hash_per_file_c6.py:39` (the pin). The equalities at :33-37 pass.
  - Got: fill `c216d68087f8444606d8dfa3fea09076af78e8943f86a244cdd3af01fd5aa44a`, walk `9f6ced5a9560c617892daa87ba916fbdb5daa526c751657f1d21a95a04dd5b9d`. These are exactly the desktop's own digests. Pinned: `185d2b9c…` / `81f71071…`.
  - Run conditions: Linux, Python 3.12.13, pytest 9.0.3, `-p no:cacheprovider`, `-B`. The base tree's `git status` was empty afterwards.
- **Q1d: the parquet writer.** *Proven.* Desktop Polars 1.40.1 re-wrote each Mac file byte-identically (6 / 6 sha256 equal). So the byte difference is wholly the values, and equal frames write equal bytes on both machines.
- **Evidence:** `desktop/desktop-run.log` (blocks `== cmp`, `== pytest -rA`, `== rewrite`) and `desktop/scripts/c6spike.py`. On each machine an instrumented re-run gave frames equal to the written files (`instrumented_frames_equal_written: true`).

**Q1a — the cause is FMA contraction in numpy's arm64 build, not libm, and the stream never shifts.** *Proven.* ⚑ 1: this refutes the draft's :39.
- **The stream.** The PCG64 state after each of the 210 rng calls is equal on both machines (calls: choice 15, gamma 9, normal 174, beta 3, integers 6, random 3). Exactly 14 calls differ in output: all 9 gamma, all 3 beta, and 2 normal calls with one element each at 1 ULP. No accept/reject test flipped.
- **The replica test.** Each of the 12 gamma / beta calls was replayed from the generator state just before it, in two pure-Python copies of numpy's Marsaglia–Tsang (`V = 1.0 + c*X`). One uses plain IEEE arithmetic. The other rounds `1 + c*X` once, which is an exact fma via `Fraction`. Both use the machine's own `math.log` for acceptance and end in numpy's state.
  - **Mac** (arm64; numpy 2.4.5 wheel built with clang 15.0.0, baseline NEON / ASIMD): the fma copy equals numpy on 12 / 12 calls. The plain copy differs on 241-410 of 2,304.
  - **Desktop** (x86_64; wheel built with gcc 14.2.1, baseline X86_V2, no FMA): the plain copy equals numpy on 12 / 12. The fma copy differs on exactly the Mac's counts (246 / 368 / 241 / 255 / 376 / 266 / 243 / 410 / 258 / 269 / 264 / 261), and these are exactly the cross-machine `n_diff` per call.
- **Conclusion.** The cause is how numpy's arm64 wheel is compiled (contraction under clang, with FMA in base AArch64), not Apple libm vs glibc. beta(2,2) = Ga / (Ga + Gb) inherits it.
- **Evidence:** `desktop/scripts/mech.py` (`probe`, `drawsall`, `env`).
  - The desktop probe ran under the lock at 12:46:36 CDT.
  - `drawsall` gave `pcg64_state_after_every_call_equal: true`.
  - The differing calls are c001 / c036 / c038 / c073 / c075 / c110 / c112 / c145 / c178 (gamma), c032 / c069 / c106 (beta), and c097 / c179 (normal).
  - `np.show_config` was read on both machines.
- **Disconfirming check.** Q4 predicted rare stream shifts through a comparison flip. None occurred in c6's 210 calls.

**Q1b — the two normal-tail 1-ULP values are unattributed.** *Best-guess.*
- Calls c097 / c179, one element each; the visible one is the `feat__trade_n_300s` value above.
- numpy's tail value (Q4 quotes it at `v2.4.5`: `ziggurat_nor_r + xx`, `xx = −ziggurat_nor_inv_r · npy_log1p(−next_double())`) can differ through FMA contraction of that expression or through libm `log1p`. The two were not separated.
- Non-blocking: 2 of ≈ 400k normal draws, and the remedy is the same (committed bytes).

**Q2 — the Mac's bytes hash on the desktop to exactly the pins.** *Proven.*
- **Transport.** The Mac c6 bytes went to the desktop as a tar stream over ssh stdin (⚑ 3). `sha256sum -c` was OK for the 6 parquet files, `draws.npz` and the script.
- **Results.** Four routes agree on both machines: `rux-ml --config <repo>/configs/base.toml data hash`, `compute_data_hash`, `_logical_hash` (per-file) and `_logical_hash_whole_source`.
  - fill: `logical_hash` `185d2b9cf3685f3544885bc30987fc67cddc137763bf06d61ebe472aced2d4d8`, `bytes_hash` `36f6cfa6fa0deccd4197e1de9e587cc287b2c851b2d356b82d9942d56cda614d`, rows 6,912.
  - walk: `logical_hash` `81f710717d194c77357005b7a347a8f3bad1dc54627c904c8080adc9629bd4e7`, `bytes_hash` `e48d49ecc1bcb2e00c6c49d5c996b973326b3125203e63157c9b87193eed3f3c`, rows 6,912.
  - Both equal `PINNED_C6` (`test_data_hash_per_file_c6.py:22-25`), and the per-file sha256 values are equal (`mac-hash.json`).
- **What it shows.** The digest is machine-independent between darwin-arm64 and linux-x86_64 on c6's dtypes (Float64 with NaN / null, Int64, Int8, String, Boolean). C10's same-bytes reading is met for c6. This is a preview: the record proper is the build's run on the git-committed fixture (Scope 4).
- **Environments.** *Proven* (`mech.py env`, `sw_vers`, `otool -L`, `ldd`; ≈ 12:43-12:46 CDT).
  - **Mac:** macOS 27.0.1 (26A434), arm64, Python 3.12.15. numpy 2.4.5 (clang 15.0.0; NEON / NEON_FP16 / NEON_VFPV4 / ASIMD), polars 1.40.1, pyarrow 24.0.0. libm: Apple `libSystem.B` 1345.120.2. rux-ml `dev` `09b9bd5`, clean tree.
  - **Desktop** (rux-MS-7D89): Ubuntu 22.04, Linux 6.8.0-138-generic x86_64, glibc 2.35, Python 3.12.13. numpy 2.4.5 (gcc 14.2.1; baseline X86_V2, X86_V3 found at runtime), polars 1.40.1, pyarrow 24.0.0. libm: `/lib/x86_64-linux-gnu/libm.so.6`. rux-ml `~/pr060-suite/base` `09b9bd5`.
  - Both machines are little-endian.
- **Desktop hygiene.** *Proven.*
  - Contention checks at 12:43, 12:44 and 12:46 CDT found no qBittorrent and no `Runner.Worker`.
  - `~/rumpy-heavy.lock` was taken by mkdir twice (12:44:39–12:44:48 and 12:46:36–12:46:37 CDT), each time released by `rmdir` in an EXIT trap, and is gone afterwards.
  - `~/pr060-suite/base` was used read-only; its `git status` was empty afterwards. `~/projects/rux-ml` and the harness were not touched. Spike files are only in `~/pr061-spike/`, which already existed, empty since 2026-10-10 15:26 CDT.
  - No repo edit, no real data, no fit, no push; `RUMPY_HARNESS_DATA_ROOT` was never exported.

**Q3 — Polars' struct hash is platform-independent on x86_64 / aarch64 by its code, not by any promise.** *Proven* (source at `py-1.40.1`, the Mac probe, and the cross-machine pins). *Convention:* there is no documented cross-platform guarantee.
- **Call path.** `versioning.py:101-103` calls `pl.struct(cols).hash(seed=0)`, so k0 = k1 = k2 = k3 = 0.
  - Polars `crates/polars-expr/src/dispatch/misc.rs:351-365` computes `PlFixedStateQuality::default().hash_one((k0, k1, k2, k3))` and passes it to `PlSeedableRandomStateQuality::seed_from_u64(seed)`.
  - `polars-utils/src/aliases.rs` maps these to `foldhash::quality::{FixedState, SeedableRandomState}`, and `seed_from_u64` is `with_seed(seed, SharedSeed::global_fixed())` (:41-44).
  - `Cargo.toml:57` sets `foldhash = "0.2.0"`. The hasher is foldhash, not ahash; the `ahash-0.8.12` string in the Mac binary belongs to another dependency.
- **Seeding is deterministic.** foldhash `seed.rs:11` defines `FIXED_GLOBAL_SEED` from compile-time constants.
- **How each dtype is hashed** (`polars-core/src/hashing/vector_hasher.rs`):
  - Numerics: `hash_one(v.to_total_ord())`.
  - Columns are combined by `hash_one(..) ^ folded_multiply(*h, MULTIPLE)` or by `_boost_hash_combine`.
  - String / binview: `xxh3_64_with_seed` (xxhash-rust 0.8.15).
  - Boolean: `hash_one(bool)`.
  - Null: `hash_one(hash_one(3188347919usize))`.
- **Platform branches.** foldhash's `folded_multiply` takes the same u128 branch under `cfg(any(target_pointer_width = "64", target_arch = "aarch64", target_arch = "x86_64"))`. Only 32-bit targets differ. Nothing depends on CPU features, and usize is 64-bit on both machines. The xxHash README (:7-8) says it "produces identical hashes on all platforms, including little- and big-endian systems".
- **What is documented** (*convention*, two sources):
  - The Polars 1.40.1 docstring says the hash "does not guarantee stable results across different Polars versions. Its stability is only guaranteed within a single version." It says nothing about platforms.
  - foldhash `lib.rs:5-12` says not to use foldhash if "you expect foldhash to have a consistent output across versions or platforms, such as for persistent file formats".
- **Q3a: float canonicalisation.** *Proven.*
  - `polars-utils/src/total_ord.rs` `canonical_f64` maps −0.0 to 0.0 and every NaN to `0x7ff8000000000000`.
  - On the Mac, 0.0 and −0.0 hash equal, and quiet, payload, signalling and negative NaNs hash equal, also inside `pl.struct`.
  - So `logical_hash` cannot see the sign of zero or NaN payloads. The composite's `bytes_hash` does see them, and every other Float64 bit change moves the row hash.
- **Empirical agreement.** The golden fixture is equal Mac = desktop (:34). `PINNED_EDGE` passed on the desktop (inferred, :35). Q2 checked c6 directly.
- **Disconfirming search.** `gh search issues --repo pola-rs/polars` ×5 ("hash aarch64", "hash platform different", "hash arm64 x86", "hash stable across", "hash deterministic machines") returned 0 hits, and a web search found no report (*best-guess* that none exists). The ahash README's "different computers … will observe different hash values" is moot, because ahash is not on this path.
- **Evidence:**
  - `q3q4/probe.py`.
  - Polars `py-1.40.1` source: the files above plus `polars-utils/src/hashing.rs:7,12`, `Cargo.toml:110` and `Cargo.lock`.
  - foldhash v0.2.0 and the xxHash README.
  - rux-ml `versioning.py:91-115`.

**Q4 — numpy's gamma / beta values depend on the machine through arithmetic contraction; libm enters only in comparisons and the normal tail.** *Proven* on the Mac and in the numpy `v2.4.5` source. The desktop half, a strong best-guess in Q4 alone, is *proven* by Q1a's desktop replica.
- **Source** (`distributions.c` @ `v2.4.5`):
  - `random_standard_gamma`, shape ≥ 1 (c6 uses 2.0): returns `b*V` from pure arithmetic (`V = 1.0 + c*X; V = V*V*V`). `sqrt` is correctly rounded, `log` only decides accept/reject, and `pow` appears only for shape < 1.
  - `random_beta(2,2)` = `Ga / (Ga + Gb)`; Johnk's branch needs a, b ≤ 1.
  - Ziggurat normal: about 99.3 % of draws return `x = rabs * wi_double[idx]` without libm. The wedge's `exp` is only a comparison. Only the tail returns a value derived from libm.
- **Mac replica** (`probe.py`, N = 20,000):
  - gamma(2,1): with fusion, 20,000 / 20,000 equal; without, 2,190 differ (10.95 %, ≤ 8 ULP).
  - beta(2,2): with fusion, 20,000 equal; without, 3,297 differ (16.5 %, ≤ 11 ULP).
- **Build flags.** The random library gets no contraction-off flag (`_core/meson.build:571` `staticlib_cflags = []`; none in `random/meson.build`). The x86_64 default baseline is `X86_V2` (`meson_cpu/meson.build:98-100`); `-mfma` belongs only to `X86_V3` (`x86/meson.build:13-27`), and the random code is not CPU-dispatched. So a default x86_64 wheel cannot fuse, as Q1a confirmed.
- **c6 columns exposed** (`conftest.py:42-131`):
  - **Directly:** `feat__spread_bp` in fill and walk; in fill, `y__queue_ahead_at_ack` and every `y__fill_frac*`; in walk, `y__half_spread_bp` and `y__walk_bp`. This is exactly Q1's list.
  - **Through thresholds,** only within a few ULP of a boundary: `filled`, `fill >= 0.8`, the masks of `y__markout_bp` / `y__capture_bp`, `y__ttf_*` and `y__through`. None were seen in Q1.
  - **Not exposed:** the `rng.normal` columns, apart from the tail (Q1b). With loc 0 or a power-of-2 scale, fused and unfused results are equal; the Mac identity check is True for all four.
- **numpy's documentation** (*convention*; `numpy.org/doc/stable/reference/random/compatibility.html`):
  - It promises the same stream only "on the same build of numpy, in the same environment, on the same machine".
  - It warns that "different CPUs implement floating point arithmetic differently, and this can cause differences in certain edge cases that cascade to the rest of the stream".
  - It names neither libm nor FMA.
  - A secondary source, arXiv 2605.05099 (Randompack), says bit-identical gamma / ziggurat across platforms needs OpenLibm plus `-ffp-contract=off`. It agrees with this mechanism.
- **Disconfirming search.** GitHub numpy/numpy ×4 (fp-contract, fma random, aarch64, gamma arm64) found no specific issue. The "cascade" numpy's text warns of did not happen in c6 (Q1a).
- **Magnitude.** c6's draws differ by ≤ 9 ULP, but the 20,000-draw beta replica reaches 11 ULP. The CHANGELOG should say "ULP-level", not "≤ 9 ULP".

**Options and recommendation.** Q1–Q4 bear on one decision, the shape of the fix, so they are weighed together.
- **A — commit the Mac bytes as the pin's fixture, with no digest change** (leans 3, 4).
  - Pros: Q2 proves the pins reproduce on the desktop from these bytes, so `PINNED_C6` and every recorded value stay. Q1d proves the bytes are stable on re-write. The numpy build and libm leave the test.
  - Cons: 6 committed files (~~size not measured~~ 3,540,091 bytes, measured at Phase 4). The other 11 c6 consumers keep generated, ULP-different data, but none of them pins a hash (:37, :41).
- **B — make generation exact everywhere** (rewrite `_c6_day`, or a numpy built with `-ffp-contract=off`).
  - Cons: it moves the data under 11 test files, including PR-060's (:59). The wheel build is outside rux-ml's control. Q1b stays open.
- **C — per-platform pins or a reference platform** (lean 3 alternatives a / b).
  - Cons: it admits a machine-dependent hash that Q2 / Q3 show does not exist.
- **D — change the digest,** for example by quantising floats.
  - Cons: it moves every stored `data_hash` (:58), and Q2 shows it is not needed.
- **Recommendation: A.** *Proven* (Q1, Q1a, Q1d, Q2, Q3).
- **Risks accepted:**
  - (i) Polars and foldhash promise nothing across versions or platforms. A Polars bump, or a foldhash change in Polars' lock, can move every logical digest (lean 6's follow-up), and a 32-bit host would differ.
  - (ii) `logical_hash` cannot see −0.0 vs 0.0 or NaN payloads (Q3a). `bytes_hash`, and so the composite `data_hash`, does.
  - (iii) The generated `c6_set` stays ULP-different across machines for its other consumers.
  - (iv) Q1b is unattributed.

**Do the Phase 1 leans hold?**
1. Tier-2 light — **holds**: the bytes are not identical.
2. Same-bytes reading — **holds**. Q2 meets it for c6 as a preview; the record proper comes from the committed fixture.
3. No digest-code change — **holds** (Q2, Q3).
4. Commit the Mac bytes — **holds**; `PINNED_C6` is unchanged (Q2).
5. Platform / libm in the C10 record only — **holds in place, but ⚑ 2: its field list grows.** libm does not name the cause, so the record also needs the numpy wheel's compiler, SIMD baseline and FMA, from `np.show_config()`.
6. Polars-version dependence as a follow-up — **holds**. Q3 adds foldhash's disclaimer to the basis for the follow-up.
7. Little-endian cast — **holds**. Both machines are little-endian, so the cast is a no-op on both.
8. PR-061, merge order by the operator — **unaffected**.

**⚑ Flags for the lead:**
- **⚑ 1 — the draft's :39 is refuted** (libm plus a stream shift).
  - Rewrite it at Phase 4: numpy's gamma / beta values differ at ULP level between the arm64 wheel (clang, FMA-contracted) and the x86_64 wheel (gcc, X86_V2), while the PCG64 stream stays aligned.
  - The same cause goes into Scope 5's CHANGELOG line, the v0.3 waiver's framing (`docs/0.3/ACCEPTANCE.md:382-385`, Notes :168) and the C10 record.
  - PR-057:55-57 attributes the golden fixture's 128-ULP difference to libm; it may share this mechanism. Not tested (*best-guess*).
- **⚑ 2 — lean 5's field list,** as above.
- **⚑ 3 — Q2's transport.**
  - Q2 was planned as a commit on a scratch branch (:83), and constraint :62 makes git the only Mac ↔ desktop route.
  - The spike instead sent the bytes as a checksummed tar stream over ssh stdin. It was scratch test data, not code, and nothing went into a repo.
  - It is the lead's ruling whether that is within "never rsync / scp". Either way, the build's C10 run uses the git-committed fixture.
- **⚑ 4 — the scratch evidence lasts only for this session** (`…/scratchpad/pr061-p3/`). To keep it, copy `desktop/desktop-run.log`, `desktop/mac-hash.json`, `desktop/scripts/` and `q3q4/probe.py` into `docs/0.5/evidence/ruxml-track/`. This is the lead's call.
- **Procedure notes:**
  - rux-ml's web-search rule (`PROCEDURE-pr-research.md:80`) is met for Q1 / Q2 through the sources of Q4 / Q3, not by separate searches.
  - The Group D MCP-verification round (after :115) is the lead's, before Phase 4 locks.

**Phase 1 statements updated:**
- :36's "got / pinned" caveat is now direct (Q1c).
- :51's "Not proven until Q1" is now proven (Q1 + Q2).
- :171's "desktop c6 files' sha256" is now measured (Q1).
- :35 (`PINNED_EDGE` on the desktop) stays inferred; it is read per test at the build (:144).

**Tier verdict: Tier-2 light stays.** Q1 found different bytes: values only, at ULP level, from numpy's arm64 FMA contraction. Q2 found that the Mac bytes hash on the desktop to exactly `PINNED_C6`, so ~~the digest is machine-independent~~ the digest agrees Mac = desktop on the same bytes (shown on c6's committed table, Polars 1.40.1). The PR is therefore a test-fixture fix plus a cross-machine record, ~~not a digest change~~ with no change to the digest's definition or values; its one code change is the value-neutral little-endian cast (Scope 3) *(corrected at Phase 4)*. Next is Phase 4 (Synthesis); halt here.

**C10 evidence values** (spike preview; Polars 1.40.1, rux-ml `09b9bd5`; full values are in Q1c / Q2):

| Bytes | Hashed on | fill `logical_hash` | fill `bytes_hash` | walk `logical_hash` | walk `bytes_hash` |
|---|---|---|---|---|---|
| Mac-generated | Mac | `185d2b9c…ced2d4d8` | `36f6cfa6…6cda614d` | `81f71071…629bd4e7` | `e48d49ec…3eed3f3c` |
| Mac-generated | desktop | `185d2b9c…ced2d4d8` | `36f6cfa6…6cda614d` | `81f71071…629bd4e7` | `e48d49ec…3eed3f3c` |
| desktop-generated | desktop | `c216d680…fd5aa44a` | `aa3df9cf…7ae56ad8` | `9f6ced5a…04dd5b9d` | `c4cf7b45…0fceb228` |

### Group D: verification (2026-10-15 ≈ 13:45 CDT, the lead — before Phase 4 locks)

- **D1 — the same bytes, the same digest, both machines (proven, re-run by the lead with rux-ml's own CLI, not the spike's script).**
  `rux-ml --config configs/base.toml data hash` on the six Mac-generated c6 files: Mac (rux-ml `dev` `09b9bd5`, its `.venv`) and desktop
  (`~/pr060-suite/base` at `09b9bd5`, its `.venv`) both print fill `logical_hash` `185d2b9c…d4d8` / `bytes_hash` `36f6cfa6…614d`, walk
  `81f71071…bd4e7` / `e48d49ec…3f3c`, `row_count` 6,912 = `PINNED_C6` (`tests/cli/test_data_hash_per_file_c6.py:23-26`); the six files'
  sha256 equal on both machines (`4a9f5b21…`, `0d0fcbc7…`, `c49c9b30…`, `4cfddc6b…`, `3c0d2e3a…`, `48bc780c…`).
- **D2 — the mechanism (proven, re-run).** `pr061-p3/mech.py probe` (sha256 `288c9aac…` on both machines; the replica read by the lead:
  numpy's Marsaglia–Tsang with `V = 1 + c·X` rounded once via `Fraction` for the fused form): Mac — the fused replica equals numpy on 12 / 12
  gamma / beta calls, the plain one differs on 246 / 368 / 241 / 255 / 376 / 266 / 243 / 410 / 258 / 269 / 264 / 261; desktop — exactly the
  mirror (plain 0 on all 12, fused those counts); the generator state equal after every call on both.
- **D3 — Polars' seed (proven from the downloaded py-1.40.1 source):** `misc.rs:359-363` hashes from `PlFixedStateQuality::default()` into
  `PlSeedableRandomStateQuality::seed_from_u64`; foldhash `seed.rs:11` `FIXED_GLOBAL_SEED` is a compile-time constant.
- No divergence from Phase 3; nothing carried to Phase 4 as an Amend candidate from Group D.

**Budget.** ~~To be re-stated at the gate with `agents × (68k + 3k × calls)`.~~ Re-stated in the Synthesis below (§Build budget). ROADMAP:31's ≈ 0.3–0.4M covered this item and the early-stopping item together; the latter is now PR-060's.

### Synthesis (Phase 4, 2026-10-15)

**Outcome: Amend — approved.** The Phase 1 premise (ROADMAP:31, :46) was a code fix that makes the digest machine-independent. The research found the digest ~~already machine-independent on the same bytes~~ already agrees Mac = desktop on the same bytes — shown on c6's committed table under Polars 1.40.1 on two 64-bit little-endian hosts, with no documented cross-platform guarantee behind it (Q2, Q3, Group D D1) — and the c6 *test data* machine-dependent (Q1). The PR narrows to a pin over committed bytes plus the C10 record, with ~~no digest-code change~~ no change to the digest's definition or values. One value-neutral code change: an explicit little-endian cast at `versioning.py:115` / :144, a no-op on both machines (lean 7, Scope 3). The amendment was put at the Phase 1 gate (leans 1 – 8, approved 2026-10-15 ≈ 12:40 CDT, :104-109), and Phase 3 confirmed it (:277-285). The operator's "go" of 2026-10-15 ≈ 13:42 CDT approved this synthesis: the corrected cause, the C10 record's longer field list (⚑ 2) and the program texts the leans call for. It is not Escalate, because v0.5's design and C10 stand: C10 is met under the same-bytes reading.

*(The critic's must-fix items folded 2026-10-15 at Phase 4: criterion 13's evidence form; the PR-060 overlap — the three docs files are shared; the cast named as a code change, with its test first at :144; "agrees Mac = desktop on the same bytes" in place of "machine-independent"; C10's precondition; ACCEPTANCE C7 / ROADMAP:29; ROADMAP:31's size.)*

**What Phase 3 changed.**
- **The cause** (⚑ 1; Q1a, Q4, D2; *proven*). ~~numpy's gamma and beta call the platform libm … one flipped accept/reject step shifts the RNG stream~~ (:39). numpy 2.4.5's arm64 wheel (clang 15.0.0; FMA is in base AArch64) contracts `random_standard_gamma`'s `V = 1.0 + c*X` into one fused multiply-add. The x86_64 wheel (gcc 14.2.1, baseline X86_V2, no FMA) does not. beta(2,2) = Ga / (Ga + Gb) inherits the difference.
  - The PCG64 state is equal after every one of c6's 210 calls, so no draw shifted the stream.
  - libm enters gamma only in the accept/reject `log`, and that `log` decided none of the differing draws.
  - Two normal-tail values differ by 1 ULP. It is not known whether FMA in the tail expression or libm `log1p` causes them (Q1b, *best-guess*). This is non-blocking: committed bytes remove both.
- **The fixture's size, measured at Phase 4.** The six Mac-generated files total **3,540,091 bytes** (≈ 3.54 MB / 3.38 MiB):
  - fill 657,479 / 658,337 / 657,643;
  - walk 522,325 / 522,089 / 522,218.
  - Measured by `ls -l` of the session-scratch copy `…/scratchpad/pr061-p3/desktop/mac-c6/set/`, whose six sha256 equal `pr061-p3/mac-hash.json`.
- **The C10 record's fields.** Beside libm, the record adds the numpy wheel's build: its compiler, SIMD baseline and FMA (⚑ 2; lean 5).
- **The ARCHITECTURE section, located** (carried from :139, :359). rux-ml `docs/ARCHITECTURE.md` §Reproducibility Architecture (per D9) is at :593, with item 5 (the composite `data_hash`) at :601. The bridge paragraph is at :266; it says the Polars row hash is guaranteed "only within one version".

**The leans as ruled** (Gate Check :104-109; Phase 3 :277-285):
1. Tier-2 light.
2. C10's "one table" = the same file bytes hashed on both machines, on a committed copy of c6's `fill/` and `walk/`.
3. No change to the digest's definition or values (ruled as "no digest-code change"; the one value-neutral code change is lean 7's little-endian cast at `versioning.py:115` / :144, a no-op on both machines). The pin test gets its own committed fixture; `_c6_day` / `c6_set` are unchanged.
4. The Mac bytes are committed, so `PINNED_C6` is unchanged.
5. Platform, libm and now the numpy build are recorded in the C10 record only. No sidecar change.
6. The Polars-version dependence is a follow-up (Notes).
7. The explicit little-endian cast: **yes**. It is now in scope, not optional (Scope 3).
8. PR-061. The operator orders the merges; both rux-ml PRs land before the `prereg-v05-probe` tag.

**Changes to ARCHITECTURE.md** (rux-ml), one sentence in the build commit, beside :266 / :601 item 5:
- The digest is equal Mac = desktop on the same bytes, pinned by `tests/cli/test_data_hash_per_file_c6.py` over `tests/fixtures/c6_pin/`.
- That equality comes from Polars 1.40.1 / foldhash 0.2.0's code on 64-bit hosts. Neither documents a cross-platform guarantee.
- `logical_hash` cannot see −0.0 vs 0.0 or NaN payloads; `bytes_hash` does (Q3, Q3a).

**Changes to CONSTRAINTS.md**: none. rux-ml `CONSTRAINTS.md:105` already allows an exact data-hash pin (:42).

**New PRs that must come first**: none. PR-060 runs in parallel. ~~the two PRs touch disjoint files (:78)~~ No PR-060 code or test file is touched: its Scope's `cli/train.py`, `tests/training/test_factory.py`, `registry/promote.py` and `runs/oos_export.py` (PR-060 draft :145-156), and the tests it names as pinning its paths, `tests/cli/test_m9_batches.py`, `tests/cli/test_train_subcommand.py` and `tests/golden/test_xgb_baseline.py` (:52-54). Both PRs add lines to the same three docs files — CHANGELOG `[Unreleased]`, `docs/ARCHITECTURE.md`, `docs/0.3/RESEARCH-BACKLOG.md` (PR-060 draft :95, :156, :194). The additions are append-only, and whichever PR merges second rebases its lines (:78).

**Research-backed details now locked:**
- **The fixture's location: `tests/fixtures/c6_pin/fill/` and `tests/fixtures/c6_pin/walk/`, each holding `2026-02-16.parquet`, `2026-02-17.parquet` and `2026-02-18.parquet`.**
  - `tests/fixtures/` holds the repo's committed non-golden test data; today that is `tests/fixtures/m9_training_schema.json`. `tests/golden/fixtures/` belongs to the golden model.
  - The file names are the ones `c6_set` writes (days 20,500 – 20,502; `tests/cli/conftest.py:122-130`).
  - No `.gitignore` rule catches the path (`git check-ignore` exit 1 at `09b9bd5`).
- **The pins:**
  - the six files' sha256 (`pr061-p3/mac-hash.json`), checked whatever the Polars version;
  - the subtrees' logical digests = `PINNED_C6` (`test_data_hash_per_file_c6.py:23-26`), still gated on Polars 1.40.1 (:21, :38).
- **The cast:** `versioning.py:115` (`row_h`, before `memoryview(row_h).cast("B")`) and :144 (`arr`, before `arr.tobytes()`).

**Risks accepted** (Phase 3 :271-275, plus one):
- (i) Polars and foldhash promise nothing across versions or platforms.
- (ii) `logical_hash` cannot see −0.0 vs 0.0 or NaN payloads; `bytes_hash` does.
- (iii) The generated `c6_set` stays ULP-different across machines for its 11 other consumers.
- (iv) Q1b is unattributed.
- (v) *(new)* A 3.54 MB binary fixture goes into git. Its bytes are a record, not something to regenerate: `_c6_day` reproduces them only on an FMA-contracting numpy build such as the Mac's.

**Decisions left for the operator at Phase 5:**
1. **The build go** on the Scope, criteria and budget below, including the one ARCHITECTURE sentence. *Lean: in.*
2. **rux-ml README :256 / :595 and `CLAUDE.md:48, :52` drift** (:55, :394). *Lean: out of this PR* (smaller PRs).

These are already settled and need no decision:
- ⚑ 3 (the transport) was ruled by the lead (:145).
- ⚑ 4 (the evidence is kept in `pr061-p3/`) is done (:145).
- Group D found no divergence (:331).
- The merge order (lean 8) is the operator's word at merge time.

---

### Gate Check (Phase 5, 2026-10-15)

- Premise still valid ✓ (amended at Phase 4: the digest agrees on the same bytes; the pin moves to committed bytes). No prerequisite PR
  (PR-053 / PR-057 on `dev`). **The operator approved the build ≈ 14:25 CDT ("go")** at the amended Scope, criteria and budget (≈ 0.3M,
  $0), the ARCHITECTURE sentence in, the README / `CLAUDE.md` drift out. The merge into `dev` is the operator's own word.
- **Implementation cleared.** Worktree `/Users/maxrux/projects/rux-ml-pr-061`, branch `pr-061-c6-pin` off `origin/dev` `09b9bd5`.

## Scope

*As amended at Phase 4 (2026-10-15; leans 1 – 8 as ruled).*

1. **A pin over committed bytes** (leans 3, 4).
   - **The fixture.** Commit c6's six Mac-generated parquet files (`fill/` and `walk/`, three days each, 3,540,091 bytes) ~~as a test fixture; the location is named at Phase 4~~ at **`tests/fixtures/c6_pin/{fill,walk}/2026-02-1{6,7,8}.parquet`**. These are the bytes whose sha256 are in `pr061-p3/mac-hash.json`. They were drawn by `_c6_day` with `default_rng(0)` on the Mac (numpy 2.4.5 arm64, Polars 1.40.1).
   - **The test.** The c6 pin test (`tests/cli/test_data_hash_per_file_c6.py`) reads the committed copy ~~:~~ instead of `c6_set`. It:
     - pins each file's sha256, whatever the Polars version;
     - pins each subtree's logical digest (`PINNED_C6`, unchanged), still gated on Polars 1.40.1;
     - keeps PR-053's four-route equality on those bytes: per-file = whole-source = `data_hashes` = `training_set_sidecar` (:32-37).
   - **Test first.** The test is written first and fails before the fixture is committed.
2. **`_c6_day` / `c6_set` unchanged** (`tests/cli/conftest.py:42-131`). The other 11 c6 consumers keep generated data.
3. ~~**Optional, digest-neutral** (gate decision 7)~~ **The little-endian cast, digest-neutral** (lean 7, ruled yes). Add an explicit `astype("<u8", copy=False)` at `versioning.py:115` (`row_h`) and :144 (`arr`). Every existing pin must reproduce: `PINNED_EDGE`, the golden manifest and `PINNED_C6`.
   - **Test first at :144.** A new test monkeypatches `pl.Series.to_numpy` to return `.astype(">u8")` and asserts that `_logical_hash_whole_source` on the committed c6 bytes gives the same digest as the unpatched call. Before the cast it fails (`arr.tobytes()` hashes big-endian bytes); after the cast it passes. The patch also reaches :111's `part.to_numpy()`, whose assignment into the native `row_h` converts by value, so the per-file path is unaffected.
   - **:115 is exempt from test first, by name.** `row_h` is allocated native `np.uint64` at `versioning.py:96`, so the cast at :115 cannot be made to fail first on a little-endian host. Its guard is the existing pins reproducing (criterion 5).
   - **The docstring at `versioning.py:122`** ("kept verbatim as the reference") is reworded in the same change, e.g. "kept as the reference, verbatim except for PR-061's explicit little-endian cast".
4. **The C10 run.** On both machines, at the merge candidate's commit, run `rux-ml --config configs/base.toml data hash` (D1's route) and the per-file sha256 on the committed table. Write the result as program `docs/0.5/evidence/ruxml-digests-<date>.json`. ~~Polars / numpy versions, rux-ml commit, platform and libm~~ For each machine the record holds:
   - per subtree: `data_hash`, `bytes_hash`, `logical_hash` and `row_count`;
   - each file's sha256;
   - the platform (OS, kernel, arch, endianness) and libm;
   - **the numpy wheel's compiler, SIMD baseline and FMA status**, from `np.show_config()` (⚑ 2);
   - the Polars, numpy, pyarrow and Python versions;
   - the rux-ml commit.

   The record belongs to the program; the run happens in rux-ml's environment (lean 5).
5. **Docs in the same commit:**
   - a CHANGELOG `[Unreleased]` `### Fixed` line naming the cause ~~found by Q1~~ as corrected at Phase 4. Proposed wording: *"The c6 digest pin now reads committed bytes (`tests/fixtures/c6_pin/`). The desktop failed the pin because the c6 test data differed, not the hash: numpy's arm64 wheel fuses `random_standard_gamma`'s `1 + c·X` into an FMA and the x86_64 wheel does not, so gamma- and beta-fed columns differ at ULP level. The same bytes hash the same on both machines."* It must not mention libm, and it says "ULP level", not "≤ 9 ULP" (:258);
   - a `docs/0.3/RESEARCH-BACKLOG.md` row (where track PRs are recorded, PR-060 draft:95);
   - the ARCHITECTURE sentence (Synthesis).

**Not in scope:**
- any change to the `bytes_hash` / `logical_hash` definitions, the composite format or the sidecar keys (the cast is a no-op on both little-endian machines);
- the Polars-version check or an exact pin (lean 6);
- `_c6_day` / `c6_set` and every PR-060 code or test file (:368; the three shared docs files take append-only lines, criterion 7);
- the harness (A13);
- cross-machine equality of fitted models or of a rebuilt table (PR-057; PR-081 Q2);
- the program texts. C10, ROADMAP:31 and (added at the critic's fold) ACCEPTANCE C7 (:93-96) and ROADMAP:29 are the lead's; their proposed text was returned with this synthesis. PR-083:56-57 and PR-084:34-36, :62 were edited at this Phase 4, with the old words struck;
- rux-ml README :256 / :595 and `CLAUDE.md:48, :52` (Phase 5 decision 2, lean out).

## Dependencies

None open in rux-ml: PR-053 (the per-file logical hash) and PR-057 (the reference platform) are merged in `dev` @ `09b9bd5`. Program side: ~~gate decision 2's ruling on C10's wording~~ C10's wording is the lead's; the proposed text was returned with this synthesis. The merge order against PR-060 and the v0.5 freeze tag is lean 8, the operator's word.

## Architecture section implemented

rux-ml `docs/ARCHITECTURE.md` §Reproducibility Architecture (per D9), :593: item 5, the composite `data_hash` (:601), and the bridge's `data_hash` paragraph (:266). ~~The section was not read in this assessment; locate it at Phase 2.~~ It was located at Phase 4.

## Verification criteria

*Restated at Phase 4. Each criterion names its evidence form.*

- [x] **1. Q1 recorded.** The desktop-generated c6 per-file sha256 against the Mac's, the first differing value per column, and the failing assertion at `09b9bd5`. *Met at Phase 3:* Findings Q1 / Q1c; `pr061-p3/desktop-run.log` (:5-39, :466-468).
- [ ] **2. Test first.** The rewritten pin test, run on the Mac before the fixture is committed, fails on the missing or unequal bytes, not on an import or collection error. It passes after the commit. *Evidence:* both `pytest -rA` outputs, with their commits, in this file's build notes.
- [ ] **3. The fixture.** All six files are at `tests/fixtures/c6_pin/{fill,walk}/2026-02-1{6,7,8}.parquet`, 3,540,091 bytes in total, with sha256 equal to `pr061-p3/mac-hash.json`. *Evidence:* `git ls-files -s tests/fixtures/c6_pin` and `shasum -a 256` output.
- [ ] **4. One table, both machines.** On the Mac and the desktop the pin test passes. `rux-ml --config configs/base.toml data hash` on each subtree prints `logical_hash` = `PINNED_C6`, the Q2 `bytes_hash` (fill `36f6cfa6…614d`, walk `e48d49ec…3f3c`; :197-198) and `row_count` 6,912. *Evidence:* the C10 record.
- [ ] **5. Existing pins.** `PINNED_EDGE` and the golden manifest digest pass on both machines. This is read per test from the junit XML; it replaces :35's inference. *Evidence:* the junit `testcase` entries, cited by name.
- [ ] **6. The digest code is unchanged except for the cast.** `git diff dev -- src/` shows only the two `astype("<u8", copy=False)` changes and the reworded docstring line at `versioning.py:122` (Scope 3), and every existing pin reproduces byte for byte. The :144 test fails before the cast and passes after; :115 is exempt by name (Scope 3). *Evidence:* the diff, both runs of the :144 test, and criterion 5.
- [ ] **7. `c6_set` unchanged; no PR-060 code or test file touched.** `git diff dev -- tests/cli/conftest.py` is empty, the 11 other c6 consumers pass with no edit, ~~and no PR-060 file is touched~~ and `git diff dev --name-only` shares no path with PR-060's code and test files (the list at :368). The three shared docs files (CHANGELOG `[Unreleased]`, `docs/ARCHITECTURE.md`, `docs/0.3/RESEARCH-BACKLOG.md`) take append-only additions; whichever PR merges second rebases its lines. *Evidence:* the two diffs and the suite runs.
- [ ] **8. The C10 record.** `docs/0.5/evidence/ruxml-digests-<date>.json` shows both machines' digests equal and has every field in Scope 4, including the numpy wheel's compiler, SIMD baseline and FMA beside libm. *Evidence:* the file; the lead re-reads the equality.
- [ ] **9. The pin test's wall** on each machine is within §Performance budget. *Evidence:* pytest's `--durations` lines on both machines.
- [ ] **10. Mac.** `uv run pytest`, `ruff check --no-cache .`, `ruff format --check src tests` and `basedpyright src/` are clean. *Evidence:* their output.
- [ ] **11. Desktop.** `uv run --frozen --all-extras pytest -n 8` **diffed against `dev`**: the 2 c6 failures are gone and there is no new failure. *Evidence:* both runs' failure lists.
- [ ] **12. Docs in the same commit** (Scope 5): the CHANGELOG `### Fixed` line with the FMA cause, the RESEARCH-BACKLOG row and the ARCHITECTURE sentence. *Evidence:* the commit's diff.
- [ ] **13. The program lead's merge to `dev`,** before the `prereg-v05-probe` tag. *Evidence:* ~~the merge commit, an ancestor of the tag~~ the rux-ml commit that the freeze record (`docs/0.5/evidence/prereg-freeze-<date>.json`, program PR-082:976, :986) records is the PR-061 merge commit or a descendant of it — `git -C ~/projects/rux-ml merge-base --is-ancestor <merge> <recorded>` exits 0 — and the merge's date is before the tag's. (`prereg-v05-probe` is a program-repo tag, PR-082:1219, so no rux-ml commit can be its ancestor.)

## Performance budget

This is a test fixture, not a per-event path. The pin test (both parameters) reads 3.54 MB and hashes 13,824 rows.

The baseline at `09b9bd5`, on the generated `c6_set`:
- **Mac:** 2 passed in **1.02 s** (pytest-reported; 1.49 s process wall). This writer measured it on 2026-10-15 with `.venv/bin/python -B -m pytest -p no:cacheprovider -q`; `git status` was clean afterwards.
- **Desktop:** 2 failed in **0.85 s** (`pr061-p3/desktop-run.log:468`).

**Budget: at most the baseline on each machine.** Reading committed bytes replaces generating them. Criterion 9 closes the budget with each machine's measured wall. A miss is reported, never waived silently.

## Build budget

By the cost model, `agents × (68k + 3k × calls)`:
- **The builder (Mac):** test first, the fixture, the cast, the docs and the Mac gates. 1 × (68k + 3k × 25) ≈ 143k.
- **Desktop:** the C10 run on both machines, the junit XML and the suite diff against `dev`. 1 × (68k + 3k × 15) ≈ 113k. Each pytest call stays ≤ 5 min, because a subagent's Bash call over 600 s is backgrounded. If the full desktop suite takes longer, the lead runs it. One mkdir heavy-lock window.
- **The lead** re-reads the record and the diff: ≈ 0.05M.
- **Total ≈ 0.3M, $0.**

Research to date, by the cost model (the actuals are the lead's):
- Phase 1: 2 × (68k + 3k × 18) ≈ 0.24M.
- Phases 2 – 3: ≈ 0.33M (:147).
- Phase 4 (this writer): ≈ 1 × (68k + 3k × 14) + the draft as its pack ≈ 0.14M.
- Total ≈ 0.7M.

Item (i) as a whole therefore costs ≈ 1.0M, against ROADMAP:31's ≈ 0.3 – 0.4M for two items.

## Research backing

- This PR's two Phase 1 lenses (18 tool calls each), re-reading rux-ml `dev` @ `09b9bd5`, with Mac probes in the same session (the golden digest, the c6 pins, `pl.Expr.hash.__doc__`).
- Program `docs/0.3/evidence/close-out/c1-ruxml-20261001.json` (the desktop runs at `48ffd4e` and `09b9bd5`; re-read for this draft at :14-17, :35-43, :50-59).
- rux-ml PR-053 (the per-file logical hash) and PR-057 (the reference platform; ~~the libm data mechanism~~ its golden data differed by up to 128 ULP, and it labels the cause "best guess, unproven" at PR-057:59 — not a proven libm mechanism).
- Program PR-081 Q2 / Q12 (libm `log` on PR-081's own probe table; pyarrow bytes within one machine).
- **Phase 3, done:**
  - Q1 / Q2: the desktop spike;
  - Q3: Polars' struct-hash platform stability (source at `py-1.40.1`, foldhash 0.2.0, web);
  - Q4: numpy's gamma / beta ~~libm dependence~~ dependence on the platform, which is the build's FMA contraction (source at `v2.4.5`, docs, web);
  - Group D, D1 – D3.
- Evidence: `docs/0.5/evidence/ruxml-track/pr061-p3/`.

## Notes

- **For the program lead, not this PR** (updated at Phase 4):
  - C10 (ACCEPTANCE:104-106): proposed text returned with this synthesis (the lead's file).
  - ROADMAP:31, item (i): proposed text returned (the lead's file), with its size cell (≈ 1.0M for (i); (ii) sized in PR-060).
  - ACCEPTANCE C7 (:93-96) and ROADMAP:29 (PR-084's row) still say "a machine-independent artifact hash", against PR-084:34-40 / :66-68 as amended. Proposed text returned (critic's fold). **An acceptance-criterion change: the operator's.**
  - PR-083:56-57: ~~"hashed on both machines", not "reproduced"~~ edited at Phase 4, with the old words struck.
  - PR-084:34-36, :62: edited at Phase 4. C10 covers the table's digest only. Whether a model file's sha256 can be equal Mac = desktop is PR-084's own open item (rux-ml PR-057:22).
  - **The v0.3 waiver's framing** (`docs/0.3/ACCEPTANCE.md:382-385`, carried at :360). It reads "platform-dependent; cause not investigated" and "(a machine-dependent data hash would break cross-machine reproducibility)". For the lead: add a note beside it rather than rewriting a cut version's record. Proposed note: *cause found 2026-10-15 (rux-ml PR-061): the hash agrees Mac = desktop on the same bytes (c6's committed table, Polars 1.40.1); the c6 test data differed (numpy's arm64 FMA contraction of gamma's `1 + c·X`); the pin now reads committed bytes.*
  - ⚑ 1's cite "Notes :168" (:290) does not resolve: `docs/0.3/ACCEPTANCE.md:168` is criterion text on Gate 1. The waiver itself is at :382-385.
- **The cause, corrected** (⚑ 1). The draft said ~~numpy's gamma and beta call the platform libm … macOS libm vs glibc … one flipped accept/reject step shifts the RNG stream~~ (:39; also :82, :85, both struck in place). The cause is FMA contraction in numpy's arm64 build of `random_standard_gamma`'s `V = 1 + c·X`; it is not libm and not a stream shift (Q1a, D2). The CHANGELOG line (Scope 5) and the C10 record carry the corrected cause.
- **Follow-up (lean 6).** A Polars bump moves every logical hash, and so might a foldhash change in Polars' lock (Q3). The sidecar records `polars_version` and the harness copies it, but nothing compares it. Promote then fails closed with a bare `data_hash` mismatch (`bridge.py:5-7, :134`; `materialize.py:832`; `promote.py:320-325`).
- **Endianness.** ~~The logical digest hashes native-endian bytes; both machines are little-endian, so there is no practical effect today (decision 7).~~ The cast (Scope 3) makes the hashed bytes explicitly little-endian. It is a no-op on both machines.
- ~~**Not measured:** the committed fixture's size on disk; the desktop c6 files' sha256.~~ Both are now measured: 3,540,091 bytes (Synthesis), and the desktop sha256 (Q1).
- **PR-057's golden-data difference** (up to 128 ULP, PR-057:57) may share the FMA mechanism through `make_classification`. Untested (*best-guess*; ⚑ 1); not this PR's.
- **Pre-existing, not this PR's:** the rux-ml README :256 / :595 (`logical_hash` description) and `CLAUDE.md:48, :52` drift (Phase 5 decision 2, lean out).
