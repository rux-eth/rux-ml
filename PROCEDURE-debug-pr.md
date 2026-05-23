# Procedure: Debug PR

## When to use

For PRs whose primary purpose is **fixing existing observed bugs in shipped code** — not adding features, not refactoring for cleanliness, not migrating dependencies. The defining trait is: the bug is already producing wrong behavior in a local or workbench environment (a reliably reproducible run), and the user wants it stopped.

**Scope: code bugs only.** This procedure is for deterministic, reproducible defects in code logic — state machines, off-by-ones, missing keys, race conditions, broken transformations, config-overlay precedence bugs, etc. The TDD-shaped failing-test-first gate, the single-root-cause file-line discipline, and the two-signoff rigor are calibrated to that kind of defect.

**Out of scope** for this procedure:

- **Prompting / LLM output quality issues** — biased outputs, instruction-following failures, low-quality completions, hallucinations, etc. LLM outputs are probabilistic; "the test fails today and passes after the fix" does not map cleanly to a stochastic generation. Even when a prompt change produces a measurable improvement in the output distribution, it's not the same shape of "defect → root cause → deterministic fix" that debug-PRs are designed around. These belong under `PROCEDURE-pr-research.md` (typically Tier-2), where the empirical baseline + research + experiment-and-measure rhythm is a better fit. rux-ml does not call LLMs in production paths at present; this clause is forward-looking for any future feature that might.
- Features, refactors, dependency migrations, performance work that isn't responding to a specific reproducible bug, infra-setup PRs.

If the work is split between "fix existing code bug" and "add new feature", split the PR. The debug-PR procedure runs only on the code-bug lane.

Branched from `PROCEDURE-pr-research.md`. Inherits the rigor (research-backed decisions, evidence-cited findings, no LLM intuition dressed as fact) but reorders and expands phases to match how bug-fixing actually works, with a TDD-style "no fix without a failing test" gate.

## Why a separate procedure

Three failure modes motivate this branch from the standard PR research procedure:

1. **Solution-jumping before specifying intent.** Multiple commits attempt fixes to "the bug" before anyone writes down what "fixed" means. The actual root cause is the Nth attempt. Costs: code clutter, real time, multiple PRs of work to clean up.
2. **Failed-attempt residue.** When a debug fix turns out NOT to address the root cause, there is no protocol for whether to revert it or leave it. Speculative fixes accumulate in the codebase even when the work that motivated them is superseded.
3. **State-assessment scope too narrow.** The standard procedure asks "what's true in the file you're modifying". For bugs that span multiple files, the actual root cause often lives in a SHARED dependency.

## Hard rules

These apply across all phases:

- **Phases 1–3 do not modify production code.** The only allowed code edits in Phases 1–3 add instrumentation / observability for debugging purposes. No "fix attempts" — no matter how obvious the fix looks.
- **No fix lands without a failing test that the fix turns green.** Tests are designed in Phase 4, implemented in Phase 5 (where they fail because nothing has been fixed), then turned green by the fix in Phase 7.
- **Two explicit user signoffs**: after Phase 4 (test design approved) and after Phase 6 (fix design approved).
- **Phase 3 ends only when the bug mechanism is known with end-to-end certainty** — file:line of the originating defect + log/empirical evidence of the failure path being taken at runtime. "I think this is the bug" is not sufficient.
- **Phase 1 includes an empirical-baseline scan** when persisted telemetry exists AND the bug class is observable in it. For rux-ml the canonical telemetry surfaces are Optuna study databases (`studies/*.db`), `trial.user_attrs` / `trial.system_attrs`, `peak_rss_mb` + Watchdog trips, run logs (`logs/*`), and any workbench `nohup`-redirected job output. Don't defer cited densities, occurrence counts, or "is this bug actually firing right now?" questions to Phase 3 if a scan can answer them now. Phase 3 is for runtime-mechanism confirmation (instrumentation, dev-mode reproduction) — not for the basic "does this bug fire and how often" question. See Phase 1 step 6. For bugs that produce no persisted-telemetry signal (CLI option parsing bugs, pre-trial config validation bugs, docs-only bugs, in-process bugs without persisted state), document that the scan is N/A with a one-line justification and move on.
- **Empirical testing in Phase 3 happens locally in dev mode** (`uv run rux-ml <verb> ...` against synthetic or tmp-path data; see `CLAUDE.md` "Build & Test Commands" for the canonical commands). Workbench reproduction (via SSH for GPU-bound bugs) is allowed; persisted-telemetry scanning happens in Phase 1; live reproduction (with instrumentation) happens in Phase 3.

## Output location

Same as `PROCEDURE-pr-research.md`: research findings appended to the PR file under `## Research findings`. Format defined in `docs/DEBUG-PR-STANDARDS.md`.

## Phases

### Phase 0: Bug Specification

> **ROADMAP note** — if an active `docs/<x.y>/ROADMAP.md` exists for the current sprint, this PR gets a row in it per `feedback_roadmap_flip_in_pr`. If no active ROADMAP exists (post-version-cut interim), the PR file is the canonical record and there's no row to flip — same "no row to flip" semantic PR-028/29 used.

**Goal**: Write down what each bug actually is — current vs expected behavior — BEFORE any state assessment, BEFORE any research, BEFORE any solution design. This is the contract the fix must satisfy.

For each suspected bug, produce a pair:

```markdown
### Bug N — [one-line title]

**Current behavior**

[What the code actually does, with file:line citations + log / reproduction evidence. No speculation about WHY it does this — only what.]

**Expected behavior**

[What the code SHOULD do, written as a contract. Acceptable solution shapes can be enumerated; specific implementations are not yet fixed. Include observable acceptance criteria — what would let the user verify the bug is fixed?]
```

**Atomic granularity rule**: if a bug has multiple distinct facets (e.g., "wrong fold count" and "wrong sampler config" and "wrong attrs recording"), split it into sub-bugs (2a, 2b, 2c, ...). Each sub-bug must be independently verifiable. Bundling reduces clarity and increases the chance research papers over a real distinction.

**Output**: `## Bugs (current vs expected behavior)` section in the PR file. See `docs/DEBUG-PR-STANDARDS.md`.

**Exit criteria**: User explicitly approves the bug specification. The bug list is now closed. New bugs found during later phases either spawn a new debug PR or trigger a STOP-and-amend with explicit user approval.

---

### Phase 1: Expanded State Assessment

**Goal**: Establish what's true RIGHT NOW in the buggy files AND in their transitive dependencies — root causes often live in shared code, not the files where symptoms appear.

Differs from `PROCEDURE-pr-research.md` Phase 1 in scope:

1. Read every file named in the bug specifications.
2. Read every file imported (transitively, ≥1 hop) by the files in step 1.
3. Identify **common imports** — files imported by ≥2 of the buggy files. These are highest-priority for inspection.
4. List the last N PRs / commits since the last formal merge that touched any file in the import closure. Recent changes are higher-priority for "did something just break this?"
5. Re-read relevant `docs/ARCHITECTURE.md` + `docs/CONSTRAINTS.md` sections.
6. **Empirical baseline scan** (conditional — see hard rules above). When persisted telemetry exists AND the bug class is observable in it, write a short one-pass script that scans every available surface for each bug's signature, and reports:
   - Match count per surface
   - First / last timestamp per surface with matches
   - Density (matches per unit time) over each surface's window
   - Cross-bug correlation evidence (Optuna trial state inspection, user_attrs field anomalies, system_attrs artifact-record absences, etc.)
   - Whether the bug fires under the **current** config / CLI-flag-default combination (a bug that doesn't fire under current defaults is real but unobservable; that information changes Phase 4's test design priority)

   Scratch scripts live in `/tmp/`; reusable scans go to `scripts/audit/<topic>.sh`. Run the scan against the canonical persisted telemetry — the local `studies/*.db` or the workbench's `~/projects/rux-ml/studies/*.db` over SSH, whichever holds the relevant runs. The script's output goes into the Phase 1 PR section under `**Empirical baseline (persisted telemetry)**`. Pulling evidence forward into Phase 1 (a) corrects auto-memory / drift on cited counts before they propagate into hypothesis design, (b) distinguishes "actively biting" bugs from "theoretical / dead-path" bugs, (c) catches cross-bug correlations the per-file read pass would miss.

   For bug classes that produce no persisted-telemetry signal (CLI option-parsing bugs, pre-trial config validation bugs, docs-only bugs, in-process bugs without persisted state), document `N/A — bug class is not observable in persisted telemetry; reproduction will happen in Phase 3` and skip step 6. This is the common case for rux-ml CLI / config bugs; the empirical scan is rare and reserved for bugs that affect persisted Optuna state.

   Anti-pattern caught by this step: deferring all empirical evidence to Phase 3 means hypotheses in Phase 2 are designed against assumed densities and assumed mechanisms. When the scan corrects an order-of-magnitude error in the spec, downstream phases were already biased. Surface the data first when the data exists.

**Approach decision (sequential vs parallel)**: by the end of Phase 1 (or early Phase 2 at latest), decide whether the PR's bug set is best worked through Phases 3–7 sequentially (one bug at a time, individually verifiable) or in parallel (all bugs batched). Sequential is safer / has smaller blast radius / clearer test mapping; parallel is faster when bugs are loosely coupled. **Empirical Step 6's findings inform this decision** — bugs that fire heavily in persisted runs sequence ahead of theoretical ones in Phase 7 implementation order. Document the decision in the PR file.

**Output appended to PR's "Research findings" section** — see `docs/DEBUG-PR-STANDARDS.md` for the template.

**Exit criteria**: every bug from Phase 0 has at least one cited candidate root-cause location, OR an explicit "root cause not yet localized; needs Phase 3 empirical investigation". User approves before Phase 2.

---

### Phase 2: Hypothesis Assembly

**Goal**: Round up every assumption and suspicion about what the problem may be — across the codebase — into one structured place. This is the input list Phase 3 will work through.

For each bug:
- List candidate root-cause hypotheses, each with file:line attributions where possible
- Rank hypotheses by likelihood given Phase 1 evidence
- For each hypothesis, define what observation would confirm or refute it
- Note dependencies between hypotheses (if H2 depends on H1's truth, sequence them)

**Output appended to PR's "Research findings" section:**

```markdown
### Phase 2 — Hypothesis Assembly (YYYY-MM-DD)

**Bug 1 hypotheses (ranked):**
1. [hypothesis] — confirm/refute via [observation]
2. [hypothesis] — confirm/refute via [observation]
3. [hypothesis (less likely)] — confirm/refute via [observation]

**Dependencies:** [if H2 depends on H1, etc.]

**Bug 2a hypotheses:** ...
```

**Exit criteria**: every bug has ≥1 hypothesis with a defined confirmation/refutation observation. User approves before Phase 3.

---

### Phase 3: Bug Confirmation (Web Research + Empirical Testing)

> If confirmation reveals this isn't a code bug (e.g., LLM-output-quality issue or a config-decision question), switch to `PROCEDURE-pr-research.md` and re-classify the PR.

**Goal**: Reach end-to-end certainty about each bug's mechanism. Phase 3 is finished only when the bug's failure path can be described from origin to symptom, citing both literature/precedent AND runtime evidence.

**Allowed activities**:
- **Web research** (per `PROCEDURE-pr-research.md` Phase 3 rigor): reputable sources, ≥2 cited candidate explanations per question, explicit disconfirming-evidence search, epistemic-status labels. Used to validate hypotheses against framework / library / production-system documentation.
- **Empirical testing in dev mode locally** (`uv run rux-ml <verb>` against synthetic/tmp-path data, or `uv run pytest <path>` for in-process repros — saves rebuild time vs workbench round-trips). Adding instrumentation / observability code is permitted (extra log lines, `print` statements in trial bodies, third-party tracing libraries, etc.). The instrumentation is allowed to land in the codebase if it's good enough to keep (e.g., a permanent log line that helps future debugging); otherwise it's reverted before Phase 7.
- **Workbench reproduction** via SSH (`ssh ssh-desktop ...`) when the bug requires GPU or large-dataset behavior unavailable locally. Reproduce on a fresh study name to avoid polluting prior baselines.
- **Isolation tests** when Phase 2 leaves ≥2 hypotheses competing on which layer fails (data layer / features pipeline / trainer / tuner / sampler / pruner / config layer / CLI dispatch / etc.). Standard debugging: write throwaway scripts that vary one variable at a time and hold the rest fixed (swap trainer family, bypass features pipeline, vary fold count, swap dataset, force-flip a CLI flag) until the failure surface is pinned to a specific layer. Don't combine hypotheses in Phase 6 without distinguishing them empirically here.
- **Reading runtime logs / Optuna study DBs / `trial.user_attrs` and `system_attrs` / xgboost callback output** to confirm the hypothesis's predicted failure path is the one taken at runtime.

**Disallowed activities**:
- Modifying production code with the intent to "fix" the bug. Not in Phase 3. Not even if the fix is "obviously right". Tests come first (Phase 4–5), then fix (Phase 6–7).
- Modifying tests to pass / hide failure (no test exists yet anyway).

**Output appended to PR's "Research findings" section:**

```markdown
### Phase 3 — Bug Confirmation (YYYY-MM-DD)

**Bug N — confirmed mechanism**

*Hypothesis tested*: [which Phase 2 hypothesis]

*Web evidence*: [cited URLs supporting the mechanism — framework docs, RFCs, post-mortems, production source]

*Empirical evidence*: [observability instrumentation added (file:line if kept, ad-hoc if reverted) + log / trial-attrs / DB excerpts confirming the predicted failure path at runtime]

*End-to-end mechanism*: [origin → propagation → observable symptom, with file:line at each hop]

*Status*: confirmed (file:line of originating defect)

[Repeat per bug. If a hypothesis is refuted, document and move to the next-most-likely hypothesis until one is confirmed.]
```

**Exit criteria**: every bug has a `Status: confirmed` with file:line + empirical evidence. User approves before Phase 4.

---

### Phase 4: Test Design

**Goal**: Design unit and/or integration tests that prove each bug exists. Tests are written so they FAIL against the current (unfixed) code — that failure is what proves the bug is real and what the fix in Phase 7 will make pass.

For each bug:
- Decide unit-test or integration-test (whichever best exercises the bug's failure path):
  - **Unit test** — isolated function / class behavior, dependencies mocked. Best when bug is in pure logic or a single component's local state.
  - **Integration test** — multi-module interaction, less mocking. Best when bug is at a contract seam (e.g., a bug that spans CLI dispatch + config overlay, or features pipeline + trainer's input shape, or HPO objective + persistence layer). For rux-ml CLI bugs the typer `CliRunner`-driven tests in `tests/cli/` are the canonical integration surface.
  - Some bugs may need both (e.g., a unit test for the local invariant + an integration test for the cross-module flow).
- Define each test's:
  - Setup (what state the test puts the code in)
  - Trigger (what action exercises the failure path)
  - Assertion (what observable outcome distinguishes "bug present" from "bug fixed")
  - Why this fails today (explicit prediction tying to Phase 3's mechanism)
- Map each test to the verification criterion it satisfies (per `docs/DEBUG-PR-STANDARDS.md`).

**Output appended to PR's "Research findings" section:**

```markdown
### Phase 4 — Test Design (YYYY-MM-DD)

**Bug N — test design**

*Test type*: unit / integration / both
*Setup*: ...
*Trigger*: ...
*Assertion*: ...
*Predicted current behavior*: [test fails because <Phase-3-mechanism>]
*Predicted post-fix behavior*: [test passes because <expected-behavior-contract>]

[Repeat per bug.]
```

**Exit criteria**: ⚠️ **USER SIGNOFF REQUIRED** before Phase 5. Test design must be explicitly approved.

---

### Phase 5: Test Implementation

**Goal**: Implement the tests designed in Phase 4. Run them. They MUST fail in the predicted way — that failure is the empirical proof that the bug exists in shipped code.

- Implement each test exactly as designed in Phase 4. No deviations from the design without going back and amending Phase 4 + getting fresh signoff.
- Run the tests against current (unfixed) code.
- Confirm each test fails with the predicted failure mode.

**If a test does NOT fail**: this is a critical signal — either the test is wrong (loop back to Phase 4) or the bug spec is wrong (loop back to Phase 0). Don't "improve" the test silently; document the discrepancy.

**Output appended to PR's "Research findings" section:**

```markdown
### Phase 5 — Test Implementation + Failure Confirmation (YYYY-MM-DD)

**Bug N — test result**

*Test file*: [path]
*Run output*: [excerpt showing failure]
*Predicted failure mode matched*: yes / no
*[if no]*: [discrepancy + loopback decision]
```

**Exit criteria**: every test fails in its predicted way. Tests are committed to the branch.

---

### Phase 6: Fix Research

**Goal**: Decide HOW to fix each bug, with research backing covering both *convention* (what does the broader software / production-system world do here?) and *project precedent* (what does the codebase already do for similar cases?).

**Required research dimensions per fix:**
1. **Convention** — web research for production-system or framework conventions for this kind of fix. ≥2 cited sources per fix decision (per `PROCEDURE-pr-research.md` Phase 3 rigor).
2. **Project precedent** — does this fix have precedent in:
   - the project's named architecture-reference repository, if one is identified in `CLAUDE.md` or project docs?
   - existing code in this repo? (e.g., is there an analogous pattern in another module — trainer family, CLI verb, config validator — that the fix should mirror?)
3. **Rollback decisions** — for each speculative-fix commit identified in `## Code changes since baseline` (if that section exists), the Phase 6 output names whether to revert / keep / refactor it as part of THIS PR. Each decision is cited.
4. **Doc updates** — if the fix implies a new coding standard, draft the `docs/CONSTRAINTS.md` or `docs/CONVENTIONS.md` amendment here. It will be committed in Phase 7 alongside the fix.

**Output appended to PR's "Research findings" section:**

```markdown
### Phase 6 — Fix Design (YYYY-MM-DD)

**Bug N — proposed fix**

*Convention research*: [findings + cited URLs, ≥2 sources]
*Project precedent*: [architecture-reference repo (if any) + existing-codebase pattern (if any)]
*Disconfirming evidence*: [what counter-arguments / known drawbacks were searched for]
*Recommended fix*: [concrete change description, files affected, status: proven/convention/best-guess-given-constraints]
*Risks accepted*: ...

[Repeat per bug.]

**Rollback decisions** (if `## Code changes since baseline` is present):
- Commit [SHA]: revert / keep / refactor — citation: [research finding above]
- ...

**CONSTRAINTS.md / CONVENTIONS.md amendments needed**:
- [draft text — committed in Phase 7]
```

**Exit criteria**: ⚠️ **USER SIGNOFF REQUIRED** before Phase 7. Fix design + rollback decisions + doc amendments must all be explicitly approved.

---

### Phase 7: Implementation + Verification

**Goal**: Implement the fixes designed in Phase 6. Re-run the tests from Phase 5. They MUST now pass.

- Implement each fix exactly as designed in Phase 6. No deviations without going back and amending Phase 6 + getting fresh signoff.
- Apply rollback decisions from Phase 6.
- Commit any drafted `CONSTRAINTS.md` / `CONVENTIONS.md` amendments alongside the fix.
- If an active `docs/<x.y>/ROADMAP.md` exists, flip the PR's row `[ ] → [x]` in the same commit per `feedback_roadmap_flip_in_pr`.
- Run the test suite. Every Phase-5 failing test must turn green.
- Run the project's gates per `CLAUDE.md` "Build & Test Commands": `uv run pytest`, `uv run ruff check .`, `uv run basedpyright src/`, and `make test-golden` if the touched surface affects the golden contract.

**Output appended to PR's "Research findings" section:**

```markdown
### Phase 7 — Implementation + Verification (YYYY-MM-DD)

**Implementation summary**:
- Files modified: [list]
- Files reverted (per Phase 6 rollback decisions): [list]
- CONSTRAINTS.md / CONVENTIONS.md amendments committed: [list]
- ROADMAP row flipped (if applicable): [yes / N/A — no active version-scoped ROADMAP]

**Test results**:
- Bug 1: [test name] — was failing (Phase 5), now passing ✓
- Bug 2a: ...
- ...

**Other gates**:
- pytest: N/N pass (+M new from Phase 5)
- ruff: clean / [issues]
- basedpyright: clean / [issues]
- make test-golden (if relevant): clean / [tolerance excursions]

**PR link**: [URL]
```

**Exit criteria**: all bugs verified green via Phase-5 tests. PR link delivered to user for review.

## Failed-attempt protocol (during Phase 7)

If, while implementing a Phase 6-approved fix, the fix doesn't turn the Phase 5 test green:

1. **Stop**. Do not commit a follow-up "real fix" speculatively.
2. Write down what evidence proves the fix didn't work (test still fails / fails in a new way).
3. Loop back to **Phase 3 or Phase 6**: did Phase 3's mechanism description miss something, or did Phase 6 pick the wrong approach?
4. Update the PR file's "Failed-attempt log" section.
5. Get fresh user approval before attempting the next fix.

Without this protocol, speculative attempts accumulate: each "obvious" fix that doesn't actually pass the test becomes residue someone has to retroactively classify and roll back.

## Anti-patterns

Inherits all from `PROCEDURE-pr-research.md`. Adds:

- **Specifying solutions before bugs.** Writing "we'll fix this with X" before Phase 0 has locked the current/expected pair.
- **Treating multi-faceted bugs as one bug.** Atomic facets (cadence, concurrency, side effects, error handling, config-overlay precedence, persistence-vs-in-process state, etc.) each get their own current/expected pair.
- **Live-debugging speculatively.** Editing code to "see if this fixes it" without a Phase-5 failing test that the change is supposed to turn green.
- **Skipping the test-fails-first gate.** Implementing a fix in Phase 7 against a Phase-5 test that already passes (because the test was actually wrong) means the fix wasn't proven necessary by evidence.
- **Letting failed-attempt residue accumulate.** Every commit that didn't fix the bug must be classified in Phase 6 (revert / keep / refactor).
- **Bundling debug + feature.** Using the bug-fix PR as cover for an unrelated improvement. Split the PR.

## Time-decay policy

Same as `PROCEDURE-pr-research.md`. State-assessment re-run if the project's defined staleness threshold has passed since the bug spec was locked (per `docs/CONSTRAINTS.md`, the rux-ml threshold is 60 days).

## Doc-alignment policy

This procedure file (and `docs/DEBUG-PR-STANDARDS.md`) MUST stay aligned with the procedure actually followed. After each phase that surfaces a refinement to the procedure (e.g., a need to add a sub-step, clarify a rule, etc.), update both docs in the same commit before proceeding to the next phase. The docs are living artifacts; drift between them and the procedure is itself an anti-pattern.
