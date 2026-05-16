# Constraints

Hard rules that must never be violated. These are enforced by Claude at all times.

---

## Structural Constraints

These apply to every project built with this template.

### No Phantom Implementations (NON-NEGOTIABLE)

A step is NOT complete if:
- A function exists but returns a default, stub, or placeholder value
- A module is declared but never called from the main flow
- Tests only verify something exists, not that it works correctly

Every PR must include:
1. A test that exercises the **actual behavior**
2. Explicit listing of any stubs or TODOs in the PR description
3. Proof of end-to-end data flow where applicable

### Documentation Accuracy (NON-NEGOTIABLE)

Every code change must include corresponding doc updates in the same commit. Before writing docs, check the actual diff — base doc updates on what changed, not memory. Docs must never describe behavior that doesn't exist in code.

### One PR, One Thing

Each PR is a single, reviewable change. No "while I'm here I'll also add..." — that's scope creep. Each PR references the specific section of the architecture it implements.

### Config Over Hardcoding

All configurable values come from config files. Zero hardcoded parameters for behavior that might change. If a value could reasonably vary between environments or over time, it belongs in config.

### Research-Backed Decisions (NON-NEGOTIABLE)

Every architectural and significant design decision must be backed by research from **reputable sources**. "I think this is right" is not sufficient.

**Reputable sources:**
- Production system documentation
- Official framework source and docs
- Published post-mortems and engineering blog posts from serious engineering teams
- Battle-tested open-source code with meaningful adoption

**Not sufficient:**
- LLM intuition
- Marketing pages
- Personal blog posts without engineering weight
- StackOverflow answers without corroborating evidence

**Process:** decisions without research backing must be explicitly flagged as unresearched in the relevant PR or design doc, and researched before implementation begins. `docs/DESIGN-log.md` tracks which decisions have research and which don't. `PROCEDURE-design-planning.md` integrates research rounds into Phase 2 (Decisions).

### PR Research Procedure Required (NON-NEGOTIABLE)

No PR is implemented until `PROCEDURE-pr-research.md` has been followed and its findings are documented in the PR file's `## Research findings` section.

**Applies to all PRs — including PRs that were research-backed at design time.** State drifts between design and implementation. The procedure's Phase 1 (State Assessment) catches drift before implementation begins.

**Enforcement:**
- Every PR file starts from `prs/PR-TEMPLATE.md`, which includes a `## Before Implementation (NON-NEGOTIABLE)` section requiring this procedure
- Every PR file has a `## Research findings` section that must be populated before implementation
- PRs without completed research findings are rejected
- Research findings include a state-assessment date; if implementation hasn't started within the project's staleness threshold, re-run state assessment per the time-decay policy in `PROCEDURE-pr-research.md`

State drifts. Research must be validated before code.

---

## Domain Constraints

These are the project-specific non-negotiables established during the design session (2026-05-14 → 2026-05-15). See `docs/DESIGN-log.md` for the research trail behind each.

### No Web UI / No Server (NON-NEGOTIABLE)

The workbench runs on a single machine and is operated via CLI (`rux-ml`) and Python imports. No long-running server processes, no web dashboards, no MLflow UI, no Optuna dashboard, no Aim UI.

**Implication:** any candidate library that requires a server to be useful is disqualified (lakeFS, Pachyderm, full MLflow Tracking server, Aim UI, etc.). Libraries with optional server modes are acceptable when their headless/CLI/Python path is the primary one.

### Single-GPU, Sequential Trials (NON-NEGOTIABLE)

HPO trials run **sequentially** with `n_jobs=1`. XGBoost `device="cuda"` saturates the RTX 4090; concurrent trials cause OOM and contention. Multi-GPU code paths are out of scope.

### CUDA + `fork` is Forbidden (NON-NEGOTIABLE)

Any subprocess that uses CUDA must be created with `spawn` start method (or via `subprocess.run` of a sibling script). The CUDA runtime cannot be re-initialized in a forked process. Subprocess-per-trial isolation (per D10/D16) uses `subprocess.run` of `python -m rux_ml._internal.trial_runner`.

### Tolerance-Based Golden Tests Only (NON-NEGOTIABLE)

XGBoost GPU `hist` is near-deterministic but not bit-exact across hardware/CUDA versions. Golden regression tests must use `np.testing.assert_allclose` with explicit `atol`/`rtol` plus metric-tolerance bands. **Exact-hash regression tests are forbidden** for any GPU-trained model output — they will break the first time CUDA/sklearn/XGBoost is upgraded.

### Zero Hardcoded Parameters (NON-NEGOTIABLE — reinforces structural rule)

Every value that could vary between datasets, problems, environments, or experiments lives in TOML config. Per-layer Pydantic models in `src/rux_ml/config/` are the single source of truth. CLI dot-path overrides (`--training.learning_rate=0.05`) and env vars (`RUXML_TRAINING__LEARNING_RATE=0.05`) are the only acceptable runtime overrides.

### Per-Trial Provenance Triple (NON-NEGOTIABLE)

Every Optuna trial — sweep or one-off — records the following in `trial.set_user_attr(...)`:

- `data_hash` — composite hash of dataset bytes + canonical projection (per D9)
- `data_cfg_hash`, `features_cfg_hash`, `training_cfg_hash`, `tuning_cfg_hash` — per-layer config hashes
- `root_cfg_hash` — full config hash (excluding elided fields)
- `git_sha` — code version
- `entropy_hex` — `SeedSequence` entropy (per D9)
- `image_digest` — container `@sha256:...` digest
- `xgboost_version`, `cuda_runtime_version`, `gpu_model`, `driver_version`
- `omp_threads`, `peak_rss_mb`

Missing any of these breaks reproducibility and the trial must not be promoted to the registry.

### Reuse Over Reinvent (NON-NEGOTIABLE)

Custom code (Python or Rust) must only be written when no mature, battle-tested library covers the need. This applies to Rust+PyO3 explicitly — the first custom Rust crate lands only when a profile shows a Python hot path consuming >5% of a real workbench task. Polars expression plugins (`pyo3-polars`) are the documented escape hatch for row-dependent feature transforms.

### No Phantom Implementations — Reaffirmed for ML

Stub model factories, unused config fields, untested transformers, and "TODO: actually train the model" placeholders are all phantom implementations. Every PR must include a test that exercises actual behavior end-to-end through the layer it touches.

### Two-File Model Bundle (NON-NEGOTIABLE)

Promoted models are stored as **two files plus a manifest**: `pipeline.skops` (sklearn FE Pipeline via skops.io) + `model.ubj` (XGBoost Booster via `save_model`) + `manifest.json` (Pydantic-validated). Combined pickles are forbidden — they tie the XGBoost booster to Python/sklearn versions unnecessarily.

### Container Digest Pinning (NON-NEGOTIABLE)

The Docker base image must be pinned by `@sha256:...` digest, not by tag. NVIDIA deletes EoL CUDA image tags. The pinned digest is recorded per-trial in `image_digest`.

---

## Project Staleness Threshold

**Research is considered stale after 60 days for this project.**

Any PR marked `fully-researched` or `state-assessed` more than 60 days before implementation begins must re-run Phase 1 (State Assessment) per `PROCEDURE-pr-research.md`. The 60-day choice is **BEST-GUESS** (no specific source) and explicitly acknowledged by the user during Phase 1 of the design session; revisit if it produces too many re-runs (lower) or too many drift surprises (raise).
