# PR-012: Container

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PR** (research-backed at design time).

## Research findings

### State Assessment (2026-05-16)

**Current state**:
- No `Dockerfile`, no `docker-compose.yml`, no `.dockerignore` exist at repo root — PR-001 deferred containerization to this PR.
- `.gitignore:55-56` already lists `.docker-image-digest` (added preemptively).
- `src/rux_ml/runs/attrs.py:75` declares `image_digest: str | None = None  # PR-012` — Optional today; PR-012 only produces the `.docker-image-digest` artifact, PR-013 wires the read-path into `TrialAttrs`.
- `src/rux_ml/_internal/env.py` (`pin_threads`) is the canonical thread-pinning surface; subprocess-per-trial sets thread env vars inside the container; no container-level conflict.
- `src/rux_ml/config/memory.py:8` watchdog threshold is `28.0 GB` — exactly the `36 - 4 OS - 4 Docker = 28 GB` budget. Compose `mem_limit: 32g` + `mem_reservation: 28g` aligns: kernel + Docker overhead consume the 4 GB headroom above the in-process watchdog.
- `docs/ARCHITECTURE.md:412-426` already documents `--memory=32g` / `--memory-reservation=28g`; this PR makes it real.
- `pyproject.toml:31` defines `rux-ml = "rux_ml.cli:app"` — `ENTRYPOINT ["tini", "--", "rux-ml"]` resolves correctly after `uv sync`.
- `pyproject.toml:2` already requires `uv_build>=0.11.14,<0.12` — uv runtime image pin should match (`0.11.14`).
- PR-008 trial_runner's strict module-loading order is unaffected by the container.

**Assumptions at PR draft time**:
1. Base image `nvidia/cuda:12.4.1-devel-ubuntu22.04` tag is still live with a current digest.
2. Astral's documented uv-in-Docker pattern is current (multi-stage copy-in).
3. XGBoost CI still aligns with CUDA 12.4.
4. NVIDIA Container Toolkit `gpus: all` Compose syntax is unchanged.
5. `tini` is available in Ubuntu 22.04 main repos via `apt`.
6. Rust toolchain + maturin install at v0 (no crate yet) is correct per D1 mandate.

**Stale assumptions**:
- **Assumption 4 (Compose `gpus: all`)**: STALE — Docker Compose v2 no longer documents `gpus: all` as a service key. Canonical syntax is now `deploy.resources.reservations.devices` with `driver: nvidia`, `capabilities: [gpu]`. CLI flag `--gpus all` still works for `docker run` / `docker compose run`.
- **Spec line 26 (`uv sync --frozen`)**: STALE — Astral's current Docker docs document `--locked` as the recommended flag (slightly stricter "lockfile must exist and not need updating" check). `--frozen` still works but the canonical example uses `--locked`.
- **Spec line 21 (system `xxhash` apt dep)**: STALE — Python `xxhash>=3.7` wheel bundles its own C extension; no system lib required. The `xxhsum` CLI is not used anywhere in the workbench. Drop the apt dep.
- All other assumptions hold (verified Phase 3 below).

**New constraints learned from prior PRs**:
- PR-007 + PR-015 added two more layers to the per-trial hash set: `cv_cfg_hash` (PR-015) and the full 8-layer `*_cfg_hash` block (PR-007). Orthogonal to this PR (container doesn't write provenance).
- PR-008's spawn-semantics requirement is honored automatically by `docker compose run` (one process per invocation).
- PR-011's `peak_rss_mb` is now required on `TrialAttrs` — the watchdog runs inside the container; the 32 GB cgroup hard cap + 28 GB soft cap is consistent with the 28 GB watchdog threshold.

### Research Questions

**Must-answer**:

1. **Q1** — Current `sha256:` digest for `nvidia/cuda:12.4.1-devel-ubuntu22.04` — success: a single `@sha256:<hex>` string to paste into `FROM` line.
2. **Q2** — Astral uv-in-Docker pattern current as of 2026-05 — success: exact `COPY --from=...` line + recommended env vars + `uv sync` command.
3. **Q3** — XGBoost CI CUDA version still 12.4 — success: confirm or report drift with cited rationale.
4. **Q4** — NVIDIA Container Toolkit / Compose v2 GPU syntax 2026 unchanged — success: exact YAML block for `services.rux-ml`.

**Dependencies**: Q1-Q4 all independent. Q3 may trigger AMEND if CI has moved; Q4 may trigger AMEND if Compose syntax has changed.

**Explicitly excluded from this round** (nice-to-have):
- CUDA 13 cutover (defer — XGBoost CI added it as additive variant, not replacement).
- Multi-arch image build (no need at v0 — single-machine).
- Pushing image to a registry (out of scope per PR-012 NOT-in-scope).

### Findings

**Q1: Current `nvidia/cuda:12.4.1-devel-ubuntu22.04` digest**

- *Options considered:*
  - **Option A: pin to current Docker Hub digest** — single live `sha256` for the 12.4.1-devel tag.
    - Sources: `https://hub.docker.com/v2/repositories/nvidia/cuda/tags/12.4.1-devel-ubuntu22.04`, `https://hub.docker.com/r/nvidia/cuda/tags?name=12.4.1-devel-ubuntu22.04`
    - Pros: D1 PROVEN base image; XGBoost CI parity; non-EoL CUDA version.
    - Cons: 2024-vintage OS packages carry stale CVEs; mitigated by `apt upgrade` in Dockerfile + host-side `nvidia-container-toolkit` providing the runtime driver.
  - **Option B: re-pick to `12.6.x` or `12.8.x`** — newer base.
    - Cons: diverges from D1 PROVEN decision; XGBoost CI still lists 12.4.1 in the user-facing examples; CUDA 13 added as additive variant (Q3) not replacement. No cited reason to move at v0.

- *Disconfirming evidence sought:* searched for "NVIDIA CUDA tag rebuilt" / "12.4.1-devel removed" — `tag_last_pushed: 2024-04-24` indicates NVIDIA has not rebuilt this minor; this is expected behavior for EoL minors (they freeze, they don't disappear). Digest is stable.

- *Recommendation:* Option A
  - **Status**: PROVEN (Docker Hub manifest API)
  - **Why**: matches D1 decision + XGBoost CI lane.
  - **Digest**: `sha256:5645fec64549cc35930eee9d85aafd2b0006c0c3f22632be5a1d85e2604e9749`
  - **Risks accepted**: 2024-era OS package CVEs; mitigated by `apt update && apt upgrade -y` step in Dockerfile.

**Q2: Astral uv-in-Docker pattern**

- *Options considered:*
  - **Option A: multi-stage `COPY --from=ghcr.io/astral-sh/uv:<ver>@sha256:<digest>`** — Astral's current canonical pattern.
    - Sources: `https://docs.astral.sh/uv/guides/integration/docker/`, `https://github.com/astral-sh/uv/releases`
    - Pros: digest-pinnable per `docs/CONSTRAINTS.md`; smallest image overhead; no pip layer needed; no curl-pipe-bash.
    - Cons: requires resolving the GHCR digest separately (resolved via HTTP API — see digest below).
  - **Option B: `pip install uv`** — pip then uv.
    - Cons: adds pip layer; uv now self-bootstraps without pip.
  - **Option C: `curl ... | sh`** — install script.
    - Cons: not pin-friendly; runs untrusted network code at build.

- *Disconfirming evidence sought:* searched Astral docs for any deprecation of the copy-in pattern — none found; the pattern is THE recommended Docker integration.

- *Recommendation:* Option A
  - **Status**: PROVEN (Astral official docs)
  - **Version pinned**: `0.11.14` (matches `pyproject.toml` `uv_build>=0.11.14,<0.12` floor exactly).
  - **GHCR index digest** (resolved via `https://ghcr.io/v2/astral-sh/uv/manifests/0.11.14` `Docker-Content-Digest` header on 2026-05-16): `sha256:1025398289b62de8269e70c45b91ffa37c373f38118d7da036fb8bb8efc85d97`
  - **`uv sync` flag**: `--locked` (current Astral canonical — supersedes `--frozen` in 2026 docs; stricter "lockfile must exist and not need updating" check)
  - **Recommended env**: `UV_COMPILE_BYTECODE=1`, `UV_LINK_MODE=copy` (per Astral Docker doc)
  - **Risks accepted**: image index pinned, not platform-specific manifest — BuildKit resolves linux/amd64 from the index automatically.

**Q3: XGBoost CI CUDA version**  ⚠ informational AMEND

- *Options considered:*
  - **Option A: stay on CUDA 12.4** — XGBoost CI still builds the CUDA 12 lane (`build-cuda.sh --cuda-version 12`); user-facing `ci.rst` shows `CUDA_VERSION_ARG: "12.4.1"`.
    - Sources: `https://github.com/dmlc/xgboost/blob/master/.github/workflows/main.yml`, `https://github.com/dmlc/xgboost/blob/master/ops/pipeline/build-cuda.sh`, `https://github.com/dmlc/xgboost/blob/master/doc/contrib/ci.rst`, `https://github.com/dmlc/xgboost/blob/master/doc/tutorials/kubernetes.rst`
    - Pros: D1 PROVEN base; no churn.
    - Cons: trails by ~1 CUDA major.
  - **Option B: move to CUDA 13** — XGBoost CI now also offers a CUDA 13 lane (additive variant, not replacement).
    - Cons: no federated/RMM support in CUDA 13 lane yet (per `build-cuda.sh` flags); changing base for no measured benefit; out of scope for this PR.

- *Disconfirming evidence sought:* none found that CUDA 12 is sunset; XGBoost CI maintains both lanes; the user-facing examples still cite CUDA 12.4.1.

- *Recommendation:* Option A (stay on 12.4.1).
  - **Status**: PROVEN (XGBoost CI workflows visible at `dmlc/xgboost@master`)
  - **Watch note** (added to `docs/0.0/RESEARCH-BACKLOG.md` Drift Watch): "XGBoost CI added CUDA 13 variant 2026-05; revisit at PR-012 + 6 months."

**Q4: NVIDIA Container Toolkit / Compose syntax**  ⚠ AMEND

- *Options considered:*
  - **Option A: `deploy.resources.reservations.devices` block** — current Compose v2 canonical.
    - Sources: `https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html`, `https://docs.docker.com/compose/how-tos/gpu-support/`
    - Pros: documented current pattern; explicit driver + capability declaration; future-compatible.
    - Cons: more verbose than the legacy single-key form.
  - **Option B: `gpus: all` top-level service key** — what PR-012 spec line 32 originally said.
    - Cons: no longer in Docker Compose docs (2026); some Compose versions still parse it as legacy shortcut but undocumented; brittle.

- *Disconfirming evidence sought:* searched Docker docs + Compose changelog for a 2026 reversal — none found; the `deploy.resources.reservations.devices` syntax is THE canonical form.

- *Recommendation:* Option A (AMEND PR-012 spec line 32).
  - **Status**: PROVEN (Docker + NVIDIA official docs)
  - **YAML block**:
    ```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ```
  - **CLI flag `--gpus all`** still works for `docker run` / `docker compose run`.
  - **Risks accepted**: none.

### Synthesis

**Outcome**: **Amend** — three findings amend the PR-012 spec; user approved each on 2026-05-16.

**Changes to this PR** from research:
- **AMEND 1**: replace `gpus: all` (spec line 32) with `deploy.resources.reservations.devices` block in `docker-compose.yml`.
- **AMEND 2**: replace `uv sync --frozen` (spec line 26) with two-step `uv sync --locked --no-install-project --no-dev` (deps-cache layer) then `uv sync --locked` (after `COPY src/`).
- **AMEND 3**: pin uv image to `ghcr.io/astral-sh/uv:0.11.14@sha256:1025398289b62de8269e70c45b91ffa37c373f38118d7da036fb8bb8efc85d97` (resolved from GHCR HTTP API on 2026-05-16; version matches `pyproject.toml` `uv_build` floor).
- **Mechanical correction**: drop `xxhash` from system apt deps (Python wheel bundles it); add `curl` (needed for rustup-init download).
- **Spec gap fix**: add `.dockerignore` (not in original spec) to exclude `.venv`, `.pytest_cache`, `.hypothesis`, `studies/`, `registry/`, `data/`, `logs/`, `.git`, `.ruff_cache`, `__pycache__`, `.basedpyright_cache` from build context.
- **Locked sub-decisions**:
  - **A1**: `COPY --from=ghcr.io/astral-sh/uv:0.11.14@sha256:1025...` to `/usr/local/bin/`
  - **B1**: `rustup-init` non-interactive minimal profile (D1 mandate; future-proof for first Rust crate)
  - **C1**: `uv tool install maturin` (matches `uv run maturin develop --uv` from CLAUDE.md)
  - **D1**: `apt install tini` (Ubuntu 22.04 ships it)
  - **G1**: `mem_limit: 32g` + `mem_reservation: 28g` at service level
  - **H confirmed**: `.docker-image-digest` written from `docker image inspect rux-ml:local --format='{{.Id}}'`

**Changes to ARCHITECTURE.md**:
- Memory & Parallelism table (line 416-417) already documents `--memory=32g` / `--memory-reservation=28g` — no change.
- Reproducibility Architecture point 2 (line 398) already says "Container pinned by `@sha256:...`" — no change.

**Changes to CONVENTIONS.md**:
- Add new "Container conventions (per PR-012)" subsection: uv-copy-in pattern + `--locked` flag + `deploy.resources.reservations.devices` over `gpus: all` + `.docker-image-digest` capture mechanism.

**Changes to CONSTRAINTS.md**:
- None (digest-pinning rule was already there).

**Changes to ROADMAP.md**:
- `[ ]` → `[x]` flip on the PR-012 row — happens in the implementation commit per [[feedback-roadmap-flip-in-pr]].

**Changes to RESEARCH-BACKLOG.md**:
- PR-012 row: `state-assessed 2026-05-16` + `implementation-cleared 2026-05-16`.
- Drift Watch addendum: "XGBoost CI added CUDA 13 variant 2026-05; revisit at PR-012 + 6 months."
- PR-012 drift-risk note updated to reflect resolved digest + locked syntax.

**New PRs that must come first**: none.

**Research-backed details now locked in this PR**:
- Base image digest: `sha256:5645fec64549cc35930eee9d85aafd2b0006c0c3f22632be5a1d85e2604e9749`
- uv image digest: `sha256:1025398289b62de8269e70c45b91ffa37c373f38118d7da036fb8bb8efc85d97` (version `0.11.14`)
- GPU Compose syntax: `deploy.resources.reservations.devices` (driver: nvidia, count: 1, capabilities: [gpu])
- uv sync flag: `--locked` (two-step deps-then-project pattern)
- System deps: `python3.12 python3.12-dev git ca-certificates tini curl` (NO `xxhash`)

### Gate Check

- Premise still valid: ✓ (containerized runtime is foundational and orthogonal to all later PRs)
- No prerequisite PRs surfaced: ✓ (depends only on PR-001, already merged)
- User approved updated spec: ✓ (2026-05-16)
- Implementation cleared: ✓ (2026-05-16)

---

## Scope

Build a reproducible container per D1 + D9 + D10. After this PR, the workbench runs in a container with pinned CUDA, GPU passthrough, and a hard memory cap.

- `Dockerfile`:
  - `FROM nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:<digest>` (state assessment must look up the current digest)
  - System deps: `python3.12`, `python3.12-dev`, `git`, `xxhash`, `ca-certificates`, `tini`
  - Install `uv` from `astral/uv` binary copy-in pattern (per Astral docs)
  - Install `rustup` + `stable` toolchain (per D1 — even though we have no Rust at v0; future-proofing the image is cheap and matches the maturin convention)
  - Non-root `rux` user
  - `WORKDIR /workbench`
  - Copy `pyproject.toml` + `uv.lock`; `uv sync --frozen`
  - `COPY src/ /workbench/src/` (and only what we need at runtime)
  - `ENTRYPOINT ["tini", "--", "rux-ml"]`
- `docker-compose.yml`:
  - `services.rux-ml`:
    - `build: .`
    - `gpus: all` (NVIDIA Container Toolkit)
    - `mem_limit: 32g` and `mem_reservation: 28g` per D10
    - Volume mounts: `./configs`, `./studies`, `./registry`, `./data`, `./logs`
    - Pass `WORKBENCH_HOME=/workbench` env
- `Makefile` targets: `docker-build`, `docker-run` (forwards args to `rux-ml`), `docker-shell`
- Tests:
  - Build the image (`make docker-build`) — must succeed
  - Run `docker compose run rux-ml --version` — must print version
  - Run `docker compose run rux-ml --help` — must show all verb groups
  - GPU passthrough sanity: `docker compose run rux-ml python -c "import xgboost as xgb; print(xgb.config_context())"` — must show CUDA
  - Memory cap sanity: `docker compose run --rm rux-ml ...` with `mem_limit=32g` honored (verify via `cat /sys/fs/cgroup/memory.max` inside container)
  - Record the digest of the built image in a `.docker-image-digest` file (gitignored) — `image_digest` in `TrialAttrs` is sourced from this for trials run inside the container

NOT in scope: shipping the image to a registry (out of scope for solo); CI image building (no CI yet).

## Dependencies

PR-001 (only — independent of layer PRs; Container can be built in parallel with Phase B-G as long as the image installs the package correctly).

## Architecture section implemented

`docs/ARCHITECTURE.md` → "System Overview" (containerized runtime), "Memory & Parallelism Architecture" (container `--memory=32g`), "Reproducibility Architecture" point 2 (image digest pinning).

## Verification criteria

- [ ] `Dockerfile` base is pinned by `@sha256:<digest>` (not a tag)
- [ ] `make docker-build` succeeds and produces an image
- [ ] `docker compose run rux-ml --version` works
- [ ] `docker compose run rux-ml --help` shows all verb groups
- [ ] GPU is visible inside the container (`nvidia-smi` or equivalent XGBoost call works)
- [ ] Memory cap of 32 GB is enforced (cgroup verification)
- [ ] Volume mounts let trials write to `studies/`, `registry/`, `data/`, `logs/` on the host
- [ ] Image digest is captured for later recording in `TrialAttrs.image_digest` (PR-013 may wire this; this PR records the file)
- [ ] `Cargo.lock` is committed when first Rust crate lands; `uv.lock --frozen` honored on every build

## Research backing

Tier 1:

- D1: [XGBoost CI uses nvidia/cuda devel](https://xgboost.readthedocs.io/en/stable/contrib/ci.html), [uv in Docker (Astral)](https://docs.astral.sh/uv/guides/integration/docker/)
- D9: [Docker digests](https://docs.docker.com/dhi/core-concepts/digests/), [Chainguard digest pinning](https://edu.chainguard.dev/chainguard/chainguard-images/how-to-use/container-image-digests/)
- D10: [Netdata cgroups v2](https://www.netdata.cloud/academy/diagnosing-linux-cgroups/)

State assessment must:
- Look up the current `nvidia/cuda:12.4.1-devel-ubuntu22.04` digest (NVIDIA may have rebuilt the image)
- Verify XGBoost CI is still on CUDA 12.4 (or update to whatever current is, with a documented justification)
- Check NVIDIA Container Toolkit installation steps for any 2026 changes

## Notes

- Per `docs/CONSTRAINTS.md`, container digest pinning is non-negotiable. If state assessment surfaces a tag rebuild, update the Dockerfile in this PR before merging.
- The Rust toolchain is installed even though no crate exists at v0 — keeping the image future-proof avoids a rebuild churn when D13's first crate lands.
- `tini` as PID 1 ensures correct signal forwarding for subprocess-per-trial under `docker compose run`.
