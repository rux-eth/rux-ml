# PR-012: Container

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PR** (research-backed at design time).

## Research findings

_To be populated by `PROCEDURE-pr-research.md`._

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
