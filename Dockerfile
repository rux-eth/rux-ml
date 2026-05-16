# syntax=docker/dockerfile:1.7
#
# rux-ml container — single-stage GPU-enabled image (per PR-012).
#
# Base image digest pinned per docs/CONSTRAINTS.md "Container Digest Pinning"
# (Docker Hub manifest API, resolved 2026-05-16):
#   nvidia/cuda:12.4.1-devel-ubuntu22.04
#
# uv image digest pinned to ghcr.io/astral-sh/uv:0.11.14 (matches the
# uv_build floor in pyproject.toml; OCI index digest from ghcr.io HTTP API
# resolved 2026-05-16 — BuildKit picks linux/amd64 automatically).

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:5645fec64549cc35930eee9d85aafd2b0006c0c3f22632be5a1d85e2604e9749

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps. No `xxhash` package — the Python `xxhash` wheel bundles its
# own C extension (per PR-012 Phase 1 finding). `curl` is required for
# rustup-init download.
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        tini \
    && rm -rf /var/lib/apt/lists/*

# Rust toolchain (minimal stable) per D1 mandate. No crate exists at v0,
# but installing now avoids an image rebuild when the first crate lands
# (per D13's profile-driven trigger). Installed system-wide so the non-root
# `rux` user inherits cargo + rustc via PATH.
ENV RUSTUP_HOME=/opt/rustup \
    CARGO_HOME=/opt/cargo \
    PATH=/opt/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable --profile minimal --no-modify-path \
    && chmod -R a+rX /opt/cargo /opt/rustup

# uv binary copy-in pattern per Astral's Docker integration guide.
COPY --from=ghcr.io/astral-sh/uv:0.11.14@sha256:1025398289b62de8269e70c45b91ffa37c373f38118d7da036fb8bb8efc85d97 \
    /uv /uvx /usr/local/bin/

# Non-root user. uid 1000 lines up with the typical host user so bind-mounted
# host directories (./studies, ./registry, ./data, ./logs) round-trip ownership.
RUN useradd --create-home --shell /bin/bash --uid 1000 rux \
    && mkdir -p /workbench /opt/uv \
    && chown -R rux:rux /workbench /opt/uv

USER rux
WORKDIR /workbench

# Astral-recommended Docker env. `UV_PYTHON_INSTALL_DIR` keeps the managed
# Python under /opt so it survives layer rebuilds; `UV_PROJECT_ENVIRONMENT`
# pins the venv path so the ENTRYPOINT can reach `rux-ml` by absolute path.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_TOOL_DIR=/opt/uv/tools \
    UV_PROJECT_ENVIRONMENT=/workbench/.venv \
    PATH=/workbench/.venv/bin:/opt/uv/tools/bin:/opt/cargo/bin:$PATH

# Python 3.12 + maturin, both uv-managed. Ubuntu 22.04 ships Python 3.10
# system-wide; we use uv-managed Python instead of the deadsnakes PPA to
# match Astral's current Docker integration recipe (single source of Python
# truth for both build and runtime).
RUN uv python install 3.12 \
    && uv tool install maturin

# Deps-only layer — cached as long as pyproject.toml + uv.lock are unchanged.
# `--no-install-project` skips the project itself; `--no-dev` excludes the
# `dev` dependency group (pytest, basedpyright, etc. — not needed at runtime).
COPY --chown=rux:rux pyproject.toml uv.lock README.md /workbench/
RUN --mount=type=cache,target=/home/rux/.cache/uv,uid=1000,gid=1000 \
    uv sync --locked --no-install-project --no-dev

# Project source + final sync (installs rux-ml into the venv).
COPY --chown=rux:rux src/ /workbench/src/
RUN --mount=type=cache,target=/home/rux/.cache/uv,uid=1000,gid=1000 \
    uv sync --locked --no-dev

# tini as PID 1 forwards SIGTERM/SIGINT correctly to the rux-ml process and
# any subprocess-per-trial children spawned via `python -m rux_ml._internal.trial_runner`
# (PR-008 spawn-semantics requirement).
ENTRYPOINT ["/usr/bin/tini", "--", "/workbench/.venv/bin/rux-ml"]
CMD ["--help"]
