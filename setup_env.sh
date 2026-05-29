#!/usr/bin/env bash
# setup_env.sh — create or update the spinhance micromamba environment
set -euo pipefail

ENV_NAME="spinhance"

# ── 1. Check for micromamba ──────────────────────────────────────────────────
if ! command -v micromamba &> /dev/null; then
  echo "micromamba not found. Installing..."
  # macOS (Apple Silicon or Intel)
  if [[ "$(uname)" == "Darwin" ]]; then
    brew install micromamba 2>/dev/null || \
      curl -Ls https://micro.mamba.pm/api/micromamba/osx-arm64/latest \
        | tar -xvj bin/micromamba --strip-components=1 -C /usr/local/bin/
  # Linux
  else
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
      | tar -xvj bin/micromamba --strip-components=1 -C /usr/local/bin/
  fi
  eval "$(micromamba shell hook --shell bash)"
fi

# ── 2. Create or update environment ─────────────────────────────────────────
if micromamba env list | grep -q "^${ENV_NAME}"; then
  echo "Updating existing '${ENV_NAME}' environment..."
  micromamba env update -n "${ENV_NAME}" -f environment.yml --prune -y
else
  echo "Creating '${ENV_NAME}' environment..."
  micromamba env create -f environment.yml -y
fi

# ── 3. Done ──────────────────────────────────────────────────────────────────
echo ""
echo "✓ Environment ready. Activate with:"
echo "    micromamba activate ${ENV_NAME}"
