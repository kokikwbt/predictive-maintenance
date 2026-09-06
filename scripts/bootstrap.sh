#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIRECTORY}/.." && pwd)"

cd "${PROJECT_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv is required but was not found in PATH." >&2
    exit 1
fi

echo "Synchronizing the locked Python 3.11 environment"
uv sync --locked

echo "Registering the Jupyter kernel"
uv run --locked python -m ipykernel install \
    --user \
    --name pdmdata \
    --display-name "Python (pdmdata)"

echo
echo "Setup complete."
echo "Download a dataset when needed: uv run --locked python scripts/download.py hydsys"
echo "Start JupyterLab with: uv run --locked jupyter lab"
