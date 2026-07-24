#!/usr/bin/env bash

set -euo pipefail

ENVIRONMENT_NAME="pmdata"
SCRIPT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIRECTORY}/.." && pwd)"

cd "${PROJECT_ROOT}"

if ! command -v conda >/dev/null 2>&1; then
    echo "Error: Conda is required but was not found in PATH." >&2
    exit 1
fi

if conda run --name "${ENVIRONMENT_NAME}" python --version >/dev/null 2>&1; then
    echo "Using existing Conda environment: ${ENVIRONMENT_NAME}"
else
    echo "Creating Conda environment: ${ENVIRONMENT_NAME}"
    conda create --yes --name "${ENVIRONMENT_NAME}" python=3.11 pip
fi

echo "Installing project requirements"
conda run --name "${ENVIRONMENT_NAME}" python -m pip install --upgrade pip
conda run --name "${ENVIRONMENT_NAME}" python -m pip install -r requirements.txt

echo "Registering the Jupyter kernel"
conda run --name "${ENVIRONMENT_NAME}" python -m ipykernel install \
    --user \
    --name "${ENVIRONMENT_NAME}" \
    --display-name "Python (${ENVIRONMENT_NAME})"

echo "Downloading all directly supported datasets"
conda run --name "${ENVIRONMENT_NAME}" python scripts/download_all.py

echo
echo "Setup complete."
echo "Activate the environment with: conda activate ${ENVIRONMENT_NAME}"
echo "Start JupyterLab with: jupyter lab"
