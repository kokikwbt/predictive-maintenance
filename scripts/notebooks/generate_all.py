#!/usr/bin/env python3
"""Regenerate all script-managed dataset, showcase, and task notebooks."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

NOTEBOOKS_SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_SCRIPT_DIR = NOTEBOOKS_SCRIPT_DIR / "datasets"
sys.path.insert(0, str(NOTEBOOKS_SCRIPT_DIR))


def _run_generator(script_path: Path) -> None:
    """Import and execute the ``main()`` entry point of a generator script."""
    spec = importlib.util.spec_from_file_location(
        script_path.stem, script_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load generator script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


def main() -> None:
    """Run dataset, visualization-showcase, and task notebook generators."""
    generators = [
        *sorted(DATASETS_SCRIPT_DIR.glob("generate_*_notebook.py")),
        NOTEBOOKS_SCRIPT_DIR / "generate_visualization_showcase.py",
        NOTEBOOKS_SCRIPT_DIR / "generate_task_notebooks.py",
    ]
    for script_path in generators:
        _run_generator(script_path)


if __name__ == "__main__":
    main()
