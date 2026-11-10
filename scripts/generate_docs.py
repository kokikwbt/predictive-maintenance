#!/usr/bin/env python3
"""Synchronize generated documentation with dataset metadata.json files."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pdmdata.catalog.docs import (  # noqa: E402
    update_all_readmes,
    update_catalog_readme,
    update_notebooks_readme,
)


def main() -> None:
    for path in update_all_readmes():
        print(path.relative_to(ROOT))
    print(update_catalog_readme().relative_to(ROOT))
    print(update_notebooks_readme().relative_to(ROOT))


if __name__ == "__main__":
    main()
