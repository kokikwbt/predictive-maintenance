#!/usr/bin/env python3
"""Synchronize generated documentation with datasets/*/metadata.json."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.docs import update_all_readmes, update_repository_readme  # noqa: E402


def main() -> None:
    for path in update_all_readmes():
        print(path.relative_to(ROOT))
    print(update_repository_readme().relative_to(ROOT))


if __name__ == "__main__":
    main()
