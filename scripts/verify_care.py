#!/usr/bin/env python3
"""Verify local CARE files and write an experiment inventory (no network)."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pdmdata.datasets.care import verify  # noqa: E402

if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
