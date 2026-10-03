#!/usr/bin/env python3
"""Verify local CARE files and write an experiment inventory (no network)."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pdmdata.datasets.care.validation import verify


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
