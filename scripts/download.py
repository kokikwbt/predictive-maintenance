#!/usr/bin/env python3
"""Download one supported dataset using its configured source."""

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import download, download_command, supported_downloads  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=supported_downloads())
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--variant",
        help="download variant, such as 2025-q1 for Backblaze",
    )
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="print the external download command without executing it",
    )
    args = parser.parse_args()

    if args.print_command:
        print(
            download_command(
                args.dataset, args.output_dir, variant=args.variant
            )
        )
        return

    result = download(
        args.dataset,
        args.output_dir,
        extract=not args.no_extract,
        overwrite=args.overwrite,
        variant=args.variant,
    )
    print("Archive: {}".format(result["archive"]))
    print("SHA-256: {}".format(result["sha256"]))
    if result["extracted"] is not None:
        print("Extracted: {}".format(result["extracted"]))


if __name__ == "__main__":
    main()
