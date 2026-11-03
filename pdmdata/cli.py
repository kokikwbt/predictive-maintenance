"""Command-line interface for catalog inspection, downloads, and verification."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl

from pdmdata.catalog import summary
from pdmdata.datasets import supported_verifications, verify
from pdmdata.io import (
    bulk_downloads,
    download,
    download_command,
    supported_downloads,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdmdata",
        description="Inspect, download, and verify predictive-maintenance datasets.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary_parser = subparsers.add_parser(
        "summary",
        help="display the dataset catalog summary table",
    )
    summary_parser.add_argument(
        "--downloadable",
        action="store_true",
        help="show only datasets with automated download support",
    )

    download_parser = subparsers.add_parser(
        "download",
        help="download one dataset or all bulk-enabled datasets",
    )
    download_parser.add_argument(
        "dataset",
        nargs="?",
        choices=supported_downloads(),
        help="dataset ID to download (omit when using --all)",
    )
    download_parser.add_argument(
        "--all",
        action="store_true",
        dest="download_all",
        help="download all bulk-enabled datasets",
    )
    download_parser.add_argument("--output-dir", type=Path)
    download_parser.add_argument("--no-extract", action="store_true")
    download_parser.add_argument("--overwrite", action="store_true")
    download_parser.add_argument(
        "--variant",
        help="download variant, such as 2025-q1 for Backblaze",
    )
    download_parser.add_argument(
        "--print-command",
        action="store_true",
        help="print the external download command without executing it",
    )

    verify_parser = subparsers.add_parser(
        "verify",
        help="run offline source verification for a downloaded dataset",
    )
    verify_parser.add_argument(
        "dataset",
        choices=supported_verifications(),
        help="dataset ID to verify",
    )
    verify_parser.add_argument(
        "--variant",
        help="dataset variant to verify (for example, Backblaze quarter)",
    )
    verify_parser.add_argument(
        "--check-archive",
        action="store_true",
        help="check extracted file CRCs against the local archive (Backblaze)",
    )
    verify_parser.add_argument(
        "--installed-only",
        action="store_true",
        help="validate only locally installed recordings (CWRU)",
    )
    verify_parser.add_argument(
        "--recording",
        type=int,
        nargs="+",
        help="specific recording ID(s) to validate (CWRU)",
    )

    return parser


def _print_download_result(result: dict[str, Any]) -> None:
    print(f"Archive: {result['archive']}")
    print(f"SHA-256: {result['sha256']}")
    if result["extracted"] is not None:
        print(f"Extracted: {result['extracted']}")


def _run_download(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> int:
    if args.download_all and args.dataset is not None:
        parser.error("cannot specify both a dataset ID and --all")
    if not args.download_all and args.dataset is None:
        parser.error("specify a dataset ID or pass --all")
    if args.download_all:
        if args.variant is not None or args.print_command:
            parser.error("--all cannot be combined with --variant or --print-command")
        dataset_ids = bulk_downloads()
        print("Datasets: {}".format(", ".join(dataset_ids)))
        for index, dataset_id in enumerate(dataset_ids, start=1):
            print(f"\n[{index}/{len(dataset_ids)}] Downloading {dataset_id}")
            result = download(
                dataset_id,
                args.output_dir,
                extract=not args.no_extract,
                overwrite=args.overwrite,
            )
            _print_download_result(result)
        return 0

    if args.print_command:
        print(
            download_command(
                args.dataset, args.output_dir, variant=args.variant
            )
        )
        return 0

    result = download(
        args.dataset,
        args.output_dir,
        extract=not args.no_extract,
        overwrite=args.overwrite,
        variant=args.variant,
    )
    _print_download_result(result)
    return 0


def _run_verify(args: argparse.Namespace) -> int:
    options: dict[str, Any] = {}
    if args.variant is not None:
        options["variant"] = args.variant
    if args.check_archive:
        options["check_archive"] = True
    if args.installed_only:
        options["installed_only"] = True
    if args.recording is not None:
        options["recording"] = (
            args.recording[0] if len(args.recording) == 1 else args.recording
        )
    report = verify(args.dataset, **options)
    if isinstance(report, pl.DataFrame):
        print(report)
    else:
        print(json.dumps(report, indent=2, default=str))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the ``pdmdata`` command-line interface."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "summary":
        summary(downloadable=True if args.downloadable else None)
        return 0
    if args.command == "download":
        return _run_download(parser, args)
    if args.command == "verify":
        return _run_verify(args)
    parser.error(f"Unknown command: {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
