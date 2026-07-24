#!/usr/bin/env python3
"""Download and extract every dataset with an automated retrieval method."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import download, supported_downloads  # noqa: E402


def main() -> None:
    dataset_ids = supported_downloads()
    print("Datasets: {}".format(", ".join(dataset_ids)))
    for index, dataset_id in enumerate(dataset_ids, start=1):
        print("\n[{}/{}] Downloading {}".format(index, len(dataset_ids), dataset_id))
        result = download(dataset_id)
        print("Archive: {}".format(result["archive"]))
        print("SHA-256: {}".format(result["sha256"]))
        print("Extracted: {}".format(result["extracted"]))


if __name__ == "__main__":
    main()
