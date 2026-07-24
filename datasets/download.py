"""Download datasets from direct URLs or through the official Kaggle CLI."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen
from zipfile import ZipFile

from .catalog import ROOT, load_catalog, load_metadata


PROJECT_ROOT = ROOT.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "raw"
CHUNK_SIZE = 1024 * 1024


def supported_downloads() -> List[str]:
    """Return dataset IDs supported by an automated download method."""
    return [item["id"] for item in load_catalog(downloadable=True)]


def download_command(dataset_id: str, output_dir: Optional[Path] = None) -> str:
    """Return the external command used to download a dataset."""
    metadata = load_metadata(dataset_id)
    resource = _resource(metadata)
    directory = _dataset_directory(dataset_id, output_dir)
    if resource.get("method", "url") == "kaggle":
        return "kaggle datasets download --dataset={} --path={}".format(
            _shell_quote(resource["dataset"]), _shell_quote(str(directory))
        )
    target = directory / resource["filename"]
    return "wget --continue --output-document={} {}".format(
        _shell_quote(str(target)), _shell_quote(resource["url"])
    )


def download(
    dataset_id: str,
    output_dir: Optional[Path] = None,
    *,
    extract: bool = True,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Download, hash, optionally extract, and validate one dataset.

    Files are stored under ``data/raw/<dataset-id>`` by default. Existing
    archives are reused unless ``overwrite`` is true.
    """
    metadata = load_metadata(dataset_id)
    resource = _resource(metadata)
    directory = _dataset_directory(dataset_id, output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / resource["filename"]

    if overwrite or not archive.exists():
        _download_resource(resource, directory, archive, overwrite=overwrite)

    digest = _sha256(archive)
    extracted = directory / "extracted"
    if extract:
        if overwrite and extracted.exists():
            shutil.rmtree(str(extracted))
        if not extracted.exists():
            extracted.mkdir(parents=True)
            _extract_zip(archive, extracted)
        _extract_nested_zips(extracted)
        _validate_expected_files(extracted, resource.get("expected_files", []))

    manifest = {
        "dataset": dataset_id,
        "source": resource.get("url", resource.get("dataset")),
        "method": resource.get("method", "url"),
        "archive": archive.name,
        "sha256": digest,
        "extracted": extract,
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "archive": archive,
        "directory": directory,
        "extracted": extracted if extract else None,
        "manifest": manifest_path,
        "sha256": digest,
    }


def _resource(metadata: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return metadata["download"]
    except KeyError:
        raise ValueError(
            "Automated download is not supported for {!r}. Supported datasets: {}".format(
                metadata["id"], ", ".join(supported_downloads())
            )
        )


def _dataset_directory(dataset_id: str, output_dir: Optional[Path]) -> Path:
    root = Path(output_dir) if output_dir is not None else DEFAULT_DATA_ROOT
    return root.expanduser().resolve() / dataset_id


def _download_file(url: str, target: Path) -> None:
    request = Request(url, headers={"User-Agent": "pmdata/0 (+dataset downloader)"})
    temporary = target.with_suffix(target.suffix + ".part")
    try:
        with urlopen(request) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream, length=CHUNK_SIZE)
        os.replace(str(temporary), str(target))
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _download_resource(
    resource: Dict[str, Any],
    directory: Path,
    archive: Path,
    *,
    overwrite: bool,
) -> None:
    method = resource.get("method", "url")
    if method == "url":
        _download_file(resource["url"], archive)
        return
    if method == "kaggle":
        _download_kaggle(
            resource["dataset"], directory, archive, overwrite=overwrite
        )
        return
    raise ValueError("Unknown download method: {!r}".format(method))


def _download_kaggle(
    dataset: str,
    directory: Path,
    archive: Path,
    *,
    overwrite: bool,
) -> None:
    executable = shutil.which("kaggle")
    if executable is None:
        raise RuntimeError(
            "The official Kaggle CLI is required. Install the project requirements "
            "and retry. Authentication, when required, is managed by the user with "
            "`kaggle auth login`."
        )
    command = [
        executable,
        "datasets",
        "download",
        "--dataset",
        dataset,
        "--path",
        str(directory),
    ]
    if overwrite:
        command.append("--force")
    try:
        subprocess.run(command, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as error:
        details = (error.stderr or error.stdout or "").strip()
        raise RuntimeError(
            "Kaggle CLI could not download {!r}. If authentication is required, "
            "run `kaggle auth login` and retry.{}".format(
                dataset, "\nKaggle CLI: {}".format(details) if details else ""
            )
        ) from error
    if not archive.is_file():
        raise RuntimeError(
            "Kaggle CLI completed but the expected archive was not created: {}".format(
                archive
            )
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_zip(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with ZipFile(str(archive)) as zipped:
        for member in zipped.infolist():
            target = (destination / member.filename).resolve()
            if destination != target and destination not in target.parents:
                raise ValueError(
                    "Unsafe archive member in {}: {}".format(archive, member.filename)
                )
        zipped.extractall(str(destination))


def _extract_nested_zips(directory: Path) -> None:
    """Extract ZIP files embedded in a downloaded archive."""
    processed = set()
    while True:
        archives = [
            path
            for path in directory.rglob("*.zip")
            if path.is_file() and path.resolve() not in processed
        ]
        if not archives:
            return
        for archive in archives:
            processed.add(archive.resolve())
            destination = archive.with_suffix("")
            if not destination.exists():
                destination.mkdir(parents=True)
                _extract_zip(archive, destination)


def _validate_expected_files(directory: Path, expected: List[str]) -> None:
    names = {path.name for path in directory.rglob("*") if path.is_file()}
    missing = [name for name in expected if name not in names]
    if missing:
        raise ValueError(
            "Downloaded archive is missing expected files: {}".format(
                ", ".join(missing)
            )
        )


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"
