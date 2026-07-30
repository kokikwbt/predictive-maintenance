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


def bulk_downloads() -> List[str]:
    """Return automated downloads that are safe to include in bootstrap."""
    return [
        item["id"]
        for item in load_catalog(downloadable=True)
        if item["download"].get("bulk", True)
    ]


def download_variants(dataset_id: str) -> List[str]:
    """Return selectable download variants for a dataset."""
    metadata = load_metadata(dataset_id)
    return sorted(metadata.get("download", {}).get("variants", {}))


def download_command(
    dataset_id: str,
    output_dir: Optional[Path] = None,
    *,
    variant: Optional[str] = None,
) -> str:
    """Return the external command used to download a dataset."""
    metadata = load_metadata(dataset_id)
    resource, selected_variant = _resource(metadata, variant)
    directory = _dataset_directory(dataset_id, output_dir, selected_variant)
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
    variant: Optional[str] = None,
) -> Dict[str, Any]:
    """Download, hash, optionally extract, and validate one dataset.

    Files are stored under ``data/raw/<dataset-id>`` by default. Existing
    archives are reused unless ``overwrite`` is true.
    """
    metadata = load_metadata(dataset_id)
    resource, selected_variant = _resource(metadata, variant)
    directory = _dataset_directory(dataset_id, output_dir, selected_variant)
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / resource["filename"]

    if overwrite or not archive.exists():
        _download_resource(resource, directory, archive, overwrite=overwrite)

    digest = _sha256(archive)
    archive_type = resource.get("archive", "zip")
    extracted = directory / "extracted"
    if extract:
        if archive_type == "zip":
            if overwrite and extracted.exists():
                shutil.rmtree(str(extracted))
            if not extracted.exists():
                extracted.mkdir(parents=True)
                _extract_zip(archive, extracted)
            _extract_nested_zips(extracted)
            validation_root = extracted
        elif archive_type == "file":
            validation_root = directory
            extracted = directory
        else:
            raise ValueError("Unknown archive type: {!r}".format(archive_type))
        _validate_expected_files(
            validation_root, resource.get("expected_files", [])
        )

    manifest = {
        "dataset": dataset_id,
        "source": resource.get("url", resource.get("dataset")),
        "method": resource.get("method", "url"),
        "archive": archive.name,
        "sha256": digest,
        "extracted": extract and archive_type == "zip",
        "variant": selected_variant,
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "archive": archive,
        "directory": directory,
        "extracted": extracted if extract and archive_type == "zip" else None,
        "manifest": manifest_path,
        "sha256": digest,
    }


def _resource(
    metadata: Dict[str, Any], variant: Optional[str] = None
) -> tuple[Dict[str, Any], Optional[str]]:
    try:
        download = metadata["download"]
    except KeyError:
        raise ValueError(
            "Automated download is not supported for {!r}. Supported datasets: {}".format(
                metadata["id"], ", ".join(supported_downloads())
            )
        )
    variants = download.get("variants")
    if not variants:
        if variant is not None:
            raise ValueError(
                "{!r} does not provide download variants".format(metadata["id"])
            )
        return download, None
    selected = variant or download.get("default_variant")
    if selected not in variants:
        raise ValueError(
            "variant must be one of: {}".format(", ".join(sorted(variants)))
        )
    common = {
        key: value
        for key, value in download.items()
        if key not in {"variants", "default_variant", "bulk"}
    }
    common.update(variants[selected])
    return common, selected


def _dataset_directory(
    dataset_id: str,
    output_dir: Optional[Path],
    variant: Optional[str] = None,
) -> Path:
    root = Path(output_dir) if output_dir is not None else DEFAULT_DATA_ROOT
    directory = root.expanduser().resolve() / dataset_id
    return directory / variant if variant else directory


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
