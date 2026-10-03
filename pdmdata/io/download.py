"""Download datasets from direct URLs or through the official Kaggle CLI."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen
from zipfile import ZipFile

from pdmdata.catalog import load_catalog, load_metadata
from pdmdata.config import get_settings


CHUNK_SIZE = 1024 * 1024


def supported_downloads() -> List[str]:
    """Return dataset IDs supported by an automated download method."""
    return [item["id"] for item in load_catalog(downloadable=True)]


def bulk_downloads() -> List[str]:
    """Return datasets included in an explicitly requested bulk download."""
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
    if resource.get("method") == "files":
        return " && ".join(
            "wget --continue --output-document={} {}".format(
                _shell_quote(str(directory / item["filename"])),
                _shell_quote(item["url"]),
            )
            for item in resource["files"]
        )
    if resource.get("method", "url") == "kaggle":
        command = "kaggle datasets download --dataset={} --path={}".format(
            _shell_quote(resource["dataset"]), _shell_quote(str(directory))
        )
        if resource.get("file"):
            command += " --file={}".format(_shell_quote(resource["file"]))
        return command
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

    Files are stored under the configured ``data_root/<dataset-id>``. Existing
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
    checksum = resource.get("checksum")
    if checksum:
        _verify_checksum(archive, checksum)
    archive_type = resource.get("archive", "zip")
    extracted = directory / "extracted"
    if extract:
        if archive_type == "zip":
            if overwrite and extracted.exists():
                shutil.rmtree(str(extracted))
            if not extracted.exists():
                _extract_archive(archive, extracted)
            _extract_nested_archives(extracted)
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
        "source": resource.get("files", resource.get("url", resource.get("dataset"))),
        "method": resource.get("method", "url"),
        "archive": archive.name,
        "sha256": digest,
        "source_checksum": checksum,
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
    root = Path(output_dir) if output_dir is not None else get_settings().data_root
    directory = root.expanduser().resolve() / dataset_id
    return directory / variant if variant else directory


def _download_file(url: str, target: Path) -> None:
    request = Request(url, headers={"User-Agent": "pdmdata/0 (+dataset downloader)"})
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
    if method == "files":
        # Bundle unmodified source files locally to reuse archive validation.
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            staging = Path(temporary)
            bundle = staging / "bundle.zip"
            with ZipFile(bundle, "w") as zipped:
                for item in resource["files"]:
                    name = item["filename"]
                    if Path(name).name != name or name in {".", "..", "bundle.zip"}:
                        raise ValueError("Invalid source filename: {!r}".format(name))
                    target = staging / name
                    _download_file(item["url"], target)
                    zipped.write(target, arcname=name)
            os.replace(bundle, archive)
        return
    if method == "url":
        _download_file(resource["url"], archive)
        return
    if method == "kaggle":
        _download_kaggle(
            resource["dataset"],
            directory,
            archive,
            overwrite=overwrite,
            file=resource.get("file"),
        )
        return
    raise ValueError("Unknown download method: {!r}".format(method))


def _download_kaggle(
    dataset: str,
    directory: Path,
    archive: Path,
    *,
    overwrite: bool,
    file: Optional[str] = None,
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
    if file:
        command.extend(["--file", file])
    if overwrite:
        command.append("--force")
    for attempt in range(2):
        try:
            result = subprocess.run(
                command, check=True, text=True, capture_output=True
            )
            _normalize_kaggle_download(directory, archive, file=file)
            if archive.is_file():
                break
            details = (result.stdout or "") + (result.stderr or "")
        except subprocess.CalledProcessError as error:
            details = (error.stdout or "") + (error.stderr or "")
        needs_auth = any(marker in details.lower() for marker in (
            "authentication required", "401", "unauthorized",
            "could not find kaggle.json", "kaggle auth login",
        ))
        if attempt == 0 and needs_auth and _interactive_kaggle_login():
            print(
                "Kaggle authentication is required. Complete the official "
                "login flow; the download will then retry.",
                flush=True,
            )
            try:
                subprocess.run(
                    [executable, "auth", "login", "--force"],
                    check=True,
                    timeout=300,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                raise RuntimeError(
                    "Kaggle login did not complete. Run `kaggle auth login` "
                    "in a terminal and retry the download."
                ) from error
            continue
        if needs_auth:
            raise RuntimeError(
                "Kaggle authentication is required. Run `kaggle auth login` in a "
                "terminal, or configure KAGGLE_API_TOKEN for unattended "
                "execution, then retry. Automatic login is attempted at most once."
            )
        raise RuntimeError(
            f"Kaggle CLI could not download {dataset!r}. Check connectivity, "
            "dataset access, and any required terms on Kaggle. "
            "No authentication retry was triggered."
        )
    if not archive.is_file():
        raise RuntimeError(
            "Kaggle CLI completed but the expected archive was not created: "
            "{}".format(archive)
        )


def _normalize_kaggle_download(
    directory: Path,
    archive: Path,
    *,
    file: Optional[str],
) -> None:
    """Move a single-file Kaggle download onto the expected archive path."""
    if archive.is_file():
        return
    candidates = []
    if file:
        candidates.append(directory / Path(file).name)
        candidates.append(directory / file)
    for candidate in candidates:
        if candidate.is_file() and candidate != archive:
            candidate.replace(archive)
            return


def _interactive_kaggle_login() -> bool:
    """Allow terminal login and browser callbacks from local desktop notebooks."""
    if os.environ.get("CI", "").lower() not in {"", "0", "false"}:
        return False
    terminal = sys.stdin is not None and sys.stdin.isatty()
    desktop_notebook = "ipykernel" in sys.modules and (
        sys.platform in {"darwin", "win32"} or bool(os.environ.get("DISPLAY"))
    )
    return terminal or desktop_notebook


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checksum(path: Path, expected: str) -> None:
    """Compare the archive against the checksum published by its provider."""
    algorithm, value = expected.split(":", 1)
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    if digest.hexdigest() != value.lower():
        raise ValueError(
            f"Source checksum mismatch for {path}. "
            "Remove the corrupt archive or retry with overwrite=True."
        )


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


def _extract_archive(archive: Path, destination: Path) -> None:
    """Publish an extracted directory only after extraction succeeds."""
    with tempfile.TemporaryDirectory(prefix=".pdmdata-extract-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "contents"
        staging.mkdir()
        if archive.suffix.lower() in {".7z", ".rar"}:
            _extract_native_archive(archive, staging)
        else:
            _extract_zip(archive, staging)
        os.replace(staging, destination)


def _extract_native_archive(archive: Path, destination: Path) -> None:
    """Stream 7z/RAR members through libarchive without changing process cwd."""
    try:
        import libarchive
    except (ImportError, OSError, AttributeError) as error:
        raise RuntimeError(
            "IMS extraction requires libarchive. Install libarchive-tools on "
            "Debian/Ubuntu, libarchive with Homebrew on macOS, or a libarchive "
            "DLL on Windows. Set LIBARCHIVE to its library path if needed."
        ) from error
    destination = destination.resolve()
    with libarchive.file_reader(str(archive)) as packed:
        for member in packed:
            target = (destination / member.pathname).resolve()
            if (target != destination and destination not in target.parents
                    or not (member.isdir or member.isfile) or member.linkpath):
                raise ValueError(f"Unsafe archive member in {archive}: {member.pathname}")
            if member.isdir:
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("wb") as stream:
                    for block in member.get_blocks():
                        stream.write(block)


def _extract_nested_archives(directory: Path) -> None:
    """Extract nested ZIP, 7z, and RAR archives, including the NASA IMS bundle."""
    processed = set()
    while True:
        archives = [
            path
            for path in directory.rglob("*")
            if path.suffix.lower() in {".zip", ".7z", ".rar"}
            and path.is_file() and path.resolve() not in processed
            and not any(part.startswith(".pdmdata-extract-") for part in path.relative_to(directory).parts)
        ]
        if not archives:
            return
        for archive in archives:
            processed.add(archive.resolve())
            destination = archive.with_suffix("")
            if not destination.exists():
                _extract_archive(archive, destination)


def _validate_expected_files(directory: Path, expected: List[str]) -> None:
    files = [path for path in directory.rglob("*") if path.is_file()]
    names = {path.name for path in files}
    relative_paths = {path.relative_to(directory).as_posix() for path in files}
    missing = [
        name for name in expected
        if name not in (relative_paths if "/" in name else names)
    ]
    if missing:
        raise ValueError(
            "Downloaded archive is missing expected files: {}".format(
                ", ".join(missing)
            )
        )


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"
