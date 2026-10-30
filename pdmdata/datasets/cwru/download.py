"""Opt-in, bounded transfer of selected official CWRU source files."""

import hashlib
import json
from urllib.request import Request, urlopen

from scipy.io import loadmat
from scipy.io.matlab import MatReadError

from pdmdata.config import get_settings

from .loader import _path, inventory


def _download_file(url, target, *, max_bytes):
    """Publish a complete bounded HTTP response without altering source bytes."""
    request = Request(url, headers={"User-Agent": "pdmdata/0 (+dataset downloader)"})
    temporary = target.with_suffix(target.suffix + ".part")
    try:
        with urlopen(request, timeout=60) as response, temporary.open("wb") as stream:
            length = response.headers.get("Content-Length")
            expected = int(length) if length is not None else None
            if expected is not None and (expected < 0 or expected > max_bytes):
                raise ValueError("Download Content-Length exceeds the byte cap")
            written = 0
            while chunk := response.read(min(1024 * 1024, max_bytes - written + 1)):
                if written + len(chunk) > max_bytes:
                    raise ValueError("Download body exceeds the byte cap")
                stream.write(chunk)
                written += len(chunk)
            if expected is not None and written != expected:
                raise OSError("Truncated HTTP response")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def download(
    *,
    recording=None,
    collection=None,
    load_hp=None,
    fault=None,
    diameter_in=None,
    outer_race_position=None,
    overwrite: bool = False,
    max_bytes: int = 1024 * 1024 * 1024,
) -> dict:
    """Download a selected set to the configured data root, with a total byte cap.

    At least one selector is required. Existing identical local files are
    reused; their size counts toward the cap. A failed transfer publishes no
    partial file, while completed files remain reusable on retry. Hashes are
    locally computed provenance, not provider-authenticated checksums.
    """
    selectors = dict(
        recording=recording,
        collection=collection,
        load_hp=load_hp,
        fault=fault,
        diameter_in=diameter_in,
        outer_race_position=outer_race_position,
    )
    if all(value is None for value in selectors.values()):
        raise ValueError(
            "Select at least one recording, collection, or condition"
        )
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    if type(overwrite) is not bool:
        raise ValueError("overwrite must be True or False")
    selected = inventory(**selectors)
    if selected.is_empty():
        raise ValueError("No CWRU recordings match the selection")
    root = get_settings().data_root / "cwru"
    destination = root / "recordings"
    destination.mkdir(parents=True, exist_ok=True)
    used = 0
    records = []
    for info in selected.iter_rows(named=True):
        try:
            path = _path(info)
        except FileNotFoundError:
            path = destination / info["filename"]
        exists = path.is_file()
        remaining = max_bytes - used
        if remaining <= 0 or (exists and path.stat().st_size > remaining):
            raise ValueError("Selected source files exceed max_bytes")
        if not exists or overwrite:
            # Validate parsing before replacing a published source. Some
            # responses can end early without an HTTP-level download error.
            staged = path.with_suffix(".mat.pending")
            try:
                for attempt in range(3):
                    try:
                        _download_file(info["url"], staged, max_bytes=remaining)
                    except OSError:
                        if attempt == 2:
                            raise
                        continue
                    try:
                        loadmat(staged)
                    except (
                        OSError,
                        ValueError,
                        IndexError,
                        EOFError,
                        MatReadError,
                    ) as error:
                        if attempt == 2:
                            raise ValueError(
                                f"Invalid or truncated MATLAB source {info['filename']}"
                            ) from error
                    else:
                        staged.replace(path)
                        break
            finally:
                staged.unlink(missing_ok=True)
        size = path.stat().st_size
        used += size
        records.append(
            {
                "recording": info["recording"],
                "url": info["url"],
                "path": path.relative_to(root).as_posix(),
                "bytes": size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = destination / "manifest.json"
    previous = json.loads(manifest.read_text()) if manifest.is_file() else {}
    all_records = {
        item["recording"]: item for item in previous.get("files", [])
    }
    all_records.update({item["recording"]: item for item in records})
    content = {
        "dataset": "cwru",
        "provider_checksum_available": False,
        "hash_origin": "Locally computed SHA-256",
        "files": [all_records[key] for key in sorted(all_records)],
    }
    temporary = manifest.with_suffix(".json.part")
    temporary.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest)
    return {
        "directory": root,
        "manifest": manifest,
        "files": records,
        "total_bytes": used,
    }
