"""Offline completeness, daily schema, and optional archive integrity checks."""

from datetime import date, timedelta
from zipfile import ZipFile
import zlib

from ..catalog import load_metadata
from ..config import get_settings
from .loader import _files, _scan, _variant


def verify(variant: str | None = None, *, check_archive: bool = False) -> dict:
    """Read each daily CSV and require complete calendar-quarter coverage.

    Archive verification compares extracted sizes and CRCs with local ZIP members;
    it does not establish authenticity against a provider-published checksum.
    """
    variant = _variant(variant)
    files = _files(variant)
    year, quarter = int(variant[:4]), int(variant[-1])
    start = date(year, (quarter - 1) * 3 + 1, 1)
    end = date(year + 1, 1, 1) if quarter == 4 else date(year, quarter * 3 + 1, 1)
    expected = {start + timedelta(days=i) for i in range((end - start).days)}
    actual = {date.fromisoformat(p.stem) for p in files}
    if actual != expected:
        raise ValueError(f"Quarter coverage mismatch: missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}")
    required = {"date", "serial_number", "model", "capacity_bytes", "failure"}
    rows = 0
    for path in files:
        frame = _scan(path).collect()
        if not required.issubset(frame.columns) or frame.is_empty():
            raise ValueError(f"Missing required columns or empty snapshot: {path}")
        if any(frame[c].null_count() for c in required):
            raise ValueError(f"Null required values: {path}")
        if set(frame["date"].to_list()) != {date.fromisoformat(path.stem)}:
            raise ValueError(f"Date does not match filename: {path}")
        if frame["serial_number"].n_unique() != frame.height:
            raise ValueError(f"Duplicate serial numbers: {path}")
        if not set(frame["failure"].to_list()).issubset({0, 1}):
            raise ValueError(f"Invalid failure flags: {path}")
        rows += frame.height
    if check_archive:
        directory = get_settings().data_root / "backblaze" / variant
        resource = load_metadata("backblaze")["download"]["variants"][variant]
        with ZipFile(directory / resource["filename"]) as archive:
            for path in files:
                member = archive.getinfo(path.relative_to(directory / "extracted").as_posix())
                crc = 0
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        crc = zlib.crc32(chunk, crc)
                if path.stat().st_size != member.file_size or crc != member.CRC:
                    raise ValueError(f"Archive integrity mismatch: {path}")
    return {"variant": variant, "files": len(files), "rows": rows,
            "archive_checked": check_archive, "complete": True}
