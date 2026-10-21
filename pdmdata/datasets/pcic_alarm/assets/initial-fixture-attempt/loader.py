"""Read original PCIC archive members without extracting or constructing tasks."""
from collections.abc import Sequence
import csv
import hashlib
from io import BytesIO, StringIO
from pathlib import PurePosixPath
import tarfile

import numpy as np
import polars as pl
from pdmdata.catalog import load_metadata
from pdmdata.io import find_raw_file

MAX_MEMBER_BYTES = 128 * 1024 * 1024

def _archive(variant):
    resources = load_metadata("pcic_alarm")["download"]["variants"]
    if not isinstance(variant, str) or variant not in resources:
        raise ValueError(f"variant must be one of: {', '.join(sorted(resources))}")
    return find_raw_file("pcic_alarm", resources[variant]["filename"])

def _validate_member(member):
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts or "\\" in member.name:
        raise ValueError("Unsafe archive member name")
    if member.issym() or member.islnk():
        raise ValueError("Archive links are not supported")
    if member.size < 0 or member.size > MAX_MEMBER_BYTES:
        raise ValueError("Archive member exceeds the supported size")

def _read_member(variant, basename):
    with tarfile.open(_archive(variant), "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            _validate_member(member)
        matches = [m for m in members if m.isfile() and PurePosixPath(m.name).name == basename]
        if len(matches) != 1:
            raise ValueError(f"Expected one {basename} member, found {len(matches)}")
        member = matches[0]
        with archive.extractfile(member) as stream:
            raw = stream.read(MAX_MEMBER_BYTES + 1)
        if len(raw) != member.size:
            raise ValueError("Archive member size mismatch")
    return member.name, raw

def source_inventory(*, variant="wireless"):
    """Inspect member names, byte hashes and CSV headers; never infer time units."""
    rows = []
    with tarfile.open(_archive(variant), "r:gz") as archive:
        for member in archive.getmembers():
            _validate_member(member)
            item = dict(member=member.name, bytes=member.size, is_file=member.isfile())
            if member.isfile():
                with archive.extractfile(member) as stream:
                    raw = stream.read(MAX_MEMBER_BYTES + 1)
                if len(raw) != member.size:
                    raise ValueError("Archive member size mismatch")
                item["sha256"] = hashlib.sha256(raw).hexdigest()
                if member.name.lower().endswith(".csv"):
                    item["header"] = next(csv.reader(StringIO(raw.decode("utf-8-sig"))), [])
            rows.append(item)
    return rows

def load(*, variant="wireless", columns: Sequence[str] | None = None, lazy=True):
    """Read Alarm.csv as strings, retaining original rows, IDs and clock text.

    Blank fields are preserved as empty strings. No sorting, deduplication,
    graph loading, timezone inference or prediction-target construction occurs.
    The archive member is read in memory; lazy=True wraps that materialized table.
    """
    if type(lazy) is not bool:
        raise ValueError("lazy must be True or False")
    _, raw = _read_member(variant, "Alarm.csv")
    frame = pl.read_csv(BytesIO(raw), infer_schema=False, try_parse_dates=False,
                        missing_utf8_is_empty_string=True, row_index_name="source_row")
    if columns is not None:
        if (not isinstance(columns, Sequence) or isinstance(columns, (str, bytes))
                or not columns or any(not isinstance(v, str) or v not in frame.columns for v in columns)
                or len(set(columns)) != len(columns)):
            raise ValueError("columns must be a nonempty unique selection of source columns")
        frame = frame.select(list(columns))
    return frame.lazy() if lazy else frame

def read_reference(*, variant="wireless", member="DAG.npy"):
    """Read a separate numeric reference array without assigning edge direction.

    This explicit opt-in API is never called by the alarm loader. Object arrays,
    pickle data and non-matrix arrays are rejected. Array meaning and orientation
    require a task-specific audit before use as a structural target.
    """
    if not isinstance(member, str) or member not in {"DAG.npy", "Topology.npy"}:
        raise ValueError("member must be DAG.npy or Topology.npy")
    _, raw = _read_member(variant, member)
    value = np.load(BytesIO(raw), allow_pickle=False)
    if not isinstance(value, np.ndarray) or value.ndim != 2 or value.dtype.kind not in "biuf":
        raise ValueError("Reference arrays must be numeric matrices")
    return value
