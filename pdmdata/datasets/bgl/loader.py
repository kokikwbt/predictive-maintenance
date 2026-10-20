"""Literal BGL log fields and raw-line provenance; no prediction targets."""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from itertools import islice
from pathlib import Path
import zipfile

import polars as pl

from pdmdata.config import get_settings

FIELDS = ['label', 'epoch', 'date', 'node', 'time', 'node_repeat', 'type', 'component', 'level', 'content']
SCHEMA = {'source_row': pl.UInt64, 'source_file': pl.String, 'raw_line': pl.String,
          'parse_ok': pl.Boolean, **{name: pl.String for name in FIELDS}}


def _raw_lines() -> Iterator[tuple[str, bytes]]:
    root = get_settings().data_root / 'bgl'
    archive = root / 'BGL.zip'
    plain = root / 'BGL.log'
    if not plain.is_file():
        plain = root / 'extracted' / 'BGL.log'
    if archive.is_file():
        with zipfile.ZipFile(archive) as handle:
            if 'BGL.log' not in handle.namelist():
                raise ValueError('BGL.zip does not contain the expected BGL.log member')
            with handle.open('BGL.log') as stream:
                for line in stream:
                    yield 'BGL.zip!BGL.log', line
    elif plain.is_file():
        with plain.open('rb') as stream:
            for line in stream:
                yield str(plain.relative_to(root)), line
    else:
        raise FileNotFoundError('No installed BGL.zip or BGL.log under ' + str(root))


def iter_batches(*, batch_rows: int = 10000, rows: int | None = None) -> Iterator[pl.DataFrame]:
    """Stream source-order rows, retaining duplicate/malformed lines verbatim.

    Raw lines include their original line ending. Fields are literal strings
    from splitting at most nine whitespace boundaries; an absent message is an empty string. No clock interpretation,
    sorting, deduplication, alert filtering, or template inference is applied.
    A malformed header is retained with parse_ok=False and null parsed fields.
    Invalid UTF-8 raises an error instead of replacing source bytes silently.
    """
    if type(batch_rows) is not int or batch_rows <= 0:
        raise ValueError('batch_rows must be a positive integer')
    if rows is not None and (type(rows) is not int or rows < 0):
        raise ValueError('rows must be a nonnegative integer or None')
    batch = []
    for index, (source, payload) in enumerate(islice(_raw_lines(), rows)):
        line = payload.decode('utf-8', errors='strict')
        fields = line.rstrip('\r\n').split(maxsplit=9)
        if len(fields) == 9:
            fields.append('')  # A complete header may have an empty message.
        parsed = len(fields) == 10
        batch.append((index, source, line, parsed, *(fields if parsed else [None] * 10)))
        if len(batch) == batch_rows:
            yield pl.DataFrame(batch, schema=SCHEMA, orient='row')
            batch = []
    if batch:
        yield pl.DataFrame(batch, schema=SCHEMA, orient='row')


def load(*, rows: int | None = None, columns: Sequence[str] | None = None) -> pl.DataFrame:
    """Materialize literal source rows; use iter_batches for a bounded-memory scan."""
    if columns is not None and (isinstance(columns, (str, bytes)) or not isinstance(columns, Sequence)
                               or not columns or any(not isinstance(v, str) for v in columns)
                               or len(set(columns)) != len(columns)):
        raise ValueError('columns must be a nonempty sequence of distinct column names')
    if columns is not None and set(columns) - set(SCHEMA):
        raise ValueError('Unknown BGL columns: ' + ', '.join(sorted(set(columns) - set(SCHEMA))))
    frames = list(iter_batches(rows=rows))
    result = pl.concat(frames) if frames else pl.DataFrame(schema=SCHEMA)
    return result.select(list(columns)) if columns is not None else result
