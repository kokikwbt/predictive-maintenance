# BGL: literal BlueGene/L alert logs

This adapter reads the full BGL archive distributed by Loghub, retaining source
order, duplicate records, original line endings, and malformed rows. It exposes
literal header strings; it does not assign time zones, infer event templates,
turn alert categories into confirmed physical failures, or define a forecasting
benchmark. The small 2,000-line parsing sample is not used as a continuous trace.

<!-- BEGIN GENERATED METADATA -->

## Dataset overview

Literal alert and non-alert messages from a real BlueGene/L supercomputer. The adapter preserves raw lines and source order; failure targets, time interpretation, templates and forecasting splits are not inferred.

| Item | Details |
|---|---|
| ID | `bgl` |
| Name | BlueGene/L alert logs |
| Provider | [Loghub v8 and USENIX HPC4](https://zenodo.org/records/8196385) |
| DOI | [10.5281/zenodo.8196385](https://doi.org/10.5281/zenodo.8196385) |
| Availability | available (checked: 2026-10-21) |
| Access | Public fixed archive; pdmdata CLI-helper transfer, provider MD5 and ZIP CRC verified |
| License | [Loghub research/academic use conditions](https://github.com/logpai/loghub/blob/dd61d0952749ee7963bde24220d1be5ede023033/LICENSE) |
| Commercial use | No |
| Redistribution | Yes |
| Data type | Real computer-system alert event logs |
| Feature dimensions | Raw text, nine literal header fields and message content; no analogue sensor channels |
| Feature counting | Source rows are not templates or confirmed failures. Clock precision, node field semantics and event-type construction require a separate audit. |
| Feature-count source | [Loghub BGL source description](https://github.com/logpai/loghub/blob/dd61d0952749ee7963bde24220d1be5ede023033/BGL/README.md) |
| Tasks | Log parsing research, Alert detection research, Event-sequence analysis |

## Experiment-task suitability

| Task | Support |
|---|---|
| Anomaly detection | Requires target derivation |
| Event or sequence forecasting | Requires target derivation |

## Attributes

| Attribute or group | Type | Role | Description |
|---|---|---|---|
| `raw_line` | String | source | Original decoded UTF-8 line, including its line ending; retained even if the header cannot be parsed. |
| `source_row` | UInt64 | provenance | Zero-based source line index, preserved across streaming batch boundaries. |
| `source_file` | String | provenance | BGL.zip!BGL.log, BGL.log, or extracted/BGL.log, retaining the installed source path. |
| `parse_ok` | Boolean | validation | Whether the nine header fields and an optional message could be split. An empty message is retained as an empty string. Malformed rows remain present. |
| `label / epoch / date / node / time / node_repeat / type / component / level / content` | String | raw-fields | Literal source tokens, without timestamp, timezone, entity, anomaly or physical-failure coercion. |

## Usage notes

- The archive was fully downloaded using the pdmdata HTTP runtime, matched the provider MD5, and passed ZIP CRC verification. No performance result or forecasting-task adoption follows from this transfer.
- The full archive is used for sequential audits; the 2,000-line parsing sample is not treated as a contiguous history.
- Loading preserves source order, duplicates, malformed rows and original line endings. Invalid UTF-8 raises an error. Neither jitter nor template inference is performed.
- The source use conditions require the Loghub URL, the relevant paper citation, and inclusion of its notice in copies. The license notice is stored in assets/SOURCE_LICENSE.
- The native text and epoch clock relationship and the meaning of the two node fields are not assumed by the loader. An alert tag is not a verified physical failure.
- A full-source audit found 34,470 complete nine-field headers with empty messages. These are parsed without dropping the row or its clock. The initial parser required a nonempty message; its source and audit were retained before this correction.

## Download

```bash
uv run --locked python scripts/download.py bgl
```

Source archive: [BGL.zip](https://zenodo.org/records/8196385/files/BGL.zip?download=1)

## Suggested citation

Oliner, A. J., and Stearley, J. (2007). What Supercomputers Say: A Study of Five System Logs. DSN. Zhu, J., He, S., He, P., Liu, J., and Lyu, M. R. (2023). Loghub: A Large Collection of System Log Datasets for AI-driven Log Analytics. ISSRE.

<!-- END GENERATED METADATA -->

```python
from pdmdata import load
from pdmdata.datasets.bgl import iter_batches

sample = load('bgl', rows=100, columns=['source_row', 'epoch', 'time', 'component', 'level'])
for batch in iter_batches(batch_rows=10000):
    # Process every source row without materializing the entire log.
    print(batch.height)
```

Install the fixed v8 `BGL.zip` under the configured raw data root's `bgl/`
directory. A decompressed `BGL.log` is also supported at that directory or under its standard `extracted/` subdirectory. If both are present, the
verified archive takes precedence. Loading never downloads missing data.
`load` materializes its requested rows as a DataFrame; `iter_batches` is the
bounded-memory API for a full scan. `rows` is an inspection prefix cap, not a
prediction-window or cohort selection rule.

The ten parsed strings follow Loghub's header layout: label, epoch, date, node,
time, repeated node, type, component, level, and content. `raw_line` preserves
the entire original line, including line endings. `source_row` is zero-based.
A complete nine-field header with no message has an empty `content` string. Unparseable headers remain present with `parse_ok=False`. No rows are sorted,
silently repaired, or discarded. The relation between the epoch and text clock,
simultaneous records, and the meaning of the two node fields require a task-level
audit before time or entity forecasting. No such benchmark is implemented here.

Loghub permits research or academic use under its attribution and notice
conditions, recorded in `assets/SOURCE_LICENSE`. It is not labeled as an
unrestricted commercial license. Cite Loghub (ISSRE 2023) and Oliner and Stearley
(DSN 2007) when using this source. Source transfer verification is recorded in
`assets/source-verification.json`; raw data are not committed.
