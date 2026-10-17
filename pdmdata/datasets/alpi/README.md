# ALPI alarm events

<!-- BEGIN GENERATED METADATA -->

## Dataset overview

Timestamped alarm events from 20 industrial packaging machines. The source provides alarm codes and machine identifiers, not verified physical-failure or repair labels.

| Item | Details |
|---|---|
| ID | `alpi` |
| Name | Alarm Logs in Packaging Industry |
| Provider | [ALPI authors on Mendeley Data](https://data.mendeley.com/datasets/4nhx2x67cd/1) |
| DOI | [10.17632/4nhx2x67cd.1](https://doi.org/10.17632/4nhx2x67cd.1) |
| Availability | available (checked: 2026-10-18) |
| Access | Direct Mendeley file download; CLI transfer and provider SHA-256 verified |
| License | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Commercial use | Yes |
| Redistribution | Yes |
| Data type | Industrial alarm event sequences |
| Feature dimensions | 1 categorical event type; timestamp and machine ID are identifiers, not sensor channels |
| Feature counting | The raw CSV has timestamp, alarm, and serial. The loader adds source_row for original-row provenance. There are no analogue sensor waveforms. |
| Feature-count source | [Provider readme and raw CSV (version 1)](https://data.mendeley.com/public-files/datasets/4nhx2x67cd/files/1c780d52-c22c-42f9-9880-89b97fa0b9cc/file_downloaded) |
| Tasks | Next-alarm prediction, Future-window alarm prediction, Event-sequence forecasting, Anomaly detection research |

## Experiment-task suitability

| Task | Support |
|---|---|
| Event or sequence forecasting | Direct |
| Time-to-event prediction | Requires target derivation |
| Anomaly detection | Requires target derivation |

## Attributes

| Attribute or group | Type | Role | Description |
|---|---|---|---|
| `timestamp` | Datetime(us), timezone unspecified | timestamp | Original alarm occurrence time. No timezone is inferred. The optional time filter is start-inclusive and end-exclusive. |
| `alarm` | Int64 | event-type | Original alarm code, 0 to 153 in the verified version. Raw ID 0 is a genuine code and is not removed as padding. |
| `serial` | Int64 | entity | Original packaging-machine identifier, 0 to 19 in the verified version. |
| `source_row` | UInt32 | provenance | Zero-based row in the original CSV, excluding the header; retained across filtering. Source order and duplicate rows are preserved. |

## Usage notes

- Version 1 was retrieved through curl without authentication or browser cookies, and the archive matched the provider SHA-256 on 2026-10-18.
- Full raw CSV check: 444,834 records, 20 machines, 154 alarm codes; first timestamp 2019-02-21 10:16:18.130, last timestamp 2020-06-17 03:53:51.885.
- The verified source contains 83 duplicate timestamp/alarm/serial rows. The loader does not deduplicate or globally sort events.
- The 8,517,314-byte ZIP expands to about 163 MB of raw and processed files. Download is opt-in; processed JSON/NPZ/pickle files are not loaded or executed by this adapter.
- Prediction windows, entity/time splits, future labels, and end-of-record handling belong to task construction. The loader does not manufacture failure, RUL, or censoring labels.

## Download

```bash
uv run --locked python scripts/download.py alpi
```

Source archive: [alarms_log_data.zip](https://data.mendeley.com/public-files/datasets/4nhx2x67cd/files/1c053e9f-e483-4ab5-925b-00d6237f24ee/file_downloaded)

## Suggested citation

Dalle Pezze, D., Tosato, D., Masiero, C., Susto, G. A., and Beghi, A. (2021). ALARM LOGS IN PACKAGING INDUSTRY (ALPI), version 1. Mendeley Data. https://doi.org/10.17632/4nhx2x67cd.1

<!-- END GENERATED METADATA -->


## Loading raw events

```python
import pdmdata
from pdmdata.datasets.alpi import inventory, validate

# Load local data lazily. Download separately with the CLI.
events = pdmdata.load("alpi")
machine = pdmdata.load("alpi", serial=4, lazy=False)
period = pdmdata.load(
    "alpi", serial=[0, 4],
    start="2019-03-01T00:00:00", end="2019-04-01T00:00:00",
    columns=["source_row", "timestamp", "alarm", "serial"],
).collect()
print(inventory())
print(validate())
```

`serial` and `alarm` accept a nonnegative integer or a nonempty sequence of
integers. An ID absent from the file produces an empty selection. `columns`
selects output fields. The time interval is `[start, end)`; timestamps have no
provider-specified timezone, so timezone-aware filter values are rejected.

The source's row order is retained, including ties and duplicates. Use
`source_row` when a downstream operation requires a stable tie-breaker. Sort
explicitly if a task requires a global chronological stream. Neither a record's
position nor an alarm code is a health-stage label.

## Validation and task scope

The [real-file validation record](assets/validation.json) links the raw CSV and
archive hashes to a standard-library CSV comparison. All original rows are
preserved, including alarm ID 0 and 83 duplicate rows. `validate()` reports
these conditions; it does not deduplicate or infer failures.

The provider ZIP also includes prepared windows in JSON, NPZ, and pickle formats.
This adapter loads only `raw/alarms.csv`. It does not execute the downloaded
`dataset.py` or deserialize prepared pickle files. Original provider
[readme](https://data.mendeley.com/public-files/datasets/4nhx2x67cd/files/1c780d52-c22c-42f9-9880-89b97fa0b9cc/file_downloaded)
and [code](https://data.mendeley.com/public-files/datasets/4nhx2x67cd/files/cc11aad8-7457-4937-97c0-21c0c1c3c409/file_downloaded)
remain available from the same versioned source.

Next-event and future-window labels can be constructed from the recorded alarms.
Prediction splits, overlapping windows, simultaneous-event semantics, and any
right-censoring assumption require an explicit task design. A machine's last
alarm is not a known observation-end or failure time. This release does not
provide verified component failures, repair interventions, RUL, or root causes.

[Dataset notebook](../../../notebooks/datasets/alpi.ipynb) demonstrates local
loading and machine-level inspection.
