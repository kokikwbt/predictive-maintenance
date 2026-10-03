# Wind Turbine SCADA Data for Early Fault Detection

<!-- BEGIN GENERATED METADATA -->

## Dataset overview

A real-world early-fault-detection benchmark containing 89 years of SCADA time series from 36 wind turbines across three wind farms.

| Item | Details |
|---|---|
| ID | `care` |
| Name | Wind Turbine SCADA Data for Early Fault Detection |
| Provider | [Fraunhofer IEE on Zenodo](https://zenodo.org/records/15846963) |
| DOI | [10.5281/zenodo.15846963](https://doi.org/10.5281/zenodo.15846963) |
| Availability | available (checked: 2026-09-05) |
| Access | Direct download from Zenodo |
| License | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| Commercial use | Yes |
| Redistribution | Yes |
| Data type | Large labeled wind-turbine SCADA time series |
| Feature dimensions | Wind Farm A: 81; B: 252; C: 952 sensor features |
| Feature counting | Counts all sensor columns in the v6 time-series files, including available statistical summaries. Excludes time_stamp, asset_id, id, train_test, and status_type_id. Total raw column counts are 86, 257, and 957 respectively; selecting only average channels reduces the dimensions. |
| Feature-count source | [Fraunhofer IEE on Zenodo](https://zenodo.org/records/15846963) |
| Tasks | Early fault detection, Anomaly detection, Failure prediction |

## Experiment-task suitability

| Task | Support |
|---|---|
| Anomaly detection | Direct |
| Fault or health-state classification | Direct |
| Condition estimation | Requires target derivation |
| Time-to-event prediction | Direct |
| Survival analysis | Requires target derivation |

## Attributes

| Attribute or group | Type | Role | Description |
|---|---|---|---|
| `time_stamp` | Datetime | timestamp | SCADA observation time within an anonymized turbine event series. |
| `sensor statistics` | Float64 | sensor | Per-timestamp average, minimum, maximum, and standard-deviation statistics; feature counts differ by wind farm. |
| `status_type_id` | Categorical | target | Turbine-status label used for training-data quality filtering and anomaly evaluation. |
| `event metadata` | Mixed | event | Fault descriptions, event timing, asset identifiers, and prediction intervals supplied with event series. |

## Usage notes

- Version 6 corrects labels, timestamps, units, and event descriptions from earlier versions.
- The collection has 95 datasets: 45 lead up to labeled turbine faults and 50 represent normal behavior.
- The provider recommends Avg signals for analysis; Min, Max, and Std can be implausible, especially in Wind Farm B.
- The 5.5 GB archive is opt-in and is not downloaded by scripts/bootstrap.sh.

## Download

```bash
uv run --locked python scripts/download.py care
```

Source archive: [CARE_To_Compare.zip](https://zenodo.org/records/15846963/files/CARE_To_Compare.zip?download=1)

## Suggested citation

Gück, C. and Roelofs, C. (2025). Wind Turbine SCADA Data For Early Fault Detection, Version 6. Zenodo. https://doi.org/10.5281/zenodo.15846963

<!-- END GENERATED METADATA -->


## Complete local dataset

`pdmdata.download("care")` downloads the complete Version 6 archive (5,503,439,673
bytes), verifies Zenodo's MD5 checksum, and checks all 103 extracted files.
The archive contains 95 time-series CSVs: 22 in Wind Farm A, 15 in B, and 58 in C.
It also contains two README files and an event table and feature description
for each farm. Extracted files occupy approximately 20 GB; allow approximately
26 GB for the archive and extracted data together.

The default location is `data/raw/care/`, controlled by the repository's
`pdmdata.toml`. All downloaded files are excluded from Git.

```python
import pdmdata

pdmdata.download("care")  # Explicit full download; existing archive is reused.
frame = pdmdata.load("care", wind_farm="A", event_id=0)
# CARE files use semicolons; the loader returns a Polars LazyFrame by default.
sample = frame.head(10).collect()

events = pdmdata.load("care", wind_farm="A", table="events", lazy=False)
features = pdmdata.load("care", wind_farm="A", table="features", lazy=False)
```

Use the supplied `train_test` split (`train` / `prediction`) when designing
experiments and consult the
farm's event table for event labels and intervals. Preserve the raw data;
feature definitions differ between farms. The event table uses `asset` in
Wind Farm A and `asset_id` in B/C; the time-series files use `asset_id`.
The website's Version 5 correction
changes event 51 to an anomaly (45 anomaly / 50 normal events); the bundled
README still mentions the older 44 / 51 counts. Use the current event tables.
For most Wind Farm B analyses the provider recommends Avg measurements because
Min, Max and Std statistics contain known inconsistencies.

To repeat the offline integrity and loading checks after downloading:

```bash
uv run --locked python scripts/verify_care.py
```

This reads all 95 recordings and verifies all 103 extracted files against the
archive's file sizes and CRCs, after checking the archive's published checksum.
It writes `inventory.csv` (rows, columns, split counts and sizes per recording)
and `verification.json` under the configured `care/` directory.


## Dataset-owned code and selectors

CARE behavior lives together in this directory:

- `loader.py`: farm/event selection, CSV types and delimiters, supporting tables,
  and training/prediction split selection.
- `validation.py`: offline completeness, integrity and full-loading checks.
- `viz.py`: CARE plot specifications.
- `metadata.json`: source, checksum, expected files, license and dataset description.

The common `pdmdata.load()` dispatches to this loader. It also supports direct use:

```python
from pdmdata.datasets.care import load

train = load(wind_farm="A", event_id=0, split="train")
prediction = load(wind_farm="A", event_id=0, split="prediction")
```

`wind_farm` accepts A/B/C (also lowercase); `event_id` is a non-negative integer.
`table="data"` is the default and requires an event ID. `table="events"` and
`table="features"` select farm-level tables and do not accept an event ID or split.
All modes return a Polars LazyFrame unless `lazy=False` is specified.

The `events` table normalizes Wind Farm A's `asset` column to `asset_id`, matching
B/C. Raw CSVs are never rewritten. The older `recording="Wind Farm A/datasets/0.csv"`
selector remains available and preserves the original table columns; use it
separately from the structured selectors above.
