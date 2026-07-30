# Backblaze Drive Stats

<!-- BEGIN GENERATED METADATA -->

> This section is generated from `metadata.json`. Do not edit it directly.

## Dataset overview

Daily fleet-scale snapshots of hard drives and SSDs, including device identity, model, failure status, and raw and normalized S.M.A.R.T. statistics.

| Item | Details |
|---|---|
| ID | `backblaze` |
| Name | Backblaze Drive Stats |
| Provider | [Backblaze](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data) |
| DOI | — |
| Availability | available (checked: 2026-07-25) |
| Access | Direct quarterly downloads |
| License | [Backblaze Drive Stats terms](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data) |
| Commercial use | Yes |
| Redistribution | Unknown |
| Data type | Fleet-scale daily reliability panel |
| Tasks | Drive failure prediction, Survival analysis, Time-to-event prediction |

## Experiment-task suitability

| Task | Support |
|---|---|
| Anomaly detection | Requires target derivation |
| Fault or health-state classification | Direct |
| Condition estimation | Requires target derivation |
| Remaining useful life prediction | Requires target derivation |
| Time-to-event prediction | Direct |
| Survival analysis | Direct |

## Attributes

| Attribute or group | Type | Role | Description |
|---|---|---|---|
| `date` | Date | time | UTC date of the daily drive snapshot. |
| `serial_number` | String | entity | Drive serial number used to construct longitudinal device histories. |
| `model / capacity_bytes` | String/UInt64 | asset-attribute | Drive model and capacity. |
| `failure` | UInt8 | target | One on the final day a drive is observed before a reported failure, otherwise zero. |
| `smart_*_raw / smart_*_normalized` | Float64 | sensor | Raw and normalized S.M.A.R.T. attributes; availability varies by drive model and quarter. |

## Usage notes

- Data are partitioned by quarter to avoid requiring a full-history download.
- The default variant is 2025-q1; specify another quarter explicitly when required.
- Schema changes occur over time and S.M.A.R.T. attributes are highly sparse across drive models.
- Backblaze permits derivative works but asks users not to sell the source data itself.
- Quarterly archives are opt-in and are not downloaded by scripts/bootstrap.sh.

## Download

```bash
python scripts/download.py backblaze --variant 2025-q1
```

Available variants: `2025-q1`, `2025-q2`, `2025-q3`, `2025-q4`.

Source page: [Backblaze](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data)

## Suggested citation

Backblaze. Drive Stats: Hard Drive Test Data. https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data

<!-- END GENERATED METADATA -->
