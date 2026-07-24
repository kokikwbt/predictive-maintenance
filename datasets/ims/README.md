# IMS Bearings

<!-- BEGIN GENERATED METADATA -->

> This section is generated from `metadata.json`. Do not edit it directly.

## Dataset overview

Run-to-failure vibration measurements from accelerated bearing degradation experiments conducted by the Center for Intelligent Maintenance Systems.

| Item | Details |
|---|---|
| ID | `ims` |
| Name | IMS Bearings |
| Provider | [NASA Prognostics Center of Excellence](https://catalog.data.gov/dataset/ims-bearings) |
| DOI | None |
| Availability | available (checked: 2026-07-24) |
| Access | Direct download from NASA |
| License | [U.S. Government Works](https://www.usa.gov/government-works) |
| Commercial use | Unknown |
| Redistribution | Unknown |
| Data type | Run-to-failure vibration time series |
| Tasks | Fault diagnosis, Degradation estimation, Prognostics |

## Attributes

| Attribute or group | Type | Role | Description |
|---|---|---|---|
| `sample` | UInt32 | time | Sample index within an individual vibration recording. |
| `channel_1..8` | Float64 | sensor | Accelerometer channels. The number and bearing assignment of channels depend on the experiment. |
| `recording_time` | Datetime | timestamp | Recording timestamp encoded in each source filename. |
| `experiment` | Categorical | entity | One of the three run-to-failure bearing experiments. |

## Usage notes

- The archive contains three experiments with separate recording files.
- Each recording contains 20,480 vibration samples; channel count and bearing assignment vary by experiment.
- Failure labels and RUL values are not stored per row and must be derived from the experiment chronology and documentation.
- The catalog record points to the U.S. Government Works policy; users should review the source terms before redistribution.

## Download

```bash
python scripts/download.py ims
```

Source archive: [IMS.zip](https://data.nasa.gov/docs/legacy/IMS.zip)

## Suggested citation

Center for Intelligent Maintenance Systems, University of Cincinnati. IMS Bearings. NASA Prognostics Center of Excellence Data Repository.

<!-- END GENERATED METADATA -->
