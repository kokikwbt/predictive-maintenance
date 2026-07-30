# Ultrasonic Flowmeter Diagnostics

<!-- BEGIN GENERATED METADATA -->

> This section is generated from `metadata.json`. Do not edit it directly.

## Dataset overview

Classification data for four liquid ultrasonic flowmeters covering healthy operation, gas injection, installation effects, and waxing.

| Item | Details |
|---|---|
| ID | `ufd` |
| Name | Ultrasonic Flowmeter Diagnostics |
| Provider | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/433/ultrasonic+flowmeter+diagnostics) |
| DOI | [10.24432/C5B895](https://doi.org/10.24432/C5B895) |
| Availability | available (checked: 2026-07-24) |
| Access | Direct download / ucimlrepo |
| License | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Commercial use | Yes |
| Redistribution | Yes |
| Data type | Multivariate tabular |
| Tasks | Fault classification, Condition diagnosis |

## Experiment-task suitability

| Task | Support |
|---|---|
| Anomaly detection | Requires target derivation |
| Fault or health-state classification | Direct |

## Attributes

| Attribute or group | Type | Role | Description |
|---|---|---|---|
| `profile/flatness factor` | Float64 | feature | Profile factor or flatness ratio describing the flow-velocity profile; naming varies by meter. |
| `symmetry` | Float64 | feature | Diagnostic value describing flow symmetry. |
| `crossflow` | Float64 | feature | Diagnostic value describing crossflow. |
| `swirl_angle` | Float64 | feature | Swirl angle, available only for Meter B. |
| `flow_velocity_1..8` | Float64 | sensor | Flow velocity along each path: eight paths for Meter A and four for Meters B through D. |
| `sound_speed_1..8` | Float64 | sensor | Speed of sound along each path. |
| `average_flow/speed` | Float64 | feature | Average flow velocity or speed of sound across all paths, available for selected meters. |
| `signal_strength_1..8` | Float64 | sensor | Signal strength at both ends of each path for Meters B through D. |
| `turbulence_1..4` | Float64 | feature | Turbulence indicator for each path, available only for Meter B. |
| `meter_performance` | Float64 | feature | Meter-performance indicator, available only for Meter B. |
| `signal_quality_1..8` | Float64 | sensor | Signal quality at both ends of each path for Meters B through D. |
| `gain_1..16` | Float64 | sensor | Gain at both ends of each path: 16 values for Meter A and eight for Meters B through D. |
| `transit_time_1..8` | Float64 | sensor | Transit time at both ends of each path for Meters B through D. |
| `health_state` | Categorical | target | Healthy, gas injection, installation effects, or waxing; available classes vary by meter. |

## Usage notes

- Meters A through D contain 540 instances in total with no missing values.
- Attribute counts and available fault classes vary by meter.

## Download

```bash
python scripts/download.py ufd
```

Source archive: [ultrasonic-flowmeter-diagnostics.zip](https://archive.ics.uci.edu/static/public/433/ultrasonic+flowmeter+diagnostics.zip)

## Suggested citation

Gyamfi, K. and Marshall, C. (2017). Ultrasonic flowmeter diagnostics. UCI Machine Learning Repository.

<!-- END GENERATED METADATA -->
