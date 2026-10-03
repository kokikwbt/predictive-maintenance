# MetroPT2: A Benchmark Dataset for Predictive Maintenance

<!-- BEGIN GENERATED METADATA -->

## Dataset overview

A 1 Hz multivariate time series from a metro-train air production unit, with analogue and digital sensor signals and documented air- and oil-leak intervals.

| Item | Details |
|---|---|
| ID | `metropt2` |
| Name | MetroPT2: A Benchmark Dataset for Predictive Maintenance |
| Provider | [INESC TEC on Zenodo](https://zenodo.org/records/7766691) |
| DOI | [10.5281/zenodo.7766691](https://doi.org/10.5281/zenodo.7766691) |
| Availability | available (checked: 2026-09-06) |
| Access | Direct download from Zenodo |
| License | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Commercial use | Yes |
| Redistribution | Yes |
| Data type | Large multivariate equipment time series |
| Feature dimensions | 20 input signals: 16 equipment signals + 4 GPS fields |
| Feature counting | Counts the columns of MetroPT2.csv excluding timestamp (21 raw columns). GPS fields are gpsLat, gpsLong, gpsSpeed, and gpsQuality; selecting equipment signals alone gives 16 dimensions. |
| Feature-count source | [INESC TEC on Zenodo](https://zenodo.org/records/7766691) |
| Tasks | Online anomaly detection, Failure prediction, RUL research |

## Experiment-task suitability

| Task | Support |
|---|---|
| Anomaly detection | Requires target derivation |
| Fault or health-state classification | Requires target derivation |
| Condition estimation | Requires target derivation |
| Remaining useful life prediction | Requires target derivation |
| Time-to-event prediction | Requires target derivation |

## Attributes

| Attribute or group | Type | Role | Description |
|---|---|---|---|
| `timestamp` | Datetime | timestamp | Observation time for measurements logged at 1 Hz. |
| `analogue sensors` | Float64 | sensor | Pressure, motor-current, oil-temperature, flowmeter, GPS, and related analogue measurements. |
| `digital signals` | Boolean/Integer | machine-state | Control and discrete signals including air-intake-valve states. |
| `failure interval` | Datetime interval | derived-target | Air-leak and oil-leak intervals documented by the provider outside the CSV labels. |

## Usage notes

- The current version contains about 7.1 million observations and 21 attributes.
- The source CSV is unlabeled; task targets must be derived from the two provider-documented failure intervals.
- The 1.2 GB CSV is opt-in and is not downloaded by scripts/bootstrap.sh.

## Download

```bash
uv run --locked python scripts/download.py metropt2
```

Source archive: [MetroPT2.csv](https://zenodo.org/records/7766691/files/MetroPT2.csv?download=1)

## Suggested citation

Veloso, B., Gama, J., Ribeiro, R. and Pereira, P. (2023). MetroPT2: A Benchmark dataset for predictive maintenance. Zenodo. https://doi.org/10.5281/zenodo.7766691

<!-- END GENERATED METADATA -->

## Explore operating cycles

The [MetroPT2 notebook](../../../notebooks/datasets/metropt2.ipynb) compares binary
controls with pressure, current, temperature, and flow over a short prefix of the
real recording. It includes a longer overview and a 20-minute zoom, with saved
Matplotlib outputs. It never downloads data implicitly.

```python
import pdmdata
from pdmdata.datasets.metropt2.viz import plot_operation

frame = pdmdata.load("metropt2").head(1200).collect()
figure = plot_operation(frame)
```

Control combinations are descriptive reference codes, not ground-truth
operating states or fault labels. The original CSV has no per-sample labels.
The provider's air/oil leak intervals are documented separately in the notebook.
