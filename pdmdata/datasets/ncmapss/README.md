# N-CMAPSS: Aircraft Engine Run-to-Failure

<!-- BEGIN GENERATED METADATA -->

## Dataset overview

A next-generation C-MAPSS run-to-failure fleet simulated under recorded commercial flight conditions, with per-cycle RUL labels and multiple failure-mode subsets (DS01–DS08).

| Item | Details |
|---|---|
| ID | `ncmapss` |
| Name | Aircraft Engine Run-to-Failure Dataset under Real Flight Conditions |
| Provider | [NASA Prognostics Center of Excellence](https://data.phmsociety.org/nasa/) |
| DOI | [10.3390/data6010005](https://doi.org/10.3390/data6010005) |
| Availability | available (checked: 2026-10-03) |
| Access | Per-subset HDF5 downloads via Kaggle mirror (official NASA bundle is a 15 GB nested ZIP) |
| License | [U.S. Government Works / provider terms](https://www.usa.gov/government-works) |
| Commercial use | Unknown |
| Redistribution | Unknown |
| Data type | Multivariate run-to-failure time series under real flight conditions |
| Feature dimensions | 14 measured sensors + 4 flight-condition inputs; optional 14 virtual sensors and 10 health-modifier channels |
| Feature counting | Default loaders expose unit, cycle, flight class, health state, four W_* flight conditions, and 14 X_s sensors. Virtual sensors (X_v) and health modifiers (T) are opt-in. Per-cycle RUL is provided in Y_*. |
| Feature-count source | [Chao et al., Data 2021](https://doi.org/10.3390/data6010005), [NASA / PHM Society mirror](https://phm-datasets.s3.amazonaws.com/NASA/17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip) |
| Tasks | RUL estimation, Prognostics, Fault diagnostics |

## Experiment-task suitability

| Task | Support |
|---|---|
| Anomaly detection | Requires target derivation |
| Fault or health-state classification | Requires target derivation |
| Condition estimation | Requires target derivation |
| Remaining useful life prediction | Direct |
| Time-to-event prediction | Direct |
| Survival analysis | Direct |

## Attributes

| Attribute or group | Type | Role | Description |
|---|---|---|---|
| `unit_number` | UInt16 | entity | Engine unit identifier within the selected subset. |
| `cycle` | UInt32 | time | Flight/cycle index along the run-to-failure trajectory. |
| `Fc` | Float64 | operating-condition | Flight-class label from the auxiliary table. |
| `hs` | Float64 | asset-attribute | Health-state indicator from the auxiliary table. |
| `alt, Mach, TRA, T2` | Float64 | operating-condition | Flight-condition / scenario descriptors (W_*). |
| `T24..Wf` | Float64 | sensor | Fourteen measured sensor channels (X_s_*). |
| `RUL` | Int64 | target | Per-cycle remaining useful life from Y_dev / Y_test. |

## Usage notes

- The official NASA distribution is a ~15.7 GB nested ZIP. PdMData downloads one HDF5 subset at a time through a Kaggle mirror of the same files (default: ds01).
- Each subset file contains development (dev) and test splits with aligned A/W/X_s/X_v/T/Y tables.
- DS01 is the smallest practical subset for smoke tests (~2.9 GB, 6 development units and 4 test units).
- Unit IDs are local to each subset file; do not join units across variants.
- Cite Chao, Kulkarni, Goebel, and Fink (2021) when using this dataset.

## Download

```bash
uv run --locked python scripts/download.py ncmapss --variant ds01
```

Available variants: `ds01`, `ds02`, `ds03`, `ds04`, `ds05`, `ds06`, `ds07`, `ds08a`, `ds08c`, `ds08d`.

Source page: [NASA Prognostics Center of Excellence](https://data.phmsociety.org/nasa/)

## Suggested citation

M. Chao, C. Kulkarni, K. Goebel and O. Fink (2021). Aircraft Engine Run-to-Failure Dataset under Real Flight Conditions. NASA Prognostics Data Repository, NASA Ames Research Center.

<!-- END GENERATED METADATA -->

## Load a subset

The full NASA bundle is about 15 GB. Download one HDF5 variant at a time
(default `ds01`, ~2.9 GB):

```python
import pdmdata
from pdmdata.datasets.ncmapss import load, inventory

pdmdata.download("ncmapss", variant="ds01")
inventory("ds01")
dev = load("ds01", "dev", unit=1, with_rul=True)
test = load("ds01", "test", with_rul=True)
```

`split="train"` is accepted as an alias of `dev`. Virtual sensors and health
modifiers are opt-in:

```python
frame = load(
    "ds01",
    "dev",
    include_virtual=True,
    include_health_modifiers=True,
)
```

Files are stored under `data/raw/ncmapss/<variant>/` by default.
