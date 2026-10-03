# XJTU-SY: Bearing Run-to-Failure Datasets

<!-- BEGIN GENERATED METADATA -->

## Dataset overview

Accelerated run-to-failure vibration records for 15 rolling-element bearings under three operating conditions, used widely for bearing RUL prognosis.

| Item | Details |
|---|---|
| ID | `xjtu_sy` |
| Name | XJTU-SY Bearing Datasets |
| Provider | [Xi'an Jiaotong University / Sumyoung Technology](https://biaowang.tech/xjtu-sy-bearing-datasets/) |
| DOI | [10.1109/TR.2018.2882682](https://doi.org/10.1109/TR.2018.2882682) |
| Availability | available (checked: 2026-10-03) |
| Access | Direct ZIP download via Hugging Face mirror of the official bundle (~5.4 GB) |
| License | [Public research use (citation requested)](https://biaowang.tech/xjtu-sy-bearing-datasets/) |
| Commercial use | Unknown |
| Redistribution | Unknown |
| Data type | Run-to-failure vibration time series |
| Feature dimensions | 2 accelerometer channels per 1.28 s snapshot (horizontal, vertical) |
| Feature counting | Each CSV snapshot has 32,768 samples at 25.6 kHz with two columns (horizontal, vertical). Snapshot-level loaders add derived RMS features and RUL in minutes; raw waveforms are opt-in. |
| Feature-count source | [XJTU-SY Bearing Datasets (Wang et al.)](https://biaowang.tech/xjtu-sy-bearing-datasets/) |
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
| `bearing` | Utf8 | entity | Bearing run id (Bearing1_1 … Bearing3_5). |
| `condition` | Utf8 | operating-condition | Operating-condition folder (35Hz12kN, 37.5Hz11kN, or 40Hz10kN). |
| `cycle` | UInt32 | time | One-based snapshot index (one CSV per minute). |
| `RUL` | Int64 | target | Minutes remaining until the final snapshot (0 at failure). |
| `rms_horizontal, rms_vertical` | Float64 | sensor | Per-snapshot RMS of the two accelerometer channels. |
| `horizontal, vertical` | Float64 | sensor | Raw waveform samples when with_waveform=True. |

## Usage notes

- Official distribution is a multi-part WinRAR set; PdMData downloads a Hugging Face ZIP mirror of the same files (~5.4 GB).
- Fifteen bearings under three conditions; sampling at 25.6 kHz with 32,768 points every minute.
- RUL is derived from complete run-to-failure trajectories (minutes until the last CSV), not a separate label file.
- Cite Wang, Lei, Li, and Li (IEEE Transactions on Reliability, 2020).

## Download

```bash
uv run --locked python scripts/download.py xjtu_sy
```

Source archive: [XJTU-SY_Bearing_Datasets.zip](https://huggingface.co/datasets/DavidNguyen/XJTU-SY_Bearing_Datasets/resolve/main/XJTU-SY_Bearing_Datasets.zip)

## Suggested citation

B. Wang, Y. Lei, N. Li and N. Li (2020). A Hybrid Prognostics Approach for Estimating Remaining Useful Life of Rolling Element Bearings. IEEE Transactions on Reliability, 69(1), 401-412.

<!-- END GENERATED METADATA -->

## Load bearings

Download the Hugging Face ZIP mirror once (~5.4 GB), then select a bearing:

```python
import pdmdata
from pdmdata.datasets.xjtu_sy import inventory, load

pdmdata.download("xjtu_sy")
inventory()
# Snapshot-level RMS features + derived RUL (minutes).
history = load("Bearing2_4", with_rul=True)
# One raw 1.28 s waveform (32,768 samples, two channels).
wave = load("Bearing2_4", cycle=1, with_waveform=True)
```

`RUL` is minutes until the final CSV of that bearing (0 at the last
snapshot). There is no separate official RUL file; labels come from complete
run-to-failure trajectories.

Files are stored under `data/raw/xjtu_sy/` by default.
