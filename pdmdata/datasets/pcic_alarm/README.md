# PCIC alarm source candidate

This adapter preserves the original PCIC alarm tables and exposes reference arrays through a separate opt-in API. It does not define forecasting targets, causal graphs, physical failures or observation censoring.

<!-- BEGIN GENERATED METADATA -->

## Dataset overview

Original real network alarm tables with separate reference arrays. Candidate adapter; source clocks, observation windows and reference graph semantics require an audit.

| Item | Details |
|---|---|
| ID | `pcic_alarm` |
| Name | PCIC 2021 real network alarm sources |
| Provider | [gcastle-hub PCIC dataset distribution](https://github.com/gcastle-hub/dataset/tree/1858e9c39d90a524e7d8ff8b9456f4b1f231ff03/alarm) |
| DOI | — |
| Availability | available (checked: 2026-10-21) |
| Access | Public pinned archives; actual transfer and source audit pending |
| License | [Repository MIT; dataset-specific terms unconfirmed](https://github.com/gcastle-hub/dataset/blob/1858e9c39d90a524e7d8ff8b9456f4b1f231ff03/LICENSE) |
| Commercial use | Unknown |
| Redistribution | Unknown |
| Data type | Real network alarm source candidate |
| Feature dimensions | Raw categorical fields and recorded clock strings; no analogue channels |
| Feature counting | All source columns remain strings until a separate clock and schema audit. Source row is provenance, not an input feature. |
| Feature-count source | [Provider README](https://github.com/gcastle-hub/dataset/blob/1858e9c39d90a524e7d8ff8b9456f4b1f231ff03/README.md) |
| Tasks | Alarm sequence analysis research, Reference graph analysis research |

## Experiment-task suitability

| Task | Support |
|---|---|
| Event or sequence forecasting | Requires target derivation |

## Attributes

| Attribute or group | Type | Role | Description |
|---|---|---|---|
| `source_row` | UInt32 | provenance | Original Alarm.csv row index, excluding the header; retained without sorting or deduplication. |
| `Original Alarm.csv fields` | String | raw-fields | Original field names, ID tokens and clock text. No automatic datetime or future-label conversion. |

## Usage notes

- The pinned distribution contains Wireless and two Microwave archives. Download is opt-in and is excluded from bulk downloads.
- The downloader stores tar.gz as an opaque file. The loader reads bounded members without filesystem extraction.
- Reference arrays are never loaded by the alarm API; explicit numeric NPY loading disables pickle.
- A repository-level MIT notice is present. Original competition conditions and the scope of that notice for data are not yet independently confirmed.
- The adapter does not establish causal ground truth, topology direction, synchronized device clocks, physical onset or survival censoring.

## Download

```bash
uv run --locked python scripts/download.py pcic_alarm --variant wireless
```

Available variants: `microwave439`, `microwave474`, `wireless`.

Source page: [gcastle-hub PCIC dataset distribution](https://github.com/gcastle-hub/dataset/tree/1858e9c39d90a524e7d8ff8b9456f4b1f231ff03/alarm)

## Suggested citation

PCIC 2021 causal discovery competition, gcastle-hub dataset distribution; Qiao et al. (2023), Structural Hawkes Processes, IJCAI, doi:10.24963/ijcai.2023/633.

<!-- END GENERATED METADATA -->

Download one explicitly selected variant with `pdmdata.download("pcic_alarm", variant="wireless")`. The pinned tar.gz is retained as an opaque source file, without filesystem extraction. `pdmdata.load("pcic_alarm", variant="wireless", lazy=False)` reads Alarm.csv entirely as strings with an original `source_row` column. Clock units and semantics must be checked before conversion. `lazy=True` wraps the materialized table; it does not stream the archive.

`source_inventory(variant="wireless")` reports archive members, hashes and CSV headers. `read_reference(variant="wireless", member="DAG.npy")` explicitly reads a numeric reference matrix with pickle disabled. Alarm loading never opens the graph. Matrix orientation, label completeness, device meaning and original competition terms remain subjects of the source audit. The repository MIT notice is retained, but a separate dataset-specific rights grant has not been confirmed.
