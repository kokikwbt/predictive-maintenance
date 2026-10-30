# CWRU Bearing Data Center

<!-- BEGIN GENERATED METADATA -->

## Dataset overview

Official MATLAB vibration recordings for normal bearings and drive-end/fan-end seeded faults, covering all 161 file links across the four provider collections with channel-aware loading.

| Item | Details |
|---|---|
| ID | `cwru` |
| Name | CWRU Bearing Data Center |
| Provider | [Case Western Reserve University Bearing Data Center](https://engineering.case.edu/bearingdatacenter/download-data-file) |
| DOI | — |
| Availability | available (checked: 2026-10-30) |
| Access | Official per-file MATLAB downloads; collection bundles and bounded recording selection |
| License | [Unspecified on the provider pages](https://engineering.case.edu/bearingdatacenter/welcome) |
| Commercial use | Unknown |
| Redistribution | Unknown |
| Data type | Seeded-fault vibration recordings |
| Feature dimensions | Up to three source accelerometer channels (DE, FE, BA), depending on recording |
| Feature counting | Long-form rows retain independent channel sample indices and unmodified source values. Channels may have different lengths. Channel rates are documented where unambiguous; otherwise sample_rate_hz and time_s are null unless explicitly overridden. |
| Feature-count source | [CWRU collection and channel documentation](https://engineering.case.edu/bearingdatacenter/download-data-file) |
| Tasks | Fault classification, Healthy/faulty discrimination, Anomaly detection research, Cross-condition adaptation, Vibration signal analysis |

## Experiment-task suitability

| Task | Support |
|---|---|
| Anomaly detection | Direct |
| Fault or health-state classification | Direct |

## Attributes

| Attribute or group | Type | Role | Description |
|---|---|---|---|
| `amplitude` | Float64 | sensor | Unmodified selected accelerometer sample from the source MATLAB array. |
| `recording, collection` | Mixed | entity | Official file identifier and collection: normal, 12k_de, 48k_de, or 12k_fe. Recordings are not independent machines. |
| `channel, source_variable` | String | sensor | DE, FE or BA and the exact MATLAB variable name. Available channels are inspected per file. |
| `sample, time_s` | Mixed | time | Original zero-based per-channel sample index; time_s is sample/rate when rate is known. Neither is an absolute timestamp. |
| `sample_rate_hz, rate_source` | Mixed | provenance | Documented or user-overridden channel rate; null with unspecified origin when ambiguous. Overrides remain explicitly labeled. |
| `load_hp` | UInt8 | operating-condition | Motor load (0, 1, 2 or 3 horsepower) transcribed from the source table. |
| `fault, fault_location` | String | target | normal, inner_race, ball or outer_race; fault_location identifies the defective bearing (DE/FE), not the accelerometer used. |
| `diameter_in, outer_race_position` | Mixed | operating-condition | Seeded diameter (0 for normal) and clock position for outer-race defects, null when not applicable. |
| `rpm` | Float64 | operating-condition | Actual scalar source RPM if present, otherwise null; approximate source-table RPM is not substituted. |

## Usage notes

- All 161 official links are indexed: normal (4), 12k_de (60), 48k_de (52), and 12k_fe (45). A missing source-table combination is not fabricated.
- Collection names identify the provider pages, not a guarantee that all channels share the heading's sampling rate. Normal rates, non-DE channels in 48k_de and the 0.028 inch files on the 12k page are left unspecified unless explicitly overridden.
- All collections are opt-in. The load0..load3 aliases preserve the original 52-file 12k_de subset (excluding 0.028 inch files), with 13 files per load. New collection variants include the complete source tables.
- The default loader remains recording 105, channel DE. Select channels='all' for long-form multi-channel output; unequal lengths are retained without padding or resampling.
- Normal and seeded-fault recordings support static healthy/faulty detection and fault-location classification. No within-run fault-onset, RUL, censoring, maintenance outcome, or naturally progressing failure labels are supplied.
- Use recording identity and operating conditions for experiment splits; random overlapping windows can leak source information. Only one test rig and four motor loads are represented.
- The provider's general apparatus narrative and collection tables use different outer-race orientation wording. The manifest retains clock positions from the collection tables without inferring a load-zone orientation.
- Official pages publish no versioned immutable release or source checksums. Local SHA-256 records identify downloaded bytes but are not provider-authenticated checksums.
- Identical copies across collection bundles and older load bundles are reused. Conflicting local copies raise an error instead of silently selecting one.
- Windowing, support/query episodes, normalization, class remapping and model training belong in task packages. Data loading never downloads implicitly.

## Download

```bash
uv run --locked python scripts/download.py cwru --variant load0
```

Available variants: `12k_de`, `12k_fe`, `48k_de`, `load0`, `load1`, `load2`, `load3`, `normal`.

Source page: [Case Western Reserve University Bearing Data Center](https://engineering.case.edu/bearingdatacenter/download-data-file)

## Suggested citation

Case Western Reserve University Bearing Data Center. Bearing fault data and experiment documentation. https://engineering.case.edu/bearingdatacenter/welcome

<!-- END GENERATED METADATA -->

## Collections and downloads

The manifest covers all 161 file links in the four official source tables:

| Collection | Recordings | Contents |
|---|---:|---|
| `normal` | 4 | Normal bearings at four loads |
| `12k_de` | 60 | Drive-end faults, including 0.028 inch defects |
| `48k_de` | 52 | Drive-end faults in the 48 kHz collection |
| `12k_fe` | 45 | Fan-end faults |

All downloads are explicit and excluded from bulk bootstrap. Use the common
API for a collection or the bounded dataset API for individual records:

```python
import pdmdata
from pdmdata.datasets import cwru

pdmdata.download("cwru", variant="normal")
cwru.download(recording=[105, 109, 278], max_bytes=64 * 1024**2)
cwru.download(collection="12k_de", load_hp=[0, 1])
```

The legacy `load0` through `load3` variants retain the original 52-recording
subset, with 13 files per load and no 0.028 inch defects. The default common
variant remains `load0`. Collection variants use separate archive filenames.

The dataset-specific `download()` requires at least one selector, supports all
inventory filters, and accepts `overwrite=False` and `max_bytes=1073741824`
(1 GiB). Existing file sizes count toward the total cap. Completed transfers
survive a later failure and are reused on retry. The local manifest contains
source URLs, byte sizes and self-computed SHA-256, not provider checksums.
New transfers are parsed before publication. An invalid/truncated MATLAB
response is retried up to three times; failed replacement preserves the
previous local source.

## Recording discovery

```python
records = cwru.inventory(
    collection=["12k_de", "48k_de"], load_hp=[0, 1],
    fault="inner_race", diameter_in=0.007,
)
local = cwru.inventory(installed=True)
missing = cwru.inventory(installed=False)
```

Filters are `recording`, `collection`, `load_hp`, `fault`, `diameter_in`,
and `outer_race_position`. Scalars and lists are supported: AND across
filters, OR within a list. Inventory requires no downloaded files unless
`installed=True/False` requests a filesystem scan.

`fault` is `normal`, `inner_race`, `ball`, or `outer_race`.
`fault_location` identifies the defective bearing (DE/FE); `channel`
identifies the accelerometer. A fan-end fault may be measured at the drive end.
Normal bearings have diameter zero and null fault location. Clock position is
null except for outer-race defects. Missing source-table combinations are not
fabricated.

## Channels, sample ranges and rates

```python
signal = pdmdata.load("cwru", recording=105)  # Default channel: DE
signals = cwru.load(278, channels=["DE", "FE", "BA"])
detail = cwru.load(105, channels="all", start=1000, stop=5000, step=2)
info = cwru.channel_info(278)
```

The default recording remains 105. Select DE, FE, BA, a list, or `"all"`.
Explicitly requesting a missing channel fails. Long-form output keeps each
channel's original sample indices and amplitude; unequal lengths are retained
without padding or resampling. Output also retains collection, fault context,
measured RPM, source variable, rate provenance and relative `time_s`.

`start/stop/step` select original zero-based sample indices (exclusive stop).
They limit output, not MATLAB file parsing. Selecting every nth sample does
not apply an anti-alias filter and is not an analysis resampling routine.

RPM remains null when absent; no approximate table RPM is substituted.
Collection headings do not establish every channel's physical rate.
The adapter uses 12 kHz for ordinary 12 kHz records and 48 kHz for DE in
the 48 kHz collection. Normal rates, non-DE rates in the 48 kHz collection,
and 0.028 inch records on the 12 kHz page remain unspecified. Their
`sample_rate_hz` and `time_s` are null until a documented override:

```python
# Use only after independently establishing this source channel's rate.
signal = cwru.load(97, sample_rates={"DE": 48000})
```

Overrides are marked `rate_source="user_override"`; they do not rewrite source
files or claim provider verification.

## Multiple recordings and raw variables

```python
for signal in cwru.iter_load(collection="12k_fe", load_hp=0, channels="all"):
    print(signal.select("recording", "channel").unique())

variables = cwru.variables(99)
original = cwru.load_variable(99, "X098_DE_time")
```

`iter_load()` accepts inventory filters plus channel, slice and rate options.
It yields one recording at a time; no continuous trajectory is invented.

Some files include auxiliary or embedded previous recordings: `99.mat`
contains both X098 and X099 series. The loader prefers variable names matching
the requested identifier; otherwise a unique suffix is accepted (for example,
`3001.mat` contains `X056_DE_time`). Ambiguous alternatives fail.
`variables()` lists every source variable and `load_variable()` returns a
named raw array with its original shape and dtype.

Identical copies in legacy and collection bundles are accepted; conflicting
copies of the same official filename fail explicitly.

## Validation and plotting

```python
checks = cwru.validate(recording=[97, 105, 109, 278, 3001])
local_checks = cwru.validate(installed_only=True)

pdmdata.visualize("cwru", "waveform", cwru.load(278, channels="all")).show()
pdmdata.visualize("cwru", "spectrum", cwru.load(105), nperseg=4096).show()
```

Validation reports per-channel samples, measured RPM, source variable and
source-file SHA-256, checking exact equality of adapter and source arrays.
It fails on malformed/non-finite vectors, ambiguous variables and missing
selected files. `installed_only=True` explicitly limits coverage to installed
files. Source and adapter use the same SciPy reader; this is not independent
parser validation or physical validation. See [validation.json](validation.json)
for the actual coverage and hashes.

Waveforms keep recording/channel traces separate and limit displayed points
(default 10,000 per trace). Welch spectra require one recording/channel, a
known or explicitly supplied `sample_rate_hz`, and uniform sample indices.
A sliced frame's step adjusts its effective rate, without undoing aliasing.
Axes use source amplitude because calibration units are not established for
every file.

The [walkthrough notebook](../../../notebooks/datasets/cwru.ipynb) demonstrates
the APIs.

## Research uses and limits

Healthy and fault recordings support static anomaly detection, fault
classification, signal analysis and condition adaptation. There is one test
rig, four loads and three fault locations plus normal bearings. No natural
degradation history, fault-onset time, RUL, censoring or maintenance outcome is
supplied.

For meta-learning, split source/target conditions before making windows, keep
recording identity and use only allowed support data for scaling/thresholds.
Compare with pooled training and ordinary fine-tuning under the same budget.
Windows and load settings are not independent machines.

The general apparatus narrative and collection tables disagree on load-zone
orientation wording. Clock positions are preserved without inferring a
corrected orientation. See the [meta-learning data guide](../../../docs/meta-learning-data.md)
for evaluation choices and complementary fleets.
