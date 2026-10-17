# Dataset candidates

Reviewed on 2026-10-18. ALPI has graduated to a [supported adapter](../pdmdata/datasets/alpi/README.md)
after an actual CLI download, provider-checksum verification, and full-file
loader validation. The other entries remain future adapter candidates; their
source files have not been validated. A dataset can be useful to PdMData without
supporting a particular research method. Research-specific eligibility and
effectiveness must be assessed separately.

[Supported catalog](../pdmdata/README.md) · [Usage guide](usage.md)

## Recommended additions

| Priority | Candidate and primary source | Tasks worth supporting | Evidence and limitations | Current status |
|---|---|---|---|---|
| 1 | [ALPI: Alarm Logs in Packaging Industry](https://data.mendeley.com/datasets/4nhx2x67cd/1), DOI 10.17632/4nhx2x67cd.1 | Next-alarm prediction, future-window alarm prediction, event-sequence forecasting | The provider describes 20 industrial packaging machines, 154 alarm codes, and logs from 2019-02-21 to 2020-06-17. Alarm prediction is a direct use of recorded events; an alarm is not automatically a physical failure or maintenance intervention. CC BY 4.0. | Supported as `alpi`: CLI transfer of the 8.5 MB ZIP succeeded without credentials; provider SHA-256 matched. The raw loader was checked against all 444,834 source events, 20 machines, and 154 codes. See the adapter validation record. Previously a reference link only. |
| 2 | [Oxford Battery Degradation Dataset 1](https://ora.ox.ac.uk/objects/uuid%3A03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac), DOI 10.5287/bodleian:KO2kdmYGg | Battery condition and capacity estimation; degradation forecasting; threshold-based RUL after defining an endpoint | The provider describes laboratory cycling of eight lithium-ion pouch cells and lists a main MATLAB file of about 254 MB. ODC-ODbL is listed on the provider record. Only eight cells; waveform samples are not independent batteries. Verify the readme, recorded fields, characterization spacing, and end conditions before assigning task support. | Recommended for a complementary battery-domain adapter feasibility check. The record is indexed, but direct page access returned HTTP 403 during this review. No automated retrieval or file validation. Previously a reference link only. |
| 3 | [REFIT electrical measurements](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements/) with [annotated load anomalies](https://pureportal.strath.ac.uk/en/datasets/annotated-load-anomalies-from-the-refit-dataset/) | Appliance-load anomaly detection; energy disaggregation as a supporting task; operating-state analysis after explicit state construction | Power records cover 20 households; a separate annotation release selects five houses with many anomalies. Both sources list CC BY 4.0. The accompanying paper includes unusual usage and settings among possible anomaly causes. These are not verified component-failure or repair records. The cleaned measurement release includes gap filling. | Recommended as an appliance condition-monitoring candidate. Verify raw/cleaned variants, annotation schema, timestamp alignment, and appliance mapping. No adapter or raw-file validation yet. |

Priority reflects a gap in the current catalog and the described task, not a
claim of model performance or an unconditional adoption decision. ALPI adds
real industrial event logs, Oxford adds battery degradation, and REFIT adds
appliance monitoring with a separate annotation source. ALPI is now registered in the supported catalog and opt-in CLI download. Oxford
and REFIT remain unimplemented candidates. ALPI is excluded from bulk downloads.

## Supporting and out-of-scope candidates

- [Intel Lab Data](https://db.csail.mit.edu/labdata/labdata.html) is a useful
  auxiliary candidate for missing-record analysis, sensor-quality monitoring,
  imputation experiments, and forecasting. The provider permits use and
  reproduction with acknowledgment. The deployment does not supply a
  documented failure/repair endpoint or fault labels, and voltage is strongly
  correlated with temperature. Do not label missing epochs as sensor failures
  or derive battery RUL solely from the final record. Keep it outside the main
  PdM adapter queue until a concrete sensor-monitoring use is specified.
- [Cortex latency observations](https://github.com/beorn7/histogram_experiments)
  describe software-service performance. They may be useful for general
  observability research, but do not fill a physical-asset maintenance gap in
  this catalog. They are not recommended as a core PdMData addition here.
- HydSys and IMS already have adapters. A new task for those datasets should
  reuse the existing dataset IDs rather than add duplicate adapters.

## Adapter acceptance criteria

1. Pin a provider version, file manifest, download URLs, checksums when supplied,
   citation, and dataset-specific terms. Retain the source terms; do not infer
   a data license from a mirror or a code repository alone.
2. Inspect source files through a source-faithful adapter. Establish entity
   identifiers, physical units, timestamps, ordering, duplicates, missingness,
   event meanings, and termination or censoring semantics. Preserve raw values;
   do not silently fill gaps, invent labels, or manufacture continuous time.
3. Implement selectors and inventory with consistent Polars schemas. Keep
   source tables separate when a join adds target information. For example,
   REFIT annotations must remain distinct from power measurements unless a
   caller explicitly requests a documented alignment.
4. Register the loader in `pdmdata/datasets/load.py`, add `metadata.json`, and
   generate dataset/catalog documentation. Mark native targets separately from
   derived targets. Task-specific labels, windows, entity splits, RUL endpoints,
   and train/test construction belong in `pdmdata/tasks/`, not the loader.
5. Validate parsing and selectors with representative fixtures, then compare
   at least one real provider file with the loader output. For multi-file data,
   check cross-file schema and entity coverage before claiming whole-dataset
   validation. Keep large downloads opt-in and out of Git.

## Candidate-specific implementation checks

**ALPI.** Preserve the provider's machine, timestamp, alarm-code, and event
fields after inspecting the actual schema. Check duplicate events, simultaneous
timestamps, timezone, machine observation windows, and any start/clear event
semantics. Next-event and future-window tasks need explicit handling of tied
events and end-of-record censoring. Do not generate a failure target from
every alarm.

**Oxford.** Preserve cells and characterization records, the recorded cycle
indices, and within-measurement curves. Verify the MATLAB version and field
structure before choosing a parser. Do not treat periodic characterization
measurements as a continuous record of all cycles. Any capacity threshold,
health normalization, or RUL construction requires an explicit task definition.

**REFIT.** Keep measurements and anomalies as separately selectable sources.
Record raw versus cleaned provenance and appliance-to-channel metadata. Inspect
the annotation readme and alignment with the chosen measurement version.
Unlisted periods are not automatically certified healthy periods; five selected
houses do not provide representative labels for all twenty houses. Preserve
the distinction between power, energy, usage anomalies, and confirmed failures.

## Sources checked

- ALPI provider page, version-1 public file API, readme, and full raw CSV,
  checked 2026-10-18. The CLI transfer matched the provider SHA-256; the
  adapter validation record reports the complete source comparison.
- Oxford provider search-index record and the [Battery Intelligence Lab data
  index](https://battery-intelligence-lab.github.io/data-and-code/), accessed
  2026-10-18. Direct file access and the provider readme were not verified.
- REFIT measurement and anomaly provider pages, plus the authors'
  [ICASSP 2019 paper](https://pure.strath.ac.uk/ws/portalfiles/portal/85978697/Rashid_etal_IEEE_ICASSP_2019_Evaluation_of_non_intrusive_load_monitoring_algorithms_for_appliance.pdf),
  accessed 2026-10-18.
- Intel Lab provider documentation, accessed 2026-10-18. Cortex provider
  README and reader code were examined for the preceding candidate review;
  these are auxiliary references, not adopted PdM datasets.
