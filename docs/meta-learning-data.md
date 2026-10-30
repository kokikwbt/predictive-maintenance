# Data for meta-learning in predictive maintenance

Meta-learning eligibility depends on the number of independent adaptation
tasks, available support labels, and the deployment question. Large numbers
of overlapping windows do not imply large numbers of independent machines.
No dataset listed here guarantees an advantage over pooled training or
ordinary pretraining followed by fine-tuning.

## Dataset choices

| Dataset | Natural task boundary | Adaptation question | Main constraint |
|---|---|---|---|
| [CWRU](../pdmdata/datasets/cwru/README.md) | Motor load, defect size, recording, fault-bearing location | Few labeled examples for an unseen load or defect configuration | One rig; 161 normal/fault source files across four collections; no RUL |
| [CARE](../pdmdata/datasets/care/README.md) | Turbine within one wind farm | A short normal history for a previously unseen turbine | Wind farms differ in sensor schema; event windows from the same turbine may overlap |
| [XJTU-SY](../pdmdata/datasets/xjtu_sy/README.md) | Bearing run and operating condition | RUL adaptation across complete training runs and held-out bearings | Only 15 bearings and three conditions; terminal fault labels do not timestamp fault onset |
| [C-MAPSS](../pdmdata/datasets/cmapss/README.md) | Engine unit and operating-condition subset | RUL adaptation to a new engine population | Simulated data; target-engine future lifetime labels are unavailable in prospective deployment |
| [ALPI](../pdmdata/datasets/alpi/README.md) | Packaging machine | Adapt next-alarm prediction to a new machine | Alarm codes are not automatically physical faults; machine histories must remain separate |

CWRU adds a small, inspectable cross-condition classification benchmark to the
existing fleet, event-log and run-to-failure datasets. Its download manifest is
transcribed from the [four official collection tables](https://engineering.case.edu/bearingdatacenter/download-data-file).
The [provider's format guide](https://engineering.case.edu/bearingdatacenter/download-data-file)
defines channel names. All 161 links are supported, including normal baselines,
12/48 kHz drive-end faults, fan-end faults and 0.028 inch defects.
Rate ambiguities are retained explicitly; a collection heading is not proof
of the acquisition rate of every channel. Different rates can themselves leak
the healthy/faulty label if preprocessing is not controlled.

## Evaluation contract

1. Define a task by machine, condition or recording before windowing. Make
   source-training, validation and held-out target groups explicit.
2. Specify the support information available at deployment: normal-only data,
   labeled faults, or labeled complete lifetime histories. These are distinct
   experimental assumptions.
3. Keep query labels and future query measurements out of adaptation,
   normalization, threshold calibration and model selection. Split by physical
   asset where identifiers permit; otherwise report recording-level limits.
4. Count shots in independent records or events as well as windows. Prevent
   overlapping or adjacent shared windows across support and query.
5. Compare no adaptation, pooled training, ordinary fine-tuning and meta-learning
   with the same backbone, data access, support budget and tuning budget.
6. Report results across held-out tasks and seeds. Use classification metrics
   for CWRU, early-warning/false-alarm metrics for CARE, and explicitly defined
   lifetime endpoints for RUL tasks. These metrics are not interchangeable.

Episode construction, targets, scaling and model-ready arrays belong in
`pdmdata/tasks/`. Dataset adapters expose source measurements and identifiers.

## Further data to consider

[Paderborn Bearing DataCenter](https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter)
is a strong next candidate: the provider describes four operating settings,
healthy bearings, artificial damage and real damage from accelerated lifetime
tests. That enables a more substantive artificial-to-real adaptation question
than window-level variation alone. Its CC BY-NC 4.0 terms, MATLAB structure and
source transfers must be checked before implementing and registering an adapter.
It is a researched candidate here, not an installed or supported dataset.
