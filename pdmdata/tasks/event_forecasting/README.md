# Recorded event-set forecasting

The initial package supports PCIC alarm source tables. It predicts the next **different recorded start tick** and the nonempty set of alarm tokens at that tick, within one literal device. Equal-tick rows have no inferred internal order. Native tick units are unspecified. This is completed-pair prediction, without physical failure, uptime or survival-censoring labels.

```python
from pdmdata.tasks.event_forecasting import RecordSetSpec, export_pcic_record_sets

spec = RecordSetSpec(train_fraction=0.6, validation_end_fraction=0.8, lookback=16)
export_pcic_record_sets("new-output-directory", variant="wireless", spec=spec)
```

The exporter never downloads missing raw data. Configure the data root and download the selected PCIC variant separately. It writes source groups with original row indices, history tensors, positive integer waits, nonempty target bits, metadata, source/raw hashes and per-file checksums. Model code remains outside this package.

The vocabulary uses only raw alarm tokens before the training calendar cut. OTHER is one presence bit for any unrecognized token; multiple unknown tokens merge into that bit, with raw and mapped target cardinalities retained separately. Device IDs are addressing metadata, not prescribed model inputs. Each history contains at most `lookback` completed groups through its origin. The first retained gap is zero; no time from a discarded prefix is silently included.

Training pairs have both origin and target strictly before the first cut. Validation pairs start at or after that cut and end strictly before the second. Test pairs start at or after the second cut and include the final recorded target. Split-crossing pairs, insufficient histories and the last group without a target are counted explicitly. Fractions refer to the full source's recorded tick range, not an inferred observation window. End timestamps and reference arrays are never used to construct features or labels.
