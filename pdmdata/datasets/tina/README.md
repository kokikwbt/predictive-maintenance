# TiNA daily source files: candidate adapter

This adapter reads installed daily Parquet files from the authors' [TiNA repository](https://github.com/ari-dasci/OD-TINA). It preserves original columns, values, duplicate rows, and within-file order. It adds `source_row` and relative `source_file` for provenance. It does not construct event onsets, infer alarm latching, parse a clock, or create failure targets.

The first three source days were transferred with the pdmdata downloader and matched the pinned GitHub blob hashes. Full-dataset validation, catalog registration, a dataset-specific public download command, and task adoption are still pending. Loading never downloads missing files.

```python
from pdmdata.datasets.tina import load
source = load(days=["2017-09-06", "2017-09-07"], lazy=True)
print(source.collect_schema())
```

The source repository uses CC BY-SA 4.0. Raw downloads remain outside Git. A sample's alarm label must not be interpreted as a new alarm occurrence without checking the original recording semantics and adjacent rows.

The [sample validation record](assets/source-verification.json) covers five fixture tests and the first three source days (246,826 rows). Source storage order is not chronological. After sorting by the raw `unix` value, the selected days have one sample per second and no missing or duplicate clocks. The 4,489 non-`none` alarm rows form eight sustained label runs. These rows are not assumed to represent 4,489 separate alarm occurrences, and the observed clock spacing is a subset observation rather than a claim about all source days.
