"""Create the MetroPT2 operating-cycle exploration notebook."""

import json
from pathlib import Path


def main():
    cells=[]
    def add(kind,source):
        c=dict(cell_type=kind,id=f'cell-{len(cells):02d}',metadata={},source=source.splitlines(True))
        if kind=='code':c.update(execution_count=None,outputs=[])
        cells.append(c)
    add('markdown','''# MetroPT2: operating cycles and sensor relationships

Explore real compressor signals alongside binary controls. Unlike GDD, MetroPT2
has no per-sample operating-state labels. Control combinations below are
**descriptive reference codes**, not ground-truth states or fault classes.

[Official source](https://zenodo.org/records/7766691)
· [Dataset guide](../../pdmdata/metropt2/README.md)

Run `pdmdata.download("metropt2")` explicitly if local data is missing. This
notebook reads a limited prefix of the downloaded CSV, without downloading or
saving another copy of the raw measurements.
''')
    add('code','''from pathlib import Path
import sys
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pdmdata").is_dir())
sys.path.insert(0,str(ROOT))
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from IPython.display import display
import pdmdata
from pdmdata.metropt2.viz import ANALOG_SIGNALS, CONTROL_SIGNALS, plot_operation

scan = pdmdata.load("metropt2")
print("Source columns:", scan.collect_schema().names())
N_SAMPLES = 21600
sample = scan.select(["timestamp", *ANALOG_SIGNALS, *CONTROL_SIGNALS]).head(N_SAMPLES).collect()
print(f"Loaded {sample.height:,} observations; {sample['timestamp'][0]} to {sample['timestamp'][-1]}")
display(sample.head())
''')
    add('markdown','''## Sampling and signal checks

The provider describes nominal 1 Hz logging; actual timestamps can be irregular.
We retain observed timestamps and do not resample. The plotted prefix is an
illustration, not a representative sample of all operating and fault conditions.
''')
    add('code','''dt = sample["timestamp"].diff().dt.total_milliseconds()/1000
print(f"Median interval: {dt.median():.3f} s; minimum: {dt.min():.3f}; maximum: {dt.max():.3f}")
print(f"Backward steps: {(dt < 0).sum()}; duplicate steps: {(dt == 0).sum()}; gaps > 5 s: {(dt > 5).sum()}")
display(sample.null_count())
display(pl.DataFrame({
    "signal": ANALOG_SIGNALS,
    "unique_values": [sample[c].n_unique() for c in ANALOG_SIGNALS],
    "standard_deviation": [sample[c].std() for c in ANALOG_SIGNALS],
}))
print("Constant analogue channels in this window:",
      [c for c in ANALOG_SIGNALS if sample[c].drop_nulls().n_unique() <= 1])
for c in CONTROL_SIGNALS:
    print(c, "observed values:", sample[c].unique().sort().to_list())
assert sample["timestamp"].is_sorted(), "Inspect timestamp reversals before plotting"
''')
    add('markdown','''## Overview: repeated control changes and sensor response

The upper panel shows each binary control separately, vertically offset for
readability. Analogue signals retain their source values. Gaps longer than five
seconds are broken in the traces; no smoothing or interpolation is applied.
''')
    add('code','''overview = plot_operation(sample)
plt.show()
''')
    add('markdown','''## Zoom into a 20-minute interval

Adjust START_MINUTES and WINDOW_MINUTES to inspect another part of the prefix.
The default starts at the beginning, without selecting for a particular fault.
''')
    add('code','''START_MINUTES, WINDOW_MINUTES = 0, 20
elapsed = (pl.col("timestamp") - sample["timestamp"][0]).dt.total_milliseconds()/60000
window = sample.filter((elapsed >= START_MINUTES) & (elapsed < START_MINUTES + WINDOW_MINUTES))
zoom = plot_operation(window)
plt.show()
''')
    add('markdown','''## Observed control combinations and their persistence

Encode `COMP + 2 * DV_eletric + 4 * Towers` only to identify combinations.
The numeric order does not mean severity, and the eight possible codes need not
all occur. Control signals can lag or lead the analogue response. Keep them out
of sensor-only model inputs if using them as evaluation references.
''')
    add('code','''for c in CONTROL_SIGNALS:
    assert sample[c].null_count() == 0 and set(sample[c].unique().to_list()) <= {0,1}
annotated = sample.with_row_index("sample_index").with_columns(
    (pl.col("COMP") + 2*pl.col("DV_eletric") + 4*pl.col("Towers")).cast(pl.UInt8).alias("control_code")
)
# Do not count transitions or segment lengths across recording gaps.
breaks = ((pl.col("control_code") != pl.col("control_code").shift(1)) |
          (pl.col("timestamp").diff().dt.total_milliseconds() > 5000)).fill_null(True)
segments = annotated.with_columns(breaks.cast(pl.UInt32).cum_sum().alias("segment"))
segments = segments.group_by("segment", maintain_order=True).agg(
    pl.col("control_code").first(), pl.col("timestamp").first().alias("start"),
    pl.col("timestamp").last().alias("last_observation"), pl.len().alias("samples")
).with_columns((pl.col("last_observation")-pl.col("start")).dt.total_milliseconds().truediv(1000).alias("observed_span_s"))
display(annotated.group_by(CONTROL_SIGNALS + ["control_code"]).len().sort("control_code"))
display(segments.group_by("control_code").agg(pl.len().alias("segments"),
    pl.col("samples").median().alias("median_samples"),
    pl.col("observed_span_s").median().alias("median_observed_span_s")).sort("control_code"))
display(segments.head(12))
''')
    add('markdown','''## Analogue profiles for each observed control combination

These descriptive medians help identify how pressure, current, temperature,
and flow vary with control signals. They are not learned dependency graphs.
''')
    add('code','''display(annotated.group_by("control_code").agg(pl.col(ANALOG_SIGNALS).median()).sort("control_code"))
''')
    add('markdown','''## Fault reports are a separate source of information

The source CSV is unlabeled. The provider lists these intervals:

| Report | Start | End |
|---|---|---|
| Air leak | 2022-06-04 10:19:24.300 | 2022-06-04 14:22:39.188 |
| Oil leak | 2022-07-11 10:10:18.948 | 2022-07-14 10:22:08.046 |

The default prefix predates both intervals. Neither absence from these intervals
nor a particular control code establishes healthy operation. To inspect another
period, filter the lazy scan by timestamp before collecting a limited window.

## Relevance to dependency-based segmentation

- Start with the eight analogue signals; GPS and binary controls are excluded.
- Examine constant channels and scale differences within each training block.
- Distinguish changes in mean/variance from changes in conditional dependencies.
- Compare sensor-only segmentation with the observed control switching, while
  recognizing that these controls are not independent ground-truth annotations.
- Evaluate multiple separated periods; this short prefix alone cannot establish
  multi-scale behavior or generalization across the whole recording.
''')
    root=Path(__file__).resolve().parents[1]
    p=root/'notebooks/datasets/metropt2.ipynb'
    p.write_text(json.dumps(dict(cells=cells,metadata={'kernelspec':{'name':'pdmdata','display_name':'Python (pdmdata)','language':'python'},'language_info':{'name':'python','version':'3.11'}},nbformat=4,nbformat_minor=5),indent=1)+'\n')


if __name__ == '__main__':
    main()
