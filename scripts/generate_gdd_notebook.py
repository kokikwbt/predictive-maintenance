"""Generate an unexecuted GDD state exploration notebook."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    cells=[]
    def add(kind, source):
        cell=dict(cell_type=kind,id=f'cell-{len(cells):02d}',metadata={},source=source.splitlines(True))
        if kind=='code':cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    add('markdown', '''# GDD: understanding machine states

Explore the real `Genesis_StateMachineLabel.csv` recording: state frequencies,
contiguous segments, transitions, motor signals, and binary control flags.

[Provider](https://www.kaggle.com/datasets/inIT-OWL/genesis-demonstrator-data-for-machine-learning)
· [Dataset guide](../../pdmdata/gdd/README.md)

The observed state IDs are 0–8. A verified mapping from these numbers to named
machine actions was not found in the downloaded files or the checked provider
information. We keep numeric IDs rather than inventing names. Descriptions below
are empirical signal profiles, not authoritative definitions of the states.

If needed, explicitly download first with `pdmdata.download("gdd")` or
`uv run --locked python scripts/download.py gdd`. This notebook only reads local data.
''')
    add('code', '''from pathlib import Path
import sys
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pdmdata").is_dir())
sys.path.insert(0, str(ROOT))
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from IPython.display import display
import pdmdata
from pdmdata.gdd.preprocessing import MOTOR_SIGNALS, state_segments
from pdmdata.gdd.viz import plot_states

state = pdmdata.load("gdd", series="state")
anomaly = pdmdata.load("gdd", series="anomaly")
assert state.drop("Label").equals(anomaly.drop("Label")), "Check alignment before comparing labels"
print(f"Observations: {state.height:,}; input fields: {state.width - 2}")
print("State IDs:", sorted(state["Label"].unique().to_list()))
print("Anomaly IDs:", sorted(anomaly["Label"].unique().to_list()))
''')
    add('markdown', '''## Timestamp quality

Preserve file order: sorting a clock reversal can interleave observations and
change the apparent state transitions. Report the actual time differences.
Use sample counts for segment lengths; multiplying by the median positive time
step is only an approximate duration, not a repaired timestamp.
''')
    add('code', '''delta = state["Timestamp"].diff()
median_dt = delta.filter(delta > 0).median()
print(f"Median positive interval: {median_dt:.4f} s")
print(f"Backward steps: {(delta < 0).sum()}; repeated timestamps: {(delta == 0).sum()}")
display(state.with_row_index("sample").with_columns(delta.alias("delta_seconds"))
        .filter(pl.col("delta_seconds") < 0).select("sample", "Timestamp", "delta_seconds"))
segments = state_segments(state)
print(f"Contiguous segments: {segments.height}; transitions: {segments.height - 1}")
''')
    add('markdown','''## Which states occur, and for how long?

Segment lengths measure consecutive equal labels. The first and last segments
may be truncated by the recording boundaries. Very brief states may be difficult
to recover with long covariance windows.
''')
    add('code','''counts = state.group_by("Label").len(name="observations")
summary = segments.group_by("Label").agg(
    pl.len().alias("segments"), pl.col("samples").min().alias("min_samples"),
    pl.col("samples").median().alias("median_samples"),
    pl.col("samples").max().alias("max_samples"),
).join(counts,on="Label").with_columns(
    (pl.col("observations") / state.height * 100).alias("observation_percent"),
    (pl.col("median_samples") * median_dt).alias("approx_median_seconds"),
).sort("Label")
display(summary)
fig, ax = plt.subplots(figsize=(9,3), layout="constrained")
ax.bar(summary["Label"], summary["observations"], color=plt.get_cmap("tab10")(np.arange(summary.height)))
ax.set(xlabel="State ID", ylabel="Observations", title="GDD state occupancy", xticks=summary["Label"].to_list())
plt.show()
''')
    add('markdown','''## State switching and waveforms

Colors identify categorical states; they do not encode severity or distance.
The first view shows the complete recording. The second shows the first 800
samples (about 38 seconds using the median interval). Adjust START and STOP.
All values are shown in source units, which are not verified physical units.
''')
    add('code','''overview = plot_states(state, start=0, stop=state.height)
plt.show()
''')
    add('code','''START, STOP = 0, 800
zoom = plot_states(state, start=START, stop=STOP)
plt.show()
''')
    add('markdown','''## Empirical signal profiles by state

Compare motor-signal medians and binary flag activation fractions. Fractions near
one mean that a flag is usually active in that labeled state. These summaries
help interpret the IDs without assigning unverified action names.
''')
    add('code','''motor_profile = state.group_by("Label").agg(pl.col(MOTOR_SIGNALS).median()).sort("Label")
display(motor_profile)
binary = [c for c in state.columns if c not in {"Timestamp", "Label"}
          and set(state[c].drop_nulls().unique().to_list()) <= {0,1}]
flags = state.group_by("Label").agg(pl.col(binary).mean()).sort("Label")
fig, ax = plt.subplots(figsize=(12,5),layout="constrained")
image = ax.imshow(flags.select(binary).to_numpy(),vmin=0,vmax=1,cmap="Blues",aspect="auto")
ax.set_yticks(range(flags.height),labels=flags["Label"].to_list())
ax.set_xticks(range(len(binary)),labels=[c.split(".")[-1] for c in binary],rotation=50,ha="right",fontsize=8)
ax.set(ylabel="State ID",title="Binary flag activation fraction by state")
fig.colorbar(image,ax=ax,label="Fraction equal to 1")
plt.show()
''')
    add('markdown','''## Transitions between segments

Counts below exclude self-transitions within a segment. They describe observed
order, not a causal graph or a complete specification of the PLC state machine.
''')
    add('code','''ids = sorted(state["Label"].unique().to_list())
index = {v:i for i,v in enumerate(ids)}
transition_counts = np.zeros((len(ids),len(ids)),dtype=int)
sequence = segments["Label"].to_list()
for a,b in zip(sequence,sequence[1:]):
    transition_counts[index[a],index[b]] += 1
fig, ax = plt.subplots(figsize=(6,5),layout="constrained")
image = ax.imshow(transition_counts,cmap="Blues")
for i in range(len(ids)):
    for j in range(len(ids)):
        if transition_counts[i,j]:
            ax.text(j,i,str(transition_counts[i,j]),ha="center",va="center",fontsize=8,
                    color="white" if transition_counts[i,j] > transition_counts.max()/2 else "black")
ax.set(xticks=range(len(ids)),yticks=range(len(ids)),xticklabels=ids,yticklabels=ids,
       xlabel="Next state",ylabel="Previous state",title="Observed segment transitions")
fig.colorbar(image,ax=ax,label="Transition count")
plt.show()
''')
    add('markdown','''## Machine state is not an anomaly label

The two labeled CSVs contain the same observations with different targets.
State IDs describe operating states; anomaly IDs belong to a separate labeling
scheme. Do not treat state ID 0 as normal and all other state IDs as faults.
''')
    add('code','''display(state.select(pl.col("Label").alias("state_id"))
        .with_columns(anomaly["Label"].alias("anomaly_id"))
        .group_by("state_id","anomaly_id").len().sort("state_id","anomaly_id"))
''')
    add('markdown','''## Implications for dependency-based segmentation

- Start with the five motor signals; inspect their distributions and near-constant
  intervals before fitting a Gaussian graphical model.
- Keep state and anomaly labels outside the input features. Binary control flags
  provide useful interpretation but can make segmentation much easier; evaluate
  their inclusion separately.
- Fit scaling and select hyperparameters using training blocks only. Use blocked
  temporal splits with a gap at least as large as the overlapping window span.
- Compare multi-scale representations against single-scale and mean/variance
  baselines. Report boundary accuracy as well as state agreement.
- Account for timestamp reversals and irregular intervals before making claims
  about physical time scales. Short states may not contain enough effectively
  independent observations to estimate a reliable local precision matrix.
- This notebook explores one recording, not independent experimental replicates.
''')
    path=ROOT/'notebooks/datasets/gdd.ipynb'
    path.write_text(json.dumps(dict(cells=cells,metadata={'kernelspec':{'display_name':'Python (pdmdata)','language':'python','name':'pdmdata'},'language_info':{'name':'python','version':'3.11'}},nbformat=4,nbformat_minor=5),indent=1)+'\n')

if __name__ == '__main__':
    main()
