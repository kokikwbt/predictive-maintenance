"""Construct completed next-recorded-tick/set samples from PCIC source tables."""
from dataclasses import asdict,dataclass
from fractions import Fraction
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone
import importlib.metadata
import numpy as np
import polars as pl
from pdmdata.datasets.pcic_alarm import load,source_inventory
from pdmdata.io import find_raw_file
from pdmdata.catalog import load_metadata

@dataclass(frozen=True)
class RecordSetSpec:
    train_fraction: float = .6
    validation_end_fraction: float = .8
    lookback: int = 16
    minimum_history: int = 1

    def validate(self):
        for name in ['train_fraction','validation_end_fraction']:
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value):
                raise ValueError(f'{name} must be a finite number')
        if not 0<self.train_fraction<self.validation_end_fraction<1:
            raise ValueError('fractions must satisfy 0 < train < validation end < 1')
        if type(self.lookback) is not int or type(self.minimum_history) is not int or not 1<=self.minimum_history<=self.lookback<=1024:
            raise ValueError('history sizes must satisfy 1 <= minimum_history <= lookback <= 1024')

def _cut(left,right,fraction):
    f=Fraction(str(fraction));return left*f.denominator+(right-left)*f.numerator,f.denominator

def prepare_pcic_record_sets(*,variant='wireless',spec=RecordSetSpec()):
    """Build fixed-calendar completed pairs; end_timestamp is never an input.

    A recorded event is the unordered set of raw alarm tokens on one device at
    one integer start tick. Native tick units are unspecified. This is not a
    physical failure, onset, resolution or censoring task. Source groups retain
    every original row index. An unknown-token presence bit represents OTHER.
    """
    if not isinstance(spec,RecordSetSpec):raise ValueError('spec must be RecordSetSpec')
    spec.validate();raw=load(variant=variant,lazy=False);required=['source_row','alarm_id','device_id','start_timestamp','end_timestamp']
    if raw.columns!=required:raise ValueError('PCIC requires the verified original four-column schema')
    if not raw['start_timestamp'].str.contains(r'^[0-9]+$').all():raise ValueError('start_timestamp must contain nonnegative integer tokens')
    ticks=raw['start_timestamp'].cast(pl.Int64,strict=True)
    if not len(raw) or ticks.max()>=2**62 or ticks.min()==ticks.max():raise ValueError('source must have a nonconstant safe integer clock')
    frame=raw.with_columns(ticks.alias('tick'));left,right=int(ticks.min()),int(ticks.max());cut1,den1=_cut(left,right,spec.train_fraction);cut2,den2=_cut(left,right,spec.validation_end_fraction)
    if right>np.iinfo(np.int64).max//max(den1,den2):raise ValueError('integer clock and fraction denominator exceed safe comparison range')
    marks=sorted(frame.filter(pl.col('tick')*den1<cut1)['alarm_id'].unique().to_list())
    if not marks:raise ValueError('training interval has no alarm tokens')
    mapping={v:i for i,v in enumerate(marks)};other=len(marks);k=other+1;nodes=sorted(raw['device_id'].unique().to_list());node_map={v:i for i,v in enumerate(nodes)}
    grouped=frame.group_by('device_id','tick').agg(pl.col('alarm_id').unique().sort().alias('marks'),pl.col('source_row').sort().alias('source_rows')).sort('device_id','tick').with_row_index('group_id');group_rows=grouped.to_dicts();bits=np.zeros((len(grouped),k),dtype=np.uint8)
    for i,row in enumerate(group_rows):
        for mark in row['marks']:bits[i,mapping.get(mark,other)]=1
    fields=['history_marks','history_length','history_gaps','history_group','target','wait_ticks','origin_tick','target_tick','device','origin_group','target_group','raw_target_cardinality','mapped_target_cardinality','other_in_target'];samples={s:{v:[] for v in fields} for s in ['train','validation','test']};by_node={node:[] for node in nodes}
    for i,row in enumerate(group_rows):by_node[row['device_id']].append(i)
    excluded_crossing=excluded_short=0
    for node,indices in by_node.items():
        for pos in range(len(indices)-1):
            origin,target=indices[pos],indices[pos+1];t0,t1=group_rows[origin]['tick'],group_rows[target]['tick'];assert t1>t0
            if t0*den1<cut1 and t1*den1<cut1:split='train'
            elif t0*den1>=cut1 and t1*den2<cut2:split='validation'
            elif t0*den2>=cut2 and t1<=right:split='test'
            else:excluded_crossing+=1;continue
            if pos+1<spec.minimum_history:excluded_short+=1;continue
            retained=indices[max(0,pos-spec.lookback+1):pos+1];length=len(retained);h=np.zeros((spec.lookback,k),dtype=np.uint8);h[:length]=bits[retained];hg=np.full(spec.lookback,-1,dtype=np.int64);hg[:length]=retained;gaps=np.zeros(spec.lookback,dtype=np.int64)
            if length>1:gaps[1:length]=np.diff([group_rows[v]['tick'] for v in retained])
            row=dict(history_marks=h,history_length=length,history_gaps=gaps,history_group=hg,target=bits[target],wait_ticks=t1-t0,origin_tick=t0,target_tick=t1,device=node_map[node],origin_group=origin,target_group=target,raw_target_cardinality=len(group_rows[target]['marks']),mapped_target_cardinality=int(bits[target].sum()),other_in_target=bool(bits[target,other]))
            for name,value in row.items():samples[split][name].append(value)
    arrays={}
    for split,data in samples.items():
        n=len(data['target']);arrays[split]={}
        for name,values in data.items():
            dtype=np.uint8 if name in ['history_marks','target'] else bool if name=='other_in_target' else np.int64
            shape=(n,spec.lookback,k) if name=='history_marks' else (n,spec.lookback) if name in ['history_gaps','history_group'] else (n,k) if name=='target' else (n,)
            arrays[split][name]=np.asarray(values,dtype=dtype).reshape(shape)
    metadata=dict(variant=variant,spec=asdict(spec),clock='nonnegative recorded integer ticks; unit unspecified',raw_rows=len(raw),record_groups=len(grouped),nodes=nodes,marks=marks,other_index=other,k=k,bounds=dict(first_recorded_tick=left,last_recorded_tick=right,train_cut_numerator=cut1,train_cut_denominator=den1,validation_cut_numerator=cut2,validation_cut_denominator=den2),rows={s:len(v['target']) for s,v in arrays.items()},excluded_split_crossing_pairs=excluded_crossing,excluded_short_history_pairs=excluded_short,last_group_per_device_without_target=len(nodes),all_pairs_accounted=sum(len(v['target']) for v in arrays.values())+excluded_crossing+excluded_short+len(nodes)==len(grouped),zero_wait_samples=0,within_tick_order_inferred=False,end_timestamp_used_as_input=False,reference_values_loaded=False,other_semantics='At least one raw alarm token absent from the training vocabulary; multiple unknown tokens merge into one bit.',first_retained_history_gap_zero=True,test_includes_last_recorded_target=True)
    return dict(groups=grouped,arrays=arrays,metadata=metadata)

def export_pcic_record_sets(output,*,variant='wireless',spec=RecordSetSpec()):
    """Export source groups and model-independent NPZ splits into a new directory."""
    output=Path(output);output.mkdir(parents=True,exist_ok=False);bundle=prepare_pcic_record_sets(variant=variant,spec=spec);bundle['groups'].write_parquet(output/'source-groups.parquet',compression='zstd')
    for split,data in bundle['arrays'].items():np.savez_compressed(output/f'{split}.npz',**data)
    root=Path(__file__).resolve().parents[3];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();files=[Path(__file__),Path(__file__).with_name('__init__.py'),root/'pdmdata/datasets/pcic_alarm/loader.py',root/'pdmdata/datasets/pcic_alarm/metadata.json'];resource=load_metadata('pcic_alarm')['download']['variants'][variant];archive=find_raw_file('pcic_alarm',resource['filename']);inventory=source_inventory(variant=variant)
    provenance=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True)),source_sha256={str(p.relative_to(root)):sha(p) for p in files},raw_archive_path=str(archive),raw_archive_sha256=sha(archive),archive_inventory=inventory,command=[sys.executable,*sys.argv],environment=dict(python=sys.version,platform=platform.platform(),packages={v:importlib.metadata.version(v) for v in ['numpy','polars']}))
    for name,value in [('metadata.json',bundle['metadata']),('provenance.json',provenance)]: (output/name).write_text(json.dumps(value,indent=2)+'\n')
    manifest=dict(outputs={p.name:sha(p) for p in output.iterdir() if p.is_file()},model_fit=False,reference_values_loaded=False);(output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return dict(output=output,metadata=bundle['metadata'],manifest=output/'manifest.json')
