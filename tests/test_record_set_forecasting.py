"""Unordered clocks, causal prefixes, train-only vocabulary and split accounting."""
from io import BytesIO
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pdmdata
from pdmdata.tasks.event_forecasting import RecordSetSpec,prepare_pcic_record_sets,export_pcic_record_sets

ROWS=[('0','a',0),('1','a',5),('2','a',5),('1','a',10),('0','a',40),('1','a',60),('0','a',70),('0','a',75),('9','a',80),('9','a',90),('8','a',90),('9','a',100),('2','b',0),('2','b',50),('8','b',90)]
class RecordSetForecastingTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.raw=self.root/'pcic_alarm/wireless/18V_55N_Wireless.tar.gz';self.raw.parent.mkdir(parents=True);self.write_rows(ROWS);p=patch('pdmdata.io.paths.get_settings',return_value=pdmdata.Settings(self.root));p.start();self.addCleanup(p.stop)
    def write_rows(self,rows):
        raw=('alarm_id,device_id,start_timestamp,end_timestamp\n'+'\n'.join(f'{mark},{node},{tick},future-end-not-an-input' for mark,node,tick in rows)+'\n').encode()
        with tarfile.open(self.raw,'w:gz') as tar:
            member=tarfile.TarInfo('Alarm.csv');member.size=len(raw);tar.addfile(member,BytesIO(raw))
    def test_tied_rows_are_one_set_and_future_tokens_do_not_define_vocabulary(self):
        bundle=prepare_pcic_record_sets(spec=RecordSetSpec(lookback=2));meta=bundle['metadata'];self.assertEqual(meta['marks'],['0','1','2']);self.assertEqual(meta['rows'],dict(train=4,validation=2,test=2));self.assertEqual(meta['excluded_split_crossing_pairs'],3);self.assertEqual(meta['record_groups'],13);self.assertTrue(meta['all_pairs_accounted']);self.assertFalse(meta['end_timestamp_used_as_input']);self.assertFalse(meta['within_tick_order_inferred'])
        train=bundle['arrays']['train'];where=np.flatnonzero((train['origin_tick']==0)&(train['device']==0))[0];np.testing.assert_array_equal(train['target'][where],[0,1,1,0]);self.assertEqual(train['wait_ticks'][where],5)
        where=np.flatnonzero((train['origin_tick']==5)&(train['device']==0))[0];np.testing.assert_array_equal(train['history_marks'][where],[[1,0,0,0],[0,1,1,0]]);np.testing.assert_array_equal(train['history_gaps'][where],[0,5]);self.assertTrue((train['wait_ticks']>0).all())
        test=bundle['arrays']['test'];where=np.flatnonzero(test['target_tick']==90)[0];self.assertEqual(test['raw_target_cardinality'][where],2);self.assertEqual(test['mapped_target_cardinality'][where],1);self.assertTrue(test['other_in_target'][where]);self.assertEqual(test['target_tick'].max(),100)
    def test_permuting_equal_tick_rows_cannot_change_forecast_arrays(self):
        first=prepare_pcic_record_sets(spec=RecordSetSpec(lookback=2))['arrays'];permuted=ROWS.copy();permuted[1],permuted[2]=permuted[2],permuted[1];permuted[9],permuted[10]=permuted[10],permuted[9];self.write_rows(permuted);second=prepare_pcic_record_sets(spec=RecordSetSpec(lookback=2))['arrays']
        for split in first:
            for field in first[split]:np.testing.assert_array_equal(first[split][field],second[split][field])
    def test_future_label_changes_do_not_change_train_or_validation_features(self):
        first=prepare_pcic_record_sets(spec=RecordSetSpec(lookback=2));self.write_rows([(mark if tick<80 else 'future-'+mark,node,tick) for mark,node,tick in ROWS]);second=prepare_pcic_record_sets(spec=RecordSetSpec(lookback=2));self.assertEqual(first['metadata']['marks'],second['metadata']['marks'])
        for split in ['train','validation']:
            for field in first['arrays'][split]:np.testing.assert_array_equal(first['arrays'][split][field],second['arrays'][split][field])
    def test_short_history_and_empty_splits_remain_well_shaped(self):
        bundle=prepare_pcic_record_sets(spec=RecordSetSpec(lookback=2,minimum_history=2));self.assertTrue(bundle['metadata']['all_pairs_accounted']);self.assertEqual(bundle['metadata']['excluded_short_history_pairs'],2)
        self.write_rows([('0','a',0),('1','a',100)]);bundle=prepare_pcic_record_sets(spec=RecordSetSpec(lookback=2));self.assertTrue(bundle['metadata']['all_pairs_accounted'])
        for split in bundle['arrays']:self.assertEqual(bundle['arrays'][split]['target'].shape,(0,2));self.assertEqual(bundle['arrays'][split]['history_marks'].shape,(0,2,2))
    def test_invalid_clock_and_spec_fail_without_jitter_or_silent_reconnection(self):
        for spec in [RecordSetSpec(train_fraction=.9),RecordSetSpec(lookback=0),RecordSetSpec(minimum_history=True),RecordSetSpec(train_fraction=float('nan'))]:
            with self.subTest(spec=spec),self.assertRaises(ValueError):prepare_pcic_record_sets(spec=spec)
        for tick in ['1.5','-1','1_000','']:
            self.write_rows([('0','a',0),('1','a',tick)])
            with self.subTest(tick=tick),self.assertRaises(ValueError):prepare_pcic_record_sets()
    def test_export_keeps_provenance_and_never_opens_reference_values(self):
        with patch('pdmdata.datasets.pcic_alarm.read_reference',side_effect=AssertionError('reference opened')):
            out=self.root/'export';result=export_pcic_record_sets(out,spec=RecordSetSpec(lookback=2))
        meta=json.loads((out/'metadata.json').read_text());self.assertTrue(meta['all_pairs_accounted']);self.assertFalse(meta['reference_values_loaded']);self.assertFalse(json.loads(result['manifest'].read_text())['model_fit']);provenance=json.loads((out/'provenance.json').read_text());self.assertEqual(len(provenance['raw_archive_sha256']),64)
        with self.assertRaises(FileExistsError):export_pcic_record_sets(out)
if __name__=='__main__':unittest.main()
