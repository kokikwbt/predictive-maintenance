"""PCIC raw token fidelity, reference isolation and archive safety."""
import importlib
from io import BytesIO
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import polars as pl
import pdmdata
from pdmdata.datasets.pcic_alarm import source_inventory, read_reference

CSV = b'Alarm ID,Device ID,Start Timestamp,End Timestamp\n00,7,001.20,2.0\n00,7,001.20,2.0\n9,0,0,\n'

class PcicAlarmTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / 'pcic_alarm/wireless/18V_55N_Wireless.tar.gz'
        self.path.parent.mkdir(parents=True)
        self.settings = patch('pdmdata.io.paths.get_settings', return_value=pdmdata.Settings(self.root))
        self.settings.start(); self.addCleanup(self.settings.stop)
        self.write_archive([('data/Alarm.csv', CSV), ('data/DAG.npy', self.array(np.eye(2)))])
    def array(self, value):
        stream = BytesIO(); np.save(stream, value); return stream.getvalue()
    def write_archive(self, files):
        with tarfile.open(self.path, 'w:gz') as archive:
            for name, raw in files:
                item = tarfile.TarInfo(name); item.size = len(raw); archive.addfile(item, BytesIO(raw))
    def test_raw_tokens_duplicates_and_order_are_preserved(self):
        frame = pdmdata.load('pcic_alarm', lazy=False)
        self.assertEqual(frame.columns, ['source_row','Alarm ID','Device ID','Start Timestamp','End Timestamp'])
        self.assertEqual(frame['Alarm ID'].to_list(), ['00','00','9'])
        self.assertEqual(frame['Start Timestamp'].to_list(), ['001.20','001.20','0'])
        self.assertEqual(frame['End Timestamp'].to_list(), ['2.0','2.0',''])
        self.assertEqual(frame['source_row'].to_list(), [0,1,2])
        self.assertTrue(all(t == pl.String for k,t in frame.schema.items() if k!='source_row'))
        self.assertIsInstance(pdmdata.load('pcic_alarm'), pl.LazyFrame)
    def test_columns_and_invalid_options(self):
        self.assertEqual(pdmdata.load('pcic_alarm',columns=['Alarm ID','source_row'],lazy=False).rows(), [('00',0),('00',1),('9',2)])
        for options in [dict(variant='unknown'),dict(variant=True),dict(lazy=1),dict(columns='Alarm ID'),dict(columns=[]),dict(columns=['x']),dict(columns=['Alarm ID','Alarm ID']),dict(columns=[[]])]:
            with self.subTest(options=options), self.assertRaises((ValueError,TypeError)):
                pdmdata.load('pcic_alarm',**options)
    def test_graph_is_separate_and_numeric_only(self):
        with patch('pdmdata.datasets.pcic_alarm.loader.np.load', side_effect=AssertionError('graph opened')):
            self.assertEqual(pdmdata.load('pcic_alarm',lazy=False).height,3)
        np.testing.assert_array_equal(read_reference(),np.eye(2))
        self.write_archive([('Alarm.csv',CSV),('DAG.npy',self.array(np.array([object()],dtype=object)))])
        with self.assertRaises(ValueError): read_reference()
        self.write_archive([('Alarm.csv',CSV),('DAG.npy',self.array(np.array([1,2])))])
        with self.assertRaises(ValueError): read_reference()
        with self.assertRaises(ValueError): read_reference(member='../DAG.npy')
    def test_source_inventory_hashes_and_headers(self):
        import hashlib
        result = source_inventory()
        self.assertEqual(result[0]['header'],['Alarm ID','Device ID','Start Timestamp','End Timestamp'])
        self.assertEqual(result[0]['sha256'],hashlib.sha256(CSV).hexdigest())
        self.assertEqual(result[0]['bytes'],len(CSV))
    def test_ambiguous_and_unsafe_members_are_rejected(self):
        self.write_archive([('a/Alarm.csv',CSV),('b/Alarm.csv',CSV)])
        with self.assertRaisesRegex(ValueError,'Expected one'): pdmdata.load('pcic_alarm')
        for name in ['../Alarm.csv','/Alarm.csv','bad\\Alarm.csv']:
            self.write_archive([(name,CSV)])
            with self.subTest(name=name), self.assertRaisesRegex(ValueError,'Unsafe'): pdmdata.load('pcic_alarm')
        with tarfile.open(self.path,'w:gz') as archive:
            item=tarfile.TarInfo('Alarm.csv');item.type=tarfile.SYMTYPE;item.linkname='/tmp/foreign';archive.addfile(item)
        with self.assertRaisesRegex(ValueError,'links'): pdmdata.load('pcic_alarm')
    def test_variants_are_pinned_and_not_bulk_downloaded(self):
        info=pdmdata.load_metadata('pcic_alarm');revision='1858e9c39d90a524e7d8ff8b9456f4b1f231ff03'
        self.assertEqual(pdmdata.download_variants('pcic_alarm'),['microwave439','microwave474','wireless'])
        self.assertNotIn('pcic_alarm',pdmdata.bulk_downloads())
        for name,resource in info['download']['variants'].items():
            self.assertIn(revision,resource['url']);self.assertEqual(resource['archive'],'file')
            self.assertIn(resource['filename'],pdmdata.download_command('pcic_alarm',self.root,variant=name))
        module=importlib.import_module('pdmdata.io.download')
        with patch.object(module,'_download_file',side_effect=lambda url,target:target.write_bytes(b'opaque archive')):
            result=pdmdata.download('pcic_alarm',self.root,variant='microwave439')
        import json
        self.assertEqual(json.loads(result['manifest'].read_text())['variant'],'microwave439')
        self.assertTrue((self.root/'pcic_alarm/microwave439/24V_439N_Microwave.tar.gz').is_file())
        self.assertFalse((self.root/'pcic_alarm/microwave439/extracted').exists())
if __name__=='__main__':unittest.main()
