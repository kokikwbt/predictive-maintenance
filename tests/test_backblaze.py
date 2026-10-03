import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from datetime import date, timedelta
from zipfile import ZipFile

import polars as pl
import pdmdata
from pdmdata.datasets.backblaze import inventory, verify
from pdmdata.datasets.backblaze.viz import plot_waveforms


class BackblazeTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        config = self.root / 'config.toml'
        config.write_text('data_root = "raw"\n')
        env = patch.dict(os.environ, {'PDMDATA_CONFIG': str(config)})
        env.start()
        self.addCleanup(env.stop)
        self.directory = self.root / 'raw/backblaze/2025-q1/extracted'
        self.directory.mkdir(parents=True)
        self.header = 'date,serial_number,model,capacity_bytes,failure,smart_5_raw\n'

    def test_selection_sparse_schemas_and_raw_preservation(self):
        a = self.directory / '2025-01-01.csv'
        a.write_text(self.header + '2025-01-01,0001,M,100,0,\n2025-01-01,0002,N,200,0,2\n')
        b = self.directory / '2025-01-02.csv'
        b.write_text(self.header.rstrip() + ',smart_9_raw\n2025-01-02,0001,M,100,1,3,9\n')
        (self.directory / 'notes.csv').write_text('not a snapshot')
        before = {p:p.read_bytes() for p in self.directory.iterdir()}
        with patch('urllib.request.urlopen', side_effect=AssertionError('network')):
            frame = pdmdata.load('backblaze', serial_number='0001', model='M',
                                 columns=['date','serial_number','failure','smart_5_raw','smart_9_raw'])
            self.assertIsInstance(frame, pl.LazyFrame)
            result = frame.collect()
            self.assertEqual(result['smart_5_raw'].to_list(), [None, 3.0])
            self.assertEqual(result['smart_9_raw'].to_list(), [None, 9.0])
            self.assertEqual(result['serial_number'].to_list(), ['0001', '0001'])
            self.assertEqual(inventory().height, 2)
            a.write_text('invalid excluded file')
            selected = pdmdata.load('backblaze', start_date='2025-01-02', end_date='2025-01-02', lazy=False)
            self.assertEqual(selected.height, 1)
            a.write_bytes(before[a])
        self.assertEqual(before, {p:p.read_bytes() for p in self.directory.iterdir()})
        figure = plot_waveforms(frame, serial_number='0001', columns=['smart_5_raw'])
        self.assertEqual(len(figure.axes[0].lines[0].get_xdata()), 2)
        self.assertEqual(len(figure.axes[0].lines), 2)
        import matplotlib.pyplot as plt
        plt.close(figure)

    def test_invalid_selections_and_missing_files(self):
        for options in ({'variant':'../other'}, {'start_date':'bad'},
                        {'start_date':'2025-02-01','end_date':'2025-01-01'},
                        {'columns':[]}, {'serial_number':''}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                pdmdata.load('backblaze', **options)
        with self.assertRaises(FileNotFoundError):
            pdmdata.load('backblaze')

    def test_quarter_coverage_content_and_archive_integrity(self):
        with self.assertRaises(FileNotFoundError):
            verify()
        for i in range(90):
            day = date(2025,1,1) + timedelta(days=i)
            (self.directory / f'{day}.csv').write_text(self.header + f'{day},A,M,100,0,1\n')
        archive = self.directory.parent / 'data_Q1_2025.zip'
        with ZipFile(archive,'w') as zipped:
            for p in self.directory.iterdir():zipped.write(p, p.name)
        self.assertEqual(verify(check_archive=True)['rows'],90)
        p = self.directory / '2025-01-01.csv'
        original = p.read_bytes()
        p.write_bytes(original.replace(b',0,1',b',0,2'))
        with self.assertRaisesRegex(ValueError,'integrity'):verify(check_archive=True)
        p.write_bytes(original.replace(b',0,1',b',2,1'))
        with self.assertRaisesRegex(ValueError,'failure'):verify()
        p.unlink()
        with self.assertRaisesRegex(ValueError,'coverage'):verify()
