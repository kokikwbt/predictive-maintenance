"""Source fidelity and selectors for the TiNA candidate adapter."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import polars as pl
from pdmdata.datasets.tina import load


class TinaTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.raw = Path(temporary.name) / 'raw'
        for day, ticks in [('2017-09-06', [100, 100, 102]), ('2017-09-07', [104, 106, 108])]:
            folder = self.raw / 'tina' / ('day=' + day)
            folder.mkdir(parents=True)
            pl.DataFrame({'unix': ticks, 'alarms': ['none', 'A1', 'A1'], 'm_id': ['none'] * 3, 'm_subid': ['none'] * 3, 'FEATURE1': [0., .5, 1.]}).write_parquet(folder / 'data.parquet')
        config = Path(temporary.name) / 'pdmdata.toml'
        config.write_text('data_root = "' + str(self.raw) + '"\n')
        env = patch.dict(os.environ, {'PDMDATA_CONFIG': str(config)})
        env.start()
        self.addCleanup(env.stop)

    def test_preserve_source_rows_and_labels(self):
        frame = load(days=['2017-09-07', '2017-09-06'], lazy=False)
        self.assertEqual(frame['unix'].to_list(), [100, 100, 102, 104, 106, 108])
        self.assertEqual(frame['alarms'].to_list(), ['none', 'A1', 'A1'] * 2)
        self.assertEqual(frame['source_row'].to_list(), [0, 1, 2] * 2)
        self.assertEqual(frame['source_file'].to_list(), ['day=2017-09-06/data.parquet'] * 3 + ['day=2017-09-07/data.parquet'] * 3)
        self.assertNotIn('day', frame.columns)

    def test_lazy_projection_and_installed_default(self):
        frame = load(columns=['source_row', 'alarms'])
        self.assertIsInstance(frame, pl.LazyFrame)
        self.assertEqual(frame.collect().columns, ['source_row', 'alarms'])
        self.assertEqual(frame.select(pl.len()).collect().item(), 6)

    def test_reject_invalid_dates(self):
        for days in [[], ['2017-09-06'] * 2, ['20170906'], [True], [{}], 1]:
            with self.subTest(days=days), self.assertRaises(ValueError):
                load(days=days)

    def test_missing_days_are_not_filled(self):
        with self.assertRaises(FileNotFoundError):
            load(days='2017-09-08')
        self.assertFalse((self.raw / 'tina/day=2017-09-08').exists())

    def test_invalid_options(self):
        for options in [{'lazy': 1}, {'columns': 'unix'}, {'columns': []}, {'columns': ['unix', 'unix']}, {'columns': ['unknown']}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                load(**options)


if __name__ == '__main__':
    unittest.main()
