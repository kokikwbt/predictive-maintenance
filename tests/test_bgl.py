"""Literal source fidelity and streaming checks for the BGL adapter."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import zipfile

from pdmdata import load as public_load
from pdmdata.datasets.bgl import load, iter_batches


class BglTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / 'raw' / 'bgl'
        self.root.mkdir(parents=True)
        config = Path(temporary.name) / 'pdmdata.toml'
        config.write_text('data_root = "' + str(self.root.parent) + '"\n')
        env = patch.dict(os.environ, {'PDMDATA_CONFIG': str(config)})
        env.start()
        self.addCleanup(env.stop)
        self.payload = (b'- 1 2005.01.01 N0 2005-01-01-00.00.01.000001 N0 RAS KERNEL INFO same  message\r\n'
                        b'A 1 2005.01.01 N1 2005-01-01-00.00.01.000001 N1 RAS LINK ERROR bad\n'
                        b'- 1 2005.01.01 N0 2005-01-01-00.00.01.000001 N0 RAS KERNEL INFO same  message\r\n'
                        b'bad header\n')
        with zipfile.ZipFile(self.root / 'BGL.zip', 'w') as archive:
            archive.writestr('BGL.log', self.payload)

    def test_raw_reconstruction_duplicates_and_malformed(self):
        frame = load()
        self.assertEqual(''.join(frame['raw_line']).encode(), self.payload)
        self.assertEqual(frame['source_row'].to_list(), [0, 1, 2, 3])
        self.assertEqual(frame['parse_ok'].to_list(), [True, True, True, False])
        self.assertEqual(frame['content'][0], 'same  message')
        self.assertEqual(frame['node'].to_list(), ['N0', 'N1', 'N0', None])

    def test_batch_boundaries_and_public_dispatch(self):
        batches = list(iter_batches(batch_rows=3))
        self.assertEqual([v.height for v in batches], [3, 1])
        self.assertEqual(batches[1]['source_row'].to_list(), [3])
        frame = public_load('bgl', rows=2, columns=['source_row', 'label'])
        self.assertEqual(frame['label'].to_list(), ['-', 'A'])

    def test_zero_rows_and_archive_precedence(self):
        self.assertEqual(load(rows=0).height, 0)
        (self.root / 'BGL.log').write_bytes(b'other\n')
        self.assertEqual(load(rows=1)['raw_line'][0].encode(), self.payload.splitlines(keepends=True)[0])
        (self.root / 'BGL.zip').unlink()
        self.assertEqual(load()['raw_line'].to_list(), ['other\n'])
        (self.root / 'extracted').mkdir()
        (self.root / 'BGL.log').rename(self.root / 'extracted/BGL.log')
        self.assertEqual(load()['source_file'].to_list(), ['extracted/BGL.log'])

    def test_complete_header_with_empty_message(self):
        payload = b'- 1 2005.01.01 N0 2005-01-01-00.00.01.000001 N0 RAS KERNEL FATAL\n'
        with zipfile.ZipFile(self.root / 'BGL.zip', 'w') as archive:
            archive.writestr('BGL.log', payload)
        frame = load()
        self.assertTrue(frame['parse_ok'][0])
        self.assertEqual(frame['content'][0], '')
        self.assertEqual(frame['node'][0], 'N0')
        self.assertEqual(frame['raw_line'][0].encode(), payload)

    def test_invalid_options(self):
        for kwargs in [{'rows': -1}, {'rows': True}, {'columns': 'node'}, {'columns': []}, {'columns': ['node', 'node']}, {'columns': ['unknown']}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                load(**kwargs)
        for batch_rows in [0, -1, True]:
            with self.subTest(batch_rows=batch_rows), self.assertRaises(ValueError):
                list(iter_batches(batch_rows=batch_rows))

    def test_missing_source_and_member(self):
        (self.root / 'BGL.zip').unlink()
        with self.assertRaises(FileNotFoundError):
            load()
        with zipfile.ZipFile(self.root / 'BGL.zip', 'w') as archive:
            archive.writestr('unexpected', b'bad')
        with self.assertRaises(ValueError):
            load()

    def test_invalid_utf8_raises(self):
        with zipfile.ZipFile(self.root / 'BGL.zip', 'w') as archive:
            archive.writestr('BGL.log', b'\xff\n')
        with self.assertRaises(UnicodeDecodeError):
            load()


if __name__ == '__main__':
    unittest.main()
