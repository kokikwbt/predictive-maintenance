"""IMS selectors, chronology, channel assignments, and extraction regressions."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
import importlib

import polars as pl
import libarchive

from pdmdata import Settings
from pdmdata.ims import channel_info, inventory, load, rms_history, verify
from pdmdata.ims import loader
from pdmdata.download import _extract_nested_archives


class ImsTest(TestCase):
    def test_experiment_selection_chronology_and_matrix_integrity(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for experiment, folder in loader.EXPERIMENT_DIRS.items():
                directory = root / "ims/extracted" / folder
                if experiment == 3:
                    directory /= "4th_test/txt"
                directory.mkdir(parents=True)
                for minute, value in [(10, 3), (0, 1)]:
                    name = f"2004.02.12.10.{minute:02d}.00"
                    row = "\t".join([str(value)] * loader.CHANNEL_COUNTS[experiment])
                    (directory / name).write_text((row + "\n") * 2)
            with patch.object(loader, "get_settings", return_value=Settings(data_root=root)), \
                 patch.object(loader, "SAMPLES_PER_RECORDING", 2), \
                 patch("pdmdata.ims.validation.EXPECTED_RECORDINGS", {1: 2, 2: 2, 3: 2}):
                self.assertEqual(inventory().height, 6)
                self.assertEqual(inventory(2)["recording_index"].to_list(), [0, 1])
                frame = load(0, experiment=1)
                self.assertEqual(len([c for c in frame.columns if c.startswith("channel_")]), 8)
                self.assertEqual(frame["time_s"].to_list(), [0, 1 / 20_000])
                self.assertEqual(load(1)["channel_1"].to_list(), [3, 3])
                self.assertEqual(load(0, experiment=3, channels=["channel_3"]).width, 6)
                history = rms_history(2, every=5)
                self.assertEqual(history["channel_1"].to_list(), [1, 3])
                self.assertEqual(verify()["recordings"].to_list(), [2, 2, 2])
                with self.assertRaisesRegex(ValueError, "matching"):
                    load("2004.02.12.10.00.00")
                for options in ({"recording": -1}, {"experiment": 4}, {"channels": ["channel_5"]}):
                    with self.subTest(options=options), self.assertRaises(ValueError):
                        load(**options)
                path = Path(inventory(2)["path"][0])
                for contents in ["1\t2\t3\t4\n", "NaN\t2\t3\t4\n" * 2, "1\t\t3\t4\n" * 2]:
                    path.write_text(contents)
                    with self.assertRaises(ValueError):
                        load()
            self.assertEqual(channel_info(1)["bearing"].to_list(), [1, 1, 2, 2, 3, 3, 4, 4])
            self.assertEqual(channel_info(2).filter(pl.col("channel") == "channel_1")["end_of_test_fault"][0], "outer race")

    def test_nested_7z_extraction_reuse_and_interruption(self):
        module = importlib.import_module("pdmdata.download")
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            archive = directory / "IMS.7z"
            with libarchive.file_writer(str(archive), "7zip") as packed:
                packed.add_file_from_memory("2nd_test/2004.02.12.10.32.39", 4, b"1\t2\n")
            with patch.object(module, "_extract_native_archive", side_effect=OSError("interrupted")):
                with self.assertRaises(OSError):
                    _extract_nested_archives(directory)
            self.assertFalse((directory / "IMS").exists())
            _extract_nested_archives(directory)
            self.assertEqual((directory / "IMS/2nd_test/2004.02.12.10.32.39").read_text(), "1\t2\n")
            with patch.object(module, "_extract_archive") as extract:
                _extract_nested_archives(directory)
                extract.assert_not_called()
            unsafe = directory / "unsafe.7z"
            with libarchive.file_writer(str(unsafe), "7zip") as packed:
                packed.add_file_from_memory("../escaped.txt", 1, b"x")
            with self.assertRaisesRegex(ValueError, "Unsafe archive member"):
                _extract_nested_archives(directory)
            self.assertFalse((directory / "unsafe").exists())
            self.assertFalse((directory / "escaped.txt").exists())
