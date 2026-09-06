"""HYDSYS cycle orientation, native rates, and target alignment."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pdmdata
from pdmdata.hydsys import loader


class HydsysTest(unittest.TestCase):
    def test_native_rate_cycles_match_matrix_rows_and_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "hydsys"
            directory.mkdir()
            for sensor, (rate, _) in loader.SENSOR_INFO.items():
                width = rate * 60
                rows = ["\t".join(str(offset + i) for i in range(width)) for offset in [0, 10000]]
                (directory / f"{sensor}.txt").write_text("\n".join(rows) + "\n")
            (directory / "profile.txt").write_text("3\t100\t0\t130\t1\n100\t73\t2\t90\t0\n")
            with patch("pdmdata.io.get_settings", return_value=pdmdata.Settings(data_root=root)), patch.object(loader, "CYCLE_COUNT", 2):
                matrix = loader.load("PS1")
                self.assertEqual(matrix.shape, (2, 6000))
                groups = loader.load_cycle(1)
                self.assertEqual({hz: frame.shape for hz, frame in groups.items()},
                                 {100: (6000, 9), 10: (600, 4), 1: (60, 10)})
                self.assertEqual(groups[100]["PS1"].to_list(), list(matrix.row(1)))
                self.assertEqual(groups[10]["FS1"][0], 10000)
                for rate, frame in groups.items():
                    self.assertAlmostEqual(frame["time_s"][-1], 60 - 1/rate)
                labels = loader.profile()
                self.assertEqual(labels["cycle"].to_list(), [0, 1])
                self.assertEqual(labels["stable_flag"].to_list(), [1, 0])
                self.assertEqual(labels["valve_condition"].to_list(), [100, 73])
                inventory = loader.inventory()
                self.assertEqual(inventory.height, 17)
                self.assertEqual(inventory["measurements"].sum(), 2 * 43680)
                with self.assertRaisesRegex(ValueError, "zero-based"):
                    loader.load_cycle(2)
                with self.assertRaisesRegex(ValueError, "once"):
                    loader.load_cycle(0, sensors=["PS1", "PS1.txt"])
                path = directory / "TS1.txt"
                path.write_text(path.read_text().replace("10000", "NaN", 1))
                with self.assertRaisesRegex(ValueError, "non-finite"):
                    loader.load("TS1", cycle=1)


if __name__ == "__main__":
    unittest.main()
