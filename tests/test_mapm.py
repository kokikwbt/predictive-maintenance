"""MAPM table selection through the common load API."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pdmdata
from pdmdata.datasets.mapm.loader import MAPM_FILES


class MapmTest(unittest.TestCase):
    def test_table_selection_and_unknown_table(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "mapm"
            directory.mkdir()
            (directory / "PdM_telemetry.csv").write_text(
                "datetime,machineID,volt,rotate,pressure,vibration\n"
                "2015-01-01 06:00:00,1,150.0,400.0,100.0,40.0\n"
            )
            (directory / "PdM_errors.csv").write_text(
                "datetime,machineID,errorID\n"
                "2015-01-01 07:00:00,1,error1\n"
            )
            with patch(
                "pdmdata.io.paths.get_settings",
                return_value=pdmdata.Settings(data_root=root),
            ):
                telemetry = pdmdata.load("mapm", table="telemetry")
                self.assertEqual(telemetry.height, 1)
                self.assertEqual(telemetry["machineID"].to_list(), [1])
                errors = pdmdata.load("mapm", table="errors")
                self.assertEqual(errors["errorID"].to_list(), ["error1"])
                self.assertEqual(set(MAPM_FILES), {
                    "telemetry", "errors", "failures", "machines", "maintenance",
                })
                with self.assertRaisesRegex(ValueError, "table must be"):
                    pdmdata.load("mapm", table="unknown")
                with self.assertRaises(FileNotFoundError):
                    pdmdata.load("mapm", table="failures")


if __name__ == "__main__":
    unittest.main()
