"""MetroPT2 lazy/eager loading through the common API."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import polars as pl

import pdmdata

CSV = (
    "timestamp,TP2,TP3,H1,DV_pressure,Reservoirs,Oil_temperature,"
    "Motor_current,COMP,DV_eletric,Towers,MPG,LPS,Pressure_switch,"
    "Oil_level,Caudal_impulses\n"
    "2022-01-01 00:00:00,0.1,0.2,0.3,0.4,0.5,60.0,1.0,1,0,0,0,0,0,0,0\n"
    "2022-01-01 00:00:01,0.1,0.2,0.3,0.4,0.5,60.1,1.1,1,0,0,0,0,0,0,0\n"
)


class Metropt2Test(unittest.TestCase):
    def test_lazy_default_and_eager_collect(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "metropt2"
            directory.mkdir()
            (directory / "MetroPT2.csv").write_text(CSV)
            with patch(
                "pdmdata.io.paths.get_settings",
                return_value=pdmdata.Settings(data_root=root),
            ):
                scan = pdmdata.load("metropt2")
                self.assertIsInstance(scan, pl.LazyFrame)
                frame = scan.collect()
                self.assertEqual(frame.height, 2)
                self.assertIn("Oil_temperature", frame.columns)
                eager = pdmdata.load("metropt2", lazy=False)
                self.assertIsInstance(eager, pl.DataFrame)
                self.assertEqual(eager.height, 2)


if __name__ == "__main__":
    unittest.main()
