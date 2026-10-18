"""OYICD filename parsing, duplicate handling, and load validation."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pdmdata
from pdmdata.datasets.oyicd import loader


def _csv(rows: int = 3) -> str:
    header = "timestamp," + ",".join(loader.SENSOR_COLUMNS)
    body = "\n".join(
        ",".join([str(i * 0.004)] + [str(0.1 * (i + j)) for j in range(8)])
        for i in range(rows)
    )
    return header + "\n" + body + "\n"


class OyicdTest(unittest.TestCase):
    def test_load_inventory_and_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            name = "01-04T184148_000_mode1.csv"
            path = root / "oyicd" / "extracted" / name
            path.parent.mkdir(parents=True)
            path.write_text(_csv(4))
            duplicate = root / "oyicd" / "copy" / name
            duplicate.parent.mkdir(parents=True)
            duplicate.write_bytes(path.read_bytes())
            with patch(
                "pdmdata.datasets.oyicd.loader.get_settings",
                return_value=pdmdata.Settings(data_root=root),
            ):
                frame = pdmdata.load("oyicd", recording=name)
                self.assertEqual(frame.height, 4)
                self.assertEqual(frame["mode"].to_list(), [1] * 4)
                self.assertEqual(frame["month"].to_list(), [1] * 4)
                listed = loader.inventory(mode=1)
                self.assertEqual(listed.height, 1)
                self.assertEqual(listed["copies"].to_list(), [2])
                with self.assertRaisesRegex(ValueError, "OYICD filename"):
                    pdmdata.load("oyicd", recording="bad.csv")
                with self.assertRaisesRegex(ValueError, "month"):
                    loader.inventory(month=0)
                bad = _csv(4).replace("0.004", "0.0", 1)
                path.write_text(bad)
                duplicate.write_text(bad)
                with self.assertRaisesRegex(ValueError, "increase strictly"):
                    pdmdata.load("oyicd", recording=name)
                duplicate.write_text(_csv(5))
                with self.assertRaisesRegex(ValueError, "Conflicting copies"):
                    pdmdata.load("oyicd", recording=name)


if __name__ == "__main__":
    unittest.main()
