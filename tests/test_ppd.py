"""PPD sequence joining, provenance, and option validation."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pdmdata
from pdmdata.datasets.ppd import loader


def _header() -> str:
    return "Timestamp," + ",".join(loader.SENSOR_COLUMNS)


def _row(index: int, value: float = 1.0) -> str:
    return ",".join([str(index)] + [str(value)] * len(loader.SENSOR_COLUMNS))


class PpdTest(unittest.TestCase):
    def test_sequence_parts_join_and_reject_bad_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "ppd"
            directory.mkdir()
            (directory / "C7-1.csv").write_text(
                _header() + "\n" + "\n".join(_row(i) for i in range(3)) + "\n"
            )
            (directory / "C7-2.csv").write_text(
                _header() + "\n" + "\n".join(_row(i, 2.0) for i in range(2)) + "\n"
            )
            (directory / "C8.csv").write_text(
                _header() + "\n" + _row(0, 3.0) + "\n" + _row(1, 3.0) + "\n"
            )
            with patch(
                "pdmdata.io.paths.get_settings",
                return_value=pdmdata.Settings(data_root=root),
            ):
                frame = pdmdata.load("ppd", sequence_id=7)
                self.assertEqual(frame.height, 5)
                self.assertEqual(frame["sample"].to_list(), list(range(5)))
                self.assertEqual(
                    frame["source_file"].unique(maintain_order=True).to_list(),
                    ["C7-1.csv", "C7-2.csv"],
                )
                self.assertEqual(frame["sequence_id"].unique().to_list(), [7])
                # Timestamp resets per part; sample stays continuous.
                self.assertEqual(frame["Timestamp"].to_list(), [0, 1, 2, 0, 1])
                single = pdmdata.load("ppd", sequence_id="C8")
                self.assertEqual(single.height, 2)
                with self.assertRaisesRegex(ValueError, "sequence_id"):
                    pdmdata.load("ppd", sequence_id=99)
                with self.assertRaisesRegex(ValueError, "experiment must be"):
                    loader.load_file("C99")


if __name__ == "__main__":
    unittest.main()
