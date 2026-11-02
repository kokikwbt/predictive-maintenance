"""GFD recording validation and load selectors."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pdmdata
from pdmdata.datasets.gfd import loader


def _write_recording(path: Path, rows: int = 4, value: float = 1.0) -> None:
    # Leading and trailing blank lines match the distributed text format.
    body = "\n".join(
        "\t".join(str(value + i + j * 0.1) for j in range(4))
        for i in range(rows)
    )
    path.write_text("\n\n" + body + "\n\n")


class GfdTest(unittest.TestCase):
    def test_load_casts_channels_and_rejects_bad_options(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "gfd"
            directory.mkdir()
            _write_recording(directory / "h30hz0.txt", rows=5, value=2.0)
            _write_recording(directory / "b30hz20.txt", rows=5, value=9.0)
            with patch(
                "pdmdata.io.paths.get_settings",
                return_value=pdmdata.Settings(data_root=root),
            ):
                healthy = pdmdata.load("gfd", condition="healthy", load=0)
                self.assertEqual(healthy.height, 5)
                self.assertEqual(
                    healthy.columns,
                    [*loader.SENSOR_COLUMNS, "condition", "load"],
                )
                self.assertEqual(healthy["condition"].to_list(), ["healthy"] * 5)
                self.assertEqual(healthy["load"].to_list(), [0] * 5)
                broken = pdmdata.load("gfd", condition="broken", load=20)
                self.assertEqual(broken["condition"].unique().to_list(), ["broken"])
                with self.assertRaisesRegex(ValueError, "healthy or broken"):
                    pdmdata.load("gfd", condition="worn")
                with self.assertRaisesRegex(ValueError, "increments of 10"):
                    pdmdata.load("gfd", load=15)
                (directory / "h30hz0.txt").write_text("\n1\t2\t3\n")
                with self.assertRaisesRegex(ValueError, "four nonempty"):
                    pdmdata.load("gfd", condition="healthy", load=0)


if __name__ == "__main__":
    unittest.main()
