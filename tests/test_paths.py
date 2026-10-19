import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pdmdata.datasets.care import load as load_care
from pdmdata.io import paths


class FindRawFileTest(unittest.TestCase):
    def test_lookup_stays_inside_the_dataset_directory(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            nested = root / "care" / "Wind Farm A" / "datasets"
            nested.mkdir(parents=True)
            (nested / "0.csv").write_text("time_stamp\n1\n")
            (root / "outside.csv").write_text("secret\n1\n")
            with patch.object(paths, "get_settings") as settings:
                settings.return_value.data_root = root
                self.assertEqual(
                    paths.find_raw_file("care", "Wind Farm A/datasets/0.csv"),
                    nested / "0.csv",
                )
                for name in ("../outside.csv", str(root / "outside.csv"), ""):
                    with self.subTest(name=name), self.assertRaises(ValueError):
                        paths.find_raw_file("care", name)
                with self.assertRaises(ValueError):
                    load_care(recording="../outside.csv")


if __name__ == "__main__":
    unittest.main()
