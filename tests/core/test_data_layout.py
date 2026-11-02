import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "pdmdata"


class DataLayoutTest(unittest.TestCase):
    def test_dataset_directories_do_not_contain_data_files(self):
        data_suffixes = {
            ".csv",
            ".gz",
            ".npz",
            ".pickle",
            ".parquet",
            ".zip",
            ".7z",
            ".rar",
        }
        files = [
            path
            for path in DATASETS.rglob("*")
            if path.is_file() and path.suffix.lower() in data_suffixes
        ]
        self.assertEqual(files, [])

    def test_auxiliary_dataset_json_files_live_under_assets(self):
        misplaced = [
            path.relative_to(ROOT).as_posix()
            for path in (DATASETS / "datasets").rglob("*.json")
            if path.name != "metadata.json" and path.parent.name != "assets"
        ]
        self.assertEqual(misplaced, [])

if __name__ == "__main__":
    unittest.main()
