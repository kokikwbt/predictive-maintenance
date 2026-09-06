import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
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

if __name__ == "__main__":
    unittest.main()
