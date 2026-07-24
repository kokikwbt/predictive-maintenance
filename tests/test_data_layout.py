import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"


class DataLayoutTest(unittest.TestCase):
    def test_dataset_directories_do_not_contain_data_files(self):
        data_suffixes = {
            ".csv",
            ".gz",
            ".npz",
            ".pickle",
            ".parquet",
            ".zip",
        }
        files = [
            path
            for path in DATASETS.rglob("*")
            if path.is_file() and path.suffix.lower() in data_suffixes
        ]
        self.assertEqual(files, [])

    def test_data_processing_does_not_reference_pandas(self):
        checked = [
            *DATASETS.rglob("*.py"),
            *(ROOT / "notebooks").rglob("*.ipynb"),
            ROOT / "requirements.txt",
        ]
        references = [
            path
            for path in checked
            if "pandas" in path.read_text(encoding="utf-8").lower()
        ]
        self.assertEqual(references, [])


if __name__ == "__main__":
    unittest.main()
