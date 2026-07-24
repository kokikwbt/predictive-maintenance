import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


class NotebookStructureTest(unittest.TestCase):
    def test_dataset_notebooks_are_removed_pending_rebuild(self):
        self.assertEqual(list((NOTEBOOKS / "datasets").glob("*.ipynb")), [])

    def test_all_notebooks_are_valid_json(self):
        for path in NOTEBOOKS.rglob("*.ipynb"):
            with self.subTest(path=path):
                document = json.loads(path.read_text(encoding="utf-8"))
                self.assertIn("cells", document)
                self.assertIn("metadata", document)

    def test_legacy_top_level_notebooks_are_removed(self):
        self.assertEqual(list(NOTEBOOKS.glob("*.ipynb")), [])

    def test_time_to_event_notebook_initializes_supported_data(self):
        path = NOTEBOOKS / "tasks" / "time-to-event-prediction.ipynb"
        document = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in document["cells"]
        )
        self.assertIn('datasets.download("cmapss")', source)
        self.assertNotIn("datasets.mapm", source)


if __name__ == "__main__":
    unittest.main()
