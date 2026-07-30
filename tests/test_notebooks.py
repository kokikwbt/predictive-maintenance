import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


class NotebookStructureTest(unittest.TestCase):
    TASK_NOTEBOOKS = {
        "anomaly-detection.ipynb": "gfd",
        "condition-estimation.ipynb": "cbm",
        "event-sequence-forecasting.ipynb": "mapm",
        "fault-classification.ipynb": "gfd",
        "operating-state-classification.ipynb": "oyicd",
        "remaining-useful-life-prediction.ipynb": "cmapss",
        "survival-analysis.ipynb": "cmapss",
        "time-to-event-prediction.ipynb": "cmapss",
    }

    def test_dataset_notebooks_contain_visualization_showcase(self):
        self.assertEqual(
            [
                path.name
                for path in sorted((NOTEBOOKS / "datasets").glob("*.ipynb"))
            ],
            ["visualization-showcase.ipynb"],
        )

    def test_all_notebooks_are_valid_json(self):
        for path in NOTEBOOKS.rglob("*.ipynb"):
            with self.subTest(path=path):
                document = json.loads(path.read_text(encoding="utf-8"))
                self.assertIn("cells", document)
                self.assertIn("metadata", document)
                for index, cell in enumerate(document["cells"]):
                    if cell.get("cell_type") == "code":
                        compile(
                            "".join(cell.get("source", [])),
                            "{}:cell-{}".format(path, index),
                            "exec",
                        )

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

    def test_supported_tasks_have_executable_notebooks(self):
        paths = {
            path.name: path for path in (NOTEBOOKS / "tasks").glob("*.ipynb")
        }
        self.assertEqual(set(paths), set(self.TASK_NOTEBOOKS))
        for filename, dataset_id in self.TASK_NOTEBOOKS.items():
            document = json.loads(paths[filename].read_text(encoding="utf-8"))
            source = "\n".join(
                "".join(cell.get("source", [])) for cell in document["cells"]
            )
            self.assertIn('datasets.download("{}")'.format(dataset_id), source)
            self.assertIn("import polars as pl", source)

    def test_task_notebooks_do_not_claim_maintenance_policy_validation(self):
        self.assertNotIn(
            "maintenance-policy-evaluation.ipynb",
            {
                path.name
                for path in (NOTEBOOKS / "tasks").glob("*.ipynb")
            },
        )


if __name__ == "__main__":
    unittest.main()
