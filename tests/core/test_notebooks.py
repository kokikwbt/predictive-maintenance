import ast
import json
import re
import unittest
from importlib import import_module
from inspect import signature
from pathlib import Path

from pdmdata.catalog import load_catalog

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = ROOT / "notebooks"


class NotebookStructureTest(unittest.TestCase):
    def test_all_notebooks_are_valid_json(self):
        dataset_ids = {item["id"] for item in load_catalog()}
        for path in NOTEBOOKS.rglob("*.ipynb"):
            with self.subTest(path=path):
                document = json.loads(path.read_text(encoding="utf-8"))
                self.assertIn("cells", document)
                self.assertIn("metadata", document)
                for index, cell in enumerate(document["cells"]):
                    if cell.get("cell_type") == "code":
                        source = "".join(cell.get("source", []))
                        compile(
                            source,
                            f"{path}:cell-{index}",
                            "exec",
                        )
                        for node in ast.walk(ast.parse(source)):
                            if not (
                                isinstance(node, ast.Call)
                                and isinstance(node.func, ast.Attribute)
                                and isinstance(node.func.value, ast.Name)
                                and node.func.value.id == "pdmdata"
                                and node.func.attr in {"load", "download"}
                                and node.args
                                and isinstance(node.args[0], ast.Constant)
                            ):
                                continue
                            dataset_id = node.args[0].value
                            self.assertIn(dataset_id, dataset_ids)
                            if node.func.attr == "load":
                                loader = import_module(
                                    f"pdmdata.datasets.{dataset_id}.loader"
                                ).load
                                signature(loader).bind_partial(
                                    **{
                                        kw.arg: None
                                        for kw in node.keywords
                                        if kw.arg
                                    }
                                )

    def test_dataset_notebooks_readme_matches_catalog_and_generators(self):
        readme = (NOTEBOOKS / "datasets" / "README.md").read_text(
            encoding="utf-8"
        )
        row_pattern = re.compile(
            r"^\| \[([^\]]+\.ipynb)\]\(([^)]+\.ipynb)\) \| `([^`]+)` \| [^|]+ \| `([^`]+)` \|$",
            re.MULTILINE,
        )
        rows = row_pattern.findall(readme)
        catalog_ids = [item["id"] for item in load_catalog()]
        self.assertEqual([dataset_id for _, _, dataset_id, _ in rows], catalog_ids)

        listed_notebooks = set()
        for label, target, dataset_id, generator_rel in rows:
            self.assertEqual(label, f"{dataset_id}.ipynb")
            self.assertEqual(target, f"{dataset_id}.ipynb")
            notebook_path = NOTEBOOKS / "datasets" / target
            self.assertTrue(notebook_path.is_file(), f"Missing {notebook_path}")
            listed_notebooks.add(notebook_path.name)

            expected_script = (
                f"scripts/notebooks/datasets/generate_{dataset_id}_notebook.py"
            )
            self.assertEqual(generator_rel, expected_script)
            self.assertTrue(
                (ROOT / generator_rel).is_file(),
                f"Missing generator script: {generator_rel}",
            )

        actual_notebooks = {
            path.name for path in (NOTEBOOKS / "datasets").glob("*.ipynb")
        }
        self.assertEqual(
            actual_notebooks,
            listed_notebooks | {"visualization-showcase.ipynb"},
        )
        self.assertTrue(
            (
                ROOT
                / "scripts/notebooks/generate_visualization_showcase.py"
            ).is_file()
        )
        self.assertTrue(
            (ROOT / "scripts/notebooks/generate_task_notebooks.py").is_file()
        )
        self.assertTrue((ROOT / "scripts/notebooks/generate_all.py").is_file())


if __name__ == "__main__":
    unittest.main()
