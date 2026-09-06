import json
import ast
from importlib import import_module
from inspect import signature
import unittest
from pathlib import Path

from pdmdata.catalog import load_catalog


ROOT = Path(__file__).resolve().parents[1]
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
                            "{}:cell-{}".format(path, index),
                            "exec",
                        )
                        for node in ast.walk(ast.parse(source)):
                            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "pdmdata"
                                    and node.func.attr in {"load", "download"} and node.args
                                    and isinstance(node.args[0], ast.Constant)):
                                continue
                            dataset_id = node.args[0].value
                            self.assertIn(dataset_id, dataset_ids)
                            if node.func.attr == "load":
                                loader = import_module(f"pdmdata.{dataset_id}.loader").load
                                signature(loader).bind_partial(**{kw.arg: None for kw in node.keywords if kw.arg})

if __name__ == "__main__":
    unittest.main()
