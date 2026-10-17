import unittest
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pdmdata
from pdmdata.catalog import load_catalog
from pdmdata.catalog import docs
from pdmdata.catalog.docs import BEGIN, END, render_metadata
from pdmdata.io.download import (
    bulk_downloads,
    download_command,
    download_variants,
)
from pdmdata.tasks import SUPPORT_LEVELS, TASKS


class CatalogTest(unittest.TestCase):
    def test_metadata_readme_section_is_renderable(self):
        for item in load_catalog():
            rendered = render_metadata(item)
            self.assertIn(BEGIN, rendered)
            self.assertIn(END, rendered)
            self.assertIn("## Attributes", rendered)

    def test_task_support_uses_canonical_ids_and_levels(self):
        task_ids = {task["id"] for task in TASKS}
        for item in load_catalog():
            support = item["task_support"]
            self.assertTrue(support)
            self.assertLessEqual(set(support), task_ids)
            self.assertLessEqual(set(support.values()), SUPPORT_LEVELS)

    def test_catalog_can_filter_by_canonical_task(self):
        survival_ids = [
            item["id"] for item in load_catalog(task="survival_analysis")
        ]
        self.assertEqual(
            survival_ids,
            [
                "backblaze",
                "care",
                "cmapss",
                "ims",
                "mapm",
                "ncmapss",
                "oyicd",
                "ppd",
                "scania_x",
                "xjtu_sy",
            ],
        )

    def test_large_download_variants_are_explicit(self):
        self.assertTrue(
            {
                "care",
                "backblaze",
                "metropt2",
                "ncmapss",
                "scania_x",
                "xjtu_sy",
            }.isdisjoint(bulk_downloads())
        )
        self.assertEqual(
            download_variants("backblaze"),
            ["2025-q1", "2025-q2", "2025-q3", "2025-q4"],
        )
        self.assertEqual(
            download_variants("ncmapss")[:3],
            ["ds01", "ds02", "ds03"],
        )
        command = download_command(
            "backblaze",
            Path("/tmp/pdmdata-test"),
            variant="2025-q2",
        )
        self.assertIn("data_Q2_2025.zip", command)
        self.assertIn("/backblaze/2025-q2/", command)
        ncmapss = download_command(
            "ncmapss",
            Path("/tmp/pdmdata-test"),
            variant="ds01",
        )
        self.assertIn("shreyaravi0/aircraft", ncmapss)
        self.assertIn("N-CMAPSS_DS01-005.h5", ncmapss)
        self.assertIn("/ncmapss/ds01", ncmapss)

    def test_summary_is_available_from_top_level_package(self):
        output = StringIO()
        result = pdmdata.summary(file=output)
        self.assertIsNone(result)
        table = output.getvalue()
        self.assertIn("Dataset", table)
        self.assertIn("cmapss", table)
        self.assertIn("Turbofan Engine Degradation Simulation Data Set", table)


class CatalogDocumentationTest(unittest.TestCase):
    def test_source_list_includes_every_dataset_in_catalog_order(self):
        catalog = load_catalog()
        lines = docs.render_dataset_sources(catalog).splitlines()
        self.assertEqual(len(lines), len(catalog))
        for item, line in zip(catalog, lines):
            self.assertTrue(line.startswith("- [" + item["short_name"] + " — "))
            self.assertIn("](" + item["source"]["landing_page"] + ")", line)

    def test_regeneration_updates_sources_and_preserves_manual_content(self):
        catalog = load_catalog()
        catalog[0]["source"]["landing_page"] = "https://example.org/updated-source"
        template = "\n\n".join([
            "# Catalog\n\nManual introduction.",
            docs.CATALOG_BEGIN + "\nOld table\n" + docs.CATALOG_END,
            docs.TASKS_BEGIN + "\nOld tasks\n" + docs.TASKS_END,
            "## Dataset sources",
            docs.SOURCES_BEGIN + "\nOld source\n" + docs.SOURCES_END,
            "## Reference links\n\n- [Manual reference](https://example.org/reference)\n",
        ])
        with TemporaryDirectory() as directory:
            readme = Path(directory) / "README.md"
            readme.write_text(template, encoding="utf-8")
            with patch.object(docs, "ROOT", Path(directory)), patch.object(
                docs, "load_catalog", return_value=catalog
            ):
                self.assertEqual(docs.update_catalog_readme(), readme)
                updated = readme.read_text(encoding="utf-8")
                self.assertIn("https://example.org/updated-source", updated)
                self.assertNotIn("Old source", updated)
                self.assertEqual(
                    updated.split(docs.SOURCES_END, 1)[1],
                    template.split(docs.SOURCES_END, 1)[1],
                )
                self.assertTrue(updated.startswith("# Catalog\n\nManual introduction."))
                docs.update_catalog_readme()
                self.assertEqual(readme.read_text(encoding="utf-8"), updated)

    def test_missing_source_markers_does_not_overwrite_readme(self):
        template = "\n".join([
            "# Catalog", docs.CATALOG_BEGIN, docs.CATALOG_END,
            docs.TASKS_BEGIN, docs.TASKS_END, "Manual sources without markers.",
        ])
        with TemporaryDirectory() as directory:
            readme = Path(directory) / "README.md"
            readme.write_text(template, encoding="utf-8")
            with patch.object(docs, "ROOT", Path(directory)):
                with self.assertRaisesRegex(ValueError, "dataset-source markers"):
                    docs.update_catalog_readme()
            self.assertEqual(readme.read_text(encoding="utf-8"), template)


if __name__ == "__main__":
    unittest.main()
