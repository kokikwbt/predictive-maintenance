import unittest
from io import StringIO
from pathlib import Path

import pdmdata
from pdmdata.catalog import load_catalog
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

if __name__ == "__main__":
    unittest.main()
