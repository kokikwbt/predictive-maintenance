import unittest
from io import StringIO
from pathlib import Path

import datasets
from datasets.catalog import format_summary, load_catalog, load_metadata
from datasets.docs import (
    BEGIN,
    END,
    render_catalog_table,
    render_metadata,
    render_task_table,
)
from datasets.download import (
    bulk_downloads,
    download_command,
    download_variants,
    supported_downloads,
)
from datasets.tasks import SUPPORT_LEVELS, TASKS


class CatalogTest(unittest.TestCase):
    def test_catalog_contains_every_dataset(self):
        self.assertEqual(
            [item["id"] for item in load_catalog()],
            [
                "alpi",
                "backblaze",
                "care",
                "cbm",
                "cmapss",
                "gdd",
                "gfd",
                "hydsys",
                "ims",
                "mapm",
                "metropt2",
                "oyicd",
                "ppd",
                "ufd",
            ],
        )

    def test_metadata_ids_match_directories(self):
        for item in load_catalog():
            self.assertEqual(load_metadata(item["id"])["id"], item["id"])

    def test_metadata_readme_section_is_renderable(self):
        for item in load_catalog():
            rendered = render_metadata(item)
            self.assertIn(BEGIN, rendered)
            self.assertIn(END, rendered)
            self.assertIn("## Attributes", rendered)

    def test_catalog_table_links_to_dataset_readmes(self):
        table = render_catalog_table(load_catalog())
        for item in load_catalog():
            self.assertIn(
                "datasets/{}/{}".format(
                    item["id"], item.get("readme", "README.md")
                ),
                table,
            )

    def test_task_support_uses_canonical_ids_and_levels(self):
        task_ids = {task["id"] for task in TASKS}
        for item in load_catalog():
            support = item["task_support"]
            self.assertTrue(support)
            self.assertLessEqual(set(support), task_ids)
            self.assertLessEqual(set(support.values()), SUPPORT_LEVELS)

    def test_task_table_contains_every_dataset_and_task(self):
        table = render_task_table(load_catalog())
        for item in load_catalog():
            self.assertIn(
                "datasets/{}/{}".format(
                    item["id"], item.get("readme", "README.md")
                ),
                table,
            )
        for task in TASKS:
            self.assertIn(task["short_name"], table)

    def test_catalog_can_filter_by_canonical_task(self):
        survival_ids = [
            item["id"] for item in load_catalog(task="survival_analysis")
        ]
        self.assertEqual(
            survival_ids,
            [
                "alpi",
                "backblaze",
                "care",
                "cmapss",
                "ims",
                "mapm",
                "oyicd",
                "ppd",
            ],
        )

    def test_automated_download_support_is_explicit(self):
        self.assertEqual(
            supported_downloads(),
            [
                "backblaze",
                "care",
                "cbm",
                "cmapss",
                "gdd",
                "gfd",
                "hydsys",
                "ims",
                "mapm",
                "metropt2",
                "oyicd",
                "ppd",
                "ufd",
            ],
        )
        self.assertEqual(
            [item["id"] for item in load_catalog(downloadable=False)],
            ["alpi"],
        )
        self.assertEqual(
            bulk_downloads(),
            [
                "cbm",
                "cmapss",
                "gdd",
                "gfd",
                "hydsys",
                "ims",
                "mapm",
                "oyicd",
                "ppd",
                "ufd",
            ],
        )

    def test_large_download_variants_are_explicit(self):
        self.assertEqual(
            download_variants("backblaze"),
            ["2025-q1", "2025-q2", "2025-q3", "2025-q4"],
        )
        command = download_command(
            "backblaze",
            Path("/tmp/pmdata-test"),
            variant="2025-q2",
        )
        self.assertIn("data_Q2_2025.zip", command)
        self.assertIn("/backblaze/2025-q2/", command)

    def test_wget_command_uses_metadata_url(self):
        command = download_command("gfd", Path("/tmp/pmdata-test"))
        self.assertIn("wget --continue", command)
        self.assertIn(load_metadata("gfd")["download"]["url"], command)

    def test_kaggle_command_uses_metadata_dataset(self):
        command = download_command("gdd", Path("/tmp/pmdata-test"))
        self.assertIn("kaggle datasets download", command)
        self.assertIn(load_metadata("gdd")["download"]["dataset"], command)

    def test_new_download_methods_match_their_sources(self):
        ims = download_command("ims", Path("/tmp/pmdata-test"))
        oyicd = download_command("oyicd", Path("/tmp/pmdata-test"))
        self.assertIn("data.nasa.gov", ims)
        self.assertIn("kaggle datasets download", oyicd)

    def test_summary_is_available_from_top_level_package(self):
        output = StringIO()
        result = datasets.summary(file=output)
        self.assertIsNone(result)
        table = output.getvalue()
        self.assertIn("Dataset", table)
        self.assertIn("cmapss", table)
        self.assertIn("Turbofan Engine Degradation Simulation Data Set", table)

    def test_summary_can_filter_direct_downloads(self):
        table = format_summary(downloadable=True)
        self.assertIn("hydsys", table)
        self.assertIn("mapm", table)


if __name__ == "__main__":
    unittest.main()
