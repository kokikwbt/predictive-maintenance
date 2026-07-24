import unittest
from io import StringIO
from pathlib import Path

import datasets
from datasets.catalog import format_summary, load_catalog, load_metadata
from datasets.docs import BEGIN, END, render_catalog_table, render_metadata
from datasets.download import download_command, supported_downloads


class CatalogTest(unittest.TestCase):
    def test_catalog_contains_every_dataset(self):
        self.assertEqual(
            [item["id"] for item in load_catalog()],
            [
                "alpi",
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

    def test_automated_download_support_is_explicit(self):
        self.assertEqual(
            supported_downloads(),
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
        self.assertEqual(
            [item["id"] for item in load_catalog(downloadable=False)],
            ["alpi"],
        )

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
