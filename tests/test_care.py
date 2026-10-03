import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import polars as pl
from polars.testing import assert_frame_equal

import pdmdata
from pdmdata.datasets.care import load as load_care


class CareLoaderTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        config = self.root / "pdmdata.toml"
        config.write_text('data_root = "raw"\n')
        environment = patch.dict(os.environ, {"PDMDATA_CONFIG": str(config)})
        environment.start()
        self.addCleanup(environment.stop)
        self.raw = self.root / "raw/care/extracted/CARE_To_Compare"
        for farm, value in (("A", 1.5), ("B", 7.5)):
            directory = self.raw / f"Wind Farm {farm}"
            (directory / "datasets").mkdir(parents=True)
            (directory / "datasets/0.csv").write_text(
                "time_stamp;asset_id;id;train_test;status_type_id;sensor_0_avg\n"
                f"2022-01-01 00:00:00;1;0;train;0;{value}\n"
                f"2022-01-01 00:10:00;1;1;prediction;4;{value + 1}\n"
            )
            asset = "asset" if farm == "A" else "asset_id"
            (directory / "event_info.csv").write_text(
                f"{asset};event_id;event_label\n1;0;anomaly\n"
            )
            (directory / "feature_description.csv").write_text(
                "sensor_name;statistics_type;unit\nsensor_0;average;Celsius\n"
            )

    def test_farm_and_event_resolve_the_correct_recording(self):
        a = pdmdata.load("care", wind_farm="a", event_id=0)
        b = pdmdata.load("care", wind_farm="B", event_id=0, lazy=False)
        self.assertIsInstance(a, pl.LazyFrame)
        self.assertEqual(a.collect()["sensor_0_avg"].to_list(), [1.5, 2.5])
        self.assertEqual(b["sensor_0_avg"].to_list(), [7.5, 8.5])
        self.assertEqual(a.collect_schema()["time_stamp"], pl.Datetime("us"))

    def test_common_direct_and_previous_apis_agree(self):
        direct = load_care(wind_farm="A", event_id=0, lazy=False)
        common = pdmdata.load("care", wind_farm="A", event_id=0, lazy=False)
        previous = pdmdata.load("care", recording="Wind Farm A/datasets/0.csv", lazy=False)
        assert_frame_equal(direct, common)
        assert_frame_equal(direct, previous)

    def test_split_filters_without_modifying_raw_data(self):
        before = {p: p.read_bytes() for p in self.raw.rglob("*.csv")}
        for split, row_id in (("train", 0), ("prediction", 1)):
            result = load_care(wind_farm="A", event_id=0, split=split).collect()
            self.assertEqual(result["id"].to_list(), [row_id])
        after = {p: p.read_bytes() for p in self.raw.rglob("*.csv")}
        self.assertEqual(before, after)

    def test_event_tables_normalize_asset_column_and_features_load(self):
        for farm in ("A", "B"):
            events = pdmdata.load("care", wind_farm=farm, table="events", lazy=False)
            self.assertEqual(events.columns, ["asset_id", "event_id", "event_label"])
        features = load_care(wind_farm="A", table="features").collect()
        self.assertEqual(features["sensor_name"].to_list(), ["sensor_0"])
        previous = load_care(recording="Wind Farm A/event_info.csv", lazy=False)
        self.assertIn("asset", previous.columns)

    def test_invalid_selectors_fail_before_reading_files(self):
        options = [
            {}, {"wind_farm": "D", "event_id": 0},
            {"wind_farm": "A", "event_id": -1},
            {"wind_farm": "A", "event_id": True},
            {"wind_farm": "A", "event_id": "../0"},
            {"wind_farm": "A", "event_id": 0, "split": "test"},
            {"wind_farm": "A", "table": "events", "event_id": 0},
            {"wind_farm": "A", "table": "features", "split": "train"},
            {"wind_farm": "A", "table": "unknown"},
            {"recording": "0.csv", "wind_farm": "A", "event_id": 0},
        ]
        with patch("pdmdata.datasets.care.loader.find_raw_file") as find:
            for option in options:
                with self.subTest(options=option), self.assertRaises(ValueError):
                    pdmdata.load("care", **option)
            find.assert_not_called()

    def test_missing_event_does_not_download(self):
        with patch("urllib.request.urlopen") as network:
            with self.assertRaises(FileNotFoundError):
                load_care(wind_farm="A", event_id=999)
            network.assert_not_called()


if __name__ == "__main__":
    unittest.main()
