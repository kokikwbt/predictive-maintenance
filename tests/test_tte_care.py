"""CARE time-to-event bundle."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import pdmdata
from pdmdata.tasks.tte import (
    EVENT_OBSERVED_COLUMN,
    TIME_TO_EVENT_COLUMN,
    eval_care,
    prepare_care,
)


def _write_farm(root: Path) -> None:
    directory = root / "care" / "extracted" / "CARE_To_Compare" / "Wind Farm A"
    datasets = directory / "datasets"
    datasets.mkdir(parents=True)
    (datasets / "0.csv").write_text(
        "time_stamp;asset_id;id;train_test;status_type_id;sensor_0_avg\n"
        "2022-01-01 00:00:00;1;0;train;0;1.0\n"
        "2022-01-01 00:10:00;1;1;train;0;1.1\n"
        "2022-01-01 00:20:00;1;2;prediction;0;1.2\n"
        "2022-01-01 00:30:00;1;3;prediction;4;1.3\n"
    )
    (datasets / "1.csv").write_text(
        "time_stamp;asset_id;id;train_test;status_type_id;sensor_0_avg\n"
        "2022-01-02 00:00:00;1;0;train;0;2.0\n"
        "2022-01-02 00:10:00;1;1;prediction;0;2.1\n"
    )
    (directory / "event_info.csv").write_text(
        "asset;event_id;event_label;event_start;event_end\n"
        "1;0;anomaly;2022-01-01 00:20:00;2022-01-01 00:30:00\n"
        "1;1;normal;;\n"
    )


class TteCareTest(TestCase):
    def test_onset_and_censored_normal_event(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_farm(root)
            with patch(
                "pdmdata.io.paths.get_settings",
                return_value=pdmdata.Settings(data_root=root),
            ):
                bundle = prepare_care("A", validation_fraction=0)
                self.assertNotIn(
                    "status_type_id", bundle.train.feature_columns
                )
                self.assertEqual(
                    bundle.train.feature_columns, ("sensor_0_avg",)
                )
                anomaly = bundle.train.frame.filter(
                    bundle.train.frame["event_id"] == 0
                )
                self.assertEqual(anomaly.height, 2)
                self.assertAlmostEqual(
                    anomaly[TIME_TO_EVENT_COLUMN][0], 20 / 60, places=6
                )
                self.assertTrue(anomaly[EVENT_OBSERVED_COLUMN].all())
                restored = bundle.test_eval_split()
                self.assertEqual(restored.frame["event_id"].to_list(), [0, 1])
                self.assertAlmostEqual(restored.y()[0], 0.0, places=6)
                self.assertAlmostEqual(restored.y()[1], 0.0, places=6)
                self.assertEqual(
                    restored.frame[EVENT_OBSERVED_COLUMN].to_list(),
                    [True, False],
                )
                durations = bundle.entity_table()["duration"].to_list()
                self.assertAlmostEqual(durations[0], 20 / 60, places=6)
                self.assertAlmostEqual(durations[1], 10 / 60, places=6)
                self.assertEqual(
                    eval_care(bundle, restored.y().to_list())["mae"], 0.0
                )
                held = prepare_care("A", validation_events=[1])
                self.assertEqual(
                    held.validation.groups().unique().to_list(), [1]
                )
                self.assertEqual(
                    held.test.groups().unique().sort().to_list(), [0, 1]
                )
