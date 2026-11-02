"""SCANIA Component X time-to-event bundle."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import pdmdata
from pdmdata.datasets.scania_x import FEATURES
from pdmdata.tasks.tte import (
    EVENT_OBSERVED_COLUMN,
    TIME_TO_EVENT_COLUMN,
    eval_scania_x,
    prepare_scania_x,
)


def _readout(vehicle: int, time_step: float, value: str = "1.0") -> str:
    return ",".join(
        [str(vehicle), str(time_step)] + [value] * len(FEATURES)
    )


def _write_split(root: Path) -> None:
    directory = root / "scania_x" / "extracted"
    directory.mkdir(parents=True)
    header = ",".join(["vehicle_id", "time_step", *FEATURES])
    train = [header]
    first = _readout(0, 0.0).split(",")
    first[2] = ""
    train.append(",".join(first))
    train += [_readout(0, time_step) for time_step in (13.0, 40.0, 55.0, 59.5)]
    train += [_readout(2, time_step) for time_step in (0.0, 1.5, 10.0)]
    (directory / "train_operational_readouts.csv").write_text(
        "\r\n".join(train) + "\r\n"
    )
    (directory / "train_tte.csv").write_text(
        "vehicle_id,length_of_study_time_step,in_study_repair\n"
        "0,60.0,1\n2,50.0,0\n"
    )
    specs = "vehicle_id," + ",".join(f"Spec_{i}" for i in range(8)) + "\n"
    (directory / "train_specifications.csv").write_text(
        specs + "0," + ",".join(["Cat0"] * 8) + "\n"
        "2," + ",".join(["Cat1"] * 8) + "\n"
    )


class TteScaniaXTest(TestCase):
    def test_holdout_keeps_censoring_and_uncapped_duration(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_split(root)
            with patch(
                "pdmdata.io.paths.get_settings",
                return_value=pdmdata.Settings(data_root=root),
            ):
                bundle = prepare_scania_x(
                    test_vehicles=[2],
                    validation_fraction=0,
                    time_cap=10,
                )
                self.assertNotIn("class_label", bundle.train.frame.columns)
                self.assertEqual(
                    bundle.train.frame[TIME_TO_EVENT_COLUMN].to_list(),
                    [10.0, 10.0, 10.0, 5.0, 0.5],
                )
                self.assertTrue(
                    bundle.train.frame[EVENT_OBSERVED_COLUMN].all()
                )
                entities = bundle.entity_table()
                self.assertEqual(entities["duration"].to_list(), [60.0, 50.0])
                self.assertEqual(
                    entities[EVENT_OBSERVED_COLUMN].to_list(), [True, False]
                )
                self.assertEqual(
                    entities["cohort"].to_list(), ["train", "test"]
                )
                restored = bundle.test_eval_split()
                self.assertEqual(
                    restored.y().to_list(), [10.0, 10.0, 10.0]
                )
                self.assertFalse(restored.frame[EVENT_OBSERVED_COLUMN].any())
                metrics = eval_scania_x(bundle, [10.0, 10.0, 10.0])
                self.assertEqual(metrics["mae"], 0.0)
                with self.assertRaises(ValueError):
                    prepare_scania_x(time_cap=0)
                with self.assertRaises(ValueError):
                    prepare_scania_x(
                        test_vehicles=[2],
                        validation_vehicles=[0],
                    )
