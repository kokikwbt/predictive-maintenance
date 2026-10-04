"""SCANIA Component X typed loading, derived targets, and challenge cost."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import polars as pl

import pdmdata
from pdmdata.datasets.scania_x import (
    FEATURES,
    challenge_cost,
    cost_table,
    histogram_columns,
    inventory,
    last_readouts,
    load,
    vehicles,
)
from pdmdata.datasets.scania_x.viz import plot_histogram_evolution


HEADER = ["vehicle_id", "time_step", *FEATURES]


def _readout(vehicle: int, time_step: float, value: str = "1.0") -> str:
    return ",".join([str(vehicle), str(time_step)] + [value] * len(FEATURES))


def _write_split(root: Path) -> None:
    directory = root / "scania_x" / "extracted"
    directory.mkdir(parents=True)
    # Vehicle 0 is repaired at 60; vehicle 2 is censored at 50. A missing
    # first value in 171_0 must stay null instead of forcing a string column.
    train = [",".join(HEADER)]
    first = _readout(0, 0.0).split(",")
    first[2] = ""
    train.append(",".join(first))
    train += [_readout(0, t) for t in (13.0, 40.0, 55.0, 59.5)]
    train += [_readout(2, t) for t in (0.0, 1.5, 10.0)]
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
    validation = [",".join(HEADER)]
    validation += [_readout(5, t) for t in (3.0, 1.0)]
    validation += [_readout(7, 2.0, "2.0")]
    (directory / "validation_operational_readouts.csv").write_text(
        "\n".join(validation) + "\n"
    )
    (directory / "validation_labels.csv").write_text(
        "vehicle_id,class_label\n5,4\n7,0\n"
    )
    (directory / "validation_specifications.csv").write_text(
        specs + "5," + ",".join(["Cat2"] * 8) + "\n"
        "7," + ",".join(["Cat0"] * 8) + "\n"
    )


class ScaniaComponentXTest(TestCase):
    def _settings(self, root: Path):
        return patch(
            "pdmdata.io.paths.get_settings",
            return_value=pdmdata.Settings(data_root=root),
        )

    def test_readouts_are_typed_lazy_and_selectable(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_split(root)
            with self._settings(root):
                readouts = pdmdata.load("scania_x")
                self.assertIsInstance(readouts, pl.LazyFrame)
                schema = readouts.collect_schema()
                self.assertEqual(len(schema), 107)
                self.assertEqual(schema["vehicle_id"], pl.Int64)
                self.assertEqual(schema["171_0"], pl.Float64)
                self.assertEqual(schema["397_35"], pl.Float64)
                frame = load("train", vehicle_id=0, lazy=False)
                self.assertEqual(frame.height, 5)
                self.assertIsNone(frame["171_0"][0])
                selected = load(
                    "train", columns=["time_step", "459_19"], lazy=False
                )
                self.assertEqual(selected.columns, ["time_step", "459_19"])
                self.assertEqual(load("train", vehicle_id=[2]).collect().height, 3)
                tte = load("train", "tte")
                self.assertIsInstance(tte, pl.DataFrame)
                self.assertEqual(tte.schema["in_study_repair"], pl.UInt8)
                self.assertEqual(
                    load("validation", "labels").schema["class_label"],
                    pl.UInt8,
                )

    def test_train_targets_follow_windows_and_censoring(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_split(root)
            with self._settings(root):
                labeled = (
                    load("train", with_labels=True)
                    .sort(["vehicle_id", "time_step"])
                    .collect()
                )
                repaired = labeled.filter(pl.col("vehicle_id") == 0)
                self.assertEqual(
                    repaired["time_to_event"].to_list(),
                    [60.0, 47.0, 20.0, 5.0, 0.5],
                )
                self.assertEqual(repaired["class_label"].to_list(), [0, 1, 2, 4, 4])
                self.assertEqual(repaired["event_observed"].unique().to_list(), [1])
                censored = labeled.filter(pl.col("vehicle_id") == 2)
                self.assertEqual(censored["class_label"].to_list(), [0, 0, None])

    def test_vehicle_tables_and_last_readouts(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_split(root)
            with self._settings(root):
                train = vehicles("train")
                self.assertEqual(train["vehicle_id"].to_list(), [0, 2])
                self.assertIn("in_study_repair", train.columns)
                last = last_readouts("validation")
                self.assertEqual(last["vehicle_id"].to_list(), [5, 7])
                self.assertEqual(last["time_step"].to_list(), [3.0, 2.0])
                self.assertEqual(last["class_label"].to_list(), [4, 0])
                self.assertEqual(last["Spec_0"].to_list(), ["Cat2", "Cat0"])
                with self.assertRaises(FileNotFoundError):
                    inventory()

    def test_invalid_selections_raise(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_split(root)
            with self._settings(root):
                with self.assertRaises(ValueError):
                    load("holdout")
                with self.assertRaises(ValueError):
                    load("validation", "tte")
                with self.assertRaises(ValueError):
                    load("train", "labels")
                with self.assertRaises(ValueError):
                    load("validation", with_labels=True)
                with self.assertRaises(ValueError):
                    load("train", columns=["unknown"])
                with self.assertRaises(ValueError):
                    load("train", vehicle_id=True)
                with self.assertRaises(FileNotFoundError):
                    load("test")
        with self.assertRaises(ValueError):
            histogram_columns("999")
        self.assertEqual(len(histogram_columns("397")), 36)

    def test_challenge_cost_matches_published_matrix(self):
        self.assertEqual(challenge_cost([0, 1, 4, 2], [0, 1, 4, 2]), 0)
        self.assertEqual(challenge_cost([0], [4]), 10)
        self.assertEqual(challenge_cost([4], [0]), 500)
        self.assertEqual(challenge_cost(pl.Series([3, 1]), pl.Series([1, 3])), 308)
        table = cost_table()
        self.assertEqual(table.height, 25)
        self.assertEqual(table["cost"].sum(), 3080)
        with self.assertRaises(ValueError):
            challenge_cost([0, 1], [0])
        with self.assertRaises(ValueError):
            challenge_cost([5], [0])

    def test_counter_trajectory_plot(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_split(root)
            with self._settings(root):
                figure = pdmdata.visualize(
                    "scania_x", "counter_trajectory", load("train"), entity=0
                )
                self.assertEqual(len(figure.data), 4)
                histogram = plot_histogram_evolution(
                    load("train", vehicle_id=0, lazy=False), variable="291"
                )
                self.assertEqual(len(histogram.axes), 2)
                with self.assertRaises(ValueError):
                    plot_histogram_evolution(load("train", lazy=False))
