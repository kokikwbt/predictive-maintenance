"""Regressions for entity-safe, model-independent sequence preparation."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import polars as pl

from pdmdata.datasets.cmapss import loader as cmapss_loader
from pdmdata.tasks.tte import (
    CmapssSequenceSpec,
    build_sequence_samples,
    common_windows,
    complete_sequences,
    export_cmapss_sequences,
    export_derived_windows,
    prepare_cmapss_sequences,
)


def write_fixture(root):
    def rows(lengths):
        lines = []
        for unit, length in enumerate(lengths, 1):
            for cycle in range(1, length + 1):
                values = [unit, cycle, 0.1, 0.2, 0.3]
                values += [float(cycle + i + (1000 if unit == 3 else 0)) for i in range(21)]
                lines.append(" ".join(map(str, values)) + "\n")
        return "".join(lines)
    (root / "train_FD001.txt").write_text(rows([6, 7, 8]))
    (root / "test_FD001.txt").write_text(rows([2, 5]))
    (root / "RUL_FD001.txt").write_text("10\n20\n")


class SequenceTest(TestCase):
    def test_train_scaling_split_final_endpoint_and_short_test(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root)
            with patch.object(cmapss_loader, "find_raw_file", side_effect=lambda _, name: root / name):
                spec = CmapssSequenceSpec(subset="FD001", feature_columns=["sensor_2", "sensor_3"],
                    seq_len=3, rul_cap=4, validation_units=[3], train_stride=3, validation_stride=3)
                arrays, metadata = prepare_cmapss_sequences(spec)
            self.assertEqual(metadata["split_entities"]["train"], [1, 2])
            self.assertEqual(metadata["split_entities"]["validation"], [3])
            self.assertEqual(metadata["normalization"]["minimum"], [2., 3.])
            self.assertEqual(metadata["normalization"]["scale"], [6., 6.])
            self.assertGreater(arrays["validation"]["x"].min(), 1.)
            self.assertEqual(arrays["train"]["endpoint"].tolist(), [3., 6., 3., 6., 7.])
            self.assertEqual(arrays["train"]["y"].tolist(), [3., 0., 4., 1., 0.])
            self.assertEqual(arrays["test"]["y"].tolist(), [4., 4.])
            self.assertEqual(arrays["test"]["times"].tolist(), [[1., 1., 2.], [3., 4., 5.]])
            np.testing.assert_array_equal(arrays["test"]["x"][0, 0], arrays["test"]["x"][0, 1])
            self.assertNotIn("points_y", arrays["test"])
            self.assertEqual(arrays["test"]["x"].dtype, np.float32)

    def test_past_only_with_fixed_normalization_and_duplicate_times_rejected(self):
        frame = pl.DataFrame({"unit_number": [1]*6, "cycle": list(range(1, 7)),
                              "sensor_2": list(range(6)), "time_to_event": list(range(5, -1, -1))})
        original = build_sequence_samples(frame, ["sensor_2"], np.array([0.]), np.array([5.]), 3, 2)
        changed = frame.with_columns(pl.when(pl.col("cycle") == 6).then(999).otherwise(pl.col("sensor_2")).alias("sensor_2"))
        altered = build_sequence_samples(changed, ["sensor_2"], np.array([0.]), np.array([5.]), 3, 2)
        np.testing.assert_array_equal(original["x"][:2], altered["x"][:2])
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            build_sequence_samples(frame.with_columns(pl.lit(1).alias("cycle")), ["sensor_2"], np.array([0.]), np.array([5.]), 3, 2)

    def test_derivation_hashes_same_endpoint_and_no_test_access(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "cmapss"
            raw.mkdir()
            write_fixture(raw)
            spec = CmapssSequenceSpec(subset="FD001", feature_columns=["sensor_2"],
                seq_len=3, rul_cap=4, validation_units=[3], train_stride=2, validation_stride=2)
            with patch.object(cmapss_loader, "find_raw_file", side_effect=lambda _, name: raw / name), \
                 patch("pdmdata.get_settings") as settings:
                settings.return_value.data_root = root
                export_cmapss_sequences(root / "source", spec)
            # Corrupt official test: a train/validation derivation must not read it.
            (root / "source/test.npz").write_bytes(b"not a valid npz")
            short = export_derived_windows(root / "source", root / "w3", length=3, minimum_endpoint=5)
            long = export_derived_windows(root / "source", root / "w5", length=5, minimum_endpoint=5)
            self.assertEqual(short["split_entities"], long["split_entities"])
            self.assertIn("generator_sha256", long["sample_generator"])
            self.assertFalse((root / "w5/test.npz").exists())
            for name in ("train", "validation"):
                with np.load(root / "w3" / f"{name}.npz") as a, np.load(root / "w5" / f"{name}.npz") as b:
                    np.testing.assert_array_equal(a["endpoint"], b["endpoint"])
                    np.testing.assert_array_equal(a["y"], b["y"])
                    np.testing.assert_array_equal(a["x"], b["x"][:, -3:])
            with self.assertRaises(FileExistsError):
                export_derived_windows(root / "source", root / "w5", length=5, minimum_endpoint=5)
            with (root / "source/train.npz").open("ab") as f:
                f.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "hash"):
                export_derived_windows(root / "source", root / "bad", length=5, minimum_endpoint=5)

    def test_unclipped_labels_and_no_validation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root)
            with patch.object(cmapss_loader, "find_raw_file", side_effect=lambda _, name: root / name):
                arrays, _ = prepare_cmapss_sequences(CmapssSequenceSpec(subset="FD001",
                    feature_columns=["sensor_2"], rul_cap=None, seq_len=3, validation_fraction=0))
            self.assertNotIn("validation", arrays)
            self.assertEqual(arrays["test"]["y"].tolist(), [10., 20.])

    def test_reordered_points_and_too_short_endpoint_rejected(self):
        x = np.arange(6, dtype=np.float32)[:, None]
        a = {"x": np.stack([x[:3], x[-3:]]), "y": np.array([3, 0]),
            "times": np.array([[1, 2, 3], [4, 5, 6]]), "entity": np.array([1, 1]), "endpoint": np.array([3, 6]),
            "points_x": x, "points_y": np.minimum(np.arange(5, -1, -1), 3), "points_entity": np.ones(6)}
        sequences = complete_sequences(a, 3)
        with self.assertRaisesRegex(ValueError, "minimum_endpoint"):
            common_windows(a, sequences, 5, 3)
        with self.assertRaisesRegex(ValueError, "point sequence"):
            complete_sequences({**a, "points_x": x[::-1]}, 3)

    def test_invalid_settings_rejected(self):
        for extra in ({"seq_len": 0}, {"train_stride": True}, {"normalization": "test_minmax"}, {"rul_cap": -1}):
            with self.assertRaises(ValueError):
                CmapssSequenceSpec(subset="FD001", feature_columns=["sensor_2"], **extra)

    def test_target_and_entity_columns_cannot_be_requested_as_features(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root)
            with patch.object(cmapss_loader, "find_raw_file", side_effect=lambda _, name: root / name):
                for name in ("time_to_event", "unit_number", "cycle"):
                    with self.assertRaisesRegex(ValueError, "candidate sensors/settings"):
                        prepare_cmapss_sequences(CmapssSequenceSpec(subset="FD001", feature_columns=[name]))

    def test_outer_split_and_official_test_is_never_read(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root)
            def training_only(_, name):
                if name != "train_FD001.txt":
                    raise AssertionError("official test must not be read")
                return root / name
            with patch.object(cmapss_loader, "find_raw_file", side_effect=training_only):
                spec = CmapssSequenceSpec(subset="FD001", feature_columns=["sensor_2"],
                    seq_len=3, validation_units=[2], evaluation_units=[3], minimum_endpoint=5)
                arrays, metadata = prepare_cmapss_sequences(spec, include_test=False)
                self.assertEqual(set(arrays), {"train", "validation", "evaluation"})
                self.assertEqual(metadata["split_entities"], {"train": [1], "validation": [2], "evaluation": [3]})
                self.assertEqual(metadata["normalization"]["minimum"], [2.])
                self.assertEqual(metadata["normalization"]["scale"], [5.])
                self.assertGreater(arrays["evaluation"]["x"].min(), 1.)
                for split in arrays.values():
                    self.assertGreaterEqual(split["endpoint"].min(), 5)
                self.assertEqual(len(arrays["train"]["points_y"]), 6)
            with patch.object(cmapss_loader, "find_raw_file", side_effect=training_only):
                with self.assertRaisesRegex(ValueError, "overlap"):
                    prepare_cmapss_sequences(CmapssSequenceSpec(subset="FD001", feature_columns=["sensor_2"],
                        validation_units=[3], evaluation_units=[3]), include_test=False)
