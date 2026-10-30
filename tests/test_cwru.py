"""CWRU collection coverage, source fidelity and bounded transfer tests."""

import json
import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import polars as pl
from scipy.io import savemat

import pdmdata
from pdmdata.config import Settings
from pdmdata.datasets import cwru
from pdmdata.io.download import bulk_downloads, download_variants


class CwruTest(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.settings = Settings(data_root=self.root)
        for module in ("loader", "download"):
            mocked = patch(
                f"pdmdata.datasets.cwru.{module}.get_settings",
                return_value=self.settings,
            )
            mocked.start()
            self.addCleanup(mocked.stop)

    def source(self, recording=105, *, data=None, directory="extracted"):
        path = self.root / "cwru" / directory / f"{recording}.mat"
        path.parent.mkdir(parents=True, exist_ok=True)
        if data is None:
            data = {"X999_DE_time": np.arange(8.0)[:, None], "X999RPM": 1796.5}
        savemat(path, data)
        return path

    def test_manifest_covers_official_collections_and_legacy_variants(self):
        frame = cwru.inventory()
        self.assertEqual(frame.height, 161)
        self.assertEqual(frame["recording"].n_unique(), 161)
        counts = dict(frame.group_by("collection").len().iter_rows())
        self.assertEqual(
            counts, {"normal": 4, "12k_de": 60, "48k_de": 52, "12k_fe": 45}
        )
        self.assertEqual(
            set(download_variants("cwru")),
            {"load0", "load1", "load2", "load3", *counts},
        )
        self.assertNotIn("cwru", bulk_downloads())
        variants = pdmdata.load_metadata("cwru")["download"]["variants"]
        for name in counts:
            self.assertEqual(
                set(variants[name]["expected_files"]),
                set(cwru.inventory(collection=name)["filename"]),
            )
        for hp in range(4):
            self.assertEqual(len(variants[f"load{hp}"]["files"]), 13)
        self.assertEqual(
            cwru.inventory(collection="normal")["diameter_in"].to_list(),
            [0.0] * 4,
        )
        self.assertEqual(
            cwru.inventory(collection="normal")["fault_location"].null_count(),
            4,
        )

    def test_combined_and_multi_value_selectors(self):
        selected = cwru.inventory(
            collection="12k_de", load_hp=1, fault="ball", diameter_in=0.014
        )
        self.assertEqual(selected["source_label"].item(), "B014_1")
        selected = cwru.inventory(recording=[105, 106], load_hp=[0, 1])
        self.assertEqual(selected["recording"].to_list(), [105, 106])
        self.assertEqual(
            cwru.inventory(fault="normal", diameter_in=0.007).height, 0
        )
        for kwargs in (
            {"load_hp": 4},
            {"collection": "unknown"},
            {"recording": True},
            {"recording": 105.0},
            {"outer_race_position": 9},
            {"fault": []},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                cwru.inventory(**kwargs)

    def test_default_dispatch_preserves_values_and_variable_identity(self):
        self.source()
        frame = pdmdata.load("cwru", recording=105)
        np.testing.assert_array_equal(
            frame["amplitude"].to_numpy(), np.arange(8.0)
        )
        self.assertEqual(
            frame["source_variable"].unique().to_list(), ["X999_DE_time"]
        )
        self.assertEqual(frame["rpm"].unique().to_list(), [1796.5])
        self.assertEqual(frame["fault"].unique().to_list(), ["inner_race"])
        self.assertEqual(frame["sample"].to_list(), list(range(8)))
        np.testing.assert_allclose(
            frame["time_s"].to_numpy(), np.arange(8) / 12000
        )

    def test_all_channels_keep_unequal_lengths_and_original_slice_indices(self):
        self.source(
            data={
                "X105_DE_time": np.arange(8.0)[:, None],
                "X105_FE_time": np.arange(5.0)[None, :],
                "X105_BA_time": np.arange(6.0)[:, None],
            }
        )
        frame = cwru.load(105, channels="all", start=1, stop=7, step=2)
        self.assertEqual(
            frame.filter(pl.col("channel") == "DE")["sample"].to_list(),
            [1, 3, 5],
        )
        self.assertEqual(
            frame.filter(pl.col("channel") == "FE")["sample"].to_list(), [1, 3]
        )
        self.assertEqual(
            frame.filter(pl.col("channel") == "BA")["sample"].to_list(),
            [1, 3, 5],
        )
        self.assertEqual(frame["rpm"].null_count(), frame.height)
        self.assertEqual(cwru.channel_info(105)["samples"].to_list(), [8, 5, 6])

    def test_ambiguous_rates_remain_null_unless_overridden(self):
        for recording in (97, 3001):
            self.source(recording)
            frame = cwru.load(recording)
            self.assertEqual(frame["sample_rate_hz"].null_count(), frame.height)
            self.assertEqual(frame["time_s"].null_count(), frame.height)
            frame = cwru.load(recording, sample_rates={"DE": 48000})
            self.assertEqual(
                frame["rate_source"].unique().to_list(), ["user_override"]
            )
            self.assertEqual(
                frame["sample_rate_hz"].unique().to_list(), [48000]
            )
        self.source(
            109,
            data={
                "X109_DE_time": np.ones((8, 1)),
                "X109_FE_time": np.ones((2, 1)),
            },
        )
        info = cwru.channel_info(109)
        self.assertEqual(info["sample_rate_hz"].to_list(), [48000, None])

    def test_embedded_previous_recording_is_not_selected_as_current(self):
        self.source(
            99,
            data={
                "X098_DE_time": np.ones((2, 1)),
                "X099_DE_time": np.full((3, 1), 2.0),
                "ans": np.array([[4.0]]),
            },
        )
        self.assertEqual(cwru.load(99)["amplitude"].to_list(), [2.0] * 3)
        self.assertEqual(
            set(cwru.variables(99)["variable"]),
            {"X098_DE_time", "X099_DE_time", "ans"},
        )
        np.testing.assert_array_equal(
            cwru.load_variable(99, "X098_DE_time"), np.ones((2, 1))
        )
        with self.assertRaises(ValueError):
            cwru.load_variable(99, "missing")

    def test_invalid_channels_slices_and_rates_fail(self):
        self.source()
        options = [
            {"channels": "FE"},
            {"channels": []},
            {"channels": ["DE", "DE"]},
            {"start": -1},
            {"start": None},
            {"step": None},
            {"step": 0},
            {"start": 3, "stop": 2},
            {"start": 100},
            {"sample_rates": {"DE": 0}},
            {"sample_rates": {"BAD": 12000}},
            {"sample_rates": 12000},
        ]
        for kwargs in options:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                cwru.load(105, **kwargs)

    def test_malformed_sources_fail_without_repair(self):
        data_cases = [
            {"X_DE_time": np.ones((2, 2))},
            {"X_DE_time": np.array([[np.nan]])},
            {"X_DE_time": np.array([[1j]])},
            {"other": np.ones((2, 1))},
            {"X_DE_time": np.ones((2, 1)), "Y_DE_time": np.ones((2, 1))},
            {"X_DE_time": np.ones((2, 1)), "XRPM": np.array([1.0, 2.0])},
        ]
        for data in data_cases:
            self.source(data=data)
            with self.assertRaises(ValueError):
                cwru.load(105)

    def test_installed_inventory_and_conflicting_duplicate_detection(self):
        path = self.source()
        self.assertEqual(
            cwru.inventory(installed=True)["recording"].to_list(), [105]
        )
        self.assertEqual(cwru.inventory(installed=False).height, 160)
        duplicate = self.root / "cwru" / "load0" / "extracted" / "105.mat"
        duplicate.parent.mkdir(parents=True)
        duplicate.write_bytes(path.read_bytes())
        self.assertEqual(cwru.load(105).height, 8)
        savemat(duplicate, {"X_DE_time": np.ones((2, 1))})
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            cwru.load(105)

    def test_iter_load_keeps_recordings_separate_and_missing_is_explicit(self):
        self.source(105)
        self.source(106)
        frames = list(cwru.iter_load(recording=[105, 106], stop=2))
        self.assertEqual(
            [f["recording"].unique().item() for f in frames], [105, 106]
        )
        with self.assertRaises(FileNotFoundError):
            list(cwru.iter_load(recording=[105, 107]))
        with self.assertRaises(ValueError):
            list(cwru.iter_load(collection="normal", fault="ball"))

    def test_bounded_download_reuses_local_files_and_records_provenance(self):
        payload_path = self.source(97)
        payload = payload_path.read_bytes()
        payload_path.unlink()

        def transfer(url, target, *, max_bytes):
            if len(payload) > max_bytes:
                raise ValueError("Download body exceeds the byte cap")
            target.write_bytes(payload)

        with patch(
            "pdmdata.datasets.cwru.download._download_file",
            side_effect=transfer,
        ) as mocked:
            result = cwru.download(
                recording=[97, 98], max_bytes=len(payload) * 2
            )
            self.assertEqual(mocked.call_count, 2)
            self.assertEqual(result["total_bytes"], len(payload) * 2)
            cwru.download(recording=97, max_bytes=len(payload))
            self.assertEqual(mocked.call_count, 2)
            manifest = json.loads(result["manifest"].read_text())
            self.assertEqual(len(manifest["files"]), 2)
            self.assertFalse(manifest["provider_checksum_available"])
            with self.assertRaises(ValueError):
                cwru.download(recording=[97, 98], max_bytes=len(payload))
            with self.assertRaises(ValueError):
                cwru.download(recording=99, max_bytes=1)
        with self.assertRaises(ValueError):
            cwru.download()

    def test_validation_reports_all_channels_and_missing_coverage(self):
        self.source()
        rows = cwru.validate(recording=105)
        self.assertTrue(rows["exact_array_match"].all())
        self.assertEqual(rows["samples"].to_list(), [8])
        self.assertEqual(cwru.validate(installed_only=True).height, 1)
        with self.assertRaises(FileNotFoundError):
            cwru.validate(recording=[105, 106])

    def test_truncated_transfer_retries_before_publish(self):
        source = self.source(97)
        payload = source.read_bytes()
        source.unlink()
        calls = []

        def transfer(url, target, *, max_bytes):
            calls.append(url)
            target.write_bytes(payload[:50] if len(calls) == 1 else payload)

        with patch(
            "pdmdata.datasets.cwru.download._download_file",
            side_effect=transfer,
        ):
            cwru.download(recording=97)
        self.assertEqual(len(calls), 2)
        self.assertEqual(cwru.load(97).height, 8)

        original = (self.root / "cwru" / "recordings" / "97.mat").read_bytes()
        with patch(
            "pdmdata.datasets.cwru.download._download_file",
            side_effect=lambda url, target, **kwargs: target.write_bytes(
                b"broken"
            ),
        ):
            with self.assertRaisesRegex(ValueError, "Invalid or truncated"):
                cwru.download(recording=97, overwrite=True)
        self.assertEqual(
            (self.root / "cwru" / "recordings" / "97.mat").read_bytes(),
            original,
        )

    def test_http_transfer_enforces_header_body_and_complete_response(self):
        from importlib import import_module

        transfer = import_module("pdmdata.datasets.cwru.download")._download_file
        target = self.root / "bounded.mat"
        target.write_bytes(b"previous")
        cases = [(b"abc", {"Content-Length": "20"}, 10, ValueError),
                 (b"abcdefghijk", {}, 10, ValueError),
                 (b"abc", {"Content-Length": "5"}, 10, OSError)]
        for payload, headers, limit, error in cases:
            response = BytesIO(payload)
            response.headers = headers
            with patch("pdmdata.datasets.cwru.download.urlopen", return_value=response):
                with self.assertRaises(error):
                    transfer("https://example.org/recording.mat", target, max_bytes=limit)
            self.assertEqual(target.read_bytes(), b"previous")
            self.assertFalse(target.with_suffix(".mat.part").exists())
        response = BytesIO(b"abc")
        response.headers = {"Content-Length": "3"}
        with patch("pdmdata.datasets.cwru.download.urlopen", return_value=response):
            transfer("https://example.org/recording.mat", target, max_bytes=3)
        self.assertEqual(target.read_bytes(), b"abc")

    def test_waveform_and_spectrum_respect_recording_channel_and_rate(self):
        # Known 100 Hz sine; loading every other source sample must retain 100 Hz.
        values = np.sin(2 * np.pi * 100 * np.arange(12000) / 12000)
        self.source(
            data={
                "X_DE_time": values[:, None],
                "X_FE_time": values[:1000, None],
            }
        )
        frame = cwru.load(105, channels="all")
        waveform = pdmdata.visualize("cwru", "waveform", frame, max_points=50)
        self.assertEqual(len(waveform.data), 2)
        self.assertEqual(len(waveform.data[0].x), 50)
        sliced = cwru.load(105, step=2)
        figure = pdmdata.visualize("cwru", "spectrum", sliced, nperseg=6000)
        peak = np.asarray(figure.data[0].x)[np.argmax(figure.data[0].y)]
        self.assertAlmostEqual(peak, 100.0)
        with self.assertRaises(ValueError):
            pdmdata.visualize("cwru", "spectrum", frame)
        self.source(97)
        with self.assertRaisesRegex(ValueError, "unspecified"):
            pdmdata.visualize("cwru", "spectrum", cwru.load(97))
        pdmdata.visualize(
            "cwru", "spectrum", cwru.load(97), sample_rate_hz=48000
        )


if __name__ == "__main__":
    unittest.main()
