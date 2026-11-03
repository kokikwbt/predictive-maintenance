"""Tests for the unified verify() dispatcher and the pdmdata CLI."""

from __future__ import annotations

import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import polars as pl

import pdmdata
from pdmdata import cli
from pdmdata.datasets import supported_verifications, verify


class VerifyDispatcherTest(unittest.TestCase):
    def test_supported_verifications_lists_all_validation_adapters(self) -> None:
        self.assertEqual(
            supported_verifications(),
            ["alpi", "backblaze", "care", "cmapss", "cwru", "ims"],
        )
        self.assertEqual(
            pdmdata.supported_verifications(),
            supported_verifications(),
        )

    def test_verify_dispatches_to_dataset_validation_module(self) -> None:
        expected = {"records": 10, "duplicates_removed": False}
        with patch(
            "pdmdata.datasets.alpi.validation.verify",
            return_value=expected,
        ) as mocked:
            result = pdmdata.verify("alpi")
            self.assertEqual(result, expected)
            mocked.assert_called_once_with()

        frame = pl.DataFrame({"recording": [105], "exact_array_match": [True]})
        with patch(
            "pdmdata.datasets.cwru.validation.verify",
            return_value=frame,
        ) as mocked_cwru:
            result_frame = verify("cwru", recording=105, installed_only=True)
            self.assertTrue(result_frame.equals(frame))
            mocked_cwru.assert_called_once_with(
                recording=105, installed_only=True
            )

    def test_verify_rejects_unknown_or_unsupported_datasets_and_bad_options(
        self,
    ) -> None:
        with self.assertRaisesRegex(KeyError, "Unknown dataset"):
            pdmdata.verify("nonexistent")
        with self.assertRaisesRegex(
            ValueError, "Offline verification is not supported for 'gfd'"
        ):
            pdmdata.verify("gfd")
        with self.assertRaisesRegex(
            TypeError, "Invalid options for 'alpi'"
        ):
            pdmdata.verify("alpi", unexpected_flag=True)


class CliTest(unittest.TestCase):
    def test_summary_command(self) -> None:
        buffer = StringIO()
        with redirect_stdout(buffer):
            code = cli.main(["summary", "--downloadable"])
        self.assertEqual(code, 0)
        output = buffer.getvalue()
        self.assertIn("cmapss", output)
        self.assertIn("Turbofan Engine Degradation Simulation Data Set", output)

    def test_download_print_command_and_single_download(self) -> None:
        buffer = StringIO()
        with redirect_stdout(buffer):
            code = cli.main(
                [
                    "download",
                    "backblaze",
                    "--variant",
                    "2025-q2",
                    "--output-dir",
                    "/tmp/pdmdata-cli",
                    "--print-command",
                ]
            )
        self.assertEqual(code, 0)
        self.assertIn("data_Q2_2025.zip", buffer.getvalue())

        fake_result = {
            "archive": Path("/tmp/pdmdata-cli/cmapss/CMAPSSData.zip"),
            "sha256": "abc123",
            "extracted": Path("/tmp/pdmdata-cli/cmapss/extracted"),
        }
        out = StringIO()
        with (
            patch("pdmdata.cli.download", return_value=fake_result) as mocked,
            redirect_stdout(out),
        ):
            code = cli.main(["download", "cmapss", "--overwrite"])
        self.assertEqual(code, 0)
        mocked.assert_called_once_with(
            "cmapss",
            None,
            extract=True,
            overwrite=True,
            variant=None,
        )
        self.assertIn("Archive:", out.getvalue())
        self.assertIn("SHA-256: abc123", out.getvalue())

    def test_download_all_and_argument_validation(self) -> None:
        fake_result = {
            "archive": Path("/tmp/archive.zip"),
            "sha256": "deadbeef",
            "extracted": Path("/tmp/extracted"),
        }
        out = StringIO()
        with (
            patch("pdmdata.cli.bulk_downloads", return_value=["cmapss", "gfd"]),
            patch("pdmdata.cli.download", return_value=fake_result) as mocked,
            redirect_stdout(out),
        ):
            code = cli.main(["download", "--all", "--no-extract"])
        self.assertEqual(code, 0)
        self.assertEqual(mocked.call_count, 2)
        self.assertIn("[1/2] Downloading cmapss", out.getvalue())

        for bad_args in (
            ["download"],
            ["download", "cmapss", "--all"],
            ["download", "--all", "--variant", "2025-q1"],
        ):
            with (
                self.subTest(bad_args=bad_args),
                redirect_stderr(StringIO()),
                self.assertRaises(SystemExit),
            ):
                cli.main(bad_args)

    def test_verify_command_formats_dict_and_dataframe_reports(self) -> None:
        out_json = StringIO()
        with (
            patch(
                "pdmdata.cli.verify",
                return_value={"variant": "2025-q1", "complete": True},
            ) as mocked_verify,
            redirect_stdout(out_json),
        ):
            code = cli.main(
                ["verify", "backblaze", "--variant", "2025-q1", "--check-archive"]
            )
        self.assertEqual(code, 0)
        mocked_verify.assert_called_once_with(
            "backblaze", variant="2025-q1", check_archive=True
        )
        parsed = json.loads(out_json.getvalue())
        self.assertTrue(parsed["complete"])

        out_df = StringIO()
        frame = pl.DataFrame({"recording": [105], "samples": [8]})
        with (
            patch("pdmdata.cli.verify", return_value=frame) as mocked_cwru,
            redirect_stdout(out_df),
        ):
            code = cli.main(
                ["verify", "cwru", "--recording", "105", "--installed-only"]
            )
        self.assertEqual(code, 0)
        mocked_cwru.assert_called_once_with(
            "cwru", installed_only=True, recording=105
        )
        self.assertIn("105", out_df.getvalue())


if __name__ == "__main__":
    unittest.main()
