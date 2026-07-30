import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from datasets.download import (
    _download_kaggle,
    _extract_nested_zips,
    _extract_zip,
    _validate_expected_files,
    download,
    download_command,
)

download_module = importlib.import_module("datasets.download")


class DownloadTest(unittest.TestCase):
    def test_unsupported_dataset_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, "Automated download is not supported"):
            download("alpi", output_dir=Path("/tmp/unused-pmdata-test"))

    def test_unknown_download_variant_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, "variant must be one of"):
            download_command("backblaze", variant="2024-q1")

    def test_missing_kaggle_cli_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with patch.object(download_module.shutil, "which", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "kaggle auth login"):
                    _download_kaggle(
                        "owner/dataset",
                        directory,
                        directory / "dataset.zip",
                        overwrite=False,
                    )

    def test_kaggle_download_delegates_authentication(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            archive = directory / "dataset.zip"

            def complete(command, **_kwargs):
                archive.write_bytes(b"archive")

            with patch.object(
                download_module.shutil, "which", return_value="/usr/bin/kaggle"
            ), patch.object(download_module.subprocess, "run") as run:
                run.side_effect = complete
                _download_kaggle(
                    "owner/dataset", directory, archive, overwrite=False
                )

                command = run.call_args.args[0]
            self.assertEqual(command[:3], ["/usr/bin/kaggle", "datasets", "download"])
            self.assertNotIn("auth", command)
            self.assertNotIn("token", " ".join(command).lower())

    def test_zip_extraction_and_file_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "dataset.zip"
            destination = root / "extracted"
            destination.mkdir()
            with ZipFile(str(archive), "w") as zipped:
                zipped.writestr("nested/data.txt", "example")

            _extract_zip(archive, destination)
            _validate_expected_files(destination, ["data.txt"])
            self.assertEqual(
                (destination / "nested" / "data.txt").read_text(),
                "example",
            )

    def test_zip_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "unsafe.zip"
            destination = root / "extracted"
            destination.mkdir()
            with ZipFile(str(archive), "w") as zipped:
                zipped.writestr("../outside.txt", "unsafe")

            with self.assertRaisesRegex(ValueError, "Unsafe archive member"):
                _extract_zip(archive, destination)

    def test_nested_zip_is_extracted_before_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            extracted = Path(temporary) / "extracted"
            extracted.mkdir()
            inner = extracted / "dataset.zip"
            with ZipFile(str(inner), "w") as zipped:
                zipped.writestr("data/train.txt", "1 2 3\n")

            _extract_nested_zips(extracted)
            _validate_expected_files(extracted, ["train.txt"])

            self.assertTrue((extracted / "dataset" / "data" / "train.txt").is_file())


if __name__ == "__main__":
    unittest.main()
