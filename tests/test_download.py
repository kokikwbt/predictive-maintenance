import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from pdmdata.io.download import (
    _download_kaggle,
    _extract_nested_archives,
    _extract_zip,
    _validate_expected_files,
    download,
    download_command,
)

download_module = importlib.import_module("pdmdata.io.download")


class DownloadTest(unittest.TestCase):
    def test_kaggle_authentication_retry_is_bounded(self):
        import subprocess

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            archive = directory / "dataset.zip"
            error = subprocess.CalledProcessError(1, ["kaggle"], stderr="401 Unauthorized")
            with patch.object(download_module.shutil, "which", return_value="kaggle"), \
                 patch.object(download_module, "_interactive_kaggle_login", return_value=True), \
                 patch.object(download_module.subprocess, "run") as run:
                def complete(*args, **kwargs):
                    if run.call_count == 1:
                        raise error
                    if run.call_count == 3:
                        archive.write_bytes(b"archive")
                    return subprocess.CompletedProcess(args[0], 0, "", "")
                run.side_effect = complete
                _download_kaggle("owner/dataset", directory, archive, overwrite=False)
                self.assertEqual(run.call_count, 3)
                self.assertEqual(run.call_args_list[1].args[0], ["kaggle", "auth", "login", "--force"])
                self.assertEqual(run.call_args_list[1].kwargs["timeout"], 300)
                archive.unlink()
                for outcomes in ([error, subprocess.TimeoutExpired("login", 300)],
                                 [error, subprocess.CompletedProcess("login", 0), error]):
                    run.reset_mock()
                    run.side_effect = outcomes
                    with self.assertRaises(RuntimeError):
                        _download_kaggle("owner/dataset", directory, archive, overwrite=False)
                    self.assertLessEqual(run.call_count, 3)

    def test_kaggle_noninteractive_and_non_auth_failures_do_not_login(self):
        import subprocess

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for interactive, message in ((False, "401 Unauthorized"),
                                         (True, "403 Forbidden"), (True, "Connection timeout")):
                with self.subTest(message=message), \
                     patch.object(download_module.shutil, "which", return_value="kaggle"), \
                     patch.object(download_module, "_interactive_kaggle_login", return_value=interactive), \
                     patch.object(download_module.subprocess, "run", side_effect=
                                  subprocess.CalledProcessError(1, ["kaggle"], stderr=message)) as run:
                    with self.assertRaises(RuntimeError):
                        _download_kaggle("owner/dataset", directory, directory / "dataset.zip", overwrite=False)
                    run.assert_called_once()
            with patch.dict(download_module.os.environ, {"CI": "true"}):
                self.assertFalse(download_module._interactive_kaggle_login())

    def test_microsoft_files_bundle_reuse_and_failed_refresh(self):
        import pdmdata

        content = b"machineID,model,age\n1,model3,18\n"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def retrieve(url, target):
                self.assertTrue(url.startswith("https://raw.githubusercontent.com/microsoft/"))
                target.write_bytes(content)

            with patch.object(download_module, "_download_file", side_effect=retrieve) as fetch:
                result = download("mapm", output_dir=root)
                self.assertEqual(fetch.call_count, 5)
                with ZipFile(result["archive"]) as archive:
                    self.assertEqual(len(archive.namelist()), 5)
                    self.assertEqual(archive.read("PdM_machines.csv"), content)
                self.assertEqual((result["extracted"] / "PdM_machines.csv").read_bytes(), content)
                download("mapm", output_dir=root)
                self.assertEqual(fetch.call_count, 5)
                before = result["archive"].read_bytes()
                fetch.side_effect = OSError("interrupted download")
                with self.assertRaisesRegex(OSError, "interrupted"):
                    download("mapm", output_dir=root, overwrite=True)
                self.assertEqual(result["archive"].read_bytes(), before)
                self.assertFalse(any(p.name.startswith("tmp") for p in result["directory"].iterdir()))
            with patch(
                "pdmdata.io.paths.get_settings",
                return_value=pdmdata.Settings(data_root=root),
            ):
                frame = pdmdata.load("mapm", table="machines")
                self.assertEqual(frame["age"].to_list(), [18])
            command = download_command("mapm", output_dir=root)
            self.assertEqual(command.count("wget --continue"), 5)
            self.assertNotIn("kaggle", command)

    def test_expected_paths_distinguish_wind_farms(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            farm = root / "Wind Farm A"
            farm.mkdir()
            (farm / "event_info.csv").write_text("event_id\n0\n")
            _validate_expected_files(root, ["Wind Farm A/event_info.csv"])
            with self.assertRaisesRegex(ValueError, "missing expected files"):
                _validate_expected_files(root, ["Wind Farm B/event_info.csv"])

    def test_download_checks_provider_checksum_before_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "care"
            directory.mkdir()
            with ZipFile(directory / "CARE_To_Compare.zip", "w") as archive:
                archive.writestr("unexpected.txt", "not the official archive")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                download("care", output_dir=root)
            self.assertFalse((directory / "extracted").exists())
            self.assertFalse((directory / "manifest.json").exists())

    def test_files_method_verifies_per_file_checksums(self):
        import hashlib

        content = b"vehicle_id,class_label\n1,0\n"
        digest = hashlib.sha256(content).hexdigest()
        resource = {
            "method": "files",
            "filename": "bundle.zip",
            "files": [
                {"filename": "a.csv", "url": "https://example.org/a.csv",
                 "checksum": "sha256:" + digest},
                {"filename": "b.csv", "url": "https://example.org/b.csv"},
            ],
        }

        def retrieve(url, target):
            target.write_bytes(content)

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            archive = directory / "bundle.zip"
            with patch.object(download_module, "_download_file", side_effect=retrieve):
                download_module._download_resource(
                    resource, directory, archive, overwrite=False
                )
                with ZipFile(archive) as zipped:
                    self.assertEqual(sorted(zipped.namelist()), ["a.csv", "b.csv"])
                archive.unlink()
                resource["files"][0]["checksum"] = "sha256:" + "0" * 64
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    download_module._download_resource(
                        resource, directory, archive, overwrite=False
                    )
            self.assertFalse(archive.exists())
            self.assertEqual(list(directory.iterdir()), [])

    def test_scania_component_x_files_are_pinned(self):
        from pdmdata.catalog import load_metadata

        download_spec = load_metadata("scania_x")["download"]
        self.assertEqual(len(download_spec["files"]), 9)
        for item in download_spec["files"]:
            self.assertTrue(item["url"].startswith(
                "https://api.researchdata.se/dataset/2024-34/3/file/data/"
            ))
            self.assertTrue(item["url"].endswith("/" + item["filename"]))
            self.assertRegex(item["checksum"], r"^sha256:[0-9a-f]{64}$")
        self.assertEqual(
            sorted(download_spec["expected_files"]),
            sorted(item["filename"] for item in download_spec["files"]),
        )
        command = download_command("scania_x", output_dir=Path("/tmp/pdmdata-test"))
        self.assertEqual(command.count("wget --continue"), 9)

    def test_unsupported_dataset_has_clear_error(self):
        with self.assertRaisesRegex(KeyError, "Unknown dataset"):
            download("unknown", output_dir=Path("/tmp/unused-pdmdata-test"))

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

            _extract_nested_archives(extracted)
            _validate_expected_files(extracted, ["train.txt"])

            self.assertTrue((extracted / "dataset" / "data" / "train.txt").is_file())


if __name__ == "__main__":
    unittest.main()
