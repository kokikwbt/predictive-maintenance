import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import pdmdata
from pdmdata.config import PROJECT_ROOT


class ConfigTest(unittest.TestCase):
    def test_repository_default_is_independent_of_working_directory(self):
        with patch.dict(os.environ, {}, clear=True), patch("pathlib.Path.cwd", return_value=Path("/tmp")):
            self.assertEqual(pdmdata.get_settings().data_root, PROJECT_ROOT / "data/raw")

    def test_configuration_is_reread_and_paths_are_relative_to_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary).resolve() / "settings.toml"
            with patch.dict(os.environ, {"PDMDATA_CONFIG": str(config)}):
                for value in ("custom/raw", "another", str(Path(temporary) / "absolute"), "~/pdmdata"):
                    config.write_text(f'data_root = "{value}"\n')
                    expected = Path(value).expanduser()
                    if not expected.is_absolute():
                        expected = config.parent / expected
                    self.assertEqual(pdmdata.get_settings().data_root, expected.resolve())
                self.assertEqual(list(config.parent.iterdir()), [config])

    def test_invalid_configuration_fails_clearly(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary).resolve() / "settings.toml"
            with patch.dict(os.environ, {"PDMDATA_CONFIG": str(config)}):
                with self.assertRaises(FileNotFoundError):
                    pdmdata.get_settings()
                for content in ('data_root = ""', 'data_root = 42', 'data_rooot = "typo"', 'invalid = ['):
                    config.write_text(content)
                    with self.subTest(content=content), self.assertRaises(ValueError):
                        pdmdata.get_settings()

    def test_download_and_load_share_configured_root_and_reuse_archive(self):
        module = importlib.import_module("pdmdata.download")
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary).resolve() / "settings.toml"
            config.write_text('data_root = "datasets"\n')

            def fake_download(url, target):
                with ZipFile(target, "w") as archive:
                    for filename in pdmdata.load_metadata("hydsys")["download"]["expected_files"]:
                        archive.writestr(filename, ("\t".join(["1"] * 6000) + "\n") * 2)

            with patch.dict(os.environ, {"PDMDATA_CONFIG": str(config)}), patch.object(module, "_download_file", side_effect=fake_download) as retrieve:
                with self.assertRaises(FileNotFoundError):
                    pdmdata.load("hydsys")
                retrieve.assert_not_called()
                result = pdmdata.download("hydsys")
                self.assertEqual(result["directory"], config.parent / "datasets/hydsys")
                self.assertEqual(pdmdata.load("hydsys").shape, (2, 6000))
                pdmdata.download("hydsys")
                retrieve.assert_called_once()
                self.assertEqual([p.name for p in (config.parent / "datasets").iterdir()], ["hydsys"])
                command = pdmdata.download_command("hydsys", output_dir=config.parent / "override")
                self.assertIn(str(config.parent / "override/hydsys"), command)

    def test_fresh_import_and_catalog_do_not_download_or_create_data(self):
        code = '''
from unittest.mock import patch
with patch("urllib.request.urlopen", side_effect=AssertionError("network")), \\
     patch("subprocess.run", side_effect=AssertionError("external command")), \\
     patch("pathlib.Path.mkdir", side_effect=AssertionError("directory creation")):
    import pdmdata
    pdmdata.load_catalog()
    pdmdata.download_command("hydsys")
'''
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary).resolve() / "settings.toml"
            config.write_text('data_root = "missing"\n')
            subprocess.run(
                [sys.executable, "-c", code], check=True,
                env={**os.environ, "PDMDATA_CONFIG": str(config)},
                cwd=PROJECT_ROOT,
            )
            self.assertFalse((config.parent / "missing").exists())


if __name__ == "__main__":
    unittest.main()
