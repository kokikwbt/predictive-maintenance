"""Unified RUL prepare dispatch coverage."""

import unittest
from unittest.mock import patch

from pdmdata.tasks import rul
from pdmdata.tasks.rul import prepare_rul, supported_rul_datasets
from pdmdata.tasks.rul import prepare as rul_prepare


class RulDispatchTest(unittest.TestCase):
    def test_supported_rul_datasets_lists_prepare_adapters(self):
        self.assertEqual(
            supported_rul_datasets(),
            ["cmapss", "ncmapss", "xjtu_sy"],
        )
        self.assertEqual(
            rul.supported_rul_datasets(),
            supported_rul_datasets(),
        )
        self.assertEqual(
            rul_prepare.supported_rul_datasets(),
            supported_rul_datasets(),
        )

    def test_prepare_rul_dispatches_kwargs(self):
        sentinel = object()
        with patch.object(
            rul_prepare, "prepare_cmapss", return_value=sentinel
        ) as mocked:
            # Re-bind dispatch map after patching the prepare module name.
            with patch.dict(
                rul_prepare._PREPARE, {"cmapss": mocked}
            ):
                result = prepare_rul(
                    "cmapss", subset="FD001", rul_cap=125
                )
        self.assertIs(result, sentinel)
        mocked.assert_called_once_with(subset="FD001", rul_cap=125)

    def test_prepare_rul_rejects_unknown_dataset(self):
        with self.assertRaisesRegex(KeyError, "alpi"):
            prepare_rul("alpi")

    def test_prepare_rul_wraps_invalid_options(self):
        with self.assertRaisesRegex(TypeError, "Invalid options"):
            prepare_rul("xjtu_sy", subset="FD001")


if __name__ == "__main__":
    unittest.main()
