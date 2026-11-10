"""Unified TTE prepare dispatch coverage."""

import unittest
from unittest.mock import patch

from pdmdata.tasks import tte
from pdmdata.tasks.tte import (
    prepare as tte_prepare,
    prepare_tte,
    supported_tte_datasets,
)


class TteDispatchTest(unittest.TestCase):
    def test_supported_tte_datasets_lists_prepare_adapters(self):
        self.assertEqual(
            supported_tte_datasets(),
            [
                "backblaze",
                "care",
                "cmapss",
                "mapm",
                "ncmapss",
                "scania_x",
                "xjtu_sy",
            ],
        )
        self.assertEqual(
            tte.supported_tte_datasets(),
            supported_tte_datasets(),
        )
        self.assertEqual(
            tte_prepare.supported_tte_datasets(),
            supported_tte_datasets(),
        )

    def test_prepare_tte_dispatches_kwargs(self):
        sentinel = object()
        with patch.object(
            tte_prepare, "prepare_mapm", return_value=sentinel
        ) as mocked:
            with patch.dict(tte_prepare._PREPARE, {"mapm": mocked}):
                result = prepare_tte("mapm", time_cap=30.0)
        self.assertIs(result, sentinel)
        mocked.assert_called_once_with(time_cap=30.0)

    def test_prepare_tte_rejects_unknown_dataset(self):
        with self.assertRaisesRegex(KeyError, "gdd"):
            prepare_tte("gdd")

    def test_prepare_tte_wraps_invalid_options(self):
        with self.assertRaisesRegex(TypeError, "Invalid options"):
            prepare_tte("mapm", subset="FD001")


if __name__ == "__main__":
    unittest.main()
