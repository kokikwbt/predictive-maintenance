"""Backblaze time-to-event bundle."""

from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import pdmdata
from pdmdata.tasks.tte import (
    EVENT_OBSERVED_COLUMN,
    TIME_TO_EVENT_COLUMN,
    eval_backblaze,
    prepare_backblaze,
)


def _write_quarter(root: Path) -> None:
    directory = root / "backblaze" / "2025-q1" / "extracted"
    directory.mkdir(parents=True)
    header = "date,serial_number,model,capacity_bytes,failure,smart_1_raw"
    rows = {
        date(2025, 1, 1): [
            "2025-01-01,drive-a,ModelA,100,0,1.0",
            "2025-01-01,drive-b,ModelB,200,0,3.0",
        ],
        date(2025, 1, 2): [
            "2025-01-02,drive-a,ModelA,100,0,1.5",
            "2025-01-02,drive-b,ModelB,200,0,3.5",
        ],
        date(2025, 1, 3): [
            "2025-01-03,drive-a,ModelA,100,1,2.0",
            "2025-01-03,drive-b,ModelB,200,0,4.0",
        ],
    }
    for day, lines in rows.items():
        path = directory / f"{day.isoformat()}.csv"
        path.write_text(header + "\n" + "\n".join(lines) + "\n")


class TteBackblazeTest(TestCase):
    def test_failure_day_and_censored_drive(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_quarter(root)
            with patch(
                "pdmdata.datasets.backblaze.loader.get_settings",
                return_value=pdmdata.Settings(data_root=root),
            ):
                bundle = prepare_backblaze(
                    "2025-q1",
                    test_serials=["drive-b"],
                    validation_fraction=0,
                )
                self.assertNotIn("failure", bundle.train.feature_columns)
                self.assertNotIn("model", bundle.train.feature_columns)
                failed = bundle.train.frame.sort("date")
                self.assertEqual(
                    failed[TIME_TO_EVENT_COLUMN].to_list(), [2, 1, 0]
                )
                self.assertTrue(failed[EVENT_OBSERVED_COLUMN].all())
                entities = bundle.entity_table().sort("serial_number")
                self.assertEqual(entities["duration"].to_list(), [2, 2])
                self.assertEqual(
                    entities[EVENT_OBSERVED_COLUMN].to_list(), [True, False]
                )
                restored = bundle.test_eval_split()
                self.assertEqual(restored.y().to_list(), [2, 1, 0])
                self.assertFalse(restored.frame[EVENT_OBSERVED_COLUMN].any())
                self.assertEqual(
                    eval_backblaze(bundle, [2, 1, 0])["mae"], 0.0
                )
