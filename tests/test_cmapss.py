"""C-MAPSS target alignment and source integrity regressions."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from pdmdata.datasets.cmapss import load, rul, inventory
from pdmdata.datasets.cmapss import loader


class CmapssTest(TestCase):
    def test_rul_alignment_selectors_and_invalid_sources(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            def rows(lengths):
                return "".join("  ".join(map(str, [unit, cycle, *([.5] * 24)])) + "  \n"
                               for unit, length in enumerate(lengths, 1)
                               for cycle in range(1, length + 1))
            train = root / "train_FD001.txt"
            test = root / "test_FD001.txt"
            target = root / "RUL_FD001.txt"
            train.write_text(rows([3, 4]))
            test.write_text(rows([2, 3]))
            target.write_text("10\n20\n")
            with patch.object(loader, "find_raw_file", side_effect=lambda _, name: root / name):
                self.assertEqual(load().width, 26)
                self.assertEqual(load(unit=2, with_rul=True)["RUL"].to_list(), [3, 2, 1, 0])
                self.assertEqual(load(split="test", unit=2, with_rul=True)["RUL"].to_list(), [22, 21, 20])
                self.assertEqual(rul()["unit_number"].to_list(), [1, 2])
                self.assertEqual(load(split="rul").columns, ["RUL"])
                self.assertEqual(load(split="rul", unit=2)["RUL"].to_list(), [20])
                self.assertEqual(inventory("FD001")["rows"].to_list(), [7, 5])
                for options in ({"unit": 3}, {"unit": 1.5}, {"subset": "FD005"}, {"split": "invalid"}):
                    with self.subTest(options=options), self.assertRaises(ValueError):
                        load(**options)
                for content in ("10\n", "10\n-1\n", "10\n20.5\n", "10\nNaN\n"):
                    target.write_text(content)
                    with self.assertRaises(ValueError):
                        load(split="test", with_rul=True)
                original = rows([3, 4])
                invalid = [original.replace("1  1  ", "1.5  1  ", 1),
                           original.replace("1  2  ", "1  1  ", 1),
                           original.replace("0.5", "inf", 1),
                           original.rstrip() + "  99\n"]
                for content in invalid:
                    train.write_text(content)
                    with self.assertRaises(ValueError):
                        load()
