import pkgutil
import subprocess
import sys
import unittest

import pdmdata.tasks


def _modules():
    yield "pdmdata.tasks"
    for info in pkgutil.walk_packages(
        pdmdata.tasks.__path__, "pdmdata.tasks."
    ):
        yield info.name


class ImportTest(unittest.TestCase):
    def test_every_task_module_imports_first_in_a_fresh_interpreter(self):
        # A fresh process per module exposes cycles that a warm
        # sys.modules (as in the rest of the suite) would hide.
        for name in _modules():
            with self.subTest(module=name):
                result = subprocess.run(
                    [sys.executable, "-c", f"import {name}"],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
