import unittest

import numpy as np

from pdmdata.tasks.tte import HigsSpec, higs_corrupt


class HigsTests(unittest.TestCase):
    def test_seed_masks_times_and_label_free_mode(self):
        x = np.arange(420, dtype=np.float32).reshape(30, 14) / 420
        spec = HigsSpec(noise_rul_factor=0)
        first = higs_corrupt(x, spec=spec, rng=np.random.default_rng(19))
        second = higs_corrupt(x, spec=spec, rng=np.random.default_rng(19), target_cycles=12)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
        values, mask, times = first
        self.assertTrue(np.isin(mask, [0, 1]).all())
        self.assertTrue((values[mask == 0] == 0).all())
        self.assertTrue((np.diff(times) >= 0).all())

    def test_label_dependent_noise_is_explicit(self):
        with self.assertRaises(ValueError):
            higs_corrupt(np.ones((30, 14), dtype=np.float32))
        a = higs_corrupt(np.ones((30, 14)), rng=np.random.default_rng(11), target_cycles=0)
        b = higs_corrupt(np.ones((30, 14)), rng=np.random.default_rng(11), target_cycles=125)
        np.testing.assert_array_equal(a[1], b[1])
        np.testing.assert_array_equal(a[2], b[2])
        self.assertGreater(np.max(np.abs(a[0] - b[0])), 0)

    def test_all_missing_is_preserved_and_no_noise_is_identity(self):
        x = np.arange(60, dtype=np.float32).reshape(30, 2)
        missing = higs_corrupt(x, spec=HigsSpec(sensor_drop=1, noise_rul_factor=0), rng=np.random.default_rng(7))
        self.assertTrue((missing[0] == 0).all())
        spec = HigsSpec(sensor_drop=0, burst_probability=0, jitter_scale=0, base_noise_sigma=0, noise_rul_factor=0)
        values, mask, time = higs_corrupt(x, spec=spec, rng=np.random.default_rng(7))
        np.testing.assert_array_equal(values, x)
        np.testing.assert_array_equal(mask, np.ones_like(x))
        np.testing.assert_allclose(time, np.arange(30) / 30)


if __name__ == "__main__":
    unittest.main()
