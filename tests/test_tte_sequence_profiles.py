import unittest
import numpy as np
import polars as pl
from pdmdata.tasks.tte import CmapssSequenceSpec, build_sequence_samples


class SequenceProfileTests(unittest.TestCase):
    def test_default_metadata_remains_compatible(self):
        spec = CmapssSequenceSpec('FD001', ['sensor_2'])
        self.assertNotIn('smoothing_alpha', spec.to_dict())
        self.assertNotIn('short_test_policy', spec.to_dict())
        with self.assertRaises(ValueError):
            CmapssSequenceSpec('FD001', ['sensor_2'], smoothing_alpha=0)

    def test_smoothing_does_not_use_future_rows(self):
        frame = pl.DataFrame({'unit_number':[1]*7, 'cycle':list(range(1,8)),
                              'sensor_2':list(range(7)), 'time_to_event':list(range(6,-1,-1))})
        changed = frame.with_columns(pl.when(pl.col('cycle') > 4).then(10000).otherwise(pl.col('sensor_2')).alias('sensor_2'))
        arguments = (['sensor_2'], np.zeros(1), np.ones(1), 3, 1)
        a = build_sequence_samples(frame, *arguments, smoothing_alpha=.8)
        b = build_sequence_samples(changed, *arguments, smoothing_alpha=.8)
        np.testing.assert_array_equal(a['x'][:2], b['x'][:2])
        self.assertTrue(np.isfinite(a['x']).all())

    def test_short_test_line_uses_only_available_history(self):
        frame = pl.DataFrame({'unit_number':[1]*3, 'cycle':[1,2,3], 'sensor_2':[1.,3.,5.]})
        a = build_sequence_samples(frame, ['sensor_2'], np.zeros(1), np.ones(1), 5, 1,
                                  targets={1:70}, short_test_policy='linear_resample')
        np.testing.assert_allclose(a['x'][0,:,0], [1,2.2,3.4,4.6,5.8], rtol=1e-6)
        self.assertEqual(a['endpoint'][0], 3)
        self.assertEqual(a['y'][0], 70)


if __name__ == '__main__':
    unittest.main()
