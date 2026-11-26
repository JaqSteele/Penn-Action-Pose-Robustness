import unittest

import numpy as np

import _common
from penn_pose import gt_preprocess, normalise


class TestNormalise(unittest.TestCase):
    def test_formula(self):
        p = np.array([[[110.0, 220.0]]])
        np.testing.assert_allclose(normalise.normalise(p, [10.0, 20.0], 100.0), [[[1.0, 2.0]]])

    def test_round_trip(self):
        rng = np.random.default_rng(1)
        p = rng.uniform(-50, 500, size=(40, 12, 2))
        c, s = np.array([231.7, 180.2]), 219.3
        np.testing.assert_allclose(normalise.denormalise(normalise.normalise(p, c, s), c, s), p,
                                   rtol=0, atol=1e-9)

    def test_same_scale_for_x_and_y_and_no_per_frame_change(self):
        p = np.zeros((3, 12, 2)); p[:, :, 0] = 100; p[:, :, 1] = 100
        u = normalise.normalise(p, [0, 0], 50)
        self.assertTrue(np.allclose(u, 2.0))

    def test_invalid_scale_rejected(self):
        for bad in (0.0, -1.0, float("nan")):
            with self.assertRaises(ValueError):
                normalise.normalise(np.zeros((1, 2)), [0, 0], bad)


@_common.requires_data
class TestNormaliseOnTrainingData(unittest.TestCase):
    def test_round_trip_all_training_videos(self):
        worst = 0.0
        for v in _common.training_videos().values():
            g = gt_preprocess.prepare_gt(v)
            back = normalise.denormalise(g.joints_normalised, g.frame.centre, g.frame.scale)
            worst = max(worst, float(np.max(np.abs(back - g.joints_repaired))))
        self.assertLess(worst, 1e-9)


if __name__ == "__main__":
    unittest.main()
