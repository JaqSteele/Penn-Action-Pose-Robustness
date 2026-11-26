import unittest

import numpy as np

import _common
from penn_pose import boxes


class TestBoxRules(unittest.TestCase):
    W, H = 480, 360   # image diagonal 600, so the minimum box diagonal is 30

    def test_validity_rules(self):
        b = np.array([
            [10, 10, 110, 210],          # valid
            [10, 10, 10, 210],           # zero width
            [50, 50, 40, 60],            # inverted
            [0, 0, np.nan, 10],          # non-finite
            [0, 0, 20, 20],              # diagonal 28.3 < 30
            [0, 0, 18, 24],              # diagonal exactly 30 -> valid
        ], dtype=float)
        np.testing.assert_array_equal(boxes.valid_box_mask(b, self.W, self.H),
                                      [True, False, False, False, False, True])

    def test_alignment_missing_and_surplus_rows(self):
        b = np.array([[0, 0, 100, 100]] * 3, dtype=float)
        a = boxes.align_boxes(b, 5)
        self.assertTrue(np.isnan(a[3:]).all())
        self.assertEqual(boxes.align_boxes(b, 2).shape, (2, 4))

    def test_nearest_valid_with_earlier_tie(self):
        good1, good2 = [0, 0, 100, 100], [200, 0, 300, 100]
        b = np.array([good1, [0, 0, 0, 0], good2, [0, 0, 0, 0], [0, 0, 0, 0]], dtype=float)
        c = boxes.crop_boxes(b, 6, self.W, self.H)   # frame 5 has no row
        np.testing.assert_array_equal(c[1], good1)   # tie between frames 0 and 2 -> earlier
        np.testing.assert_array_equal(c[3], good2)
        np.testing.assert_array_equal(c[5], good2)

    def test_no_valid_box_uses_full_image(self):
        b = np.array([[0, 0, 5, 5]] * 4, dtype=float)
        c = boxes.crop_boxes(b, 4, self.W, self.H)
        np.testing.assert_array_equal(c, np.tile([0, 0, self.W, self.H], (4, 1)))
        f = boxes.video_centre_scale(b, 4, self.W, self.H)
        self.assertTrue(f.used_fallback)
        np.testing.assert_allclose(f.centre, [240, 180])
        self.assertAlmostEqual(f.scale, 600.0)

    def test_medians_use_valid_boxes_only(self):
        b = np.array([[0, 0, 30, 40], [100, 100, 130, 140], [0, 0, 60, 80], [1, 1, 1, 1]], dtype=float)
        f = boxes.video_centre_scale(b, 4, self.W, self.H)
        self.assertEqual(f.n_valid, 3)
        np.testing.assert_allclose(f.centre, [np.median([15, 115, 30]), np.median([20, 120, 40])])
        self.assertAlmostEqual(f.scale, 50.0)


@_common.requires_data
class TestVideo1516(unittest.TestCase):
    """Protocol Section 5.2.2: video 1516's placeholder box triggers the full-image fallback."""

    def test_full_image_fallback(self):
        v = _common.training_videos()["1516"]
        self.assertEqual((v.image_width, v.image_height), (480, 351))
        aligned = boxes.align_boxes(v.bbox, v.n_frames)
        self.assertFalse(boxes.valid_box_mask(aligned, v.image_width, v.image_height).any())
        f = boxes.video_centre_scale(v.bbox, v.n_frames, v.image_width, v.image_height)
        self.assertTrue(f.used_fallback)
        np.testing.assert_allclose(f.centre, [240.0, 175.5])
        self.assertAlmostEqual(f.scale, float(np.hypot(480, 351)), places=9)   # 594.6436...
        self.assertEqual(round(f.scale, 1), 594.6)                             # protocol value
        crop = boxes.crop_boxes(v.bbox, v.n_frames, v.image_width, v.image_height)
        np.testing.assert_array_equal(crop, np.tile([0, 0, 480, 351], (v.n_frames, 1)))

    def test_only_1516_lacks_valid_boxes_in_training(self):
        fallback = [vid for vid, v in _common.training_videos().items()
                    if boxes.video_centre_scale(v.bbox, v.n_frames, v.image_width, v.image_height).used_fallback]
        self.assertEqual(fallback, ["1516"])


if __name__ == "__main__":
    unittest.main()
