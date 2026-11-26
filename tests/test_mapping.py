import unittest

import numpy as np

import _common  # noqa: F401  (adds src/ to the path)
from penn_pose import mapping


class TestMapping(unittest.TestCase):
    def test_primary_order_is_coco_5_to_16(self):
        self.assertEqual(mapping.COCO_INDEX_FOR_PRIMARY, tuple(range(5, 17)))
        self.assertEqual(mapping.PRIMARY_JOINTS, mapping.COCO_KEYPOINTS[5:17])

    def test_penn_columns_follow_protocol_table(self):
        # Protocol Section 6.2: each anatomical joint comes from the opposite-labelled Penn column.
        expected = {
            "left_shoulder": 2, "right_shoulder": 1, "left_elbow": 4, "right_elbow": 3,
            "left_wrist": 6, "right_wrist": 5, "left_hip": 8, "right_hip": 7,
            "left_knee": 10, "right_knee": 9, "left_ankle": 12, "right_ankle": 11,
        }
        got = dict(zip(mapping.PRIMARY_JOINTS, mapping.PENN_COLUMN_FOR_PRIMARY))
        self.assertEqual(got, expected)
        self.assertNotIn(0, mapping.PENN_COLUMN_FOR_PRIMARY, "head must be excluded")

    def test_every_pair_is_swapped(self):
        for j, col in zip(mapping.PRIMARY_JOINTS, mapping.PENN_COLUMN_FOR_PRIMARY):
            penn_label = mapping.PENN_COLUMN_LABELS[col]
            self.assertNotEqual(j.split("_")[0], penn_label.split("_")[0], (j, penn_label))
            self.assertEqual(j.split("_", 1)[1], penn_label.split("_", 1)[1])

    def test_partner_is_involution(self):
        for j, p in enumerate(mapping.PARTNER):
            self.assertNotEqual(j, p)
            self.assertEqual(mapping.PARTNER[p], j)
            self.assertEqual(mapping.PRIMARY_JOINTS[j].split("_", 1)[1],
                             mapping.PRIMARY_JOINTS[p].split("_", 1)[1])

    def test_penn_to_primary_values(self):
        x = np.arange(13, dtype=float)[None, :] + np.array([[0.0], [100.0]])
        y = x + 1000
        out = mapping.penn_to_primary(x, y)
        self.assertEqual(out.shape, (2, 12, 2))
        for j, col in enumerate(mapping.PENN_COLUMN_FOR_PRIMARY):
            np.testing.assert_array_equal(out[:, j, 0], x[:, col])
            np.testing.assert_array_equal(out[:, j, 1], y[:, col])

    def test_coco_to_primary(self):
        kp = np.random.default_rng(0).normal(size=(3, 17, 2))
        np.testing.assert_array_equal(mapping.coco_to_primary(kp), kp[:, 5:17])
        with self.assertRaises(ValueError):
            mapping.coco_to_primary(kp, keypoint_names=list(reversed(mapping.COCO_KEYPOINTS)))

    def test_bad_shapes_rejected(self):
        with self.assertRaises(ValueError):
            mapping.penn_to_primary(np.zeros((2, 12)), np.zeros((2, 12)))


if __name__ == "__main__":
    unittest.main()
