import unittest

import numpy as np

import _common  # noqa: F401
from penn_pose import e1, mapping


class TestE1(unittest.TestCase):
    def test_longest_run(self):
        self.assertEqual(e1.longest_run([0, 1, 1, 0, 1, 1, 1, 0]), 3)
        self.assertEqual(e1.longest_run([0, 0]), 0)

    def test_video_diagnostics(self):
        v = np.ones((6, 12), dtype=bool)
        v[1:4, 0] = False                  # left_shoulder: run of 3 invalid frames
        v[:, 10] = False                   # left_ankle never valid; partner right_ankle valid
        d = e1.video_diagnostics(v)
        self.assertEqual(d["n_invalid"], 3 + 6)
        self.assertEqual(d["longest_invalid_run_by_joint"]["left_shoulder"], 3)
        self.assertEqual(d["never_valid_tracks"], ["left_ankle"])
        self.assertEqual(d["never_valid_tracks_partner_also_never"], [])
        self.assertEqual(d["frames_without_valid_primary"], 0)
        self.assertFalse(d["needs_step4"])
        self.assertAlmostEqual(d["invalid_rate_by_joint"]["left_ankle"], 1.0)

    def test_step3_and_step4_flags(self):
        v = np.ones((4, 12), dtype=bool)
        v[:, 10] = v[:, 11] = False        # both ankles never valid -> step 3
        self.assertEqual(sorted(e1.video_diagnostics(v)["never_valid_tracks_partner_also_never"]),
                         ["left_ankle", "right_ankle"])
        none = np.zeros((3, 12), dtype=bool)
        d = e1.video_diagnostics(none)
        self.assertTrue(d["needs_step4"])
        self.assertEqual(d["frames_without_valid_primary"], 3)

    def test_summary_pools_correctly(self):
        a = np.ones((10, 12), dtype=bool); a[:5, 2] = False
        b = np.ones((30, 12), dtype=bool)
        s = e1.summarise({"0001": e1.video_diagnostics(a), "0002": e1.video_diagnostics(b)})
        self.assertEqual(s["frames"], 40)
        self.assertEqual(s["invalid_keypoints"], 5)
        self.assertAlmostEqual(s["invalid_rate_by_joint"][mapping.PRIMARY_JOINTS[2]], 5 / 40)
        self.assertEqual(s["videos_needing_steps_2_or_3"], [])

    def test_shape_checked(self):
        with self.assertRaises(ValueError):
            e1.video_diagnostics(np.ones((3, 17), dtype=bool))


if __name__ == "__main__":
    unittest.main()
