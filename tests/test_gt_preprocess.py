import unittest

import numpy as np

import _common
from penn_pose import gt_preprocess, mapping


def _pose(n=5, value=100.0):
    j = np.full((n, mapping.N_PRIMARY, 2), value)
    j[..., 0] += np.arange(mapping.N_PRIMARY)          # distinct x per joint
    j[..., 1] += np.arange(n)[:, None]                  # moves over time
    return j


class TestRepairRules(unittest.TestCase):
    centre = np.array([240.0, 180.0])

    def test_non_placeholders_untouched(self):
        j = _pose()
        out, rep = gt_preprocess.repair_placeholders(j, self.centre)
        np.testing.assert_array_equal(out, j)
        self.assertEqual(rep.n_placeholder_points, 0)

    def test_placeholder_definition(self):
        j = _pose(3)
        j[0, 0] = [0.0, 0.0]; j[1, 0] = [0.75, 0.67]; j[2, 0] = [0.9, 1.0]   # last: y not < 1
        np.testing.assert_array_equal(gt_preprocess.placeholder_mask(j)[:, 0], [True, True, False])

    def test_partly_track_interpolated_and_held(self):
        j = _pose(6)
        orig = j.copy()
        j[[0, 2, 3, 5], 4] = [0.75, 0.75]       # wrist placeholders: start, interior pair, end
        out, rep = gt_preprocess.repair_placeholders(j, self.centre)
        self.assertEqual(rep.partly_tracks, [4])
        np.testing.assert_allclose(out[0, 4], orig[1, 4])          # held at start
        np.testing.assert_allclose(out[5, 4], orig[4, 4])          # held at end
        np.testing.assert_allclose(out[2, 4], orig[1, 4] + (orig[4, 4] - orig[1, 4]) / 3)
        np.testing.assert_allclose(out[3, 4], orig[1, 4] + 2 * (orig[4, 4] - orig[1, 4]) / 3)

    def test_entire_track_copies_clean_partner(self):
        j = _pose()
        j[:, 10] = [0.75, 0.75]                 # left_ankle entirely placeholder
        out, rep = gt_preprocess.repair_placeholders(j, self.centre)
        self.assertEqual(rep.partner_copied, [10])
        np.testing.assert_array_equal(out[:, 10], j[:, mapping.PARTNER[10]])

    def test_entire_track_copies_partner_after_rule3(self):
        j = _pose()
        p = mapping.PARTNER[10]
        j[:, 10] = [0.75, 0.75]
        j[2, p] = [0.5, 0.5]                    # partner partly placeholder -> repaired first
        out, rep = gt_preprocess.repair_placeholders(j, self.centre)
        self.assertEqual(rep.partner_copied, [10])
        self.assertEqual(rep.partly_tracks, [p])
        np.testing.assert_array_equal(out[:, 10], out[:, p])
        self.assertFalse(gt_preprocess.placeholder_mask(out).any())

    def test_both_partners_entire_use_video_centre(self):
        j = _pose()
        j[:, 10] = [0.75, 0.75]; j[:, 11] = [0.67, 0.67]
        out, rep = gt_preprocess.repair_placeholders(j, self.centre)
        self.assertEqual(sorted(rep.centre_fallback), [10, 11])
        self.assertTrue(np.all(out[:, 10] == self.centre) and np.all(out[:, 11] == self.centre))

    def test_input_not_modified(self):
        j = _pose(); j[:, 0] = [0.5, 0.5]
        before = j.copy()
        gt_preprocess.repair_placeholders(j, self.centre)
        np.testing.assert_array_equal(j, before)


@_common.requires_data
class TestRepairCountsOnTraining(unittest.TestCase):
    """Protocol Section 5.5.3: training-partition counts."""

    @classmethod
    def setUpClass(cls):
        cls.reports = {vid: gt_preprocess.prepare_gt(v).report for vid, v in _common.training_videos().items()}
        cls.summary = gt_preprocess.summarise_reports(cls.reports)

    def test_counts(self):
        s = self.summary
        self.assertEqual(s["videos"], 1258)
        self.assertEqual(s["partly_tracks"], 14)
        self.assertEqual(s["partner_copied"], 601)
        self.assertEqual(s["centre_fallback"], 494)
        self.assertEqual(s["partner_copied"] + s["centre_fallback"], 1095)
        self.assertEqual(s["placeholder_points"], 78166)

    def test_no_placeholder_remains(self):
        for vid, v in _common.training_videos().items():
            g = gt_preprocess.prepare_gt(v)
            self.assertFalse(gt_preprocess.placeholder_mask(g.joints_repaired).any(), vid)
            self.assertTrue(np.all(np.isfinite(g.joints_normalised)), vid)

    def test_visible_joints_are_never_placeholders(self):
        for vid, v in _common.training_videos().items():
            g = gt_preprocess.prepare_gt(v)
            self.assertFalse((gt_preprocess.placeholder_mask(g.joints_raw) & g.visibility).any(), vid)


if __name__ == "__main__":
    unittest.main()
