import dataclasses
import unittest

import numpy as np

import _common
from penn_pose import boxes, est_preprocess, labels, mapping, pose_io

SIZE = np.array([480, 360])
E1_VIDEOS = ["0359", "0362", "0373", "0462", "0710", "1516", "1905"]


def _est(n=6):
    kp = np.zeros((n, mapping.N_PRIMARY, 2))
    kp[..., 0] = 100.0 + 10.0 * np.arange(mapping.N_PRIMARY)       # distinct x per joint
    kp[..., 1] = 50.0 + 3.0 * np.arange(n)[:, None]                 # moves over time
    return kp, np.ones((n, mapping.N_PRIMARY), dtype=bool)


def _invalidate(kp, valid, frames, joint, sentinel=(-0.5, -0.5)):
    """Mark entries invalid and give them a sentinel location that must never survive."""
    valid[frames, joint] = False
    kp[frames, joint] = sentinel


class TestImputationSteps(unittest.TestCase):
    def test_all_valid_unchanged(self):
        kp, valid = _est()
        out, rep = est_preprocess.impute_estimates(kp, valid, SIZE)
        np.testing.assert_array_equal(out, kp)
        self.assertEqual((rep.n_invalid, rep.step1_points, rep.step2_tracks, rep.step3_tracks, rep.step4_video),
                         (0, 0, [], [], False))

    def test_step1_interior_start_and_end(self):
        kp, valid = _est(8)
        orig = kp.copy()
        _invalidate(kp, valid, [0, 1], 2)          # start gap
        _invalidate(kp, valid, [3, 4], 2)          # interior gap
        _invalidate(kp, valid, [7], 2)             # end gap
        out, rep = est_preprocess.impute_estimates(kp, valid, SIZE)
        np.testing.assert_array_equal(out[[0, 1], 2], orig[[2, 2], 2])              # held at start
        np.testing.assert_array_equal(out[7, 2], orig[6, 2])                         # held at end
        np.testing.assert_allclose(out[3, 2], orig[2, 2] + (orig[5, 2] - orig[2, 2]) / 3)
        np.testing.assert_allclose(out[4, 2], orig[2, 2] + 2 * (orig[5, 2] - orig[2, 2]) / 3)
        self.assertEqual((rep.step1_points, rep.step1_tracks), (5, [2]))
        np.testing.assert_array_equal(out[valid], kp[valid])                         # valid untouched

    def test_step1_ignores_nonfinite_invalid(self):
        kp, valid = _est()
        _invalidate(kp, valid, [2], 0, sentinel=(np.nan, np.inf))
        out, _ = est_preprocess.impute_estimates(kp, valid, SIZE)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_step2_copies_partner_after_step1(self):
        kp, valid = _est()
        j, p = 4, mapping.PARTNER[4]
        _invalidate(kp, valid, slice(None), j)       # never valid
        _invalidate(kp, valid, [1], p)               # partner has a gap, filled by step 1 first
        out, rep = est_preprocess.impute_estimates(kp, valid, SIZE)
        self.assertEqual((rep.step2_tracks, rep.step1_tracks, rep.step3_tracks), ([j], [p], []))
        np.testing.assert_array_equal(out[:, j], out[:, p])
        self.assertEqual(rep.step2_points, 6)

    def test_step3_uses_original_valid_observations_only(self):
        kp, valid = _est()
        _invalidate(kp, valid, slice(None), 10)
        _invalidate(kp, valid, slice(None), 11)      # both ankles never valid
        _invalidate(kp, valid, [0, 1, 2], 0, sentinel=(1e6, 1e6))   # step-1 track with sentinels
        expected = kp[valid].mean(axis=0)            # only original valid observations
        out, rep = est_preprocess.impute_estimates(kp, valid, SIZE)
        self.assertEqual(sorted(rep.step3_tracks), [10, 11])
        np.testing.assert_allclose(out[:, 10], np.tile(expected, (6, 1)))
        np.testing.assert_allclose(out[:, 11], np.tile(expected, (6, 1)))
        # Including imputed values (all frames of joint 0) would give a different mean.
        with_imputed = np.concatenate([out[:, :10].reshape(-1, 2)])
        self.assertFalse(np.allclose(with_imputed.mean(axis=0), expected))
        np.testing.assert_allclose(rep.step3_mean, expected)

    def test_step4_image_centre(self):
        kp, valid = _est()
        valid[:] = False
        out, rep = est_preprocess.impute_estimates(kp, valid, np.array([482 - 2, 351]))
        self.assertTrue(rep.step4_video)
        self.assertTrue(np.all(out == np.array([240.0, 175.5])))
        self.assertEqual((rep.step4_points, rep.step1_points, rep.step2_tracks, rep.step3_tracks),
                         (6 * 12, 0, [], []))

    def test_single_frame_video(self):
        # All valid: unchanged.
        kp, valid = _est(1)
        out, rep = est_preprocess.impute_estimates(kp, valid, SIZE)
        np.testing.assert_array_equal(out, kp)
        self.assertEqual((rep.n_frames, rep.n_invalid), (1, 0))
        # Invalid joints in the only frame are never-valid tracks: step 1 cannot apply.
        kp, valid = _est(1)
        _invalidate(kp, valid, [0], 4)                      # partner 5 valid -> step 2
        _invalidate(kp, valid, [0], 10)
        _invalidate(kp, valid, [0], 11)                     # both ankles -> step 3
        expected_mean = kp[valid].mean(axis=0)
        out, rep = est_preprocess.impute_estimates(kp, valid, SIZE)
        self.assertEqual((rep.step1_points, rep.step2_tracks, sorted(rep.step3_tracks), rep.step4_video),
                         (0, [4], [10, 11], False))
        np.testing.assert_array_equal(out[0, 4], kp[0, mapping.PARTNER[4]])
        np.testing.assert_allclose(out[0, 10], expected_mean)
        np.testing.assert_allclose(out[0, 11], expected_mean)
        # Fully invalid single frame: step 4.
        kp, valid = _est(1)
        valid[:] = False
        out, rep = est_preprocess.impute_estimates(kp, valid, SIZE)
        self.assertTrue(rep.step4_video)
        self.assertTrue(np.all(out == np.array([240.0, 180.0])))
        self.assertEqual(rep.step4_points, 12)

    def test_step2_and_step3_in_one_video(self):
        kp, valid = _est(7)
        _invalidate(kp, valid, slice(None), 0)              # left_shoulder never valid
        p0 = mapping.PARTNER[0]
        _invalidate(kp, valid, [3], p0, sentinel=(9e5, 9e5))   # partner gap filled by step 1
        _invalidate(kp, valid, slice(None), 8)
        _invalidate(kp, valid, slice(None), 9)              # both knees never valid -> step 3
        expected_mean = kp[valid].mean(axis=0)              # original valid observations only
        out, rep = est_preprocess.impute_estimates(kp, valid, SIZE)
        self.assertEqual((rep.step1_tracks, rep.step2_tracks, sorted(rep.step3_tracks)), ([p0], [0], [8, 9]))
        self.assertEqual((rep.step1_points, rep.step2_points, rep.step3_points), (1, 7, 14))
        np.testing.assert_array_equal(out[:, 0], out[:, p0])            # copied after step 1
        self.assertFalse(np.any(out == 9e5))
        np.testing.assert_allclose(out[:, 8], np.tile(expected_mean, (7, 1)))
        np.testing.assert_allclose(out[:, 9], np.tile(expected_mean, (7, 1)))
        # The step-3 mean must not include step-1 or step-2 values: a mean over the imputed
        # tracks (all joints except the step-3 ones) would differ.
        others = [i for i in range(mapping.N_PRIMARY) if i not in (8, 9)]
        self.assertFalse(np.allclose(out[:, others].reshape(-1, 2).mean(axis=0), expected_mean))
        np.testing.assert_allclose(rep.step3_mean, expected_mean)

    def test_partner_with_exactly_one_valid_observation(self):
        kp, valid = _est(5)
        j, p = 2, mapping.PARTNER[2]
        _invalidate(kp, valid, slice(None), j)              # never valid
        only = kp[3, p].copy()
        _invalidate(kp, valid, [0, 1, 2, 4], p)            # partner valid only at frame 3
        out, rep = est_preprocess.impute_estimates(kp, valid, SIZE)
        self.assertEqual((rep.step1_tracks, rep.step1_points, rep.step2_tracks, rep.step3_tracks), ([p], 4, [j], []))
        self.assertTrue(np.all(out[:, p] == only))          # held in both directions
        self.assertTrue(np.all(out[:, j] == only))          # step 2 copies the step-1 track

    def test_no_sentinel_survives(self):
        kp, valid = _est()
        _invalidate(kp, valid, [0, 5], 3, sentinel=(-12345.0, -12345.0))
        out, _ = est_preprocess.impute_estimates(kp, valid, SIZE)
        self.assertFalse(np.any(out == -12345.0))

    def test_input_not_modified(self):
        kp, valid = _est(); _invalidate(kp, valid, [1], 1)
        kb, vb = kp.copy(), valid.copy()
        est_preprocess.impute_estimates(kp, valid, SIZE)
        np.testing.assert_array_equal(kp, kb); np.testing.assert_array_equal(valid, vb)

    def test_rejects_bad_inputs(self):
        kp, valid = _est()
        bad = kp.copy(); bad[0, 0, 0] = np.nan          # valid but non-finite
        with self.assertRaises(ValueError):
            est_preprocess.impute_estimates(bad, valid, SIZE)
        with self.assertRaises(ValueError):
            est_preprocess.impute_estimates(kp, valid.astype(int), SIZE)
        for size in (np.array([480.0, 360.0]), np.array([0, 360]), np.array([480])):
            with self.assertRaises(ValueError):
                est_preprocess.impute_estimates(kp, valid, size)


def _video(n=6, w=480, h=360, bbox=None):
    nan = np.full((n, 13), np.nan)
    return labels.PennVideo(video_id="0001", action="bowl", pose="front", is_train=True,
                            x=nan, y=nan, visibility=np.zeros((n, 13), bool),
                            bbox=bbox if bbox is not None else np.tile([100.0, 50.0, 300.0, 330.0], (n, 1)),
                            n_frames=n, image_width=w, image_height=h, dims_width=w)


def _arrays(video):
    kp, valid = _est(video.n_frames)
    return {"frame_index": np.arange(1, video.n_frames + 1), "image_size": np.array([video.image_width, video.image_height]),
            "bbox_annotation": boxes.align_boxes(video.bbox, video.n_frames), "keypoints12": kp, "valid12": valid}


class TestConsistency(unittest.TestCase):
    def test_synthetic_pipeline_uses_boxes_not_joints(self):
        video = _video()                                   # GT joints are NaN: must not matter
        prep = est_preprocess.prepare_estimated_from(video, _arrays(video))
        frame = boxes.video_centre_scale(video.bbox, 6, 480, 360)
        np.testing.assert_allclose(prep.keypoints_normalised, (prep.keypoints_imputed - frame.centre) / frame.scale)
        self.assertTrue(np.all(np.isfinite(prep.keypoints_normalised)))

    def test_mismatches_rejected(self):
        video = _video()
        a = _arrays(video); a["frame_index"] = np.arange(1, 6)
        with self.assertRaises(est_preprocess.EstimateConsistencyError):
            est_preprocess.check_consistency(video, a)
        a = _arrays(video); a["image_size"] = np.array([482, 360])
        with self.assertRaises(est_preprocess.EstimateConsistencyError):
            est_preprocess.check_consistency(video, a)
        a = _arrays(video); a["bbox_annotation"] = a["bbox_annotation"] + 1
        with self.assertRaises(est_preprocess.EstimateConsistencyError):
            est_preprocess.check_consistency(video, a)

    def test_missing_box_row_matches_nan_alignment(self):
        video = _video(bbox=np.tile([100.0, 50.0, 300.0, 330.0], (5, 1)))   # one row short
        est_preprocess.check_consistency(video, _arrays(video))

    def test_non_training_video_rejected(self):
        video = dataclasses.replace(_video(), is_train=False)
        with self.assertRaises(labels.TestPartitionAccessError):
            est_preprocess.prepare_estimated_from(video, _arrays(video))


@_common.requires_poses
class TestOnTrainingPartition(unittest.TestCase):
    """Independent check of the imputation against measurement E1 (protocol Section 5.5.4)."""

    @classmethod
    def setUpClass(cls):
        cls.videos = _common.training_videos()
        cls.prepared = {}
        for vid, video in cls.videos.items():
            arrays, _ = pose_io.load_video_output(_common.POSE_DIR, vid)
            cls.prepared[vid] = est_preprocess.prepare_estimated_from(video, arrays)
        from penn_pose import splits
        cls.fitting = set(splits.load_split("fitting"))

    def test_e1_reproduced(self):
        reps = {v: p.report for v, p in self.prepared.items()}
        total = est_preprocess.summarise_reports(reps)
        fit = est_preprocess.summarise_reports({v: r for v, r in reps.items() if v in self.fitting})
        val = est_preprocess.summarise_reports({v: r for v, r in reps.items() if v not in self.fitting})
        self.assertEqual(total["videos"], 1258)
        self.assertEqual((fit["videos"], fit["frames"]), (1070, 73793))
        self.assertEqual(fit["frames"] * 12, 885516)
        for s, step1 in ((total, 20), (fit, 20), (val, 0)):
            self.assertEqual(s["invalid_observations"], step1)
            self.assertEqual(s["step1_points"], step1)
            self.assertEqual((s["step2_points"], s["step3_points"], s["step4_points"]), (0, 0, 0))
        self.assertEqual(total["step1_videos"], E1_VIDEOS)

    def test_per_video_counts_and_valid_untouched(self):
        for vid, p in self.prepared.items():
            self.assertEqual(p.report.step1_points, int((~p.valid).sum()), vid)
            np.testing.assert_array_equal(p.keypoints_imputed[p.valid], p.keypoints_raw[p.valid])
            self.assertTrue(np.all(np.isfinite(p.keypoints_normalised)), vid)

    def test_same_normalisation_frame_as_ground_truth(self):
        from penn_pose import gt_preprocess
        for vid in ("0002", "1516", "1865", "2173"):
            g = gt_preprocess.prepare_gt(self.videos[vid]).frame
            e = self.prepared[vid].frame
            np.testing.assert_array_equal(g.centre, e.centre)
            self.assertEqual(g.scale, e.scale)
        np.testing.assert_allclose(self.prepared["1516"].frame.centre, [240.0, 175.5])

    def test_ground_truth_joints_and_visibility_not_used(self):
        for vid in ("0359", "1516", "0002"):
            video = self.videos[vid]
            poisoned = dataclasses.replace(video, x=np.full_like(video.x, np.nan), y=np.full_like(video.y, np.nan),
                                           visibility=~video.visibility)
            arrays, _ = pose_io.load_video_output(_common.POSE_DIR, vid)
            out = est_preprocess.prepare_estimated_from(poisoned, arrays)
            np.testing.assert_array_equal(out.keypoints_normalised, self.prepared[vid].keypoints_normalised)

    def test_prepare_estimated_loads_from_disk(self):
        p = est_preprocess.prepare_estimated("0359", _common.DATA_ROOT, _common.POSE_DIR)
        np.testing.assert_array_equal(p.keypoints_normalised, self.prepared["0359"].keypoints_normalised)


if __name__ == "__main__":
    unittest.main()
