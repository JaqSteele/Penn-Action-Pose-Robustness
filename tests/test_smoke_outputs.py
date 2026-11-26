"""Checks on the saved five-video smoke-test outputs (skipped if they have not been produced)."""

import unittest

import numpy as np

import _common
from penn_pose import inference_config as ic, pose_io, splits

SMOKE = _common.DATA_ROOT / "poses" / "rtmpose_m" / "smoke"


@unittest.skipUnless((SMOKE / "run1").is_dir() and _common.HAVE_DATA, "smoke-test outputs not present")
class TestSmokeOutputs(unittest.TestCase):
    def setUp(self):
        self.ids = ic.smoke_test_video_ids(splits.load_split("fitting"))

    def test_outputs_complete_and_consistent(self):
        for vid in self.ids:
            arr, meta = pose_io.load_video_output(SMOKE / "run1", vid)   # validates format and D-34a
            v = _common.training_videos()[vid]
            self.assertEqual(arr["keypoints17"].shape, (v.n_frames, 17, 2))
            self.assertEqual(arr["keypoints12"].shape, (v.n_frames, 12, 2))
            self.assertEqual(tuple(arr["image_size"]), (v.image_width, v.image_height))
            self.assertTrue(np.isfinite(arr["keypoints17"][arr["valid17"]]).all())
            self.assertEqual(meta["checkpoint_sha256"], ic.CHECKPOINT_SHA256)
            self.assertEqual(meta["settings"]["flip_test"], True)
            self.assertEqual(meta["partition"], "train")

    @unittest.skipUnless((SMOKE / "run2").is_dir(), "repeat run not present")
    def test_repeat_run_bit_identical(self):
        for vid in self.ids:
            a, _ = pose_io.load_video_output(SMOKE / "run1", vid)
            b, _ = pose_io.load_video_output(SMOKE / "run2", vid)
            for k in a:
                np.testing.assert_array_equal(a[k], b[k], err_msg=f"{vid}:{k}")


if __name__ == "__main__":
    unittest.main()
