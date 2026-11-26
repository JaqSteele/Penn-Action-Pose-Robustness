import tempfile
import unittest

import numpy as np

import _common  # noqa: F401
from penn_pose import mapping, pose_io


def _arrays(n=4, seed=0):
    rng = np.random.default_rng(seed)
    kp = rng.uniform(0, 400, size=(n, 17, 2)).astype(np.float32)
    sc = rng.uniform(0.1, 1.0, size=(n, 17)).astype(np.float32)
    sc[1, 3] = 0.0                       # invalid: score <= 0
    kp[2, 7] = np.nan                    # invalid: non-finite coordinate (left_elbow)
    v17 = pose_io.keypoint_validity(kp, sc)
    cols = list(mapping.COCO_INDEX_FOR_PRIMARY)
    return {
        "frame_index": np.arange(1, n + 1, dtype=np.int32), "image_size": np.array([480, 360], dtype=np.int32),
        "bbox_annotation": np.tile([10.0, 10.0, 200.0, 300.0], (n, 1)), "bbox_used": np.tile([10.0, 10.0, 200.0, 300.0], (n, 1)),
        "bbox_replaced": np.zeros(n, dtype=bool), "keypoints17": kp, "scores17": sc, "valid17": v17,
        "keypoints12": mapping.coco_to_primary(kp), "scores12": sc[:, cols], "valid12": v17[:, cols],
    }


class TestValidity(unittest.TestCase):
    def test_d34a_rules(self):
        kp = np.zeros((1, 6, 2))
        kp[0, 4, 1] = np.inf
        kp[0, 5, 0] = np.nan
        sc = np.array([[0.5, 0.0, -0.2, np.nan, 0.9, 0.9]])
        np.testing.assert_array_equal(pose_io.keypoint_validity(kp, sc), [[True, False, False, False, False, False]])

    def test_finite_coordinate_with_zero_score_is_invalid(self):
        self.assertFalse(pose_io.keypoint_validity(np.array([[100.0, 50.0]]), np.array([0.0]))[0])


class TestRoundTrip(unittest.TestCase):
    def test_save_load_and_no_overwrite(self):
        arr = _arrays()
        with tempfile.TemporaryDirectory() as d:
            pose_io.save_video_output(d, "0002", arr, {"video_id": "0002", "note": "test"})
            back, meta = pose_io.load_video_output(d, "0002")
            for k in arr:
                np.testing.assert_array_equal(back[k], arr[k])
                self.assertEqual(back[k].dtype, np.asarray(arr[k]).dtype)
            self.assertEqual(meta["note"], "test")
            with self.assertRaises(FileExistsError):
                pose_io.save_video_output(d, "0002", arr, {"video_id": "0002"})

    def test_inconsistent_arrays_rejected(self):
        bad_valid = _arrays(); bad_valid["valid17"] = np.ones_like(bad_valid["valid17"])
        bad_map = _arrays(); bad_map["keypoints12"] = bad_map["keypoints12"][:, ::-1]
        bad_shape = _arrays(); bad_shape["scores17"] = bad_shape["scores17"][:, :16]
        for bad in (bad_valid, bad_map, bad_shape):
            with self.assertRaises(pose_io.OutputFormatError):
                pose_io.validate_arrays(bad)


if __name__ == "__main__":
    unittest.main()
