import dataclasses
import unittest

import numpy as np

import _common
from penn_pose import dataset, est_preprocess, gt_preprocess, mapping, pose_io, splits, temporal


class TestLayout(unittest.TestCase):
    def test_channel_names_and_layout(self):
        self.assertEqual(dataset.N_CHANNELS, 24)
        self.assertEqual(dataset.CHANNEL_NAMES[:4], ("left_shoulder_x", "left_shoulder_y",
                                                     "right_shoulder_x", "right_shoulder_y"))
        self.assertEqual(dataset.CHANNEL_NAMES[-1], "right_ankle_y")
        t = np.arange(64)
        joints = np.empty((64, 12, 2))
        for j in range(12):
            joints[:, j, 0] = 1000 * j + t           # x encodes joint and time
            joints[:, j, 1] = -1000 * j - t - 1      # y encodes joint and time
        x = dataset.to_channels(joints)
        self.assertEqual(x.shape, (24, 64))
        for j, name in enumerate(mapping.PRIMARY_JOINTS):
            np.testing.assert_array_equal(x[2 * j], joints[:, j, 0])
            np.testing.assert_array_equal(x[2 * j + 1], joints[:, j, 1])
            self.assertEqual(dataset.CHANNEL_NAMES[2 * j], f"{name}_x")
            self.assertEqual(dataset.CHANNEL_NAMES[2 * j + 1], f"{name}_y")
        np.testing.assert_array_equal(dataset.from_channels(x), joints)

    def test_joint_order_is_coco_5_to_16(self):
        self.assertEqual(mapping.COCO_INDEX_FOR_PRIMARY, tuple(range(5, 17)))

    def test_video_input_shape_dtype(self):
        out = dataset.video_input(np.random.default_rng(0).normal(size=(57, 12, 2)))
        self.assertEqual((out.shape, out.dtype), ((24, 64), np.float32))
        with self.assertRaises(ValueError):
            dataset.video_input(np.full((10, 12, 2), np.nan))

    def test_class_map(self):
        self.assertEqual(len(dataset.CLASS_NAMES), 15)
        self.assertEqual(list(dataset.CLASS_NAMES), sorted(dataset.CLASS_NAMES))
        self.assertEqual([dataset.CLASS_INDEX[n] for n in dataset.CLASS_NAMES], list(range(15)))
        with self.assertRaises(dataset.DatasetError):
            dataset.class_names_from_actions(["bowl", "squat"])

    def test_invalid_source_and_duplicates(self):
        with self.assertRaises(ValueError):
            dataset.build_inputs("both", ["0002"])
        with self.assertRaises(dataset.DatasetError):
            dataset.check_ids(["0002", "0002"])
        with self.assertRaises(dataset.DatasetError):
            dataset.check_ids([])


@_common.requires_data
class TestClassMapOnData(unittest.TestCase):
    def test_matches_training_actions(self):
        actions = [v.action for v in _common.training_videos().values()]
        self.assertEqual(tuple(dataset.class_names_from_actions(actions)), dataset.CLASS_NAMES)


@_common.requires_poses
class TestTrainingInputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gt, cls.est = _common.training_inputs()
        cls.fitting = splits.load_split("fitting")
        cls.validation = splits.load_split("validation")

    def test_every_training_video_exactly_once(self):
        ids = self.gt.ids.tolist()
        self.assertEqual(len(ids), 1258)
        self.assertEqual(len(set(ids)), 1258)
        self.assertEqual(ids, sorted(self.fitting + self.validation))
        self.assertEqual(set(ids), set(_common.training_videos()))

    def test_shapes_dtypes_finite(self):
        for s in (self.gt, self.est):
            dataset.check_input_set(s)
            self.assertEqual(s.X.shape, (1258, 24, 64))
            self.assertEqual((s.X.dtype, s.y.dtype, s.ids.dtype.str), (np.float32, np.int64, "<U4"))
            self.assertTrue(np.all(np.isfinite(s.X)))

    def test_subsets(self):
        sub = dict(zip(self.gt.ids.tolist(), self.gt.subset.tolist()))
        self.assertTrue(all(sub[v] == "fitting" for v in self.fitting))
        self.assertTrue(all(sub[v] == "validation" for v in self.validation))
        self.assertEqual((sum(s == "fitting" for s in sub.values()), sum(s == "validation" for s in sub.values())),
                         (1070, 188))

    def test_alignment_and_labels(self):
        dataset.check_alignment(self.gt, self.est)
        vids = _common.training_videos()
        for vid, label, n in zip(self.gt.ids, self.gt.y, self.gt.n_frames):
            self.assertEqual(dataset.CLASS_NAMES[label], vids[vid].action)
            self.assertEqual(n, vids[vid].n_frames)
        self.assertEqual(int((self.gt.y == dataset.CLASS_INDEX["bowl"]).sum()), 135)
        self.assertEqual(int(np.bincount(self.gt.y).min()), 40)

    def test_alignment_detects_mismatch(self):
        broken = dataclasses.replace(self.est, n_frames=self.est.n_frames + 1)
        with self.assertRaises(dataset.DatasetError):
            dataset.check_alignment(self.gt, broken)

    def test_matches_manual_pipeline(self):
        vids = _common.training_videos()
        index = {v: i for i, v in enumerate(self.gt.ids.tolist())}
        for vid in ("0002", "0359", "1516", "1865", "2173"):
            g = gt_preprocess.prepare_gt(vids[vid]).joints_normalised
            g_expected = temporal.resample_to_length(g).reshape(64, 24).T.astype(np.float32)
            np.testing.assert_array_equal(self.gt.X[index[vid]], g_expected)
            arrays, _ = pose_io.load_video_output(_common.POSE_DIR, vid)
            e = est_preprocess.prepare_estimated_from(vids[vid], arrays).keypoints_normalised
            e_expected = temporal.resample_to_length(e).reshape(64, 24).T.astype(np.float32)
            np.testing.assert_array_equal(self.est.X[index[vid]], e_expected)
            # first and last resampled frames are the first and last original frames
            np.testing.assert_array_equal(self.est.X[index[vid]][:, 0], e[0].reshape(24).astype(np.float32))
            np.testing.assert_array_equal(self.est.X[index[vid]][:, -1], e[-1].reshape(24).astype(np.float32))

    def test_gt_and_est_are_close_for_a_visible_video(self):
        # Sanity check of shared joint order and normalisation, not a pose-error result.
        i = self.gt.ids.tolist().index("0002")
        self.assertLess(float(np.median(np.abs(self.gt.X[i] - self.est.X[i]))), 0.1)


CACHE_DIR = _common.DATA_ROOT / "inputs" / "release1"
CACHE_RECORD = _common.REPO_ROOT / "configs" / "inputs" / "release1-cache.json"


class TestCacheRecord(unittest.TestCase):
    """The durable repository record of the accepted D1 cache (read-only checks)."""

    def setUp(self):
        import json
        self.record = json.loads(CACHE_RECORD.read_text(encoding="utf-8"))

    def test_record_structure(self):
        self.assertEqual(self.record["manifest"], "inputs-sha256.txt")
        self.assertEqual(sorted(self.record["files"]),
                         ["build_report.json", "class_map.json", "est_train.npz", "gt_train.npz"])
        for digest in [self.record["manifest_sha256"], *self.record["files"].values()]:
            self.assertRegex(digest, r"^[0-9a-f]{64}$")

    @unittest.skipUnless((CACHE_DIR / "inputs-sha256.txt").is_file(), "D1 cache not present")
    def test_cache_manifest_matches_record(self):
        from penn_pose import integrity
        manifest = CACHE_DIR / "inputs-sha256.txt"
        self.assertEqual(integrity.sha256_file(manifest), self.record["manifest_sha256"])
        self.assertEqual(integrity.read_manifest(manifest), self.record["files"])


if __name__ == "__main__":
    unittest.main()
