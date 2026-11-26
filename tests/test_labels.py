import unittest

import numpy as np

import _common
from penn_pose import labels


class TestValidation(unittest.TestCase):
    def test_bad_video_id_rejected(self):
        for bad in ("1", "abcd", 1516, "01516"):
            with self.assertRaises((ValueError, labels.LabelFormatError)):
                labels.load_video(bad, _common.DATA_ROOT)

    def test_missing_file_reported(self, ):
        with self.assertRaises(labels.LabelFormatError):
            labels.load_video("9999", _common.DATA_ROOT)


@_common.requires_data
class TestPartitionsAndLock(unittest.TestCase):
    """Test-partition protection (protocol Section 4.3). Only `train` flags of test files are read."""

    @classmethod
    def setUpClass(cls):
        cls.flags = labels.partition_flags(_common.DATA_ROOT)

    def test_partition_sizes(self):
        self.assertEqual(len(self.flags), 2326)
        self.assertEqual(sum(f == labels.TRAIN_FLAG for f in self.flags.values()), 1258)
        self.assertEqual(sum(f == labels.TEST_FLAG for f in self.flags.values()), 1068)

    def test_listing_test_requires_opt_in(self):
        for part in ("test", "all"):
            with self.assertRaises(labels.TestPartitionAccessError):
                labels.list_video_ids(part, _common.DATA_ROOT)
        self.assertEqual(len(labels.list_video_ids("train", _common.DATA_ROOT)), 1258)

    def test_loading_test_video_blocked_by_default(self):
        test_ids = [v for v, f in self.flags.items() if f == labels.TEST_FLAG]
        for vid in (test_ids[0], test_ids[len(test_ids) // 2], test_ids[-1]):
            with self.assertRaises(labels.TestPartitionAccessError):
                labels.load_video(vid, _common.DATA_ROOT)


@_common.requires_data
class TestTrainingLabels(unittest.TestCase):
    def test_all_training_videos_load_and_are_consistent(self):
        vids = _common.training_videos()
        self.assertEqual(len(vids), 1258)
        for vid, v in vids.items():
            self.assertTrue(v.is_train)
            self.assertEqual(v.x.shape, (v.n_frames, 13))
            self.assertEqual(v.visibility.dtype, np.bool_)
            self.assertEqual(v.bbox.shape[1], 4)
            self.assertIn(v.image_width, (v.dims_width, 480))
        self.assertEqual(len({v.action for v in vids.values()}), 15)

    def test_known_quirks(self):
        vids = _common.training_videos()
        self.assertEqual((vids["1865"].bbox.shape[0], vids["1865"].n_frames), (72, 73))
        corrected = [v for v in vids.values() if v.dims_width in (481, 482)]
        self.assertTrue(corrected and all(v.image_width == 480 for v in corrected))


if __name__ == "__main__":
    unittest.main()
