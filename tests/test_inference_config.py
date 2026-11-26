import unittest

import numpy as np

import _common
from penn_pose import boxes, inference_config as ic, labels, splits


class TestSettings(unittest.TestCase):
    def test_protocol_settings(self):
        s = ic.SETTINGS
        self.assertEqual(s["model"], "RTMPose-M")
        self.assertEqual(s["input_size_wh"], [192, 256])
        self.assertIs(s["flip_test"], True)
        self.assertEqual(s["precision"], "fp32")
        self.assertEqual(s["boxes_per_call"], 1)
        self.assertEqual(s["bbox_format"], "xyxy")
        self.assertEqual((s["cudnn_benchmark"], s["cudnn_deterministic"], s["cudnn_allow_tf32"], s["matmul_allow_tf32"]),
                         (False, True, False, False))
        self.assertEqual(ic.CHECKPOINT_SHA256, "929f00f2d84764aa0297decf13a84556642e43cbf5fca8cfb16e7326d92dfd97")

    def test_apply_determinism(self):
        import torch
        saved = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
                 torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
        try:
            flags = ic.apply_determinism(torch)
            self.assertEqual(flags, {"cudnn_benchmark": False, "cudnn_deterministic": True,
                                     "cudnn_allow_tf32": False, "matmul_allow_tf32": False})
        finally:
            (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
             torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32) = saved

    def test_smoke_selection_is_deterministic(self):
        fit = splits.load_split("fitting")
        sel = ic.smoke_test_video_ids(fit)
        self.assertEqual(sel, ["0002", "0516", "1101", "1746", "2325"])
        self.assertEqual(sel, ic.smoke_test_video_ids(list(reversed(fit))))
        self.assertTrue(set(sel) <= set(fit))


@_common.requires_data
class TestDataGuards(unittest.TestCase):
    def test_inference_ids_reject_test_partition(self):
        flags = labels.partition_flags(_common.DATA_ROOT)
        test_vid = next(v for v, f in flags.items() if f == labels.TEST_FLAG)
        self.assertEqual(ic.check_inference_ids(["0002"], _common.DATA_ROOT), ["0002"])
        with self.assertRaises(labels.TestPartitionAccessError):
            ic.check_inference_ids(["0002", test_vid], _common.DATA_ROOT)

    def test_box_integration_missing_row(self):
        v = _common.training_videos()["1865"]           # 72 box rows for 73 frames
        used = boxes.crop_boxes(v.bbox, v.n_frames, v.image_width, v.image_height)
        self.assertEqual(used.shape, (73, 4))
        np.testing.assert_array_equal(used[72], v.bbox[71])   # last frame takes the nearest valid box


if __name__ == "__main__":
    unittest.main()
