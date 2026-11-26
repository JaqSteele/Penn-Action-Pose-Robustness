import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import _common
from penn_pose import frames, labels


def _video(vid="0001", n=3, w=40, h=30):
    z = np.zeros((n, 13))
    return labels.PennVideo(video_id=vid, action="a", pose="front", is_train=True, x=z, y=z,
                            visibility=z.astype(bool), bbox=np.zeros((n, 4)), n_frames=n,
                            image_width=w, image_height=h, dims_width=w)


class TestFrameChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dir = frames.video_frame_dir("0001", self.root)
        self.dir.mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, names, size=(40, 30)):
        for n in names:
            Image.new("RGB", size).save(self.dir / n, format="JPEG")

    def test_naming_and_order(self):
        self.assertEqual(frames.frame_name(7), "000007.jpg")
        paths = frames.frame_paths("0001", 3, self.root)
        self.assertEqual([p.name for p in paths], ["000001.jpg", "000002.jpg", "000003.jpg"])

    def test_good_folder_passes(self):
        self._write(["000001.jpg", "000002.jpg", "000003.jpg"])
        r = frames.verify_video_frames(_video(), self.root)
        self.assertTrue(r.ok, r.issues)
        self.assertEqual(r.image_size, (40, 30))

    def test_missing_extra_and_unexpected(self):
        self._write(["000001.jpg", "000003.jpg", "000004.jpg"])
        (self.dir / "notes.txt").write_text("x")
        r = frames.verify_video_frames(_video(), self.root)
        text = " ".join(r.issues)
        self.assertIn("missing frames", text)
        self.assertIn("beyond nframes", text)
        self.assertIn("unexpected files", text)

    def test_wrong_size_and_unreadable(self):
        self._write(["000001.jpg", "000002.jpg"], size=(41, 30))
        (self.dir / "000003.jpg").write_bytes(b"not a jpeg")
        r = frames.verify_video_frames(_video(), self.root)
        text = " ".join(r.issues)
        self.assertIn("unreadable", text)
        self.assertIn("image size", text)

    def test_missing_folder(self):
        r = frames.verify_video_frames(_video("0002"), self.root)
        self.assertIn("frame folder missing", r.issues)


@unittest.skipUnless((_common.DATA_ROOT / "frames").is_dir(), "frames not extracted")
class TestRealFrames(_common.unittest.TestCase):
    def test_selected_training_videos_pass(self):
        for vid in ("1516", "1865", "0002"):          # 1516: placeholder box; 1865: box row missing
            r = frames.verify_video_frames(_common.training_videos()[vid], _common.DATA_ROOT)
            self.assertTrue(r.ok, (vid, r.issues))

    def test_no_test_partition_frames_extracted(self):
        flags = labels.partition_flags(_common.DATA_ROOT)
        present = {p.name for p in frames.frames_root(_common.DATA_ROOT).iterdir()}
        self.assertFalse({v for v in present if flags.get(v) == labels.TEST_FLAG})


if __name__ == "__main__":
    unittest.main()
