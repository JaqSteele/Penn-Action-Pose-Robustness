"""Independent check of the COCO keypoint names used by ``mapping``.

``configs/pose/rtmpose_m.yaml`` records ``inference.keypoint_id2name`` as read from the RTMPose
checkpoint's own metadata (verified by execution during setup). Comparing against it means a
consistent but wrong edit of both name lists in ``mapping.py`` would still be caught. The YAML is
only read here, never modified. PyYAML is already pinned in requirements.txt.
"""

import unittest

import yaml

import _common
from penn_pose import mapping

CONFIG = _common.REPO_ROOT / "configs" / "pose" / "rtmpose_m.yaml"


class TestKeypointNames(unittest.TestCase):
    def test_coco_names_match_checkpoint_record(self):
        with open(CONFIG, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        recorded = tuple(cfg["inference"]["keypoint_id2name"])
        self.assertEqual(mapping.COCO_KEYPOINTS, recorded)
        self.assertEqual(mapping.PRIMARY_JOINTS, recorded[5:17])


if __name__ == "__main__":
    unittest.main()
