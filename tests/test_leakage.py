"""Batch D1 leakage and test-isolation checks (protocol Sections 4.1-4.3, D-07a rule 4)."""

import re
import unittest

import _common
from penn_pose import dataset, est_preprocess, labels, splits

D1_FILES = [
    _common.REPO_ROOT / "src" / "penn_pose" / "temporal.py",
    _common.REPO_ROOT / "src" / "penn_pose" / "est_preprocess.py",
    _common.REPO_ROOT / "src" / "penn_pose" / "dataset.py",
    _common.REPO_ROOT / "scripts" / "build_inputs.py",
]


class TestStaticSource(unittest.TestCase):
    def test_no_test_partition_opt_in(self):
        for path in D1_FILES:
            text = path.read_text(encoding="utf-8")
            self.assertNotRegex(text, r"allow_test\s*=\s*True", path.name)
            self.assertNotRegex(text, r"list_video_ids\(\s*[\"'](test|all)[\"']", path.name)
            self.assertNotRegex(text, r"--allow-test", path.name)

    def test_estimated_pipeline_never_reads_gt_joints_or_visibility(self):
        text = (_common.REPO_ROOT / "src" / "penn_pose" / "est_preprocess.py").read_text(encoding="utf-8")
        code = re.sub(r'"""(.|\n)*?"""', "", text)          # ignore docstrings
        for pattern in (r"\.x\b", r"\.y\b", r"\.visibility\b", r"penn_to_primary", r"gt_preprocess",
                        r"placeholder"):
            self.assertNotRegex(code, pattern)

    def test_default_pose_dir_is_training(self):
        self.assertEqual(est_preprocess.DEFAULT_POSE_DIR.name, "train")


@_common.requires_data
class TestRuntimeLock(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        flags = labels.partition_flags(_common.DATA_ROOT)       # reads only the `train` variable
        test_ids = [v for v, f in flags.items() if f == labels.TEST_FLAG]
        cls.test_ids = [test_ids[0], test_ids[len(test_ids) // 2], test_ids[-1]]

    def test_build_inputs_rejects_test_ids(self):
        for source in dataset.SOURCES:
            for vid in self.test_ids:
                with self.assertRaises(labels.TestPartitionAccessError):
                    dataset.build_inputs(source, ["0002", vid], _common.DATA_ROOT, _common.POSE_DIR)

    def test_prepare_estimated_rejects_test_ids(self):
        for vid in self.test_ids:
            with self.assertRaises(labels.TestPartitionAccessError):
                est_preprocess.prepare_estimated(vid, _common.DATA_ROOT, _common.POSE_DIR)

    def test_no_test_pose_outputs_exist(self):
        self.assertFalse((_common.DATA_ROOT / "poses" / "rtmpose_m" / "test").exists())

    def test_split_disjoint_and_complete(self):
        fitting, validation = splits.load_split("fitting"), splits.load_split("validation")
        self.assertFalse(set(fitting) & set(validation))
        self.assertEqual(sorted(fitting + validation), labels.list_video_ids("train", _common.DATA_ROOT))
        membership = dataset.training_split()
        self.assertEqual(len(membership), 1258)


if __name__ == "__main__":
    unittest.main()
