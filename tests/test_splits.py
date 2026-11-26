import unittest

import _common
from penn_pose import labels, splits

EXPECTED_PER_CLASS = {
    "baseball_pitch": (104, 16), "baseball_swing": (115, 17), "bench_press": (69, 10),
    "bowl": (135, 20), "clean_and_jerk": (43, 6), "golf_swing": (89, 13), "jump_rope": (40, 6),
    "jumping_jacks": (56, 8), "pullup": (98, 15), "pushup": (104, 16), "situp": (50, 8),
    "squat": (114, 17), "strum_guitar": (46, 7), "tennis_forehand": (80, 12), "tennis_serve": (115, 17),
}


class TestSplitFormat(unittest.TestCase):
    def test_rounding(self):
        self.assertEqual([splits.n_validation(n) for n in (40, 43, 46, 135)], [6, 6, 7, 20])

    def test_canonical_text_and_hash(self):
        self.assertEqual(splits.canonical_text(["0001", "0002"]), "0001\n0002\n")
        self.assertEqual(splits.list_sha256([]), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")

    def test_strict_parser(self):
        self.assertEqual(splits.parse_split_text("0001\r\n0002\r\n"), ["0001", "0002"])
        for bad in ("0001\n0002", "0002\n0001\n", "0001\n0001\n", "1\n", "0001\n\n0002\n"):
            with self.assertRaises(splits.SplitError):
                splits.parse_split_text(bad)

    def test_stored_files_match_protocol(self):
        fit = splits.load_split("fitting")
        val = splits.load_split("validation")
        self.assertEqual((len(fit), len(val)), (1070, 188))
        self.assertFalse(set(fit) & set(val))


@_common.requires_data
class TestSplitGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.actions = labels.load_training_actions(_common.DATA_ROOT)
        cls.fitting, cls.validation = splits.generate_split(cls.actions)

    def test_counts(self):
        self.assertEqual(len(self.actions), 1258)
        self.assertEqual(len(self.fitting), 1070)
        self.assertEqual(len(self.validation), 188)
        self.assertEqual(sorted(self.fitting + self.validation), sorted(self.actions))

    def test_hashes(self):
        self.assertEqual(splits.list_sha256(self.validation), splits.EXPECTED_SHA256["validation"])
        self.assertEqual(splits.list_sha256(self.fitting), splits.EXPECTED_SHA256["fitting"])

    def test_per_class_counts(self):
        self.assertEqual(splits.per_class_counts(self.actions, self.validation), EXPECTED_PER_CLASS)

    def test_deterministic_and_matches_stored_files(self):
        again = splits.generate_split(self.actions)
        self.assertEqual(again, (self.fitting, self.validation))
        self.assertEqual(splits.load_split("fitting"), self.fitting)
        self.assertEqual(splits.load_split("validation"), self.validation)


if __name__ == "__main__":
    unittest.main()
