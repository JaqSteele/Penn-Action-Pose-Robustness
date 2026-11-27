import unittest

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import _common  # noqa: F401
from penn_pose import metrics


def _sk_macro(y, p):
    return f1_score(y, p, labels=list(range(15)), average="macro", zero_division=0)


class TestMetrics(unittest.TestCase):
    def test_matches_scikit_learn_on_random_predictions(self):
        rng = np.random.default_rng(0)
        for trial in range(200):
            n = int(rng.integers(15, 200))
            y = rng.integers(0, 15, n)
            p = rng.integers(0, 6 if trial % 2 else 15, n)          # often leaves classes unpredicted
            self.assertAlmostEqual(metrics.macro_f1(y, p), _sk_macro(y, p), places=12)
            self.assertEqual(metrics.accuracy(y, p), accuracy_score(y, p))
            np.testing.assert_array_equal(metrics.confusion(y, p), confusion_matrix(y, p, labels=list(range(15))))

    def test_all_fifteen_classes_and_zero_division(self):
        y = np.array([0, 0, 1, 1, 2, 2])
        p = np.array([0, 0, 1, 1, 1, 1])                             # class 2 never predicted
        f1 = metrics.per_class_f1(y, p)
        self.assertEqual(f1.shape, (15,))
        self.assertEqual(f1[2], 0.0)
        self.assertTrue(np.all(f1[3:] == 0.0))                       # absent and unpredicted classes score 0
        self.assertAlmostEqual(metrics.macro_f1(y, p), (1 + 2 / 3) / 15, places=15)
        self.assertAlmostEqual(metrics.macro_f1(y, p), _sk_macro(y, p), places=15)

    def test_perfect_and_float64(self):
        y = np.arange(15).repeat(3)
        self.assertEqual(metrics.macro_f1(y, y), 1.0)
        self.assertIsInstance(metrics.macro_f1(y, y), float)
        self.assertEqual(metrics.accuracy(y, y), 1.0)

    def test_majority_baseline(self):
        y = np.array([3, 3, 3, 1, 1, 14, 0])
        self.assertEqual(metrics.majority_class(y), 3)
        self.assertEqual(metrics.majority_class(np.array([5, 5, 2, 2])), 2)     # tie: smallest index
        p = metrics.majority_predictions(y, 4)
        self.assertEqual((p.tolist(), p.dtype), ([3, 3, 3, 3], np.int64))

    def test_summary(self):
        y = np.arange(15).repeat(2)
        s = metrics.summary(y, np.roll(y, 1))
        self.assertEqual(set(s), {"macro_f1", "accuracy", "per_class_f1", "confusion", "n_videos"})
        self.assertEqual((len(s["per_class_f1"]), len(s["confusion"]), s["n_videos"]), (15, 15, 30))

    def test_invalid_inputs(self):
        for y, p in (([0, 1], [0]), ([0, 15], [0, 1]), ([0, -1], [0, 1]), ([], []), ([0.0, 1.0], [0, 1])):
            with self.assertRaises(ValueError):
                metrics.macro_f1(np.array(y), np.array(p))


if __name__ == "__main__":
    unittest.main()
