import inspect
import unittest

import numpy as np

import _common  # noqa: F401
from penn_pose import bootstrap, metrics

B = 300


def _labels():
    return np.repeat(np.arange(15), [6, 7, 8, 9, 10, 6, 7, 8, 9, 10, 6, 7, 8, 9, 20])


class TestResampling(unittest.TestCase):
    def test_seed_required(self):
        y = _labels()
        for bad in (None, True, -1, 1.5, "1"):
            with self.assertRaises(ValueError):
                bootstrap.resample_indices(y, 5, bad)
        for fn in (bootstrap.paired_comparison, bootstrap.condition_summary):
            self.assertIs(inspect.signature(fn).parameters["seed"].default, inspect.Parameter.empty)
        self.assertEqual(bootstrap.N_RESAMPLES, 10_000)

    def test_class_counts_preserved_and_deterministic(self):
        y = _labels()
        idx = bootstrap.resample_indices(y, 50, 123)
        self.assertEqual(idx.shape, (50, y.size))
        for row in idx:
            np.testing.assert_array_equal(np.bincount(y[row], minlength=15), np.bincount(y, minlength=15))
        np.testing.assert_array_equal(idx, bootstrap.resample_indices(y, 50, 123))
        self.assertFalse(np.array_equal(idx, bootstrap.resample_indices(y, 50, 124)))

    def test_no_default_seed_anywhere(self):
        for fn in (bootstrap.resample_indices, bootstrap.paired_comparison, bootstrap.condition_summary):
            self.assertIs(inspect.signature(fn).parameters["seed"].default, inspect.Parameter.empty, fn.__name__)


class TestConventions(unittest.TestCase):
    """D-26, D-27, D-46 and D-47 (protocol v0.15, Section 9.5.9)."""

    def test_constants(self):
        self.assertEqual(bootstrap.N_RESAMPLES, 10_000)
        self.assertEqual(bootstrap.LEVELS, {"ci95": (2.5, 97.5), "ci97_5": (1.25, 98.75)})
        self.assertEqual(bootstrap.PERCENTILE_METHOD, "linear")
        self.assertIs(type(bootstrap.SD_DDOF), int)
        self.assertEqual(bootstrap.SD_DDOF, 1)

    def test_every_percentile_call_names_the_method(self):
        import ast
        tree = ast.parse(inspect.getsource(bootstrap))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr in ("percentile", "quantile", "nanpercentile", "nanquantile")]
        self.assertTrue(calls)
        for call in calls:
            self.assertEqual(call.func.attr, "percentile")
            kw = {k.arg: k.value for k in call.keywords}
            self.assertIn("method", kw)
            self.assertIsInstance(kw["method"], ast.Name)
            self.assertEqual(kw["method"].id, "PERCENTILE_METHOD")
        stds = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr in ("std", "var")]
        for call in stds:
            kw = {k.arg: k.value for k in call.keywords}
            self.assertIsInstance(kw.get("ddof"), ast.Name)
            self.assertEqual(kw["ddof"].id, "SD_DDOF")

    def test_linear_interpolation_is_used(self):
        values = np.arange(10, dtype=float)                       # methods disagree on this input
        got = bootstrap._percentiles(values)
        for name, q in bootstrap.LEVELS.items():
            np.testing.assert_array_equal(got[name], np.percentile(values, q, method="linear"))
        self.assertAlmostEqual(got["ci95"][0], 0.225, places=12)        # (n - 1) * 0.025 = 0.225
        self.assertAlmostEqual(got["ci97_5"][1], 8.8875, places=12)    # (n - 1) * 0.9875
        self.assertNotEqual(got["ci95"], [float(v) for v in np.percentile(values, (2.5, 97.5), method="inverted_cdf")])
        self.assertNotEqual(got["ci95"], [float(v) for v in np.percentile(values, (2.5, 97.5), method="median_unbiased")])

    def test_b10000_positions(self):
        values = np.arange(10_000, dtype=float)                   # value == order statistic index
        got = bootstrap._percentiles(values)
        np.testing.assert_allclose(got["ci95"], [249.975, 9749.025], rtol=0, atol=1e-9)
        np.testing.assert_allclose(got["ci97_5"], [124.9875, 9874.0125], rtol=0, atol=1e-9)


class TestComparisons(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.y = _labels()
        self.a = {s: np.where(rng.random(self.y.size) < 0.8, self.y, rng.integers(0, 15, self.y.size)) for s in (0, 1, 2)}
        self.b = {s: np.where(rng.random(self.y.size) < 0.6, self.y, rng.integers(0, 15, self.y.size)) for s in (0, 1, 2)}

    def test_identical_conditions_give_zero(self):
        r = bootstrap.paired_comparison(self.y, self.a, self.a, seed=1, n_resamples=B)
        self.assertEqual(r["point_estimate"], 0.0)
        self.assertEqual(r["ci95"], [0.0, 0.0])

    def test_procedure(self):
        r = bootstrap.paired_comparison(self.y, self.a, self.b, seed=7, n_resamples=B)
        diffs = [metrics.macro_f1(self.y, self.a[s]) - metrics.macro_f1(self.y, self.b[s]) for s in (0, 1, 2)]
        self.assertAlmostEqual(r["point_estimate"], float(np.mean(diffs)), places=15)
        idx = bootstrap.resample_indices(self.y, B, 7)
        avg = np.mean([[metrics.macro_f1(self.y[i], self.a[s][i]) - metrics.macro_f1(self.y[i], self.b[s][i])
                        for i in idx] for s in (0, 1, 2)], axis=0)
        np.testing.assert_allclose(r["ci95"], np.percentile(avg, [2.5, 97.5]), rtol=0, atol=1e-15)
        np.testing.assert_allclose(r["ci97_5"], np.percentile(avg, [1.25, 98.75]), rtol=0, atol=1e-15)
        self.assertLessEqual(r["ci97_5"][0], r["ci95"][0])
        self.assertGreaterEqual(r["ci97_5"][1], r["ci95"][1])
        sv = r["seed_variation"]
        self.assertAlmostEqual(sv["range"], max(diffs) - min(diffs), places=15)
        self.assertAlmostEqual(sv["sd"], float(np.std(diffs, ddof=1)), places=15)
        self.assertEqual(sv["sd_ddof"], 1)
        self.assertNotAlmostEqual(sv["sd"], float(np.std(diffs, ddof=0)), places=12)   # not the population SD
        self.assertAlmostEqual(sv["sd"], float(np.sqrt(np.sum((np.array(diffs) - np.mean(diffs)) ** 2) / 2)), places=15)
        self.assertEqual(r["percentile_method"], "linear")
        np.testing.assert_allclose(r["ci95"], np.percentile(avg, [2.5, 97.5], method="linear"), rtol=0, atol=1e-15)
        self.assertEqual(set(sv["per_seed_ci95"]), {"0", "1", "2"})
        self.assertEqual((r["bootstrap_seed"], r["n_resamples"]), (7, B))
        self.assertEqual(bootstrap.paired_comparison(self.y, self.a, self.b, seed=7, n_resamples=B), r)

    def test_condition_summary(self):
        r = bootstrap.condition_summary(self.y, self.a, seed=3, n_resamples=B)
        mf = r["macro_f1"]
        self.assertAlmostEqual(mf["per_seed"]["0"]["value"], metrics.macro_f1(self.y, self.a[0]), places=15)
        self.assertAlmostEqual(mf["mean"]["value"],
                               float(np.mean([metrics.macro_f1(self.y, self.a[s]) for s in (0, 1, 2)])), places=15)
        lo, hi = r["accuracy"]["mean"]["ci95"]
        self.assertLessEqual(lo, r["accuracy"]["mean"]["value"])
        self.assertGreaterEqual(hi, r["accuracy"]["mean"]["value"])

    def test_shared_resamples_across_comparisons_and_summaries(self):
        """One seed gives identical resamples for every condition, seed, comparison and summary."""
        seed = 20261128
        idx = bootstrap.resample_indices(self.y, B, seed)
        c = {s: self.a[(s + 1) % 3] for s in (0, 1, 2)}
        for x, z in ((self.a, self.b), (c, self.b)):                  # e.g. H1 (C1 vs C2) and H2 (C3 vs C2)
            r = bootstrap.paired_comparison(self.y, x, z, seed=seed, n_resamples=B)
            avg = np.mean([[metrics.macro_f1(self.y[i], x[s][i]) - metrics.macro_f1(self.y[i], z[s][i])
                            for i in idx] for s in (0, 1, 2)], axis=0)
            np.testing.assert_array_equal(r["ci95"], np.percentile(avg, [2.5, 97.5], method="linear"))
            self.assertEqual(r["bootstrap_seed"], seed)
        summary = bootstrap.condition_summary(self.y, self.b, seed=seed, n_resamples=B)
        acc = np.mean([[metrics.accuracy(self.y[i], self.b[s][i]) for i in idx] for s in (0, 1, 2)], axis=0)
        np.testing.assert_array_equal(summary["accuracy"]["mean"]["ci95"], np.percentile(acc, [2.5, 97.5], method="linear"))
        self.assertEqual(summary["percentile_method"], "linear")
        flipped = bootstrap.paired_comparison(self.y, self.b, self.a, seed=seed, n_resamples=B)
        forward = bootstrap.paired_comparison(self.y, self.a, self.b, seed=seed, n_resamples=B)
        np.testing.assert_allclose(flipped["ci95"], [-forward["ci95"][1], -forward["ci95"][0]], rtol=0, atol=1e-12)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            bootstrap.paired_comparison(self.y, self.a, {0: self.b[0]}, seed=1, n_resamples=B)
        with self.assertRaises(ValueError):
            bootstrap.paired_comparison(self.y, {0: self.a[0][:-1]}, {0: self.b[0]}, seed=1, n_resamples=B)
        with self.assertRaises(ValueError):
            bootstrap.paired_comparison(self.y, {}, {}, seed=1, n_resamples=B)
        with self.assertRaises(ValueError):
            bootstrap.resample_indices(self.y, 0, 1)


if __name__ == "__main__":
    unittest.main()
