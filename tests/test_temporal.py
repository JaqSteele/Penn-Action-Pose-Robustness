import unittest

import numpy as np

import _common  # noqa: F401  (puts src on the path)
from penn_pose import temporal


def _seq(n, seed=0):
    return np.random.default_rng(seed).normal(size=(n, 12, 2)) * 50 + 200


def _reference(seq, length=64):
    """Independent reference: np.interp per coordinate at t_k = k (N - 1) / (length - 1)."""
    n = seq.shape[0]
    t = np.array([k * (n - 1) / (length - 1) for k in range(length)])
    flat = seq.reshape(n, -1)
    out = np.column_stack([np.interp(t, np.arange(n), flat[:, c]) for c in range(flat.shape[1])])
    return out.reshape((length,) + seq.shape[1:])


class TestResample(unittest.TestCase):
    def test_sample_positions_formula(self):
        for n in (18, 57, 64, 657):
            t = temporal.sample_positions(n)
            self.assertEqual(t.shape, (64,))
            self.assertEqual(t[0], 0.0)
            self.assertEqual(t[-1], n - 1)
            np.testing.assert_allclose(t, np.arange(64) * (n - 1) / 63, rtol=0, atol=1e-12)

    def test_lengths_match_reference_and_endpoints(self):
        for n in (18, 57, 64, 657):
            seq = _seq(n, n)
            out = temporal.resample_to_length(seq)
            self.assertEqual(out.shape, (64, 12, 2))
            np.testing.assert_allclose(out, _reference(seq), rtol=0, atol=1e-9)
            np.testing.assert_array_equal(out[0], seq[0])
            np.testing.assert_array_equal(out[-1], seq[-1])

    def test_n64_is_identity(self):
        seq = _seq(64)
        np.testing.assert_array_equal(temporal.resample_to_length(seq), seq)

    def test_single_frame_repeated(self):
        seq = _seq(1)
        out = temporal.resample_to_length(seq)
        self.assertEqual(out.shape, (64, 12, 2))
        self.assertTrue(np.all(out == seq[0]))

    def test_linear_motion_preserved(self):
        n = 57
        t = np.arange(n, dtype=float)
        seq = np.stack([3.0 + 0.5 * t, -2.0 + 1.25 * t], axis=-1)[:, None, :]   # (n, 1, 2)
        out = temporal.resample_to_length(seq)
        tk = np.arange(64) * (n - 1) / 63
        np.testing.assert_allclose(out[:, 0, 0], 3.0 + 0.5 * tk, atol=1e-12)
        np.testing.assert_allclose(out[:, 0, 1], -2.0 + 1.25 * tk, atol=1e-12)

    def test_two_frames(self):
        seq = np.array([[0.0], [63.0]])
        np.testing.assert_allclose(temporal.resample_to_length(seq)[:, 0], np.arange(64.0), atol=1e-12)

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            temporal.resample_to_length(np.zeros((0, 12, 2)))
        bad = _seq(5); bad[2, 3, 0] = np.nan
        with self.assertRaises(ValueError):
            temporal.resample_to_length(bad)

    def test_rejects_invalid_lengths(self):
        for n in (1, 2, 57):                    # including single-frame input
            for length in (1, 0, -3):
                with self.assertRaises(ValueError):
                    temporal.resample_to_length(_seq(n), length)
        for length in (1, 0):
            with self.assertRaises(ValueError):
                temporal.sample_positions(57, length)
        with self.assertRaises(ValueError):
            temporal.sample_positions(0)

    def test_minimum_length_two(self):
        seq = _seq(57)
        out = temporal.resample_to_length(seq, 2)
        np.testing.assert_array_equal(out, seq[[0, -1]])
        self.assertEqual(temporal.resample_to_length(_seq(1), 2).shape, (2, 12, 2))

    def test_input_not_modified(self):
        seq = _seq(20); before = seq.copy()
        temporal.resample_to_length(seq)
        np.testing.assert_array_equal(seq, before)


if __name__ == "__main__":
    unittest.main()
