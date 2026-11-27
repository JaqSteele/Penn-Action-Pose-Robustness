import hashlib
import os
import unittest
from unittest import mock

import torch

import _synthetic
from penn_pose import determinism, model


class TestEnvironmentAndSettings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _synthetic.configure_cpu()

    def test_cublas_variable_set_on_import(self):
        self.assertEqual(os.environ.get("CUBLAS_WORKSPACE_CONFIG"), ":4096:8")

    def test_conflicting_cublas_value_rejected(self):
        with mock.patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}):
            with self.assertRaises(determinism.DeterminismError):
                determinism.set_cuda_environment()

    def test_cpu_configuration(self):
        s = determinism.configure("cpu")
        self.assertTrue(s["deterministic_algorithms"])
        self.assertFalse(s["deterministic_warn_only"])
        self.assertTrue(s["cudnn_deterministic"])
        self.assertFalse(s["cudnn_benchmark"] or s["cudnn_allow_tf32"] or s["matmul_allow_tf32"])
        self.assertEqual((s["num_threads"], s["num_interop_threads"], s["data_loader_workers"]), (1, 1, 0))
        self.assertEqual(s["float32_matmul_precision"], "highest")

    def test_check_detects_each_violation(self):
        for attr_set, undo in [
            (lambda: setattr(torch.backends.cudnn, "benchmark", True), lambda: setattr(torch.backends.cudnn, "benchmark", False)),
            (lambda: setattr(torch.backends.cudnn, "allow_tf32", True), lambda: setattr(torch.backends.cudnn, "allow_tf32", False)),
            (lambda: setattr(torch.backends.cudnn, "deterministic", False), lambda: setattr(torch.backends.cudnn, "deterministic", True)),
            (lambda: torch.use_deterministic_algorithms(False), lambda: torch.use_deterministic_algorithms(True)),
            (lambda: torch.set_num_threads(2), lambda: torch.set_num_threads(1)),
        ]:
            attr_set()
            try:
                with self.assertRaises(determinism.DeterminismError):
                    determinism.check_configured("cpu")
            finally:
                undo()
        determinism.check_configured("cpu")

    def test_cuda_initialised_too_early_rejected(self):
        with mock.patch.object(torch.cuda, "is_initialized", return_value=True):
            with self.assertRaises(determinism.DeterminismError):
                determinism.configure("cuda:0")
        determinism.configure("cpu")

    def test_unsupported_device(self):
        with self.assertRaises((determinism.DeterminismError, RuntimeError)):
            determinism.configure("mps")


class TestStreamsAndHashes(unittest.TestCase):
    def test_stream_seeds(self):
        seeds = {(s, n): determinism.stream_seed(s, n) for s in (0, 1, 2) for n in determinism.STREAMS}
        self.assertEqual(len(set(seeds.values())), 9)               # all streams and seeds distinct
        expected = int.from_bytes(hashlib.sha256(b"penn_pose/release1/order/1").digest()[:8], "little") & ((1 << 63) - 1)
        self.assertEqual(determinism.stream_seed(1, "order"), expected)
        for bad in (-1, 1.0, True, "0"):
            with self.assertRaises(ValueError):
                determinism.stream_seed(bad, "init")
        with self.assertRaises(ValueError):
            determinism.stream_seed(0, "other")

    def test_state_hash(self):
        _synthetic.configure_cpu()
        a, b = model.build_model(0.1, 0), model.build_model(0.1, 0)
        self.assertEqual(determinism.state_hash(a), determinism.state_hash(b.state_dict()))
        b.features[1].running_mean += 1e-6                          # buffers are included
        self.assertNotEqual(determinism.state_hash(a), determinism.state_hash(b))
        b = model.build_model(0.1, 0)
        b.features[1].num_batches_tracked += 1
        self.assertNotEqual(determinism.state_hash(a), determinism.state_hash(b))

    def test_environment_record_and_lock(self):
        _synthetic.configure_cpu()
        rec = determinism.environment_record("cpu")
        for key in ("device", "device_name", "torch", "cuda", "cudnn", "numpy", "scipy", "scikit_learn", "settings"):
            self.assertIn(key, rec)
        self.assertEqual(rec["device"], "cpu")
        determinism.assert_same_environment(rec, dict(rec))
        for key, value in (("device", "cuda:0"), ("torch", "0.0"), ("cudnn", 1), ("device_name", "other")):
            with self.assertRaises(determinism.DeterminismError):
                determinism.assert_same_environment(rec, {**rec, key: value})


if __name__ == "__main__":
    unittest.main()
