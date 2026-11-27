import hashlib
import re
import unittest

import numpy as np
import torch

import _common
import _synthetic
from penn_pose import determinism, training
from penn_pose.training import TrainConfig

CFG = TrainConfig(0.003, 0.0001, 0.3)


class TestComponents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _synthetic.configure_cpu()

    def test_class_weights(self):
        y = np.array([0] * 4 + [1] * 2 + list(range(2, 15)), dtype=np.int64)    # N = 19
        w = training.class_weights(y)
        self.assertEqual(w.dtype, torch.float32)
        expected = np.array([19 / (15 * 4), 19 / (15 * 2)] + [19 / 15] * 13)
        np.testing.assert_allclose(w.numpy(), expected.astype(np.float32), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            training.class_weights(np.arange(14, dtype=np.int64))                 # class 14 missing

    def test_weighted_mean_reduction(self):
        torch.manual_seed(0)
        logits = torch.randn(10, 15)
        target = torch.tensor([0, 1, 1, 2, 3, 3, 3, 4, 5, 14])
        w = torch.rand(15) + 0.5
        loss = training.make_loss(w)(logits, target)
        per = torch.nn.functional.cross_entropy(logits, target, reduction="none")
        manual = (w[target] * per).sum() / w[target].sum()
        torch.testing.assert_close(loss, manual)
        self.assertEqual(training.make_loss(w).reduction, "mean")

    def test_optimizer_defaults_and_all_parameters_decayed(self):
        m = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.BatchNorm1d(2))
        opt = training.make_optimizer(m, CFG)
        self.assertIsInstance(opt, torch.optim.AdamW)
        self.assertEqual(len(opt.param_groups), 1)
        g = opt.param_groups[0]
        self.assertEqual(len(g["params"]), len(list(m.parameters())))         # biases and BN included
        self.assertEqual((g["lr"], g["weight_decay"], g["betas"], g["eps"], g["amsgrad"]),
                         (0.003, 0.0001, (0.9, 0.999), 1e-8, False))

    def test_batches_keep_last_partial(self):
        for n, n_batches, last in ((1070, 34, 14), (1258, 40, 10), (32, 1, 32), (33, 2, 1)):
            s = training.batch_slices(n)
            self.assertEqual((len(s), s[-1].stop - s[-1].start), (n_batches, last))
            self.assertEqual(sum(x.stop - x.start for x in s), n)
        with self.assertRaises(ValueError):
            training.batch_slices(0)

    def test_epoch_orders(self):
        a, b = training.epoch_orders(50, 1, 3), training.epoch_orders(50, 1, 3)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
            self.assertEqual(sorted(x.tolist()), list(range(50)))
        self.assertFalse(np.array_equal(a[0], a[1]))
        self.assertFalse(np.array_equal(a[0], training.epoch_orders(50, 2, 1)[0]))

    def test_no_unapproved_training_features(self):
        text = (_common.REPO_ROOT / "src" / "penn_pose" / "training.py").read_text(encoding="utf-8")
        code = re.sub(r'"""(.|\n)*?"""', "", text)
        for pattern in ("lr_scheduler", "clip_grad", "autocast", "GradScaler", "label_smoothing", "half()",
                        "bfloat16", "num_workers", "DataLoader"):
            self.assertNotIn(pattern, code)


class TestTrainModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _synthetic.configure_cpu()
        cls.X, cls.y = _synthetic.data(45)
        cls.Xv, cls.yv = _synthetic.data(30, seed=5)

    def _train(self, X=None, seed=0, epochs=4, **kw):
        return training.train_model(self.X if X is None else X, self.y, CFG, seed, epochs, "cpu", **kw)

    def test_repeatable(self):
        a, b = self._train(checkpoint_epochs=(2, 4)), self._train(checkpoint_epochs=(2, 4))
        self.assertEqual(a.epochs, b.epochs)
        self.assertEqual(a.final_state_sha256, b.final_state_sha256)
        self.assertEqual({e: c["state_sha256"] for e, c in a.checkpoints.items()},
                         {e: c["state_sha256"] for e, c in b.checkpoints.items()})
        self.assertNotEqual(a.final_state_sha256, self._train(seed=1).final_state_sha256)

    def test_checkpoints_and_metrics(self):
        r = self._train(checkpoint_epochs=(2, 4))
        self.assertEqual(sorted(r.checkpoints), [2, 4])
        self.assertEqual(r.checkpoints[4]["state_sha256"], r.final_state_sha256)
        self.assertEqual([e["epoch"] for e in r.epochs], [1, 2, 3, 4])
        for e in r.epochs:
            self.assertTrue(np.isfinite(e["train_loss"]) and 0.0 <= e["train_accuracy"] <= 1.0)

    def test_evaluation_does_not_change_training(self):
        plain = self._train(checkpoint_epochs=(1, 2, 3, 4))
        evaluator = training.make_evaluator(self.Xv, self.yv, "cpu")
        evaluated = self._train(checkpoint_epochs=(1, 2, 3, 4), evaluate=evaluator)
        self.assertEqual(plain.epochs, evaluated.epochs)
        self.assertEqual(plain.final_state_sha256, evaluated.final_state_sha256)
        self.assertEqual(plain.checkpoints[2]["state_sha256"], evaluated.checkpoints[2]["state_sha256"])
        c = evaluated.checkpoints[4]
        self.assertEqual(c["y_pred"].shape, (30,))
        self.assertTrue(0.0 <= c["macro_f1"] <= 1.0 and 0.0 <= c["accuracy"] <= 1.0)

    def test_epoch_loss_is_epoch_level_weighted_mean(self):
        # With lr = 0 and dropout = 0 the model never changes, so the epoch loss can be recomputed
        # independently: sum(w_i * l_i) / sum(w_i) over every video, batch by batch in training mode.
        X, y = _synthetic.data(45)                            # 45 videos: batches of 32 and 13
        cfg = TrainConfig(0.0, 0.0, 0.0)
        got = training.train_model(X, y, cfg, 3, 1, "cpu").epochs[0]
        m = training.build_model(0.0, 3).train()
        w = training.class_weights(y)
        order = training.epoch_orders(45, 3, 1)[0]
        num = den = 0.0
        correct = 0
        batch_means = []
        with torch.no_grad():
            for s in training.batch_slices(45):
                idx = torch.from_numpy(order[s])
                logits = m(torch.from_numpy(X)[idx])
                per = torch.nn.functional.cross_entropy(logits, torch.from_numpy(y)[idx], reduction="none")
                wi = w[torch.from_numpy(y)[idx]]
                num += float((wi * per).sum()); den += float(wi.sum())
                batch_means.append(float((wi * per).sum() / wi.sum()))
                correct += int((logits.argmax(1) == torch.from_numpy(y)[idx]).sum())
        self.assertAlmostEqual(got["train_loss"], num / den, places=5)
        self.assertEqual(got["train_accuracy"], correct / 45)
        # The old definition (unweighted mean of the batch means) gives a different value here.
        self.assertEqual([s.stop - s.start for s in training.batch_slices(45)], [32, 13])
        unweighted = float(np.mean(batch_means))
        self.assertGreater(abs(got["train_loss"] - unweighted), 1e-4)

    def test_matched_seeds_for_c1_and_c3(self):
        X_est = self.X + np.float32(0.05)                    # a different pose source, same videos
        c1, c3 = self._train(seed=2), self._train(X=X_est, seed=2)
        self.assertEqual(c1.initial_state_sha256, c3.initial_state_sha256)
        self.assertEqual([e["order_sha256"] for e in c1.epochs], [e["order_sha256"] for e in c3.epochs])
        self.assertNotEqual(c1.final_state_sha256, c3.final_state_sha256)
        expected = [hashlib.sha256(o.astype(np.int64).tobytes()).hexdigest()
                    for o in training.epoch_orders(45, 2, 4)]
        self.assertEqual([e["order_sha256"] for e in c1.epochs], expected)

    def test_independent_of_global_rng_state(self):
        a = self._train(epochs=2)
        torch.manual_seed(12345); np.random.seed(7); torch.rand(100)
        b = self._train(epochs=2)
        self.assertEqual(a.final_state_sha256, b.final_state_sha256)

    def test_prediction_modes_and_rng(self):
        r = self._train(epochs=1)
        m = r.model.train()
        before = torch.random.get_rng_state().clone()
        p1, p2 = training.predict(m, self.Xv, "cpu"), training.predict(m, self.Xv, "cpu")
        np.testing.assert_array_equal(p1, p2)
        self.assertTrue(m.training)                          # mode restored
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        x = torch.from_numpy(self.Xv[:4])
        with torch.no_grad():
            self.assertFalse(torch.equal(m(x), m(x)))        # dropout active in training mode

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            training.train_model(self.X.astype(np.float64), self.y, CFG, 0, 1, "cpu")
        with self.assertRaises(ValueError):
            training.train_model(self.X[:, :, :32].copy(), self.y, CFG, 0, 1, "cpu")
        with self.assertRaises(ValueError):
            training.train_model(self.X, self.y.astype(np.int32), CFG, 0, 1, "cpu")
        bad = self.y.copy(); bad[0] = 15
        with self.assertRaises(ValueError):
            training.train_model(self.X, bad, CFG, 0, 1, "cpu")
        nan = self.X.copy(); nan[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            training.train_model(nan, self.y, CFG, 0, 1, "cpu")
        with self.assertRaises(ValueError):
            training.train_model(self.X, self.y, CFG, 0, 0, "cpu")
        with self.assertRaises(ValueError):
            training.train_model(self.X, self.y, CFG, 0, 2, "cpu", checkpoint_epochs=(3,))

    def test_requires_deterministic_configuration(self):
        torch.backends.cudnn.benchmark = True
        try:
            with self.assertRaises(determinism.DeterminismError):
                training.train_model(self.X, self.y, CFG, 0, 1, "cpu")
        finally:
            torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    unittest.main()
