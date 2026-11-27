import unittest

import numpy as np
import torch
from torch import nn

import _synthetic
from penn_pose import determinism, model


class TestArchitecture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _synthetic.configure_cpu()

    def test_parameter_count(self):
        m = model.TemporalCNN(0.1)
        self.assertEqual(model.count_parameters(m), 133_135)
        self.assertEqual(model.EXPECTED_PARAMETERS, 133_135)
        by_layer = {n: p.numel() for n, p in m.named_parameters()}
        self.assertEqual(by_layer, {
            "features.0.weight": 7680, "features.1.weight": 64, "features.1.bias": 64,
            "features.5.weight": 40960, "features.6.weight": 128, "features.6.bias": 128,
            "features.10.weight": 81920, "features.11.weight": 128, "features.11.bias": 128,
            "classifier.weight": 1920, "classifier.bias": 15})

    def test_layer_specification(self):
        m = model.TemporalCNN(0.3)
        convs = [l for l in m.features if isinstance(l, nn.Conv1d)]
        self.assertEqual([(c.in_channels, c.out_channels) for c in convs], [(24, 64), (64, 128), (128, 128)])
        for c in convs:
            self.assertEqual((c.kernel_size, c.padding, c.bias), ((5,), (2,), None))
        bns = [l for l in m.features if isinstance(l, nn.BatchNorm1d)]
        self.assertTrue(all(b.momentum == 0.1 and b.eps == 1e-5 and b.affine and b.track_running_stats for b in bns))
        pools = [i for i, l in enumerate(m.features) if isinstance(l, nn.MaxPool1d)]
        self.assertEqual(pools, [4, 9])                      # after blocks 1 and 2 only
        self.assertEqual([type(l).__name__ for l in m.features][:5],
                         ["Conv1d", "BatchNorm1d", "ReLU", "Dropout", "MaxPool1d"])
        drops = [l.p for l in m.modules() if isinstance(l, nn.Dropout)]
        self.assertEqual(drops, [0.3, 0.3, 0.3, 0.3])        # three blocks + before the classifier
        self.assertEqual((m.classifier.in_features, m.classifier.out_features), (128, 15))

    def test_shapes_and_global_average(self):
        m = model.TemporalCNN(0.1).eval()
        x = torch.randn(5, 24, 64)
        with torch.no_grad():
            feats = m.features(x)
            self.assertEqual(tuple(feats.shape), (5, 128, 16))
            out = m(x)
            manual = m.classifier(feats.mean(dim=2))
        self.assertEqual(tuple(out.shape), (5, 15))
        torch.testing.assert_close(out, manual, rtol=0, atol=0)

    def test_rejects_bad_input_and_dropout(self):
        m = model.TemporalCNN(0.1)
        for shape in [(5, 24, 63), (5, 12, 64), (24, 64)]:
            with self.assertRaises(ValueError):
                m(torch.randn(*shape))
        for p in (1.0, -0.1, 1, "0.1"):
            with self.assertRaises(ValueError):
                model.TemporalCNN(p)


class TestInitialisation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _synthetic.configure_cpu()

    def test_same_seed_same_state(self):
        a, b = model.build_model(0.1, 1), model.build_model(0.1, 1)
        self.assertEqual(determinism.state_hash(a), determinism.state_hash(b))
        self.assertNotEqual(determinism.state_hash(a), determinism.state_hash(model.build_model(0.1, 2)))

    def test_init_independent_of_dropout(self):
        self.assertEqual(determinism.state_hash(model.build_model(0.1, 0)),
                         determinism.state_hash(model.build_model(0.3, 0)))

    def test_does_not_touch_global_rng(self):
        before = torch.random.get_rng_state().clone()
        model.build_model(0.1, 0)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_default_initialisation_used(self):
        # PyTorch default (kaiming-uniform style) bounds: |w| <= 1/sqrt(fan_in); BN weight 1, bias 0.
        m = model.build_model(0.1, 0)
        conv = m.features[0]
        self.assertLessEqual(float(conv.weight.detach().abs().max()), 1 / np.sqrt(24 * 5) + 1e-7)
        self.assertTrue(torch.all(m.features[1].weight == 1) and torch.all(m.features[1].bias == 0))
        self.assertTrue(torch.all(m.features[1].running_mean == 0) and torch.all(m.features[1].running_var == 1))


if __name__ == "__main__":
    unittest.main()
