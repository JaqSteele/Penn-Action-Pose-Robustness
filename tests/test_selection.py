import math
import unittest

import numpy as np

import _common  # noqa: F401
from penn_pose import selection
from penn_pose.selection import RELEASE1, Plan, Score
from penn_pose.training import TrainConfig


def _scores(plan, fn):
    return [Score(c, s, e, float(fn(c, s, e))) for c in plan.grid for s in plan.seeds for e in plan.eval_epochs]


class TestPlan(unittest.TestCase):
    def test_release1_plan(self):
        self.assertEqual(len(RELEASE1.grid), 8)
        self.assertEqual({c.learning_rate for c in RELEASE1.grid}, {0.001, 0.003})
        self.assertEqual({c.weight_decay for c in RELEASE1.grid}, {0.0001, 0.001})
        self.assertEqual({c.dropout for c in RELEASE1.grid}, {0.1, 0.3})
        self.assertEqual(RELEASE1.seeds, (0, 1, 2))
        self.assertEqual(RELEASE1.eval_epochs, tuple(range(5, 151, 5)))
        self.assertEqual(len(RELEASE1.eval_epochs), 30)

    def test_invalid_plans(self):
        c = TrainConfig(0.001, 0.0001, 0.1)
        for plan in (Plan((), (0,), 10, 5), Plan((c, c), (0,), 10, 5), Plan((c,), (0, 0), 10, 5),
                     Plan((c,), (0,), 12, 5), Plan((c,), (0,), 0, 5)):
            with self.assertRaises(ValueError):
                plan.validate()


class TestSelection(unittest.TestCase):
    def test_selects_highest_mean(self):
        target = RELEASE1.grid[5]
        sel = selection.select(_scores(RELEASE1, lambda c, s, e: 0.9 if (c == target and e == 40) else 0.5 + 0.001 * s))
        self.assertEqual((sel.selected.config, sel.selected.epoch), (target, 40))
        self.assertEqual(len(sel.tied), 1)
        self.assertEqual(len(sel.candidates), 240)

    def test_mean_over_seeds(self):
        sel = selection.select(_scores(RELEASE1, lambda c, s, e: [0.6, 0.7, 0.95][s] if e == 100 and c == RELEASE1.grid[0]
                                       else 0.7))
        self.assertEqual(sel.selected.epoch, 100)
        self.assertEqual(sel.selected.mean_macro_f1, math.fsum([0.6, 0.7, 0.95]) / 3)
        self.assertEqual(sel.selected.per_seed, (0.6, 0.7, 0.95))

    def test_tie_breaking_order(self):
        g = RELEASE1.grid
        # All equal: smallest E, then smallest learning rate, weight decay, dropout.
        self.assertEqual(selection.select(_scores(RELEASE1, lambda c, s, e: 0.5)).selected.epoch, 5)
        self.assertEqual(selection.select(_scores(RELEASE1, lambda c, s, e: 0.5)).selected.config, g[0])
        # Tie between a later epoch of config 0 and an earlier epoch of config 7: smaller E wins.
        f = lambda c, s, e: 0.8 if (c == g[0] and e == 50) or (c == g[7] and e == 20) else 0.5
        sel = selection.select(_scores(RELEASE1, f))
        self.assertEqual((sel.selected.config, sel.selected.epoch, len(sel.tied)), (g[7], 20, 2))
        # Same E: learning rate, then weight decay, then dropout decide.
        for pair, winner in (((g[4], g[3]), g[3]), ((g[2], g[1]), g[1]), ((g[1], g[0]), g[0])):
            f = lambda c, s, e, pair=pair: 0.8 if c in pair and e == 30 else 0.5
            self.assertEqual(selection.select(_scores(RELEASE1, f)).selected.config, winner)

    def test_exact_equality_defines_ties(self):
        g = RELEASE1.grid
        hi = 0.8
        lo = np.nextafter(hi, 0.0)                                     # one ulp lower: not a tie
        f = lambda c, s, e: hi if (c == g[7] and e == 150) else (lo if (c == g[0] and e == 5) else 0.1)
        sel = selection.select(_scores(RELEASE1, f))
        self.assertEqual((sel.selected.config, sel.selected.epoch, len(sel.tied)), (g[7], 150, 1))

    def test_mean_is_order_independent(self):
        vals = [0.1, 0.2, 0.30000000000000004]
        self.assertEqual(math.fsum(vals) / 3, math.fsum(vals[::-1]) / 3)

    def test_rejects_bad_score_tables(self):
        good = _scores(RELEASE1, lambda c, s, e: 0.5)
        bad_tables = [
            good[:-1],                                                                   # incomplete
            good + [good[0]],                                                            # duplicate
            good[:-1] + [Score(good[-1].config, good[-1].seed, 151, 0.5)],               # epoch outside plan
            good[:-1] + [Score(TrainConfig(0.01, 0.0001, 0.1), 0, 5, 0.5)],              # config outside grid
            good[:-1] + [Score(good[-1].config, good[-1].seed, good[-1].epoch, float("nan"))],
            good[:-1] + [Score(good[-1].config, good[-1].seed, good[-1].epoch, 1.5)],
            good[:-1] + [Score(good[-1].config, 3, good[-1].epoch, 0.5)],                # seed outside plan
        ]
        for table in bad_tables:
            with self.assertRaises(ValueError):
                selection.select(table)


if __name__ == "__main__":
    unittest.main()
