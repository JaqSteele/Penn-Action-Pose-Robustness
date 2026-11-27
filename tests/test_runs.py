import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import _synthetic
from penn_pose import integrity, runs


class TestRunWriter(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, name="run1"):
        w = runs.RunWriter(self.root / "stage" / name)
        w.write_json("run.json", {"a": 1})
        w.write_npz("predictions.npz", y=np.arange(3))
        w.write_state_dict("model.pt", {"w": torch.ones(2)})
        return w, w.finalise()

    def test_atomic_publish_and_manifest(self):
        w, sha = self._run()
        final = self.root / "stage" / "run1"
        self.assertTrue(final.is_dir())
        self.assertFalse((self.root / "stage" / "run1.incomplete").exists())
        self.assertEqual(sha, integrity.sha256_file(final / runs.RUN_MANIFEST))
        self.assertEqual(set(integrity.read_manifest(final / runs.RUN_MANIFEST)), {"run.json", "predictions.npz", "model.pt"})
        self.assertEqual(runs.verify_run(final), sha)
        self.assertEqual(runs.read_json(final / "run.json"), {"a": 1})
        self.assertTrue(torch.equal(torch.load(final / "model.pt", weights_only=True)["w"], torch.ones(2)))
        with self.assertRaises(RuntimeError):
            w.write_json("late.json", {})

    def test_refuses_overwrite(self):
        self._run()
        with self.assertRaises(runs.OutputExistsError):
            runs.RunWriter(self.root / "stage" / "run1")
        (self.root / "stage" / "run2.incomplete").mkdir()
        with self.assertRaises(runs.OutputExistsError):
            runs.RunWriter(self.root / "stage" / "run2")

    def test_file_level_protection_and_names(self):
        w = runs.RunWriter(self.root / "stage" / "r")
        w.write_json("a.json", {})
        with self.assertRaises(FileExistsError):
            w.write_json("a.json", {})
        for bad in ("", "x/y.json", runs.RUN_MANIFEST, "z.incomplete"):
            with self.assertRaises(ValueError):
                w.write_json(bad, {})
        with self.assertRaises(ValueError):
            w.write_json("nan.json", {"x": float("nan")})          # strict JSON only
        empty = runs.RunWriter(self.root / "stage" / "empty")
        with self.assertRaises(RuntimeError):
            empty.finalise()

    def test_tampering_detected(self):
        self._run()
        final = self.root / "stage" / "run1"
        (final / "run.json").write_text('{"a": 2}\n', encoding="utf-8")
        with self.assertRaises(ValueError):
            runs.verify_run(final)
        self._run("run3")
        (self.root / "stage" / "run3" / "extra.txt").write_text("x", encoding="utf-8")
        with self.assertRaises(ValueError):
            runs.verify_run(self.root / "stage" / "run3")

    def test_stage_manifest(self):
        self._run("a")
        self._run("b")
        with self.assertRaises(ValueError):
            runs.write_stage_manifest(self.root / "stage", ["a"])
        sha = runs.write_stage_manifest(self.root / "stage", ["a", "b"])
        self.assertEqual(len(sha), 64)
        with self.assertRaises(FileExistsError):
            runs.write_stage_manifest(self.root / "stage", ["a", "b"])
        (self.root / "stage" / "c.incomplete").mkdir()
        with self.assertRaises(runs.OutputExistsError):
            runs.stage_runs(self.root / "stage")

    def test_record_never_overwritten(self):
        p = self.root / "records" / "r.json"
        runs.write_record(p, {"x": 1})
        with self.assertRaises(FileExistsError):
            runs.write_record(p, {"x": 2})
        self.assertEqual(runs.read_json(p), {"x": 1})


class TestCommitGate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        _synthetic.init_repo(self.repo)
        self.rec = self.repo / "configs" / "sel.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_gate(self):
        with self.assertRaises(runs.CommitGateError):
            runs.require_committed(self.rec, self.repo)                     # missing
        runs.write_record(self.rec, {"selected": 1})
        with self.assertRaises(runs.CommitGateError):
            runs.require_committed(self.rec, self.repo)                     # untracked
        _synthetic.commit(self.repo, self.rec)
        commit = runs.require_committed(self.rec, self.repo)
        self.assertEqual(len(commit), 40)
        self.rec.write_text('{"selected": 2}\n', encoding="utf-8")
        with self.assertRaises(runs.CommitGateError):
            runs.require_committed(self.rec, self.repo)                     # modified
        _synthetic.git(self.repo, "add", "configs/sel.json")
        with self.assertRaises(runs.CommitGateError):
            runs.require_committed(self.rec, self.repo)                     # staged, not committed


if __name__ == "__main__":
    unittest.main()
