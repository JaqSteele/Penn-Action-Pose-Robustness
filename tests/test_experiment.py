"""Synthetic end-to-end tests of the Release 1.0 orchestration (CPU; temporary folders and a
temporary Git repository; no Penn Action data except the read-only cache checks at the end)."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import _common
import _synthetic
from _synthetic import DEVICE, TINY_PLAN
from penn_pose import determinism, experiment as ex, runs


class TestQualification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _synthetic.configure_cpu()
        cls.X, cls.y = _synthetic.data(45)

    def _cpu_repetition(self, device):
        if device != "cpu":
            raise determinism.DeterminismError("CUDA is not available (simulated)")
        return ex.qualification_repetition(self.X, self.y, "cpu")

    def test_repetition_records_training_only_metrics(self):
        r = ex.qualification_repetition(self.X, self.y, "cpu")
        self.assertEqual(set(r["checkpoint_state_sha256"]), {"5"})
        self.assertEqual(set(r["training_metrics"][0]), {"epoch", "train_loss", "train_accuracy", "order_sha256"})
        text = json.dumps(r)
        self.assertNotIn("macro_f1", text)
        self.assertNotIn("validation", text)
        with self.assertRaises(ValueError):
            ex.qualification_repetition(self.X, self.y, "cpu", epochs=4)

    def test_gpu_failure_falls_back_to_cpu(self):
        result = ex.qualify(self._cpu_repetition)
        self.assertEqual(result["qualified_device"], "cpu")
        self.assertEqual([a["device"] for a in result["attempts"]], ["cuda:0", "cpu"])
        self.assertFalse(result["attempts"][0]["qualified"])
        self.assertTrue(result["attempts"][1]["qualified"])
        self.assertEqual(result["environment"]["settings"]["num_threads"], 1)

    def test_nonidentical_repetitions_fail_and_both_failing_stops(self):
        calls = {"n": 0}

        def flaky(device):
            calls["n"] += 1
            r = ex.qualification_repetition(self.X, self.y, "cpu")
            r["device"] = device
            if calls["n"] % 2 == 0:
                r["training_metrics"][0] = {**r["training_metrics"][0], "train_loss": 0.0}
            return r
        with self.assertRaises(ex.QualificationError):
            ex.qualify(flaky)
        self.assertEqual(calls["n"], 4)                                  # GPU twice, CPU twice

    def test_record_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = ex.qualify(self._cpu_repetition)
            sha = ex.record_qualification(result, Path(tmp) / "out", Path(tmp) / "records")
            rec = runs.read_json(Path(tmp) / "records" / "release1-device.json")
            self.assertEqual(rec["qualified_device"], "cpu")
            self.assertEqual(rec["run_manifest_sha256"], runs.verify_run(Path(tmp) / "out" / "qualification" / "device"))
            self.assertEqual(len(sha), 64)


class TestEndToEnd(unittest.TestCase):
    """Stage A C1 -> selection -> commit gate -> C2 reruns -> C3 -> Stage B, on a tiny plan, in a
    temporary repository whose protected code/configuration is committed."""

    @classmethod
    def setUpClass(cls):
        _synthetic.configure_cpu()
        cls.gt = _synthetic.inputs("gt")
        cls.est = _synthetic.inputs("est", shift=0.05)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.repo = base / "repo"
        self.repo.mkdir()
        _synthetic.init_repo(self.repo)
        self.out = base / "experiments"
        self.records = self.repo / "records"
        self.record_path = self.repo / "configs" / "selection.json"

    def tearDown(self):
        self.tmp.cleanup()

    def _stage_a_c1(self, plan=TINY_PLAN):
        return ex.stage_a_c1(self.gt, DEVICE, self.out, plan=plan, prov=_synthetic.prov(self.repo), repo_root=self.repo)

    def _select(self, table=None):
        return ex.select_and_record(self.out, self.gt.y, self.record_path, table, plan=TINY_PLAN,
                                    prov=_synthetic.prov(self.repo), repo_root=self.repo)

    def _stage_a(self):
        self._stage_a_c1()
        return self._select(self.repo / "table.csv")

    def _common_args(self):
        return dict(device=DEVICE, out_root=self.out, record_path=self.record_path, repo_root=self.repo,
                    records_dir=self.records, plan=TINY_PLAN, prov=_synthetic.prov(self.repo))

    def test_full_flow(self):
        names = self._stage_a_c1()
        self.assertEqual(len(names), 4)
        manifests = runs.stage_runs(self.out / "stage_a_c1")
        self.assertEqual(self._stage_a_c1(), names)                                  # resume: no rerun
        self.assertEqual(runs.stage_runs(self.out / "stage_a_c1"), manifests)
        run = runs.read_json(self.out / "stage_a_c1" / names[0] / "run.json")
        self.assertEqual([c["epoch"] for c in run["checkpoints"]], [2, 4])
        self.assertEqual(run["provenance"]["code"], runs.code_identity(self.repo))
        with np.load(self.out / "stage_a_c1" / names[0] / "predictions.npz") as p:
            self.assertEqual(p["y_pred"].shape, (2, 30))

        record = self._select(self.repo / "table.csv")
        sel = record["selected"]
        self.assertIn(sel["epoch"], (2, 4))
        self.assertEqual(set(sel["per_seed_state_sha256"]), {"0", "1"})
        self.assertEqual(record["stage_a_implementation"], run["provenance"]["code"])
        self.assertEqual(record["stage_b"]["majority_class"]["index"], ex.metrics.majority_class(self.gt.y))
        self.assertTrue((self.out / "stage_a_c1" / "stage-sha256.txt").is_file())
        self.assertEqual(len((self.repo / "table.csv").read_text().splitlines()), 1 + 4)
        with self.assertRaises(FileExistsError):
            self._select()

        # D-41: nothing after selection may run before the record is committed.
        for fn, args in ((ex.stage_a_c2, (self.gt, self.est)), (ex.stage_a_c3, (self.est,)),
                         (ex.stage_b, (self.gt, self.est))):
            with self.assertRaises(runs.CommitGateError):
                fn(*args, **self._common_args())
        self.assertFalse((self.out / "stage_a_c2").exists())
        _synthetic.commit(self.repo, self.record_path)          # a legitimate record commit: HEAD moves

        c2 = ex.stage_a_c2(self.gt, self.est, **self._common_args())
        self.assertTrue(c2["descriptive_only"])
        rerun = runs.read_json(next((self.out / "stage_a_c2").glob("*_seed0")) / "run.json")
        self.assertEqual(rerun["rerun_check"]["state_sha256"], sel["per_seed_state_sha256"]["0"])
        self.assertEqual(rerun["rerun_check"]["gt_validation_macro_f1"], sel["per_seed_macro_f1"]["0"])
        self.assertNotEqual(rerun["provenance"]["code"]["commit"], record["stage_a_implementation"]["commit"])
        self.assertEqual(rerun["provenance"]["code"]["protected_tree_sha256"],
                         record["stage_a_implementation"]["protected_tree_sha256"])

        c3 = ex.stage_a_c3(self.est, **self._common_args())
        c3_run = runs.read_json(next((self.out / "stage_a_c3").glob("*_seed1")) / "run.json")
        self.assertEqual(c3_run["epoch"], sel["epoch"])                  # D-43: evaluated only at E
        self.assertEqual(len(c3_run["training_metrics"]), sel["epoch"])
        self.assertTrue(c3["descriptive_only"])

        b = ex.stage_b(self.gt, self.est, **self._common_args())
        for cond in ("c1", "c3"):
            run_dirs = sorted((self.out / f"stage_b_{cond}").glob("*_seed*"))
            self.assertEqual(len(run_dirs), 2)
            for d in run_dirs:
                rec = runs.read_json(d / "run.json")
                self.assertIsNone(rec["evaluation"])
                self.assertEqual(rec["training_videos"], 75)
                self.assertTrue((d / "model.pt").is_file())
        c1_0 = runs.read_json(next((self.out / "stage_b_c1").glob("*_seed0")) / "run.json")
        c3_0 = runs.read_json(next((self.out / "stage_b_c3").glob("*_seed0")) / "run.json")
        self.assertEqual(c1_0["initial_state_sha256"], c3_0["initial_state_sha256"])        # matched seeds
        self.assertEqual([e["order_sha256"] for e in c1_0["training_metrics"]],
                         [e["order_sha256"] for e in c3_0["training_metrics"]])
        self.assertEqual(set(b["conditions"]), {"C1", "C3"})
        selection = runs.read_json(self.record_path)
        models = ex.load_stage_b_models(self.out, b, DEVICE, selection_record=selection, repo_root=self.repo)
        self.assertEqual({k: sorted(v) for k, v in models.items()}, {"C1": [0, 1], "C3": [0, 1]})
        for cond in ("C1", "C3"):
            for seed, m in models[cond].items():
                self.assertFalse(m.training)
                self.assertEqual(determinism.state_hash(m), b["conditions"][cond]["final_state_sha256"][str(seed)])
        with self.assertRaises(TypeError):                               # the selection record is required
            ex.load_stage_b_models(self.out, b, DEVICE)
        victim = next((self.out / "stage_b_c3").glob("*_seed1")) / "model.pt"
        victim.write_bytes(victim.read_bytes()[:-8] + b"tampered")
        with self.assertRaises(ValueError):
            ex.load_stage_b_models(self.out, b, DEVICE, selection_record=selection, repo_root=self.repo)
        for name in ("stage-a-c2", "stage-a-c3", "stage-b"):
            self.assertTrue(ex.stage_record_path(self.records, name).is_file())

        # Nothing is overwritten on a second attempt.
        for fn, args in ((ex.stage_a_c2, (self.gt, self.est)), (ex.stage_a_c3, (self.est,)),
                         (ex.stage_b, (self.gt, self.est))):
            with self.assertRaises(runs.OutputExistsError):
                fn(*args, **self._common_args())

    def test_failed_reproduction_stops_before_c2(self):
        record = self._stage_a()
        tampered = dict(record)
        tampered["selected"] = {**record["selected"],
                                "per_seed_state_sha256": {**record["selected"]["per_seed_state_sha256"], "1": "0" * 64}}
        self.record_path.write_text(json.dumps(tampered), encoding="utf-8")
        _synthetic.commit(self.repo, self.record_path)
        with self.assertRaises(ex.ReproductionError):
            ex.stage_a_c2(self.gt, self.est, **self._common_args())
        self.assertFalse((self.out / "stage_a_c2").exists() and any((self.out / "stage_a_c2").iterdir()))

    def test_corrupted_macro_f1_stops_before_c2(self):
        # The state hash stays correct; only the recorded ground-truth validation macro-F1 is off by one ulp.
        record = self._stage_a()
        f1 = record["selected"]["per_seed_macro_f1"]["0"]
        tampered = dict(record)
        tampered["selected"] = {**record["selected"],
                                "per_seed_macro_f1": {**record["selected"]["per_seed_macro_f1"],
                                                      "0": float(np.nextafter(f1, 2.0))}}
        self.assertEqual(tampered["selected"]["per_seed_state_sha256"], record["selected"]["per_seed_state_sha256"])
        self.record_path.write_text(json.dumps(tampered), encoding="utf-8")
        _synthetic.commit(self.repo, self.record_path)
        with self.assertRaises(ex.ReproductionError):
            ex.stage_a_c2(self.gt, self.est, **self._common_args())
        self.assertFalse((self.out / "stage_a_c2").exists() and any((self.out / "stage_a_c2").iterdir()))
        self.assertFalse(ex.stage_record_path(self.records, "stage-a-c2").exists())

    def test_existing_records_block_before_any_output(self):
        self._stage_a_c1()
        runs.write_record(self.record_path, {"placeholder": True})
        with self.assertRaises(runs.OutputExistsError):
            self._select()
        self.assertFalse((self.out / "stage_a_c1" / "stage-sha256.txt").exists())   # nothing written
        self.record_path.unlink()
        self._select()
        _synthetic.commit(self.repo, self.record_path)
        runs.write_record(ex.stage_record_path(self.records, "stage-b"), {"placeholder": True})
        with self.assertRaises(runs.OutputExistsError):
            ex.stage_b(self.gt, self.est, **self._common_args())
        self.assertFalse((self.out / "stage_b_c1").exists())                        # no model trained

    def test_selection_requires_every_run(self):
        self._stage_a_c1()
        victim = sorted((self.out / "stage_a_c1").iterdir())[0]
        victim.rename(victim.with_name(victim.name + ".incomplete"))
        with self.assertRaises(runs.OutputExistsError):
            self._select()

    def test_mismatched_inputs_rejected(self):
        bad = ex.Inputs("est", self.est.X, np.roll(self.est.y, 1), self.est.ids, self.est.subset)
        with self.assertRaises(ValueError):
            ex.check_paired(self.gt, bad)


class TestNoTestPartitionAccess(unittest.TestCase):
    D2_FILES = [_common.REPO_ROOT / "src" / "penn_pose" / f"{m}.py"
                for m in ("model", "determinism", "training", "metrics", "selection", "bootstrap", "runs", "experiment",
                          "batch_e")]
    D2_FILES.append(_common.REPO_ROOT / "scripts" / "run_release1.py")

    def test_static(self):
        import re
        for path in self.D2_FILES:
            text = path.read_text(encoding="utf-8")
            self.assertNotRegex(text, r"allow_test\s*=\s*True", path.name)
            self.assertNotRegex(text, r"list_video_ids\(", path.name)
            self.assertNotRegex(text, r"load_video\(|loadmat\(|partition_flags\(", path.name)
            self.assertNotRegex(text, r"rtmpose_m[\\/]+test|_test\.npz", path.name)


class TestCommandLineGuards(unittest.TestCase):
    def test_training_commands_need_confirmation(self):
        import subprocess, sys
        script = _common.REPO_ROOT / "scripts" / "run_release1.py"
        for command in ("qualify", "qualify-repetition", "stage-a-c1", "stage-a-c2", "stage-a-c3", "stage-b"):
            proc = subprocess.run([sys.executable, "-B", str(script), command, "--device", "cpu"],
                                  cwd=_common.REPO_ROOT, capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0, command)
            self.assertIn("--confirm", proc.stderr, command)


class TestDeviceLockAndConfig(unittest.TestCase):
    def test_prepare_device(self):
        _synthetic.configure_cpu()
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            _synthetic.init_repo(repo)
            path = repo / "device.json"
            env = determinism.environment_record("cpu")
            runs.write_record(path, {"qualified_device": "cpu", "environment": env})
            with self.assertRaises(runs.CommitGateError):
                ex.prepare_device(path, repo)
            _synthetic.commit(repo, path)
            self.assertEqual(ex.prepare_device(path, repo), "cpu")
            other = repo / "device2.json"
            runs.write_record(other, {"qualified_device": "cpu", "environment": {**env, "torch": "0.0"}})
            _synthetic.commit(repo, other)
            with self.assertRaises(determinism.DeterminismError):
                ex.prepare_device(other, repo)

    def test_committed_config_matches_code(self):
        cfg = ex.load_config()
        self.assertEqual(cfg["bootstrap"], {                             # D-45 resolved, D-46, D-47 (v0.15)
            "n_resamples": 10000, "seed": 20261128, "sd_ddof": 1, "percentile_method": "linear",
            "seed_status": "fixed (D-45, protocol v0.15) after Stage B and before any test-partition output "
                           "existed; D-46 sample SD; D-47 linear percentiles"})
        self.assertIs(type(cfg["bootstrap"]["seed"]), int)
        self.assertEqual(cfg["protocol"], ex.PROTOCOL_VERSION)
        self.assertEqual(ex.PROTOCOL_VERSION, "v0.15")
        self.assertEqual(cfg["plan"]["eval_epochs"], list(range(5, 151, 5)))
        with tempfile.TemporaryDirectory() as tmp:
            altered = dict(cfg)
            altered["batch_size"] = 64
            p = Path(tmp) / "c.json"
            p.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaises(ValueError):
                ex.load_config(p)


@unittest.skipUnless((ex.CACHE_DIR / "inputs-sha256.txt").is_file(), "D1 cache not present")
class TestRealInputLoading(unittest.TestCase):
    """Read-only checks of the D1 training-partition cache (no training)."""

    def test_load_and_pair(self):
        gt, est = ex.load_inputs("gt"), ex.load_inputs("est")
        ex.check_paired(gt, est)
        self.assertEqual(gt.X.shape, (1258, 24, 64))
        for which, n in (("fitting", 1070), ("validation", 188), ("all", 1258)):
            X, y, ids = gt.part(which)
            self.assertEqual((X.shape[0], y.shape[0], ids.shape[0]), (n, n, n))
        with self.assertRaises(ValueError):
            ex.load_inputs("test")


if __name__ == "__main__":
    unittest.main()
