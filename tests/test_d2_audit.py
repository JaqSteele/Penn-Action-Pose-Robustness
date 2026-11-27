"""Regression tests for the D2 audit: process isolation and import order (point 1), output paths
(point 3), the device record (point 4) and qualification inputs (point 5). CPU and synthetic only;
no qualification, no Penn Action training."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

import _common
import _synthetic
from _synthetic import DEVICE, TINY_PLAN
from penn_pose import determinism, experiment as ex, runs

SCRIPT = _common.REPO_ROOT / "scripts" / "run_release1.py"


def _load_cli():
    spec = importlib.util.spec_from_file_location("run_release1_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)                   # defines functions only; main() is not called
    return module


class TestImportOrderInFreshProcess(unittest.TestCase):
    """Point 1: in a new interpreter, importing the CLI sets CUBLAS_WORKSPACE_CONFIG before torch is
    first imported, and nothing up to and including the D-36 configuration initialises CUDA."""

    def test_cli_import_path(self):
        child = textwrap.dedent(f"""
            import importlib.abc, importlib.util, json, os, sys
            seen = {{}}
            class Watch(importlib.abc.MetaPathFinder):
                def find_spec(self, name, path, target=None):
                    if name == "torch" and "torch" not in seen:
                        seen["torch"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
                    return None
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            sys.meta_path.insert(0, Watch())
            spec = importlib.util.spec_from_file_location("cli", {str(SCRIPT)!r})
            cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
            import torch
            after_import = torch.cuda.is_initialized()
            cli.ex.load_config()
            after_config_load = torch.cuda.is_initialized()
            cli.determinism.configure("cuda:0")         # flags only; must not initialise CUDA
            after_configure = torch.cuda.is_initialized()
            print(json.dumps({{"cublas_at_first_torch_import": seen.get("torch"),
                               "cuda_initialised": [after_import, after_config_load, after_configure],
                               "penn_pose_init_imports_torch": "torch" in open(
                                   cli.REPO_ROOT / "src" / "penn_pose" / "__init__.py").read()}}))
        """)
        env = {k: v for k, v in os.environ.items() if k != "CUBLAS_WORKSPACE_CONFIG"}
        proc = subprocess.run([sys.executable, "-B", "-c", child], cwd=_common.REPO_ROOT, env=env,
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        out = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertEqual(out["cublas_at_first_torch_import"], ":4096:8")
        self.assertEqual(out["cuda_initialised"], [False, False, False])
        self.assertFalse(out["penn_pose_init_imports_torch"])


class TestFreshProcessPerRepetition(unittest.TestCase):
    """Point 1: every repetition, including the CPU fallback after a GPU failure, is a new process
    started from the real launcher, with --confirm and the qualify-repetition command."""

    def test_real_command_and_launcher(self):
        cli = _load_cli()
        cmd = cli.repetition_command("cpu")
        self.assertEqual(cmd[0], sys.executable)
        self.assertEqual(cmd[2:], [str(SCRIPT.resolve()), "qualify-repetition", "--device", "cpu", "--confirm"])

        with tempfile.TemporaryDirectory() as tmp:
            stub = Path(tmp) / "stub.py"
            stub.write_text(textwrap.dedent("""
                import json, os, sys
                device = sys.argv[1]
                if device != "cpu":
                    sys.exit("simulated GPU qualification failure")
                env = {"device": "cpu", "device_type": "cpu", "device_name": "x", "torch": "t", "cuda": "c", "cudnn": 1}
                print(json.dumps({"device": "cpu", "config": {}, "seed": 0, "epochs": 5,
                                  "initial_state_sha256": "a", "final_state_sha256": "b",
                                  "checkpoint_state_sha256": {"5": "b"}, "training_metrics": [],
                                  "environment": env, "process_id": os.getpid()}))
            """), encoding="utf-8")
            launch = lambda d: cli._repetition_in_subprocess(d, lambda dev: [sys.executable, "-B", str(stub), dev])
            result = ex.qualify(launch)
        self.assertEqual(result["qualified_device"], "cpu")
        gpu, cpu = result["attempts"]
        self.assertFalse(gpu["qualified"])
        self.assertIn("simulated GPU qualification failure", gpu["error"])
        pids = [r["process_id"] for r in cpu["repetitions"]]
        self.assertEqual(len(set(pids)), 2)                            # two distinct processes
        self.assertNotIn(os.getpid(), pids)                            # neither is this process


class TestOutputPaths(unittest.TestCase):
    """Point 3: bulky outputs go outside the repository; only compact records inside it."""

    def test_resolved_locations(self):
        self.assertEqual(runs.EXPERIMENT_ROOT, _common.DATA_ROOT / "experiments" / "release1")
        ex.check_external(runs.EXPERIMENT_ROOT)
        for inside in (ex.RECORDS_DIR, ex.TABLES_DIR, ex.SELECTION_RECORD):
            inside.resolve().relative_to(_common.REPO_ROOT.resolve())
        self.assertEqual(ex.RECORDS_DIR, _common.REPO_ROOT / "configs" / "recognition")
        self.assertEqual(ex.device_record_path(), _common.REPO_ROOT / "configs" / "recognition" / "release1-device.json")

    def test_writers_refuse_repository_paths(self):
        _synthetic.configure_cpu()
        inside = _common.REPO_ROOT / "experiments" / "release1"
        gt = _synthetic.inputs("gt")
        with self.assertRaises(runs.OutputExistsError):
            ex.stage_a_c1(gt, DEVICE, inside, plan=TINY_PLAN, prov={}, repo_root=_common.REPO_ROOT)
        with self.assertRaises(runs.OutputExistsError):
            ex.record_qualification({"qualified_device": "cpu", "environment": {}, "attempts": []},
                                    inside, Path(tempfile.gettempdir()) / "unused-records")
        self.assertFalse(inside.exists())                              # nothing was created


class TestDeviceRecord(unittest.TestCase):
    """Point 4: content, no error text or paths, and immutability across stages."""

    @classmethod
    def setUpClass(cls):
        _synthetic.configure_cpu()
        cls.X, cls.y = _synthetic.data(45)

    def test_compact_record(self):
        def rep(device):
            if device != "cpu":
                raise RuntimeError(r"failed at C:\Users\someone\secret\path.py")
            return ex.qualification_repetition(self.X, self.y, "cpu")
        result = ex.qualify(rep)
        compact = ex.compact_device_record(result, "f" * 64)
        text = json.dumps(compact)
        self.assertNotIn("Users", text)
        self.assertNotIn("secret", text)
        self.assertEqual(compact["attempts"][0], {"device": "cuda:0", "qualified": False, "failure": "RuntimeError",
                                                  "differing_fields": []})
        for key in ("torch", "cuda", "cudnn", "numpy", "scipy", "scikit_learn", "device", "device_name", "settings"):
            self.assertIn(key, compact["environment"])
        q = compact["qualification"]
        self.assertEqual((q["seed"], q["epochs"]), (0, 5))
        self.assertEqual(set(q["checkpoint_state_sha256"]), {"5"})
        self.assertRegex(compact["created"], r"[+-]\d\d:\d\d$")           # time-zone aware
        self.assertNotIn("repetitions", text)
        self.assertNotIn("process_id", text)

    def test_changed_device_record_blocks_c2_c3_and_stage_b(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"; repo.mkdir()
            _synthetic.init_repo(repo)
            gt, est = _synthetic.inputs("gt"), _synthetic.inputs("est", shift=0.05)
            ex.stage_a_c1(gt, DEVICE, base / "out", plan=TINY_PLAN, prov=_synthetic.prov(repo, "a" * 64),
                          repo_root=repo)
            rec_path = repo / "configs" / "selection.json"
            ex.select_and_record(base / "out", gt.y, rec_path, None, plan=TINY_PLAN,
                                 prov=_synthetic.prov(repo, "a" * 64), repo_root=repo)
            _synthetic.commit(repo, rec_path)
            args = dict(device=DEVICE, out_root=base / "out", record_path=rec_path, repo_root=repo,
                        records_dir=repo / "records", plan=TINY_PLAN, prov=_synthetic.prov(repo, "b" * 64))
            for fn, a in ((ex.stage_a_c2, (gt, est)), (ex.stage_a_c3, (est,)), (ex.stage_b, (gt, est))):
                with self.assertRaises(ex.ProvenanceError):
                    fn(*a, **args)
            self.assertFalse(any((base / "out").glob("stage_a_c2*")) or any((base / "out").glob("stage_b*")))

    def test_record_tracking_and_commit_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            _synthetic.init_repo(repo)
            path = ex.device_record_path(repo / "configs" / "recognition")
            env = determinism.environment_record("cpu")
            runs.write_record(path, {"qualified_device": "cpu", "environment": env})
            with self.assertRaises(runs.CommitGateError):              # untracked
                ex.prepare_device(path, repo)
            _synthetic.commit(repo, path)
            self.assertEqual(ex.prepare_device(path, repo), "cpu")
            with self.assertRaises(FileExistsError):                    # never overwritten by the code
                runs.write_record(path, {"qualified_device": "cuda:0"})
            path.write_text(path.read_text(encoding="utf-8").replace('"cpu"', '"cuda:0"', 1), encoding="utf-8")
            with self.assertRaises(runs.CommitGateError):              # edited after commit
                ex.prepare_device(path, repo)


class TestQualificationInputs(unittest.TestCase):
    """Point 5: the qualification uses ground-truth fitting inputs only."""

    def test_fitting_only(self):
        gt = _synthetic.inputs("gt", n_fit=45, n_val=30)
        X, y = ex.qualification_inputs(gt)
        self.assertEqual((X.shape[0], y.shape[0]), (45, 45))
        X_fit, y_fit, _ = gt.part("fitting")
        np.testing.assert_array_equal(X, X_fit)
        with self.assertRaises(ValueError):
            ex.qualification_inputs(_synthetic.inputs("est"))


if __name__ == "__main__":
    unittest.main()
