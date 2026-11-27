"""Provenance and code-identity enforcement, plus the review fixes for the C3 epoch check and
boolean argument validation. CPU, synthetic data and temporary Git repositories only."""

import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import _common
import _synthetic
from _synthetic import DEVICE, TINY_PLAN
from penn_pose import bootstrap, determinism, experiment as ex, runs, training
from penn_pose.training import TrainConfig


class _RepoCase(unittest.TestCase):
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

    def stage_a(self, prov, plan=TINY_PLAN):
        return ex.stage_a_c1(self.gt, DEVICE, self.out, plan=plan, prov=prov, repo_root=self.repo)

    def select(self, prov):
        return ex.select_and_record(self.out, self.gt.y, self.record_path, None, plan=TINY_PLAN,
                                    prov=prov, repo_root=self.repo)

    def later_args(self, prov):
        return dict(device=DEVICE, out_root=self.out, record_path=self.record_path, repo_root=self.repo,
                    records_dir=self.records, plan=TINY_PLAN, prov=prov)

    def run_json_bytes(self):
        return {p.parent.name: p.read_bytes() for p in sorted((self.out / "stage_a_c1").glob("*/run.json"))}


class TestCodeIdentity(_RepoCase):
    def test_clean_scope_identifies_head(self):
        ident = runs.code_identity(self.repo)
        head = _synthetic.git(self.repo, "rev-parse", "HEAD").stdout.strip()
        self.assertEqual(ident["commit"], head)
        self.assertEqual(ident["protected_paths"], list(runs.PROTECTED_PATHS))
        self.assertEqual(ident["protected_tree_sha256"], runs.protected_tree_sha256(self.repo, head))

    def test_untracked_modified_staged_and_ignored_files_are_rejected(self):
        untracked = self.repo / "src" / "penn_pose" / "extra.py"
        untracked.write_text("X = 1\n", encoding="utf-8")
        with self.assertRaises(runs.CodeIdentityError):
            runs.code_identity(self.repo)
        untracked.unlink()
        code = self.repo / "src" / "penn_pose" / "code.py"
        code.write_text("VALUE = 3\n", encoding="utf-8")
        with self.assertRaises(runs.CodeIdentityError):               # modified, not staged
            runs.code_identity(self.repo)
        _synthetic.git(self.repo, "add", "src/penn_pose/code.py")
        with self.assertRaises(runs.CodeIdentityError):               # staged, not committed
            runs.code_identity(self.repo)
        _synthetic.git(self.repo, "reset", "-q", "--hard")
        (self.repo / ".gitignore").write_text("*.local\n", encoding="utf-8")
        _synthetic.commit(self.repo, self.repo / ".gitignore", "ignore rule")
        hidden = self.repo / "scripts" / "settings.local"
        hidden.write_text("secret switch\n", encoding="utf-8")
        with self.assertRaises(runs.CodeIdentityError):               # ignored files are not exempt
            runs.code_identity(self.repo)
        hidden.unlink()
        cache = self.repo / "src" / "penn_pose" / "__pycache__"
        cache.mkdir()
        (cache / "code.cpython-311.pyc").write_bytes(b"\0")
        (self.repo / ".gitignore").write_text("*.local\n__pycache__/\n", encoding="utf-8")
        _synthetic.commit(self.repo, self.repo / ".gitignore", "ignore caches")
        runs.code_identity(self.repo)                                # bytecode caches are allowed

    def test_records_outside_scope_do_not_change_identity(self):
        before = runs.code_identity(self.repo)
        rec = self.repo / "configs" / "recognition" / "release1-selection.json"
        rec.write_text("{}\n", encoding="utf-8")
        (self.repo / "results").mkdir()
        (self.repo / "results" / "table.csv").write_text("a\n", encoding="utf-8")
        self.assertEqual(runs.code_identity(self.repo)["protected_tree_sha256"], before["protected_tree_sha256"])
        _synthetic.commit(self.repo, rec)
        after = runs.code_identity(self.repo)
        self.assertNotEqual(after["commit"], before["commit"])
        self.assertEqual(after["protected_tree_sha256"], before["protected_tree_sha256"])
        self.assertTrue(runs.is_ancestor(self.repo, before["commit"], after["commit"]))

    def test_protected_change_changes_identity(self):
        before = runs.code_identity(self.repo)
        for rel in ("src/penn_pose/code.py", "scripts/run.py", "configs/recognition/release1.json",
                    "configs/inputs/release1-cache.json", "configs/splits/fitting.txt", "requirements.txt",
                    "docs/RESEARCH_PROTOCOL.md"):
            path = self.repo / rel
            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            _synthetic.commit(self.repo, path, f"change {rel}")
            after = runs.code_identity(self.repo)
            self.assertNotEqual(after["protected_tree_sha256"], before["protected_tree_sha256"], rel)
            before = after

    def test_missing_protected_path_rejected(self):
        _synthetic.git(self.repo, "rm", "-q", "requirements.txt")
        _synthetic.git(self.repo, "commit", "-q", "-m", "remove")
        with self.assertRaises(runs.CodeIdentityError):
            runs.code_identity(self.repo)


class TestValidateProvenance(_RepoCase):
    def test_complete_and_current(self):
        ex.validate_provenance(_synthetic.prov(self.repo), self.repo)

    def test_missing_incomplete_or_stale(self):
        good = _synthetic.prov(self.repo)
        for bad in (None, {}, {k: v for k, v in good.items() if k != "code"},
                    {k: v for k, v in good.items() if k != "environment"}, {**good, "device_record_sha256": None}):
            with self.assertRaises(ex.ProvenanceError):
                ex.validate_provenance(bad, self.repo)
        _synthetic.change_code(self.repo)                               # HEAD and tree move on
        with self.assertRaises(ex.ProvenanceError):
            ex.validate_provenance(good, self.repo)

    def test_every_production_step_requires_provenance(self):
        _synthetic.commit(self.repo, self._write_record(), "record")
        steps = [
            lambda p: ex.stage_a_c1(self.gt, DEVICE, self.out, plan=TINY_PLAN, prov=p, repo_root=self.repo),
            lambda p: ex.select_and_record(self.out, self.gt.y, self.repo / "configs" / "other.json", None,
                                           plan=TINY_PLAN, prov=p, repo_root=self.repo),
            lambda p: ex.stage_a_c2(self.gt, self.est, **self.later_args(p)),
            lambda p: ex.stage_a_c3(self.est, **self.later_args(p)),
            lambda p: ex.stage_b(self.gt, self.est, **self.later_args(p)),
        ]
        partial = {"device_record_sha256": "d" * 64}
        for step in steps:
            for bad in (None, partial):
                with self.assertRaises(ex.ProvenanceError):
                    step(bad)
        self.assertFalse(self.out.exists())                             # nothing created by any step

    def _write_record(self):
        runs.write_record(self.record_path, {"selected": {"config": TINY_PLAN.grid[0].as_dict(), "epoch": 2},
                                             "provenance": _synthetic.prov(self.repo)})
        return self.record_path


class TestStageAProvenance(_RepoCase):
    def test_resume_rejects_old_consistent_runs_after_code_change(self):
        partial_plan = _synthetic.Plan(grid=TINY_PLAN.grid[:1], seeds=TINY_PLAN.seeds, max_epochs=4, eval_every=2)
        self.stage_a(_synthetic.prov(self.repo), plan=partial_plan)    # two consistent runs at commit A
        before = self.run_json_bytes()
        _synthetic.change_code(self.repo)                               # commit B changes protected code
        with self.assertRaises(ex.ProvenanceError):
            self.stage_a(_synthetic.prov(self.repo))                    # resume under the current provenance
        self.assertEqual(self.run_json_bytes(), before)                 # old runs untouched, none added

    def test_resume_rejects_other_device_record_or_environment(self):
        partial_plan = _synthetic.Plan(grid=TINY_PLAN.grid[:1], seeds=TINY_PLAN.seeds, max_epochs=4, eval_every=2)
        self.stage_a(_synthetic.prov(self.repo, "a" * 64), plan=partial_plan)
        before = self.run_json_bytes()
        with self.assertRaises(ex.ProvenanceError):
            self.stage_a(_synthetic.prov(self.repo, "b" * 64))
        env = dict(_synthetic.prov(self.repo)["environment"], torch="0.0")
        with self.assertRaises(ex.ProvenanceError):
            self.stage_a(_synthetic.prov(self.repo, "a" * 64, environment=env))
        self.assertEqual(self.run_json_bytes(), before)

    def test_resume_with_matching_provenance_completes(self):
        partial_plan = _synthetic.Plan(grid=TINY_PLAN.grid[:1], seeds=TINY_PLAN.seeds, max_epochs=4, eval_every=2)
        self.stage_a(_synthetic.prov(self.repo), plan=partial_plan)
        before = self.run_json_bytes()
        self.assertEqual(len(self.stage_a(_synthetic.prov(self.repo))), 4)
        after = self.run_json_bytes()
        self.assertEqual({k: after[k] for k in before}, before)        # resumed runs not rewritten
        self.assertEqual(len(after), 4)

    def test_selection_rejects_runs_from_another_implementation(self):
        self.stage_a(_synthetic.prov(self.repo))                         # complete, consistent, commit A
        _synthetic.change_code(self.repo)
        with self.assertRaises(ex.ProvenanceError):
            self.select(_synthetic.prov(self.repo))
        self.assertFalse(self.record_path.exists())
        self.assertFalse((self.out / "stage_a_c1" / "stage-sha256.txt").exists())

    def test_selection_rejects_after_unrelated_commit(self):
        # Stage A and selection must use one implementation commit (D-41 reruns, Section 9.5.7).
        self.stage_a(_synthetic.prov(self.repo))
        note = self.repo / "notes.txt"
        note.write_text("a note\n", encoding="utf-8")
        _synthetic.commit(self.repo, note, "unrelated")
        with self.assertRaises(ex.ProvenanceError):
            self.select(_synthetic.prov(self.repo))

    def test_stage_a_requires_committed_implementation(self):
        (self.repo / "src" / "penn_pose" / "new_module.py").write_text("X = 1\n", encoding="utf-8")
        with self.assertRaises(runs.CodeIdentityError):
            _synthetic.prov(self.repo)
        with self.assertRaises((ex.ProvenanceError, runs.CodeIdentityError)):
            self.stage_a({"device_record_sha256": "d" * 64})
        self.assertFalse(self.out.exists())


class TestLaterStageCodeIdentity(_RepoCase):
    def _through_selection(self):
        self.stage_a(_synthetic.prov(self.repo))
        self.select(_synthetic.prov(self.repo))
        _synthetic.commit(self.repo, self.record_path)                 # legitimate record commit

    def _steps(self):
        return ((ex.stage_a_c2, (self.gt, self.est)), (ex.stage_a_c3, (self.est,)), (ex.stage_b, (self.gt, self.est)))

    def test_record_commits_allowed(self):
        self._through_selection()
        other = self.repo / "configs" / "recognition" / "release1-notes.json"
        other.write_text("{}\n", encoding="utf-8")
        _synthetic.commit(self.repo, other, "another record")
        ex.stage_a_c3(self.est, **self.later_args(_synthetic.prov(self.repo)))
        self.assertTrue((self.out / "stage_a_c3" / "stage-sha256.txt").is_file())

    def test_committed_code_change_rejected(self):
        self._through_selection()
        _synthetic.change_code(self.repo)
        for fn, args in self._steps():
            with self.assertRaises(ex.ProvenanceError):
                fn(*args, **self.later_args(_synthetic.prov(self.repo)))
        self.assertFalse(any(self.out.glob("stage_a_c2*")) or any(self.out.glob("stage_a_c3*"))
                         or any(self.out.glob("stage_b*")))

    def test_uncommitted_or_untracked_change_rejected(self):
        self._through_selection()
        prov = _synthetic.prov(self.repo)
        extra = self.repo / "scripts" / "patch.py"
        extra.write_text("X = 1\n", encoding="utf-8")                  # untracked file in scope
        for fn, args in self._steps():
            with self.assertRaises((ex.ProvenanceError, runs.CodeIdentityError)):
                fn(*args, **self.later_args(prov))
        extra.unlink()
        (self.repo / "configs" / "splits" / "fitting.txt").write_text("0002\n", encoding="utf-8")   # modified
        for fn, args in self._steps():
            with self.assertRaises((ex.ProvenanceError, runs.CodeIdentityError)):
                fn(*args, **self.later_args(prov))
        self.assertFalse(any(self.out.glob("stage_a_c2*")) or any(self.out.glob("stage_b*")))

    def test_selection_from_unrelated_history_rejected(self):
        self._through_selection()
        record = runs.read_json(self.record_path)
        record["provenance"]["code"]["commit"] = "0" * 40               # not an ancestor of HEAD
        self.record_path.write_text(__import__("json").dumps(record), encoding="utf-8")
        _synthetic.commit(self.repo, self.record_path)
        with self.assertRaises(ex.ProvenanceError):
            ex.stage_a_c3(self.est, **self.later_args(_synthetic.prov(self.repo)))


class TestCanonicalProtectedPaths(_RepoCase):
    """Item 1: only the canonical protected-path list is ever accepted or used."""

    def _with_paths(self, paths):
        """Otherwise valid provenance whose code identity is computed for ``paths``."""
        prov = _synthetic.prov(self.repo)
        prov["code"] = runs.code_identity(self.repo, tuple(paths))
        return prov

    def test_canonical_list_includes_the_protocol(self):
        self.assertIn("docs/RESEARCH_PROTOCOL.md", runs.PROTECTED_PATHS)
        self.assertNotIn("docs/RECOGNITION.md", runs.PROTECTED_PATHS)
        self.assertEqual(_synthetic.prov(self.repo)["code"]["protected_paths"], list(runs.PROTECTED_PATHS))

    def test_narrowed_altered_reordered_or_missing_lists_rejected(self):
        canonical = list(runs.PROTECTED_PATHS)
        variants = {
            "narrowed": [p for p in canonical if p != "src"],
            "altered": [("README" if p == "requirements.txt" else p) for p in canonical],
            "reordered": list(reversed(canonical)),
        }
        for name, paths in variants.items():
            prov = self._with_paths(paths)                           # internally consistent identity
            with self.assertRaisesRegex(ex.ProvenanceError, "code.protected_paths must be exactly", msg=name):
                ex.validate_provenance(prov, self.repo)
        prov = _synthetic.prov(self.repo)
        del prov["code"]["protected_paths"]
        with self.assertRaisesRegex(ex.ProvenanceError, "code.protected_paths must be exactly"):
            ex.validate_provenance(prov, self.repo)

    def test_recorded_non_canonical_paths_rejected(self):
        expected = _synthetic.prov(self.repo)
        recorded = self._with_paths([p for p in runs.PROTECTED_PATHS if p != "scripts"])
        with self.assertRaisesRegex(ex.ProvenanceError, r"code\.protected_paths"):
            ex.check_provenance(recorded, expected, self.repo, same_commit=True)
        with self.assertRaisesRegex(ex.ProvenanceError, "canonical protected paths"):
            ex.check_provenance(expected, recorded, self.repo, same_commit=True)


class TestProtocolDocumentProtected(_RepoCase):
    """Item 2: the research protocol is part of the protected identity."""

    def _change_protocol(self):
        path = self.repo / "docs" / "RESEARCH_PROTOCOL.md"
        path.write_text(path.read_text(encoding="utf-8") + "Amended.\n", encoding="utf-8")
        _synthetic.commit(self.repo, path, "protocol amendment")

    def test_protocol_change_changes_identity_and_invalidates_provenance(self):
        before = _synthetic.prov(self.repo)
        self._change_protocol()
        self.assertNotEqual(runs.code_identity(self.repo)["protected_tree_sha256"],
                            before["code"]["protected_tree_sha256"])
        with self.assertRaisesRegex(ex.ProvenanceError, "does not describe the repository now"):
            ex.validate_provenance(before, self.repo)
        with self.assertRaisesRegex(ex.ProvenanceError, r"code\.protected_tree_sha256"):
            ex.check_provenance(before, _synthetic.prov(self.repo), self.repo, same_commit=False)

    def test_protocol_change_blocks_resume_and_later_stages(self):
        partial_plan = _synthetic.Plan(grid=TINY_PLAN.grid[:1], seeds=TINY_PLAN.seeds, max_epochs=4, eval_every=2)
        self.stage_a(_synthetic.prov(self.repo), plan=partial_plan)
        before = self.run_json_bytes()
        self._change_protocol()
        with self.assertRaisesRegex(ex.ProvenanceError, r"code\.protected_tree_sha256"):
            self.stage_a(_synthetic.prov(self.repo))
        self.assertEqual(self.run_json_bytes(), before)

    def test_protocol_change_after_selection_blocks_c2_c3_and_stage_b(self):
        self.stage_a(_synthetic.prov(self.repo))
        self.select(_synthetic.prov(self.repo))
        _synthetic.commit(self.repo, self.record_path)                 # still allowed: a record commit
        self._change_protocol()
        for fn, args in ((ex.stage_a_c2, (self.gt, self.est)), (ex.stage_a_c3, (self.est,)),
                         (ex.stage_b, (self.gt, self.est))):
            with self.assertRaisesRegex(ex.ProvenanceError, r"code\.protected_tree_sha256"):
                fn(*args, **self.later_args(_synthetic.prov(self.repo)))
        self.assertFalse(any(self.out.glob("stage_a_c2*")) or any(self.out.glob("stage_a_c3*"))
                         or any(self.out.glob("stage_b*")))


class TestEnvironmentAndHashCompleteness(_RepoCase):
    """Item 3: locked environment fields and hashes must be present and well formed."""

    def assert_invalid(self, prov, pattern):
        with self.assertRaisesRegex(ex.ProvenanceError, pattern):
            ex.validate_provenance(prov, self.repo)

    def test_fixture_environment_is_valid(self):
        prov = _synthetic.prov(self.repo)
        self.assertEqual(determinism.locked_environment_problems(prov["environment"]), [])
        ex.validate_provenance(prov, self.repo)

    def test_missing_or_null_locked_fields(self):
        for field in determinism.LOCKED_FIELDS:
            prov = _synthetic.prov(self.repo)
            del prov["environment"][field]
            self.assert_invalid(prov, rf"environment\.{field} is missing")
            prov = _synthetic.prov(self.repo)
            prov["environment"][field] = None
            self.assert_invalid(prov, rf"environment\.{field} is null")
        self.assert_invalid(_synthetic.prov(self.repo, environment=None), "environment is missing")

    def test_malformed_locked_fields(self):
        cases = [("device", "gpu", "not a CPU or CUDA device"), ("device", 0, "not a CPU or CUDA device"),
                 ("device_type", "cuda", "does not match device"), ("device_name", "  ", "device_name must be"),
                 ("torch", 2.15, "torch must be"), ("cuda", "", "cuda must be"),
                 ("cudnn", "92600", "cudnn must be"), ("cudnn", True, "cudnn must be"), ("cudnn", 0, "cudnn must be")]
        for field, value, pattern in cases:
            prov = _synthetic.prov(self.repo)
            prov["environment"][field] = value
            self.assert_invalid(prov, pattern)

    def test_hash_formats(self):
        for key in ("device_record_sha256", "inputs_record_sha256"):
            for bad in (None, "", "x" * 64, "a" * 63, "a" * 65, 12345):
                self.assert_invalid(_synthetic.prov(self.repo, **{key: bad}), f"{key} is not a SHA-256 hash")
        ex.validate_provenance(_synthetic.prov(self.repo, device_record_sha256="AB" * 32), self.repo)   # any case
        good = _synthetic.prov(self.repo)["split_sha256"]
        self.assert_invalid(_synthetic.prov(self.repo, split_sha256={"fitting": good["fitting"]}), "exactly the keys")
        self.assert_invalid(_synthetic.prov(self.repo, split_sha256={**good, "test": "a" * 64}), "exactly the keys")
        self.assert_invalid(_synthetic.prov(self.repo, split_sha256={**good, "fitting": "z" * 64}),
                            r"split_sha256\.fitting is not a SHA-256 hash")
        self.assert_invalid(_synthetic.prov(self.repo, split_sha256=None), "exactly the keys")


class TestC3EpochCheck(unittest.TestCase):
    def test_explicit_check(self):
        ex.require_only_epoch({4: {}}, 4)
        for bad in ({2: {}, 4: {}}, {2: {}}, {}):
            with self.assertRaises(RuntimeError):
                ex.require_only_epoch(bad, 4)

    def test_enforced_under_python_optimisation(self):
        code = ("import sys; sys.path.insert(0, 'src'); from penn_pose import experiment as ex\n"
                "assert False, 'asserts are active'\n")
        code_check = ("import sys; sys.path.insert(0, 'src'); from penn_pose import experiment as ex\n"
                      "try:\n    ex.require_only_epoch({2: {}, 4: {}}, 4)\nexcept RuntimeError:\n    print('raised')\n")
        asserts = subprocess.run([sys.executable, "-O", "-B", "-c", code], cwd=_common.REPO_ROOT,
                                 capture_output=True, text=True)
        self.assertEqual(asserts.returncode, 0, "the subprocess must run with assertions disabled")
        proc = subprocess.run([sys.executable, "-O", "-B", "-c", code_check], cwd=_common.REPO_ROOT,
                              capture_output=True, text=True)
        self.assertEqual(proc.stdout.strip(), "raised", proc.stderr[-2000:])

    def test_no_assert_statements_in_experiment_code(self):
        files = list((_common.REPO_ROOT / "src" / "penn_pose").glob("*.py")) + list((_common.REPO_ROOT / "scripts").glob("*.py"))
        for path in files:
            self.assertIsNone(re.search(r"^\s*assert\s", path.read_text(encoding="utf-8"), re.M), path.name)
        text = (_common.REPO_ROOT / "src" / "penn_pose" / "experiment.py").read_text(encoding="utf-8")
        self.assertIn("require_only_epoch(result.checkpoints, E)", text)


class TestBooleanArguments(unittest.TestCase):
    def test_bootstrap_resample_count(self):
        y = np.repeat(np.arange(15), 3)
        for bad in (True, False, 0, -1, 1.5, "10", None):
            with self.assertRaises(ValueError):
                bootstrap.resample_indices(y, bad, 1)
        self.assertEqual(bootstrap.resample_indices(y, np.int64(5), 1).shape, (5, 45))
        with self.assertRaises(ValueError):
            bootstrap.paired_comparison(y, {0: y}, {0: y}, seed=1, n_resamples=True)

    def test_training_epochs(self):
        _synthetic.configure_cpu()
        X, y = _synthetic.data(45)
        for bad in (True, False):
            with self.assertRaises(ValueError):
                training.train_model(X, y, TrainConfig(0.001, 0.0001, 0.1), 0, bad, "cpu")


if __name__ == "__main__":
    unittest.main()
