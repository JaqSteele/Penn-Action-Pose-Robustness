"""Batch E preparation (protocol v0.15, Section 9.5.9): the final-evaluation provenance gate, the
hardened Stage B loader, test-input pairing safeguards, and protocol/configuration consistency.
Synthetic data and temporary Git repositories only; no test-partition data is read."""

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

import _common
import _synthetic
from _synthetic import DEVICE, TINY_PLAN
from penn_pose import batch_e, bootstrap, experiment as ex, integrity, runs

EXP_V1 = '''"""Synthetic experiment module."""

import json

PROTOCOL_VERSION = "v0.14"


def config_dict():
    return {"seed": None}


def stage_b():
    return "train"


def load_stage_b_models():
    return "load"
'''

FILES = {
    "src/penn_pose/experiment.py": EXP_V1,
    "src/penn_pose/bootstrap.py": "METHOD = None\n",
    "src/penn_pose/model.py": "LAYERS = 3\n",
    "src/penn_pose/training.py": "EPOCHS = 130\n",
    "scripts/run.py": "print('run')\n",
    "configs/recognition/release1.json": '{"seed": null}\n',
    "configs/inputs/release1-cache.json": "{}\n",
    "configs/splits/fitting.txt": "0001\n",
    "requirements.txt": "numpy\n",
    "docs/RESEARCH_PROTOCOL.md": "# Protocol v0.14\n",
    "docs/RECOGNITION.md": "# Recognition (seed null)\n",
}
SELECTION = "configs/recognition/release1-selection.json"
STAGE_B = "configs/recognition/release1-stage-b.json"
RECOGNITION = "docs/RECOGNITION.md"
ALLOWED = ("docs/RESEARCH_PROTOCOL.md", "configs/recognition/release1.json", "src/penn_pose/bootstrap.py",
           "src/penn_pose/experiment.py", "src/penn_pose/batch_e.py", RECOGNITION)
#: The allowed paths the protected-scope change rule can see (RECOGNITION.md is outside it).
ALLOWED_PROTECTED = tuple(p for p in ALLOWED if p != RECOGNITION)


class TestGate(unittest.TestCase):
    """``batch_e.check_code`` in a temporary repository: training commit -> Stage B record commit."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        g = self.git
        g("init", "-q")
        for rel, text in FILES.items():
            self.write(rel, text)
        g("add", "-A")
        g("commit", "-q", "-m", "training implementation")
        self.training = self.head()
        self.tree = runs.protected_tree_sha256(self.repo, self.training)
        self.write(SELECTION, json.dumps({"stage_a_implementation": {
            "commit": self.training, "protected_paths": list(runs.PROTECTED_PATHS),
            "protected_tree_sha256": self.tree}}) + "\n")
        self.commit_all("selection record")
        self.write(STAGE_B, '{"stage": "B"}\n')
        self.commit_all("stage B record")
        self.stage_b = self.head()

    def tearDown(self):
        self.tmp.cleanup()

    def git(self, *args):
        return _synthetic.git(self.repo, *args)

    def head(self):
        return self.git("rev-parse", "HEAD").stdout.strip()

    def write(self, rel, text):
        p = self.repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="\n")

    def commit_all(self, message="change"):
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)

    def gate(self, **kw):
        args = dict(training_commit=self.training, training_tree=self.tree, stage_b_commit=self.stage_b,
                    frozen_records=(SELECTION, STAGE_B), allowed_paths=ALLOWED,
                    experiment_path="src/penn_pose/experiment.py", experiment_mutable=batch_e.EXPERIMENT_MUTABLE,
                    selection_record=SELECTION)
        args.update(kw)
        return batch_e.check_code(self.repo, **args)

    def assert_rejected(self, exc=batch_e.BatchEGateError):
        with self.assertRaises(exc):
            self.gate()

    def allowed_changes(self):
        self.write("docs/RESEARCH_PROTOCOL.md", "# Protocol v0.15\n")
        self.write("configs/recognition/release1.json", '{"seed": 20261128}\n')
        self.write("src/penn_pose/bootstrap.py", "METHOD = 'linear'\n")
        self.write("src/penn_pose/batch_e.py", "GATE = True\n")
        self.write(RECOGNITION, "# Recognition (seed 20261128)\n")
        self.write("src/penn_pose/experiment.py", EXP_V1
                   .replace('"""Synthetic experiment module."""', '"""Synthetic experiment module (v0.15)."""')
                   .replace('"v0.14"', '"v0.15"')
                   .replace('return {"seed": None}', 'return {"seed": 20261128}')
                   .replace('return "load"', 'return "load and verify"'))
        self.commit_all("batch E preparation")

    def test_unchanged_repository_passes(self):
        ident = self.gate()
        self.assertEqual(ident["changed_protected_paths"], [])
        self.assertEqual(ident["batch_e_code"]["commit"], self.stage_b)

    def test_allowed_changes_pass_and_identity_is_recorded(self):
        self.allowed_changes()
        ident = self.gate()
        self.assertEqual(ident["changed_protected_paths"], sorted(ALLOWED_PROTECTED))
        self.assertEqual(ident["controlled_documents_sha256"],
                         {RECOGNITION: hashlib.sha256((self.repo / RECOGNITION).read_bytes()).hexdigest()})
        self.assertEqual(ident["batch_e_code"], runs.code_identity(self.repo))
        self.assertNotEqual(ident["batch_e_code"]["protected_tree_sha256"], self.tree)
        self.assertEqual((ident["training_commit"], ident["stage_b_commit"]), (self.training, self.stage_b))
        self.assertEqual(ident["training_protected_tree_sha256"], self.tree)
        self.assertEqual(ident["frozen_records_sha256"][STAGE_B],
                         hashlib.sha256((self.repo / STAGE_B).read_bytes()).hexdigest())

    def test_training_model_preprocessing_split_and_dependency_changes_rejected(self):
        for rel, text in (("src/penn_pose/model.py", "LAYERS = 4\n"), ("src/penn_pose/training.py", "EPOCHS = 131\n"),
                          ("configs/splits/fitting.txt", "0002\n"), ("configs/inputs/release1-cache.json", '{"x": 1}\n'),
                          ("requirements.txt", "numpy==9\n"), ("scripts/run.py", "print('other')\n"),
                          ("src/penn_pose/test_inputs.py", "NEW = 1\n"), ("configs/splits/validation.txt", "0003\n")):
            with self.subTest(rel=rel):
                self.allowed_changes() if rel == "src/penn_pose/model.py" else None
                before = self.head()
                self.write(rel, text)
                self.commit_all(f"change {rel}")
                with self.assertRaises(batch_e.BatchEGateError) as ctx:
                    self.gate()
                self.assertIn(rel, str(ctx.exception))
                self.git("reset", "-q", "--hard", before)

    def test_recognition_document_is_controlled_but_not_part_of_code_identity(self):
        tree_before = runs.code_identity(self.repo)["protected_tree_sha256"]
        self.write(RECOGNITION, "# Recognition (seed 20261128)\n")
        self.commit_all("document the bootstrap settings")
        ident = self.gate()
        self.assertEqual(ident["changed_protected_paths"], [])                       # outside the protected scope
        self.assertEqual(ident["batch_e_code"]["protected_tree_sha256"], tree_before)
        self.assertEqual(ident["controlled_documents_sha256"][RECOGNITION],
                         hashlib.sha256(b"# Recognition (seed 20261128)\n").hexdigest())
        self.write(RECOGNITION, "# Recognition (uncommitted edit)\n")
        self.assert_rejected(runs.CommitGateError)                                 # must be committed
        self.git("checkout", "--", RECOGNITION)
        with self.assertRaises(batch_e.BatchEGateError):                           # only listed documents
            self.gate(allowed_paths=ALLOWED_PROTECTED)
        self.git("rm", "-q", RECOGNITION)
        self.git("commit", "-q", "-m", "remove document")
        self.assert_rejected(runs.CommitGateError)

    def test_deleted_protected_file_rejected(self):
        self.git("rm", "-q", "src/penn_pose/model.py")
        self.git("commit", "-q", "-m", "remove model")
        self.assert_rejected()

    def test_experiment_module_outside_mutable_definitions_rejected(self):
        for new in (EXP_V1.replace('return "train"', 'return "train differently"'),       # frozen function
                    EXP_V1 + "\n\ndef extra():\n    return 1\n",                           # added definition
                    EXP_V1.replace("import json\n", "import json, os\n"),                  # changed import
                    EXP_V1.replace('def stage_b():\n    return "train"\n', ""),           # removed function
                    EXP_V1 + "\nVALUE = 1\n"):                                              # added assignment
            with self.subTest(new=new[-40:]):
                before = self.head()
                self.write("src/penn_pose/experiment.py", new)
                self.commit_all("experiment change")
                self.assert_rejected()
                self.git("reset", "-q", "--hard", before)

    def test_formatting_only_change_to_frozen_function_allowed(self):
        self.write("src/penn_pose/experiment.py", EXP_V1.replace('def stage_b():\n    return "train"',
                                                               'def stage_b():\n    # comment\n    return "train"'))
        self.commit_all("comment")
        self.assertEqual(self.gate()["changed_protected_paths"], ["src/penn_pose/experiment.py"])

    def test_uncommitted_or_untracked_changes_rejected(self):
        self.write("docs/RESEARCH_PROTOCOL.md", "# uncommitted\n")
        self.assert_rejected(runs.CodeIdentityError)
        self.git("checkout", "--", "docs/RESEARCH_PROTOCOL.md")
        self.write("src/penn_pose/batch_e.py", "UNTRACKED = 1\n")
        self.assert_rejected(runs.CodeIdentityError)

    def test_frozen_records_must_be_unchanged(self):
        self.write(STAGE_B, '{"stage": "B", "edited": true}\n')
        self.commit_all("rewrite stage B record")
        self.assert_rejected()
        self.git("reset", "-q", "--hard", "HEAD~1")
        self.write(STAGE_B, '{"stage": "B", "dirty": true}\n')                 # uncommitted edit
        self.assert_rejected(runs.CommitGateError)
        self.git("checkout", "--", STAGE_B)
        self.git("rm", "-q", SELECTION)
        self.git("commit", "-q", "-m", "drop selection")
        self.assert_rejected()

    def test_stage_b_commit_must_be_an_ancestor(self):
        self.git("checkout", "-q", "-b", "elsewhere", self.training)
        self.write("docs/RESEARCH_PROTOCOL.md", "# Protocol v0.15\n")
        self.write(SELECTION, (self.repo / SELECTION).read_text() if (self.repo / SELECTION).exists() else "{}\n")
        self.commit_all("history without Stage B")
        self.assert_rejected()

    def test_training_tree_and_selection_record_must_agree(self):
        with self.assertRaises(batch_e.BatchEGateError):
            self.gate(training_tree="0" * 64)
        self.write(SELECTION, json.dumps({"stage_a_implementation": {"commit": "f" * 40,
                                                                       "protected_tree_sha256": self.tree}}) + "\n")
        self.commit_all("selection edited")
        with self.assertRaises(batch_e.BatchEGateError):
            self.gate(frozen_records=(STAGE_B,))

    def test_worktree_preview(self):
        self.write("docs/RESEARCH_PROTOCOL.md", "# Protocol v0.15\n")
        self.write("src/penn_pose/model.py", "LAYERS = 5\n")
        self.assertEqual(batch_e.changed_protected_paths(self.repo, self.training, None),
                         ["docs/RESEARCH_PROTOCOL.md", "src/penn_pose/model.py"])
        problems = batch_e.code_change_problems(self.repo, None, training_commit=self.training, allowed_paths=ALLOWED)
        self.assertEqual(len(problems), 1)
        self.assertIn("src/penn_pose/model.py", problems[0])


class TestRealRepositoryConstants(unittest.TestCase):
    """The gate's constants against the committed records (read-only)."""

    def setUp(self):
        self.root = _common.REPO_ROOT
        have = _synthetic.subprocess.run(["git", "cat-file", "-e", batch_e.STAGE_B_COMMIT], cwd=self.root,
                                         capture_output=True).returncode == 0
        if not have:
            self.skipTest("Stage B commit not present in this clone")

    def test_constants_match_records(self):
        sel = runs.read_json(ex.SELECTION_RECORD)
        self.assertEqual(sel["stage_a_implementation"]["commit"], batch_e.TRAINING_COMMIT)
        self.assertEqual(sel["stage_a_implementation"]["protected_tree_sha256"], batch_e.TRAINING_PROTECTED_TREE)
        self.assertEqual(runs.protected_tree_sha256(self.root, batch_e.TRAINING_COMMIT), batch_e.TRAINING_PROTECTED_TREE)
        self.assertTrue(runs.is_ancestor(self.root, batch_e.STAGE_B_COMMIT, "HEAD"))
        self.assertTrue(runs.is_ancestor(self.root, batch_e.TRAINING_COMMIT, batch_e.STAGE_B_COMMIT))
        for rel in batch_e.FROZEN_RECORDS:
            self.assertEqual(batch_e._blob(self.root, batch_e.STAGE_B_COMMIT, rel), batch_e._blob(self.root, "HEAD", rel), rel)
        stage_b = runs.read_json(ex.RECORDS_DIR / "release1-stage-b.json")
        self.assertEqual(set(stage_b["conditions"]), {"C1", "C3"})

    def test_allow_list_is_exact(self):
        self.assertEqual(batch_e.ALLOWED_PATHS, ALLOWED)
        self.assertEqual(batch_e.CONTROLLED_DOCUMENTS, (RECOGNITION,))
        self.assertNotIn(RECOGNITION, runs.PROTECTED_PATHS)                     # documentation, not code identity
        self.assertEqual(batch_e.EXPERIMENT_MUTABLE, ("__doc__", "PROTOCOL_VERSION", "config_dict", "load_stage_b_models"))
        self.assertEqual(batch_e.FROZEN_RECORDS, tuple(f"configs/recognition/release1-{n}.json" for n in
                                                       ("device", "selection", "stage-a-c2", "stage-a-c3", "stage-b")))

    def test_working_tree_satisfies_the_change_rule(self):
        self.assertEqual(batch_e.code_change_problems(self.root, None), [])


class TestHardenedStageBLoader(unittest.TestCase):
    """``experiment.load_stage_b_models`` checks run records and provenance, not only hashes."""

    @classmethod
    def setUpClass(cls):
        _synthetic.configure_cpu()
        cls.tmp = tempfile.TemporaryDirectory()
        base = Path(cls.tmp.name)
        cls.repo = base / "repo"
        cls.repo.mkdir()
        _synthetic.init_repo(cls.repo)
        cls.out = base / "experiments"
        records = cls.repo / "records"
        cls.record_path = cls.repo / "configs" / "selection.json"
        gt, est = _synthetic.inputs("gt"), _synthetic.inputs("est", shift=0.05)
        ex.stage_a_c1(gt, DEVICE, cls.out, plan=TINY_PLAN, prov=_synthetic.prov(cls.repo), repo_root=cls.repo)
        ex.select_and_record(cls.out, gt.y, cls.record_path, None, plan=TINY_PLAN, prov=_synthetic.prov(cls.repo),
                             repo_root=cls.repo)
        _synthetic.commit(cls.repo, cls.record_path)
        cls.stage_b = ex.stage_b(gt, est, device=DEVICE, out_root=cls.out, record_path=cls.record_path,
                                 repo_root=cls.repo, records_dir=records, plan=TINY_PLAN, prov=_synthetic.prov(cls.repo))
        cls.selection = runs.read_json(cls.record_path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.work = Path(tempfile.mkdtemp(dir=self.tmp.name))
        for cond in ("c1", "c3"):
            shutil.copytree(self.out / f"stage_b_{cond}", self.work / f"stage_b_{cond}")
        self.record = json.loads(json.dumps(self.stage_b))

    def load(self, record=None, selection=None):
        return ex.load_stage_b_models(self.work, record or self.record, DEVICE,
                                      selection_record=selection or self.selection, repo_root=self.repo)

    def forge(self, cond, seed, edit=None, extra_file=None):
        """Edit one run, then rewrite its run manifest, the stage manifest and the record's stage-manifest
        hash, so that only the semantic checks can catch the change."""
        stage_dir = self.work / f"stage_b_{cond.lower()}"
        run_dir = next(stage_dir.glob(f"*_seed{seed}"))
        if edit:
            run = runs.read_json(run_dir / "run.json")
            edit(run)
            (run_dir / "run.json").write_bytes(runs._dump_json(run))
        if extra_file:
            (run_dir / extra_file).write_bytes(b"extra")
        files = sorted(p.name for p in run_dir.iterdir() if p.name != runs.RUN_MANIFEST)
        (run_dir / runs.RUN_MANIFEST).write_text("".join(f"{integrity.sha256_file(run_dir / n)} *{n}\n" for n in files),
                                                 encoding="ascii", newline="\n")
        (stage_dir / "stage-sha256.txt").unlink()
        self.record["conditions"][cond]["stage_manifest_sha256"] = runs.write_stage_manifest(
            stage_dir, sorted(p.name for p in stage_dir.iterdir() if p.is_dir()))

    def test_valid_outputs_load(self):
        models = self.load()
        self.assertEqual({k: sorted(v) for k, v in models.items()}, {"C1": [0, 1], "C3": [0, 1]})

    def test_forged_manifests_alone_do_not_hide_semantic_changes(self):
        self.forge("C1", 0)                                            # re-manifest without edits: still valid
        self.load()

    def test_run_record_changes_rejected(self):
        E = self.record["epochs"]
        edits = {
            "epochs": lambda r: r.update(epochs=E + 1),
            "training_metrics": lambda r: r.update(training_metrics=r["training_metrics"][:-1]),
            "evaluation": lambda r: r.update(evaluation={"macro_f1": 1.0}),
            "early_stopping": lambda r: r.update(early_stopping=True),
            "config": lambda r: r["config"].update(dropout=0.3),
            "seed": lambda r: r.update(seed=5),
            "training_videos": lambda r: r.update(training_videos=r["training_videos"] - 1),
            "train_inputs": lambda r: r.update(train_inputs="est all training videos"),
            "condition": lambda r: r.update(condition="C3"),
            "selection_record_commit": lambda r: r.update(selection_record_commit="0" * 40),
        }
        for name, edit in edits.items():
            with self.subTest(name=name):
                self.setUp()
                self.forge("C1", 1, edit)
                with self.assertRaises((ValueError, ex.ProvenanceError)):
                    self.load()

    def test_provenance_changes_rejected(self):
        edits = {
            "protocol": lambda r: r["provenance"].update(protocol="v9.99"),
            "device record": lambda r: r["provenance"].update(device_record_sha256="e" * 64),
            "environment": lambda r: r["provenance"]["environment"].update(torch="0.0"),
            "protected tree": lambda r: r["provenance"]["code"].update(protected_tree_sha256="a" * 64),
            "commit": lambda r: r["provenance"]["code"].update(commit="0" * 40),
            "missing": lambda r: r.pop("provenance"),
        }
        for name, edit in edits.items():
            with self.subTest(name=name):
                self.setUp()
                self.forge("C3", 0, edit)
                with self.assertRaises((ValueError, ex.ProvenanceError)):
                    self.load()

    def test_unmatched_seed_initial_state_rejected(self):
        self.forge("C3", 1, lambda r: r.update(initial_state_sha256="b" * 64))
        with self.assertRaises(ValueError):
            self.load()

    def test_extra_file_in_run_rejected(self):
        self.forge("C1", 0, extra_file="predictions.npz")
        with self.assertRaises(ValueError):
            self.load()

    def test_record_inconsistent_with_selection_rejected(self):
        for edit in (lambda r: r.update(epochs=r["epochs"] + 1),
                     lambda r: r["config"].update(learning_rate=0.1),
                     lambda r: r["conditions"].pop("C3"),
                     lambda r: r["conditions"]["C1"]["final_state_sha256"].pop("1")):
            with self.subTest():
                record = json.loads(json.dumps(self.stage_b))
                edit(record)
                with self.assertRaises(ValueError):
                    self.load(record)


class TestTestPairing(unittest.TestCase):
    def setUp(self):
        self.ids = np.array(["0003", "0010", "0101", "1000"])
        self.y = np.array([0, 4, 14, 7], dtype=np.int64)

    def test_valid_pair(self):
        ids, y = batch_e.check_test_pairing(self.ids, self.y, self.ids.copy(), self.y.copy(), expected_ids=list(self.ids[::-1]))
        np.testing.assert_array_equal(ids, self.ids)
        np.testing.assert_array_equal(y, self.y)

    def test_rejections(self):
        cases = {
            "unsorted": (self.ids[::-1], self.y[::-1], self.ids[::-1], self.y[::-1], None),
            "duplicate": (np.array(["0003", "0003", "0101", "1000"]), self.y, np.array(["0003", "0003", "0101", "1000"]), self.y, None),
            "ids differ": (self.ids, self.y, np.array(["0003", "0010", "0101", "1001"]), self.y, None),
            "labels differ": (self.ids, self.y, self.ids, np.array([0, 4, 14, 8]), None),
            "label range": (self.ids, np.array([0, 4, 15, 7]), self.ids, np.array([0, 4, 15, 7]), None),
            "length": (self.ids, self.y[:3], self.ids, self.y[:3], None),
            "expected": (self.ids, self.y, self.ids, self.y, ["0003", "0010", "0101"]),
            "not strings": (np.array([3, 10, 101, 1000]), self.y, np.array([3, 10, 101, 1000]), self.y, None),
            "empty": (np.array([], dtype="<U4"), np.array([], dtype=np.int64), np.array([], dtype="<U4"), np.array([], dtype=np.int64), None),
        }
        for name, (gi, gy, ei, ey, expected) in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    batch_e.check_test_pairing(gi, gy, ei, ey, expected_ids=expected)

    def test_string_valued_object_arrays_supported(self):
        obj = np.array(list(self.ids), dtype=object)
        ids, y = batch_e.check_test_pairing(obj, self.y, obj.copy(), self.y.copy())
        self.assertEqual(ids.dtype.kind, "U")
        self.assertEqual(ids.tolist(), self.ids.tolist())
        mixed_str = np.array([str(self.ids[0]), np.str_(self.ids[1]), str(self.ids[2]), np.str_(self.ids[3])], dtype=object)
        batch_e.check_test_pairing(mixed_str, self.y, self.ids, self.y)      # numpy.str_ is a str
        batch_e.check_test_pairing(self.ids, self.y, obj, self.y)            # Unicode and object arrays agree

    def test_non_string_identifiers_rejected_without_conversion(self):
        cases = {
            "object ints": np.array([3, 10, 101, 1000], dtype=object),
            "object mixed": np.array(["0003", 10, "0101", "1000"], dtype=object),
            "object float": np.array(["0003", "0010", 101.0, "1000"], dtype=object),
            "object bytes": np.array(["0003", b"0010", "0101", "1000"], dtype=object),
            "object None": np.array(["0003", "0010", None, "1000"], dtype=object),
            "byte strings": np.array([b"0003", b"0010", b"0101", b"1000"]),
            "integers": np.array([3, 10, 101, 1000]),
            "floats": np.array([3.0, 10.0, 101.0, 1000.0]),
            "2-D": np.array([["0003", "0010"], ["0101", "1000"]]),
        }
        for name, bad in cases.items():
            for args in ((bad, self.y, self.ids, self.y), (self.ids, self.y, bad, self.y)):
                with self.subTest(name=name, side="gt" if args[0] is bad else "est"):
                    with self.assertRaises(ValueError):
                        batch_e.check_test_pairing(*args)
        with self.assertRaises(ValueError):                                 # expected IDs must be strings too
            batch_e.check_test_pairing(self.ids, self.y, self.ids, self.y, expected_ids=[3, 10, 101, 1000])
        with self.assertRaises(ValueError):
            batch_e.check_test_pairing(self.ids, self.y, self.ids, self.y, expected_ids=["0003", "0010", "0101", 1000])

    def test_checked_bootstrap_wrappers(self):
        rng = np.random.default_rng(1)
        y = np.repeat(np.arange(15), 3)
        ids = np.array([f"{i:04d}" for i in range(y.size)])
        a = {s: np.where(rng.random(y.size) < 0.8, y, 0) for s in (0, 1, 2)}
        b = {s: np.where(rng.random(y.size) < 0.6, y, 0) for s in (0, 1, 2)}
        got = batch_e.checked_paired_comparison(ids, y, ids, y, a, b, seed=20261128, n_resamples=50)
        self.assertEqual(got, bootstrap.paired_comparison(y, a, b, seed=20261128, n_resamples=50))
        summary = batch_e.checked_condition_summary(ids, y, ids, y, a, seed=20261128, n_resamples=50)
        self.assertEqual(summary, bootstrap.condition_summary(y, a, seed=20261128, n_resamples=50))
        with self.assertRaises(ValueError):
            batch_e.checked_paired_comparison(ids, y, ids[::-1], y[::-1], a, b, seed=20261128, n_resamples=50)
        with self.assertRaises(TypeError):                                 # still no default seed
            batch_e.checked_paired_comparison(ids, y, ids, y, a, b)

    def test_bootstrap_seed_from_configuration(self):
        cfg = ex.load_config()
        self.assertEqual(batch_e.bootstrap_seed(cfg), 20261128)
        for change in ({"seed": None}, {"seed": True}, {"sd_ddof": 0}, {"percentile_method": "nearest"},
                       {"n_resamples": 1000}):
            with self.subTest(change=change):
                altered = json.loads(json.dumps(cfg))
                altered["bootstrap"].update(change)
                with self.assertRaises(ValueError):
                    batch_e.bootstrap_seed(altered)


class TestProtocolConsistency(unittest.TestCase):
    """The committed protocol, configuration, bootstrap module and gate agree (v0.15)."""

    @classmethod
    def setUpClass(cls):
        cls.text = (_common.REPO_ROOT / "docs" / "RESEARCH_PROTOCOL.md").read_text(encoding="utf-8")
        cls.cfg = ex.load_config()

    def _section(self, start, end):
        i = self.text.index(start)
        return self.text[i:self.text.index(end, i + len(start))]

    def test_version(self):
        self.assertIn(f"**Document status:** Draft {ex.PROTOCOL_VERSION} for review.", self.text)
        self.assertEqual(self.cfg["protocol"], ex.PROTOCOL_VERSION)
        self.assertIn(f"| {ex.PROTOCOL_VERSION} | 2026-11-28 |", self.text)

    def test_decisions_recorded(self):
        seed = self.cfg["bootstrap"]["seed"]
        resolved = self._section("### 14.1 Resolved decisions", "### 14.2 Open")
        open_ = self._section("### 14.2 Open", "### 14.3")
        self.assertIn(f"| D-45 *(v0.14; resolved v0.15)* | Bootstrap seed value | **{seed}**", resolved)
        self.assertIn("| D-46 *(v0.15, clarification)* | Seed-variation standard deviation | Sample standard deviation (n − 1, `ddof = 1`)", resolved)
        self.assertIn('| D-47 *(v0.15, clarification)* | Percentile interpolation | Linear interpolation', resolved)
        self.assertNotIn("| D-45", open_)
        stats = self._section("### 10.7 Release 1.0 statistical plan", "## 11.")
        for ref in (f"**{seed}**", "`D-45`", "`D-46`", "`D-47`", 'method="linear"', "n − 1"):
            self.assertIn(ref, stats)
        for unchanged in ("**B = 10,000**", "The 2.5th and 97.5th percentiles", "the 1.25th and 98.75th",
                          "**H1:** C1 versus C2", "**H2:** C3 versus C2", "the mean of d<sub>s</sub> over the three seeds"):
            self.assertIn(unchanged, stats)

    def test_recognition_document_matches_configuration(self):
        """docs/RECOGNITION.md states the bootstrap settings; they must equal the configuration."""
        import re
        doc = (_common.REPO_ROOT / RECOGNITION).read_text(encoding="utf-8")
        i = doc.index("## Final-evaluation bootstrap settings (protocol ")
        section = doc[i:doc.index("\n## ", i + 1)]
        self.assertIn(f"(protocol {ex.PROTOCOL_VERSION})", section.splitlines()[0])
        rows = {m.group(1).strip(): (m.group(2).strip(), m.group(3).strip())
                for m in re.finditer(r"^\| ([^|]+) \| ([^|]+) \| (D-\d+) \|$", section, re.M)}
        b = self.cfg["bootstrap"]
        self.assertEqual(rows, {
            "Resamples (B)": (f"`{b['n_resamples']}`", "D-26"),
            "Seed": (f"`{b['seed']}`", "D-45"),
            "Seed-variation standard deviation": (f"sample SD, `ddof={b['sd_ddof']}`", "D-46"),
            "Percentile interpolation": (f'`method="{b["percentile_method"]}"`', "D-47"),
        })
        self.assertEqual((b["n_resamples"], b["seed"], b["sd_ddof"], b["percentile_method"]),
                         (10000, 20261128, 1, "linear"))
        self.assertNotIn("seed is `null`", doc)

    def test_configuration_matches_bootstrap_module_and_protocol(self):
        b = self.cfg["bootstrap"]
        self.assertEqual((b["n_resamples"], b["sd_ddof"], b["percentile_method"]),
                         (bootstrap.N_RESAMPLES, bootstrap.SD_DDOF, bootstrap.PERCENTILE_METHOD))
        section = self._section("#### 9.5.9", "\n---\n")
        self.assertIn(f"The bootstrap seed is **{b['seed']}**", section)
        self.assertIn("`ddof = 1`", section)
        self.assertIn('method="linear"', section)
        for const in (batch_e.TRAINING_COMMIT[:7], batch_e.STAGE_B_COMMIT[:7], batch_e.TRAINING_PROTECTED_TREE[:8]):
            self.assertIn(const, section)


if __name__ == "__main__":
    unittest.main()
