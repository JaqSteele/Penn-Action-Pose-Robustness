import tempfile
import unittest
from pathlib import Path

import _common
from penn_pose import integrity


class TestManifestParsing(unittest.TestCase):
    def test_parse_and_reject(self):
        with tempfile.TemporaryDirectory() as d:
            good = Path(d) / "m.txt"
            good.write_text("a" * 64 + " *0001.mat\n" + "b" * 64 + "  0002.mat\n", encoding="ascii")
            self.assertEqual(integrity.read_manifest(good), {"0001.mat": "a" * 64, "0002.mat": "b" * 64})
            bad = Path(d) / "bad.txt"
            bad.write_text("abc *0001.mat\n", encoding="ascii")
            with self.assertRaises(ValueError):
                integrity.read_manifest(bad)


@_common.requires_data
class TestLabelIntegrity(unittest.TestCase):
    def test_all_label_files_match_manifest(self):
        result = integrity.verify_labels(_common.DATA_ROOT)
        self.assertTrue(result.ok, (result.mismatched[:5], result.missing[:5], result.unexpected[:5]))
        self.assertEqual(result.checked, integrity.EXPECTED_LABEL_FILES)


if __name__ == "__main__":
    unittest.main()
