import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("download_freefall", ROOT / "scripts/download_freefall.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RestoreTests(unittest.TestCase):
    def test_restores_expected_path_and_replaces_stale_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cached = root / "cache.npy"
            cached.write_bytes(b"good")
            target = root / "paper/basis.npy"
            target.parent.mkdir()
            target.write_bytes(b"bad!")

            def download(**kwargs):
                self.assertEqual(kwargs, {"repo_id": "test/project", "filename": "artifacts/basis.npy"})
                return cached

            result = MODULE.restore(root, {"path": "paper/basis.npy", "filename": "artifacts/basis.npy"}, "test/project", download)
            self.assertEqual(result, target)
            self.assertEqual(target.read_bytes(), b"good")

    def test_rejects_path_outside_repository(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                MODULE.restore(Path(tmp), {"path": "../outside", "filename": "unused"}, "test/project", None)


if __name__ == "__main__":
    unittest.main()
