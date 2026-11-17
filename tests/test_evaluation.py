import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
SPEC = importlib.util.spec_from_file_location("judge_eval", ROOT / "eval/judge_qwenlm.py")
JUDGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(JUDGE)


class EvaluationTests(unittest.TestCase):
    def test_exhausted_judge_requests_do_not_become_incorrect_predictions(self):
        with patch("openai.OpenAI") as client, patch.object(JUDGE.time, "sleep"):
            client.return_value.chat.completions.create.side_effect = ConnectionError("unavailable")
            with self.assertRaisesRegex(RuntimeError, "failed after retries"):
                JUDGE.judge_via_api(["question"], "http://localhost:1/v1", "EMPTY", "judge", 16, 1)

    def test_accuracy_rejects_invalid_verdicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "judge.json"
            path.write_text(json.dumps([{"judge": "API_ERROR"}]))
            result = subprocess.run([sys.executable, str(ROOT / "eval/cal_acc.py"), "--benchmark", "mmstar", "--judge_json", str(path)], text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("valid Yes/No", result.stderr)


if __name__ == "__main__":
    unittest.main()
