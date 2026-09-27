import json
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import review_completion_audit as review  # noqa: E402


class ReviewCompletionAuditTests(unittest.TestCase):
    def test_transport_success_with_truncated_or_empty_answer_is_incomplete(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "reviews.json"
            path.write_text(json.dumps([
                {"status": "ok", "finish_reason": "length", "answer": "partial", "role": "skeptic"},
                {"status": "ok", "finish_reason": "stop", "answer": "  ", "role": "designer"},
                {"status": "ok", "finish_reason": "stop", "answer": "complete review", "role": "auditor"},
            ]))
            result = review.audit(path)
            self.assertEqual((result["complete_count"], result["incomplete_count"]), (1, 2))
            self.assertEqual([item["reason"] for item in result["entries"]],
                             ["nonstop_finish", "empty_final_answer", None])
            self.assertNotIn("complete review", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
