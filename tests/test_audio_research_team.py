import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    "audio_research_team", Path(__file__).resolve().parents[1] / "scripts/audio_research_team.py")
team = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(team)


class AudioTeamTests(unittest.TestCase):
    def test_incomplete_or_wrong_causal_receipt_cannot_enter_synthesis(self):
        valid = {"status": "ok", "finish_reason": "stop",
                 "requested_model": "deepseek/deepseek-v4.1-flash",
                 "returned_model": "deepseek/deepseek-v4.1-flash", "answer_json": {
            "summary": "Bounded evidence review.", "context_receipt": dict(team.RECEIPT)}}
        self.assertTrue(team.audit(valid)["eligible_for_synthesis"])
        for field, value in (("finish_reason", "length"), ("status", "incomplete")):
            broken = copy.deepcopy(valid)
            broken[field] = value
            self.assertFalse(team.audit(broken)["eligible_for_synthesis"])
        for wrong in (True, 0, None):
            broken = copy.deepcopy(valid)
            broken["answer_json"]["context_receipt"]["direct_reference_input"] = wrong
            self.assertFalse(team.audit(broken)["eligible_for_synthesis"])
        empty = copy.deepcopy(valid)
        empty["answer_json"]["summary"] = ""
        self.assertFalse(team.audit(empty)["eligible_for_synthesis"])
        changed_model = copy.deepcopy(valid)
        changed_model["returned_model"] = "another/model"
        self.assertFalse(team.audit(changed_model)["eligible_for_synthesis"])

    def test_selected_context_keeps_source_lines_and_refuses_truncation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "source.rs").write_text("one\ntwo\nthree\n")
            builder = team.ContextBuilder(root)
            text, identity = builder.selection("source.rs:2-3")
            self.assertIn("2: two\n3: three", text)
            self.assertEqual(identity["sha256"], team.digest(b"two\nthree\n"))
            with self.assertRaises(ValueError):
                builder.selection("source.rs:1-4")
            with self.assertRaises(ValueError):
                builder.build({"core": "source.rs", "evidence": [], "max_request_bytes": 8192})
            protected = root / ".agents"
            protected.mkdir()
            (protected / "private.txt").write_text("not selected")
            with self.assertRaises(ValueError):
                builder.selection(".agents/private.txt")


if __name__ == "__main__":
    unittest.main()
