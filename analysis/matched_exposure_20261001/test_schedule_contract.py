import hashlib
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ScheduleContractTests(unittest.TestCase):
    def test_schedules_are_contiguous_and_common_to_all_arms(self):
        receipt = json.loads((HERE / "matched_exposure_receipt.json").read_text())
        seen = set()
        for seed, item in receipt["schedules"].items():
            path = HERE / f"schedule_{seed}.json"
            self.assertEqual(sha(path), item["sha256"])
            schedule = json.loads(path.read_text())
            self.assertEqual(schedule["slots"], [f"slot_{i:02}.wav" for i in range(6)])
            cursor = 58107
            slots = []
            for episode in schedule["episodes"]:
                self.assertEqual(episode["start_step"], cursor)
                self.assertEqual(episode["chunks"], 256)
                self.assertEqual(episode["source_frame"] % 4096, 0)
                slot = episode["slot"]
                slots.append(slot)
                for arm in "abc":
                    source = receipt["arms"][arm]["slots"][slot]
                    self.assertLessEqual(episode["source_frame"] + 256 * 4096, source["frames"])
                    self.assertEqual(source["alias"], f"slot_{slot:02}.wav")
                cursor += 256
            self.assertEqual(cursor, 59131)
            self.assertEqual(len(set(slots)), 4)
            seen.update(slots)
        self.assertEqual(seen, set(range(6)))

    def test_all_manifests_have_training_only_slots_and_matching_holdouts(self):
        receipt = json.loads((HERE / "matched_exposure_receipt.json").read_text())
        heldout = receipt["heldout_sha256"]
        for arm in "abc":
            corpus = HERE / "runs/corpus" / arm
            manifest = HERE / "runs/corpus" / f"{arm}_manifest.json"
            self.assertEqual(sha(manifest), receipt["arms"][arm]["manifest_sha256"])
            entries = json.loads(manifest.read_text())["entries"]
            train = [item for item in entries if item["role"] == "train"]
            self.assertEqual({item["file"] for item in train},
                             {f"slot_{i:02}.wav" for i in range(6)})
            self.assertEqual({item["family"] for item in train},
                             {f"slot_{i:02}" for i in range(6)})
            self.assertEqual({item["file"] for item in entries if item["role"] != "train"},
                             set(heldout))
            for name, expected in heldout.items():
                self.assertEqual(sha(corpus / name), expected)
            self.assertFalse({sha(corpus / item["file"]) for item in train} & set(heldout.values()))


if __name__ == "__main__":
    unittest.main()
