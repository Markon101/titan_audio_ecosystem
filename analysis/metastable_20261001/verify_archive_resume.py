#!/usr/bin/env python3
"""Compare uninterrupted archive replay with an exact JSON save/resume."""

import hashlib
import json
from pathlib import Path
import argparse

from regime_archive import Archive, read_capture

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, default=HERE / "observer_capture.jsonl.gz")
    parser.add_argument("--second", type=Path, default=HERE / "observer_capture_segment2.jsonl.gz")
    parser.add_argument("--calibration", type=Path, default=HERE / "regime_calibration.json")
    parser.add_argument("--out", type=Path, default=HERE / "archive_resume_gate.json")
    args = parser.parse_args()
    first, second, calpath = args.first, args.second, args.calibration
    cal = json.loads(calpath.read_text())
    a, b = list(read_capture([first])), list(read_capture([second]))
    uninterrupted = Archive(cal)
    expected = [uninterrupted.process(row) for row in a + b]
    split = Archive(cal)
    actual = [split.process(row) for row in a]
    restored = Archive(cal, json.loads(json.dumps(split.state, sort_keys=True)))
    actual += [restored.process(row) for row in b]
    trace_exact = json.dumps(expected, sort_keys=True) == json.dumps(actual, sort_keys=True)
    state_exact = uninterrupted.state == restored.state
    report = {"schema": 1, "first_capture_sha256": sha(first),
              "second_capture_sha256": sha(second), "calibration_sha256": sha(calpath),
              "samples": len(a) + len(b), "first_last_step": a[-1]["global_step"],
              "second_first_step": b[0]["global_step"],
              "trace_exact": trace_exact, "state_exact": state_exact,
              "final_regimes": len(restored.state["regimes"]),
              "final_transitions": restored.state["transition_count"]}
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report))
    if not (trace_exact and state_exact):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
