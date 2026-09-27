#!/usr/bin/env python3
"""Audit external research-review completion without copying answer contents."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_entry(entry: dict, index: int) -> dict:
    answer = entry.get("answer")
    text = answer if isinstance(answer, str) else ""
    status = entry.get("status")
    finish = entry.get("finish_reason")
    complete = status == "ok" and finish == "stop" and bool(text.strip())
    usage = entry.get("usage") if isinstance(entry.get("usage"), dict) else {}
    return {
        "index": index,
        "role": entry.get("research_role") or entry.get("role"),
        "model": entry.get("returned_model") or entry.get("model"),
        "status": status,
        "finish_reason": finish,
        "answer_nonempty": bool(text.strip()),
        "answer_chars": len(text),
        "answer_sha256": hashlib.sha256(text.encode()).hexdigest() if text else None,
        "prompt_tokens": usage.get("prompt_tokens") or entry.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "complete_for_research": complete,
        "reason": None if complete else (
            "nonstop_finish" if finish != "stop" else
            "transport_failed" if status != "ok" else "empty_final_answer"
        ),
    }


def audit(path: Path) -> dict:
    source = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(source, dict):
        entries = [source]
    elif isinstance(source, list) and all(isinstance(item, dict) for item in source):
        entries = source
    else:
        raise ValueError("review source must be a JSON object or list of objects")
    reviewed = [inspect_entry(item, index) for index, item in enumerate(entries)]
    complete = sum(item["complete_for_research"] for item in reviewed)
    return {
        "schema": "research_review_completion_audit_v1",
        "source_path": str(path.resolve()),
        "source_sha256": sha256(path),
        "entries": reviewed,
        "complete_count": complete,
        "incomplete_count": len(reviewed) - complete,
        "claim_boundary": "Transport status alone is not a usable research answer; content and finish_reason=stop are required. Completion does not certify scientific correctness.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        parser.exit(1, "output already exists\n")
    report = audit(args.input_json)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"{report['complete_count']} complete, {report['incomplete_count']} incomplete")


if __name__ == "__main__":
    main()
