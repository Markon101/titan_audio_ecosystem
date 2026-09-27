#!/usr/bin/env python3
"""Audio-specific high-context team using the installed bounded DeepSeek helper.

prepare is offline. run makes the eight independent requests; challenge makes
one final review with the shared packet and accepted independent answers.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
HELPER = Path.home() / ".codex/skills/deepseek-flash/scripts/deepseek_flash.py"
ROLES = {
    "independent-investigator": ("researcher", "Reconstruct the causal audio pipeline and what the observations establish. Locate missing evidence and propose the cheapest separating measurement."),
    "falsification-arbiter": ("experiment-designer", "Give competing claims, why they differ, and experiments whose outcomes distinguish them. Specify an inconclusive outcome."),
    "experiment-designer": ("experiment-designer", "Design the smallest reproducible audio comparison for harshness and width with waveform controls, level matching, transients, and listening. Bound compute."),
    "dynamics-agent": ("dynamical-systems-critic", "Compare waveform heat diffusion, latent diffusion, and a learned decoder. State which experiment can test which mechanism and the preservation risks."),
    "rust-audit-agent": ("code-reviewer", "Audit source-level causes and confounds in renderer, losses, controls, postprocessing, and prime selection. Cite actual lines and prioritize minimal interventions."),
    "statistical-agent": ("researcher", "Audit the measurement design, mid-only spectra, sequential-run dependence, prime selection, level matching, and listening inference. Define defensible units and limits."),
    "ideation-agent": ("mechanistic-competitor", "Generate distinct conventional and unconventional hypotheses about useful stereo/timbre variation. Preserve one unusual but testable idea with a cheap falsifier."),
    "skeptical-agent": ("result-skeptic", "Find the strongest mundane explanations and unsupported leaps, including center collapse, harshness, output smoothing, and practical priming. Offer decisive controls."),
}
RECEIPT = {
    "project": "titan_audio_ecosystem",
    "direct_reference_input": False,
    "prompt_is_post_render_output": True,
    "waveform_filter_tests_latent_diffusion": False,
    "canonical_mutation_allowed": False,
}
OUTPUT_CONTRACT = (
    " Return a JSON object with conclusions, evidence, uncertainties, proposed_experiments, "
    "and a context_receipt object reconstructing these facts from the source: project (string), "
    "direct_reference_input (boolean), prompt_is_post_render_output (boolean), "
    "waveform_filter_tests_latent_diffusion (boolean), canonical_mutation_allowed (boolean). "
    "The receipt is mandatory. Cite original source paths and line numbers."
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


class ContextBuilder:
    """Explicit selections, complete excerpts, original line numbers; no crawling."""
    def __init__(self, root=ROOT):
        self.root = root.resolve()

    def selection(self, selection):
        name, separator, span = selection.rpartition(":")
        if not separator or "-" not in span:
            name, span = selection, None
        path = (self.root / name).resolve(strict=True)
        relative = path.relative_to(self.root)
        if any(p in {".git", ".agents", ".codex"} or p.startswith(".env") for p in relative.parts):
            raise ValueError("protected path in context selections")
        lines = path.read_text(encoding="utf-8").splitlines()
        start, end = (map(int, span.split("-")) if span else (1, len(lines)))
        if not 1 <= start <= end <= len(lines):
            raise ValueError(f"invalid context span: {selection}")
        raw = "\n".join(lines[start - 1:end]) + "\n"
        numbered = "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, end + 1))
        return f"\n## EVIDENCE {relative}:{start}-{end}\nSHA256 {digest(raw.encode())}\n\n{numbered}\n", {
            "selection": selection, "sha256": digest(raw.encode()), "bytes": len(raw.encode()),
        }

    def build(self, config):
        sections, identities = [], []
        for selection in [config["core"], *config["evidence"]]:
            text, identity = self.selection(selection)
            sections.append(text)
            identities.append(identity)
        packet = "# Shared Audio evidence snapshot\n" + "".join(sections)
        if len(packet.encode()) > config["max_request_bytes"] - 8192:
            raise ValueError("selected context exceeds budget; revise explicit selections, no truncation")
        return packet, identities


def command(packet, config, role, task, dry=False):
    result = [sys.executable, str(HELPER), "run", "--root", str(ROOT),
              "--context-file", str(packet), "--role", role, "--task", task + OUTPUT_CONTRACT,
              "--model", config["model"], "--max-tokens", str(config["max_tokens"]),
              "--max-request-bytes", str(config["max_request_bytes"]),
              "--timeout", "180", "--retries", "1", "--json-answer"]
    if dry:
        result.append("--dry-run")
    return result


def invoke(arguments):
    completed = subprocess.run(arguments, cwd=ROOT, capture_output=True, text=True, timeout=240)
    try:
        result = json.loads(completed.stdout)
        if not isinstance(result, dict):
            raise ValueError("not an envelope")
        return result
    except (ValueError, json.JSONDecodeError):
        return {"status": "launcher_error", "error": "helper did not return a JSON envelope", "exit_code": completed.returncode}


def audit(envelope):
    answer = envelope.get("answer_json")
    reasons = []
    if envelope.get("status") != "ok" or envelope.get("finish_reason") != "stop":
        reasons.append("transport or final-answer completion failed")
    if envelope.get("requested_model") != envelope.get("returned_model"):
        reasons.append("returned model differs from requested model")
    if not isinstance(answer, dict) or not (answer.get("summary") or answer.get("conclusions")):
        reasons.append("nonempty structured summary missing")
    receipt = answer.get("context_receipt", {}) if isinstance(answer, dict) else {}
    for key, expected in RECEIPT.items():
        actual = receipt.get(key)
        if actual != expected or type(actual) is not type(expected):
            reasons.append(f"incorrect context receipt: {key}")
    return {"eligible_for_synthesis": not reasons, "reasons": reasons,
            "scope": "completion and factual receipt checks only; scientific claims require independent verification"}


def accepted_or_original(directory, name):
    original = json.loads((directory / "independent" / f"{name}.json").read_text())
    retry = directory / "independent" / f"{name}.retry.json"
    if retry.is_file():
        corrected = json.loads(retry.read_text())
        if audit(corrected)["eligible_for_synthesis"]:
            return corrected
    return original


def repair_receipts(directory):
    manifest, packet = load_packet(directory)
    for name, (role, task) in ROLES.items():
        current = accepted_or_original(directory, name)
        if audit(current)["eligible_for_synthesis"]:
            continue
        target = directory / "independent" / f"{name}.retry.json"
        if target.exists():
            raise ValueError(f"bounded retry already exists: {name}")
        # Same raw evidence, explicit task-level output contract; no other reviews.
        preflight = invoke(command(packet, manifest["config"], role, task, dry=True))
        write_json(directory / f"{name}.retry.dry-run.json", preflight)
        if preflight.get("status") != "dry_run":
            raise ValueError(f"retry preflight failed: {name}")
        result = invoke(command(packet, manifest["config"], role, task))
        write_json(target, result)
        print(f"{name} retry: {result.get('status')}, receipt {audit(result)['eligible_for_synthesis']}", flush=True)
    write_json(directory / "independent_audit.json", {
        name: audit(accepted_or_original(directory, name)) for name in ROLES})


def prepare(directory, config_path):
    config = json.loads(config_path.read_text())
    packet, identities = ContextBuilder().build(config)
    directory.mkdir(parents=True, exist_ok=False)
    packet_path = directory / "context_packet.md"
    packet_path.write_text(packet)
    manifest = {"schema_version": 1, "config": config, "selections": identities,
                "packet_sha256": digest(packet.encode()), "packet_bytes": len(packet.encode()),
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "roles": list(ROLES), "synthesis_role": "synthesis-challenger"}
    write_json(directory / "context_manifest.json", manifest)
    for name, (role, task) in ROLES.items():
        result = invoke(command(packet_path, config, role, task, dry=True))
        write_json(directory / f"{name}.dry-run.json", result)
        if result.get("status") != "dry_run":
            raise ValueError(f"context preflight failed for {name}")
    print(f"Prepared {len(ROLES)} independent roles + synthesis challenger; {len(packet.encode())} context bytes; {config['max_tokens']} output tokens per role", flush=True)


def load_packet(directory):
    manifest = json.loads((directory / "context_manifest.json").read_text())
    packet = directory / "context_packet.md"
    if digest(packet.read_bytes()) != manifest["packet_sha256"]:
        raise ValueError("context snapshot has changed")
    return manifest, packet


def run(directory):
    manifest, packet = load_packet(directory)
    results_dir = directory / "independent"
    results_dir.mkdir(exist_ok=False)
    config = manifest["config"]
    outcomes = {}
    with ThreadPoolExecutor(max_workers=min(3, config["workers"])) as executor:
        futures = {executor.submit(invoke, command(packet, config, role, task)): name
                   for name, (role, task) in ROLES.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
            except (OSError, subprocess.TimeoutExpired):
                result = {"status": "launcher_error", "error": "helper failed or exceeded wall timeout"}
            write_json(results_dir / f"{name}.json", result)
            outcomes[name] = audit(result)
            print(f"{name}: {result.get('status')}, receipt {'pass' if outcomes[name]['eligible_for_synthesis'] else 'fail'}", flush=True)
    write_json(directory / "independent_audit.json", outcomes)
    if any(not item["eligible_for_synthesis"] for item in outcomes.values()):
        raise RuntimeError("independent review is partial; inspect audit and use one bounded receipt repair")


def challenge(directory):
    manifest, packet = load_packet(directory)
    target = directory / "synthesis-challenger.json"
    if target.exists():
        raise ValueError("synthesis output already exists")
    parts, accepted = [packet.read_text()], []
    for name in ROLES:
        path = directory / "independent" / f"{name}.json"
        if not path.is_file():
            continue
        result = accepted_or_original(directory, name)
        if audit(result)["eligible_for_synthesis"]:
            parts.append(f"\n## ADVISORY REVIEW {name}; not empirical evidence\n{result['answer']}\n")
            accepted.append(name)
    if len(accepted) != len(ROLES):
        raise ValueError("full-team synthesis requires every independent role to be complete and context-valid")
    context = directory / "synthesis_context.md"
    context.write_text("\n".join(parts))
    task = ("Challenge the independent team reviews against the shared source evidence. "
            "Return the required context_receipt and summary, accepted and rejected claims, "
            "CLAIM A / CLAIM B / WHY THEY DIFFER / DISCRIMINATING EXPERIMENT for disagreements, "
            "and a bounded next action. Agent agreement is advisory, not evidence. "
            "Do not promote offline filtering to proof about latent dynamics or generative diffusion. "
            f"Include included_role_count exactly {len(accepted)} and included_roles as the exact list of role names.")
    config = manifest["config"]
    preflight = invoke(command(context, config, "adversarial-reviewer", task, dry=True))
    write_json(directory / "synthesis.dry-run.json", preflight)
    if preflight.get("status") != "dry_run":
        raise ValueError("synthesis context failed preflight; no truncation applied")
    result = invoke(command(context, config, "adversarial-reviewer", task))
    write_json(target, result)
    synthesis_audit = audit(result)
    answer = result.get("answer_json") or {}
    if answer.get("included_role_count") != len(accepted) or answer.get("included_roles") != accepted:
        synthesis_audit["eligible_for_synthesis"] = False
        synthesis_audit["reasons"].append("synthesis did not enumerate the exact included team")
    write_json(directory / "synthesis_audit.json", {**synthesis_audit, "included_roles": accepted})
    print(f"synthesis-challenger: {result.get('status')}; {len(accepted)} independent reviews included", flush=True)
    if not synthesis_audit["eligible_for_synthesis"]:
        raise RuntimeError("synthesis completion or factual team count failed; inspect audit")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "repair-receipts", "challenge"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "research/audio_team.json")
    args = parser.parse_args()
    directory = args.directory.resolve()
    directory.relative_to(ROOT)  # helper evidence must live in the reviewed repo
    if args.mode == "prepare":
        prepare(directory, args.config)
    elif args.mode == "run":
        run(directory)
    elif args.mode == "repair-receipts":
        repair_receipts(directory)
    else:
        challenge(directory)


if __name__ == "__main__":
    main()
