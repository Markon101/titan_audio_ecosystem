#!/usr/bin/env python3
"""Run a preregistered sequential, restart-safe frozen weight/world matrix."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TAG = "v10-msfield-fresh-20260930-02"
RUNS = HERE / "runs"
P = ROOT / "analysis/metastable_20261001/frozen_step_58107"
C = RUNS / "frozen_60919"
E = RUNS / "frozen_703"
CORPUS = ROOT / "analysis/matched_exposure_20261001/runs/corpus"
CELLS = [("L16_WP_SP", P, P, 16), ("L16_WC_SP", C, P, 16),
         ("L16_WP_SC", P, C, 16), ("L16_WC_SC", C, C, 16),
         ("L16_WE_SP", E, P, 16), ("L16_WE_SC", E, C, 16),
         ("L1_WE_SP", E, P, 1), ("L1_WP_SP", P, P, 1)]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def header(path):
    with path.open("rb") as source:
        size = struct.unpack("<Q", source.read(8))[0]
        value = json.loads(source.read(size))
    return {key: (item["dtype"], item["shape"]) for key, item in value.items() if key != "__metadata__"}


def artifact(root, stem, ext):
    return root / f"titan_{stem}_v10_msfield_{TAG}.{ext}"


def mem():
    value = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, rest = line.partition(":")
        if key in ("MemAvailable", "SwapFree"):
            value[key] = int(rest.split()[0]) / 1024
    return value


def verify_sources():
    receipts = [ROOT / "analysis/metastable_20261001/frozen_step_58107_receipt.json",
                RUNS / "frozen_703_receipt.json", RUNS / "frozen_60919_receipt.json"]
    inventories = {}
    for root, receipt in zip((P, E, C), receipts):
        data = json.loads(receipt.read_text())
        for item in data["files"]:
            if sha(root / item["name"]) != item["sha256"]:
                raise RuntimeError(f"source checkpoint changed: {root / item['name']}")
        inventories[str(data["global_step"])] = {"root": str(root), "receipt_sha256": sha(receipt),
            "model_sha256": sha(artifact(root, "model", "safetensors")),
            "world_sha256": sha(artifact(root, "world", "bin"))}
    if not (header(artifact(P,"model","safetensors")) == header(artifact(C,"model","safetensors")) == header(artifact(E,"model","safetensors"))):
        raise RuntimeError("checkpoint tensor names/shapes/dtypes differ; no transplantation allowed")
    return inventories


def prepare():
    inventory = verify_sources()
    corpus_receipt=json.loads((ROOT/'analysis/matched_exposure_20261001/matched_exposure_receipt.json').read_text())
    fresh_receipt=json.loads((ROOT/'analysis/msfield_fresh_20260930/run_receipt.json').read_text())
    manifest_path=Path(fresh_receipt['files']['manifest']['path'])
    if not (sha(manifest_path)==fresh_receipt['files']['manifest']['sha256']==corpus_receipt['source_manifest_sha256']):
        raise RuntimeError('historical source manifest identity changed; reassess holdout provenance')
    for name,expected in corpus_receipt['heldout_sha256'].items():
        if sha(CORPUS/'a'/name)!=expected:raise RuntimeError(f'heldout audio changed: {name}')
    path = HERE / "schedule.json"
    if not path.exists():
        plan = json.loads((ROOT / "analysis/matched_exposure_20261001/matched_exposure_receipt.json").read_text())
        previous = json.loads((ROOT / "analysis/matched_exposure_20261001/schedule_20261002.json").read_text())
        known = {entry["slot"]: entry["source_frame"] for entry in previous["episodes"]}
        episodes = []
        for index, slot in enumerate((3, 0, 2, 1, 4, 5)):
            source = plan["arms"]["a"]["slots"][slot]
            frame = known.get(slot, 4096 * 32)
            if frame + 64 * 4096 > source["frames"]:
                raise RuntimeError("scheduled frame range exceeds source")
            episodes.append({"start_step": index * 64, "chunks": 64, "slot": slot, "source_frame": frame})
        path.write_text(json.dumps({"schema":1,"seed":20261002,"slots":[f"slot_{i:02}.wav" for i in range(6)],"episodes":episodes}, indent=2)+"\n")
    receipt = {"schema": 1, "sources": inventory, "schedule_sha256": sha(path),
               "historical_source_manifest_sha256":sha(manifest_path),"heldout_audio_hashes_verified":len(corpus_receipt['heldout_sha256']),
               "cells": [cell[0] for cell in CELLS], "chunks":384,"stride":8,
               "analysis_seed":20261002,"optimizer_steps":0}
    (HERE / "preflight_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    return path


def command(name, weights, world, depth, chunks, output, metrics=True, common=True, binary=None):
    cmd = [str(binary or ROOT / "target/release/titan"), "--analysis-only", "--analysis-substrate", "msfield",
           "--base-dir", str(world), "--model", str(artifact(weights,"model","safetensors")),
           "--state", str(artifact(world,"world","bin")), "--run-tag",TAG,
           "--corpus-dir",str(CORPUS / "a"),"--corpus-manifest",str(CORPUS / "a_manifest.json"),
           "--analysis-target-schedule",str(HERE / "schedule.json"),
           "--analysis-seed","20261002","--dynamics-ablation",str(chunks),"--ablation","none",
           "--analysis-stride","8","--analysis-dir",str(output),"--analysis-terminal","quiet","--threads","2"]
    if common: cmd += ["--analysis-common-rng","--analysis-target-origin","0","--analysis-active-depth",str(depth)]
    if metrics: cmd += ["--analysis-evaluation-metrics"]
    return cmd


def checked_complete(output, chunks, expected_command=None):
    report = json.loads((output / "analysis_report.json").read_text())
    suite = json.loads((output / "ablations/summary.json").read_text())
    full = json.loads((output / "ablations/full/summary.json").read_text())
    if expected_command is not None:
        provenance=json.loads((output/'provenance.json').read_text())
        if provenance['analysis_configuration']['invocation'][1:]!=expected_command[1:]:
            raise RuntimeError('completed output invocation does not match requested cell')
    if (output / "INCOMPLETE").exists() or not (report["non_mutation"]["unchanged"] and suite["no_op_clone_exact"] and
          report["analysis"]["optimizer_steps"] == 0 and full["horizon_chunks"] == chunks):
        raise RuntimeError("frozen run did not pass required gates")
    if full.get("weights_hash_before") != full.get("weights_hash_after"):
        raise RuntimeError("parameters mutated")
    return {"report_sha256":sha(output / "analysis_report.json"),"summary_sha256":sha(output / "ablations/full/summary.json"),
            "audio_sha256":sha(output / "ablations/full/post_dsp.wav"), "no_op_exact":True}


def run(cmd, name, chunks):
    output = Path(cmd[cmd.index("--analysis-dir") + 1])
    if (output / "analysis_report.json").exists():
        value=checked_complete(output, chunks,cmd)
        resource_path=RUNS/f'{name}_resource.json'
        if resource_path.exists():value['resource']=json.loads(resource_path.read_text())
        value['command']=json.loads((output/'provenance.json').read_text())['analysis_configuration']['invocation']
        return value
    if output.exists():
        raise RuntimeError(f"preserving partial output {output}; inspect before continuing")
    initial = mem()
    executable_hash=sha(Path(cmd[0]))
    if initial["MemAvailable"] < 1500:
        raise RuntimeError(f"resource gate: only {initial['MemAvailable']:.0f} MiB available")
    log = RUNS / f"{name}.log"
    if log.exists(): raise RuntimeError(f"preserving existing log {log}")
    start = time.monotonic(); peak = 0; minimum = initial["MemAvailable"]; stopped = False
    with log.open("xb") as stream:
        process = subprocess.Popen(cmd, stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        while process.poll() is None:
            current = mem(); minimum = min(minimum,current["MemAvailable"])
            try:
                lines = Path(f"/proc/{process.pid}/status").read_text().splitlines()
                peak = max(peak,next((int(line.split()[1])/1024 for line in lines if line.startswith("VmRSS:")),0))
            except FileNotFoundError: pass
            if current["MemAvailable"] < 768 and not stopped:
                stopped = True; os.killpg(process.pid,signal.SIGTERM)
            time.sleep(0.5)
    resource = {"initial":initial,"binary_sha256":executable_hash,"minimum_available_mib":minimum,"peak_rss_mib":peak,
                "elapsed_seconds":time.monotonic()-start,"exit_code":process.returncode,"safety_stop":stopped}
    (RUNS / f"{name}_resource.json").write_text(json.dumps(resource,indent=2)+"\n")
    if process.returncode or stopped: raise RuntimeError(f"{name} incomplete; see {log}")
    return {**checked_complete(output,chunks,cmd),"resource":resource,"command":cmd}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--cell",choices=[cell[0] for cell in CELLS])
    parser.add_argument("--binary",type=Path,default=ROOT/'target/release/titan')
    args=parser.parse_args();prepare();RUNS.mkdir(exist_ok=True)
    if args.prepare_only: print("exact tensor-schema and source hashes verified; schedule prepared");return
    cells = CELLS[:1] if args.smoke else ([cell for cell in CELLS if cell[0]==args.cell] if args.cell else CELLS)
    receipt_path=HERE / ("smoke_receipt.json" if args.smoke else "matrix_receipt.json")
    receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else {"schema":1,"cells":{}}
    current_binary_hash=sha(args.binary)
    missing=any(not (RUNS/('smoke16' if args.smoke else cell[0])/'analysis_report.json').exists() for cell in cells)
    if receipt.get('binary_sha256') and receipt['binary_sha256']!=current_binary_hash and missing:
        raise RuntimeError('preserve matrix binary identity: use --binary runs/matrix_titan for missing cells')
    if 'binary_sha256' not in receipt and receipt['cells'] and missing:
        raise RuntimeError('existing partial campaign has no binary identity; recover provenance before adding cells')
    if 'binary_sha256' not in receipt and not receipt['cells']:receipt['binary_sha256']=current_binary_hash
    for name,weights,world,depth in cells:
        cell_name = "smoke16" if args.smoke else name
        chunks=16 if args.smoke else 384
        cmd=command(name,weights,world,depth,chunks,RUNS / cell_name,binary=args.binary)
        receipt["cells"][cell_name]=run(cmd,cell_name,chunks)
        receipt_path.write_text(json.dumps(receipt,indent=2)+"\n")
        print(json.dumps({"cell":cell_name,"complete":True}),flush=True)
    verify_sources()


if __name__=="__main__":main()
