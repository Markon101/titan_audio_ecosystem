#!/usr/bin/env python3
"""Calibrate and replay the opt-in v10 regime capture without affecting Titan.

Calibration uses only a specified observer-only training interval. Analysis
uses causal online normalization and a separate, persistent regime archive.
No development/validation probe field is admitted as input.
"""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def read_capture(paths):
    last = -1
    for path in paths:
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt") as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("schema") != 1 or not row.get("views"):
                    raise ValueError("unsupported or empty regime capture")
                if any("validation" in key or "development" in key for key in row):
                    raise ValueError("probe-derived fields are forbidden in regime capture")
                if any("validation" in key or "development" in key for key in row["views"]):
                    raise ValueError("probe-derived views are forbidden in regime capture")
                if int(row["global_step"]) <= last:
                    raise ValueError("captures must have strictly increasing global steps")
                last = int(row["global_step"])
                for key, value in row["views"].items():
                    if not value or not np.isfinite(value).all():
                        raise ValueError(f"non-finite or empty view: {key}")
                yield row


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_stats(rows):
    """Fit reference scales from calibration rows, floor tiny dimensions."""
    names = sorted(rows[0]["views"])
    result = {}
    for name in names:
        values = np.asarray([r["views"][name] for r in rows], dtype=np.float64)
        if values.ndim != 2:
            raise ValueError(f"ragged view: {name}")
        sd = values.std(axis=0, ddof=1)
        positive = sd[sd > 1e-8]
        floor = max(1e-5, float(np.median(positive) * 0.1) if len(positive) else 1e-5)
        result[name] = {"mean": values.mean(axis=0).tolist(),
                        "scale": np.maximum(sd, floor).tolist(), "floor": floor}
    return result


def view_weights(names):
    """Give field, GRU, Morphic, decoder, and audio equal total weight."""
    def domain(name):
        if name.startswith("field_"):
            return "field"
        if name == "gru":
            return "gru"
        if name.startswith("morphic_"):
            return "morphic"
        if name == "decoder_control":
            return "decoder"
        if name == "audio_behavior":
            return "audio"
        raise ValueError(f"unregistered regime descriptor view: {name}")
    counts = {}
    for name in names:
        group = domain(name)
        counts[group] = counts.get(group, 0) + 1
    return {name: 1.0 / (len(counts) * counts[domain(name)]) for name in names}


def vector(row, stats):
    pieces = []
    weights = view_weights(sorted(stats))
    for name in sorted(stats):
        if name not in row["views"]:
            raise ValueError(f"missing calibrated view: {name}")
        values = np.asarray(row["views"][name], dtype=np.float64)
        mean = np.asarray(stats[name]["mean"])
        scale = np.asarray(stats[name]["scale"])
        if values.shape != mean.shape:
            raise ValueError(f"view shape changed: {name}")
        z = np.clip((values - mean) / scale, -6.0, 6.0)
        pieces.append(z * math.sqrt(weights[name] / len(z)))
    return np.concatenate(pieces)


def raw_vector(row, stats):
    pieces = []
    for name in sorted(stats):
        if name not in row["views"] or len(row["views"][name]) != len(stats[name]["mean"]):
            raise ValueError(f"missing or changed calibrated view: {name}")
        pieces.extend(row["views"][name])
    return np.asarray(pieces, dtype=np.float64)


def initial_online_stats(cal):
    names = sorted(cal["normalization"])
    weights = view_weights(names)
    mean = [v for name in names for v in cal["normalization"][name]["mean"]]
    scales = [v for name in names for v in cal["normalization"][name]["scale"]]
    floors = [cal["normalization"][name]["floor"] for name in names
              for _ in cal["normalization"][name]["mean"]]
    count = max(32, cal["source_sample_count"])
    return {"count": count, "mean": mean,
            "m2": [(count - 1) * v * v for v in scales],
            "floor": floors, "sizes": [len(cal["normalization"][name]["mean"]) for name in names],
            "view_weights": [weights[name] for name in names]}


def online_vector(raw, stats):
    mean = np.asarray(stats["mean"])
    sd = np.sqrt(np.asarray(stats["m2"]) / max(1, stats["count"] - 1))
    scale = np.maximum(sd, np.asarray(stats["floor"]))
    z = np.clip((raw - mean) / scale, -6.0, 6.0)
    offset = 0
    for size, weight in zip(stats["sizes"], stats["view_weights"]):
        z[offset:offset + size] *= math.sqrt(weight / size)
        offset += size
    return z


def update_online(stats, raw):
    count = stats["count"] + 1
    mean = np.asarray(stats["mean"])
    delta = raw - mean
    mean += delta / count
    m2 = np.asarray(stats["m2"]) + delta * (raw - mean)
    stats.update(count=count, mean=mean.tolist(), m2=m2.tolist())


def distance(a, b):
    return float(np.linalg.norm(a - b))


def calibrate(rows):
    if len(rows) < 32:
        raise ValueError("calibration needs at least 32 observer samples")
    reference = rows[:max(32, len(rows) // 2)]
    stats = normalize_stats(reference)
    vec = [vector(r, stats) for r in reference]
    # Local recurrent motion sets a residence scale. Time-shuffled pairs are
    # an explicit separability null; they must not cap the threshold, because
    # a recurrent trajectory legitimately contains nonlocal close returns.
    local = [distance(vec[i], vec[i - lag]) for i in range(4, len(vec)) for lag in (1, 2, 4)]
    rng = np.random.default_rng(0xA17AC70)
    perm = rng.permutation(len(vec))
    shuffled = [distance(vec[i], vec[int(perm[i])]) for i in range(len(vec))]
    med = float(np.median(local))
    mad = float(np.median(np.abs(np.asarray(local) - med)))
    threshold = float(np.quantile(local, 0.9))
    if threshold <= 0 or not np.isfinite(threshold):
        raise ValueError("calibrated residence threshold is degenerate")
    cal = {"schema": 1, "method": "local_lags_1_2_4_p90_with_shuffled_overlap_diagnostic",
            "threshold": threshold, "local_median": med, "local_mad": mad,
            "quiet_novelty_threshold": float(np.quantile(local, 0.1)),
            "shuffled_p10": float(np.quantile(shuffled, 0.1)),
            "shuffled_overlap_fraction": float(np.mean(np.asarray(shuffled) <= threshold)),
            "min_persistence_samples": 4, "source_steps": [reference[0]["global_step"],
                                                       reference[-1]["global_step"]],
            "source_sample_count": len(reference),
            "normalization": stats,
            "null_seed": 0xA17AC70}
    # Run the descriptive archive on the calibration half to set its alert
    # threshold. It cannot influence the Titan trajectory or select regimes.
    cal["trap_threshold"] = 2.0
    cal["dwell_threshold_samples"] = 64
    cal["trap_component_thresholds"] = {}
    cal["trap_confirmation_samples"] = 16
    probe = Archive(cal)
    replay = [probe.process(row) for row in reference]
    scores = [item["trap_score"] for item in replay]
    cal["trap_threshold"] = float(np.quantile(scores, 0.95))
    dwells = [item["regime_dwell_samples"] for item in replay if item["regime_id"] is not None]
    if dwells:
        cal["dwell_threshold_samples"] = max(16, int(np.quantile(dwells, 0.9)))
    dynamic = ("low_novelty", "dimension_contraction", "low_topology_change",
               "controller_repetition", "recurrence")
    cal["trap_component_thresholds"] = {
        key: float(np.quantile([item["trap_components"][key] for item in replay], 0.75))
        for key in dynamic}
    cal["trap_confirmation_samples"] = max(8, cal["dwell_threshold_samples"] // 4)
    return cal


def participation_ratio(points):
    if len(points) < 4:
        return None
    x = np.asarray(points, dtype=np.float64)
    x -= x.mean(axis=0, keepdims=True)
    gram = x @ x.T / max(1, len(x) - 1)
    trace = np.trace(gram)
    square = np.sum(gram * gram)
    return float(trace * trace / square) if square > 1e-12 else 0.0


class Archive:
    def __init__(self, calibration, state=None):
        self.cal = calibration
        self.threshold = calibration["threshold"]
        calibration_hash = hashlib.sha256(json.dumps(calibration, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest()
        self.state = state or {
            "schema": 1, "last_step": -1, "next_id": 1, "regimes": [],
            "calibration_sha256": calibration_hash,
            "normalizer": initial_online_stats(calibration),
            "current_id": None, "dwell": 0, "pending": [], "pending_target": None, "recent": [],
            "history": [], "transitions": {}, "transition_count": 0,
            "revisit_count": 0, "trap_streak": 0, "trap_active": False,
            "trap_release_streak": 0,
            "prev_motif_candidates": 0, "prev_motif_similarity": 0,
            "prev_motif_quality": 0, "motif_deltas": [], "control_recent": [],
            "recent_transition": None,
        }
        if self.state["schema"] != 1:
            raise ValueError("unsupported regime archive state")
        if self.state["calibration_sha256"] != calibration_hash:
            raise ValueError("archive resume calibration fingerprint changed")

    def process(self, row):
        s = self.state
        step = int(row["global_step"])
        if step <= s["last_step"]:
            raise ValueError("archive resume would duplicate or reverse time")
        raw = raw_vector(row, self.cal["normalization"])
        normalizer = s["normalizer"]
        x = online_vector(raw, normalizer)
        regimes = s["regimes"]
        ranked = sorted(((distance(x, online_vector(np.asarray(r["center_raw"]), normalizer)), r) for r in regimes),
                        key=lambda pair: pair[0])
        nearest = ranked[0][0] if ranked else None
        nearest_match = ranked[0][1] if ranked and nearest <= self.threshold else None
        old = s["current_id"]
        candidate_created = False
        unresolved = False
        if nearest_match is not None and nearest_match["id"] == old:
            match = nearest_match
            s["pending"].clear()
            s["pending_target"] = None
        else:
            target = nearest_match["id"] if nearest_match is not None else "new"
            pending = s["pending"]
            if s["pending_target"] != target or (target == "new" and pending and
                distance(x, online_vector(np.mean(np.asarray(pending), axis=0), normalizer)) > self.threshold):
                pending.clear()
            s["pending_target"] = target
            pending.append(raw.tolist())
            if len(pending) >= self.cal["min_persistence_samples"]:
                if nearest_match is None:
                    center = np.mean(np.asarray(pending), axis=0).tolist()
                    match = {"id": s["next_id"], "center_raw": center, "visits": 0,
                             "entries": 0, "first_step": step, "last_step": step,
                             "healthy_visits": 0, "persistent": False,
                             "persistent_step": None}
                    s["next_id"] += 1
                    regimes.append(match)
                    candidate_created = True
                else:
                    match = nearest_match
                pending.clear()
                s["pending_target"] = None
            else:
                match = next((r for r in regimes if r["id"] == old), None)
                unresolved = True
        current = match["id"] if match else None
        if current != old:
            if current is not None and old is not None:
                edge = f"{old}->{current}"
                s["transitions"][edge] = s["transitions"].get(edge, 0) + 1
                s["transition_count"] += 1
            if current is not None:
                if match["entries"]:
                    s["revisit_count"] += 1
                match["entries"] += 1
                s["recent_transition"] = {"old": old, "new": current, "sample_step": step}
            s["dwell"] = 0
        s["current_id"] = current
        s["dwell"] = s["dwell"] + 1 if current is not None else 0
        if match is not None and not unresolved:
            match["visits"] += 1
            match["last_step"] = step
            match["healthy_visits"] += int(row["health"] >= 0.55)
            # A slow centroid update preserves historical identity while
            # permitting deep development inside a known regime.
            rate = 1.0 / min(match["visits"], 128)
            center = np.asarray(match["center_raw"])
            match["center_raw"] = (center + rate * (raw - center)).tolist()
        persistent_new = False
        if match is not None and not match["persistent"] and (
            s["dwell"] >= self.cal["dwell_threshold_samples"] and
            match["healthy_visits"] / max(1, match["visits"]) >= 0.75):
            match["persistent"] = True
            match["persistent_step"] = step
            persistent_new = True
        s["recent"].append(raw.tolist())
        s["recent"] = s["recent"][-128:]
        recent = [online_vector(np.asarray(v), normalizer) for v in s["recent"]]
        short = recent[-16:]
        long = recent[-64:]
        novelty_short = float(np.mean([distance(x, np.asarray(v)) for v in short[:-1]])) if len(short) > 1 else None
        novelty_long = float(np.mean([distance(x, np.asarray(v)) for v in long[:-1]])) if len(long) > 1 else None
        pr = participation_ratio(long)
        earlier = participation_ratio(recent[-128:-64]) if len(recent) >= 96 else None
        expansion = (pr / earlier - 1.0) if pr is not None and earlier and earlier > 0 else None
        recurrence = (sum(distance(x, np.asarray(v)) <= self.threshold for v in recent[:-8]) /
                      max(1, len(recent[:-8]))) if len(recent) > 8 else None
        mc = int(row["motif_candidates"])
        mr = int(row["motif_rejected_similarity"])
        mq = int(row["motif_rejected_quality"])
        dc = max(0, mc - s["prev_motif_candidates"])
        dr = max(0, mr - s["prev_motif_similarity"])
        dq = max(0, mq - s["prev_motif_quality"])
        if dc:
            s["motif_deltas"].append([dc, dr])
            s["motif_deltas"] = s["motif_deltas"][-64:]
        total_candidates = sum(item[0] for item in s["motif_deltas"])
        rejection_fraction = (sum(item[1] for item in s["motif_deltas"]) / total_candidates
                              if total_candidates else None)
        rejection_reason = "similarity" if dr else "quality" if dq else "stored_or_unassessed" if dc else None
        s["prev_motif_candidates"], s["prev_motif_similarity"] = mc, mr
        s["prev_motif_quality"] = mq
        s["control_recent"].append(row["views"]["decoder_control"][-9:])
        s["control_recent"] = s["control_recent"][-64:]
        control_change = (float(np.mean(np.std(np.asarray(s["control_recent"][-16:]), axis=0)))
                          if len(s["control_recent"]) >= 16 else None)
        s["history"].append({"step": step, "regime_id": current,
                             "novelty": nearest, "pr": pr, "health": row["health"],
                             "entropy": row["views"]["audio_behavior"][0],
                             "field_entropy": row["views"]["audio_behavior"][9]})
        s["history"] = s["history"][-256:]
        h = s["history"]
        entropy_change = (h[-1]["entropy"] - h[-17]["entropy"]) / 16 if len(h) >= 17 else None
        field_entropy_change = (h[-1]["field_entropy"] - h[-17]["field_entropy"]) / 16 if len(h) >= 17 else None
        offset = 0
        field_positions = []
        for name, size in zip(sorted(self.cal["normalization"]), normalizer["sizes"]):
            if name.startswith("field_"):
                field_positions.extend(range(offset, offset + size))
            offset += size
        topology_change = distance(x[field_positions], np.asarray(recent[-17])[field_positions]) if len(recent) >= 17 else None
        transition = s["recent_transition"]
        persistence = None
        if transition is not None and step > transition["sample_step"]:
            age = (step - transition["sample_step"]) / max(1, row["sample_stride"])
            if age <= self.cal["dwell_threshold_samples"]:
                old_regime = next((r for r in regimes if r["id"] == transition["old"]), None)
                if old_regime is not None:
                    separation = distance(x, online_vector(np.asarray(old_regime["center_raw"]), normalizer))
                    persistence = float(current == transition["new"]) * min(
                        1.0, separation / self.threshold) * min(
                        1.0, age / self.cal["dwell_threshold_samples"])
            else:
                s["recent_transition"] = None
        components = {
            "dwell": min(1.0, s["dwell"] / 64),
            "low_novelty": max(0.0, 1.0 - (nearest or self.threshold) / self.threshold),
            "motif_rejection": rejection_fraction or 0.0,
            "dimension_contraction": max(0.0, min(1.0, -expansion)) if expansion is not None else 0.0,
            "low_topology_change": max(0.0, 1.0 - topology_change / self.threshold) if topology_change is not None else 0.0,
            "confidence": row["model_confidence"],
            "controller_repetition": max(0.0, 1.0 - control_change / 0.1) if control_change is not None else 0.0,
            "recurrence": recurrence or 0.0,
            "few_transitions": float(len({v["regime_id"] for v in h[-64:] if v["regime_id"] is not None}) <= 1),
        }
        trap_score = sum(components.values()) / len(components)
        dynamic_thresholds = self.cal.get("trap_component_thresholds", {})
        evidence_count = sum(components[key] > threshold for key, threshold in dynamic_thresholds.items())
        # A healthy long dwell is not a trap by itself. Require independent
        # change/novelty/recurrence evidence over a sustained confirmation window.
        over_resident = (len(h) >= 64 and s["dwell"] >= self.cal["dwell_threshold_samples"] and
                         row["health"] >= 0.55 and trap_score >= self.cal["trap_threshold"] and
                         evidence_count >= 3)
        s["trap_streak"] = s["trap_streak"] + 1 if over_resident else 0
        if s["trap_streak"] >= self.cal.get("trap_confirmation_samples", 16):
            s["trap_active"] = True
        release = not over_resident and (trap_score < 0.75 * self.cal["trap_threshold"] or evidence_count <= 1)
        s["trap_release_streak"] = s["trap_release_streak"] + 1 if release else 0
        if s["trap_release_streak"] >= max(4, self.cal.get("trap_confirmation_samples", 16) // 2):
            s["trap_active"] = False
        unstable = row["health"] < 0.4 or row["stagnation"] > 0.85
        quiet = (not unstable and not s["trap_active"] and current is not None and
                 novelty_short is not None and novelty_short < self.cal["quiet_novelty_threshold"])
        state = "unstable" if unstable else "over_resident" if s["trap_active"] else \
            "temporarily_quiet" if quiet else "healthy_stable" if current is not None else "transient"
        s["last_step"] = step
        update_online(normalizer, raw)
        return {
            "global_step": step, "regime_id": current, "regime_dwell_samples": s["dwell"],
            "regime_dwell_chunks": s["dwell"] * row["sample_stride"],
            "nearest_archive_distance": nearest, "novelty_short": novelty_short,
            "novelty_long": novelty_long, "archive_size": len(regimes),
            "regime_visitation_count": match["visits"] if match else 0,
            "regime_persistent": bool(match["persistent"]) if match else False,
            "revisit_count": s["revisit_count"], "transition_count": s["transition_count"],
            "trajectory_participation_ratio": pr, "manifold_expansion_rate": expansion,
            "recurrence_rate": recurrence, "audio_entropy_change_rate": entropy_change,
            "field_entropy_change_rate": field_entropy_change,
            "field_topology_change": topology_change,
            "motif_nearest_distance": row["motif_nearest_distance"],
            "motif_similarity_rejection_fraction": rejection_fraction,
            "motif_rejection_reason": rejection_reason,
            "trap_score": trap_score, "trap_components": components,
            "trap_evidence_count": evidence_count,
            "trap_confirmation_streak": s["trap_streak"],
            "residence_state": state, "candidate_regime_created": candidate_created,
            "persistent_new_regime": persistent_new,
            "intervention_trigger": False, "candidate_interventions": [],
            "candidate_scores": [], "chosen_intervention": None, "no_op_score": None,
            "persistence_score_after_transition": persistence, "cooldown_remaining": 0,
            "health": row["health"], "stagnation": row["stagnation"],
            "per_scale_msfield_summaries": {
                k: v[-4:] for k, v in row["views"].items() if k.startswith("field_")},
            "active_morph_depth": row["active_morph_depth"],
            "optimizer_update_count": row["optimizer_updates"],
        }


def write_atomic_json(path, value):
    target = Path(path)
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    temp.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pcal = sub.add_parser("calibrate")
    pcal.add_argument("captures", nargs="+", type=Path)
    pcal.add_argument("--out", required=True, type=Path)
    pan = sub.add_parser("analyze")
    pan.add_argument("captures", nargs="+", type=Path)
    pan.add_argument("--calibration", required=True, type=Path)
    pan.add_argument("--state-in", type=Path)
    pan.add_argument("--state-out", required=True, type=Path)
    pan.add_argument("--trace-out", required=True, type=Path)
    pan.add_argument("--graph-out", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "calibrate":
        cal = calibrate(list(read_capture(args.captures)))
        cal["captures"] = [{"path": str(path), "sha256": sha256(path)} for path in args.captures]
        write_atomic_json(args.out, cal)
        print(json.dumps({k: cal[k] for k in ("threshold", "local_median", "shuffled_p10", "source_steps")}))
        return
    cal = json.loads(args.calibration.read_text())
    state = json.loads(args.state_in.read_text()) if args.state_in else None
    archive = Archive(cal, state)
    if args.trace_out.exists() and not args.state_in:
        parser.error("trace already exists; use a fresh output or state-in")
    with args.trace_out.open("a" if args.state_in else "x") as stream:
        for row in read_capture(args.captures):
            result = archive.process(row)
            stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
    write_atomic_json(args.state_out, archive.state)
    graph = {"schema": 1, "regimes": [{k: r[k] for k in ("id", "visits", "entries", "first_step", "last_step", "healthy_visits", "persistent", "persistent_step")}
                                      for r in archive.state["regimes"]],
             "transitions": archive.state["transitions"]}
    write_atomic_json(args.graph_out, graph)
    dot = ["digraph titan_regimes {", "  rankdir=LR;", "  node [shape=circle];"]
    for regime in graph["regimes"]:
        dot.append(f'  r{regime["id"]} [label="R{regime["id"]}\\nvisits {regime["visits"]}"];')
    for edge, count in graph["transitions"].items():
        source, target = edge.split("->")
        dot.append(f'  r{source} -> r{target} [label="{count}"];')
    dot.append("}")
    args.graph_out.with_suffix(".dot").write_text("\n".join(dot) + "\n")
    print(json.dumps({"regimes": len(graph["regimes"]), "transitions": sum(graph["transitions"].values()),
                      "last_step": archive.state["last_step"]}))


if __name__ == "__main__":
    main()
