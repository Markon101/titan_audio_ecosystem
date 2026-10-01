#!/usr/bin/env python3
"""Bounded v10 SafeTensors/world geometry analysis with explicit nulls.

This measures matrix spectra and Adam pressure, plus optional captured
activation sketches and exported field maps. A good log-log fit is never used
as evidence of a fractal or power law.
"""

import argparse
import gzip
import hashlib
import json
import mmap
import os
from pathlib import Path
import re
import struct

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np


class SafeTensorStore:
    def __init__(self, path):
        self.path = Path(path)
        self.file = self.path.open("rb")
        self.map = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        size = struct.unpack_from("<Q", self.map)[0]
        self.header = json.loads(self.map[8:8 + size])
        self.start = 8 + size

    def array(self, name):
        item = self.header[name]
        dtype = {"F32": "<f4", "I64": "<i8"}.get(item["dtype"])
        if dtype is None:
            raise ValueError(f"unsupported tensor dtype {item['dtype']}")
        offset, end = item["data_offsets"]
        count = int(np.prod(item["shape"])) if item["shape"] else 1
        if end - offset != count * np.dtype(dtype).itemsize:
            raise ValueError(f"bad SafeTensors offsets for {name}")
        return np.ndarray(tuple(item["shape"]), dtype=dtype,
                          buffer=self.map, offset=self.start + offset)

    def close(self):
        self.map.close()
        self.file.close()


def sha256(path):
    sha = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def group(name):
    if name.startswith("model.msfield."):
        return "msfield"
    found = re.search(r"model\.morphic\.(?:l|norm)(\d+)", name)
    if found:
        return f"morphic_l{int(found.group(1)) + 1:02d}"
    if name.startswith("model.gru_memory."):
        return "gru"
    if "temporal_decoder" in name:
        return "temporal_decoder"
    if any(term in name for term in ("pitch_head", "wave_morph", "wavefolder", "stereo_width",
                                      "spatial_panner", "partial_", "fm_mod_")):
        return "synthesis_heads"
    return "host_auxiliary"


def matrix_view(a):
    if a.ndim == 2:
        return np.asarray(a, dtype=np.float64)
    if a.ndim == 4:
        return np.asarray(a, dtype=np.float64).reshape(a.shape[0], -1)
    return None


def spectral_metrics(a):
    s = np.linalg.svd(a, full_matrices=False, compute_uv=False)
    energy = s * s
    total = energy.sum()
    if total <= 1e-20:
        return {"stable_rank": 0.0, "participation_rank": 0.0, "effective_rank": 0.0,
                "spectral_entropy": 0.0, "condition_number": None,
                "top_energy": {str(k): None for k in (1, 4, 16, 64)}}
    p = energy / total
    entropy = -float(np.sum(p[p > 0] * np.log(p[p > 0])))
    rank = min(a.shape)
    positive = s[s > max(1e-10, s[0] * 1e-6)]
    return {"shape": list(a.shape), "stable_rank": float(total / energy[0]),
            "participation_rank": float(total * total / np.sum(energy * energy)),
            "effective_rank": float(np.exp(entropy)),
            "spectral_entropy": float(entropy / np.log(rank)) if rank > 1 else 0.0,
            "condition_number": float(s[0] / positive[-1]) if len(positive) == rank else None,
            "top_energy": {str(k): float(energy[:k].sum() / total) for k in (1, 4, 16, 64)}}


def adam_pressure(weight, moment, second):
    w = np.asarray(weight, dtype=np.float64).reshape(-1)
    m = np.asarray(moment, dtype=np.float64).reshape(-1)
    v = np.asarray(second, dtype=np.float64).reshape(-1)
    update = m / (np.sqrt(np.maximum(v, 0.0)) + 1e-8)
    denom = np.linalg.norm(w) * np.linalg.norm(update)
    return {"moment_rms": float(np.sqrt(np.mean(m * m))),
            "second_moment_rms": float(np.sqrt(np.mean(v * v))),
            "adam_direction_rms": float(np.sqrt(np.mean(update * update))),
            "weight_update_cosine": float(np.dot(w, update) / denom) if denom > 0 else None,
            "weight_rms": float(np.sqrt(np.mean(w * w)))}


def cka(x, y):
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)
    cross = np.linalg.norm(x.T @ y, "fro") ** 2
    denom = np.linalg.norm(x.T @ x, "fro") * np.linalg.norm(y.T @ y, "fro")
    return float(cross / denom) if denom > 1e-12 else None


def covariance_pr(x):
    centered = x - x.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular * singular
    return float(energy.sum() ** 2 / np.sum(energy * energy)) if energy.sum() > 1e-12 else 0.0


def detrend(x):
    time = np.linspace(-1.0, 1.0, len(x))
    design = np.column_stack([np.ones(len(x)), time])
    return x - design @ np.linalg.lstsq(design, x, rcond=None)[0]


def activation_geometry(path, repeats, seed):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) < 8:
        return {"valid": False, "reason": "at least eight captured samples are required"}
    names = sorted(set.intersection(*(set(row["views"]) for row in rows)))
    metrics = {}
    rng = np.random.default_rng(seed)
    arrays = {}
    for name in names:
        x = np.asarray([row["views"][name] for row in rows], dtype=np.float64)
        arrays[name] = x
        pr = covariance_pr(x)
        sd = x.std(axis=0)
        shuffled_pr = []
        gaussian_pr = []
        for _ in range(repeats):
            shuffled = np.column_stack([rng.permutation(x[:, i]) for i in range(x.shape[1])])
            gaussian = rng.normal(x.mean(axis=0), sd, size=x.shape)
            shuffled_pr.append(covariance_pr(shuffled))
            gaussian_pr.append(covariance_pr(gaussian))
        metrics[name] = {"samples": len(x), "sketch_dimensions": x.shape[1],
                         "covariance_participation": pr,
                         "first_difference_participation": covariance_pr(np.diff(x, axis=0)),
                         "detrended_participation": covariance_pr(detrend(x)),
                         "independent_channel_shuffle_null_pr": shuffled_pr,
                         "matched_gaussian_null_pr": gaussian_pr,
                         "near_constant_sketch_channels": int(np.sum(sd < max(1e-6, np.median(sd) * 0.01))),
                         "median_channel_std": float(np.median(sd))}
    overlaps = {}
    differenced = {}
    deltas = {}
    for i in (1, 4, 8, 12):
        left, right = f"morphic_l{i:02d}", f"morphic_l{i + (3 if i == 1 else 4):02d}"
        if left in names and right in names:
            overlaps[f"{left}:{right}"] = cka(arrays[left], arrays[right])
            differenced[f"{left}:{right}"] = cka(np.diff(arrays[left], axis=0),
                                                  np.diff(arrays[right], axis=0))
            dl, dr = left.replace("morphic_l", "morphic_delta_l"), right.replace("morphic_l", "morphic_delta_l")
            if dl in arrays and dr in arrays:
                deltas[f"{dl}:{dr}"] = cka(np.diff(arrays[dl], axis=0), np.diff(arrays[dr], axis=0))
    return {"valid": True, "views": metrics, "activation_cka": overlaps,
            "first_difference_activation_cka": differenced,
            "first_difference_delta_cka": deltas,
            "scope": "32-dimensional orthogonal sketches during continuing training; lower-bound proxies for full 512D activations"}


def autocorr(a, lag):
    a = a - a.mean()
    denom = np.mean(a * a)
    if denom < 1e-14:
        return None
    return float((np.mean(a[:, :-lag] * a[:, lag:]) +
                  np.mean(a[:-lag, :] * a[lag:, :])) / (2 * denom))


def box_counts(a):
    mask = a > np.quantile(a, 0.75)
    result = {}
    for size in (1, 2, 4, 8, 16):
        if size > a.shape[0]:
            continue
        blocks = mask.reshape(a.shape[0] // size, size, a.shape[1] // size, size)
        result[str(size)] = int(blocks.any(axis=(1, 3)).sum())
    return result


def radial_power(a):
    power = np.abs(np.fft.fft2(a - a.mean())) ** 2
    fy = np.fft.fftfreq(a.shape[0])[:, None]
    fx = np.fft.fftfreq(a.shape[1])[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    edges = np.linspace(0, 0.71, 9)
    return [float(power[(radius >= edges[i]) & (radius < edges[i + 1])].mean())
            if ((radius >= edges[i]) & (radius < edges[i + 1])).any() else None
            for i in range(8)]


def spatial_summary(a):
    return {"lag1_correlation": autocorr(a, 1), "lag2_correlation": autocorr(a, 2),
            "lag4_correlation": autocorr(a, 4), "box_counts_top_quartile": box_counts(a),
            "radial_power": radial_power(a), "rms": float(np.sqrt(np.mean(a * a)))}


def phase_randomize(a, rng):
    centered = a - a.mean()
    magnitude = np.abs(np.fft.fft2(centered))
    phase = np.fft.fft2(rng.normal(size=a.shape))
    return np.fft.ifft2(magnitude * phase / np.maximum(np.abs(phase), 1e-12)).real + a.mean()


def world_geometry(path, repeats, seed):
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text())
    rng = np.random.default_rng(seed)
    results = {}
    for name, shape in manifest["shape"].items():
        field = np.fromfile(root / f"{name}.f32le", dtype="<f4")
        if field.size != int(np.prod(shape)):
            raise ValueError(f"world export shape mismatch: {name}")
        field = field.reshape(shape)
        # RMS across channels retains spatial localization without signed
        # cross-channel cancellation. This is a descriptive scalar map.
        a = np.sqrt(np.mean(field.astype(np.float64) ** 2, axis=0))
        pixel_null = []
        phase_null = []
        for _ in range(repeats):
            pixel_null.append(spatial_summary(rng.permutation(a.reshape(-1)).reshape(a.shape)))
            phase_null.append(spatial_summary(phase_randomize(a, rng)))
        results[name] = {"observed": spatial_summary(a), "pixel_shuffle_null": pixel_null,
                         "phase_randomized_null": phase_null,
                         "phase_null_scope": "periodic Fourier power is preserved; non-wrapping lag correlations can differ at boundaries"}
    return {"global_step": manifest["global_step"], "scales": results,
            "scope": "one checkpoint, scalar channel-RMS maps; box counts are not fractal dimension evidence"}


def checkpoint_change(first, first_optimizer, second, second_optimizer, repeats, seed):
    if set(first.header) != set(second.header):
        raise ValueError("comparison checkpoint has a different tensor schema")
    rng = np.random.default_rng(seed)
    result = {"second_model_sha256": sha256(second.path),
              "second_optimizer_sha256": sha256(second_optimizer.path),
              "second_global_step": int(second_optimizer.array("optimizer.global_step")),
              "second_optimizer_updates": int(second_optimizer.array("optimizer.step_t")),
              "groups": {}, "tensors": {}}
    group_sums = {}
    for name in sorted(first.header):
        if name == "__metadata__":
            continue
        before = np.asarray(first.array(name), dtype=np.float64)
        after = np.asarray(second.array(name), dtype=np.float64)
        delta = after - before
        g = group(name)
        sums = group_sums.setdefault(g, {"delta_sq": 0.0, "weight_sq": 0.0,
                                          "parameters": 0, "changed": 0})
        sums["delta_sq"] += float(np.sum(delta * delta))
        sums["weight_sq"] += float(np.sum(before * before))
        sums["parameters"] += int(before.size)
        sums["changed"] += int(np.count_nonzero(delta))
        dnorm = float(np.linalg.norm(delta.reshape(-1)))
        pressure = None
        mname, vname = "optimizer.m." + name, "optimizer.v." + name
        if mname in first_optimizer.header:
            m = np.asarray(first_optimizer.array(mname), dtype=np.float64).reshape(-1)
            v = np.asarray(first_optimizer.array(vname), dtype=np.float64).reshape(-1)
            direction = -m / (np.sqrt(np.maximum(v, 0.0)) + 1e-8)
            denom = np.linalg.norm(direction) * dnorm
            pressure = float(np.dot(direction, delta.reshape(-1)) / denom) if denom > 0 else None
        record = {"group": g, "relative_l2_delta": dnorm / max(1e-12, float(np.linalg.norm(before.reshape(-1)))),
                  "adam_initial_direction_cosine": pressure}
        matrix = matrix_view(delta)
        if matrix is not None and min(matrix.shape) >= 8 and dnorm > 1e-12:
            record["delta_spectrum"] = spectral_metrics(matrix)
            record["entry_shuffle_delta_null_pr"] = [
                spectral_metrics(rng.permutation(matrix.reshape(-1)).reshape(matrix.shape))["participation_rank"]
                for _ in range(repeats)]
        result["tensors"][name] = record
    for g, sums in group_sums.items():
        result["groups"][g] = {"parameters": sums["parameters"],
                               "changed_parameters": sums["changed"],
                               "relative_l2_delta": float(np.sqrt(sums["delta_sq"] / max(sums["weight_sq"], 1e-24)))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--optimizer", required=True, type=Path)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--world-export", type=Path)
    parser.add_argument("--compare-model", type=Path)
    parser.add_argument("--compare-optimizer", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--null-repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=73091)
    args = parser.parse_args()
    if not 1 <= args.null_repeats <= 16:
        parser.error("--null-repeats must be 1..16")
    if bool(args.compare_model) != bool(args.compare_optimizer):
        parser.error("--compare-model and --compare-optimizer must be provided together")
    model, optimizer = SafeTensorStore(args.model), SafeTensorStore(args.optimizer)
    rng = np.random.default_rng(args.seed)
    report = {"schema": 1, "model_sha256": sha256(args.model),
              "optimizer_sha256": sha256(args.optimizer),
              "optimizer_global_step": int(optimizer.array("optimizer.global_step")),
              "optimizer_updates": int(optimizer.array("optimizer.step_t")),
              "null_repeats": args.null_repeats, "seed": args.seed, "matrices": {}, "groups": {},
              "claim_boundary": "No fractal, power-law, or unused-capacity claim follows from spectra alone."}
    group_summaries = {}
    bases = {}
    for name in sorted(model.header):
        if name == "__metadata__":
            continue
        weight = model.array(name)
        g = group(name)
        state = group_summaries.setdefault(g, {"parameters": 0, "matrix_count": 0,
                                               "stable_ranks": [], "participation_ranks": [],
                                               "adam_update_rms": []})
        state["parameters"] += int(weight.size)
        m_name, v_name = "optimizer.m." + name, "optimizer.v." + name
        if m_name in optimizer.header and v_name in optimizer.header:
            pressure = adam_pressure(weight, optimizer.array(m_name), optimizer.array(v_name))
            state["adam_update_rms"].append(pressure["adam_direction_rms"])
        else:
            pressure = None
        a = matrix_view(weight)
        if a is None or min(a.shape) < 8:
            continue
        observed = spectral_metrics(a)
        nulls = []
        for _ in range(args.null_repeats):
            shuffled = rng.permutation(a.reshape(-1)).reshape(a.shape)
            gaussian = rng.normal(a.mean(), a.std(), size=a.shape)
            nulls.append({"entry_shuffle": spectral_metrics(shuffled),
                          "matched_gaussian": spectral_metrics(gaussian)})
        row_sd = a.std(axis=1)
        report["matrices"][name] = {"group": g, "observed": observed, "nulls": nulls,
                                     "adam_pressure": pressure,
                                     "near_constant_rows": int(np.sum(row_sd < max(1e-8, np.median(row_sd) * 0.01)))}
        state["matrix_count"] += 1
        state["stable_ranks"].append(observed["stable_rank"])
        state["participation_ranks"].append(observed["participation_rank"])
        if re.search(r"model\.morphic\.l\d+_1\.weight", name):
            _, _, vt = np.linalg.svd(a, full_matrices=False)
            bases[g] = vt[:16].T
    for name, data in group_summaries.items():
        report["groups"][name] = {"parameters": data["parameters"],
                                  "matrix_count": data["matrix_count"],
                                  "median_stable_rank": float(np.median(data["stable_ranks"])) if data["stable_ranks"] else None,
                                  "median_participation_rank": float(np.median(data["participation_ranks"])) if data["participation_ranks"] else None,
                                  "median_adam_direction_rms": float(np.median(data["adam_update_rms"])) if data["adam_update_rms"] else None}
    report["adjacent_morphic_subspace_overlap"] = {}
    for i in range(1, 16):
        left, right = f"morphic_l{i:02d}", f"morphic_l{i + 1:02d}"
        if left in bases and right in bases:
            singular = np.linalg.svd(bases[left].T @ bases[right], compute_uv=False)
            report["adjacent_morphic_subspace_overlap"][f"{left}:{right}"] = float(np.mean(singular ** 2))
    if args.capture:
        report["activations"] = activation_geometry(args.capture, args.null_repeats, args.seed + 2)
    if args.world_export:
        report["spatial"] = world_geometry(args.world_export, args.null_repeats, args.seed + 1)
    if args.compare_model:
        future_model = SafeTensorStore(args.compare_model)
        future_optimizer = SafeTensorStore(args.compare_optimizer)
        report["checkpoint_change"] = checkpoint_change(
            model, optimizer, future_model, future_optimizer, args.null_repeats, args.seed + 3)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"matrix_count": len(report["matrices"]), "groups": len(report["groups"]),
                      "world": bool(args.world_export), "capture": bool(args.capture)}))
    # mmap remains live while ndarray views are in scope. Process exit closes it.


if __name__ == "__main__":
    main()
