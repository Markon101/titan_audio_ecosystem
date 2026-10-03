"""Opt-in offline generation selection. Never applies an intervention to a world."""
import argparse
import json
from pathlib import Path
import shutil
import math
import hashlib
from predictive import reward


def choose(candidates, pd_weight=0, naive=False):
    if not math.isfinite(pd_weight) or pd_weight < 0:
        raise ValueError('nonnegative experimental coefficient required')
    # Exact bypass: no metric, covariance, null, or audit data is consulted.
    if pd_weight == 0 and not naive:
        return {'chosen': 'baseline', 'mode': 'disabled', 'scores': {'baseline': 0.0}}
    if not naive and any(item['coding'].get('protocol') != 'v2_affine_periodic' for item in candidates.values()):
        raise ValueError('nonzero selection requires the hardened affine-periodic protocol')
    scores = {}
    for name, item in candidates.items():
        if naive:
            scores[name] = item['naive_selection_score']
        elif item['selection_physical']['eligible']:
            scores[name] = -item['selection_waveform_displacement'] + pd_weight * reward(item['coding'])['value']
        else:
            scores[name] = -1e12
        if not math.isfinite(scores[name]):
            raise ValueError('nonfinite candidate score')
    # A calibrated-margin replacement requires stronger data; this pilot uses
    # the prespecified conservative displacement cost and exact tie-to-no-op.
    winner = max(scores, key=lambda name: (scores[name], name == 'baseline', name))
    return {'chosen': winner, 'mode': 'naive' if naive else 'predictive_density',
            'pd_weight': pd_weight, 'scores': scores, 'audit_not_used': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--snapshot', choices=['w0', 'w128'], required=True)
    parser.add_argument('--pd-weight', type=float, default=0)
    parser.add_argument('--naive', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    values = json.loads(args.results.read_text())['snapshots'][args.snapshot]['candidates']
    decision = choose(values, args.pd_weight, args.naive)
    source = Path(values[decision['chosen']]['post_dsp_wav'])
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != values[decision['chosen']]['post_dsp_sha256']:
        raise RuntimeError('candidate audio changed after measurement')
    if args.output.exists() or args.output.with_suffix('.receipt.json').exists():
        raise FileExistsError('preserve prior selected audio')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as read, args.output.open('xb') as write:
        shutil.copyfileobj(read, write)
    if hashlib.sha256(args.output.read_bytes()).hexdigest() != digest:
        raise RuntimeError('selected copy differs')
    decision.update({'source_sha256': digest, 'output_sha256': digest,
                     'coding_protocol': values[decision['chosen']]['coding']['protocol']})
    args.output.with_suffix('.receipt.json').write_text(json.dumps(decision, indent=2)+'\n')
    print(json.dumps(decision))


if __name__ == '__main__':
    main()
