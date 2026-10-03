"""Read-only matched-bank coding, internal nulls, selection and held-out audit."""
import argparse
import csv
import json
from pathlib import Path
import time
from functools import partial
import numpy as np
from audio import CHUNK, describe, read_wav, phase_surrogate
from predictive import SPLITS, diagnostics, measure as coding_measure, reward
from selection import choose
from run_frozen import HERE, ROOT, sha, parent_hashes

# Revised estimator only; original W0_RESULTS.json and v1 controls are retained.
measure = partial(coding_measure, affine_controls=True)


def curve(report, probe='output_slow', segment='audit'):
    return {h: ({'gain_bits': v['probes'][probe][segment]['gain_bits'],
                 'net_gain_bits': v['probes'][probe][segment]['net_gain_bits'],
                 'over_history_bits': v['probes'][probe][segment]['incremental_over_history_bits'],
                 'over_unpenalized_baseline_bits': v['probes'][probe][segment]['incremental_over_unpenalized_baseline_bits'],
                 'pd': v['probes'][probe][segment]['pd_bits_saved_per_coefficient_bit'],
                 'baseline': v['selected_simple_baseline']}
                if v['supported'] else {'supported': False})
            for h, v in report['horizons'].items()}


def shuffled_view(value, seed):
    """Artificial negative control: alignment shuffled separately inside each split."""
    rng = np.random.default_rng(seed)
    result = value.copy()
    boundaries = [0] + [int(len(value) * fraction) for fraction in SPLITS] + [len(value)]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        result[start:end] = value[start:end][rng.permutation(end - start)]
    return result


def internal(output, summary, y):
    capture = [json.loads(line) for line in Path(summary['regime_capture_path']).read_text().splitlines()]
    steps = {row['absolute_step']: row for row in summary['sampled_steps']}
    capture = [row for row in capture if steps[row['global_step']]['rollout_offset'] % 8 == 0]
    indices = [steps[row['global_step']]['rollout_offset'] - 1 for row in capture]
    if len(indices) != len(y) // 8 or any(b-a != 8 for a,b in zip(indices[:-1], indices[1:])):
        raise RuntimeError('trajectory grid/audio alignment differs')
    views = {name: np.asarray([row['views'][name] for row in capture]) for name in capture[0]['views']
             if name != 'audio_behavior'}
    host_keys = ('energy', 'temperature', 'radiation_probability', 'radiation_realized',
                 'shear_amplitude', 'kick_amplitude', 'model_confidence_raw', 'target_error_feedback')
    views['host'] = np.asarray([[float(steps[row['global_step']].get(key) or 0) for key in host_keys]
                                for row in capture])
    motif_keys = ('motif_nearest_distance', 'motif_candidates', 'motif_rejected_similarity',
                  'motif_rejected_quality')
    views['motif'] = np.asarray([[float(row.get(key) or 0) for key in motif_keys] +
                                [float(steps[row['global_step']]['motif_occupancy'])]
                                 for row in capture])
    phases = views['decoder_control'][:, -4:]
    views['synthesis_phase'] = np.concatenate((np.sin(phases), np.cos(phases)), axis=1)
    requested = ('gru', 'field_fine', 'field_meso', 'field_coarse', 'morphic_delta_l01',
                 'morphic_delta_l08', 'morphic_delta_l16', 'morphic_output', 'host', 'motif',
                 'decoder_control', 'synthesis_phase')
    report = measure(y[indices], {name: views[name] for name in requested},
                     horizons=(1, 2, 8, 16), sample_chunks=8)
    nulls = {}
    for seed in (20261003, 20261004):
        controls = {name: shuffled_view(views[name], seed) for name in ('gru', 'field_coarse', 'host',
                                                                       'decoder_control', 'synthesis_phase')}
        nulls[str(seed)] = measure(y[indices], controls, horizons=(1, 2, 8, 16), sample_chunks=8)
    target_groups = {'envelope': [0, 1], 'spectral_bands': list(range(2, 10)),
                     'flux': [10], 'stereo_geometry': [11, 12]}
    grouped = {name: measure(y[np.asarray(indices)][:, columns],
                            {view: views[view] for view in ('gru', 'field_coarse', 'host')},
                            horizons=(1, 2, 8, 16), sample_chunks=8)
               for name, columns in target_groups.items()}
    return {'coding': report, 'split_shuffled_view_nulls': nulls,
            'future_output_representation_sensitivity': grouped,
            'capture_sha256': sha(Path(summary['regime_capture_path'])),
            'state_semantics': 'intact forward-proposed state at current chunk; future output target',
            'motif_host_semantics': 'observational summaries, not complete memory/ecology contents',
            'null_semantics': 'artificial broken-alignment controls, not deployable causal predictors'}


def ordinary(summary):
    rows = [row for row in summary['sampled_steps'] if row.get('evaluation')]
    keys = ('validation_mean_spectral', 'validation_mean_chroma', 'development_mean_spectral',
            'development_mean_chroma', 'health', 'stagnation', 'fine_delta_rms',
            'motif_candidates', 'motif_rejected_similarity')
    return {'evaluation_never_used_for_choice': True,
            'means': {key: float(np.mean([row['evaluation'][key] for row in rows
                                       if key in row['evaluation']]))
                      for key in keys if any(key in row['evaluation'] for row in rows)},
            'predictor_mse_mean': float(np.mean([row['evaluation']['predictor']['mse'] for row in rows
                                              if row['evaluation'].get('predictor')])) if any(row['evaluation'].get('predictor') for row in rows) else None,
            'final_horizon': summary['horizons'][-1]}


def snapshot(name):
    receipt = json.loads((HERE / f'{name}_receipt.json').read_text())
    output = HERE / 'runs' / name
    folder = output / 'perturbations'
    if not receipt['checks']['no_op_exact']:
        raise RuntimeError('bank no-op gate not met')
    baseline = read_wav(folder / 'baseline/post_dsp.wav')
    n = len(baseline) // CHUNK
    b2, b3 = [int(n * fraction) for fraction in SPLITS[1:]]
    selection_audio = baseline[b2*CHUNK:b3*CHUNK]
    candidates = {}
    started = time.monotonic()
    for export in ('baseline', 'energy_state_s00000000', 'energy_state_s00010000', 'energy_state_s00030000'):
        audio = read_wav(folder / export / 'post_dsp.wav')
        y, physical = describe(audio)
        report = measure(y)
        selected = audio[b2*CHUNK:b3*CHUNK]
        _, selection_physical = describe(selected)
        displacement = float(np.sqrt(np.mean((selected - selection_audio)**2)) /
                             max(np.sqrt(np.mean(selection_audio**2)), 1e-9))
        summary = json.loads((folder / export / 'summary.json').read_text())
        candidates[export] = {'post_dsp_wav': str(folder / export / 'post_dsp.wav'),
            'post_dsp_sha256': sha(folder / export / 'post_dsp.wav'), 'coding': report,
            'physical': physical, 'diagnostics': diagnostics(y), 'selection_physical': selection_physical,
            'selection_waveform_displacement': displacement,
            'naive_selection_score': -float(np.mean(np.diff(y[b2:b3], axis=0)**2)),
            'ordinary': ordinary(summary), 'audit_curve': curve(report),
            'selection_reward': reward(report), 'audit_reward': reward(report, 'audit')}
        if export == 'baseline':
            baseline_y = y.copy()
            baseline_internal = internal(output, summary, y)
            candidates[export]['output_representation_sensitivity'] = {
                name: measure(y[:, columns], horizons=(1, 4, 16, 64, 128)) for name, columns in
                {'envelope': [0, 1], 'spectral_bands': list(range(2, 10)),
                 'flux': [10], 'stereo_geometry': [11, 12]}.items()}
    # No-op analysis is deterministic too, excluding elapsed time fields.
    if candidates['baseline']['post_dsp_sha256'] != candidates['energy_state_s00000000']['post_dsp_sha256']:
        raise RuntimeError('no-op audio not exact')
    transforms = {}
    rng = np.random.default_rng(20261003)
    for label, audio in [('chunk_shuffle', baseline.reshape(n, CHUNK, 2)[rng.permutation(n)].reshape(-1, 2)),
                         ('shared_phase', phase_surrogate(baseline, 20261003))]:
        y, physical = describe(audio)
        report = measure(y)
        transforms[label] = {'coding': report, 'physical': physical, 'diagnostics': diagnostics(y),
                             'eligible_reward': reward(report)['value'] if physical['eligible'] else 0}
    decisions = {'A_baseline': choose(candidates, 0), 'B_measurement_only': choose(candidates, 0),
                 'C_weak_PD': choose(candidates, .02), 'D_naive_predictability': choose(candidates, 0, naive=True)}
    for decision in decisions.values():
        decision['selected_audio_sha256'] = candidates[decision['chosen']]['post_dsp_sha256']
        decision['held_out_audit'] = candidates[decision['chosen']]['audit_curve']
    return {'receipt': receipt, 'candidates': candidates, 'decisions': decisions,
            'internal_baseline': baseline_internal, 'output_nulls': transforms,
            'analysis_elapsed_seconds': time.monotonic()-started,
            'replication': 'declared snapshots of one model/world, same candidate-bank selection, not independent model seeds'}


def reused_lesions():
    result = {}
    for warmup in (0, 128):
        root = ROOT / f'analysis/frozen_host_clamp_20261002/runs/w{warmup}_256/ablations'
        prior = ROOT / f'analysis/frozen_v10_20261002/runs/panel_exact_rng_w{warmup}_256/ablations'
        for source, folder in [('host_clamped', root), ('closed_loop', prior)]:
            for export in ('full', 'coarse_hold', 'gru_hold', 'morphic_upper_bypass'):
                actual = export if source == 'closed_loop' or export == 'full' else export+'_control_energy_replay'
                path = folder / actual / 'post_dsp.wav'
                y, physical = describe(read_wav(path))
                report = measure(y, short_protocol=True)
                result[f'w{warmup}_{source}_{export}'] = {'wav_sha256': sha(path), 'coding': report,
                    'physical': physical, 'diagnostics': diagnostics(y),
                    'scope': '256-chunk reused short-history sensitivity, not comparable to primary 1024-chunk protocol'}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', choices=['w0', 'w128', 'both'], default='both')
    parser.add_argument('--include-lesions', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    hashes = parent_hashes()
    result = {'schema': 1, 'snapshots': {}, 'parent_sha256': hashes,
              'method': 'held-out quantized predictive coding, tiny ridge, no optimizer',
              'coding_protocol': 'v2_affine_periodic',
              'v1_first_readout_sha256': sha(HERE/'W0_RESULTS.json'),
              'pd_weight': .02, 'parameter_bits_assumed': 16}
    for name in ('w0', 'w128') if args.snapshot == 'both' else (args.snapshot,):
        result['snapshots'][name] = snapshot(name)
    if args.include_lesions:
        result['reused_lesions'] = reused_lesions()
    if hashes != parent_hashes():
        raise RuntimeError('parent changed during read-only analysis')
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({name: {role: decision['chosen'] for role, decision in value['decisions'].items()}
                      for name, value in result['snapshots'].items()}))


if __name__ == '__main__':
    main()
