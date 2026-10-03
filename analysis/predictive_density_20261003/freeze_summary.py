"""Freeze curves, descriptive novelty, overlap and a compact result ledger."""
import csv
import json
from pathlib import Path
import numpy as np
from audio import CHUNK, read_wav, describe
from predictive import diagnostics
from run_frozen import HERE, sha


def main():
    targets = [HERE/'SUMMARY.json', HERE/'density_curves.csv', HERE/'NOVELTY_RESULTS.json', HERE/'overlap_receipt.json']
    if any(path.exists() for path in targets):
        raise FileExistsError('preserve existing frozen summary')
    result = json.loads((HERE/'RESULTS.json').read_text())
    overlap = {'schema': 1, 'snapshots_are_not_independent': True, 'overlap_chunks': 896,
               'overlap_fraction_of_each_baseline': .875, 'raw_post_overlap_exact': {}}
    for filename in ('raw_renderer.wav', 'post_dsp.wav'):
        a = read_wav(HERE/'runs/w0/perturbations/baseline'/filename)
        b = read_wav(HERE/'runs/w128/perturbations/baseline'/filename)
        exact = np.array_equal(a[128*CHUNK:], b[:896*CHUNK])
        if not exact:
            raise RuntimeError('baseline overlap hypothesis failed; preserve and investigate')
        overlap['raw_post_overlap_exact'][filename] = bool(exact)
    summary = {'schema': 1, 'results_sha256': sha(HERE/'RESULTS.json'), 'snapshots': {},
               'overlap': overlap, 'no_generation_benefit_demonstrated': True,
               'no_supported_net_predictive_depth': True, 'qualitative_audio_ratings': None}
    novelty = {'schema': 1, 'descriptive_only_not_reward': True, 'candidates': {}}
    with (HERE/'density_curves.csv').open('x', newline='') as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerow(['snapshot','domain','probe','horizon_chunks','horizon_seconds','forecasts',
                         'gain_bits','gain_bits_per_feature_forecast','coefficient_cost_bits','net_gain_bits',
                         'pd_total','pd_per_feature_forecast','over_history_bits','over_unpenalized_baseline_bits'])
        for name, snapshot in result['snapshots'].items():
            candidate = snapshot['candidates']['baseline']
            domains = {'output': candidate['coding'], 'internal_to_output': snapshot['internal_baseline']['coding']}
            for domain, report in domains.items():
                for h, item in report['horizons'].items():
                    if item['supported']:
                        for probe, scores in item['probes'].items():
                            a = scores['audit']; cost = max(a['coefficient_cost_bits'],16)
                            writer.writerow([name,domain,probe,h,item['horizon_seconds'],a['forecasts'],
                                a['gain_bits'],a['bits_per_feature_forecast'],a['coefficient_cost_bits'],a['net_gain_bits'],
                                a['pd_bits_saved_per_coefficient_bit'],a['bits_per_feature_forecast']/cost,
                                a['incremental_over_history_bits'],a['incremental_over_unpenalized_baseline_bits']])
            for key, item in snapshot['candidates'].items():
                y, physical = describe(read_wav(item['post_dsp_wav']))
                novelty['candidates'][name+'_'+key] = {'diagnostics': diagnostics(y), 'physical':physical}
            summary['snapshots'][name] = {'decisions': {role: decision['chosen'] for role,decision in snapshot['decisions'].items()},
                'baseline_audit_curve': candidate['audit_curve'],
                'naive_selected': snapshot['candidates'][snapshot['decisions']['D_naive_predictability']['chosen']]['physical'],
                'baseline_rms': candidate['physical']['rms'], 'analysis_elapsed_seconds':snapshot['analysis_elapsed_seconds'],
                'runtime_resource':snapshot['receipt']['resource']}
    targets[0].write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    targets[2].write_text(json.dumps(novelty,indent=2,allow_nan=False)+'\n')
    targets[3].write_text(json.dumps(overlap,indent=2)+'\n')
    print(json.dumps({'baseline_overlap_exact':True,'overlap_fraction':.875,'frozen_curve_csv':str(targets[1])}))


if __name__ == '__main__':
    main()
