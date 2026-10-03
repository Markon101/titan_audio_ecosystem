"""Compact readout and fixed-normalization drift/mean diagnostics, no new fit."""
import gzip
import json
from pathlib import Path
import numpy as np
from transfer import FrozenBank
from run_transfer import HERE,ROOT,PREVIOUS,load_episode,VIEWS
from run_frozen import sha


def drift(bank,y,views):
    s=bank.state
    def assess(x,sc):
        normalized=(x-np.asarray(sc['mean']))/np.asarray(sc['scale'])
        return {'rms_standardized_distance':float(np.sqrt(np.mean(normalized**2))),
                'clip_fraction':float(np.mean(np.abs(normalized)>30)),
                'outside_training_range_fraction':float(np.mean((x<np.asarray(sc['minimum']))|
                                                                 (x>np.asarray(sc['maximum']))))}
    return {'target':assess(y,s['y_scaler']),
            'views':{name:assess(views[name],s['views'][name]['before']) for name in VIEWS}}


def main():
    if (HERE/'SUMMARY.json').exists() or (HERE/'DRIFT_DIAGNOSTIC.json').exists():
        raise FileExistsError('preserve frozen readout')
    result=json.loads((HERE/'RESULTS.json').read_text())
    bank=FrozenBank.load(HERE/'runs/state_probe.json')
    summary={'schema':1,'no_training_reward_promoted':True,'episodes':{},
             'prior_source_v2_parity':result['source_v2_code_parity'],
             'actual_serialized_probe_bytes':sum(p.stat().st_size for p in (HERE/'runs').glob('*_probe.json')),
             'elapsed_seconds':result['elapsed_seconds'],'peak_rss_mib':result['peak_python_rss_mib']}
    diagnostics={'schema':1,'source_fixed_normalization':True,'episodes':{}}
    for name,e in result['episodes'].items():
        meta=e['metadata']
        y,sy,views,_,_=load_episode(Path(meta['root']),meta['folder'])
        diagnostics['episodes'][name]=drift(bank,sy,views)
        item={'scope':e['scope'],'source_fit_calibration_chunk_overlap':e['exact_chunks_reused_from_source_fit_calibration'],
              'state_gain_over_history':{},'adaptation':{}}
        for h,row in e['strict']['state_to_output']['horizons'].items():
            if not row['supported']:continue
            item['state_gain_over_history'][h]={}
            for view in ('field_fine','field_meso','field_coarse','gru','host','synthesis_phase'):
                score=row['probes'][view];base=row['probes']['history']
                item['state_gain_over_history'][h][view]={
                    'gain_bits':score['over_history_bits'],
                    'mean_only_bits':score['mean_only_common_history_sigma_gain_bits'],
                    'mse_improvement_percent':100*(base['scaled_mse']-score['scaled_mse'])/base['scaled_mse'],
                    'null_gain_bits':[n['strict']['horizons'][h]['probes'][view]['over_history_bits'] for n in e['nulls'].values()],
                    'net_bits_after_16bit_added_parameter_cost':score['over_history_bits']-16*(
                        bank.state['horizons'][str(int(h)//8)]['models'][view]['parameters']-
                        bank.state['horizons'][str(int(h)//8)]['models']['history']['parameters'])}
        if 'sigma_only_adapted' in e:
            for h,row in e['sigma_only_adapted']['state_to_output']['horizons'].items():
                if not row['supported']:continue
                original=e['strict_matched_suffix']['state_to_output']['horizons'][h]
                item['adaptation'][h]={}
                for view in ('gru','field_coarse','host','synthesis_phase'):
                    a,b=original['probes'][view],row['probes'][view]
                    if a['scaled_mse']!=b['scaled_mse']:raise RuntimeError('sigma adaptation changed mean predictions')
                    item['adaptation'][h][view]={'strict_history_gain_bits':a['over_history_bits'],
                        'adapted_history_gain_bits':b['over_history_bits'],'scaled_mse_unchanged':a['scaled_mse'],
                        'strict_mean_only_bits':a['mean_only_common_history_sigma_gain_bits'],
                        'adapted_mean_only_bits':b['mean_only_common_history_sigma_gain_bits']}
        summary['episodes'][name]=item
    (HERE/'SUMMARY.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    (HERE/'DRIFT_DIAGNOSTIC.json').write_text(json.dumps(diagnostics,indent=2,allow_nan=False)+'\n')
    # Preserve the local readable original; track a deterministic compressed full report.
    archived=HERE/'RESULTS.json.gz'
    if archived.exists():raise FileExistsError(archived)
    with archived.open('xb') as output:output.write(gzip.compress((HERE/'RESULTS.json').read_bytes(),mtime=0))
    (HERE/'archive_receipt.json').write_text(json.dumps({'raw_sha256':sha(HERE/'RESULTS.json'),
        'gzip_sha256':sha(archived),'gzip_mtime':0,'raw_bytes':(HERE/'RESULTS.json').stat().st_size,
        'compressed_bytes':archived.stat().st_size},indent=2)+'\n')
    print(json.dumps({'full_archive_bytes':archived.stat().st_size,'summary':str(HERE/'SUMMARY.json')}))


if __name__=='__main__':main()
