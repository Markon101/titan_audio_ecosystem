"""Stream existing frozen artifacts through source-trained, persisted probes."""
import csv
import hashlib
import json
from pathlib import Path
import resource
import time
import wave
import numpy as np
from transfer import FrozenBank,PREVIOUS,plain
from audio import read_wav,describe,CHUNK
from run_frozen import parent_hashes,sha

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
VIEWS=('gru','field_fine','field_meso','field_coarse','morphic_delta_l01','morphic_delta_l08',
       'morphic_delta_l16','morphic_output','host','motif','decoder_control','synthesis_phase')


def load_episode(root,folder):
    directory=root/folder
    summary=json.loads((directory/'summary.json').read_text())
    if summary['optimizer_steps'] or not summary['fixed_weights'] or not summary['fixed_morphology']:
        raise RuntimeError('not frozen inference')
    if summary['weights_hash_before']!=summary['weights_hash_after']:
        raise RuntimeError('weights changed')
    capture_path=Path(summary['regime_capture_path'])
    capture=[json.loads(line) for line in capture_path.read_text().splitlines()]
    step={row['absolute_step']:row for row in summary['sampled_steps']}
    capture=[row for row in capture if step[row['global_step']]['rollout_offset']%8==0]
    indices=np.asarray([step[row['global_step']]['rollout_offset']-1 for row in capture])
    y,physical=describe(read_wav(directory/'post_dsp.wav'))
    if len(indices)!=len(y)//8 or not np.all(np.diff(indices)==8):
        raise RuntimeError('state/audio grid mismatch')
    views={name:np.asarray([row['views'][name] for row in capture]) for name in capture[0]['views']}
    host=('energy','temperature','radiation_probability','radiation_realized','shear_amplitude',
          'kick_amplitude','model_confidence_raw','target_error_feedback')
    views['host']=np.asarray([[float(step[row['global_step']].get(k) or 0) for k in host] for row in capture])
    motif=('motif_nearest_distance','motif_candidates','motif_rejected_similarity','motif_rejected_quality')
    views['motif']=np.asarray([[float(row.get(k) or 0) for k in motif]+[step[row['global_step']]['motif_occupancy']]
                              for row in capture])
    phases=views['decoder_control'][:,-4:]
    views['synthesis_phase']=np.concatenate((np.sin(phases),np.cos(phases)),axis=1)
    views={name:views[name] for name in VIEWS}
    with wave.open(str(directory/'post_dsp.wav'),'rb') as source:
        chunks=[hashlib.sha256(source.readframes(CHUNK)).hexdigest() for _ in range(len(y))]
    report=json.loads((root/'analysis_report.json').read_text())
    if not report['non_mutation']['unchanged']:
        raise RuntimeError('prior input non-mutation gate failed')
    provenance=json.loads((root/'provenance.json').read_text())
    invocation=provenance['analysis_configuration']['invocation']
    seed=invocation[invocation.index('--analysis-seed')+1]
    metadata={'root':str(root),'folder':folder,'warmup_chunks':summary['warmup_chunks'],
              'audio_sha256':sha(directory/'post_dsp.wav'),'capture_sha256':sha(capture_path),
              'summary_sha256':sha(directory/'summary.json'),'report_sha256':sha(root/'analysis_report.json'),
              'initial_world_fingerprint':summary['initial_world_fingerprint'],
              'weight_hash':summary['weights_hash_before'],'analysis_seed':seed,'physical':physical}
    return y,y[indices],views,chunks,metadata


def shuffle(values,seed,boundaries):
    result={}
    rng=np.random.default_rng(seed)
    for name,x in values.items():
        value=x.copy()
        for start,end in zip(boundaries[:-1],boundaries[1:]):
            value[start:end]=x[start:end][rng.permutation(end-start)]
        result[name]=value
    return result


def domains(audio_bank,state_bank,y,sy,views,warmup,start_audio=None,start_state=None,adapt=False):
    return {'output':audio_bank.score(y,warmup=warmup,start=start_audio,adapt_sigma=adapt),
            'state_to_output':state_bank.score(sy,views,warmup=warmup,start=start_state,adapt_sigma=adapt)}


def persist_verified(bank,path):
    if path.exists():
        if json.loads(path.read_text())!=plain(bank.state):
            raise RuntimeError('preserve incompatible partial probe artifact')
    else:
        bank.save(path)


def main():
    if (HERE/'RESULTS.json').exists() or (HERE/'RESULTS.json.gz').exists():
        raise FileExistsError('preserve completed transfer study')
    mem=next(int(line.split()[1])/1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    if mem<1200:raise RuntimeError('postpone offline fits below 1.2 GiB available RAM')
    initial=parent_hashes();started=time.monotonic()
    (HERE/'runs').mkdir(exist_ok=True)
    source_root=PREVIOUS/'runs/w0'
    y,sy,views,source_chunks,source_meta=load_episode(source_root,'perturbations/baseline')
    audio=FrozenBank.train(y)
    state=FrozenBank.train(sy,views,horizons=(1,2,8,16),sample_chunks=8)
    probe_paths=[HERE/'runs/audio_probe.json',HERE/'runs/state_probe.json']
    persist_verified(audio,probe_paths[0]);persist_verified(state,probe_paths[1])
    restored_audio,restored_state=[FrozenBank.load(p) for p in probe_paths]
    source_score=domains(audio,state,y,sy,views,0,int(len(y)*.8),int(len(sy)*.8))
    restored_score=domains(restored_audio,restored_state,y,sy,views,0,int(len(y)*.8),int(len(sy)*.8))
    if source_score!=restored_score:raise RuntimeError('probe restoration changed source predictions')
    audio,state=restored_audio,restored_state
    old=json.loads((PREVIOUS/'RESULTS.json').read_text())['snapshots']['w0']
    for domain,current,prior in [('output',source_score['output'],old['candidates']['baseline']['coding']),
                                 ('state_to_output',source_score['state_to_output'],old['internal_baseline']['coding'])]:
        for h,item in current['horizons'].items():
            for name,score in item['probes'].items():
                if name=='history':
                    entry=prior['horizons'][h]['probes']['output_slow']['audit']
                    previous=entry['gain_bits']-entry['incremental_over_history_bits']
                else:
                    previous=prior['horizons'][h]['probes'][name]['audit']['gain_bits']
                if not np.isclose(score['gain_bits'],previous,rtol=0,atol=1e-8):
                    raise RuntimeError(f'source v2 parity failed: {domain}/{h}/{name}')
    source_boundaries=[0,int(len(sy)*.4),int(len(sy)*.6),len(sy)]
    null_banks={}
    for seed in (20261003,20261004,20261005):
        bank=FrozenBank.train(sy,shuffle(views,seed,source_boundaries),horizons=(1,2,8,16),sample_chunks=8)
        path=HERE/'runs'/f'null_{seed}.json';persist_verified(bank,path)
        null_banks[seed]=FrozenBank.load(path)
    hash_before={p.name:sha(p) for p in (HERE/'runs').glob('*.json')}
    registry=[('source_audit',source_root,'perturbations/baseline','source chronological holdout'),
              ('overlap_w128',PREVIOUS/'runs/w128','perturbations/baseline','known overlapping reference')]
    matrix=ROOT/'analysis/weight_world_20261002/runs'
    registry += [(name,matrix/name,'ablations/full',scope) for name,scope in
                 [('L16_WP_SP','primary different RNG, same parent weights/world'),
                  ('L16_WP_SC','world-state transfer, parent weights'),
                  ('L16_WC_SP','weight shift, parent world'),('L16_WC_SC','weight and world shift'),
                  ('L16_WE_SP','distant early-weight OOD; upper depth was untrained')]]
    registry += [(f'{snapshot}_{label}',PREVIOUS/'runs'/snapshot,'perturbations/'+stem,'host energy intervention transfer')
                 for snapshot in ('w0','w128') for label,stem in
                 [('energy01','energy_state_s00010000'),('energy03','energy_state_s00030000')]]
    result={'schema':1,'source':source_meta,'fixed_probes_sha256':hash_before,'episodes':{},
            'source_v2_code_parity':True,'save_reload_exact':True,'null_seeds':list(null_banks),
            'no_titan_training_render_or_build':True}
    rows=[]
    for name,root,folder,scope in registry:
        ty,tsy,tv,chunks,meta=load_episode(root,folder)
        source_audit=name=='source_audit'
        warmup=meta['warmup_chunks']
        start_audio=int(len(ty)*.8) if source_audit else None
        start_state=int(len(tsy)*.8) if source_audit else None
        strict=domains(audio,state,ty,tsy,tv,warmup,start_audio,start_state)
        entry={'scope':scope,'metadata':meta,'strict':strict,
               'exact_chunks_reused_from_source_fit_calibration':sum(c in set(source_chunks[:int(len(y)*.6)]) for c in chunks),
               'exact_chunks_reused_from_source_full':sum(c in set(source_chunks) for c in chunks),
               'nulls':{}}
        if not source_audit:
            entry['strict_matched_suffix']=domains(audio,state,ty,tsy,tv,warmup,256,32)
            entry['sigma_only_adapted']=domains(audio,state,ty,tsy,tv,warmup,adapt=True)
        for seed,bank in null_banks.items():
            boundaries=[0,int(len(tsy)*.4),int(len(tsy)*.6),len(tsy)] if source_audit else [0,32,len(tsy)]
            null_views=shuffle(tv,seed,boundaries)
            entry['nulls'][str(seed)]={'strict':bank.score(tsy,null_views,warmup=warmup,start=start_state)}
            if not source_audit:
                entry['nulls'][str(seed)]['strict_matched_suffix']=bank.score(tsy,null_views,warmup=warmup,start=32)
                entry['nulls'][str(seed)]['sigma_only_adapted']=bank.score(tsy,null_views,warmup=warmup,adapt_sigma=True)
        result['episodes'][name]=entry
        for mode in ('strict','strict_matched_suffix','sigma_only_adapted'):
            for domain,report in entry.get(mode,{}).items():
                for h,item in report['horizons'].items():
                    if item['supported']:
                        for probe,metrics in item['probes'].items():
                            rows.append([name,mode,domain,probe,h,item['forecasts'],metrics['gain_bits'],
                                metrics['bits_per_feature_forecast'],metrics['over_history_bits'],
                                metrics['over_unpenalized_baseline_bits'],metrics['over_same_capacity_slow_history_bits'],
                                metrics['mean_only_common_history_sigma_gain_bits'],metrics['scaled_mse'],metrics['net_gain_bits']])
        print(json.dumps({'episode':name,'source_fit_calibration_chunk_overlap':entry['exact_chunks_reused_from_source_fit_calibration']}),flush=True)
    if initial!=parent_hashes():raise RuntimeError('canonical parent changed')
    if hash_before!={p.name:sha(p) for p in (HERE/'runs').glob('*.json')}:
        raise RuntimeError('evaluation mutated the fixed probes')
    result['parent_unchanged']=True
    result['elapsed_seconds']=time.monotonic()-started
    result['peak_python_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    (HERE/'RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (HERE/'curves.csv').open('x',newline='') as output:
        writer=csv.writer(output,lineterminator='\n')
        writer.writerow(['episode','mode','domain','probe','horizon_chunks','forecasts','gain_bits',
                         'gain_per_feature_forecast','over_history_bits','over_unpenalized_baseline_bits',
                         'over_same_capacity_slow_history_bits','mean_only_common_sigma_gain_bits','scaled_mse','net_gain_bits'])
        writer.writerows(rows)
    (HERE/'verification_receipt.json').write_text(json.dumps({'source_v2_code_parity':True,'save_reload_exact':True,
        'fixed_probes_unchanged':True,'parent_unchanged':True,'episodes':len(registry),
        'peak_rss_mib':result['peak_python_rss_mib'],'elapsed_seconds':result['elapsed_seconds'],
        'source_sha256':source_meta['audio_sha256'],'result_sha256':sha(HERE/'RESULTS.json')},indent=2)+'\n')


if __name__=='__main__':main()
