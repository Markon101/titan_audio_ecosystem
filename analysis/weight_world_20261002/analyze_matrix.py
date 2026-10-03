#!/usr/bin/env python3
"""Freeze paired weight/world effects; never infer independent replicates."""
import copy
import json
from pathlib import Path
import sys
import numpy as np
import run_matrix as matrix

sys.path.insert(0,str(matrix.ROOT/'analysis/metastable_20261001'))
import regime_archive as regimes
sys.path.insert(0,str(matrix.ROOT/'analysis/frozen_v10_20261002'))
from analyze_frozen_panel import read_wav, geometry, rms, spectral_distribution, correlation

PRIMARY=('L16_WP_SP','L16_WC_SP','L16_WP_SC','L16_WC_SC')


def pr(values):return regimes.participation_ratio(np.asarray(values,dtype=float).copy())


def measured(row):
    e=row['evaluation'];p=e.get('predictor') or {}
    values={key:float(value) for key,value in e.items() if type(value) in (int,float) and key!='evaluation_clock'}
    for kind in ('development','validation','target'):
        for index,name in enumerate(('side_mid','correlation','level')):
            values[f'{kind}_stereo_{name}_mse']=e[f'{kind}_stereo_squared_errors'][index]
    for key,value in p.items():values[f'predictor_{key}']=value
    values.update({key:row[key] for key in ('micro_rms','macro_rms','coarse_rms','recurrent_rms','micro_near_bound_fraction',
                                         'macro_near_bound_fraction','recurrent_proposed_delta_rms')})
    return values


def effects(cells,key):
    a,b,c,d=[cells[name]['episode_weighted_means'][key] for name in PRIMARY]
    return {'weight_effect_in_parent_world':b-a,'weight_effect_in_child_world':d-c,
            'mean_weight_effect':((b-a)+(d-c))/2,'world_effect_with_parent_weights':c-a,
            'world_effect_with_child_weights':d-b,'mean_world_effect':((c-a)+(d-b))/2,
            'interaction':(d-c)-(b-a),
            'relative_weight_effect':(((b-a)+(d-c))/2)/max(1e-12,abs((a+c)/2))}


def archive_summary(capture,calibration):
    archive=regimes.Archive(copy.deepcopy(calibration))
    trace=[archive.process(row) for row in capture]
    return {'candidate_regions':len(archive.state['regimes']),
            'persistent_counter':sum(regime['persistent'] for regime in archive.state['regimes']),
            'revisits':archive.state['revisit_count'],
            'transitions':archive.state['transition_count'],
            'mean_recurrence_rate':float(np.mean([row['recurrence_rate'] for row in trace if row['recurrence_rate'] is not None])),
            'trap_events':sum(row['residence_state']=='over_resident' for row in trace)}


def main():
    schedule=json.loads((matrix.HERE/'schedule.json').read_text())
    cal=json.loads((matrix.ROOT/'analysis/metastable_20261001/regime_calibration.json').read_text())
    result={'schema':1,'cells':{},'factorial_effects':{},'limitations':[
        'one lineage, two mature worlds; temporal blocks are correlated observations, not model seeds',
        'fixed probe distance is distributional support, not reconstruction of uniquely identified phrases',
        'early L16 activates previously untrained upper blocks; L1 sensitivity changes the mature decoder coordinate context',
        'persistent region discovery is not estimable in 384 chunks with the original observer residence window',
        'predictor offset-1 scores share identical history within a world; later predictor scores use each model own trajectory']}
    captures={};audio={}
    for name,weights,world,depth in matrix.CELLS:
        output=matrix.RUNS/name
        receipt=matrix.checked_complete(output,384)
        summary=json.loads((output/'ablations/full/summary.json').read_text())
        evaluated=[row for row in summary['sampled_steps'] if row.get('evaluation')]
        if len(evaluated)!=49:raise RuntimeError(f'incomplete sampled evaluation in {name}')
        for row in evaluated:
            clock=row['evaluation']['evaluation_clock']
            episode=next(entry for entry in schedule['episodes'] if entry['start_step']<=clock<entry['start_step']+entry['chunks'])
            if row['target_file']!=f"slot_{episode['slot']:02}.wav" or row['target_frame']!=episode['source_frame']+(clock-episode['start_step'])*4096:
                raise RuntimeError(f'target mismatch in {name}')
        metrics=[measured(row) for row in evaluated]
        keys=sorted(set.intersection(*(set(row) for row in metrics)))
        episode_values={}
        for key in keys:
            episode_values[key]=[float(np.mean([metrics[index][key] for index,row in enumerate(evaluated)
                                    if episode['start_step']<=row['evaluation']['evaluation_clock']<episode['start_step']+64]))
                                 for episode in schedule['episodes']]
        capture=list(regimes.read_capture([Path(summary['regime_capture_path'])]));captures[name]=capture
        views={}
        for view in sorted(capture[0]['views']):
            points=np.asarray([row['views'][view] for row in capture]);views[view]={
                'trajectory_participation_ratio':pr(points),'delta_participation_ratio':pr(np.diff(points,axis=0))}
            if view.startswith('field_'):views[view]['final_rms_signedmean_railfraction_neighbor_delta']=points[-1,-4:].tolist()
        wav=read_wav(output/'ablations/full/post_dsp.wav');audio[name]=wav
        last=evaluated[-1]['evaluation'];first=evaluated[0]['evaluation']
        candidates=last['motif_candidates']-first['motif_candidates']
        item={'receipt':receipt,'active_depth':depth,'initial_world_fingerprint':summary['initial_world_fingerprint'],
              'weights_hash':summary['weights_hash_before'],'probes':summary['probe_manifest'],
              'episode_weighted_means':{key:float(np.mean(value)) for key,value in episode_values.items()},
              'episode_metrics':episode_values,'first_common_history_predictor':first['predictor'],
              'first_probe_scores':{key:first[key] for key in ('validation_mean_spectral','validation_mean_chroma','development_mean_spectral','development_mean_chroma')},
              'health_range':[min(row['evaluation']['health'] for row in evaluated),max(row['evaluation']['health'] for row in evaluated)],
              'bounded':summary['horizons'][-1]['bounded'],'nonfinite':summary['horizons'][-1]['nonfinite_detected'],
              'clipping':summary['horizons'][-1]['audio']['clipping_fraction'],
              'motif_candidate_delta':candidates,'motif_store_delta':last['motif_stored_total']-first['motif_stored_total'],
              'motif_similarity_reject_delta':last['motif_rejected_similarity']-first['motif_rejected_similarity'],
              'motif_quality_reject_delta':last['motif_rejected_quality']-first['motif_rejected_quality'],
              'trajectory_views':views,'audio':geometry(wav)}
        if depth==16:
            observed=archive_summary(capture,cal)
            rng=np.random.default_rng(20261002);permutation=rng.permutation(len(capture));shuffled=copy.deepcopy(capture);gaussian=copy.deepcopy(capture)
            for i,row in enumerate(shuffled):row['views']=copy.deepcopy(capture[int(permutation[i])]['views'])
            for view in capture[0]['views']:
                x=np.asarray([row['views'][view] for row in capture]);noise=rng.normal(x.mean(0),x.std(0),size=x.shape)
                for row,point in zip(gaussian,noise):row['views'][view]=point.tolist()
            item['exploratory_archive']={'observed':observed,'time_shuffle':archive_summary(shuffled,cal),
                'matched_gaussian':archive_summary(gaussian,cal),
                'persistent_discovery_not_estimable':True,'calibration_sha256':matrix.sha(matrix.ROOT/'analysis/metastable_20261001/regime_calibration.json')}
        result['cells'][name]=item
    for world in ('SP','SC'):
        selected=[value['initial_world_fingerprint'] for name,value in result['cells'].items() if name.startswith('L16_') and name.endswith(world)]
        if len(set(selected))!=1:raise RuntimeError(f'initial state differs across weights under {world}')
    if result['cells']['L1_WE_SP']['initial_world_fingerprint']!=result['cells']['L1_WP_SP']['initial_world_fingerprint']:
        raise RuntimeError('L1 prefix common-state gate failed')
    if len({json.dumps(cell['probes'],sort_keys=True) for cell in result['cells'].values()})!=1:
        raise RuntimeError('fixed probes differ across cells')
    keys=set.intersection(*(set(result['cells'][name]['episode_weighted_means']) for name in PRIMARY))
    for key in sorted(keys):result['factorial_effects'][key]=effects(result['cells'],key)
    result['paired_audio_distances']={}
    for a,b in [('L16_WP_SP','L16_WC_SP'),('L16_WP_SC','L16_WC_SC'),('L16_WP_SP','L16_WP_SC'),
                ('L16_WC_SP','L16_WC_SC'),('L16_WE_SP','L16_WP_SP'),('L16_WE_SC','L16_WP_SC'),('L1_WE_SP','L1_WP_SP')]:
        aa,bb=audio[a],audio[b]
        view_distances={view:float(np.sqrt(np.mean((np.asarray(captures[a][-1]['views'][view])-np.asarray(captures[b][-1]['views'][view]))**2)))
                        for view in set(captures[a][-1]['views'])&set(captures[b][-1]['views'])}
        result['paired_audio_distances'][f'{a}__{b}']={'waveform_rms_difference':rms(bb-aa),
            'waveform_correlation':correlation(aa,bb),'average_log_spectral_rms':rms(spectral_distribution(aa)-spectral_distribution(bb)),
            'final_compact_view_rms_differences':view_distances}
    result['initial_state_matched_across_weights']=True
    result['fixed_probes_matched_all_cells']=True
    (matrix.HERE/'RESULTS.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    for name,cell in result['cells'].items():
        x=cell['episode_weighted_means'];print(name,round(x['validation_mean_spectral'],5),round(x['validation_mean_chroma'],5),round(x['predictor_mse'],7))
    for key in ('validation_mean_spectral','validation_mean_chroma','predictor_mse'):
        print(key,result['factorial_effects'][key])


if __name__=='__main__':main()
