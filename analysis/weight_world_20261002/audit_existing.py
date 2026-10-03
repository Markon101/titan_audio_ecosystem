#!/usr/bin/env python3
"""Audit the supplied telemetry objections by named columns, without training."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
MATCHED=ROOT/'analysis/matched_exposure_20261001'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    x=np.asarray(values,dtype=float)
    if not len(x):return None
    return {'n':len(x),'min':float(x.min()),'median':float(np.median(x)),
            'p95':float(np.quantile(x,.95)),'max':float(x.max()),'mean':float(x.mean()),'std':float(x.std())}


def rows(path, allow_incomplete_tail=False):
    with path.open(newline='') as source:
        reader=csv.DictReader(source);value=list(reader)
    bad=[index for index,row in enumerate(value) if None in row or any(item is None for item in row.values())]
    if bad and allow_incomplete_tail and bad==[len(value)-1]:value=value[:-1]
    elif bad:
        raise RuntimeError(f'shifted/truncated CSV schema: {path}')
    return value


def number(row,key):return float(row[key])


def main():
    plan=json.loads((MATCHED/'matched_exposure_receipt.json').read_text())
    roots=list((MATCHED/'runs').glob('seed_*'))
    roots += [ROOT/'analysis/metastable_20261001/runs/observer_step_58107',
              ROOT/'analysis/metastable_20261001/runs/segment1_step_59513',
              Path('/sdcard/Download/TITAN_v10_msfield_fresh_20260930_02_cont'),
              Path('/sdcard/Download/TITAN_v10_msfield_fresh_20260930_02')]
    out={'schema':1,'scope':'provided Space Bunny checklist; original anonymous document was not supplied',
         'substrate_semantics':{'msfield_active_spatial_rings':0,'msfield_far_ring_gain':0,
             'msfield_manifold_depth':4,'meaning':'legacy CA ring controls are inapplicable; all 64 scale channels stay active'},
         'trace_results':{},'limitations':[
             'sampled HOLD runs do not prove uninterrupted per-chunk lock',
             'exact training advantage cannot be recovered without pre-update bandit reward EMA; centered reward is a separate measure',
             'update detection is interval-censored by sparse sampling; latest grad_norm/clip_scale repeat between updates',
             'fine deltas are proposed learned field steps; current forcing and exact hidden deltas are not in these training CSVs',
             'no pathological threshold is invented from RMS or confidence alone']}
    for root in roots:
        trace=next(root.glob('uncertainty_trace_rust*.csv'),None)
        metadata=next(root.glob('titan_run_metadata_v10*.json'),None)
        if trace is None or metadata is None:continue
        before=sha(trace);r=rows(trace);meta=json.loads(metadata.read_text())
        if not r:continue
        if sha(trace)!=before:raise RuntimeError('live trace changed during audit')
        scales=np.asarray([number(row,'clip_scale') for row in r])
        norms=np.asarray([number(row,'grad_norm') for row in r])
        expected=np.minimum(1,5/np.maximum(norms,1e-6))
        confidence=np.asarray([number(row,'model_confidence') for row in r])
        log_conf=np.asarray([-2.8*number(row,'prediction_error')-1.4*number(row,'calibration_error') for row in r])
        unclamped=np.exp(log_conf)
        triple=[all(row[key]=='HOLD' for key in ('planner_proposal','bandit_proposal','action')) for row in r]
        streak=0;longest=0;start=0;span=0
        for index,value in enumerate(triple):
            if value:
                if streak==0:start=index
                streak+=1
                if streak>longest:longest=streak;span=int(r[index]['step'])-int(r[start]['step'])
            else:streak=0
        actions={}
        for action in sorted({row['action'] for row in r}):
            group=[row for row in r if row['action']==action]
            actions[action]={'reward':stats([number(row,'reward') for row in group]),
                'reward_minus_adaptive_reward_mean':stats([number(row,'reward')-number(row,'reward_mean') for row in group])}
        families={}
        arm=root.name.rsplit('_',1)[-1] if root.name.startswith('seed_') else None
        aliases={slot['alias']:slot['source_family'] for slot in plan['arms'][arm]['slots']} if arm in plan['arms'] else {}
        manifest_path=Path(meta.get('corpus',{}).get('manifest','/sdcard/Download/titan_corpus_manifest_v7_sml.json'))
        if manifest_path.is_file():
            manifest=json.loads(manifest_path.read_text());aliases.update({entry['file']:entry['family'] for entry in manifest['entries'] if not entry['file'].startswith('slot_')})
        for row in r:
            family=aliases.get(row['target_file'],row['target_file'])
            families.setdefault(family,[]).append(number(row,'output_low_band_ratio')-number(row,'target_low_band_ratio'))
        updates={int(row['optimizer_updates']):row for row in r if int(row['optimizer_updates_run'])>0}
        item={'path':str(trace),'sha256':before,'rows':len(r),'named_schema_valid':True,
              'field_rail_excess':stats([number(row,'field_rail_excess') for row in r]),
              'field_rms_heuristic_not_rail_fraction':stats([number(row,'micro_amp') for row in r]),
              'confidence':stats(confidence),'confidence_at_upper_clamp_fraction':float(np.mean(abs(confidence-.98)<1e-6)),
              'confidence_unclamped_reconstructed':stats(unclamped),
              'confidence_formula_max_absolute_error':float(np.max(abs(confidence-np.clip(unclamped,.02,.98)))),
              'confidence_formula_first_row_error':float(abs(confidence[0]-np.clip(unclamped[0],.02,.98))),
              'confidence_formula_max_error_after_first':float(np.max(abs(confidence[1:]-np.clip(unclamped[1:],.02,.98)))) if len(r)>1 else None,
              'clip_scale':stats(scales),'clip_zero_rows':int(np.sum(scales==0)),
              'clip_below_1e_minus_6_rows':int(np.sum(scales<=1e-6)),
              'clip_formula_max_absolute_error':float(np.max(abs(scales-expected))),
              'sampled_unique_optimizer_updates':len(updates),
              'sampled_unique_updates_clipped':sum(number(row,'clip_scale')<1 for row in updates.values()),
              'triple_hold_sampled_rows':sum(triple),'longest_triple_hold_sampled_streak':longest,
              'longest_triple_hold_observed_span_chunks':span,
              'action_conditioned_reward':actions,
              'low_band_output_minus_target_by_family':{family:stats(values) for family,values in families.items()}}
        field_path=next(root.glob('msfield_trace_v10*.csv'),None)
        if field_path:
            with field_path.open(newline='') as source:raw_field=list(csv.DictReader(source))
            tail_incomplete=bool(raw_field and any(value is None for value in raw_field[-1].values()))
            f=rows(field_path,allow_incomplete_tail=True)
            field_ids=sorted({row['run_id'] for row in f})
            compatible=field_ids==[meta['run_id']]
            tape=meta['invocation']['autograd_tape_chunks'];horizon=meta['invocation']['bptt_requested_chunks']
            top=sorted(f,key=lambda row:number(row,'fine_delta_rms'),reverse=True)[:8]
            item['field_trace_sha256']=sha(field_path)
            item['field_trace_run_ids']=field_ids
            item['field_trace_metadata_matches']=compatible
            item['field_trace_incomplete_tail']=tail_incomplete
            item['fine_delta_rms']=stats([number(row,'fine_delta_rms') for row in f])
            item['largest_sampled_fine_deltas']=[{'step':int(row['global_step']),'run_step':int(row['run_step']),
                'fine_delta_rms':number(row,'fine_delta_rms'),
                'on_tape_boundary':(int(row['run_step'])+1)%tape==0 if compatible else None,
                'on_optimizer_horizon':(int(row['run_step'])+1)%horizon==0 if compatible else None,
                'first_resume_sample':int(row['run_step'])==0} for row in top]
            item['sampling_modulo_tape_counts']={str(rem):sum((int(row['run_step'])+1)%tape==rem for row in f) for rem in range(tape)} if compatible else None
        out['trace_results'][root.name]=item
    (HERE/'telemetry_audit.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    for name,item in out['trace_results'].items():
        print(name,'rail max',item['field_rail_excess']['max'],'confidence clamp',round(item['confidence_at_upper_clamp_fraction'],3),
              'clip min',item['clip_scale']['min'],'triple HOLD longest samples',item['longest_triple_hold_sampled_streak'])


if __name__=='__main__':main()
