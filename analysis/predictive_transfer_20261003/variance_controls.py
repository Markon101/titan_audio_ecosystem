"""Exact mean/sigma decomposition of fixed-code gains, plus raw mean errors."""
import json
from pathlib import Path
import numpy as np
from transfer import FrozenBank
from run_transfer import HERE,load_episode


def main():
    path=HERE/'VARIANCE_CONTROLS.json'
    if path.exists():raise FileExistsError(path)
    result=json.loads((HERE/'RESULTS.json').read_text())
    banks={'output':FrozenBank.load(HERE/'runs/audio_probe.json'),
           'state_to_output':FrozenBank.load(HERE/'runs/state_probe.json')}
    controls={'schema':1,'no_new_fit':True,'mean_and_sigma_decomposition_exact':True,'episodes':{}}
    for name,episode in result['episodes'].items():
        meta=episode['metadata']
        y,sy,views,_,_=load_episode(Path(meta['root']),meta['folder'])
        entry={}
        for domain,bank in banks.items():
            target,inputs=(y,{}) if domain=='output' else (sy,views)
            mean=np.asarray(bank.state['y_scaler']['mean']);scale=np.asarray(bank.state['y_scaler']['scale'])
            entry[domain]={}
            for h,spec in bank.state['horizons'].items():
                ids,actual,pred=bank.predictions(target,inputs,int(h),meta['warmup_chunks'])
                cutoff=int(len(target)*.8) if name=='source_audit' else bank.state['history_start']
                mask=ids>=cutoff
                hs=str(int(h)*bank.state['sample_chunks'])
                expected=episode['strict'][domain]['horizons'][hs]
                if not expected['supported']:continue
                models=spec['models'];q=bank.quantizer
                history=q.gaussian_bits(actual,pred['history'],models['history']['scale'])
                details={}
                for probe in ('output_slow','field_fine','field_meso','field_coarse','gru','host','synthesis_phase'):
                    if probe not in models:continue
                    model=models[probe]
                    changed_sigma=q.gaussian_bits(actual,pred['history'],model['scale'])
                    full=q.gaussian_bits(actual,pred[probe],model['scale'])
                    variance_gain=float((history[mask]-changed_sigma[mask]).sum())
                    mean_gain=float((changed_sigma[mask]-full[mask]).sum())
                    expected_gain=expected['probes'][probe]['over_history_bits']
                    if not np.isclose(variance_gain+mean_gain,expected_gain,rtol=0,atol=1e-8):
                        raise RuntimeError('decomposition does not reproduce saved code gain')
                    raw_mean=pred[probe][mask]*scale+mean
                    raw_mse=float(np.mean((target[ids[mask]+int(h)]-raw_mean)**2))
                    details[probe]={'full_gain_over_history_bits':expected_gain,
                        'history_mean_with_probe_sigma_gain_bits':variance_gain,
                        'probe_mean_gain_at_probe_sigma_bits':mean_gain,
                        'raw_descriptor_mse':raw_mse,'raw_descriptor_mse_is_not_perceptual_quality':True,
                        'per_target_variance_only_gain_bits':(history[mask]-changed_sigma[mask]).sum(0).tolist(),
                        'per_target_mean_at_probe_sigma_gain_bits':(changed_sigma[mask]-full[mask]).sum(0).tolist()}
                entry[domain][hs]=details
        controls['episodes'][name]=entry
    path.write_text(json.dumps(controls,indent=2,allow_nan=False)+'\n')
    example=controls['episodes']['L16_WP_SP']['state_to_output']['128']['synthesis_phase']
    print(json.dumps({'primary_phase_128':example}))


if __name__=='__main__':main()
