#!/usr/bin/env python3
"""Score external noise/phase controls with the exact existing Rust probe bank."""
import json
from pathlib import Path
import subprocess
import numpy as np
import run_matrix as matrix


def main():
    receipt=json.loads((matrix.HERE/'audio_null_receipt.json').read_text())
    paths=list(receipt['files'])
    paths.append(paths[0])  # Exact repeated scoring control.
    output=matrix.RUNS/'audio_probe_scoring'
    if output.exists():raise FileExistsError('preserve completed/partial audio-probe output')
    cmd=[str(matrix.ROOT/'target/release/titan'),'--analysis-only','--analysis-substrate','msfield',
         '--base-dir',str(matrix.P),'--run-tag',matrix.TAG,'--corpus-dir',str(matrix.CORPUS/'a'),
         '--corpus-manifest',str(matrix.CORPUS/'a_manifest.json'),'--analysis-stride','8',
         '--analysis-dir',str(output),'--analysis-terminal','quiet','--threads','2']
    for path in paths:
        path=Path(path)
        if matrix.sha(path)!=receipt['files'][str(path)]['sha256']:raise RuntimeError('null WAV changed')
        cmd.extend(['--analysis-audio-probe',str(path.resolve())])
    subprocess.run(cmd,check=True)
    data=json.loads((output/'audio_probe_scores.json').read_text())
    report=json.loads((output/'analysis_report.json').read_text())
    if not report['non_mutation']['unchanged'] or data['model_forward_passes']!=0:
        raise RuntimeError('external probe purity gate failed')
    if data['audio_files'][0]['sampled_metrics']!=data['audio_files'][-1]['sampled_metrics']:
        raise RuntimeError('repeated external scoring differs')
    matrix_result=json.loads((matrix.HERE/'RESULTS.json').read_text())
    if data['probe_manifest']!=matrix_result['cells']['L16_WP_SP']['probes']:
        raise RuntimeError('null and model probe banks differ')
    result={'schema':1,'deterministic_repeat_exact':True,'same_probe_bank_as_matrix':True,
            'model_forward_passes':0,'backward_passes':0,'optimizer_steps':0,
            'input_non_mutation':True,'report_sha256':matrix.sha(output/'analysis_report.json'),
            'binary_sha256':matrix.sha(matrix.ROOT/'target/release/titan'),'command':cmd,'clips':{}}
    for path,item in zip(paths[:-1],data['audio_files'][:-1]):
        identity=receipt['files'][path]['identity'];rows=item['sampled_metrics']
        metrics={key:float(np.mean([np.mean([row[key] for row in rows if episode*64<=row['offset']-1<(episode+1)*64])
                                           for episode in range(6)]))
                 for key in ['development_mean_spectral','development_mean_chroma','validation_mean_spectral','validation_mean_chroma']}
        entry={'sha256':receipt['files'][path]['sha256'],'scores':metrics}
        if identity.endswith('_raw'):
            name={'parent_raw':'L16_WP_SP','child_raw':'L16_WC_SP','early_raw':'L16_WE_SP'}[identity]
            online=json.loads((matrix.RUNS/name/'ablations/full/summary.json').read_text())['sampled_steps']
            entry['pcm16_online_score_max_errors']={key:max(abs(row[key]-original['evaluation'][key]) for row,original in zip(rows,online))
                                                   for key in metrics}
        else:entry['matching_properties']=receipt['files'][path]
        result['clips'][identity]=entry
    (matrix.HERE/'NULL_RESULTS.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    for name,value in result['clips'].items():print(name,value['scores'])


if __name__=='__main__':main()
