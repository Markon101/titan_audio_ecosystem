#!/usr/bin/env python3
"""Check disabled-binary parity and measurement-only parity on exact clones."""
import json
from pathlib import Path
import run_matrix as matrix


def main():
    matrix.prepare()
    records={}
    for label,old in [('legacy_flags_old',True),('legacy_flags_new',False)]:
        output=matrix.RUNS/label
        cmd=matrix.command(label,matrix.P,matrix.P,16,16,output,metrics=False,common=False)
        cmd[cmd.index('--analysis-target-schedule')+1]=str(matrix.ROOT/'analysis/matched_exposure_20261001/schedule_20261002.json')
        if old:cmd[0]=str(matrix.RUNS/'prior_titan')
        records[label]=matrix.run(cmd,label,16)
    output=matrix.RUNS/'metrics_off_standardized'
    cmd=matrix.command('metrics_off_standardized',matrix.P,matrix.P,16,16,output,metrics=False)
    records['metrics_off_standardized']=matrix.run(cmd,'metrics_off_standardized',16)
    pairs=[('legacy_flags_old','legacy_flags_new'),('metrics_off_standardized','smoke16')]
    for a,b in pairs:
        for name in ('post_dsp.wav','raw_renderer.wav'):
            if matrix.sha(matrix.RUNS/a/'ablations/full'/name)!=matrix.sha(matrix.RUNS/b/'ablations/full'/name):
                raise RuntimeError(f'audio parity failed {a}/{b}/{name}')
        aa=json.loads((matrix.RUNS/a/'ablations/full/summary.json').read_text())
        bb=json.loads((matrix.RUNS/b/'ablations/full/summary.json').read_text())
        if aa['final_world_fingerprint']!=bb['final_world_fingerprint']:
            raise RuntimeError(f'world parity failed {a}/{b}')
    result={'schema':1,'disabled_binary_raw_post_and_world_exact':True,
            'measurement_enabled_raw_post_and_world_exact':True,'runs':records}
    (matrix.HERE/'regression_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print('disabled path and measurement-only raw/post/world parity: exact')


if __name__=='__main__':main()
