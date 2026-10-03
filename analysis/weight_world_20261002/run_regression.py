#!/usr/bin/env python3
"""Check disabled-binary parity and measurement-only parity on exact clones."""
import argparse
import json
from pathlib import Path
import run_matrix as matrix


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suffix',default='')
    args=parser.parse_args()
    if args.suffix and not args.suffix.isalnum():raise ValueError('invalid regression suffix')
    suffix='_'+args.suffix if args.suffix else ''
    matrix.prepare()
    records={}
    for label,old in [('legacy_flags_old',True),('legacy_flags_new',False)]:
        label+=suffix
        output=matrix.RUNS/label
        cmd=matrix.command(label,matrix.P,matrix.P,16,16,output,metrics=False,common=False)
        cmd[cmd.index('--analysis-target-schedule')+1]=str(matrix.ROOT/'analysis/matched_exposure_20261001/schedule_20261002.json')
        if old:cmd[0]=str(matrix.RUNS/'prior_titan')
        records[label]=matrix.run(cmd,label,16)
    off='metrics_off_standardized'+suffix
    output=matrix.RUNS/off
    cmd=matrix.command(off,matrix.P,matrix.P,16,16,output,metrics=False)
    records[off]=matrix.run(cmd,off,16)
    pairs=[('legacy_flags_old'+suffix,'legacy_flags_new'+suffix),(off,'smoke16')]
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
    receipt_path=matrix.HERE/f'regression{suffix}_receipt.json'
    previous=json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    result['binary_sha256_by_run']=previous.get('binary_sha256_by_run',{
        name:value.get('resource',{}).get('binary_sha256') for name,value in records.items()})
    receipt_path.write_text(json.dumps(result,indent=2)+'\n')
    print('disabled path and measurement-only raw/post/world parity: exact')


if __name__=='__main__':main()
