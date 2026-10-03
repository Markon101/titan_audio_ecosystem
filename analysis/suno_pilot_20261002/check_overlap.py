#!/usr/bin/env python3
"""Check selected WAVs for shared audio before releasing condition identities."""
import json
import os
import subprocess
import numpy as np
from freeze_intake import HERE, sha

RATE=3000


def decode(path):
    env=os.environ.copy();env['LD_PRELOAD']='/data/data/com.termux/files/usr/lib/libc++_shared.so'
    data=subprocess.run(['ffmpeg','-v','error','-nostdin','-i',str(path),'-ar',str(RATE),'-ac','1',
                         '-f','f32le','pipe:1'],env=env,capture_output=True,check=True).stdout
    return np.frombuffer(data,'<f4').astype(float)


def match(source,target,anchor):
    template=source[int(anchor*RATE):int((anchor+4)*RATE)].copy();template-=template.mean()
    length=len(template);n=1<<(len(target)+length-1).bit_length()
    product=np.fft.irfft(np.fft.rfft(target,n)*np.conj(np.fft.rfft(template,n)),n)[:len(target)-length+1]
    sums=np.concatenate(([0.],np.cumsum(target)));squares=np.concatenate(([0.],np.cumsum(target*target)))
    window_sum=sums[length:]-sums[:-length]
    variance=np.maximum(0,squares[length:]-squares[:-length]-window_sum**2/length)
    denom=np.sqrt(np.sum(template*template)*variance)
    valid=variance/length>1e-6
    score=np.zeros_like(product);score[valid]=product[valid]/np.maximum(denom[valid],1e-12)
    index=int(np.argmax(abs(score)))
    return {'anchor_seconds':anchor,'match_seconds':index/RATE,
            'normalized_correlation':float(score[index]),'absolute_correlation':float(abs(score[index])),
            'match_offset_seconds':index/RATE-anchor}


def main():
    receipt=json.loads((HERE/'intake_receipt.json').read_text());snapshot=HERE/'runs/intake_snapshot/generated_wavs'
    audio={code:decode(snapshot/name) for name,code in receipt['output_input_mapping'].items()}
    result={'schema':1,'condition_identities_inspected':False,'rate':RATE,'mono':True,
            'anchor_seconds':[20,60,100,150],'template_seconds':4,'pairs':{},'self_controls':{},
            'interpretation':'search maxima diagnose audio overlap only; repeated musical patterns may match without duplicate files'}
    for code,value in audio.items():
        control=match(value,value,20)
        if control['absolute_correlation']<.999999:raise RuntimeError('self-match control failed')
        result['self_controls'][code]=control
    for a,b in [('S01','S02'),('S01','S03'),('S02','S03')]:
        result['pairs'][f'{a}__{b}']=[match(audio[a],audio[b],anchor) for anchor in result['anchor_seconds']]
    (HERE/'OVERLAP_CHECK.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,values in result['pairs'].items():
        print(name,[(round(row['absolute_correlation'],4),round(row['match_offset_seconds'],3)) for row in values])


if __name__=='__main__':main()
