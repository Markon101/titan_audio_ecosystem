#!/usr/bin/env python3
"""Hash and describe anonymous manual Suno outputs without opening the key."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np

HERE=Path(__file__).resolve().parent
DEFAULT=Path('/sdcard/Download/TITAN_Suno_Pilot_20261002')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--intake',type=Path,default=DEFAULT)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError('preserve previous analysis; use a new output path')
    inputs=json.loads((args.intake/'input_receipt.json').read_text())
    for entry in inputs['inputs'].values():
        if sha(args.intake/entry['file'])!=entry['sha256']:raise RuntimeError('anonymous input changed')
    wavs=sorted((args.intake/'generated_wavs').glob('*.wav'))
    if not wavs:
        print(json.dumps({'status':'awaiting_manual_outputs','directory':str(args.intake/'generated_wavs')}));return
    env=os.environ.copy();env['LD_PRELOAD']='/data/data/com.termux/files/usr/lib/libc++_shared.so'
    result={'schema':1,'unblinded':False,'input_receipt_sha256':sha(args.intake/'input_receipt.json'),
            'notes':[{'path':str(path),'sha256':sha(path)} for path in sorted((args.intake/'notes').glob('*.md'))],
            'outputs':[],'limitations':['acoustic descriptors do not rate preference or musical usefulness',
                'generations from one prompt/model are not independent trained-model replicates']}
    for path in wavs:
        identity=sha(path)
        probe=subprocess.run(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)],env=env,capture_output=True,check=True)
        metadata=json.loads(probe.stdout)
        decoded=subprocess.run(['ffmpeg','-v','error','-i',str(path),'-map','0:a:0','-ar','24000','-ac','2','-f','f32le','pipe:1'],
                               env=env,capture_output=True,check=True)
        values=np.frombuffer(decoded.stdout,dtype='<f4').reshape(-1,2)
        if not len(values) or not np.isfinite(values).all():raise RuntimeError(f'nonfinite/empty output: {path}')
        left,right=values.T;mid=(left+right)*.5;side=(left-right)*.5
        rms=lambda x:float(np.sqrt(np.mean(x.astype(np.float64)**2)))
        width=2048;power=np.zeros(width//2+1);count=0;window=np.hanning(width)
        for start in range(0,len(mid)-width+1,1024):
            power+=abs(np.fft.rfft(mid[start:start+width]*window))**2;count+=1
        hz=np.fft.rfftfreq(width,1/24000);total=max(1e-12,float(power.sum()))
        envelope=np.sqrt(np.mean(mid[:len(mid)//2400*2400].reshape(-1,2400).astype(float)**2,axis=1))
        modulation=abs(np.fft.rfft(envelope-envelope.mean()))**2
        mod_hz=np.fft.rfftfreq(len(envelope),.1);mod_total=max(1e-12,float(modulation[mod_hz>=.1].sum()))
        if sha(path)!=identity:raise RuntimeError('output changed during ingestion')
        result['outputs'].append({'file':str(path),'sha256':identity,'original_metadata':metadata,
            'descriptor_sample_rate':24000,'frames':len(values),'duration_seconds':len(values)/24000,
            'rms':rms(values),'peak':float(abs(values).max()),'near_fullscale_fraction':float(np.mean(abs(values)>=.999)),
            'stereo_correlation':float(np.corrcoef(left,right)[0,1]) if left.std()>1e-9 and right.std()>1e-9 else None,
            'side_mid_ratio':rms(side)/max(1e-12,rms(mid)),
            'spectral_centroid_hz':float(np.dot(hz,power)/total),
            'spectral_power_bands':{f'{lo}_{hi}':float(power[(hz>=lo)&(hz<hi)].sum()/total)
                for lo,hi in [(0,80),(80,250),(250,2000),(2000,8000),(8000,12000)]},
            'envelope_modulation_power':{f'{lo}_{hi}':float(modulation[(mod_hz>=lo)&(mod_hz<hi)].sum()/mod_total)
                for lo,hi in [(.1,1),(1,4),(4,10)]}})
    args.out.mkdir(parents=True)
    (args.out/'anonymous_output_analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'outputs':len(wavs),'unblinded':False,'analysis':str(args.out)}))


if __name__=='__main__':main()
