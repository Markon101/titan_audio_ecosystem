#!/usr/bin/env python3
"""Matched external waveform controls for checking whether probes reward noise."""
import json
import shutil
import wave
import numpy as np
import run_matrix as matrix


def read(path):
    with wave.open(str(path),'rb') as w:
        assert (w.getnchannels(),w.getframerate(),w.getsampwidth())==(2,48000,2)
        x=np.frombuffer(w.readframes(w.getnframes()),'<i2').copy()
    return x.reshape(-1,2).astype(float)/32768


def square_root(cov,inverse=False):
    values,vectors=np.linalg.eigh(cov)
    scales=1/np.sqrt(np.maximum(values,1e-12)) if inverse else np.sqrt(np.maximum(values,0))
    return (vectors*scales)@vectors.T


def write(path,data):
    peak=float(abs(data).max());gain=min(1,.95/max(1e-12,peak))
    with wave.open(str(path),'wb') as w:
        w.setnchannels(2);w.setframerate(48000);w.setsampwidth(2)
        w.writeframes(np.round(np.clip(data*gain,-1,1)*32767).astype('<i2').tobytes())
    return gain


def main():
    directory=matrix.RUNS/'audio_nulls'
    if directory.exists():raise FileExistsError('preserve prior null waveforms')
    directory.mkdir()
    result={'schema':1,'scope':'external measurement controls, never injected into Titan', 'files':{}}
    rng=np.random.default_rng(20261002)
    for label,cell in [('parent','L16_WP_SP'),('child','L16_WC_SP'),('early','L16_WE_SP')]:
        source=matrix.RUNS/cell/'ablations/full/raw_renderer.wav';raw=read(source)
        destination=directory/f'{label}_raw.wav';shutil.copyfile(source,destination)
        result['files'][str(destination)]={'identity':f'{label}_raw','sha256':matrix.sha(destination)}
        if label=='child':continue
        centered=raw-raw.mean(0);target_cov=centered.T@centered/len(raw)
        z=rng.normal(size=raw.shape);z-=z.mean(0);z=z@square_root(z.T@z/len(z),inverse=True)@square_root(target_cov)+raw.mean(0)
        spectrum=np.fft.rfft(raw,axis=0);phase=rng.uniform(-np.pi,np.pi,len(spectrum));phase[0]=phase[-1]=0
        randomized=spectrum*np.exp(1j*phase)[:,None]
        surrogate=np.fft.irfft(randomized,n=len(raw),axis=0)
        for kind,data in [('covariance_gaussian',z),('shared_phase',surrogate)]:
            path=directory/f'{label}_{kind}.wav';gain=write(path,data);decoded=read(path)
            result['files'][str(path)]={'identity':f'{label}_{kind}','sha256':matrix.sha(path),
                'peak_safety_gain':gain,'rms_ratio_to_source':float(np.sqrt(np.mean(decoded**2)/np.mean(raw**2))),
                'stereo_correlation':float(np.corrcoef(decoded.T)[0,1]),
                'source_stereo_correlation':float(np.corrcoef(raw.T)[0,1]),
                'preencoding_covariance_max_error':float(abs((data-data.mean(0)).T@(data-data.mean(0))/len(data)-target_cov).max())}
    (matrix.HERE/'audio_null_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({item['identity']:item.get('rms_ratio_to_source') for item in result['files'].values()}))


if __name__=='__main__':main()
