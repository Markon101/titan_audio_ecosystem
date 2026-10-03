#!/usr/bin/env python3
"""Describe the corrected, still-anonymous selected pilot and shuffle controls."""
import json
import os
from pathlib import Path
import subprocess
import wave
import numpy as np
from freeze_intake import HERE, sha

RATE=24000
FFT=4096
EDGES=np.geomspace(20,12000,17)


def decode(path):
    env=os.environ.copy();env['LD_PRELOAD']='/data/data/com.termux/files/usr/lib/libc++_shared.so'
    result=subprocess.run(['ffmpeg','-v','error','-nostdin','-i',str(path),'-ar',str(RATE),'-ac','2',
                           '-f','f32le','pipe:1'],env=env,capture_output=True,check=True)
    values=np.frombuffer(result.stdout,'<f4').reshape(-1,2)
    if not len(values) or not np.isfinite(values).all():raise ValueError('empty/nonfinite audio')
    return values


def rms(x):return float(np.sqrt(np.mean(np.asarray(x,dtype=float)**2)))


def corr(a,b):
    a=np.asarray(a,dtype=float).copy();b=np.asarray(b,dtype=float).copy()
    a-=a.mean();b-=b.mean();denom=float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.dot(a,b)/denom) if denom>1e-12 else None


def native_peaks(path):
    with wave.open(str(path),'rb') as wav:
        if (wav.getnchannels(),wav.getframerate(),wav.getsampwidth())!=(2,48000,2):
            raise ValueError('pilot inputs/outputs expected stereo 48-kHz PCM16')
        values=np.frombuffer(wav.readframes(wav.getnframes()),'<i2').reshape(-1,2)
    rail=(values==32767)|(values==-32768)
    longest=0
    for channel in range(2):
        flags=rail[:,channel]
        if flags.any():
            switches=np.diff(np.concatenate(([0],flags.astype(np.int8),[0])))
            longest=max(longest,int(np.max(np.flatnonzero(switches==-1)-np.flatnonzero(switches==1))))
    return {'peak':max(float(values.max()),-float(values.min()))/32768,
            'exact_fullscale_sample_fraction':float(rail.mean()),
            'longest_fullscale_run_samples_per_channel':longest,
            'interpretation':'sample rails alone do not identify audible artifacts or limiter intent'}


def window_spectrum(mono):
    frames=np.lib.stride_tricks.sliding_window_view(mono,FFT)[::FFT//2]
    return np.mean(abs(np.fft.rfft(frames*np.hanning(FFT),axis=1))**2,axis=0)


def series(audio):
    hz=np.fft.rfftfreq(FFT,1/RATE)
    bands=np.zeros((len(hz),16))
    for index in range(16):bands[(hz>=EDGES[index])&(hz<EDGES[index+1]),index]=1
    chroma=np.zeros((len(hz),12));valid=(hz>=55)&(hz<=5000)
    notes=np.rint(69+12*np.log2(hz[valid]/440)).astype(int)%12
    chroma[np.flatnonzero(valid),notes]=1
    values=[];log_features=[];total_power=np.zeros(len(hz))
    for second in range(len(audio)//RATE):
        block=audio[second*RATE:(second+1)*RATE];left,right=block.T
        mid=(left+right)*.5;side=(left-right)*.5
        power=window_spectrum(mid);total_power+=power;total=max(1e-12,float(power.sum()))
        band=power@bands;band/=max(1e-12,float(band.sum()))
        tone=power@chroma;tone/=max(1e-12,float(tone.sum()))
        values.append([rms(mid),corr(left,right) or 0.,rms(side)/max(1e-12,rms(mid)),
                       float(np.log((rms(left)+1e-9)/(rms(right)+1e-9))),float(np.dot(hz,power)/total)])
        log_features.append(np.concatenate((np.log(np.maximum(band,1e-8)),tone)))
    return np.asarray(values),np.asarray(log_features),total_power,hz


def continuity(features,seed):
    z=(features-features.mean(0))/np.maximum(features.std(0),.05)
    rng=np.random.default_rng(seed);result={}
    for lag in (1,5,10,30):
        if len(z)-lag<8:continue
        observed=float(np.mean((z[lag:]-z[:-lag])**2));null=[]
        for _ in range(32):
            shuffled=z[rng.permutation(len(z))];null.append(float(np.mean((shuffled[lag:]-shuffled[:-lag])**2)))
        result[str(lag)]={'pairs':len(z)-lag,'observed_mean_squared_feature_change':observed,
            'shuffled_mean':float(np.mean(null)),'shuffled_q05':float(np.quantile(null,.05)),
            'shuffled_q95':float(np.quantile(null,.95)),
            'observed_to_shuffle_ratio':observed/max(1e-12,float(np.mean(null)))}
    return result


def describe(audio,seed):
    left,right=audio.T;mid=(left+right)*.5;side=(left-right)*.5
    time,features,power,hz=series(audio);total=max(1e-12,float(power.sum()))
    band_power={f'{low}_{high}':float(power[(hz>=low)&(hz<high)].sum()/total)
                for low,high in [(0,80),(80,250),(250,2000),(2000,8000),(8000,12000)]}
    probs=np.maximum(power[hz>=20],1e-20);flatness=float(np.exp(np.log(probs).mean())/probs.mean())
    envelope=np.sqrt(np.mean(mid[:len(mid)//2400*2400].reshape(-1,2400).astype(float)**2,axis=1))
    modulation=abs(np.fft.rfft(envelope-envelope.mean()))**2;mod_hz=np.fft.rfftfreq(len(envelope),.1)
    valid=(mod_hz>=.5)&(mod_hz<=4);peak_hz=float(mod_hz[valid][np.argmax(modulation[valid])])
    summary={'duration_seconds':len(audio)/RATE,'rms':rms(audio),'mid_rms':rms(mid),
        'peak_at_24k':float(abs(audio).max()),'crest_factor':float(abs(audio).max())/max(1e-12,rms(audio)),
        'stereo_correlation':corr(left,right),'side_mid_ratio':rms(side)/max(1e-12,rms(mid)),
        'spectral_centroid_hz':float(np.dot(hz,power)/total),'spectral_flatness_above_20_hz':flatness,
        'spectral_power_bands':band_power,'one_second_mid_rms_cv':float(time[:,0].std()/max(1e-12,time[:,0].mean())),
        'one_second_stereo_correlation_median':float(np.median(time[:,1])),
        'one_second_stereo_correlation_iqr':float(np.quantile(time[:,1],.75)-np.quantile(time[:,1],.25)),
        'median_per_second_stereo_correlation_change':float(np.median(abs(np.diff(time[:,1])))),
        'median_per_second_log_channel_balance_change':float(np.median(abs(np.diff(time[:,3])))),
        'dominant_envelope_modulation_0p5_to_4_hz':peak_hz,
        'quiet_one_second_fraction_below_minus50_dbfs':float(np.mean(time[:,0]<10**(-50/20))),
        'temporal_continuity_vs_window_shuffle':continuity(features,seed),
        'mean_normalized_logband':np.exp(features[:,:16]).mean(0).tolist(),
        'mean_chroma_power_distribution':features[:,16:].mean(0).tolist()}
    return summary,time,features


def js(a,b):
    a=np.maximum(np.asarray(a,dtype=float),1e-12).copy();b=np.maximum(np.asarray(b,dtype=float),1e-12).copy();a/=a.sum();b/=b.sum();m=(a+b)*.5
    return float(.5*np.sum(a*np.log(a/m))+.5*np.sum(b*np.log(b/m)))


def spectrogram(path,target,start,duration):
    env=os.environ.copy();env['LD_PRELOAD']='/data/data/com.termux/files/usr/lib/libc++_shared.so'
    filters=f'atrim=start={start}:duration={duration},asetpts=PTS-STARTPTS,aresample=24000,pan=mono|c0=0.5*c0+0.5*c1,showspectrumpic=s=1024x512:legend=1:scale=log:fscale=log:start=20:stop=12000:drange=80:limit=0:color=viridis'
    subprocess.run(['ffmpeg','-v','error','-nostdin','-n','-i',str(path),'-lavfi',filters,
                    '-frames:v','1','-threads','1',str(target)],env=env,check=True)


def main():
    result_path=HERE/'ANONYMOUS_RESULTS.json'
    directory=HERE/'runs/structure_v1'
    if directory.exists() or result_path.exists():raise FileExistsError('preserve completed/partial analysis')
    receipt=json.loads((HERE/'intake_receipt.json').read_text())
    for item in receipt['files']:
        if sha(Path(item['snapshot']))!=item['sha256']:raise RuntimeError('frozen pilot changed')
    snapshot=HERE/'runs/intake_snapshot'
    mapping=receipt['output_input_mapping']
    ingestion=json.loads((HERE/'runs/anonymous_ingestion/anonymous_output_analysis.json').read_text())
    shortest=min(item['duration_seconds'] for item in ingestion['outputs'])
    start,end=(20.,140.) if shortest>=140 else (10.,min(shortest-10.,130.))
    if end-start<30:raise ValueError('not enough common audio for a core comparison')
    directory.mkdir()
    result={'schema':1,'condition_identities_inspected':False,'declared_settings':receipt['settings'],
        'selection':receipt['selection'],'completed_blind_preference':receipt['completed_blind_preference'],
        'core_interval_seconds':[start,end],'descriptor_rate':RATE,'fft':FFT,'fft_hop':FFT//2,
        'feature_window_seconds':1,'shuffle_repetitions':32,'outputs':{},'inputs':{},
        'limitations':['one selected favorite per condition; no within-condition variance or independent batch replication',
                       'no prompt-only output; no causal separation from the Suno/prompt prior',
                       'within-track windows and shuffle repetitions are not generation replicates',
                       'temporal continuity describes the output and cannot establish source timing transfer',
                       'chroma/spectral descriptors do not rate beauty, preference, cleanliness, aliveness, or vocal interest']}
    for code in ('S01','S02','S03'):
        path=snapshot/'anonymous_inputs'/f'{code}.wav';audio=decode(path)
        summary,time,features=describe(audio,20261002)
        result['inputs'][code]={'sha256':sha(path),'summary':summary,'native_peaks':native_peaks(path)}
        np.savez_compressed(directory/f'{code}_input_features.npz',time=time,features=features)
    for filename,code in mapping.items():
        path=snapshot/'generated_wavs'/filename;audio=decode(path)
        full,time,features=describe(audio,20261002)
        core,core_time,core_features=describe(audio[int(start*RATE):int(end*RATE)],20261002)
        result['outputs'][code]={'observed_filename':filename,'corrected_input_id':code,'sha256':sha(path),
            'full':full,'core':core,'native_peaks':native_peaks(path)}
        np.savez_compressed(directory/f'{code}_output_features.npz',time=time,features=features,
                            core_time=core_time,core_features=core_features)
        target=directory/f'{code}_core_spectrogram.png';spectrogram(path,target,start,end-start)
        result['outputs'][code]['core_spectrogram']={'path':str(target),'sha256':sha(target),
            'display_range_dbfs':[-80,0],'mono_average':True,'frequency_hz':[20,12000]}
    result['aggregate_source_output_distances']={}
    for output,values in result['outputs'].items():
        result['aggregate_source_output_distances'][output]={}
        for source,input_item in result['inputs'].items():
            a,b=values['full'],input_item['summary']
            result['aggregate_source_output_distances'][output][source]={
                'normalized_logband_js':js(a['mean_normalized_logband'],b['mean_normalized_logband']),
                'chroma_js':js(a['mean_chroma_power_distribution'],b['mean_chroma_power_distribution'])}
    result['between_input_distances']={f'{a}__{b}':{
        'normalized_logband_js':js(result['inputs'][a]['summary']['mean_normalized_logband'],result['inputs'][b]['summary']['mean_normalized_logband']),
        'chroma_js':js(result['inputs'][a]['summary']['mean_chroma_power_distribution'],result['inputs'][b]['summary']['mean_chroma_power_distribution'])}
        for a,b in [('S01','S02'),('S01','S03'),('S02','S03')]}
    for item in receipt['files']:
        if sha(Path(item['snapshot']))!=item['sha256']:raise RuntimeError('snapshot mutated during analysis')
    result_path.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({'outputs':3,'core_seconds':[start,end],'condition_key_inspected':False}))


if __name__=='__main__':main()
