#!/usr/bin/env python3
"""Freeze user pilot files and the stated label correction without the key."""
import hashlib
import json
import shutil
from pathlib import Path

HERE=Path(__file__).resolve().parent
SOURCE=Path('/sdcard/Download/TITAN_Suno_Pilot_20261002')
MAPPING={'Machinery Ethereal S03.wav':'S01','Máquina Ética - S02.wav':'S02','Máquina Ética S03.wav':'S03'}


def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(4*1024*1024),b''):value.update(block)
    return value.hexdigest()


def main():
    destination=HERE/'runs/intake_snapshot'
    if destination.exists():raise FileExistsError('preserve the existing frozen snapshot')
    public=json.loads((SOURCE/'input_receipt.json').read_text())
    for item in public['inputs'].values():
        if sha(SOURCE/item['file'])!=item['sha256']:raise RuntimeError('anonymous input changed')
    destination.mkdir(parents=True)
    for name in ('anonymous_inputs','generated_wavs','notes'):(destination/name).mkdir()
    files=[SOURCE/'input_receipt.json',SOURCE/'README.md',SOURCE/'suggested_upload_order.csv']
    files+=sorted((SOURCE/'anonymous_inputs').glob('*.wav'))
    files+=sorted((SOURCE/'generated_wavs').glob('*.wav'))
    files+=sorted((SOURCE/'notes').glob('*.md'))
    if {path.name for path in (SOURCE/'generated_wavs').glob('*.wav')}!=set(MAPPING):
        raise RuntimeError('output inventory differs from the corrected three-file pilot')
    receipt={'schema':1,'key_opened':False,'files':[], 'output_input_mapping':MAPPING,
        'correction_source':'user explicitly corrected Machinery Ethereal S03.wav to S01',
        'completed_blind_preference':{'preferred_input_ids':['S03','S01'],'ordering_between_preferred_ids':None,
            'numerical_dimension_ratings':None},
        'selection':'one selected favorite from each two-track input batch; discarded siblings unavailable',
        'declared_generation_batches':3,'declared_candidate_outputs':6,'provided_outputs':3,
        'unprovided_sibling_outputs':3,'actual_attempt_order':None,'generation_seeds_exposed':False,
        'prompt_only_provided':False,
        'settings':{'date':'2026-10-02','model':'Suno V6','custom_model':'V6 plain','mode':'cover',
            'prompt':'Rich idm electronica mixed with beautiful synths, ethereal vibes, grandiose cinematics, large soundstage',
            'audio_strength_percent':25,'style_strength_percent':50,'weirdness_percent':50,
            'lyrics_field':'empty','instrumental_requested':True,'prompt_rewrite':False,'input_trim':'full'}}
    for path in files:
        before=sha(path);target=destination/path.relative_to(SOURCE)
        shutil.copyfile(path,target)
        if sha(path)!=before or sha(target)!=before:raise RuntimeError('file changed during freezing')
        target.chmod(0o444)
        receipt['files'].append({'source':str(path),'snapshot':str(target),'bytes':path.stat().st_size,'sha256':before})
    (HERE/'intake_receipt.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'snapshot':str(destination),'outputs':3,'condition_key_opened':False}))


if __name__=='__main__':main()
