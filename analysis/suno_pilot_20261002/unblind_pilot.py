#!/usr/bin/env python3
"""Release identities only after frozen preference, descriptors, and overlap checks."""
import copy
import json
from pathlib import Path
from freeze_intake import HERE, sha


def main():
    root=HERE.parents[1]
    key_path=root/'analysis/weight_world_20261002/runs/suno_intake_private/answer_key.json'
    source_receipt=json.loads((root/'analysis/weight_world_20261002/suno_intake_receipt.json').read_text())
    anonymous_path=HERE/'ANONYMOUS_RESULTS.json';overlap_path=HERE/'OVERLAP_CHECK.json'
    if (HERE/'RESULTS.json').exists() or (HERE/'UNBLINDING_RECEIPT.json').exists():
        raise FileExistsError('preserve completed identity release')
    anonymous=json.loads(anonymous_path.read_text());overlap=json.loads(overlap_path.read_text())
    if anonymous['condition_identities_inspected'] or overlap['condition_identities_inspected']:
        raise RuntimeError('anonymous analysis identity flag is inconsistent')
    intake=json.loads((HERE/'intake_receipt.json').read_text())
    if intake['output_input_mapping']['Machinery Ethereal S03.wav']!='S01':
        raise RuntimeError('user correction not applied')
    if sha(key_path)!=source_receipt['private_answer_key_sha256']:
        raise RuntimeError('private key changed')
    key=json.loads(key_path.read_text())
    for code,item in key['conditions'].items():
        if anonymous['inputs'][code]['sha256']!=item['sha256']:
            raise RuntimeError('key does not match actual uploaded input')
    result=copy.deepcopy(anonymous);result['condition_identities_inspected']=True
    preferred=set(intake['completed_blind_preference']['preferred_input_ids'])
    mapping={code:item['identity'] for code,item in key['conditions'].items()}
    for code,item in result['inputs'].items():item['condition']=mapping[code]
    for code,item in result['outputs'].items():
        item['condition']=mapping[code]
        item['named_in_user_preferred_pair']=code in preferred
    result['overlap_check_sha256']=sha(overlap_path)
    receipt={'schema':1,'condition_identities_released_after_anonymous_descriptors':True,
        'completed_qualitative_blind_assessment':intake['completed_blind_preference'],
        'intake_receipt_sha256':sha(HERE/'intake_receipt.json'),
        'anonymous_results_sha256':sha(anonymous_path),'overlap_check_sha256':sha(overlap_path),
        'private_answer_key_sha256':sha(key_path),'input_condition_mapping':mapping,
        'input_hashes_verified':True,
        'note':'qualitative assessment complete per user ready message and frozen note; no numerical ratings were fabricated'}
    (HERE/'UNBLINDING_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (HERE/'RESULTS.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({'mapping':mapping,'preferred_pair':sorted(preferred)}))


if __name__=='__main__':main()
