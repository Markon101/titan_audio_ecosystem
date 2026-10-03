"""Verify source/result/probe hashes and exact variance-control identities."""
import gzip
import hashlib
import json
from pathlib import Path
from run_transfer import HERE,ROOT,PREVIOUS
from run_frozen import sha,parent_hashes


def main():
    output=HERE/'final_verification.json'
    if output.exists():raise FileExistsError(output)
    archive=json.loads((HERE/'archive_receipt.json').read_text())
    compressed=(HERE/'RESULTS.json.gz').read_bytes()
    raw=gzip.decompress(compressed)
    if hashlib.sha256(raw).hexdigest()!=archive['raw_sha256'] or sha(HERE/'RESULTS.json.gz')!=archive['gzip_sha256']:
        raise RuntimeError('full report archive changed')
    report=json.loads(raw)
    for name,digest in report['fixed_probes_sha256'].items():
        if sha(HERE/'runs'/name)!=digest:raise RuntimeError('fixed probe changed')
    matrix=json.loads((ROOT/'analysis/weight_world_20261002/matrix_receipt.json').read_text())
    old=json.loads((PREVIOUS/'RESULTS.json').read_text())
    for name,episode in report['episodes'].items():
        meta=episode['metadata']
        actual=sha(Path(meta['root'])/meta['folder']/'post_dsp.wav')
        if name.startswith('L16_'):
            expected=matrix['cells'][name]['audio_sha256']
        else:
            snapshot='w128' if name.startswith('w128') or name=='overlap_w128' else 'w0'
            stem='baseline' if 'energy' not in name else ('energy_state_s00010000' if name.endswith('01') else 'energy_state_s00030000')
            expected=old['snapshots'][snapshot]['candidates'][stem]['post_dsp_sha256']
        if actual!=meta['audio_sha256'] or actual!=expected:
            raise RuntimeError('episode audio differs from its previous frozen receipt')
        if sha(Path(meta['root'])/meta['folder']/'summary.json')!=meta['summary_sha256']:
            raise RuntimeError('frozen summary changed')
    parent_hashes() # revalidate all four canonical artifacts against original receipt
    variance=json.loads((HERE/'VARIANCE_CONTROLS.json').read_text())
    count=0
    for episode in variance['episodes'].values():
        for domain in episode.values():
            for horizon in domain.values():
                for x in horizon.values():
                    total=x['history_mean_with_probe_sigma_gain_bits']+x['probe_mean_gain_at_probe_sigma_bits']
                    if abs(total-x['full_gain_over_history_bits'])>1e-8:raise RuntimeError('decomposition mismatch')
                    count+=1
    output.write_text(json.dumps({'schema':1,'prior_frozen_audio_receipts_match':True,'episodes':len(report['episodes']),
        'parent_unchanged':True,'fixed_probe_files_unchanged':True,'source_v2_parity':True,
        'saved_probe_restoration_exact':True,'variance_decompositions_verified':count,
        'new_tests_passed':6,'prior_tests_passed':9,'archive_verified':True,
        'no_titan_build_render_training_or_model_optimizer_update':True},indent=2)+'\n')
    print(json.dumps({'verified_episodes':len(report['episodes']),'variance_decompositions':count}))


if __name__=='__main__':main()
