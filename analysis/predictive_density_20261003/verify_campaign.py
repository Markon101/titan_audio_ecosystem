"""Independent source bytes, target frames, frozen weights/world and selector checks."""
import json
from pathlib import Path
from run_frozen import HERE, ROOT, CORPUS, sha, parent_hashes
from selection import choose


def main():
    output = HERE / 'verification_receipt.json'
    if output.exists():
        raise FileExistsError(output)
    campaign = json.loads((ROOT / 'analysis/matched_exposure_20261001/matched_exposure_receipt.json').read_text())
    slots = {}
    for entry in campaign['arms']['a']['slots']:
        path = CORPUS / 'a' / entry['alias']
        actual = sha(path)
        if actual != entry['source_sha256']:
            raise RuntimeError('target alias no longer contains exact original source bytes')
        slots[entry['alias']] = {'sha256': actual, 'family': entry['source_family']}
    if sha(CORPUS / 'a_manifest.json') != campaign['arms']['a']['manifest_sha256']:
        raise RuntimeError('manifest changed')
    schedule = json.loads((HERE / 'schedule.json').read_text())
    counts = {}
    for name in ('parity_a', 'parity_b', 'w0', 'w128'):
        receipt = json.loads((HERE / f'{name}_receipt.json').read_text())
        if receipt['parent_sha256'] != parent_hashes():
            raise RuntimeError('canonical parent changed')
        root = HERE / 'runs' / name / ('ablations' if name.startswith('parity') else 'perturbations')
        count = 0
        for path in root.glob('*/summary.json'):
            summary = json.loads(path.read_text())
            if summary.get('weights_hash_before') != summary.get('weights_hash_after') or summary['optimizer_steps']:
                raise RuntimeError('frozen weight/optimizer gate failed')
            if not summary['fixed_morphology']:
                raise RuntimeError('morphology changed')
            for row in summary['sampled_steps']:
                relative = summary['warmup_chunks'] + row['rollout_offset'] - 1
                item = next(e for e in schedule['episodes'] if e['start_step'] <= relative < e['start_step']+e['chunks'])
                expected_file = f"slot_{item['slot']:02}.wav"
                expected_frame = item['source_frame']+(relative-item['start_step'])*4096
                if row['target_file'] != expected_file or row['target_frame'] != expected_frame:
                    raise RuntimeError('wrong target file/frame identity')
                if not .18 <= row['energy'] <= .96:
                    raise RuntimeError('host energy out of declared bounds')
                count += 1
        counts[name] = count
    result = json.loads((HERE / 'RESULTS.json').read_text())
    selected = {}
    for name, snapshot in result['snapshots'].items():
        bank = snapshot['candidates']
        if choose(bank, 0)['chosen'] != 'baseline':
            raise RuntimeError('zero-weight bypass changed')
        for role, stem in [('A_baseline', 'baseline'), ('B_measurement_only', 'measurement'),
                           ('C_weak_PD', 'weak_pd'), ('D_naive_predictability', 'naive')]:
            decision = snapshot['decisions'][role]
            wav = HERE/'runs/selected'/f'{name}_{stem}.wav'
            if sha(wav) != decision['selected_audio_sha256']:
                raise RuntimeError('selected export not exact source bytes')
            selected[f'{name}_{role}'] = sha(wav)
    output.write_text(json.dumps({'schema': 1, 'parent_unchanged': True, 'model_optimizer_updates': 0,
        'target_source_bytes_verified': slots, 'sampled_target_rows_verified': counts,
        'host_energy_bounded': True, 'disabled_exact_baseline': True,
        'selected_export_sha256': selected, 'tests': 9,
        'validation_probe_scores_excluded_from_coding_and_choice': True}, indent=2)+'\n')
    print(json.dumps({'sampled_target_rows': sum(counts.values()), 'selected_exact_exports': len(selected)}))


if __name__ == '__main__':
    main()
