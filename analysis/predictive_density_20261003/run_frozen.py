"""Sequential bounded inference, restart-safe; never write a parent checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = ROOT / 'analysis/metastable_20261001/frozen_step_58107'
TAG = 'v10-msfield-fresh-20260930-02'
RUNS = HERE / 'runs'
CORPUS = ROOT / 'analysis/matched_exposure_20261001/runs/corpus'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def memory():
    return {key: int(rest.split()[0]) / 1024 for line in Path('/proc/meminfo').read_text().splitlines()
            for key, rest in [line.split(':', 1)] if key in ('MemAvailable', 'SwapFree')}


def parent_hashes():
    receipt = json.loads((PARENT.parent / 'frozen_step_58107_receipt.json').read_text())
    result = {}
    for item in receipt['files']:
        actual = sha(PARENT / item['name'])
        if actual != item['sha256']:
            raise RuntimeError('archived parent identity changed')
        result[item['name']] = actual
    return result


def prepare():
    metric = json.loads((HERE / 'metric_validation.json').read_text())
    if not metric['all_trivial_noise_rewards_zero']:
        raise RuntimeError('predictive metric has not passed falsification gates')
    RUNS.mkdir(exist_ok=True)
    binary = RUNS / 'frozen_titan'
    if not binary.exists():
        shutil.copyfile(ROOT / 'target/release/titan', binary)
        binary.chmod(0o500)
    schedule = HERE / 'schedule.json'
    if not schedule.exists():
        previous = json.loads((ROOT / 'analysis/weight_world_20261002/schedule.json').read_text())
        episodes = []
        for index in range(20):
            episode = dict(previous['episodes'][index % len(previous['episodes'])])
            episode['start_step'] = index * 64
            episodes.append(episode)
        schedule.write_text(json.dumps({'schema': 1, 'seed': 20261003,
            'slots': previous['slots'], 'episodes': episodes}, indent=2) + '\n')
    return binary, parent_hashes()


def command(binary, output, warmup=0, measurement=True, perturb=False):
    cmd = [str(binary), '--analysis-only', '--analysis-substrate', 'msfield',
           '--base-dir', str(PARENT), '--run-tag', TAG,
           '--corpus-dir', str(CORPUS / 'a'), '--corpus-manifest', str(CORPUS / 'a_manifest.json'),
           '--analysis-target-schedule', str(HERE / 'schedule.json'),
           '--analysis-seed', '20261003', '--analysis-common-rng', '--analysis-target-origin', '0',
           '--analysis-active-depth', '16', '--analysis-warmup', str(warmup),
           '--analysis-stride', '8', '--analysis-dir', str(output),
           '--analysis-terminal', 'quiet', '--threads', '2']
    if measurement:
        cmd += ['--analysis-evaluation-metrics']
    if perturb:
        cmd += ['--perturbation-analysis', '1024', '--perturbation', 'energy_state',
                '--perturbation-scales', '0,0.01,0.03']
    else:
        cmd += ['--dynamics-ablation', '64', '--ablation', 'none']
    return cmd


def check(output, cmd, perturb):
    report = json.loads((output / 'analysis_report.json').read_text())
    provenance = json.loads((output / 'provenance.json').read_text())
    if provenance['analysis_configuration']['invocation'] != cmd:
        raise RuntimeError('completed arm configuration differs; preserve it')
    if (output / 'INCOMPLETE').exists() or not report['non_mutation']['unchanged'] or report['analysis']['optimizer_steps']:
        raise RuntimeError('non-mutation/optimizer gate failed')
    root = output / ('perturbations' if perturb else 'ablations')
    # All candidate summaries certify fixed weights, bounds, and no backward/optimizer.
    summaries = {p.parent.name: json.loads(p.read_text()) for p in root.glob('*/summary.json')}
    for name, summary in summaries.items():
        if not summary['fixed_weights'] or summary['optimizer_steps'] != 0:
            raise RuntimeError(f'candidate parameters updated: {name}')
        if summary.get('weights_hash_before') != summary.get('weights_hash_after'):
            raise RuntimeError('parameter hash mismatch')
        if summary['horizons'][-1]['nonfinite_detected'] or not summary['horizons'][-1]['bounded']:
            raise RuntimeError('candidate nonfinite or unbounded')
    baseline = 'full'
    noop = 'energy_state_s00000000' if perturb else 'none'
    # Discover baseline export stem deliberately rather than inferring condition aliases.
    if perturb:
        candidates = [name for name, summary in summaries.items() if summary['condition'] == 'perturbation_baseline']
        if len(candidates) != 1:
            raise RuntimeError(f'missing perturbation baseline: {list(summaries)}')
        baseline = candidates[0]
    for filename in ('post_dsp.wav', 'raw_renderer.wav'):
        if sha(root / baseline / filename) != sha(root / noop / filename):
            raise RuntimeError('zero perturbation/no-op audio differs')
    if summaries[baseline]['final_world_fingerprint'] != summaries[noop]['final_world_fingerprint']:
        raise RuntimeError('zero perturbation/no-op full world differs')
    return {'report_sha256': sha(output / 'analysis_report.json'), 'no_op_exact': True,
            'optimizer_steps': 0, 'baseline_export': baseline,
            'baseline_post_sha256': sha(root / baseline / 'post_dsp.wav'),
            'baseline_raw_sha256': sha(root / baseline / 'raw_renderer.wav'),
            'world_fingerprint': summaries[baseline]['final_world_fingerprint'],
            'candidate_exports': sorted(summaries)}


def run(binary, name, warmup, measurement, perturb, hashes):
    output = RUNS / name
    cmd = command(binary, output, warmup, measurement, perturb)
    receipt = HERE / f'{name}_receipt.json'
    if (output / 'analysis_report.json').exists():
        checked = check(output, cmd, perturb)
        if not receipt.exists():
            raise RuntimeError('finished output lacks resource/binary receipt; audit before adoption')
        existing = json.loads(receipt.read_text())
        if existing['binary_sha256'] != sha(binary) or existing['checks'] != checked:
            raise RuntimeError('completed provenance mismatch')
        return existing
    if output.exists() or receipt.exists() or (RUNS / f'{name}.log').exists():
        raise FileExistsError(f'preserve partial arm: {name}')
    processes = subprocess.run(['ps', '-ef'], capture_output=True, text=True, check=True).stdout
    if any(' --analysis-only ' in line or ('/titan ' in line and ' --' in line)
           for line in processes.splitlines() if 'ps -ef' not in line):
        raise RuntimeError('another Titan process is active; do not duplicate or compete')
    initial = memory()
    if initial['MemAvailable'] < 1500 or shutil.disk_usage(RUNS).free < 2 * 1024 ** 3:
        raise RuntimeError('resource gate: require 1.5 GiB available RAM and 2 GiB storage')
    start = time.monotonic(); minimum = initial['MemAvailable']; peak = 0; stopped = False
    with (RUNS / f'{name}.log').open('xb') as log:
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while process.poll() is None:
            current = memory(); minimum = min(minimum, current['MemAvailable'])
            try:
                status = Path(f'/proc/{process.pid}/status').read_text().splitlines()
                peak = max(peak, next((int(s.split()[1])/1024 for s in status if s.startswith('VmRSS:')), 0))
            except FileNotFoundError:
                pass
            if current['MemAvailable'] < 768 and not stopped:
                stopped = True; os.killpg(process.pid, signal.SIGTERM)
            time.sleep(.5)
    resource = {'initial': initial, 'minimum_available_mib': minimum, 'peak_rss_mib': peak,
                'elapsed_seconds': time.monotonic() - start, 'exit_code': process.returncode,
                'safety_stop': stopped}
    if process.returncode or stopped:
        (HERE / f'{name}_failed_resource.json').write_text(json.dumps(resource, indent=2) + '\n')
        raise RuntimeError('preserved interrupted arm; see log')
    if parent_hashes() != hashes:
        raise RuntimeError('canonical parent changed')
    value = {'schema': 1, 'command': cmd, 'binary_sha256': sha(binary), 'parent_sha256': hashes,
             'schedule_sha256': sha(HERE / 'schedule.json'), 'resource': resource,
             'checks': check(output, cmd, perturb)}
    receipt.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps({'arm': name, 'checks': value['checks'], 'resource': resource}), flush=True)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parity', action='store_true')
    parser.add_argument('--snapshot', choices=['w0', 'w128'])
    args = parser.parse_args()
    if not args.parity and not args.snapshot:
        parser.error('choose --parity or --snapshot')
    binary, hashes = prepare()
    if args.parity:
        a = run(binary, 'parity_a', 0, False, False, hashes)
        b = run(binary, 'parity_b', 0, True, False, hashes)
        if a['checks'] != b['checks']:
            # Report hashes differ because the sidecars differ; compare evolution only.
            for key in ('baseline_post_sha256', 'baseline_raw_sha256', 'world_fingerprint', 'no_op_exact'):
                if a['checks'][key] != b['checks'][key]:
                    raise RuntimeError('measurement-only changed evolution')
        (HERE / 'parity_receipt.json').write_text(json.dumps({'schema': 1, 'exact_audio_world_parity': True,
            'measurement_elapsed_ratio': b['resource']['elapsed_seconds']/a['resource']['elapsed_seconds']}, indent=2)+'\n')
    else:
        run(binary, args.snapshot, 0 if args.snapshot == 'w0' else 128, True, True, hashes)


if __name__ == '__main__':
    main()
