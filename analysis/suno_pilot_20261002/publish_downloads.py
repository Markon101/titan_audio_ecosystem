#!/usr/bin/env python3
"""Publish a new readout and figures beside the user-provided pilot files."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DESTINATION = Path('/sdcard/Download/TITAN_Suno_Pilot_20261002')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    receipt_path = HERE / 'downloads_report_receipt.json'
    files = [(HERE / 'DOWNLOADS_SUMMARY.md', DESTINATION / 'ANALYSIS_20261003.md')]
    files += [(HERE / 'runs/structure_v1' / f'{sample}_core_spectrogram.png',
               DESTINATION / 'analysis_figures' / f'{sample}_core_spectrogram.png')
              for sample in ('S01', 'S02', 'S03')]
    if receipt_path.exists() or any(target.exists() for _, target in files):
        raise FileExistsError('preserve existing publication; inspect it before repeating')
    if not DESTINATION.is_dir():
        raise FileNotFoundError(DESTINATION)
    intake = json.loads((HERE / 'intake_receipt.json').read_text())
    for item in intake['files']:
        if sha(Path(item['source'])) != item['sha256']:
            raise RuntimeError('original intake changed before publication')
    # Read all publication sources before creating any outputs.
    payloads = [(source, target, source.read_bytes()) for source, target in files]
    (DESTINATION / 'analysis_figures').mkdir(exist_ok=True)
    published = []
    for source, target, payload in payloads:
        with target.open('xb') as output:
            output.write(payload)
        digest = hashlib.sha256(payload).hexdigest()
        if sha(target) != digest:
            raise RuntimeError('publication copy mismatch')
        published.append({'source': str(source), 'published': str(target),
                          'bytes': len(payload), 'sha256': digest})
    for item in intake['files']:
        if sha(Path(item['source'])) != item['sha256']:
            raise RuntimeError('original intake changed during publication')
    receipt = {'schema': 1, 'originals_unchanged': True, 'files': published}
    with receipt_path.open('x') as output:
        output.write(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'report': str(files[0][1]), 'figures': 3, 'originals_unchanged': True}))


if __name__ == '__main__':
    main()
