"""Additional noise/ordering challenges; never silently tune after seeing them."""
import json
import argparse
from pathlib import Path
import numpy as np
from audio import CHUNK, RATE, describe
from predictive import measure, reward, diagnostics

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hardened', action='store_true')
    args = parser.parse_args()
    path = HERE / ('adversarial_controls_v2.json' if args.hardened else 'adversarial_controls.json')
    if path.exists():
        raise FileExistsError(path)
    rng = np.random.default_rng(101)
    result = {'schema': 1, 'controls': {}, 'metric_was_not_retuned': True}
    n = 1024 * CHUNK
    for name in ('noise_with_chirped_amplitude', 'random_walk_amplitude_noise',
                 'single_chirped_oscillator'):
        t = np.arange(n) / RATE
        if name == 'random_walk_amplitude_noise':
            walk = np.cumsum(rng.normal(size=1024))
            walk = .06 + .09 * (walk-walk.min()) / np.ptp(walk)
            envelope = np.interp(np.arange(n)/CHUNK, np.arange(1024), walk)
        else:
            envelope = .08 + .05 * np.sin(2*np.pi*(.035*t + .0007*t*t))
        if name == 'single_chirped_oscillator':
            audio = np.repeat((envelope*np.sin(2*np.pi*(375*t+.05*t*t)))[:, None], 2, axis=1)
        else:
            audio = envelope[:, None] * rng.normal(size=(n, 2))
        y, physical = describe(audio)
        report = measure(y, affine_controls=args.hardened)
        allowed = reward(report)['value'] if physical['eligible'] else 0
        result['controls'][name] = {'coding': report, 'physical': physical,
                                    'diagnostics': diagnostics(y), 'eligible_reward': allowed}
    result['all_rewards_zero'] = all(item['eligible_reward'] == 0 for item in result['controls'].values())
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({name: item['eligible_reward'] for name, item in result['controls'].items()}))
    if not result['all_rewards_zero']:
        raise SystemExit('challenge exposed a reward vulnerability; keep selection disabled')


if __name__ == '__main__':
    main()
