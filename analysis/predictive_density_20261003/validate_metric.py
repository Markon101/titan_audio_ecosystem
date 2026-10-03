"""Freeze trivial/noise and known complementary-prediction controls first."""
import json
import argparse
from pathlib import Path
import time
import numpy as np
from audio import CHUNK, RATE, describe
from predictive import measure, reward, diagnostics

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hardened', action='store_true')
    args = parser.parse_args()
    path = HERE / ('metric_validation_v2.json' if args.hardened else 'metric_validation.json')
    if path.exists():
        raise FileExistsError(path)
    rng = np.random.default_rng(20261003)
    started = time.monotonic()
    result = {'schema': 1, 'controls': {}, 'subjective_quality_not_measured': True}
    for name in ('silence', 'constant', 'sinusoid', 'amplitude_modulated_tone',
                 'repeated_texture_37', 'white_noise', 'colored_noise'):
        n = 1024 * CHUNK
        if name == 'silence':
            audio = np.zeros((n, 2))
        elif name == 'constant':
            audio = np.full((n, 2), .15)
        elif name in ('sinusoid', 'amplitude_modulated_tone'):
            t = np.arange(n) / RATE
            envelope = .2 if name == 'sinusoid' else .15 + .05 * np.sin(2 * np.pi * .37 * t)
            audio = np.repeat((envelope * np.sin(2 * np.pi * 375 * t))[:, None], 2, axis=1)
        elif name == 'repeated_texture_37':
            texture = rng.normal(0, .08, size=(37 * CHUNK, 2))
            audio = texture[np.arange(n) % len(texture)]
        else:
            audio = rng.normal(0, .07, size=(n, 2))
            if name == 'colored_noise':
                # Fixed finite filter; no new neural model/dependency.
                original = audio.copy()
                for lag in (1, 2, 4, 8, 16):
                    audio[lag:] += .4 * original[:-lag]
                audio *= .7
        y, physical = describe(audio)
        report = measure(y, affine_controls=args.hardened)
        raw_reward = reward(report)
        allowed = raw_reward['value'] if physical['eligible'] else 0
        result['controls'][name] = {'physical': physical, 'coding': report,
                                    'diagnostics': diagnostics(y), 'raw_reward': raw_reward,
                                    'eligible_reward': allowed,
                                    'naive_negative_persistence_mse': -float(np.mean(np.diff(y, axis=0) ** 2))}
        if allowed != 0:
            raise RuntimeError(f'trivial/noise reward gate failed: {name}; preserve output in a new audit')
    a, b = rng.normal(size=(2, 1536, 4))
    target = np.roll(a[:, :2] + b[:, :2], 4, axis=0) + .08 * rng.normal(size=(1536, 2))
    known = measure(target, {'gru': a, 'field_coarse': b}, horizons=(4,), affine_controls=args.hardened)
    joint = known['horizons']['4']['synergy']['gru+field_coarse']['audit']
    if joint['joint_beyond_best_individual_bits'] <= 100:
        raise RuntimeError('known complementary internal contribution not detected')
    result['known_delayed_complementary_process'] = known
    if args.hardened:
        # Retain the failure witness, not just the corrected result.
        arng = np.random.default_rng(19)
        loop = arng.normal(size=(1024, 1))
        for t in range(65, len(loop)):
            loop[t] = -.92*loop[t-65]+.12*arng.normal()
        v1, v2 = measure(loop), measure(loop, affine_controls=True)
        result['affine_loop_failure_witness'] = {'v1': v1, 'v2': v2,
                                                'v1_reward': reward(v1), 'v2_reward': reward(v2)}
        if reward(v1)['value'] <= 0 or reward(v2)['value'] != 0:
            raise RuntimeError('affine-loop discrimination failed')
        # Exploratory constructed positive control, NOT a Titan/music result.
        arng = np.random.default_rng(31)
        structured = arng.normal(0, .1, size=(4096, 1))
        for t in range(65, len(structured)):
            structured[t] = .55*structured[t-65]-.4*structured[t-33]+.07*arng.normal()
        positive = measure(structured, affine_controls=True)
        result['long_memory_gaussian_positive_control'] = {'coding': positive, 'reward': reward(positive),
            'interpretation': 'compact stochastic predictive relationships, not music; warns that positive PD cannot certify artistic organization',
            'exploratory_fixture_search': 'three coefficient pairs were tried: (0.5,0.35), (0.65,0.25), (0.55,-0.4); metric was not retuned'}
        if reward(positive)['value'] <= 0:
            raise RuntimeError('hardened metric did not detect constructed long-context relation')
    result['all_trivial_noise_rewards_zero'] = True
    result['naive_control_degeneracy'] = 'silence and constants tie at perfect persistence; not meaningful organization'
    result['initial_test_failure_fixed'] = 'circular history padding leaked suffix into prefix normalization; causal padding and audit-mutation regression fixed it'
    result['elapsed_seconds'] = time.monotonic() - started
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'trivial_noise_controls': len(result['controls']), 'all_rewards_zero': True,
                      'known_joint_extra_bits': joint['joint_beyond_best_individual_bits']}))


if __name__ == '__main__':
    main()
