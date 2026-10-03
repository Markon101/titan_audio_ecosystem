"""Small, chronological predictive coding probes. No Titan or optimizer writes.

Code lengths concern quantized descriptors, not waveforms or Shannon MI.
Coefficients are charged at an explicitly assumed 16 bits each; this is an
operational cost, not a serialized executable compressor.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
from dataclasses import dataclass
import hashlib
import time
import zlib
import numpy as np

SPLITS = (0.40, 0.60, 0.80)
COEFFICIENT_BITS = 16
RIDGE = 10.0


def projection(values, width, name):
    """Fixed orthonormal columns; independent of data and Titan RNG."""
    values = np.asarray(values, dtype=np.float64)
    seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], 'little')
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(values.shape[1],
                                                              min(width, values.shape[1]))))
    return values @ q


def normalize(values, end):
    values = np.asarray(values, dtype=np.float64)
    mean = values[:end].mean(0)
    scale = np.maximum(values[:end].std(0), 1e-6)
    return np.clip((values - mean) / scale, -30, 30)


def cdf(values):
    """Vectorized normal CDF approximation, absolute error below 8e-8."""
    a = np.asarray(values, dtype=np.float64)
    x = np.abs(a)
    t = 1 / (1 + 0.2316419 * x)
    polynomial = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 +
                  t * (-1.821255978 + t * 1.330274429))))
    upper = np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi) * polynomial
    return np.where(a >= 0, 1 - upper, upper)


class Quantizer:
    def __init__(self, training):
        # Repeated quantiles collapse to fewer bins, including one for constants.
        self.edges = []
        for column in np.asarray(training).T:
            if np.ptp(column) <= 1e-9:
                self.edges.append(np.empty(0))
            else:
                self.edges.append(np.unique(np.quantile(column, np.arange(1, 16) / 16)))
        self.marginals = []
        for column, edges in zip(np.asarray(training).T, self.edges):
            counts = np.bincount(np.searchsorted(edges, column, side='right'),
                                 minlength=len(edges) + 1) + 0.5
            self.marginals.append(counts / counts.sum())

    def iid_bits(self, actual):
        return np.column_stack([-np.log2(prob[np.searchsorted(edges, column, side='right')])
                                for column, edges, prob in zip(np.asarray(actual).T,
                                                              self.edges, self.marginals)])

    def gaussian_bits(self, actual, mean, scale):
        result = []
        for j, edges in enumerate(self.edges):
            interior = cdf((edges[None, :] - mean[:, j, None]) / scale[j])
            boundaries = np.column_stack((np.zeros(len(mean)), interior, np.ones(len(mean))))
            probabilities = np.maximum(np.diff(boundaries, axis=1), 1e-12)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            bins = np.searchsorted(edges, actual[:, j], side='right')
            result.append(-np.log2(probabilities[np.arange(len(mean)), bins]))
        return np.column_stack(result)


@dataclass
class Linear:
    weights: np.ndarray
    scale: np.ndarray
    def predict(self, x):
        return np.column_stack((np.ones(len(x)), x)) @ self.weights
    @property
    def parameters(self):
        return self.weights.size + self.scale.size


def fit(x, y, train, calibration):
    a = np.column_stack((np.ones(len(x)), x))
    penalty = np.eye(a.shape[1]) * RIDGE
    penalty[0, 0] = 1e-6
    weights = np.linalg.solve(a[train].T @ a[train] + penalty, a[train].T @ y[train])
    residual = y[calibration] - a[calibration] @ weights
    scale = np.maximum(np.sqrt(np.mean(residual * residual, axis=0)), 0.15)
    return Linear(weights, scale)


def history(y, lags, width, name):
    # No circular padding: otherwise end-of-track data contaminates the prefix
    # normalizer even when the wrapped rows are later purged from model fits.
    delayed = [y if lag == 0 else np.concatenate((np.repeat(y[:1], lag, axis=0),
                                                 y[:-lag]), axis=0) for lag in lags]
    return projection(np.concatenate(delayed, axis=1), width, name)


def forecast_indices(n, horizon, history_start):
    b1, b2, b3 = [int(n * fraction) for fraction in SPLITS]
    indices = np.arange(n - horizon)
    masks = {'train': (indices >= history_start) & (indices + horizon < b1),
             'calibration': (indices >= b1) & (indices + horizon < b2),
             'selection': (indices >= b2) & (indices + horizon < b3),
             'audit': (indices >= b3) & (indices + horizon < n)}
    if any(int(mask.sum()) < 8 for mask in masks.values()):
        raise ValueError('insufficient disjoint temporal support for this horizon')
    return indices, masks


def summaries(gain, extra_parameters, x_width, targets, training_rows):
    bits = float(np.sum(gain))
    cost = max(extra_parameters, 0) * COEFFICIENT_BITS
    halves = np.array_split(gain, 2)
    return {'gain_bits': bits, 'bits_per_feature_forecast': float(np.mean(gain)),
            'coefficient_cost_bits': int(cost), 'net_gain_bits': bits - cost,
            'pd_bits_saved_per_coefficient_bit': bits / max(cost, COEFFICIENT_BITS),
            'half_gain_bits': [float(np.sum(part)) for part in halves],
            'extra_parameters': int(max(extra_parameters, 0)),
            'predict_multiply_adds_per_forecast': int((x_width + 1) * targets),
            'fit_multiply_add_proxy': int(training_rows * (x_width + 1) ** 2 +
                                          (x_width + 1) ** 3),
            'forecasts': len(gain)}


def measure(y, views=None, horizons=(1, 4, 16, 64), sample_chunks=1, short_protocol=False,
            affine_controls=False):
    """No test-time fitting. Selection and audit scores are kept separate."""
    started = time.monotonic()
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 2 or not np.isfinite(y).all():
        raise ValueError('finite two-dimensional target required')
    n = len(y)
    end = int(n * SPLITS[0])
    yn = normalize(y, end)
    quantizer = Quantizer(yn[:end])
    # Time is an available nuisance baseline for gradual drift, not a state view.
    clock = ((np.arange(n) - end / 2) / max(end, 1))[:, None]
    short_lags = (0, 1, 4, 16) if sample_chunks == 1 else (0, 1, 2, 4)
    slow_lags = ((8, 16, 32) if short_protocol else (32, 64, 128)) if sample_chunks == 1 else (8, 16)
    short = normalize(history(yn, short_lags, 8, 'short_history'), end)
    slow = normalize(history(yn, slow_lags, 4, 'slow_history'), end)
    short = np.column_stack((short, clock))
    augmented = {'output_slow': np.column_stack((short, slow))}
    for name, value in (views or {}).items():
        value = np.asarray(value, dtype=np.float64)
        if len(value) != n or not np.isfinite(value).all():
            raise ValueError(f'invalid view {name}')
        # Scale before the fixed projection so coordinates do not dominate by units.
        sketch = normalize(projection(normalize(value, end), 4, name), end)
        augmented[name] = np.column_stack((short, sketch))
    if views:
        for left, right in (('gru', 'field_coarse'), ('morphic_delta_l01', 'morphic_delta_l16'),
                            ('host', 'field_coarse'), ('motif', 'gru')):
            if left in augmented and right in augmented:
                augmented[left + '+' + right] = np.column_stack((short,
                    augmented[left][:, short.shape[1]:], augmented[right][:, short.shape[1]:]))
    result = {'schema': 1, 'samples': n, 'sample_chunks': sample_chunks,
              'split_boundaries': [int(n * f) for f in SPLITS], 'history_lags': list(short_lags + slow_lags),
              'short_protocol': short_protocol,
              'protocol': 'v2_affine_periodic' if affine_controls else 'v1_copy_periodic',
              'bins_per_target': [len(e) + 1 for e in quantizer.edges],
              'horizons': {}, 'units': 'proper code lengths of quantized descriptors; not MI'}
    for h in horizons:
        try:
            ids, masks = forecast_indices(n, h, max(slow_lags))
        except ValueError as error:
            result['horizons'][str(h * sample_chunks)] = {'supported': False, 'reason': str(error)}
            continue
        future = yn[ids + h]
        models = {}
        parameters = {'iid': 0, 'persistence': y.shape[1], 'trend': y.shape[1]}
        models['iid'] = quantizer.iid_bits(future)
        for name, prediction in [('persistence', yn[ids]),
                                  ('trend', yn[ids] + h * (yn[ids] - yn[np.maximum(ids - 1, 0)]))]:
            scale = np.maximum(np.sqrt(np.mean((future[masks['calibration']] -
                                                 prediction[masks['calibration']]) ** 2, axis=0)), 0.15)
            models[name] = quantizer.gaussian_bits(future, prediction, scale)
        periods = list((4, 16, 64, 128) if sample_chunks == 1 else (2, 4, 8, 16))
        # Detect arbitrary exact/near periods on the training prefix, including
        # non-power-of-two loops; a fixed period list alone is easy to game.
        limit = end // 3 if affine_controls else min(max(slow_lags), end // 3)
        empirical_period = min(range(1, limit + 1), key=lambda lag:
            float(np.mean((yn[lag:end] - yn[:end-lag]) ** 2)))
        periods = sorted(set(periods + [empirical_period]))
        for period in periods:
            # Closest past observation with the same future phase; never t+1.
            lag = (h - 1) // period * period + period - h
            prediction = yn[np.maximum(ids - lag, 0)]
            name = f'period_{period * sample_chunks}'
            scale = np.maximum(np.sqrt(np.mean((future[masks['calibration']] -
                                                 prediction[masks['calibration']]) ** 2, axis=0)), 0.15)
            models[name] = quantizer.gaussian_bits(future, prediction, scale)
            parameters[name] = y.shape[1] + 1
        if affine_controls:
            def affine_period(period):
                past, current = yn[:end-period], yn[period:end]
                pm, cm = past.mean(0), current.mean(0)
                slope = np.sum((past-pm)*(current-cm), axis=0) / np.maximum(
                    np.sum((past-pm)**2, axis=0), 1e-8)
                offset = cm - slope*pm
                residual = current - past*slope-offset
                return slope, offset, float(np.mean(residual**2))
            affine_periods = [p for p in periods if p <= end-8] + [
                min(range(1, limit+1), key=lambda p: affine_period(p)[2])]
            for period in sorted(set(affine_periods)):
                slope, offset, _ = affine_period(period)
                cycles = (h-1)//period + 1
                lag = cycles*period-h
                prediction = yn[np.maximum(ids-lag, 0)].copy()
                for _ in range(cycles):
                    prediction = prediction*slope+offset
                scale = np.maximum(np.sqrt(np.mean((future[masks['calibration']] -
                                                     prediction[masks['calibration']])**2, axis=0)), .15)
                name = f'affine_period_{period*sample_chunks}'
                models[name] = quantizer.gaussian_bits(future, prediction, scale)
                parameters[name] = 3*y.shape[1]+1
        local = fit(short[ids], future, masks['train'], masks['calibration'])
        parameters['history'] = local.parameters
        models['history'] = quantizer.gaussian_bits(future, local.predict(short[ids]), local.scale)
        # Baseline selection uses calibration only, with a declared parameter charge.
        best = min(models, key=lambda name: (float(models[name][masks['calibration']].sum()) +
                                             COEFFICIENT_BITS * parameters[name], name))
        # Additional skeptical readout: do not let a parameter charge conceal
        # that an affordable simple predictor forecasts better than iid.
        strongest = min(models, key=lambda name: (float(models[name][masks['calibration']].sum()), name))
        item = {'supported': True, 'horizon_chunks': h * sample_chunks,
                'horizon_seconds': h * sample_chunks * 4096 / 48000,
                'selected_simple_baseline': best,
                'unpenalized_calibration_baseline': strongest,
                'baseline_calibration_bits_with_parameter_cost': {
                    name: float(code[masks['calibration']].sum()) + COEFFICIENT_BITS * parameters[name]
                    for name, code in models.items()}, 'probes': {}, 'synergy': {}}
        for name, x in augmented.items():
            model = fit(x[ids], future, masks['train'], masks['calibration'])
            code = quantizer.gaussian_bits(future, model.predict(x[ids]), model.scale)
            entry = {'parameters': model.parameters, 'input_width': x.shape[1]}
            for segment in ('selection', 'audit'):
                mask = masks[segment]
                entry[segment] = summaries(models[best][mask] - code[mask],
                    model.parameters - parameters[best], x.shape[1], y.shape[1], int(masks['train'].sum()))
                entry[segment]['iid_gain_bits'] = float((models['iid'][mask] - code[mask]).sum())
                entry[segment]['incremental_over_history_bits'] = float((models['history'][mask] - code[mask]).sum())
                entry[segment]['incremental_over_unpenalized_baseline_bits'] = float((models[strongest][mask] - code[mask]).sum())
                entry[segment]['predictor_code_bits'] = float(code[mask].sum())
                entry[segment]['baseline_code_bits'] = float(models[best][mask].sum())
            # Fixed future targets isolate probe acquisition from drifting evaluation data.
            train_ids = np.flatnonzero(masks['train'])
            small_train = np.zeros(len(ids), dtype=bool)
            small_train[train_ids[:max(1, len(train_ids) // 2)]] = True
            small = fit(x[ids], future, small_train, masks['calibration'])
            small_bits = quantizer.gaussian_bits(future, small.predict(x[ids]), small.scale)
            entry['fixed_audit_probe_learning_progress_bits'] = float(
                (small_bits[masks['audit']] - code[masks['audit']]).sum())
            item['probes'][name] = entry
        for name, entry in item['probes'].items():
            if '+' in name:
                left, right = name.split('+')
                item['synergy'][name] = {}
                for segment in ('selection', 'audit'):
                    joint = entry[segment]['gain_bits']
                    a = item['probes'][left][segment]['gain_bits']
                    b = item['probes'][right][segment]['gain_bits']
                    item['synergy'][name][segment] = {
                        'operational_joint_minus_sum_bits': joint - a - b,
                        'joint_beyond_best_individual_bits': joint - max(a, b),
                        'not_PID': True}
        result['horizons'][str(h * sample_chunks)] = item
    result['elapsed_seconds'] = time.monotonic() - started
    return result


def reward(report, segment='selection'):
    """A conservative evidence gate, not a music/entropy objective."""
    components = []
    for h, item in report['horizons'].items():
        if not item['supported']:
            continue
        evidence = item['probes']['output_slow'][segment]
        # Pay for the declared compact predictor and require persistence within segment.
        gain = min(evidence['net_gain_bits'], 2 * min(evidence['half_gain_bits']))
        components.append({'horizon_chunks': int(h), 'persistent_net_bits': gain,
                           'reward': float(np.clip(gain / max(evidence['coefficient_cost_bits'], 16), 0, 1))})
    return {'value': float(np.mean([p['reward'] for p in components])) if components else 0.0,
            'components': components}


def diagnostics(y):
    y = np.asarray(y, dtype=np.float64)
    quantizer = Quantizer(y[:int(len(y) * SPLITS[0])])
    tokens = np.column_stack([np.searchsorted(edge, column, side='right')
                             for edge, column in zip(quantizer.edges, y.T)]).astype(np.uint8)
    entropy = []
    for column in tokens.T:
        counts = np.bincount(column)
        p = counts[counts > 0] / len(column)
        entropy.append(float(-np.sum(p * np.log2(p))))
    normalized = normalize(y, int(len(y) * SPLITS[0]))
    change = np.sqrt(np.mean(np.diff(normalized, axis=0) ** 2, axis=1))
    recurrence = {}
    for period in (1, 4, 16, 64, 128):
        if len(y) > period:
            recurrence[str(period)] = float(np.mean((normalized[period:] - normalized[:-period]) ** 2))
    nearest = []
    for index in range(32, len(y)):
        previous = normalized[max(0, index-128):index-16]
        nearest.append(float(np.sqrt(np.min(np.mean((previous-normalized[index])**2, axis=1)))))
    covariance = np.cov(normalized, rowvar=False)
    energy = np.maximum(np.linalg.eigvalsh(np.atleast_2d(covariance)), 0)
    participation = float(energy.sum()**2 / max(float(np.sum(energy**2)), 1e-12))
    return {'descriptor_entropy_bits_per_feature': entropy,
            'training_active_features': int(sum(len(edges) > 0 for edges in quantizer.edges)),
            'descriptor_token_zlib_ratio': len(zlib.compress(tokens.tobytes(), 9)) / max(tokens.nbytes, 1),
            'median_normalized_change': float(np.median(change)),
            'last_quarter_normalized_change': float(np.mean(change[-max(1, len(change) // 4):])),
            'repeat_mean_square_by_lag': recurrence,
            'normalized_descriptor_covariance_participation_ratio': participation,
            'recent_novelty_window': 128, 'recent_novelty_exclusion': 16,
            'recent_nearest_distance_mean': float(np.mean(nearest)),
            'recent_nearest_distance_last_quarter': float(np.mean(nearest[-max(1,len(nearest)//4):])),
            'not_waveform_compression': True}
