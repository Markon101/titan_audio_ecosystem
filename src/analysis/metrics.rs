use rustfft::{num_complex::Complex, FftPlanner};
use serde::Serialize;

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct AudioMetrics {
    pub frames: usize,
    pub rms_left: f32,
    pub rms_right: f32,
    pub rms_mid: f32,
    pub peak: f32,
    pub crest: Option<f32>,
    pub dc_left: f32,
    pub dc_right: f32,
    pub clipping_fraction: f32,
    pub nonfinite: usize,
    pub stereo_correlation: Option<f32>,
    pub side_mid_ratio: Option<f32>,
    pub side_energy_width: Option<f32>,
    pub correlation_aware_width: Option<f32>,
    pub centroid_hz: Option<f32>,
    pub rolloff_85_hz: Option<f32>,
    pub spectral_flatness: Option<f32>,
    pub low_band_ratio: Option<f32>,
    pub ultrasonic_ratio: Option<f32>,
    pub onset_strength: Option<f32>,
    pub temporal_envelope_rms: Option<f32>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct AudioDistance {
    pub waveform_l2: Option<f32>,
    pub waveform_correlation: Option<f32>,
    pub multiresolution_log_spectral: Option<f32>,
    pub temporal_envelope: Option<f32>,
    pub rms_delta: f32,
    pub centroid_delta_hz: Option<f32>,
    pub stereo_correlation_delta: Option<f32>,
    pub valid_waveform_alignment: bool,
    pub invalid_reason: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct StateDistance {
    pub micro_relative_l2: Option<f32>,
    pub macro_relative_l2: Option<f32>,
    pub recurrent_relative_l2: Option<f32>,
    pub combined_relative_l2: Option<f32>,
    pub valid: bool,
    pub reason: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct RecurrenceCandidate {
    pub step: u64,
    pub nearest_prior_step: Option<u64>,
    pub normalized_distance: Option<f32>,
    pub theiler_window: usize,
    pub recurrence_distance_valid: bool,
    pub label: &'static str,
}

pub(crate) fn audio_metrics(left: &[f32], right: &[f32]) -> AudioMetrics {
    let frames = left.len().min(right.len());
    if frames == 0 {
        return AudioMetrics::default();
    }
    let mut left_sum = 0.0f64;
    let mut right_sum = 0.0f64;
    let mut left_sq = 0.0f64;
    let mut right_sq = 0.0f64;
    let mut mid_sq = 0.0f64;
    let mut side_sq = 0.0f64;
    let mut cross = 0.0f64;
    let mut peak = 0.0f32;
    let mut clipping = 0usize;
    let mut nonfinite = 0usize;
    let mut mono = Vec::with_capacity(frames);
    for (&left, &right) in left.iter().zip(right).take(frames) {
        if !left.is_finite() || !right.is_finite() {
            nonfinite += 1;
            mono.push(0.0);
            continue;
        }
        let mid = (left + right) * 0.5;
        let side = (left - right) * 0.5;
        mono.push(mid);
        left_sum += left as f64;
        right_sum += right as f64;
        left_sq += left as f64 * left as f64;
        right_sq += right as f64 * right as f64;
        mid_sq += mid as f64 * mid as f64;
        side_sq += side as f64 * side as f64;
        cross += left as f64 * right as f64;
        peak = peak.max(left.abs()).max(right.abs());
        clipping += usize::from(left.abs() >= 1.0) + usize::from(right.abs() >= 1.0);
    }
    let n = frames as f64;
    let rms_left = (left_sq / n).sqrt() as f32;
    let rms_right = (right_sq / n).sqrt() as f32;
    let rms_mid = (mid_sq / n).sqrt() as f32;
    let stereo_correlation = normalized_ratio(cross, (left_sq * right_sq).sqrt());
    let side_mid_ratio = normalized_ratio(side_sq.sqrt(), mid_sq.sqrt());
    let side_energy_width = normalized_ratio((4.0 * side_sq).sqrt(), (left_sq + right_sq).sqrt());
    let correlation_aware_width = side_energy_width
        .zip(stereo_correlation)
        .map(|(side, corr)| super::super::stereo::correlation_aware_width(side, corr));
    let spectrum = spectrum_metrics(&mono);
    let envelope = temporal_envelope(&mono, 128);
    let onset_strength = if envelope.len() > 1 {
        Some(
            envelope
                .windows(2)
                .map(|window| (window[1] - window[0]).max(0.0))
                .sum::<f32>()
                / (envelope.len() - 1) as f32,
        )
    } else {
        None
    };
    AudioMetrics {
        frames,
        rms_left,
        rms_right,
        rms_mid,
        peak,
        crest: (rms_mid > 1e-9).then_some(peak / rms_mid),
        dc_left: (left_sum / n) as f32,
        dc_right: (right_sum / n) as f32,
        clipping_fraction: clipping as f32 / (frames * 2) as f32,
        nonfinite,
        stereo_correlation,
        side_mid_ratio,
        side_energy_width,
        correlation_aware_width,
        centroid_hz: spectrum.centroid_hz,
        rolloff_85_hz: spectrum.rolloff_85_hz,
        spectral_flatness: spectrum.flatness,
        low_band_ratio: spectrum.low_band_ratio,
        ultrasonic_ratio: spectrum.ultrasonic_ratio,
        onset_strength,
        temporal_envelope_rms: (!envelope.is_empty()).then(|| {
            (envelope.iter().map(|value| value * value).sum::<f32>() / envelope.len() as f32).sqrt()
        }),
    }
}

pub(crate) fn audio_distance(
    left_a: &[f32],
    right_a: &[f32],
    left_b: &[f32],
    right_b: &[f32],
) -> AudioDistance {
    let metrics_a = audio_metrics(left_a, right_a);
    let metrics_b = audio_metrics(left_b, right_b);
    let aligned = left_a.len() == left_b.len()
        && right_a.len() == right_b.len()
        && left_a.len() == right_a.len()
        && !left_a.is_empty();
    let mono_a: Vec<f32> = left_a
        .iter()
        .zip(right_a)
        .map(|(left, right)| (left + right) * 0.5)
        .collect();
    let mono_b: Vec<f32> = left_b
        .iter()
        .zip(right_b)
        .map(|(left, right)| (left + right) * 0.5)
        .collect();
    let (waveform_l2, waveform_correlation) = if aligned {
        (relative_l2(&mono_a, &mono_b), correlation(&mono_a, &mono_b))
    } else {
        (None, None)
    };
    AudioDistance {
        waveform_l2,
        waveform_correlation,
        multiresolution_log_spectral: aligned
            .then(|| multiresolution_spectral_distance(&mono_a, &mono_b)),
        temporal_envelope: aligned.then(|| {
            relative_l2(
                &temporal_envelope(&mono_a, 128),
                &temporal_envelope(&mono_b, 128),
            )
            .unwrap_or(0.0)
        }),
        rms_delta: (metrics_a.rms_mid - metrics_b.rms_mid).abs(),
        centroid_delta_hz: metrics_a
            .centroid_hz
            .zip(metrics_b.centroid_hz)
            .map(|(a, b)| (a - b).abs()),
        stereo_correlation_delta: metrics_a
            .stereo_correlation
            .zip(metrics_b.stereo_correlation)
            .map(|(a, b)| (a - b).abs()),
        valid_waveform_alignment: aligned,
        invalid_reason: (!aligned).then(|| "waveforms are not sample-aligned".to_string()),
    }
}

pub(crate) fn state_distance(
    micro_a: &[f32],
    macro_a: &[f32],
    hidden_a: &[f32],
    micro_b: &[f32],
    macro_b: &[f32],
    hidden_b: &[f32],
) -> StateDistance {
    if micro_a.len() != micro_b.len()
        || macro_a.len() != macro_b.len()
        || hidden_a.len() != hidden_b.len()
    {
        return StateDistance {
            valid: false,
            reason: Some("state components have different shapes".to_string()),
            ..StateDistance::default()
        };
    }
    let micro = relative_l2(micro_a, micro_b);
    let macro_t = relative_l2(macro_a, macro_b);
    let hidden = relative_l2(hidden_a, hidden_b);
    let combined = relative_l2_three(micro_a, macro_a, hidden_a, micro_b, macro_b, hidden_b);
    StateDistance {
        micro_relative_l2: micro,
        macro_relative_l2: macro_t,
        recurrent_relative_l2: hidden,
        combined_relative_l2: combined,
        valid: micro.is_some() && macro_t.is_some() && hidden.is_some(),
        reason: None,
    }
}

pub(crate) fn recurrence_candidate(
    step: u64,
    signature: &[f32],
    history: &[(u64, Vec<f32>)],
    theiler_window: usize,
) -> RecurrenceCandidate {
    let mut nearest: Option<(u64, f32)> = None;
    for (prior_step, prior) in history {
        if step.saturating_sub(*prior_step) < theiler_window as u64
            || prior.len() != signature.len()
        {
            continue;
        }
        if let Some(distance) = relative_l2(signature, prior) {
            if nearest.is_none_or(|(_, best)| distance < best) {
                nearest = Some((*prior_step, distance));
            }
        }
    }
    RecurrenceCandidate {
        step,
        nearest_prior_step: nearest.map(|(prior, _)| prior),
        normalized_distance: nearest.map(|(_, distance)| distance),
        theiler_window,
        recurrence_distance_valid: nearest.is_some(),
        label: "approximate_cycle_candidate",
    }
}

pub(crate) fn compact_signature(micro: &[f32], macro_t: &[f32], hidden: &[f32]) -> Vec<f32> {
    let mut output = Vec::new();
    append_block_signature(&mut output, micro, 64);
    append_block_signature(&mut output, macro_t, 32);
    append_block_signature(&mut output, hidden, 32);
    output
}

fn append_block_signature(output: &mut Vec<f32>, values: &[f32], bins: usize) {
    if values.is_empty() {
        output.resize(output.len() + bins, 0.0);
        return;
    }
    for bin in 0..bins {
        let start = bin * values.len() / bins;
        let end = ((bin + 1) * values.len() / bins)
            .max(start + 1)
            .min(values.len());
        output.push(values[start..end].iter().sum::<f32>() / (end - start) as f32);
    }
}

fn normalized_ratio(numerator: f64, denominator: f64) -> Option<f32> {
    (denominator > 1e-12).then_some((numerator / denominator) as f32)
}

fn relative_l2(left: &[f32], right: &[f32]) -> Option<f32> {
    if left.len() != right.len() || left.is_empty() {
        return None;
    }
    let mut delta = 0.0f64;
    let mut scale = 0.0f64;
    for (&left, &right) in left.iter().zip(right) {
        if !left.is_finite() || !right.is_finite() {
            return None;
        }
        let difference = left as f64 - right as f64;
        delta += difference * difference;
        scale += right as f64 * right as f64;
    }
    Some((delta / scale.max(1e-20)).sqrt() as f32)
}

fn relative_l2_three(
    a0: &[f32],
    a1: &[f32],
    a2: &[f32],
    b0: &[f32],
    b1: &[f32],
    b2: &[f32],
) -> Option<f32> {
    let mut delta = 0.0f64;
    let mut scale = 0.0f64;
    for (left, right) in [(a0, b0), (a1, b1), (a2, b2)] {
        for (&left, &right) in left.iter().zip(right) {
            if !left.is_finite() || !right.is_finite() {
                return None;
            }
            delta += (left as f64 - right as f64).powi(2);
            scale += (right as f64).powi(2);
        }
    }
    Some((delta / scale.max(1e-20)).sqrt() as f32)
}

fn correlation(left: &[f32], right: &[f32]) -> Option<f32> {
    if left.len() != right.len() || left.is_empty() {
        return None;
    }
    let mean_left = left.iter().sum::<f32>() / left.len() as f32;
    let mean_right = right.iter().sum::<f32>() / right.len() as f32;
    let mut cross = 0.0f64;
    let mut left_sq = 0.0f64;
    let mut right_sq = 0.0f64;
    for (&left, &right) in left.iter().zip(right) {
        let left = (left - mean_left) as f64;
        let right = (right - mean_right) as f64;
        cross += left * right;
        left_sq += left * left;
        right_sq += right * right;
    }
    normalized_ratio(cross, (left_sq * right_sq).sqrt())
}

fn temporal_envelope(signal: &[f32], frames: usize) -> Vec<f32> {
    if signal.is_empty() || frames == 0 {
        return Vec::new();
    }
    (0..frames)
        .map(|frame| {
            let start = frame * signal.len() / frames;
            let end = ((frame + 1) * signal.len() / frames)
                .max(start + 1)
                .min(signal.len());
            (signal[start..end]
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                / (end - start) as f32)
                .sqrt()
        })
        .collect()
}

fn multiresolution_spectral_distance(left: &[f32], right: &[f32]) -> f32 {
    [256usize, 1024, 4096]
        .iter()
        .filter(|size| **size <= left.len().min(right.len()))
        .map(|size| log_spectral_distance(&left[..*size], &right[..*size]))
        .sum::<f32>()
        / [256usize, 1024, 4096]
            .iter()
            .filter(|size| **size <= left.len().min(right.len()))
            .count()
            .max(1) as f32
}

fn log_spectral_distance(left: &[f32], right: &[f32]) -> f32 {
    let left = magnitude_spectrum(left);
    let right = magnitude_spectrum(right);
    let count = left.len().min(right.len()).max(1);
    (left
        .iter()
        .zip(right.iter())
        .map(|(left, right)| ((left + 1e-7).ln() - (right + 1e-7).ln()).powi(2))
        .sum::<f32>()
        / count as f32)
        .sqrt()
}

fn magnitude_spectrum(signal: &[f32]) -> Vec<f32> {
    if signal.len() < 2 {
        return Vec::new();
    }
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(signal.len());
    let mut buffer: Vec<Complex<f32>> = signal
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let window =
                0.5 - 0.5 * (super::super::TWO_PI * index as f32 / (signal.len() - 1) as f32).cos();
            Complex::new(value * window, 0.0)
        })
        .collect();
    fft.process(&mut buffer);
    buffer
        .iter()
        .take(signal.len() / 2)
        .map(|value| value.norm())
        .collect()
}

struct SpectrumMetrics {
    centroid_hz: Option<f32>,
    rolloff_85_hz: Option<f32>,
    flatness: Option<f32>,
    low_band_ratio: Option<f32>,
    ultrasonic_ratio: Option<f32>,
}

fn spectrum_metrics(signal: &[f32]) -> SpectrumMetrics {
    let magnitudes = magnitude_spectrum(signal);
    let sum = magnitudes.iter().sum::<f32>();
    if magnitudes.is_empty() || sum <= 1e-12 {
        return SpectrumMetrics {
            centroid_hz: None,
            rolloff_85_hz: None,
            flatness: None,
            low_band_ratio: None,
            ultrasonic_ratio: None,
        };
    }
    let bin_hz = super::super::SAMPLE_RATE as f32 / signal.len() as f32;
    let centroid = magnitudes
        .iter()
        .enumerate()
        .map(|(index, magnitude)| index as f32 * bin_hz * magnitude)
        .sum::<f32>()
        / sum;
    let threshold = sum * 0.85;
    let mut cumulative = 0.0;
    let mut rolloff = 0.0;
    for (index, magnitude) in magnitudes.iter().enumerate() {
        cumulative += magnitude;
        if cumulative >= threshold {
            rolloff = index as f32 * bin_hz;
            break;
        }
    }
    let geometric = (magnitudes
        .iter()
        .map(|magnitude| (magnitude + 1e-12).ln())
        .sum::<f32>()
        / magnitudes.len() as f32)
        .exp();
    let arithmetic = sum / magnitudes.len() as f32;
    let band_ratio = |minimum: f32, maximum: f32| {
        magnitudes
            .iter()
            .enumerate()
            .filter(|(index, _)| {
                let frequency = *index as f32 * bin_hz;
                frequency >= minimum && frequency < maximum
            })
            .map(|(_, magnitude)| magnitude)
            .sum::<f32>()
            / sum
    };
    SpectrumMetrics {
        centroid_hz: Some(centroid),
        rolloff_85_hz: Some(rolloff),
        flatness: Some((geometric / arithmetic.max(1e-12)).clamp(0.0, 1.0)),
        low_band_ratio: Some(band_ratio(20.0, 250.0)),
        ultrasonic_ratio: Some(band_ratio(20_000.0, 24_001.0)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panned_mono_is_correlated_not_truthfully_wide() {
        let left = vec![0.5f32, -0.5, 0.25, -0.25];
        let right: Vec<f32> = left.iter().map(|value| value * 0.2).collect();
        let metrics = audio_metrics(&left, &right);
        assert!(metrics.stereo_correlation.unwrap() > 0.99);
        assert!(metrics.correlation_aware_width.unwrap() < 0.05);
    }

    #[test]
    fn antiphase_has_negative_correlation_and_side_energy() {
        let left = vec![0.5f32, -0.5, 0.25, -0.25];
        let right: Vec<f32> = left.iter().map(|value| -*value).collect();
        let metrics = audio_metrics(&left, &right);
        assert!(metrics.stereo_correlation.unwrap() < -0.99);
        assert!(metrics.side_energy_width.unwrap() > 0.9);
    }

    #[test]
    fn identical_states_have_zero_distance() {
        let values = vec![1.0, 2.0, 3.0];
        let distance = state_distance(&values, &values, &values, &values, &values, &values);
        assert_eq!(distance.combined_relative_l2, Some(0.0));
    }
}
