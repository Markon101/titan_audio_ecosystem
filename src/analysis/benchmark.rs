use super::metrics::{self, AudioDistance, AudioMetrics};
use rand::{Rng, SeedableRng};
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub(crate) struct SignalDefinition {
    pub id: String,
    pub family: String,
    pub generator_version: u32,
    pub seed: u64,
    pub sample_rate: u32,
    pub frames: usize,
    pub channels: usize,
    pub formula: String,
    pub parameters: serde_json::Value,
    pub float_bytes_sha256: String,
}

#[derive(Clone, Debug)]
pub(crate) struct ProceduralSignal {
    pub definition: SignalDefinition,
    pub left: Vec<f32>,
    pub right: Vec<f32>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct BenchmarkComparison {
    pub reference_id: String,
    pub reference_family: String,
    pub reference_metrics: AudioMetrics,
    pub generated_distance: Option<AudioDistance>,
    pub comparison_semantics: String,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct BenchmarkReport {
    pub generator_version: u32,
    pub direct_reference_input: bool,
    pub optimizer_steps: usize,
    pub entered_training_corpus: bool,
    pub supported_modes: Vec<String>,
    pub deferred_modes: Vec<String>,
    pub identical_first_chunk_negative_control: String,
    pub target_label_permutation_control: String,
    pub definitions: Vec<SignalDefinition>,
    pub comparisons: Vec<BenchmarkComparison>,
    pub interpretation_constraints: Vec<String>,
}

pub(crate) fn generate(seed: u64) -> Vec<ProceduralSignal> {
    let frames = super::super::SAMPLE_RATE as usize;
    let sample_rate = super::super::SAMPLE_RATE as f32;
    let mut signals = Vec::new();
    let mut add = |id: &str,
                   family: &str,
                   formula: &str,
                   parameters: serde_json::Value,
                   left: Vec<f32>,
                   right: Vec<f32>| {
        let mut bytes = Vec::with_capacity((left.len() + right.len()) * 4);
        for (left, right) in left.iter().zip(&right) {
            bytes.extend_from_slice(&left.to_le_bytes());
            bytes.extend_from_slice(&right.to_le_bytes());
        }
        signals.push(ProceduralSignal {
            definition: SignalDefinition {
                id: id.to_string(),
                family: family.to_string(),
                generator_version: 1,
                seed,
                sample_rate: super::super::SAMPLE_RATE,
                frames: left.len().min(right.len()),
                channels: 2,
                formula: formula.to_string(),
                parameters,
                float_bytes_sha256: super::super::provenance::sha256_bytes(&bytes),
            },
            left,
            right,
        });
    };
    for frequency in [110.0f32, 440.0, 8_000.0] {
        let mono = tone(frames, sample_rate, frequency, 0.35);
        add(
            &format!("pure_tone_{frequency:.0}"),
            "pure_tone",
            "0.35*sin(2*pi*f*t)",
            serde_json::json!({"frequency_hz": frequency}),
            mono.clone(),
            mono,
        );
    }
    let harmonic = stack(frames, sample_rate, &[1.0, 2.0, 3.0, 4.0, 5.0], 137.0);
    add(
        "harmonic_stack",
        "harmonic_stack",
        "normalized sum sin(2*pi*base*ratio*t)/index",
        serde_json::json!({"base_hz": 137.0, "ratios": [1,2,3,4,5]}),
        harmonic.clone(),
        harmonic,
    );
    let inharmonic = stack(
        frames,
        sample_rate,
        &[1.0, std::f32::consts::SQRT_2, 2.17, std::f32::consts::PI],
        173.0,
    );
    add(
        "inharmonic_stack",
        "inharmonic_stack",
        "normalized sum of irrational-ratio sinusoids",
        serde_json::json!({"base_hz": 173.0, "ratios": [1.0, std::f32::consts::SQRT_2, 2.17, std::f32::consts::PI]}),
        inharmonic.clone(),
        inharmonic,
    );
    let chirp_linear = chirp(frames, sample_rate, 40.0, 12_000.0, false);
    add(
        "linear_chirp",
        "chirp",
        "sin(2*pi*integral(linear_frequency(t)))",
        serde_json::json!({"start_hz":40.0,"end_hz":12000.0,"law":"linear"}),
        chirp_linear.clone(),
        chirp_linear,
    );
    let chirp_log = chirp(frames, sample_rate, 40.0, 12_000.0, true);
    add(
        "log_chirp",
        "chirp",
        "sin(2*pi*integral(log_frequency(t)))",
        serde_json::json!({"start_hz":40.0,"end_hz":12000.0,"law":"logarithmic"}),
        chirp_log.clone(),
        chirp_log,
    );
    let mut impulse = vec![0.0f32; frames];
    impulse[0] = 0.9;
    for (index, sample) in impulse.iter_mut().enumerate().take(12_000).skip(1) {
        *sample += 0.35 * (-(index as f32) / 1_600.0).exp() * (0.19 * index as f32).sin();
    }
    add(
        "impulse_decay",
        "impulse",
        "unit impulse plus deterministic damped sinusoidal decay",
        serde_json::json!({"decay_samples":1600}),
        impulse.clone(),
        impulse,
    );
    let am: Vec<f32> = (0..frames)
        .map(|index| {
            let time = index as f32 / sample_rate;
            0.32 * (super::super::TWO_PI * 330.0 * time).sin()
                * (0.55 + 0.45 * (super::super::TWO_PI * 7.0 * time).sin())
        })
        .collect();
    add(
        "am_tone",
        "amplitude_modulation",
        "carrier*(0.55+0.45*sin(modulator))",
        serde_json::json!({"carrier_hz":330,"modulator_hz":7}),
        am.clone(),
        am,
    );
    let fm: Vec<f32> = (0..frames)
        .map(|index| {
            let time = index as f32 / sample_rate;
            0.32 * (super::super::TWO_PI * 220.0 * time
                + 3.5 * (super::super::TWO_PI * 31.0 * time).sin())
            .sin()
        })
        .collect();
    add(
        "fm_tone",
        "frequency_modulation",
        "sin(carrier_phase+index*sin(modulator_phase))",
        serde_json::json!({"carrier_hz":220,"modulator_hz":31,"index":3.5}),
        fm.clone(),
        fm,
    );
    let pulse = impulse_train(frames, sample_rate, &[4.0], 0.55);
    add(
        "rhythmic_impulse_train",
        "rhythm",
        "decaying impulses at 4 Hz",
        serde_json::json!({"rate_hz":4}),
        pulse.clone(),
        pulse,
    );
    let poly_left = impulse_train(frames, sample_rate, &[3.0, 5.0], 0.42);
    let poly_right = impulse_train(frames, sample_rate, &[4.0, 7.0], 0.42);
    add(
        "polyrhythm_3_5_4_7",
        "polyrhythm",
        "left 3+5 Hz impulse trains; right 4+7 Hz",
        serde_json::json!({"left_hz":[3,5],"right_hz":[4,7]}),
        poly_left,
        poly_right,
    );
    let envelope: Vec<f32> = (0..frames)
        .map(|index| {
            let time = index as f32 / sample_rate;
            let attack = (time / 0.08).min(1.0);
            let release = ((1.0 - time) / 0.25).clamp(0.0, 1.0);
            0.5 * attack * release * (super::super::TWO_PI * 261.6256 * time).sin()
        })
        .collect();
    add(
        "adsr_like_envelope",
        "envelope",
        "bounded attack and release multiplied by C4 tone",
        serde_json::json!({"attack_s":0.08,"release_s":0.25}),
        envelope.clone(),
        envelope,
    );
    let mut rng = super::super::RuntimeRng::seed_from_u64(seed ^ 0xB3AC_4A7E);
    let mut lowpass = 0.0f32;
    let filtered_noise: Vec<f32> = (0..frames)
        .map(|_| {
            lowpass += 0.035 * (rng.gen_range(-1.0f32..1.0) - lowpass);
            lowpass * 0.45
        })
        .collect();
    add(
        "filtered_noise",
        "deterministic_filtered_noise",
        "seeded white noise through one-pole low-pass",
        serde_json::json!({"pole_update":0.035}),
        filtered_noise.clone(),
        filtered_noise,
    );
    let carrier = tone(frames, sample_rate, 523.25, 0.35);
    let mut motion_left = Vec::with_capacity(frames);
    let mut motion_right = Vec::with_capacity(frames);
    for (index, sample) in carrier.iter().enumerate() {
        let pan = (super::super::TWO_PI * 0.75 * index as f32 / sample_rate).sin();
        motion_left.push(sample * ((1.0 - pan) * 0.5).sqrt());
        motion_right.push(sample * ((1.0 + pan) * 0.5).sqrt());
    }
    add(
        "stereo_motion",
        "stereo_motion",
        "equal-power sinusoidal pan trajectory",
        serde_json::json!({"carrier_hz":523.25,"motion_hz":0.75}),
        motion_left,
        motion_right,
    );
    let antiphase = tone(frames, sample_rate, 880.0, 0.25);
    add(
        "stereo_antiphase",
        "stereo_geometry",
        "right=-left",
        serde_json::json!({"carrier_hz":880}),
        antiphase.clone(),
        antiphase.iter().map(|value| -*value).collect(),
    );
    let nested: Vec<f32> = (0..frames)
        .map(|index| {
            let time = index as f32 / sample_rate;
            let slow = 0.55 + 0.45 * (super::super::TWO_PI * 1.5 * time).sin();
            let fast = 0.65 + 0.35 * (super::super::TWO_PI * 13.0 * time).sin();
            0.35 * slow * fast * (super::super::TWO_PI * 196.0 * time).sin()
        })
        .collect();
    add(
        "nested_temporal_modulation",
        "nested_temporal_structure",
        "carrier multiplied by slow and fast envelopes",
        serde_json::json!({"carrier_hz":196,"envelopes_hz":[1.5,13]}),
        nested.clone(),
        nested,
    );
    let branching = impulse_train(frames, sample_rate, &[2.0, 4.0, 8.0, 16.0], 0.24);
    add(
        "branching_multiscale_rhythm",
        "branching_multiscale_rhythm",
        "sum of octave-related deterministic impulse trains",
        serde_json::json!({"rates_hz":[2,4,8,16]}),
        branching.clone(),
        branching,
    );
    signals
}

pub(crate) fn report(
    signals: &[ProceduralSignal],
    generated_audio: Option<(&[f32], &[f32])>,
) -> BenchmarkReport {
    let comparisons = signals
        .iter()
        .map(|signal| BenchmarkComparison {
            reference_id: signal.definition.id.clone(),
            reference_family: signal.definition.family.clone(),
            reference_metrics: metrics::audio_metrics(&signal.left, &signal.right),
            generated_distance: generated_audio.map(|(left, right)| {
                let frames = signal.left.len().min(left.len());
                metrics::audio_distance(
                    &signal.left[..frames],
                    &signal.right[..frames],
                    &left[..frames],
                    &right[..frames],
                )
            }),
            comparison_semantics:
                "transparent descriptor proximity; not reconstruction or transfer".to_string(),
        })
        .collect();
    BenchmarkReport {
        generator_version: 1,
        direct_reference_input: false,
        optimizer_steps: 0,
        entered_training_corpus: false,
        supported_modes: vec![
            "descriptor_coverage".to_string(),
            "target_error_feedback_probe".to_string(),
        ],
        deferred_modes: vec!["held_out_reconstruction_or_direct_conditioning".to_string()],
        identical_first_chunk_negative_control:
            "required: target signals are evaluated only after model.forward; reference identity cannot affect the current chunk"
                .to_string(),
        target_label_permutation_control:
            "descriptor labels are metadata only; permutation must not affect generated audio"
                .to_string(),
        definitions: signals
            .iter()
            .map(|signal| signal.definition.clone())
            .collect(),
        comparisons,
        interpretation_constraints: vec![
            "benchmark proximity is not semantic understanding".to_string(),
            "the current architecture has no direct target/reference input".to_string(),
            "procedural signals are never written into OLD_WAVS or the corpus manifest".to_string(),
        ],
    }
}

fn tone(frames: usize, sample_rate: f32, frequency: f32, amplitude: f32) -> Vec<f32> {
    (0..frames)
        .map(|index| {
            amplitude * (super::super::TWO_PI * frequency * index as f32 / sample_rate).sin()
        })
        .collect()
}

fn stack(frames: usize, sample_rate: f32, ratios: &[f32], base: f32) -> Vec<f32> {
    (0..frames)
        .map(|index| {
            let time = index as f32 / sample_rate;
            ratios
                .iter()
                .enumerate()
                .map(|(partial, ratio)| {
                    (super::super::TWO_PI * base * ratio * time).sin() / (partial + 1) as f32
                })
                .sum::<f32>()
                * 0.35
                / ratios.len().max(1) as f32
        })
        .collect()
}

fn chirp(frames: usize, sample_rate: f32, start: f32, end: f32, logarithmic: bool) -> Vec<f32> {
    let duration = frames as f32 / sample_rate;
    let mut phase = 0.0f32;
    (0..frames)
        .map(|index| {
            let fraction = index as f32 / frames.max(1) as f32;
            let frequency = if logarithmic {
                start * (end / start).powf(fraction)
            } else {
                start + (end - start) * fraction
            };
            phase += super::super::TWO_PI * frequency / sample_rate;
            0.32 * phase.sin()
                * (index as f32 / (0.01 * sample_rate)).min(1.0)
                * ((duration - index as f32 / sample_rate) / 0.01).clamp(0.0, 1.0)
        })
        .collect()
}

fn impulse_train(frames: usize, sample_rate: f32, rates: &[f32], amplitude: f32) -> Vec<f32> {
    let mut output = vec![0.0f32; frames];
    for rate in rates {
        let period = (sample_rate / rate).round().max(1.0) as usize;
        for start in (0..frames).step_by(period) {
            for offset in 0..256.min(frames - start) {
                output[start + offset] +=
                    amplitude * (-(offset as f32) / 45.0).exp() / rates.len().max(1) as f32;
            }
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn procedural_suite_is_deterministic_and_modality_honest() {
        let first = generate(123);
        let second = generate(123);
        assert_eq!(first.len(), second.len());
        assert_eq!(
            first
                .iter()
                .map(|signal| &signal.definition.float_bytes_sha256)
                .collect::<Vec<_>>(),
            second
                .iter()
                .map(|signal| &signal.definition.float_bytes_sha256)
                .collect::<Vec<_>>()
        );
        let report = report(&first, None);
        assert!(!report.direct_reference_input);
        assert!(!report.entered_training_corpus);
    }
}
