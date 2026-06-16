// =====================================================================
// TITAN AUDIO ECOSYSTEM — RUST EDITION (GRADIENT-COHERENT RELEASE)
// =====================================================================
//
// Fixes relative to the previous version:
//
//  1. END-TO-END DIFFERENTIABLE SYNTHESIS. Frequencies, FM ratio/index,
//     formants, pan, filter openness and metabolic blending are kept as
//     graph tensors (broadcast ops) instead of being extracted via
//     to_scalar (which detaches). The CA -> GRU -> synthesis pipeline now
//     receives real gradients from the audio losses, instead of only the
//     two KAN wavefolders learning.
//  2. REVIVED TOP-DOWN PATHWAY. NeuralCA1D now applies the metabolic
//     field AND the macro modulation multiplicatively, so macro_mod and
//     the AsymptoticContractionLayer are no longer dead code.
//  3. LIVE LOSS TERMS. movement_loss and empowerment_loss are built from
//     graph tensors, so they actually shape the dynamics.
//  4. TRAINED CONTROLLERS. The Arbiter is trained on learning progress
//     (it learns to allocate loss budget toward objectives that are
//     improving) plus its entropy regularizer. The Defibrillator gains a
//     predictive head (next movement / next mimic drift) trained by MSE,
//     so its trunk learns a real model of the system.
//  5. PROPER RG COARSE-GRAINING. Block decimation (average adjacent
//     pairs) replaces the antipodal half-fold, both in the CA's
//     multiscale branch (with aligned nearest-neighbour upsampling) and
//     in the rg_loss pooling. Scale invariance now means scale.
//  6. PHASE CONTINUITY. Carrier and modulator phases are carried
//     analytically (phi += 2*pi*f*CHUNK/SR), the FM modulator no longer
//     leaks into the carried carrier phase, and amplitude-like
//     per-chunk parameters (FM index, openness, pan gains) are ramped
//     across the chunk to remove zipper noise.
//  7. PERCEPTUAL MIMIC LOSS. Waveform MSE against randomly-positioned
//     target chunks is nearly meaningless (phase misalignment). The
//     mimic loss is now a log-magnitude spectral loss on 96 log-spaced
//     bins, computed with fixed DFT projection matrices (differentiable,
//     cheap: two 4096x96 matmuls).
//  8. K-STEP TRUNCATED BPTT (window = 4). The GRU/CA finally get
//     multi-step temporal credit assignment.
//  9. ROBUST AUDIO LOADING. Handles 16/24/32-bit int and 32-bit float
//     WAVs, downmixes stereo, linearly resamples to 48 kHz, and skips
//     files shorter than one chunk (the old version panicked).
// 10. STABLE SOFTPLUS (no exp overflow), criticality-SEEKING Choptuik
//     learning-rate law (plasticity peaks near the critical surface
//     instead of freezing on it — flip CRITICALITY_SEEKING to restore
//     the "time dilation at the horizon" behaviour), QNM damping derived
//     from a physical Q (audible ringdown instead of subtle EQ), and a
//     damped one-pole in the FDN feedback loop for natural HF decay.
// 11. SNAPDRAGON TUNING. TAPE_LEN 512->256, CA hidden 48x->8x (the conv
//     was ~0.5 GMAC per call), t_steps/ramp/DFT matrices precomputed
//     once, optimizer step every BPTT_WINDOW chunks, periodic
//     checkpointing. See build notes below.
//
// ---------------------------------------------------------------------
// Cargo.toml (unchanged deps):
//   anyhow, candle-core, candle-nn, rustfft, rand, hound, serde_json,
//   csv, rayon
//
// Build for the S25 Ultra (Termux, aarch64) — this matters a lot, it
// enables NEON/SVE codegen in the gemm kernels:
//
//   RUSTFLAGS="-C target-cpu=native" cargo build --release
//
// and in Cargo.toml:
//   [profile.release]
//   lto = "fat"
//   codegen-units = 1
//   opt-level = 3
//
// Tuning knobs, cheapest quality/speed trade first:
//   CA_HIDDEN_MULT (8 -> 4), TAPE_LEN (256 -> 128), BPTT_WINDOW (4 -> 2),
//   SPEC_BINS (96 -> 64).
// =====================================================================

use anyhow::Result;
use candle_core::{Device, Tensor, DType, D, Result as CResult};
use candle_nn::{Linear, Conv1dConfig, VarBuilder as VBV, VarMap, Optimizer, AdamW, Module};
use std::collections::VecDeque;
use rustfft::{FftPlanner, num_complex::Complex};
use rand::Rng;

// ==========================================
// ECOSYSTEM CONFIGURATION
// ==========================================
const SAMPLE_RATE: u32 = 48000;
const DURATION_SECONDS: f32 = 240.0;
const CHUNK_SIZE: usize = 4096;
const TAPE_LEN: usize = 256;          // was 512 — halves CA cost
const CA_CHANNELS: usize = 64;
const CA_HIDDEN_MULT: usize = 8;      // was 48 — ~6x cheaper conv, still expressive
const KAN_BASIS_FUNCTIONS: usize = 64;
const MEMORY_DIM: usize = 256;
const BPTT_WINDOW: usize = 4;         // truncated BPTT length (chunks)
const SPEC_BINS: usize = 96;          // log-spaced spectral-loss bins

// Physics-Driven Constants
const CHOPTUIK_EXPONENT: f32 = 0.37413;
const CRITICAL_D0: f32 = 0.02;        // distance scale of the critical surface
const CRITICALITY_SEEKING: bool = true; // true: plasticity peaks AT criticality
const LARGE_D_DIM: usize = 512;
const FDN_DELAY_LINES: usize = 4;
const FDN_DELAYS: [usize; 4] = [149, 263, 431, 701];
const QNM_Q_BASE: f32 = 40.0;         // physical resonator Q at phi = 0

const BASE_FREQ_L: f32 = 48.0;
const BASE_FREQ_R: f32 = 69.0;
const METABOLIC_DECAY: f32 = 0.999999;
const FREQ_GLIDE_SPEED: f32 = 0.0711;
const BASE_LR: f64 = 1.3e-3;
const RESONANT_AUTONOMY: f32 = 0.2;
const TWO_PI: f32 = 2.0 * std::f32::consts::PI;

// ==========================================
// SEMANTIC-FIELD INTEGRATION (from token_universe V8.3-PRIME)
// ==========================================
// Morphological plasticity: the observer's depth tracks how hard the
// target sonic field is to mimic. Implemented as depth-gating over a
// pre-allocated block pool so the VarMap/optimizer/checkpoint stay
// fixed-shape (candle has no dynamic param groups; this is the
// static-graph-friendly form of neurogenesis/pruning).
const MORPH_MAX_BLOCKS: usize = 12;          // token_universe MAX_LAYERS=60; capped for mobile
const MORPH_START_DEPTH: usize = 1;
const MORPH_PATIENCE_BASE: usize = 10;       // patience = BASE + active_depth*2
// The growth/prune decision compares a patience-window average of the
// bounded mimic signal against a baseline the program CALIBRATES from a
// warmup window (instead of magic absolute numbers, whose right value
// depends on the log-spectral scale we can't know ahead of time). The
// REL multipliers form a dead-band around that baseline, so the depth
// only changes when mimic sustains a real departure from the calibrated
// norm — it sits stable at equilibrium rather than thrashing.
const MORPH_WARMUP: usize = 48;              // observe before adapting; also calibrates the baseline
const MORPH_GROWTH_REL: f32 = 1.10;          // grow if window avg > baseline * this (struggling)
const MORPH_PRUNE_REL: f32 = 0.55;           // prune if window avg < baseline * this (mastering)

// Edge-of-chaos radiation homeostat.
const RAD_AMP_INIT: f32 = 0.8;
const RAD_AMP_MIN: f32 = 0.10;
const RAD_AMP_MAX: f32 = 0.98;
const RAD_COOL: f32 = 0.7;   // on neurogenesis (observer struggling): calm the world
const RAD_HEAT: f32 = 1.3;   // on pruning (observer mastering): agitate the world
const RADIATE_PROB: f32 = 0.12;     // per-chunk chance a radiation burst occurs
const RADIATE_SPARSITY: f32 = 0.95; // only cells above this uniform draw mutate
const CAUCHY_CLAMP: f32 = 8.0;      // bound the heavy tail so a draw can't detonate the tape
const CHAOS_LAMBDA: f32 = 3.99;     // logistic-map chaos pressure (deep chaotic regime)

// Dual-lane quantile codec alphabets (value density / gradient motion).
const VAL_SYMS: [&str; 8] = [" ", "·", "░", "▒", "▓", "█", "▪", "■"];
const GRAD_SYMS: [&str; 8] = [" ", "˙", "·", "∘", "o", "O", "◎", "●"];

// Absolute-threshold archetypes over the [0,1]-mapped field state.
const ARCHETYPES: [&str; 8] = [
    "VOID", "LATENT", "DRIFT", "NEXUS", "PULSE", "SIGNAL", "AXIOM", "SINGULARITY",
];
const ARCH_BOUNDS: [f32; 9] = [0.0, 0.13, 0.26, 0.40, 0.52, 0.65, 0.78, 0.90, 1.001];

// Epistemic phase labels keyed off the bounded mimic signal.
const PHASE_MAP: [(f32, &str); 7] = [
    (0.20, "MASTERY"),
    (0.27, "COHERENT"),
    (0.35, "CONVERGING"),
    (0.45, "LEARNING"),
    (0.55, "TURBULENT"),
    (0.70, "CHAOTIC"),
    (f32::MAX, "PRIMORDIAL"),
];

// ==========================================
// TARGET AUDIO LOADER (format-robust, resampling)
// ==========================================
/// Stereo target buffers: (left, right), both resampled to 48 kHz.
struct TargetAudioLoader {
    buffers: Vec<(Vec<f32>, Vec<f32>)>,
}

impl TargetAudioLoader {
    fn new(path: &str) -> Result<Self> {
        let mut buffers = Vec::new();
        let entries = std::fs::read_dir(path)?;
        for entry in entries {
            let entry = entry?;
            let p = entry.path();
            let is_output = p.file_name()
                .and_then(|name| name.to_str())
                .map(|name| name == "rust_ecosystem_out.wav")
                .unwrap_or(false);

            if p.extension().map_or(false, |ext| ext == "wav") && !is_output {
                match Self::load_wav(&p) {
                    Ok((l, r)) if l.len() >= CHUNK_SIZE => {
                        println!("--> Loaded target audio: {:?} ({} samples/ch @ 48k stereo)", p, l.len());
                        buffers.push((l, r));
                    }
                    Ok((l, _)) => {
                        println!("--> Skipping {:?}: only {} samples after resample (< one chunk)", p, l.len());
                    }
                    Err(e) => {
                        println!("--> Skipping {:?}: {}", p, e);
                    }
                }
            }
        }
        if buffers.is_empty() {
            anyhow::bail!("No usable training audio found in {}", path);
        }
        Ok(Self { buffers })
    }

    /// Returns (left, right) at 48 kHz. Mono files are duplicated to both
    /// channels; files with more than two channels use the first two.
    fn load_wav(p: &std::path::Path) -> Result<(Vec<f32>, Vec<f32>)> {
        let mut reader = hound::WavReader::open(p)?;
        let spec = reader.spec();

        // Decode to f32 regardless of on-disk format.
        let raw: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
            (hound::SampleFormat::Float, 32) => {
                reader.samples::<f32>().filter_map(|s| s.ok()).collect()
            }
            (hound::SampleFormat::Int, 16) => {
                reader.samples::<i16>().filter_map(|s| s.ok())
                    .map(|s| s as f32 / 32768.0).collect()
            }
            (hound::SampleFormat::Int, bits @ (24 | 32)) => {
                let scale = (1i64 << (bits - 1)) as f32;
                reader.samples::<i32>().filter_map(|s| s.ok())
                    .map(|s| s as f32 / scale).collect()
            }
            (fmt, bits) => anyhow::bail!("unsupported WAV format {:?}/{} bits", fmt, bits),
        };
        if raw.is_empty() {
            anyhow::bail!("no samples decoded");
        }

        // De-interleave to stereo — these targets ARE stereo, and the model
        // synthesizes a stereo field, so keep the image instead of downmixing.
        let ch = spec.channels as usize;
        let (left, right): (Vec<f32>, Vec<f32>) = if ch >= 2 {
            let l = raw.iter().step_by(ch).copied().collect();
            let r = raw.iter().skip(1).step_by(ch).copied().collect();
            (l, r)
        } else {
            (raw.clone(), raw)
        };

        // Linear resample each channel to SAMPLE_RATE if needed.
        if spec.sample_rate == SAMPLE_RATE {
            return Ok((left, right));
        }
        Ok((Self::resample(&left, spec.sample_rate), Self::resample(&right, spec.sample_rate)))
    }

    fn resample(x: &[f32], from_rate: u32) -> Vec<f32> {
        let ratio = SAMPLE_RATE as f64 / from_rate as f64;
        let out_len = (x.len() as f64 * ratio) as usize;
        let mut out = Vec::with_capacity(out_len);
        for i in 0..out_len {
            let pos = i as f64 / ratio;
            let i0 = pos.floor() as usize;
            let frac = (pos - i0 as f64) as f32;
            let a = x[i0.min(x.len() - 1)];
            let b = x[(i0 + 1).min(x.len() - 1)];
            out.push(a + (b - a) * frac);
        }
        out
    }

    /// Returns a stereo target chunk of shape (2, CHUNK_SIZE).
    fn sample_chunk(&self, device: &Device) -> CResult<Tensor> {
        let mut rng = rand::thread_rng();
        let buf_idx = rng.gen_range(0..self.buffers.len());
        let (l, r) = &self.buffers[buf_idx];
        // Loader guarantees l.len() >= CHUNK_SIZE, so this cannot panic.
        let start = if l.len() == CHUNK_SIZE {
            0
        } else {
            rng.gen_range(0..(l.len() - CHUNK_SIZE + 1))
        };
        let mut data = Vec::with_capacity(2 * CHUNK_SIZE);
        data.extend_from_slice(&l[start..start + CHUNK_SIZE]);
        data.extend_from_slice(&r[start..start + CHUNK_SIZE]);
        Tensor::from_vec(data, (2, CHUNK_SIZE), device)
    }
}

// ==========================================
// CUSTOM MODULES & IT UTILS
// ==========================================
struct Tanh;
impl Module for Tanh {
    fn forward(&self, xs: &Tensor) -> CResult<Tensor> { xs.tanh() }
}

struct Sigmoid;
impl Module for Sigmoid {
    fn forward(&self, xs: &Tensor) -> CResult<Tensor> { candle_nn::ops::sigmoid(xs) }
}

/// Numerically stable softplus: relu(x) + ln(1 + exp(-|x|)).
/// (The naive exp(x) form overflows for large pre-activations.)
struct Softplus;
impl Module for Softplus {
    fn forward(&self, xs: &Tensor) -> CResult<Tensor> {
        let relu = xs.relu()?;
        let neg_abs = xs.abs()?.neg()?;
        let log1p = neg_abs.exp()?.affine(1.0, 1.0)?.log()?;
        relu.add(&log1p)
    }
}

fn var_all(x: &Tensor) -> CResult<Tensor> {
    let mean = x.mean_all()?;
    let diff = x.broadcast_sub(&mean)?;
    diff.sqr()?.mean_all()
}

/// Partial checkpoint load. candle's VarMap::load only sets variables that
/// already exist in the map (and errors on any key the file lacks), so the
/// previous "load into an empty VarMap before building the model" pattern
/// was a silent no-op — the checkpoint never actually loaded. This reads
/// the safetensors directly and sets only the vars whose name AND shape
/// match, leaving new modules (e.g. the morphic stack added this revision)
/// at their fresh init. Returns (loaded, missing, shape_mismatch).
fn load_into_varmap(varmap: &VarMap, path: &str, device: &Device) -> Result<(usize, usize, usize)> {
    let loaded = candle_core::safetensors::load(path, device).map_err(anyhow::Error::msg)?;
    let data = varmap.data().lock().unwrap();
    let (mut hit, mut miss, mut mismatch) = (0usize, 0usize, 0usize);
    for (name, var) in data.iter() {
        match loaded.get(name) {
            Some(t) if t.dims() == var.as_tensor().dims() => {
                var.set(t).map_err(anyhow::Error::msg)?;
                hit += 1;
            }
            Some(_) => mismatch += 1,
            None => miss += 1,
        }
    }
    Ok((hit, miss, mismatch))
}

/// True block coarse-graining: average ADJACENT pairs and decimate.
/// (b, c, l) -> (b, c, l/2). This is the RG block-spin step; the old
/// version averaged antipodal halves, which enforces fold symmetry,
/// not scale invariance.
fn decimate2(x: &Tensor) -> CResult<Tensor> {
    let (b, c, l) = x.dims3()?;
    let half = l / 2;
    x.reshape((b, c, half, 2))?.mean(D::Minus1)
}

/// Nearest-neighbour upsample by 2, aligned with decimate2:
/// coarse cell i maps to fine cells 2i and 2i+1.
fn upsample2(x: &Tensor) -> CResult<Tensor> {
    let (b, c, h) = x.dims3()?;
    let u = x.unsqueeze(D::Minus1)?;
    Tensor::cat(&[&u, &u], D::Minus1)?.reshape((b, c, h * 2))
}

/// Gaussian mutual-information estimate (-0.5 ln(1 - r^2)) between the
/// micro and macro channel-activation profiles. Note: this measures
/// REDUNDANCY (shared information across scales), not PID-synergy.
fn calculate_cross_layer_synergy(micro: &Tensor, macro_t: &Tensor) -> CResult<f32> {
    let micro_mean = micro.mean(D::Minus1)?;
    let macro_mean = macro_t.mean(D::Minus1)?;

    let micro_norm = micro_mean.broadcast_sub(&micro_mean.mean_all()?)?;
    let macro_norm = macro_mean.broadcast_sub(&macro_mean.mean_all()?)?;

    let cross_cov = micro_norm.mul(&macro_norm)?.mean_all()?;
    let micro_var = micro_norm.sqr()?.mean_all()?.add(&Tensor::new(1e-6f32, micro.device())?)?;
    let macro_var = macro_norm.sqr()?.mean_all()?.add(&Tensor::new(1e-6f32, micro.device())?)?;

    let r_sq = cross_cov.sqr()?.broadcast_div(&(&micro_var * &macro_var)?)?;
    let correlation = r_sq.reshape(())?.to_scalar::<f32>()?;

    let synergy = -0.5 * (1.0 - correlation + 1e-5).ln();
    Ok(synergy.clamp(0.0, 5.0))
}

fn conv1d_circular(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    vb: VBV,
) -> Result<Box<dyn Fn(&Tensor) -> CResult<Tensor>>> {
    let padding = kernel_size / 2;
    let config = Conv1dConfig { padding: 0, stride: 1, dilation: 1, groups: 1 };
    let conv = candle_nn::conv1d(in_channels, out_channels, kernel_size, config, vb)?;
    Ok(Box::new(move |x: &Tensor| {
        let l = x.dim(D::Minus1)?;
        let left = x.narrow(D::Minus1, l - padding, padding)?;
        let right = x.narrow(D::Minus1, 0, padding)?;
        let padded = Tensor::cat(&[&left, x, &right], D::Minus1)?;
        conv.forward(&padded)
    }))
}

// ==========================================
// DIFFERENTIABLE SPECTRAL PROJECTOR
// ==========================================
// Fixed Hann window + cos/sin DFT bases at SPEC_BINS log-spaced
// frequencies (40 Hz .. 8 kHz). log-magnitude spectra are compared by
// MSE. This is the perceptual replacement for raw waveform MSE against
// randomly positioned target chunks, and it is fully differentiable.
struct SpectralProjector {
    window: Tensor,   // (CHUNK,)
    cos_m: Tensor,    // (CHUNK, SPEC_BINS)
    sin_m: Tensor,    // (CHUNK, SPEC_BINS)
}

impl SpectralProjector {
    fn new(device: &Device) -> CResult<Self> {
        let n = CHUNK_SIZE;
        let mut win = Vec::with_capacity(n);
        for i in 0..n {
            win.push(0.5 - 0.5 * (TWO_PI * i as f32 / (n as f32 - 1.0)).cos());
        }
        let f_lo = 40.0f32;
        let f_hi = 8000.0f32;
        let mut cos_v = vec![0.0f32; n * SPEC_BINS];
        let mut sin_v = vec![0.0f32; n * SPEC_BINS];
        for k in 0..SPEC_BINS {
            let frac = k as f32 / (SPEC_BINS as f32 - 1.0);
            let f = f_lo * (f_hi / f_lo).powf(frac);
            let omega = TWO_PI * f / SAMPLE_RATE as f32;
            for i in 0..n {
                cos_v[i * SPEC_BINS + k] = (omega * i as f32).cos();
                sin_v[i * SPEC_BINS + k] = (omega * i as f32).sin();
            }
        }
        Ok(Self {
            window: Tensor::new(win, device)?,
            cos_m: Tensor::from_vec(cos_v, (n, SPEC_BINS), device)?,
            sin_m: Tensor::from_vec(sin_v, (n, SPEC_BINS), device)?,
        })
    }

    /// x: (1, CHUNK) -> log-magnitude (1, SPEC_BINS)
    fn log_mag(&self, x: &Tensor) -> CResult<Tensor> {
        let xw = x.broadcast_mul(&self.window.unsqueeze(0)?)?;
        let re = xw.matmul(&self.cos_m)?;
        let im = xw.matmul(&self.sin_m)?;
        let mag = re.sqr()?.add(&im.sqr()?)?.affine(1.0, 1e-8)?.sqrt()?;
        mag.affine(1.0, 1e-4)?.log()
    }
}

// ==========================================
// ASYMPTOTIC DIMENSION CONTRACTION LAYER
// ==========================================
struct AsymptoticContractionLayer {
    expand_proj: Linear,
    contract_proj: Linear,
}

impl AsymptoticContractionLayer {
    fn new(in_dim: usize, hyper_dim: usize, out_dim: usize, vb: VBV) -> Result<Self> {
        let expand_proj = candle_nn::linear(in_dim, hyper_dim, vb.pp("expand"))?;
        let contract_proj = candle_nn::linear(hyper_dim, out_dim, vb.pp("contract"))?;
        Ok(Self { expand_proj, contract_proj })
    }

    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let hyper_state = self.expand_proj.forward(x)?.tanh()?;
        let scale_factor = 1.0 / (LARGE_D_DIM as f32).sqrt();
        let contracted = self.contract_proj.forward(&hyper_state)?;
        contracted.affine(scale_factor as f64, 0.0)
    }
}

// ==========================================
// QUASINORMAL MODE (QNM) RESONATOR BANK
// ==========================================
// Damping is now derived from a physical Q: d = pi * f / (Q * SR), with
// Q falling as phi rises (the old per-sample damping of 0.04+ gave ring
// times under a millisecond — coloration, not ringdown). At Q ~ 40 the
// 220 Hz mode rings for tens of milliseconds: audible decay.
struct QNMFilterBank {
    states_l: Vec<[f32; 2]>,
    states_r: Vec<[f32; 2]>,
}

impl QNMFilterBank {
    fn new() -> Self {
        Self {
            states_l: vec![[0.0, 0.0]; 3],
            states_r: vec![[0.0, 0.0]; 3],
        }
    }

    fn process(&mut self, samples_l: &mut [f32], samples_r: &mut [f32], phi: f32) {
        let q_eff = (QNM_Q_BASE / (1.0 + 0.5 * phi)).max(4.0);
        let qnm_freqs = [220.0f32, 550.0, 1200.0];

        for (idx, &freq) in qnm_freqs.iter().enumerate() {
            let damping = std::f32::consts::PI * freq / (q_eff * SAMPLE_RATE as f32);
            let omega = TWO_PI * freq / SAMPLE_RATE as f32;
            let r = (-damping).exp();
            let c1 = 2.0 * r * omega.cos();
            let c2 = -r * r;
            let input_scale = (1.0 - r) * 0.5;

            let s_l = &mut self.states_l[idx];
            for sample in samples_l.iter_mut() {
                let next_v = (*sample * input_scale) + c1 * s_l[0] + c2 * s_l[1];
                s_l[1] = s_l[0];
                s_l[0] = next_v;
                *sample = (*sample + next_v * 0.12).clamp(-1.0, 1.0);
            }

            let s_r = &mut self.states_r[idx];
            for sample in samples_r.iter_mut() {
                let next_v = (*sample * input_scale) + c1 * s_r[0] + c2 * s_r[1];
                s_r[1] = s_r[0];
                s_r[0] = next_v;
                *sample = (*sample + next_v * 0.12).clamp(-1.0, 1.0);
            }
        }
    }
}

// ==========================================
// DISCRETE SELF-SIMILAR FRACTAL FDN
// ==========================================
// Hadamard/2 mixing (orthogonal), prime delays, and now a one-pole
// lowpass in each feedback path so high frequencies decay faster than
// lows — natural-sounding tails instead of metallic ringing.
struct FractalFDN {
    buffers: Vec<VecDeque<f32>>,
    lp_states: [f32; FDN_DELAY_LINES],
}

impl FractalFDN {
    fn new() -> Self {
        let mut buffers = Vec::new();
        for &delay in &FDN_DELAYS {
            buffers.push(VecDeque::from(vec![0.0; delay]));
        }
        Self { buffers, lp_states: [0.0; FDN_DELAY_LINES] }
    }

    fn process(&mut self, samples: &mut [f32], echo_weight: f32) {
        let mix_matrix = [
            [0.5,  0.5,  0.5,  0.5],
            [0.5, -0.5,  0.5, -0.5],
            [0.5,  0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5,  0.5],
        ];
        let lp_a = 0.35; // feedback HF damping

        for sample in samples.iter_mut() {
            let mut outputs = [0.0; FDN_DELAY_LINES];
            for i in 0..FDN_DELAY_LINES {
                let raw = self.buffers[i].pop_front().unwrap_or(0.0);
                self.lp_states[i] = self.lp_states[i] * lp_a + raw * (1.0 - lp_a);
                outputs[i] = self.lp_states[i];
            }

            for i in 0..FDN_DELAY_LINES {
                let mut sum = 0.0;
                for j in 0..FDN_DELAY_LINES {
                    sum += mix_matrix[i][j] * outputs[j];
                }
                self.buffers[i].push_back(*sample + sum * (0.42 * echo_weight));
            }

            let fdn_out = (outputs[0] + outputs[1] + outputs[2] + outputs[3]) * 0.25;
            *sample = *sample * (1.0 - echo_weight * 0.2) + fdn_out * (echo_weight * 0.4);
        }
    }
}

// ==========================================
// 1. FRACTAL NEURAL CA (Scale-Invariant 1D)
// ==========================================
struct NeuralCA1D {
    rule: Box<dyn Fn(&Tensor) -> CResult<Tensor>>,
    mutate: Linear,
    ln: candle_nn::LayerNorm,
}

impl NeuralCA1D {
    fn new(channels: usize, hidden_mult: usize, vb: VBV) -> Result<Self> {
        let hidden_dim = channels * hidden_mult;
        let rule = conv1d_circular(channels, hidden_dim, 5, vb.pp("rule"))?;
        let mutate = candle_nn::linear(hidden_dim, channels, vb.pp("mutate"))?;
        let ln = candle_nn::layer_norm(channels, 1e-5, vb.pp("ln"))?;
        Ok(Self { rule, mutate, ln })
    }

    fn apply_rule(&self, x: &Tensor) -> CResult<Tensor> {
        let neighborhood = (self.rule)(x)?.sin()?;
        let neighborhood_t = neighborhood.transpose(1, 2)?;
        self.mutate.forward(&neighborhood_t)?.transpose(1, 2)?.tanh()
    }

    /// Both modulation channels now coexist: the metabolic field sets the
    /// baseline decay/speed, and macro_mod (top-down causation through the
    /// AsymptoticContractionLayer) modulates them multiplicatively. In the
    /// old version the if/else-if made macro_mod dead code whenever a
    /// metabolic field was present — which was always.
    fn forward(&self, x: &Tensor, macro_mod: Option<&Tensor>, metabolic_field: Option<&Tensor>) -> CResult<Tensor> {
        let mut delta = self.apply_rule(x)?;

        // True multiscale branch: run the same rule on a block-decimated
        // copy (genuinely coarser scale), then upsample position-aligned.
        if x.dim(D::Minus1)? >= 4 {
            let coarse_x = decimate2(x)?;
            let coarse_delta = self.apply_rule(&coarse_x)?;
            let upscaled = upsample2(&coarse_delta)?;
            if upscaled.shape() == delta.shape() {
                delta = delta.affine(0.618, 0.0)?.add(&upscaled.affine(0.382, 0.0)?)?;
            }
        }

        let mut decay = match metabolic_field {
            Some(m_field) => m_field.broadcast_as(x.shape())?,
            None => Tensor::new(METABOLIC_DECAY, x.device())?.broadcast_as(x.shape())?,
        };
        let mut evolution_speed = Tensor::new(0.25f32, x.device())?.broadcast_as(x.shape())?;

        if let Some(m_mod) = macro_mod {
            let sig = candle_nn::ops::sigmoid(&m_mod.unsqueeze(D::Minus1)?)?; // (1, C, 1)
            // decay modulated within ~[0.98, 1.00], speed within [0.2x, 1.8x]
            decay = decay.broadcast_mul(&sig.affine(0.02, 0.98)?)?;
            evolution_speed = evolution_speed.broadcast_mul(&sig.affine(1.6, 0.2)?)?;
        }

        let res = x.mul(&decay)?.add(&delta.mul(&evolution_speed)?)?;

        let res_t = res.transpose(1, 2)?;
        let ln_out = self.ln.forward(&res_t)?.transpose(1, 2)?;

        let part1 = res.affine(0.3, 0.0)?;
        let part2 = ln_out.affine(0.7, 0.0)?;
        let normalized = part1.add(&part2)?;

        let anti_stagnation = Tensor::randn(0.0f32, 1.0f32, x.shape(), x.device())?.affine(0.005, 0.0)?;
        normalized.add(&anti_stagnation)?.clamp(-1.0, 1.0)
    }
}

// ==========================================
// 2. TAPE CODEC (Dual-Lane: Value + Signed Gradient)
// ==========================================
#[allow(dead_code)] // superseded by the quantile dual-lane codec; kept for reference
struct TapeCodec;
impl TapeCodec {
    fn encode(values: &[f32], gradients: &[f32]) -> String {
        let chars = [" ", "·", "▪", "▒", "▓", "█"];
        let mut tape = String::new();
        for (v, g) in values.iter().zip(gradients.iter()) {
            let v_idx = (((v + 1.0) * 0.5) * (chars.len() - 1) as f32).round() as usize;
            tape.push_str(chars[v_idx.min(chars.len() - 1)]);
            // Direction now carries sign, not just magnitude.
            let mag = g.abs();
            let arrow = if mag <= 0.05 { " " }
                else if mag > 0.5 { "!" }
                else if *g > 0.2 { "↑" }
                else if *g > 0.0 { "↗" }
                else if *g < -0.2 { "↓" }
                else { "↘" };
            tape.push_str(arrow);
        }
        tape
    }
}

// ==========================================
// MORPHIC STACK — neurogenesis / pruning (from token_universe)
// ==========================================
// GELU via its standard sigmoid approximation x * sigmoid(1.702 x). Uses
// only candle_nn::ops::sigmoid + basic tensor ops that this codebase
// already relies on, so it doesn't depend on Tensor::gelu() being present
// in whatever candle build is installed. Max abs error vs exact GELU is
// ~0.02 — immaterial for a hidden activation.
fn gelu_approx(x: &Tensor) -> CResult<Tensor> {
    let g = candle_nn::ops::sigmoid(&x.affine(1.702, 0.0)?)?;
    x.mul(&g)
}

// A residual block: x + gelu(linear(norm(x))), token_universe's ResBlock.
// (Orthogonal init isn't in candle's standard kit; default init is used.)
struct ResBlock {
    linear: Linear,
    norm: candle_nn::LayerNorm,
}
impl ResBlock {
    fn new(dim: usize, vb: VBV) -> Result<Self> {
        let linear = candle_nn::linear(dim, dim, vb.pp("linear"))?;
        let norm = candle_nn::layer_norm(dim, 1e-5, vb.pp("norm"))?;
        Ok(Self { linear, norm })
    }
    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let h = gelu_approx(&self.linear.forward(&self.norm.forward(x)?)?)?;
        x.add(&h)
    }
}

// The observer's adjustable representational depth. ALL blocks are
// allocated up front (fixed VarMap); only the first `active_depth` run in
// the forward graph, so growth/pruning never reshapes the checkpoint and
// never needs the optimizer rebuilt. Inactive blocks receive no gradient
// and are frozen until activated — the static-graph form of the Python
// version's grow()/prune() on an nn.ModuleList.
struct MorphicStack {
    blocks: Vec<ResBlock>,
    active_depth: usize,
}
impl MorphicStack {
    fn new(dim: usize, max_blocks: usize, vb: VBV) -> Result<Self> {
        let mut blocks = Vec::with_capacity(max_blocks);
        for i in 0..max_blocks {
            blocks.push(ResBlock::new(dim, vb.pp(format!("block_{i}")))?);
        }
        Ok(Self { blocks, active_depth: MORPH_START_DEPTH.clamp(1, max_blocks) })
    }
    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let mut out = x.clone();
        for block in self.blocks.iter().take(self.active_depth) {
            out = block.forward(&out)?;
        }
        Ok(out)
    }
    fn grow(&mut self) -> bool {
        if self.active_depth < self.blocks.len() { self.active_depth += 1; true } else { false }
    }
    fn prune(&mut self) -> bool {
        if self.active_depth > 1 { self.active_depth -= 1; true } else { false }
    }
}

// ==========================================
// LÉVY / CAUCHY RADIATION (from token_universe AdaptiveUniverse.radiate)
// ==========================================
// Sparse, heavy-tailed mutation of the substrate, scaled by rad_amp. Each
// radiated cell also takes one step toward the logistic map's image: we
// map the cell v in [-1,1] to p in [0,1], apply f(p)=lambda*p*(1-p), map
// the result back to [-1,1] as v_next, and nudge toward (v_next - v).
// Doing the whole thing in v-space keeps it dimensionally consistent
// (the earlier form mixed a [0,1] deviation into a [-1,1] value, which
// was directionally right but half-magnitude). This is a kick along the
// logistic (edge-of-chaos) dynamics, not convergence to a fixed point.
fn levy_radiate(tape: &Tensor, rad_amp: f32) -> CResult<Tensor> {
    let (b, c, l) = tape.dims3()?;
    let mut data = tape.flatten_all()?.to_vec1::<f32>()?;
    let mut rng = rand::thread_rng();
    for v in data.iter_mut() {
        if rng.gen::<f32>() > RADIATE_SPARSITY {
            let u: f32 = rng.gen::<f32>();
            let cauchy = (std::f32::consts::PI * (u - 0.5)).tan().clamp(-CAUCHY_CLAMP, CAUCHY_CLAMP);
            let p = (*v + 1.0) * 0.5;                                  // [-1,1] -> [0,1]
            let v_next = 2.0 * (CHAOS_LAMBDA * p * (1.0 - p)) - 1.0;   // logistic image, back in [-1,1]
            let logistic_dev = v_next - *v;                           // deviation toward the chaotic map
            *v = (*v + rad_amp * (0.6 * cauchy + 0.4 * logistic_dev)).clamp(-1.0, 1.0);
        }
    }
    Tensor::from_vec(data, (b, c, l), tape.device())
}

// ==========================================
// QUANTILE DUAL-LANE CODEC (from token_universe)
// ==========================================
// Quantile encoding guarantees the full alphabet is used every frame, so
// the tape stays legible whether the field is near-uniform or spiky —
// unlike absolute binning, which collapses to one glyph in calm regimes.
fn qencode(arr: &[f32], syms: &[&str]) -> String {
    let n = syms.len();
    let mut sorted = arr.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let last = sorted.len().saturating_sub(1);
    // Degenerate (flat) field: all quantile breakpoints collapse to one
    // value, and `v >= qs[i]` would push every cell to the TOP glyph — i.e.
    // a perfectly calm field would read as maximum turbulence. Map it to
    // the lowest (calm/empty) glyph instead.
    if last == 0 || (sorted[last] - sorted[0]).abs() < 1e-9 {
        return std::iter::repeat(syms[0]).take(arr.len()).collect();
    }
    let qs: Vec<f32> = (1..n)
        .map(|k| {
            let pos = (k as f32 / n as f32) * last as f32;
            sorted[(pos.floor() as usize).min(last)]
        })
        .collect();
    arr.iter()
        .map(|&v| {
            let mut idx = 0;
            while idx < qs.len() && v >= qs[idx] { idx += 1; }
            syms[idx.min(n - 1)]
        })
        .collect()
}

fn quantile_dual_tape(pooled: &[f32]) -> (String, String) {
    let val_lane = qencode(pooled, &VAL_SYMS);
    let mut grad = vec![0.0f32; pooled.len()];
    for i in 0..pooled.len() {
        let prev = if i == 0 { pooled[pooled.len() - 1] } else { pooled[i - 1] };
        grad[i] = (pooled[i] - prev).abs();
    }
    let grad_lane = qencode(&grad, &GRAD_SYMS);
    (val_lane, grad_lane)
}

// ==========================================
// SEMANTIC FIELD — phase / archetype / entropy / commentary narrator
// ==========================================
// The epistemic-narration layer ported from token_universe. It reads the
// substrate and the mimic signal and produces the phase vocabulary,
// archetype histogram, Shannon entropy (bits), trend arrow, and a rolling
// commentary line — all of which also seed the final priming prompt.
struct SemanticField {
    mimic_window: VecDeque<f32>,
    cmt_idx: std::collections::HashMap<&'static str, usize>,
    phase_counts: std::collections::HashMap<&'static str, usize>,
    archetype_counts: [u64; 8],
}
impl SemanticField {
    fn new() -> Self {
        Self {
            mimic_window: VecDeque::with_capacity(40),
            cmt_idx: std::collections::HashMap::new(),
            phase_counts: std::collections::HashMap::new(),
            archetype_counts: [0; 8],
        }
    }

    fn phase(mimic: f32) -> &'static str {
        for (thresh, label) in PHASE_MAP.iter() {
            if mimic < *thresh { return label; }
        }
        "PRIMORDIAL"
    }

    /// (top-3 summary, Shannon entropy in bits, dominant archetype index)
    fn archetype_field(state01: &[f32]) -> (String, f32, usize) {
        let mut counts = [0u64; 8];
        for &v in state01 {
            let v = v.clamp(0.0, 1.0);
            let mut idx = 0;
            for b in 1..9 {
                if v < ARCH_BOUNDS[b] { idx = b - 1; break; }
            }
            counts[idx] += 1;
        }
        let total: u64 = counts.iter().sum::<u64>().max(1);
        let mut h = 0.0f32;
        for &c in &counts {
            if c > 0 {
                let p = c as f32 / total as f32;
                h -= p * p.log2();
            }
        }
        let mut order: Vec<usize> = (0..8).collect();
        order.sort_by(|&a, &b| counts[b].cmp(&counts[a]));
        let summary = order.iter().take(3)
            .filter(|&&k| counts[k] > 0)
            .map(|&k| format!("{}:{}%", ARCHETYPES[k], 100 * counts[k] / total))
            .collect::<Vec<_>>().join(" ");
        (summary, h, order[0])
    }

    fn trend(&self) -> &'static str {
        if self.mimic_window.len() < 6 { return "→"; }
        let recent = self.mimic_window[self.mimic_window.len() - 1];
        let past = self.mimic_window[self.mimic_window.len() - 6];
        let slope = (recent - past) / past.abs().max(1e-9);
        if slope < -0.15 { "↓↓" } else if slope < -0.03 { "↓" }
        else if slope > 0.15 { "↑↑" } else if slope > 0.03 { "↑" }
        else { "─" }
    }

    fn commentary(&mut self, phase: &'static str, event: Option<&str>, rad_amp: f32, depth: usize) -> String {
        if let Some(ev) = event {
            return match ev {
                "NEUROGENESIS" => format!(
                    "depth threshold breached · L:{:02} online · new representational layer · rad cooling to {:.3}",
                    depth, rad_amp),
                "PRUNING" => format!(
                    "complexity reduced · distilled to L:{:02} · lean manifold · rad ascending to {:.3}",
                    depth, rad_amp),
                _ => String::new(),
            };
        }
        let pool: &[&str] = match phase {
            "PRIMORDIAL" => &[
                "observer initializing · high entropy regime · first gradients forming",
                "universe in maximum disorder · manifold has no map · all paths open",
                "null prior · substrate awaiting first imprint",
            ],
            "CHAOTIC" => &[
                "Lévy pressure dominant · universe resists prediction · surprise maximal",
                "signal overwhelms structure · compression failing · depth required",
                "observer bandwidth saturated · loss landscape steep · gradients unstable",
            ],
            "TURBULENT" => &[
                "surprise above threshold · manifold actively restructuring",
                "attractor basins forming · gradient field pulling toward order",
                "observer acquiring new invariants · engaging field complexity",
            ],
            "LEARNING" => &[
                "surprise collapsing · first stable attractors crystallizing",
                "manifold mapping the underlying grammar of chaos",
                "loss curvature flattening · attractor landscape sharpening",
            ],
            "CONVERGING" => &[
                "deep attractor basins forming · surprise approaching irreducible floor",
                "observer and universe approaching mutual invariance",
                "Kolmogorov complexity of the map approaching that of the territory",
            ],
            "COHERENT" => &[
                "manifold has mapped the substrate · surprise near minimum",
                "attractor fully formed · observer resonating with universe rhythm",
                "free energy minimized · observer model compressed and stable",
            ],
            "MASTERY" => &[
                "observer crystallized · universe internalized · PRUNE eligible",
                "maximum compression achieved · the map has become the territory",
                "all extractable structure extracted · model is a mirror of the field",
            ],
            _ => &[],
        };
        if pool.is_empty() { return String::new(); }
        let idx = *self.cmt_idx.get(phase).unwrap_or(&0);
        self.cmt_idx.insert(phase, idx + 1);
        pool[idx % pool.len()].to_string()
    }

    /// Records per-step stats for the end-of-run priming summary.
    fn record(&mut self, mimic: f32, phase: &'static str, dom_archetype: usize) {
        self.mimic_window.push_back(mimic);
        if self.mimic_window.len() > 40 { self.mimic_window.pop_front(); }
        *self.phase_counts.entry(phase).or_insert(0) += 1;
        self.archetype_counts[dom_archetype] += 1;
    }

    fn dominant_phase(&self) -> &'static str {
        self.phase_counts.iter().max_by_key(|(_, &c)| c).map(|(&p, _)| p).unwrap_or("PRIMORDIAL")
    }
    fn dominant_archetype(&self) -> &'static str {
        let mut best = 0;
        for i in 1..8 { if self.archetype_counts[i] > self.archetype_counts[best] { best = i; } }
        ARCHETYPES[best]
    }
}

// ==========================================
// 3. KAN LAYER (Dynamic Adaptive Wavefolder)
// ==========================================
// Centers gain a trainable offset (adaptive grid, in the spirit of real
// KAN grid refinement) on top of the fixed linspace base.
struct KANLayer {
    centers_base: Tensor,
    center_shift: Tensor,
    weights: Tensor,
    variance: Tensor,
}

impl KANLayer {
    fn new(in_features: usize, out_features: usize, num_basis: usize, vb: VBV) -> Result<Self> {
        let mut c_vec = Vec::with_capacity(num_basis);
        for i in 0..num_basis {
            c_vec.push(-1.0 + 2.0 * (i as f32) / (num_basis as f32 - 1.0));
        }
        let centers_base = Tensor::new(c_vec, vb.device())?.reshape((1, 1, num_basis))?;
        let center_shift = vb.get_with_hints((1, 1, num_basis), "center_shift", candle_nn::Init::Const(0.0))?;
        let weights = vb.get_with_hints((out_features, in_features, num_basis), "weights", candle_nn::Init::Randn { mean: 0.0, stdev: 0.2 })?;
        let variance = vb.get_with_hints((1,), "variance", candle_nn::Init::Const(0.5))?;
        Ok(Self { centers_base, center_shift, weights, variance })
    }

    fn forward(&self, x: &Tensor, work: f32) -> CResult<Tensor> {
        let centers = self.centers_base.add(&self.center_shift)?;

        let mean = x.mean_all()?;
        let std_dev = x.broadcast_sub(&mean)?.sqr()?.mean_all()?.sqrt()?.add(&Tensor::new(1e-5f32, x.device())?)?;
        let x_bounded = x.broadcast_sub(&mean)?.broadcast_div(&std_dev)?;

        let x_expanded = x_bounded.unsqueeze(D::Minus1)?;
        let diff = x_expanded.broadcast_sub(&centers)?;

        let work_mod = ((-5.0 * work).exp() as f64).max(0.1);
        let var_sq = self.variance.sqr()?.affine(work_mod, 1e-4)?;

        let phi = diff.sqr()?.broadcast_div(&var_sq)?.neg()?.exp()?;
        let weights_reshaped = self.weights.reshape((self.weights.dim(0)?, self.weights.dim(2)?))?;
        let folded = phi.matmul(&weights_reshaped.transpose(0, 1)?.unsqueeze(0)?)?;

        let secondary_diff = folded.broadcast_sub(&centers)?;
        let secondary_phi = secondary_diff.sqr()?.broadcast_div(&var_sq)?.neg()?.exp()?;
        let refined_fold = secondary_phi.matmul(&weights_reshaped.transpose(0, 1)?.unsqueeze(0)?)?;

        let part1 = folded.affine(0.7, 0.0)?;
        let part2 = refined_fold.affine(0.3, 0.0)?;
        part1.add(&part2)
    }
}

// ==========================================
// 4. MONITOR AGENTS
// ==========================================
struct SpectralEntropyMonitor {
    history: VecDeque<f32>,
    window: usize,
}

impl SpectralEntropyMonitor {
    fn new(window: usize) -> Self { Self { history: VecDeque::with_capacity(window), window } }
    fn analyze(&mut self, stereo_chunk: &Tensor) -> Result<serde_json::Value> {
        let mono = stereo_chunk.mean(0)?.reshape((CHUNK_SIZE,))?.to_vec1::<f32>()?;
        let n = mono.len();
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n);
        let mut buffer: Vec<Complex<f32>> = mono.iter().map(|&x| Complex::new(x, 0.0)).collect();
        fft.process(&mut buffer);
        let magnitudes: Vec<f32> = buffer.iter().take(n / 2).map(|c| c.norm()).collect();
        let sum_mag: f32 = magnitudes.iter().sum::<f32>() + 1e-8;
        let mut entropy = 0.0;
        for m in magnitudes {
            let p = m / sum_mag;
            if p > 1e-7 { entropy -= p * p.ln(); }
        }
        entropy /= 2048.0_f32.ln();
        self.history.push_back(entropy);
        if self.history.len() > self.window { self.history.pop_front(); }
        let avg_entropy: f32 = self.history.iter().sum::<f32>() / self.history.len() as f32;
        Ok(serde_json::json!({"signal": entropy, "avg": avg_entropy, "trigger": entropy < (3.0 / 2048.0_f32.ln()), "type": "spectral_entropy"}))
    }
}

struct MovementCoherenceMonitor {
    history: VecDeque<f32>,
    window: usize,
}

impl MovementCoherenceMonitor {
    fn new(window: usize) -> Self { Self { history: VecDeque::with_capacity(window), window } }
    fn analyze(&mut self, movement: f32) -> Result<serde_json::Value> {
        self.history.push_back(movement);
        if self.history.len() > self.window { self.history.pop_front(); }
        let mut trigger = false;
        let mut trend = 0.0;
        if self.history.len() >= 10 {
            let n = self.history.len() as f32;
            let x_mean = (n - 1.0) / 2.0;
            let y_mean: f32 = self.history.iter().sum::<f32>() / n;
            let mut num = 0.0;
            let mut den = 0.0;
            for (i, &y) in self.history.iter().enumerate() {
                let dx = i as f32 - x_mean;
                num += dx * (y - y_mean);
                den += dx * dx;
            }
            trend = num / (den + 1e-8);
            trigger = trend < -0.001;
        }
        Ok(serde_json::json!({"signal": movement, "trend": trend, "trigger": trigger, "type": "movement_coherence"}))
    }
}

// ==========================================
// 5. AUDIO UNCERTAINTY STATE
// ==========================================
struct AudioUncertaintyState {
    spectral: f32,
    movement: f32,
    mimic: f32,
    compositional: f32,
    phi: f32,
    synergy: f32,
    empowerment: f32,
}

impl AudioUncertaintyState {
    fn new() -> Self {
        Self {
            spectral: 0.0, movement: 0.0, mimic: 0.0, compositional: 0.0,
            phi: 0.0, synergy: 0.0, empowerment: 0.0
        }
    }
    fn update(&mut self, spectral_sig: &serde_json::Value, movement_sig: &serde_json::Value, mimic_sig: Option<&serde_json::Value>, synergy_val: f32, empowerment_val: f32) {
        let s_sig = spectral_sig["signal"].as_f64().unwrap_or(0.0) as f32;
        let avg_s = spectral_sig["avg"].as_f64().unwrap_or(1.0) as f32;
        let m_trend = movement_sig["trend"].as_f64().unwrap_or(0.0) as f32;

        let resonance = (avg_s / (s_sig + 1e-6)).clamp(0.1, 5.0);
        self.phi = (s_sig * resonance).clamp(0.0, 10.0);

        self.spectral = (1.0 - s_sig).max(0.0);
        self.movement = (-m_trend * 200.0).max(0.0);
        if let Some(ms) = mimic_sig {
            let drift = ms["drift"].as_f64().unwrap_or(0.0) as f32;
            self.mimic = (drift * 10.0).max(0.0);
        }
        self.synergy = synergy_val;
        self.empowerment = empowerment_val;

        self.compositional = (self.compositional * 0.92) + (self.spectral.max(self.movement) * 0.08);
        self.compositional = self.compositional.min(1.0);
    }
    fn branch_aperture(&self) -> f32 {
        let raw = (self.spectral * 0.20) + (self.movement * 0.25) + (self.mimic * 0.15) + (self.compositional * 0.10) + (self.synergy * 0.30);
        raw.clamp(0.05, 1.0)
    }
}

// ==========================================
// 6. GRU CELL
// ==========================================
struct GRUCell {
    w_ih: Linear,
    w_hh: Linear,
    ln: candle_nn::LayerNorm,
}

impl GRUCell {
    fn new(input_size: usize, hidden_size: usize, vb: VBV) -> Result<Self> {
        let w_ih = candle_nn::linear(input_size, 3 * hidden_size, vb.pp("w_ih"))?;
        let w_hh = candle_nn::linear(hidden_size, 3 * hidden_size, vb.pp("w_hh"))?;
        let ln = candle_nn::layer_norm(hidden_size, 1e-5, vb.pp("ln"))?;
        Ok(Self { w_ih, w_hh, ln })
    }

    fn forward(&self, x: &Tensor, h: &Tensor) -> CResult<Tensor> {
        let hidden_size = h.dim(D::Minus1)?;
        let gi = self.w_ih.forward(x)?;
        let gh = self.w_hh.forward(h)?;
        let i_r = gi.narrow(D::Minus1, 0, hidden_size)?;
        let i_z = gi.narrow(D::Minus1, hidden_size, hidden_size)?;
        let i_n = gi.narrow(D::Minus1, 2 * hidden_size, hidden_size)?;
        let h_r = gh.narrow(D::Minus1, 0, hidden_size)?;
        let h_z = gh.narrow(D::Minus1, hidden_size, hidden_size)?;
        let h_n = gh.narrow(D::Minus1, 2 * hidden_size, hidden_size)?;
        let reset_gate = candle_nn::ops::sigmoid(&i_r.add(&h_r)?)?;
        let update_gate = candle_nn::ops::sigmoid(&i_z.add(&h_z)?)?;
        let new_gate = i_n.add(&reset_gate.broadcast_mul(&h_n)?)?.tanh()?;
        let h_next = (update_gate.affine(-1.0, 1.0)?.broadcast_mul(&new_gate)? + update_gate.broadcast_mul(h)?)?;
        self.ln.forward(&h_next)
    }
}

// ==========================================
// SELF-MODEL MONITOR HEAD
// ==========================================
struct MonitorHead {
    net: candle_nn::Sequential,
}

impl MonitorHead {
    fn new(vb: VBV) -> Result<Self> {
        let net = candle_nn::seq()
            .add(candle_nn::linear(MEMORY_DIM, 32, vb.pp("net_0"))?)
            .add(candle_nn::Activation::Relu)
            .add(candle_nn::linear(32, 5, vb.pp("net_2"))?)
            .add(Sigmoid);
        Ok(Self { net })
    }

    fn forward(&self, hidden_mem: &Tensor) -> CResult<Tensor> {
        self.net.forward(hidden_mem)
    }
}

// ==========================================
// 7. COMPLEX AUDIO ECOSYSTEM (Titan Engine)
// ==========================================
// All synthesis parameters are graph tensors now; only the carried
// phases and ramp baselines are scalar STATE (detached by design — they
// are initial conditions of the next chunk, not learnable quantities).
struct ComplexAudioEcosystem {
    micro_ca: NeuralCA1D,
    macro_ca: NeuralCA1D,
    gru_memory: GRUCell,
    morphic: MorphicStack,
    asymptotic_contraction: AsymptoticContractionLayer,
    spatial_panner: candle_nn::Sequential,
    fm_mod_ratio: candle_nn::Sequential,
    fm_mod_index: candle_nn::Sequential,
    wavefolder_l: KANLayer,
    wavefolder_r: KANLayer,
    base_freq_l: Tensor,
    base_freq_r: Tensor,
    // Precomputed constants
    t_steps: Tensor, // (CHUNK,) seconds
    ramp: Tensor,    // (CHUNK,) 0..1 linear, for zipper-free param ramps
    // Scalar dynamical state (carried, detached by design)
    current_freq_l: f32,
    current_freq_r: f32,
    current_mod_freq_l: f32,
    current_mod_freq_r: f32,
    prev_fm_idx_l: f32,
    prev_fm_idx_r: f32,
    prev_openness: f32,
    prev_gain_l: f32,
    prev_gain_r: f32,
    prev_theta: f32,
}

struct ForwardOut {
    stereo: Tensor,        // (2, CHUNK), in graph
    next_micro: Tensor,
    next_macro: Tensor,
    next_hidden: Tensor,   // raw GRU memory — carried as recurrent state
    refined_hidden: Tensor,// morphic-stack readout — drives heads + monitor
    movement_t: Tensor,    // 0-dim, in graph
    theta: f32,
}

impl ComplexAudioEcosystem {
    fn new(vb: VBV, device: &Device) -> Result<Self> {
        let micro_ca = NeuralCA1D::new(CA_CHANNELS, CA_HIDDEN_MULT, vb.pp("micro_ca"))?;
        let macro_ca = NeuralCA1D::new(CA_CHANNELS, CA_HIDDEN_MULT, vb.pp("macro_ca"))?;
        let gru_memory = GRUCell::new(CA_CHANNELS, MEMORY_DIM, vb.pp("gru_memory"))?;
        let morphic = MorphicStack::new(MEMORY_DIM, MORPH_MAX_BLOCKS, vb.pp("morphic"))?;
        let asymptotic_contraction = AsymptoticContractionLayer::new(MEMORY_DIM, LARGE_D_DIM, CA_CHANNELS, vb.pp("asymp_contract"))?;
        let spatial_panner = candle_nn::seq().add(candle_nn::linear(MEMORY_DIM, 1, vb.pp("spatial_panner_0"))?).add(Tanh);
        let fm_mod_ratio = candle_nn::seq().add(candle_nn::linear(MEMORY_DIM, 2, vb.pp("fm_mod_ratio_0"))?).add(Softplus);
        let fm_mod_index = candle_nn::seq().add(candle_nn::linear(MEMORY_DIM, 2, vb.pp("fm_mod_index_0"))?).add(Sigmoid);
        let wavefolder_l = KANLayer::new(1, 1, KAN_BASIS_FUNCTIONS, vb.pp("wavefolder_l"))?;
        let wavefolder_r = KANLayer::new(1, 1, KAN_BASIS_FUNCTIONS, vb.pp("wavefolder_r"))?;
        let base_freq_l = vb.get_with_hints((1,), "base_freq_l", candle_nn::Init::Const(BASE_FREQ_L as f64))?;
        let base_freq_r = vb.get_with_hints((1,), "base_freq_r", candle_nn::Init::Const(BASE_FREQ_R as f64))?;

        // Hoisted constants (the old version rebuilt t_steps every chunk).
        let steps_vec: Vec<f32> = (0..CHUNK_SIZE).map(|i| i as f32 / SAMPLE_RATE as f32).collect();
        let t_steps = Tensor::new(steps_vec, device)?;
        let ramp_vec: Vec<f32> = (0..CHUNK_SIZE).map(|i| i as f32 / (CHUNK_SIZE as f32 - 1.0)).collect();
        let ramp = Tensor::new(ramp_vec, device)?;

        Ok(Self {
            micro_ca, macro_ca, gru_memory, morphic, asymptotic_contraction, spatial_panner, fm_mod_ratio, fm_mod_index,
            wavefolder_l, wavefolder_r, base_freq_l, base_freq_r,
            t_steps, ramp,
            current_freq_l: BASE_FREQ_L, current_freq_r: BASE_FREQ_R,
            current_mod_freq_l: BASE_FREQ_L, current_mod_freq_r: BASE_FREQ_R,
            prev_fm_idx_l: 0.0, prev_fm_idx_r: 0.0,
            prev_openness: 0.7, prev_gain_l: 0.707, prev_gain_r: 0.707,
            prev_theta: 0.0,
        })
    }

    /// Ramp a per-chunk-constant parameter from its previous value to the
    /// new (in-graph) value across the chunk. Returns a (CHUNK,) tensor;
    /// gradient flows through `new_val`.
    fn ramp_param(&self, new_val: &Tensor, prev_val: f32) -> CResult<Tensor> {
        let delta = new_val.affine(1.0, -(prev_val as f64))?; // new - prev (0-dim)
        self.ramp.broadcast_mul(&delta)?.affine(1.0, prev_val as f64)
    }

    fn grow(&mut self) -> bool { self.morphic.grow() }
    fn prune(&mut self) -> bool { self.morphic.prune() }
    fn depth(&self) -> usize { self.morphic.active_depth }
    fn set_depth(&mut self, d: usize) {
        self.morphic.active_depth = d.clamp(1, self.morphic.blocks.len());
    }

    fn forward(
        &mut self,
        micro_tape: &Tensor,
        macro_tape: &Tensor,
        hidden_mem: &Tensor,
        phase_c_l: f32, phase_c_r: f32,
        phase_m_l: f32, phase_m_r: f32,
        force_macro_update: bool,
        mimic_loss_val: f32,
    ) -> Result<(ForwardOut, f32, f32, f32, f32)> {
        let device = micro_tape.device();

        // ---- Macro CA (slow timescale) ----
        let mut next_macro = macro_tape.clone();
        if force_macro_update {
            let m_field = macro_tape.abs().map_err(anyhow::Error::msg)?
                .affine(-0.01, METABOLIC_DECAY as f64).map_err(anyhow::Error::msg)?;
            next_macro = self.macro_ca.forward(macro_tape, None, Some(&m_field)).map_err(anyhow::Error::msg)?;
        }

        // Metabolic rate stays in-graph (was a detached scalar before).
        let macro_act_t = next_macro.abs().map_err(anyhow::Error::msg)?.mean_all().map_err(anyhow::Error::msg)?;
        let metab_rate_t = macro_act_t.affine(5.0, 0.0).map_err(anyhow::Error::msg)?
            .clamp(0.01f32, 1.0f32).map_err(anyhow::Error::msg)?;
        let inv_rate_t = metab_rate_t.affine(-1.0, 1.0).map_err(anyhow::Error::msg)?;

        // ---- Top-down modulation (now actually wired in) ----
        let contracted_mem = self.asymptotic_contraction.forward(hidden_mem).map_err(anyhow::Error::msg)?;
        let macro_mod = contracted_mem.add(&next_macro.mean(D::Minus1).map_err(anyhow::Error::msg)?)?;

        // ---- Micro CA (fast timescale) ----
        let micro_m_field = micro_tape.abs().map_err(anyhow::Error::msg)?
            .affine(-0.01, METABOLIC_DECAY as f64).map_err(anyhow::Error::msg)?;
        let raw_next_micro = self.micro_ca.forward(micro_tape, Some(&macro_mod), Some(&micro_m_field)).map_err(anyhow::Error::msg)?;

        let next_micro = micro_tape.broadcast_mul(&inv_rate_t).map_err(anyhow::Error::msg)?
            .add(&raw_next_micro.broadcast_mul(&metab_rate_t).map_err(anyhow::Error::msg)?)?
            .clamp(-1.0f32, 1.0f32).map_err(anyhow::Error::msg)?;

        // ---- Observables (graph tensors) ----
        let movement_t = next_micro.sub(micro_tape).map_err(anyhow::Error::msg)?
            .abs().map_err(anyhow::Error::msg)?.mean_all().map_err(anyhow::Error::msg)?;
        let pop_l_t = next_micro.narrow(1, 0, 1).map_err(anyhow::Error::msg)?.mean_all().map_err(anyhow::Error::msg)?;
        let pop_r_t = next_micro.narrow(1, 1, 1).map_err(anyhow::Error::msg)?.mean_all().map_err(anyhow::Error::msg)?;

        // Kuramoto-style order parameter (kept as scalar neuromodulation:
        // it is a phase offset, and atan2 isn't worth the graph cost).
        let micro_vec = next_micro.mean(D::Minus1).map_err(anyhow::Error::msg)?
            .reshape((CA_CHANNELS,)).map_err(anyhow::Error::msg)?.to_vec1::<f32>().map_err(anyhow::Error::msg)?;
        let mut sum_real = 0.0f32;
        let mut sum_imag = 0.0f32;
        for i in (0..CA_CHANNELS).step_by(2) {
            sum_real += micro_vec[i];
            sum_imag += micro_vec[i + 1];
        }
        let theta = sum_imag.atan2(sum_real + 1e-6);

        // ---- Memory ----
        let tape_features = next_micro.mean(D::Minus1).map_err(anyhow::Error::msg)?;
        let next_hidden = self.gru_memory.forward(&tape_features, hidden_mem).map_err(anyhow::Error::msg)?;
        // Morphic readout: the observer's adjustable-depth refinement of
        // its own memory. Depth grows/prunes with mimic-mastery (see main).
        // The RAW GRU state is carried recurrently; the REFINED state
        // drives every synthesis head and the self-model monitor, so added
        // depth changes the sound and is trained by the mimic loss.
        let refined_hidden = self.morphic.forward(&next_hidden).map_err(anyhow::Error::msg)?;

        // ---- Frequencies (graph tensors; glide blends in-graph target
        //      with the detached previous value: cur = g*target + (1-g)*prev) ----
        let b_l = self.base_freq_l.reshape(()).map_err(anyhow::Error::msg)?.abs().map_err(anyhow::Error::msg)?;
        let b_r = self.base_freq_r.reshape(()).map_err(anyhow::Error::msg)?.abs().map_err(anyhow::Error::msg)?;
        let target_l = b_l.add(&pop_l_t.affine(200.0, 0.0)?)?.add(&movement_t.affine(100.0, 0.0)?)?
            .clamp(20.0f32, 4000.0f32).map_err(anyhow::Error::msg)?;
        let target_r = b_r.add(&pop_r_t.affine(200.0, 0.0)?)?.add(&movement_t.affine(-100.0, 0.0)?)?
            .clamp(20.0f32, 4000.0f32).map_err(anyhow::Error::msg)?;
        let g = FREQ_GLIDE_SPEED as f64;
        let cur_l = target_l.affine(g, (self.current_freq_l * (1.0 - FREQ_GLIDE_SPEED)) as f64)?;
        let cur_r = target_r.affine(g, (self.current_freq_r * (1.0 - FREQ_GLIDE_SPEED)) as f64)?;

        // ---- FM parameters (graph tensors) ----
        let fm_ratios = self.fm_mod_ratio.forward(&refined_hidden).map_err(anyhow::Error::msg)?.affine(4.0, 0.0)?;
        let fm_indices = self.fm_mod_index.forward(&refined_hidden).map_err(anyhow::Error::msg)?.affine(5.0, 0.0)?;
        let ratio_l = fm_ratios.narrow(1, 0, 1).map_err(anyhow::Error::msg)?.reshape(()).map_err(anyhow::Error::msg)?;
        let ratio_r = fm_ratios.narrow(1, 1, 1).map_err(anyhow::Error::msg)?.reshape(()).map_err(anyhow::Error::msg)?;
        let idx_l = fm_indices.narrow(1, 0, 1).map_err(anyhow::Error::msg)?.reshape(()).map_err(anyhow::Error::msg)?;
        let idx_r = fm_indices.narrow(1, 1, 1).map_err(anyhow::Error::msg)?.reshape(()).map_err(anyhow::Error::msg)?;

        let mod_f_l = cur_l.mul(&ratio_l).map_err(anyhow::Error::msg)?
            .clamp(0.0f32, 6000.0f32).map_err(anyhow::Error::msg)?; // keep the modulator well below Nyquist
        let mod_f_r = cur_r.mul(&ratio_r).map_err(anyhow::Error::msg)?
            .clamp(0.0f32, 6000.0f32).map_err(anyhow::Error::msg)?;

        // ---- Phase trajectories ----
        // Modulator: theta_m(t) = 2*pi*f_m*t + phase_m. Carried phase is
        // its own analytic integral; index is ramped to kill zipper noise.
        let omega_m_l = mod_f_l.affine(TWO_PI as f64, 0.0)?;
        let omega_m_r = mod_f_r.affine(TWO_PI as f64, 0.0)?;
        let ph_m_l = self.t_steps.broadcast_mul(&omega_m_l).map_err(anyhow::Error::msg)?.affine(1.0, phase_m_l as f64)?;
        let ph_m_r = self.t_steps.broadcast_mul(&omega_m_r).map_err(anyhow::Error::msg)?.affine(1.0, phase_m_r as f64)?;
        let idx_curve_l = self.ramp_param(&idx_l, self.prev_fm_idx_l).map_err(anyhow::Error::msg)?;
        let idx_curve_r = self.ramp_param(&idx_r, self.prev_fm_idx_r).map_err(anyhow::Error::msg)?;
        let modulator_l = ph_m_l.sin().map_err(anyhow::Error::msg)?.mul(&idx_curve_l).map_err(anyhow::Error::msg)?;
        let modulator_r = ph_m_r.sin().map_err(anyhow::Error::msg)?.mul(&idx_curve_r).map_err(anyhow::Error::msg)?;

        // Carrier: base phase carries ONLY the carrier integral. theta (the
        // Kuramoto order-parameter angle, range +/-pi) is applied as a phase
        // offset RAMPED from the previous chunk's theta to this one across
        // the chunk, so chunk boundaries stay phase-continuous (a raw scalar
        // theta offset jumps up to 2*pi per boundary -> ~12 Hz clicking). The
        // ramp's only side effect is a sub-audible transient pitch glide when
        // theta swings hard, which is exactly when the CA is incoherent.
        let mut dtheta = theta - self.prev_theta;
        dtheta -= TWO_PI * (dtheta / TWO_PI).round(); // shortest angular path, |dtheta| <= pi
        let theta_curve = self.ramp.affine(dtheta as f64, self.prev_theta as f64)
            .map_err(anyhow::Error::msg)?;
        let omega_c_l = cur_l.affine(TWO_PI as f64, 0.0)?;
        let omega_c_r = cur_r.affine(TWO_PI as f64, 0.0)?;
        let ph_c_l = self.t_steps.broadcast_mul(&omega_c_l).map_err(anyhow::Error::msg)?
            .affine(1.0, phase_c_l as f64)?.add(&theta_curve)?.add(&modulator_l)?;
        let ph_c_r = self.t_steps.broadcast_mul(&omega_c_r).map_err(anyhow::Error::msg)?
            .affine(1.0, phase_c_r as f64)?.add(&theta_curve)?.add(&modulator_r)?;

        let mut audio_l = ph_c_l.sin().map_err(anyhow::Error::msg)?;
        let mut audio_r = ph_c_r.sin().map_err(anyhow::Error::msg)?;

        // ---- Formants (graph tensors driven by pop/movement) ----
        let f1_l = pop_l_t.affine(700.0, 300.0)?;
        let f1_r = pop_r_t.affine(700.0, 300.0)?;
        let f2 = movement_t.affine(1700.0, 800.0)?;
        let f3_l = pop_l_t.affine(-500.0, 2000.0)?;
        let f3_r = pop_r_t.affine(-500.0, 2000.0)?;
        for (f_l, f_r) in [(&f1_l, &f1_r), (&f2, &f2), (&f3_l, &f3_r)] {
            let w_l = f_l.affine(TWO_PI as f64, 0.0)?;
            let w_r = f_r.affine(TWO_PI as f64, 0.0)?;
            let p_l = self.t_steps.broadcast_mul(&w_l).map_err(anyhow::Error::msg)?.affine(1.0, phase_c_l as f64)?.add(&theta_curve)?;
            let p_r = self.t_steps.broadcast_mul(&w_r).map_err(anyhow::Error::msg)?.affine(1.0, phase_c_r as f64)?.add(&theta_curve)?;
            audio_l = audio_l.add(&p_l.sin().map_err(anyhow::Error::msg)?.affine(0.3, 0.0)?)?;
            audio_r = audio_r.add(&p_r.sin().map_err(anyhow::Error::msg)?.affine(0.3, 0.0)?)?;
        }

        let audio_l = audio_l.unsqueeze(0).map_err(anyhow::Error::msg)?; // (1, CHUNK)
        let audio_r = audio_r.unsqueeze(0).map_err(anyhow::Error::msg)?;

        // ---- KAN wavefolders ----
        // The KAN emits (1, CHUNK, out_features=1); flatten the trailing
        // singleton immediately. Without this, broadcasting (1, CHUNK, 1)
        // against the (1, CHUNK) openness curve aligns from the right and
        // produces a (1, CHUNK, CHUNK) outer product — the shape-mismatch
        // crash at the stereo cat.
        let audio_l = self.wavefolder_l.forward(&audio_l, mimic_loss_val).map_err(anyhow::Error::msg)?
            .reshape((1, CHUNK_SIZE)).map_err(anyhow::Error::msg)?;
        let audio_r = self.wavefolder_r.forward(&audio_r, mimic_loss_val).map_err(anyhow::Error::msg)?
            .reshape((1, CHUNK_SIZE)).map_err(anyhow::Error::msg)?;

        // ---- Filter openness (graph tensor, ramped) ----
        let open_t = refined_hidden.abs().map_err(anyhow::Error::msg)?.mean_all().map_err(anyhow::Error::msg)?
            .affine(5.0, 0.0)?.add(&movement_t)?
            .clamp(0.4f32, 1.0f32).map_err(anyhow::Error::msg)?;
        let open_curve = self.ramp_param(&open_t, self.prev_openness).map_err(anyhow::Error::msg)?;
        let audio_l = audio_l.broadcast_mul(&open_curve.unsqueeze(0).map_err(anyhow::Error::msg)?)?;
        let audio_r = audio_r.broadcast_mul(&open_curve.unsqueeze(0).map_err(anyhow::Error::msg)?)?;

        // ---- Equal-power pan (graph tensors, ramped) ----
        let pan_t = self.spatial_panner.forward(&refined_hidden).map_err(anyhow::Error::msg)?
            .reshape(()).map_err(anyhow::Error::msg)?
            .clamp(-0.5f32, 0.5f32).map_err(anyhow::Error::msg)?;
        let gain_l_t = pan_t.affine(-0.5, 0.5)?.sqrt().map_err(anyhow::Error::msg)?; // sqrt((1-pan)/2)
        let gain_r_t = pan_t.affine(0.5, 0.5)?.sqrt().map_err(anyhow::Error::msg)?;  // sqrt((1+pan)/2)
        let gain_curve_l = self.ramp_param(&gain_l_t, self.prev_gain_l).map_err(anyhow::Error::msg)?;
        let gain_curve_r = self.ramp_param(&gain_r_t, self.prev_gain_r).map_err(anyhow::Error::msg)?;
        let audio_l = audio_l.broadcast_mul(&gain_curve_l.unsqueeze(0).map_err(anyhow::Error::msg)?)?.affine(1.414, 0.0)?;
        let audio_r = audio_r.broadcast_mul(&gain_curve_r.unsqueeze(0).map_err(anyhow::Error::msg)?)?.affine(1.414, 0.0)?;

        let stereo = Tensor::cat(&[&audio_l, &audio_r], 0).map_err(anyhow::Error::msg)?
            .reshape((2, CHUNK_SIZE)).map_err(anyhow::Error::msg)?;

        // ---- Analytic phase carry (full N-sample advance, carrier only) ----
        let chunk_dt = CHUNK_SIZE as f32 / SAMPLE_RATE as f32;
        let cur_l_s = cur_l.to_scalar::<f32>().map_err(anyhow::Error::msg)?;
        let cur_r_s = cur_r.to_scalar::<f32>().map_err(anyhow::Error::msg)?;
        let mod_l_s = mod_f_l.to_scalar::<f32>().map_err(anyhow::Error::msg)?;
        let mod_r_s = mod_f_r.to_scalar::<f32>().map_err(anyhow::Error::msg)?;
        let next_phase_c_l = (phase_c_l + TWO_PI * cur_l_s * chunk_dt) % TWO_PI;
        let next_phase_c_r = (phase_c_r + TWO_PI * cur_r_s * chunk_dt) % TWO_PI;
        let next_phase_m_l = (phase_m_l + TWO_PI * mod_l_s * chunk_dt) % TWO_PI;
        let next_phase_m_r = (phase_m_r + TWO_PI * mod_r_s * chunk_dt) % TWO_PI;

        // ---- Update scalar ramp/glide state from detached values ----
        self.current_freq_l = cur_l_s;
        self.current_freq_r = cur_r_s;
        self.current_mod_freq_l = mod_l_s;
        self.current_mod_freq_r = mod_r_s;
        self.prev_fm_idx_l = idx_l.to_scalar::<f32>().map_err(anyhow::Error::msg)?;
        self.prev_fm_idx_r = idx_r.to_scalar::<f32>().map_err(anyhow::Error::msg)?;
        self.prev_openness = open_t.to_scalar::<f32>().map_err(anyhow::Error::msg)?;
        self.prev_gain_l = gain_l_t.to_scalar::<f32>().map_err(anyhow::Error::msg)?;
        self.prev_gain_r = gain_r_t.to_scalar::<f32>().map_err(anyhow::Error::msg)?;
        self.prev_theta = theta;

        Ok((
            ForwardOut { stereo, next_micro, next_macro, next_hidden, refined_hidden, movement_t, theta },
            next_phase_c_l, next_phase_c_r, next_phase_m_l, next_phase_m_r,
        ))
    }
}

// ==========================================
// 8. DEFIBRILLATOR CONTROLLER (now predictive, hence trainable)
// ==========================================
// Outputs [pred_movement_norm, pred_mimic, thresh_raw, noise_raw, lr_raw].
// The first two are trained by next-step MSE against observations, so
// the trunk learns a forward model; the control outputs share that trunk.
struct DefibrillatorController { net: candle_nn::Sequential }
impl DefibrillatorController {
    fn new(vb: VBV) -> Result<Self> {
        let net = candle_nn::seq()
            .add(candle_nn::linear(7, 24, vb.pp("net_0"))?)
            .add(candle_nn::Activation::Relu)
            .add(candle_nn::linear(24, 16, vb.pp("net_2"))?)
            .add(candle_nn::Activation::Relu)
            .add(candle_nn::linear(16, 5, vb.pp("net_4"))?);
        Ok(Self { net })
    }
    /// Returns (prediction tensor (1,2) in-graph, threshold, noise_scale, lr_multiplier)
    fn forward(&self, features: &Tensor) -> Result<(Tensor, f32, f32, f32)> {
        let raw = self.net.forward(features).map_err(anyhow::Error::msg)?;
        let pred = candle_nn::ops::sigmoid(&raw.narrow(1, 0, 2).map_err(anyhow::Error::msg)?).map_err(anyhow::Error::msg)?;
        let ctrl = raw.narrow(1, 2, 3).map_err(anyhow::Error::msg)?
            .reshape((3,)).map_err(anyhow::Error::msg)?
            .to_vec1::<f32>().map_err(anyhow::Error::msg)?;
        let threshold = (1.0 / (1.0 + (-ctrl[0]).exp())) * 0.20 + 0.05;
        let noise_scale = (1.0 / (1.0 + (-ctrl[1]).exp())) * 1.5 + 0.2;
        let lr_multiplier = 1.0 + (1.0 / (1.0 + (-ctrl[2]).exp())) * 7.0;
        Ok((pred, threshold, noise_scale, lr_multiplier))
    }
}

// ==========================================
// 9. AUDIO ARBITER (now trained on learning progress)
// ==========================================
// The softmax weights are used DETACHED to weight the losses (using them
// in-graph would just teach the arbiter to zero-out hard objectives).
// Instead the arbiter is trained to place weight on objectives that are
// currently improving (learning-progress allocation), plus the entropy
// regularizer that keeps the budget from collapsing onto one term.
struct AudioArbiter { net: candle_nn::Sequential }
impl AudioArbiter {
    fn new(vb: VBV) -> Result<Self> {
        let net = candle_nn::seq()
            .add(candle_nn::linear(14, 32, vb.pp("net_0"))?)
            .add(candle_nn::Activation::Relu)
            .add(candle_nn::linear(32, 16, vb.pp("net_2"))?)
            .add(candle_nn::Activation::Relu)
            .add(candle_nn::linear(16, 7, vb.pp("net_4"))?);
        Ok(Self { net })
    }
    /// Returns (softmax weights (1,7) in-graph, entropy penalty tensor)
    fn forward(&self, features: &Tensor) -> Result<(Tensor, Tensor)> {
        let raw = self.net.forward(features).map_err(anyhow::Error::msg)?;
        let exp_w = raw.exp().map_err(anyhow::Error::msg)?;
        let sum_exp = exp_w.sum_all().map_err(anyhow::Error::msg)?;
        let p = exp_w.broadcast_div(&sum_exp).map_err(anyhow::Error::msg)?;
        let entropy_penalty = p.log().map_err(anyhow::Error::msg)?
            .broadcast_mul(&p)?
            .sum_all().map_err(anyhow::Error::msg)?
            .affine(0.05, 0.0).map_err(anyhow::Error::msg)?;
        Ok((p, entropy_penalty))
    }
}

// ==========================================
// MAIN RUNTIME LOGIC
// ==========================================
fn main() -> Result<()> {
    let n_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
    rayon::ThreadPoolBuilder::new().num_threads(n_threads).build_global()?;
    let device = Device::Cpu;
    println!("=== TITAN AUDIO ECOSYSTEM: RUST EDITION (GRADIENT-COHERENT RELEASE) ===");
    println!("Threads: {} | BPTT window: {} | Tape: {}x{} | CA hidden: {}",
        n_threads, BPTT_WINDOW, CA_CHANNELS, TAPE_LEN, CA_CHANNELS * CA_HIDDEN_MULT);

    // Base directory configurable: first CLI arg, else /sdcard/Download.
    let base_dir = std::env::args().nth(1).unwrap_or_else(|| "/sdcard/Download".to_string());
    let wav_dir = format!("{}/OLD_WAVS", base_dir);
    let model_path = format!("{}/titan_model_beta.safetensors", base_dir);

    let target_loader = TargetAudioLoader::new(&wav_dir)?;
    let varmap = VarMap::new();
    let vb = VBV::from_varmap(&varmap, DType::F32, &device);
    let mut model = ComplexAudioEcosystem::new(vb.pp("model"), &device)?;
    let defib_ctrl = DefibrillatorController::new(vb.pp("defib"))?;
    let arbiter = AudioArbiter::new(vb.pp("arbiter"))?;
    let monitor_head = MonitorHead::new(vb.pp("monitor_head"))?;
    let spec_proj = SpectralProjector::new(&device).map_err(anyhow::Error::msg)?;

    // Load the checkpoint AFTER the vars exist (see load_into_varmap). The
    // optimizer is built afterward and holds the same Vars, so the loaded
    // values are picked up regardless.
    if std::path::Path::new(&model_path).exists() {
        match load_into_varmap(&varmap, &model_path, &device) {
            Ok((hit, miss, mismatch)) => println!(
                "--> Loaded {} tensors from {} ({} new/uninitialized, {} shape-mismatched)",
                hit, model_path, miss, mismatch),
            Err(e) => println!("--> Could not load {}: {} — starting fresh", model_path, e),
        }
    }

    // Restore observer morphology + radiation homeostat from sidecar.
    let morph_path = format!("{}/titan_morph_state.json", base_dir);
    let mut rad_amp = RAD_AMP_INIT;
    if let Ok(txt) = std::fs::read_to_string(&morph_path) {
        if let Ok(j) = serde_json::from_str::<serde_json::Value>(&txt) {
            if let Some(d) = j["active_depth"].as_u64() { model.set_depth(d as usize); }
            if let Some(r) = j["rad_amp"].as_f64() { rad_amp = (r as f32).clamp(RAD_AMP_MIN, RAD_AMP_MAX); }
        }
    }
    println!("--> Observer depth: L{:02} / {}  ·  rad_amp: {:.3}", model.depth(), MORPH_MAX_BLOCKS, rad_amp);

    let mut qnm_resonators = QNMFilterBank::new();
    let mut fractal_fdn_l = FractalFDN::new();
    let mut fractal_fdn_r = FractalFDN::new();

    let mut spectral_mon = SpectralEntropyMonitor::new(20);
    let mut movement_mon = MovementCoherenceMonitor::new(20);
    let mut uncertainty = AudioUncertaintyState::new();
    let mut semantic = SemanticField::new();
    let mut morph_history: Vec<f32> = Vec::new();
    let mut morph_baseline: Option<f32> = None; // calibrated from the warmup window
    let mut warmup_sum = 0.0f32;
    let mut field_entropy_sum = 0.0f64;
    let mut field_entropy_n = 0u64;
    let mut optimizer = AdamW::new_lr(varmap.all_vars(), BASE_LR).map_err(anyhow::Error::msg)?;

    let mut micro_tape = Tensor::randn(0.0f32, 1.0f32, (1, CA_CHANNELS, TAPE_LEN), &device).map_err(anyhow::Error::msg)?;
    let mut macro_tape = Tensor::randn(0.0f32, 1.0f32, (1, CA_CHANNELS, TAPE_LEN), &device).map_err(anyhow::Error::msg)?;
    let mut hidden_mem = Tensor::zeros((1, MEMORY_DIM), DType::F32, &device).map_err(anyhow::Error::msg)?;
    let mut phase_c_l = 0.0f32; let mut phase_c_r = 0.0f32; let mut phase_m_l = 0.0f32; let mut phase_m_r = 0.0f32;

    let total_chunks = (SAMPLE_RATE as f32 * DURATION_SECONDS / CHUNK_SIZE as f32) as usize;
    let mut audio_frames: Vec<i16> = Vec::with_capacity(total_chunks * CHUNK_SIZE * 2);
    let mut topology_history = Vec::new();
    let mut uncertainty_trace = Vec::new();

    let mut burst_ticks = 0;
    let mut burst_energy = 0.0f32;
    let mut phi = 0.0f32;
    let mut total_complexity = 0.0f32;
    let mut boost_state = 1.0f32;

    // BPTT-window accumulators
    let mut window_loss: Option<Tensor> = None;
    let mut steps_in_window = 0usize;
    let mut latest_lr_gain = 1.0f64;

    // Controller training state
    let mut prev_pred: Option<Tensor> = None;          // defib prediction from last step
    let mut prev_loss_vec = [0.0f32; 7];               // arbiter learning-progress baseline
    let mut prev_movement = 0.0f32;

    for step in 0..total_chunks {
        let aperture = uncertainty.branch_aperture();
        let force_macro = rand::thread_rng().gen_range(0.0..1.0) < (0.2 + aperture * 0.6);
        let prev_mimic_loss = if step == 0 { 0.5f32 } else { uncertainty.mimic / 10.0 };

        let (out, nc_l, nc_r, nm_l, nm_r) = model.forward(
            &micro_tape, &macro_tape, &hidden_mem,
            phase_c_l, phase_c_r, phase_m_l, phase_m_r,
            force_macro, prev_mimic_loss,
        )?;
        let ForwardOut { stereo: stereo_chunk, next_micro, next_macro, next_hidden, refined_hidden, movement_t, theta } = out;
        let movement = movement_t.reshape(())?.to_scalar::<f32>().map_err(anyhow::Error::msg)?;

        total_complexity += movement;
        let age_factor = (total_complexity / 500.0).min(0.6);
        let audio_for_loss = stereo_chunk.tanh().map_err(anyhow::Error::msg)?.affine((1.0 - age_factor) as f64, 0.0)?;

        // --------------------------------------------------
        // INFORMATION-THEORETIC RUNTIME ANALYTICS
        // --------------------------------------------------
        let synergy_val = calculate_cross_layer_synergy(&next_micro, &next_macro).unwrap_or(0.0);

        // Empowerment proxy stays IN-GRAPH now: differential entropy of
        // the state transition, so empowerment_loss genuinely shapes the
        // dynamics instead of being a constant.
        let memory_delta = next_hidden.sub(&hidden_mem)?;
        let tape_delta = next_micro.sub(&micro_tape)?;
        let trans_var = var_all(&memory_delta)?.add(&var_all(&tape_delta)?)?.affine(1.0, 1e-6)?;
        let cont_entropy_t = trans_var.log().map_err(anyhow::Error::msg)?.affine(0.5, 0.0)?;
        let empowerment_t = cont_entropy_t.affine(1.0, 7.0)?
            .clamp(0.0f32, 5.0f32).map_err(anyhow::Error::msg)?
            .mul(&movement_t.affine(1.0, 1.0)?)?;
        let empowerment_val = empowerment_t.reshape(())?.to_scalar::<f32>().unwrap_or(0.0);

        // Scale Invariance Constraint (true RG block decimation now)
        let coarse_micro = decimate2(&next_micro).map_err(anyhow::Error::msg)?;
        let coarse_macro = decimate2(&next_macro).map_err(anyhow::Error::msg)?;
        let detached_macro = coarse_macro.detach().map_err(anyhow::Error::msg)?;
        let rg_loss = coarse_micro.sub(&detached_macro)?.sqr()?.mean_all()?;

        // --------------------------------------------------
        // PERCEPTUAL MIMIC LOSS (log-magnitude spectral MSE)
        // --------------------------------------------------
        // Stereo targets, per-channel spectral comparison: the model's L/R
        // voices and pan field get real supervision instead of a mono
        // downmix erasing the stereo image.
        let target_chunk = target_loader.sample_chunk(&device).map_err(anyhow::Error::msg)?; // (2, CHUNK)
        let out_spec_l = spec_proj.log_mag(&audio_for_loss.narrow(0, 0, 1).map_err(anyhow::Error::msg)?).map_err(anyhow::Error::msg)?;
        let out_spec_r = spec_proj.log_mag(&audio_for_loss.narrow(0, 1, 1).map_err(anyhow::Error::msg)?).map_err(anyhow::Error::msg)?;
        let tgt_spec_l = spec_proj.log_mag(&target_chunk.narrow(0, 0, 1).map_err(anyhow::Error::msg)?).map_err(anyhow::Error::msg)?
            .detach().map_err(anyhow::Error::msg)?;
        let tgt_spec_r = spec_proj.log_mag(&target_chunk.narrow(0, 1, 1).map_err(anyhow::Error::msg)?).map_err(anyhow::Error::msg)?
            .detach().map_err(anyhow::Error::msg)?;
        let mimic_l = out_spec_l.sub(&tgt_spec_l)?.sqr()?.mean_all()?;
        let mimic_r = out_spec_r.sub(&tgt_spec_r)?.sqr()?.mean_all()?;
        let mimic_loss = mimic_l.add(&mimic_r)?.affine(0.5, 0.0).map_err(anyhow::Error::msg)?;
        let mimic_drift = mimic_loss.reshape(())?.to_scalar::<f32>().unwrap_or(0.0);
        // Bounded control signal: log-spectral MSE is unbounded above, so
        // every downstream controller sees drift/(1+drift) in [0, 1).
        let mimic_drift_n = mimic_drift / (1.0 + mimic_drift);

        // --------------------------------------------------
        // SEMANTIC FIELD + MORPHOLOGICAL HOMEOSTASIS (token_universe)
        // --------------------------------------------------
        // Read the freshly evolved substrate as an archetype field, place
        // the system on the epistemic phase ladder, and let the observer's
        // depth + the world's radiation co-adapt toward the edge of chaos.
        let field01: Vec<f32> = next_micro.flatten_all().map_err(anyhow::Error::msg)?
            .to_vec1::<f32>().map_err(anyhow::Error::msg)?
            .iter().map(|&x| (x + 1.0) * 0.5).collect();
        let (arch_summary, field_entropy, dom_arch) = SemanticField::archetype_field(&field01);
        let phase = SemanticField::phase(mimic_drift_n);
        semantic.record(mimic_drift_n, phase, dom_arch);
        let trend = semantic.trend();
        field_entropy_sum += field_entropy as f64;
        field_entropy_n += 1;

        // Patience-gated neurogenesis / pruning against a self-calibrated
        // baseline. During warmup we only observe (and accumulate the mean);
        // after warmup the baseline is fixed and the depth adapts when the
        // patience-window average leaves the dead-band around it. Patience
        // lengthens with depth (deeper observers restructure more slowly).
        let mut morph_event: Option<&str> = None;
        if step < MORPH_WARMUP {
            warmup_sum += mimic_drift_n;
        } else {
            if morph_baseline.is_none() {
                let b = (warmup_sum / MORPH_WARMUP as f32).max(1e-4);
                morph_baseline = Some(b);
                println!("--> Morph baseline calibrated: mimic≈{:.3}  (grow>{:.3}, prune<{:.3})",
                    b, b * MORPH_GROWTH_REL, b * MORPH_PRUNE_REL);
            }
            morph_history.push(mimic_drift_n);
            let patience = MORPH_PATIENCE_BASE + model.depth() * 2;
            if morph_history.len() >= patience {
                let avg = morph_history.iter().sum::<f32>() / morph_history.len() as f32;
                morph_history.clear();
                let base = morph_baseline.unwrap();
                if avg > base * MORPH_GROWTH_REL {
                    if model.grow() {
                        rad_amp = (rad_amp * RAD_COOL).max(RAD_AMP_MIN); // struggling: calm the world
                        morph_event = Some("NEUROGENESIS");
                    }
                } else if avg < base * MORPH_PRUNE_REL {
                    if model.prune() {
                        rad_amp = (rad_amp * RAD_HEAT).min(RAD_AMP_MAX);  // mastering: agitate the world
                        morph_event = Some("PRUNING");
                    }
                }
            }
        }
        if let Some(ev) = morph_event {
            let line = semantic.commentary(phase, Some(ev), rad_amp, model.depth());
            println!("  ◄ {} ►  {}", ev, line);
        }

        // --------------------------------------------------
        // SHAPING LOSSES (all live in the graph)
        // --------------------------------------------------
        let current_var = var_all(&audio_for_loss).map_err(anyhow::Error::msg)?;
        let var_loss = current_var.affine(1.0, -0.12)?.sqr().map_err(anyhow::Error::msg)?;

        // Single RMS target (the old saturation_loss/energy_loss pair
        // pulled toward 0.28 and 0.25 simultaneously).
        let rms = audio_for_loss.sqr().map_err(anyhow::Error::msg)?
            .mean_all().map_err(anyhow::Error::msg)?
            .affine(1.0, 1e-5)?.sqrt().map_err(anyhow::Error::msg)?;
        let rms_val = rms.reshape(())?.to_scalar::<f32>().unwrap_or(0.0);
        let saturation_loss = rms.affine(1.0, -0.28)?.sqr().map_err(anyhow::Error::msg)?;

        // movement_loss is a graph tensor now (was exp of a detached f32:
        // a constant with zero gradient).
        let movement_loss = movement_t.neg().map_err(anyhow::Error::msg)?.exp().map_err(anyhow::Error::msg)?;

        let diff = audio_for_loss.narrow(1, 1, CHUNK_SIZE - 1).map_err(anyhow::Error::msg)?
            .sub(&audio_for_loss.narrow(1, 0, CHUNK_SIZE - 1).map_err(anyhow::Error::msg)?)?;
        let roughness_loss = diff.sqr().map_err(anyhow::Error::msg)?.mean_all().map_err(anyhow::Error::msg)?;
        let reg_loss = stereo_chunk.sqr().map_err(anyhow::Error::msg)?.mean_all().map_err(anyhow::Error::msg)?;

        let empowerment_loss = empowerment_t.affine(1.0, -2.5)?.sqr().map_err(anyhow::Error::msg)?;

        // --------------------------------------------------
        // MONITORS FIRST, then the self-model predicts FRESH observations
        // (the old order made the monitor head predict stale state).
        // --------------------------------------------------
        let m_sig = movement_mon.analyze(movement)?;
        let s_sig = spectral_mon.analyze(&stereo_chunk)?;
        uncertainty.update(&s_sig, &m_sig, Some(&serde_json::json!({"drift": mimic_drift_n, "theta": theta})), synergy_val, empowerment_val);
        phi = uncertainty.phi;

        let pred_state = monitor_head.forward(&refined_hidden)?;
        let observed_state = Tensor::new(
            &[
                uncertainty.spectral.clamp(0.0, 1.0),
                (uncertainty.movement / 2.0).clamp(0.0, 1.0),
                uncertainty.mimic.clamp(0.0, 1.0),
                aperture.clamp(0.0, 1.0),
                (synergy_val / 5.0).clamp(0.0, 1.0),
            ],
            &device,
        )?.unsqueeze(0)?;
        let self_model_loss = pred_state.sub(&observed_state)?.sqr()?.mean_all()?;

        // --------------------------------------------------
        // ARBITER: real features (no placeholder zeros), learning-progress meta-loss
        // --------------------------------------------------
        let rg_v = rg_loss.reshape(())?.to_scalar::<f32>().unwrap_or(0.0);
        let arb_features = Tensor::new(&[
            rms_val,
            mimic_drift_n,
            movement / 0.3,
            synergy_val / 5.0,
            empowerment_val / 5.0,
            rg_v,
            uncertainty.spectral,
            uncertainty.movement,
            uncertainty.mimic,
            uncertainty.compositional,
            aperture,
            step as f32 / total_chunks as f32,
            phi / 10.0,
            theta / std::f32::consts::PI,
        ], &device).map_err(anyhow::Error::msg)?.unsqueeze(0).map_err(anyhow::Error::msg)?;

        let (w_graph, arb_entropy_loss) = arbiter.forward(&arb_features)?;
        // Detached copies weight the actual losses (stochastic controller).
        let lw_raw = w_graph.reshape((7,))?.to_vec1::<f32>().map_err(anyhow::Error::msg)?;
        let lw: Vec<f32> = lw_raw.iter().map(|p| p * 7.0).collect();

        // Learning-progress meta-objective: reward weight on improving losses.
        let cur_loss_vec = [
            current_var.reshape(())?.to_scalar::<f32>().unwrap_or(0.0),
            mimic_drift_n,
            movement_loss.reshape(())?.to_scalar::<f32>().unwrap_or(0.0),
            roughness_loss.reshape(())?.to_scalar::<f32>().unwrap_or(0.0),
            rg_v,
            self_model_loss.reshape(())?.to_scalar::<f32>().unwrap_or(0.0),
            empowerment_loss.reshape(())?.to_scalar::<f32>().unwrap_or(0.0),
        ];
        let improvement: Vec<f32> = (0..7).map(|i| (prev_loss_vec[i] - cur_loss_vec[i]).clamp(-1.0, 1.0)).collect();
        prev_loss_vec = cur_loss_vec;
        let improvement_t = Tensor::new(improvement, &device).map_err(anyhow::Error::msg)?;
        let arb_progress_loss = w_graph.reshape((7,))?
            .mul(&improvement_t).map_err(anyhow::Error::msg)?
            .sum_all().map_err(anyhow::Error::msg)?
            .affine(-0.5, 0.0).map_err(anyhow::Error::msg)?;

        // --------------------------------------------------
        // DEFIBRILLATOR: prediction loss from last step's forecast
        // --------------------------------------------------
        let defib_features = Tensor::new(&[
            movement,
            movement - prev_movement,
            mimic_drift_n,
            rms_val,
            aperture,
            phi / 10.0,
            step as f32 / total_chunks as f32,
        ], &device).map_err(anyhow::Error::msg)?.unsqueeze(0).map_err(anyhow::Error::msg)?;
        prev_movement = movement;
        let (pred_t, thresh, n_scale, lr_mult) = defib_ctrl.forward(&defib_features)?;

        let defib_pred_loss = if let Some(p) = prev_pred.take() {
            let obs = Tensor::new(&[(movement / 0.3).clamp(0.0, 1.0), mimic_drift_n], &device)
                .map_err(anyhow::Error::msg)?.unsqueeze(0).map_err(anyhow::Error::msg)?;
            Some(p.sub(&obs)?.sqr()?.mean_all()?)
        } else { None };
        prev_pred = Some(pred_t);

        // --------------------------------------------------
        // TOTAL LOSS ASSEMBLY
        // --------------------------------------------------
        let mut total_loss = mimic_loss.affine((lw[1] * (1.0 - RESONANT_AUTONOMY)) as f64, 0.0).map_err(anyhow::Error::msg)?;
        total_loss = total_loss.add(&var_loss.affine((lw[0] * 2.5) as f64, 0.0).map_err(anyhow::Error::msg)?)?;
        total_loss = total_loss.add(&saturation_loss.affine(2.0, 0.0).map_err(anyhow::Error::msg)?)?;
        total_loss = total_loss.add(&movement_loss.affine((lw[2] * RESONANT_AUTONOMY) as f64, 0.0).map_err(anyhow::Error::msg)?)?;
        total_loss = total_loss.add(&roughness_loss.affine(lw[3] as f64, 0.0).map_err(anyhow::Error::msg)?)?;
        total_loss = total_loss.add(&reg_loss.affine(0.01, 0.0).map_err(anyhow::Error::msg)?)?;
        total_loss = total_loss.add(&rg_loss.affine((0.15 * lw[4].max(0.2)) as f64, 0.0).map_err(anyhow::Error::msg)?)?;
        total_loss = total_loss.add(&self_model_loss.affine((0.30 * lw[5].max(0.2)) as f64, 0.0).map_err(anyhow::Error::msg)?)?;
        total_loss = total_loss.add(&empowerment_loss.affine(lw[6] as f64, 0.0).map_err(anyhow::Error::msg)?)?;
        total_loss = total_loss.add(&arb_entropy_loss)?;
        total_loss = total_loss.add(&arb_progress_loss)?;
        if let Some(dl) = defib_pred_loss {
            total_loss = total_loss.add(&dl.affine(0.2, 0.0).map_err(anyhow::Error::msg)?)?;
        }

        // --------------------------------------------------
        // K-STEP TRUNCATED BPTT
        // --------------------------------------------------
        window_loss = Some(match window_loss.take() {
            None => total_loss,
            Some(w) => w.add(&total_loss).map_err(anyhow::Error::msg)?,
        });
        steps_in_window += 1;

        // Choptuik learning-rate law. CRITICALITY_SEEKING = true puts
        // peak plasticity AT the critical surface (SOC); false restores
        // the "frozen at the horizon" time-dilation reading.
        let distance_to_horizon = (movement - thresh).abs() + 1e-4;
        let choptuik_gain = if CRITICALITY_SEEKING {
            (CRITICAL_D0 / distance_to_horizon).powf(CHOPTUIK_EXPONENT).clamp(0.3, 3.0)
        } else {
            distance_to_horizon.powf(CHOPTUIK_EXPONENT)
        };

        if (movement < thresh || mimic_drift_n > 0.6) && burst_ticks == 0 {
            burst_ticks = 8;
            burst_energy = n_scale;
        }

        let phi_gate = 1.0 / (1.0 + phi);
        let burst_env = (burst_ticks as f32 / 8.0).sqrt();
        latest_lr_gain = if burst_ticks > 0 {
            ((1.0 + (lr_mult - 1.0) * burst_env) * phi_gate * choptuik_gain) as f64
        } else {
            (phi_gate * choptuik_gain) as f64
        };
        latest_lr_gain = latest_lr_gain.max(0.1);

        if steps_in_window >= BPTT_WINDOW || step == total_chunks - 1 {
            if let Some(w) = window_loss.take() {
                let scaled = w.affine(1.0 / steps_in_window as f64, 0.0).map_err(anyhow::Error::msg)?;
                optimizer.set_learning_rate(BASE_LR * latest_lr_gain);
                optimizer.backward_step(&scaled).map_err(anyhow::Error::msg)?;
            }
            steps_in_window = 0;
            // Detach carried state: window boundary of truncated BPTT.
            micro_tape = next_micro.detach().map_err(anyhow::Error::msg)?;
            macro_tape = next_macro.detach().map_err(anyhow::Error::msg)?;
            hidden_mem = next_hidden.detach().map_err(anyhow::Error::msg)?;
            prev_pred = match prev_pred.take() {
                Some(p) => Some(p.detach().map_err(anyhow::Error::msg)?),
                None => None,
            };
            // Edge-of-chaos radiation: sparse heavy-tailed mutation of the
            // substrate, scaled by the homeostat's rad_amp. Done only at the
            // window boundary — the tape is a detached leaf here, so it is a
            // physical kick to the world, not a torn gradient path.
            if rand::thread_rng().gen::<f32>() < RADIATE_PROB {
                micro_tape = levy_radiate(&micro_tape, rad_amp).map_err(anyhow::Error::msg)?;
            }
        } else {
            // Keep graph alive within the window: multi-step credit assignment.
            micro_tape = next_micro;
            macro_tape = next_macro;
            hidden_mem = next_hidden;
        }
        phase_c_l = nc_l; phase_c_r = nc_r; phase_m_l = nm_l; phase_m_r = nm_r;

        // Defibrillation noise is applied AFTER the state carry. The old
        // order added it to the stale tape and then overwrote that tape
        // with next_micro — every burst was silently discarded.
        if burst_ticks > 0 {
            let noise = Tensor::randn(0.0f32, 1.0f32, micro_tape.shape(), micro_tape.device())
                .map_err(anyhow::Error::msg)?
                .affine((burst_energy * burst_env * choptuik_gain) as f64, 0.0)?;
            micro_tape = micro_tape.add(&noise).map_err(anyhow::Error::msg)?.clamp(-1.0f32, 1.0f32).map_err(anyhow::Error::msg)?;
            if step % 20 == 0 { println!("[PACEMAKER CHOPTUIK BURST] step {} (env: {:.2}, gain: {:.3})", step, burst_env, choptuik_gain); }
            burst_ticks -= 1;
        }

        // --------------------------------------------------
        // RENDER PATH (post-graph DSP; the spectral mimic loss is
        // computed pre-FX by design — the FX chain is non-differentiable
        // sample-recursive IIR, and we want the model to learn the voice,
        // not the room)
        // --------------------------------------------------
        let audio_t = stereo_chunk.tanh().map_err(anyhow::Error::msg)?.affine((1.0 - age_factor) as f64, 0.0)?;
        let abs_max = audio_t.abs().map_err(anyhow::Error::msg)?
            .flatten_all().map_err(anyhow::Error::msg)?
            .max(0).map_err(anyhow::Error::msg)?
            .to_scalar::<f32>().map_err(anyhow::Error::msg)?;
        let boost_target = if abs_max < 0.25 { 0.25 / (abs_max + 1e-6) } else { 1.0 };
        boost_state = boost_state * 0.9 + boost_target * 0.1; // smoothed: no per-chunk gain pumping
        let audio_normalized = audio_t.affine(boost_state as f64, 0.0).map_err(anyhow::Error::msg)?;

        let mut audio_l = audio_normalized.narrow(0, 0, 1).map_err(anyhow::Error::msg)?.to_vec2::<f32>().map_err(anyhow::Error::msg)?[0].clone();
        let mut audio_r = audio_normalized.narrow(0, 1, 1).map_err(anyhow::Error::msg)?.to_vec2::<f32>().map_err(anyhow::Error::msg)?[0].clone();

        qnm_resonators.process(&mut audio_l, &mut audio_r, phi);

        let echo_aperture = aperture.min(0.7);
        fractal_fdn_l.process(&mut audio_l, echo_aperture);
        fractal_fdn_r.process(&mut audio_r, echo_aperture);

        for i in 0..CHUNK_SIZE {
            let sample_l = (audio_l[i] * 32767.0).clamp(-32768.0, 32767.0) as i16;
            let sample_r = (audio_r[i] * 32767.0).clamp(-32768.0, 32767.0) as i16;
            audio_frames.push(sample_l);
            audio_frames.push(sample_r);
        }

        // --------------------------------------------------
        // TRACES & LOGGING
        // --------------------------------------------------
        if step % 10 == 0 {
            let topology_state = macro_tape.mean(1).map_err(anyhow::Error::msg)?
                .reshape((TAPE_LEN,)).map_err(anyhow::Error::msg)?
                .to_vec1::<f32>().map_err(anyhow::Error::msg)?;
            topology_history.push(topology_state);
            uncertainty_trace.push(serde_json::json!({
                "step": step, "spectral": uncertainty.spectral, "movement": uncertainty.movement,
                "compositional": uncertainty.compositional, "aperture": aperture, "phi": phi,
                "synergy": synergy_val, "empowerment": empowerment_val
            }));
        }
        if step % 50 == 0 {
            println!("Chunk {}/{} | Move: {:.3} | Mimic: {:.3} {} | Phase: {} | L{:02} rad:{:.2} | Phi: {:.2} | LRx: {:.2}",
                step, total_chunks, movement, mimic_drift_n, trend, phase, model.depth(), rad_amp, phi, latest_lr_gain);
            println!("  field H:{:.2}b · {} · synergy:{:.2} empower:{:.2}",
                field_entropy, arch_summary, synergy_val, empowerment_val);
            // Quantile dual-lane tape: value density over gradient motion.
            let cols = 64.min(TAPE_LEN);
            if let Ok(v) = micro_tape.mean(1).and_then(|m| m.reshape((TAPE_LEN,))) {
                if let Ok(full) = v.to_vec1::<f32>() {
                    let pooled: Vec<f32> = full.iter().take(cols).copied().collect();
                    let (val_lane, grad_lane) = quantile_dual_tape(&pooled);
                    println!("  v {}", val_lane);
                    println!("  ∂ {}", grad_lane);
                }
            }
            let comment = semantic.commentary(phase, None, rad_amp, model.depth());
            if !comment.is_empty() { println!("  · {}", comment); }
        }
        // Periodic checkpoint: phones die, batteries drain.
        if step > 0 && step % 500 == 0 {
            if varmap.save(&model_path).is_ok() {
                let _ = std::fs::write(&morph_path, serde_json::json!({
                    "active_depth": model.depth(), "rad_amp": rad_amp
                }).to_string());
                println!("--> Checkpoint saved at step {} (L{:02}, rad {:.3})", step, model.depth(), rad_amp);
            }
        }
    }

    // Creative Prompt Formulation for External Generation Engine
    let avg_phi = uncertainty_trace.iter().map(|t| t["phi"].as_f64().unwrap_or(0.0)).sum::<f64>() / uncertainty_trace.len() as f64;
    let avg_aperture = uncertainty_trace.iter().map(|t| t["aperture"].as_f64().unwrap_or(0.0)).sum::<f64>() / uncertainty_trace.len() as f64;
    let avg_synergy = uncertainty_trace.iter().map(|t| t["synergy"].as_f64().unwrap_or(0.0)).sum::<f64>() / uncertainty_trace.len() as f64;
    let avg_field_h = if field_entropy_n > 0 { field_entropy_sum / field_entropy_n as f64 } else { 0.0 };
    let dom_phase = semantic.dominant_phase();
    let dom_archetype = semantic.dominant_archetype();
    let final_depth = model.depth();

    let prompt = format!(
        "Style: {}, {}, {}, {}. Texture: {}. Field: {} regime · {} archetype · depth L{:02}. \
         [Informational Phi: {:.2}, Aperture: {:.2}, Synergy: {:.2}, Field-Entropy: {:.2}b, Rad: {:.2}]",
        if avg_phi > 0.65 { "Hyper-Resonant" } else { "Chaotic" },
        if avg_aperture > 0.5 { "Evolving" } else { "Stable" },
        if total_complexity > 500.0 { "Dense" } else { "Minimal" },
        "Information-Theoretic Glitch",
        if avg_synergy > 1.5 { "Crystalline-Autonomous" } else if avg_phi > 0.52 { "Organic" } else { "Grit" },
        dom_phase, dom_archetype, final_depth,
        avg_phi, avg_aperture, avg_synergy, avg_field_h, rad_amp
    );
    println!("\n=== GENERATIVE PRIMING PROMPT ===");
    println!("{}", prompt);
    std::fs::write(format!("{}/suno_priming_prompt.txt", base_dir), &prompt)?;

    // Serialize Exporters (CSV + Safetensors)
    let mut topo_writer = csv::Writer::from_path(format!("{}/ca_topology_rust.csv", base_dir))?;
    for row in topology_history { topo_writer.write_record(row.iter().map(|f| f.to_string()))?; }
    topo_writer.flush()?;

    let mut unc_writer = csv::Writer::from_path(format!("{}/uncertainty_trace_rust.csv", base_dir))?;
    unc_writer.write_record(&["step", "spectral", "movement", "compositional", "aperture", "synergy", "empowerment"])?;
    for trace in uncertainty_trace {
        unc_writer.write_record(&[
            trace["step"].to_string(), trace["spectral"].to_string(), trace["movement"].to_string(),
            trace["compositional"].to_string(), trace["aperture"].to_string(),
            trace["synergy"].to_string(), trace["empowerment"].to_string()
        ])?;
    }
    unc_writer.flush()?;
    println!("Topology and uncertainty trace saved to {}.", base_dir);

    let spec = hound::WavSpec { channels: 2, sample_rate: SAMPLE_RATE, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut writer = hound::WavWriter::create(format!("{}/rust_ecosystem_out.wav", base_dir), spec)?;
    for sample in audio_frames { writer.write_sample(sample)?; }
    writer.finalize()?;
    println!("Audio saved to {}/rust_ecosystem_out.wav", base_dir);

    varmap.save(&model_path).map_err(anyhow::Error::msg)?;
    let _ = std::fs::write(&morph_path, serde_json::json!({
        "active_depth": model.depth(), "rad_amp": rad_amp
    }).to_string());
    let metadata = std::fs::metadata(&model_path)?;
    println!("Model saved to {} (L{:02}, rad {:.3}). Size: {:.2} MB",
        model_path, model.depth(), rad_amp, metadata.len() as f32 / 1_048_576.0);

    Ok(())
}
