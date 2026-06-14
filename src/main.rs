use anyhow::Result;
use candle_core::{Device, Tensor, DType, D, Result as CResult};
use candle_nn::{Linear, Conv1dConfig, VarBuilder as VBV, VarMap, Optimizer, AdamW, Module};
use std::collections::VecDeque;
use rustfft::{FftPlanner, num_complex::Complex};
use rand::Rng;
use std::f32::consts::PI;

// ==========================================
// ECOSYSTEM CONFIGURATION
// ==========================================
const SAMPLE_RATE: u32 = 48000;
const DURATION_SECONDS: f32 = 420.0;
const CHUNK_SIZE: usize = 2048;
const TAPE_LEN: usize = 512;
const CA_CHANNELS: usize = 144;
const CA_HIDDEN_MULT: usize = 128;
const KAN_BASIS_FUNCTIONS: usize = 244;
const MEMORY_DIM: usize = 512;
const NUM_PARTIALS: usize = 32;        // additive synthesis partials (CA channels → harmonics)
const FM_OPERATORS: usize = 4;         // DX7-style FM operators (algorithm 5: 2×[mod→carrier])
const NUM_GRAINS: usize = 8;           // concurrent granular synthesis grains
const GRAIN_BUFFER_LEN: usize = 96000; // 2-second grain buffer at 48 kHz
const RD_TAPE_LEN: usize = 128;        // Gray-Scott reaction-diffusion tape length

// Physics-Driven Constants (from cutting-edge physics research)
const CHOPTUIK_EXPONENT: f32 = 0.37413; // universal critical exponent near black-hole formation horizon
const LARGE_D_DIM: usize = 512;         // effective dimension for asymptotic large-D gravity contraction
const FDN_DELAY_LINES: usize = 4;
const FDN_DELAYS: [usize; 4] = [149, 263, 431, 701]; // prime-spaced delay lines for FDN

// Tuning: L/R differ by 7.83 Hz (Schumann resonance / alpha brainwave binaural beat)
const BASE_FREQ_L: f32 = 49.0;
const BASE_FREQ_R: f32 = 56.83;
const METABOLIC_DECAY: f32 = 0.9999;
const FREQ_GLIDE_SPEED: f32 = 0.0554;
const BASE_LR: f64 = 1e-4;
const RESONANT_AUTONOMY: f32 = 0.114;

// Mix ratios for the three synthesis voices (should sum to <= 1.0)
const ADDITIVE_MIX: f32 = 0.50;  // CA-channel-driven additive harmonic series
const FM_MIX: f32 = 0.25;         // 4-op FM synthesis
const GRANULAR_MIX: f32 = 0.25;  // granular resynthesis from past audio

// ==========================================
// TARGET AUDIO LOADER
// ==========================================
struct TargetAudioLoader {
    buffers: Vec<Vec<f32>>,
}

impl TargetAudioLoader {
    fn new(path: &str) -> Result<Self> {
        let mut buffers = Vec::new();
        let entries = std::fs::read_dir(path)?;
        for entry in entries {
            let entry = entry?;
            let p = entry.path();
            if p.extension().map_or(false, |ext| ext == "wav")
                && p.file_name().unwrap() != "rust_ecosystem_out.wav"
            {
                println!("--> Loading target audio: {:?}", p);
                let mut reader = hound::WavReader::open(p)?;
                let samples: Vec<i16> =
                    reader.samples::<i16>().map(|s| s.unwrap_or(0)).collect();
                let f32_samples: Vec<f32> =
                    samples.iter().map(|&s| s as f32 / 32768.0).collect();
                if !f32_samples.is_empty() {
                    buffers.push(f32_samples);
                }
            }
        }
        if buffers.is_empty() {
            anyhow::bail!("No training audio found in {}", path);
        }
        Ok(Self { buffers })
    }

    fn sample_chunk(&self, device: &Device) -> CResult<Tensor> {
        let mut rng = rand::thread_rng();
        let buf_idx = rng.gen_range(0..self.buffers.len());
        let buf = &self.buffers[buf_idx];
        let start = rng.gen_range(0..(buf.len().saturating_sub(CHUNK_SIZE)));
        let slice = &buf[start..start + CHUNK_SIZE];
        Tensor::new(slice, device)?.unsqueeze(0)
    }
}

// ==========================================
// CUSTOM MODULES
// ==========================================
struct Tanh;
impl Module for Tanh {
    fn forward(&self, xs: &Tensor) -> CResult<Tensor> { xs.tanh() }
}

struct Sigmoid;
impl Module for Sigmoid {
    fn forward(&self, xs: &Tensor) -> CResult<Tensor> { candle_nn::ops::sigmoid(xs) }
}

struct Softplus;
impl Module for Softplus {
    fn forward(&self, xs: &Tensor) -> CResult<Tensor> {
        let exp = xs.exp()?;
        let one = Tensor::new(1.0f32, xs.device())?.broadcast_as(exp.shape())?;
        exp.add(&one)?.log()
    }
}

// ==========================================
// UTILS
// ==========================================
fn var_all(x: &Tensor) -> CResult<Tensor> {
    let mean = x.mean_all()?;
    let diff = x.broadcast_sub(&mean)?;
    diff.sqr()?.mean_all()
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
// CHAOTIC NOISE SOURCE — Lorenz Attractor
// ==========================================
/// Replaces uniform anti-stagnation noise with structured chaos.
/// Lorenz dynamics produce long-range correlated noise that sounds more
/// musical than white noise — small perturbations that grow and collapse.
struct LorenzAttractor {
    x: f32, y: f32, z: f32,
    sigma: f32, rho: f32, beta: f32,
    dt: f32,
}

impl LorenzAttractor {
    fn new() -> Self {
        Self {
            x: 1.0, y: 0.0, z: 0.0,
            sigma: 10.0, rho: 28.0, beta: 8.0 / 3.0,
            dt: 0.01,
        }
    }

    /// Advance one step; return normalized (x, y, z) each approximately in [-1, 1]
    fn step(&mut self) -> (f32, f32, f32) {
        let dx = self.sigma * (self.y - self.x);
        let dy = self.x * (self.rho - self.z) - self.y;
        let dz = self.x * self.y - self.beta * self.z;
        self.x += dx * self.dt;
        self.y += dy * self.dt;
        self.z += dz * self.dt;
        // Attractor bounds: x in [-20,20], y in [-30,30], z in [0,50]
        (self.x / 22.0, self.y / 30.0, (self.z - 25.0) / 25.0)
    }

    fn noise_scalar(&mut self) -> f32 {
        let (x, y, _) = self.step();
        ((x + y) * 0.5).clamp(-1.0, 1.0)
    }
}

// ==========================================
// EUCLIDEAN RHYTHM GENERATOR
// ==========================================
/// Bjorklund / Bresenham approximation: distributes `pulses` as evenly
/// as possible across `steps`. E(7,24) gives a dense syncopated feel;
/// the pulse count is re-derived every 96 chunks from the GRU state.
fn euclidean_rhythm(pulses: usize, steps: usize) -> Vec<bool> {
    if steps == 0 { return vec![]; }
    let pulses = pulses.min(steps);
    if pulses == 0 { return vec![false; steps]; }
    let mut pattern = vec![false; steps];
    for i in 0..pulses {
        pattern[(i * steps) / pulses] = true;
    }
    pattern
}

// ==========================================
// REACTION-DIFFUSION TEXTURE — Gray-Scott
// ==========================================
/// 1-D Gray-Scott system. Produces evolving spot/stripe/labyrinth patterns
/// whose variance is used to scale the KAN wavefolder's input — more
/// complex R-D patterns → more aggressive wavefolding → richer harmonics.
struct ReactionDiffusion {
    u: Vec<f32>,   // activator field
    v: Vec<f32>,   // inhibitor field
    len: usize,
    f: f32,        // feed rate (F)
    k: f32,        // kill rate (k)
    du: f32,       // U diffusion coefficient
    dv: f32,       // V diffusion coefficient
    dt: f32,       // integration step
}

impl ReactionDiffusion {
    fn new() -> Self {
        let len = RD_TAPE_LEN;
        let mut u = vec![1.0f32; len];
        let mut v = vec![0.0f32; len];
        let mid = len / 2;
        for i in mid.saturating_sub(8)..(mid + 8).min(len) {
            let t = (i as f32 - mid as f32) / 8.0;
            u[i] = 0.50 + t * 0.05;
            v[i] = 0.25;
        }
        Self { u, v, len, f: 0.055, k: 0.062, du: 0.16, dv: 0.08, dt: 1.0 }
    }

    /// Advance the system; returns the standard deviation of V in [0, ~0.5]
    /// (zero = uniform / no pattern, high = rich spatial texture)
    fn step(&mut self, input_mod: f32) -> f32 {
        let len = self.len;
        let mut new_u = self.u.clone();
        let mut new_v = self.v.clone();
        let f_mod = (self.f + input_mod * 0.004).clamp(0.01, 0.095);

        for i in 0..len {
            let left  = if i == 0       { len - 1 } else { i - 1 };
            let right = if i == len - 1 { 0       } else { i + 1 };
            let lapl_u = self.u[left] - 2.0 * self.u[i] + self.u[right];
            let lapl_v = self.v[left] - 2.0 * self.v[i] + self.v[right];
            let uvv = self.u[i] * self.v[i] * self.v[i];
            new_u[i] = (self.u[i] + self.dt * (self.du * lapl_u - uvv + f_mod * (1.0 - self.u[i]))).clamp(0.0, 1.0);
            new_v[i] = (self.v[i] + self.dt * (self.dv * lapl_v + uvv - (f_mod + self.k) * self.v[i])).clamp(0.0, 1.0);
        }

        self.u = new_u;
        self.v = new_v;

        // Return std-dev of V field as a texture complexity scalar in [0, ~0.5]
        let mean_v: f32 = self.v.iter().sum::<f32>() / len as f32;
        let var_v: f32 = self.v.iter().map(|&x| (x - mean_v).powi(2)).sum::<f32>() / len as f32;
        var_v.sqrt()
    }
}

// ==========================================
// GRANULAR SYNTHESIS LAYER
// ==========================================
/// Reads windowed grains from a ring buffer of recently generated audio.
/// Grain density, scatter depth, and pitch spread are driven by the
/// uncertainty state — more entropy → more scattered, pitch-shifted grains.
struct GranularLayer {
    ring_buffer: Vec<f32>,
    write_head: usize,
    grain_read_pos: Vec<f32>,
    grain_length: Vec<usize>,
    grain_elapsed: Vec<usize>,
    grain_pitch: Vec<f32>,
    grain_active: Vec<bool>,
}

impl GranularLayer {
    fn new() -> Self {
        Self {
            ring_buffer: vec![0.0; GRAIN_BUFFER_LEN],
            write_head: 0,
            grain_read_pos: vec![0.0; NUM_GRAINS],
            grain_length: vec![CHUNK_SIZE; NUM_GRAINS],
            grain_elapsed: vec![0; NUM_GRAINS],
            grain_pitch: vec![1.0; NUM_GRAINS],
            grain_active: vec![false; NUM_GRAINS],
        }
    }

    fn write(&mut self, samples: &[f32]) {
        for &s in samples {
            self.ring_buffer[self.write_head] = s;
            self.write_head = (self.write_head + 1) % GRAIN_BUFFER_LEN;
        }
    }

    fn render(&mut self, grain_density: f32, grain_scatter: f32, pitch_spread: f32, rng: &mut impl Rng) -> Vec<f32> {
        let mut output = vec![0.0f32; CHUNK_SIZE];
        let inv_sqrt_n = 1.0 / (NUM_GRAINS as f32).sqrt();

        for i in 0..NUM_GRAINS {
            let finished = !self.grain_active[i] || self.grain_elapsed[i] >= self.grain_length[i];
            if finished && rng.gen::<f32>() < grain_density {
                let max_look = (GRAIN_BUFFER_LEN - CHUNK_SIZE).max(1);
                let look_back = ((grain_scatter * max_look as f32) as usize + CHUNK_SIZE)
                    .min(GRAIN_BUFFER_LEN - 1);
                let start = (self.write_head + GRAIN_BUFFER_LEN - look_back) % GRAIN_BUFFER_LEN;
                self.grain_read_pos[i] = start as f32;
                self.grain_length[i] = 1024 + rng.gen_range(0..(CHUNK_SIZE * 2));
                self.grain_elapsed[i] = 0;
                self.grain_pitch[i] = 1.0 + (rng.gen::<f32>() - 0.5) * pitch_spread * 0.6;
                self.grain_active[i] = true;
            } else if finished {
                self.grain_active[i] = false;
            }
        }

        for i in 0..NUM_GRAINS {
            if !self.grain_active[i] { continue; }
            let gl = self.grain_length[i];
            let pitch = self.grain_pitch[i];
            let read_base = self.grain_read_pos[i];
            for s in 0..CHUNK_SIZE {
                let elapsed = self.grain_elapsed[i] + s;
                if elapsed >= gl { break; }
                // Hann window: sin^2(pi * t / L)
                let env = ((elapsed as f32 / gl as f32) * PI).sin().powi(2);
                let read_f = read_base + elapsed as f32 * pitch;
                let read_i = read_f as usize % GRAIN_BUFFER_LEN;
                let read_j = (read_i + 1) % GRAIN_BUFFER_LEN;
                let frac = read_f - read_f.floor();
                let sample = self.ring_buffer[read_i] * (1.0 - frac)
                           + self.ring_buffer[read_j] * frac;
                output[s] += sample * env * inv_sqrt_n;
            }
            self.grain_elapsed[i] += CHUNK_SIZE;
        }

        output
    }
}

// ==========================================
// ADDITIVE SYNTHESIS HELPER
// ==========================================
/// Computes a harmonic series where partial n has amplitude partial_amps[n-1].
/// Phase continuity: phase_offset for partial n = n * reference_phase,
/// so all harmonics stay coherent across chunks.
fn additive_synthesis_chunk(
    partial_amps: &Tensor,
    base_freq: f32,
    t_steps: &Tensor,
    phase_ref: f32,
    device: &Device,
) -> CResult<Tensor> {
    let freq_mults: Vec<f32> = (1..=NUM_PARTIALS).map(|n| n as f32).collect();
    let fm_t = Tensor::new(freq_mults.as_slice(), device)?.reshape((NUM_PARTIALS, 1))?;
    let t_exp = t_steps.unsqueeze(0)?;
    let time_phases = fm_t.broadcast_mul(&t_exp)?.affine((2.0 * PI * base_freq) as f64, 0.0)?;
    let init_phases = fm_t.affine(phase_ref as f64, 0.0)?;
    let phases = time_phases.broadcast_add(&init_phases)?;
    let sin_mat = phases.sin()?;
    let amps_col = partial_amps.abs()?.reshape((NUM_PARTIALS, 1))?;
    let amp_sum = amps_col.sum_all()?.affine(1.0, 1e-6)?;
    let weighted = amps_col.broadcast_mul(&sin_mat)?;
    weighted.sum(0)?.broadcast_div(&amp_sum)
}

// ==========================================
// 4-OP FM NETWORK — DX7 Algorithm 5
// ==========================================
struct FMOpNetwork {
    ratio_net: Linear,
    index_net: Linear,
    carrier_weights: Tensor,
}

impl FMOpNetwork {
    fn new(vb: VBV) -> Result<Self> {
        let ratio_net = candle_nn::linear(MEMORY_DIM, FM_OPERATORS * 2, vb.pp("ratio_net"))?;
        let index_net = candle_nn::linear(MEMORY_DIM, FM_OPERATORS * 2, vb.pp("index_net"))?;
        let carrier_weights = vb.get_with_hints(
            (2,), "carrier_weights",
            candle_nn::Init::Const(0.5),
        )?;
        Ok(Self { ratio_net, index_net, carrier_weights })
    }

    fn synthesize_channel(
        ratios: &[f32],
        indices: &[f32],
        carrier_freq: f32,
        carrier_w: &[f32],
        phase_offsets: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        let n = CHUNK_SIZE;
        let dt = 1.0 / SAMPLE_RATE as f32;
        let mut output = vec![0.0f32; n];
        let mut next_phases = phase_offsets.to_vec();
        let omega: Vec<f32> = ratios.iter().map(|r| 2.0 * PI * carrier_freq * r).collect();

        for i in 0..n {
            let t = i as f32 * dt;
            let ph0 = omega[0] * t + phase_offsets[0];
            let ph2 = omega[2] * t + phase_offsets[2];
            let mod0 = indices[0] * ph0.sin();
            let mod2 = indices[2] * ph2.sin();
            let ph1 = omega[1] * t + phase_offsets[1] + mod0;
            let ph3 = omega[3] * t + phase_offsets[3] + mod2;
            output[i] = carrier_w[0] * ph1.sin() + carrier_w[1] * ph3.sin();
        }

        let advance = CHUNK_SIZE as f32 * dt;
        for op in 0..FM_OPERATORS {
            next_phases[op] = (phase_offsets[op] + omega[op] * advance) % (2.0 * PI);
        }

        (output, next_phases)
    }
}

// ==========================================
// ASYMPTOTIC DIMENSION CONTRACTION LAYER
// ==========================================
/// Inspired by large-D gravity: projects into a higher-dimensional space
/// then contracts back, scaled by 1/sqrt(LARGE_D_DIM) to maintain signal
/// magnitude as dimensionality grows. Acts as a learned non-linear
/// bottleneck that captures long-range correlations in the memory state.
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
/// Models the ringdown of a perturbed black hole.
/// Each mode has a characteristic frequency and damping rate that depends
/// on the horizon field phi. As phi grows (more resonant state),
/// damping decreases and the resonator rings longer — organically
/// lengthening the reverb tail during high-complexity passages.
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
        let qnm_modes = [
            (220.0,  0.04 * (1.0 + phi)),
            (550.0,  0.07 * (1.0 + phi)),
            (1200.0, 0.15 * (1.0 + phi)),
        ];

        for (idx, &(freq, damping)) in qnm_modes.iter().enumerate() {
            let omega = 2.0 * PI * freq / SAMPLE_RATE as f32;
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
/// Feedback Delay Network with a Hadamard mixing matrix.
/// Prime-spaced delay lines create an inharmonic reverb tail that avoids
/// metallic flutter. echo_weight (= branch aperture) controls feedback
/// energy recycling — aperture drives both neural exploration AND acoustic space.
struct FractalFDN {
    buffers: Vec<VecDeque<f32>>,
}

impl FractalFDN {
    fn new() -> Self {
        let mut buffers = Vec::new();
        for &delay in &FDN_DELAYS {
            buffers.push(VecDeque::from(vec![0.0; delay]));
        }
        Self { buffers }
    }

    fn process(&mut self, samples: &mut [f32], echo_weight: f32) {
        let mix_matrix = [
            [ 0.5,  0.5,  0.5,  0.5],
            [ 0.5, -0.5,  0.5, -0.5],
            [ 0.5,  0.5, -0.5, -0.5],
            [ 0.5, -0.5, -0.5,  0.5],
        ];

        for sample in samples.iter_mut() {
            let mut outputs = [0.0f32; FDN_DELAY_LINES];
            for i in 0..FDN_DELAY_LINES {
                outputs[i] = self.buffers[i].pop_front().unwrap_or(0.0);
            }

            let mut next_inputs = [0.0f32; FDN_DELAY_LINES];
            for i in 0..FDN_DELAY_LINES {
                let mut sum = 0.0;
                for j in 0..FDN_DELAY_LINES {
                    sum += mix_matrix[i][j] * outputs[j];
                }
                next_inputs[i] = *sample + sum * (0.42 * echo_weight);
                self.buffers[i].push_back(next_inputs[i]);
            }

            let fdn_out = (outputs[0] + outputs[1] + outputs[2] + outputs[3]) * 0.25;
            *sample = *sample * (1.0 - echo_weight * 0.2) + fdn_out * (echo_weight * 0.4);
        }
    }
}

// ==========================================
// 1. NEURAL CA (1D Scale-Invariant Fractal)
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

    fn forward(
        &self,
        x: &Tensor,
        macro_mod: Option<&Tensor>,
        metabolic_field: Option<&Tensor>,
        lorenz_noise: f32,
    ) -> CResult<Tensor> {
        let neighborhood = (self.rule)(x)?.sin()?;
        let neighborhood_t = neighborhood.transpose(1, 2)?;
        let mut delta = self.mutate.forward(&neighborhood_t)?.transpose(1, 2)?.tanh()?;

        if x.dim(D::Minus1)? >= 4 {
            let half_len = x.dim(D::Minus1)? / 2;
            let coarse_x = x.narrow(D::Minus1, 0, half_len)?;
            let coarse_rule = (self.rule)(&coarse_x)?.sin()?;
            let coarse_delta = self.mutate.forward(&coarse_rule.transpose(1, 2)?)?.transpose(1, 2)?.tanh()?;
            let upscale_coarse = coarse_delta.repeat(&[1, 1, 2])?;
            if upscale_coarse.shape() == delta.shape() {
                delta = delta.affine(0.618, 0.0)?.add(&upscale_coarse.affine(0.382, 0.0)?)?;
            }
        }

        let (decay, evolution_speed) = if let Some(m_field) = metabolic_field {
            (
                m_field.broadcast_as(x.shape())?,
                Tensor::new(0.25f32, x.device())?.broadcast_as(x.shape())?,
            )
        } else if let Some(m_mod) = macro_mod {
            let sig = candle_nn::ops::sigmoid(m_mod)?;
            let d = sig.affine((1.0 - METABOLIC_DECAY) as f64, (2.0 * METABOLIC_DECAY - 1.0) as f64)?;
            let s = sig.affine(0.5, 0.05)?;
            (d, s)
        } else {
            (
                Tensor::new(METABOLIC_DECAY, x.device())?.broadcast_as(x.shape())?,
                Tensor::new(0.25f32, x.device())?.broadcast_as(x.shape())?,
            )
        };

        let res = x.broadcast_mul(&decay)?.add(&delta.broadcast_mul(&evolution_speed)?)?;
        let normalized = self.ln.forward(&res.transpose(1, 2)?)?.transpose(1, 2)?;

        let lorenz_amp = (lorenz_noise.abs() * 0.008 + 0.002).clamp(0.001, 0.025) as f64;
        let noise = Tensor::randn_like(x, 0.0, 1.0)?
            .affine(lorenz_amp, lorenz_noise as f64 * 0.003)?;
        normalized.add(&noise)?.clamp(-1.0f32, 1.0f32)
    }
}

// ==========================================
// 2. TAPE CODEC (Dual-Lane: Value + Gradient)
// ==========================================
struct TapeCodec;
impl TapeCodec {
    fn encode(values: &[f32], gradients: &[f32]) -> String {
        let chars = [" ", "\u{00b7}", "\u{25aa}", "\u{2592}", "\u{2593}", "\u{2588}"];
        let g_chars = [" ", "\u{2198}", "\u{2192}", "\u{2197}", "\u{2191}", "!"];
        let mut tape = String::new();
        for (v, g) in values.iter().zip(gradients.iter()) {
            let v_idx = (((v + 1.0) * 0.5) * (chars.len() - 1) as f32).round() as usize;
            let g_idx = (((g.abs() * 5.0).min(1.0)) * (g_chars.len() - 1) as f32).round() as usize;
            tape.push_str(chars[v_idx.min(chars.len() - 1)]);
            tape.push_str(if g.abs() > 0.05 { g_chars[g_idx.min(g_chars.len() - 1)] } else { " " });
        }
        tape
    }
}

// ==========================================
// 3. KAN LAYER — Thermodynamic Wavefolder
// ==========================================
/// Fixed: correct in_features contraction over both in_features and num_basis.
struct KANLayer {
    centers: Tensor,
    weights: Tensor,
    variance: Tensor,
    in_features: usize,
    out_features: usize,
    num_basis: usize,
}

impl KANLayer {
    fn new(in_features: usize, out_features: usize, num_basis: usize, vb: VBV) -> Result<Self> {
        let c_vec: Vec<f32> = (0..num_basis)
            .map(|i| -1.0 + 2.0 * i as f32 / (num_basis as f32 - 1.0))
            .collect();
        let centers = Tensor::new(c_vec.as_slice(), vb.device())?.reshape((1, 1, num_basis))?;
        let weights = vb.get_with_hints(
            (out_features, in_features, num_basis), "weights",
            candle_nn::Init::Randn { mean: 0.0, stdev: 0.2 },
        )?;
        let variance = vb.get_with_hints((1,), "variance", candle_nn::Init::Const(0.5))?;
        Ok(Self { centers, weights, variance, in_features, out_features, num_basis })
    }

    fn forward(&self, x: &Tensor, work: f32) -> CResult<Tensor> {
        let orig_shape = x.shape().clone();
        let n = x.elem_count() / self.in_features;
        let x_flat = x.reshape((n, self.in_features))?;
        let x_exp = x_flat.unsqueeze(D::Minus1)?;
        let work_mod = ((-5.0 * work).exp() as f64).max(0.1);
        let var_sq = self.variance.powf(2.0)?.affine(work_mod, 1e-4)?;
        let diff = x_exp.broadcast_sub(&self.centers)?;
        let phi = diff.powf(2.0)?.broadcast_div(&var_sq)?.neg()?.exp()?;
        let phi_flat = phi.reshape((n, self.in_features * self.num_basis))?;
        let w_flat = self.weights.reshape((self.out_features, self.in_features * self.num_basis))?;
        let folded = phi_flat.matmul(&w_flat.t()?)?;
        let folded_exp = folded.unsqueeze(D::Minus1)?;
        let sec_diff = folded_exp.broadcast_sub(&self.centers)?;
        let sec_phi = sec_diff.powf(2.0)?.broadcast_div(&var_sq)?.neg()?.exp()?;
        let sec_phi_flat = sec_phi.reshape((n, self.out_features * self.num_basis))?;
        let refined = if self.in_features == self.out_features {
            sec_phi_flat.matmul(&w_flat.t()?)?
        } else {
            folded.clone()
        };
        let result = folded.affine(0.7, 0.0)?.add(&refined.affine(0.3, 0.0)?)?;
        result.reshape(orig_shape)
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
        let entropy: f32 = magnitudes.iter().fold(0.0, |acc, &m| {
            let p = m / sum_mag;
            if p > 1e-7 { acc - p * p.ln() } else { acc }
        });
        self.history.push_back(entropy);
        if self.history.len() > self.window { self.history.pop_front(); }
        let avg: f32 = self.history.iter().sum::<f32>() / self.history.len() as f32;
        Ok(serde_json::json!({ "signal": entropy, "avg": avg, "trigger": entropy < 3.0, "type": "spectral_entropy" }))
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
            let (mut num, mut den) = (0.0f32, 0.0f32);
            for (i, &y) in self.history.iter().enumerate() {
                let dx = i as f32 - x_mean;
                num += dx * (y - y_mean);
                den += dx * dx;
            }
            trend = num / (den + 1e-8);
            trigger = trend < -0.001;
        }
        Ok(serde_json::json!({ "signal": movement, "trend": trend, "trigger": trigger, "type": "movement_coherence" }))
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
}

impl AudioUncertaintyState {
    fn new() -> Self { Self { spectral: 0.0, movement: 0.0, mimic: 0.0, compositional: 0.0, phi: 0.0 } }

    fn update(
        &mut self,
        spectral_sig: &serde_json::Value,
        movement_sig: &serde_json::Value,
        mimic_sig: Option<&serde_json::Value>,
    ) {
        let s_sig   = spectral_sig["signal"].as_f64().unwrap_or(0.0) as f32;
        let avg_s   = spectral_sig["avg"].as_f64().unwrap_or(1.0) as f32;
        let m_trend = movement_sig["trend"].as_f64().unwrap_or(0.0) as f32;
        let resonance = (avg_s / (s_sig + 1e-6)).clamp(0.1, 5.0);
        self.phi = (s_sig * resonance).clamp(0.0, 10.0);
        self.spectral = (1.0 - s_sig / 8.0).max(0.0);
        self.movement = (-m_trend * 200.0).max(0.0);
        if let Some(ms) = mimic_sig {
            let drift = ms["drift"].as_f64().unwrap_or(0.0) as f32;
            self.mimic = (drift * 10.0).max(0.0);
        }
        self.compositional = (self.compositional * 0.92 + self.spectral.max(self.movement) * 0.08).min(1.0);
    }

    fn branch_aperture(&self) -> f32 {
        let raw = self.spectral * 0.35 + self.movement * 0.40 + self.mimic * 0.15 + self.compositional * 0.10;
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
        let reset  = candle_nn::ops::sigmoid(&i_r.add(&h_r)?)?;
        let update = candle_nn::ops::sigmoid(&i_z.add(&h_z)?)?;
        let new    = i_n.add(&reset.broadcast_mul(&h_n)?)?.tanh()?;
        let h_next = (update.affine(-1.0, 1.0)?.broadcast_mul(&new)? + update.broadcast_mul(h)?)?;
        self.ln.forward(&h_next)
    }
}

// ==========================================
// 7. COMPLEX AUDIO ECOSYSTEM (Titan Engine)
// ==========================================
struct ComplexAudioEcosystem {
    micro_ca: NeuralCA1D,
    macro_ca: NeuralCA1D,
    gru_memory: GRUCell,
    memory_to_macro: Linear,                           // fast linear path: mem -> CA mod
    asymptotic_contraction: AsymptoticContractionLayer, // deep non-linear large-D path
    spatial_panner: candle_nn::Sequential,
    fm_op_net: FMOpNetwork,
    partial_proj: Linear,
    wavefolder_l: KANLayer,
    wavefolder_r: KANLayer,
    current_freq_l: Tensor,
    current_freq_r: Tensor,
    base_freq_l: Tensor,
    base_freq_r: Tensor,
    op_phases_l: Vec<f32>,
    op_phases_r: Vec<f32>,
}

impl ComplexAudioEcosystem {
    fn new(vb: VBV) -> Result<Self> {
        let dev = vb.device();
        let micro_ca      = NeuralCA1D::new(CA_CHANNELS, CA_HIDDEN_MULT, vb.pp("micro_ca"))?;
        let macro_ca      = NeuralCA1D::new(CA_CHANNELS, CA_HIDDEN_MULT, vb.pp("macro_ca"))?;
        let gru_memory    = GRUCell::new(CA_CHANNELS, MEMORY_DIM, vb.pp("gru_memory"))?;
        let memory_to_macro = candle_nn::linear(MEMORY_DIM, CA_CHANNELS, vb.pp("memory_to_macro"))?;
        let asymptotic_contraction = AsymptoticContractionLayer::new(
            MEMORY_DIM, LARGE_D_DIM, CA_CHANNELS, vb.pp("asymp_contract"),
        )?;
        let spatial_panner = candle_nn::seq()
            .add(candle_nn::linear(MEMORY_DIM, 1, vb.pp("spatial_panner_0"))?)
            .add(Tanh);
        let fm_op_net    = FMOpNetwork::new(vb.pp("fm_op_net"))?;
        let partial_proj = candle_nn::linear(CA_CHANNELS, NUM_PARTIALS, vb.pp("partial_proj"))?;
        let wavefolder_l = KANLayer::new(1, 1, KAN_BASIS_FUNCTIONS, vb.pp("wavefolder_l"))?;
        let wavefolder_r = KANLayer::new(1, 1, KAN_BASIS_FUNCTIONS, vb.pp("wavefolder_r"))?;
        let base_freq_l  = vb.get_with_hints((1,), "base_freq_l", candle_nn::Init::Const(BASE_FREQ_L as f64))?;
        let base_freq_r  = vb.get_with_hints((1,), "base_freq_r", candle_nn::Init::Const(BASE_FREQ_R as f64))?;
        let current_freq_l = Tensor::new(&[BASE_FREQ_L], dev)?;
        let current_freq_r = Tensor::new(&[BASE_FREQ_R], dev)?;

        Ok(Self {
            micro_ca, macro_ca, gru_memory, memory_to_macro, asymptotic_contraction,
            spatial_panner, fm_op_net, partial_proj, wavefolder_l, wavefolder_r,
            current_freq_l, current_freq_r, base_freq_l, base_freq_r,
            op_phases_l: vec![0.0f32; FM_OPERATORS],
            op_phases_r: vec![0.0f32; FM_OPERATORS],
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &mut self,
        micro_tape: &Tensor,
        macro_tape: &Tensor,
        hidden_mem: &Tensor,
        phase_c_l: f32, phase_c_r: f32,
        force_macro_update: bool,
        mimic_loss_val: f32,
        lorenz_noise: f32,
        rd_complexity: f32,
        rhythm_gate: f32,
        granular_l: &[f32],
        granular_r: &[f32],
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, f32, f32, f32, f32)> {
        let dev = micro_tape.device();

        // ---- MACRO CA ----
        let mut next_macro_tape = macro_tape.clone();
        if force_macro_update {
            let m_field = macro_tape.abs().map_err(anyhow::Error::msg)?
                .affine(-0.01, METABOLIC_DECAY as f64).map_err(anyhow::Error::msg)?;
            next_macro_tape = self.macro_ca
                .forward(macro_tape, None, Some(&m_field), lorenz_noise)
                .map_err(anyhow::Error::msg)?;
        }

        let macro_activation = next_macro_tape.abs().map_err(anyhow::Error::msg)?
            .mean_all().map_err(anyhow::Error::msg)?;
        let metabolic_rate = (macro_activation * 5.0).map_err(anyhow::Error::msg)?
            .clamp(0.01f32, 1.0f32)?;

        // ---- MICRO CA: dual-path macro modulation ----
        // Fast linear path + deep non-linear asymptotic contraction, summed
        let linear_mod = self.memory_to_macro.forward(hidden_mem).map_err(anyhow::Error::msg)?;
        let contracted_mod = self.asymptotic_contraction.forward(hidden_mem).map_err(anyhow::Error::msg)?;
        let macro_mod = (linear_mod
            + contracted_mod
            + next_macro_tape.mean(D::Minus1).map_err(anyhow::Error::msg)?)?;

        let micro_m_field = micro_tape.abs().map_err(anyhow::Error::msg)?
            .affine(-0.01, METABOLIC_DECAY as f64).map_err(anyhow::Error::msg)?;
        let raw_next_micro = self.micro_ca
            .forward(micro_tape, Some(&macro_mod), Some(&micro_m_field), lorenz_noise)
            .map_err(anyhow::Error::msg)?;

        let one_t = Tensor::new(1.0f32, dev)?;
        let active_decay = one_t.sub(&metabolic_rate)?;
        let next_micro_tape = micro_tape.broadcast_mul(&active_decay)?
            .add(&raw_next_micro.broadcast_mul(&metabolic_rate)?)?
            .clamp(-1.0f32, 1.0f32)?;

        // ---- CA STATE SUMMARY ----
        let ca_means   = next_micro_tape.mean(D::Minus1).map_err(anyhow::Error::msg)?;
        let micro_flat = ca_means.reshape((CA_CHANNELS,)).map_err(anyhow::Error::msg)?;
        let micro_pairs = micro_flat.reshape((CA_CHANNELS / 2, 2)).map_err(anyhow::Error::msg)?;
        let sums = micro_pairs.sum(0).map_err(anyhow::Error::msg)?.to_vec1::<f32>().map_err(anyhow::Error::msg)?;
        let theta = sums[1].atan2(sums[0] + 1e-6);

        // ---- GRU MEMORY ----
        let next_hidden_mem = self.gru_memory.forward(&ca_means, hidden_mem).map_err(anyhow::Error::msg)?;

        // ---- MOVEMENT ----
        let movement_t = next_micro_tape.sub(micro_tape).map_err(anyhow::Error::msg)?
            .abs().map_err(anyhow::Error::msg)?.mean_all().map_err(anyhow::Error::msg)?;
        let movement_val = movement_t.to_scalar::<f32>().map_err(anyhow::Error::msg)?;

        let pop_l = next_micro_tape.narrow(1, 0, 1).map_err(anyhow::Error::msg)?
            .mean_all().map_err(anyhow::Error::msg)?;
        let pop_r = next_micro_tape.narrow(1, 1, 1).map_err(anyhow::Error::msg)?
            .mean_all().map_err(anyhow::Error::msg)?;
        let pop_l_val = pop_l.to_scalar::<f32>().map_err(anyhow::Error::msg)?;
        let pop_r_val = pop_r.to_scalar::<f32>().map_err(anyhow::Error::msg)?;

        // ---- FREQUENCY GLIDE ----
        let target_freq_l = self.base_freq_l
            .broadcast_add(&pop_l.affine(200.0, 0.0)?)?
            .broadcast_add(&movement_t.affine(100.0, 0.0)?)?
            .clamp(20.0f32, 4000.0f32)?;
        let target_freq_r = self.base_freq_r
            .broadcast_add(&pop_r.affine(200.0, 0.0)?)?
            .broadcast_sub(&movement_t.affine(100.0, 0.0)?)?
            .clamp(20.0f32, 4000.0f32)?;

        let gs  = Tensor::new(&[FREQ_GLIDE_SPEED], dev)?;
        let omg = Tensor::new(&[1.0f32 - FREQ_GLIDE_SPEED], dev)?;
        self.current_freq_l = self.current_freq_l.broadcast_mul(&omg)?
            .add(&target_freq_l.broadcast_mul(&gs)?)?.detach();
        self.current_freq_r = self.current_freq_r.broadcast_mul(&omg)?
            .add(&target_freq_r.broadcast_mul(&gs)?)?.detach();

        let freq_l = self.current_freq_l.reshape(())?.to_scalar::<f32>().map_err(anyhow::Error::msg)?;
        let freq_r = self.current_freq_r.reshape(())?.to_scalar::<f32>().map_err(anyhow::Error::msg)?;

        // ---- TIME BASE ----
        let t_steps = Tensor::arange(0.0f32, CHUNK_SIZE as f32, dev).map_err(anyhow::Error::msg)?
            .affine(1.0 / SAMPLE_RATE as f64, 0.0)?;

        // ================================================================
        // SYNTHESIS VOICES
        // ================================================================

        // Voice 1: ADDITIVE SYNTHESIS
        let partial_amps_raw = self.partial_proj.forward(&ca_means).map_err(anyhow::Error::msg)?;
        let partial_amps = partial_amps_raw.reshape((NUM_PARTIALS,))?.tanh()?;
        let additive_l = additive_synthesis_chunk(&partial_amps, freq_l, &t_steps, phase_c_l + theta, dev)
            .map_err(anyhow::Error::msg)?.unsqueeze(0)?;
        let additive_r = additive_synthesis_chunk(&partial_amps, freq_r, &t_steps, phase_c_r + theta, dev)
            .map_err(anyhow::Error::msg)?.unsqueeze(0)?;

        // Voice 2: 4-OP FM SYNTHESIS
        let fm_ratios_raw = self.fm_op_net.ratio_net.forward(&next_hidden_mem).map_err(anyhow::Error::msg)?;
        let fm_ratios = candle_nn::ops::sigmoid(&fm_ratios_raw)?.affine(6.875, 0.125)?;
        let fm_ratios_v = fm_ratios.reshape((FM_OPERATORS * 2,))?.to_vec1::<f32>().map_err(anyhow::Error::msg)?;
        let fm_indices_raw = self.fm_op_net.index_net.forward(&next_hidden_mem).map_err(anyhow::Error::msg)?;
        let fm_indices = candle_nn::ops::sigmoid(&fm_indices_raw)?.affine(8.0, 0.0)?;
        let fm_indices_v = fm_indices.reshape((FM_OPERATORS * 2,))?.to_vec1::<f32>().map_err(anyhow::Error::msg)?;
        let cw = candle_nn::ops::sigmoid(&self.fm_op_net.carrier_weights)?.to_vec1::<f32>()
            .map_err(anyhow::Error::msg)?;

        let (fm_l_vec, next_phases_l) = FMOpNetwork::synthesize_channel(
            &fm_ratios_v[..FM_OPERATORS], &fm_indices_v[..FM_OPERATORS],
            freq_l, &cw, &self.op_phases_l,
        );
        let (fm_r_vec, next_phases_r) = FMOpNetwork::synthesize_channel(
            &fm_ratios_v[FM_OPERATORS..], &fm_indices_v[FM_OPERATORS..],
            freq_r, &cw, &self.op_phases_r,
        );
        self.op_phases_l = next_phases_l;
        self.op_phases_r = next_phases_r;

        let fm_l = Tensor::new(fm_l_vec.as_slice(), dev)?.unsqueeze(0)?;
        let fm_r = Tensor::new(fm_r_vec.as_slice(), dev)?.unsqueeze(0)?;

        // Voice 3: GRANULAR RESYNTHESIS
        let gran_l = Tensor::new(granular_l, dev)?.unsqueeze(0)?;
        let gran_r = Tensor::new(granular_r, dev)?.unsqueeze(0)?;

        // ---- MIX WITH EUCLIDEAN RHYTHM GATING ----
        let gate_synth = if rhythm_gate > 0.5 { 1.0f64 } else { 0.35f64 };
        let gate_gran  = if rhythm_gate > 0.5 { 0.7f64  } else { 1.0f64 };

        let raw_l = additive_l.affine(ADDITIVE_MIX as f64 * gate_synth, 0.0)?
            .add(&fm_l.affine(FM_MIX as f64 * gate_synth, 0.0)?)?
            .add(&gran_l.affine(GRANULAR_MIX as f64 * gate_gran, 0.0)?)?;
        let raw_r = additive_r.affine(ADDITIVE_MIX as f64 * gate_synth, 0.0)?
            .add(&fm_r.affine(FM_MIX as f64 * gate_synth, 0.0)?)?
            .add(&gran_r.affine(GRANULAR_MIX as f64 * gate_gran, 0.0)?)?;

        // ---- REACTION-DIFFUSION PRE-GAIN SCALE ----
        let rd_scale = (1.0 + rd_complexity as f64 * 1.5).clamp(0.8, 3.0);
        let pre_fold_l = raw_l.affine(rd_scale, 0.0)?;
        let pre_fold_r = raw_r.affine(rd_scale, 0.0)?;

        // ---- KAN WAVEFOLDER ----
        let audio_l = self.wavefolder_l.forward(&pre_fold_l, mimic_loss_val).map_err(anyhow::Error::msg)?;
        let audio_r = self.wavefolder_r.forward(&pre_fold_r, mimic_loss_val).map_err(anyhow::Error::msg)?;

        // ---- DYNAMIC FILTER OPENNESS ----
        let filter_open = (next_hidden_mem.abs().map_err(anyhow::Error::msg)?
            .mean_all().map_err(anyhow::Error::msg)?.to_scalar::<f32>().map_err(anyhow::Error::msg)? * 5.0
            + movement_val).clamp(0.4, 1.0);
        let audio_l = audio_l.affine(filter_open as f64, 0.0).map_err(anyhow::Error::msg)?;
        let audio_r = audio_r.affine(filter_open as f64, 0.0).map_err(anyhow::Error::msg)?;

        // ---- SPATIAL PANNING ----
        let pan_raw = self.spatial_panner.forward(&next_hidden_mem).map_err(anyhow::Error::msg)?
            .reshape(())?.to_scalar::<f32>().map_err(anyhow::Error::msg)?;
        let pan = pan_raw.clamp(-0.5, 0.5);
        let gain_l = (0.5 * (1.0 - pan)).sqrt();
        let gain_r = (0.5 * (1.0 + pan)).sqrt();
        let audio_l = audio_l.affine((gain_l * 1.414) as f64, 0.0).map_err(anyhow::Error::msg)?;
        let audio_r = audio_r.affine((gain_r * 1.414) as f64, 0.0).map_err(anyhow::Error::msg)?;

        let stereo_chunk = Tensor::cat(&[&audio_l, &audio_r], 0)
            .map_err(anyhow::Error::msg)?.reshape((2, CHUNK_SIZE)).map_err(anyhow::Error::msg)?;

        // ---- PHASE CONTINUITY ----
        let chunk_dt = CHUNK_SIZE as f32 / SAMPLE_RATE as f32;
        let next_phase_c_l = (phase_c_l + 2.0 * PI * freq_l * chunk_dt) % (2.0 * PI);
        let next_phase_c_r = (phase_c_r + 2.0 * PI * freq_r * chunk_dt) % (2.0 * PI);

        let _ = (pop_l_val, pop_r_val);

        Ok((
            stereo_chunk,
            next_micro_tape,
            next_macro_tape,
            next_hidden_mem,
            next_phase_c_l,
            next_phase_c_r,
            movement_val,
            theta,
        ))
    }
}

// ==========================================
// 8. DEFIBRILLATOR CONTROLLER
// ==========================================
struct DefibrillatorController { net: candle_nn::Sequential }
impl DefibrillatorController {
    fn new(vb: VBV) -> Result<Self> {
        let net = candle_nn::seq()
            .add(candle_nn::linear(7, 24, vb.pp("net_0"))?)
            .add(candle_nn::Activation::Relu)
            .add(candle_nn::linear(24, 16, vb.pp("net_2"))?)
            .add(candle_nn::Activation::Relu)
            .add(candle_nn::linear(16, 3, vb.pp("net_4"))?);
        Ok(Self { net })
    }
    fn forward(&self, features: &Tensor) -> Result<(f32, f32, f32)> {
        let raw = self.net.forward(features).map_err(anyhow::Error::msg)?;
        let v = raw.reshape((3,))?.to_vec1::<f32>().map_err(anyhow::Error::msg)?;
        let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
        Ok((
            sigmoid(v[0]) * 0.20 + 0.05,
            sigmoid(v[1]) * 1.50 + 0.20,
            1.0 + sigmoid(v[2]) * 7.0,
        ))
    }
}

// ==========================================
// 9. AUDIO ARBITER (8 loss weights)
// ==========================================
struct AudioArbiter { net: candle_nn::Sequential }
impl AudioArbiter {
    fn new(vb: VBV) -> Result<Self> {
        let net = candle_nn::seq()
            .add(candle_nn::linear(12, 32, vb.pp("net_0"))?)
            .add(candle_nn::Activation::Relu)
            .add(candle_nn::linear(32, 16, vb.pp("net_2"))?)
            .add(candle_nn::Activation::Relu)
            .add(candle_nn::linear(16, 8, vb.pp("net_4"))?);
        Ok(Self { net })
    }
    fn forward(&self, features: &Tensor) -> Result<Tensor> {
        let raw = self.net.forward(features).map_err(anyhow::Error::msg)?;
        let w = raw.exp().map_err(anyhow::Error::msg)?.affine(1.0, 1.0)?.log()
            .map_err(anyhow::Error::msg)?.affine(1.0, 0.5).map_err(anyhow::Error::msg)?;
        Ok(w)
    }
}

// ==========================================
// MAIN RUNTIME
// ==========================================
fn main() -> Result<()> {
    rayon::ThreadPoolBuilder::new().num_threads(6).build_global()?;

    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    if device.is_cuda() {
        println!("--> Success: Running on GPU Workstation (CUDA Accelerated Platform)");
    } else {
        println!("--> Fallback: GPU not detected, leveraging local host CPU layer");
    }

    println!("=== TITAN AUDIO ECOSYSTEM v3: RUST EDITION (RESONANT CHAOS + PHYSICS MERGE) ===");
    println!("    Additive CA synthesis ({} partials)", NUM_PARTIALS);
    println!("    4-op DX7-style FM (Algorithm 5)");
    println!("    Granular resynthesis ({} grains, {}s buffer)", NUM_GRAINS, GRAIN_BUFFER_LEN / SAMPLE_RATE as usize);
    println!("    Gray-Scott reaction-diffusion texture");
    println!("    Lorenz attractor chaos noise");
    println!("    Euclidean rhythm gating");
    println!("    Binaural beat: {:.2} Hz (L={:.1} Hz, R={:.1} Hz)", BASE_FREQ_R - BASE_FREQ_L, BASE_FREQ_L, BASE_FREQ_R);
    println!("    QNM resonator bank (black-hole ringdown physics)");
    println!("    Fractal FDN reverb (Hadamard mixing, prime delays)");
    println!("    Asymptotic large-D contraction layer (LARGE_D_DIM={})", LARGE_D_DIM);
    println!("    Choptuik critical exponent burst scaling ({:.5})", CHOPTUIK_EXPONENT);
    println!("    Multi-scale temporal + spectral flatness + saturation loss");
    println!("    KANLayer dimension bug fixed");

    let target_loader = TargetAudioLoader::new("/home/anon/Downloads/")?;
    let mut varmap = VarMap::new();
    let model_path = "/home/anon/Downloads/titan_model.safetensors";
    if std::path::Path::new(model_path).exists() {
        println!("--> Loading existing matrix weights from {}", model_path);
        varmap.load(model_path).map_err(anyhow::Error::msg)?;
    }
    let vb = VBV::from_varmap(&varmap, DType::F32, &device);
    let mut model      = ComplexAudioEcosystem::new(vb.pp("model"))?;
    let defib_ctrl     = DefibrillatorController::new(vb.pp("defib"))?;
    let arbiter        = AudioArbiter::new(vb.pp("arbiter"))?;
    let mut spectral_mon   = SpectralEntropyMonitor::new(20);
    let mut movement_mon   = MovementCoherenceMonitor::new(20);
    let mut uncertainty    = AudioUncertaintyState::new();
    let mut optimizer      = AdamW::new_lr(varmap.all_vars(), BASE_LR).map_err(anyhow::Error::msg)?;

    // Physics / DSP runtime systems
    let mut lorenz          = LorenzAttractor::new();
    let mut rd_layer        = ReactionDiffusion::new();
    let mut granular_l      = GranularLayer::new();
    let mut granular_r      = GranularLayer::new();
    let mut qnm_resonators  = QNMFilterBank::new();
    let mut fractal_fdn_l   = FractalFDN::new();
    let mut fractal_fdn_r   = FractalFDN::new();
    let mut rng             = rand::thread_rng();

    // Euclidean rhythm: E(7,24)
    let rhythm_steps = 24usize;
    let mut rhythm_pattern = euclidean_rhythm(7, rhythm_steps);
    let mut rhythm_pos = 0usize;

    let mut micro_tape = Tensor::randn(0.0f32, 1.0f32, (1, CA_CHANNELS, TAPE_LEN), &device)
        .map_err(anyhow::Error::msg)?;
    let mut macro_tape = Tensor::randn(0.0f32, 1.0f32, (1, CA_CHANNELS, TAPE_LEN), &device)
        .map_err(anyhow::Error::msg)?;
    let mut hidden_mem = Tensor::zeros((1, MEMORY_DIM), DType::F32, &device)
        .map_err(anyhow::Error::msg)?;
    let mut phase_c_l = 0.0f32;
    let mut phase_c_r = 0.0f32;

    let total_chunks = (SAMPLE_RATE as f32 * DURATION_SECONDS / CHUNK_SIZE as f32) as usize;
    let mut audio_frames: Vec<i16> = Vec::with_capacity(total_chunks * CHUNK_SIZE * 2);
    let mut topology_history  = Vec::new();
    let mut uncertainty_trace = Vec::new();

    let mut burst_ticks  = 0;
    let mut burst_energy = 0.0f32;
    let mut total_complexity = 0.0f32;
    let mut phi_current  = 0.0f32;

    for step in 0..total_chunks {
        let aperture    = uncertainty.branch_aperture();
        let force_macro = rng.gen_range(0.0..1.0_f32) < (0.2 + aperture * 0.6);
        let prev_mimic  = if step == 0 { 0.5f32 } else { uncertainty.mimic / 10.0 };

        let lorenz_noise  = lorenz.noise_scalar();
        let rd_complexity = rd_layer.step(lorenz_noise);

        let rhythm_gate = if rhythm_pattern[rhythm_pos % rhythm_steps] { 1.0f32 } else { 0.0f32 };
        rhythm_pos = (rhythm_pos + 1) % rhythm_steps;

        let prev_move    = movement_mon.history.back().copied().unwrap_or(0.05);
        let grain_density = (0.3 + aperture * 0.5).clamp(0.0, 1.0);
        let grain_scatter = (prev_move * 8.0).clamp(0.05, 0.95);
        let pitch_spread  = (uncertainty.spectral * 0.5).clamp(0.0, 1.0);
        let gran_l_vec = granular_l.render(grain_density, grain_scatter, pitch_spread, &mut rng);
        let gran_r_vec = granular_r.render(grain_density, grain_scatter, pitch_spread, &mut rng);

        let (stereo_chunk, next_micro, next_macro, next_hidden, nc_l, nc_r, movement, theta) =
            model.forward(
                &micro_tape, &macro_tape, &hidden_mem,
                phase_c_l, phase_c_r,
                force_macro, prev_mimic,
                lorenz_noise, rd_complexity, rhythm_gate,
                &gran_l_vec, &gran_r_vec,
            )?;

        total_complexity += movement;
        let age_factor = (total_complexity / 500.0).min(0.6);

        let audio_for_loss = stereo_chunk.tanh().map_err(anyhow::Error::msg)?
            .affine((1.0 - age_factor) as f64, 0.0)?;

        // ================================================================
        // LOSS COMPUTATION
        // ================================================================
        let target_chunk = target_loader.sample_chunk(&device)?;
        let target_mono  = target_chunk.reshape((CHUNK_SIZE,)).map_err(anyhow::Error::msg)?;
        let audio_mono   = audio_for_loss.mean(0).map_err(anyhow::Error::msg)?
            .reshape((CHUNK_SIZE,)).map_err(anyhow::Error::msg)?;

        // L1: Waveform mimic loss
        let mimic_loss = audio_mono.sub(&target_mono)?
            .sqr().map_err(anyhow::Error::msg)?.mean_all().map_err(anyhow::Error::msg)?;

        // L2: Variance loss (negative = maximise)
        let var_loss = var_all(&audio_for_loss).map_err(anyhow::Error::msg)?.neg().map_err(anyhow::Error::msg)?;

        // L3: Movement loss (negative = encourage CA dynamics)
        let movement_loss = Tensor::new(movement, &device).map_err(anyhow::Error::msg)?
            .neg().map_err(anyhow::Error::msg)?;

        // L4: Multi-scale temporal loss
        let mut ms_loss = Tensor::zeros((), DType::F32, &device).map_err(anyhow::Error::msg)?;
        for &scale in &[1usize, 4, 16, 64] {
            if CHUNK_SIZE > scale {
                let diff = audio_for_loss.narrow(1, scale, CHUNK_SIZE - scale).map_err(anyhow::Error::msg)?
                    .sub(&audio_for_loss.narrow(1, 0, CHUNK_SIZE - scale).map_err(anyhow::Error::msg)?)?;
                let w = 1.0 / (1.0 + (scale as f64).sqrt());
                ms_loss = ms_loss.add(&diff.sqr()?.mean_all()?.affine(w, 0.0)?)?;
            }
        }

        // L5: Regularisation
        let reg_loss = stereo_chunk.sqr().map_err(anyhow::Error::msg)?.mean_all().map_err(anyhow::Error::msg)?;

        // L6: Energy loss (keep RMS near 0.25) — epsilon-guarded sqrt
        let rms_sq = audio_for_loss.sqr().map_err(anyhow::Error::msg)?.mean_all().map_err(anyhow::Error::msg)?;
        let eps    = Tensor::new(1e-5f32, &device).map_err(anyhow::Error::msg)?;
        let rms    = rms_sq.add(&eps).map_err(anyhow::Error::msg)?.sqrt().map_err(anyhow::Error::msg)?;
        let energy_loss = rms.sub(&Tensor::new(0.25f32, &device).map_err(anyhow::Error::msg)?)?.sqr()?;

        // L7: Spectral flatness incentive
        let log_sq   = audio_for_loss.sqr()?.affine(1.0, 1e-8)?.log()?;
        let mean_log = log_sq.mean_all()?;
        let log_mean = audio_for_loss.sqr()?.mean_all()?.affine(1.0, 1e-8)?.log()?;
        let flatness_loss = mean_log.sub(&log_mean)?.neg()?;

        // L8: Saturation loss — penalises clipping above 0.92 amplitude
        let saturation_target = Tensor::new(0.92f32, &device).map_err(anyhow::Error::msg)?;
        let saturation_loss   = audio_for_loss
            .sqr().map_err(anyhow::Error::msg)?
            .broadcast_sub(&saturation_target).map_err(anyhow::Error::msg)?
            .clamp(0.0f64, f64::MAX).map_err(anyhow::Error::msg)?
            .mean_all().map_err(anyhow::Error::msg)?;

        let mimic_scalar = mimic_loss.to_scalar::<f32>().unwrap_or(0.0);

        let arb_feats = Tensor::new(
            &[0.5f32, mimic_scalar, movement / 0.3, 0.0, 0.0, 0.0,
              uncertainty.spectral, uncertainty.movement, uncertainty.mimic,
              uncertainty.compositional, aperture, step as f32 / total_chunks as f32],
            &device,
        ).map_err(anyhow::Error::msg)?.unsqueeze(0).map_err(anyhow::Error::msg)?;
        let lw_t = arbiter.forward(&arb_feats)?;
        let lw = lw_t.reshape((8,))?.to_vec1::<f32>().map_err(anyhow::Error::msg)?;

        // lw[0]=var, lw[1]=mimic, lw[2]=move, lw[3]=multiscale, lw[4]=flatness, lw[5]=saturation, lw[6..7]=reserved
        let l1 = mimic_loss.affine((lw[1] * (1.0 - RESONANT_AUTONOMY)) as f64, 0.0)?;
        let l2 = var_loss.affine((lw[0] * (1.0 + RESONANT_AUTONOMY)) as f64, 0.0)?;
        let l3 = movement_loss.affine((lw[2] * RESONANT_AUTONOMY) as f64, 0.0)?;
        let l4 = ms_loss.affine(lw[3] as f64 * 0.4, 0.0)?;
        let l5 = reg_loss.affine(0.01, 0.0)?;
        let l6 = energy_loss.affine(2.0, 0.0)?;
        let l7 = flatness_loss.affine(lw[4] as f64 * 0.08, 0.0)?;
        let l8 = saturation_loss.affine((lw[5] * 2.0) as f64, 0.0)?;
        let total_loss = l1.add(&l2)?.add(&l3)?.add(&l4)?.add(&l5)?.add(&l6)?.add(&l7)?.add(&l8)?;

        optimizer.backward_step(&total_loss).map_err(anyhow::Error::msg)?;

        // ================================================================
        // STATE UPDATES
        // ================================================================
        micro_tape = next_micro.detach();
        macro_tape = next_macro.detach();
        hidden_mem = next_hidden.detach();
        phase_c_l = nc_l;
        phase_c_r = nc_r;

        let al = audio_for_loss.narrow(0, 0, 1).map_err(anyhow::Error::msg)?.to_vec2::<f32>().map_err(anyhow::Error::msg)?[0].clone();
        let ar = audio_for_loss.narrow(0, 1, 1).map_err(anyhow::Error::msg)?.to_vec2::<f32>().map_err(anyhow::Error::msg)?[0].clone();
        granular_l.write(&al);
        granular_r.write(&ar);

        let m_sig = movement_mon.analyze(movement)?;
        let s_sig = spectral_mon.analyze(&audio_for_loss)?;
        uncertainty.update(&s_sig, &m_sig,
            Some(&serde_json::json!({ "drift": mimic_scalar, "theta": theta })));

        phi_current = uncertainty.phi;

        if step % 96 == 0 {
            let h_act = hidden_mem.abs()?.mean_all().map_err(anyhow::Error::msg)?
                .to_scalar::<f32>().map_err(anyhow::Error::msg)?;
            let new_pulses = (3 + (h_act * 12.0) as usize).clamp(2, rhythm_steps - 1);
            rhythm_pattern = euclidean_rhythm(new_pulses, rhythm_steps);
        }

        // Defibrillator with Choptuik critical horizon scaling
        let defib_feats = Tensor::new(
            &[movement, 0.1f32, mimic_scalar, 0.5, aperture, phi_current / 10.0, step as f32 / total_chunks as f32],
            &device,
        ).map_err(anyhow::Error::msg)?.unsqueeze(0).map_err(anyhow::Error::msg)?;
        let (thresh, n_scale, lr_mult) = defib_ctrl.forward(&defib_feats)?;

        // Choptuik scalar: distance to criticality raised to the universal critical exponent.
        // As movement approaches the threshold, the scalar rises — larger burst/LR kick.
        let distance_to_horizon = (movement - thresh).abs() + 1e-6;
        let choptuik_scalar = distance_to_horizon.powf(CHOPTUIK_EXPONENT);

        if (movement < thresh || mimic_scalar > 0.8) && phi_current < 4.0 && burst_ticks == 0 {
            burst_ticks = 8;
            burst_energy = n_scale;
        }

        if burst_ticks > 0 {
            let env = (burst_ticks as f32 / 8.0).sqrt();
            let noise = Tensor::randn_like(&micro_tape, 0.0, 1.0)?
                .affine((burst_energy * env * choptuik_scalar) as f64, 0.0)?;
            micro_tape = micro_tape.add(&noise)?.clamp(-1.0f32, 1.0f32)?;
            let phi_gate = 1.0 / (1.0 + phi_current);
            optimizer.set_learning_rate(BASE_LR * (1.0 + (lr_mult - 1.0) * env) as f64 * phi_gate as f64 * choptuik_scalar as f64);
            if step % 20 == 0 {
                println!("[PACEMAKER CHOPTUIK BURST] step {} (env:{:.2} phi_gate:{:.2} scalar:{:.3})", step, env, phi_gate, choptuik_scalar);
            }
            burst_ticks -= 1;
        } else {
            let phi_gate = 1.0 / (1.0 + phi_current);
            optimizer.set_learning_rate(BASE_LR * phi_gate as f64 * choptuik_scalar as f64);
        }

        // ---- OUTPUT AUDIO ----
        // Normalise to float headroom first; QNM and FDN operate in float
        let audio_t = stereo_chunk.tanh().map_err(anyhow::Error::msg)?
            .affine((1.0 - age_factor) as f64, 0.0)?;
        let abs_max = audio_t.abs().map_err(anyhow::Error::msg)?
            .flatten_all().map_err(anyhow::Error::msg)?.max(0).map_err(anyhow::Error::msg)?
            .to_scalar::<f32>().map_err(anyhow::Error::msg)?;
        let boost = if abs_max < 0.25 { 0.25 / (abs_max + 1e-6) } else { 1.0 };
        let audio_normalized = audio_t.affine(boost as f64, 0.0).map_err(anyhow::Error::msg)?;

        let mut audio_l_vec = audio_normalized.narrow(0, 0, 1).map_err(anyhow::Error::msg)?.to_vec2::<f32>().map_err(anyhow::Error::msg)?[0].clone();
        let mut audio_r_vec = audio_normalized.narrow(0, 1, 1).map_err(anyhow::Error::msg)?.to_vec2::<f32>().map_err(anyhow::Error::msg)?[0].clone();

        // QNM resonators: black-hole ringdown colour
        qnm_resonators.process(&mut audio_l_vec, &mut audio_r_vec, phi_current);

        // Fractal FDN: aperture-driven reverb space
        let echo_aperture = aperture.min(0.7);
        fractal_fdn_l.process(&mut audio_l_vec, echo_aperture);
        fractal_fdn_r.process(&mut audio_r_vec, echo_aperture);

        // Scale to 16-bit ONLY at the final output stage
        for i in 0..CHUNK_SIZE {
            let sample_l = (audio_l_vec[i] * 32767.0).clamp(-32768.0, 32767.0) as i16;
            let sample_r = (audio_r_vec[i] * 32767.0).clamp(-32768.0, 32767.0) as i16;
            audio_frames.push(sample_l);
            audio_frames.push(sample_r);
        }

        if step % 10 == 0 {
            let topo = macro_tape.mean(1).map_err(anyhow::Error::msg)?
                .reshape((TAPE_LEN,)).map_err(anyhow::Error::msg)?.to_vec1::<f32>().map_err(anyhow::Error::msg)?;
            topology_history.push(topo);
            uncertainty_trace.push(serde_json::json!({
                "step": step,
                "spectral": uncertainty.spectral, "movement": uncertainty.movement,
                "compositional": uncertainty.compositional, "aperture": aperture,
                "phi": phi_current, "rd_complexity": rd_complexity,
                "lorenz": lorenz_noise, "rhythm_gate": rhythm_gate,
                "choptuik_scalar": choptuik_scalar,
            }));
        }
        if step % 50 == 0 {
            println!(
                "Chunk {}/{} | Move:{:.3} Mimic:{:.3} Phi:{:.2} Ap:{:.2} | R-D:{:.3} Lorenz:{:.3} Horizon:{:.4} | Rhythm:{}",
                step, total_chunks, movement, mimic_scalar, phi_current, aperture,
                rd_complexity, lorenz_noise, distance_to_horizon,
                if rhythm_gate > 0.5 { "BEAT" } else { "----" },
            );
            if let Ok(v) = micro_tape.narrow(1, 0, 1).unwrap_or(micro_tape.clone())
                .narrow(2, 0, 64).unwrap_or(micro_tape.clone()).reshape((64,))
            {
                if let Ok(vals) = v.to_vec1::<f32>() {
                    let mut grads = vec![0.0f32; 64];
                    for i in 0..63 { grads[i] = vals[i + 1] - vals[i]; }
                    println!("Tape: [{}]", TapeCodec::encode(&vals, &grads));
                }
            }
        }
    }

    // ====== POST-RUN OUTPUTS ======

    let avg_phi = uncertainty_trace.iter()
        .map(|t| t["phi"].as_f64().unwrap_or(0.0)).sum::<f64>() / uncertainty_trace.len() as f64;
    let avg_ap = uncertainty_trace.iter()
        .map(|t| t["aperture"].as_f64().unwrap_or(0.0)).sum::<f64>() / uncertainty_trace.len() as f64;
    let avg_rd = uncertainty_trace.iter()
        .map(|t| t["rd_complexity"].as_f64().unwrap_or(0.0)).sum::<f64>() / uncertainty_trace.len() as f64;

    let prompt = format!(
        "Style: {}, {}, {}, Granular Texture. Timbre: {}. Binaural: {:.1} Hz. [Phi:{:.2} Aperture:{:.2} R-D:{:.3}]",
        if avg_phi > 5.0 { "Hyper-Resonant" } else { "Chaotic" },
        if avg_ap  > 0.5 { "Evolving" } else { "Stable" },
        if total_complexity > 500.0 { "Dense" } else { "Minimal" },
        if avg_phi > 7.0 { "Crystalline" } else if avg_phi > 4.0 { "Organic" } else { "Grit" },
        BASE_FREQ_R - BASE_FREQ_L,
        avg_phi, avg_ap, avg_rd,
    );
    println!("\n=== SUNO/UDIO PRIMING PROMPT ===\n{}", prompt);
    std::fs::write("/home/anon/Downloads/suno_priming_prompt.txt", &prompt)?;

    let mut topo_w = csv::Writer::from_path("/home/anon/Downloads/ca_topology_rust.csv")?;
    for row in topology_history { topo_w.write_record(row.iter().map(|f| f.to_string()))?; }
    topo_w.flush()?;

    let mut unc_w = csv::Writer::from_path("/home/anon/Downloads/uncertainty_trace_rust.csv")?;
    unc_w.write_record(&["step","spectral","movement","compositional","aperture","phi","rd_complexity","lorenz","rhythm_gate","choptuik_scalar"])?;
    for t in uncertainty_trace {
        unc_w.write_record(&[
            t["step"].to_string(), t["spectral"].to_string(), t["movement"].to_string(),
            t["compositional"].to_string(), t["aperture"].to_string(), t["phi"].to_string(),
            t["rd_complexity"].to_string(), t["lorenz"].to_string(), t["rhythm_gate"].to_string(),
            t["choptuik_scalar"].to_string(),
        ])?;
    }
    unc_w.flush()?;
    println!("Topology and uncertainty trace saved.");

    let spec = hound::WavSpec {
        channels: 2, sample_rate: SAMPLE_RATE,
        bits_per_sample: 16, sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create("/home/anon/Downloads/rust_ecosystem_out.wav", spec)?;
    for sample in audio_frames { writer.write_sample(sample)?; }
    writer.finalize()?;
    println!("Audio saved to /home/anon/Downloads/rust_ecosystem_out.wav");

    varmap.save(model_path).map_err(anyhow::Error::msg)?;
    let sz = std::fs::metadata(model_path)?.len() as f32 / 1_048_576.0;
    println!("Model saved to {}. Size: {:.2} MB", model_path, sz);

    Ok(())
}
