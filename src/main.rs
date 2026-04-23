use anyhow::{Result, Context};
use candle_core::{Device, Tensor, DType, D, Result as CResult};
use candle_nn::{Linear, Conv1d, Conv1dConfig, VarBuilder as VBV, VarMap, Optimizer, AdamW, Module};
use std::collections::VecDeque;
use rustfft::{FftPlanner, num_complex::Complex};
use rand::Rng;

// ==========================================
// ECOSYSTEM CONFIGURATION
// ==========================================
const SAMPLE_RATE: u32 = 48000;
const DURATION_SECONDS: f32 = 300.0; 
const CHUNK_SIZE: usize = 2048;
const TAPE_LEN: usize = 512;
const CA_CHANNELS: usize = 32; 
const CA_HIDDEN_MULT: usize = 16;
const KAN_BASIS_FUNCTIONS: usize = 64;
const MEMORY_DIM: usize = 64;

const BASE_FREQ_L: f32 = 41.0;
const BASE_FREQ_R: f32 = 69.0;
const METABOLIC_DECAY: f32 = 0.9995; // Increased to prevent rapid death
const FREQ_GLIDE_SPEED: f32 = 0.0554;
const BASE_LR: f64 = 1e-3;

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
            if p.extension().map_or(false, |ext| ext == "wav") && p.file_name().unwrap() != "rust_ecosystem_out.wav" {
                println!("--> Loading target audio: {:?}", p);
                let mut reader = hound::WavReader::open(p)?;
                let samples: Vec<f32> = reader.samples::<i16>().map(|s| s.unwrap_or(0) as f32 / 32768.0).collect();
                if !samples.is_empty() {
                    buffers.push(samples);
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
// 1. NEURAL CA (1D)
// ==========================================
struct NeuralCA1D {
    rule: Box<dyn Fn(&Tensor) -> CResult<Tensor>>,
    mutate: Linear,
}

impl NeuralCA1D {
    fn new(channels: usize, hidden_mult: usize, vb: VBV) -> Result<Self> {
        let hidden_dim = channels * hidden_mult;
        let rule = conv1d_circular(channels, hidden_dim, 5, vb.pp("rule"))?;
        let mutate = candle_nn::linear(hidden_dim, channels, vb.pp("mutate"))?;
        Ok(Self { rule, mutate })
    }

    fn forward(&self, x: &Tensor, macro_mod: Option<&Tensor>) -> CResult<Tensor> {
        let neighborhood = (self.rule)(x)?.sin()?;
        let neighborhood_t = neighborhood.transpose(1, 2)?;
        let delta = self.mutate.forward(&neighborhood_t)?.transpose(1, 2)?.tanh()?;
        
        let decay = if let Some(m_mod) = macro_mod {
             let sig = candle_nn::ops::sigmoid(&m_mod.unsqueeze(D::Minus1)?)?;
             sig.affine(METABOLIC_DECAY as f64, 0.0)?
        } else {
            Tensor::new(METABOLIC_DECAY, x.device())?.broadcast_as(x.shape())?
        };

        let anti_stagnation = (Tensor::randn_like(x, 0.0, 1.0)? * 0.005)?;
        let res = ((x.broadcast_mul(&decay)?) + (delta * 0.15)?)?; // Reduced delta impact for stability
        let res = (res + anti_stagnation)?;
        res.clamp(-1.0, 1.0)
    }
}

// ==========================================
// 2. KAN LAYER
// ==========================================
struct KANLayer {
    centers: Tensor,
    weights: Tensor,
    variance: Tensor,
}

impl KANLayer {
    fn new(in_features: usize, out_features: usize, num_basis: usize, vb: VBV) -> Result<Self> {
        let mut c_vec = Vec::with_capacity(num_basis);
        for i in 0..num_basis {
            c_vec.push(-1.0 + 2.0 * (i as f32) / (num_basis as f32 - 1.0));
        }
        let centers = Tensor::new(c_vec, vb.device())?.reshape((1, 1, num_basis))?;
        let weights = vb.get_with_hints((out_features, in_features, num_basis), "weights", candle_nn::Init::Randn { mean: 0.0, stdev: 0.2 })?;
        let variance = vb.get_with_hints((1,), "variance", candle_nn::Init::Const(0.5))?;
        Ok(Self { centers, weights, variance })
    }

    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let x_expanded = x.unsqueeze(D::Minus1)?;
        let diff = x_expanded.broadcast_sub(&self.centers)?;
        let var_sq = (self.variance.powf(2.0)? + 1e-4)?;
        let phi = diff.powf(2.0)?.broadcast_div(&var_sq)?.neg()?.exp()?;
        let weights_reshaped = self.weights.reshape((self.weights.dim(0)?, self.weights.dim(2)?))?;
        let folded = phi.matmul(&weights_reshaped.transpose(0, 1)?.unsqueeze(0)?)?;
        Ok(folded)
    }
}

// ==========================================
// 3. MONITOR AGENTS
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
        let magnitudes: Vec<f32> = buffer.iter().take(n/2).map(|c| c.norm()).collect();
        let sum_mag: f32 = magnitudes.iter().sum::<f32>() + 1e-8;
        let mut entropy = 0.0;
        for m in magnitudes {
            let p = m / sum_mag;
            if p > 1e-7 { entropy -= p * p.ln(); }
        }
        self.history.push_back(entropy);
        if self.history.len() > self.window { self.history.pop_front(); }
        let avg_entropy: f32 = self.history.iter().sum::<f32>() / self.history.len() as f32;
        Ok(serde_json::json!({"signal": entropy, "avg": avg_entropy, "trigger": entropy < 3.0, "type": "spectral_entropy"}))
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
// 4. AUDIO UNCERTAINTY STATE
// ==========================================
struct AudioUncertaintyState {
    spectral: f32,
    movement: f32,
    mimic: f32,
    compositional: f32,
}

impl AudioUncertaintyState {
    fn new() -> Self { Self { spectral: 0.0, movement: 0.0, mimic: 0.0, compositional: 0.0 } }
    fn update(&mut self, spectral_sig: &serde_json::Value, movement_sig: &serde_json::Value, mimic_sig: Option<&serde_json::Value>) {
        let s_sig = spectral_sig["signal"].as_f64().unwrap_or(0.0) as f32;
        let m_trend = movement_sig["trend"].as_f64().unwrap_or(0.0) as f32;
        self.spectral = (1.0 - (s_sig / 8.0)).max(0.0);
        self.movement = (-m_trend * 200.0).max(0.0);
        if let Some(ms) = mimic_sig {
             let drift = ms["drift"].as_f64().unwrap_or(0.0) as f32;
             self.mimic = (drift * 10.0).max(0.0);
        }
        self.compositional = (self.compositional * 0.92) + (self.spectral.max(self.movement) * 0.08);
        self.compositional = self.compositional.min(1.0);
    }
    fn branch_aperture(&self) -> f32 {
        let raw = (self.spectral * 0.35) + (self.movement * 0.40) + (self.mimic * 0.15) + (self.compositional * 0.10);
        raw.clamp(0.05, 1.0)
    }
}

// ==========================================
// 5. GRU CELL
// ==========================================
struct GRUCell {
    w_ih: Linear,
    w_hh: Linear,
}

impl GRUCell {
    fn new(input_size: usize, hidden_size: usize, vb: VBV) -> Result<Self> {
        let w_ih = candle_nn::linear(input_size, 3 * hidden_size, vb.pp("w_ih"))?;
        let w_hh = candle_nn::linear(hidden_size, 3 * hidden_size, vb.pp("w_hh"))?;
        Ok(Self { w_ih, w_hh })
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
        Ok(h_next)
    }
}

// ==========================================
// 6. COMPLEX AUDIO ECOSYSTEM
// ==========================================
struct ComplexAudioEcosystem {
    micro_ca: NeuralCA1D,
    macro_ca: NeuralCA1D,
    gru_memory: GRUCell,
    memory_to_macro: Linear,
    spatial_panner: candle_nn::Sequential,
    fm_mod_ratio: candle_nn::Sequential,
    fm_mod_index: candle_nn::Sequential,
    wavefolder_l: KANLayer,
    wavefolder_r: KANLayer,
    current_freq_l: f32,
    current_freq_r: f32,
    base_freq_l: Tensor,
    base_freq_r: Tensor,
}

impl ComplexAudioEcosystem {
    fn new(vb: VBV) -> Result<Self> {
        let micro_ca = NeuralCA1D::new(CA_CHANNELS, CA_HIDDEN_MULT, vb.pp("micro_ca"))?;
        let macro_ca = NeuralCA1D::new(CA_CHANNELS, CA_HIDDEN_MULT, vb.pp("macro_ca"))?;
        let gru_memory = GRUCell::new(CA_CHANNELS, MEMORY_DIM, vb.pp("gru_memory"))?;
        let memory_to_macro = candle_nn::linear(MEMORY_DIM, CA_CHANNELS, vb.pp("memory_to_macro"))?;
        let spatial_panner = candle_nn::seq().add(candle_nn::linear(MEMORY_DIM, 1, vb.pp("spatial_panner_0"))?).add(Tanh); 
        let fm_mod_ratio = candle_nn::seq().add(candle_nn::linear(MEMORY_DIM, 2, vb.pp("fm_mod_ratio_0"))?).add(Softplus);
        let fm_mod_index = candle_nn::seq().add(candle_nn::linear(MEMORY_DIM, 2, vb.pp("fm_mod_index_0"))?).add(Sigmoid);
        let wavefolder_l = KANLayer::new(1, 1, KAN_BASIS_FUNCTIONS, vb.pp("wavefolder_l"))?;
        let wavefolder_r = KANLayer::new(1, 1, KAN_BASIS_FUNCTIONS, vb.pp("wavefolder_r"))?;
        let base_freq_l = vb.get_with_hints((1,), "base_freq_l", candle_nn::Init::Const(BASE_FREQ_L as f64))?;
        let base_freq_r = vb.get_with_hints((1,), "base_freq_r", candle_nn::Init::Const(BASE_FREQ_R as f64))?;
        Ok(Self {
            micro_ca, macro_ca, gru_memory, memory_to_macro, spatial_panner, fm_mod_ratio, fm_mod_index,
            wavefolder_l, wavefolder_r, current_freq_l: BASE_FREQ_L, current_freq_r: BASE_FREQ_R, base_freq_l, base_freq_r,
        })
    }

    fn forward(
        &mut self,
        micro_tape: &Tensor,
        macro_tape: &Tensor,
        hidden_mem: &Tensor,
        phase_c_l: f32, phase_c_r: f32,
        phase_m_l: f32, phase_m_r: f32,
        force_macro_update: bool,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, f32, f32, f32, f32, f32)> {
        let mut next_macro_tape = macro_tape.clone();
        if force_macro_update {
            next_macro_tape = self.macro_ca.forward(macro_tape, None).map_err(anyhow::Error::msg)?;
        }
        let macro_mod = (self.memory_to_macro.forward(hidden_mem).map_err(anyhow::Error::msg)? + next_macro_tape.mean(D::Minus1).map_err(anyhow::Error::msg)?)?;
        let next_micro_tape = self.micro_ca.forward(micro_tape, Some(&macro_mod)).map_err(anyhow::Error::msg)?;
        let tape_features = next_micro_tape.mean(D::Minus1).map_err(anyhow::Error::msg)?;
        let next_hidden_mem = self.gru_memory.forward(&tape_features, hidden_mem).map_err(anyhow::Error::msg)?;
        let movement = next_micro_tape.sub(micro_tape).map_err(anyhow::Error::msg)?.abs().map_err(anyhow::Error::msg)?.mean_all().map_err(anyhow::Error::msg)?.reshape(())?.to_scalar::<f32>().map_err(anyhow::Error::msg)?;
        let pop_l = next_micro_tape.narrow(1, 0, 1).map_err(anyhow::Error::msg)?.mean_all().map_err(anyhow::Error::msg)?.reshape(())?.to_scalar::<f32>().map_err(anyhow::Error::msg)?;
        let pop_r = next_micro_tape.narrow(1, 1, 1).map_err(anyhow::Error::msg)?.mean_all().map_err(anyhow::Error::msg)?.reshape(())?.to_scalar::<f32>().map_err(anyhow::Error::msg)?;
        let b_freq_l = self.base_freq_l.reshape(())?.to_scalar::<f32>().map_err(anyhow::Error::msg)?.abs();
        let b_freq_r = self.base_freq_r.reshape(())?.to_scalar::<f32>().map_err(anyhow::Error::msg)?.abs();
        let target_freq_l = (b_freq_l + pop_l * 200.0 + movement * 100.0).clamp(20.0, 4000.0);
        let target_freq_r = (b_freq_r + pop_r * 200.0 - movement * 100.0).clamp(20.0, 4000.0);
        self.current_freq_l = self.current_freq_l * (1.0 - FREQ_GLIDE_SPEED) + target_freq_l * FREQ_GLIDE_SPEED;
        self.current_freq_r = self.current_freq_r * (1.0 - FREQ_GLIDE_SPEED) + target_freq_r * FREQ_GLIDE_SPEED;
        let fm_ratios = (self.fm_mod_ratio.forward(&next_hidden_mem).map_err(anyhow::Error::msg)?.affine(4.0, 0.0))?;
        let fm_indices = (self.fm_mod_index.forward(&next_hidden_mem).map_err(anyhow::Error::msg)?.affine(5.0, 0.0))?;
        let device = micro_tape.device();
        let t_steps = (Tensor::arange(0.0f32, CHUNK_SIZE as f32, device).map_err(anyhow::Error::msg)?.affine(1.0 / SAMPLE_RATE as f64, 0.0))?;
        let fm_ratio_l = fm_ratios.narrow(1, 0, 1).map_err(anyhow::Error::msg)?.reshape(())?.to_scalar::<f32>().map_err(anyhow::Error::msg)?;
        let fm_ratio_r = fm_ratios.narrow(1, 1, 1).map_err(anyhow::Error::msg)?.reshape(())?.to_scalar::<f32>().map_err(anyhow::Error::msg)?;
        let fm_idx_l = fm_indices.narrow(1, 0, 1).map_err(anyhow::Error::msg)?.reshape(())?.to_scalar::<f32>().map_err(anyhow::Error::msg)?;
        let fm_idx_r = fm_indices.narrow(1, 1, 1).map_err(anyhow::Error::msg)?.reshape(())?.to_scalar::<f32>().map_err(anyhow::Error::msg)?;
        let mod_freq_l = self.current_freq_l * fm_ratio_l;
        let mod_freq_r = self.current_freq_r * fm_ratio_r;
        let phases_m_l = (t_steps.affine((2.0 * std::f32::consts::PI * mod_freq_l) as f64, phase_m_l as f64))?;
        let phases_m_r = (t_steps.affine((2.0 * std::f32::consts::PI * mod_freq_r) as f64, phase_m_r as f64))?;
        let modulator_l = phases_m_l.sin().map_err(anyhow::Error::msg)?.affine(fm_idx_l as f64, 0.0)?;
        let modulator_r = phases_m_r.sin().map_err(anyhow::Error::msg)?.affine(fm_idx_r as f64, 0.0)?;
        let phases_c_l = (t_steps.affine((2.0 * std::f32::consts::PI * self.current_freq_l) as f64, phase_c_l as f64)? + modulator_l)?; 
        let phases_c_r = (t_steps.affine((2.0 * std::f32::consts::PI * self.current_freq_r) as f64, phase_c_r as f64)? + modulator_r)?;
        let next_phase_c_l = phases_c_l.narrow(0, CHUNK_SIZE - 1, 1).map_err(anyhow::Error::msg)?.reshape(())?.to_scalar::<f32>().map_err(anyhow::Error::msg)? % (2.0 * std::f32::consts::PI);
        let next_phase_c_r = phases_c_r.narrow(0, CHUNK_SIZE - 1, 1).map_err(anyhow::Error::msg)?.reshape(())?.to_scalar::<f32>().map_err(anyhow::Error::msg)? % (2.0 * std::f32::consts::PI);
        let next_phase_m_l = phases_m_l.narrow(0, CHUNK_SIZE - 1, 1).map_err(anyhow::Error::msg)?.reshape(())?.to_scalar::<f32>().map_err(anyhow::Error::msg)? % (2.0 * std::f32::consts::PI);
        let next_phase_m_r = phases_m_r.narrow(0, CHUNK_SIZE - 1, 1).map_err(anyhow::Error::msg)?.reshape(())?.to_scalar::<f32>().map_err(anyhow::Error::msg)? % (2.0 * std::f32::consts::PI);
        let osc_l = phases_c_l.sin().map_err(anyhow::Error::msg)?.unsqueeze(0).map_err(anyhow::Error::msg)?;
        let osc_r = phases_c_r.sin().map_err(anyhow::Error::msg)?.unsqueeze(0).map_err(anyhow::Error::msg)?;
        let audio_l = self.wavefolder_l.forward(&osc_l).map_err(anyhow::Error::msg)?;
        let audio_r = self.wavefolder_r.forward(&osc_r).map_err(anyhow::Error::msg)?;
        let filter_openness = (next_hidden_mem.abs().map_err(anyhow::Error::msg)?.mean_all().map_err(anyhow::Error::msg)?.reshape(())?.to_scalar::<f32>().map_err(anyhow::Error::msg)? * 5.0 + movement).clamp(0.4, 1.0);
        let audio_l = audio_l.affine(filter_openness as f64, 0.0).map_err(anyhow::Error::msg)?;
        let audio_r = audio_r.affine(filter_openness as f64, 0.0).map_err(anyhow::Error::msg)?;
        
        // Panning: Max 75% to one side
        let pan_raw = self.spatial_panner.forward(&next_hidden_mem).map_err(anyhow::Error::msg)?.reshape(())?.to_scalar::<f32>().map_err(anyhow::Error::msg)?;
        let pan = pan_raw.clamp(-0.5, 0.5); // Maps to roughly 0.25 - 0.75 range
        let gain_l = (0.5 * (1.0 - pan)).sqrt();
        let gain_r = (0.5 * (1.0 + pan)).sqrt();
        let audio_l = audio_l.affine((gain_l * 1.414) as f64, 0.0).map_err(anyhow::Error::msg)?;
        let audio_r = audio_r.affine((gain_r * 1.414) as f64, 0.0).map_err(anyhow::Error::msg)?;
        let stereo_chunk = Tensor::cat(&[&audio_l, &audio_r], 0).map_err(anyhow::Error::msg)?.reshape((2, CHUNK_SIZE)).map_err(anyhow::Error::msg)?;
        Ok((stereo_chunk, next_micro_tape, next_macro_tape, next_hidden_mem, next_phase_c_l, next_phase_c_r, next_phase_m_l, next_phase_m_r, movement))
    }
}

// ==========================================
// 7. DEFIBRILLATOR CONTROLLER
// ==========================================
struct DefibrillatorController { net: candle_nn::Sequential }
impl DefibrillatorController {
    fn new(vb: VBV) -> Result<Self> {
        let net = candle_nn::seq().add(candle_nn::linear(7, 24, vb.pp("net_0"))?).add(candle_nn::Activation::Relu).add(candle_nn::linear(24, 16, vb.pp("net_2"))?).add(candle_nn::Activation::Relu).add(candle_nn::linear(16, 3, vb.pp("net_4"))?);
        Ok(Self { net })
    }
    fn forward(&self, features: &Tensor) -> Result<(f32, f32, f32)> {
        let raw = self.net.forward(features).map_err(anyhow::Error::msg)?;
        let raw_v = raw.reshape((3,))?.to_vec1::<f32>().map_err(anyhow::Error::msg)?;
        let threshold = (1.0 / (1.0 + (-raw_v[0]).exp())) * 0.20 + 0.05;
        let noise_scale = (1.0 / (1.0 + (-raw_v[1]).exp())) * 1.5 + 0.2;
        let lr_multiplier = 1.0 + (1.0 / (1.0 + (-raw_v[2]).exp())) * 7.0;
        Ok((threshold, noise_scale, lr_multiplier))
    }
}

// ==========================================
// 8. AUDIO ARBITER
// ==========================================
struct AudioArbiter { net: candle_nn::Sequential }
impl AudioArbiter {
    fn new(vb: VBV) -> Result<Self> {
        let net = candle_nn::seq().add(candle_nn::linear(12, 32, vb.pp("net_0"))?).add(candle_nn::Activation::Relu).add(candle_nn::linear(32, 16, vb.pp("net_2"))?).add(candle_nn::Activation::Relu).add(candle_nn::linear(16, 6, vb.pp("net_4"))?);
        Ok(Self { net })
    }
    fn forward(&self, features: &Tensor) -> Result<Tensor> {
        let raw = self.net.forward(features).map_err(anyhow::Error::msg)?;
        let weights = raw.exp().map_err(anyhow::Error::msg)?.affine(1.0, 1.0)?.log().map_err(anyhow::Error::msg)?.affine(1.0, 0.5).map_err(anyhow::Error::msg)?;
        Ok(weights)
    }
}

fn main() -> Result<()> {
    rayon::ThreadPoolBuilder::new().num_threads(6).build_global()?;
    let device = Device::Cpu; 
    println!("=== TITAN AUDIO ECOSYSTEM: RUST EDITION ===");

    let target_loader = TargetAudioLoader::new("/sdcard/Download")?;
    let varmap = VarMap::new();
    let vb = VBV::from_varmap(&varmap, DType::F32, &device);
    let mut model = ComplexAudioEcosystem::new(vb.pp("model"))?;
    let defib_ctrl = DefibrillatorController::new(vb.pp("defib"))?;
    let arbiter = AudioArbiter::new(vb.pp("arbiter"))?;
    let mut spectral_mon = SpectralEntropyMonitor::new(20);
    let mut movement_mon = MovementCoherenceMonitor::new(20);
    let mut uncertainty = AudioUncertaintyState::new();
    let mut optimizer = AdamW::new_lr(varmap.all_vars(), BASE_LR).map_err(anyhow::Error::msg)?;

    let mut micro_tape = Tensor::randn(0.0f32, 1.0f32, (1, CA_CHANNELS, TAPE_LEN), &device).map_err(anyhow::Error::msg)?;
    let mut macro_tape = Tensor::randn(0.0f32, 1.0f32, (1, CA_CHANNELS, TAPE_LEN), &device).map_err(anyhow::Error::msg)?;
    let mut hidden_mem = Tensor::zeros((1, MEMORY_DIM), DType::F32, &device).map_err(anyhow::Error::msg)?;
    let mut phase_c_l = 0.0f32; let mut phase_c_r = 0.0f32; let mut phase_m_l = 0.0f32; let mut phase_m_r = 0.0f32;

    let total_chunks = (SAMPLE_RATE as f32 * DURATION_SECONDS / CHUNK_SIZE as f32) as usize;
    let mut audio_frames: Vec<i16> = Vec::with_capacity(total_chunks * CHUNK_SIZE * 2);
    let mut topology_history = Vec::new();
    let mut uncertainty_trace = Vec::new();

    for step in 0..total_chunks {
        let aperture = uncertainty.branch_aperture();
        let force_macro = rand::thread_rng().gen_range(0.0..1.0) < (0.2 + aperture * 0.6);
        let (stereo_chunk, next_micro_tape, next_macro_tape, next_hidden_mem, nc_l, nc_r, nm_l, nm_r, movement) = model.forward(&micro_tape, &macro_tape, &hidden_mem, phase_c_l, phase_c_r, phase_m_l, phase_m_r, force_macro)?;

        // Losses
        let target_chunk = target_loader.sample_chunk(&device)?;
        let mimic_loss = stereo_chunk.mean(0)?.sub(&target_chunk.reshape((CHUNK_SIZE,))?)?.sqr()?.mean_all()?;
        let var_loss = var_all(&stereo_chunk).map_err(anyhow::Error::msg)?.neg().map_err(anyhow::Error::msg)?;
        let movement_loss = Tensor::new(movement, &device).map_err(anyhow::Error::msg)?.neg().map_err(anyhow::Error::msg)?;
        let diff = stereo_chunk.narrow(1, 1, CHUNK_SIZE - 1).map_err(anyhow::Error::msg)?.sub(&stereo_chunk.narrow(1, 0, CHUNK_SIZE - 1).map_err(anyhow::Error::msg)?)?;
        let roughness_loss = diff.sqr().map_err(anyhow::Error::msg)?.mean_all().map_err(anyhow::Error::msg)?;

        let arb_features = Tensor::new(&[0.5, mimic_loss.to_scalar::<f32>().unwrap_or(0.0), movement / 0.3, 0.0, 0.0, 0.0, uncertainty.spectral, uncertainty.movement, uncertainty.mimic, uncertainty.compositional, aperture, step as f32 / total_chunks as f32], &device).map_err(anyhow::Error::msg)?.unsqueeze(0).map_err(anyhow::Error::msg)?;
        let loss_weights = arbiter.forward(&arb_features)?;
        let lw = loss_weights.reshape((6,))?.to_vec1::<f32>().map_err(anyhow::Error::msg)?;

        let total_loss = (mimic_loss.affine(lw[1] as f64, 0.0).map_err(anyhow::Error::msg)? + var_loss.affine(lw[0] as f64, 0.0).map_err(anyhow::Error::msg)? + movement_loss.affine(lw[2] as f64, 0.0).map_err(anyhow::Error::msg)? + roughness_loss.affine(lw[3] as f64, 0.0).map_err(anyhow::Error::msg)?)?;
        optimizer.backward_step(&total_loss).map_err(anyhow::Error::msg)?;

        micro_tape = next_micro_tape.detach().map_err(anyhow::Error::msg)?;
        macro_tape = next_macro_tape.detach().map_err(anyhow::Error::msg)?;
        hidden_mem = next_hidden_mem.detach().map_err(anyhow::Error::msg)?;
        phase_c_l = nc_l; phase_c_r = nc_r; phase_m_l = nm_l; phase_m_r = nm_r;

        let m_sig = movement_mon.analyze(movement)?;
        let s_sig = spectral_mon.analyze(&stereo_chunk)?;
        let mimic_drift = mimic_loss.to_scalar::<f32>().unwrap_or(0.0);
        uncertainty.update(&s_sig, &m_sig, Some(&serde_json::json!({"drift": mimic_drift})));

        let defib_features = Tensor::new(&[movement, 0.1, mimic_drift, 0.5, aperture, 0.0, step as f32 / total_chunks as f32], &device).map_err(anyhow::Error::msg)?.unsqueeze(0).map_err(anyhow::Error::msg)?;
        let (thresh, n_scale, lr_mult) = defib_ctrl.forward(&defib_features)?;
        if movement < thresh || mimic_drift > 0.8 {
            let noise = (Tensor::randn_like(&micro_tape, 0.0, 1.0).map_err(anyhow::Error::msg)?.affine(n_scale as f64, 0.0))?;
            micro_tape = (micro_tape + noise).map_err(anyhow::Error::msg)?.clamp(-1.0, 1.0).map_err(anyhow::Error::msg)?;
            optimizer.set_learning_rate(BASE_LR * lr_mult as f64);
            if step % 20 == 0 { println!("[DEFIBRILLATOR FIRED] at chunk {}", step); }
        } else { optimizer.set_learning_rate(BASE_LR); }

        let audio_data = stereo_chunk.tanh().map_err(anyhow::Error::msg)?.affine(32767.0, 0.0).map_err(anyhow::Error::msg)?;
        let audio_l = audio_data.narrow(0, 0, 1).map_err(anyhow::Error::msg)?.to_vec2::<f32>().map_err(anyhow::Error::msg)?[0].clone();
        let audio_r = audio_data.narrow(0, 1, 1).map_err(anyhow::Error::msg)?.to_vec2::<f32>().map_err(anyhow::Error::msg)?[0].clone();
        for i in 0..CHUNK_SIZE { audio_frames.push(audio_l[i] as i16); audio_frames.push(audio_r[i] as i16); }

        if step % 10 == 0 {
            let topology_state = macro_tape.mean(1).map_err(anyhow::Error::msg)?.reshape((TAPE_LEN,)).map_err(anyhow::Error::msg)?.to_vec1::<f32>().map_err(anyhow::Error::msg)?;
            topology_history.push(topology_state);
            uncertainty_trace.push(serde_json::json!({"step": step, "spectral": uncertainty.spectral, "movement": uncertainty.movement, "compositional": uncertainty.compositional, "aperture": aperture}));
        }
        if step % 50 == 0 { println!("Chunk {}/{} | Move: {:.3} | Mimic: {:.3} | Aperture: {:.2}", step, total_chunks, movement, mimic_drift, aperture); }
    }

    // Save CSVs
    let mut topo_writer = csv::Writer::from_path("/sdcard/Download/ca_topology_rust.csv")?;
    for row in topology_history { topo_writer.write_record(row.iter().map(|f| f.to_string()))?; }
    topo_writer.flush()?;
    let mut unc_writer = csv::Writer::from_path("/sdcard/Download/uncertainty_trace_rust.csv")?;
    unc_writer.write_record(&["step", "spectral", "movement", "compositional", "aperture"])?;
    for trace in uncertainty_trace { unc_writer.write_record(&[trace["step"].to_string(), trace["spectral"].to_string(), trace["movement"].to_string(), trace["compositional"].to_string(), trace["aperture"].to_string()])?; }
    unc_writer.flush()?;
    println!("Topology and uncertainty trace saved to /sdcard/Download/.");

    let spec = hound::WavSpec { channels: 2, sample_rate: SAMPLE_RATE, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut writer = hound::WavWriter::create("/sdcard/Download/rust_ecosystem_out.wav", spec)?;
    for sample in audio_frames { writer.write_sample(sample)?; }
    writer.finalize()?;
    println!("Audio saved to /sdcard/Download/rust_ecosystem_out.wav");
    Ok(())
}
