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
const DURATION_SECONDS: f32 = 420.0; 
const CHUNK_SIZE: usize = 2048;
const TAPE_LEN: usize = 512;
const CA_CHANNELS: usize = 144;
const CA_HIDDEN_MULT: usize = 64;
const KAN_BASIS_FUNCTIONS: usize = 144;
const MEMORY_DIM: usize = 256;

const BASE_FREQ_L: f32 = 49.0;
const BASE_FREQ_R: f32 = 69.0;
const METABOLIC_DECAY: f32 = 0.99999; 
const FREQ_GLIDE_SPEED: f32 = 0.0554;
const BASE_LR: f64 = 1e-4;
const RESONANT_AUTONOMY: f32 = 0.314; 

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
                let samples: Vec<i16> = reader.samples::<i16>().map(|s| s.unwrap_or(0)).collect();
                let f32_samples: Vec<f32> = samples.iter().map(|&s| s as f32 / 32768.0).collect();
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

    fn forward(&self, x: &Tensor, macro_mod: Option<&Tensor>, metabolic_field: Option<&Tensor>) -> CResult<Tensor> {
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
                let d_scaled = delta.affine(0.618, 0.0)?;
                let u_scaled = upscale_coarse.affine(0.382, 0.0)?;
                delta = d_scaled.add(&u_scaled)?;
            }
        }

        let (decay, evolution_speed) = if let Some(m_field) = metabolic_field {
             (m_field.broadcast_as(x.shape())?, Tensor::new(0.25f32, x.device())?.broadcast_as(x.shape())?)
        } else if let Some(m_mod) = macro_mod {
             let sig = candle_nn::ops::sigmoid(m_mod)?;
             let d = sig.affine((1.0 - METABOLIC_DECAY) as f64, (2.0 * METABOLIC_DECAY - 1.0) as f64)?;
             let s = sig.affine(0.5, 0.05)?; 
             (d, s)
        } else {
            (Tensor::new(METABOLIC_DECAY, x.device())?.broadcast_as(x.shape())?, Tensor::new(0.25f32, x.device())?.broadcast_as(x.shape())?)
        };

        let decayed_x = x.broadcast_mul(&decay)?;
        let evolved_delta = delta.broadcast_mul(&evolution_speed)?;
        let res = decayed_x.add(&evolved_delta)?;
        
        let res_t = res.transpose(1, 2)?;
        let normalized = self.ln.forward(&res_t)?.transpose(1, 2)?;
        
        let rand_noise = Tensor::randn_like(x, 0.0, 1.0)?;
        let anti_stagnation = rand_noise.affine(0.005, 0.0)?;
        normalized.add(&anti_stagnation)?.clamp(-1.0f32, 1.0f32)
    }
}

// ==========================================
// 2. TAPE CODEC (Dual-Lane: Value + Gradient)
// ==========================================
struct TapeCodec;
impl TapeCodec {
    fn encode(values: &[f32], gradients: &[f32]) -> String {
        let chars = [" ", "·", "▪", "▒", "▓", "█"];
        let g_chars = [" ", "↘", "→", "↗", "↑", "!" ];
        let mut tape = String::new();
        for (v, g) in values.iter().zip(gradients.iter()) {
            let v_idx = (((v + 1.0) * 0.5) * (chars.len() - 1) as f32).round() as usize;
            let g_idx = (((g.abs() * 5.0).min(1.0)) * (g_chars.len() - 1) as f32).round() as usize;
            tape.push_str(chars[v_idx.min(chars.len()-1)]);
            if g.abs() > 0.05 {
                tape.push_str(g_chars[g_idx.min(g_chars.len()-1)]);
            } else {
                tape.push_str(" ");
            }
        }
        tape
    }
}

// ==========================================
// 3. KAN LAYER (Recursive Fused Wavefolder)
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

    fn forward(&self, x: &Tensor, work: f32) -> CResult<Tensor> {
        let x_expanded = x.unsqueeze(D::Minus1)?;
        let diff = x_expanded.broadcast_sub(&self.centers)?;
        
        let work_mod = ((-5.0 * work).exp() as f64).max(0.1);
        let var_sq = (self.variance.powf(2.0)?.affine(work_mod, 1e-4))?;
        
        let phi = diff.powf(2.0)?.broadcast_div(&var_sq)?.neg()?.exp()?;
        let weights_reshaped = self.weights.reshape((self.weights.dim(0)?, self.weights.dim(2)?))?;
        let folded = phi.matmul(&weights_reshaped.transpose(0, 1)?.unsqueeze(0)?)?;
        
        let secondary_diff = folded.broadcast_sub(&self.centers)?;
        let secondary_phi = secondary_diff.powf(2.0)?.broadcast_div(&var_sq)?.neg()?.exp()?;
        let refined_fold = secondary_phi.matmul(&weights_reshaped.transpose(0, 1)?.unsqueeze(0)?)?;

        let f_scaled = folded.affine(0.7, 0.0)?;
        let r_scaled = refined_fold.affine(0.3, 0.0)?;
        f_scaled.add(&r_scaled)
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
    fn update(&mut self, spectral_sig: &serde_json::Value, movement_sig: &serde_json::Value, mimic_sig: Option<&serde_json::Value>) {
        let s_sig = spectral_sig["signal"].as_f64().unwrap_or(0.0) as f32;
        let avg_s = spectral_sig["avg"].as_f64().unwrap_or(1.0) as f32;
        let m_trend = movement_sig["trend"].as_f64().unwrap_or(0.0) as f32;
        
        let resonance = (avg_s / (s_sig + 1e-6)).clamp(0.1, 5.0);
        self.phi = (s_sig * resonance).clamp(0.0, 10.0);

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
// 7. COMPLEX AUDIO ECOSYSTEM (Titan GPU Model)
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
    current_freq_l: Tensor,
    current_freq_r: Tensor,
    base_freq_l: Tensor,
    base_freq_r: Tensor,
}

impl ComplexAudioEcosystem {
    fn new(vb: VBV) -> Result<Self> {
        let dev = vb.device();
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
        
        let current_freq_l = Tensor::new(&[BASE_FREQ_L], dev)?;
        let current_freq_r = Tensor::new(&[BASE_FREQ_R], dev)?;

        Ok(Self {
            micro_ca, macro_ca, gru_memory, memory_to_macro, spatial_panner, fm_mod_ratio, fm_mod_index,
            wavefolder_l, wavefolder_r, current_freq_l, current_freq_r, base_freq_l, base_freq_r,
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
        mimic_loss_val: f32,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, f32, f32, f32, f32, f32, f32)> {
        let dev = micro_tape.device();
        let mut next_macro_tape = macro_tape.clone();
        if force_macro_update {
            let m_field = macro_tape.abs().map_err(anyhow::Error::msg)?.affine(-0.01, METABOLIC_DECAY as f64).map_err(anyhow::Error::msg)?;
            next_macro_tape = self.macro_ca.forward(macro_tape, None, Some(&m_field)).map_err(anyhow::Error::msg)?;
        }
        
        let macro_activation = next_macro_tape.abs().map_err(anyhow::Error::msg)?.mean_all().map_err(anyhow::Error::msg)?;
        let metabolic_rate = (macro_activation * 5.0).map_err(anyhow::Error::msg)?.clamp(0.01f32, 1.0f32)?;

        let macro_mod = (self.memory_to_macro.forward(hidden_mem).map_err(anyhow::Error::msg)? + next_macro_tape.mean(D::Minus1).map_err(anyhow::Error::msg)?)?;
        let micro_m_field = micro_tape.abs().map_err(anyhow::Error::msg)?.affine(-0.01, METABOLIC_DECAY as f64).map_err(anyhow::Error::msg)?;
        let raw_next_micro = self.micro_ca.forward(micro_tape, Some(&macro_mod), Some(&micro_m_field)).map_err(anyhow::Error::msg)?;
        
        let one_t = Tensor::new(1.0f32, dev)?;
        let active_decay = one_t.sub(&metabolic_rate)?;
        let decayed_micro = micro_tape.broadcast_mul(&active_decay)?;
        let evolved_micro = raw_next_micro.broadcast_mul(&metabolic_rate)?;
        let next_micro_tape = decayed_micro.add(&evolved_micro)?.clamp(-1.0f32, 1.0f32)?;

        let micro_avg = next_micro_tape.mean(D::Minus1)?;
        let micro_flat = micro_avg.reshape((CA_CHANNELS,))?;
        
        let micro_reshaped = micro_flat.reshape((CA_CHANNELS / 2, 2)).map_err(anyhow::Error::msg)?;
        let component_sums = micro_reshaped.sum(0).map_err(anyhow::Error::msg)?.to_vec1::<f32>().map_err(anyhow::Error::msg)?;
        let sum_real = component_sums[0];
        let sum_imag = component_sums[1];
        let theta = sum_imag.atan2(sum_real + 1e-6);

        let tape_features = next_micro_tape.mean(D::Minus1).map_err(anyhow::Error::msg)?;
        let next_hidden_mem = self.gru_memory.forward(&tape_features, hidden_mem).map_err(anyhow::Error::msg)?;
        
        let movement = next_micro_tape.sub(micro_tape).map_err(anyhow::Error::msg)?.abs().map_err(anyhow::Error::msg)?.mean_all().map_err(anyhow::Error::msg)?;
        let movement_val = movement.to_scalar::<f32>().map_err(anyhow::Error::msg)?;
        
        let pop_l = next_micro_tape.narrow(1, 0, 1).map_err(anyhow::Error::msg)?.mean_all().map_err(anyhow::Error::msg)?;
        let pop_r = next_micro_tape.narrow(1, 1, 1).map_err(anyhow::Error::msg)?.mean_all().map_err(anyhow::Error::msg)?;
        let pop_l_val = pop_l.to_scalar::<f32>().map_err(anyhow::Error::msg)?;
        let pop_r_val = pop_r.to_scalar::<f32>().map_err(anyhow::Error::msg)?;
        
        let formant_freqs = [
            (300.0 + pop_l_val * 700.0, 300.0 + pop_r_val * 700.0),   
            (800.0 + movement_val * 1700.0, 800.0 + movement_val * 1700.0), 
            (2000.0 - pop_l_val * 500.0, 2000.0 - pop_r_val * 500.0), 
        ];

        let t_steps = (Tensor::arange(0.0f32, CHUNK_SIZE as f32, dev).map_err(anyhow::Error::msg)?.affine(1.0 / SAMPLE_RATE as f64, 0.0))?;
        
        let fm_ratios = (self.fm_mod_ratio.forward(&next_hidden_mem).map_err(anyhow::Error::msg)?.affine(4.0, 0.0))?;
        let fm_indices = (self.fm_mod_index.forward(&next_hidden_mem).map_err(anyhow::Error::msg)?.affine(5.0, 0.0))?;
        
        // FIX: Explicitly reshape the narrowed matrices [1, 1] down into 1D vectors [1] to allow clean matching
        let fm_ratio_l = fm_ratios.narrow(1, 0, 1)?.reshape((1,))?;
        let fm_ratio_r = fm_ratios.narrow(1, 1, 1)?.reshape((1,))?;
        let fm_idx_l = fm_indices.narrow(1, 0, 1)?.reshape((1,))?;
        let fm_idx_r = fm_indices.narrow(1, 1, 1)?.reshape((1,))?;

        let mut final_audio_l = Tensor::zeros((1, CHUNK_SIZE), DType::F32, dev).map_err(anyhow::Error::msg)?;
        let mut final_audio_r = Tensor::zeros((1, CHUNK_SIZE), DType::F32, dev).map_err(anyhow::Error::msg)?;

        let scaled_pop_l = pop_l.affine(200.0, 0.0)?;
        let scaled_pop_r = pop_r.affine(200.0, 0.0)?;
        let scaled_move = movement.affine(100.0, 0.0)?;
        
        let target_freq_l = self.base_freq_l.broadcast_add(&scaled_pop_l)?.broadcast_add(&scaled_move)?.clamp(20.0f32, 4000.0f32)?;
        let target_freq_r = self.base_freq_r.broadcast_add(&scaled_pop_r)?.broadcast_sub(&scaled_move)?.clamp(20.0f32, 4000.0f32)?;
        
        // FIX: Allocate glide buffers as explicit 1D arrays to prevent shape mismatches against frequencies
        let glide_speed = Tensor::new(&[FREQ_GLIDE_SPEED], dev)?;
        let one_minus_glide = Tensor::new(&[1.0 - FREQ_GLIDE_SPEED], dev)?;
        
        let left_glide = self.current_freq_l.broadcast_mul(&one_minus_glide)?;
        let left_target = target_freq_l.broadcast_mul(&glide_speed)?;
        self.current_freq_l = left_glide.add(&left_target)?.detach();

        let right_glide = self.current_freq_r.broadcast_mul(&one_minus_glide)?;
        let right_target = target_freq_r.broadcast_mul(&glide_speed)?;
        self.current_freq_r = right_glide.add(&right_target)?.detach();

        let mod_freq_l = self.current_freq_l.broadcast_mul(&fm_ratio_l)?;
        let mod_freq_r = self.current_freq_r.broadcast_mul(&fm_ratio_r)?;
        
        let mod_freq_l_val = mod_freq_l.reshape(())?.to_scalar::<f32>().map_err(anyhow::Error::msg)?;
        let mod_freq_r_val = mod_freq_r.reshape(())?.to_scalar::<f32>().map_err(anyhow::Error::msg)?;
        let fm_idx_l_val = fm_idx_l.reshape(())?.to_scalar::<f32>().map_err(anyhow::Error::msg)?;
        let fm_idx_r_val = fm_idx_r.reshape(())?.to_scalar::<f32>().map_err(anyhow::Error::msg)?;
        let current_freq_l_val = self.current_freq_l.reshape(())?.to_scalar::<f32>().map_err(anyhow::Error::msg)?;
        let current_freq_r_val = self.current_freq_r.reshape(())?.to_scalar::<f32>().map_err(anyhow::Error::msg)?;

        let phases_m_l = (t_steps.affine((2.0 * std::f32::consts::PI * mod_freq_l_val) as f64, phase_m_l as f64))?;
        let phases_m_r = (t_steps.affine((2.0 * std::f32::consts::PI * mod_freq_r_val) as f64, phase_m_r as f64))?;
        let modulator_l = phases_m_l.sin()?.affine(fm_idx_l_val as f64, 0.0)?;
        let modulator_r = phases_m_r.sin()?.affine(fm_idx_r_val as f64, 0.0)?;
        
        let phases_c_l = t_steps.affine((2.0 * std::f32::consts::PI * current_freq_l_val) as f64, (phase_c_l + theta) as f64)?;
        let phases_c_l = phases_c_l.add(&modulator_l)?;
        
        let phases_c_r = t_steps.affine((2.0 * std::f32::consts::PI * current_freq_r_val) as f64, (phase_c_r + theta) as f64)?;
        let phases_c_r = phases_c_r.add(&modulator_r)?;
        
        let sin_l = phases_c_l.sin()?.unsqueeze(0)?;
        let sin_r = phases_c_r.sin()?.unsqueeze(0)?;
        final_audio_l = final_audio_l.add(&sin_l)?;
        final_audio_r = final_audio_r.add(&sin_r)?;

        for (f_l, f_r) in formant_freqs.iter() {
            let p_l = t_steps.affine((2.0 * std::f32::consts::PI * f_l) as f64, (phase_c_l + theta) as f64)?;
            let p_r = t_steps.affine((2.0 * std::f32::consts::PI * f_r) as f64, (phase_c_r + theta) as f64)?;
            
            let voice_l = p_l.sin()?.unsqueeze(0)?.affine(0.3, 0.0)?;
            let voice_r = p_r.sin()?.unsqueeze(0)?.affine(0.3, 0.0)?;
            final_audio_l = final_audio_l.add(&voice_l)?;
            final_audio_r = final_audio_r.add(&voice_r)?;
        }

        let next_phase_c_l = phases_c_l.narrow(0, CHUNK_SIZE - 1, 1).map_err(anyhow::Error::msg)?.reshape(())?.to_scalar::<f32>().map_err(anyhow::Error::msg)? % (2.0 * std::f32::consts::PI);
        let next_phase_c_r = phases_c_r.narrow(0, CHUNK_SIZE - 1, 1).map_err(anyhow::Error::msg)?.reshape(())?.to_scalar::<f32>().map_err(anyhow::Error::msg)? % (2.0 * std::f32::consts::PI);
        let next_phase_m_l = phases_m_l.narrow(0, CHUNK_SIZE - 1, 1).map_err(anyhow::Error::msg)?.reshape(())?.to_scalar::<f32>().map_err(anyhow::Error::msg)? % (2.0 * std::f32::consts::PI);
        let next_phase_m_r = phases_m_r.narrow(0, CHUNK_SIZE - 1, 1).map_err(anyhow::Error::msg)?.reshape(())?.to_scalar::<f32>().map_err(anyhow::Error::msg)? % (2.0 * std::f32::consts::PI);

        let audio_l = self.wavefolder_l.forward(&final_audio_l, mimic_loss_val).map_err(anyhow::Error::msg)?;
        let audio_r = self.wavefolder_r.forward(&final_audio_r, mimic_loss_val).map_err(anyhow::Error::msg)?;
        
        let filter_openness = (next_hidden_mem.abs().map_err(anyhow::Error::msg)?.mean_all().map_err(anyhow::Error::msg)?.to_scalar::<f32>().map_err(anyhow::Error::msg)? * 5.0 + movement_val).clamp(0.4, 1.0);
        let audio_l = audio_l.affine(filter_openness as f64, 0.0).map_err(anyhow::Error::msg)?;
        let audio_r = audio_r.affine(filter_openness as f64, 0.0).map_err(anyhow::Error::msg)?;
        
        let pan_raw = self.spatial_panner.forward(&next_hidden_mem).map_err(anyhow::Error::msg)?.reshape(())?.to_scalar::<f32>().map_err(anyhow::Error::msg)?;
        let pan = pan_raw.clamp(-0.5, 0.5); 
        let gain_l = (0.5 * (1.0 - pan)).sqrt();
        let gain_r = (0.5 * (1.0 + pan)).sqrt();
        let audio_l = audio_l.affine((gain_l * 1.414) as f64, 0.0).map_err(anyhow::Error::msg)?;
        let audio_r = audio_r.affine((gain_r * 1.414) as f64, 0.0).map_err(anyhow::Error::msg)?;
        let stereo_chunk = Tensor::cat(&[&audio_l, &audio_r], 0).map_err(anyhow::Error::msg)?.reshape((2, CHUNK_SIZE)).map_err(anyhow::Error::msg)?;
        Ok((stereo_chunk, next_micro_tape, next_macro_tape, next_hidden_mem, next_phase_c_l, next_phase_c_r, next_phase_m_l, next_phase_m_r, movement_val, theta))
    }
}

// ==========================================
// 8. DEFIBRILLATOR CONTROLLER
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
// 9. AUDIO ARBITER
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

// ==========================================
// MAIN INDUSTRIAL RUNTIME LOGIC
// ==========================================
fn main() -> Result<()> {
    rayon::ThreadPoolBuilder::new().num_threads(6).build_global()?;
    
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    if device.is_cuda() {
        println!("--> Success: Running on GPU Workstation (CUDA Accelerated Platform)");
    } else {
        println!("--> Fallback: GPU not detected, leveraging local host CPU layer");
    }
    println!("=== TITAN AUDIO ECOSYSTEM: RUST EDITION (RESONANT BETA) ===");

    let target_loader = TargetAudioLoader::new("/home/anon/Downloads/")?;
    let mut varmap = VarMap::new();
    let model_path = "/home/anon/Downloads/titan_model.safetensors";
    if std::path::Path::new(model_path).exists() {
        println!("--> Loading existing matrix weights from {}", model_path);
        varmap.load(model_path).map_err(anyhow::Error::msg)?;
    }
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
    
    let mut burst_ticks = 0;
    let mut burst_energy = 0.0f32;
    let mut total_complexity = 0.0f32;

    for step in 0..total_chunks {
        let aperture = uncertainty.branch_aperture();
        let force_macro = rand::thread_rng().gen_range(0.0..1.0) < (0.2 + aperture * 0.6);
        
        let prev_mimic_loss = if step == 0 { 0.5f32 } else { uncertainty.mimic / 10.0 };
        
        let (stereo_chunk, next_micro_tape, next_macro_tape, next_hidden_mem, nc_l, nc_r, nm_l, nm_r, movement, theta) = model.forward(&micro_tape, &macro_tape, &hidden_mem, phase_c_l, phase_c_r, phase_m_l, phase_m_r, force_macro, prev_mimic_loss)?;

        total_complexity += movement;
        let age_factor = (total_complexity / 500.0).min(0.6);
        let audio_for_loss = stereo_chunk.tanh().map_err(anyhow::Error::msg)?.affine((1.0 - age_factor) as f64, 0.0)?;
        
        let target_chunk = target_loader.sample_chunk(&device)?;
        let mimic_loss = audio_for_loss.mean(0).map_err(anyhow::Error::msg)?.sub(&target_chunk.reshape((CHUNK_SIZE,)).map_err(anyhow::Error::msg)?)?.sqr().map_err(anyhow::Error::msg)?.mean_all().map_err(anyhow::Error::msg)?;
        let var_loss = var_all(&audio_for_loss).map_err(anyhow::Error::msg)?.neg().map_err(anyhow::Error::msg)?;
        let movement_loss = Tensor::new(movement, &device).map_err(anyhow::Error::msg)?.neg().map_err(anyhow::Error::msg)?;
        let diff = audio_for_loss.narrow(1, 1, CHUNK_SIZE - 1).map_err(anyhow::Error::msg)?.sub(&audio_for_loss.narrow(1, 0, CHUNK_SIZE - 1).map_err(anyhow::Error::msg)?)?;
        let roughness_loss = diff.sqr().map_err(anyhow::Error::msg)?.mean_all().map_err(anyhow::Error::msg)?;
        let reg_loss = stereo_chunk.sqr().map_err(anyhow::Error::msg)?.mean_all().map_err(anyhow::Error::msg)?;
        
        let rms = audio_for_loss.sqr().map_err(anyhow::Error::msg)?.mean_all().map_err(anyhow::Error::msg)?.sqrt().map_err(anyhow::Error::msg)?;
        let energy_loss = rms.sub(&Tensor::new(0.25f32, &device).map_err(anyhow::Error::msg)?)?.sqr().map_err(anyhow::Error::msg)?;

        let mimic_loss_scalar = mimic_loss.to_scalar::<f32>().unwrap_or(0.0);
        let arb_features = Tensor::new(&[0.5, mimic_loss_scalar, movement / 0.3, 0.0, 0.0, 0.0, uncertainty.spectral, uncertainty.movement, uncertainty.mimic, uncertainty.compositional, aperture, step as f32 / total_chunks as f32], &device).map_err(anyhow::Error::msg)?.unsqueeze(0).map_err(anyhow::Error::msg)?;
        let loss_weights = arbiter.forward(&arb_features)?;
        let lw = loss_weights.reshape((6,))?.to_vec1::<f32>().map_err(anyhow::Error::msg)?;

        let l1 = mimic_loss.affine((lw[1] * (1.0 - RESONANT_AUTONOMY)) as f64, 0.0)?;
        let l2 = var_loss.affine((lw[0] * (1.0 + RESONANT_AUTONOMY)) as f64, 0.0)?;
        let l3 = movement_loss.affine((lw[2] * RESONANT_AUTONOMY) as f64, 0.0)?;
        let l4 = roughness_loss.affine(lw[3] as f64, 0.0)?;
        let l5 = reg_loss.affine(0.01, 0.0)?;
        let l6 = energy_loss.affine(2.0, 0.0)?;
        let total_loss = l1.add(&l2)?.add(&l3)?.add(&l4)?.add(&l5)?.add(&l6)?;

        optimizer.backward_step(&total_loss).map_err(anyhow::Error::msg)?;

        micro_tape = next_micro_tape.detach();
        macro_tape = next_macro_tape.detach();
        hidden_mem = next_hidden_mem.detach();
        phase_c_l = nc_l; phase_c_r = nc_r; phase_m_l = nm_l; phase_m_r = nm_r;

        let m_sig = movement_mon.analyze(movement)?;
        let s_sig = spectral_mon.analyze(&audio_for_loss)?;
        uncertainty.update(&s_sig, &m_sig, Some(&serde_json::json!({"drift": mimic_loss_scalar, "theta": theta})));

        let phi_current = uncertainty.phi;

        let defib_features = Tensor::new(&[movement, 0.1, mimic_loss_scalar, 0.5, aperture, phi_current / 10.0, step as f32 / total_chunks as f32], &device).map_err(anyhow::Error::msg)?.unsqueeze(0).map_err(anyhow::Error::msg)?;
        let (thresh, n_scale, lr_mult) = defib_ctrl.forward(&defib_features)?;
        
        if (movement < thresh || mimic_loss_scalar > 0.8) && phi_current < 4.0 && burst_ticks == 0 {
            burst_ticks = 8;
            burst_energy = n_scale;
        }

        if burst_ticks > 0 {
            let env = (burst_ticks as f32 / 8.0).sqrt(); 
            let noise = Tensor::randn_like(&micro_tape, 0.0, 1.0)?.affine((burst_energy * env) as f64, 0.0)?;
            micro_tape = micro_tape.add(&noise)?.clamp(-1.0f32, 1.0f32)?;
            
            let phi_gate = 1.0 / (1.0 + phi_current);
            optimizer.set_learning_rate(BASE_LR * (1.0 + (lr_mult - 1.0) * env) as f64 * phi_gate as f64);
            if step % 20 == 0 { println!("[PACEMAKER BURST] step {} (env: {:.2}, phi_gate: {:.2})", step, env, phi_gate); }
            burst_ticks -= 1;
        } else { 
            let phi_gate = 1.0 / (1.0 + phi_current);
            optimizer.set_learning_rate(BASE_LR * phi_gate as f64); 
        }

        let age_factor = (total_complexity / 500.0).min(0.6);
        let audio_t = stereo_chunk.tanh().map_err(anyhow::Error::msg)?.affine((1.0 - age_factor) as f64, 0.0)?;
        let abs_max = audio_t.abs().map_err(anyhow::Error::msg)?.flatten_all().map_err(anyhow::Error::msg)?.max(0).map_err(anyhow::Error::msg)?.to_scalar::<f32>().map_err(anyhow::Error::msg)?;
        let boost = if abs_max < 0.25 { 0.25 / (abs_max + 1e-6) } else { 1.0 };
        let audio_data = audio_t.affine(boost as f64 * 32767.0, 0.0).map_err(anyhow::Error::msg)?;
        
        let audio_l = audio_data.narrow(0, 0, 1).map_err(anyhow::Error::msg)?.to_vec2::<f32>().map_err(anyhow::Error::msg)?[0].clone();
        let audio_r = audio_data.narrow(0, 1, 1).map_err(anyhow::Error::msg)?.to_vec2::<f32>().map_err(anyhow::Error::msg)?[0].clone();
        for i in 0..CHUNK_SIZE { audio_frames.push(audio_l[i] as i16); audio_frames.push(audio_r[i] as i16); }

        if step % 10 == 0 {
            let topology_state = macro_tape.mean(1).map_err(anyhow::Error::msg)?.reshape((TAPE_LEN,)).map_err(anyhow::Error::msg)?.to_vec1::<f32>().map_err(anyhow::Error::msg)?;
            topology_history.push(topology_state);
            uncertainty_trace.push(serde_json::json!({"step": step, "spectral": uncertainty.spectral, "movement": uncertainty.movement, "compositional": uncertainty.compositional, "aperture": aperture, "phi": phi_current}));
        }
        if step % 50 == 0 { 
            println!("Chunk {}/{} | Move: {:.3} | Mimic: {:.3} | Phi: {:.2} | Aperture: {:.2}", step, total_chunks, movement, mimic_loss_scalar, phi_current, aperture);
            if let Ok(v) = micro_tape.narrow(1, 0, 1).unwrap_or(micro_tape.clone()).narrow(2, 0, 64).unwrap_or(micro_tape.clone()).reshape((64,)) {
                if let Ok(vals) = v.to_vec1::<f32>() {
                    let mut grads = vec![0.0; 64];
                    for i in 0..63 { grads[i] = vals[i+1] - vals[i]; }
                    println!("Tape: [{}]", TapeCodec::encode(&vals, &grads));
                }
            }
        }
    }

    // Generate Priming Prompt for Suno/Udio
    let avg_phi = uncertainty_trace.iter().map(|t| t["phi"].as_f64().unwrap_or(0.0)).sum::<f64>() / uncertainty_trace.len() as f64;
    let avg_aperture = uncertainty_trace.iter().map(|t| t["aperture"].as_f64().unwrap_or(0.0)).sum::<f64>() / uncertainty_trace.len() as f64;
    
    let prompt = format!(
        "Style: {}, {}, {}, {}. Texture: {}. [Informational Phi: {:.2}, Aperture: {:.2}]",
        if avg_phi > 5.0 { "Hyper-Resonant" } else { "Chaotic" },
        if avg_aperture > 0.5 { "Evolving" } else { "Stable" },
        if total_complexity > 500.0 { "Dense" } else { "Minimal" },
        "Information-Theoretic Glitch",
        if avg_phi > 7.0 { "Crystalline" } else if avg_phi > 4.0 { "Organic" } else { "Grit" },
        avg_phi, avg_aperture
    );
    println!("\n=== SUNO/UDIO PRIMING PROMPT ===");
    println!("{}", prompt);
    std::fs::write("/home/anon/Downloads/suno_priming_prompt.txt", prompt)?;

    // Save CSVs
    let mut topo_writer = csv::Writer::from_path("/home/anon/Downloads/ca_topology_rust.csv")?;
    for row in topology_history { topo_writer.write_record(row.iter().map(|f| f.to_string()))?; }
    topo_writer.flush()?;
    let mut unc_writer = csv::Writer::from_path("/home/anon/Downloads/uncertainty_trace_rust.csv")?;
    unc_writer.write_record(&["step", "spectral", "movement", "compositional", "aperture"])?;
    for trace in uncertainty_trace { unc_writer.write_record(&[trace["step"].to_string(), trace["spectral"].to_string(), trace["movement"].to_string(), trace["compositional"].to_string(), trace["aperture"].to_string()])?; }
    unc_writer.flush()?;
    println!("Topology and uncertainty trace saved to /home/anon/Downloads/.");

    let spec = hound::WavSpec { channels: 2, sample_rate: SAMPLE_RATE, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut writer = hound::WavWriter::create("/home/anon/Downloads/rust_ecosystem_out.wav", spec)?;
    for sample in audio_frames { writer.write_sample(sample)?; }
    writer.finalize()?;
    println!("Audio saved to /home/anon/Downloads/rust_ecosystem_out.wav");

    // Save weights safely
    varmap.save(model_path).map_err(anyhow::Error::msg)?;
    let metadata = std::fs::metadata(model_path)?;
    println!("Model saved to {}. Size: {:.2} MB", model_path, metadata.len() as f32 / 1_048_576.0);

    Ok(())
}
