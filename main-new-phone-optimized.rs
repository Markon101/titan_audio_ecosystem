// =====================================================================
// TITAN AUDIO ECOSYSTEM — RUST EDITION (GRADIENT-COHERENT PHONE RELEASE)
// =====================================================================

use anyhow::Result;
use candle_core::{Device, Tensor, DType, D, Result as CResult};
use candle_nn::{Linear, Conv1dConfig, VarBuilder as VBV, VarMap, Optimizer, AdamW, Module};
use std::collections::VecDeque;
use rustfft::{FftPlanner, num_complex::Complex};
use rand::Rng;

// --- ECOSYSTEM CONSTANTS ---
const SAMPLE_RATE: u32 = 48000;
const DURATION_SECONDS: f32 = 240.0;
const CHUNK_SIZE: usize = 4096;
const TAPE_LEN: usize = 512;          
const CA_CHANNELS: usize = 96;
const CA_HIDDEN_MULT: usize = 32;      
const KAN_BASIS_FUNCTIONS: usize = 128;
const MEMORY_DIM: usize = 512;
const BPTT_WINDOW: usize = 4;         
const SPEC_BINS: usize = 96;          

const CHOPTUIK_EXPONENT: f32 = 0.37413;
const CRITICAL_D0: f32 = 0.02;        
const CRITICALITY_SEEKING: bool = true; 
const LARGE_D_DIM: usize = 512;
const FDN_DELAY_LINES: usize = 4;
const FDN_DELAYS: [usize; 4] = [149, 263, 431, 701];
const QNM_Q_BASE: f32 = 40.0;         

const BASE_FREQ_L: f32 = 48.0;
const BASE_FREQ_R: f32 = 69.0;
const METABOLIC_DECAY: f32 = 0.999999;
const FREQ_GLIDE_SPEED: f32 = 0.0711;
const BASE_LR: f64 = 1.3e-3;
const RESONANT_AUTONOMY: f32 = 0.37;
const TWO_PI: f32 = 2.0 * std::f32::consts::PI;

// --- DISRUPTION-AVOIDANCE (tokamak q-factor) CONTROLLER ---
// Predictive avoidance replaces reactive defibrillation: we steer the macro
// field's operating point away from the cold-quench boundary BEFORE it crosses,
// using a Kruskal-Shafranov-style safety factor q_titan and a time-to-quench
// estimate from a dual-EMA precursor signal.
const DISRUPT_EMA_FAST: f32 = 0.30;       // fast EMA weight on macro variance
const DISRUPT_EMA_SLOW: f32 = 0.05;       // slow EMA weight on macro variance
const DISRUPT_WARMUP: usize = 32;         // steps to self-calibrate q baseline
const DISRUPT_Q_FLOOR_REL: f32 = 0.40;    // quench boundary = 40% of healthy q baseline
const DISRUPT_RATE_GAIN: f32 = 20.0;      // sensitivity of time-to-quench to approach velocity
const DISRUPT_HORIZON: f32 = 8.0;         // steps of lookahead for the avoidance ramp
const SHEAR_AMP_MIN: f32 = 0.030;         // always-on structured dither (anti-weld floor; raised for stronger baseline momentum)
const SHEAR_AMP_MAX: f32 = 0.45;          // max predictive macro shear when quench is imminent
const SHEAR_OCTAVES: usize = 4;           // fBm octaves for structured (non-white) shear
const SHEAR_PHASE_VEL: f32 = 0.30;        // traveling-wave phase advance per step

// --- COUPLING BAND + LOCK-BREAK (H-mode push) ---
// The system's loss previously rewarded coupling -> 1 (zero magnetic shear, the L-mode
// lock). We instead target a coupling BAND (correlated, not identical) and give the
// controller an ABSOLUTE lock detector, because the self-calibrated q baseline can be
// poisoned when the lock forms inside the warmup window.
const SYNERGY_TARGET: f32 = 0.50;         // center of the healthy coupling band
const SYNERGY_BAND_W: f32 = 0.50;         // fixed weight of the band penalty in total loss
const COUPLE_CEILING: f32 = 0.80;         // absolute coupling above which decorrelating shear engages
const LOCK_SHEAR_SCALE: f32 = 0.75;       // a lock is firm-but-not-quench; cap its shear below full

// --- METABOLIC HOMEOSTAT ---
// energy_state was pinning at the 1.0 ceiling (recharge >> cost), removing all scarcity
// pressure. A gentle setpoint pull keeps it in a band with headroom both ways.
const ENERGY_SETPOINT: f32 = 0.65;        // metabolic operating point (was effectively 1.0)
const ENERGY_HOMEO_RATE: f32 = 0.05;      // pull strength toward setpoint (energy settles ~0.72)

// --- MACRO SLOW-FIELD AMPLITUDE HOMEOSTAT ---
// Telemetry showed the macro field collapsing to a uniform near-zero state: the blind per-step
// 0.95 contraction, balanced against the very weak (0.1x) restoring field, had an equilibrium
// abs-mean of only ~0.04. A dead macro field also freezes the micro tape, because the micro
// metabolic gate metab = clamp(5*|macro|, .01, 1) collapses to ~0.2 (80% inertia) when |macro|
// is tiny. This homeostat targets a healthy abs-mean with enough gain to dominate the decay,
// replacing the fixed contraction and serving as anti-rail AND anti-collapse in one controller.
const MACRO_AMP_SETPOINT: f32 = 0.40;     // target abs-mean for the slow macro field (metab saturates >0.2)
const MACRO_AMP_RATE: f32 = 0.35;         // per-step multiplicative pull toward the setpoint

const MORPH_MAX_BLOCKS: usize = 12;          
const MORPH_START_DEPTH: usize = 1;
const MORPH_PATIENCE_BASE: usize = 10;       
const MORPH_WARMUP: usize = 48;              
const MORPH_GROWTH_REL: f32 = 1.10;          
const MORPH_PRUNE_REL: f32 = 0.55;           

const RAD_AMP_INIT: f32 = 0.8;
const RAD_AMP_MIN: f32 = 0.10;
const RAD_AMP_MAX: f32 = 0.98;
const RAD_COOL: f32 = 0.7;   
const RAD_HEAT: f32 = 1.3;   
const RADIATE_PROB: f32 = 0.12;     
const RADIATE_SPARSITY: f32 = 0.95; 
const CAUCHY_CLAMP: f32 = 8.0;      
const CHAOS_LAMBDA: f32 = 3.99;     

const DEFIB_BURST_TICKS: usize = 8;        
const DEFIB_REFRACTORY_BASE: usize = 16;   
const DEFIB_REFRACTORY_MAX: usize = 96;    
const DEFIB_FIRE_DEBOUNCE: usize = 3;      
const DEFIB_RATE_DECAY: f32 = 0.97;        
const DEFIB_RATE_SENSITIVITY: f32 = 6.0;   

const VAL_SYMS: [&str; 8] = [" ", "·", "░", "▒", "▓", "█", "▪", "■"];
const GRAD_SYMS: [&str; 8] = [" ", "˙", "·", "∘", "o", "O", "◎", "●"];
const ARCHETYPES: [&str; 8] = ["VOID", "LATENT", "DRIFT", "NEXUS", "PULSE", "SIGNAL", "AXIOM", "SINGULARITY"];
const ARCH_BOUNDS: [f32; 9] = [0.0, 0.13, 0.26, 0.40, 0.52, 0.65, 0.78, 0.90, 1.001];
const PHASE_MAP: [(f32, &str); 7] = [
    (0.20, "MASTERY"), (0.27, "COHERENT"), (0.35, "CONVERGING"),
    (0.45, "LEARNING"), (0.55, "TURBULENT"), (0.70, "CHAOTIC"), (f32::MAX, "PRIMORDIAL")
];

// --- AUDIO TARGET LOADER ---
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
            let is_out = p.file_name().and_then(|n| n.to_str()).map_or(false, |n| n == "rust_ecosystem_out.wav");
            if p.extension().map_or(false, |ext| ext == "wav") && !is_out {
                match Self::load_wav(&p) {
                    Ok((l, r)) if l.len() >= CHUNK_SIZE => {
                        println!("--> Loaded target audio: {:?} ({} samples/ch @ 48k stereo)", p, l.len());
                        buffers.push((l, r));
                    }
                    Ok((l, _)) => println!("--> Skipping {:?}: only {} samples after resample", p, l.len()),
                    Err(e) => println!("--> Skipping {:?}: {}", p, e),
                }
            }
        }
        if buffers.is_empty() { anyhow::bail!("No usable training audio found in {}", path); }
        Ok(Self { buffers })
    }

    fn load_wav(p: &std::path::Path) -> Result<(Vec<f32>, Vec<f32>)> {
        let mut reader = hound::WavReader::open(p)?;
        let spec = reader.spec();
        let raw: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
            (hound::SampleFormat::Float, 32) => reader.samples::<f32>().filter_map(|s| s.ok()).collect(),
            (hound::SampleFormat::Int, 16) => reader.samples::<i16>().filter_map(|s| s.ok()).map(|s| s as f32 / 32768.0).collect(),
            (hound::SampleFormat::Int, bits @ (24 | 32)) => {
                let scale = (1i64 << (bits - 1)) as f32;
                reader.samples::<i32>().filter_map(|s| s.ok()).map(|s| s as f32 / scale).collect()
            }
            (fmt, bits) => anyhow::bail!("unsupported WAV format {:?}/{} bits", fmt, bits),
        };
        if raw.is_empty() { anyhow::bail!("no samples decoded"); }
        let ch = spec.channels as usize;
        let (left, right): (Vec<f32>, Vec<f32>) = if ch >= 2 {
            let l = raw.iter().step_by(ch).copied().collect();
            let r = raw.iter().skip(1).step_by(ch).copied().collect();
            (l, r)
        } else {
            (raw.clone(), raw)
        };
        if spec.sample_rate == SAMPLE_RATE { return Ok((left, right)); }
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

    fn sample_chunk(&self, device: &Device) -> CResult<Tensor> {
        let mut rng = rand::thread_rng();
        let idx = rng.gen_range(0..self.buffers.len());
        let (l, r) = &self.buffers[idx];
        let start = if l.len() == CHUNK_SIZE { 0 } else { rng.gen_range(0..(l.len() - CHUNK_SIZE + 1)) };
        let mut data = Vec::with_capacity(2 * CHUNK_SIZE);
        data.extend_from_slice(&l[start..start + CHUNK_SIZE]);
        data.extend_from_slice(&r[start..start + CHUNK_SIZE]);
        Tensor::from_vec(data, (2, CHUNK_SIZE), device)
    }
}

// --- CUSTOM MODULES & MATH ---
struct Tanh;
impl Module for Tanh { fn forward(&self, xs: &Tensor) -> CResult<Tensor> { xs.tanh() } }

struct Sigmoid;
impl Module for Sigmoid { fn forward(&self, xs: &Tensor) -> CResult<Tensor> { candle_nn::ops::sigmoid(xs) } }

struct Relu;
impl Module for Relu { fn forward(&self, xs: &Tensor) -> CResult<Tensor> { xs.relu() } }

fn var_all(x: &Tensor) -> CResult<Tensor> {
    let mean = x.mean_all()?;
    x.broadcast_sub(&mean)?.sqr()?.mean_all()
}

fn load_into_varmap(varmap: &VarMap, path: &str, device: &Device) -> Result<(usize, usize, usize)> {
    let loaded = candle_core::safetensors::load(path, device).map_err(anyhow::Error::msg)?;
    let data = varmap.data().lock().unwrap();
    let (mut hit, mut miss, mut mismatch) = (0, 0, 0);
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

fn decimate2(x: &Tensor) -> CResult<Tensor> {
    let (b, c, l) = x.dims3()?;
    x.reshape((b, c, l / 2, 2))?.mean(D::Minus1)
}

fn calculate_cross_layer_synergy_tensor(micro: &Tensor, macro_t: &Tensor) -> CResult<Tensor> {
    let micro_flat = micro.flatten_all()?;
    let macro_flat = macro_t.flatten_all()?;
    let micro_mean = micro_flat.mean_all()?;
    let macro_mean = macro_flat.mean_all()?;
    let micro_norm = micro_flat.broadcast_sub(&micro_mean)?;
    let macro_norm = macro_flat.broadcast_sub(&macro_mean)?;
    
    // Pearson correlation has a mathematically explosive gradient (1/v^2) if variance nears zero.
    // By using direct covariance mapped through tanh, we measure cross-layer synergy
    // with an absolutely safe, bounded gradient profile.
    let cross_cov = micro_norm.mul(&macro_norm)?.mean_all()?;
    cross_cov.affine(10.0, 0.0)?.tanh()
}

fn conv1d_circular(in_c: usize, out_c: usize, k: usize, vb: VBV) -> Result<Box<dyn Fn(&Tensor) -> CResult<Tensor>>> {
    let padding = k / 2;
    let config = Conv1dConfig { padding: 0, stride: 1, dilation: 1, groups: 1 };
    let conv = candle_nn::conv1d(in_c, out_c, k, config, vb)?;
    Ok(Box::new(move |x: &Tensor| {
        let l = x.dim(D::Minus1)?;
        let left = x.narrow(D::Minus1, l - padding, padding)?;
        let right = x.narrow(D::Minus1, 0, padding)?;
        let padded = Tensor::cat(&[&left, x, &right], D::Minus1)?;
        conv.forward(&padded)
    }))
}

// --- INNOVATIVE MATH & SIGNAL HELPERS ---

fn morph_wave(phase: &Tensor, morph: &Tensor) -> CResult<Tensor> {
    let s = phase.sin()?;
    let t = phase.affine(3.0, 0.0)?.sin()?.affine(-0.11, 0.0)?;
    let tri_approx = s.add(&t)?;
    let one = Tensor::new(1.0f32, phase.device())?;
    let one_minus_morph = one.sub(morph)?;
    let s_part = s.broadcast_mul(&one_minus_morph)?;
    let t_part = tri_approx.broadcast_mul(morph)?;
    s_part.add(&t_part)
}

fn apply_haas_delay(x: &Tensor, delay_samples: usize) -> CResult<Tensor> {
    let len = x.dim(candle_core::D::Minus1)?;
    if delay_samples == 0 {
        return Ok(x.clone());
    }
    let dev = x.device();
    let zero = Tensor::zeros((1, delay_samples), DType::F32, dev)?;
    let cut = x.narrow(candle_core::D::Minus1, 0, len - delay_samples)?;
    Tensor::cat(&[&zero, &cut], candle_core::D::Minus1)
}

fn clamp_weights(varmap: &VarMap, bound: f32) -> Result<()> {
    let vars = varmap.all_vars();
    for var in vars.iter() {
        let t = var.as_tensor();
        let clamped = t.clamp(-bound, bound)?;
        var.set(&clamped).map_err(anyhow::Error::msg)?;
    }
    Ok(())
}

// --- SPECTRAL PROJECTOR ---
struct SpectralProjector {
    window: Tensor,
    cos_m: Tensor,
    sin_m: Tensor,
}
impl SpectralProjector {
    fn new(device: &Device) -> CResult<Self> {
        let n = CHUNK_SIZE;
        let mut win = Vec::with_capacity(n);
        for i in 0..n { win.push(0.5 - 0.5 * (TWO_PI * i as f32 / (n as f32 - 1.0)).cos()); }
        let f_lo = 40.0f32;
        let f_hi = 8000.0f32;
        let mut cos_v = vec![0.0f32; n * SPEC_BINS];
        let mut sin_v = vec![0.0f32; n * SPEC_BINS];
        for k in 0..SPEC_BINS {
            let frac = k as f32 / (SPEC_BINS as f32 - 1.0);
            let omega = TWO_PI * f_lo * (f_hi / f_lo).powf(frac) / SAMPLE_RATE as f32;
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
    
    fn log_mag(&self, x: &Tensor) -> CResult<Tensor> {
        let xw = x.broadcast_mul(&self.window.unsqueeze(0)?)?;
        let re = xw.matmul(&self.cos_m)?;
        let im = xw.matmul(&self.sin_m)?;
        re.sqr()?.add(&im.sqr()?)?.affine(1.0, 1e-3)?.log()?.affine(0.5, 0.0)
    }
}

// --- CONTRACTION & FILTER STACKS ---
struct AsymptoticContractionLayer { expand: Linear, contract: Linear }
impl AsymptoticContractionLayer {
    fn new(in_d: usize, hyper_d: usize, out_d: usize, vb: VBV) -> Result<Self> {
        Ok(Self { expand: candle_nn::linear(in_d, hyper_d, vb.pp("expand"))?, contract: candle_nn::linear(hyper_d, out_d, vb.pp("contract"))? })
    }
    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        self.contract.forward(&self.expand.forward(x)?.tanh()?)?.affine(1.0 / (LARGE_D_DIM as f32).sqrt() as f64, 0.0)
    }
}

struct QNMFilterBank { states_l: [[f32; 2]; 3], states_r: [[f32; 2]; 3] }
impl QNMFilterBank {
    fn new() -> Self { Self { states_l: [[0.0; 2]; 3], states_r: [[0.0; 2]; 3] } }
    fn process(&mut self, samples_l: &mut [f32], samples_r: &mut [f32], phi: f32) {
        let q_eff = (QNM_Q_BASE / (1.0 + 0.5 * phi)).max(4.0);
        let qnm_freqs = [220.0f32, 550.0, 1200.0];
        for (idx, &freq) in qnm_freqs.iter().enumerate() {
            let damping = std::f32::consts::PI * freq / (q_eff * SAMPLE_RATE as f32);
            let omega = TWO_PI * freq / SAMPLE_RATE as f32;
            let r = (-damping).exp();
            let c1 = 2.0 * r * omega.cos();
            let c2 = -r * r;
            let scale = (1.0 - r) * 0.5;

            let s_l = &mut self.states_l[idx];
            for x in samples_l.iter_mut() {
                let next_v = (*x * scale) + c1 * s_l[0] + c2 * s_l[1];
                s_l[1] = s_l[0]; s_l[0] = next_v;
                *x = (*x + next_v * 0.12).clamp(-1.0, 1.0);
            }
            let s_r = &mut self.states_r[idx];
            for x in samples_r.iter_mut() {
                let next_v = (*x * scale) + c1 * s_r[0] + c2 * s_r[1];
                s_r[1] = s_r[0]; s_r[0] = next_v;
                *x = (*x + next_v * 0.12).clamp(-1.0, 1.0);
            }
        }
    }
}

struct FractalFDN { buffers: Vec<Vec<f32>>, indices: Vec<usize>, lp_states: [f32; FDN_DELAY_LINES] }
impl FractalFDN {
    fn new() -> Self {
        let mut buffers = Vec::new(); let mut indices = Vec::new();
        for &d in &FDN_DELAYS { buffers.push(vec![0.0; d]); indices.push(0); }
        Self { buffers, indices, lp_states: [0.0; FDN_DELAY_LINES] }
    }
    fn process(&mut self, samples: &mut [f32], echo: f32) {
        let mix = [[0.5, 0.5, 0.5, 0.5], [0.5, -0.5, 0.5, -0.5], [0.5, 0.5, -0.5, -0.5], [0.5, -0.5, -0.5, 0.5]];
        let lp_a = 0.35; let lp_b = 1.0 - lp_a; let scale = 0.42 * echo;
        for x in samples.iter_mut() {
            let mut outs = [0.0; FDN_DELAY_LINES];
            for i in 0..FDN_DELAY_LINES {
                let idx = self.indices[i];
                self.lp_states[i] = self.lp_states[i] * lp_a + self.buffers[i][idx] * lp_b;
                outs[i] = self.lp_states[i];
            }
            for i in 0..FDN_DELAY_LINES {
                let mut sum = 0.0;
                for j in 0..FDN_DELAY_LINES { sum += mix[i][j] * outs[j]; }
                let idx = self.indices[i];
                self.buffers[i][idx] = *x + sum * scale;
                let next_idx = idx + 1;
                self.indices[i] = if next_idx >= self.buffers[i].len() { 0 } else { next_idx };
            }
            let fdn_out = (outs[0] + outs[1] + outs[2] + outs[3]) * 0.25;
            *x = *x * (1.0 - echo * 0.2) + fdn_out * (echo * 0.4);
        }
    }
}

// --- MONITORS ---
struct SpectralEntropyMonitor { history: VecDeque<f32>, window: usize, fft: std::sync::Arc<dyn rustfft::Fft<f32>> }
impl SpectralEntropyMonitor {
    fn new(w: usize) -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(CHUNK_SIZE);
        Self { history: VecDeque::with_capacity(w), window: w, fft }
    }
    fn analyze(&mut self, stereo: &Tensor) -> Result<serde_json::Value> {
        let mono = stereo.mean(0)?.reshape((CHUNK_SIZE,))?.to_vec1::<f32>()?;
        let n = mono.len();
        let mut buf: Vec<Complex<f32>> = mono.iter().map(|&x| Complex::new(x, 0.0)).collect();
        self.fft.process(&mut buf);
        let mags: Vec<f32> = buf.iter().take(n / 2).map(|c| c.norm()).collect();
        let sum: f32 = mags.iter().sum::<f32>() + 1e-8;
        let mut entropy = 0.0;
        for m in mags {
            let p = m / sum; if p > 1e-7 { entropy -= p * p.ln(); }
        }
        self.history.push_back(entropy);
        if self.history.len() > self.window { self.history.pop_front(); }
        let avg = self.history.iter().sum::<f32>() / self.history.len() as f32;
        Ok(serde_json::json!({"signal": entropy, "avg": avg, "trigger": entropy < 3.0, "type": "spectral_entropy"}))
    }
}

struct MovementCoherenceMonitor { history: VecDeque<f32>, window: usize }
impl MovementCoherenceMonitor {
    fn new(w: usize) -> Self { Self { history: VecDeque::with_capacity(w), window: w } }
    fn analyze(&mut self, m: f32) -> Result<serde_json::Value> {
        self.history.push_back(m);
        if self.history.len() > self.window { self.history.pop_front(); }
        let mut trigger = false; let mut trend = 0.0;
        if self.history.len() >= 10 {
            let n = self.history.len() as f32;
            let x_mean = (n - 1.0) / 2.0;
            let y_mean: f32 = self.history.iter().sum::<f32>() / n;
            let (mut num, mut den) = (0.0, 0.0);
            for (i, &y) in self.history.iter().enumerate() {
                let dx = i as f32 - x_mean;
                num += dx * (y - y_mean); den += dx * dx;
            }
            trend = num / (den + 1e-8); trigger = trend < -0.001;
        }
        Ok(serde_json::json!({"signal": m, "trend": trend, "trigger": trigger, "type": "movement_coherence"}))
    }
}

struct AudioUncertaintyState { spectral: f32, movement: f32, mimic: f32, compositional: f32, phi: f32, synergy: f32, empowerment: f32 }
impl AudioUncertaintyState {
    fn new() -> Self { Self { spectral: 0.0, movement: 0.0, mimic: 0.0, compositional: 0.0, phi: 0.0, synergy: 0.0, empowerment: 0.0 } }
    fn update(&mut self, spec_sig: &serde_json::Value, move_sig: &serde_json::Value, mimic_sig: Option<&serde_json::Value>, syn: f32, emp: f32) {
        let s_sig = spec_sig["signal"].as_f64().unwrap_or(0.0) as f32;
        let avg_s = spec_sig["avg"].as_f64().unwrap_or(1.0) as f32;
        let m_trend = move_sig["trend"].as_f64().unwrap_or(0.0) as f32;
        // Normalize raw spectral entropy (nats) by the max possible entropy ln(N/2),
        // landing phi in ~[0,1.5] so it tracks the field-entropy swings instead of
        // pinning to a constant. The SAME phi feeds phi_gate = 1/(1+phi) downstream.
        let phi_norm = (CHUNK_SIZE as f32 / 2.0).ln();
        let s_n = s_sig / phi_norm;
        let avg_n = avg_s / phi_norm;
        self.phi = (s_n * (avg_n / (s_n + 1e-6)).clamp(0.1, 5.0)).clamp(0.0, 1.5);
        self.spectral = (1.0 - (s_sig / 8.0)).max(0.0);
        self.movement = (-m_trend * 200.0).max(0.0);
        if let Some(ms) = mimic_sig { self.mimic = (ms["drift"].as_f64().unwrap_or(0.0) as f32 * 10.0).max(0.0); }
        self.synergy = syn; self.empowerment = emp;
        self.compositional = (self.compositional * 0.92) + (self.spectral.max(self.movement) * 0.08);
        self.compositional = self.compositional.min(1.0);
    }
    fn branch_aperture(&self) -> f32 {
        ((self.spectral * 0.20) + (self.movement * 0.25) + (self.mimic * 0.15) + (self.compositional * 0.10) + (self.synergy * 0.30)).clamp(0.05, 1.0)
    }
}

// --- SUBSYSTEMS ---
fn mod_2pi(x: &Tensor) -> CResult<Tensor> {
    let two_pi = 2.0 * std::f32::consts::PI;
    let mut v = x.to_scalar::<f32>()? % two_pi;
    if v < 0.0 { v += two_pi; }
    Tensor::new(v, x.device())
}

fn levy_radiate(tape: &Tensor, amp: f32) -> CResult<Tensor> {
    let dev = tape.device();
    let shape = tape.shape();
    let n1 = Tensor::randn(0.0f32, 1.0f32, shape, dev)?;
    let n2 = Tensor::randn(0.0f32, 1.0f32, shape, dev)?.abs()?.affine(1.0, 1e-3)?;
    let cauchy = n1.broadcast_div(&n2)?.clamp(-CAUCHY_CLAMP, CAUCHY_CLAMP)?;
    let mask = Tensor::rand(0.0f32, 1.0f32, shape, dev)?;
    let mask_t = mask.affine(1.0, -RADIATE_SPARSITY as f64)?.relu()?.affine(10000.0, 0.0)?.clamp(0.0f32, 1.0f32)?;
    tape.add(&cauchy.broadcast_mul(&mask_t)?.affine(amp as f64, 0.0)?)?.clamp(-1.0f32, 1.0f32)
}

// Structured, multi-scale, traveling-wave perturbation for the SLOW macro field.
// Unlike white noise, this injects coherent fBm structure (octave-summed sines with
// a per-octave traveling phase and per-channel dephasing), so it breaks rail-lock
// while preserving the macro field's slow-field character. This is the "magnetic
// shear" injected to keep field lines winding instead of welding to the rail.
// Structured, multi-scale, traveling-wave perturbation for the SLOW macro field.
// Precomputes the static spatial sine/cosine tables once; per step it advances the
// traveling phase via the angle-addition identity sin(A+B)=sinA cosB+cosA sinB, so the
// hot path is table lookups + multiply-adds with NO per-element sin() calls. This is the
// exact same fBm field as a naive per-element implementation, ~20x cheaper per step.
struct ShearField {
    sin_a: Vec<f32>,   // [octave * channels * len] sin(2pi*freq*x + c_phase)
    cos_a: Vec<f32>,   // [octave * channels * len] cos(2pi*freq*x + c_phase)
    freqs: [f32; SHEAR_OCTAVES],
    weights: [f32; SHEAR_OCTAVES],
    channels: usize,
    len: usize,
    scratch: Vec<f32>, // reusable [channels*len] accumulator — avoids a fresh heap Vec every step
}
impl ShearField {
    fn new(channels: usize, len: usize) -> Self {
        let cl = channels * len;
        let mut sin_a = vec![0.0f32; SHEAR_OCTAVES * cl];
        let mut cos_a = vec![0.0f32; SHEAR_OCTAVES * cl];
        let mut freqs = [0.0f32; SHEAR_OCTAVES];
        let mut weights = [0.0f32; SHEAR_OCTAVES];
        let mut freq = 1.0f32; let mut weight = 1.0f32;
        for oct in 0..SHEAR_OCTAVES {
            freqs[oct] = freq; weights[oct] = weight;
            for c in 0..channels {
                let c_phase = c as f32 * 0.1;
                for i in 0..len {
                    let x = i as f32 / len as f32;
                    let a = TWO_PI * freq * x + c_phase;
                    let idx = oct * cl + c * len + i;
                    sin_a[idx] = a.sin();
                    cos_a[idx] = a.cos();
                }
            }
            freq *= 2.0; weight *= 0.5;
        }
        Self { sin_a, cos_a, freqs, weights, channels, len, scratch: vec![0.0f32; cl] }
    }
    fn generate(&mut self, amp: f32, phase: f32, device: &Device) -> CResult<Tensor> {
        let cl = self.channels * self.len;
        for v in self.scratch.iter_mut() { *v = 0.0; }
        for oct in 0..SHEAR_OCTAVES {
            let b = phase * self.freqs[oct];
            let (sb, cb) = (b.sin(), b.cos());
            let w = self.weights[oct];
            let base = oct * cl;
            for k in 0..cl {
                // sin(a + b) = sin a cos b + cos a sin b
                self.scratch[k] += w * (self.sin_a[base + k] * cb + self.cos_a[base + k] * sb);
            }
        }
        for v in self.scratch.iter_mut() { *v *= amp; }
        // from_slice copies the reused buffer into tensor storage instead of consuming a
        // freshly-allocated Vec each call.
        Tensor::from_slice(&self.scratch, (1, self.channels, self.len), device)
    }
}

// Telemetry + control output of one disruption-avoidance step.
struct DisruptState {
    q: f32,          // raw safety factor: shear / (coupling * rail_proximity)
    q_norm: f32,     // q normalized to healthy baseline (1.0 = nominal, <0.4 = quenching)
    urgency: f32,    // [0,1] total shear drive (max of quench-avoidance and lock-break)
    shear_amp: f32,  // graded macro shear amplitude to apply this step
    recovered: bool, // true on the edge where q climbs back above the quench floor
    lock: f32,       // [0,1] absolute over-coupling pressure (L-mode lock-break drive)
}

// Kruskal-Shafranov-style avoidance controller. q_titan > floor => field lines wind
// (varied, loosely coupled, off-rail). q_titan < floor => kink/quench (pinned, locked,
// at-rail). A dual-EMA of macro variance gives the rate of approach to the boundary;
// time-to-quench drives a graded shear ramp that acts BEFORE the crossing.
struct DisruptionController {
    macrov_fast: f32,
    macrov_slow: f32,
    couple_ema: f32,
    q_baseline: Option<f32>,
    q_warmup_sum: f32,
    q_warmup_n: usize,
    was_disrupting: bool,
    initialized: bool,
}
impl DisruptionController {
    fn new() -> Self {
        Self { macrov_fast: 0.0, macrov_slow: 0.0, couple_ema: 0.0, q_baseline: None, q_warmup_sum: 0.0, q_warmup_n: 0, was_disrupting: false, initialized: false }
    }
    fn update(&mut self, macrov: f32, coupling: f32, rail_prox: f32, step: usize) -> DisruptState {
        let coupling = coupling.abs();
        if !self.initialized {
            self.macrov_fast = macrov; self.macrov_slow = macrov; self.couple_ema = coupling; self.initialized = true;
        }
        self.macrov_fast += DISRUPT_EMA_FAST * (macrov - self.macrov_fast);
        self.macrov_slow += DISRUPT_EMA_SLOW * (macrov - self.macrov_slow);
        self.couple_ema += DISRUPT_EMA_SLOW * (coupling - self.couple_ema);

        // q_titan: dynamical winding over (coupling x rail-proximity). High shear,
        // loose coupling, and distance from the rail all raise the safety factor.
        let q = self.macrov_fast / (coupling * rail_prox + 1e-4);

        // Self-calibrating baseline (matches the codebase's warmup-baseline idiom).
        if step < DISRUPT_WARMUP {
            self.q_warmup_sum += q; self.q_warmup_n += 1;
        } else if self.q_baseline.is_none() {
            self.q_baseline = Some((self.q_warmup_sum / self.q_warmup_n.max(1) as f32).max(1e-3));
        }
        let q_ref = self.q_baseline.unwrap_or(q.max(1e-3));
        let q_floor = q_ref * DISRUPT_Q_FLOOR_REL;
        let q_norm = q / q_ref;

        // Approach velocity: positive when fast EMA falls below slow EMA (variance dropping
        // toward the quench). This is the disruption precursor.
        let approach = (self.macrov_slow - self.macrov_fast).max(0.0);
        let q_margin = (q - q_floor).max(0.0);
        let ttq = q_margin / (approach * DISRUPT_RATE_GAIN + 1e-4); // time-to-quench (steps)
        let mut quench_urgency = 1.0 / (1.0 + ttq / DISRUPT_HORIZON);
        let disrupting = q < q_floor;
        if disrupting { quench_urgency = 1.0; } // already past the boundary -> full shear

        // ABSOLUTE lock pressure: persistent over-coupling (couple_ema above the ceiling)
        // is the L-mode lock. This is independent of the (poisonable) q baseline, so it
        // fires even when the lock formed before warmup finished.
        let lock = ((self.couple_ema - COUPLE_CEILING) / (1.0 - COUPLE_CEILING)).clamp(0.0, 1.0);

        // Total shear drive: whichever of quench-avoidance or lock-break is more urgent.
        let urgency = quench_urgency.max(lock * LOCK_SHEAR_SCALE);

        let recovered = self.was_disrupting && !disrupting;
        self.was_disrupting = disrupting;

        let shear_amp = SHEAR_AMP_MIN + (SHEAR_AMP_MAX - SHEAR_AMP_MIN) * urgency;
        DisruptState { q, q_norm, urgency, shear_amp, recovered, lock }
    }
}

fn quantile_dual_tape(vals: &[f32]) -> (String, String) {
    let mut v_lane = String::with_capacity(vals.len());
    let mut g_lane = String::with_capacity(vals.len());
    for &val in vals {
        let v_idx = ((val + 1.0) * 4.0).clamp(0.0, 7.99) as usize;
        v_lane.push_str(VAL_SYMS[v_idx]);
        let g_idx = ((val.abs()) * 8.0).clamp(0.0, 7.99) as usize;
        g_lane.push_str(GRAD_SYMS[g_idx]);
    }
    (v_lane, g_lane)
}

struct SemanticField {
    history: VecDeque<f32>,
    arch_history: VecDeque<usize>,
}
impl SemanticField {
    fn new() -> Self { Self { history: VecDeque::new(), arch_history: VecDeque::new() } }
    fn archetype_field(field01: &[f32]) -> (String, f32, usize) {
        let mut bins = [0usize; 8];
        let mut sum = 0.0;
        for &v in field01 {
            let mut a = 7;
            for (i, &b) in ARCH_BOUNDS.iter().skip(1).enumerate() {
                if v < b { a = i; break; }
            }
            bins[a] += 1; sum += v;
        }
        let n = field01.len() as f32 + 1e-6;
        let mut entropy = 0.0; let mut max_b = 0; let mut max_c = 0;
        let mut syms = String::with_capacity(8);
        for (i, &c) in bins.iter().enumerate() {
            if c > max_c { max_c = c; max_b = i; }
            let p = c as f32 / n;
            if p > 0.0 { entropy -= p * p.log2(); }
            let idx = (p * 16.0).clamp(0.0, 7.99) as usize;
            syms.push_str(VAL_SYMS[idx]);
        }
        (syms, entropy, max_b)
    }
    fn phase(drift: f32) -> &'static str {
        for &(bound, name) in &PHASE_MAP {
            if drift <= bound { return name; }
        }
        "PRIMORDIAL"
    }
    fn record(&mut self, drift: f32, _phase: &str, dom_arch: usize) {
        self.history.push_back(drift);
        if self.history.len() > 20 { self.history.pop_front(); }
        self.arch_history.push_back(dom_arch);
        if self.arch_history.len() > 20 { self.arch_history.pop_front(); }
    }
    fn trend(&self) -> &'static str {
        if self.history.len() < 2 { return "→"; }
        let a = self.history[0]; let b = *self.history.back().unwrap();
        if b > a + 0.05 { "↑" } else if b < a - 0.05 { "↓" } else { "→" }
    }
    fn dominant_phase(&self) -> &'static str {
        if self.history.is_empty() { return "PRIMORDIAL"; }
        let avg = self.history.iter().copied().sum::<f32>() / self.history.len() as f32;
        Self::phase(avg)
    }
    fn dominant_archetype(&self) -> &'static str {
        if self.arch_history.is_empty() { return ARCHETYPES[0]; }
        let mut counts = [0usize; 8];
        for &a in &self.arch_history { counts[a] += 1; }
        let mut max_c = 0; let mut max_a = 0;
        for (i, &c) in counts.iter().enumerate() {
            if c > max_c { max_c = c; max_a = i; }
        }
        ARCHETYPES[max_a]
    }
    fn commentary(&self, phase: &str, ev: Option<&str>, rad_amp: f32, depth: usize) -> String {
        if let Some(e) = ev {
            format!("❖ {} ❖  Depth L{:02} | Rad {:.2}", e, depth, rad_amp)
        } else {
            String::new()
        }
    }
}

struct DefibrillatorController {
    net: candle_nn::Sequential,
}
impl DefibrillatorController {
    fn new(vb: VBV) -> Result<Self> {
        let net = candle_nn::seq()
            .add(candle_nn::linear(7, 32, vb.pp("fc1"))?)
            .add(Tanh)
            .add(candle_nn::linear(32, 5, vb.pp("fc2"))?)
            .add(Sigmoid); // Replaced unbounded Softplus with strictly bounded Sigmoid
        Ok(Self { net })
    }
    fn forward(&self, features: &Tensor) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let out = self.net.forward(features)?;
        let pred = out.narrow(1, 0, 2)?; 
        
        // Dynamically scaled into safe ranges guaranteeing MSE loss gradients never explode
        let thresh_t = out.narrow(1, 2, 1)?; // Maps cleanly [0, 1]
        let n_scale_t = out.narrow(1, 3, 1)?.affine(5.0, 0.0)?; // Maps cleanly [0, 5.0]
        let lr_mult_t = out.narrow(1, 4, 1)?.affine(9.9, 0.1)?; // Maps cleanly [0.1, 10.0]
        
        Ok((pred, thresh_t, n_scale_t, lr_mult_t))
    }
}

struct AudioArbiter {
    net: candle_nn::Sequential,
}
impl AudioArbiter {
    fn new(vb: VBV) -> Result<Self> {
        let net = candle_nn::seq()
            .add(candle_nn::linear(14, 64, vb.pp("fc1"))?)
            .add(Tanh)
            .add(candle_nn::linear(64, 7, vb.pp("fc2"))?);
        Ok(Self { net })
    }
    fn forward(&self, features: &Tensor) -> Result<(Tensor, Tensor)> {
        let logits = self.net.forward(features)?;
        // Softmax intrinsically bounded and gradient-safe, removing division-by-sum singularity
        let w_norm = candle_nn::ops::softmax(&logits, D::Minus1)?;
        let entropy = w_norm.mul(&w_norm.affine(1.0, 1e-4)?.log()?)?.sum_all()?.neg()?;
        Ok((w_norm, entropy))
    }
}

struct MonitorHead {
    net: candle_nn::Sequential,
}
impl MonitorHead {
    fn new(vb: VBV) -> Result<Self> {
        let net = candle_nn::seq()
            .add(candle_nn::linear(MEMORY_DIM, 64, vb.pp("fc1"))?)
            .add(Tanh)
            .add(candle_nn::linear(64, 5, vb.pp("fc2"))?)
            .add(Sigmoid);
        Ok(Self { net })
    }
    fn forward(&self, features: &Tensor) -> Result<Tensor> {
        Ok(self.net.forward(features)?)
    }
}

struct KANLayer {
    basis_fn: usize,
    w: Tensor,
    mod_proj: Linear,
    freqs: Tensor, // (1, basis_fn) = [1, 2, ..., basis_fn], precomputed once
}
impl KANLayer {
    fn new(_in_dim: usize, _out_dim: usize, basis_fn: usize, vb: VBV) -> Result<Self> {
        let w = vb.get_with_hints((basis_fn,), "weights", candle_nn::Init::Randn { mean: 0.0, stdev: 0.1 })?;
        let mod_proj = candle_nn::linear(MEMORY_DIM, basis_fn, vb.pp("mod_proj"))?;
        let freq_vec: Vec<f32> = (1..=basis_fn).map(|i| i as f32).collect();
        let freqs = Tensor::from_vec(freq_vec, (1, basis_fn), vb.device())?;
        Ok(Self { basis_fn, w, mod_proj, freqs })
    }
    fn forward(&self, x: &Tensor, mem: &Tensor) -> CResult<Tensor> {
        // Vectorized identity of: sum_i active_w[i] * sin((i+1) * x) / sqrt(basis_fn).
        // The old per-basis loop launched 128 small kernels per call (x2 channels per
        // step); this collapses to a handful of batched ops with the same gradient.
        let (d0, d1) = x.dims2()?;
        let delta_w = self.mod_proj.forward(mem)?.tanh()?.reshape((self.basis_fn,))?;
        let active_w = self.w.add(&delta_w.affine(0.15, 0.0)?)?.reshape((1, self.basis_fn))?;
        let xf = x.reshape((d0 * d1, 1))?;                         // (N, 1)
        let basis = xf.broadcast_mul(&self.freqs)?.sin()?;        // (N, basis): sin((i+1)*x)
        let summed = basis.broadcast_mul(&active_w)?.sum(D::Minus1)?; // (N,)
        summed.reshape((d0, d1))?.affine(1.0 / (self.basis_fn as f64).sqrt(), 0.0)
    }
}

struct MorphicStack {
    layers: Vec<candle_nn::Sequential>,
    active_depth: usize,
}
impl MorphicStack {
    fn new(dim: usize, max_depth: usize, vb: VBV) -> Result<Self> {
        let mut layers = Vec::new();
        for i in 0..max_depth {
            let seq = candle_nn::seq()
                .add(candle_nn::linear(dim, dim, vb.pp(&format!("l{}_1", i)))?)
                .add(Relu) // Safe replacement for Softplus in stacked residuals
                .add(candle_nn::linear(dim, dim, vb.pp(&format!("l{}_2", i)))?)
                .add(Tanh);
            layers.push(seq);
        }
        Ok(Self { layers, active_depth: MORPH_START_DEPTH })
    }
    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let mut out = x.clone();
        for i in 0..self.active_depth {
            out = out.add(&self.layers[i].forward(&out)?)?;
        }
        Ok(out)
    }
    fn depth(&self) -> usize { self.active_depth }
    fn set_depth(&mut self, d: usize) { self.active_depth = d.clamp(1, self.layers.len()); }
    fn grow(&mut self) -> bool {
        if self.active_depth < self.layers.len() {
            self.active_depth += 1;
            true
        } else { false }
    }
    fn prune(&mut self) -> bool {
        if self.active_depth > 1 {
            self.active_depth -= 1;
            true
        } else { false }
    }
}

struct GRUCell {
    w_ir: Linear, w_hr: Linear,
    w_iz: Linear, w_hz: Linear,
    w_in: Linear, w_hn: Linear,
}
impl GRUCell {
    fn new(in_d: usize, hidden_d: usize, vb: VBV) -> Result<Self> {
        Ok(Self {
            w_ir: candle_nn::linear(in_d, hidden_d, vb.pp("w_ir"))?,
            w_hr: candle_nn::linear(hidden_d, hidden_d, vb.pp("w_hr"))?,
            w_iz: candle_nn::linear(in_d, hidden_d, vb.pp("w_iz"))?,
            w_hz: candle_nn::linear(hidden_d, hidden_d, vb.pp("w_hz"))?,
            w_in: candle_nn::linear(in_d, hidden_d, vb.pp("w_in"))?,
            w_hn: candle_nn::linear(hidden_d, hidden_d, vb.pp("w_hn"))?,
        })
    }
    fn forward(&self, x: &Tensor, h: &Tensor) -> CResult<Tensor> {
        let r = self.w_ir.forward(x)?.add(&self.w_hr.forward(h)?)?;
        let r = candle_nn::ops::sigmoid(&r)?;
        let z = self.w_iz.forward(x)?.add(&self.w_hz.forward(h)?)?;
        let z = candle_nn::ops::sigmoid(&z)?;
        let n = self.w_in.forward(x)?.add(&r.mul(&self.w_hn.forward(h)?)?)?.tanh()?;
        
        let one_minus_z = z.affine(-1.0, 1.0)?; 
        one_minus_z.mul(h)?.add(&z.mul(&n)?)
    }
}

struct NeuralCA1D {
    conv1: Box<dyn Fn(&Tensor) -> CResult<Tensor>>,
    conv2: Box<dyn Fn(&Tensor) -> CResult<Tensor>>,
    anisotropic_mask: Tensor,
}
impl NeuralCA1D {
    fn new(channels: usize, mult: usize, vb: VBV) -> Result<Self> {
        let hidden = channels * mult;
        let conv1 = conv1d_circular(channels, hidden, 3, vb.pp("c1"))?;
        let conv2 = conv1d_circular(hidden, channels, 1, vb.pp("c2"))?;
        
        let mut pattern = vec![0.0f32; channels * TAPE_LEN];
        for c in 0..channels {
            for i in 0..TAPE_LEN {
                let phase = (i as f32 / TAPE_LEN as f32) * TWO_PI;
                pattern[c * TAPE_LEN + i] = 0.8 + 0.4 * ((phase * (1.0 + (c % 3) as f32) + (c as f32)).sin());
            }
        }
        let anisotropic_mask = Tensor::from_vec(pattern, (1, channels, TAPE_LEN), vb.device())?;

        Ok(Self { conv1, conv2, anisotropic_mask })
    }
    fn forward(&self, x: &Tensor, ext_mod: Option<&Tensor>, field_bias: Option<&Tensor>) -> CResult<Tensor> {
        let h = (self.conv1)(x)?.relu()?;
        let mut out = (self.conv2)(&h)?;
        out = out.broadcast_mul(&self.anisotropic_mask)?;
        if let Some(m) = ext_mod {
            out = out.broadcast_mul(&m.unsqueeze(2)?)?;
        }
        if let Some(f) = field_bias {
            out = out.add(f)?;
        }
        x.add(&out.affine(0.1, 0.0)?)
    }
}

// --- TENSOR MODEL STRUCTS ---
struct ForwardOut {
    stereo: Tensor,
    next_micro: Tensor,
    next_macro: Tensor,
    next_hidden: Tensor,
    refined_hidden: Tensor,
    movement_t: Tensor,
    theta: f32,
}

struct ComplexAudioEcosystem {
    micro_ca: NeuralCA1D, macro_ca: NeuralCA1D, gru_memory: GRUCell, morphic: MorphicStack,
    asymptotic_contraction: AsymptoticContractionLayer, spatial_panner: candle_nn::Sequential,
    fm_mod_ratio: candle_nn::Sequential, fm_mod_index: candle_nn::Sequential,
    wave_morph_head: candle_nn::Sequential,
    wavefolder_l: KANLayer, wavefolder_r: KANLayer, base_freq_l: Tensor, base_freq_r: Tensor,
    t_steps: Tensor, ramp: Tensor, current_freq_l: Tensor, current_freq_r: Tensor,
    current_mod_freq_l: Tensor, current_mod_freq_r: Tensor, prev_fm_idx_l: Tensor, prev_fm_idx_r: Tensor,
    prev_openness: Tensor, prev_gain_l: Tensor, prev_gain_r: Tensor, prev_theta: f32,
}

impl ComplexAudioEcosystem {
    fn new(vb: VBV, dev: &Device) -> Result<Self> {
        let micro_ca = NeuralCA1D::new(CA_CHANNELS, CA_HIDDEN_MULT, vb.pp("micro_ca"))?;
        let macro_ca = NeuralCA1D::new(CA_CHANNELS, CA_HIDDEN_MULT, vb.pp("macro_ca"))?;
        let gru_memory = GRUCell::new(CA_CHANNELS, MEMORY_DIM, vb.pp("gru_memory"))?;
        let morphic = MorphicStack::new(MEMORY_DIM, MORPH_MAX_BLOCKS, vb.pp("morphic"))?;
        let asymptotic_contraction = AsymptoticContractionLayer::new(MEMORY_DIM, LARGE_D_DIM, CA_CHANNELS, vb.pp("asymp_contract"))?;
        let spatial_panner = candle_nn::seq().add(candle_nn::linear(MEMORY_DIM, 1, vb.pp("spatial_panner_0"))?).add(Tanh);
        let fm_mod_ratio = candle_nn::seq().add(candle_nn::linear(MEMORY_DIM, 2, vb.pp("fm_mod_ratio_0"))?).add(Relu);
        let fm_mod_index = candle_nn::seq().add(candle_nn::linear(MEMORY_DIM, 2, vb.pp("fm_mod_index_0"))?).add(Sigmoid);
        let wave_morph_head = candle_nn::seq().add(candle_nn::linear(MEMORY_DIM, 2, vb.pp("wave_morph_head_0"))?).add(Sigmoid);
        let wavefolder_l = KANLayer::new(1, 1, KAN_BASIS_FUNCTIONS, vb.pp("wavefolder_l"))?;
        let wavefolder_r = KANLayer::new(1, 1, KAN_BASIS_FUNCTIONS, vb.pp("wavefolder_r"))?;
        let base_freq_l = vb.get_with_hints((1,), "base_freq_l", candle_nn::Init::Const(BASE_FREQ_L as f64))?;
        let base_freq_r = vb.get_with_hints((1,), "base_freq_r", candle_nn::Init::Const(BASE_FREQ_R as f64))?;
        let steps_vec: Vec<f32> = (0..CHUNK_SIZE).map(|i| i as f32 / SAMPLE_RATE as f32).collect();
        let ramp_vec: Vec<f32> = (0..CHUNK_SIZE).map(|i| i as f32 / (CHUNK_SIZE as f32 - 1.0)).collect();
        Ok(Self {
            micro_ca, macro_ca, gru_memory, morphic, asymptotic_contraction, spatial_panner, fm_mod_ratio, fm_mod_index,
            wave_morph_head, wavefolder_l, wavefolder_r, base_freq_l, base_freq_r, t_steps: Tensor::new(steps_vec, dev)?, ramp: Tensor::new(ramp_vec, dev)?,
            current_freq_l: Tensor::new(BASE_FREQ_L, dev)?, current_freq_r: Tensor::new(BASE_FREQ_R, dev)?,
            current_mod_freq_l: Tensor::new(BASE_FREQ_L, dev)?, current_mod_freq_r: Tensor::new(BASE_FREQ_R, dev)?,
            prev_fm_idx_l: Tensor::new(0.0f32, dev)?, prev_fm_idx_r: Tensor::new(0.0f32, dev)?,
            prev_openness: Tensor::new(0.7f32, dev)?, prev_gain_l: Tensor::new(0.707f32, dev)?, prev_gain_r: Tensor::new(0.707f32, dev)?,
            prev_theta: 0.0,
        })
    }
    fn depth(&self) -> usize { self.morphic.depth() }
    fn set_depth(&mut self, d: usize) { self.morphic.set_depth(d); }
    fn grow(&mut self) -> bool { self.morphic.grow() }
    fn prune(&mut self) -> bool { self.morphic.prune() }

    fn ramp_param(&self, new_val: &Tensor, prev_val: &Tensor) -> CResult<Tensor> {
        let delta = new_val.sub(prev_val)?;
        self.ramp.broadcast_mul(&delta.reshape((1,))?)?.broadcast_add(prev_val)
    }
    fn forward(&mut self, micro: &Tensor, macro_t: &Tensor, mem: &Tensor, pc_l: &Tensor, pc_r: &Tensor, pm_l: &Tensor, pm_r: &Tensor, force: bool, energy: f32) -> Result<(ForwardOut, Tensor, Tensor, Tensor, Tensor)> {
        let mut next_macro = macro_t.clone();
        if force {
            // Restoring (anti-rail) metabolic field: +bias where |macro| is small,
            // -bias where |macro| is large. Removes the old constant +0.999 DC drive
            // that welded cells to +1. Soft-saturate (0.95*tanh) instead of hard clip
            // so cells can never sit exactly on the rail.
            let field = macro_t.abs()?.affine(-0.04, 0.02)?; // 0.02 - 0.04*|macro|
            next_macro = self.macro_ca.forward(macro_t, None, Some(&field))?.tanh()?.affine(0.95, 0.0)?;
        }
        let macro_act = next_macro.abs()?.mean_all()?;
        let metab = macro_act.affine(5.0, 0.0)?.clamp(0.01f32, 1.0f32)?;
        let inv_metab = metab.affine(-1.0, 1.0)?;
        let contracted_mem = self.asymptotic_contraction.forward(mem)?;
        let macro_mod = contracted_mem.add(&next_macro.mean(D::Minus1)?)?;
        let micro_field = micro.abs()?.affine(-0.01, METABOLIC_DECAY as f64)?;
        let raw_next_micro = self.micro_ca.forward(micro, Some(&macro_mod), Some(&micro_field))?;
        let next_micro = micro.broadcast_mul(&inv_metab)?.add(&raw_next_micro.broadcast_mul(&metab)?)?.clamp(-1.0f32, 1.0f32)?;
        let movement_t = next_micro.sub(micro)?.abs()?.mean_all()?;
        let pop_l = next_micro.narrow(1, 0, 1)?.mean_all()?;
        let pop_r = next_micro.narrow(1, 1, 1)?.mean_all()?;
        let micro_feats = next_micro.mean(D::Minus1)?.reshape((CA_CHANNELS,))?;
        let paired = micro_feats.reshape((CA_CHANNELS / 2, 2))?;
        let sums = paired.sum(0)?.to_vec1::<f32>()?;
        let theta = sums[1].atan2(sums[0] + 1e-6);
        let tape_feats = next_micro.mean(D::Minus1)?;
        let next_hidden = self.gru_memory.forward(&tape_feats, mem)?;
        let refined_hidden = self.morphic.forward(&next_hidden)?;
        let fm_ratios = self.fm_mod_ratio.forward(&refined_hidden)?.affine(4.0, 0.0)?;
        let fm_indices = self.fm_mod_index.forward(&refined_hidden)?.affine(5.0, 0.0)?;
        let ratio_l = fm_ratios.narrow(1, 0, 1)?.reshape(())?;
        let ratio_r = fm_ratios.narrow(1, 1, 1)?.reshape(())?;
        let idx_l = fm_indices.narrow(1, 0, 1)?.reshape(())?;
        let idx_r = fm_indices.narrow(1, 1, 1)?.reshape(())?;
        let b_l = self.base_freq_l.reshape(())?.abs()?;
        let b_r = self.base_freq_r.reshape(())?.abs()?;
        
        let energy_factor = energy.clamp(0.15, 1.0);
        let target_l = b_l.add(&pop_l.affine(200.0, 0.0)?)?.add(&movement_t.affine(100.0, 0.0)?)?.clamp(20.0f32, 4000.0f32)?.affine(energy_factor as f64, 0.0)?;
        let target_r = b_r.add(&pop_r.affine(200.0, 0.0)?)?.add(&movement_t.affine(-100.0, 0.0)?)?.clamp(20.0f32, 4000.0f32)?.affine(energy_factor as f64, 0.0)?;
        
        let g = FREQ_GLIDE_SPEED as f64;
        let cur_l = target_l.affine(g, 0.0)?.add(&self.current_freq_l.affine(1.0 - g, 0.0)?)?;
        let cur_r = target_r.affine(g, 0.0)?.add(&self.current_freq_r.affine(1.0 - g, 0.0)?)?;
        let mod_f_l = cur_l.mul(&ratio_l)?.clamp(0.0f32, 6000.0f32)?;
        let mod_f_r = cur_r.mul(&ratio_r)?.clamp(0.0f32, 6000.0f32)?;
        let omega_m_l = mod_f_l.affine(TWO_PI as f64, 0.0)?;
        let omega_m_r = mod_f_r.affine(TWO_PI as f64, 0.0)?;
        let ph_m_l = self.t_steps.broadcast_mul(&omega_m_l)?.broadcast_add(pm_l)?;
        let ph_m_r = self.t_steps.broadcast_mul(&omega_m_r)?.broadcast_add(pm_r)?;
        let idx_curve_l = self.ramp_param(&idx_l, &self.prev_fm_idx_l)?;
        let idx_curve_r = self.ramp_param(&idx_r, &self.prev_fm_idx_r)?;
        let modulator_l = ph_m_l.sin()?.mul(&idx_curve_l)?;
        let modulator_r = ph_m_r.sin()?.mul(&idx_curve_r)?;
        let mut dtheta = theta - self.prev_theta; dtheta -= TWO_PI * (dtheta / TWO_PI).round();
        let theta_curve = self.ramp.affine(dtheta as f64, self.prev_theta as f64)?;
        let omega_c_l = cur_l.affine(TWO_PI as f64, 0.0)?;
        let omega_c_r = cur_r.affine(TWO_PI as f64, 0.0)?;
        let ph_c_l = self.t_steps.broadcast_mul(&omega_c_l)?.broadcast_add(pc_l)?.add(&theta_curve)?.add(&modulator_l)?;
        let ph_c_r = self.t_steps.broadcast_mul(&omega_c_r)?.broadcast_add(pc_r)?.add(&theta_curve)?.add(&modulator_r)?;
        
        let morphs = self.wave_morph_head.forward(&refined_hidden)?;
        let morph_l = morphs.narrow(1, 0, 1)?.reshape(())?;
        let morph_r = morphs.narrow(1, 1, 1)?.reshape(())?;
        
        let mut audio_l = morph_wave(&ph_c_l, &morph_l)?;
        let mut audio_r = morph_wave(&ph_c_r, &morph_r)?;
        for (f_l, f_r) in [(&pop_l.affine(700.0, 300.0)?, &pop_r.affine(700.0, 300.0)?), (&movement_t.affine(1700.0, 800.0)?, &movement_t.affine(1700.0, 800.0)?), (&pop_l.affine(-500.0, 2000.0)?, &pop_r.affine(-500.0, 2000.0)?)] {
            let p_l = self.t_steps.broadcast_mul(&f_l.affine(TWO_PI as f64, 0.0)?)?.broadcast_add(pc_l)?.add(&theta_curve)?;
            let p_r = self.t_steps.broadcast_mul(&f_r.affine(TWO_PI as f64, 0.0)?)?.broadcast_add(pc_r)?.add(&theta_curve)?;
            audio_l = audio_l.add(&morph_wave(&p_l, &morph_l)?.affine(0.3, 0.0)?)?;
            audio_r = audio_r.add(&morph_wave(&p_r, &morph_r)?.affine(0.3, 0.0)?)?;
        }
        
        let audio_l = self.wavefolder_l.forward(&audio_l.unsqueeze(0)?, &refined_hidden)?.reshape((1, CHUNK_SIZE))?;
        let audio_r = self.wavefolder_r.forward(&audio_r.unsqueeze(0)?, &refined_hidden)?.reshape((1, CHUNK_SIZE))?;
        
        let open_t = refined_hidden.abs()?.mean_all()?.affine(5.0, 0.0)?.add(&movement_t)?.clamp(0.4f32, 1.0f32)?.affine(energy_factor as f64, 0.0)?;
        let open_curve = self.ramp_param(&open_t, &self.prev_openness)?;
        let audio_l = audio_l.broadcast_mul(&open_curve.unsqueeze(0)?)?;
        let audio_r = audio_r.broadcast_mul(&open_curve.unsqueeze(0)?)?;
        
        let mid = audio_l.add(&audio_r)?.affine(0.5, 0.0)?;
        let side = audio_l.sub(&audio_r)?.affine(0.5, 0.0)?;
        
        let pan_t = self.spatial_panner.forward(&refined_hidden)?.reshape(())?.clamp(-0.5f32, 0.5f32)?;
        let pan_t_scalar = pan_t.to_scalar::<f32>().unwrap_or(0.0);
        let width_val = 1.0 + pan_t_scalar.abs() * 0.8;
        let side_wide = side.affine(width_val as f64, 0.0)?;
        let side_delayed = apply_haas_delay(&side_wide, 16)?;
        
        let audio_l = mid.add(&side_delayed)?;
        let audio_r = mid.sub(&side_delayed)?;
        
        let gain_l = pan_t.affine(-0.5, 0.5)?.sqrt()?; let gain_r = pan_t.affine(0.5, 0.5)?.sqrt()?;
        let gain_curve_l = self.ramp_param(&gain_l, &self.prev_gain_l)?;
        let gain_curve_r = self.ramp_param(&gain_r, &self.prev_gain_r)?;
        let audio_l = audio_l.broadcast_mul(&gain_curve_l.unsqueeze(0)?)?.affine(1.414, 0.0)?;
        let audio_r = audio_r.broadcast_mul(&gain_curve_r.unsqueeze(0)?)?.affine(1.414, 0.0)?;
        
        let stereo = Tensor::cat(&[&audio_l, &audio_r], 0)?.reshape((2, CHUNK_SIZE))?;
        let chunk_dt = CHUNK_SIZE as f32 / SAMPLE_RATE as f32;
        let next_phase_c_l = mod_2pi(&pc_l.add(&cur_l.affine(TWO_PI as f64 * chunk_dt as f64, 0.0)?)?)?;
        let next_phase_c_r = mod_2pi(&pc_r.add(&cur_r.affine(TWO_PI as f64 * chunk_dt as f64, 0.0)?)?)?;
        let next_phase_m_l = mod_2pi(&pm_l.add(&mod_f_l.affine(TWO_PI as f64 * chunk_dt as f64, 0.0)?)?)?;
        let next_phase_m_r = mod_2pi(&pm_r.add(&mod_f_r.affine(TWO_PI as f64 * chunk_dt as f64, 0.0)?)?)?;
        self.current_freq_l = cur_l.detach()?; self.current_freq_r = cur_r.detach()?;
        self.current_mod_freq_l = mod_f_l.detach()?; self.current_mod_freq_r = mod_f_r.detach()?;
        self.prev_fm_idx_l = idx_l.detach()?; self.prev_fm_idx_r = idx_r.detach()?;
        self.prev_openness = open_t.detach()?; self.prev_gain_l = gain_l.detach()?; self.prev_gain_r = gain_r.detach()?;
        self.prev_theta = theta;
        Ok((ForwardOut { stereo, next_micro, next_macro, next_hidden, refined_hidden, movement_t, theta }, next_phase_c_l, next_phase_c_r, next_phase_m_l, next_phase_m_r))
    }
}

// --- MAIN RUNTIME LOGIC ---
fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut base_dir = "/sdcard/Download".to_string();
    let mut n_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
    let mut target_lr = BASE_LR;
    let mut sim_duration = DURATION_SECONDS;
    let mut bptt_window = BPTT_WINDOW;
    let mut arg_idx = 1;
    while arg_idx < args.len() {
        match args[arg_idx].as_str() {
            "--base-dir" | "-b" => { if arg_idx + 1 < args.len() { base_dir = args[arg_idx + 1].clone(); arg_idx += 2; } else { anyhow::bail!("Missing value for --base-dir"); } }
            "--threads" | "-t" => { if arg_idx + 1 < args.len() { n_threads = args[arg_idx + 1].parse::<usize>()?; arg_idx += 2; } else { anyhow::bail!("Missing value for --threads"); } }
            "--lr" | "-l" => { if arg_idx + 1 < args.len() { target_lr = args[arg_idx + 1].parse::<f64>()?; arg_idx += 2; } else { anyhow::bail!("Missing value for --lr"); } }
            "--duration" | "-d" => { if arg_idx + 1 < args.len() { sim_duration = args[arg_idx + 1].parse::<f32>()?; arg_idx += 2; } else { anyhow::bail!("Missing value for --duration"); } }
            "--bptt" | "-w" => { if arg_idx + 1 < args.len() { bptt_window = args[arg_idx + 1].parse::<usize>()?; arg_idx += 2; } else { anyhow::bail!("Missing value for --bptt"); } }
            _ => { if arg_idx == 1 && !args[arg_idx].starts_with('-') { base_dir = args[arg_idx].clone(); arg_idx += 1; } else { println!("Unknown parameter: {}", args[arg_idx]); arg_idx += 1; } }
        }
    }
    rayon::ThreadPoolBuilder::new().num_threads(n_threads).build_global()?;
    let device = Device::Cpu;
    println!("=== TITAN AUDIO ECOSYSTEM: RUST EDITION (GRADIENT-COHERENT RELEASE) ===");
    println!("Threads: {} | BPTT window: {} | Tape: {}x{} | CA hidden: {} | Base LR: {:.2e} | Duration: {}s", n_threads, bptt_window, CA_CHANNELS, TAPE_LEN, CA_CHANNELS * CA_HIDDEN_MULT, target_lr, sim_duration);

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

    if std::path::Path::new(&model_path).exists() {
        match load_into_varmap(&varmap, &model_path, &device) {
            Ok((hit, miss, mismatch)) => println!("--> Loaded {} tensors from {} ({} new/uninitialized, {} shape-mismatched)", hit, model_path, miss, mismatch),
            Err(e) => println!("--> Could not load {}: {} — starting fresh", model_path, e),
        }
    }

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
    let mut disruptor = DisruptionController::new();
    let mut shear_gen = ShearField::new(CA_CHANNELS, TAPE_LEN);
    let mut shear_phase = 0.0f32;
    let mut uncertainty = AudioUncertaintyState::new();
    let mut semantic = SemanticField::new();
    let mut morph_history = Vec::new();
    let mut morph_baseline: Option<f32> = None;
    let mut warmup_sum = 0.0f32; let mut field_entropy_sum = 0.0f64; let mut field_entropy_n = 0u64;
    let mut optimizer = AdamW::new_lr(varmap.all_vars(), target_lr).map_err(anyhow::Error::msg)?;

    let mut micro_tape = Tensor::randn(0.0f32, 1.0f32, (1, CA_CHANNELS, TAPE_LEN), &device).map_err(anyhow::Error::msg)?;
    let mut macro_tape = Tensor::randn(0.0f32, 1.0f32, (1, CA_CHANNELS, TAPE_LEN), &device).map_err(anyhow::Error::msg)?;
    let mut hidden_mem = Tensor::zeros((1, MEMORY_DIM), DType::F32, &device).map_err(anyhow::Error::msg)?;
    let mut phase_c_l = Tensor::new(0.0f32, &device).map_err(anyhow::Error::msg)?;
    let mut phase_c_r = Tensor::new(0.0f32, &device).map_err(anyhow::Error::msg)?;
    let mut phase_m_l = Tensor::new(0.0f32, &device).map_err(anyhow::Error::msg)?;
    let mut phase_m_r = Tensor::new(0.0f32, &device).map_err(anyhow::Error::msg)?;

    let total_chunks = (SAMPLE_RATE as f32 * sim_duration / CHUNK_SIZE as f32) as usize;
    let mut audio_frames = Vec::with_capacity(total_chunks * CHUNK_SIZE * 2);
    let mut topology_history = Vec::new();
    let mut uncertainty_trace = Vec::new();

    let mut burst_ticks = 0; let mut burst_energy = 0.0f32; let mut refractory = 0;
    let mut fire_rate_ema = 0.0f32; let mut alarm_streak = 0; let mut bursts_fired = 0;
    let mut phi = 0.0f32; let mut total_complexity = 0.0f32; let mut boost_state = 1.0f32;

    let mut window_loss: Option<Tensor> = None;
    let mut steps_in_window = 0; let mut latest_lr_gain = 1.0;
    let mut weight_clamp_tick = 0usize; // gate the O(params) weight-clamp to run periodically, not every window
    let mut prev_pred: Option<Tensor> = None;
    let mut prev_loss_vec = [0.0f32; 7]; let mut prev_movement = 0.0f32;

    // --- REVENUE-PACS STAGNATION & METABOLISM VALUES ---
    let mut energy_state = 1.0f32;
    let mut prev_archetype = "";
    let mut stagnation_ticks = 0usize;

    let timer_start = std::time::Instant::now();
    let mut profiling_lap = std::time::Instant::now();

    for step in 0..total_chunks {
        
        // --- BIO-RESET SAFETY MECHANISM ---
        // Prevents a NaN bomb from crashing the organism forever by re-seeding the primordial soup.
        let state_check = micro_tape.mean_all()?.add(&macro_tape.mean_all()?)?.to_scalar::<f32>().unwrap_or(f32::NAN);
        if !state_check.is_finite() {
            println!("! BIO-RESET: Tape corruption detected (NaN). Re-seeding primordial soup.");
            micro_tape = Tensor::randn(0.0f32, 1.0f32, (1, CA_CHANNELS, TAPE_LEN), &device)?;
            macro_tape = Tensor::randn(0.0f32, 1.0f32, (1, CA_CHANNELS, TAPE_LEN), &device)?;
            hidden_mem = Tensor::zeros((1, MEMORY_DIM), DType::F32, &device)?;
        }

        let aperture = uncertainty.branch_aperture();
        let force_macro = rand::thread_rng().gen_range(0.0..1.0) < (0.2 + aperture * 0.6);

        let (out, nc_l, nc_r, nm_l, nm_r) = model.forward(
            &micro_tape, &macro_tape, &hidden_mem,
            &phase_c_l, &phase_c_r, &phase_m_l, &phase_m_r,
            force_macro, energy_state,
        )?;
        let ForwardOut { stereo: stereo_chunk, next_micro, next_macro, next_hidden, refined_hidden, movement_t, theta } = out;

        let synergy_tensor = calculate_cross_layer_synergy_tensor(&next_micro, &next_macro)?;
        let synergy_val = synergy_tensor.to_scalar::<f32>().unwrap_or(0.0);

        // --- DISRUPTION-AVOIDANCE: measure operating point, predict approach to quench ---
        let macro_var_val = var_all(&next_macro)?.to_scalar::<f32>().unwrap_or(0.0);
        let rail_prox = next_macro.abs()?.mean_all()?.to_scalar::<f32>().unwrap_or(0.0);
        let ds = disruptor.update(macro_var_val, synergy_val, rail_prox, step);
        let memory_delta = next_hidden.sub(&hidden_mem)?;
        let tape_delta = next_micro.sub(&micro_tape)?;
        
        // Safety epsilon raised
        let trans_var = var_all(&memory_delta)?.add(&var_all(&tape_delta)?)?.affine(1.0, 1e-4)?;
        let cont_entropy_t = trans_var.log().map_err(anyhow::Error::msg)?.affine(0.5, 0.0)?;
        let empowerment_t = cont_entropy_t.affine(1.0, 7.0)?.clamp(0.0f32, 5.0f32)?.mul(&movement_t.affine(1.0, 1.0)?)?;
        let empowerment_val = empowerment_t.reshape(())?.to_scalar::<f32>().unwrap_or(0.0);
        let coarse_micro = decimate2(&next_micro)?;
        let coarse_macro = decimate2(&next_macro)?;
        let detached_macro = coarse_macro.detach()?;
        let rg_loss = coarse_micro.sub(&detached_macro)?.sqr()?.mean_all()?;

        let target_chunk = target_loader.sample_chunk(&device)?;
        let age_factor = (total_complexity / 500.0).min(0.6);
        let audio_for_loss = stereo_chunk.tanh()?.affine((1.0 - age_factor) as f64, 0.0)?;
        let out_spec_l = spec_proj.log_mag(&audio_for_loss.narrow(0, 0, 1)?)?;
        let out_spec_r = spec_proj.log_mag(&audio_for_loss.narrow(0, 1, 1)?)?;
        let tgt_spec_l = spec_proj.log_mag(&target_chunk.narrow(0, 0, 1)?)?.detach()?;
        let tgt_spec_r = spec_proj.log_mag(&target_chunk.narrow(0, 1, 1)?)?.detach()?;
        let mimic_l = out_spec_l.sub(&tgt_spec_l)?.sqr()?.mean_all()?;
        let mimic_r = out_spec_r.sub(&tgt_spec_r)?.sqr()?.mean_all()?;
        let mimic_loss = mimic_l.add(&mimic_r)?.affine(0.5, 0.0)?;

        let current_var = var_all(&audio_for_loss)?;
        let var_loss = current_var.affine(1.0, -0.12)?.sqr()?;
        
        let rms = audio_for_loss.sqr()?.mean_all()?.affine(1.0, 1e-4)?.sqrt()?;
        let saturation_loss = rms.affine(1.0, -0.28)?.sqr()?;
        let movement_loss = movement_t.neg()?.exp()?;
        let diff = audio_for_loss.narrow(1, 1, CHUNK_SIZE - 1)?.sub(&audio_for_loss.narrow(1, 0, CHUNK_SIZE - 1)?)?;
        let roughness_loss = diff.sqr()?.mean_all()?;
        let reg_loss = stereo_chunk.sqr()?.mean_all()?;
        let empowerment_loss = empowerment_t.affine(1.0, -2.5)?.sqr()?;
        
        // Target-band coupling: penalize deviation from SYNERGY_TARGET in BOTH directions.
        // The old `-synergy` reward drove coupling -> 1 (the zero-shear L-mode lock).
        let synergy_loss = synergy_tensor.affine(1.0, -(SYNERGY_TARGET as f64))?.sqr()?;

        let abs_max_t = audio_for_loss.abs()?.flatten_all()?.max(0)?;
        let first_metrics = Tensor::cat(&[
            &movement_t.reshape((1,))?, &mimic_loss.reshape((1,))?, &rms.reshape((1,))?,
            &abs_max_t.reshape((1,))?, &rg_loss.reshape((1,))?, &empowerment_loss.reshape((1,))?,
            &roughness_loss.reshape((1,))?, &current_var.reshape((1,))?, &movement_loss.reshape((1,))?,
        ], 0)?;
        let first_metrics_vec = first_metrics.to_vec1::<f32>()?;
        let movement = first_metrics_vec[0]; let mimic_drift = first_metrics_vec[1];
        let rms_val = first_metrics_vec[2]; let abs_max = first_metrics_vec[3];
        let rg_v = first_metrics_vec[4]; let empowerment_loss_val = first_metrics_vec[5];
        let roughness_loss_val = first_metrics_vec[6]; let current_var_val = first_metrics_vec[7];
        let movement_loss_val = first_metrics_vec[8];
        let mimic_drift_n = mimic_drift / (1.0 + mimic_drift);
        total_complexity += movement;

        let boost_target = if abs_max < 0.25 { (0.25 / (abs_max + 1e-6)).clamp(1.0, 4.0) } else { 1.0 };
        boost_state = boost_state * 0.9 + boost_target * 0.1;
        let audio_normalized = audio_for_loss.affine(boost_state as f64, 0.0)?;

        let m_sig = movement_mon.analyze(movement)?;
        let s_sig = spectral_mon.analyze(&stereo_chunk)?;
        uncertainty.update(&s_sig, &m_sig, Some(&serde_json::json!({"drift": mimic_drift_n, "theta": theta})), synergy_val, empowerment_val);
        phi = uncertainty.phi;

        // Telemetry decoupling: derive the archetype/field-entropy readout from the
        // per-channel summary (96 values) rather than syncing all CHANNELS*TAPE_LEN cells
        // to CPU every step. This 512x-smaller sync also aligns the archetype signal with
        // the channel projection the synthesis actually consumes.
        let field_summary = next_micro.mean(D::Minus1)?.reshape((CA_CHANNELS,))?.to_vec1::<f32>()?;
        let field01: Vec<f32> = field_summary.iter().map(|&x| (x + 1.0) * 0.5).collect();
        let (arch_summary, field_entropy, dom_arch) = SemanticField::archetype_field(&field01);
        let phase = SemanticField::phase(mimic_drift_n);
        semantic.record(mimic_drift_n, phase, dom_arch);
        let trend = semantic.trend();
        field_entropy_sum += field_entropy as f64; field_entropy_n += 1;

        // --- TRACK STAGNATON & CURIOSTIY REGIMES ---
        let current_arch = semantic.dominant_archetype();
        if current_arch == prev_archetype {
            stagnation_ticks += 1;
        } else {
            stagnation_ticks = 0;
            prev_archetype = current_arch;
        }
        let curiosity_factor = (stagnation_ticks as f32 / 12.0).min(1.0);

        // --- COMPUTE METABOLIC CHARGE SYSTEMS ---
        let freq_l_val = model.current_freq_l.to_scalar::<f32>().unwrap_or(BASE_FREQ_L);
        let freq_r_val = model.current_freq_r.to_scalar::<f32>().unwrap_or(BASE_FREQ_R);
        let metabolic_cost = (rms_val * 0.4 + (freq_l_val + freq_r_val) / 6000.0) * 0.012;
        energy_state = (energy_state - metabolic_cost).max(0.12);
        
        let energy_recharge = (1.0 - rms_val).max(0.0) * 0.008;
        energy_state = (energy_state + energy_recharge).min(1.0);
        // Homeostat: gently pull energy toward a setpoint so it stops welding to the 1.0
        // ceiling (which removed all metabolic scarcity pressure). Keeps headroom both ways.
        energy_state = energy_state + ENERGY_HOMEO_RATE * (ENERGY_SETPOINT - energy_state);
        energy_state = energy_state.clamp(0.12, 1.0);

        let mut morph_event = None;
        if step < MORPH_WARMUP { warmup_sum += mimic_drift_n; } else {
            if morph_baseline.is_none() {
                let b = (warmup_sum / MORPH_WARMUP as f32).max(1e-4);
                morph_baseline = Some(b);
                println!("--> Morph baseline calibrated: mimic≈{:.3}  (grow>{:.3}, prune<{:.3})", b, b * MORPH_GROWTH_REL, b * MORPH_PRUNE_REL);
            }
            morph_history.push(mimic_drift_n);
            let patience = MORPH_PATIENCE_BASE + model.depth() * 2;
            if morph_history.len() >= patience {
                let avg = morph_history.iter().sum::<f32>() / morph_history.len() as f32;
                morph_history.clear();
                let base = morph_baseline.unwrap();
                if avg > base * MORPH_GROWTH_REL {
                    if model.grow() { rad_amp = (rad_amp * RAD_COOL).max(RAD_AMP_MIN); morph_event = Some("NEUROGENESIS"); }
                } else if avg < base * MORPH_PRUNE_REL {
                    if model.prune() { rad_amp = (rad_amp * RAD_HEAT).min(RAD_AMP_MAX); morph_event = Some("PRUNING"); }
                }
            }
        }
        if let Some(ev) = morph_event {
            let line = semantic.commentary(phase, Some(ev), rad_amp, model.depth());
            println!("  ◄ {} ►  {}", ev, line);
        }

        let pred_state = monitor_head.forward(&refined_hidden)?;
        let observed_state = Tensor::new(&[uncertainty.spectral.clamp(0.0, 1.0), (uncertainty.movement / 2.0).clamp(0.0, 1.0), uncertainty.mimic.clamp(0.0, 1.0), aperture.clamp(0.0, 1.0), (synergy_val / 5.0).clamp(0.0, 1.0)], &device)?.unsqueeze(0)?;
        let self_model_loss = pred_state.sub(&observed_state)?.sqr()?.mean_all()?;

        let arb_features = Tensor::new(&[rms_val, mimic_drift_n, movement / 0.3, synergy_val / 5.0, empowerment_val / 5.0, rg_v, uncertainty.spectral, uncertainty.movement, uncertainty.mimic, uncertainty.compositional, aperture, step as f32 / total_chunks as f32, phi / 10.0, theta / std::f32::consts::PI], &device)?.unsqueeze(0)?;
        let (w_graph, arb_entropy_loss) = arbiter.forward(&arb_features)?;
        let lw_raw = w_graph.reshape((7,))?.to_vec1::<f32>()?;
        let lw: Vec<f32> = lw_raw.iter().map(|p| p * 7.0).collect();

        let defib_features = Tensor::new(&[movement, movement - prev_movement, mimic_drift_n, rms_val, aperture, phi / 10.0, step as f32 / total_chunks as f32], &device)?.unsqueeze(0)?;
        prev_movement = movement;
        let (pred_t, thresh_t, n_scale_t, lr_mult_t) = defib_ctrl.forward(&defib_features)?;
        let defib_pred_loss = if let Some(p) = prev_pred.take() {
            let obs = Tensor::new(&[(movement / 0.3).clamp(0.0, 1.0), mimic_drift_n], &device)?.unsqueeze(0)?;
            Some(p.sub(&obs)?.sqr()?.mean_all()?)
        } else { None };
        prev_pred = Some(pred_t);

        let second_metrics = Tensor::cat(&[&self_model_loss.reshape((1,))?, &thresh_t.reshape((1,))?, &n_scale_t.reshape((1,))?, &lr_mult_t.reshape((1,))?], 0)?;
        let second_metrics_vec = second_metrics.to_vec1::<f32>()?;
        let self_model_loss_val = second_metrics_vec[0];
        
        let thresh = second_metrics_vec[1].clamp(0.01, 1.0); 
        let n_scale = second_metrics_vec[2].clamp(0.0, 5.0); 
        let lr_mult = second_metrics_vec[3].clamp(0.1, 10.0);

        let cur_loss_vec = [current_var_val, mimic_drift_n, movement_loss_val, roughness_loss_val, rg_v, self_model_loss_val, empowerment_loss_val];
        let improvement: Vec<f32> = (0..7).map(|i| (prev_loss_vec[i] - cur_loss_vec[i]).clamp(-1.0, 1.0)).collect();
        prev_loss_vec = cur_loss_vec;
        let improvement_t = Tensor::new(improvement, &device)?;
        let arb_progress_loss = w_graph.reshape((7,))?.mul(&improvement_t)?.sum_all()?.affine(-0.5, 0.0)?;

        let mut total_loss = mimic_loss.affine((lw[1] * (1.0 - RESONANT_AUTONOMY)) as f64, 0.0)?;
        total_loss = total_loss.add(&var_loss.affine((lw[0] * 2.5) as f64, 0.0)?)?;
        total_loss = total_loss.add(&saturation_loss.affine(2.0, 0.0)?)?;
        total_loss = total_loss.add(&movement_loss.affine((lw[2] * RESONANT_AUTONOMY) as f64, 0.0)?)?;
        total_loss = total_loss.add(&roughness_loss.affine(lw[3] as f64, 0.0)?)?;
        total_loss = total_loss.add(&reg_loss.affine(0.01, 0.0)?)?;
        total_loss = total_loss.add(&rg_loss.affine((0.15 * lw[4].max(0.2)) as f64, 0.0)?)?;
        total_loss = total_loss.add(&self_model_loss.affine((0.30 * lw[5].max(0.2)) as f64, 0.0)?)?;
        total_loss = total_loss.add(&empowerment_loss.affine(lw[6] as f64, 0.0)?)?;
        // Dynamic synergy relaxation: when the controller reports absolute over-coupling
        // (ds.lock) or the system is stagnating, back the band penalty off so micro and macro
        // can differentiate instead of being welded together — welding collapses the safety
        // factor q toward the quench boundary, which is exactly the pinned regime observed.
        // Dynamic synergy relaxation. The absolute lock detector (ds.lock) almost never fires
        // in practice (telemetry showed lock==0 throughout), so key the release on whatever is
        // actually signalling trouble: absolute lock, a depressed safety factor (q below ~0.6
        // of baseline), or stagnation. Any of these eases the band penalty so micro/macro can
        // differentiate instead of being pulled together while q sits near the quench floor.
        let q_release = (1.0 - ds.q_norm / 0.6).clamp(0.0, 1.0);
        let synergy_release = ds.lock.max(q_release).max(curiosity_factor);
        let synergy_w = (SYNERGY_BAND_W * (1.0 - 0.7 * synergy_release)).max(0.1f32);
        total_loss = total_loss.add(&synergy_loss.affine(synergy_w as f64, 0.0)?)?;
        total_loss = total_loss.add(&arb_entropy_loss)?;
        total_loss = total_loss.add(&arb_progress_loss)?;
        if let Some(dl) = defib_pred_loss { total_loss = total_loss.add(&dl.affine(0.2, 0.0)?)?; }

        window_loss = Some(match window_loss.take() {
            None => total_loss,
            Some(w) => w.add(&total_loss)?,
        });
        steps_in_window += 1;

        let distance_to_horizon = (movement - thresh).abs() + 1e-4;
        let choptuik_gain = if CRITICALITY_SEEKING { (CRITICAL_D0 / distance_to_horizon).powf(CHOPTUIK_EXPONENT).clamp(0.3, 3.0) } else { distance_to_horizon.powf(CHOPTUIK_EXPONENT) };

        let alarm = (movement < thresh || mimic_drift_n > 0.6) && burst_ticks == 0 && refractory == 0;
        if alarm { alarm_streak += 1; } else { alarm_streak = 0; }

        let mut fired_now = false;
        if burst_ticks == 0 && refractory == 0 && alarm_streak >= DEFIB_FIRE_DEBOUNCE {
            let span = (DEFIB_REFRACTORY_MAX - DEFIB_REFRACTORY_BASE) as f32;
            let grow = 1.0 - (-DEFIB_RATE_SENSITIVITY * fire_rate_ema).exp();
            refractory = DEFIB_REFRACTORY_BASE + (span * grow) as usize;
            burst_ticks = DEFIB_BURST_TICKS;
            burst_energy = n_scale;
            alarm_streak = 0;
            fired_now = true;
            bursts_fired += 1;
            // Deliberately do NOT reset stagnation_ticks here. Sustained defibrillation is
            // itself a stagnation signal (the system keeps flatlining), so zeroing the clock
            // on every shock starved curiosity/neurogenesis and pinned depth at L01. We let
            // the escape drive (curiosity + shear momentum) keep building instead; only a
            // genuine archetype change (novelty) resets it.
            println!("[DEFIB FIRE #{}] chunk {} · {} · energy:{:.2} · gain:{:.3} · refractory:{} · rate:{:.3}", bursts_fired, step, if movement < thresh { "flatline" } else { "mimic-drift" }, burst_energy, choptuik_gain, refractory, fire_rate_ema);
        }

        fire_rate_ema = (DEFIB_RATE_DECAY * fire_rate_ema + (1.0 - DEFIB_RATE_DECAY) * if fired_now { 1.0 } else { 0.0 }).clamp(0.0, 1.0);
        if burst_ticks == 0 && refractory > 0 { refractory -= 1; }

        let curiosity_lr_gain = 1.0 + (curiosity_factor * 1.5) as f64;
        let phi_gate = 1.0 / (1.0 + phi);
        let burst_env = (burst_ticks as f32 / DEFIB_BURST_TICKS as f32).sqrt();
        latest_lr_gain = if burst_ticks > 0 {
            ((1.0 + (lr_mult - 1.0) * burst_env) * phi_gate * choptuik_gain) as f64
        } else {
            (phi_gate * choptuik_gain) as f64
        };
        latest_lr_gain *= curiosity_lr_gain;

        if steps_in_window >= bptt_window || step == total_chunks - 1 {
            if let Some(w) = window_loss.take() {
                let scaled = w.affine(1.0 / steps_in_window as f64, 0.0)?;
                let bounded_loss = scaled.clamp(0.0, 10.0)?;
                
                if let Ok(loss_val) = bounded_loss.to_scalar::<f32>() {
                    if loss_val.is_finite() && latest_lr_gain.is_finite() {
                        optimizer.set_learning_rate(target_lr * latest_lr_gain);
                        if optimizer.backward_step(&bounded_loss).is_ok() {
                            // clamp_weights clones+clamps every parameter (~10M floats, ~40MB)
                            // each call — by far the largest per-step allocation in the loop,
                            // dwarfing the shear/feature tensors. With a ±100 bound it almost
                            // never actually clamps anything, so run it periodically as a pure
                            // blow-up guardrail; the non-finite-loss check and NaN bio-reset
                            // catch true runaways between clamps.
                            weight_clamp_tick += 1;
                            if weight_clamp_tick % 32 == 0 { let _ = clamp_weights(&varmap, 100.0); }
                        }
                    } else {
                        println!("! WARNING: Non-finite loss step detected. Dropping mathematical BPTT window.");
                    }
                }
            }
            steps_in_window = 0;
            micro_tape = next_micro.detach()?; macro_tape = next_macro.detach()?; hidden_mem = next_hidden.detach()?;
            prev_pred = match prev_pred.take() { Some(p) => Some(p.detach()?), None => None };
            phase_c_l = nc_l.detach()?; phase_c_r = nc_r.detach()?; phase_m_l = nm_l.detach()?; phase_m_r = nm_r.detach()?;
            
            let rad_probability = RADIATE_PROB + curiosity_factor * 0.15;
            if rand::thread_rng().gen::<f32>() < rad_probability { 
                micro_tape = levy_radiate(&micro_tape, rad_amp * (1.0 + curiosity_factor * 0.5))?; 
            }
        } else {
            micro_tape = next_micro; macro_tape = next_macro; hidden_mem = next_hidden;
            phase_c_l = nc_l; phase_c_r = nc_r; phase_m_l = nm_l; phase_m_r = nm_r;
        }

        if burst_ticks > 0 {
            let noise = Tensor::randn(0.0f32, 1.0f32, micro_tape.shape(), micro_tape.device())?.affine((burst_energy * burst_env * choptuik_gain) as f64, 0.0)?;
            micro_tape = micro_tape.add(&noise)?.clamp(-1.0f32, 1.0f32)?;
            burst_ticks -= 1;
        }

        // --- PREDICTIVE MACRO SHEAR (disruption avoidance) ---
        // Graded, structured perturbation injected into the SLOW field every step.
        // Amplitude ramps with avoidance urgency (gentle dither when safe, hard shear
        // when quench is imminent). Soft-saturation keeps cells off the rail. This is
        // the direct macro-tape injection the reactive defibrillator never performed.
        shear_phase += SHEAR_PHASE_VEL;
        // Stagnation-driven momentum: a uniform, stuck field (curiosity_factor -> 1) gets
        // escalating structured shear on TOP of the q-estimate's drive, so it has enough kick
        // to skate over the rail boundary rather than settling into a flat ▒▒▒▒ basin.
        let shear_amp_eff = (ds.shear_amp + curiosity_factor * SHEAR_AMP_MAX * 0.5).min(SHEAR_AMP_MAX * 1.5);
        let shear = shear_gen.generate(shear_amp_eff, shear_phase, &device)?;
        // tanh hard-bounds the field; the old fixed *0.95 contraction is replaced by an
        // amplitude homeostat whose equilibrium is a healthy abs-mean (not ~0.04). When the
        // field collapses it scales up (reviving macro modulation AND unfreezing the micro
        // metab gate); when it approaches the rail it scales down. The shear seeds spatial
        // structure that the homeostat then sustains instead of erasing it every step.
        macro_tape = macro_tape.add(&shear)?.tanh()?;
        let macro_abs = macro_tape.abs()?.mean_all()?.to_scalar::<f32>().unwrap_or(MACRO_AMP_SETPOINT);
        let amp_gain = (1.0 + MACRO_AMP_RATE * (MACRO_AMP_SETPOINT - macro_abs)).clamp(0.5, 1.5);
        macro_tape = macro_tape.affine(amp_gain as f64, 0.0)?;
        if ds.recovered {
            stagnation_ticks = 0; // successful avoidance -> restart the patience clock
            println!("  \u{27FF} DISRUPTION AVERTED · q recovered to {:.2}x baseline · shear backing off", ds.q_norm);
        }

        // --------------------------------------------------
        // RENDER DSP PATH
        // --------------------------------------------------
        let audio_normalized_vec = audio_normalized.to_vec2::<f32>()?;
        let mut audio_l = audio_normalized_vec[0].clone();
        let mut audio_r = audio_normalized_vec[1].clone();

        qnm_resonators.process(&mut audio_l, &mut audio_r, phi);
        let echo_aperture = aperture.min(0.7);
        fractal_fdn_l.process(&mut audio_l, echo_aperture);
        fractal_fdn_r.process(&mut audio_r, echo_aperture);

        for i in 0..CHUNK_SIZE {
            let sample_l = (audio_l[i] * 32767.0).clamp(-32768.0, 32767.0) as i16;
            let sample_r = (audio_r[i] * 32767.0).clamp(-32768.0, 32767.0) as i16;
            audio_frames.push(sample_l); audio_frames.push(sample_r);
        }

        if step % 10 == 0 {
            let topology_state = macro_tape.mean(1)?.reshape((TAPE_LEN,))?.to_vec1::<f32>()?;
            topology_history.push(topology_state);
            uncertainty_trace.push(serde_json::json!({"step": step, "spectral": uncertainty.spectral, "movement": uncertainty.movement, "compositional": uncertainty.compositional, "aperture": aperture, "phi": phi, "synergy": synergy_val, "empowerment": empowerment_val, "q_titan": ds.q, "q_norm": ds.q_norm, "shear_urgency": ds.urgency, "lock": ds.lock}));
        }
        if step % 50 == 0 {
            let rolling_sec = profiling_lap.elapsed().as_secs_f32();
            let rolling_sps = if step > 0 && rolling_sec > 1e-4 { 50.0 / rolling_sec } else { 0.0 };
            profiling_lap = std::time::Instant::now();

            println!("Chunk {}/{} [SPS: {:.2}] | Move: {:.3} | Mimic: {:.3} {} | Phase: {} | L{:02} rad:{:.2} | Phi: {:.2} | LRx: {:.2} | q:{:.2} shear:{:.2} lock:{:.2}", step, total_chunks, rolling_sps, movement, mimic_drift_n, trend, phase, model.depth(), rad_amp, phi, latest_lr_gain, ds.q_norm, ds.shear_amp, ds.lock);
            println!("  field H:{:.2}b · {} · synergy:{:.2} empower:{:.2} | metabolic energy: {:.2} | curiosity stagnation: {}", field_entropy, arch_summary, synergy_val, empowerment_val, energy_state, stagnation_ticks);
            let cols = 64.min(TAPE_LEN);
            if let Ok(v) = micro_tape.mean(1).and_then(|m| m.reshape((TAPE_LEN,))) {
                if let Ok(full) = v.to_vec1::<f32>() {
                    let pooled: Vec<f32> = full.iter().take(cols).copied().collect();
                    let (val_lane, grad_lane) = quantile_dual_tape(&pooled);
                    println!("  v {}", val_lane); println!("  ∂ {}", grad_lane);
                }
            }
            let comment = semantic.commentary(phase, None, rad_amp, model.depth());
            if !comment.is_empty() { println!("  · {}", comment); }
        }
        if step > 0 && step % 500 == 0 {
            if varmap.save(&model_path).is_ok() {
                let _ = std::fs::write(&morph_path, serde_json::json!({"active_depth": model.depth(), "rad_amp": rad_amp}).to_string());
                println!("--> Checkpoint saved at step {} (L{:02}, rad {:.3})", step, model.depth(), rad_amp);
            }
        }
    }

    let total_elapsed = timer_start.elapsed().as_secs_f32();
    let overall_sps = total_chunks as f32 / total_elapsed;
    println!("\n=== PERFORMANCE REPORT ===");
    println!("Total simulation elapsed: {:.2}s", total_elapsed);
    println!("Overall performance speed: {:.2} steps/sec", overall_sps);

    let avg_phi = uncertainty_trace.iter().map(|t| t["phi"].as_f64().unwrap_or(0.0)).sum::<f64>() / uncertainty_trace.len() as f64;
    let avg_aperture = uncertainty_trace.iter().map(|t| t["aperture"].as_f64().unwrap_or(0.0)).sum::<f64>() / uncertainty_trace.len() as f64;
    let avg_synergy = uncertainty_trace.iter().map(|t| t["synergy"].as_f64().unwrap_or(0.0)).sum::<f64>() / uncertainty_trace.len() as f64;
    let avg_field_h = if field_entropy_n > 0 { field_entropy_sum / field_entropy_n as f64 } else { 0.0 };
    let dom_phase = semantic.dominant_phase();
    let dom_archetype = semantic.dominant_archetype();
    let final_depth = model.depth();

    let prompt = format!(
        "Style: {}, {}, {}, {}. Texture: {}. Field: {} regime · {} archetype · depth L{:02}. [Informational Phi: {:.2}, Aperture: {:.2}, Synergy: {:.2}, Field-Entropy: {:.2}b, Rad: {:.2}, Metabolic-Energy: {:.2}, Stagnation-Ticks: {}]",
        // phi is normalized to ~[0,1.5] and synergy is tanh-bounded to (-1,1), so the old
        // >5.0/>4.0/>1.5 thresholds were dead code (always "Chaotic"/"Grit"). Rescaled to the
        // real ranges.
        if avg_phi > 0.9 { "Hyper-Resonant" } else { "Chaotic" }, if avg_aperture > 0.5 { "Evolving" } else { "Stable" }, if total_complexity > 500.0 { "Dense" } else { "Minimal" }, "Information-Theoretic Glitch",
        if avg_synergy > 0.6 { "Crystalline-Autonomous" } else if avg_phi > 0.6 { "Organic" } else { "Grit" }, dom_phase, dom_archetype, final_depth, avg_phi, avg_aperture, avg_synergy, avg_field_h, rad_amp, energy_state, stagnation_ticks
    );
    println!("\n=== GENERATIVE PRIMING PROMPT ===\n{}", prompt);
    std::fs::write(format!("{}/suno_priming_prompt.txt", base_dir), &prompt)?;

    let mut topo_writer = csv::Writer::from_path(format!("{}/ca_topology_rust.csv", base_dir))?;
    for row in topology_history { topo_writer.write_record(row.iter().map(|f| f.to_string()))?; }
    topo_writer.flush()?;

    let mut unc_writer = csv::Writer::from_path(format!("{}/uncertainty_trace_rust.csv", base_dir))?;
    unc_writer.write_record(&["step", "spectral", "movement", "compositional", "aperture", "synergy", "empowerment", "phi", "q_titan", "q_norm", "shear_urgency", "lock"])?;
    for trace in uncertainty_trace {
        unc_writer.write_record(&[trace["step"].to_string(), trace["spectral"].to_string(), trace["movement"].to_string(), trace["compositional"].to_string(), trace["aperture"].to_string(), trace["synergy"].to_string(), trace["empowerment"].to_string(), trace["phi"].to_string(), trace["q_titan"].to_string(), trace["q_norm"].to_string(), trace["shear_urgency"].to_string(), trace["lock"].to_string()])?;
    }
    unc_writer.flush()?;
    println!("Topology and uncertainty trace saved to {}.", base_dir);

    let spec = hound::WavSpec { channels: 2, sample_rate: SAMPLE_RATE, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut writer = hound::WavWriter::create(format!("{}/rust_ecosystem_out.wav", base_dir), spec)?;
    for sample in audio_frames { writer.write_sample(sample)?; }
    writer.finalize()?;
    println!("Audio saved to {}/rust_ecosystem_out.wav", base_dir);

    varmap.save(&model_path).map_err(anyhow::Error::msg)?;
    let _ = std::fs::write(&morph_path, serde_json::json!({"active_depth": model.depth(), "rad_amp": rad_amp}).to_string());
    let metadata = std::fs::metadata(&model_path)?;
    println!("Model saved to {} (L{:02}, rad {:.3}). Size: {:.2} MB", model_path, model.depth(), rad_amp, metadata.len() as f32 / 1_048_576.0);
    
    if bursts_fired > 0 {
        println!("Defibrillator: {} bursts over {} chunks (avg 1 every {:.0} chunks).", bursts_fired, total_chunks, total_chunks as f32 / bursts_fired as f32);
    } else {
        println!("Defibrillator: 0 bursts (system never sustained a flatline).");
    }
    Ok(())
}
