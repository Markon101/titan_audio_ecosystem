#![recursion_limit = "512"]

// =====================================================================
// TITAN AUDIO ECOSYSTEM — RUST EDITION v8 ("CONFINEMENT REACTOR")
// =====================================================================
// BUILD (Termux / Snapdragon 8 Elite):
//   RUSTFLAGS="-C target-cpu=native" cargo build --release
//
// DESIGN RECORD (v8) — recursive reactor with staged confinement control:
//
// 1. RECURSIVE RENORMALIZATION OBSERVER. The 4x4 regional field is
//    repeatedly coarse-grained into 2x2 and global descriptions. Entropy,
//    temporal change, spatial correlation, scale invariance, disagreement,
//    and active-scale count become both a host-side control signal and a
//    differentiable anti-collapse loss for the neural CA.
//
// 2. MOVING CRITICAL MANIFOLD. "Criticality" is no longer one sigma target.
//    The controller tracks an 8-dimensional surface: branching, correlation
//    length, temporal memory, perturbation susceptibility, entropy rate,
//    prediction horizon, regional diversity, and scale invariance. Its target
//    drifts slowly so the organism orbits the edge rather than freezing on a
//    single optimum.
//
// 3. EXPLICIT REACTOR STATE. Final post-DSP audio, ecological health, the
//    critical vector, RG features, state-space novelty, recurrence, and
//    recoverability are compressed into 48 bounded values. The planner never
//    predicts samples and never clones the 64x64x64 neural CA.
//
// 4. ONLINE WORLD-MODEL ENSEMBLE. Three tiny 58->64->48 MLPs learn
//    (reactor_state, action) -> next_reactor_state from a prioritized replay
//    buffer. Ensemble disagreement supplies epistemic uncertainty; shadow
//    rollouts estimate a cheap Lyapunov-like divergence and prediction horizon.
//
// 5. EVENT-TRIGGERED BEAM PLANNER. Short model-based rollouts run only at
//    meaningful transitions. Actions are scored by advantage over a HOLD
//    rollout of the same horizon, viable complexity, critical proximity,
//    information gain, collapse risk, and option value: the diversity of
//    healthy states still reachable afterward. cool/balanced/max profiles
//    trade depth for heat and latency.
//
// 6. MODEL-BASED / MODEL-FREE ARBITRATION. The existing learned bandit handles
//    fast reactions and unfamiliar regimes; the compact planner handles
//    medium-horizon steering. Raw predictor confidence is gated by ecological
//    health and ensemble readiness, preventing a frozen but easy-to-predict
//    world from taking control.
//
// 7. ATTRACTORS, NOT JUST CLIPS. Long-term memory stores reactor-state
//    attractors with entry actions, synthesis controls, recoverability, and
//    learned exit values. Recall has cooldown and can function as a bridge
//    between regimes instead of becoming a single repeating motif.
//
// 8. ACTIVE SYSTEM IDENTIFICATION. Small, infrequent WIDEN/RESONATE/
//    INHARMONIC/TURBULENCE probes compare the real response with a predicted
//    HOLD trajectory. Their response estimates susceptibility and makes
//    self-measurement part of the composition.
//
// 9. PHONE-FIRST COMPUTE SCHEDULING. Heavy neural CA/audio work remains in
//    Candle; planning and world-model learning use compact host arrays.
//    Planning, training, motif maintenance, RG analysis, and checkpoints run
//    at different cadences. Audio streams to disk, and Ctrl-C finishes the
//    active chunk before finalizing WAVs and saving the organism.
//
// 10. CONTINUITY. Versioned, checksummed checkpoints preserve CA tensors, DSP
//    delay memory, RNG, controller, RG state, transition ensemble/replay,
//    state-space occupancy, attractors, and probes. V6 and V5 worlds migrate
//    into V8 without overwriting their original files. AdamW moment buffers
//    remain process-local and are intentionally not serialized.

use anyhow::Result;
use candle_core::{DType, Device, Result as CResult, Tensor, D};
use candle_nn::{AdamW, Conv2dConfig, Linear, Module, Optimizer, VarBuilder as VBV, VarMap};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use rustfft::{num_complex::Complex, FftPlanner};
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;
use std::collections::{BTreeMap, VecDeque};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::sync::{
    atomic::{AtomicBool, Ordering as AtomicOrdering},
    Arc,
};

type RuntimeRng = ChaCha8Rng;

struct WakeLockGuard;
impl WakeLockGuard {
    fn acquire() -> Self {
        let _ = std::process::Command::new("termux-wake-lock").status();
        Self
    }
}
impl Drop for WakeLockGuard {
    fn drop(&mut self) {
        let _ = std::process::Command::new("termux-wake-unlock").status();
    }
}

// --- ECOSYSTEM CONSTANTS ---
const SAMPLE_RATE: u32 = 48000;
const DURATION_SECONDS: f32 = 240.0;
const CHUNK_SIZE: usize = 4096;

// 2D field: 64 channels on a 64x64 torus = 262,144 scalar cells.  This is
// deliberately larger than v3's 96x512 ring: the S25 Ultra can sustain the
// added spatial vocabulary, while --threads and --bptt remain available as
// thermal/RAM controls for other devices.
const GRID_H: usize = 64;
const GRID_W: usize = 64;
const CA_CHANNELS: usize = 96;
const CA_HIDDEN: usize = 128;
const CA_UPDATE_PROB: f32 = 0.71;
// Reusable stochastic fields live on the compute device. 64 slots consume
// about 288 MiB at the current field size (uniform + two normal pools), which
// is deliberately small relative to the 11 GiB desktop target while avoiding
// several large host allocations and H2D copies on every chunk.
const GPU_RANDOM_POOL_SLOTS: usize = 64;

const MEMORY_DIM: usize = 768;
const BPTT_WINDOW: usize = 8;
const SPEC_BINS: usize = 128;
const FINE_SPEC_BINS: usize = 48;
const KAN_BASIS_FUNCTIONS: usize = 128;

// Scan-synth (the 2D field made directly audible)
const SCAN_PARTIALS: usize = 16; // 4x4 regional agents -> partial amplitudes
const SCAN_GAIN: f64 = 0.42;
const REGION_ROWS: usize = 4;
const REGION_COLS: usize = 4;
const REGION_COUNT: usize = REGION_ROWS * REGION_COLS;
const REGION_H: usize = GRID_H / REGION_ROWS;
const REGION_W: usize = GRID_W / REGION_COLS;

// Recursive self-model / hybrid control
const OBS_DIM: usize = 12;
const ACTION_COUNT: usize = 10;
const WORLD_VERSION: u32 = 8;
const WORLD_MAGIC: [u8; 8] = *b"TITANW8\0";
const LEGACY_WORLD_MAGIC_V7: [u8; 8] = *b"TITANW7\0";
const LEGACY_WORLD_MAGIC_V6: [u8; 8] = *b"TITANW6\0";
const LEGACY_WORLD_MAGIC_V5: [u8; 8] = *b"TITANW5\0";
const WORLD_SAVE_EVERY: usize = 1_000;
const MOTIF_SLOTS: usize = 128;
const MOTIF_EVERY: usize = 16;
const MOTIF_MIN_AGE: u64 = 128;
const STAGNATION_WARMUP: u64 = 48;
const STAGNATION_ESCAPE_AFTER: u32 = 96;
const STAGNATION_HARD_AFTER: u32 = 224;
const SUBCRITICAL_SIGMA_TRIGGER: f32 = 0.68;
const LOW_MOTION_TRIGGER: f32 = 0.009;
const ACTION_USAGE_DECAY: f32 = 0.965;
const MODAL_MODES: usize = 8;
const NOISE_BANDS: usize = 4;

// Tokamak-inspired staged confinement recovery. Each phase changes actuator
// class instead of indefinitely increasing the same stochastic heat.
const RECOVERY_EDGE_CHUNKS: u64 = 160;
const RECOVERY_ROTATION_CHUNKS: u64 = 160;
const RECOVERY_GUIDED_CHUNKS: u64 = 128;
const RECOVERY_COOLDOWN_CHUNKS: u64 = 192;
const CONFINEMENT_CALIBRATION_SAMPLES: u64 = 128;

// Episodic memory
const EPI_SLOTS: usize = 16; // snapshots retained (~16*64 steps ≈ 87 s span)
const EPI_SNAP_EVERY: usize = 64;
const EPI_DIM: usize = 64; // attention readout width

// Novelty pressure (anti-self-similarity)
const NOVELTY_SLOTS: usize = 24;
const NOVELTY_EVERY: usize = 4;
const NOVELTY_MARGIN: f64 = 0.35;
const NOVELTY_W: f64 = 0.25;

// Min-of-K target sampling
const TARGET_K: usize = 3;

// Criticality / plasticity
const CHOPTUIK_EXPONENT: f32 = 0.3747;
const CRITICAL_D0: f32 = 0.02;

// --- POTENTIAL CONTROLLER V(s) ---
// Bowls (quadratic wells at setpoints), barriers (1/(1.02-a) rail walls),
// a Gaussian ridge at movement=0, and a criticality bowl at sigma=1.
const POT_MICRO_SET: f32 = 0.50;
const POT_MACRO_SET: f32 = 0.40;
const POT_COUPLE_SET: f32 = 0.50;
const POT_ENERGY_SET: f32 = 0.72;
const POT_K_AMP: f32 = 1.2;
const POT_K_RHO: f32 = 1.0;
const POT_K_E: f32 = 0.4;
const POT_K_SIG: f32 = 0.75;
const POT_BARRIER: f32 = 0.004; // b/(1.02-a): negligible mid-range, ~10 at the rail
const POT_RIDGE_G: f32 = 0.6; // height of the movement=0 ridge
const POT_RIDGE_W: f32 = 0.05; // width of the ridge
const POT_STEP: f32 = 0.35; // eta: gain = clamp(1 - eta * dV/da)
const POT_GAIN_LO: f32 = 0.6;
const POT_GAIN_HI: f32 = 1.4;
// Temperature: T = smooth(stuck + subcritical + stagnation), in [0,1].
const TEMP_STUCK_SPEED: f32 = 0.010; // state-speed scale below which "flat" ⇒ heat
const TEMP_EXCESS_SCALE: f32 = 0.5; // V-above-floor normalization
const TEMP_SMOOTH: f32 = 0.10; // EMA rate on T
const TEMP_HOT: f32 = 0.57; // "hot episode" edge for the anneal printout
const TEMP_COOL: f32 = 0.25;
const SHEAR_AMP_MIN: f32 = 0.03; // structured macro shear at T=0 (anti-weld floor)
const SHEAR_AMP_MAX: f32 = 0.45; // at T=1
const MICRO_KICK_MAX: f32 = 0.11; // white micro kick amplitude at T=1
const LR_HEAT_MAX: f64 = 1.5; // lr multiplier reaches 1+this at T=1
const ARB_TAU_MAX: f32 = 2.0; // arbiter softmax temp reaches 1+this at T=1
const SHEAR_OCTAVES: usize = 7;
const SHEAR_PHASE_VEL: f32 = 0.314;

// Coupling band (kept from v3: correlated, not identical)
const SYNERGY_TARGET: f32 = 0.51;
const SYNERGY_BAND_W: f32 = 0.50;

const BASE_FREQ_L: f32 = 48.0;
const BASE_FREQ_R: f32 = 69.0;
const FREQ_GLIDE_SPEED: f32 = 0.07131;
const BASE_LR: f64 = 1.5e-3;
const RESONANT_AUTONOMY: f32 = 0.31;
const GRAD_NORM_MAX: f32 = 5.0; // global-norm ceiling; applied as an LR scale (see note at use)
const TWO_PI: f32 = 2.0 * std::f32::consts::PI;

const ENERGY_HOMEO_RATE: f32 = 0.025; // energy bowl strength applied directly to the scalar

const LARGE_D_DIM: usize = 512;
const FDN_DELAY_LINES: usize = 4;
const FDN_DELAYS: [usize; 4] = [149, 263, 431, 701];

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

const VAL_SYMS: [&str; 8] = [" ", "·", "░", "▒", "▓", "█", "▪", "■"];
const GRAD_SYMS: [&str; 8] = [" ", "˙", "·", "∘", "o", "O", "◎", "●"];
const ARCHETYPES: [&str; 8] = [
    "VOID",
    "LATENT",
    "DRIFT",
    "NEXUS",
    "PULSE",
    "SIGNAL",
    "AXIOM",
    "SINGULARITY",
];
const ARCH_BOUNDS: [f32; 9] = [0.0, 0.13, 0.26, 0.40, 0.52, 0.65, 0.78, 0.90, 1.001];
const PHASE_MAP: [(f32, &str); 7] = [
    (0.20, "MASTERY"),
    (0.27, "COHERENT"),
    (0.35, "CONVERGING"),
    (0.45, "LEARNING"),
    (0.55, "TURBULENT"),
    (0.70, "CHAOTIC"),
    (f32::MAX, "PRIMORDIAL"),
];

// --- DETERMINISTIC RANDOM TENSOR HELPERS ---
// All runtime randomness flows through the caller's RuntimeRng. Tensors are built
// host-side via from_vec, so candle's (unseedable-on-CPU) internal RNG is
// never consulted after init. Box-Muller keeps us off extra deps.
fn rng_normal(rng: &mut RuntimeRng) -> f32 {
    let u1: f32 = rng.gen_range(1e-7f32..1.0);
    let u2: f32 = rng.gen_range(0.0f32..1.0);
    (-2.0 * u1.ln()).sqrt() * (TWO_PI * u2).cos()
}
fn randn_t(rng: &mut RuntimeRng, shape: &[usize], std: f32, device: &Device) -> CResult<Tensor> {
    let n: usize = shape.iter().product();
    let v: Vec<f32> = (0..n).map(|_| rng_normal(rng) * std).collect();
    Tensor::from_vec(v, shape, device)
}
struct DeviceRandomPool {
    uniform: Tensor,
    normal_a: Tensor,
    normal_b: Tensor,
}

impl DeviceRandomPool {
    fn new(seed: u64, device: &Device) -> CResult<Self> {
        let field_len = CA_CHANNELS * GRID_H * GRID_W;
        let total = GPU_RANDOM_POOL_SLOTS * field_len;
        // Derive an independent deterministic stream so pool construction does
        // not advance the checkpointed runtime RNG. Indexing by global step
        // then gives identical fields across stop/resume boundaries.
        let seeds: Vec<u64> = (0..GPU_RANDOM_POOL_SLOTS)
            .map(|slot| seed ^ 0xD1B5_4A32_D192_ED03u64.wrapping_mul(slot as u64 + 1))
            .collect();
        let mut uniform = vec![0.0f32; total];
        let mut normal_a = vec![0.0f32; total];
        let mut normal_b = vec![0.0f32; total];
        uniform
            .par_chunks_mut(field_len)
            .zip(normal_a.par_chunks_mut(field_len))
            .zip(normal_b.par_chunks_mut(field_len))
            .zip(seeds.par_iter())
            .for_each(|(((u_slot, a_slot), b_slot), &slot_seed)| {
                let mut slot_rng = RuntimeRng::seed_from_u64(slot_seed);
                for ((u, a), b) in u_slot
                    .iter_mut()
                    .zip(a_slot.iter_mut())
                    .zip(b_slot.iter_mut())
                {
                    *u = slot_rng.gen_range(0.0f32..1.0);
                    *a = rng_normal(&mut slot_rng);
                    *b = rng_normal(&mut slot_rng);
                }
            });
        let shape = (GPU_RANDOM_POOL_SLOTS, CA_CHANNELS, GRID_H, GRID_W);
        Ok(Self {
            uniform: Tensor::from_vec(uniform, shape, device)?,
            normal_a: Tensor::from_vec(normal_a, shape, device)?,
            normal_b: Tensor::from_vec(normal_b, shape, device)?,
        })
    }

    fn slot(tensor: &Tensor, index: usize) -> CResult<Tensor> {
        tensor.narrow(0, index % GPU_RANDOM_POOL_SLOTS, 1)
    }

    fn keep_mask(&self, index: usize) -> CResult<Tensor> {
        // relu/clamp implements (u >= 1-p) without a host-generated mask.
        Self::slot(&self.uniform, index)?
            .affine(1.0, -(1.0 - CA_UPDATE_PROB) as f64)?
            .affine(10000.0, 0.0)?
            .clamp(0.0f32, 1.0f32)
    }

    fn normal(&self, index: usize) -> CResult<Tensor> {
        Self::slot(&self.normal_a, index)
    }

    fn cauchy(&self, index: usize) -> CResult<Tensor> {
        let n1 = Self::slot(&self.normal_a, index)?;
        let n2 = Self::slot(&self.normal_b, index)?
            .abs()?
            .affine(1.0, 1e-3)?;
        n1.broadcast_div(&n2)?.clamp(-CAUCHY_CLAMP, CAUCHY_CLAMP)
    }

    fn sparse_mask(&self, index: usize) -> CResult<Tensor> {
        Self::slot(&self.uniform, index)?
            .affine(1.0, -RADIATE_SPARSITY as f64)?
            .relu()?
            .affine(10000.0, 0.0)?
            .clamp(0.0f32, 1.0f32)
    }
}

// =====================================================================
// RECURSIVE CONTROL PLANE
// =====================================================================
// The neural CA remains the differentiable organism.  The structures below
// are intentionally small host-side models: they plan over compact
// observations instead of cloning the 64x64x64 Candle graph for every action.

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[repr(u8)]
enum ControlAction {
    Hold = 0,
    Crystallize = 1,
    Explore = 2,
    Inharmonic = 3,
    Harmonic = 4,
    Resonate = 5,
    Turbulence = 6,
    Widen = 7,
    Contract = 8,
    Recall = 9,
}
impl ControlAction {
    const ALL: [Self; ACTION_COUNT] = [
        Self::Hold,
        Self::Crystallize,
        Self::Explore,
        Self::Inharmonic,
        Self::Harmonic,
        Self::Resonate,
        Self::Turbulence,
        Self::Widen,
        Self::Contract,
        Self::Recall,
    ];
    fn index(self) -> usize {
        self as usize
    }
    fn from_index(i: usize) -> Self {
        Self::ALL[i.min(ACTION_COUNT - 1)]
    }
    fn label(self) -> &'static str {
        match self {
            Self::Hold => "HOLD",
            Self::Crystallize => "CRYSTALLIZE",
            Self::Explore => "EXPLORE",
            Self::Inharmonic => "INHARMONIC",
            Self::Harmonic => "HARMONIC",
            Self::Resonate => "RESONATE",
            Self::Turbulence => "TURBULENCE",
            Self::Widen => "WIDEN",
            Self::Contract => "CONTRACT",
            Self::Recall => "RECALL",
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct SynthesisControl {
    shear_mult: f32,
    kick_mult: f32,
    inharmonicity: f32,
    spectral_tilt: f32,
    resonator_drive: f32,
    noise_level: f32,
    echo_delta: f32,
    width_mult: f32,
    recall_mix: f32,
}
impl Default for SynthesisControl {
    fn default() -> Self {
        Self {
            shear_mult: 1.0,
            kick_mult: 1.0,
            inharmonicity: 0.20,
            spectral_tilt: 0.0,
            resonator_drive: 0.35,
            noise_level: 0.05,
            echo_delta: 0.0,
            width_mult: 1.0,
            recall_mix: 0.0,
        }
    }
}
impl SynthesisControl {
    fn for_action(action: ControlAction) -> Self {
        let mut c = Self::default();
        match action {
            ControlAction::Hold => {}
            ControlAction::Crystallize => {
                // Crystallization now organizes without switching off the
                // very fluctuations needed to leave the attractor later.
                c.shear_mult = 0.86;
                c.kick_mult = 0.78;
                c.inharmonicity = 0.05;
                c.spectral_tilt = -0.10;
                c.echo_delta = -0.07;
            }
            ControlAction::Explore => {
                c.shear_mult = 1.42;
                c.kick_mult = 1.55;
                c.inharmonicity = 0.48;
                c.noise_level = 0.11;
                c.spectral_tilt = 0.20;
            }
            ControlAction::Inharmonic => {
                c.inharmonicity = 0.88;
                c.resonator_drive = 0.55;
                c.spectral_tilt = 0.12;
            }
            ControlAction::Harmonic => {
                c.inharmonicity = 0.0;
                c.noise_level = 0.025;
                c.resonator_drive = 0.42;
            }
            ControlAction::Resonate => {
                c.resonator_drive = 0.95;
                c.echo_delta = 0.10;
                c.inharmonicity = 0.34;
            }
            ControlAction::Turbulence => {
                c.shear_mult = 1.72;
                c.kick_mult = 1.82;
                c.noise_level = 0.18;
                c.inharmonicity = 0.66;
                c.spectral_tilt = 0.30;
            }
            ControlAction::Widen => {
                c.width_mult = 1.45;
                c.echo_delta = 0.08;
                c.inharmonicity = 0.32;
            }
            ControlAction::Contract => {
                c.width_mult = 0.62;
                c.echo_delta = -0.10;
                c.noise_level = 0.02;
                c.inharmonicity = 0.12;
            }
            ControlAction::Recall => {
                c.recall_mix = 0.75;
                c.resonator_drive = 0.62;
            }
        }
        c
    }
    fn blend(self, other: Self, amount: f32) -> Self {
        let a = amount.clamp(0.0, 1.0);
        let mix = |x: f32, y: f32| x + (y - x) * a;
        Self {
            shear_mult: mix(self.shear_mult, other.shear_mult),
            kick_mult: mix(self.kick_mult, other.kick_mult),
            inharmonicity: mix(self.inharmonicity, other.inharmonicity),
            spectral_tilt: mix(self.spectral_tilt, other.spectral_tilt),
            resonator_drive: mix(self.resonator_drive, other.resonator_drive),
            noise_level: mix(self.noise_level, other.noise_level),
            echo_delta: mix(self.echo_delta, other.echo_delta),
            width_mult: mix(self.width_mult, other.width_mult),
            recall_mix: mix(self.recall_mix, other.recall_mix),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct AudioObservation {
    // entropy, flatness, centroid, flux, rms, crest, width, movement,
    // synergy, field entropy, predictive structure, criticality health
    values: [f32; OBS_DIM],
}
impl Default for AudioObservation {
    fn default() -> Self {
        Self {
            values: [0.0; OBS_DIM],
        }
    }
}
impl AudioObservation {
    fn distance(&self, other: &Self) -> f32 {
        (self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / OBS_DIM as f32)
            .sqrt()
    }
    fn structured_complexity(&self) -> f32 {
        let entropy = self.values[0];
        let flatness = self.values[1];
        let flux = self.values[3];
        let pi = self.values[10];
        let critical = self.values[11];
        let flatness_shape = (-((flatness - 0.22) / 0.24).powi(2)).exp();
        (0.28 * entropy + 0.22 * flatness_shape + 0.20 * flux + 0.20 * pi + 0.10 * critical)
            .clamp(0.0, 1.0)
    }
    fn reward_against(
        &self,
        prev: Option<&Self>,
        recurrence: f32,
        ecology: &AdaptiveDynamics,
        raw_model_confidence: f32,
        action_age: u32,
    ) -> f32 {
        // Reward is deliberately centered and contrastive. v5's mostly
        // positive score compressed every action into ~0.22, leaving the
        // bandit almost no evidence about which intervention helped.
        let base = self.structured_complexity();
        let novelty = prev.map(|p| self.distance(p)).unwrap_or(0.12);
        let novelty_score = (novelty / 0.10).clamp(0.0, 1.0);
        let rms_penalty = ((self.values[4] - 0.25).abs() / 0.45).clamp(0.0, 1.0);
        let crest_penalty = ((self.values[5] - 0.82) / 0.25).clamp(0.0, 1.0);
        // Reward the same finite critical band used by AdaptiveDynamics.
        // A monotonic sigma/0.75 term incorrectly gave full credit to
        // supercritical persistence and partial credit to welded states.
        let critical_health = (-((ecology.sigma_ema - 1.0) / 0.42).powi(2))
            .exp()
            .clamp(0.0, 1.0);
        let repetition_penalty = (action_age as f32 / 28.0).clamp(0.0, 1.0);
        let predictability_trap = raw_model_confidence * ecology.stagnation;
        let score = 0.58 * base
            + 0.16 * novelty_score
            + 0.10 * recurrence
            + 0.18 * ecology.activity_health
            + 0.10 * critical_health
            - 0.36 * ecology.stagnation
            - 0.16 * predictability_trap
            - 0.12 * repetition_penalty
            - 0.10 * rms_penalty
            - 0.08 * crest_penalty
            - 0.34;
        score.clamp(-1.0, 1.0)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct AdaptiveDynamics {
    samples: u64,
    movement_fast: f32,
    movement_slow: f32,
    movement_peak: f32,
    region_fast: f32,
    region_slow: f32,
    region_peak: f32,
    observation_delta_ema: f32,
    observation_delta_peak: f32,
    complexity_ema: f32,
    sigma_ema: f32,
    activity_health: f32,
    stagnation: f32,
    low_motion_run: u32,
    escape_cooldown: u32,
    action_use_ema: [f32; ACTION_COUNT],
    effective_model_weight: f32,
    reward_mean: f32,
    reward_variance: f32,
}
impl Default for AdaptiveDynamics {
    fn default() -> Self {
        Self {
            samples: 0,
            movement_fast: 0.01,
            movement_slow: 0.01,
            movement_peak: 0.012,
            region_fast: 0.01,
            region_slow: 0.01,
            region_peak: 0.012,
            observation_delta_ema: 0.04,
            observation_delta_peak: 0.06,
            complexity_ema: 0.45,
            sigma_ema: 0.75,
            activity_health: 0.65,
            stagnation: 0.0,
            low_motion_run: 0,
            escape_cooldown: 0,
            action_use_ema: [0.0; ACTION_COUNT],
            effective_model_weight: 0.15,
            reward_mean: 0.0,
            reward_variance: 0.05,
        }
    }
}
impl AdaptiveDynamics {
    fn observe(
        &mut self,
        movement: f32,
        region_change: f32,
        observation_delta: f32,
        complexity: f32,
        sigma: f32,
        raw_model_confidence: f32,
    ) {
        self.samples = self.samples.saturating_add(1);
        self.movement_fast += 0.12 * (movement - self.movement_fast);
        self.movement_slow += 0.012 * (movement - self.movement_slow);
        self.movement_peak = (self.movement_peak * 0.9985).max(movement);
        self.region_fast += 0.12 * (region_change - self.region_fast);
        self.region_slow += 0.012 * (region_change - self.region_slow);
        self.region_peak = (self.region_peak * 0.9985).max(region_change);
        self.observation_delta_ema += 0.10 * (observation_delta - self.observation_delta_ema);
        self.observation_delta_peak = (self.observation_delta_peak * 0.9985).max(observation_delta);
        self.complexity_ema += 0.04 * (complexity - self.complexity_ema);
        self.sigma_ema += 0.05 * (sigma - self.sigma_ema);

        // Mix absolute and self-relative health. The absolute terms detect a
        // truly welded CA; the relative terms let naturally quieter organisms
        // retain their own operating scale.
        let move_abs = (self.movement_fast / 0.006).clamp(0.0, 1.0);
        let move_rel = (self.movement_fast / (0.62 * self.movement_peak + 1e-6)).clamp(0.0, 1.0);
        let region_abs = (self.region_fast / 0.004).clamp(0.0, 1.0);
        let region_rel = (self.region_fast / (0.62 * self.region_peak + 1e-6)).clamp(0.0, 1.0);
        let delta_abs = (self.observation_delta_ema / 0.025).clamp(0.0, 1.0);
        let delta_rel = (self.observation_delta_ema / (0.58 * self.observation_delta_peak + 1e-6))
            .clamp(0.0, 1.0);
        let move_health = 0.52 * move_abs + 0.48 * move_rel;
        let region_health = 0.48 * region_abs + 0.52 * region_rel;
        let temporal_health = 0.50 * delta_abs + 0.50 * delta_rel;
        // Critical health is a band around sigma=1, not a monotonic reward.
        // The previous sigma/0.75 score treated a deeply subcritical but
        // self-consistent field as healthy and prevented the escape timer.
        let critical_health = (-((self.sigma_ema - 1.0) / 0.42).powi(2))
            .exp()
            .clamp(0.0, 1.0);
        let complexity_health = ((self.complexity_ema - 0.18) / 0.42).clamp(0.0, 1.0);
        let target_health = (0.30 * move_health
            + 0.24 * region_health
            + 0.20 * temporal_health
            + 0.14 * critical_health
            + 0.12 * complexity_health)
            .clamp(0.0, 1.0);
        self.activity_health += 0.07 * (target_health - self.activity_health);

        let easy_but_empty =
            raw_model_confidence * (1.0 - self.activity_health) * (1.0 - complexity_health);
        let stagnation_target =
            (1.0 - self.activity_health + 0.35 * easy_but_empty).clamp(0.0, 1.0);
        self.stagnation += 0.055 * (stagnation_target - self.stagnation);
        let subcritical_low_motion =
            self.sigma_ema < SUBCRITICAL_SIGMA_TRIGGER && self.movement_fast < LOW_MOTION_TRIGGER;
        if self.samples > STAGNATION_WARMUP && (self.stagnation > 0.56 || subcritical_low_motion) {
            self.low_motion_run = self.low_motion_run.saturating_add(1);
        } else {
            self.low_motion_run = self.low_motion_run.saturating_sub(2);
        }
        self.escape_cooldown = self.escape_cooldown.saturating_sub(1);

        // An accurate predictor of a nearly frozen, subcritical world should
        // not earn full control authority merely because it is easy to model.
        let critical_confidence_gate = 0.20 + 0.80 * critical_health;
        let confidence_gate = (0.12 + 0.88 * self.activity_health)
            * (1.0 - 0.38 * self.stagnation)
            * critical_confidence_gate;
        self.effective_model_weight = (raw_model_confidence * confidence_gate).clamp(0.03, 0.92);
    }

    fn escape_strength(&self) -> f32 {
        if self.low_motion_run <= STAGNATION_ESCAPE_AFTER {
            return 0.0;
        }
        let duration = ((self.low_motion_run - STAGNATION_ESCAPE_AFTER) as f32
            / (STAGNATION_HARD_AFTER - STAGNATION_ESCAPE_AFTER) as f32)
            .clamp(0.0, 1.0);
        let subcritical = ((SUBCRITICAL_SIGMA_TRIGGER - self.sigma_ema) / 0.40).clamp(0.0, 1.0);
        let motion_deficit =
            ((LOW_MOTION_TRIGGER - self.movement_fast) / LOW_MOTION_TRIGGER).clamp(0.0, 1.0);
        (0.35 * self.stagnation + 0.30 * duration + 0.25 * subcritical + 0.10 * motion_deficit)
            .clamp(0.0, 1.0)
    }

    fn record_action(&mut self, action: ControlAction) {
        for x in &mut self.action_use_ema {
            *x *= ACTION_USAGE_DECAY;
        }
        self.action_use_ema[action.index()] += 1.0 - ACTION_USAGE_DECAY;
    }

    fn action_usage_penalty(&self, action: ControlAction) -> f32 {
        (self.action_use_ema[action.index()] * 0.38).clamp(0.0, 0.30)
    }

    fn observe_reward(&mut self, reward: f32) {
        let delta = reward - self.reward_mean;
        self.reward_mean += 0.05 * delta;
        self.reward_variance += 0.05 * (delta * delta - self.reward_variance);
    }

    fn reward_std(&self) -> f32 {
        self.reward_variance.max(1e-6).sqrt()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct MetaModel {
    error_ema: f32,
    variance_ema: f32,
    calibration_error: f32,
    error_trend: f32,
    confidence: f32,
}
impl Default for MetaModel {
    fn default() -> Self {
        Self {
            error_ema: 0.5,
            variance_ema: 0.5,
            calibration_error: 0.5,
            error_trend: 0.0,
            confidence: 0.15,
        }
    }
}
impl MetaModel {
    fn update(&mut self, actual: &AudioObservation, mean: &[f32], log_var: &[f32]) {
        if mean.len() < OBS_DIM || log_var.len() < OBS_DIM {
            return;
        }
        let mut mse = 0.0f32;
        let mut predicted_var = 0.0f32;
        for i in 0..OBS_DIM {
            let d = actual.values[i] - mean[i];
            mse += d * d;
            predicted_var += log_var[i].clamp(-6.0, 2.0).exp();
        }
        mse /= OBS_DIM as f32;
        predicted_var /= OBS_DIM as f32;
        let old = self.error_ema;
        self.error_ema += 0.08 * (mse - self.error_ema);
        self.variance_ema += 0.08 * (predicted_var - self.variance_ema);
        self.calibration_error += 0.08 * ((mse - predicted_var).abs() - self.calibration_error);
        self.error_trend += 0.12 * ((self.error_ema - old) - self.error_trend);
        self.confidence = (-2.8 * self.error_ema - 1.4 * self.calibration_error)
            .exp()
            .clamp(0.02, 0.98);
    }
    fn surprise(&self) -> f32 {
        (self.error_ema.sqrt() * 2.5).clamp(0.0, 1.0)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ModelFreeBandit {
    q: [f32; ACTION_COUNT],
    visits: [u32; ACTION_COUNT],
    reward_ema: f32,
}
impl Default for ModelFreeBandit {
    fn default() -> Self {
        Self {
            q: [0.0; ACTION_COUNT],
            visits: [0; ACTION_COUNT],
            reward_ema: 0.0,
        }
    }
}
impl ModelFreeBandit {
    fn update(&mut self, action: ControlAction, reward: f32) {
        let i = action.index();
        self.visits[i] = self.visits[i].saturating_add(1);
        let baseline = self.reward_ema;
        self.reward_ema += 0.06 * (reward - self.reward_ema);
        let advantage = (reward - baseline).clamp(-1.0, 1.0);
        let alpha = (1.0 / (self.visits[i] as f32).sqrt()).clamp(0.04, 0.35);
        self.q[i] += alpha * (advantage - self.q[i]);
    }

    fn apply_recovery_tax(&mut self, strength: f32) {
        // Forced Explore/Turbulence actions are interventions, so do not blame
        // them for the unhealthy state that caused the intervention. Instead,
        // slowly unwind the ordering habits that maintained that state.
        for (action, scale) in [
            (ControlAction::Recall, 0.30),
            (ControlAction::Contract, 0.25),
            (ControlAction::Crystallize, 0.20),
            (ControlAction::Hold, 0.12),
        ] {
            let q = &mut self.q[action.index()];
            let target = -scale * strength;
            *q += 0.03 * (target - *q);
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct HybridController {
    bandit: ModelFreeBandit,
    meta: MetaModel,
    current_action: ControlAction,
    action_age: u32,
    cached_model_scores: [f32; ACTION_COUNT],
}
impl Default for HybridController {
    fn default() -> Self {
        Self {
            bandit: ModelFreeBandit::default(),
            meta: MetaModel::default(),
            current_action: ControlAction::Hold,
            action_age: 0,
            cached_model_scores: [0.0; ACTION_COUNT],
        }
    }
}
impl HybridController {
    fn choose(
        &mut self,
        temp: f32,
        ecology: &mut AdaptiveDynamics,
        motif_available: bool,
        forced_action: Option<ControlAction>,
        rng: &mut RuntimeRng,
    ) -> ControlAction {
        let model_weight = ecology.effective_model_weight.clamp(0.03, 0.92);
        let escape = ecology.escape_strength();
        let epsilon =
            (0.05 + 0.18 * temp + 0.20 * (1.0 - model_weight) + 0.24 * ecology.stagnation)
                .clamp(0.05, 0.58);

        let policy_choice = if let Some(forced) = forced_action {
            forced
        } else if escape > 0.56
            && ecology.escape_cooldown == 0
            && rng.gen::<f32>() < (0.35 + 0.55 * escape)
        {
            let rescue = [
                ControlAction::Explore,
                ControlAction::Turbulence,
                ControlAction::Resonate,
                ControlAction::Inharmonic,
                ControlAction::Widen,
            ];
            ecology.escape_cooldown = 24;
            rescue[rng.gen_range(0..rescue.len())]
        } else if rng.gen::<f32>() < epsilon {
            let mut action = ControlAction::Hold;
            for _ in 0..12 {
                let candidate = ControlAction::from_index(rng.gen_range(0..ACTION_COUNT));
                if candidate == ControlAction::Recall && !motif_available {
                    continue;
                }
                if ecology.stagnation > 0.65
                    && matches!(
                        candidate,
                        ControlAction::Crystallize | ControlAction::Contract
                    )
                {
                    continue;
                }
                action = candidate;
                break;
            }
            action
        } else {
            let normalize = |xs: &[f32; ACTION_COUNT], i: usize| {
                let lo = xs.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let hi = xs.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                if hi - lo < 1e-6 {
                    0.5
                } else {
                    (xs[i] - lo) / (hi - lo)
                }
            };
            let total_visits = self.bandit.visits.iter().map(|&v| v as f32).sum::<f32>() + 1.0;
            let mut best_i = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for i in 0..ACTION_COUNT {
                let action = ControlAction::from_index(i);
                if action == ControlAction::Recall && !motif_available {
                    continue;
                }
                let planned = normalize(&self.cached_model_scores, i);
                let habitual = normalize(&self.bandit.q, i);
                let current_tax = if i == self.current_action.index() {
                    (self.action_age as f32 / 20.0).clamp(0.0, 0.38)
                } else {
                    0.0
                };
                let usage_tax = ecology.action_usage_penalty(action);
                let ucb = ((total_visits.ln() / (self.bandit.visits[i] as f32 + 1.0)).sqrt()
                    * 0.045
                    * (1.0 - model_weight))
                    .clamp(0.0, 0.16);
                let escape_bias = match action {
                    ControlAction::Explore => 0.34 * escape,
                    ControlAction::Turbulence => 0.42 * escape,
                    ControlAction::Resonate => 0.20 * escape,
                    ControlAction::Inharmonic => 0.16 * escape,
                    ControlAction::Widen => 0.12 * escape,
                    ControlAction::Crystallize => -0.55 * ecology.stagnation,
                    ControlAction::Contract => -0.45 * ecology.stagnation,
                    ControlAction::Hold => -0.28 * ecology.stagnation,
                    _ => 0.0,
                };
                let v =
                    model_weight * planned + (1.0 - model_weight) * habitual + ucb + escape_bias
                        - current_tax
                        - usage_tax;
                if v > best_v {
                    best_v = v;
                    best_i = i;
                }
            }
            ControlAction::from_index(best_i)
        };
        // A hard ecological recovery is a safety override, not another score
        // for a confident planner or a recall-heavy bandit to outvote.
        let chosen = policy_choice;
        if chosen == self.current_action {
            self.action_age = self.action_age.saturating_add(1);
        } else {
            self.current_action = chosen;
            self.action_age = 0;
        }
        ecology.record_action(chosen);
        chosen
    }

    fn model_proposal(&self) -> ControlAction {
        let mut best_i = 0usize;
        let mut best = f32::NEG_INFINITY;
        for (i, &v) in self.cached_model_scores.iter().enumerate() {
            if v > best {
                best = v;
                best_i = i;
            }
        }
        ControlAction::from_index(best_i)
    }

    fn bandit_proposal(&self) -> ControlAction {
        let mut best_i = 0usize;
        let mut best = f32::NEG_INFINITY;
        for (i, &v) in self.bandit.q.iter().enumerate() {
            if v > best {
                best = v;
                best_i = i;
            }
        }
        ControlAction::from_index(best_i)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Motif {
    observation: AudioObservation,
    control: SynthesisControl,
    born_step: u64,
    quality: f32,
}
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
struct MotifMemory {
    entries: VecDeque<Motif>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct MotifDiagnostics {
    candidates: u64,
    stored_total: u64,
    rejected_quality: u64,
    rejected_similarity: u64,
    recall_attempts: u64,
    recall_hits: u64,
    last_store_step: u64,
    last_quality: f32,
    last_nearest_distance: f32,
    best_quality: f32,
}
impl Default for MotifDiagnostics {
    fn default() -> Self {
        Self {
            candidates: 0,
            stored_total: 0,
            rejected_quality: 0,
            rejected_similarity: 0,
            recall_attempts: 0,
            recall_hits: 0,
            last_store_step: 0,
            last_quality: 0.0,
            last_nearest_distance: 1.0,
            best_quality: 0.0,
        }
    }
}

impl MotifMemory {
    fn diversity_score(distance: f32) -> f32 {
        // Accepted motifs are at least ~0.03-0.045 apart. A distance of 0.20
        // is already strongly distinct in the normalized 12-D observation.
        (distance / 0.20).clamp(0.0, 1.0)
    }

    fn retention_score(&self, index: usize) -> f32 {
        let motif = &self.entries[index];
        let nearest_other = self
            .entries
            .iter()
            .enumerate()
            .filter(|(other, _)| *other != index)
            .map(|(_, other)| motif.observation.distance(&other.observation))
            .fold(1.0f32, |a, b| a.min(b));
        0.72 * motif.quality + 0.28 * Self::diversity_score(nearest_other)
    }

    fn maybe_store(
        &mut self,
        obs: &AudioObservation,
        control: SynthesisControl,
        step: u64,
        ecology: &AdaptiveDynamics,
        diagnostics: &mut MotifDiagnostics,
    ) -> bool {
        diagnostics.candidates = diagnostics.candidates.saturating_add(1);
        let quality = (0.78 * obs.structured_complexity()
            + 0.12 * obs.values[3]
            + 0.10 * ecology.activity_health)
            .clamp(0.0, 1.0);
        let nearest = self
            .entries
            .iter()
            .map(|m| obs.distance(&m.observation))
            .fold(1.0f32, |a, b| a.min(b));
        diagnostics.last_quality = quality;
        diagnostics.last_nearest_distance = nearest;
        diagnostics.best_quality = diagnostics.best_quality.max(quality);

        let long_drought = step.saturating_sub(diagnostics.last_store_step) > 1024;
        let quality_floor = if self.entries.is_empty() {
            0.25
        } else {
            (0.34 - 0.07 * ecology.stagnation - if long_drought { 0.04 } else { 0.0 })
                .clamp(0.24, 0.36)
        };
        if quality < quality_floor {
            diagnostics.rejected_quality = diagnostics.rejected_quality.saturating_add(1);
            return false;
        }
        let separation_floor = if self.entries.len() < 4 {
            0.025
        } else if long_drought {
            0.030
        } else {
            0.045
        };
        if !self.entries.is_empty() && nearest < separation_floor {
            diagnostics.rejected_similarity = diagnostics.rejected_similarity.saturating_add(1);
            return false;
        }
        let candidate = Motif {
            observation: obs.clone(),
            control,
            born_step: step,
            quality,
        };
        if self.entries.len() < MOTIF_SLOTS {
            self.entries.push_back(candidate);
        } else if let Some((weakest_index, weakest_score)) = (0..self.entries.len())
            .map(|index| (index, self.retention_score(index)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        {
            let candidate_score = 0.72 * quality + 0.28 * Self::diversity_score(nearest);
            if candidate_score <= weakest_score {
                diagnostics.rejected_quality = diagnostics.rejected_quality.saturating_add(1);
                return false;
            }
            self.entries[weakest_index] = candidate;
        }
        diagnostics.stored_total = diagnostics.stored_total.saturating_add(1);
        diagnostics.last_store_step = step;
        true
    }

    fn has_recallable(&self, step: u64) -> bool {
        self.entries
            .iter()
            .any(|m| step.saturating_sub(m.born_step) >= MOTIF_MIN_AGE)
    }

    fn recall(
        &self,
        current: Option<&AudioObservation>,
        step: u64,
        diagnostics: &mut MotifDiagnostics,
    ) -> Option<(SynthesisControl, f32)> {
        diagnostics.recall_attempts = diagnostics.recall_attempts.saturating_add(1);
        let current = current?;
        let result = self
            .entries
            .iter()
            .filter(|m| step.saturating_sub(m.born_step) >= MOTIF_MIN_AGE)
            .map(|m| {
                let similarity = (1.0 - current.distance(&m.observation)).clamp(0.0, 1.0);
                (m.control, similarity * m.quality)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        if result.is_some() {
            diagnostics.recall_hits = diagnostics.recall_hits.saturating_add(1);
        }
        result
    }

    fn recurrence(&self, current: &AudioObservation, step: u64) -> f32 {
        self.entries
            .iter()
            .filter(|m| step.saturating_sub(m.born_step) >= MOTIF_MIN_AGE)
            .map(|m| (1.0 - current.distance(&m.observation)).clamp(0.0, 1.0) * m.quality)
            .fold(0.0f32, |a, b| a.max(b))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CriticalityEstimator {
    history: VecDeque<f32>,
    sigma: f32,
    confidence: f32,
    correlation: f32,
}
impl Default for CriticalityEstimator {
    fn default() -> Self {
        Self {
            history: VecDeque::with_capacity(96),
            sigma: 1.0,
            confidence: 0.0,
            correlation: 0.0,
        }
    }
}
impl CriticalityEstimator {
    fn update(&mut self, movement: f32) -> f32 {
        self.history.push_back(movement);
        if self.history.len() > 96 {
            self.history.pop_front();
        }
        if self.history.len() < 16 {
            return self.sigma;
        }
        let x: Vec<f32> = self.history.iter().copied().collect();
        let mean = x.iter().sum::<f32>() / x.len() as f32;
        let variance = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / x.len() as f32 + 1e-8;
        let corr = |lag: usize| -> f32 {
            let mut cov = 0.0f32;
            for i in lag..x.len() {
                cov += (x[i] - mean) * (x[i - lag] - mean);
            }
            (cov / ((x.len() - lag) as f32 * variance)).clamp(-0.99, 0.99)
        };
        let c1 = corr(1).max(0.0);
        let c2 = corr(2).max(0.0).sqrt();
        let c4 = corr(4).max(0.0).sqrt().sqrt();
        let estimate = (0.55 * c1 + 0.28 * c2 + 0.17 * c4).clamp(0.0, 1.25);
        self.sigma += 0.12 * (estimate - self.sigma);
        self.correlation += 0.10 * (c1 - self.correlation);
        let consistency = 1.0 - ((c1 - c2).abs() + (c2 - c4).abs()).min(1.0) * 0.5;
        self.confidence += 0.08 * (consistency - self.confidence);
        self.sigma
    }
}

// =====================================================================
// V8 RECURSIVE CONFINEMENT REACTOR
// =====================================================================
// This host-side subsystem is deliberately compact. It never predicts raw
// samples or clones the neural CA. Instead it learns transitions in a bounded
// 48-dimensional reactor state assembled from final-audio, ecological,
// criticality and renormalization-group measurements.

const REACTOR_DIM: usize = 48;
const ACTION_FEATURE_DIM: usize = ACTION_COUNT;
const RG_FEATURE_DIM: usize = 16;
const CRITICAL_DIM: usize = 8;
const TRANSITION_HIDDEN: usize = 64;
const TRANSITION_ENSEMBLE_SIZE: usize = 3;
const TRANSITION_REPLAY_CAPACITY: usize = 2048;
const TRANSITION_DELTA_SCALE: f32 = 0.35;
const ATTRACTOR_SLOTS: usize = 48;
const ATTRACTOR_MIN_AGE: u64 = 128;
const ATTRACTOR_RECALL_COOLDOWN: u64 = 192;
const STATE_SPACE_MAX_BINS: usize = 2048;
const STATE_SPACE_RECENT: usize = 256;
const PROBE_EVERY: u64 = 192;
const PROBE_DELAY: u64 = 4;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
enum PlannerProfile {
    Cool,
    Balanced,
    Max,
}
impl PlannerProfile {
    fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "cool" | "low" => Ok(Self::Cool),
            "balanced" | "default" | "medium" => Ok(Self::Balanced),
            "max" | "deep" | "high" => Ok(Self::Max),
            _ => anyhow::bail!("unknown planner profile '{s}' (use cool, balanced, or max)"),
        }
    }
    fn label(self) -> &'static str {
        match self {
            Self::Cool => "cool",
            Self::Balanced => "balanced",
            Self::Max => "max",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct PlannerConfig {
    horizon: usize,
    beam_width: usize,
    min_interval: u64,
    max_interval: u64,
    train_every: u64,
    train_batch: usize,
}
impl PlannerConfig {
    fn for_profile(profile: PlannerProfile) -> Self {
        match profile {
            PlannerProfile::Cool => Self {
                horizon: 2,
                beam_width: 2,
                min_interval: 8,
                max_interval: 24,
                train_every: 16,
                train_batch: 4,
            },
            PlannerProfile::Balanced => Self {
                horizon: 4,
                beam_width: 4,
                min_interval: 4,
                max_interval: 16,
                train_every: 8,
                train_batch: 6,
            },
            PlannerProfile::Max => Self {
                horizon: 5,
                beam_width: 6,
                min_interval: 2,
                max_interval: 10,
                train_every: 4,
                train_batch: 8,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct ActionCommand {
    action: ControlAction,
    intensity: f32,
}
impl Default for ActionCommand {
    fn default() -> Self {
        Self {
            action: ControlAction::Hold,
            intensity: 0.0,
        }
    }
}
impl ActionCommand {
    fn new(action: ControlAction, intensity: f32) -> Self {
        Self {
            action,
            intensity: if action == ControlAction::Hold {
                0.0
            } else {
                intensity.clamp(0.08, 1.0)
            },
        }
    }
    fn control(self) -> SynthesisControl {
        SynthesisControl::default().blend(SynthesisControl::for_action(self.action), self.intensity)
    }
    fn features(self) -> [f32; ACTION_FEATURE_DIM] {
        control_features(self.action, self.control())
    }
}

fn normalized_entropy(values: &[f32]) -> f32 {
    if values.len() <= 1 {
        return 0.0;
    }
    let shifted: Vec<f32> = values.iter().map(|v| v.max(0.0) + 1e-6).collect();
    let sum = shifted.iter().sum::<f32>().max(1e-6);
    let h = shifted
        .iter()
        .map(|v| {
            let p = *v / sum;
            -p * p.ln()
        })
        .sum::<f32>();
    (h / (values.len() as f32).ln()).clamp(0.0, 1.0)
}

fn mean_std(values: &[f32]) -> (f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
    (mean, var.max(0.0).sqrt())
}

fn neighbor_correlation_4x4(values: &[f32; REGION_COUNT]) -> f32 {
    let (mean, std) = mean_std(values);
    if std < 1e-6 {
        return 0.0;
    }
    let mut cov = 0.0f32;
    let mut n = 0usize;
    for y in 0..REGION_ROWS {
        for x in 0..REGION_COLS {
            let a = values[y * REGION_COLS + x] - mean;
            for (nx, ny) in [((x + 1) % REGION_COLS, y), (x, (y + 1) % REGION_ROWS)] {
                cov += a * (values[ny * REGION_COLS + nx] - mean);
                n += 1;
            }
        }
    }
    ((cov / n.max(1) as f32) / (std * std + 1e-6)).clamp(-1.0, 1.0)
}

fn pool_4x4_to_2x2(values: &[f32; REGION_COUNT]) -> [f32; 4] {
    let mut out = [0.0f32; 4];
    for by in 0..2 {
        for bx in 0..2 {
            let mut sum = 0.0;
            for dy in 0..2 {
                for dx in 0..2 {
                    sum += values[(by * 2 + dy) * REGION_COLS + bx * 2 + dx];
                }
            }
            out[by * 2 + bx] = sum * 0.25;
        }
    }
    out
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RgObserver {
    features: [f32; RG_FEATURE_DIM],
    prev_fine: [f32; REGION_COUNT],
    prev_coarse: [f32; 4],
    initialized: bool,
    entropy_rate_ema: f32,
    scale_invariance_ema: f32,
    disagreement_ema: f32,
    active_scales_ema: f32,
}
impl Default for RgObserver {
    fn default() -> Self {
        Self {
            features: [0.0; RG_FEATURE_DIM],
            prev_fine: [0.0; REGION_COUNT],
            prev_coarse: [0.0; 4],
            initialized: false,
            entropy_rate_ema: 0.35,
            scale_invariance_ema: 0.5,
            disagreement_ema: 0.35,
            active_scales_ema: 0.5,
        }
    }
}
impl RgObserver {
    fn update(&mut self, activity: &[f32; REGION_COUNT], change: &[f32; REGION_COUNT]) {
        let fine: [f32; REGION_COUNT] =
            std::array::from_fn(|i| activity[i].abs().clamp(0.0, 1.5) / 1.5);
        let delta: [f32; REGION_COUNT] =
            std::array::from_fn(|i| change[i].abs().clamp(0.0, 0.08) / 0.08);
        let coarse = pool_4x4_to_2x2(&fine);
        let coarse_delta = pool_4x4_to_2x2(&delta);
        let (fine_mean, fine_std) = mean_std(&fine);
        let (coarse_mean, coarse_std) = mean_std(&coarse);
        let (delta_mean, delta_std) = mean_std(&delta);
        let (coarse_delta_mean, coarse_delta_std) = mean_std(&coarse_delta);
        let fine_entropy = normalized_entropy(&fine);
        let coarse_entropy = normalized_entropy(&coarse);
        let spatial_corr = ((neighbor_correlation_4x4(&fine) + 1.0) * 0.5).clamp(0.0, 1.0);
        let active_fraction = fine
            .iter()
            .filter(|&&v| v > fine_mean + 0.25 * fine_std)
            .count() as f32
            / REGION_COUNT as f32;
        let temporal_fine = if self.initialized {
            fine.iter()
                .zip(self.prev_fine.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>()
                / REGION_COUNT as f32
        } else {
            delta_mean
        };
        let temporal_coarse = if self.initialized {
            coarse
                .iter()
                .zip(self.prev_coarse.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>()
                * 0.25
        } else {
            coarse_delta_mean
        };
        let scale_invariance =
            (1.0 - (fine_entropy - coarse_entropy).abs() - 0.7 * (fine_std - coarse_std).abs())
                .clamp(0.0, 1.0);
        let disagreement = ((temporal_fine - temporal_coarse).abs() * 4.0
            + (fine_std - coarse_std).abs())
        .clamp(0.0, 1.0);
        let active_scales = ((delta_mean > 0.06) as u8 as f32
            + (coarse_delta_mean > 0.045) as u8 as f32
            + (temporal_coarse > 0.025) as u8 as f32)
            / 3.0;
        let entropy_rate =
            (0.55 * temporal_fine * 5.0 + 0.45 * temporal_coarse * 6.0).clamp(0.0, 1.0);
        self.entropy_rate_ema += 0.10 * (entropy_rate - self.entropy_rate_ema);
        // Static fields look perfectly scale-invariant because every scale is
        // equally unchanged. Require temporal support before treating that
        // agreement as evidence of healthy multi-scale organization.
        let temporal_support = (entropy_rate / 0.10).clamp(0.0, 1.0);
        let dynamic_scale_invariance = scale_invariance * (0.20 + 0.80 * temporal_support);
        self.scale_invariance_ema += 0.08 * (dynamic_scale_invariance - self.scale_invariance_ema);
        self.disagreement_ema += 0.08 * (disagreement - self.disagreement_ema);
        self.active_scales_ema += 0.08 * (active_scales - self.active_scales_ema);
        self.features = [
            fine_mean.clamp(0.0, 1.0),
            (fine_std * 3.0).clamp(0.0, 1.0),
            fine_entropy,
            delta_mean.clamp(0.0, 1.0),
            (delta_std * 2.5).clamp(0.0, 1.0),
            coarse_mean.clamp(0.0, 1.0),
            (coarse_std * 3.0).clamp(0.0, 1.0),
            coarse_entropy,
            coarse_delta_mean.clamp(0.0, 1.0),
            (coarse_delta_std * 2.5).clamp(0.0, 1.0),
            spatial_corr,
            active_fraction,
            self.entropy_rate_ema,
            self.scale_invariance_ema,
            self.disagreement_ema,
            self.active_scales_ema,
        ];
        self.prev_fine = fine;
        self.prev_coarse = coarse;
        self.initialized = true;
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CriticalManifold {
    vector: [f32; CRITICAL_DIM],
    health: f32,
    order_risk: f32,
    chaos_risk: f32,
    target_phase: f32,
    lyapunov_proxy: f32,
    prediction_horizon: f32,
    susceptibility: f32,
}
impl Default for CriticalManifold {
    fn default() -> Self {
        Self {
            vector: [0.5; CRITICAL_DIM],
            health: 0.55,
            order_risk: 0.25,
            chaos_risk: 0.20,
            target_phase: 0.0,
            lyapunov_proxy: 0.5,
            prediction_horizon: 0.5,
            susceptibility: 0.5,
        }
    }
}
impl CriticalManifold {
    fn target(&self) -> [f32; CRITICAL_DIM] {
        let p = self.target_phase;
        [
            0.60 + 0.10 * p.sin(),
            0.54 + 0.10 * (p + 0.8).sin(),
            0.55 + 0.08 * (p + 1.6).sin(),
            0.52 + 0.12 * (p + 2.2).sin(),
            0.52 + 0.12 * (p + 2.8).sin(),
            0.58 + 0.10 * (p + 3.4).sin(),
            0.60 + 0.08 * (p + 4.0).sin(),
            0.56 + 0.10 * (p + 4.7).sin(),
        ]
    }
    fn score_vector(&self, values: &[f32; CRITICAL_DIM]) -> f32 {
        let target = self.target();
        let widths = [0.28, 0.30, 0.30, 0.34, 0.30, 0.32, 0.30, 0.32];
        let d = (0..CRITICAL_DIM)
            .map(|i| ((values[i] - target[i]) / widths[i]).powi(2))
            .sum::<f32>()
            / CRITICAL_DIM as f32;
        (-0.75 * d).exp().clamp(0.0, 1.0)
    }
    #[allow(clippy::too_many_arguments)]
    fn update(
        &mut self,
        sigma: f32,
        criticality: &CriticalityEstimator,
        rg: &RgObserver,
        adaptive: &AdaptiveDynamics,
        observation_delta: f32,
        prediction_horizon: f32,
        lyapunov_proxy: f32,
        susceptibility: f32,
        occupancy_novelty: f32,
    ) {
        self.target_phase = (self.target_phase
            + 0.0016
            + 0.0012 * (1.0 - adaptive.activity_health)
            + 0.0008 * occupancy_novelty)
            .rem_euclid(TWO_PI);
        self.prediction_horizon += 0.08 * (prediction_horizon - self.prediction_horizon);
        self.lyapunov_proxy += 0.08 * (lyapunov_proxy - self.lyapunov_proxy);
        self.susceptibility += 0.08 * (susceptibility - self.susceptibility);
        let branching = (sigma / 1.05).clamp(0.0, 1.0);
        let correlation_length =
            (0.58 * criticality.correlation.max(0.0) + 0.42 * rg.features[10]).clamp(0.0, 1.0);
        let temporal_memory =
            (0.52 * criticality.confidence + 0.48 * self.prediction_horizon).clamp(0.0, 1.0);
        let entropy_rate = (0.72 * rg.entropy_rate_ema
            + 0.28 * (observation_delta / 0.10).clamp(0.0, 1.0))
        .clamp(0.0, 1.0);
        let regional_diversity =
            (0.50 * rg.features[2] + 0.28 * rg.features[1] + 0.22 * rg.features[11])
                .clamp(0.0, 1.0);
        self.vector = [
            branching,
            correlation_length,
            temporal_memory,
            self.susceptibility,
            entropy_rate,
            self.prediction_horizon,
            regional_diversity,
            rg.scale_invariance_ema,
        ];
        let target_health = self.score_vector(&self.vector);
        self.health += 0.08 * (target_health - self.health);
        let order_target = (0.42 * (1.0 - entropy_rate)
            + 0.30 * (1.0 - branching)
            + 0.18 * adaptive.stagnation
            + 0.10 * (1.0 - self.lyapunov_proxy))
            .clamp(0.0, 1.0);
        let chaos_target = (0.34 * entropy_rate
            + 0.24 * self.lyapunov_proxy
            + 0.22 * (1.0 - self.prediction_horizon)
            + 0.20 * rg.disagreement_ema)
            .clamp(0.0, 1.0);
        self.order_risk += 0.08 * (order_target - self.order_risk);
        self.chaos_risk += 0.08 * (chaos_target - self.chaos_risk);
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ReactorState {
    #[serde(with = "BigArray")]
    values: [f32; REACTOR_DIM],
}
impl Default for ReactorState {
    fn default() -> Self {
        Self {
            values: [0.0; REACTOR_DIM],
        }
    }
}
impl ReactorState {
    #[allow(clippy::too_many_arguments)]
    fn assemble(
        observation: &AudioObservation,
        adaptive: &AdaptiveDynamics,
        manifold: &CriticalManifold,
        rg: &RgObserver,
        energy: f32,
        occupancy_novelty: f32,
        occupancy_entropy: f32,
        recurrence: f32,
        recoverability: f32,
    ) -> Self {
        let mut v = [0.0f32; REACTOR_DIM];
        v[..OBS_DIM].copy_from_slice(&observation.values);
        v[12] = (adaptive.movement_fast / 0.015).clamp(0.0, 1.0);
        v[13] = (adaptive.region_fast / 0.012).clamp(0.0, 1.0);
        v[14] = (adaptive.observation_delta_ema / 0.12).clamp(0.0, 1.0);
        v[15] = adaptive.complexity_ema.clamp(0.0, 1.0);
        v[16] = adaptive.activity_health.clamp(0.0, 1.0);
        v[17] = adaptive.stagnation.clamp(0.0, 1.0);
        v[18] = energy.clamp(0.0, 1.0);
        v[19] = adaptive.effective_model_weight.clamp(0.0, 1.0);
        v[20..20 + CRITICAL_DIM].copy_from_slice(&manifold.vector);
        v[28..28 + RG_FEATURE_DIM].copy_from_slice(&rg.features);
        v[44] = occupancy_novelty.clamp(0.0, 1.0);
        v[45] = occupancy_entropy.clamp(0.0, 1.0);
        v[46] = recurrence.clamp(0.0, 1.0);
        v[47] = recoverability.clamp(0.0, 1.0);
        Self { values: v }
    }
    fn distance(&self, other: &Self) -> f32 {
        (self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / REACTOR_DIM as f32)
            .sqrt()
    }
    fn structured_complexity(&self) -> f32 {
        AudioObservation {
            values: std::array::from_fn(|i| self.values[i]),
        }
        .structured_complexity()
    }
    fn temporal_continuity(&self) -> f32 {
        let delta = self.values[14];
        (-((delta - 0.34) / 0.34).powi(2)).exp().clamp(0.0, 1.0)
    }
    fn multi_scale_activity(&self) -> f32 {
        (0.24 * self.values[30]
            + 0.20 * self.values[31]
            + 0.20 * self.values[40]
            + 0.18 * self.values[42]
            + 0.18 * self.values[43])
            .clamp(0.0, 1.0)
    }
    fn viable_complexity(&self, manifold: &CriticalManifold) -> f32 {
        let structured = self.structured_complexity().max(0.02);
        let recoverability = self.values[47].max(0.05);
        let multiscale = self.multi_scale_activity().max(0.05);
        let continuity = self.temporal_continuity().max(0.05);
        let critical = manifold
            .score_vector(&std::array::from_fn(|i| self.values[20 + i]))
            .max(0.05);
        (structured * recoverability * multiscale * continuity * critical)
            .powf(0.2)
            .clamp(0.0, 1.0)
    }
    fn collapse_risk(&self) -> f32 {
        let order = 0.32 * self.values[17]
            + 0.24 * (1.0 - self.values[12])
            + 0.20 * (1.0 - self.values[32])
            + 0.14 * (1.0 - self.values[24])
            + 0.10 * (1.0 - self.values[45]);
        let chaos = 0.24 * self.values[32]
            + 0.20 * self.values[42]
            + 0.20 * (1.0 - self.values[25])
            + 0.18 * self.values[1]
            + 0.18 * self.values[14];
        order.max(chaos).clamp(0.0, 1.0)
    }
    fn hash_signature(&self) -> u64 {
        const IDX: [usize; 12] = [0, 1, 3, 7, 12, 16, 17, 20, 24, 32, 42, 46];
        let mut h = 0u64;
        for (slot, &i) in IDX.iter().enumerate() {
            let q = (self.values[i].clamp(0.0, 0.9999) * 16.0) as u64;
            h |= q.min(15) << (slot * 4);
        }
        h
    }
}

// =====================================================================
// CONFINEMENT CONTROL — a moving viable shell, not a fixed target point
// =====================================================================

const CONFINEMENT_MODES: [(usize, usize); 4] = [(1, 0), (0, 1), (1, 1), (1, 3)];

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ConfinementObserver {
    #[serde(with = "BigArray")]
    healthy_mean: [f32; REACTOR_DIM],
    #[serde(with = "BigArray")]
    healthy_var: [f32; REACTOR_DIM],
    #[serde(with = "BigArray")]
    previous_q: [f32; REACTOR_DIM],
    previous_micro_modes: [[f32; 2]; 4],
    previous_macro_modes: [[f32; 2]; 4],
    healthy_samples: u64,
    radius: f32,
    radial_velocity: f32,
    tangential_velocity: f32,
    modal_rotation: f32,
    modal_coherence: f32,
    micro_macro_lock: f32,
    health: f32,
}
impl Default for ConfinementObserver {
    fn default() -> Self {
        Self {
            healthy_mean: [0.5; REACTOR_DIM],
            healthy_var: [0.04; REACTOR_DIM],
            previous_q: [0.0; REACTOR_DIM],
            previous_micro_modes: [[0.0; 2]; 4],
            previous_macro_modes: [[0.0; 2]; 4],
            healthy_samples: 0,
            radius: 1.0,
            radial_velocity: 0.0,
            tangential_velocity: 0.0,
            modal_rotation: 0.0,
            modal_coherence: 0.0,
            micro_macro_lock: 0.5,
            health: 0.5,
        }
    }
}
impl ConfinementObserver {
    fn spatial_modes(regions: &[f32; REGION_COUNT]) -> [[f32; 2]; 4] {
        std::array::from_fn(|m| {
            let (kx, ky) = CONFINEMENT_MODES[m];
            let mut re = 0.0f32;
            let mut im = 0.0f32;
            for y in 0..REGION_ROWS {
                for x in 0..REGION_COLS {
                    let phase = TWO_PI
                        * (kx as f32 * x as f32 / REGION_COLS as f32
                            + ky as f32 * y as f32 / REGION_ROWS as f32);
                    let value = regions[y * REGION_COLS + x];
                    re += value * phase.cos();
                    im -= value * phase.sin();
                }
            }
            [re / REGION_COUNT as f32, im / REGION_COUNT as f32]
        })
    }

    fn update(
        &mut self,
        state: &ReactorState,
        micro_regions: &[f32; REGION_COUNT],
        macro_regions: &[f32; REGION_COUNT],
        learn_healthy_shell: bool,
    ) {
        if learn_healthy_shell {
            self.healthy_samples = self.healthy_samples.saturating_add(1);
            let rate = if self.healthy_samples < CONFINEMENT_CALIBRATION_SAMPLES {
                1.0 / self.healthy_samples as f32
            } else {
                0.002
            };
            for i in 0..REACTOR_DIM {
                let delta = state.values[i] - self.healthy_mean[i];
                self.healthy_mean[i] += rate * delta;
                self.healthy_var[i] += rate * (delta * delta - self.healthy_var[i]);
                self.healthy_var[i] = self.healthy_var[i].clamp(0.0004, 0.25);
            }
        }

        let q: [f32; REACTOR_DIM] = std::array::from_fn(|i| {
            ((state.values[i] - self.healthy_mean[i]) / self.healthy_var[i].sqrt()).clamp(-6.0, 6.0)
        });
        let radius = (q.iter().map(|v| v * v).sum::<f32>() / REACTOR_DIM as f32).sqrt();
        let delta: [f32; REACTOR_DIM] = std::array::from_fn(|i| q[i] - self.previous_q[i]);
        let speed_sq = delta.iter().map(|v| v * v).sum::<f32>() / REACTOR_DIM as f32;
        let radial = q
            .iter()
            .zip(delta.iter())
            .map(|(position, velocity)| position * velocity)
            .sum::<f32>()
            / (REACTOR_DIM as f32 * radius.max(1e-4));
        let tangential = (speed_sq - radial * radial).max(0.0).sqrt();

        let micro_modes = Self::spatial_modes(micro_regions);
        let macro_modes = Self::spatial_modes(macro_regions);
        let mut rotation_sum = 0.0f32;
        let mut coherence_sum = 0.0f32;
        let mut lock_sum = 0.0f32;
        let mut lock_modes = 0.0f32;
        let mut weight_sum = 0.0f32;
        for i in 0..4 {
            let [re, im] = micro_modes[i];
            let [pre, pim] = self.previous_micro_modes[i];
            let amp = (re * re + im * im).sqrt();
            let prev_amp = (pre * pre + pim * pim).sqrt();
            let weight = (amp * prev_amp).sqrt();
            if weight > 1e-5 {
                let cross = pre * im - pim * re;
                let dot = pre * re + pim * im;
                let angle = cross.atan2(dot);
                rotation_sum += angle.abs() * weight;
                coherence_sum += angle.cos().max(0.0) * weight;
                weight_sum += weight;
            }
            let [mre, mim] = macro_modes[i];
            let macro_amp = (mre * mre + mim * mim).sqrt();
            if amp > 1e-5 && macro_amp > 1e-5 {
                lock_sum += ((re * mre + im * mim) / (amp * macro_amp)).abs();
                lock_modes += 1.0;
            }
        }
        let rotation = if weight_sum > 1e-6 {
            rotation_sum / weight_sum
        } else {
            0.0
        };
        let coherence = if weight_sum > 1e-6 {
            coherence_sum / weight_sum
        } else {
            0.0
        };
        let lock = if lock_modes > 0.0 {
            lock_sum / lock_modes
        } else {
            0.0
        };

        self.radius += 0.08 * (radius - self.radius);
        self.radial_velocity += 0.12 * (radial - self.radial_velocity);
        self.tangential_velocity += 0.12 * (tangential - self.tangential_velocity);
        self.modal_rotation += 0.12 * (rotation - self.modal_rotation);
        self.modal_coherence += 0.10 * (coherence - self.modal_coherence);
        self.micro_macro_lock += 0.10 * (lock - self.micro_macro_lock);
        let calibrated =
            (self.healthy_samples as f32 / CONFINEMENT_CALIBRATION_SAMPLES as f32).clamp(0.0, 1.0);
        let shell_health = (-((self.radius - 1.0) / 0.85).powi(2)).exp();
        let tangent_health = (self.tangential_velocity / 0.08).clamp(0.0, 1.0);
        let rotation_health = (self.modal_rotation / 0.035).clamp(0.0, 1.0);
        let target_health = calibrated
            * (0.40 * shell_health
                + 0.25 * tangent_health
                + 0.20 * rotation_health
                + 0.15 * self.micro_macro_lock)
            + (1.0 - calibrated) * 0.5;
        self.health += 0.08 * (target_health - self.health);
        self.previous_q = q;
        self.previous_micro_modes = micro_modes;
        self.previous_macro_modes = macro_modes;
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
enum RecoveryPhase {
    Nominal,
    EdgeAgitation,
    RotationCapture,
    GuidedPulse,
    PartialReseed,
    Cooldown,
}
impl RecoveryPhase {
    fn label(self) -> &'static str {
        match self {
            Self::Nominal => "NOMINAL",
            Self::EdgeAgitation => "EDGE",
            Self::RotationCapture => "ROTATE",
            Self::GuidedPulse => "GUIDED",
            Self::PartialReseed => "RESEED",
            Self::Cooldown => "COOLDOWN",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RecoveryController {
    phase: RecoveryPhase,
    entered_step: u64,
    growth_used: bool,
    healthy_dwell: u32,
    progress_ema: f32,
    previous_health: f32,
    just_entered: bool,
}
impl Default for RecoveryController {
    fn default() -> Self {
        Self {
            phase: RecoveryPhase::Nominal,
            entered_step: 0,
            growth_used: false,
            healthy_dwell: 0,
            progress_ema: 0.0,
            previous_health: 0.5,
            just_entered: false,
        }
    }
}
impl RecoveryController {
    fn enter(&mut self, phase: RecoveryPhase, step: u64) {
        self.phase = phase;
        self.entered_step = step;
        self.just_entered = true;
    }

    fn update(&mut self, step: u64, hard_recovery: bool, confinement_health: f32) {
        self.just_entered = false;
        let progress = confinement_health - self.previous_health;
        self.progress_ema += 0.08 * (progress - self.progress_ema);
        self.previous_health = confinement_health;
        if !hard_recovery {
            self.healthy_dwell = self.healthy_dwell.saturating_add(1);
            if self.phase != RecoveryPhase::Nominal {
                self.enter(RecoveryPhase::Nominal, step);
            }
            if self.healthy_dwell >= 256 {
                self.growth_used = false;
            }
            return;
        }
        self.healthy_dwell = 0;
        let elapsed = step.saturating_sub(self.entered_step);
        match self.phase {
            RecoveryPhase::Nominal => self.enter(RecoveryPhase::EdgeAgitation, step),
            RecoveryPhase::EdgeAgitation if elapsed >= RECOVERY_EDGE_CHUNKS => {
                self.enter(RecoveryPhase::RotationCapture, step)
            }
            RecoveryPhase::RotationCapture if elapsed >= RECOVERY_ROTATION_CHUNKS => {
                self.enter(RecoveryPhase::GuidedPulse, step)
            }
            RecoveryPhase::GuidedPulse if elapsed >= RECOVERY_GUIDED_CHUNKS => {
                self.enter(RecoveryPhase::PartialReseed, step)
            }
            RecoveryPhase::PartialReseed if elapsed >= 1 => {
                self.enter(RecoveryPhase::Cooldown, step)
            }
            RecoveryPhase::Cooldown if elapsed >= RECOVERY_COOLDOWN_CHUNKS => {
                self.enter(RecoveryPhase::EdgeAgitation, step)
            }
            _ => {}
        }
    }

    fn elapsed(&self, step: u64) -> u64 {
        step.saturating_sub(self.entered_step)
    }
}

#[derive(Clone, Copy, Debug)]
struct CoreControl {
    update_drive: f32,
    macro_coupling: f32,
    rotational_drive: f32,
    coherent_pulse: f32,
    stochastic_heat: f32,
}
impl CoreControl {
    fn for_recovery(phase: RecoveryPhase, pressure: f32) -> Self {
        let p = pressure.clamp(0.0, 1.0);
        match phase {
            RecoveryPhase::Nominal => Self {
                update_drive: 1.0,
                macro_coupling: 1.0,
                rotational_drive: 0.0,
                coherent_pulse: 0.0,
                stochastic_heat: 1.0,
            },
            RecoveryPhase::EdgeAgitation => Self {
                update_drive: 1.04 + 0.06 * p,
                macro_coupling: 0.88,
                rotational_drive: 0.012 + 0.015 * p,
                coherent_pulse: 0.035 + 0.045 * p,
                stochastic_heat: 0.72,
            },
            RecoveryPhase::RotationCapture => Self {
                update_drive: 1.08 + 0.06 * p,
                macro_coupling: 1.05,
                rotational_drive: 0.045 + 0.055 * p,
                coherent_pulse: 0.025,
                stochastic_heat: 0.44,
            },
            RecoveryPhase::GuidedPulse => Self {
                update_drive: 1.10 + 0.08 * p,
                macro_coupling: 1.12,
                rotational_drive: 0.075 + 0.055 * p,
                coherent_pulse: 0.10 + 0.08 * p,
                stochastic_heat: 0.32,
            },
            RecoveryPhase::PartialReseed => Self {
                update_drive: 1.08,
                macro_coupling: 0.95,
                rotational_drive: 0.04,
                coherent_pulse: 0.04,
                stochastic_heat: 0.20,
            },
            RecoveryPhase::Cooldown => Self {
                update_drive: 1.04,
                macro_coupling: 1.0,
                rotational_drive: 0.02,
                coherent_pulse: 0.0,
                stochastic_heat: 0.28,
            },
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct StateSpaceTracker {
    counts: BTreeMap<u64, u32>,
    transitions: BTreeMap<u64, u32>,
    recent: VecDeque<u64>,
    last_hash: Option<u64>,
    visits: u64,
    novel_ema: f32,
    occupancy_entropy: f32,
    transition_diversity: f32,
    return_time_ema: f32,
}
impl Default for StateSpaceTracker {
    fn default() -> Self {
        Self {
            counts: BTreeMap::new(),
            transitions: BTreeMap::new(),
            recent: VecDeque::new(),
            last_hash: None,
            visits: 0,
            novel_ema: 0.5,
            occupancy_entropy: 0.0,
            transition_diversity: 0.0,
            return_time_ema: 0.5,
        }
    }
}
impl StateSpaceTracker {
    fn trim_lowest(map: &mut BTreeMap<u64, u32>, max: usize) {
        if map.len() <= max {
            return;
        }
        if let Some((&key, _)) = map.iter().min_by_key(|(_, count)| **count) {
            map.remove(&key);
        }
    }
    fn update(&mut self, state: &ReactorState) {
        let h = state.hash_signature();
        self.visits = self.visits.saturating_add(1);
        let novel = if self.counts.contains_key(&h) {
            0.0
        } else {
            1.0
        };
        *self.counts.entry(h).or_insert(0) += 1;
        Self::trim_lowest(&mut self.counts, STATE_SPACE_MAX_BINS);
        self.novel_ema += 0.04 * (novel - self.novel_ema);
        if let Some(prev) = self.last_hash {
            let edge = prev.rotate_left(17) ^ h;
            *self.transitions.entry(edge).or_insert(0) += 1;
            Self::trim_lowest(&mut self.transitions, STATE_SPACE_MAX_BINS * 2);
        }
        let return_time = self
            .recent
            .iter()
            .rev()
            .position(|&x| x == h)
            .map(|p| (p as f32 / STATE_SPACE_RECENT as f32).clamp(0.0, 1.0))
            .unwrap_or(1.0);
        self.return_time_ema += 0.05 * (return_time - self.return_time_ema);
        self.recent.push_back(h);
        if self.recent.len() > STATE_SPACE_RECENT {
            self.recent.pop_front();
        }
        self.last_hash = Some(h);
        if self.visits % 16 == 0 {
            let total = self
                .counts
                .values()
                .map(|&v| v as f32)
                .sum::<f32>()
                .max(1.0);
            let h_raw = self
                .counts
                .values()
                .map(|&v| {
                    let p = v as f32 / total;
                    -p * p.ln()
                })
                .sum::<f32>();
            self.occupancy_entropy = if self.counts.len() > 1 {
                (h_raw / (self.counts.len() as f32).ln()).clamp(0.0, 1.0)
            } else {
                0.0
            };
            self.transition_diversity = (self.transitions.len() as f32
                / (self.visits as f32).sqrt().max(1.0))
            .clamp(0.0, 1.0);
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TransitionSample {
    state: ReactorState,
    action: [f32; ACTION_FEATURE_DIM],
    next: ReactorState,
    reward: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TinyTransitionModel {
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    loss_ema: f32,
    steps: u64,
}
impl TinyTransitionModel {
    fn new(rng: &mut RuntimeRng) -> Self {
        let input = REACTOR_DIM + ACTION_FEATURE_DIM;
        let b1_bound = 1.0 / (input as f32).sqrt();
        let b2_bound = 1.0 / (TRANSITION_HIDDEN as f32).sqrt();
        Self {
            w1: (0..TRANSITION_HIDDEN * input)
                .map(|_| rng.gen_range(-b1_bound..b1_bound))
                .collect(),
            b1: vec![0.0; TRANSITION_HIDDEN],
            w2: (0..REACTOR_DIM * TRANSITION_HIDDEN)
                .map(|_| rng.gen_range(-b2_bound..b2_bound))
                .collect(),
            b2: vec![0.0; REACTOR_DIM],
            loss_ema: 0.2,
            steps: 0,
        }
    }
    fn forward_parts(
        &self,
        state: &ReactorState,
        action: &[f32; ACTION_FEATURE_DIM],
    ) -> (Vec<f32>, Vec<f32>, ReactorState) {
        let input_dim = REACTOR_DIM + ACTION_FEATURE_DIM;
        let mut input = vec![0.0f32; input_dim];
        input[..REACTOR_DIM].copy_from_slice(&state.values);
        input[REACTOR_DIM..].copy_from_slice(action);
        let mut hidden = vec![0.0f32; TRANSITION_HIDDEN];
        for h in 0..TRANSITION_HIDDEN {
            let row = &self.w1[h * input_dim..(h + 1) * input_dim];
            hidden[h] = (self.b1[h]
                + row
                    .iter()
                    .zip(input.iter())
                    .map(|(w, x)| w * x)
                    .sum::<f32>())
            .tanh();
        }
        let mut raw = vec![0.0f32; REACTOR_DIM];
        let mut next = [0.0f32; REACTOR_DIM];
        for o in 0..REACTOR_DIM {
            let row = &self.w2[o * TRANSITION_HIDDEN..(o + 1) * TRANSITION_HIDDEN];
            raw[o] = self.b2[o]
                + row
                    .iter()
                    .zip(hidden.iter())
                    .map(|(w, x)| w * x)
                    .sum::<f32>();
            next[o] = (state.values[o] + raw[o].tanh() * TRANSITION_DELTA_SCALE).clamp(0.0, 1.0);
        }
        (hidden, raw, ReactorState { values: next })
    }
    fn predict(&self, state: &ReactorState, action: &[f32; ACTION_FEATURE_DIM]) -> ReactorState {
        self.forward_parts(state, action).2
    }
    fn train(&mut self, sample: &TransitionSample, lr: f32) -> f32 {
        let input_dim = REACTOR_DIM + ACTION_FEATURE_DIM;
        let mut input = vec![0.0f32; input_dim];
        input[..REACTOR_DIM].copy_from_slice(&sample.state.values);
        input[REACTOR_DIM..].copy_from_slice(&sample.action);
        let (hidden, raw, predicted) = self.forward_parts(&sample.state, &sample.action);
        let mut grad_out = vec![0.0f32; REACTOR_DIM];
        let mut loss = 0.0f32;
        for o in 0..REACTOR_DIM {
            let err = predicted.values[o] - sample.next.values[o];
            loss += err * err;
            let t = raw[o].tanh();
            grad_out[o] = (2.0 * err / REACTOR_DIM as f32) * TRANSITION_DELTA_SCALE * (1.0 - t * t);
        }
        loss /= REACTOR_DIM as f32;
        let mut grad_hidden = vec![0.0f32; TRANSITION_HIDDEN];
        for o in 0..REACTOR_DIM {
            for h in 0..TRANSITION_HIDDEN {
                grad_hidden[h] += self.w2[o * TRANSITION_HIDDEN + h] * grad_out[o];
            }
        }
        for o in 0..REACTOR_DIM {
            let g = grad_out[o].clamp(-0.25, 0.25);
            self.b2[o] -= lr * g;
            for h in 0..TRANSITION_HIDDEN {
                let idx = o * TRANSITION_HIDDEN + h;
                self.w2[idx] -= lr * (g * hidden[h] + 1e-5 * self.w2[idx]);
            }
        }
        for h in 0..TRANSITION_HIDDEN {
            let g = (grad_hidden[h] * (1.0 - hidden[h] * hidden[h])).clamp(-0.25, 0.25);
            self.b1[h] -= lr * g;
            for i in 0..input_dim {
                let idx = h * input_dim + i;
                self.w1[idx] -= lr * (g * input[i] + 1e-5 * self.w1[idx]);
            }
        }
        self.steps = self.steps.saturating_add(1);
        self.loss_ema += 0.03 * (loss - self.loss_ema);
        loss
    }
}

#[derive(Clone, Debug)]
struct EnsemblePrediction {
    mean: ReactorState,
    disagreement: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TransitionEnsemble {
    models: Vec<TinyTransitionModel>,
    replay: VecDeque<TransitionSample>,
    rng: RuntimeRng,
    train_loss_ema: f32,
    updates: u64,
}
impl TransitionEnsemble {
    fn new(seed: u64) -> Self {
        let mut rng = RuntimeRng::seed_from_u64(seed ^ 0xA771_C0DE_5EED_0007);
        let models = (0..TRANSITION_ENSEMBLE_SIZE)
            .map(|_| TinyTransitionModel::new(&mut rng))
            .collect();
        Self {
            models,
            replay: VecDeque::with_capacity(TRANSITION_REPLAY_CAPACITY),
            rng,
            train_loss_ema: 0.2,
            updates: 0,
        }
    }
    fn predict(&self, state: &ReactorState, command: ActionCommand) -> EnsemblePrediction {
        let features = command.features();
        let predictions: Vec<ReactorState> = self
            .models
            .iter()
            .map(|m| m.predict(state, &features))
            .collect();
        let mut mean = [0.0f32; REACTOR_DIM];
        for p in &predictions {
            for i in 0..REACTOR_DIM {
                mean[i] += p.values[i] / predictions.len() as f32;
            }
        }
        let mut variance = 0.0f32;
        for p in &predictions {
            for i in 0..REACTOR_DIM {
                variance += (p.values[i] - mean[i]).powi(2);
            }
        }
        variance /= (predictions.len() * REACTOR_DIM).max(1) as f32;
        EnsemblePrediction {
            mean: ReactorState { values: mean },
            disagreement: (variance.sqrt() * 8.0).clamp(0.0, 1.0),
        }
    }
    fn readiness(&self) -> f32 {
        let data = (self.replay.len() as f32 / 256.0).clamp(0.0, 1.0);
        let updates = (self.updates as f32 / 768.0).clamp(0.0, 1.0);
        let accuracy = (-4.0 * self.train_loss_ema).exp().clamp(0.0, 1.0);
        (data * updates * accuracy).powf(1.0 / 3.0).clamp(0.0, 1.0)
    }
    fn observe(
        &mut self,
        state: ReactorState,
        command: ActionCommand,
        next: ReactorState,
        reward: f32,
    ) {
        if self.replay.len() >= TRANSITION_REPLAY_CAPACITY {
            self.replay.pop_front();
        }
        self.replay.push_back(TransitionSample {
            state,
            action: command.features(),
            next,
            reward,
        });
    }
    fn train_if_due(&mut self, step: u64, config: PlannerConfig) {
        if self.replay.len() < 32 || step % config.train_every != 0 {
            return;
        }
        let mut total = 0.0f32;
        let mut n = 0usize;
        let model_len = self.models.len().max(1);
        for _ in 0..config.train_batch {
            // Tiny tournament-prioritized replay: regime changes, rare-state
            // transitions and strong outcomes are learned sooner without the
            // memory/maintenance cost of a full sum tree on a phone.
            let mut idx = self.rng.gen_range(0..self.replay.len());
            let mut best_priority = f32::NEG_INFINITY;
            for _ in 0..3 {
                let candidate = self.rng.gen_range(0..self.replay.len());
                let sample = &self.replay[candidate];
                let transition_size = sample.state.distance(&sample.next);
                let priority = 0.44 * transition_size
                    + 0.26 * sample.reward.abs()
                    + 0.18 * sample.next.values[44]
                    + 0.12 * sample.next.collapse_risk();
                if priority > best_priority {
                    best_priority = priority;
                    idx = candidate;
                }
            }
            let sample = self.replay[idx].clone();
            for (mi, model) in self.models.iter_mut().enumerate() {
                if self.rng.gen::<f32>() < 0.82 || mi == idx % model_len {
                    let novelty_weight = 0.75 + 0.50 * sample.reward.abs();
                    total += model.train(&sample, 0.0035 * novelty_weight);
                    n += 1;
                }
            }
        }
        if n > 0 {
            let loss = total / n as f32;
            self.train_loss_ema += 0.08 * (loss - self.train_loss_ema);
            self.updates = self.updates.saturating_add(1);
        }
    }
    fn prediction_horizon(&self, state: &ReactorState) -> f32 {
        let mut s = state.clone();
        let mut horizon = 0usize;
        for _ in 0..8 {
            let p = self.predict(&s, ActionCommand::default());
            if p.disagreement > 0.24 {
                break;
            }
            let d = s.distance(&p.mean);
            if d > 0.22 {
                break;
            }
            s = p.mean;
            horizon += 1;
        }
        (horizon as f32 / 8.0).clamp(0.0, 1.0)
    }
    fn lyapunov_proxy(&self, state: &ReactorState) -> f32 {
        let mut a = state.clone();
        let mut b = state.clone();
        b.values[3] = (b.values[3] + 0.005).clamp(0.0, 1.0);
        b.values[12] = (b.values[12] + 0.005).clamp(0.0, 1.0);
        let d0 = a.distance(&b).max(1e-6);
        for _ in 0..4 {
            a = self.predict(&a, ActionCommand::default()).mean;
            b = self.predict(&b, ActionCommand::default()).mean;
        }
        let growth = (a.distance(&b).max(1e-6) / d0).ln() / 4.0;
        (0.5 + 0.35 * growth).clamp(0.0, 1.0)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct AttractorEntry {
    state: ReactorState,
    control: SynthesisControl,
    born_step: u64,
    last_recall_step: u64,
    quality: f32,
    recoverability: f32,
    recurrence_count: u32,
    entry_action: ControlAction,
    exit_values: [f32; ACTION_COUNT],
}
#[derive(Clone, Debug, Serialize, Deserialize)]
struct AttractorMemory {
    entries: VecDeque<AttractorEntry>,
    candidates: u64,
    stored_total: u64,
    recalls: u64,
    replacements: u64,
    cooldown_rejects: u64,
}
impl Default for AttractorMemory {
    fn default() -> Self {
        Self {
            entries: VecDeque::new(),
            candidates: 0,
            stored_total: 0,
            recalls: 0,
            replacements: 0,
            cooldown_rejects: 0,
        }
    }
}
impl AttractorMemory {
    fn nearest_index(&self, state: &ReactorState) -> Option<(usize, f32)> {
        self.entries
            .iter()
            .enumerate()
            .map(|(i, a)| (i, state.distance(&a.state)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }
    fn recurrence(&self, state: &ReactorState, step: u64) -> f32 {
        self.entries
            .iter()
            .filter(|a| step.saturating_sub(a.born_step) >= ATTRACTOR_MIN_AGE)
            .map(|a| (1.0 - state.distance(&a.state) * 5.0).clamp(0.0, 1.0) * a.quality)
            .fold(0.0f32, |a, b| a.max(b))
    }
    fn maybe_store(
        &mut self,
        state: &ReactorState,
        control: SynthesisControl,
        action: ControlAction,
        step: u64,
        quality: f32,
        recoverability: f32,
    ) {
        self.candidates = self.candidates.saturating_add(1);
        let nearest = self.nearest_index(state).map(|(_, d)| d).unwrap_or(1.0);
        if quality < 0.48 || nearest < 0.075 {
            return;
        }
        let entry = AttractorEntry {
            state: state.clone(),
            control,
            born_step: step,
            // The `recurrence_count == 0` exemption below marks this as an
            // unconsumed bridge; the cooldown starts only after first recall.
            last_recall_step: step,
            quality,
            recoverability,
            recurrence_count: 0,
            entry_action: action,
            exit_values: [0.0; ACTION_COUNT],
        };
        if self.entries.len() < ATTRACTOR_SLOTS {
            self.entries.push_back(entry);
        } else if let Some((idx, _)) = self.entries.iter().enumerate().min_by(|(_, a), (_, b)| {
            (a.quality * (0.6 + 0.4 * a.recoverability))
                .partial_cmp(&(b.quality * (0.6 + 0.4 * b.recoverability)))
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            self.entries[idx] = entry;
            self.replacements = self.replacements.saturating_add(1);
        }
        self.stored_total = self.stored_total.saturating_add(1);
    }
    fn update_exit(
        &mut self,
        previous: &ReactorState,
        action: ControlAction,
        reward: f32,
        recoverability: f32,
    ) {
        if let Some((idx, distance)) = self.nearest_index(previous) {
            if distance < 0.10 {
                let a = &mut self.entries[idx];
                a.exit_values[action.index()] += 0.12 * (reward - a.exit_values[action.index()]);
                a.recoverability += 0.08 * (recoverability - a.recoverability);
                a.recurrence_count = a.recurrence_count.saturating_add(1);
            }
        }
    }
    fn has_recallable(&self, step: u64) -> bool {
        self.entries.iter().any(|a| {
            step.saturating_sub(a.born_step) >= ATTRACTOR_MIN_AGE
                && (a.recurrence_count == 0
                    || step.saturating_sub(a.last_recall_step) >= ATTRACTOR_RECALL_COOLDOWN)
        })
    }
    fn recall(&mut self, current: &ReactorState, step: u64) -> Option<(SynthesisControl, f32)> {
        let choice = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, a)| step.saturating_sub(a.born_step) >= ATTRACTOR_MIN_AGE)
            .filter(|(_, a)| {
                a.recurrence_count == 0
                    || step.saturating_sub(a.last_recall_step) >= ATTRACTOR_RECALL_COOLDOWN
            })
            .map(|(i, a)| {
                let similarity = (1.0 - current.distance(&a.state) * 3.0).clamp(0.0, 1.0);
                let exit_option = a
                    .exit_values
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max)
                    .max(0.0);
                (
                    i,
                    similarity * a.quality * (0.65 + 0.35 * a.recoverability) + 0.12 * exit_option,
                )
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        if let Some((idx, strength)) = choice {
            self.entries[idx].last_recall_step = step;
            self.entries[idx].recurrence_count =
                self.entries[idx].recurrence_count.saturating_add(1);
            self.recalls = self.recalls.saturating_add(1);
            Some((self.entries[idx].control, strength.clamp(0.0, 1.0)))
        } else {
            if !self.entries.is_empty() {
                self.cooldown_rejects = self.cooldown_rejects.saturating_add(1);
            }
            None
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ProbeEvent {
    start_step: u64,
    command: ActionCommand,
    predicted_baseline: ReactorState,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
struct ProbeController {
    next_probe_step: u64,
    active: Option<ProbeEvent>,
    susceptibility_ema: f32,
    completed: u64,
    enabled: bool,
}
impl ProbeController {
    fn new(enabled: bool) -> Self {
        Self {
            next_probe_step: PROBE_EVERY,
            active: None,
            susceptibility_ema: 0.5,
            completed: 0,
            enabled,
        }
    }
    fn maybe_begin(
        &mut self,
        step: u64,
        state: Option<&ReactorState>,
        ensemble: &TransitionEnsemble,
        health: f32,
        escape: f32,
    ) {
        if !self.enabled
            || self.active.is_some()
            || step < self.next_probe_step
            || health < 0.50
            || escape > 0.05
        {
            return;
        }
        let Some(state) = state else {
            return;
        };
        let action = match self.completed % 4 {
            0 => ControlAction::Widen,
            1 => ControlAction::Resonate,
            2 => ControlAction::Inharmonic,
            _ => ControlAction::Turbulence,
        };
        let command = ActionCommand::new(action, 0.14);
        let mut predicted = state.clone();
        for _ in 0..PROBE_DELAY {
            predicted = ensemble.predict(&predicted, ActionCommand::default()).mean;
        }
        self.active = Some(ProbeEvent {
            start_step: step,
            command,
            predicted_baseline: predicted,
        });
    }
    fn overlay(&self, step: u64) -> Option<ActionCommand> {
        self.active
            .as_ref()
            .filter(|p| step < p.start_step + PROBE_DELAY)
            .map(|p| p.command)
    }
    fn observe(&mut self, step: u64, actual: &ReactorState) {
        // The chunk indexed `step` has just completed. A probe beginning at S
        // and lasting D chunks therefore finishes after chunks S..S+D-1.
        let done = self
            .active
            .as_ref()
            .map(|p| step.saturating_add(1) >= p.start_step.saturating_add(PROBE_DELAY))
            .unwrap_or(false);
        if !done {
            return;
        }
        if let Some(p) = self.active.take() {
            let response = (actual.distance(&p.predicted_baseline) / p.command.intensity.max(0.05)
                * 2.5)
                .clamp(0.0, 1.0);
            self.susceptibility_ema += 0.16 * (response - self.susceptibility_ema);
            self.completed = self.completed.saturating_add(1);
            self.next_probe_step = step.saturating_add(PROBE_EVERY);
        }
    }
}

#[derive(Clone, Debug)]
struct BeamNode {
    state: ReactorState,
    score: f32,
    first: ActionCommand,
    uncertainty: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ReactorPlanner {
    ensemble: TransitionEnsemble,
    cached_scores: [f32; ACTION_COUNT],
    cached_intensities: [f32; ACTION_COUNT],
    cached_option_values: [f32; ACTION_COUNT],
    last_plan_step: u64,
    plan_count: u64,
    last_trigger: String,
    last_disagreement: f32,
    last_option_value: f32,
    last_prediction_horizon: f32,
    last_lyapunov: f32,
    last_advantage: f32,
}
impl ReactorPlanner {
    fn new(seed: u64) -> Self {
        Self {
            ensemble: TransitionEnsemble::new(seed),
            cached_scores: [0.0; ACTION_COUNT],
            cached_intensities: [0.55; ACTION_COUNT],
            cached_option_values: [0.5; ACTION_COUNT],
            last_plan_step: 0,
            plan_count: 0,
            last_trigger: "bootstrap".to_string(),
            last_disagreement: 1.0,
            last_option_value: 0.5,
            last_prediction_horizon: 0.0,
            last_lyapunov: 0.5,
            last_advantage: 0.0,
        }
    }
    fn should_plan(
        &self,
        step: u64,
        state: Option<&ReactorState>,
        manifold: &CriticalManifold,
        adaptive: &AdaptiveDynamics,
        meta: &MetaModel,
        action_age: u32,
        config: PlannerConfig,
    ) -> Option<&'static str> {
        let elapsed = step.saturating_sub(self.last_plan_step);
        let Some(state) = state else {
            return if elapsed >= config.min_interval {
                Some("bootstrap")
            } else {
                None
            };
        };
        if elapsed >= config.max_interval {
            return Some("deadline");
        }
        if elapsed < config.min_interval {
            return None;
        }
        if state.collapse_risk() > 0.58 {
            return Some("collapse-risk");
        }
        if manifold.health < 0.62 {
            return Some("critical-drift");
        }
        if adaptive.stagnation > 0.38 {
            return Some("stagnation");
        }
        if meta.surprise() > 0.18 {
            return Some("self-surprise");
        }
        if self.last_disagreement > 0.22 {
            return Some("model-disagreement");
        }
        if state.values[44] > 0.72 {
            return Some("novel-regime");
        }
        if state.values[46] > 0.72 {
            return Some("attractor-window");
        }
        if action_age > 14 {
            return Some("action-residence");
        }
        if manifold.order_risk > 0.58 || manifold.chaos_risk > 0.62 {
            return Some("risk-boundary");
        }
        None
    }
    fn intensity_candidates(
        action: ControlAction,
        manifold: &CriticalManifold,
        adaptive: &AdaptiveDynamics,
    ) -> [f32; 3] {
        if action == ControlAction::Hold {
            return [0.0, 0.0, 0.0];
        }
        let base = match action {
            ControlAction::Explore | ControlAction::Turbulence => {
                0.42 + 0.48 * manifold.order_risk.max(adaptive.stagnation)
            }
            ControlAction::Crystallize | ControlAction::Contract | ControlAction::Harmonic => {
                0.36 + 0.42 * manifold.chaos_risk
            }
            ControlAction::Recall => 0.45 + 0.25 * adaptive.activity_health,
            _ => 0.50 + 0.22 * (1.0 - manifold.health),
        }
        .clamp(0.22, 0.92);
        [
            (base * 0.62).clamp(0.12, 1.0),
            base,
            (base * 1.28).clamp(0.12, 1.0),
        ]
    }
    fn state_score(state: &ReactorState, manifold: &CriticalManifold, uncertainty: f32) -> f32 {
        let viable = state.viable_complexity(manifold);
        let critical = manifold.score_vector(&std::array::from_fn(|i| state.values[20 + i]));
        let novelty = 0.55 * state.values[44] + 0.45 * state.values[45];
        let information_gain = uncertainty * (1.0 - state.collapse_risk());
        0.48 * viable
            + 0.22 * critical
            + 0.12 * novelty
            + 0.10 * state.values[47]
            + 0.08 * information_gain
            - 0.48 * state.collapse_risk()
    }
    fn option_value(
        &self,
        state: &ReactorState,
        manifold: &CriticalManifold,
        adaptive: &AdaptiveDynamics,
        motif_available: bool,
    ) -> f32 {
        let mut successors = Vec::with_capacity(ACTION_COUNT);
        let mut healthy = 0.0f32;
        for action in ControlAction::ALL {
            if action == ControlAction::Recall && !motif_available {
                continue;
            }
            let intensity = Self::intensity_candidates(action, manifold, adaptive)[1];
            let p = self
                .ensemble
                .predict(state, ActionCommand::new(action, intensity));
            if p.mean.collapse_risk() < 0.55 {
                healthy += 1.0;
            }
            successors.push(p.mean);
        }
        if successors.len() < 2 {
            return 0.0;
        }
        let mut diversity = 0.0f32;
        let mut pairs = 0usize;
        for i in 0..successors.len() {
            for j in i + 1..successors.len() {
                diversity += successors[i].distance(&successors[j]);
                pairs += 1;
            }
        }
        let diversity = (diversity / pairs.max(1) as f32 * 5.0).clamp(0.0, 1.0);
        let healthy_fraction = healthy / successors.len() as f32;
        (0.55 * healthy_fraction + 0.45 * diversity).clamp(0.0, 1.0)
    }
    #[allow(clippy::too_many_arguments)]
    fn plan(
        &mut self,
        step: u64,
        state: &ReactorState,
        neural_prior: [f32; ACTION_COUNT],
        manifold: &CriticalManifold,
        adaptive: &AdaptiveDynamics,
        motif_available: bool,
        config: PlannerConfig,
        trigger: &str,
    ) {
        let readiness = self.ensemble.readiness();
        // Before the compact world model has enough transitions, expensive
        // deep rollouts mostly amplify random initialization. Grow planning
        // depth with earned readiness so interesting behavior appears quickly
        // and the phone spends early compute on learning real transitions.
        let plan_horizon = if readiness < 0.15 {
            1
        } else if readiness < 0.40 {
            config.horizon.min(2)
        } else {
            config.horizon
        };
        let plan_beam = if readiness < 0.15 {
            config.beam_width.min(2).max(1)
        } else {
            config.beam_width.max(1)
        };
        let mut best_commands = [ActionCommand::default(); ACTION_COUNT];
        // Compute HOLD over the same horizon used by candidate beams. Without
        // this, a multi-step candidate is compared against a one-step baseline
        // and receives a systematic artificial advantage.
        let hold_command = ActionCommand::default();
        let mut hold_pred = self.ensemble.predict(state, hold_command);
        let mut hold_state = hold_pred.mean.clone();
        let mut hold_uncertainty = hold_pred.disagreement;
        let mut hold_score = Self::state_score(&hold_state, manifold, hold_uncertainty);
        for depth in 1..plan_horizon {
            hold_pred = self.ensemble.predict(&hold_state, hold_command);
            hold_state = hold_pred.mean;
            hold_uncertainty = hold_uncertainty.max(hold_pred.disagreement);
            hold_score += 0.82f32.powi(depth as i32)
                * Self::state_score(&hold_state, manifold, hold_pred.disagreement);
        }
        let hold_option = self.option_value(&hold_state, manifold, adaptive, motif_available);
        hold_score += 0.34 * hold_option - 0.12 * hold_uncertainty * readiness;
        let prior_min = neural_prior.iter().copied().fold(f32::INFINITY, f32::min);
        let prior_max = neural_prior
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let prior_norm = |i: usize| {
            if prior_max > prior_min + 1e-6 {
                (neural_prior[i] - prior_min) / (prior_max - prior_min)
            } else {
                0.5
            }
        };
        let mut roots = Vec::new();
        for action in ControlAction::ALL {
            if action == ControlAction::Recall && !motif_available {
                self.cached_scores[action.index()] = -1.0;
                continue;
            }
            let mut best = f32::NEG_INFINITY;
            let mut best_command = ActionCommand::new(action, 0.5);
            let mut best_state = state.clone();
            let mut best_uncertainty = 0.0;
            for intensity in Self::intensity_candidates(action, manifold, adaptive) {
                let command = ActionCommand::new(action, intensity);
                let prediction = self.ensemble.predict(state, command);
                let action_cost = 0.035 * intensity * intensity
                    + if action == ControlAction::Hold {
                        0.025 * adaptive.stagnation
                    } else {
                        0.0
                    };
                let score = Self::state_score(&prediction.mean, manifold, prediction.disagreement)
                    + 0.10 * prior_norm(action.index()) * (1.0 - readiness)
                    - action_cost;
                if score > best {
                    best = score;
                    best_command = command;
                    best_state = prediction.mean;
                    best_uncertainty = prediction.disagreement;
                }
            }
            best_commands[action.index()] = best_command;
            roots.push(BeamNode {
                state: best_state,
                score: best,
                first: best_command,
                uncertainty: best_uncertainty,
            });
        }
        roots.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        roots.truncate(plan_beam);
        let expand_actions = [
            ControlAction::Hold,
            ControlAction::Explore,
            ControlAction::Resonate,
            ControlAction::Turbulence,
            ControlAction::Widen,
            ControlAction::Contract,
            ControlAction::Recall,
        ];
        let mut beam = roots;
        for depth in 1..plan_horizon {
            let mut next_beam = Vec::new();
            for node in &beam {
                for &action in &expand_actions {
                    if action == ControlAction::Recall && !motif_available {
                        continue;
                    }
                    let command = best_commands[action.index()];
                    let prediction = self.ensemble.predict(&node.state, command);
                    let discount = 0.82f32.powi(depth as i32);
                    let incremental =
                        Self::state_score(&prediction.mean, manifold, prediction.disagreement)
                            - 0.025 * command.intensity.powi(2);
                    next_beam.push(BeamNode {
                        state: prediction.mean,
                        score: node.score + discount * incremental,
                        first: node.first,
                        uncertainty: node.uncertainty.max(prediction.disagreement),
                    });
                }
            }
            next_beam.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            next_beam.truncate(plan_beam);
            beam = next_beam;
        }
        self.cached_scores.fill(-1.0);
        self.cached_option_values.fill(0.0);
        for node in &beam {
            let option = self.option_value(&node.state, manifold, adaptive, motif_available);
            let final_score = node.score + 0.34 * option - 0.12 * node.uncertainty * readiness;
            let i = node.first.action.index();
            let advantage = final_score - hold_score;
            if advantage > self.cached_scores[i] {
                self.cached_scores[i] = advantage;
                self.cached_intensities[i] = node.first.intensity;
                self.cached_option_values[i] = option;
            }
        }
        // Every action retains a one-step estimate even if beam pruning removed it.
        for action in ControlAction::ALL {
            let i = action.index();
            if self.cached_scores[i] <= -0.999
                && !(action == ControlAction::Recall && !motif_available)
            {
                let command = best_commands[i];
                let p = self.ensemble.predict(state, command);
                let option = self.option_value(&p.mean, manifold, adaptive, motif_available);
                self.cached_scores[i] = Self::state_score(&p.mean, manifold, p.disagreement)
                    + 0.28 * option
                    - hold_score;
                self.cached_intensities[i] = command.intensity;
                self.cached_option_values[i] = option;
            }
        }
        // Blend in the neural self-model as a prior until the online world model earns trust.
        for i in 0..ACTION_COUNT {
            self.cached_scores[i] = readiness * self.cached_scores[i]
                + (1.0 - readiness) * (prior_norm(i) - 0.5) * 0.55;
        }
        let best_i = self
            .cached_scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        self.last_advantage = self.cached_scores[best_i];
        self.last_option_value = self.cached_option_values[best_i];
        self.last_disagreement = beam
            .iter()
            .filter(|node| node.first.action.index() == best_i)
            .max_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|node| node.uncertainty)
            .unwrap_or_else(|| {
                self.ensemble
                    .predict(state, best_commands[best_i])
                    .disagreement
            });
        self.last_prediction_horizon = self.ensemble.prediction_horizon(state);
        self.last_lyapunov = self.ensemble.lyapunov_proxy(state);
        self.last_plan_step = step;
        self.plan_count = self.plan_count.saturating_add(1);
        self.last_trigger = trigger.to_string();
    }
    fn command_for(&self, action: ControlAction, adaptive: &AdaptiveDynamics) -> ActionCommand {
        let planned = self.cached_intensities[action.index()];
        let fallback = match action {
            ControlAction::Explore | ControlAction::Turbulence => 0.50 + 0.38 * adaptive.stagnation,
            ControlAction::Hold => 0.0,
            _ => 0.52,
        };
        let readiness = self.ensemble.readiness();
        ActionCommand::new(action, readiness * planned + (1.0 - readiness) * fallback)
    }
}

fn json_put<T: Serialize>(
    map: &mut serde_json::Map<String, serde_json::Value>,
    key: &str,
    value: T,
) {
    map.insert(
        key.to_string(),
        serde_json::to_value(value).unwrap_or(serde_json::Value::Null),
    );
}

// Deterministic re-initialization of a fresh varmap. candle's default init
// draws from a thread-local RNG we cannot seed portably on CPU, so on a
// fresh start we overwrite every tensor: Kaiming-uniform for weights
// (bound = 1/sqrt(fan_in)), zeros for biases, N(0, 0.1) for the KAN "weights"
// vectors, and the frequency constants are skipped (they are Const inits).
fn deterministic_reinit(varmap: &VarMap, seed: u64, device: &Device) -> Result<usize> {
    let mut rng = RuntimeRng::seed_from_u64(seed ^ 0x5EED_1417);
    let data = varmap.data().lock().unwrap();
    let mut names: Vec<String> = data.keys().cloned().collect();
    names.sort(); // fixed iteration order => fixed draws
    let mut count = 0usize;
    for name in names {
        let var = &data[&name];
        let dims = var.as_tensor().dims().to_vec();
        if name.contains("base_freq") {
            continue;
        }
        let n: usize = dims.iter().product();
        let vals: Vec<f32> = if name.ends_with("weights") {
            (0..n).map(|_| rng_normal(&mut rng) * 0.1).collect()
        } else if name.contains("bias") {
            vec![0.0; n]
        } else {
            let fan_in: usize = if dims.len() >= 2 {
                dims[1..].iter().product()
            } else {
                dims[0]
            };
            let bound = 1.0 / (fan_in.max(1) as f32).sqrt();
            (0..n).map(|_| rng.gen_range(-bound..bound)).collect()
        };
        var.set(&Tensor::from_vec(vals, dims, device)?)
            .map_err(anyhow::Error::msg)?;
        count += 1;
    }
    Ok(count)
}

// --- AUDIO TARGET LOADER ---
struct TargetAudioLoader {
    buffers: Vec<(Vec<f32>, Vec<f32>)>,
}

impl TargetAudioLoader {
    fn new(path: &str) -> Result<Self> {
        let mut buffers = Vec::new();
        let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(path)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .collect();
        paths.sort(); // deterministic buffer indexing across runs
        for p in paths {
            let is_out = p
                .file_name()
                .and_then(|n| n.to_str())
                .map_or(false, |n| n == "rust_ecosystem_out.wav");
            if p.extension().map_or(false, |ext| ext == "wav") && !is_out {
                match Self::load_wav(&p) {
                    Ok((l, r)) if l.len() >= CHUNK_SIZE => {
                        println!(
                            "--> Loaded target audio: {:?} ({} samples/ch @ 48k stereo)",
                            p,
                            l.len()
                        );
                        buffers.push((l, r));
                    }
                    Ok((l, _)) => println!(
                        "--> Skipping {:?}: only {} samples after resample",
                        p,
                        l.len()
                    ),
                    Err(e) => println!("--> Skipping {:?}: {}", p, e),
                }
            }
        }
        if buffers.is_empty() {
            anyhow::bail!("No usable training audio found in {}", path);
        }
        Ok(Self { buffers })
    }

    fn load_wav(p: &std::path::Path) -> Result<(Vec<f32>, Vec<f32>)> {
        let mut reader = hound::WavReader::open(p)?;
        let spec = reader.spec();
        let raw: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
            (hound::SampleFormat::Float, 32) => {
                reader.samples::<f32>().filter_map(|s| s.ok()).collect()
            }
            (hound::SampleFormat::Int, 16) => reader
                .samples::<i16>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / 32768.0)
                .collect(),
            (hound::SampleFormat::Int, bits @ (24 | 32)) => {
                let scale = (1i64 << (bits - 1)) as f32;
                reader
                    .samples::<i32>()
                    .filter_map(|s| s.ok())
                    .map(|s| s as f32 / scale)
                    .collect()
            }
            (fmt, bits) => anyhow::bail!("unsupported WAV format {:?}/{} bits", fmt, bits),
        };
        if raw.is_empty() {
            anyhow::bail!("no samples decoded");
        }
        let ch = spec.channels as usize;
        let (left, right): (Vec<f32>, Vec<f32>) = if ch >= 2 {
            let l = raw.iter().step_by(ch).copied().collect();
            let r = raw.iter().skip(1).step_by(ch).copied().collect();
            (l, r)
        } else {
            (raw.clone(), raw)
        };
        if spec.sample_rate == SAMPLE_RATE {
            return Ok((left, right));
        }
        Ok((
            Self::resample(&left, spec.sample_rate),
            Self::resample(&right, spec.sample_rate),
        ))
    }

    fn resample(x: &[f32], from_rate: u32) -> Vec<f32> {
        // NOTE: linear interpolation with no anti-alias prefilter — >48k sources
        // will fold a little HF hash into the targets. Acceptable for now.
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

    // Min-of-K sampling: return K candidate chunks stacked (K, 2, CHUNK). The
    // caller computes a cheap coarse mimic per candidate and keeps the nearest —
    // regressing to A mode of the target set instead of the blur of all modes.
    fn sample_chunks(&self, k: usize, rng: &mut RuntimeRng, device: &Device) -> CResult<Tensor> {
        let mut data = Vec::with_capacity(k * 2 * CHUNK_SIZE);
        for _ in 0..k {
            let idx = rng.gen_range(0..self.buffers.len());
            let (l, r) = &self.buffers[idx];
            let start = if l.len() == CHUNK_SIZE {
                0
            } else {
                rng.gen_range(0..(l.len() - CHUNK_SIZE + 1))
            };
            data.extend_from_slice(&l[start..start + CHUNK_SIZE]);
            data.extend_from_slice(&r[start..start + CHUNK_SIZE]);
        }
        Tensor::from_vec(data, (k, 2, CHUNK_SIZE), device)
    }
}

// --- CUSTOM MODULES & MATH ---
struct Tanh;
impl Module for Tanh {
    fn forward(&self, xs: &Tensor) -> CResult<Tensor> {
        xs.tanh()
    }
}

struct Sigmoid;
impl Module for Sigmoid {
    fn forward(&self, xs: &Tensor) -> CResult<Tensor> {
        candle_nn::ops::sigmoid(xs)
    }
}

struct Relu;
impl Module for Relu {
    fn forward(&self, xs: &Tensor) -> CResult<Tensor> {
        xs.relu()
    }
}

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

// 2x2 average pooling for the renormalization-group loss on 2D fields.
fn decimate2_2d(x: &Tensor) -> CResult<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    x.reshape((b, c, h / 2, 2, w / 2, 2))?.mean(5)?.mean(3)
}

fn calculate_cross_layer_synergy_tensor(micro: &Tensor, macro_t: &Tensor) -> CResult<Tensor> {
    let micro_flat = micro.flatten_all()?;
    let macro_flat = macro_t.flatten_all()?;
    let micro_mean = micro_flat.mean_all()?;
    let macro_mean = macro_flat.mean_all()?;
    let micro_norm = micro_flat.broadcast_sub(&micro_mean)?;
    let macro_norm = macro_flat.broadcast_sub(&macro_mean)?;
    // Covariance through tanh: bounded gradient (Pearson's 1/v^2 explodes near
    // zero variance). Same device as v3.
    let cross_cov = micro_norm.mul(&macro_norm)?.mean_all()?;
    cross_cov.affine(10.0, 0.0)?.tanh()
}

// Circular (torus) 2D convolution: wrap-pad both spatial dims, then conv with
// padding 0. The torus removes boundary artifacts that would otherwise pin
// patterns to the edges.
fn conv2d_torus(
    in_c: usize,
    out_c: usize,
    k: usize,
    vb: VBV,
) -> Result<Box<dyn Fn(&Tensor) -> CResult<Tensor>>> {
    let pad = k / 2;
    let config = Conv2dConfig {
        padding: 0,
        stride: 1,
        ..Default::default()
    };
    let conv = candle_nn::conv2d(in_c, out_c, k, config, vb)?;
    Ok(Box::new(move |x: &Tensor| {
        if pad == 0 {
            return conv.forward(x);
        }
        let h = x.dim(D::Minus2)?;
        let top = x.narrow(D::Minus2, h - pad, pad)?;
        let bot = x.narrow(D::Minus2, 0, pad)?;
        let xv = Tensor::cat(&[&top, x, &bot], D::Minus2)?;
        let w = xv.dim(D::Minus1)?;
        let left = xv.narrow(D::Minus1, w - pad, pad)?;
        let right = xv.narrow(D::Minus1, 0, pad)?;
        let xh = Tensor::cat(&[&left, &xv, &right], D::Minus1)?;
        conv.forward(&xh)
    }))
}

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
    let len = x.dim(D::Minus1)?;
    if delay_samples == 0 {
        return Ok(x.clone());
    }
    let dev = x.device();
    let zero = Tensor::zeros((1, delay_samples), DType::F32, dev)?;
    let cut = x.narrow(D::Minus1, 0, len - delay_samples)?;
    Tensor::cat(&[&zero, &cut], D::Minus1)
}

// --- SPECTRAL PROJECTOR ---
struct SpectralProjector {
    window: Tensor, // pre-unsqueezed (1, n)
    cos_m: Tensor,
    sin_m: Tensor,
}
impl SpectralProjector {
    fn new(device: &Device) -> CResult<Self> {
        Self::new_with(CHUNK_SIZE, SPEC_BINS, device)
    }
    fn new_with(n: usize, bins: usize, device: &Device) -> CResult<Self> {
        let mut win = Vec::with_capacity(n);
        for i in 0..n {
            win.push(0.5 - 0.5 * (TWO_PI * i as f32 / (n as f32 - 1.0)).cos());
        }
        let f_lo = 40.0f32;
        let f_hi = 8000.0f32;
        let mut cos_v = vec![0.0f32; n * bins];
        let mut sin_v = vec![0.0f32; n * bins];
        for k in 0..bins {
            let frac = k as f32 / (bins as f32 - 1.0);
            let omega = TWO_PI * f_lo * (f_hi / f_lo).powf(frac) / SAMPLE_RATE as f32;
            for i in 0..n {
                cos_v[i * bins + k] = (omega * i as f32).cos();
                sin_v[i * bins + k] = (omega * i as f32).sin();
            }
        }
        Ok(Self {
            window: Tensor::new(win, device)?.unsqueeze(0)?,
            cos_m: Tensor::from_vec(cos_v, (n, bins), device)?,
            sin_m: Tensor::from_vec(sin_v, (n, bins), device)?,
        })
    }
    fn log_mag(&self, x: &Tensor) -> CResult<Tensor> {
        let xw = x.broadcast_mul(&self.window)?;
        let re = xw.matmul(&self.cos_m)?;
        let im = xw.matmul(&self.sin_m)?;
        re.sqr()?
            .add(&im.sqr()?)?
            .affine(1.0, 1e-3)?
            .log()?
            .affine(0.5, 0.0)
    }
}

// =====================================================================
// POST-NEURAL SPECTRAL ECOLOGY
// =====================================================================
// These layers are deliberately host-side.  They enrich the final signal and
// are included in the post-DSP self-observation loop, while the differentiable
// core still learns from its pre-DSP signal.

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ModalResonatorBank {
    states_l: [[f32; 2]; MODAL_MODES],
    states_r: [[f32; 2]; MODAL_MODES],
}
impl Default for ModalResonatorBank {
    fn default() -> Self {
        Self {
            states_l: [[0.0; 2]; MODAL_MODES],
            states_r: [[0.0; 2]; MODAL_MODES],
        }
    }
}
impl ModalResonatorBank {
    fn new() -> Self {
        Self::default()
    }
    fn process(
        &mut self,
        samples_l: &mut [f32],
        samples_r: &mut [f32],
        regions: &[f32; REGION_COUNT],
        control: &SynthesisControl,
        surprise: f32,
    ) {
        const RATIOS: [f32; MODAL_MODES] = [1.0, 1.4142, 1.875, 2.618, 3.0, 3.732, 4.236, 5.125];
        let root = 82.0 + 62.0 * regions.iter().take(4).copied().sum::<f32>() * 0.25;
        let wet =
            (0.035 + 0.16 * control.resonator_drive + 0.08 * surprise + 0.06 * control.recall_mix)
                .clamp(0.02, 0.30);
        let process_channel = |samples: &mut [f32],
                               states: &mut [[f32; 2]; MODAL_MODES],
                               right: bool| {
            for mode in 0..MODAL_MODES {
                let regional = regions[(mode * 2) % REGION_COUNT].clamp(-1.0, 1.0);
                let warp = 1.0 + control.inharmonicity * (0.025 * mode as f32 + 0.08 * regional);
                let freq = (root * RATIOS[mode] * warp).clamp(35.0, 9200.0);
                let q = (18.0
                    + 58.0 * (1.0 - surprise)
                    + 24.0 * regions[(mode * 2 + 1) % REGION_COUNT].abs())
                .clamp(6.0, 110.0);
                let damping = std::f32::consts::PI * freq / (q * SAMPLE_RATE as f32);
                let omega = TWO_PI * freq / SAMPLE_RATE as f32;
                let r = (-damping).exp();
                let c1 = 2.0 * r * omega.cos();
                let c2 = -r * r;
                let scale = (1.0 - r) * (0.20 + 0.45 * control.resonator_drive);
                let pan = mode as f32 / (MODAL_MODES - 1) as f32 * 2.0 - 1.0;
                let gain = if right {
                    ((1.0 + pan) * 0.5).sqrt()
                } else {
                    ((1.0 - pan) * 0.5).sqrt()
                };
                let state = &mut states[mode];
                for sample in samples.iter_mut() {
                    let next = *sample * scale + c1 * state[0] + c2 * state[1];
                    state[1] = state[0];
                    state[0] = next;
                    *sample = (*sample + next * wet * gain).clamp(-1.4, 1.4);
                }
            }
        };
        rayon::join(
            || process_channel(samples_l, &mut self.states_l, false),
            || process_channel(samples_r, &mut self.states_r, true),
        );
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SpectralNoiseBank {
    low_l: [f32; NOISE_BANDS],
    band_l: [f32; NOISE_BANDS],
    low_r: [f32; NOISE_BANDS],
    band_r: [f32; NOISE_BANDS],
}
impl Default for SpectralNoiseBank {
    fn default() -> Self {
        Self {
            low_l: [0.0; NOISE_BANDS],
            band_l: [0.0; NOISE_BANDS],
            low_r: [0.0; NOISE_BANDS],
            band_r: [0.0; NOISE_BANDS],
        }
    }
}
impl SpectralNoiseBank {
    fn process(
        &mut self,
        samples_l: &mut [f32],
        samples_r: &mut [f32],
        regions: &[f32; REGION_COUNT],
        control: &SynthesisControl,
        rng: &mut RuntimeRng,
    ) {
        if control.noise_level <= 1e-4 {
            return;
        }
        const BASE_FREQS: [f32; NOISE_BANDS] = [110.0, 430.0, 1650.0, 5200.0];
        let mut coeff = [0.0f32; NOISE_BANDS];
        let mut damping = [0.0f32; NOISE_BANDS];
        let mut pans = [0.0f32; NOISE_BANDS];
        for b in 0..NOISE_BANDS {
            let region = regions[b * 4..b * 4 + 4].iter().copied().sum::<f32>() * 0.25;
            let freq = (BASE_FREQS[b]
                * (1.0 + 0.38 * region + 0.12 * control.inharmonicity * b as f32))
                .clamp(45.0, 9000.0);
            coeff[b] =
                (2.0 * (std::f32::consts::PI * freq / SAMPLE_RATE as f32).sin()).clamp(0.001, 0.92);
            damping[b] = (0.16 + 0.58 * (1.0 - region.abs())).clamp(0.12, 0.85);
            pans[b] = (region * 0.8 + (b as f32 / 3.0 * 2.0 - 1.0) * 0.35).clamp(-1.0, 1.0);
        }
        let gain = (0.012 + 0.11 * control.noise_level).clamp(0.0, 0.16);
        let noise: Vec<f32> = (0..samples_l.len())
            .map(|_| rng.gen_range(-1.0f32..1.0f32))
            .collect();
        let process_channel = |samples: &mut [f32],
                               low: &mut [f32; NOISE_BANDS],
                               band: &mut [f32; NOISE_BANDS],
                               right: bool| {
            for (sample, &white) in samples.iter_mut().zip(noise.iter()) {
                let input = if right { -white } else { white };
                let mut add = 0.0f32;
                for b in 0..NOISE_BANDS {
                    let high = input - low[b] - damping[b] * band[b];
                    band[b] = (band[b] + coeff[b] * high).clamp(-4.0, 4.0);
                    low[b] = (low[b] + coeff[b] * band[b]).clamp(-4.0, 4.0);
                    let pan_gain = if right {
                        ((1.0 + pans[b]) * 0.5).sqrt()
                    } else {
                        ((1.0 - pans[b]) * 0.5).sqrt()
                    };
                    add += band[b] * pan_gain;
                }
                *sample = (*sample + add * gain).clamp(-1.5, 1.5);
            }
        };
        rayon::join(
            || process_channel(samples_l, &mut self.low_l, &mut self.band_l, false),
            || process_channel(samples_r, &mut self.low_r, &mut self.band_r, true),
        );
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct FractalFDN {
    buffers: Vec<Vec<f32>>,
    indices: Vec<usize>,
    lp_states: [f32; FDN_DELAY_LINES],
}
impl FractalFDN {
    fn new() -> Self {
        let mut buffers = Vec::new();
        let mut indices = Vec::new();
        for &d in &FDN_DELAYS {
            buffers.push(vec![0.0; d]);
            indices.push(0);
        }
        Self {
            buffers,
            indices,
            lp_states: [0.0; FDN_DELAY_LINES],
        }
    }
    fn is_valid(&self) -> bool {
        self.buffers.len() == FDN_DELAY_LINES
            && self.indices.len() == FDN_DELAY_LINES
            && self
                .buffers
                .iter()
                .zip(FDN_DELAYS.iter())
                .all(|(b, &d)| b.len() == d && !b.is_empty())
            && self
                .indices
                .iter()
                .zip(self.buffers.iter())
                .all(|(&i, b)| i < b.len())
    }
    fn process(&mut self, samples: &mut [f32], echo: f32, damping: f32) {
        let mix = [
            [0.5, 0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5, 0.5],
        ];
        let lp_a = damping.clamp(0.08, 0.92);
        let lp_b = 1.0 - lp_a;
        let scale = (0.18 + 0.34 * echo).clamp(0.0, 0.54);
        for x in samples.iter_mut() {
            let mut outs = [0.0; FDN_DELAY_LINES];
            for i in 0..FDN_DELAY_LINES {
                let idx = self.indices[i];
                self.lp_states[i] = self.lp_states[i] * lp_a + self.buffers[i][idx] * lp_b;
                outs[i] = self.lp_states[i];
            }
            for i in 0..FDN_DELAY_LINES {
                let mut sum = 0.0;
                for j in 0..FDN_DELAY_LINES {
                    sum += mix[i][j] * outs[j];
                }
                let idx = self.indices[i];
                self.buffers[i][idx] = (*x + sum * scale).clamp(-2.0, 2.0);
                self.indices[i] = (idx + 1) % self.buffers[i].len();
            }
            let fdn_out = (outs[0] + outs[1] + outs[2] + outs[3]) * 0.25;
            *x = *x * (1.0 - echo * 0.18) + fdn_out * (echo * 0.44);
        }
    }
}

// --- MONITORS ---
// Order-4 permutation entropy at a given ordinal lag over a signal slice.
// Lag 1 = fastest temporal structure; larger lags probe slower structure.
// The MULTI-LAG SPREAD is the predictive-structure instrument: white noise
// scores high at every lag (no structure), frozen tones score low at every
// lag; the interesting regime is high short-lag entropy WITH lower long-lag
// entropy (surprise now, order over time). Logged, not yet in the loss.
fn perm_entropy4(x: &[f32], lag: usize) -> f32 {
    if x.len() < 3 * lag + 1 {
        return 0.0;
    }
    let mut hist = [0u32; 24];
    let n = x.len() - 3 * lag;
    for i in 0..n {
        let w = [x[i], x[i + lag], x[i + 2 * lag], x[i + 3 * lag]];
        let mut c0 = 0usize;
        for k in 1..4 {
            if w[k] < w[0] {
                c0 += 1;
            }
        }
        let mut c1 = 0usize;
        for k in 2..4 {
            if w[k] < w[1] {
                c1 += 1;
            }
        }
        let c2 = if w[3] < w[2] { 1usize } else { 0 };
        hist[c0 * 6 + c1 * 2 + c2] += 1;
    }
    let total = n as f32;
    let mut pe = 0.0f32;
    for &c in &hist {
        if c > 0 {
            let p = c as f32 / total;
            pe -= p * p.ln();
        }
    }
    pe / (24f32).ln()
}

#[derive(Clone, Debug)]
struct PostDspAnalysis {
    observation: AudioObservation,
    json: serde_json::Value,
}

struct SpectralEntropyMonitor {
    history: VecDeque<f32>,
    window: usize,
    fft: std::sync::Arc<dyn rustfft::Fft<f32>>,
    hann: Vec<f32>,
    prev_mags: Vec<f32>,
}
impl SpectralEntropyMonitor {
    fn new(w: usize) -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(CHUNK_SIZE);
        let hann = (0..CHUNK_SIZE)
            .map(|i| 0.5 - 0.5 * (TWO_PI * i as f32 / (CHUNK_SIZE as f32 - 1.0)).cos())
            .collect();
        Self {
            history: VecDeque::with_capacity(w),
            window: w,
            fft,
            hann,
            prev_mags: vec![0.0; CHUNK_SIZE / 2],
        }
    }
    fn analyze(
        &mut self,
        left: &[f32],
        right: &[f32],
        movement: f32,
        synergy: f32,
        field_entropy: f32,
        sigma: f32,
    ) -> PostDspAnalysis {
        let n = left.len().min(right.len()).min(CHUNK_SIZE);
        let mut mono = Vec::with_capacity(n);
        let mut buf = Vec::with_capacity(n);
        let mut sum_sq = 0.0f32;
        let mut peak = 0.0f32;
        let mut side_e = 0.0f32;
        let mut total_e = 0.0f32;
        for i in 0..n {
            let m = (left[i] + right[i]) * 0.5;
            mono.push(m);
            buf.push(Complex::new(m * self.hann[i], 0.0));
            sum_sq += m * m;
            peak = peak.max(m.abs());
            let side = left[i] - right[i];
            side_e += side * side;
            total_e += left[i] * left[i] + right[i] * right[i];
        }
        self.fft.process(&mut buf);
        let mags: Vec<f32> = buf.iter().take(n / 2).map(|c| c.norm()).collect();
        let sum = mags.iter().sum::<f32>() + 1e-8;
        let nb = mags.len().max(1) as f32;
        let mut entropy = 0.0f32;
        let mut log_sum = 0.0f32;
        let (mut c_num, mut c_den) = (0.0f32, 0.0f32);
        let mut flux_num = 0.0f32;
        let bin_hz = SAMPLE_RATE as f32 / n.max(1) as f32;
        for (k, &m) in mags.iter().enumerate() {
            let p = m / sum;
            if p > 1e-8 {
                entropy -= p * p.ln();
            }
            log_sum += (m + 1e-9).ln();
            c_num += m * (k as f32 * bin_hz);
            c_den += m;
            let d = (m - self.prev_mags[k]).max(0.0);
            flux_num += d * d;
        }
        self.prev_mags[..mags.len()].copy_from_slice(&mags);
        let entropy_norm = (entropy / nb.ln().max(1e-6)).clamp(0.0, 1.0);
        let flatness = ((log_sum / nb).exp() / (sum / nb)).clamp(0.0, 1.0);
        let brightness_hz = if c_den > 1e-6 { c_num / c_den } else { 0.0 };
        let centroid_norm = (brightness_hz / 8000.0).clamp(0.0, 1.0);
        let flux = (flux_num.sqrt() / sum).clamp(0.0, 1.0);
        let rms = (sum_sq / n.max(1) as f32).sqrt();
        let crest = (peak / (rms + 1e-6) / 10.0).clamp(0.0, 1.0);
        let width = if total_e > 1e-8 {
            (side_e / total_e).sqrt().clamp(0.0, 1.0)
        } else {
            0.0
        };

        let dec: Vec<f32> = mono.iter().step_by(4).copied().collect();
        let pe1 = perm_entropy4(&dec, 1);
        let pe4 = perm_entropy4(&dec, 4);
        let pe16 = perm_entropy4(&dec, 16);
        let pi_proxy = (pe1 * (1.0 - pe16)).clamp(0.0, 1.0);
        let critical_health = (-4.0 * (sigma - 1.0).abs()).exp().clamp(0.0, 1.0);

        self.history.push_back(entropy_norm);
        if self.history.len() > self.window {
            self.history.pop_front();
        }
        let avg = self.history.iter().sum::<f32>() / self.history.len().max(1) as f32;
        let observation = AudioObservation {
            values: [
                entropy_norm,
                flatness,
                centroid_norm,
                flux,
                rms.clamp(0.0, 1.0),
                crest,
                width,
                (movement / 0.35).clamp(0.0, 1.0),
                synergy.abs().clamp(0.0, 1.0),
                (field_entropy / 3.0).clamp(0.0, 1.0),
                pi_proxy,
                critical_health,
            ],
        };
        let json = serde_json::json!({
            "signal": entropy_norm, "avg": avg, "type": "post_dsp_spectral_ecology",
            "flatness": flatness, "brightness": brightness_hz, "centroid_norm": centroid_norm,
            "flux": flux, "rms": rms, "crest": crest, "width": width,
            "pe1": pe1, "pe4": pe4, "pe16": pe16, "pi_proxy": pi_proxy,
            "structured_complexity": observation.structured_complexity(),
        });
        PostDspAnalysis { observation, json }
    }
    fn history_snapshot(&self) -> Vec<f32> {
        self.history.iter().copied().collect()
    }
    fn prev_mags_snapshot(&self) -> Vec<f32> {
        self.prev_mags.clone()
    }
    fn restore_history(&mut self, values: &[f32]) {
        self.history.clear();
        for &v in values
            .iter()
            .rev()
            .take(self.window)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
        {
            self.history.push_back(v);
        }
    }
    fn restore_prev_mags(&mut self, values: &[f32]) {
        if values.len() == self.prev_mags.len() {
            self.prev_mags.copy_from_slice(values);
        }
    }
}

struct MovementCoherenceMonitor {
    history: VecDeque<f32>,
    window: usize,
}
impl MovementCoherenceMonitor {
    fn new(w: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(w),
            window: w,
        }
    }
    fn analyze(&mut self, m: f32) -> Result<serde_json::Value> {
        self.history.push_back(m);
        if self.history.len() > self.window {
            self.history.pop_front();
        }
        let mut trend = 0.0;
        if self.history.len() >= 10 {
            let n = self.history.len() as f32;
            let x_mean = (n - 1.0) / 2.0;
            let y_mean: f32 = self.history.iter().sum::<f32>() / n;
            let (mut num, mut den) = (0.0, 0.0);
            for (i, &y) in self.history.iter().enumerate() {
                let dx = i as f32 - x_mean;
                num += dx * (y - y_mean);
                den += dx * dx;
            }
            trend = num / (den + 1e-8);
        }
        Ok(serde_json::json!({"signal": m, "trend": trend, "type": "movement_coherence"}))
    }
    fn history_snapshot(&self) -> Vec<f32> {
        self.history.iter().copied().collect()
    }
    fn restore_history(&mut self, values: &[f32]) {
        self.history.clear();
        for &v in values
            .iter()
            .rev()
            .take(self.window)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
        {
            self.history.push_back(v);
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct AudioUncertaintyState {
    spectral: f32,
    movement: f32,
    mimic: f32,
    compositional: f32,
    phi: f32,
    synergy: f32,
    empowerment: f32,
    flatness: f32,
    prediction_error: f32,
    confidence: f32,
}
impl AudioUncertaintyState {
    fn new() -> Self {
        Self {
            spectral: 0.0,
            movement: 0.0,
            mimic: 0.0,
            compositional: 0.0,
            phi: 0.0,
            synergy: 0.0,
            empowerment: 0.0,
            flatness: 0.0,
            prediction_error: 0.5,
            confidence: 0.15,
        }
    }
    fn update(
        &mut self,
        spec_sig: &serde_json::Value,
        move_sig: &serde_json::Value,
        mimic_drift_n: f32,
        syn: f32,
        emp: f32,
        meta: &MetaModel,
    ) {
        let m_trend = move_sig["trend"].as_f64().unwrap_or(0.0) as f32;
        let pe = spec_sig["pe1"].as_f64().unwrap_or(0.5) as f32;
        let pi = spec_sig["pi_proxy"].as_f64().unwrap_or(0.0) as f32;
        let entropy = spec_sig["signal"].as_f64().unwrap_or(0.5) as f32;
        let flux = spec_sig["flux"].as_f64().unwrap_or(0.0) as f32;
        self.flatness = spec_sig["flatness"].as_f64().unwrap_or(0.5) as f32;
        let flatness_shape = (-((self.flatness - 0.22) / 0.24).powi(2)).exp();
        self.phi = (0.55 * pi + 0.30 * pe + 0.15 * entropy).clamp(0.0, 1.0);
        self.spectral =
            (0.34 * entropy + 0.26 * flatness_shape + 0.22 * flux + 0.18 * pi).clamp(0.0, 1.0);
        self.movement = ((-m_trend * 160.0).max(0.0) + meta.surprise() * 0.35).clamp(0.0, 1.0);
        self.mimic = (mimic_drift_n * 10.0).clamp(0.0, 1.0);
        self.synergy = syn;
        self.empowerment = emp;
        self.prediction_error = meta.surprise();
        self.confidence = meta.confidence;
        let instant = self
            .spectral
            .max(self.movement)
            .max(self.prediction_error * 0.75);
        self.compositional = self.compositional * 0.92 + instant * 0.08;
        self.compositional = self.compositional.min(1.0);
    }
    fn branch_aperture(&self) -> f32 {
        (self.spectral.clamp(0.0, 1.0) * 0.18
            + self.movement.clamp(0.0, 1.0) * 0.18
            + self.mimic.clamp(0.0, 1.0) * 0.10
            + self.compositional.clamp(0.0, 1.0) * 0.10
            + self.synergy.abs().clamp(0.0, 1.0) * 0.19
            + self.prediction_error * 0.25)
            .clamp(0.05, 1.0)
    }
}

// =====================================================================
// POTENTIAL CONTROLLER — the whole crash cart as one function
// =====================================================================
// V(s) = k_a(a_mic - 0.5)^2 + B/(1.02 - a_mic)          [micro bowl + rail wall]
//      + k_a(a_mac - 0.4)^2 + B/(1.02 - a_mac)          [macro bowl + rail wall]
//      + k_rho(rho - 0.5)^2                             [coupling band]
//      + G * exp(-(m / w)^2)                            [ridge: flatline is a hilltop]
//      + k_e(e - 0.65)^2                                [metabolic bowl]
//      + k_sig(sigma - 1)^2                             [criticality bowl]
//
// Drift: each actuated coordinate gets gain = clamp(1 - eta * dV/da, lo, hi),
// a multiplicative pull downhill. The barrier derivative B/(1.02-a)^2 is
// negligible mid-range and diverges at the rail — cells cannot weld.
//
// Temperature: T = smooth( stuck + subcritical + stagnation ), where
// stuck = (V above recent floor) * exp(-state_speed / s0). "Stuck" is not a
// threshold event with debounce and refractory clocks; it is DEFINED as
// being uphill while the state barely moves, and it CAUSES heat until the
// system falls off the plateau. Heat is spent four ways: structured macro
// shear, white micro kicks, an LR multiplier, and arbiter softmax temp.
struct PotOut {
    micro_gain: f32,
    macro_gain: f32,
    shear_amp: f32,
    micro_kick: f32,
    temp: f32,
    lr_heat: f64,
    couple_release: f32, // eases the synergy band when over-coupled or hot
    v: f32,
    annealed: bool, // edge: hot episode just cooled below TEMP_COOL
}
#[derive(Clone, Debug, Serialize, Deserialize)]
struct PotentialController {
    prev_state: Option<[f32; 5]>,
    speed_ema: f32,
    v_floor: f32,
    sigma_ema: f32,
    temp: f32,
    was_hot: bool,
    init: bool,
}
impl PotentialController {
    fn new() -> Self {
        Self {
            prev_state: None,
            speed_ema: 0.02,
            v_floor: 0.0,
            sigma_ema: 1.0,
            temp: 0.0,
            was_hot: false,
            init: false,
        }
    }
    fn update(
        &mut self,
        a_mic: f32,
        a_mac: f32,
        coupling: f32,
        movement: f32,
        energy: f32,
        sigma: f32,
        curiosity: f32,
        stagnation: f32,
    ) -> PotOut {
        let rho = coupling.abs();
        self.sigma_ema += 0.05 * (sigma - self.sigma_ema);
        let sig = self.sigma_ema;

        // --- V(s) and its actuated gradients ---
        let bar = |a: f32| POT_BARRIER / (1.02 - a.min(1.01));
        let dbar = |a: f32| {
            let d = 1.02 - a.min(1.01);
            POT_BARRIER / (d * d)
        };
        let v = POT_K_AMP * (a_mic - POT_MICRO_SET).powi(2)
            + bar(a_mic)
            + POT_K_AMP * (a_mac - POT_MACRO_SET).powi(2)
            + bar(a_mac)
            + POT_K_RHO * (rho - POT_COUPLE_SET).powi(2)
            + POT_RIDGE_G * (-(movement / POT_RIDGE_W).powi(2)).exp()
            + POT_K_E * (energy - POT_ENERGY_SET).powi(2)
            + POT_K_SIG * (sig - 1.0).powi(2);
        let g_mic = 2.0 * POT_K_AMP * (a_mic - POT_MICRO_SET) + dbar(a_mic);
        let g_mac = 2.0 * POT_K_AMP * (a_mac - POT_MACRO_SET) + dbar(a_mac);
        let micro_gain = (1.0 - POT_STEP * g_mic).clamp(POT_GAIN_LO, POT_GAIN_HI);
        let macro_gain = (1.0 - POT_STEP * g_mac).clamp(POT_GAIN_LO, POT_GAIN_HI);

        // --- state speed (dual role of the old dual-EMA precursor) ---
        let s = [a_mic, a_mac, rho, movement, energy];
        let ds = match self.prev_state {
            Some(p) => s
                .iter()
                .zip(p.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>(),
            None => 0.05,
        };
        self.prev_state = Some(s);
        self.speed_ema += 0.2 * (ds - self.speed_ema);

        // --- recent-best floor of V (slowly forgets upward so old lows expire) ---
        if !self.init {
            self.v_floor = v;
            self.init = true;
        }
        // Floor drifts up 0.002/step (old lows expire) but snaps down to any new low.
        self.v_floor = v.min(self.v_floor + 0.002);
        let excess = ((v - self.v_floor) / TEMP_EXCESS_SCALE).clamp(0.0, 1.0);

        // --- temperature ---
        let stuck = excess * (-self.speed_ema / TEMP_STUCK_SPEED).exp();
        let subcrit = (1.0 - sig).clamp(0.0, 1.0);
        let movement_deficit =
            ((LOW_MOTION_TRIGGER - movement) / LOW_MOTION_TRIGGER).clamp(0.0, 1.0);
        let t_target = (stuck * 0.65
            + subcrit * 0.48
            + curiosity * 0.30
            + stagnation * 0.62
            + movement_deficit * 0.34)
            .clamp(0.0, 1.0);
        self.temp += TEMP_SMOOTH * (t_target - self.temp);
        let temp = self.temp.clamp(0.0, 1.0);

        let annealed = self.was_hot && temp < TEMP_COOL;
        if temp > TEMP_HOT {
            self.was_hot = true;
        }
        if annealed {
            self.was_hot = false;
        }

        let couple_release = ((rho - 0.6) * 2.5)
            .clamp(0.0, 1.0)
            .max(temp * 0.5)
            .max(curiosity);

        PotOut {
            micro_gain,
            macro_gain,
            shear_amp: SHEAR_AMP_MIN + (SHEAR_AMP_MAX - SHEAR_AMP_MIN) * temp,
            micro_kick: MICRO_KICK_MAX * temp,
            temp,
            lr_heat: 1.0 + LR_HEAT_MAX * temp as f64,
            couple_release,
            v,
            annealed,
        }
    }
}

// Structured 2D fBm shear for the macro field: octave-summed plane waves with
// per-octave direction and traveling phase, advanced by the angle-addition
// identity so the hot path is table lookups + FMAs, zero per-element sin().
struct ShearField2D {
    sin_a: Tensor,
    cos_a: Tensor,
    freqs: [f32; SHEAR_OCTAVES],
    weights: [f32; SHEAR_OCTAVES],
}
impl ShearField2D {
    fn new(channels: usize, h: usize, w: usize, device: &Device) -> CResult<Self> {
        let cl = channels * h * w;
        let mut sin_a = vec![0.0f32; SHEAR_OCTAVES * cl];
        let mut cos_a = vec![0.0f32; SHEAR_OCTAVES * cl];
        let mut freqs = [0.0f32; SHEAR_OCTAVES];
        let mut weights = [0.0f32; SHEAR_OCTAVES];
        let mut freq = 1.0f32;
        let mut weight = 1.0f32;
        for oct in 0..SHEAR_OCTAVES {
            freqs[oct] = freq;
            weights[oct] = weight;
            let theta = 0.7 + oct as f32 * 1.1; // per-octave wave direction
            let (dx, dy) = (theta.cos(), theta.sin());
            for c in 0..channels {
                let c_phase = c as f32 * 0.1;
                for i in 0..h {
                    for j in 0..w {
                        let a =
                            TWO_PI * freq * (i as f32 / h as f32 * dy + j as f32 / w as f32 * dx)
                                + c_phase;
                        let idx = oct * cl + c * h * w + i * w + j;
                        sin_a[idx] = a.sin();
                        cos_a[idx] = a.cos();
                    }
                }
            }
            freq *= 2.0;
            weight *= 0.5;
        }
        Ok(Self {
            sin_a: Tensor::from_vec(sin_a, (SHEAR_OCTAVES, channels, h, w), device)?,
            cos_a: Tensor::from_vec(cos_a, (SHEAR_OCTAVES, channels, h, w), device)?,
            freqs,
            weights,
        })
    }
    fn generate(&self, amp: f32, phase: f32) -> CResult<Tensor> {
        let mut field = Tensor::zeros_like(&self.sin_a.narrow(0, 0, 1)?)?;
        for oct in 0..SHEAR_OCTAVES {
            let b = phase * self.freqs[oct];
            let (sb, cb) = (b.sin(), b.cos());
            let w = self.weights[oct] as f64;
            let term = self
                .sin_a
                .narrow(0, oct, 1)?
                .affine((cb as f64) * w, 0.0)?
                .add(&self.cos_a.narrow(0, oct, 1)?.affine((sb as f64) * w, 0.0)?)?;
            field = field.add(&term)?;
        }
        field.affine(amp as f64, 0.0)
    }
}

// =====================================================================
// EPISODIC MEMORY — "return of the theme"
// =====================================================================
// Ring buffer of detached refined_hidden snapshots + a learned attention
// readout. Snapshots span ~87 s at defaults, an order of magnitude past the
// GRU's horizon. Gradients flow through Wq/Wk/Wv and the current query only.
struct EpisodicMemory {
    slots: VecDeque<Tensor>, // each (1, MEMORY_DIM), detached
    wq: Linear,
    wk: Linear,
    wv: Linear,
}
impl EpisodicMemory {
    fn new(vb: VBV) -> Result<Self> {
        Ok(Self {
            slots: VecDeque::with_capacity(EPI_SLOTS),
            wq: candle_nn::linear(MEMORY_DIM, EPI_DIM, vb.pp("wq"))?,
            wk: candle_nn::linear(MEMORY_DIM, EPI_DIM, vb.pp("wk"))?,
            wv: candle_nn::linear(MEMORY_DIM, EPI_DIM, vb.pp("wv"))?,
        })
    }
    fn snapshot(&mut self, hidden: &Tensor) {
        if self.slots.len() >= EPI_SLOTS {
            self.slots.pop_front();
        }
        self.slots.push_back(hidden.detach());
    }
    fn read(&self, hidden: &Tensor, device: &Device) -> CResult<Tensor> {
        if self.slots.is_empty() {
            return Tensor::zeros((1, EPI_DIM), DType::F32, device);
        }
        let refs: Vec<&Tensor> = self.slots.iter().collect();
        let snaps = Tensor::cat(&refs, 0)?; // (N, 512)
        let k = self.wk.forward(&snaps)?; // (N, 64)
        let v = self.wv.forward(&snaps)?; // (N, 64)
        let q = self.wq.forward(hidden)?; // (1, 64)
        let scores = q
            .matmul(&k.t()?)?
            .affine(1.0 / (EPI_DIM as f64).sqrt(), 0.0)?;
        let attn = candle_nn::ops::softmax(&scores, D::Minus1)?;
        attn.matmul(&v) // (1, 64)
    }
    fn snapshot_host(&self) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(self.slots.len());
        for slot in &self.slots {
            out.push(slot.reshape((MEMORY_DIM,))?.to_vec1::<f32>()?);
        }
        Ok(out)
    }
    fn restore_host(&mut self, slots: &[Vec<f32>], device: &Device) -> Result<()> {
        self.slots.clear();
        for values in slots
            .iter()
            .rev()
            .take(EPI_SLOTS)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
        {
            if values.len() == MEMORY_DIM {
                self.slots
                    .push_back(Tensor::from_vec(values.clone(), (1, MEMORY_DIM), device)?);
            }
        }
        Ok(())
    }
}

// --- SEMANTIC FIELD / TELEMETRY TEXT ---
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

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SemanticField {
    history: VecDeque<f32>,
    arch_history: VecDeque<usize>,
}
impl SemanticField {
    fn new() -> Self {
        Self {
            history: VecDeque::new(),
            arch_history: VecDeque::new(),
        }
    }
    fn archetype_field(field01: &[f32]) -> (String, f32, usize) {
        let mut bins = [0usize; 8];
        for &v in field01 {
            let mut a = 7;
            for (i, &b) in ARCH_BOUNDS.iter().skip(1).enumerate() {
                if v < b {
                    a = i;
                    break;
                }
            }
            bins[a] += 1;
        }
        let n = field01.len() as f32 + 1e-6;
        let mut entropy = 0.0;
        let mut max_b = 0;
        let mut max_c = 0;
        let mut syms = String::with_capacity(8);
        for (i, &c) in bins.iter().enumerate() {
            if c > max_c {
                max_c = c;
                max_b = i;
            }
            let p = c as f32 / n;
            if p > 0.0 {
                entropy -= p * p.log2();
            }
            let idx = (p * 16.0).clamp(0.0, 7.99) as usize;
            syms.push_str(VAL_SYMS[idx]);
        }
        (syms, entropy, max_b)
    }
    fn phase(drift: f32) -> &'static str {
        for &(bound, name) in &PHASE_MAP {
            if drift <= bound {
                return name;
            }
        }
        "PRIMORDIAL"
    }
    fn record(&mut self, drift: f32, dom_arch: usize) {
        self.history.push_back(drift);
        if self.history.len() > 20 {
            self.history.pop_front();
        }
        self.arch_history.push_back(dom_arch);
        if self.arch_history.len() > 20 {
            self.arch_history.pop_front();
        }
    }
    fn trend(&self) -> &'static str {
        if self.history.len() < 2 {
            return "→";
        }
        let a = self.history[0];
        let b = *self.history.back().unwrap();
        if b > a + 0.05 {
            "↑"
        } else if b < a - 0.05 {
            "↓"
        } else {
            "→"
        }
    }
    fn dominant_phase(&self) -> &'static str {
        if self.history.is_empty() {
            return "PRIMORDIAL";
        }
        let avg = self.history.iter().copied().sum::<f32>() / self.history.len() as f32;
        Self::phase(avg)
    }
    fn dominant_archetype(&self) -> &'static str {
        if self.arch_history.is_empty() {
            return ARCHETYPES[0];
        }
        let mut counts = [0usize; 8];
        for &a in &self.arch_history {
            counts[a] += 1;
        }
        let mut max_c = 0;
        let mut max_a = 0;
        for (i, &c) in counts.iter().enumerate() {
            if c > max_c {
                max_c = c;
                max_a = i;
            }
        }
        ARCHETYPES[max_a]
    }
}

// --- ARBITER (sign-fixed, temperature-aware) ---
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
    // v3 BUG FIX: the old code ADDED (-sum w log w) = +H's negation... net
    // effect was minimizing entropy — the arbiter was rewarded for collapsing
    // all weight onto one loss, and the progress term accelerated the
    // collapse. Now: softmax at temperature tau (hot system => flatter
    // mixing), and we return -H so that ADDING (coef * neg_entropy) to the
    // loss is an entropy BONUS.
    fn forward(&self, features: &Tensor, tau: f32) -> Result<(Tensor, Tensor)> {
        let logits = self
            .net
            .forward(features)?
            .affine(1.0 / tau.max(0.25) as f64, 0.0)?;
        let w_norm = candle_nn::ops::softmax(&logits, D::Minus1)?;
        let neg_entropy = w_norm.mul(&w_norm.affine(1.0, 1e-4)?.log()?)?.sum_all()?;
        Ok((w_norm, neg_entropy))
    }
}

struct MonitorHead {
    net: candle_nn::Sequential,
}
impl MonitorHead {
    fn new(vb: VBV) -> Result<Self> {
        let net = candle_nn::seq()
            .add(candle_nn::linear(
                MEMORY_DIM + ACTION_COUNT,
                96,
                vb.pp("fc1"),
            )?)
            .add(Tanh)
            .add(candle_nn::linear(96, OBS_DIM * 2, vb.pp("fc2"))?);
        Ok(Self { net })
    }
    fn forward(&self, features: &Tensor) -> Result<(Tensor, Tensor)> {
        let raw = self.net.forward(features)?;
        let mean = candle_nn::ops::sigmoid(&raw.narrow(1, 0, OBS_DIM)?)?;
        let log_var = raw.narrow(1, OBS_DIM, OBS_DIM)?.tanh()?.affine(3.0, -3.0)?;
        Ok((mean, log_var))
    }
}

fn control_features(action: ControlAction, c: SynthesisControl) -> [f32; ACTION_COUNT] {
    [
        (c.shear_mult / 1.75).clamp(0.0, 1.0),
        (c.kick_mult / 1.80).clamp(0.0, 1.0),
        c.inharmonicity.clamp(0.0, 1.0),
        ((c.spectral_tilt + 0.30) / 0.60).clamp(0.0, 1.0),
        c.resonator_drive.clamp(0.0, 1.0),
        (c.noise_level / 0.20).clamp(0.0, 1.0),
        ((c.echo_delta + 0.12) / 0.24).clamp(0.0, 1.0),
        (c.width_mult / 1.50).clamp(0.0, 1.0),
        c.recall_mix.clamp(0.0, 1.0),
        action.index() as f32 / (ACTION_COUNT - 1) as f32,
    ]
}

fn predictor_input(
    hidden: &Tensor,
    action: ControlAction,
    control: SynthesisControl,
    device: &Device,
) -> CResult<Tensor> {
    let control = Tensor::from_vec(
        control_features(action, control).to_vec(),
        (1, ACTION_COUNT),
        device,
    )?;
    Tensor::cat(&[hidden, &control], 1)
}

fn predicted_interest(values: &[f32], ecology: &AdaptiveDynamics) -> f32 {
    if values.len() < OBS_DIM {
        return 0.0;
    }
    let obs = AudioObservation {
        values: std::array::from_fn(|i| values[i].clamp(0.0, 1.0)),
    };
    let recovery = ecology.stagnation
        * (0.42 * obs.values[7]
            + 0.24 * obs.values[3]
            + 0.20 * obs.values[11]
            + 0.14 * obs.values[10]);
    obs.structured_complexity() + recovery
}

fn plan_action_scores(
    head: &MonitorHead,
    hidden: &Tensor,
    device: &Device,
    ecology: &AdaptiveDynamics,
) -> Result<[f32; ACTION_COUNT]> {
    let refs: Vec<&Tensor> = (0..ACTION_COUNT).map(|_| hidden).collect();
    let hidden_batch = Tensor::cat(&refs, 0)?;
    let mut control_rows = Vec::with_capacity(ACTION_COUNT * ACTION_COUNT);
    for action in ControlAction::ALL {
        control_rows.extend_from_slice(&control_features(
            action,
            SynthesisControl::for_action(action),
        ));
    }
    let action_batch = Tensor::from_vec(control_rows, (ACTION_COUNT, ACTION_COUNT), device)?;
    let input = Tensor::cat(&[&hidden_batch, &action_batch], 1)?;
    let (mean, log_var) = head.forward(&input)?;
    let means = mean.to_vec2::<f32>()?;
    let vars = log_var.to_vec2::<f32>()?;
    let mut scores = [0.0f32; ACTION_COUNT];
    for i in 0..ACTION_COUNT {
        let uncertainty = vars[i].iter().map(|v| v.exp()).sum::<f32>() / OBS_DIM as f32;
        scores[i] = predicted_interest(&means[i], ecology) - 0.08 * uncertainty.sqrt();
    }
    Ok(scores)
}

struct KANLayer {
    basis_fn: usize,
    w: Tensor,
    mod_proj: Linear,
    freqs: Tensor,
    tilt: Tensor,
}
impl KANLayer {
    fn new(basis_fn: usize, vb: VBV) -> Result<Self> {
        let w = vb.get_with_hints(
            (basis_fn,),
            "weights",
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 0.1,
            },
        )?;
        let mod_proj = candle_nn::linear(MEMORY_DIM, basis_fn, vb.pp("mod_proj"))?;
        let freq_vec: Vec<f32> = (1..=basis_fn).map(|i| i as f32).collect();
        let freqs = Tensor::from_vec(freq_vec, (1, basis_fn), vb.device())?;
        let tilt_vec: Vec<f32> = (1..=basis_fn).map(|i| 1.0 / (i as f32).sqrt()).collect();
        let tilt = Tensor::from_vec(tilt_vec, (1, basis_fn), vb.device())?;
        Ok(Self {
            basis_fn,
            w,
            mod_proj,
            freqs,
            tilt,
        })
    }
    fn forward(&self, x: &Tensor, mem: &Tensor) -> CResult<Tensor> {
        let (d0, d1) = x.dims2()?;
        let delta_w = self
            .mod_proj
            .forward(mem)?
            .tanh()?
            .reshape((self.basis_fn,))?;
        let active_w = self
            .w
            .add(&delta_w.affine(0.15, 0.0)?)?
            .reshape((1, self.basis_fn))?
            .broadcast_mul(&self.tilt)?;
        let xf = x.reshape((d0 * d1, 1))?;
        let basis = xf.broadcast_mul(&self.freqs)?.sin()?;
        let summed = basis.broadcast_mul(&active_w)?.sum(D::Minus1)?;
        summed
            .reshape((d0, d1))?
            .affine(1.0 / (self.basis_fn as f64).sqrt(), 0.0)
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
                .add(Relu)
                .add(candle_nn::linear(dim, dim, vb.pp(&format!("l{}_2", i)))?)
                .add(Tanh);
            layers.push(seq);
        }
        Ok(Self {
            layers,
            active_depth: MORPH_START_DEPTH,
        })
    }
    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let mut out = x.clone();
        for i in 0..self.active_depth {
            out = out.add(&self.layers[i].forward(&out)?)?;
        }
        Ok(out)
    }
    fn depth(&self) -> usize {
        self.active_depth
    }
    fn set_depth(&mut self, d: usize) {
        self.active_depth = d.clamp(1, self.layers.len());
    }
    fn grow(&mut self) -> bool {
        if self.active_depth < self.layers.len() {
            self.active_depth += 1;
            true
        } else {
            false
        }
    }
    fn prune(&mut self) -> bool {
        if self.active_depth > 1 {
            self.active_depth -= 1;
            true
        } else {
            false
        }
    }
}

struct GRUCell {
    w_ir: Linear,
    w_hr: Linear,
    w_iz: Linear,
    w_hz: Linear,
    w_in: Linear,
    w_hn: Linear,
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
        let n = self
            .w_in
            .forward(x)?
            .add(&r.mul(&self.w_hn.forward(h)?)?)?
            .tanh()?;
        let one_minus_z = z.affine(-1.0, 1.0)?;
        one_minus_z.mul(h)?.add(&z.mul(&n)?)
    }
}

// --- 2D NEURAL CA on a torus ---
struct NeuralCA2D {
    conv1: Box<dyn Fn(&Tensor) -> CResult<Tensor>>,
    conv2: Box<dyn Fn(&Tensor) -> CResult<Tensor>>,
    anisotropic_mask: Tensor,
}
impl NeuralCA2D {
    fn new(channels: usize, hidden: usize, vb: VBV) -> Result<Self> {
        let conv1 = conv2d_torus(channels, hidden, 3, vb.pp("c1"))?;
        let conv2 = conv2d_torus(hidden, channels, 1, vb.pp("c2"))?;
        // Fixed anisotropy: a gentle per-channel 2D interference pattern so the
        // learned rule is not forced to break symmetry from pure noise.
        let mut pattern = vec![0.0f32; channels * GRID_H * GRID_W];
        for c in 0..channels {
            for i in 0..GRID_H {
                for j in 0..GRID_W {
                    let pi = (i as f32 / GRID_H as f32) * TWO_PI;
                    let pj = (j as f32 / GRID_W as f32) * TWO_PI;
                    pattern[c * GRID_H * GRID_W + i * GRID_W + j] = 0.8
                        + 0.4
                            * ((pi * (1.0 + (c % 3) as f32)
                                + pj * (1.0 + (c % 2) as f32)
                                + c as f32)
                                .sin());
                }
            }
        }
        let anisotropic_mask =
            Tensor::from_vec(pattern, (1, channels, GRID_H, GRID_W), vb.device())?;
        Ok(Self {
            conv1,
            conv2,
            anisotropic_mask,
        })
    }
    // Stochastic cell clock: the keep-mask is supplied by the caller (built
    // from the master seeded RNG), constant w.r.t. the graph.
    fn forward(
        &self,
        x: &Tensor,
        ext_mod: Option<&Tensor>,
        field_bias: Option<&Tensor>,
        keep: &Tensor,
    ) -> CResult<Tensor> {
        let h = (self.conv1)(x)?.relu()?;
        let mut out = (self.conv2)(&h)?;
        out = out.broadcast_mul(&self.anisotropic_mask)?;
        if let Some(m) = ext_mod {
            out = out.broadcast_mul(&m.unsqueeze(2)?.unsqueeze(3)?)?;
        }
        if let Some(f) = field_bias {
            out = out.add(f)?;
        }
        x.add(&out.mul(keep)?.affine(0.1, 0.0)?)
    }
}

// Pairwise skew rotation is approximately norm preserving and creates
// coherent tangential motion without raising every cell's amplitude. This is
// the core analogue of rotational drive rather than another white-noise kick.
fn rotate_channel_pairs(field: &Tensor, omega: f32) -> CResult<Tensor> {
    if omega.abs() <= 1e-6 {
        return Ok(field.clone());
    }
    let paired = field.reshape((1, CA_CHANNELS / 2, 2, GRID_H, GRID_W))?;
    let a = paired.narrow(2, 0, 1)?;
    let b = paired.narrow(2, 1, 1)?;
    let norm = (1.0 + omega * omega).sqrt() as f64;
    let a_rot = a
        .affine(1.0 / norm, 0.0)?
        .add(&b.affine(-(omega as f64) / norm, 0.0)?)?;
    let b_rot = b
        .affine(1.0 / norm, 0.0)?
        .add(&a.affine((omega as f64) / norm, 0.0)?)?;
    Tensor::cat(&[&a_rot, &b_rot], 2)?.reshape((1, CA_CHANNELS, GRID_H, GRID_W))
}

// --- TENSOR MODEL STRUCTS ---
struct ForwardOut {
    stereo: Tensor,
    next_micro: Tensor,
    next_macro: Tensor,
    next_hidden: Tensor,
    refined_hidden: Tensor,
    movement_t: Tensor,
    cur_freq_l: Tensor, // (1,) — read back in the batched metrics sync
    cur_freq_r: Tensor,
    mod_freq_l: Tensor,
    mod_freq_r: Tensor,
    pan: Tensor,                   // (1,) — feeds the last_pan host mirror
    pair_sums: Tensor,             // (2,) — theta for the NEXT step (1-step lag, no sync)
    region_activity: Tensor,       // (16,) 4x4 spatial summary for DSP/control
    region_change: Tensor,         // (16,) local temporal activity
    macro_region_activity: Tensor, // (16,) slow-field profile for confinement control
}

struct ComplexAudioEcosystem {
    micro_ca: NeuralCA2D,
    macro_ca: NeuralCA2D,
    gru_memory: GRUCell,
    morphic: MorphicStack,
    asymptotic_contraction: AsymptoticContractionLayer,
    spatial_panner: candle_nn::Sequential,
    fm_mod_ratio: candle_nn::Sequential,
    fm_mod_index: candle_nn::Sequential,
    wave_morph_head: candle_nn::Sequential,
    wavefolder_l: KANLayer,
    wavefolder_r: KANLayer,
    base_freq_l: Tensor,
    base_freq_r: Tensor,
    t_steps: Tensor,
    ramp: Tensor,
    scan_harmonic: Tensor,
    scan_inharmonic: Tensor,
    scan_detune: Tensor,
    scan_pan_l: Tensor,
    scan_pan_r: Tensor,
    scan_brightness: Tensor,
    scan_phase_offsets: Tensor,
    scan_interp: Tensor, // (CHUNK_SIZE, GRID_W) linear-interp upsampler for the column envelope
    // Host-side mirrors (phases and glide frequencies live OFF the device: the
    // v3 mod_2pi tensor round-trip synced every step and carried no gradient
    // anyway — these are pure f32 now).
    current_freq_l: f32,
    current_freq_r: f32,
    last_pan: f32,
    prev_fm_idx_l: Tensor,
    prev_fm_idx_r: Tensor,
    prev_openness: Tensor,
    prev_gain_l: Tensor,
    prev_gain_r: Tensor,
}

struct AsymptoticContractionLayer {
    expand: Linear,
    contract: Linear,
}
impl AsymptoticContractionLayer {
    fn new(in_d: usize, hyper_d: usize, out_d: usize, vb: VBV) -> Result<Self> {
        Ok(Self {
            expand: candle_nn::linear(in_d, hyper_d, vb.pp("expand"))?,
            contract: candle_nn::linear(hyper_d, out_d, vb.pp("contract"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        self.contract
            .forward(&self.expand.forward(x)?.tanh()?)?
            .affine(1.0 / (LARGE_D_DIM as f64).sqrt(), 0.0)
    }
}

impl ComplexAudioEcosystem {
    fn new(vb: VBV, dev: &Device) -> Result<Self> {
        let micro_ca = NeuralCA2D::new(CA_CHANNELS, CA_HIDDEN, vb.pp("micro_ca"))?;
        let macro_ca = NeuralCA2D::new(CA_CHANNELS, CA_HIDDEN, vb.pp("macro_ca"))?;
        let gru_memory = GRUCell::new(CA_CHANNELS + EPI_DIM, MEMORY_DIM, vb.pp("gru_memory"))?;
        let morphic = MorphicStack::new(MEMORY_DIM, MORPH_MAX_BLOCKS, vb.pp("morphic"))?;
        let asymptotic_contraction = AsymptoticContractionLayer::new(
            MEMORY_DIM,
            LARGE_D_DIM,
            CA_CHANNELS,
            vb.pp("asymp_contract"),
        )?;
        let spatial_panner = candle_nn::seq()
            .add(candle_nn::linear(MEMORY_DIM, 1, vb.pp("spatial_panner_0"))?)
            .add(Tanh);
        let fm_mod_ratio = candle_nn::seq()
            .add(candle_nn::linear(MEMORY_DIM, 2, vb.pp("fm_mod_ratio_0"))?)
            .add(Relu);
        let fm_mod_index = candle_nn::seq()
            .add(candle_nn::linear(MEMORY_DIM, 2, vb.pp("fm_mod_index_0"))?)
            .add(Sigmoid);
        let wave_morph_head = candle_nn::seq()
            .add(candle_nn::linear(
                MEMORY_DIM,
                2,
                vb.pp("wave_morph_head_0"),
            )?)
            .add(Sigmoid);
        let wavefolder_l = KANLayer::new(KAN_BASIS_FUNCTIONS, vb.pp("wavefolder_l"))?;
        let wavefolder_r = KANLayer::new(KAN_BASIS_FUNCTIONS, vb.pp("wavefolder_r"))?;
        let base_freq_l = vb.get_with_hints(
            (1,),
            "base_freq_l",
            candle_nn::Init::Const(BASE_FREQ_L as f64),
        )?;
        let base_freq_r = vb.get_with_hints(
            (1,),
            "base_freq_r",
            candle_nn::Init::Const(BASE_FREQ_R as f64),
        )?;
        let steps_vec: Vec<f32> = (0..CHUNK_SIZE)
            .map(|i| i as f32 / SAMPLE_RATE as f32)
            .collect();
        let ramp_vec: Vec<f32> = (0..CHUNK_SIZE)
            .map(|i| i as f32 / (CHUNK_SIZE as f32 - 1.0))
            .collect();
        let harmonic_vec: Vec<f32> = (1..=SCAN_PARTIALS).map(|i| i as f32).collect();
        let inharmonic_vec = vec![
            1.0f32, 1.4142, 2.0, 2.6180, 3.0, 3.7321, 4.2361, 5.0, 5.3852, 6.2832, 7.0711, 7.8540,
            8.4853, 9.1925, 10.0, 11.0902,
        ];
        let detune_vec: Vec<f32> = (0..SCAN_PARTIALS)
            .map(|i| (((i * 7 + 3) % 11) as f32 / 10.0 - 0.5) * 0.18)
            .collect();
        let brightness_vec: Vec<f32> = (0..SCAN_PARTIALS)
            .map(|i| i as f32 / (SCAN_PARTIALS - 1) as f32 * 2.0 - 1.0)
            .collect();
        let phase_vec: Vec<f32> = (0..SCAN_PARTIALS)
            .map(|i| (i as f32 * 2.3999632).rem_euclid(TWO_PI))
            .collect();
        let mut pan_l = Vec::with_capacity(SCAN_PARTIALS);
        let mut pan_r = Vec::with_capacity(SCAN_PARTIALS);
        for i in 0..SCAN_PARTIALS {
            let x = i as f32 / (SCAN_PARTIALS - 1) as f32 * 2.0 - 1.0;
            pan_l.push(((1.0 - x) * 0.5).sqrt());
            pan_r.push(((1.0 + x) * 0.5).sqrt());
        }
        // Linear-interp matrix upsampling the (GRID_W,) column profile to CHUNK
        // samples: env[t] = (1-f)*col[j] + f*col[j+1]. Precomputed once so the
        // scan-synth envelope is a single matmul at runtime.
        let mut interp = vec![0.0f32; CHUNK_SIZE * GRID_W];
        for t in 0..CHUNK_SIZE {
            let pos = t as f32 / CHUNK_SIZE as f32 * (GRID_W as f32 - 1.0);
            let j = pos.floor() as usize;
            let f = pos - j as f32;
            interp[t * GRID_W + j] += 1.0 - f;
            interp[t * GRID_W + (j + 1).min(GRID_W - 1)] += f;
        }
        Ok(Self {
            micro_ca,
            macro_ca,
            gru_memory,
            morphic,
            asymptotic_contraction,
            spatial_panner,
            fm_mod_ratio,
            fm_mod_index,
            wave_morph_head,
            wavefolder_l,
            wavefolder_r,
            base_freq_l,
            base_freq_r,
            t_steps: Tensor::new(steps_vec, dev)?,
            ramp: Tensor::new(ramp_vec, dev)?,
            scan_harmonic: Tensor::from_vec(harmonic_vec, (SCAN_PARTIALS, 1), dev)?,
            scan_inharmonic: Tensor::from_vec(inharmonic_vec, (SCAN_PARTIALS, 1), dev)?,
            scan_detune: Tensor::from_vec(detune_vec, (SCAN_PARTIALS, 1), dev)?,
            scan_pan_l: Tensor::from_vec(pan_l, (SCAN_PARTIALS, 1), dev)?,
            scan_pan_r: Tensor::from_vec(pan_r, (SCAN_PARTIALS, 1), dev)?,
            scan_brightness: Tensor::from_vec(brightness_vec, (SCAN_PARTIALS, 1), dev)?,
            scan_phase_offsets: Tensor::from_vec(phase_vec, (SCAN_PARTIALS, 1), dev)?,
            scan_interp: Tensor::from_vec(interp, (CHUNK_SIZE, GRID_W), dev)?,
            current_freq_l: BASE_FREQ_L,
            current_freq_r: BASE_FREQ_R,
            last_pan: 0.0,
            prev_fm_idx_l: Tensor::new(0.0f32, dev)?,
            prev_fm_idx_r: Tensor::new(0.0f32, dev)?,
            prev_openness: Tensor::new(0.7f32, dev)?,
            prev_gain_l: Tensor::new(0.707f32, dev)?,
            prev_gain_r: Tensor::new(0.707f32, dev)?,
        })
    }
    fn depth(&self) -> usize {
        self.morphic.depth()
    }
    fn set_depth(&mut self, d: usize) {
        self.morphic.set_depth(d);
    }
    fn grow(&mut self) -> bool {
        self.morphic.grow()
    }
    fn prune(&mut self) -> bool {
        self.morphic.prune()
    }

    fn ramp_param(&self, new_val: &Tensor, prev_val: &Tensor) -> CResult<Tensor> {
        let delta = new_val.sub(prev_val)?;
        self.ramp
            .broadcast_mul(&delta.reshape((1,))?)?
            .broadcast_add(prev_val)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &mut self,
        micro: &Tensor,
        macro_t: &Tensor,
        mem: &Tensor,
        epi_out: &Tensor,
        phases: [f32; 4], // [carrier_l, carrier_r, mod_l, mod_r] — host f32
        theta_prev: f32,  // theta from last step's batched readout (1-step lag)
        theta_prev2: f32,
        force: bool,
        energy: f32,
        control: &SynthesisControl,
        core_control: &CoreControl,
        random_pool: &DeviceRandomPool,
        random_index: usize,
    ) -> Result<ForwardOut> {
        let [pc_l, pc_r, pm_l, pm_r] = phases;

        let mut next_macro = macro_t.clone();
        if force {
            // Local anti-rail restoring field (per-cell health; the global-mean
            // barrier lives in the PotentialController).
            let field = macro_t.abs()?.affine(-0.04, 0.02)?;
            let keep = random_pool.keep_mask(random_index * 2)?;
            let proposed = self.macro_ca.forward(macro_t, None, Some(&field), &keep)?;
            next_macro = macro_t
                .add(
                    &proposed
                        .sub(macro_t)?
                        .affine(core_control.update_drive as f64, 0.0)?,
                )?
                .tanh()?
                .affine(0.95, 0.0)?;
        }
        let macro_act = next_macro.abs()?.mean_all()?;
        let metab = macro_act.affine(5.0, 0.0)?.clamp(0.01f32, 1.0f32)?;
        let inv_metab = metab.affine(-1.0, 1.0)?;
        let contracted_mem = self.asymptotic_contraction.forward(mem)?;
        let macro_ch = next_macro.mean(D::Minus1)?.mean(D::Minus1)?; // (1, C)
        let macro_mod = contracted_mem
            .add(&macro_ch)?
            .affine(core_control.macro_coupling as f64, 0.0)?;
        let micro_field = micro.abs()?.affine(-0.04, 0.02)?;
        let keep_m = random_pool.keep_mask(random_index * 2 + 1)?;
        let proposed_micro =
            self.micro_ca
                .forward(micro, Some(&macro_mod), Some(&micro_field), &keep_m)?;
        let driven_micro = micro.add(
            &proposed_micro
                .sub(micro)?
                .affine(core_control.update_drive as f64, 0.0)?,
        )?;
        let next_micro = micro
            .broadcast_mul(&inv_metab)?
            .add(&driven_micro.broadcast_mul(&metab)?)?;
        let next_micro = rotate_channel_pairs(&next_micro, core_control.rotational_drive)?
            .clamp(-1.0f32, 1.0f32)?;
        let movement_t = next_micro.sub(micro)?.abs()?.mean_all()?;

        // Channel features (spatial mean) — population readouts + GRU input.
        let micro_feats = next_micro.mean(D::Minus1)?.mean(D::Minus1)?; // (1, C)
        let pop_l = micro_feats.narrow(1, 0, 1)?.reshape(())?;
        let pop_r = micro_feats.narrow(1, 1, 1)?.reshape(())?;
        // Theta pair sums, read back in the batched metrics sync (no host sync here).
        let paired = micro_feats.reshape((CA_CHANNELS / 2, 2))?;
        let pair_sums = paired.sum(0)?; // (2,)

        // GRU input = channel features ++ episodic attention readout.
        let gru_in = Tensor::cat(&[&micro_feats, epi_out], 1)?; // (1, C + EPI_DIM)
        let next_hidden = self.gru_memory.forward(&gru_in, mem)?;
        let refined_hidden = self.morphic.forward(&next_hidden)?;

        let fm_ratios = self
            .fm_mod_ratio
            .forward(&refined_hidden)?
            .affine(4.0, 0.0)?;
        let fm_indices = self
            .fm_mod_index
            .forward(&refined_hidden)?
            .affine(5.0, 0.0)?;
        let ratio_l = fm_ratios.narrow(1, 0, 1)?.reshape(())?;
        let ratio_r = fm_ratios.narrow(1, 1, 1)?.reshape(())?;
        let idx_l = fm_indices.narrow(1, 0, 1)?.reshape(())?;
        let idx_r = fm_indices.narrow(1, 1, 1)?.reshape(())?;
        let b_l = self.base_freq_l.reshape(())?.abs()?;
        let b_r = self.base_freq_r.reshape(())?.abs()?;

        let energy_factor = energy.clamp(0.15, 1.0);
        let target_l = b_l
            .add(&pop_l.affine(200.0, 0.0)?)?
            .add(&movement_t.affine(100.0, 0.0)?)?
            .clamp(20.0f32, 4000.0f32)?
            .affine(energy_factor as f64, 0.0)?;
        let target_r = b_r
            .add(&pop_r.affine(200.0, 0.0)?)?
            .add(&movement_t.affine(-100.0, 0.0)?)?
            .clamp(20.0f32, 4000.0f32)?
            .affine(energy_factor as f64, 0.0)?;

        let g = FREQ_GLIDE_SPEED as f64;
        let cur_l = target_l.affine(g, self.current_freq_l as f64 * (1.0 - g))?;
        let cur_r = target_r.affine(g, self.current_freq_r as f64 * (1.0 - g))?;
        let mod_f_l = cur_l.mul(&ratio_l)?.clamp(0.0f32, 6000.0f32)?;
        let mod_f_r = cur_r.mul(&ratio_r)?.clamp(0.0f32, 6000.0f32)?;
        let omega_m_l = mod_f_l.affine(TWO_PI as f64, 0.0)?;
        let omega_m_r = mod_f_r.affine(TWO_PI as f64, 0.0)?;
        let ph_m_l = self
            .t_steps
            .broadcast_mul(&omega_m_l)?
            .affine(1.0, pm_l as f64)?;
        let ph_m_r = self
            .t_steps
            .broadcast_mul(&omega_m_r)?
            .affine(1.0, pm_r as f64)?;
        let idx_curve_l = self.ramp_param(&idx_l, &self.prev_fm_idx_l)?;
        let idx_curve_r = self.ramp_param(&idx_r, &self.prev_fm_idx_r)?;
        let modulator_l = ph_m_l.sin()?.mul(&idx_curve_l)?;
        let modulator_r = ph_m_r.sin()?.mul(&idx_curve_r)?;

        let mut dtheta = theta_prev - theta_prev2;
        dtheta -= TWO_PI * (dtheta / TWO_PI).round();
        let theta_curve = self.ramp.affine(dtheta as f64, theta_prev2 as f64)?;
        let omega_c_l = cur_l.affine(TWO_PI as f64, 0.0)?;
        let omega_c_r = cur_r.affine(TWO_PI as f64, 0.0)?;
        let ph_c_l = self
            .t_steps
            .broadcast_mul(&omega_c_l)?
            .affine(1.0, pc_l as f64)?
            .add(&theta_curve)?
            .add(&modulator_l)?;
        let ph_c_r = self
            .t_steps
            .broadcast_mul(&omega_c_r)?
            .affine(1.0, pc_r as f64)?
            .add(&theta_curve)?
            .add(&modulator_r)?;

        let morphs = self.wave_morph_head.forward(&refined_hidden)?;
        let morph_l = morphs.narrow(1, 0, 1)?.reshape(())?;
        let morph_r = morphs.narrow(1, 1, 1)?.reshape(())?;

        let mut audio_l = morph_wave(&ph_c_l, &morph_l)?;
        let mut audio_r = morph_wave(&ph_c_r, &morph_r)?;
        for (f_l, f_r) in [
            (&pop_l.affine(700.0, 300.0)?, &pop_r.affine(700.0, 300.0)?),
            (
                &movement_t.affine(1700.0, 800.0)?,
                &movement_t.affine(1700.0, 800.0)?,
            ),
            (
                &pop_l.affine(-500.0, 2000.0)?,
                &pop_r.affine(-500.0, 2000.0)?,
            ),
        ] {
            let p_l = self
                .t_steps
                .broadcast_mul(&f_l.affine(TWO_PI as f64, 0.0)?)?
                .affine(1.0, pc_l as f64)?
                .add(&theta_curve)?;
            let p_r = self
                .t_steps
                .broadcast_mul(&f_r.affine(TWO_PI as f64, 0.0)?)?
                .affine(1.0, pc_r as f64)?
                .add(&theta_curve)?;
            audio_l = audio_l.add(&morph_wave(&p_l, &morph_l)?.affine(0.3, 0.0)?)?;
            audio_r = audio_r.add(&morph_wave(&p_r, &morph_r)?.affine(0.3, 0.0)?)?;
        }

        // --- REGIONAL SPECTRAL FIELD ---
        // The previous row/column projection discarded most 2-D topology.  A
        // 4x4 regional readout now drives sixteen independently panned partial
        // agents.  The ratios continuously interpolate between harmonic and
        // inharmonic modal sets; local temporal change adds micro-detuning.
        let field_cm = next_micro.mean(1)?; // (1, H, W)
        let region_grid = field_cm
            .reshape((REGION_ROWS, REGION_H, REGION_COLS, REGION_W))?
            .mean(D::Minus1)?
            .mean(1)?; // (4,4)
        let region_activity = region_grid.reshape((REGION_COUNT,))?;
        let delta_cm = next_micro.sub(micro)?.mean(1)?;
        let region_change = delta_cm
            .reshape((REGION_ROWS, REGION_H, REGION_COLS, REGION_W))?
            .abs()?
            .mean(D::Minus1)?
            .mean(1)?
            .reshape((REGION_COUNT,))?;
        let macro_region_activity = next_macro
            .mean(1)?
            .reshape((REGION_ROWS, REGION_H, REGION_COLS, REGION_W))?
            .mean(D::Minus1)?
            .mean(1)?
            .reshape((REGION_COUNT,))?;

        let amps = region_activity
            .affine(0.5, 0.5)?
            .add(&region_change.affine(0.55, 0.0)?)?
            .relu()?
            .reshape((SCAN_PARTIALS, 1))?;
        let tilt = self
            .scan_brightness
            .affine(control.spectral_tilt as f64, 1.0)?
            .clamp(0.18f32, 1.82f32)?;
        let amps = amps.broadcast_mul(&tilt)?;
        let amp_sum = amps.sum_all()?.affine(1.0, 1e-4)?;
        let amps_n = amps.broadcast_div(&amp_sum)?;

        let inh = control.inharmonicity.clamp(0.0, 1.0);
        let ratios = self
            .scan_harmonic
            .affine((1.0 - inh) as f64, 0.0)?
            .add(&self.scan_inharmonic.affine(inh as f64, 0.0)?)?
            .add(
                &region_change
                    .reshape((REGION_COUNT, 1))?
                    .broadcast_mul(&self.scan_detune)?
                    .affine((0.7 + 0.8 * inh) as f64, 0.0)?,
            )?;

        // Keep one global column envelope as a slow, coherent breath while the
        // regional agents retain spatially independent spectra.
        let cols = field_cm.mean(1)?.reshape((GRID_W, 1))?;
        let env = self
            .scan_interp
            .matmul(&cols)?
            .affine(0.30, 0.70)?
            .clamp(0.18f32, 1.25f32)?
            .reshape((1, CHUNK_SIZE))?;
        let base_phase_l = self
            .t_steps
            .broadcast_mul(&omega_c_l)?
            .affine(1.0, pc_l as f64)?
            .reshape((1, CHUNK_SIZE))?;
        let base_phase_r = self
            .t_steps
            .broadcast_mul(&omega_c_r)?
            .affine(1.0, pc_r as f64)?
            .reshape((1, CHUNK_SIZE))?;
        let ph_l = ratios
            .broadcast_mul(&base_phase_l)?
            .broadcast_add(&self.scan_phase_offsets)?;
        let ph_r = ratios
            .broadcast_mul(&base_phase_r)?
            .broadcast_add(&self.scan_phase_offsets.affine(-1.0, 0.0)?)?;
        let partials_l = ph_l
            .sin()?
            .broadcast_mul(&amps_n)?
            .broadcast_mul(&self.scan_pan_l)?;
        let partials_r = ph_r
            .sin()?
            .broadcast_mul(&amps_n)?
            .broadcast_mul(&self.scan_pan_r)?;
        let scan_l = partials_l.sum(0)?.reshape((1, CHUNK_SIZE))?.mul(&env)?;
        let scan_r = partials_r.sum(0)?.reshape((1, CHUNK_SIZE))?.mul(&env)?;
        audio_l = audio_l.add(&scan_l.affine(SCAN_GAIN, 0.0)?.reshape((CHUNK_SIZE,))?)?;
        audio_r = audio_r.add(&scan_r.affine(SCAN_GAIN, 0.0)?.reshape((CHUNK_SIZE,))?)?;

        let audio_l = self
            .wavefolder_l
            .forward(&audio_l.unsqueeze(0)?, &refined_hidden)?
            .reshape((1, CHUNK_SIZE))?;
        let audio_r = self
            .wavefolder_r
            .forward(&audio_r.unsqueeze(0)?, &refined_hidden)?
            .reshape((1, CHUNK_SIZE))?;

        let open_t = refined_hidden
            .abs()?
            .mean_all()?
            .affine(5.0, 0.0)?
            .add(&movement_t)?
            .clamp(0.4f32, 1.0f32)?
            .affine(energy_factor as f64, 0.0)?;
        let open_curve = self.ramp_param(&open_t, &self.prev_openness)?;
        let audio_l = audio_l.broadcast_mul(&open_curve.unsqueeze(0)?)?;
        let audio_r = audio_r.broadcast_mul(&open_curve.unsqueeze(0)?)?;

        let mid = audio_l.add(&audio_r)?.affine(0.5, 0.0)?;
        let side = audio_l.sub(&audio_r)?.affine(0.5, 0.0)?;
        let pan_t = self
            .spatial_panner
            .forward(&refined_hidden)?
            .reshape(())?
            .clamp(-0.5f32, 0.5f32)?;
        // Haas width from LAST step's pan (host mirror, updated by the batched
        // readback) — removes the one remaining per-step to_scalar sync. Width
        // is a slow spatial parameter; the 85 ms lag is inaudible.
        let width_val = (1.0 + self.last_pan.abs() * 0.8) * control.width_mult.clamp(0.5, 1.6);
        let side_wide = side.affine(width_val as f64, 0.0)?;
        let side_delayed = apply_haas_delay(&side_wide, 16)?;
        let audio_l = mid.add(&side_delayed)?;
        let audio_r = mid.sub(&side_delayed)?;

        let gain_l = pan_t.affine(-0.5, 0.5)?.sqrt()?;
        let gain_r = pan_t.affine(0.5, 0.5)?.sqrt()?;
        let gain_curve_l = self.ramp_param(&gain_l, &self.prev_gain_l)?;
        let gain_curve_r = self.ramp_param(&gain_r, &self.prev_gain_r)?;
        let audio_l = audio_l
            .broadcast_mul(&gain_curve_l.unsqueeze(0)?)?
            .affine(1.414, 0.0)?;
        let audio_r = audio_r
            .broadcast_mul(&gain_curve_r.unsqueeze(0)?)?
            .affine(1.414, 0.0)?;

        let stereo = Tensor::cat(&[&audio_l, &audio_r], 0)?.reshape((2, CHUNK_SIZE))?;

        self.prev_fm_idx_l = idx_l.detach();
        self.prev_fm_idx_r = idx_r.detach();
        self.prev_openness = open_t.detach();
        self.prev_gain_l = gain_l.detach();
        self.prev_gain_r = gain_r.detach();

        Ok(ForwardOut {
            stereo,
            next_micro,
            next_macro,
            next_hidden,
            refined_hidden,
            movement_t,
            cur_freq_l: cur_l.reshape((1,))?,
            cur_freq_r: cur_r.reshape((1,))?,
            mod_freq_l: mod_f_l.reshape((1,))?,
            mod_freq_r: mod_f_r.reshape((1,))?,
            pan: pan_t.reshape((1,))?,
            pair_sums,
            region_activity,
            region_change,
            macro_region_activity,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ModelRuntimeState {
    current_freq_l: f32,
    current_freq_r: f32,
    last_pan: f32,
    prev_fm_idx_l: f32,
    prev_fm_idx_r: f32,
    prev_openness: f32,
    prev_gain_l: f32,
    prev_gain_r: f32,
}
impl ComplexAudioEcosystem {
    fn runtime_state(&self) -> Result<ModelRuntimeState> {
        Ok(ModelRuntimeState {
            current_freq_l: self.current_freq_l,
            current_freq_r: self.current_freq_r,
            last_pan: self.last_pan,
            prev_fm_idx_l: self.prev_fm_idx_l.to_scalar::<f32>()?,
            prev_fm_idx_r: self.prev_fm_idx_r.to_scalar::<f32>()?,
            prev_openness: self.prev_openness.to_scalar::<f32>()?,
            prev_gain_l: self.prev_gain_l.to_scalar::<f32>()?,
            prev_gain_r: self.prev_gain_r.to_scalar::<f32>()?,
        })
    }
    fn restore_runtime_state(&mut self, state: &ModelRuntimeState, device: &Device) -> Result<()> {
        self.current_freq_l = state.current_freq_l;
        self.current_freq_r = state.current_freq_r;
        self.last_pan = state.last_pan;
        self.prev_fm_idx_l = Tensor::new(state.prev_fm_idx_l, device)?;
        self.prev_fm_idx_r = Tensor::new(state.prev_fm_idx_r, device)?;
        self.prev_openness = Tensor::new(state.prev_openness, device)?;
        self.prev_gain_l = Tensor::new(state.prev_gain_l, device)?;
        self.prev_gain_r = Tensor::new(state.prev_gain_r, device)?;
        Ok(())
    }
}

fn levy_radiate(
    tape: &Tensor,
    amp: f32,
    random_pool: &DeviceRandomPool,
    random_index: usize,
) -> CResult<Tensor> {
    let cauchy = random_pool.cauchy(random_index)?;
    let mask_t = random_pool.sparse_mask(random_index + 17)?;
    tape.add(&cauchy.broadcast_mul(&mask_t)?.affine(amp as f64, 0.0)?)?
        .clamp(-1.0f32, 1.0f32)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct HostRuntimeState {
    morph_history: Vec<f32>,
    morph_baseline: Option<f32>,
    warmup_sum: f32,
    field_entropy_sum: f64,
    field_entropy_n: u64,
    total_complexity: f32,
    boost_state: f32,
    prev_loss_vec: [f32; 7],
    prev_archetype: String,
    stagnation_ticks: usize,
    last_temp: f32,
    smoothed_control: SynthesisControl,
    dc_x1_l: f32,
    dc_y1_l: f32,
    dc_x1_r: f32,
    dc_y1_r: f32,
}
impl Default for HostRuntimeState {
    fn default() -> Self {
        Self {
            morph_history: Vec::new(),
            morph_baseline: None,
            warmup_sum: 0.0,
            field_entropy_sum: 0.0,
            field_entropy_n: 0,
            total_complexity: 0.0,
            boost_state: 1.0,
            prev_loss_vec: [0.0; 7],
            prev_archetype: String::new(),
            stagnation_ticks: 0,
            last_temp: 0.0,
            smoothed_control: SynthesisControl::default(),
            dc_x1_l: 0.0,
            dc_y1_l: 0.0,
            dc_x1_r: 0.0,
            dc_y1_r: 0.0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct WorldCheckpoint {
    version: u32,
    global_step: u64,
    seed: u64,
    grid_h: usize,
    grid_w: usize,
    ca_channels: usize,
    rng: RuntimeRng,
    micro_tape: Vec<f32>,
    macro_tape: Vec<f32>,
    hidden_mem: Vec<f32>,
    phases: [f32; 4],
    theta_prev: f32,
    theta_prev2: f32,
    model_runtime: ModelRuntimeState,
    rad_amp: f32,
    active_depth: usize,
    energy_state: f32,
    shear_phase: f32,
    uncertainty: AudioUncertaintyState,
    potential: PotentialController,
    semantic: SemanticField,
    criticality: CriticalityEstimator,
    episodic_slots: Vec<Vec<f32>>,
    novelty_spectra: Vec<Vec<f32>>,
    modal_resonators: ModalResonatorBank,
    fdn_l: FractalFDN,
    fdn_r: FractalFDN,
    noise_bank: SpectralNoiseBank,
    controller: HybridController,
    motifs: MotifMemory,
    last_observation: Option<AudioObservation>,
    pending_predictor_input: Option<Vec<f32>>,
    spectral_history: Vec<f32>,
    spectral_prev_mags: Vec<f32>,
    movement_history: Vec<f32>,
    adaptive_dynamics: AdaptiveDynamics,
    motif_diagnostics: MotifDiagnostics,
    host_runtime: HostRuntimeState,
    rg_observer: RgObserver,
    critical_manifold: CriticalManifold,
    state_space: StateSpaceTracker,
    reactor_planner: ReactorPlanner,
    attractors: AttractorMemory,
    probe_controller: ProbeController,
    confinement_observer: ConfinementObserver,
    recovery_controller: RecoveryController,
    last_reactor_state: Option<ReactorState>,
    pending_reactor_command: Option<ActionCommand>,
}

// Exact v7 layout for migration into the confinement-aware V8 control plane.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct LegacyWorldCheckpointV7 {
    version: u32,
    global_step: u64,
    seed: u64,
    grid_h: usize,
    grid_w: usize,
    ca_channels: usize,
    rng: RuntimeRng,
    micro_tape: Vec<f32>,
    macro_tape: Vec<f32>,
    hidden_mem: Vec<f32>,
    phases: [f32; 4],
    theta_prev: f32,
    theta_prev2: f32,
    model_runtime: ModelRuntimeState,
    rad_amp: f32,
    active_depth: usize,
    energy_state: f32,
    shear_phase: f32,
    uncertainty: AudioUncertaintyState,
    potential: PotentialController,
    semantic: SemanticField,
    criticality: CriticalityEstimator,
    episodic_slots: Vec<Vec<f32>>,
    novelty_spectra: Vec<Vec<f32>>,
    modal_resonators: ModalResonatorBank,
    fdn_l: FractalFDN,
    fdn_r: FractalFDN,
    noise_bank: SpectralNoiseBank,
    controller: HybridController,
    motifs: MotifMemory,
    last_observation: Option<AudioObservation>,
    pending_predictor_input: Option<Vec<f32>>,
    spectral_history: Vec<f32>,
    spectral_prev_mags: Vec<f32>,
    movement_history: Vec<f32>,
    adaptive_dynamics: AdaptiveDynamics,
    motif_diagnostics: MotifDiagnostics,
    host_runtime: HostRuntimeState,
    rg_observer: RgObserver,
    critical_manifold: CriticalManifold,
    state_space: StateSpaceTracker,
    reactor_planner: ReactorPlanner,
    attractors: AttractorMemory,
    probe_controller: ProbeController,
    last_reactor_state: Option<ReactorState>,
    pending_reactor_command: Option<ActionCommand>,
}
impl LegacyWorldCheckpointV7 {
    fn validate(&self) -> Result<()> {
        if self.version != 7 {
            anyhow::bail!(
                "legacy checkpoint reports version {} instead of 7",
                self.version
            );
        }
        let ca_n = CA_CHANNELS * GRID_H * GRID_W;
        if self.grid_h != GRID_H
            || self.grid_w != GRID_W
            || self.ca_channels != CA_CHANNELS
            || self.micro_tape.len() != ca_n
            || self.macro_tape.len() != ca_n
            || self.hidden_mem.len() != MEMORY_DIM
            || self.spectral_prev_mags.len() != CHUNK_SIZE / 2
        {
            anyhow::bail!("legacy v7 world dimensions or tensor sizes are invalid");
        }
        Ok(())
    }

    fn migrate(self) -> WorldCheckpoint {
        WorldCheckpoint {
            version: WORLD_VERSION,
            global_step: self.global_step,
            seed: self.seed,
            grid_h: self.grid_h,
            grid_w: self.grid_w,
            ca_channels: self.ca_channels,
            rng: self.rng,
            micro_tape: self.micro_tape,
            macro_tape: self.macro_tape,
            hidden_mem: self.hidden_mem,
            phases: self.phases,
            theta_prev: self.theta_prev,
            theta_prev2: self.theta_prev2,
            model_runtime: self.model_runtime,
            rad_amp: self.rad_amp,
            active_depth: self.active_depth,
            energy_state: self.energy_state,
            shear_phase: self.shear_phase,
            uncertainty: self.uncertainty,
            potential: self.potential,
            semantic: self.semantic,
            criticality: self.criticality,
            episodic_slots: self.episodic_slots,
            novelty_spectra: self.novelty_spectra,
            modal_resonators: self.modal_resonators,
            fdn_l: self.fdn_l,
            fdn_r: self.fdn_r,
            noise_bank: self.noise_bank,
            controller: self.controller,
            motifs: self.motifs,
            last_observation: self.last_observation,
            pending_predictor_input: self.pending_predictor_input,
            spectral_history: self.spectral_history,
            spectral_prev_mags: self.spectral_prev_mags,
            movement_history: self.movement_history,
            adaptive_dynamics: self.adaptive_dynamics,
            motif_diagnostics: self.motif_diagnostics,
            host_runtime: self.host_runtime,
            rg_observer: self.rg_observer,
            critical_manifold: self.critical_manifold,
            state_space: self.state_space,
            reactor_planner: self.reactor_planner,
            attractors: self.attractors,
            probe_controller: self.probe_controller,
            confinement_observer: ConfinementObserver::default(),
            recovery_controller: RecoveryController::default(),
            last_reactor_state: self.last_reactor_state,
            pending_reactor_command: self.pending_reactor_command,
        }
    }
}

// Exact v6 layout for one-time migration into the current reactor.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct LegacyWorldCheckpointV6 {
    version: u32,
    global_step: u64,
    seed: u64,
    grid_h: usize,
    grid_w: usize,
    ca_channels: usize,
    rng: RuntimeRng,
    micro_tape: Vec<f32>,
    macro_tape: Vec<f32>,
    hidden_mem: Vec<f32>,
    phases: [f32; 4],
    theta_prev: f32,
    theta_prev2: f32,
    model_runtime: ModelRuntimeState,
    rad_amp: f32,
    active_depth: usize,
    energy_state: f32,
    shear_phase: f32,
    uncertainty: AudioUncertaintyState,
    potential: PotentialController,
    semantic: SemanticField,
    criticality: CriticalityEstimator,
    episodic_slots: Vec<Vec<f32>>,
    novelty_spectra: Vec<Vec<f32>>,
    modal_resonators: ModalResonatorBank,
    fdn_l: FractalFDN,
    fdn_r: FractalFDN,
    noise_bank: SpectralNoiseBank,
    controller: HybridController,
    motifs: MotifMemory,
    last_observation: Option<AudioObservation>,
    pending_predictor_input: Option<Vec<f32>>,
    spectral_history: Vec<f32>,
    spectral_prev_mags: Vec<f32>,
    movement_history: Vec<f32>,
    adaptive_dynamics: AdaptiveDynamics,
    motif_diagnostics: MotifDiagnostics,
    host_runtime: HostRuntimeState,
}

// Exact v5 layout for one-time world migration. Neural tensor shapes are
// unchanged in v6, so the learned organism can continue while the new
// anti-stagnation state starts from calibrated defaults.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct LegacyWorldCheckpointV5 {
    version: u32,
    global_step: u64,
    seed: u64,
    grid_h: usize,
    grid_w: usize,
    ca_channels: usize,
    rng: RuntimeRng,
    micro_tape: Vec<f32>,
    macro_tape: Vec<f32>,
    hidden_mem: Vec<f32>,
    phases: [f32; 4],
    theta_prev: f32,
    theta_prev2: f32,
    model_runtime: ModelRuntimeState,
    rad_amp: f32,
    active_depth: usize,
    energy_state: f32,
    shear_phase: f32,
    uncertainty: AudioUncertaintyState,
    potential: PotentialController,
    semantic: SemanticField,
    criticality: CriticalityEstimator,
    episodic_slots: Vec<Vec<f32>>,
    novelty_spectra: Vec<Vec<f32>>,
    modal_resonators: ModalResonatorBank,
    fdn_l: FractalFDN,
    fdn_r: FractalFDN,
    noise_bank: SpectralNoiseBank,
    controller: HybridController,
    motifs: MotifMemory,
    last_observation: Option<AudioObservation>,
    pending_predictor_input: Option<Vec<f32>>,
    spectral_history: Vec<f32>,
    spectral_prev_mags: Vec<f32>,
    movement_history: Vec<f32>,
    host_runtime: HostRuntimeState,
}
impl LegacyWorldCheckpointV6 {
    fn validate(&self) -> Result<()> {
        if self.version != 6 {
            anyhow::bail!(
                "legacy checkpoint reports version {} instead of 6",
                self.version
            );
        }
        if self.grid_h != GRID_H || self.grid_w != GRID_W || self.ca_channels != CA_CHANNELS {
            anyhow::bail!("legacy v6 world dimensions do not match this build");
        }
        let ca_n = CA_CHANNELS * GRID_H * GRID_W;
        if self.micro_tape.len() != ca_n
            || self.macro_tape.len() != ca_n
            || self.hidden_mem.len() != MEMORY_DIM
            || self.spectral_prev_mags.len() != CHUNK_SIZE / 2
        {
            anyhow::bail!("legacy v6 world checkpoint tensor sizes are invalid or truncated");
        }
        if self.active_depth == 0
            || self.active_depth > MORPH_MAX_BLOCKS
            || self.episodic_slots.iter().any(|v| v.len() != MEMORY_DIM)
            || self.novelty_spectra.iter().any(|v| v.len() != SPEC_BINS)
            || !self.fdn_l.is_valid()
            || !self.fdn_r.is_valid()
        {
            anyhow::bail!("legacy v6 world checkpoint adaptive state is malformed");
        }
        Ok(())
    }
    fn migrate(self) -> WorldCheckpoint {
        WorldCheckpoint {
            version: WORLD_VERSION,
            global_step: self.global_step,
            seed: self.seed,
            grid_h: self.grid_h,
            grid_w: self.grid_w,
            ca_channels: self.ca_channels,
            rng: self.rng,
            micro_tape: self.micro_tape,
            macro_tape: self.macro_tape,
            hidden_mem: self.hidden_mem,
            phases: self.phases,
            theta_prev: self.theta_prev,
            theta_prev2: self.theta_prev2,
            model_runtime: self.model_runtime,
            rad_amp: self.rad_amp,
            active_depth: self.active_depth,
            energy_state: self.energy_state,
            shear_phase: self.shear_phase,
            uncertainty: self.uncertainty,
            potential: self.potential,
            semantic: self.semantic,
            criticality: self.criticality,
            episodic_slots: self.episodic_slots,
            novelty_spectra: self.novelty_spectra,
            modal_resonators: self.modal_resonators,
            fdn_l: self.fdn_l,
            fdn_r: self.fdn_r,
            noise_bank: self.noise_bank,
            controller: self.controller,
            motifs: self.motifs,
            last_observation: self.last_observation,
            pending_predictor_input: self.pending_predictor_input,
            spectral_history: self.spectral_history,
            spectral_prev_mags: self.spectral_prev_mags,
            movement_history: self.movement_history,
            adaptive_dynamics: self.adaptive_dynamics,
            motif_diagnostics: self.motif_diagnostics,
            host_runtime: self.host_runtime,
            rg_observer: RgObserver::default(),
            critical_manifold: CriticalManifold::default(),
            state_space: StateSpaceTracker::default(),
            reactor_planner: ReactorPlanner::new(self.seed),
            attractors: AttractorMemory::default(),
            probe_controller: ProbeController::new(true),
            confinement_observer: ConfinementObserver::default(),
            recovery_controller: RecoveryController::default(),
            last_reactor_state: None,
            pending_reactor_command: None,
        }
    }
}

impl LegacyWorldCheckpointV5 {
    fn validate(&self) -> Result<()> {
        if self.version != 5 {
            anyhow::bail!(
                "legacy checkpoint reports version {} instead of 5",
                self.version
            );
        }
        if self.grid_h != GRID_H || self.grid_w != GRID_W || self.ca_channels != CA_CHANNELS {
            anyhow::bail!("legacy v5 world dimensions do not match this build");
        }
        let ca_n = CA_CHANNELS * GRID_H * GRID_W;
        if self.micro_tape.len() != ca_n
            || self.macro_tape.len() != ca_n
            || self.hidden_mem.len() != MEMORY_DIM
            || self.spectral_prev_mags.len() != CHUNK_SIZE / 2
        {
            anyhow::bail!("legacy v5 world checkpoint tensor sizes are invalid or truncated");
        }
        if self.active_depth == 0
            || self.active_depth > MORPH_MAX_BLOCKS
            || self.episodic_slots.iter().any(|v| v.len() != MEMORY_DIM)
            || self.novelty_spectra.iter().any(|v| v.len() != SPEC_BINS)
            || !self.fdn_l.is_valid()
            || !self.fdn_r.is_valid()
        {
            anyhow::bail!("legacy v5 world checkpoint adaptive state is malformed");
        }
        Ok(())
    }
    fn migrate(self) -> WorldCheckpoint {
        let mut controller = self.controller;
        controller.bandit = ModelFreeBandit::default();
        controller.action_age = 0;
        let motif_count = self.motifs.entries.len() as u64;
        WorldCheckpoint {
            version: WORLD_VERSION,
            global_step: self.global_step,
            seed: self.seed,
            grid_h: self.grid_h,
            grid_w: self.grid_w,
            ca_channels: self.ca_channels,
            rng: self.rng,
            micro_tape: self.micro_tape,
            macro_tape: self.macro_tape,
            hidden_mem: self.hidden_mem,
            phases: self.phases,
            theta_prev: self.theta_prev,
            theta_prev2: self.theta_prev2,
            model_runtime: self.model_runtime,
            rad_amp: self.rad_amp,
            active_depth: self.active_depth,
            energy_state: self.energy_state,
            shear_phase: self.shear_phase,
            uncertainty: self.uncertainty,
            potential: self.potential,
            semantic: self.semantic,
            criticality: self.criticality,
            episodic_slots: self.episodic_slots,
            novelty_spectra: self.novelty_spectra,
            modal_resonators: self.modal_resonators,
            fdn_l: self.fdn_l,
            fdn_r: self.fdn_r,
            noise_bank: self.noise_bank,
            controller,
            motifs: self.motifs,
            last_observation: self.last_observation,
            pending_predictor_input: self.pending_predictor_input,
            spectral_history: self.spectral_history,
            spectral_prev_mags: self.spectral_prev_mags,
            movement_history: self.movement_history,
            adaptive_dynamics: AdaptiveDynamics::default(),
            motif_diagnostics: MotifDiagnostics {
                stored_total: motif_count,
                ..MotifDiagnostics::default()
            },
            host_runtime: self.host_runtime,
            rg_observer: RgObserver::default(),
            critical_manifold: CriticalManifold::default(),
            state_space: StateSpaceTracker::default(),
            reactor_planner: ReactorPlanner::new(self.seed),
            attractors: AttractorMemory::default(),
            probe_controller: ProbeController::new(true),
            confinement_observer: ConfinementObserver::default(),
            recovery_controller: RecoveryController::default(),
            last_reactor_state: None,
            pending_reactor_command: None,
        }
    }
}

impl WorldCheckpoint {
    fn validate(&self) -> Result<()> {
        if self.version != WORLD_VERSION {
            anyhow::bail!(
                "world checkpoint version {} is not supported by v{}",
                self.version,
                WORLD_VERSION
            );
        }
        if self.grid_h != GRID_H || self.grid_w != GRID_W || self.ca_channels != CA_CHANNELS {
            anyhow::bail!(
                "world dimensions {}ch x {}x{} do not match this build ({}ch x {}x{})",
                self.ca_channels,
                self.grid_h,
                self.grid_w,
                CA_CHANNELS,
                GRID_H,
                GRID_W
            );
        }
        let ca_n = CA_CHANNELS * GRID_H * GRID_W;
        if self.micro_tape.len() != ca_n
            || self.macro_tape.len() != ca_n
            || self.hidden_mem.len() != MEMORY_DIM
        {
            anyhow::bail!("world checkpoint tensor sizes are invalid or truncated");
        }
        if self.spectral_prev_mags.len() != CHUNK_SIZE / 2 {
            anyhow::bail!("world checkpoint spectral memory has the wrong size");
        }
        if self.active_depth == 0 || self.active_depth > MORPH_MAX_BLOCKS {
            anyhow::bail!("world checkpoint morphic depth is invalid");
        }
        if self.episodic_slots.iter().any(|v| v.len() != MEMORY_DIM)
            || self.novelty_spectra.iter().any(|v| v.len() != SPEC_BINS)
        {
            anyhow::bail!("world checkpoint memory slots are malformed");
        }
        if !self.fdn_l.is_valid() || !self.fdn_r.is_valid() {
            anyhow::bail!("world checkpoint feedback-delay state is malformed");
        }
        let transition_input = REACTOR_DIM + ACTION_FEATURE_DIM;
        let malformed_transition = self.reactor_planner.ensemble.models.iter().any(|m| {
            m.w1.len() != TRANSITION_HIDDEN * transition_input
                || m.b1.len() != TRANSITION_HIDDEN
                || m.w2.len() != REACTOR_DIM * TRANSITION_HIDDEN
                || m.b2.len() != REACTOR_DIM
        });
        if self.reactor_planner.ensemble.models.len() != TRANSITION_ENSEMBLE_SIZE
            || malformed_transition
            || self.reactor_planner.ensemble.replay.len() > TRANSITION_REPLAY_CAPACITY
            || self.attractors.entries.len() > ATTRACTOR_SLOTS
            || self.state_space.counts.len() > STATE_SPACE_MAX_BINS
            || self.state_space.transitions.len() > STATE_SPACE_MAX_BINS * 2
        {
            anyhow::bail!(
                "world checkpoint reactor planner, state-space, or attractor memory is malformed"
            );
        }
        Ok(())
    }
}

fn ensure_parent_dir(path: &str) -> Result<()> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ComputeBackend {
    Auto,
    Cpu,
    Cuda,
}

impl ComputeBackend {
    fn parse(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "cuda" | "gpu" => Ok(Self::Cuda),
            _ => anyhow::bail!("unknown device '{value}' (use auto, cpu, or cuda)"),
        }
    }
}

fn select_device(backend: ComputeBackend, cuda_ordinal: usize) -> Result<(Device, &'static str)> {
    match backend {
        ComputeBackend::Cpu => Ok((Device::Cpu, "cpu")),
        ComputeBackend::Cuda => Ok((Device::new_cuda(cuda_ordinal)?, "cuda")),
        ComputeBackend::Auto => match Device::new_cuda(cuda_ordinal) {
            Ok(device) => Ok((device, "cuda")),
            Err(error) => {
                eprintln!(
                    "--> CUDA device {} unavailable ({}); falling back to CPU.",
                    cuda_ordinal, error
                );
                Ok((Device::Cpu, "cpu"))
            }
        },
    }
}

fn checkpoint_checksum(bytes: &[u8]) -> u64 {
    // FNV-1a is not cryptographic; it is a fast corruption/truncation guard.
    let mut h = 0xcbf29ce484222325u64;
    for &b in bytes {
        h = (h ^ b as u64).wrapping_mul(0x100000001b3);
    }
    h
}

fn atomic_save_world(path: &str, checkpoint: &WorldCheckpoint) -> Result<()> {
    let tmp = format!("{}.tmp", path);
    let payload = bincode::serialize(checkpoint)?;
    let mut bytes = Vec::with_capacity(24 + payload.len());
    bytes.extend_from_slice(&WORLD_MAGIC);
    bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&checkpoint_checksum(&payload).to_le_bytes());
    bytes.extend_from_slice(&payload);
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn load_world(path: &str) -> Result<WorldCheckpoint> {
    let bytes = std::fs::read(path)?;
    if bytes.len() < 24 {
        anyhow::bail!("world checkpoint has an invalid or truncated header");
    }
    let magic = &bytes[..8];
    if magic != WORLD_MAGIC.as_slice()
        && magic != LEGACY_WORLD_MAGIC_V7.as_slice()
        && magic != LEGACY_WORLD_MAGIC_V6.as_slice()
        && magic != LEGACY_WORLD_MAGIC_V5.as_slice()
    {
        anyhow::bail!("world checkpoint is neither TITAN v8 nor a migratable v7/v6/v5 world");
    }
    let payload_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    let expected_checksum = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
    if payload_len != bytes.len() - 24 {
        anyhow::bail!("world checkpoint is truncated");
    }
    let payload = &bytes[24..];
    if checkpoint_checksum(payload) != expected_checksum {
        anyhow::bail!("world checkpoint checksum mismatch");
    }
    if magic == WORLD_MAGIC.as_slice() {
        let checkpoint: WorldCheckpoint = bincode::deserialize(payload)?;
        checkpoint.validate()?;
        Ok(checkpoint)
    } else if magic == LEGACY_WORLD_MAGIC_V7.as_slice() {
        let legacy: LegacyWorldCheckpointV7 = bincode::deserialize(payload)?;
        legacy.validate()?;
        println!("--> Migrating TITAN v7 organism into the v8 confinement controller.");
        let checkpoint = legacy.migrate();
        checkpoint.validate()?;
        Ok(checkpoint)
    } else if magic == LEGACY_WORLD_MAGIC_V6.as_slice() {
        let legacy: LegacyWorldCheckpointV6 = bincode::deserialize(payload)?;
        legacy.validate()?;
        println!("--> Migrating TITAN v6 organism into the v8 recursive confinement reactor.");
        let checkpoint = legacy.migrate();
        checkpoint.validate()?;
        Ok(checkpoint)
    } else {
        let legacy: LegacyWorldCheckpointV5 = bincode::deserialize(payload)?;
        legacy.validate()?;
        println!("--> Migrating TITAN v5 organism through the v8 compatibility path.");
        let checkpoint = legacy.migrate();
        checkpoint.validate()?;
        Ok(checkpoint)
    }
}

fn flatten_tensor(t: &Tensor) -> Result<Vec<f32>> {
    Ok(t.flatten_all()?.to_vec1::<f32>()?)
}

fn novelty_snapshot(buf: &VecDeque<Tensor>) -> Result<Vec<Vec<f32>>> {
    let mut out = Vec::with_capacity(buf.len());
    for t in buf {
        out.push(t.flatten_all()?.to_vec1::<f32>()?);
    }
    Ok(out)
}

fn restore_novelty(values: &[Vec<f32>], device: &Device) -> Result<VecDeque<Tensor>> {
    let mut out = VecDeque::with_capacity(NOVELTY_SLOTS);
    for v in values
        .iter()
        .rev()
        .take(NOVELTY_SLOTS)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
    {
        if v.len() == SPEC_BINS {
            out.push_back(Tensor::from_vec(v.clone(), (1, SPEC_BINS), device)?);
        }
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn capture_world(
    global_step: u64,
    seed: u64,
    rng: &RuntimeRng,
    micro_tape: &Tensor,
    macro_tape: &Tensor,
    hidden_mem: &Tensor,
    phases: [f32; 4],
    theta_prev: f32,
    theta_prev2: f32,
    model: &ComplexAudioEcosystem,
    rad_amp: f32,
    energy_state: f32,
    shear_phase: f32,
    uncertainty: &AudioUncertaintyState,
    potential: &PotentialController,
    semantic: &SemanticField,
    criticality: &CriticalityEstimator,
    episodic: &EpisodicMemory,
    novelty_buf: &VecDeque<Tensor>,
    modal_resonators: &ModalResonatorBank,
    fdn_l: &FractalFDN,
    fdn_r: &FractalFDN,
    noise_bank: &SpectralNoiseBank,
    controller: &HybridController,
    motifs: &MotifMemory,
    last_observation: &Option<AudioObservation>,
    pending_predictor_input: &Option<Tensor>,
    spectral_mon: &SpectralEntropyMonitor,
    movement_mon: &MovementCoherenceMonitor,
    adaptive_dynamics: &AdaptiveDynamics,
    motif_diagnostics: &MotifDiagnostics,
    host_runtime: &HostRuntimeState,
    rg_observer: &RgObserver,
    critical_manifold: &CriticalManifold,
    state_space: &StateSpaceTracker,
    reactor_planner: &ReactorPlanner,
    attractors: &AttractorMemory,
    probe_controller: &ProbeController,
    confinement_observer: &ConfinementObserver,
    recovery_controller: &RecoveryController,
    last_reactor_state: &Option<ReactorState>,
    pending_reactor_command: &Option<ActionCommand>,
) -> Result<WorldCheckpoint> {
    Ok(WorldCheckpoint {
        version: WORLD_VERSION,
        global_step,
        seed,
        grid_h: GRID_H,
        grid_w: GRID_W,
        ca_channels: CA_CHANNELS,
        rng: rng.clone(),
        micro_tape: flatten_tensor(micro_tape)?,
        macro_tape: flatten_tensor(macro_tape)?,
        hidden_mem: flatten_tensor(hidden_mem)?,
        phases,
        theta_prev,
        theta_prev2,
        model_runtime: model.runtime_state()?,
        rad_amp,
        active_depth: model.depth(),
        energy_state,
        shear_phase,
        uncertainty: uncertainty.clone(),
        potential: potential.clone(),
        semantic: semantic.clone(),
        criticality: criticality.clone(),
        episodic_slots: episodic.snapshot_host()?,
        novelty_spectra: novelty_snapshot(novelty_buf)?,
        modal_resonators: modal_resonators.clone(),
        fdn_l: fdn_l.clone(),
        fdn_r: fdn_r.clone(),
        noise_bank: noise_bank.clone(),
        controller: controller.clone(),
        motifs: motifs.clone(),
        last_observation: last_observation.clone(),
        pending_predictor_input: match pending_predictor_input {
            Some(t) => Some(flatten_tensor(t)?),
            None => None,
        },
        spectral_history: spectral_mon.history_snapshot(),
        spectral_prev_mags: spectral_mon.prev_mags_snapshot(),
        movement_history: movement_mon.history_snapshot(),
        adaptive_dynamics: adaptive_dynamics.clone(),
        motif_diagnostics: motif_diagnostics.clone(),
        host_runtime: host_runtime.clone(),
        rg_observer: rg_observer.clone(),
        critical_manifold: critical_manifold.clone(),
        state_space: state_space.clone(),
        reactor_planner: reactor_planner.clone(),
        attractors: attractors.clone(),
        probe_controller: probe_controller.clone(),
        confinement_observer: confinement_observer.clone(),
        recovery_controller: recovery_controller.clone(),
        last_reactor_state: last_reactor_state.clone(),
        pending_reactor_command: *pending_reactor_command,
    })
}

// --- MAIN RUNTIME LOGIC ---
fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut base_dir = "/sdcard/Download".to_string();
    let mut n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8)
        .min(6);
    let mut target_lr = BASE_LR;
    let mut sim_duration = DURATION_SECONDS;
    let mut bptt_window = BPTT_WINDOW;
    let mut fresh_model = false;
    let mut fresh_world = false;
    let mut state_override: Option<String> = None;
    let mut model_override: Option<String> = None;
    let mut import_model_override: Option<String> = None;
    let mut planner_profile = PlannerProfile::Balanced;
    let mut probes_enabled = true;
    let mut backend = ComputeBackend::Auto;
    let mut cuda_ordinal = 0usize;
    let mut seed: u64 = 42;
    let mut arg_idx = 1;
    while arg_idx < args.len() {
        match args[arg_idx].as_str() {
            "--base-dir" | "-b" => {
                if arg_idx + 1 < args.len() {
                    base_dir = args[arg_idx + 1].clone();
                    arg_idx += 2;
                } else {
                    anyhow::bail!("Missing value for --base-dir");
                }
            }
            "--threads" | "-t" => {
                if arg_idx + 1 < args.len() {
                    n_threads = args[arg_idx + 1].parse::<usize>()?;
                    arg_idx += 2;
                } else {
                    anyhow::bail!("Missing value for --threads");
                }
            }
            "--lr" | "-l" => {
                if arg_idx + 1 < args.len() {
                    target_lr = args[arg_idx + 1].parse::<f64>()?;
                    arg_idx += 2;
                } else {
                    anyhow::bail!("Missing value for --lr");
                }
            }
            "--duration" | "-d" => {
                if arg_idx + 1 < args.len() {
                    sim_duration = args[arg_idx + 1].parse::<f32>()?;
                    arg_idx += 2;
                } else {
                    anyhow::bail!("Missing value for --duration");
                }
            }
            "--bptt" | "-w" => {
                if arg_idx + 1 < args.len() {
                    bptt_window = args[arg_idx + 1].parse::<usize>()?;
                    arg_idx += 2;
                } else {
                    anyhow::bail!("Missing value for --bptt");
                }
            }
            "--seed" | "-s" => {
                if arg_idx + 1 < args.len() {
                    seed = args[arg_idx + 1].parse::<u64>()?;
                    arg_idx += 2;
                } else {
                    anyhow::bail!("Missing value for --seed");
                }
            }
            "--state" => {
                if arg_idx + 1 < args.len() {
                    state_override = Some(args[arg_idx + 1].clone());
                    arg_idx += 2;
                } else {
                    anyhow::bail!("Missing value for --state");
                }
            }
            "--model" => {
                if arg_idx + 1 < args.len() {
                    model_override = Some(args[arg_idx + 1].clone());
                    arg_idx += 2;
                } else {
                    anyhow::bail!("Missing value for --model");
                }
            }
            "--import-model" => {
                if arg_idx + 1 < args.len() {
                    import_model_override = Some(args[arg_idx + 1].clone());
                    arg_idx += 2;
                } else {
                    anyhow::bail!("Missing value for --import-model");
                }
            }
            "--planner-profile" => {
                if arg_idx + 1 < args.len() {
                    planner_profile = PlannerProfile::parse(&args[arg_idx + 1])?;
                    arg_idx += 2;
                } else {
                    anyhow::bail!("Missing value for --planner-profile");
                }
            }
            "--device" => {
                if arg_idx + 1 < args.len() {
                    backend = ComputeBackend::parse(&args[arg_idx + 1])?;
                    arg_idx += 2;
                } else {
                    anyhow::bail!("Missing value for --device");
                }
            }
            "--cuda-device" => {
                if arg_idx + 1 < args.len() {
                    cuda_ordinal = args[arg_idx + 1].parse::<usize>()?;
                    arg_idx += 2;
                } else {
                    anyhow::bail!("Missing value for --cuda-device");
                }
            }
            "--no-probes" => {
                probes_enabled = false;
                arg_idx += 1;
            }
            "--fresh-world" => {
                fresh_world = true;
                arg_idx += 1;
            }
            "--fresh-model" | "--fresh" | "-f" => {
                fresh_model = true;
                fresh_world = true;
                arg_idx += 1;
            }
            "--help" | "-h" => {
                println!(
                    "TITAN v8 Confinement Reactor\n\n\
Usage: titan [BASE_DIR] [options]\n\n\
  -b, --base-dir DIR   Output/training root (default /sdcard/Download)\n\
  -d, --duration SEC   Render duration (default 240)\n\
  -t, --threads N      Rayon/Candle CPU threads (S25 Ultra default 6)\n\
  -w, --bptt N         Truncated-BPTT window (default 8)\n\
  -l, --lr VALUE       Base AdamW learning rate\n\
  -s, --seed N         Seed for a fresh deterministic organism\n\
      --planner-profile P  cool, balanced, or max (default balanced)\n\
      --device BACKEND    auto, cuda, or cpu (default auto)\n\
      --cuda-device N     CUDA device ordinal (default 0)\n\
      --no-probes      Disable subtle perturbation-response experiments\n\
      --state PATH     World-checkpoint path\n\
      --model PATH     v8 model output/resume path\n\
      --import-model P Import compatible v4/v5/v6/v7 weights\n\
      --fresh-world    Reset CA/DSP/reactor state while retaining weights\n\
  -f, --fresh-model    Reset both learned weights and the world\n\
\nCtrl-C finishes the active chunk, finalizes audio, and saves the reactor.\n"
                );
                return Ok(());
            }
            _ => {
                if arg_idx == 1 && !args[arg_idx].starts_with('-') {
                    base_dir = args[arg_idx].clone();
                    arg_idx += 1;
                } else {
                    println!("Unknown parameter: {}", args[arg_idx]);
                    arg_idx += 1;
                }
            }
        }
    }
    let available_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);
    n_threads = n_threads.max(1).min(available_threads.max(1));
    bptt_window = bptt_window.max(1);
    sim_duration = sim_duration.max(CHUNK_SIZE as f32 / SAMPLE_RATE as f32);
    rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global()?;
    let (device, device_label) = select_device(backend, cuda_ordinal)?;
    let mut rng = RuntimeRng::seed_from_u64(seed);
    let _wake_lock = WakeLockGuard::acquire();
    let keep_running = Arc::new(AtomicBool::new(true));
    {
        let keep_running = Arc::clone(&keep_running);
        ctrlc::set_handler(move || {
            keep_running.store(false, AtomicOrdering::SeqCst);
        })?;
    }
    std::fs::create_dir_all(&base_dir)?;
    let planner_config = PlannerConfig::for_profile(planner_profile);
    println!("=== TITAN AUDIO ECOSYSTEM: RUST EDITION v8 (CONFINEMENT REACTOR) ===");
    println!("Seed: {} | Device: {} | Threads: {} | BPTT: {} | Planner: {} h{} b{} | Field: {}ch x {}x{} torus | LR: {:.2e} | Duration: {}s", seed, device_label, n_threads, bptt_window, planner_profile.label(), planner_config.horizon, planner_config.beam_width, CA_CHANNELS, GRID_H, GRID_W, target_lr, sim_duration);
    println!("NOTE: CA/DSP/RNG state resumes from the world checkpoint; AdamW moment buffers restart per process. Fresh reproducibility also requires the same --threads value.");

    let wav_dir = format!("{}/OLD_WAVS", base_dir);
    let explicit_model_path = model_override.is_some();
    let explicit_world_path = state_override.is_some();
    let model_path =
        model_override.unwrap_or_else(|| format!("{}/titan_model_v8.safetensors", base_dir));
    let legacy_model_path = format!("{}/titan_model_v7.safetensors", base_dir);
    let legacy_model_path_v6 = format!("{}/titan_model_v6.safetensors", base_dir);
    let legacy_model_path_v5 = format!("{}/titan_model_v5.safetensors", base_dir);
    let importing_model = import_model_override.is_some();
    let load_model_path = if let Some(path) = import_model_override {
        path
    } else if std::path::Path::new(&model_path).exists() {
        model_path.clone()
    } else if !explicit_model_path && std::path::Path::new(&legacy_model_path).exists() {
        println!(
            "--> No v8 model yet; importing compatible v7 weights from {}.",
            legacy_model_path
        );
        legacy_model_path.clone()
    } else if !explicit_model_path && std::path::Path::new(&legacy_model_path_v6).exists() {
        println!(
            "--> No v8/v7 model yet; importing compatible v6 weights from {}.",
            legacy_model_path_v6
        );
        legacy_model_path_v6.clone()
    } else if !explicit_model_path && std::path::Path::new(&legacy_model_path_v5).exists() {
        println!(
            "--> No v8/v7/v6 model yet; importing compatible v5 weights from {}.",
            legacy_model_path_v5
        );
        legacy_model_path_v5.clone()
    } else {
        model_path.clone()
    };
    let world_path = state_override.unwrap_or_else(|| format!("{}/titan_world_v8.bin", base_dir));
    let legacy_world_path = format!("{}/titan_world_v7.bin", base_dir);
    let legacy_world_path_v6 = format!("{}/titan_world_v6.bin", base_dir);
    let legacy_world_path_v5 = format!("{}/titan_world_v5.bin", base_dir);
    let load_world_path = if std::path::Path::new(&world_path).exists() {
        world_path.clone()
    } else if !explicit_world_path && std::path::Path::new(&legacy_world_path).exists() {
        legacy_world_path.clone()
    } else if !explicit_world_path && std::path::Path::new(&legacy_world_path_v6).exists() {
        legacy_world_path_v6.clone()
    } else if !explicit_world_path && std::path::Path::new(&legacy_world_path_v5).exists() {
        legacy_world_path_v5.clone()
    } else {
        world_path.clone()
    };
    let morph_path = format!("{}/titan_morph_state_v8.json", base_dir);
    ensure_parent_dir(&model_path)?;
    ensure_parent_dir(&world_path)?;
    let target_loader = TargetAudioLoader::new(&wav_dir)?;
    let varmap = VarMap::new();
    let vb = VBV::from_varmap(&varmap, DType::F32, &device);
    let mut model = ComplexAudioEcosystem::new(vb.pp("model"), &device)?;
    let arbiter = AudioArbiter::new(vb.pp("arbiter"))?;
    let monitor_head = MonitorHead::new(vb.pp("monitor_head"))?;
    let mut episodic = EpisodicMemory::new(vb.pp("episodic"))?;
    let spec_proj = SpectralProjector::new(&device).map_err(anyhow::Error::msg)?;
    let spec_proj_fine =
        SpectralProjector::new_with(1024, FINE_SPEC_BINS, &device).map_err(anyhow::Error::msg)?;

    // Initialize the full V8 neural parameter set deterministically first, then
    // load every compatible tensor from an older or current checkpoint on top.
    // V8 adds host-side confinement control, so compatible V7/V6/V5 neural weights can
    // continue without discarding the learned CA and synthesis mappings.
    let initialized = deterministic_reinit(&varmap, seed, &device)?;
    let mut loaded_full = false;
    let mut loaded_any = false;
    if fresh_model {
        println!("==> FRESH MODEL: ignoring model and world checkpoints.");
        println!(
            "--> Deterministic init: {} tensors seeded from {}.",
            initialized, seed
        );
        fresh_world = true;
    } else if std::path::Path::new(&load_model_path).exists() {
        match load_into_varmap(&varmap, &load_model_path, &device) {
            Ok((hit, miss, mismatch)) => {
                loaded_any = hit > 0;
                loaded_full = miss == 0 && mismatch == 0 && loaded_any;
                println!("--> Loaded {} compatible tensors from {} ({} new/missing, {} shape-mismatched)",
                    hit, load_model_path, miss, mismatch);
                if !loaded_full {
                    println!("--> Migrated checkpoint: compatible learned weights retained; new v8 tensors use deterministic seed {}.", seed);
                    fresh_world = true; // old dynamical state does not match the migrated control plane
                }
            }
            Err(e) => {
                println!(
                    "--> Could not load {}: {} — using deterministic fresh weights",
                    load_model_path, e
                );
                fresh_world = true;
            }
        }
    } else {
        println!(
            "--> Deterministic init: {} tensors seeded from {}.",
            initialized, seed
        );
        fresh_world = true;
    }
    if !loaded_any && !fresh_model && std::path::Path::new(&load_model_path).exists() {
        println!("--> No compatible tensors were found; this run starts as a fresh v8 model.");
    }
    if importing_model && loaded_any {
        println!(
            "--> Import source retained at {}; future checkpoints will be written to {}.",
            load_model_path, model_path
        );
        fresh_world = true;
    }

    let mut rad_amp = RAD_AMP_INIT;
    if !fresh_world && loaded_full {
        if let Ok(txt) = std::fs::read_to_string(&morph_path) {
            if let Ok(j) = serde_json::from_str::<serde_json::Value>(&txt) {
                if let Some(d) = j["active_depth"].as_u64() {
                    model.set_depth(d as usize);
                }
                if let Some(r) = j["rad_amp"].as_f64() {
                    rad_amp = (r as f32).clamp(RAD_AMP_MIN, RAD_AMP_MAX);
                }
            }
        }
    }

    let mut modal_resonators = ModalResonatorBank::new();
    let mut spectral_noise = SpectralNoiseBank::default();
    let mut fractal_fdn_l = FractalFDN::new();
    let mut fractal_fdn_r = FractalFDN::new();
    let mut spectral_mon = SpectralEntropyMonitor::new(20);
    let mut movement_mon = MovementCoherenceMonitor::new(20);
    let mut potential = PotentialController::new();
    let shear_gen = ShearField2D::new(CA_CHANNELS, GRID_H, GRID_W, &device)?;
    let mut shear_phase = 0.0f32;
    let mut uncertainty = AudioUncertaintyState::new();
    let mut semantic = SemanticField::new();
    let mut criticality = CriticalityEstimator::default();
    let mut controller = HybridController::default();
    let mut adaptive_dynamics = AdaptiveDynamics::default();
    let mut motifs = MotifMemory::default();
    let mut motif_diagnostics = MotifDiagnostics::default();
    let mut rg_observer = RgObserver::default();
    let mut critical_manifold = CriticalManifold::default();
    let mut state_space = StateSpaceTracker::default();
    let mut reactor_planner = ReactorPlanner::new(seed);
    let mut attractors = AttractorMemory::default();
    let mut probe_controller = ProbeController::new(probes_enabled);
    let mut confinement_observer = ConfinementObserver::default();
    let mut recovery_controller = RecoveryController::default();
    let mut last_reactor_state: Option<ReactorState> = None;
    let mut pending_reactor_command: Option<ActionCommand> = None;
    let mut last_observation: Option<AudioObservation> = None;
    let mut pending_predictor_input: Option<Tensor> = None;
    let mut global_step: u64 = 0;
    let mut novelty_buf: VecDeque<Tensor> = VecDeque::with_capacity(NOVELTY_SLOTS);

    let mut host_runtime = HostRuntimeState::default();
    let mut optimizer = AdamW::new_lr(varmap.all_vars(), target_lr).map_err(anyhow::Error::msg)?;

    let mut micro_tape = randn_t(&mut rng, &[1, CA_CHANNELS, GRID_H, GRID_W], 1.0, &device)
        .map_err(anyhow::Error::msg)?;
    let mut macro_tape = randn_t(&mut rng, &[1, CA_CHANNELS, GRID_H, GRID_W], 1.0, &device)
        .map_err(anyhow::Error::msg)?;
    let mut hidden_mem =
        Tensor::zeros((1, MEMORY_DIM), DType::F32, &device).map_err(anyhow::Error::msg)?;
    let mut phases = [0.0f32; 4];
    let mut theta_prev = 0.0f32;
    let mut theta_prev2 = 0.0f32;
    let mut energy_state = POT_ENERGY_SET;

    let mut loaded_world = false;
    if !fresh_world && std::path::Path::new(&load_world_path).exists() {
        match load_world(&load_world_path) {
            Ok(world) => {
                global_step = world.global_step;
                seed = world.seed;
                rng = world.rng;
                micro_tape =
                    Tensor::from_vec(world.micro_tape, (1, CA_CHANNELS, GRID_H, GRID_W), &device)?;
                macro_tape =
                    Tensor::from_vec(world.macro_tape, (1, CA_CHANNELS, GRID_H, GRID_W), &device)?;
                hidden_mem = Tensor::from_vec(world.hidden_mem, (1, MEMORY_DIM), &device)?;
                phases = world.phases;
                theta_prev = world.theta_prev;
                theta_prev2 = world.theta_prev2;
                model.restore_runtime_state(&world.model_runtime, &device)?;
                model.set_depth(world.active_depth);
                rad_amp = world.rad_amp.clamp(RAD_AMP_MIN, RAD_AMP_MAX);
                energy_state = world.energy_state.clamp(0.18, 0.96);
                shear_phase = world.shear_phase;
                uncertainty = world.uncertainty;
                potential = world.potential;
                semantic = world.semantic;
                criticality = world.criticality;
                episodic.restore_host(&world.episodic_slots, &device)?;
                novelty_buf = restore_novelty(&world.novelty_spectra, &device)?;
                modal_resonators = world.modal_resonators;
                fractal_fdn_l = world.fdn_l;
                fractal_fdn_r = world.fdn_r;
                spectral_noise = world.noise_bank;
                controller = world.controller;
                adaptive_dynamics = world.adaptive_dynamics;
                motifs = world.motifs;
                motif_diagnostics = world.motif_diagnostics;
                rg_observer = world.rg_observer;
                critical_manifold = world.critical_manifold;
                state_space = world.state_space;
                reactor_planner = world.reactor_planner;
                attractors = world.attractors;
                probe_controller = world.probe_controller;
                confinement_observer = world.confinement_observer;
                recovery_controller = world.recovery_controller;
                probe_controller.enabled = probes_enabled;
                last_reactor_state = world.last_reactor_state;
                pending_reactor_command = world.pending_reactor_command;
                last_observation = world.last_observation;
                if let Some(v) = world.pending_predictor_input {
                    if v.len() == MEMORY_DIM + ACTION_COUNT {
                        pending_predictor_input = Some(Tensor::from_vec(
                            v,
                            (1, MEMORY_DIM + ACTION_COUNT),
                            &device,
                        )?);
                    }
                }
                spectral_mon.restore_history(&world.spectral_history);
                spectral_mon.restore_prev_mags(&world.spectral_prev_mags);
                movement_mon.restore_history(&world.movement_history);
                host_runtime = world.host_runtime;
                loaded_world = true;
                println!("--> Resumed world {} at global step {} (L{:02}, motifs {}, attractors {}, reactor-ready {:.2}, raw/effective model confidence {:.2}/{:.2}).",
                    load_world_path, global_step, model.depth(), motifs.entries.len(), attractors.entries.len(),
                    reactor_planner.ensemble.readiness(), controller.meta.confidence, adaptive_dynamics.effective_model_weight);
            }
            Err(e) => println!(
                "--> World checkpoint could not be loaded: {} — starting a fresh world",
                e
            ),
        }
    }
    if fresh_world {
        println!(
            "==> FRESH WORLD: retaining compatible model weights but resetting organism state."
        );
    }
    println!(
        "--> Observer depth: L{:02} / {} · rad_amp {:.3} · world {}",
        model.depth(),
        MORPH_MAX_BLOCKS,
        rad_amp,
        if loaded_world { "resumed" } else { "new" }
    );
    let random_pool = DeviceRandomPool::new(seed, &device)?;
    println!(
        "--> Device-resident stochastic pool: {} slots ({:.0} MiB stable VRAM)",
        GPU_RANDOM_POOL_SLOTS,
        (GPU_RANDOM_POOL_SLOTS * CA_CHANNELS * GRID_H * GRID_W * 3 * 4) as f64 / (1024.0 * 1024.0)
    );

    let total_chunks = ((SAMPLE_RATE as f32 * sim_duration / CHUNK_SIZE as f32) as usize).max(1);
    // Mobile-safe output path: stream DC-blocked f32 samples to a temporary
    // file, then perform one normalization/transcode pass.  A 16-minute run
    // no longer retains ~350 MB of stereo f32 audio in RAM.
    let raw_audio_path = format!("{}/.titan_audio_f32.tmp", base_dir);
    let raw_audio_file = File::create(&raw_audio_path)?;
    let mut raw_audio_writer = BufWriter::with_capacity(1 << 20, raw_audio_file);
    let mut raw_chunk_bytes = Vec::with_capacity(CHUNK_SIZE * 2 * std::mem::size_of::<f32>());
    let (mut dc_x1_l, mut dc_y1_l, mut dc_x1_r, mut dc_y1_r) = (
        host_runtime.dc_x1_l,
        host_runtime.dc_y1_l,
        host_runtime.dc_x1_r,
        host_runtime.dc_y1_r,
    );
    let dc_pole = 0.998f32;
    let mut raw_peak = 1e-6f32;
    let mut chunk_scores: Vec<f32> = Vec::with_capacity(total_chunks);
    let mut topology_history = Vec::new();
    let mut uncertainty_trace = Vec::new();

    let mut phi = uncertainty.phi;
    let mut window_loss: Option<Tensor> = None;
    let mut steps_in_window = 0;
    // Persisted host-side adaptive state is unpacked into local scalars for
    // the hot loop, then packed again only when checkpointing.
    let mut morph_history = std::mem::take(&mut host_runtime.morph_history);
    let mut morph_baseline = host_runtime.morph_baseline;
    let mut warmup_sum = host_runtime.warmup_sum;
    let mut field_entropy_sum = host_runtime.field_entropy_sum;
    let mut field_entropy_n = host_runtime.field_entropy_n;
    let mut total_complexity = host_runtime.total_complexity;
    let mut boost_state = host_runtime.boost_state;
    let mut prev_loss_vec = host_runtime.prev_loss_vec;
    let mut prev_archetype = std::mem::take(&mut host_runtime.prev_archetype);
    let mut stagnation_ticks = host_runtime.stagnation_ticks;
    let mut last_temp = if loaded_world {
        host_runtime.last_temp
    } else {
        potential.temp
    };
    let mut smoothed_control = host_runtime.smoothed_control;
    let mut sigma = criticality.sigma;
    let chunk_dt = CHUNK_SIZE as f32 / SAMPLE_RATE as f32;

    let timer_start = std::time::Instant::now();
    let mut profiling_lap = std::time::Instant::now();
    let mut completed_chunks = 0usize; // chunks committed to the raw audio stream
    let mut evolved_chunks = 0usize; // includes a NaN-triggered ecological reset

    for step in 0..total_chunks {
        if !keep_running.load(AtomicOrdering::SeqCst) {
            println!(
                "\n--> Stop requested: finalizing {} completed chunks and saving the organism.",
                completed_chunks
            );
            break;
        }
        let absolute_step = global_step + step as u64;
        let aperture = uncertainty.branch_aperture();
        let escape_strength = adaptive_dynamics.escape_strength();
        let subcritical_pressure = ((0.78 - sigma) / (0.78 - 0.30)).clamp(0.0, 1.0);
        // Deep subcriticality gets an immediate, bounded rescue pressure;
        // prolonged low motion still ramps the stronger stateful escape path.
        let recovery_pressure = escape_strength.max(0.55 * subcritical_pressure);
        let recovery_exit_ready = sigma > 0.72
            && adaptive_dynamics.movement_fast > 0.007
            && rg_observer.entropy_rate_ema > 0.015;
        let recovery_latched =
            adaptive_dynamics.low_motion_run > STAGNATION_ESCAPE_AFTER && !recovery_exit_ready;
        let hard_recovery = recovery_pressure >= 0.50 || recovery_latched;
        recovery_controller.update(absolute_step, hard_recovery, confinement_observer.health);
        if recovery_controller.just_entered {
            println!(
                "  ⟿ CONFINEMENT PHASE {} · radial {:.3} tangential {:.3} rotation {:.3}",
                recovery_controller.phase.label(),
                confinement_observer.radial_velocity,
                confinement_observer.tangential_velocity,
                confinement_observer.modal_rotation
            );
        }
        let core_control = CoreControl::for_recovery(recovery_controller.phase, recovery_pressure);
        let curiosity_factor = (stagnation_ticks as f32 / 12.0)
            .min(1.0)
            .max(adaptive_dynamics.stagnation * 0.85);

        // Causally correct self-model: the state/action pair retained from the
        // previous chunk predicts the post-DSP observation that is now known.
        let (self_model_loss, predicted_mean_t, predicted_log_var_t) = if let (
            Some(input),
            Some(actual_obs),
        ) =
            (&pending_predictor_input, &last_observation)
        {
            let (mean, log_var) = monitor_head.forward(input)?;
            let actual = Tensor::from_vec(actual_obs.values.to_vec(), (1, OBS_DIM), &device)?;
            let nll = mean
                .sub(&actual)?
                .sqr()?
                .mul(&log_var.neg()?.exp()?)?
                .add(&log_var)?
                .mean_all()?
                .affine(0.5, 0.0)?;
            (nll, mean, log_var)
        } else {
            (
                Tensor::new(0.0f32, &device)?,
                Tensor::zeros((1, OBS_DIM), DType::F32, &device)?,
                Tensor::zeros((1, OBS_DIM), DType::F32, &device)?,
            )
        };

        // Recursive planner: the Candle self-model supplies a fast neural prior,
        // while a tiny online ensemble plans in the compressed reactor state.
        // Planning is event-triggered and therefore does not tax every chunk.
        let motif_available =
            motifs.has_recallable(absolute_step) || attractors.has_recallable(absolute_step);
        if let Some(trigger) = reactor_planner.should_plan(
            absolute_step,
            last_reactor_state.as_ref(),
            &critical_manifold,
            &adaptive_dynamics,
            &controller.meta,
            controller.action_age,
            planner_config,
        ) {
            let neural_prior = match plan_action_scores(
                &monitor_head,
                &hidden_mem.detach(),
                &device,
                &adaptive_dynamics,
            ) {
                Ok(scores) => scores,
                Err(e) => {
                    println!(
                        "! neural planner prior skipped at step {}: {}",
                        absolute_step, e
                    );
                    controller.cached_model_scores
                }
            };
            if let Some(state) = last_reactor_state.as_ref() {
                reactor_planner.plan(
                    absolute_step,
                    state,
                    neural_prior,
                    &critical_manifold,
                    &adaptive_dynamics,
                    motif_available,
                    planner_config,
                    trigger,
                );
                controller.cached_model_scores = reactor_planner.cached_scores;
            } else {
                controller.cached_model_scores = neural_prior;
                reactor_planner.last_plan_step = absolute_step;
                reactor_planner.last_trigger = trigger.to_string();
            }
        }

        probe_controller.maybe_begin(
            absolute_step,
            last_reactor_state.as_ref(),
            &reactor_planner.ensemble,
            adaptive_dynamics.activity_health,
            escape_strength,
        );
        let forced_recovery_action = hard_recovery.then_some(match recovery_controller.phase {
            RecoveryPhase::EdgeAgitation => ControlAction::Explore,
            RecoveryPhase::RotationCapture => ControlAction::Resonate,
            RecoveryPhase::GuidedPulse | RecoveryPhase::PartialReseed => ControlAction::Turbulence,
            RecoveryPhase::Cooldown | RecoveryPhase::Nominal => ControlAction::Explore,
        });
        let current_action = controller.choose(
            last_temp,
            &mut adaptive_dynamics,
            motif_available,
            forced_recovery_action,
            &mut rng,
        );
        let current_command = if hard_recovery {
            // Do not inherit a near-zero planner intensity for an action the
            // safety controller has explicitly forced.
            ActionCommand::new(
                current_action,
                (0.62 + 0.30 * recovery_pressure).clamp(0.62, 0.92),
            )
        } else {
            reactor_planner.command_for(current_action, &adaptive_dynamics)
        };
        let mut target_control = current_command.control();
        let mut recall_strength = 0.0f32;
        if current_action == ControlAction::Recall {
            let attractor_recall = last_reactor_state
                .as_ref()
                .and_then(|state| attractors.recall(state, absolute_step));
            if let Some((remembered, strength)) = attractor_recall {
                recall_strength = strength;
                target_control =
                    target_control.blend(remembered, (0.30 + 0.58 * strength).clamp(0.0, 0.92));
            } else if let Some((remembered, strength)) = motifs.recall(
                last_observation.as_ref(),
                absolute_step,
                &mut motif_diagnostics,
            ) {
                recall_strength = strength;
                target_control =
                    target_control.blend(remembered, (0.35 + 0.55 * strength).clamp(0.0, 0.90));
            }
        }
        // Subtle system-identification probes are part of the music. They
        // compare the observed response against an ensemble-predicted HOLD
        // trajectory and estimate local susceptibility.
        if let Some(probe_command) = probe_controller.overlay(absolute_step) {
            target_control = target_control.blend(probe_command.control(), 0.22);
        }
        // A prolonged low-information attractor adds a bounded rescue bias
        // even when the learned controller still proposes an ordering action.
        if recovery_pressure > 0.0 {
            let rescue = SynthesisControl::for_action(ControlAction::Turbulence);
            target_control =
                target_control.blend(rescue, (0.12 + 0.62 * recovery_pressure).clamp(0.0, 0.82));
        }
        let control_slew =
            (0.10 + 0.18 * controller.meta.surprise() + 0.14 * recovery_pressure).clamp(0.08, 0.38);
        let current_control = smoothed_control.blend(target_control, control_slew);
        smoothed_control = current_control;
        let current_predictor_input = predictor_input(
            &hidden_mem.detach(),
            current_action,
            current_control,
            &device,
        )?
        .detach();
        let force_probability = (0.18
            + aperture * 0.48
            + 0.12 * (current_control.shear_mult - 1.0).max(0.0)
            + 0.10 * controller.meta.surprise()
            + 0.25 * recovery_pressure
            + 0.12 * subcritical_pressure)
            .clamp(0.05, 0.98);
        let force_macro = rng.gen_range(0.0f32..1.0) < force_probability;

        // Episodic attention readout from the current memory (before this step's GRU).
        let epi_out = episodic.read(&hidden_mem, &device)?;

        let out = model.forward(
            &micro_tape,
            &macro_tape,
            &hidden_mem,
            &epi_out,
            phases,
            theta_prev,
            theta_prev2,
            force_macro,
            energy_state,
            &current_control,
            &core_control,
            &random_pool,
            absolute_step as usize,
        )?;
        let ForwardOut {
            stereo: stereo_chunk,
            next_micro,
            next_macro,
            next_hidden,
            refined_hidden,
            movement_t,
            cur_freq_l,
            cur_freq_r,
            mod_freq_l,
            mod_freq_r,
            pan,
            pair_sums,
            region_activity,
            region_change,
            macro_region_activity,
        } = out;

        let synergy_tensor = calculate_cross_layer_synergy_tensor(&next_micro, &next_macro)?;
        let memory_delta = next_hidden.sub(&hidden_mem)?;
        let tape_delta = next_micro.sub(&micro_tape)?;
        let trans_var = var_all(&memory_delta)?
            .add(&var_all(&tape_delta)?)?
            .affine(1.0, 1e-4)?;
        let cont_entropy_t = trans_var
            .log()
            .map_err(anyhow::Error::msg)?
            .affine(0.5, 0.0)?;
        let empowerment_t = cont_entropy_t
            .affine(1.0, 7.0)?
            .clamp(0.0f32, 5.0f32)?
            .mul(&movement_t.affine(1.0, 1.0)?)?;
        let coarse_micro = decimate2_2d(&next_micro)?;
        let coarse_macro = decimate2_2d(&next_macro)?;
        let coarse2_micro = decimate2_2d(&coarse_micro)?;
        let coarse2_macro = decimate2_2d(&coarse_macro)?;
        let rg_match_l1 = coarse_micro
            .sub(&coarse_macro.detach())?
            .sqr()?
            .mean_all()?;
        let rg_match_l2 = coarse2_micro
            .sub(&coarse2_macro.detach())?
            .sqr()?
            .mean_all()?;
        // Renormalization should preserve meaningful organization, not erase
        // all variation. The floor penalizes a coarse field that becomes too
        // uniform while scale matching keeps micro/macro descriptions coupled.
        let coarse_var = var_all(&coarse_micro)?;
        let rg_activity_floor = coarse_var.affine(-20.0, 1.0)?.relu()?.sqr()?;
        let rg_loss = rg_match_l1
            .add(&rg_match_l2.affine(0.55, 0.0)?)?
            .add(&rg_activity_floor.affine(0.30, 0.0)?)?;

        // --- MIN-OF-K TARGET SELECTION ---
        // K candidate chunks; the coarse mimic picks the NEAREST, so the model
        // matches a mode of the target set instead of the blur of all modes.
        let age_factor = (total_complexity / 500.0).min(0.6);
        let audio_for_loss = stereo_chunk.tanh()?;
        let out_spec_l = spec_proj.log_mag(&audio_for_loss.narrow(0, 0, 1)?)?;
        let out_spec_r = spec_proj.log_mag(&audio_for_loss.narrow(0, 1, 1)?)?;
        let targets = target_loader.sample_chunks(TARGET_K, &mut rng, &device)?;
        let tgt_specs = spec_proj
            .log_mag(&targets.reshape((TARGET_K * 2, CHUNK_SIZE))?)?
            .detach(); // (K*2, bins)
                       // (2, bins), grad-carrying. Score all K references on-device and let
                       // min route the gradient to the closest one. This removes the mid-step
                       // CUDA synchronization and produces the same min-of-K objective.
        let out_spec = Tensor::cat(&[&out_spec_l, &out_spec_r], 0)?;
        let coarse_per_target = tgt_specs
            .reshape((TARGET_K, 2, SPEC_BINS))?
            .broadcast_sub(&out_spec.unsqueeze(0)?)?
            .sqr()?
            .mean(D::Minus1)?
            .mean(D::Minus1)?;
        let out_fine = spec_proj_fine.log_mag(&audio_for_loss.reshape((8, 1024))?)?;
        let tgt_fine = spec_proj_fine
            .log_mag(&targets.reshape((TARGET_K * 8, 1024))?)?
            .detach()
            .reshape((TARGET_K, 8, FINE_SPEC_BINS))?;
        let fine_per_target = tgt_fine
            .broadcast_sub(&out_fine.unsqueeze(0)?)?
            .sqr()?
            .mean(D::Minus1)?
            .mean(D::Minus1)?;
        let mimic_coarse = coarse_per_target.min(0)?;
        let best_k = coarse_per_target.argmin(0)?.reshape((1,))?;
        let mimic_fine = fine_per_target.gather(&best_k, 0)?.reshape(())?;
        let mimic_loss = mimic_coarse.add(&mimic_fine.affine(0.5, 0.0)?)?;

        // --- NOVELTY PRESSURE: the teacher gets bored ---
        // Hinge on distance to the nearest of the system's OWN recent spectra.
        let mono_spec = out_spec_l.add(&out_spec_r)?.affine(0.5, 0.0)?; // (1, bins)
        let (novelty_loss, novelty_dmin_t) = if !novelty_buf.is_empty() {
            let refs: Vec<&Tensor> = novelty_buf.iter().collect();
            let past = Tensor::cat(&refs, 0)?; // (M, bins)
            let d = past.broadcast_sub(&mono_spec)?.sqr()?.mean(D::Minus1)?; // (M,)
            let d_min = d.min(0)?;
            (Some(d_min.affine(-1.0, NOVELTY_MARGIN)?.relu()?), d_min)
        } else {
            (None, Tensor::new(1.0f32, &device)?)
        };

        let current_var = var_all(&audio_for_loss)?;
        let var_loss = current_var.affine(1.0, -0.12)?.sqr()?;
        let rms = audio_for_loss
            .sqr()?
            .mean_all()?
            .affine(1.0, 1e-4)?
            .sqrt()?;
        let saturation_loss = rms.affine(1.0, -0.28)?.sqr()?;
        let movement_loss = movement_t.neg()?.exp()?;
        // Differentiable anti-weld floors. Targets adapt to the organism's
        // own decaying peaks, with small absolute minima so a collapsed world
        // cannot redefine zero movement as its healthy baseline.
        let movement_floor = (0.003 + 0.28 * adaptive_dynamics.movement_peak).clamp(0.003, 0.012);
        let region_floor = (0.002 + 0.24 * adaptive_dynamics.region_peak).clamp(0.002, 0.010);
        let movement_floor_loss = movement_t
            .affine(-(1.0 / movement_floor) as f64, 1.0)?
            .relu()?
            .sqr()?;
        let region_change_mean_t = region_change.mean_all()?;
        let regional_floor_loss = region_change_mean_t
            .affine(-(1.0 / region_floor) as f64, 1.0)?
            .relu()?
            .sqr()?;
        let diff = audio_for_loss
            .narrow(1, 1, CHUNK_SIZE - 1)?
            .sub(&audio_for_loss.narrow(1, 0, CHUNK_SIZE - 1)?)?;
        let roughness_loss = diff.sqr()?.mean_all()?;
        let reg_loss = stereo_chunk.sqr()?.mean_all()?;
        let empowerment_loss = empowerment_t.affine(1.0, -2.5)?.sqr()?;
        let synergy_loss = synergy_tensor
            .affine(1.0, -(SYNERGY_TARGET as f64))?
            .sqr()?;

        let arb_features = Tensor::new(
            &[
                prev_loss_vec[0],
                prev_loss_vec[1],
                prev_loss_vec[2],
                prev_loss_vec[3],
                uncertainty.spectral,
                uncertainty.movement,
                uncertainty.mimic,
                uncertainty.compositional,
                aperture,
                step as f32 / total_chunks as f32,
                phi,
                last_temp,
                sigma - 1.0,
                energy_state,
            ],
            &device,
        )?
        .unsqueeze(0)?;
        // Sign-fixed entropy handling; softmax temperature rides the system heat.
        let (w_graph, neg_entropy) =
            arbiter.forward(&arb_features, 1.0 + ARB_TAU_MAX * last_temp)?;

        // --- BATCHED METRICS READBACK: one sync for the whole control plane ---
        let abs_max_t = audio_for_loss.abs()?.flatten_all()?.max(0)?;
        let micro_feats_flat = next_micro
            .mean(D::Minus1)?
            .mean(D::Minus1)?
            .reshape((CA_CHANNELS,))?;
        let metrics = Tensor::cat(
            &[
                &movement_t.reshape((1,))?,
                &mimic_coarse.reshape((1,))?,
                &rms.reshape((1,))?,
                &abs_max_t.reshape((1,))?,
                &rg_loss.reshape((1,))?,
                &empowerment_loss.reshape((1,))?,
                &roughness_loss.reshape((1,))?,
                &current_var.reshape((1,))?,
                &movement_loss.reshape((1,))?,
                &synergy_tensor.reshape((1,))?,
                &next_micro.abs()?.mean_all()?.reshape((1,))?,
                &next_macro.abs()?.mean_all()?.reshape((1,))?,
                &empowerment_t.reshape((1,))?,
                &self_model_loss.reshape((1,))?,
                &cur_freq_l,
                &cur_freq_r,
                &mod_freq_l,
                &mod_freq_r,
                &pan,
                &pair_sums,
                &micro_feats_flat,
                &w_graph.reshape((7,))?,
                &predicted_mean_t.reshape((OBS_DIM,))?,
                &predicted_log_var_t.reshape((OBS_DIM,))?,
                &region_activity,
                &region_change,
                &macro_region_activity,
                &novelty_dmin_t.reshape((1,))?,
            ],
            0,
        )?
        .to_vec1::<f32>()?;
        let movement = metrics[0];
        let mimic_drift = metrics[1];
        let rms_val = metrics[2];
        let abs_max = metrics[3];
        let rg_v = metrics[4];
        let empowerment_loss_val = metrics[5];
        let roughness_loss_val = metrics[6];
        let current_var_val = metrics[7];
        let movement_loss_val = metrics[8];
        let synergy_val = metrics[9];
        let micro_abs = metrics[10];
        let macro_abs = metrics[11];
        let empowerment_val = metrics[12];
        let self_model_loss_val = metrics[13];
        let (f_l, f_r, mf_l, mf_r) = (metrics[14], metrics[15], metrics[16], metrics[17]);
        model.last_pan = metrics[18];
        theta_prev2 = theta_prev;
        theta_prev = metrics[20].atan2(metrics[19] + 1e-6);
        let field_start = 21;
        let field_summary = &metrics[field_start..field_start + CA_CHANNELS];
        let lw_start = field_start + CA_CHANNELS;
        let lw_raw = &metrics[lw_start..lw_start + 7];
        let pred_mean_start = lw_start + 7;
        let pred_log_start = pred_mean_start + OBS_DIM;
        let region_start = pred_log_start + OBS_DIM;
        let predicted_mean_host = &metrics[pred_mean_start..pred_mean_start + OBS_DIM];
        let predicted_log_host = &metrics[pred_log_start..pred_log_start + OBS_DIM];
        let region_activity_host: [f32; REGION_COUNT] = metrics
            [region_start..region_start + REGION_COUNT]
            .try_into()
            .unwrap();
        let region_change_host: [f32; REGION_COUNT] = metrics
            [region_start + REGION_COUNT..region_start + REGION_COUNT * 2]
            .try_into()
            .unwrap();
        let macro_region_activity_host: [f32; REGION_COUNT] = metrics
            [region_start + REGION_COUNT * 2..region_start + REGION_COUNT * 3]
            .try_into()
            .unwrap();
        let novelty_dmin_val = metrics[region_start + REGION_COUNT * 3];
        if let (Some(actual), Some(_)) = (&last_observation, &pending_predictor_input) {
            controller
                .meta
                .update(actual, predicted_mean_host, predicted_log_host);
        }

        // NaN bio-reset rides the same readback — no dedicated check sync.
        if !movement.is_finite() || !micro_abs.is_finite() {
            println!("! BIO-RESET: Tape corruption detected (NaN). Re-seeding primordial soup.");
            micro_tape = randn_t(&mut rng, &[1, CA_CHANNELS, GRID_H, GRID_W], 1.0, &device)?;
            macro_tape = randn_t(&mut rng, &[1, CA_CHANNELS, GRID_H, GRID_W], 1.0, &device)?;
            hidden_mem = Tensor::zeros((1, MEMORY_DIM), DType::F32, &device)?;
            window_loss = None;
            steps_in_window = 0;
            adaptive_dynamics.low_motion_run = 0;
            adaptive_dynamics.stagnation = 0.0;
            adaptive_dynamics.escape_cooldown = 0;
            controller.action_age = 0;
            evolved_chunks = step + 1;
            continue;
        }
        let mimic_drift_n = mimic_drift / (1.0 + mimic_drift);
        total_complexity += movement;

        // Host mirrors for phase advance + metabolism.
        model.current_freq_l = f_l;
        model.current_freq_r = f_r;
        phases[0] = (phases[0] + TWO_PI * f_l * chunk_dt).rem_euclid(TWO_PI);
        phases[1] = (phases[1] + TWO_PI * f_r * chunk_dt).rem_euclid(TWO_PI);
        phases[2] = (phases[2] + TWO_PI * mf_l * chunk_dt).rem_euclid(TWO_PI);
        phases[3] = (phases[3] + TWO_PI * mf_r * chunk_dt).rem_euclid(TWO_PI);

        // Mean-centered multi-lag criticality estimate.  Unlike the old
        // through-origin lag-1 regression, this does not mistake a non-zero
        // movement baseline for branching persistence.
        sigma = criticality.update(movement);

        let boost_target = if abs_max < 0.25 {
            (0.25 / (abs_max + 1e-6)).clamp(1.0, 4.0)
        } else {
            1.0
        };
        boost_state = boost_state * 0.9 + boost_target * 0.1;
        let audio_normalized = audio_for_loss.affine(boost_state as f64, 0.0)?;

        let m_sig = movement_mon.analyze(movement)?;

        // Archetype / stagnation from the channel summary (already in the readback).
        let field01: Vec<f32> = field_summary.iter().map(|&x| (x + 1.0) * 0.5).collect();
        let (arch_summary, field_entropy, dom_arch) = SemanticField::archetype_field(&field01);
        semantic.record(mimic_drift_n, dom_arch);
        let trend = semantic.trend();
        field_entropy_sum += field_entropy as f64;
        field_entropy_n += 1;
        let current_arch = semantic.dominant_archetype();
        if prev_archetype == current_arch {
            stagnation_ticks += 1;
        } else {
            stagnation_ticks = 0;
            prev_archetype = current_arch.to_string();
        }

        // Metabolism now has a genuine interior fixed point. In v5 the flat
        // recharge term overwhelmed cost and pinned energy at 0.99.
        let metabolic_cost = 0.0020
            + rms_val.clamp(0.0, 1.0) * 0.0060
            + movement.clamp(0.0, 0.20) * 0.045
            + (current_control.kick_mult - 1.0).max(0.0) * 0.0015;
        energy_state = (energy_state - metabolic_cost).max(0.18);
        let recharge_capacity = (1.0 - energy_state).max(0.0);
        let energy_recharge = recharge_capacity * (1.0 - rms_val).clamp(0.0, 1.0) * 0.012;
        energy_state += energy_recharge;
        energy_state += ENERGY_HOMEO_RATE * (POT_ENERGY_SET - energy_state);
        energy_state = energy_state.clamp(0.18, 0.96);

        // Morphic growth/pruning (unchanged policy).
        let mut morph_event = None;
        if step < MORPH_WARMUP {
            warmup_sum += mimic_drift_n;
        } else {
            if morph_baseline.is_none() {
                let b = (warmup_sum / MORPH_WARMUP as f32).max(1e-4);
                morph_baseline = Some(b);
                println!(
                    "--> Morph baseline calibrated: mimic≈{:.3}  (grow>{:.3}, prune<{:.3})",
                    b,
                    b * MORPH_GROWTH_REL,
                    b * MORPH_PRUNE_REL
                );
            }
            morph_history.push(mimic_drift_n);
            let patience = MORPH_PATIENCE_BASE + model.depth() * 2;
            if morph_history.len() >= patience {
                let avg = morph_history.iter().sum::<f32>() / morph_history.len() as f32;
                morph_history.clear();
                let base = morph_baseline.unwrap();
                if avg > base * MORPH_GROWTH_REL {
                    if model.grow() {
                        rad_amp = (rad_amp * RAD_COOL).max(RAD_AMP_MIN);
                        morph_event = Some("NEUROGENESIS");
                    }
                } else if avg < base * MORPH_PRUNE_REL {
                    if model.prune() {
                        rad_amp = (rad_amp * RAD_HEAT).min(RAD_AMP_MAX);
                        morph_event = Some("PRUNING");
                    }
                }
            }
        }
        if let Some(ev) = morph_event {
            println!(
                "  ◄ {} ►  Depth L{:02} | Rad {:.2}",
                ev,
                model.depth(),
                rad_amp
            );
        }
        // Capacity growth is a late recovery tool, not a periodic substitute
        // for control authority. Use it once per recovery episode, after edge
        // agitation and rotation capture have both failed.
        if morph_event.is_none()
            && recovery_pressure > 0.62
            && recovery_controller.phase == RecoveryPhase::GuidedPulse
            && recovery_controller.elapsed(absolute_step) >= 96
            && !recovery_controller.growth_used
            && model.grow()
        {
            recovery_controller.growth_used = true;
            println!(
                "  ◄ RECOVERY NEUROGENESIS ►  Depth L{:02} | Rad {:.2}",
                model.depth(),
                rad_amp
            );
        }

        // --- POTENTIAL CONTROLLER: the entire crash cart, one call ---
        let recursive_curiosity = curiosity_factor
            .max(controller.meta.surprise() * 0.80)
            .max(adaptive_dynamics.stagnation * 0.90);
        let pot = potential.update(
            micro_abs,
            macro_abs,
            synergy_val,
            movement,
            energy_state,
            sigma,
            recursive_curiosity,
            adaptive_dynamics.stagnation,
        );
        // Radiation amplitude becomes a slow ecological state instead of
        // remaining at its initial value until a rare morphic event.
        let rad_target = (0.38
            + 0.36 * recovery_pressure
            + 0.22 * subcritical_pressure
            + 0.18 * controller.meta.surprise()
            + 0.10 * pot.temp)
            .clamp(RAD_AMP_MIN, RAD_AMP_MAX);
        rad_amp += 0.012 * (rad_target - rad_amp);
        last_temp = pot.temp;
        if pot.annealed {
            println!(
                "  ⟿ ANNEALED · V={:.3} T={:.2} · heat spent, plateau escaped",
                pot.v, pot.temp
            );
        }

        // Arbiter mixing weights (host values arrived in the batched readback).
        let lw: Vec<f32> = lw_raw.iter().map(|p| p * 7.0).collect();
        let cur_loss_vec = [
            current_var_val,
            mimic_drift_n,
            movement_loss_val,
            roughness_loss_val,
            rg_v,
            self_model_loss_val,
            empowerment_loss_val,
        ];
        let improvement: Vec<f32> = (0..7)
            .map(|i| (prev_loss_vec[i] - cur_loss_vec[i]).clamp(-1.0, 1.0))
            .collect();
        prev_loss_vec = cur_loss_vec;
        let improvement_t = Tensor::new(improvement, &device)?;
        let arb_progress_loss = w_graph
            .reshape((7,))?
            .mul(&improvement_t)?
            .sum_all()?
            .affine(-0.5, 0.0)?;

        // --- TOTAL LOSS ---
        let mut total_loss = mimic_loss.affine(
            (lw[1] * (1.0 - RESONANT_AUTONOMY) * (1.0 - age_factor)) as f64,
            0.0,
        )?;
        total_loss = total_loss.add(&var_loss.affine((lw[0] * 2.5) as f64, 0.0)?)?;
        total_loss = total_loss.add(&saturation_loss.affine(2.0, 0.0)?)?;
        total_loss =
            total_loss.add(&movement_loss.affine((lw[2] * RESONANT_AUTONOMY) as f64, 0.0)?)?;
        let anti_weld_weight = 0.06 + 0.20 * adaptive_dynamics.stagnation;
        let regional_weld_weight = 0.04 + 0.12 * adaptive_dynamics.stagnation;
        total_loss = total_loss.add(&movement_floor_loss.affine(anti_weld_weight as f64, 0.0)?)?;
        total_loss =
            total_loss.add(&regional_floor_loss.affine(regional_weld_weight as f64, 0.0)?)?;
        total_loss = total_loss.add(&roughness_loss.affine(lw[3] as f64, 0.0)?)?;
        total_loss = total_loss.add(&reg_loss.affine(0.01, 0.0)?)?;
        total_loss = total_loss.add(&rg_loss.affine((0.15 * lw[4].max(0.2)) as f64, 0.0)?)?;
        total_loss =
            total_loss.add(&self_model_loss.affine((0.30 * lw[5].max(0.2)) as f64, 0.0)?)?;
        total_loss = total_loss.add(&empowerment_loss.affine(lw[6] as f64, 0.0)?)?;
        // Coupling band, released by the controller's over-coupling/heat signal.
        let synergy_w = (SYNERGY_BAND_W * (1.0 - 0.7 * pot.couple_release)).max(0.1f32);
        total_loss = total_loss.add(&synergy_loss.affine(synergy_w as f64, 0.0)?)?;
        // Entropy BONUS (v3 minimized it — see AudioArbiter): adding neg_entropy
        // with a positive coefficient maximizes mixing entropy.
        total_loss = total_loss.add(&neg_entropy.affine(0.05, 0.0)?)?;
        total_loss = total_loss.add(&arb_progress_loss)?;
        if let Some(nl) = novelty_loss {
            total_loss = total_loss.add(&nl.affine(NOVELTY_W, 0.0)?)?;
        }

        window_loss = Some(match window_loss.take() {
            None => total_loss,
            Some(w) => w.add(&total_loss)?,
        });
        steps_in_window += 1;

        // --- CRITICALITY-DRIVEN PLASTICITY (replaces the movement-threshold Choptuik) ---
        let crit_gain =
            ((CRITICAL_D0 / ((sigma - 1.0).abs() + 1e-3)).powf(CHOPTUIK_EXPONENT)).clamp(0.3, 3.0);
        let phi_gate = 1.0 / (1.0 + phi);
        let curiosity_lr_gain = 1.0 + (curiosity_factor * 1.5) as f64;
        let latest_lr_gain = crit_gain as f64 * pot.lr_heat * phi_gate as f64 * curiosity_lr_gain;

        if steps_in_window >= bptt_window || step == total_chunks - 1 {
            if let Some(w) = window_loss.take() {
                let scaled = w.affine(1.0 / steps_in_window as f64, 0.0)?;
                let bounded_loss = scaled.clamp(0.0, 10.0)?;
                if let Ok(loss_val) = bounded_loss.to_scalar::<f32>() {
                    if loss_val.is_finite() && latest_lr_gain.is_finite() {
                        // Global grad norm -> LR scale. NOTE: for AdamW this is not
                        // bitwise-identical to true clipping (the moments still see
                        // raw grads), but as a blow-up guardrail it is equivalent in
                        // effect and replaces v3's no-op ±100 weight clamp.
                        match bounded_loss.backward() {
                            Ok(grads) => {
                                let mut sq = Tensor::zeros((), DType::F32, &device)?;
                                for var in varmap.all_vars() {
                                    if let Some(g) = grads.get(var.as_tensor()) {
                                        sq = sq.add(&g.sqr()?.sum_all()?)?;
                                    }
                                }
                                let gnorm = sq.to_scalar::<f32>().unwrap_or(f32::INFINITY).sqrt();
                                if gnorm.is_finite() {
                                    let clip_scale =
                                        (GRAD_NORM_MAX / gnorm.max(1e-6)).min(1.0) as f64;
                                    optimizer
                                        .set_learning_rate(target_lr * latest_lr_gain * clip_scale);
                                    let _ = optimizer.step(&grads);
                                } else {
                                    println!("! WARNING: non-finite grad norm — skipping window.");
                                }
                            }
                            Err(e) => {
                                println!("! WARNING: backward failed: {} — skipping window.", e)
                            }
                        }
                    } else {
                        println!("! WARNING: Non-finite loss step detected. Dropping BPTT window.");
                    }
                }
            }
            steps_in_window = 0;
            micro_tape = next_micro.detach();
            macro_tape = next_macro.detach();
            hidden_mem = next_hidden.detach();
            let rad_probability = ((RADIATE_PROB
                + curiosity_factor * 0.12
                + (current_control.kick_mult - 1.0).max(0.0) * 0.08
                + controller.meta.surprise() * 0.06
                + recovery_pressure * 0.36
                + subcritical_pressure * 0.16)
                * core_control.stochastic_heat)
                .clamp(0.0, 0.82);
            let escape_pulse = recovery_controller.phase == RecoveryPhase::EdgeAgitation
                && recovery_pressure > 0.60
                && absolute_step % 64 == 0;
            if escape_pulse || rng.gen::<f32>() < rad_probability {
                micro_tape = levy_radiate(
                    &micro_tape,
                    rad_amp
                        * (1.0
                            + curiosity_factor * 0.4
                            + controller.meta.surprise() * 0.25
                            + recovery_pressure * 0.50
                            + subcritical_pressure * 0.25),
                    &random_pool,
                    absolute_step as usize + 31,
                )?;
            }
        } else {
            micro_tape = next_micro;
            macro_tape = next_macro;
            hidden_mem = next_hidden;
        }
        if recovery_controller.phase == RecoveryPhase::PartialReseed
            && recovery_controller.just_entered
        {
            // Sparse Cauchy reseed affects only a small mask and preserves the
            // macro field, recurrent memory, motifs, and learned weights.
            micro_tape =
                levy_radiate(&micro_tape, 0.42, &random_pool, absolute_step as usize + 79)?;
            println!("  ⟿ PARTIAL CORE RESEED · coarse world and memory preserved");
        }

        // --- LANGEVIN STEP: drift (-grad V gains) + temperature noise ---
        shear_phase += SHEAR_PHASE_VEL * (0.82 + 0.28 * current_control.shear_mult);
        let controlled_shear = (pot.shear_amp * current_control.shear_mult).clamp(0.0, 0.75);
        let shear = shear_gen.generate(controlled_shear, shear_phase)?;
        macro_tape = macro_tape
            .add(&shear)?
            .tanh()?
            .affine(pot.macro_gain as f64, 0.0)?;
        if core_control.coherent_pulse > 1e-4 {
            // Reuse the smooth multiscale shear as a correlated micro-field
            // pulse. Unlike white kicks, neighboring cells receive a coherent
            // direction that the rotation phase can capture.
            micro_tape = micro_tape
                .add(&shear.affine(core_control.coherent_pulse as f64, 0.0)?)?
                .clamp(-1.0f32, 1.0f32)?;
        }
        let controlled_kick =
            (pot.micro_kick * current_control.kick_mult * core_control.stochastic_heat)
                .clamp(0.0, 0.27);
        if controlled_kick > 1e-3 {
            let kick = random_pool
                .normal(absolute_step as usize + 47)?
                .affine(controlled_kick as f64, 0.0)?;
            micro_tape = micro_tape.add(&kick)?;
        }
        micro_tape = micro_tape
            .affine(pot.micro_gain as f64, 0.0)?
            .clamp(-1.0f32, 1.0f32)?;

        // Episodic snapshot cadence.
        if absolute_step % EPI_SNAP_EVERY as u64 == 0 && absolute_step > 0 {
            episodic.snapshot(&refined_hidden);
        }
        // Novelty buffer cadence.
        if absolute_step % NOVELTY_EVERY as u64 == 0 {
            if novelty_buf.len() >= NOVELTY_SLOTS {
                novelty_buf.pop_front();
            }
            novelty_buf.push_back(mono_spec.detach());
        }

        // --- POST-DSP RECURSIVE AUDIO PATH ---
        // Every controller reward and self-observation describes the final
        // pre-master signal, including noise, resonators, FDN, saturation and
        // DC blocking. End-of-run global gain and i16 quantization remain a
        // transparent mastering step outside the recursive loop.
        let audio_normalized_vec = audio_normalized.to_vec2::<f32>()?;
        let mut audio_l = audio_normalized_vec[0].clone();
        let mut audio_r = audio_normalized_vec[1].clone();
        spectral_noise.process(
            &mut audio_l,
            &mut audio_r,
            &region_activity_host,
            &current_control,
            &mut rng,
        );
        modal_resonators.process(
            &mut audio_l,
            &mut audio_r,
            &region_activity_host,
            &current_control,
            controller.meta.surprise(),
        );
        let echo_aperture =
            (aperture.min(0.72) + current_control.echo_delta + 0.05 * current_control.recall_mix)
                .clamp(0.02, 0.84);
        let fdn_damping =
            (0.28 + 0.42 * (1.0 - uncertainty.flatness) + 0.12 * current_control.inharmonicity)
                .clamp(0.12, 0.88);
        rayon::join(
            || fractal_fdn_l.process(&mut audio_l, echo_aperture, fdn_damping),
            || fractal_fdn_r.process(&mut audio_r, echo_aperture, fdn_damping),
        );
        let finish_channel = |samples: &mut [f32], mut x1: f32, mut y1: f32| {
            let mut peak = 0.0f32;
            for sample in samples.iter_mut() {
                *sample = (*sample * 0.92).tanh();
                // Stateful DC blocker is part of the observed signal path.
                let x = *sample;
                let y = x - x1 + dc_pole * y1;
                x1 = x;
                y1 = y;
                *sample = y;
                peak = peak.max(y.abs());
            }
            (x1, y1, peak)
        };
        let (left_state, right_state) = rayon::join(
            || finish_channel(&mut audio_l, dc_x1_l, dc_y1_l),
            || finish_channel(&mut audio_r, dc_x1_r, dc_y1_r),
        );
        (dc_x1_l, dc_y1_l) = (left_state.0, left_state.1);
        (dc_x1_r, dc_y1_r) = (right_state.0, right_state.1);
        raw_peak = raw_peak.max(left_state.2).max(right_state.2);

        let post = spectral_mon.analyze(
            &audio_l,
            &audio_r,
            movement,
            synergy_val,
            field_entropy,
            sigma,
        );
        let s_sig = &post.json;
        uncertainty.update(
            s_sig,
            &m_sig,
            mimic_drift_n,
            synergy_val,
            empowerment_val,
            &controller.meta,
        );
        phi = uncertainty.phi;

        let region_change_mean = region_change_host.iter().sum::<f32>() / REGION_COUNT as f32;
        let observation_delta = last_observation
            .as_ref()
            .map(|prev| post.observation.distance(prev))
            .unwrap_or(0.08);
        let structured_complexity = post.observation.structured_complexity();
        adaptive_dynamics.observe(
            movement,
            region_change_mean,
            observation_delta,
            structured_complexity,
            sigma,
            controller.meta.confidence,
        );

        // Recursive renormalization observer: 4x4 regional fields are pooled
        // into 2x2 and global descriptions, then compared across space, time
        // and scale. The critical target itself drifts slowly, preventing a
        // single scalar setpoint from becoming the next frozen attractor.
        rg_observer.update(&region_activity_host, &region_change_host);
        let planner_horizon = if reactor_planner.ensemble.readiness() > 0.05 {
            reactor_planner.last_prediction_horizon
        } else {
            0.45
        };
        let lyapunov_proxy = if reactor_planner.ensemble.readiness() > 0.05 {
            reactor_planner.last_lyapunov
        } else {
            0.50
        };
        critical_manifold.update(
            sigma,
            &criticality,
            &rg_observer,
            &adaptive_dynamics,
            observation_delta,
            planner_horizon,
            lyapunov_proxy,
            probe_controller.susceptibility_ema,
            state_space.novel_ema,
        );

        let old_motif_recurrence = motifs.recurrence(&post.observation, absolute_step);
        let provisional_state = ReactorState::assemble(
            &post.observation,
            &adaptive_dynamics,
            &critical_manifold,
            &rg_observer,
            energy_state,
            state_space.novel_ema,
            state_space.occupancy_entropy,
            old_motif_recurrence,
            reactor_planner.last_option_value,
        );
        let attractor_recurrence = attractors.recurrence(&provisional_state, absolute_step);
        let recurrence = old_motif_recurrence.max(attractor_recurrence);
        let mut reactor_state = ReactorState::assemble(
            &post.observation,
            &adaptive_dynamics,
            &critical_manifold,
            &rg_observer,
            energy_state,
            state_space.novel_ema,
            state_space.occupancy_entropy,
            recurrence,
            reactor_planner.last_option_value,
        );
        state_space.update(&reactor_state);
        reactor_state.values[44] = state_space.novel_ema;
        reactor_state.values[45] = state_space.occupancy_entropy;
        let viable_complexity = reactor_state.viable_complexity(&critical_manifold);
        let learn_healthy_shell = !hard_recovery
            && adaptive_dynamics.activity_health > 0.62
            && adaptive_dynamics.movement_fast > 0.006
            && rg_observer.entropy_rate_ema > 0.012
            && critical_manifold.health > 0.42;
        confinement_observer.update(
            &reactor_state,
            &region_activity_host,
            &macro_region_activity_host,
            learn_healthy_shell,
        );

        let reward_recurrence = recurrence * (1.0 - 0.90 * recovery_pressure);
        let recall_reward = 0.08 * recall_strength * (1.0 - recovery_pressure);
        let recovery_action_penalty = if matches!(
            current_action,
            ControlAction::Recall | ControlAction::Contract | ControlAction::Crystallize
        ) {
            0.18 * recovery_pressure
        } else {
            0.0
        };
        let reward = (post.observation.reward_against(
            last_observation.as_ref(),
            reward_recurrence,
            &adaptive_dynamics,
            controller.meta.confidence,
            controller.action_age,
        ) + recall_reward
            + 0.16 * (viable_complexity - 0.50)
            + 0.08 * (critical_manifold.health - 0.50)
            + 0.06 * (state_space.transition_diversity - 0.35)
            - 0.10 * reactor_state.collapse_risk()
            - recovery_action_penalty)
            .clamp(-1.0, 1.0);
        if hard_recovery {
            controller.bandit.apply_recovery_tax(recovery_pressure);
        } else {
            controller.bandit.update(current_action, reward);
        }
        adaptive_dynamics.observe_reward(reward);

        // Each chunk is a free transition-training example. Tiny bootstrap
        // models learn in short bursts and remain orders of magnitude cheaper
        // than a full neural-CA rollout.
        if let Some(previous) = last_reactor_state.as_ref() {
            attractors.update_exit(
                previous,
                current_command.action,
                reward,
                reactor_planner.last_option_value,
            );
            reactor_planner.ensemble.observe(
                previous.clone(),
                current_command,
                reactor_state.clone(),
                reward,
            );
        }
        reactor_planner
            .ensemble
            .train_if_due(absolute_step, planner_config);
        probe_controller.observe(absolute_step, &reactor_state);

        if absolute_step % MOTIF_EVERY as u64 == 0 {
            motifs.maybe_store(
                &post.observation,
                current_control,
                absolute_step,
                &adaptive_dynamics,
                &mut motif_diagnostics,
            );
            attractors.maybe_store(
                &reactor_state,
                current_control,
                current_action,
                absolute_step,
                viable_complexity,
                reactor_planner.last_option_value,
            );
        }
        last_observation = Some(post.observation.clone());
        pending_predictor_input = Some(current_predictor_input);
        last_reactor_state = Some(reactor_state.clone());
        pending_reactor_command = Some(current_command);

        raw_chunk_bytes.clear();
        for (&left, &right) in audio_l.iter().zip(audio_r.iter()) {
            raw_chunk_bytes.extend_from_slice(&left.to_le_bytes());
            raw_chunk_bytes.extend_from_slice(&right.to_le_bytes());
        }
        raw_audio_writer.write_all(&raw_chunk_bytes)?;
        chunk_scores.push(
            field_entropy * (0.20 + 0.35 * adaptive_dynamics.activity_health)
                + structured_complexity * 0.45
                + viable_complexity * 0.55
                + critical_manifold.health * 0.20
                - adaptive_dynamics.stagnation * 0.25,
        );
        completed_chunks += 1;
        evolved_chunks = step + 1;

        if step % 10 == 0 {
            let topology_state = macro_tape
                .mean(1)?
                .reshape((GRID_H * GRID_W,))?
                .to_vec1::<f32>()?;
            topology_history.push(topology_state);
            let mut record = serde_json::Map::new();
            json_put(&mut record, "step", absolute_step);
            json_put(&mut record, "spectral", uncertainty.spectral);
            json_put(&mut record, "movement", uncertainty.movement);
            json_put(&mut record, "compositional", uncertainty.compositional);
            json_put(&mut record, "aperture", aperture);
            json_put(&mut record, "phi", phi);
            json_put(&mut record, "synergy", synergy_val);
            json_put(&mut record, "empowerment", empowerment_val);
            json_put(&mut record, "V", pot.v);
            json_put(&mut record, "temp", pot.temp);
            json_put(&mut record, "micro_gain", pot.micro_gain);
            json_put(&mut record, "macro_gain", pot.macro_gain);
            json_put(&mut record, "micro_amp", micro_abs);
            json_put(&mut record, "macro_amp", macro_abs);
            json_put(&mut record, "sigma", sigma);
            json_put(
                &mut record,
                "criticality_confidence",
                criticality.confidence,
            );
            json_put(&mut record, "crit_gain", crit_gain);
            json_put(&mut record, "flatness", uncertainty.flatness);
            json_put(&mut record, "pe1", s_sig["pe1"].as_f64().unwrap_or(0.0));
            json_put(&mut record, "pe4", s_sig["pe4"].as_f64().unwrap_or(0.0));
            json_put(&mut record, "pe16", s_sig["pe16"].as_f64().unwrap_or(0.0));
            json_put(
                &mut record,
                "pi_proxy",
                s_sig["pi_proxy"].as_f64().unwrap_or(0.0),
            );
            json_put(
                &mut record,
                "brightness",
                s_sig["brightness"].as_f64().unwrap_or(0.0),
            );
            json_put(&mut record, "flux", s_sig["flux"].as_f64().unwrap_or(0.0));
            json_put(&mut record, "width", s_sig["width"].as_f64().unwrap_or(0.0));
            json_put(&mut record, "structured_complexity", structured_complexity);
            json_put(&mut record, "viable_complexity", viable_complexity);
            json_put(&mut record, "novelty_dmin", novelty_dmin_val);
            json_put(&mut record, "action", current_action.label());
            json_put(&mut record, "action_intensity", current_command.intensity);
            json_put(
                &mut record,
                "planner_proposal",
                controller.model_proposal().label(),
            );
            json_put(
                &mut record,
                "bandit_proposal",
                controller.bandit_proposal().label(),
            );
            json_put(&mut record, "reward", reward);
            json_put(&mut record, "recurrence", recurrence);
            json_put(&mut record, "model_confidence", controller.meta.confidence);
            json_put(
                &mut record,
                "effective_model_weight",
                adaptive_dynamics.effective_model_weight,
            );
            json_put(&mut record, "prediction_error", controller.meta.error_ema);
            json_put(
                &mut record,
                "calibration_error",
                controller.meta.calibration_error,
            );
            json_put(&mut record, "region_change", region_change_mean);
            json_put(&mut record, "observation_delta", observation_delta);
            json_put(
                &mut record,
                "activity_health",
                adaptive_dynamics.activity_health,
            );
            json_put(&mut record, "stagnation", adaptive_dynamics.stagnation);
            json_put(
                &mut record,
                "low_motion_run",
                adaptive_dynamics.low_motion_run,
            );
            json_put(
                &mut record,
                "escape_strength",
                adaptive_dynamics.escape_strength(),
            );
            json_put(&mut record, "subcritical_pressure", subcritical_pressure);
            json_put(&mut record, "recovery_pressure", recovery_pressure);
            json_put(&mut record, "hard_recovery", hard_recovery);
            json_put(
                &mut record,
                "recovery_phase",
                recovery_controller.phase.label(),
            );
            json_put(
                &mut record,
                "confinement_health",
                confinement_observer.health,
            );
            json_put(
                &mut record,
                "confinement_radius",
                confinement_observer.radius,
            );
            json_put(
                &mut record,
                "radial_velocity",
                confinement_observer.radial_velocity,
            );
            json_put(
                &mut record,
                "tangential_velocity",
                confinement_observer.tangential_velocity,
            );
            json_put(
                &mut record,
                "modal_rotation",
                confinement_observer.modal_rotation,
            );
            json_put(
                &mut record,
                "micro_macro_lock",
                confinement_observer.micro_macro_lock,
            );
            json_put(&mut record, "reward_mean", adaptive_dynamics.reward_mean);
            json_put(&mut record, "reward_std", adaptive_dynamics.reward_std());
            json_put(&mut record, "motifs", motifs.entries.len());
            json_put(
                &mut record,
                "motif_candidates",
                motif_diagnostics.candidates,
            );
            json_put(
                &mut record,
                "motif_stored_total",
                motif_diagnostics.stored_total,
            );
            json_put(
                &mut record,
                "motif_rejected_quality",
                motif_diagnostics.rejected_quality,
            );
            json_put(
                &mut record,
                "motif_rejected_similarity",
                motif_diagnostics.rejected_similarity,
            );
            json_put(
                &mut record,
                "motif_last_quality",
                motif_diagnostics.last_quality,
            );
            json_put(
                &mut record,
                "motif_last_distance",
                motif_diagnostics.last_nearest_distance,
            );
            json_put(&mut record, "rg_entropy_rate", rg_observer.entropy_rate_ema);
            json_put(
                &mut record,
                "rg_scale_invariance",
                rg_observer.scale_invariance_ema,
            );
            json_put(&mut record, "rg_disagreement", rg_observer.disagreement_ema);
            json_put(
                &mut record,
                "rg_active_scales",
                rg_observer.active_scales_ema,
            );
            json_put(
                &mut record,
                "critical_manifold_health",
                critical_manifold.health,
            );
            json_put(&mut record, "order_risk", critical_manifold.order_risk);
            json_put(&mut record, "chaos_risk", critical_manifold.chaos_risk);
            json_put(
                &mut record,
                "susceptibility",
                critical_manifold.susceptibility,
            );
            json_put(
                &mut record,
                "prediction_horizon",
                critical_manifold.prediction_horizon,
            );
            json_put(
                &mut record,
                "lyapunov_proxy",
                critical_manifold.lyapunov_proxy,
            );
            json_put(&mut record, "state_novelty", state_space.novel_ema);
            json_put(&mut record, "state_entropy", state_space.occupancy_entropy);
            json_put(
                &mut record,
                "transition_diversity",
                state_space.transition_diversity,
            );
            json_put(
                &mut record,
                "reactor_model_ready",
                reactor_planner.ensemble.readiness(),
            );
            json_put(
                &mut record,
                "reactor_model_loss",
                reactor_planner.ensemble.train_loss_ema,
            );
            json_put(
                &mut record,
                "planner_trigger",
                reactor_planner.last_trigger.clone(),
            );
            json_put(
                &mut record,
                "planner_advantage",
                reactor_planner.last_advantage,
            );
            json_put(
                &mut record,
                "planner_option_value",
                reactor_planner.last_option_value,
            );
            json_put(
                &mut record,
                "planner_disagreement",
                reactor_planner.last_disagreement,
            );
            json_put(&mut record, "attractors", attractors.entries.len());
            json_put(&mut record, "attractor_candidates", attractors.candidates);
            json_put(&mut record, "attractor_recalls", attractors.recalls);
            json_put(
                &mut record,
                "probe_susceptibility",
                probe_controller.susceptibility_ema,
            );
            json_put(&mut record, "probe_completed", probe_controller.completed);
            uncertainty_trace.push(serde_json::Value::Object(record));
        }
        if step % 50 == 0 {
            let rolling_sec = profiling_lap.elapsed().as_secs_f32();
            let rolling_sps = if step > 0 && rolling_sec > 1e-4 {
                50.0 / rolling_sec
            } else {
                0.0
            };
            profiling_lap = std::time::Instant::now();
            println!("Chunk {}/{} (global {}) [SPS: {:.2}] | Move:{:.3} Mimic:{:.3} {} | {}@{:.2} | L{:02} rad:{:.2} | σ:{:.3}/{:.2} | V:{:.3} T:{:.2} LRx:{:.2}",
                step, total_chunks, absolute_step, rolling_sps, movement, mimic_drift_n, trend,
                current_action.label(), current_command.intensity, model.depth(), rad_amp,
                sigma, criticality.confidence, pot.v, pot.temp, latest_lr_gain);
            println!("  field H:{:.2}b · {} · synergy:{:.2} empower:{:.2} | energy:{:.2} | φ:{:.2} PI:{:.2} | self raw/eff:{:.2}/{:.2} err:{:.3}",
                field_entropy, arch_summary, synergy_val, empowerment_val, energy_state, phi,
                s_sig["pi_proxy"].as_f64().unwrap_or(0.0), controller.meta.confidence,
                adaptive_dynamics.effective_model_weight, controller.meta.error_ema);
            println!("  ecology health:{:.2} stagnation:{:.2} escape/recovery:{:.2}/{:.2} hard:{} | viable:{:.2} reward:{:+.3} μ:{:+.3} σr:{:.3}",
                adaptive_dynamics.activity_health, adaptive_dynamics.stagnation,
                adaptive_dynamics.escape_strength(), recovery_pressure, hard_recovery,
                viable_complexity, reward,
                adaptive_dynamics.reward_mean, adaptive_dynamics.reward_std());
            println!("  edge health:{:.2} order:{:.2} chaos:{:.2} | RG Hdot:{:.2} scale:{:.2} Δs:{:.2} | horizon:{:.2} λ:{:.2} χ:{:.2}",
                critical_manifold.health, critical_manifold.order_risk, critical_manifold.chaos_risk,
                rg_observer.entropy_rate_ema, rg_observer.scale_invariance_ema,
                rg_observer.disagreement_ema, critical_manifold.prediction_horizon,
                critical_manifold.lyapunov_proxy, critical_manifold.susceptibility);
            println!("  confinement {} health:{:.2} R:{:.2} vr:{:+.3} vt:{:.3} rot:{:.3} lock:{:.2} | core drive:{:.2} couple:{:.2} heat:{:.2}",
                recovery_controller.phase.label(), confinement_observer.health,
                confinement_observer.radius, confinement_observer.radial_velocity,
                confinement_observer.tangential_velocity, confinement_observer.modal_rotation,
                confinement_observer.micro_macro_lock, core_control.update_drive,
                core_control.macro_coupling, core_control.stochastic_heat);
            println!("  planner {} ready:{:.2} loss:{:.4} trigger:{} adv:{:+.3} option:{:.2} | model:{} bandit:{} selected:{}",
                planner_profile.label(), reactor_planner.ensemble.readiness(),
                reactor_planner.ensemble.train_loss_ema, reactor_planner.last_trigger,
                reactor_planner.last_advantage, reactor_planner.last_option_value,
                controller.model_proposal().label(), controller.bandit_proposal().label(),
                current_action.label());
            println!("  memory motifs:{}/{} attractors:{}/{} recalls:{} | state novel:{:.2} H:{:.2} transitions:{:.2} | action age:{} usage:{:.2}",
                motifs.entries.len(), motif_diagnostics.candidates, attractors.entries.len(),
                attractors.candidates, attractors.recalls, state_space.novel_ema,
                state_space.occupancy_entropy, state_space.transition_diversity,
                controller.action_age, adaptive_dynamics.action_use_ema[current_action.index()]);
            // Middle row of the channel-mean field as the visual tape.
            if let Ok(v) = micro_tape
                .mean(1)
                .and_then(|m| m.reshape((GRID_H, GRID_W)))
                .and_then(|m| m.narrow(0, GRID_H / 2, 1))
                .and_then(|m| m.reshape((GRID_W,)))
            {
                if let Ok(row) = v.to_vec1::<f32>() {
                    let (val_lane, grad_lane) = quantile_dual_tape(&row);
                    println!("  v {}", val_lane);
                    println!("  ∂ {}", grad_lane);
                }
            }
        }
        if absolute_step > global_step && (absolute_step + 1) % WORLD_SAVE_EVERY as u64 == 0 {
            // Save the learned model first, then the world that refers to it.
            // Each file is atomic. If Termux is killed between the two renames,
            // the model may be one checkpoint ahead of the world; that state is
            // recoverable, but the next run is not mathematically bit-exact.
            let model_tmp = format!("{}.tmp", model_path);
            if varmap.save(&model_tmp).is_ok() && std::fs::rename(&model_tmp, &model_path).is_ok() {
                match capture_world(
                    absolute_step + 1,
                    seed,
                    &rng,
                    &micro_tape,
                    &macro_tape,
                    &hidden_mem,
                    phases,
                    theta_prev,
                    theta_prev2,
                    &model,
                    rad_amp,
                    energy_state,
                    shear_phase,
                    &uncertainty,
                    &potential,
                    &semantic,
                    &criticality,
                    &episodic,
                    &novelty_buf,
                    &modal_resonators,
                    &fractal_fdn_l,
                    &fractal_fdn_r,
                    &spectral_noise,
                    &controller,
                    &motifs,
                    &last_observation,
                    &pending_predictor_input,
                    &spectral_mon,
                    &movement_mon,
                    &adaptive_dynamics,
                    &motif_diagnostics,
                    &HostRuntimeState {
                        morph_history: morph_history.clone(),
                        morph_baseline,
                        warmup_sum,
                        field_entropy_sum,
                        field_entropy_n,
                        total_complexity,
                        boost_state,
                        prev_loss_vec,
                        prev_archetype: prev_archetype.clone(),
                        stagnation_ticks,
                        last_temp,
                        smoothed_control,
                        dc_x1_l,
                        dc_y1_l,
                        dc_x1_r,
                        dc_y1_r,
                    },
                    &rg_observer,
                    &critical_manifold,
                    &state_space,
                    &reactor_planner,
                    &attractors,
                    &probe_controller,
                    &confinement_observer,
                    &recovery_controller,
                    &last_reactor_state,
                    &pending_reactor_command,
                )
                .and_then(|world| atomic_save_world(&world_path, &world))
                {
                    Ok(()) => {
                        let _ = std::fs::write(
                            &morph_path,
                            serde_json::json!({"active_depth": model.depth(), "rad_amp": rad_amp})
                                .to_string(),
                        );
                        println!("--> Model + reactor checkpoint saved at global step {} (L{:02}, motifs {}, attractors {}, planner ready {:.2}, conf {:.2})",
                            absolute_step + 1, model.depth(), motifs.entries.len(), attractors.entries.len(),
                            reactor_planner.ensemble.readiness(), controller.meta.confidence);
                    }
                    Err(e) => println!("! world checkpoint failed: {}", e),
                }
            }
        }
    }

    let total_elapsed = timer_start.elapsed().as_secs_f32();
    let overall_sps = completed_chunks as f32 / total_elapsed.max(1e-6);
    println!("\n=== PERFORMANCE REPORT ===");
    println!("Total simulation elapsed: {:.2}s", total_elapsed);
    println!("Overall performance speed: {:.2} steps/sec", overall_sps);

    // ---- STREAMED MASTERING + PRIME EXTRACTION ----
    raw_audio_writer.flush()?;
    drop(raw_audio_writer);
    let norm = 0.891 / raw_peak.max(1e-6);
    println!(
        "Mastering: streamed DC-blocked peak {:.3} normalized to -1 dBFS (gain {:.2}x).",
        raw_peak, norm
    );

    let total_frames = completed_chunks * CHUNK_SIZE;
    let rendered_seconds = total_frames as f32 / SAMPLE_RATE as f32;
    let prime_secs = 60.0f32.min(rendered_seconds.max(CHUNK_SIZE as f32 / SAMPLE_RATE as f32));
    let win = ((SAMPLE_RATE as f32 * prime_secs / CHUNK_SIZE as f32) as usize)
        .max(1)
        .min(chunk_scores.len().max(1));
    let (best_start, best_sum) = if !chunk_scores.is_empty() {
        let mut run: f32 = chunk_scores.iter().take(win).sum();
        let mut best_start = 0usize;
        let mut best_sum = run;
        if chunk_scores.len() > win {
            for start in 1..=(chunk_scores.len() - win) {
                run += chunk_scores[start + win - 1] - chunk_scores[start - 1];
                if run > best_sum {
                    best_sum = run;
                    best_start = start;
                }
            }
        }
        (best_start, best_sum)
    } else {
        (0usize, 0.0f32)
    };
    let prime_start_frame = best_start * CHUNK_SIZE;
    let prime_end_frame = ((best_start + win) * CHUNK_SIZE).min(total_frames);
    let fade = 2048usize.min(prime_end_frame.saturating_sub(prime_start_frame) / 4);

    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let output_path = format!("{}/rust_ecosystem_out.wav", base_dir);
    let prime_path = format!("{}/titan_prime_{}s.wav", base_dir, prime_secs as u32);
    let mut output_writer = hound::WavWriter::create(&output_path, spec)?;
    let mut prime_writer = hound::WavWriter::create(&prime_path, spec)?;
    let mut raw_reader = BufReader::with_capacity(1 << 20, File::open(&raw_audio_path)?);
    let mut audio_frames =
        Vec::with_capacity(prime_end_frame.saturating_sub(prime_start_frame) * 2);
    let mut bytes = [0u8; 4];
    for frame in 0..total_frames {
        raw_reader.read_exact(&mut bytes)?;
        let left = f32::from_le_bytes(bytes) * norm;
        raw_reader.read_exact(&mut bytes)?;
        let right = f32::from_le_bytes(bytes) * norm;
        output_writer.write_sample((left * 32767.0).clamp(-32768.0, 32767.0) as i16)?;
        output_writer.write_sample((right * 32767.0).clamp(-32768.0, 32767.0) as i16)?;
        if frame >= prime_start_frame && frame < prime_end_frame {
            let g = if fade == 0 {
                1.0
            } else if frame < prime_start_frame + fade {
                (frame - prime_start_frame) as f32 / fade as f32
            } else if frame >= prime_end_frame.saturating_sub(fade) {
                (prime_end_frame - frame) as f32 / fade as f32
            } else {
                1.0
            };
            let pl = left * g;
            let pr = right * g;
            prime_writer.write_sample((pl * 32767.0).clamp(-32768.0, 32767.0) as i16)?;
            prime_writer.write_sample((pr * 32767.0).clamp(-32768.0, 32767.0) as i16)?;
            audio_frames.push(pl);
            audio_frames.push(pr);
        }
    }
    output_writer.finalize()?;
    prime_writer.finalize()?;
    let _ = std::fs::remove_file(&raw_audio_path);
    println!("Audio saved to {}", output_path);
    println!(
        "Priming segment: chunks {}..{} (avg score {:.2}) -> {}",
        best_start,
        best_start + win,
        best_sum / win.max(1) as f32,
        prime_path
    );
    let n_frames = audio_frames.len() / 2;

    // ---- AUDIO FEATURE ANALYSIS (for the priming prompt) ----
    let hop = 1024usize;
    let n_hops = n_frames / hop;
    let mut env = Vec::with_capacity(n_hops);
    for h in 0..n_hops {
        let mut acc = 0.0f32;
        for i in 0..hop {
            let s = h * hop + i;
            acc += (audio_frames[2 * s] + audio_frames[2 * s + 1]).abs();
        }
        env.push(acc / hop as f32);
    }
    let onset: Vec<f32> = env.windows(2).map(|w| (w[1] - w[0]).max(0.0)).collect();
    let lag_lo = (60.0 * SAMPLE_RATE as f32 / (hop as f32 * 180.0)) as usize;
    let lag_hi = (60.0 * SAMPLE_RATE as f32 / (hop as f32 * 46.0)) as usize;
    let mut best_lag = 0usize;
    let mut best_ac = 0.0f32;
    for lag in lag_lo..=lag_hi.min(onset.len().saturating_sub(1)) {
        let mut ac = 0.0f32;
        for i in lag..onset.len() {
            ac += onset[i] * onset[i - lag];
        }
        if ac > best_ac {
            best_ac = ac;
            best_lag = lag;
        }
    }
    // v3 bug fix: guard the silent/flat case so the prompt never reads "~0 BPM".
    let bpm = if best_lag > 0 && best_ac > 1e-6 {
        60.0 * SAMPLE_RATE as f32 / (hop as f32 * best_lag as f32)
    } else {
        0.0
    };
    let mut cf_planner = FftPlanner::new();
    let cfft = cf_planner.plan_fft_forward(CHUNK_SIZE);
    let mut centroid_sum = 0.0f32;
    let mut centroid_n = 0u32;
    for w in 0..32usize {
        let start = (w * n_frames.saturating_sub(CHUNK_SIZE)) / 31usize.max(1);
        if start + CHUNK_SIZE > n_frames {
            break;
        }
        let mut buf: Vec<Complex<f32>> = (0..CHUNK_SIZE)
            .map(|i| {
                Complex::new(
                    (audio_frames[2 * (start + i)] + audio_frames[2 * (start + i) + 1]) * 0.5,
                    0.0,
                )
            })
            .collect();
        cfft.process(&mut buf);
        let (mut num, mut den) = (0.0f32, 0.0f32);
        for k in 1..CHUNK_SIZE / 2 {
            let m = buf[k].norm();
            num += m * (k as f32 * SAMPLE_RATE as f32 / CHUNK_SIZE as f32);
            den += m;
        }
        if den > 1e-6 {
            centroid_sum += num / den;
            centroid_n += 1;
        }
    }
    let centroid = if centroid_n > 0 {
        centroid_sum / centroid_n as f32
    } else {
        0.0
    };
    let tone = if centroid < 900.0 {
        "dark subterranean low-end"
    } else if centroid < 2200.0 {
        "warm midrange body"
    } else {
        "bright glassy upper spectrum"
    };
    let mut side_e = 0.0f32;
    let mut tot_e = 0.0f32;
    for i in 0..n_frames {
        let l = audio_frames[2 * i];
        let r = audio_frames[2 * i + 1];
        side_e += (l - r) * (l - r);
        tot_e += l * l + r * r;
    }
    let width = if tot_e > 1e-6 {
        (side_e / tot_e).sqrt()
    } else {
        0.0
    };
    let width_word = if width > 0.5 {
        "ultra-wide stereo field"
    } else if width > 0.2 {
        "wide stereo image"
    } else {
        "focused center image"
    };
    let tempo_txt = if bpm > 0.0 {
        format!("~{:.0} BPM", bpm)
    } else {
        "free time, no fixed pulse".to_string()
    };
    println!(
        "Features: {} · centroid {:.0} Hz ({}) · width {:.2} ({})",
        tempo_txt, centroid, tone, width, width_word
    );

    let n_trace = uncertainty_trace.len().max(1) as f64;
    let avg_phi = uncertainty_trace
        .iter()
        .map(|t| t["phi"].as_f64().unwrap_or(0.0))
        .sum::<f64>()
        / n_trace;
    let avg_aperture = uncertainty_trace
        .iter()
        .map(|t| t["aperture"].as_f64().unwrap_or(0.0))
        .sum::<f64>()
        / n_trace;
    let avg_synergy = uncertainty_trace
        .iter()
        .map(|t| t["synergy"].as_f64().unwrap_or(0.0))
        .sum::<f64>()
        / n_trace;
    let avg_temp = uncertainty_trace
        .iter()
        .map(|t| t["temp"].as_f64().unwrap_or(0.0))
        .sum::<f64>()
        / n_trace;
    let avg_sigma = uncertainty_trace
        .iter()
        .map(|t| t["sigma"].as_f64().unwrap_or(1.0))
        .sum::<f64>()
        / n_trace;
    let avg_pi = uncertainty_trace
        .iter()
        .map(|t| t["pi_proxy"].as_f64().unwrap_or(0.0))
        .sum::<f64>()
        / n_trace;
    let avg_field_h = if field_entropy_n > 0 {
        field_entropy_sum / field_entropy_n as f64
    } else {
        0.0
    };
    let dom_phase = semantic.dominant_phase();
    let dom_archetype = semantic.dominant_archetype();
    let final_depth = model.depth();

    let prompt = format!(
        "Style: {}, {}, {}, {}. Texture: {}. Tempo: {}. Tone: {}. Space: {}. Field: {} regime · {} archetype · depth L{:02}. [Phi: {:.2}, Sigma: {:.3}, Temp: {:.2}, PI-proxy: {:.2}, Aperture: {:.2}, Synergy: {:.2}, Field-Entropy: {:.2}b, Energy: {:.2}, Self-Confidence raw/effective: {:.2}/{:.2}, Ecology-Health: {:.2}, Critical-Manifold: {:.2}, RG-Scale-Invariance: {:.2}, State-Entropy: {:.2}, Viable-Attractors: {}, Motifs: {}, Planner-Ready: {:.2}, Seed: {}]",
        if avg_phi > 0.85 { "Hyper-Resonant" } else { "Chaotic" },
        if avg_aperture > 0.5 { "Evolving" } else { "Stable" },
        if total_complexity > 500.0 { "Dense" } else { "Minimal" },
        if avg_sigma > 0.50 && avg_sigma < 0.90 { "Critical-Edge Reactor" } else { "Information-Theoretic Glitch" },
        if avg_synergy > 0.6 { "Crystalline-Autonomous" } else if avg_phi > 0.55 { "Organic" } else { "Grit" },
        tempo_txt, tone, width_word, dom_phase, dom_archetype, final_depth,
        avg_phi, avg_sigma, avg_temp, avg_pi, avg_aperture, avg_synergy, avg_field_h,
        energy_state, controller.meta.confidence, adaptive_dynamics.effective_model_weight,
        adaptive_dynamics.activity_health, critical_manifold.health,
        rg_observer.scale_invariance_ema, state_space.occupancy_entropy,
        attractors.entries.len(), motifs.entries.len(), reactor_planner.ensemble.readiness(), seed
    );
    println!("\n=== GENERATIVE PRIMING PROMPT ===\n{}", prompt);
    std::fs::write(format!("{}/suno_priming_prompt.txt", base_dir), &prompt)?;

    let mut topo_writer = csv::Writer::from_path(format!("{}/ca_topology_rust.csv", base_dir))?;
    for row in topology_history {
        topo_writer.write_record(row.iter().map(|f| f.to_string()))?;
    }
    topo_writer.flush()?;

    let mut unc_writer =
        csv::Writer::from_path(format!("{}/uncertainty_trace_rust.csv", base_dir))?;
    unc_writer.write_record(&[
        "step",
        "spectral",
        "movement",
        "compositional",
        "aperture",
        "synergy",
        "empowerment",
        "phi",
        "V",
        "temp",
        "micro_gain",
        "macro_gain",
        "micro_amp",
        "macro_amp",
        "sigma",
        "criticality_confidence",
        "crit_gain",
        "flatness",
        "pe1",
        "pe4",
        "pe16",
        "pi_proxy",
        "brightness",
        "flux",
        "width",
        "structured_complexity",
        "viable_complexity",
        "novelty_dmin",
        "action",
        "action_intensity",
        "planner_proposal",
        "bandit_proposal",
        "reward",
        "recurrence",
        "model_confidence",
        "effective_model_weight",
        "prediction_error",
        "calibration_error",
        "region_change",
        "observation_delta",
        "activity_health",
        "stagnation",
        "low_motion_run",
        "escape_strength",
        "subcritical_pressure",
        "recovery_pressure",
        "hard_recovery",
        "recovery_phase",
        "confinement_health",
        "confinement_radius",
        "radial_velocity",
        "tangential_velocity",
        "modal_rotation",
        "micro_macro_lock",
        "reward_mean",
        "reward_std",
        "motifs",
        "motif_candidates",
        "motif_stored_total",
        "motif_rejected_quality",
        "motif_rejected_similarity",
        "motif_last_quality",
        "motif_last_distance",
        "rg_entropy_rate",
        "rg_scale_invariance",
        "rg_disagreement",
        "rg_active_scales",
        "critical_manifold_health",
        "order_risk",
        "chaos_risk",
        "susceptibility",
        "prediction_horizon",
        "lyapunov_proxy",
        "state_novelty",
        "state_entropy",
        "transition_diversity",
        "reactor_model_ready",
        "reactor_model_loss",
        "planner_trigger",
        "planner_advantage",
        "planner_option_value",
        "planner_disagreement",
        "attractors",
        "attractor_candidates",
        "attractor_recalls",
        "probe_susceptibility",
        "probe_completed",
    ])?;
    for t in &uncertainty_trace {
        unc_writer.write_record(&[
            t["step"].to_string(),
            t["spectral"].to_string(),
            t["movement"].to_string(),
            t["compositional"].to_string(),
            t["aperture"].to_string(),
            t["synergy"].to_string(),
            t["empowerment"].to_string(),
            t["phi"].to_string(),
            t["V"].to_string(),
            t["temp"].to_string(),
            t["micro_gain"].to_string(),
            t["macro_gain"].to_string(),
            t["micro_amp"].to_string(),
            t["macro_amp"].to_string(),
            t["sigma"].to_string(),
            t["criticality_confidence"].to_string(),
            t["crit_gain"].to_string(),
            t["flatness"].to_string(),
            t["pe1"].to_string(),
            t["pe4"].to_string(),
            t["pe16"].to_string(),
            t["pi_proxy"].to_string(),
            t["brightness"].to_string(),
            t["flux"].to_string(),
            t["width"].to_string(),
            t["structured_complexity"].to_string(),
            t["viable_complexity"].to_string(),
            t["novelty_dmin"].to_string(),
            t["action"].as_str().unwrap_or("").to_string(),
            t["action_intensity"].to_string(),
            t["planner_proposal"].as_str().unwrap_or("").to_string(),
            t["bandit_proposal"].as_str().unwrap_or("").to_string(),
            t["reward"].to_string(),
            t["recurrence"].to_string(),
            t["model_confidence"].to_string(),
            t["effective_model_weight"].to_string(),
            t["prediction_error"].to_string(),
            t["calibration_error"].to_string(),
            t["region_change"].to_string(),
            t["observation_delta"].to_string(),
            t["activity_health"].to_string(),
            t["stagnation"].to_string(),
            t["low_motion_run"].to_string(),
            t["escape_strength"].to_string(),
            t["subcritical_pressure"].to_string(),
            t["recovery_pressure"].to_string(),
            t["hard_recovery"].to_string(),
            t["recovery_phase"].as_str().unwrap_or("").to_string(),
            t["confinement_health"].to_string(),
            t["confinement_radius"].to_string(),
            t["radial_velocity"].to_string(),
            t["tangential_velocity"].to_string(),
            t["modal_rotation"].to_string(),
            t["micro_macro_lock"].to_string(),
            t["reward_mean"].to_string(),
            t["reward_std"].to_string(),
            t["motifs"].to_string(),
            t["motif_candidates"].to_string(),
            t["motif_stored_total"].to_string(),
            t["motif_rejected_quality"].to_string(),
            t["motif_rejected_similarity"].to_string(),
            t["motif_last_quality"].to_string(),
            t["motif_last_distance"].to_string(),
            t["rg_entropy_rate"].to_string(),
            t["rg_scale_invariance"].to_string(),
            t["rg_disagreement"].to_string(),
            t["rg_active_scales"].to_string(),
            t["critical_manifold_health"].to_string(),
            t["order_risk"].to_string(),
            t["chaos_risk"].to_string(),
            t["susceptibility"].to_string(),
            t["prediction_horizon"].to_string(),
            t["lyapunov_proxy"].to_string(),
            t["state_novelty"].to_string(),
            t["state_entropy"].to_string(),
            t["transition_diversity"].to_string(),
            t["reactor_model_ready"].to_string(),
            t["reactor_model_loss"].to_string(),
            t["planner_trigger"].as_str().unwrap_or("").to_string(),
            t["planner_advantage"].to_string(),
            t["planner_option_value"].to_string(),
            t["planner_disagreement"].to_string(),
            t["attractors"].to_string(),
            t["attractor_candidates"].to_string(),
            t["attractor_recalls"].to_string(),
            t["probe_susceptibility"].to_string(),
            t["probe_completed"].to_string(),
        ])?;
    }
    unc_writer.flush()?;
    println!("Topology and uncertainty trace saved to {}.", base_dir);

    // Final atomic model + organism save.
    let final_global_step = global_step + evolved_chunks as u64;
    let tmp = format!("{}.tmp", model_path);
    varmap.save(&tmp).map_err(anyhow::Error::msg)?;
    std::fs::rename(&tmp, &model_path)?;
    let world = capture_world(
        final_global_step,
        seed,
        &rng,
        &micro_tape,
        &macro_tape,
        &hidden_mem,
        phases,
        theta_prev,
        theta_prev2,
        &model,
        rad_amp,
        energy_state,
        shear_phase,
        &uncertainty,
        &potential,
        &semantic,
        &criticality,
        &episodic,
        &novelty_buf,
        &modal_resonators,
        &fractal_fdn_l,
        &fractal_fdn_r,
        &spectral_noise,
        &controller,
        &motifs,
        &last_observation,
        &pending_predictor_input,
        &spectral_mon,
        &movement_mon,
        &adaptive_dynamics,
        &motif_diagnostics,
        &HostRuntimeState {
            morph_history: morph_history.clone(),
            morph_baseline,
            warmup_sum,
            field_entropy_sum,
            field_entropy_n,
            total_complexity,
            boost_state,
            prev_loss_vec,
            prev_archetype: prev_archetype.clone(),
            stagnation_ticks,
            last_temp,
            smoothed_control,
            dc_x1_l,
            dc_y1_l,
            dc_x1_r,
            dc_y1_r,
        },
        &rg_observer,
        &critical_manifold,
        &state_space,
        &reactor_planner,
        &attractors,
        &probe_controller,
        &confinement_observer,
        &recovery_controller,
        &last_reactor_state,
        &pending_reactor_command,
    )?;
    atomic_save_world(&world_path, &world)?;
    let _ = std::fs::write(
        &morph_path,
        serde_json::json!({"active_depth": model.depth(), "rad_amp": rad_amp}).to_string(),
    );
    let metadata = std::fs::metadata(&model_path)?;
    println!(
        "Model saved to {} (L{:02}, rad {:.3}). Size: {:.2} MB",
        model_path,
        model.depth(),
        rad_amp,
        metadata.len() as f32 / 1_048_576.0
    );
    println!("Reactor saved to {} at global step {} ({} motifs, {} attractors, planner ready {:.2}, critical health {:.2}, raw/effective confidence {:.2}/{:.2}, ecology health {:.2}).",
        world_path, final_global_step, motifs.entries.len(), attractors.entries.len(),
        reactor_planner.ensemble.readiness(), critical_manifold.health,
        controller.meta.confidence, adaptive_dynamics.effective_model_weight,
        adaptive_dynamics.activity_health);
    println!("Continue normally; use --fresh-world to keep learned weights but reset the ecology, or --fresh-model to reset everything.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rich_observation() -> AudioObservation {
        AudioObservation {
            values: [
                0.82, // entropy
                0.22, // flatness near the structured-complexity optimum
                0.48, // centroid
                0.46, // flux
                0.25, // rms
                0.78, // crest
                0.52, // width
                0.35, // movement
                0.50, // synergy
                0.78, // field entropy
                0.62, // predictive structure
                0.72, // criticality health
            ],
        }
    }

    #[test]
    fn frozen_regime_reduces_effective_model_authority() {
        let mut dynamics = AdaptiveDynamics::default();
        for _ in 0..420 {
            dynamics.observe(0.0004, 0.0003, 0.002, 0.16, 0.12, 0.92);
        }
        assert!(dynamics.stagnation > 0.55);
        assert!(dynamics.low_motion_run > STAGNATION_ESCAPE_AFTER);
        assert!(dynamics.escape_strength() > 0.0);
        assert!(dynamics.effective_model_weight < 0.55);
    }

    #[test]
    fn subcritical_low_motion_triggers_escape_despite_relative_health() {
        let mut dynamics = AdaptiveDynamics::default();
        for _ in 0..300 {
            // Absolute movement is close to the old "healthy" threshold and
            // self-relative terms remain strong, but sigma is deeply low.
            dynamics.observe(0.006, 0.004, 0.025, 0.58, 0.40, 0.98);
        }
        assert!(dynamics.low_motion_run > STAGNATION_ESCAPE_AFTER);
        assert!(dynamics.escape_strength() > 0.20);
        assert!(dynamics.effective_model_weight < 0.45);
    }

    #[test]
    fn planner_authority_is_higher_near_criticality() {
        let mut subcritical = AdaptiveDynamics::default();
        let mut critical = AdaptiveDynamics::default();
        for _ in 0..160 {
            subcritical.observe(0.010, 0.008, 0.040, 0.62, 0.40, 0.98);
            critical.observe(0.010, 0.008, 0.040, 0.62, 1.00, 0.98);
        }
        assert!(critical.effective_model_weight > subcritical.effective_model_weight + 0.30);
    }

    #[test]
    fn hard_recovery_action_overrides_planner_and_bandit() {
        let mut controller = HybridController::default();
        controller.cached_model_scores[ControlAction::Recall.index()] = 10.0;
        controller.bandit.q[ControlAction::Recall.index()] = 10.0;
        let mut dynamics = AdaptiveDynamics::default();
        let mut rng = RuntimeRng::seed_from_u64(123);
        let action = controller.choose(
            1.0,
            &mut dynamics,
            true,
            Some(ControlAction::Turbulence),
            &mut rng,
        );
        assert_eq!(action, ControlAction::Turbulence);
        assert_eq!(controller.current_action, ControlAction::Turbulence);
        let recall_before = controller.bandit.q[ControlAction::Recall.index()];
        controller.bandit.apply_recovery_tax(0.8);
        assert!(controller.bandit.q[ControlAction::Recall.index()] < recall_before);
        assert_eq!(
            controller.bandit.visits[ControlAction::Turbulence.index()],
            0
        );
    }

    #[test]
    fn static_rg_agreement_does_not_saturate_scale_health() {
        let mut observer = RgObserver::default();
        let activity = [0.5f32; REGION_COUNT];
        let no_change = [0.0f32; REGION_COUNT];
        for _ in 0..100 {
            observer.update(&activity, &no_change);
        }
        assert!(observer.entropy_rate_ema < 0.01);
        assert!(observer.scale_invariance_ema < 0.30);
    }

    #[test]
    fn confinement_observer_detects_coherent_spatial_rotation() {
        let mut observer = ConfinementObserver::default();
        let state = ReactorState {
            values: [0.5; REACTOR_DIM],
        };
        let first: [f32; REGION_COUNT] = std::array::from_fn(|i| {
            let x = i % REGION_COLS;
            (TWO_PI * x as f32 / REGION_COLS as f32).cos()
        });
        let shifted: [f32; REGION_COUNT] = std::array::from_fn(|i| {
            let x = (i % REGION_COLS + 1) % REGION_COLS;
            (TWO_PI * x as f32 / REGION_COLS as f32).cos()
        });
        observer.update(&state, &first, &first, false);
        for _ in 0..12 {
            observer.update(&state, &shifted, &shifted, false);
        }
        assert!(observer.modal_rotation > 0.02);
        assert!(observer.micro_macro_lock > 0.70);
    }

    #[test]
    fn recovery_changes_actuator_class_and_limits_growth_per_episode() {
        let mut recovery = RecoveryController::default();
        recovery.update(10, true, 0.2);
        assert_eq!(recovery.phase, RecoveryPhase::EdgeAgitation);
        recovery.update(10 + RECOVERY_EDGE_CHUNKS, true, 0.2);
        assert_eq!(recovery.phase, RecoveryPhase::RotationCapture);
        recovery.update(
            10 + RECOVERY_EDGE_CHUNKS + RECOVERY_ROTATION_CHUNKS,
            true,
            0.2,
        );
        assert_eq!(recovery.phase, RecoveryPhase::GuidedPulse);
        recovery.growth_used = true;
        for step in 0..256 {
            recovery.update(1_000 + step, false, 0.7);
        }
        assert_eq!(recovery.phase, RecoveryPhase::Nominal);
        assert!(!recovery.growth_used);
    }

    #[test]
    fn channel_rotation_preserves_field_norm() {
        let device = Device::Cpu;
        let values: Vec<f32> = (0..CA_CHANNELS * GRID_H * GRID_W)
            .map(|i| ((i % 31) as f32 - 15.0) / 15.0)
            .collect();
        let field = Tensor::from_vec(values, (1, CA_CHANNELS, GRID_H, GRID_W), &device).unwrap();
        let rotated = rotate_channel_pairs(&field, 0.12).unwrap();
        // Accumulate in f64 on the host: the normal f32 parallel reduction
        // changes order after cat/reshape and its rounding error is larger
        // than the actual rotation error over ~400k values.
        let before_values = field.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let after_values = rotated.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let before = before_values
            .iter()
            .map(|v| (*v as f64).powi(2))
            .sum::<f64>();
        let after = after_values
            .iter()
            .map(|v| (*v as f64).powi(2))
            .sum::<f64>();
        let relative_error = (before - after).abs() / before.max(1e-12);
        assert!(
            relative_error < 1e-6,
            "pair rotation changed mean-square norm by {relative_error:e}"
        );
    }

    #[test]
    fn motif_memory_admits_a_good_first_candidate() {
        let mut memory = MotifMemory::default();
        let mut diagnostics = MotifDiagnostics::default();
        let stored = memory.maybe_store(
            &rich_observation(),
            SynthesisControl::default(),
            256,
            &AdaptiveDynamics::default(),
            &mut diagnostics,
        );
        assert!(stored);
        assert_eq!(memory.entries.len(), 1);
        assert_eq!(diagnostics.candidates, 1);
        assert_eq!(diagnostics.stored_total, 1);
    }

    #[test]
    fn motif_retention_rewards_diversity_as_well_as_quality() {
        let duplicate = AudioObservation {
            values: [0.2; OBS_DIM],
        };
        let isolated = AudioObservation {
            values: [0.8; OBS_DIM],
        };
        let mut memory = MotifMemory::default();
        for (observation, quality) in [
            (duplicate.clone(), 0.80),
            (duplicate, 0.80),
            (isolated, 0.65),
        ] {
            memory.entries.push_back(Motif {
                observation,
                control: SynthesisControl::default(),
                born_step: 0,
                quality,
            });
        }
        assert!(memory.retention_score(2) > memory.retention_score(0));
    }

    #[test]
    fn full_motif_memory_replaces_weakest_redundant_entry() {
        let mut memory = MotifMemory::default();
        for index in 0..MOTIF_SLOTS {
            memory.entries.push_back(Motif {
                observation: AudioObservation {
                    values: [0.0; OBS_DIM],
                },
                control: SynthesisControl::default(),
                born_step: index as u64,
                quality: if index == 0 { 0.10 } else { 0.90 },
            });
        }
        let mut diagnostics = MotifDiagnostics::default();
        let stored = memory.maybe_store(
            &rich_observation(),
            SynthesisControl::default(),
            10_000,
            &AdaptiveDynamics::default(),
            &mut diagnostics,
        );
        assert!(stored);
        assert_eq!(memory.entries.len(), MOTIF_SLOTS);
        assert!(memory.entries.iter().any(|motif| motif.born_step == 10_000));
        assert!(!memory.entries.iter().any(|motif| motif.quality == 0.10));
    }

    #[test]
    fn stagnation_scores_below_healthy_activity() {
        let observation = rich_observation();
        let previous = AudioObservation {
            values: std::array::from_fn(|i| (observation.values[i] - 0.025).clamp(0.0, 1.0)),
        };

        let mut healthy = AdaptiveDynamics::default();
        healthy.activity_health = 0.82;
        healthy.stagnation = 0.08;
        healthy.sigma_ema = 0.72;

        let mut welded = healthy.clone();
        welded.activity_health = 0.10;
        welded.stagnation = 0.90;
        welded.sigma_ema = 0.14;

        let healthy_reward = observation.reward_against(Some(&previous), 0.2, &healthy, 0.75, 2);
        let welded_reward = observation.reward_against(Some(&previous), 0.2, &welded, 0.75, 40);
        assert!(healthy_reward > welded_reward + 0.35);
    }

    #[test]
    fn transition_model_learns_a_compact_state_delta() {
        let mut rng = RuntimeRng::seed_from_u64(7);
        let mut model = TinyTransitionModel::new(&mut rng);
        let mut state = ReactorState::default();
        for i in 0..REACTOR_DIM {
            state.values[i] = 0.20 + (i % 5) as f32 * 0.03;
        }
        let command = ActionCommand::new(ControlAction::Explore, 0.65);
        let mut next = state.clone();
        next.values[3] = (next.values[3] + 0.12).clamp(0.0, 1.0);
        next.values[12] = (next.values[12] + 0.10).clamp(0.0, 1.0);
        next.values[17] = (next.values[17] - 0.08).clamp(0.0, 1.0);
        let sample = TransitionSample {
            state: state.clone(),
            action: command.features(),
            next: next.clone(),
            reward: 0.2,
        };
        let before = model.predict(&state, &sample.action).distance(&next);
        for _ in 0..8_000 {
            model.train(&sample, 0.004);
        }
        let after = model.predict(&state, &sample.action).distance(&next);
        assert!(after < before * 0.65, "before={before} after={after}");
    }

    #[test]
    fn critical_manifold_prefers_an_interior_band() {
        let manifold = CriticalManifold::default();
        let target = manifold.target();
        let ordered = [0.02; CRITICAL_DIM];
        let chaotic = [0.98; CRITICAL_DIM];
        let target_score = manifold.score_vector(&target);
        assert!(target_score > manifold.score_vector(&ordered));
        assert!(target_score > manifold.score_vector(&chaotic));
    }

    #[test]
    fn state_space_tracker_detects_new_regions() {
        let mut tracker = StateSpaceTracker::default();
        for k in 0..64 {
            let mut state = ReactorState::default();
            state.values[0] = (k % 16) as f32 / 16.0;
            state.values[3] = ((k * 3) % 16) as f32 / 16.0;
            state.values[12] = ((k * 5) % 16) as f32 / 16.0;
            state.values[20] = ((k * 7) % 16) as f32 / 16.0;
            tracker.update(&state);
        }
        assert!(tracker.counts.len() > 8);
        assert!(tracker.occupancy_entropy > 0.45);
        assert!(tracker.transition_diversity > 0.0);
    }

    #[test]
    fn attractor_recall_obeys_cooldown() {
        let mut memory = AttractorMemory::default();
        let mut state = ReactorState::default();
        state.values[0] = 0.7;
        state.values[16] = 0.8;
        memory.maybe_store(
            &state,
            SynthesisControl::default(),
            ControlAction::Resonate,
            0,
            0.8,
            0.8,
        );
        assert!(memory.has_recallable(ATTRACTOR_MIN_AGE));
        assert!(memory.recall(&state, ATTRACTOR_MIN_AGE).is_some());
        assert!(memory.recall(&state, ATTRACTOR_MIN_AGE + 1).is_none());
    }
}
