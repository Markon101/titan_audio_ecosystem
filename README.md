# Titan Audio Ecosystem v2 (Rust Edition)

A self-evolving generative audio engine combining **Neural Cellular Automata (NCA)**, a **4-operator FM synthesis network**, **KAN-based Wavefolding**, **granular resynthesis**, and **Gray-Scott Reaction-Diffusion** dynamics — all running concurrently with continuous online learning against target audio samples.

## Overview

The Titan Audio Ecosystem treats audio synthesis as an **emergent biological process**. Training WAVs act as "genetic attractors" that pull the system's chaotic internal dynamics toward interesting timbral territory, but never fully constrain it. The result is a system that hallucinates variations rather than playing back recordings.

### Core Architecture (v2)

- **Micro/Macro NCA:** 144 channels, 128× hidden multiplier, scale-invariant fractal forward pass (coarse + fine blend).
- **GRU Memory:** 512-dim hidden state providing infinite-context temporal coherence.
- **4-Op FM Network (NEW):** DX7 Algorithm 5 topology (Op0→Op1, Op2→Op3); ratios and modulation indices driven dynamically from GRU output.
- **Additive CA Synthesis (NEW):** 32 partial harmonic series with amplitudes derived from a learned linear projection of the CA channel means. Each step the CA literally defines the timbre.
- **Granular Layer (NEW):** 8 concurrent grains reading from a 2-second ring buffer of past audio, with Hann windowing, linear interpolation, and pitch spread controlled by uncertainty state.
- **Gray-Scott Reaction-Diffusion (NEW):** 1D R-D system drives wavefolder pre-gain — complex spatial patterns → more aggressive harmonic distortion.
- **Lorenz Attractor Noise (NEW):** Replaces uniform anti-stagnation noise with structured chaotic noise (long-range correlated, musically interesting).
- **Euclidean Rhythm Gating (NEW):** E(k,24) patterns gate the synthesis voices — on-beat → full synth, off-beat → granular swells. Pattern re-derived from GRU state every 96 chunks.
- **KAN Wavefolding (FIXED):** Kolmogorov-Arnold Network basis function wavefolder; bug fix for in_features dimension handling (was silently dropped, now properly contracted).
- **Binaural Beat (NEW):** L=49.0 Hz, R=56.83 Hz — 7.83 Hz difference targets the Schumann resonance / alpha brainwave range.
- **Multi-Scale Temporal Loss (NEW):** Roughness computed at 4 scales (1, 4, 16, 64 samples) to approximate multi-resolution STFT magnitude loss — far superior to single-scale MSE.
- **Spectral Flatness Incentive (NEW):** Additional loss term that rewards spectrally flat (broad-spectrum) output.
- **Audio Arbiter:** 8-weight dynamic loss rebalancer (was 6).
- **Pacemaker Defibrillator:** 8-chunk burst system for anti-stagnation; phi-gated learning rate.

## Synthesis Signal Flow

```
CA Channels (144) ──── partial_proj (Linear) ──→ Additive Synth (32 partials)
                  \                                                              \
GRU Memory (512) ──── FM ratio/index nets ──→ 4-Op FM Synthesis                ├── MIX ──→ KAN Wavefold ──→ Pan → Stereo Out
                  \                                                              /
Past Audio Ring Buffer ──────────────────────── Granular Layer (8 grains) ─────/
                       ↑                                                    ↑
                 R-D complexity ────────── modulates wavefolder pre-gain ───┘
                 Euclidean rhythm ─────── gates synth mix (on/off beat) ──────┘
                 Lorenz attractor ──────── CA anti-stagnation noise ──────────┘
```

## System Components

- **LorenzAttractor:** Chaotic 3-variable ODE (σ=10, ρ=28, β=8/3); outputs correlated noise.
- **EuclideanRhythm:** Bjorklund/Bresenham approximation for E(k,n) patterns.
- **ReactionDiffusion:** Gray-Scott 1D system; `f`=0.055, `k`=0.062; spatially evolving texture driver.
- **GranularLayer:** Ring-buffer grain engine with Hann windowing and fractional interpolation.
- **FMOpNetwork:** 4-operator DX7 Algorithm 5 with GRU-derived ratios (0.125–7.0) and mod indices (0–8).
- **AdditiveBank:** 32-partial harmonic series; amplitudes from CA channel projections via tanh.
- **KANLayer (fixed):** Proper (N, in×basis) × (out, in×basis)ᵀ contraction. Second recursive fold.
- **SpectralEntropyMonitor / MovementCoherenceMonitor:** Uncertainty signal analysis.
- **AudioUncertaintyState:** Phi (resonance), aperture (branch probability), compositional tracking.
- **DefibrillatorController / AudioArbiter:** Meta-level controllers for stagnation prevention and loss weighting.

## Getting Started

### Prerequisites

- Rust and Cargo installed.
- CUDA toolkit (GTX 1080 Ti: use `CUDA_COMPUTE_CAP=86` override — see below).
- Target audio `.wav` files in `/home/anon/Downloads/`.

### Building and Running

```bash
CUDA_COMPUTE_CAP=86 cargo build --release
./target/release/titan_audio_ecosystem
```

> **Note for GTX 1080 Ti users:** The auto-detected compute cap (6.1) isn't supported by newer nvcc targets. Override with `CUDA_COMPUTE_CAP=86` to use the closest available target.

### Outputs

| File | Description |
|---|---|
| `/home/anon/Downloads/rust_ecosystem_out.wav` | 7-minute stereo audio |
| `/home/anon/Downloads/titan_model.safetensors` | Persisted model weights (cumulative) |
| `/home/anon/Downloads/ca_topology_rust.csv` | Macro-CA state history |
| `/home/anon/Downloads/uncertainty_trace_rust.csv` | Spectral, movement, R-D, Lorenz, rhythm per step |
| `/home/anon/Downloads/suno_priming_prompt.txt` | AI prompt derived from session statistics |

## Technical Details

| Parameter | Value |
|---|---|
| Sample Rate | 48,000 Hz |
| Chunk Size | 2,048 samples (≈42.7ms) |
| Duration | 420s (7 minutes) |
| CA Channels | 144 (micro + macro) |
| GRU Hidden Dim | 512 |
| Partials | 32 |
| FM Operators | 4 (Algorithm 5) |
| Grains | 8 concurrent |
| Grain Buffer | 96,000 samples (2 seconds) |
| R-D Tape | 128 cells (Gray-Scott) |
| Rhythm Steps | 24 (re-derived every 96 chunks) |
| Binaural Beat | 7.83 Hz (L=49 Hz, R=56.83 Hz) |

## Theory

See [`THEORY_OF_CHAOS_2026_05_16.md`](THEORY_OF_CHAOS_2026_05_16.md) for the full theoretical framing of this system as an emergent biological organism rather than a conventional generative model.

## License

[Specify License if applicable]
