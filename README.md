# Titan Audio Ecosystem v3 (Gradient-Coherent Release)

A self-evolving generative audio engine combining **Neural Cellular Automata (NCA)**, a **Morphic Stack (Neurogenesis/Pruning)**, and **End-to-End Differentiable FM Synthesis** — all running concurrently with continuous online learning against target audio samples.

## Overview

The Titan Audio Ecosystem treats audio synthesis as an **emergent biological process**. Training WAVs act as "genetic attractors" that pull the system's chaotic internal dynamics toward interesting timbral territory, but never fully constrain it. Version 3 represents a massive leap in learning coherence: the entire synthesis pipeline is fully differentiable, allowing the system to gradient-descend its parameters (frequencies, FM ratios, pan, etc.) using K-step Truncated BPTT.

### Core Architecture (v3)

- **End-to-End Differentiable Graph:** Every synthesis parameter (frequencies, FM ratios, modulation indices, pan, filter openness) is maintained as a computational graph tensor.
- **Micro/Macro NCA:** 144 channels, 128× hidden multiplier, scale-invariant fractal forward pass (with true RG block decimation).
- **Morphic Stack (Neurogenesis/Pruning):** The network autonomously grows or prunes residual blocks (up to 12) based on a patience-gated homeostasis loop. Deeper networks learn slower but represent more complex functions.
- **K-Step Truncated BPTT:** Multi-step temporal credit assignment (window=4) for the recurrent units (NCA + GRU).
- **Perceptual Spectral Mimic Loss:** Raw waveform MSE is replaced by a log-magnitude spectral loss evaluated on 96 log-spaced bins. Stereo targets are loaded and resampled directly.
- **Semantic Field & Lévy Radiation:** The internal cellular state is continuously analyzed via its Shannon entropy and mapped to archetypes. At window boundaries, heavy-tailed Lévy radiation occasionally mutates the substrate.
- **Information-Theoretic Regularization:**
  - **Synergy / Empowerment:** Mutual information and transition entropy are calculated and optimized within the graph.
  - **Self-Model (Monitor Head):** A sub-network attempts to predict the system's own interoceptive state.
- **Learning-Progress Arbiter:** The multi-objective loss function is dynamically re-weighted by an Arbiter that receives meta-rewards for improving losses. The weights it applies to the losses are **detached** — the Arbiter is trained purely on learning-progress allocation plus an entropy regularizer, so it cannot cheat by zeroing out hard objectives.
- **Predictive Defibrillator:** Foresees stagnation and applies targeted Choptuik criticality-seeking learning rate bursts.
- **Cross-Run Continuity:** The dynamical substrate (CA tapes, GRU memory, oscillator phases, carried synthesis scalars) **persists across runs** alongside the weights, so the organism continues its trajectory instead of cold-starting from noise each launch.
- **Predictive Safety (q-Factor) Disruption Controller & fBm Shear:** Predicts cellular q-collapse based on macro variance, coupling, and rail proximity. Injects traveling-wave fBm shear perturbations to break phase locks, automatically backing off once q recovers.
- **Metabolic Energy & Stagnation Homeostats:** Features a metabolic charge tracker (`energy_state`) that scales frequencies/openness under heavy loads, macro **and** micro tape amplitude homeostats that hold the fields off their saturation rails, and an archetype stagnation tracker that triggers curiosity-driven learning rate boosts.
- **Permutation-Entropy Phi:** The system-complexity signal `phi` is order-4 permutation entropy (ordinal temporal structure, bounded [0,1]) computed alongside spectral flatness and brightness from a single per-chunk FFT — replacing the Shannon spectral entropy that saturated on broadband output.
- **Mid/Side Haas Delay Stereo & Wave Morphing:** Upgraded spatialization with a Mid/Side panning matrix and a 16-sample Haas delay on the wide side channel. Supports learned wave morphing that projects oscillator carriers and formants dynamically between pure sines and rounded triangle waves.
- **CUDA-Native Latency Grouping & Lévy Radiation:** Vectorized `levy_radiate` runs entirely on the GPU. All per-step control-plane scalars (metrics, synergy, empowerment, macro variance, rail proximity, carried frequencies, controller outputs) ride two combined GPU→CPU copies (`first_metrics` and `second_metrics`), and the two tape homeostats share a third — the training loop performs no other per-step scalar readbacks.
- **Deferred Mastering & Priming Exports:** Audio accumulates in f32 and is mastered once at the end of the run (15 Hz DC-block, peak-normalize to −1 dBFS) instead of being hard-clipped per sample. The mastered render is then analyzed for tempo (onset autocorrelation), spectral centroid, and stereo width — these lead the generative priming prompt — and the best-scoring contiguous 60 s (field entropy × movement-in-band) is exported as a separate faded priming WAV.
- **Divergence-Safe Persistence & Bio-Reset:** A non-finite loss never reaches the optimizer, checkpoints are written atomically and only **promoted** when finite and non-regressing (generational `.prev` backups make any bad save recoverable), and a per-step bio-reset re-seeds the CA tapes if the carried state ever goes NaN mid-run.

## Synthesis Signal Flow

```
CA Channels (144) ──── decimate2 ───> Multi-Scale Topology ─────────┐
                                                                    ↓
GRU Memory (512) ──── Morphic Stack (dynamic depth) ────> Differentiable Synthesis:
                                                            - FM Ratio / Index
                                                            - Frequencies / Pan
                                                            - KAN Wavefolding
                                                            - Stereo Output
```

## System Components

- **MorphicStack:** Dynamically sized stack of `ResBlock` units using `gelu_approx`.
- **SpectralProjector:** Computes differentiable log-magnitude spectrograms.
- **SemanticField:** Maps states to epistemic phases (PRIMORDIAL → MASTERY).
- **MonitorHead:** Predicts the system's own structural uncertainty.
- **KANLayer:** Kolmogorov-Arnold Network basis function wavefolder.

## Getting Started

### Prerequisites

- Rust and Cargo installed.
- A capable CUDA device (e.g., GTX 1080 Ti, `sm_61`).
- Target audio `.wav` files placed **directly in the base/working directory** you pass on the command line. All outputs and checkpoints are also written there.

### Building and Running

Use the build wrapper, which activates the micromamba CUDA-12.4 / gcc-12 / `sm_61`
toolchain and sets the cudarc/bindgen environment:

```bash
./build.sh                 # cargo build --release
./build.sh run -- /path/to/base_dir
./build.sh check           # fast type-check, no CUDA kernel compile
```

The first non-flag argument is the base/working directory (also the WAV source).
If omitted, it defaults to `/home/anon/Downloads`.

**Flags:**

| Flag | Effect |
|---|---|
| `--base-dir DIR`, `-b DIR` | Base/working directory (equivalent to the positional argument). |
| `--duration SECS`, `-d SECS` | Simulation length in seconds (default `160`). Also bounds the priming-segment length. |
| `--lr RATE`, `-l RATE` | Base AdamW learning rate (default `1.3e-3`); the Choptuik/defib gains multiply this. |
| `--bptt N`, `-w N` | Truncated-BPTT window in chunks (default `4`). Larger = more temporal credit, more VRAM. |
| `--threads N`, `-t N` | Rayon thread count (default: all cores). Mostly affects the CPU DSP/analysis path. |
| `--fx on\|off` | Render FX chain — QNM resonator bank + fractal FDN reverb (default `on`). `off` renders the raw synthesized voice dry and skips the per-sample IIR work. Training is unaffected either way: the mimic loss is computed pre-FX by design. |
| `--fresh`, `-f` | Ignore the weight checkpoint and morph sidecar — train from random init. Implies `--fresh-substrate`. |
| `--fresh-substrate` | Cold-start the dynamical state (ignore any saved substrate). |
| `--no-substrate-kick` | Skip the on-load Lévy nudge applied to the restored substrate. |

Example:

```bash
./build.sh run -- /path/to/base_dir -d 420 -w 8 --lr 8e-4
```

### Outputs

All written to the base directory:

| File | Description |
|---|---|
| `rust_ecosystem_out.wav` | Full mastered generative render (DC-blocked, peak-normalized to −1 dBFS). |
| `titan_prime_60s.wav` | Highlight priming segment: the best-scoring contiguous 60 s of the render, edge-faded, for conditioning external audio models. |
| `titan_model_beta.safetensors` | Persisted model weights. |
| `titan_substrate.safetensors` | Persisted dynamical substrate (CA tapes, GRU memory, phases, carried synthesis scalars) — enables cross-run continuity. |
| `titan_morph_state.json` | Morphic-stack active depth, radiation amplitude (`rad_amp`), and the run-health metric used by the non-regression checkpoint gate. |
| `*.prev` | Generational backup of the previous good weights / substrate / morph state. |
| `ca_topology_rust.csv` | Macro-CA state history. |
| `uncertainty_trace_rust.csv` | System health metrics: spectral/movement/compositional uncertainty, aperture, synergy, empowerment, phi, flatness, brightness, disruption q-norm/lock, tape amplitudes, and the branching-ratio σ (criticality proxy: <1 subcritical, ~1 critical, >1 supercritical). |
| `suno_priming_prompt.txt` | Generative primer text — leads with perceptual features (BPM, tonal centroid, stereo width) followed by engine analytics. |

The console additionally reports a rolling steps-per-second figure every 50 chunks
and a total performance summary at the end of the run.
