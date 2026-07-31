# Titan Audio Ecosystem v7 (Recursive Critical Reactor)

The active implementation is the v7 recursive critical reactor ported from
the current phone release. Its control and synthesis equations are retained:
the 96-channel 64×64 toroidal neural CA, 48-value reactor state, RG observer,
8-dimensional moving critical manifold, online transition-model ensemble,
event-triggered beam planner, attractor memory, and active probes all run
unchanged. CUDA changes the tensor backend only; it does not substitute the
previous desktop v3 1D architecture.

The runtime selects CUDA device 0 when available and otherwise falls back to
CPU. Use `--device cuda`, `--device cpu`, or `--cuda-device N` to override
that choice. GPU and CPU results follow the same equations, though floating
point reduction order can make them numerically non-bit-identical.

A self-evolving generative audio engine combining **Neural Cellular Automata (NCA)**, a **Morphic Stack (Neurogenesis/Pruning)**, and **End-to-End Differentiable FM Synthesis** — all running concurrently with continuous online learning against target audio samples.

## Overview

The Titan Audio Ecosystem treats audio synthesis as an **emergent biological process**. Training WAVs act as "genetic attractors" that pull the system's chaotic internal dynamics toward interesting timbral territory, but never fully constrain it. Version 7 combines the differentiable synthesis core with a recursive critical reactor, online world model, event-triggered planning, persistent attractors, and a post-DSP observation loop.

### Core Architecture (v7)

- **End-to-End Differentiable Graph:** Every synthesis parameter (frequencies, FM ratios, modulation indices, pan, filter openness) is maintained as a computational graph tensor.
- **Micro/Macro NCA:** Two 96-channel 64×64 toroidal fields with 128 hidden convolution channels and recursive RG block decimation.
- **Morphic Stack (Neurogenesis/Pruning):** The network autonomously grows or prunes residual blocks (up to 12) based on a patience-gated homeostasis loop. Deeper networks learn slower but represent more complex functions.
- **K-Step Truncated BPTT:** Multi-step temporal credit assignment (default window `8`) for the recurrent units (NCA + 768-value GRU memory).
- **Perceptual Spectral Mimic Loss:** Raw waveform MSE is replaced by coarse 128-bin and fine 48-bin log-magnitude spectral losses. Three stereo target candidates are evaluated per chunk and the closest coarse match is selected entirely on-device.
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
- **CUDA-Native Latency Grouping & Resident Stochastic Fields:** CA clocks, Langevin kicks, Lévy radiation, and the structured shear basis reuse a deterministic device-resident field pool (about 288 MiB at the default 64-slot setting), eliminating their per-chunk host generation and H2D uploads. Min-of-K target selection uses an on-device argmin/gather. Control-plane metrics use one combined GPU→CPU readback per chunk.
- **Parallel Host DSP:** Independent left/right modal resonators, spectral-noise filters, FDN reverbs, saturation, and DC blocking run through the configured Rayon pool. Each interleaved f32 chunk is emitted with one buffered write instead of thousands of tiny writes.
- **Deferred Mastering & Priming Exports:** Audio accumulates in f32 and is mastered once at the end of the run (15 Hz DC-block, peak-normalize to −1 dBFS) instead of being hard-clipped per sample. The mastered render is then analyzed for tempo (onset autocorrelation), spectral centroid, and stereo width — these lead the generative priming prompt — and the best-scoring contiguous 60 s (field entropy × movement-in-band) is exported as a separate faded priming WAV.
- **Divergence-Safe Persistence & Bio-Reset:** A non-finite loss never reaches the optimizer, checkpoints are written atomically and only **promoted** when finite and non-regressing (generational `.prev` backups make any bad save recoverable), and a per-step bio-reset re-seeds the CA tapes if the carried state ever goes NaN mid-run.

## Synthesis Signal Flow

```
CA Channels (96, 64×64) ── decimate2 ──> Multi-Scale Topology ─────┐
                                                                    ↓
GRU Memory (768) ──── Morphic Stack (dynamic depth) ────> Differentiable Synthesis:
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
- Target audio `.wav` files placed in `BASE_DIR/OLD_WAVS`. Outputs and checkpoints are written directly to `BASE_DIR`.

### Building and Running

Use the build wrapper, which activates the micromamba CUDA-12.4 / gcc-12 / `sm_61`
toolchain and sets the cudarc/bindgen environment:

```bash
./build.sh                 # cargo build --release
./build.sh run -- /path/to/base_dir
./build.sh check           # fast type-check, no CUDA kernel compile
```

The first non-flag argument is the base/working directory. If omitted, v7
uses `/sdcard/Download`; pass `--base-dir` on desktop systems.

**Flags:**

| Flag | Effect |
|---|---|
| `--base-dir DIR`, `-b DIR` | Base/working directory (equivalent to the positional argument). |
| `--duration SECS`, `-d SECS` | Render duration (default `240`). |
| `--lr RATE`, `-l RATE` | Base AdamW learning rate (default `1.5e-3`). |
| `--bptt N`, `-w N` | Truncated-BPTT window (default `8`). |
| `--threads N`, `-t N` | Rayon/Candle CPU worker count (default: up to `6` available cores). |
| `--seed N`, `-s N` | Seed for a fresh deterministic organism (default `42`). |
| `--planner-profile cool\|balanced\|max` | Planner depth/beam compute profile. |
| `--device auto\|cuda\|cpu` | Select compute backend; `auto` prefers CUDA. |
| `--cuda-device N` | CUDA device ordinal (default `0`). |
| `--no-probes` | Disable subtle system-identification probes. |
| `--state PATH` | Override the world-checkpoint path. |
| `--model PATH` | Override the v7 model resume/output path. |
| `--import-model PATH` | Import compatible v4/v5/v6/v7 weights. |
| `--fresh-world` | Reset CA/DSP/reactor state but retain weights. |
| `--fresh-model`, `-f` | Reset both learned weights and reactor state. |

Example:

```bash
./build.sh run -- --base-dir /path/to/base_dir --device cuda --planner-profile max -d 420
```

### GPU and CPU Performance

The CUDA path deliberately keeps approximately 288 MiB of deterministic
stochastic fields resident on the selected device. The 64-slot pool supplies
the CA cell clocks, Langevin kicks, and Lévy radiation without regenerating and
uploading full 96×64×64 tensors every chunk. The structured fBm shear sine and
cosine bases are also persistent GPU tensors, and target selection uses a
device-side argmin/gather instead of pausing for a CPU decision.

The host post-processing path uses the configured Rayon pool. Left and right
spectral-noise filters, modal resonators, FDN reverbs, saturation, and DC
blockers run concurrently where their state is independent. Completed stereo
chunks are packed into a reusable byte buffer and written in one operation.

VRAM will still rise during each truncated-BPTT window and drop after the
optimizer step. This is expected: the autograd graph retains eight chunks of
activations by default and releases them when the window is detached. Lower
`--bptt` if peak VRAM approaches the device limit; increase it only when there
is sufficient headroom and the longer temporal credit window is valuable.

For profiling, force CUDA so an unavailable device is reported as an error
instead of silently falling back to CPU:

```bash
./build.sh run -- --base-dir /path/to/base_dir --device cuda --threads 6
```

### Outputs

All written to the base directory:

| File | Description |
|---|---|
| `rust_ecosystem_out.wav` | Full mastered generative render (DC-blocked, peak-normalized to −1 dBFS). |
| `titan_prime_<seconds>s.wav` | Best-scoring contiguous priming segment, up to 60 seconds and edge-faded. |
| `titan_model_v7.safetensors` | Persisted neural model weights. |
| `titan_world_v7.bin` | Versioned reactor checkpoint: CA tapes, GRU memory, DSP state, RNG, controllers, world-model ensemble/replay, attractors, probes, and runtime continuity state. |
| `titan_morph_state_v7.json` | Active morphic depth, radiation amplitude, and save-health metadata. |
| `ca_topology_rust.csv` | Macro-CA state history. |
| `uncertainty_trace_rust.csv` | System health metrics: spectral/movement/compositional uncertainty, aperture, synergy, empowerment, phi, flatness, brightness, disruption q-norm/lock, tape amplitudes, and the branching-ratio σ (criticality proxy: <1 subcritical, ~1 critical, >1 supercritical). |
| `suno_priming_prompt.txt` | Generative primer text — leads with perceptual features (BPM, tonal centroid, stereo width) followed by engine analytics. |

The console additionally reports a rolling steps-per-second figure every 50 chunks
and a total performance summary at the end of the run.
