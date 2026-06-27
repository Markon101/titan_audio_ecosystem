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
- **Divergence-Safe Persistence:** A non-finite loss never reaches the optimizer, and checkpoints are written atomically and only **promoted** when finite and non-regressing — a generational `.prev` backup makes any bad save recoverable.

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
| `--fresh-substrate` | Cold-start the dynamical state (ignore any saved substrate). |
| `--no-substrate-kick` | Skip the on-load Lévy nudge applied to the restored substrate. |

### Outputs

All written to the base directory:

| File | Description |
|---|---|
| `rust_ecosystem_out.wav` | Resulting generative audio. |
| `titan_model_beta.safetensors` | Persisted model weights. |
| `titan_substrate.safetensors` | Persisted dynamical substrate (CA tapes, GRU memory, phases, carried synthesis scalars) — enables cross-run continuity. |
| `titan_morph_state.json` | Morphic-stack active depth, radiation amplitude (`rad_amp`), and the run-health metric used by the non-regression checkpoint gate. |
| `*.prev` | Generational backup of the previous good weights / substrate / morph state. |
| `ca_topology_rust.csv` | Macro-CA state history. |
| `uncertainty_trace_rust.csv` | System health metrics (Spectral, Movement, etc.). |
| `suno_priming_prompt.txt` | Generative primer text based on engine analytics. |
