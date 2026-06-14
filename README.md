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
- **Learning-Progress Arbiter:** The multi-objective loss function is dynamically re-weighted by an Arbiter that receives meta-rewards for improving losses.
- **Predictive Defibrillator:** Foresees stagnation and applies targeted Choptuik criticality-seeking learning rate bursts.

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
- Target audio `.wav` files in `Desktop/OLD_WAVS/` (or specify via CLI).
- A capable CUDA device (e.g., GTX 1080 Ti).

### Building and Running

```bash
cargo build --release
./target/release/titan_audio_ecosystem /path/to/base_dir
```

*(By default, the engine will look for WAVs in `/sdcard/Download/Desktop/OLD_WAVS/` if no path is provided. Pass your workspace path as the first argument).*

### Outputs

| File | Description |
|---|---|
| `rust_ecosystem_out.wav` | Resulting generative audio. |
| `titan_model_beta.safetensors` | Persisted model weights. |
| `ca_topology_rust.csv` | Macro-CA state history. |
| `uncertainty_trace_rust.csv` | System health metrics (Spectral, Movement, etc.). |
| `suno_priming_prompt.txt` | Generative primer text based on engine analytics. |
| `morph_state.json` | Stores the current active depth of the Morphic Stack. |
