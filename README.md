# Titan Audio Ecosystem (Rust Edition)

A generative audio engine that utilizes **Neural Cellular Automata (NCA)**, **FM Synthesis**, and **KAN-based Wavefolding** for continuous online learning and autonomous audio evolution.

## Overview

The Titan Audio Ecosystem is designed to mimic target audio samples while maintaining an evolving internal state. It features a complex architecture that combines deep learning with traditional synthesis techniques to create organic, breathing soundscapes.

### Core Architecture

- **Micro/Macro NCA:** 144 channels with a 16x hidden multiplier for complex spatial-temporal dynamics.
- **Memory:** 256-dimension GRU cell for long-term temporal coherence.
- **Synthesis Engine:** FM Synthesis coupled with KAN (Kolmogorov-Arnold Network) based wavefolding.
- **Stabilization:** Integrated Recurrent LayerNorm to maintain manifold stability and prevent signal collapse.
- **Pacemaker:** A rhythmic "burst" defibrillator system that prevents stagnation through periodic energy injections.

## System Components

- **Neural Cellular Automata:** Governs the "metabolic" and "topology" of the sound.
- **Adaptive Metabolic Field:** Implements a dynamic, per-cell learning rate that adapts based on local entropy, replacing static decay.
- **FM Synthesis:** Provides the base timbral generation.
- **KAN Wavefolding:** Adds non-linear complexity and "grit" to the output.
- **Volume Governance:** Real-time gain normalization and energy loss optimization to maintain consistent output levels.
- **TapeCodec Visualization:** A dual-lane (Value + Gradient) ASCII codec for real-time visualization of internal state dynamics.

## Files & Directories

- `src/main.rs`: Primary engine logic and implementation.
- `titan_model.safetensors`: Persisted model weights (located in `/sdcard/Download/` by default).
- `rust_ecosystem_out.wav`: The most recently generated audio output.
- `ca_topology_rust.csv`: Macro-CA state history for analysis.
- `uncertainty_trace_rust.csv`: System health metrics including spectral entropy and movement.

## Getting Started

### Prerequisites

- Rust and Cargo installed.
- Target audio files (.wav) placed in the target directory (default: `/sdcard/Download/`).

### Running the Engine

```bash
cargo build --release
./target/release/titan_audio_ecosystem
```

The engine will:
1. Load existing weights from `titan_model.safetensors`.
2. Load target audio samples to guide evolution.
3. Run for the duration specified in the source code.
4. Save updated weights and output the generated audio to `rust_ecosystem_out.wav`.

## Technical Details

- **Sample Rate:** 48,000 Hz
- **Architecture Stability:** Uses `METABOLIC_DECAY` and `LayerNorm` to ensure the system doesn't collapse or explode.
- **Rhythmic Pulse:** Employs an 8-chunk decay envelope for musical "breathing."

## License

[Specify License if applicable]
