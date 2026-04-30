# Titan Audio Ecosystem: Rust Edition - Agent Briefing

## Project Overview
This is a generative audio engine that uses **Neural Cellular Automata (NCA)** coupled with **FM Synthesis** and **KAN-based Wavefolding**. The system is designed for "online" continuous learning, mimicking target audio samples while maintaining an autonomous, evolving internal state.

## Current System State (As of April 24, 2026)
- **Model Size:** ~16.55 MB (`titan_model.safetensors`).
- **Architecture:** 
    - **Micro/Macro NCA:** 144 channels, 16x hidden multiplier.
    - **Memory:** 256-dimension GRU cell.
    - **Normalization:** Full Recurrent LayerNorm integrated into NCA and GRU transitions for manifold stability.
    - **Pacemaker:** Rhythmic "burst" defibrillator (8-chunk decay envelope) to prevent stagnation.
- **Persistence:** The model persists in `/sdcard/Download/titan_model.safetensors`. Training is cumulative across runs.

## Key Improvements & Fixes
1.  **Manifold Stabilization:** Added LayerNorm to prevent the internal CA states from collapsing to zero.
2.  **Metabolic Logic Fix:** Corrected a critical bug where macro-modulation was incorrectly scaling the `METABOLIC_DECAY`, leading to signal death.
3.  **Rhythmic Pacemaker:** Upgraded the single-shock defibrillator to a decaying burst system, creating more musical "breathing" and "pulse" effects.
4.  **Volume Governance:**
    *   `energy_loss` term in the optimizer keeps RMS volume around 0.25.
    *   Real-time gain boost ensures at least 25% peak volume in the final output.
5.  **Target Mimicry:** The model samples from multiple `.wav` files in `/sdcard/Download/` to guide its spectral evolution.

## Files & Directories
- `src/main.rs`: Primary engine logic.
- `/sdcard/Download/titan_model.safetensors`: The weights.
- `/sdcard/Download/rust_ecosystem_out.wav`: Most recent audio output.
- `/sdcard/Download/ca_topology_rust.csv`: Macro-CA state history.
- `/sdcard/Download/uncertainty_trace_rust.csv`: System health metrics (Spectral Entropy, Movement, etc.).

## How to Run
```bash
cargo build --release
./target/release/titan_audio_ecosystem
```
The engine will load the existing model, train for the duration specified in `DURATION_SECONDS`, save the updated weights, and output the new audio track.

## Future Directions for Agents
- **Aperture Refinement:** The `branch_aperture` logic controls how often the macro-CA updates. Tweaking this can lead to different "temporal textures."
- **Normalization Groups:** Currently using full LayerNorm. Exploring GroupNorm might allow for more independent "species" of signals within the CA.
- **KAN Optimization:** The `KANLayer` uses basis functions for wavefolding; increasing `KAN_BASIS_FUNCTIONS` further could add more grit/complexity to the timbre.
