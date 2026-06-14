# Titan Audio Ecosystem: Rust Edition - Agent Briefing

## Project Overview
This is a generative audio engine that uses **Neural Cellular Automata (NCA)** coupled with an **End-to-End Differentiable Synthesis Pipeline** (FM, KAN Wavefolding). The system is designed for "online" continuous learning against target audio samples.

## Current System State (As of June 2026: v3 Gradient-Coherent)
- **Architecture:** 
    - **Micro/Macro NCA:** 144 channels, 128x hidden multiplier.
    - **Memory:** 512-dimension GRU cell.
    - **Morphic Stack:** Autonomously grows/prunes ResBlocks (up to 12) to match task complexity.
- **Learning Mechanics:**
    - **BPTT:** K-step truncated BPTT (window=4) for multi-step temporal credit assignment.
    - **Loss:** Log-magnitude spectral MSE (96 bins) replacing meaningless waveform MSE.
    - **Meta-Learning:** An Arbiter network dynamically weights loss terms based on learning progress.
    - **Self-Model:** A MonitorHead predicts structural uncertainty, trained via interoceptive MSE.
- **Persistence:** Weights (`titan_model_beta.safetensors`) and morphological depth (`morph_state.json`) persist across runs.

## Key Improvements in v3
1. **Fully Differentiable Graph:** Fixed a major bug where `to_scalar()` broke gradient flow to frequencies, FM params, pan, and filter openness. The entire pipeline now learns.
2. **Scale-Invariant RG Coarse-Graining:** Swapped broken antipodal folds for true block decimation.
3. **Morphic Homeostasis:** The model self-calibrates a baseline mimic loss during warmup, then grows (neurogenesis) if struggling or prunes if mastering the targets.
4. **Desktop/CUDA Optimized:** Adapted from a mobile-focused branch to fully leverage a GTX 1080 Ti (2048 chunk size, 512 tape length, 144 CA channels).

## Files & Directories
- `src/main.rs`: Primary engine logic.
- `Cargo.toml`: Includes `candle-core` with CUDA support.
- `main-new-phone-optimized.rs`: The historical mobile-reference architecture from which v3 features were ported.

## How to Run
```bash
cargo run --release -- /path/to/working/dir
```

## Future Directions for Agents
- **KAN Width:** The `KANLayer` currently uses 244 basis functions; tweaking this affects wavefolder resolution.
- **Synergy / Empowerment Tuning:** The proxies for mutual information and transition entropy are currently soft regularizers.
- **Choptuik Criticality Exponent:** Experimenting with the `CHOPTUIK_EXPONENT` (0.37413) affects the "burstiness" of the learning rate.
