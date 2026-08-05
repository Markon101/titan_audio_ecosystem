# TITAN Audio Ecosystem v7

An adaptive, deterministic 2D neural-cellular-automata audio ecosystem for
CPU-based training and rendering on Termux. TITAN couples a 64-channel 2D CA,
episodic and motif memory, adaptive control, target-audio grounding, and a
differentiable, phase-continuous synthesis path.

v7 is a clean training generation; it does not load or overwrite earlier model
or world files. Its defaults are `titan_model_v7.safetensors` and
`titan_world_v7.bin`. Start it with fresh weights for the new renderer/loss
geometry:

Training WAV files are read from `OLD_WAVS` beneath the selected base directory.
Model, world-state, telemetry, and rendered WAV files are saved alongside it.

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
./target/release/titan --base-dir /sdcard/Download --duration 240 --seed 42 --fresh-model
```

Use `./target/release/titan --help` for checkpoint, reset, thread, learning-rate,
and truncated-BPTT options.

Target WAVs are indexed, not loaded into RAM. At the beginning of a ~21.9 s
episode, min-of-K chooses one nearby source passage; subsequent chunks follow
that passage contiguously. Supervision combines 1,024-sample, 4,096-sample,
and tape-scale full-band spectral views, time-aligned energy envelopes, and an
explicit chunk-seam term. This gives temporal patterns a causal target while
remaining phase-invariant where exact waveform phase would be misleading.

All oscillator families, the field scan, and stereo delay carry state across
chunks. The audible post path is intentionally transparent: learned rendering,
bounded saturation, and a stateful DC blocker. Discrete control can no longer
inject untrained noise, resonators, or echo after the source loss.

`--bptt` selects the optimizer's gradient-averaging horizon. The differentiable
recurrent tape is capped at 8 chunks and longer horizons accumulate detached
8-chunk gradient segments, so `--bptt 64` no longer retains a 64-chunk CA graph
in memory. Use the default `--bptt 8` for fresh models and fastest adaptation;
larger values trade update frequency for lower-variance gradients and are most
useful for already-developed organisms.

Morphic depth has two checkpoint-compatible development paths. Sustained
ecological or mimic pressure may add one layer at a 512-chunk boundary. A
healthy organism with both low normalized field entropy and low predictive
structure may add one layer at the slower 2,048-chunk boundary. Pruning uses
the slow boundary and requires a healthy, non-stagnant, structurally rich
regime. Every structural event prints its reason. Motif memory retains up to
64 diverse observations for long-horizon recall.

Telemetry names and their scientific limitations are documented in
[`METRICS.md`](METRICS.md). Schema v3 separates raw CA movement from the
uncertainty movement feature, records exact morph events, supplies a topology
row index, and adds oscillator, source-episode, seam, level, and fine-scale
loss diagnostics. Run the verification
suite with `cargo test` and `cargo clippy --all-targets -- -D warnings`.
