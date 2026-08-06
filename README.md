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

Target WAVs are indexed, not loaded into RAM. On first use Titan creates
`titan_corpus_manifest_v7.json`: generated Titan audio is quarantined, files
are explicitly assigned to train/validation/exclude roles, and mastered/remix
variants share a sampling family. At the beginning of a ~21.9 s episode, Titan
samples a family independently of its own output and follows one passage
contiguously. This removes the old nearest-target feedback loop.

Supervision now covers 20 Hz--20 kHz at 1,024-, 4,096-, and tape scales, plus
relative band energy, chroma/pitch salience, onset envelopes, modulation
spectrum, 0.68/2.73/5.46-second recurrence geometry, level, and chunk seams.
Exact waveform phase is not a target. The decoder is a phase-continuous modal
bank whose carrier pitch, auxiliary modes, ratios, amplitudes, damping,
family gains, and stereo width are continuous functions of recurrent memory
and regional CA state.

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

AdamW first/second moments and cumulative update count are saved in
`titan_optimizer_v7.safetensors`. They resume only when their global step
matches the world checkpoint. `--fresh-decoder` resets the audible decoder but
retains CA, GRU, morphic, arbiter, and episodic weights. `--run-tag NAME`
isolates all experiment artifacts, making current-decoder, fresh-decoder, and
fresh-model comparisons safe.

Morphic depth has two checkpoint-compatible development paths. Sustained
ecological or mimic pressure may add one layer at a 512-chunk boundary. A
healthy organism with both low normalized field entropy and low predictive
structure may add one layer at the slower 2,048-chunk boundary. Pruning uses
the slow boundary and requires a healthy, non-stagnant, structurally rich
regime. Every structural event prints its reason. Motif memory retains up to
64 diverse observations for long-horizon recall.

Telemetry names and their scientific limitations are documented in
[`METRICS.md`](METRICS.md). Schema v4 separates raw CA movement from the
uncertainty movement feature, records exact morph events, supplies a topology
row index, and adds source-band, chroma, onset, modulation, recurrence,
held-out-validation, sub-bass, and optimizer-continuity diagnostics. Run the
verification suite with `cargo test` and
`cargo clippy --all-targets -- -D warnings`.
