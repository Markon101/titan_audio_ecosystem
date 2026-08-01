# TITAN Audio Ecosystem v6

An adaptive, deterministic 2D neural-cellular-automata audio ecosystem for
CPU-based training and rendering on Termux. TITAN couples a 64-channel 2D CA,
episodic and motif memory, adaptive control, target-audio grounding, and a
differentiable synthesis path.

Training WAV files are read from `OLD_WAVS` beneath the selected base directory.
Model, world-state, telemetry, and rendered WAV files are saved alongside it.

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
./target/release/titan --base-dir /sdcard/Download --duration 240 --seed 42
```

Use `./target/release/titan --help` for checkpoint, reset, thread, learning-rate,
and truncated-BPTT options.

Telemetry names and their scientific limitations are documented in
[`METRICS.md`](METRICS.md). Run the verification suite with `cargo test` and
`cargo clippy --all-targets -- -D warnings`.
