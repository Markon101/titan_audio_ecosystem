# TITAN Audio Ecosystem v8

TITAN is an adaptive, deterministic neural-cellular-automata audio ecosystem
for sustained CPU training and rendering in Termux. v8 is a phone-first model:
it keeps corpus and telemetry memory bounded, reduces the expensive spatial CA
rule, and spends most of its approximately 14.9 million parameters in a
channel-aware decoder that runs at 32 control frames per audio chunk.

v8 writes independent artifacts and never overwrites v7 model or world state:

- `titan_model_v8.safetensors`
- `titan_optimizer_v8.safetensors`
- `titan_world_v8.bin`
- `titan_morph_state_v8.json`
- `titan_run_metadata_v8.json`

The corpus classification remains in `titan_corpus_manifest_v7.json`. Its
train/development/validation family semantics are data policy rather than model
architecture, so v8 deliberately reuses it.

## Build and run

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
./target/release/titan --base-dir /sdcard/Download --duration 240 --seed 42 --fresh-model
```

Training WAVs live in `OLD_WAVS` under the selected base directory. Use
`./target/release/titan --help` for all checkpoint, reset, thread, learning-rate,
run-tag, BPTT, and core-update options.

To migrate compatible recurrent/control tensors from v7 while deliberately
resetting the incompatible audible path:

```bash
./target/release/titan \
  --base-dir /sdcard/Download \
  --import-model /sdcard/Download/titan_model_v7.safetensors \
  --fresh-decoder
```

## Architecture

The organism has two 64-channel toroidal neural cellular automata:

- a 64x64 micro field updated every chunk;
- a 32x32 macro field eligible to evolve on a four-chunk clock.

Each CA uses a depthwise 3x3 spatial perception kernel followed by learned
64-to-128 and 128-to-64 pointwise channel mixing. This reduces the dominant
spatial multiply count by roughly 4.8x relative to v7's dense 3x3 rule.
Precomputed deterministic cell-clock masks remove per-chunk mask allocation and
random-number generation.

Global micro-channel means and a 64-dimensional episodic attention readout feed
a 512-unit GRU. Its output passes through an RMS-normalized, Swish residual
MorphicStack with up to 12 blocks. The audible decoder also reads the fields
directly: 64 micro tokens from an 8x8 grid and 16 macro tokens from a 4x4 grid
are projected and attended by recurrent memory.

The resulting context is decoded into 32 temporal control frames per 4,096
samples. Six low-rate residual blocks produce independent time-varying controls
for carrier phase and drive, 32 regional partials, differentiable mid/side
excitation, channel openness, KAN drive, and mid/side dynamics. Linear
interpolation expands those controls to sample rate. Phase-continuous carrier,
FM, auxiliary and modal oscillators remain the primary sound source.

Separate left and right eight-basis KAN-inspired sinusoidal wavefolders remain
active. A bounded global pan is only a residual after independently decoded
left/right and mid/side structure.

## Online training and memory behavior

Target audio is indexed by manifest, not loaded into RAM. A coherent source
episode is seek-decoded in blocks of at most 16 chunks (about 0.5 MiB of stereo
FP32 audio), amortizing file-open and resampling work while preserving constant
memory. Generated Titan audio remains quarantined from the training corpus.

`--bptt` controls the optimizer's gradient-averaging horizon; the differentiable
tape remains capped at eight chunks. `--core-update-every N` performs full
CA/GRU/Morphic backward on one of every N tapes and trains the decoder on the
intervening tapes. The default is 4. Use `1` for full end-to-end BPTT on every
tape when maximum adaptation matters more than speed.

The phase profiler reports model forward, target I/O, loss/metrics, backward,
optimizer, output I/O and checkpoint time per completed chunk. These values are
also written to run metadata, so thread counts and architecture changes can be
compared after thermal soak rather than from cold-start speed.

Large topology and scalar trace records are streamed during the run and
converted to their stable CSV formats at finalization. Audio is streamed to a
temporary FP32 file and normalized in a second pass. Full, mutually consistent
model/optimizer/world checkpoints are written every 1,024 chunks and on clean
shutdown, limiting storage traffic from the larger model.

## Supervision and verification

Supervision covers 20 Hz--20 kHz at 1,024-, 4,096-, and tape scales, together
with relative band energy, chroma/pitch salience, onset envelopes, modulation,
multi-lag recurrence, level, chunk seams, and correlation-aware stereo
geometry. Target phase is not supplied to the renderer.

Telemetry semantics and their scientific limitations are documented in
[`METRICS.md`](METRICS.md). Verify a build with:

```bash
cargo test
cargo clippy --all-targets -- -D warnings
```
