# Decoder research queue

This is a design queue, not an implemented decoder change. TITAN v9 deliberately
keeps the v8.1 audible renderer intact so the manifold intervention can be
measured without changing both the organism and its instrument at once.

## Best next avenue: morphable spectral residual

The strongest fit is a small causal spectral residual branch beside the
existing oscillator/DDSP renderer:

1. Render TITAN's present phase-continuous stereo signal.
2. Predict a bounded complex STFT or MDCT residual at the existing low control
   rate.
3. Reconstruct with overlap-add state carried in the world checkpoint.
4. Begin with a zero output projection, so enabling the branch is exactly
   output-preserving.

[Vocos](https://arxiv.org/abs/2306.00814) shows that a non-autoregressive model
can generate Fourier coefficients directly instead of synthesizing a waveform
sample by sample. [StreamCodec](https://arxiv.org/abs/2504.06561) is an even
closer efficiency reference: it uses a fully causal MDCT design, reports a
7-million-parameter model, 20 ms algorithmic latency, and roughly 20x real-time
CPU inference. TITAN should borrow the representation and causality ideas, not
import either full codec architecture.

This branch could itself morph. Extra frequency groups, temporal residual
blocks, or stereo-residual rank can activate as MorphicStack depth grows. Each
new group should use a zero-initialized output projection and receive its own
telemetry, preserving the current model before the new capacity learns.

## Low-risk avenue: alias-free periodic nonlinearities

[BigVGAN](https://arxiv.org/abs/2206.04658) combines periodic activations with
explicit anti-aliasing. That is directly relevant to TITAN's sinusoidal
wavefolders and its existing ultrasonic/foldback guardrail. A later isolated
experiment could replace one wavefold stage with an oversampled, low-pass
periodic block while leaving oscillator phases and all stereo controls intact.
This is smaller and easier to ablate than replacing the decoder.

Success should mean lower ultrasonic ratio and lower multiscale spectral loss
without reduced transient flux, stereo incoherence, or blinded preference. A
lower ultrasonic scalar alone is not enough.

## Medium-risk avenue: lightweight subband refinement

[LDCodec](https://arxiv.org/abs/2510.15364) is explicitly designed around a
low-complexity smartphone decoder and uses lightweight residual units with
subband and full-band discrimination. TITAN could borrow the subband residual
layout while avoiding a codec token bottleneck. Four or eight learned bands
would be enough for an initial experiment; the current renderer would remain
the source and the new network would only refine its spectral envelope and
transients.

Adversarial training should not be the first version. It would add another
moving system to an already online, non-stationary organism and could make the
cause of improvements difficult to identify. Start with the existing
multiscale spectral, onset, level, seam, and stereo losses.

## Complementary avenue: richer differentiable resonators

[Differentiable Digital Signal Processing](https://arxiv.org/abs/2001.04643)
demonstrates the value of placing known signal structure inside a trainable
model. TITAN already follows this philosophy. A compact extension would add a
small stable pole/zero or modal filter bank controlled at decoder-frame rate,
rather than adding a generic waveform network. It could model evolving
formants, damping, and material resonance with few parameters and preserve the
phone-first renderer.

## Recommended order

1. Run and evaluate the v9 manifold change with the decoder frozen as an
   architectural variable.
2. Test the alias-free wavefold stage as the smallest decoder-only ablation.
3. Prototype a zero-initialized causal STFT/MDCT residual with fixed capacity.
4. If it earns its cost, couple spectral groups or residual depth to morphic
   growth using append-preserving activation.
5. Consider subband refinement or stable learned resonators only after the
   simpler residual establishes a measurable gap.

Reject an avenue if it materially harms chunks per second or memory stability,
if improvement exists only against training families, or if it increases
stereo width through panned-mono imbalance rather than lower correlation with
balanced channel levels.
