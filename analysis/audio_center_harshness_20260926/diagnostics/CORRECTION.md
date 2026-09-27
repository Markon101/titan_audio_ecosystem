# Correction to the 2026-09-26 Gemini audio diagnostic

`diagnostic_report.json` and `RESEARCH_DIAGNOSTIC_FINDINGS.md` are preserved as the original investigation, but their E2, E3, prime-origin, and causal conclusions are superseded. Use `diagnostic_report.corrected.json` for the checked measurements. Reproduce it with:

```sh
python scripts/audio_diagnostic_experiments.py
```

The immutable `9bcf582d4b13` WAV has 28,798,976 frames (599.9787 seconds). The available tagged trace has only 12 sampled rows, steps 81,371–81,481, and matches a later run whose metadata says 112 chunks, 458,752 frames, and 9.5573 seconds. The archived metadata file named for `9bcf582d4b13` also contains those later run fields. Therefore E2 cannot attribute target, width register, or oscillator behavior to the 600-second WAV. E3 cannot reconstruct its prime-selection score from that trace; it also omitted the `+ observation.width() * 0.40` score term in `src/main.rs`. The corrected report marks E2 and E3 unavailable.

An exact match of the prime's unfaded PCM16 interior places it at frames 17,031,168–19,910,656, or 354.816–414.805 seconds. The original report's 359.02–419.01-second localization was wrong. The original E1 used integer division and omitted the final 540–599.9787-second interval. That interval has left/right correlation 0.8841, side/mid power −11.95 dB, and a 2–6 kHz power fraction of 47.32%. The earlier 300–360-second fixed window is more centered than the extracted prime by these metrics; the “single most mono-centered window” claim is unsupported.

The 2–6 kHz fraction is a descriptive band fraction from sampled 4096-frame FFTs. It has no matched target or listening control, so it cannot establish that the renderer's learned weights caused perceived harshness. E4 measures distortion from `tanh(0.92 x)` on synthetic 1-kHz pure sines. The pre-tanh renderer waveform was not saved, so those values cannot measure distortion added to this recording or exclude the post path as a contributor.

The existing EQ sidecars are reproducible transforms of their listed inputs: their output SHA-256 hashes match the receipts. The −3 dB bell lowers the 2–6 kHz fraction and leaves broad stereo correlation and side/mid ratio similar in these files, while lowering RMS by about 10%. These measurements do not establish that musical detail, transients, or downstream priming quality were preserved. Listen to level-matched raw and processed clips before adopting the transform.
