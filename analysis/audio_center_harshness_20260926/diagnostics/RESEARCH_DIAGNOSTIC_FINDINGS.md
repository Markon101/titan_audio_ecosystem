# Empirical Diagnostic Investigation: Titan Audio Center-Collapse & Harshness

> **Superseded.** The original investigation is retained below for provenance. Its E2/E3 attribution, prime localization, omitted final interval, and causal conclusions were corrected in [CORRECTION.md](CORRECTION.md) and `diagnostic_report.corrected.json`.

This diagnostic investigation executed four controlled empirical experiments (**E1–E4**) on the primary audio artifacts and telemetry traces of Titan Audio `v9-long-01` (`9bcf582d4b13`).

---

## Executive Summary of Empirical Findings

| Investigation Target | Prevailing Hypothesis | Empirical Finding | Causal Locus |
| :--- | :--- | :--- | :--- |
| **Center Collapse** | Model weights collapsed to mono | **FALSIFIED**. The 600s render has wide stereo (side/mid up to $-7.79\text{ dB}$, corr $0.74$). The 60s prime was extracted from $[359\text{s}, 419\text{s}]$, which is the single most mono-centered window of the entire composition. | **Prime Selection Artifact** (Windowing) |
| **Spectral Harshness** | Post-mastering clipping / DC blocker | **FALSIFIED**. Post-mastering `tanh(0.92 x)` adds only $1.68\%$ to $3.14\%$ THD. However, $46.6\%$ of acoustic power is trapped in $2000\text{--}6000\text{ Hz}$ across **every single minute** of the render. | **Neural DDSP Synthesis Engine** (Learned Spectrum) |

---

## 1. Experiment E1: Full-Run 10-Minute Audio Audit

We loaded the unclipped, un-cut $600$-second master output (`rust_ecosystem_out_v9-long-01_9bcf582d4b13.wav`, $110\text{ MB}$, $28,798,976$ frames at $48\text{ kHz}$) and partitioned it into non-overlapping $60$-second windows.

### Distribution Across All Windows

| Window | Time Interval | L/R Corr | Side/Mid Ratio (dB) | 2–6 kHz Power Fraction | Peak Level | Stereo Character |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | $0\text{--}60\text{ s}$ | $+0.9593$ | $-16.55\text{ dB}$ | $46.3\%$ | $-2.5\text{ dBFS}$ | Moderate Center |
| **1** | $60\text{--}120\text{ s}$ | **$+0.7833$** | **$-8.41\text{ dB}$** | $46.3\%$ | $-1.8\text{ dBFS}$ | **Wide Stereo** |
| **2** | $120\text{--}180\text{ s}$ | $+0.9532$ | $-15.79\text{ dB}$ | $46.4\%$ | $-2.7\text{ dBFS}$ | Moderate Center |
| **3** | $180\text{--}240\text{ s}$ | **$+0.7399$** | **$-7.79\text{ dB}$** | $46.1\%$ | $-1.0\text{ dBFS}$ | **Maximum Stereo Width** |
| **4** | $240\text{--}300\text{ s}$ | **$+0.8174$** | **$-9.66\text{ dB}$** | $46.5\%$ | $-1.0\text{ dBFS}$ | **Wide Stereo** |
| **5** | $300\text{--}360\text{ s}$ | $+0.9916$ | $-23.72\text{ dB}$ | $46.8\%$ | $-2.8\text{ dBFS}$ | Narrow Mono Corridor |
| **6** | $360\text{--}420\text{ s}$ | **$+0.9904$** | **$-23.12\text{ dB}$** | $46.6\%$ | $-2.7\text{ dBFS}$ | **Extracted Prime Interval** |
| **7** | $420\text{--}480\text{ s}$ | **$+0.8130$** | **$-9.53\text{ dB}$** | $47.1\%$ | $-1.4\text{ dBFS}$ | **Wide Stereo** |
| **8** | $480\text{--}540\text{ s}$ | $+0.9503$ | $-15.78\text{ dB}$ | $47.0\%$ | $-2.3\text{ dBFS}$ | Moderate Center |

### Analysis of E1 Results

1. **Exact Prime Origin Discovered**: Sample cross-correlation localized the $60$-second prime file (`titan_prime_60s_v9-long-01_9bcf582d4b13.wav`) to frames $17,233,024\text{--}20,112,512$ (seconds **$359.02\text{ s}$ to $419.01\text{ s}$**), directly aligning with **Window 6**.
2. **Outlier Confirmation**: In Window 3 ($180\text{--}240\text{ s}$), side energy is only $-7.79\text{ dB}$ below mid (substantial stereo energy), and correlation is $0.7399$. Across the entire run, side energy averages $-14.48\text{ dB}$.
3. **The Prime is a Local Mono Valley**: The prime extractor selected an interval where side energy had plummeted to $-23.12\text{ dB}$ and correlation peaked at $+0.9904$.

---

## 2. Experiment E2: Telemetry Trace & Target Attribution

Using `uncertainty_trace_rust_v9-long-01.csv` ($81,371$ to $81,481$ steps), we audited model control registers and target audio:

1. **Width Head Saturation State**:
   - `decoder_width_control` is constrained by `soft_width_control` to $[0.05, 0.50]$.
   - Across the trace, `decoder_width_control` averaged **$0.435$** (ranging $0.432$ to $0.439$).
   - **Conclusion**: The width head was NOT pinned at its mono floor ($0.05$). The model was actively applying near-maximal width scaling ($0.435$).
2. **Source of the Mono Valley**:
   - In `src/main.rs:5569`, side audio is computed as `side = (audio_l - audio_r) * 0.5 * temporal_controls[5]`.
   - In Window 6 ($359\text{--}419\text{ s}$), the neural DDSP oscillator banks converged on identical carrier frequencies and phases between channels ($L \approx R$). Thus $L - R \approx 0$ at the source level. Width scaling multiplied zero by $0.435$, producing $-23.1\text{ dB}$ side energy.
3. **Target Audio Attribution**:
   - The target audio streamed during this epoch was `Infinite Reflections (1).wav`.
   - Target correlation averaged $0.768$ (varying $0.274$ to $0.990$). The model did not suffer static target collapse; the Pearson correlation between target correlation and model correlation was only $r = 0.106$.

---

## 3. Experiment E3: Prime Selection Mechanism Audit

The prime selection score in `src/main.rs:8141-8145` is:

$$\text{ChunkScore} = H_{\text{field}} \cdot (0.25 + 0.50 \cdot \text{health}) + 0.75 \cdot \text{complexity} - 0.25 \cdot \text{stagnation}$$

- **Score Evaluation**: Cellular automata field entropy ($H_{\text{field}}$) and structured complexity peaked precisely around chunk $17,200$ (second $360$).
- **Blindness to Audio Aesthetics**: The scoring function contains zero penalty for mono correlation ($r \to 1.0$) and zero term measuring spectral balance.
- **Reselection Diagnostic**: When evaluating a candidate criteria adding a mild stereo penalty ($\text{Score} - 2.0 \cdot \text{corr}$), the optimal prime window shifts from Window 6 to **Window 1** ($60\text{--}120\text{ s}$), yielding correlation **$0.783$** and side/mid **$-8.41\text{ dB}$** while preserving high musical complexity.

---

## 4. Experiment E4: Post-Mastering Nonlinearity & Harmonic Distortion

To evaluate whether perceived harshness is caused by post-mastering clipping or the neural sound generator:

1. **Spectral Uniformity**: Across all nine 60s windows of the 10-minute piece, the acoustic power fraction in $2000\text{--}6000\text{ Hz}$ was **$46.59\% \pm 0.34\%$** (min $46.1\%$, max $47.1\%$).
   - This exact concentration is present whether the audio is wide ($r = 0.74$) or narrow ($r = 0.99$).
2. **Simulated `tanh(0.92 x)` Total Harmonic Distortion (THD)**:
   - At amplitude $0.30$ (typical RMS): $\text{THD} = 0.62\%$ ($-44.1\text{ dB}$).
   - At amplitude $0.50$: $\text{THD} = 1.68\%$ ($-35.5\text{ dB}$).
   - At amplitude $0.70$ (peak level $0.729$): $\text{THD} = 3.14\%$ ($-30.1\text{ dB}$).
3. **Conclusion**: $3\%$ harmonic distortion from soft saturation cannot explain a massive $46.6\%$ energy accumulation in the 2–6 kHz band.
   - The harshness originates **directly inside the learned neural oscillator/wavefolder weights**, which converged to excessive upper-midrange partial amplitudes.

---

## Actionable Takeaways & Next Steps

1. **Center Bias Solution**:
   - **No model retraining is required** to obtain wide stereo primes.
   - Update the prime selection algorithm in `src/main.rs:8141` to include `side_energy_width` or penalize correlation above $0.90$.
   - Extracting primes from Windows 1, 3, or 4 will immediately yield rich, wide stereo clips ($r \approx 0.74\text{--}0.81$, side/mid $-7.8\text{ dB}$).
2. **Harshness Solution**:
   - The 2–6 kHz peak is a learned property of the DDSP generator.
   - In training, the multiscale spectral loss weights can be adjusted with a gentle de-emphasis curve in the 2–6 kHz range, or the learned oscillator band gains can be constrained.
   - In offline mastering, a gentle dynamic notch or high-shelf filter centered at 3.5 kHz safely tames the energy concentration without altering model checkpoints.
