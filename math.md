# Titan dynamics: mathematical status, edge-of-chaos hypotheses, and attractor tests

## Executive conclusion

Titan is presently a **bounded adaptive stochastic neural cellular automaton** with a recursive audio/control loop.  The implementation has several ingredients that plausibly support long-lived complex transients and metastable regimes:

- a local nonlinear 2-D toroidal rule;
- asynchronous cell updates;
- coupled micro and macro fields;
- recurrent memory;
- bounded heavy-tailed perturbations and temperature-dependent forcing;
- a potential-shaped homeostat;
- state-dependent action selection and structural depth changes.

Those ingredients make edge-of-chaos behavior possible, but they do not prove it.  In particular:

1. Boundedness can be proved for the frozen-weight runtime state.
2. Existence of at least one statistical invariant regime is strongly supported and can be proved under standard continuity/Feller assumptions.
3. The current scalar `sigma` is a movement-autocorrelation persistence estimate.  It is **not** a branching ratio and **not** a Lyapunov exponent.
4. The potential `V` is a useful homeostatic shaping function, but it is not currently a global Lyapunov function for the complete dynamics.
5. The last long pre-v7 trajectory was not at `sigma ~= 1` even by Titan's
   present proxy. A 30-second v7.1 modal-gain test materially reduced acoustic
   collapse and improved fixed-probe spectral distance, but is far too short
   to establish an attractor class or musical convergence. That run preceded
   the manifest's strict family-boundary repair, so it is not held-out evidence.
6. A deterministic strange attractor cannot be inferred from a noisy audio waveform or a single scalar trace.  With ongoing random forcing, the more appropriate objects are a random attractor, stationary measure, or metastable family of random attractors.

v7 deliberately starts a new weight generation because the earlier objective
could not observe important audible degrees of freedom and allowed an
after-loss noise shortcut. The best scientific next step after v7 establishes
a nontrivial trained regime is **instrumentation with frozen weights**: estimate
common-noise Lyapunov exponents, damage spreading, correlation lengths,
recurrence, attractor dimension, and basin structure without optimizer drift.

This document uses the following evidence labels:

- **Theorem:** follows from the implemented equations under explicitly stated assumptions.
- **Conditional proposition:** follows if the listed regularity assumptions hold.
- **Empirical result:** calculated from a recorded Titan run.
- **Conjecture:** plausible but not established.
- **Falsifier:** an observation that would reject the associated conjecture.

## 1. What mathematical object is Titan?

Let one audio chunk be one dynamical time step, with

\[
    \Delta t = \frac{4096}{48000} \approx 0.085333\ \mathrm{s}.
\]

The principal runtime variables are

\[
z_t = (x_t,y_t,h_t,e_t,c_t,q_t,\varphi_t,d_t,r_t),
\]

where:

- \(x_t\in\mathbb{R}^{64\times64\times64}\) is the micro field;
- \(y_t\in\mathbb{R}^{64\times64\times64}\) is the macro field;
- \(h_t\in\mathbb{R}^{512}\) is GRU memory;
- \(e_t\) is metabolic energy;
- \(c_t\) is the compact adaptive-controller state;
- \(q_t\) contains motif and episodic memories;
- \(\varphi_t\in\mathbb{T}^{42}\) contains carrier, FM, auxiliary, and regional-partial phases;
- \(d_t\) contains the bounded stateful Haas and DC-blocker states;
- \(r_t\) is the pseudorandom-generator state.

For fixed network weights \(\theta\), the chunk map can be written

\[
    z_{t+1}=F_\theta(z_t,\omega_t),
\]

where \(\omega_t\) collects the stochastic cell masks, target selection, control exploration, radiation event, and Langevin kick at time \(t\).  This is naturally a **random dynamical system**.  For a fixed seed and saved RNG state it is also a deterministic skew-product system on the enlarged state \((z_t,r_t)\).

During learning, the weights change:

\[
    (\theta_{k+1},m_{k+1},v_{k+1})
    =\operatorname{AdamW}(\theta_k,\nabla_\theta \mathcal L_k,m_k,v_k).
\]

The learning process is consequently non-autonomous unless
\((\theta,m,v,k)\) is included in the state. v7.1 persists Adam moments and
the cumulative update count, accepting them only when their global-step stamp
matches the world. A missing/mismatched optimizer still changes the full
learning dynamical system because it restarts behind a 32-update warmup. Claims
about an attractor should therefore be made in one of two clearly separated
regimes:

1. **Frozen organism:** fixed \(\theta\), no optimizer steps; appropriate for attractor and criticality analysis.
2. **Adaptive organism:** changing \(\theta\); appropriate for studying tracking, drift, and a possible pullback or slowly moving statistical attractor.

Mixing these regimes makes an attractor claim ambiguous.

## 2. Implemented neural-CA equations

Ignoring batching notation, each neural CA has a residual local rule

\[
R_\theta(x)=W_2*\operatorname{ReLU}(W_1*x),
\]

where \(*\) is a circular 2-D convolution on the torus.  A fixed anisotropy field \(A\), optional macro modulation \(u\), local restoring bias \(b(x)\), and asynchronous Bernoulli mask \(M_t\) produce an update of the form

\[
\widetilde x_{t+1}
=x_t+\alpha M_t\odot\left[A\odot u_t\odot R_\theta(x_t)+b(x_t)\right],
\qquad \alpha=0.1.
\]

The micro field is blended according to macro metabolic activity \(m_t\in[0.01,1]\):

\[
x_{t+1}^{(0)}
=(1-m_t)x_t+m_t\widetilde x_{t+1}.
\]

It is then clamped, mean-damped, possibly radiated, kicked, gain-scaled, and clamped again.  Schematically,

\[
x_{t+1}
=C_1\!\left[g_x(t)\left(D(C_1(x_{t+1}^{(0)}))+\ell_t+\eta_t\right)\right],
\]

where:

- \(C_a(v)=\operatorname{clip}(v,-a,a)\);
- \(D(v)=v-\gamma\langle v\rangle\), with \(\gamma=0.035\);
- \(\ell_t\) is sparse, bounded Cauchy-like radiation;
- \(\eta_t\) is the temperature-dependent Gaussian kick;
- \(g_x(t)\in[0.6,1.4]\) is the potential controller's micro gain.

The macro map is intermittently advanced, passed through `tanh`, damped, driven by a normalized multioctave shear field, and gain-scaled:

\[
y_{t+1}
=g_y(t)\tanh\!\left(D(\widehat F_\theta(y_t,M_t))+s_t\right),
\qquad g_y(t)\in[0.6,1.4].
\]

The GRU memory obeys

\[
h_{t+1}=(1-z_t)\odot h_t+z_t\odot n_t,
\]

with \(z_t\in(0,1)^{512}\) and \(n_t\in[-1,1]^{512}\).

These equations are not a classical finite-state cellular automaton.  They are a continuous-state, stochastic, learned coupled-map lattice with recurrent global summaries.

## 3. Results that can be proved now

### 3.1 Bounded absorbing runtime state

**Theorem 1 — frozen-weight forward boundedness.**  For fixed finite weights, finite controller memories, and the implemented clamps, Titan's runtime field, recurrent, oscillator, and DSP states remain in a bounded set.

**Proof.**

1. The micro field is explicitly clamped to \([-1,1]\) after radiation, kick, and potential gain.  Therefore

   \[
   \|x_t\|_\infty\le 1.
   \]

2. Before the macro field is retained, its driven value passes through `tanh` and is multiplied by \(g_y\le1.4\).  Therefore

   \[
   \|y_t\|_\infty\le1.4.
   \]

   On entry to the next neural update it is damped and clamped to \([-1,1]\), so no unbounded accumulation occurs.

3. If \(h_0\in[-1,1]^{512}\), the GRU update is a coordinatewise convex combination of \(h_t\in[-1,1]\) and \(n_t\in[-1,1]\).  Induction gives

   \[
   \|h_t\|_\infty\le1.
   \]

4. Each morphic residual is `tanh` bounded.  Active depth is at most 12 and the added residual gains have finite sum.  Hence the refined hidden vector is bounded even though it need not remain in \([-1,1]\).

5. Energy is clamped to \([0.18,0.96]\); temperature and controller features are clamped to compact intervals; phases are reduced modulo \(2\pi\); motif and episodic buffers have finite capacity.

6. Final audio samples pass through `tanh`. The stateful 16-sample Haas tail is
   finite, and the stable DC blocker has pole \(0.998<1\), so bounded input
   produces bounded renderer state.

The finite Cartesian product of these bounded closed sets is compact in the implementation's finite-dimensional state space.  Thus the frozen-weight runtime enters and remains in a compact absorbing set. \(\square\)

**Limitation.** This theorem does not include the optimizer.  There is no proved compact bound on all learned weights or Adam moment tensors.  Full adaptive-runtime boundedness therefore remains unproved, even though gradient clipping and AdamW make numerical explosion less likely.

### 3.2 Exact radiation-hazard invariance

**Theorem 2 — BPTT-independent event probability.**  If a reference window of \(W\) chunks should contain at least one radiation event with probability \(P\), the implemented per-chunk probability

\[
p=1-(1-P)^{1/W}
\]

preserves that probability exactly under independent event draws.

**Proof.** The probability of no event in \(W\) chunks is

\[
(1-p)^W=\left((1-P)^{1/W}\right)^W=1-P.
\]

Therefore the probability of at least one event is \(P\). \(\square\)

This is a genuine mathematical improvement over tying ecological event frequency to optimizer horizon.

### 3.3 Existence of a statistical invariant regime

**Conditional Proposition 1.**  Freeze the weights and structural depth.  If the induced Markov transition kernel on the compact runtime state is Feller, then at least one invariant Borel probability measure exists.

**Reason.** Compactness gives tightness of empirical occupation measures, and the Feller property permits a Krylov–Bogolyubov limiting argument.  This proves existence, not uniqueness, mixing, criticality, or strangeness.

Discrete control decisions can make the exact real-arithmetic transition map piecewise continuous at action ties.  With randomized exploration, ties of continuously distributed scores should have probability zero, but this regularity should be checked rather than silently assumed.

At literal finite machine precision, the complete frozen program plus finite RNG state has a finite state space.  It must eventually enter a recurrent class or periodic orbit in the enlarged machine state.  That fact is mathematically true but scientifically weak: the recurrence time can be astronomical and says nothing about useful macroscopic organization.

### 3.4 What the potential controller does and does not prove

Titan defines

\[
\begin{aligned}
V(s)={}&k_a(a_x-a_x^*)^2+B(a_x)
       +k_a(a_y-a_y^*)^2+B(a_y)\\
     &+k_\rho(\rho-\rho^*)^2
       +G\exp[-(m/w)^2]
       +k_e(e-e^*)^2
       +k_\sigma(\sigma-1)^2.
\end{aligned}
\]

For the amplitude coordinates it calculates

\[
g(a)=\operatorname{clip}\left(1-\eta\frac{\partial V}{\partial a},g_{\min},g_{\max}\right)
\]

and approximately applies \(a^+=g(a)a\).  If all other dynamics are temporarily removed and clipping is inactive,

\[
\Delta a=-\eta a V'(a),
\]

so a first-order Taylor expansion yields

\[
\Delta V\approx V'(a)\Delta a=-\eta a[V'(a)]^2\le0
\]

for \(a\ge0\).  This establishes **local conditional descent** of the isolated amplitude subsystem for sufficiently small effective steps.

It does **not** make \(V\) a Lyapunov function for Titan because:

- the neural rule, shear, radiation, kick, optimization, and action switching add non-gradient forcing;
- only micro and macro amplitude coordinates use the explicit derivative of \(V\);
- coupling, movement, energy, and `sigma` are controlled indirectly;
- clamping and `tanh` make the global map nonsmooth;
- the implemented barrier uses `min(1.01)`, so it is finite rather than mathematically divergent;
- measured amplitudes are not exact scalar state coordinates under the field map.

The scientifically correct name is therefore **potential-shaped homeostatic controller**, not global energy minimization.

**Conditional Proposition 2 — isolated amplitude fixed points are locally stable.**  Remove neural forcing, noise, clipping, and cross-coordinate changes, and let

\[
a^+=a[1-\eta V'(a)].
\]

The controller's amplitude equilibrium solves

\[
2k_a(a-a^*)+\frac{B}{(1.02-a)^2}=0.
\]

With the implemented constants, the micro equilibrium is approximately `0.49398` rather than the nominal `0.50`, and the macro equilibrium is approximately `0.39572` rather than `0.40`.  The barrier shifts both equilibria slightly away from the rail.  Linearization gives

\[
\frac{d a^+}{da}\bigg|_{a=\bar a}
=1-\eta\bar a V''(\bar a).
\]

The resulting slopes are approximately `0.576` and `0.663`, both strictly inside the unit circle, so these isolated scalar fixed points are locally asymptotically stable.  This is a useful controller-level result, but external field forcing can and does move the observed amplitudes away from these equilibria.

Temperature has a separate exact invariant interval.  Its update is a convex combination

\[
T^+=(1-\beta)T+\beta T_{\mathrm{target}},
\qquad \beta\in\{0.035,0.10\},
\]

so `T in [0,1]` and `T_target in [0,1]` imply `T+ in [0,1]`.  This proves bounded temperature, not convergence, because the target itself changes with the organism.

## 4. Edge of chaos: the criterion Titan actually needs

### 4.1 Tangent dynamics

For a differentiable deterministic map \(z_{t+1}=F(z_t)\), an infinitesimal perturbation obeys

\[
\delta z_{t+1}=J_t\delta z_t,
\qquad J_t=DF(z_t).
\]

The maximal Lyapunov exponent is

\[
\lambda_{\max}
=\lim_{T\to\infty}\frac1T
 \log\frac{\|J_{T-1}\cdots J_0\delta z_0\|}{\|\delta z_0\|}.
\]

For the unclipped interior of the neural CA, the micro-field contribution has the schematic Jacobian

\[
J_t^{(x)}
\approx D_t\left[I+\alpha\operatorname{diag}(M_t)
  D\!\left(A\odot u_t\odot R_\theta\right)(x_t)
  +\alpha\operatorname{diag}(M_t)Db(x_t)\right],
\]

where \(D_t\) includes metabolic mixing, global-mean damping, gain, and derivatives of clamps.  Saturated clamp coordinates have zero local derivative; unsaturated residual paths retain the identity term.  This makes the balance subtle:

- residual identity promotes perturbation persistence;
- ReLU/convolution and macro modulation can expand perturbations;
- local anti-rail bias, mean damping, `tanh`, and clipping contract them;
- random masks intermittently remove local expansion;
- common stochastic forcing can synchronize or desynchronize depending on the local Jacobian.

An operational classification is:

\[
\lambda_{\max}<0 \Rightarrow \text{ordered/contractive},
\]

\[
\lambda_{\max}>0 \Rightarrow \text{sensitive/chaotic tangent dynamics},
\]

\[
\lambda_{\max}\approx0 \Rightarrow \text{candidate edge of chaos}.
\]

Near zero is necessary for the usual edge-of-chaos claim, but not sufficient for useful computation.  A system can have \(\lambda\approx0\) because it is nearly frozen, intermittently clipped, quasiperiodic, or dominated by an external drive.

### 4.2 Common-noise Lyapunov exponent

Titan is stochastically forced, so paired trajectories must receive the **same** random masks, actions, target selections, radiation events, and kicks.  With common forcing,

\[
z'_{t+1}=F(z'_t,\omega_t),\qquad
z_{t+1}=F(z_t,\omega_t).
\]

The separation then measures sensitivity to state, not sensitivity to different noise realizations.  A Benettin-style estimator is:

1. Initialize \(z'_0=z_0+\varepsilon v_0\).
2. Advance both states with the same \(\omega_t\).
3. Measure \(d_t=\|z'_t-z_t\|\).
4. Accumulate \(\log(d_t/\varepsilon)\).
5. Rescale the shadow perturbation to length \(\varepsilon\) in the measured tangent direction.

Then

\[
\widehat\lambda_{\max}
=\frac1T\sum_{t=1}^{T}\log\frac{d_t}{\varepsilon}.
\]

Independent random streams answer a different question—noise response—not chaos.  They must not be used for this test.

### 4.3 Why current `sigma` is not this quantity

Titan's estimator stores scalar movement \(m_t\), fits mean-centered regressions at lags \(k\in\{1,2,4\}\), and converts positive slopes to per-step persistence:

\[
\widehat\sigma_k
=\left(\frac{\operatorname{Cov}(m_t,m_{t+k})}
              {\operatorname{Var}(m_t)}\right)^{1/k}.
\]

It averages valid lag estimates and confidence-weights the result using dynamic range, stationarity, lag agreement, and sample support.

This is better than a through-origin autocorrelation, but it measures persistence in one aggregate observable.  A slowly varying but entirely stable system can have \(\widehat\sigma\approx1\).  A high-dimensional chaotic system can have low movement autocorrelation.  Therefore:

\[
\widehat\sigma\approx1\ \not\Rightarrow\ \lambda_{\max}\approx0.
\]

`sigma` should remain a useful controller sensor, but the console should describe it as **movement persistence** until a damage-spreading or tangent-space calibration connects it to a branching process.

### 4.4 No finite-system singularity has been established

Titan has a finite field, bounded continuous states, explicit noise, clipping, and bounded controller gains.  These features generally round sharp transitions.  The plasticity multiplier

\[
g_{\mathrm{crit}}
=\operatorname{clip}\left[
 \left(\frac{D_0}{|\sigma-1|+10^{-3}}\right)^{0.3747},
 0.3,3.0\right]
\]

is finite everywhere because of the \(10^{-3}\) regularizer and clipping.  It has a cusp at \(\sigma=1\), not a divergence.

The exponent `0.3747` is inspired by Choptuik's gravitational-collapse scaling exponent.  No derivation currently places Titan in that universality class.  It should be treated as a design exponent, not evidence of gravitational-style critical scaling.

A genuine edge singularity would require a parameterized family \(F_g\), an order parameter \(O(g,L)\), and finite-size evidence that a susceptibility or correlation length sharpens as field width \(L\) increases.  For example,

\[
\chi_L(g)=L^2\operatorname{Var}[O(g,L)]
\]

should develop a growing peak near \(g_c\), while a correlation length obeys a scaling law such as

\[
\xi(g)\sim|g-g_c|^{-\nu}
\]

over a defensible range.  Without sweeps over \(g\) and \(L\), “singularity” is a metaphor.

## 5. Fixed points, strange attractors, and random attractors

### 5.1 What would count as a deterministic strange attractor?

For frozen weights and forcing disabled or made periodic, a set \(A\) would need to be:

1. compact;
2. invariant, \(F(A)=A\);
3. attracting for an open neighborhood;
4. aperiodic and sensitive, normally with \(\lambda_{\max}>0\);
5. geometrically nontrivial, with a stable noninteger correlation dimension or related fractal evidence.

Bounded irregular output is insufficient.  Filtered stochastic noise can imitate broadband spectra, recurrence plots, and apparent fractal slopes over short ranges.

### 5.2 The more appropriate stochastic object

With cell masks, Langevin kicks, radiation, and randomized control active, Titan should be described using a random cocycle

\[
\varphi(t,\omega,z_0).
\]

The candidate object is a random attractor \(A(\omega)\) satisfying invariance under the noise shift and pullback attraction, or more modestly a stationary measure \(\mu\).  A single stationary measure can contain several metastable lobes corresponding to recognizable regimes.  Conversely, distinct initial worlds may converge to different invariant measures, implying multiple basins or broken ergodicity.

For generative music, the most useful target is probably neither one fixed point nor unconstrained chaos.  It is a **bounded metastable statistical attractor with several communicating coherent regimes**:

- enough contraction to preserve motifs and identity;
- enough expansion to create novelty and long memory;
- transitions that are state-dependent rather than pure noise;
- a stationary envelope over long runs so the organism does not drift into rails or hiss.

This can be one global random attractor with internal lobes.  It does not require forcing the system into one tonal behavior.

### 5.3 One attractor versus multiple attractors

Freeze weights and launch many worlds \(z_0^{(i)}\).  After burn-in, compare distributions of field summaries, recurrence features, controller states, and audio features.  Let \(\widehat\mu_i^T\) be the empirical occupation measure of run \(i\).

- If distances such as MMD or Wasserstein satisfy

  \[
  d(\widehat\mu_i^T,\widehat\mu_j^T)\to0
  \]

  with increasing \(T\), the evidence favors one statistical attractor.

- If the measures remain separated, show hysteresis, and do not mix under common parameters, the evidence favors multiple basins or extremely long metastability.

Finite runs cannot perfectly distinguish separate attractors from slow transitions.  Transition-rate estimates and progressively longer runs are required.

## 6. Current empirical status

### 6.1 Long pre-v7 dynamical baseline

The latest available telemetry inspected on 2026-08-04 is run `1785888816989-p29543-s42`, produced by build `27ea8a0f63e2`.  It is a 360-second resumed L06 run from global step 23,808 to 28,026.  The following values come from 422 trace samples:

| Quantity | Mean | Standard deviation | Last | Interpretation |
|---|---:|---:|---:|---|
| raw movement | 0.00516 | 0.00130 | 0.00315 | active but often below the 0.006 absolute health scale |
| `sigma` | 0.612 | 0.159 | 0.629 | subcritical under the current persistence proxy |
| criticality confidence | 0.148 | 0.127 | 0.417 | usually weak evidence |
| field entropy | 1.921 bits | 0.285 | 1.664 | diverse, but below the 3-bit bin maximum |
| field rail excess | 0.00814 | 0.00071 | 0.00783 | rails are well controlled |
| activity health | 0.748 | 0.054 | 0.690 | viable ecology |
| stagnation | 0.312 | 0.0667 | 0.374 | moderate stagnation pressure |
| temperature | 0.488 | 0.128 | 0.612 | substantial continuing external forcing |
| PI proxy | 0.00981 | 0.00944 | 0.00427 | little measured separation of fast and slow temporal order |
| reward | -0.0472 | 0.0821 | -0.0186 | controller objective is not strongly satisfied |
| gradient clip scale | 0.572 | 0.303 | 1.0 | 78.4% of sampled updates were clipped |

Additional facts:

- No sampled `sigma` value fell in `[0.95, 1.05]`.
- Only about 20.9% of samples had criticality confidence above `0.25`.
- Temperature exceeded the hot threshold `0.57` in about 24.9% of samples.
- The audio monitor measured high entropy and very high flatness, while the CA's PI proxy stayed low.

**Empirical conclusion.**  This trajectory is bounded, ecologically serviceable, and not welded to the rails.  It is not evidence of an edge-of-chaos singularity or deterministic strange attractor.  The combination of low movement persistence, low PI proxy, and appreciable temperature is more consistent with a subcritical organism kept active by adaptive forcing.  Acoustic noisiness is not proof of internal chaos.

This diagnosis does **not** imply that the weights should be discarded.  It implies that the present measurements are insufficient and that the control loop may be compensating for an overly contractive internal map.

### 6.2 v7.1 modal-gain intervention (2026-08-05)

**Empirical result.** Run `1785980433755-p21961-s42` used the real retained v7
weights, a fresh world, schema-v4 objectives, and newly initialized continuous
oscillator-gain and stereo-width heads. It generated 351 chunks (29.95 s) and
44 AdamW updates. It is an intervention test, not an attractor measurement.
It also preceded the automatic repair that prevents variants of one normalized
family from straddling train and validation. Its fixed-probe values remain
useful within-run diagnostics, but they are not strict family-held-out
generalization evidence and must not be used to claim corpus generalization.

Independent measurements of the final WAV, compared with the immediately
preceding untagged output, were:

| Audio quantity | Prior output | v7.1 modal gain | Direction |
|---|---:|---:|---|
| power from 20--80 Hz | 0.6277 | 0.3436 | less sub-bass concentration |
| 256-sample envelope CV | 0.1855 | 0.2658 | more amplitude modulation |
| mean 1 s log-band recurrence | 0.9970 | 0.5080 | stationary comb substantially broken |
| stereo correlation | -0.2406 | 0.8102 | severe antiphase removed |
| mono RMS retention | 0.6534 | 0.9200 | substantially safer mono projection |

Within the run, the last-five-trace means were:

| Training quantity | Last-five mean |
|---|---:|
| coarse source loss | 0.3351 |
| band loss | 0.6464 |
| output/target first-band ratio | 0.0258 / 0.0250 |
| fixed-probe mean spectral distance | 0.4854 |
| gradient norm | 2.3415 |
| clip scale | 1.0000 |

The mean gradient norm over the whole run was 44.47 because the new gain heads
adapted sharply during their first updates; by the end, clipping was inactive.
Fixed-probe mean spectral distance moved from 0.7390 at the first trace to
0.4969 at the last. Validation chroma did not show the same clear improvement,
so tonal learning remains unresolved.

**Interpretation.** This rejects the hypothesis that the 7.9-million-parameter
organism was simply too small. The binding defect was decoder controllability:
carrier, auxiliary, regional, and side gains were fixed. Once recurrent state
could attenuate those modes, the low-frequency stationary/antiphase attractor
lost much of its acoustic dominance without source passthrough. The remaining
64.5 Hz spectral peak and 34.36% sub-80 Hz power show that convergence is not
complete. No conclusion about edge-of-chaos status follows from this audio
improvement.

## 7. Falsifiable hypotheses

### H1 — self-organized edge

**Conjecture.** With frozen weights and normal common forcing, the controller drives the maximal conditional Lyapunov exponent toward zero from either side.

**Predictions.**

- \(\lambda_{\max}(t)\) approaches a narrow neighborhood of zero after burn-in.
- Perturbing residual gain upward causes heat/contraction responses that lower \(\lambda\).
- Perturbing gain downward causes exploration/shear responses that raise \(\lambda\).

**Falsifier.** \(\lambda\) remains substantially negative while temperature merely injects activity, or remains substantially positive despite controller intervention.

### H2 — computational rather than noisy criticality

**Conjecture.** Near the zero crossing of \(\lambda\), active information storage and local transfer are jointly elevated while spectral flatness remains source-compatible.

**Falsifier.** Entropy peaks but predictive information, transfer entropy, and source similarity do not.  That outcome is a noise maximum, not computational criticality.

### H3 — coherent metastable attractor

**Conjecture.** Frozen-weight runs from different worlds converge to one stationary measure containing several recurrent regime lobes with nonzero state-dependent transition information.

**Falsifier.** Long-run distributions remain seed-separated, or regime transitions are statistically indistinguishable from shuffled/noise-driven transitions.

### H4 — useful morphic depth

**Conjecture.** Increasing active morphic depth enlarges memory or predictive structure without systematically increasing \(\lambda\) far above zero.

**Falsifier.** Depth correlates mainly with gradient clipping, spectral noise, or predictor confidence while field transfer and source similarity do not improve.

## 8. Experiments that do not change the weights

### Phase A — frozen shadow dynamics

Add an analysis mode that loads the real weights and world but disables optimizer steps and checkpoint writes.

1. Clone the complete runtime state into reference and shadow worlds.
2. Perturb only the micro field by normalized \(\varepsilon v\), starting with \(\varepsilon\in\{10^{-7},10^{-6},10^{-5},10^{-4}\}\).
3. Feed both worlds identical RNG draws and identical controller actions.
4. Estimate \(\lambda_{\max}\) with periodic renormalization.
5. Repeat from at least 16 perturbation directions and several starting points.
6. Bootstrap confidence intervals over time blocks.

Measure field-only, memory-only, and combined norms separately.  Audio-only divergence is not enough because the oscillator renderer can magnify small frequency changes.

### Phase B — damage spreading

Create a finite perturbation in a compact spatial patch and track

\[
D_t(\epsilon)
=\frac{1}{N}\sum_i
 \mathbf 1\{|x'_{t,i}-x_{t,i}|>\epsilon\}.
\]

Define a damage branching estimate

\[
B_t=\frac{D_{t+1}+\delta}{D_t+\delta}.
\]

Unlike current `sigma`, this directly observes propagation of a perturbation.  At an edge candidate, early-time median \(B_t\) should be near one before finite-size saturation.

### Phase C — parameter and finite-size sweeps

Introduce a runtime-only residual multiplier \(g\) around the existing `0.1` CA step:

\[
\alpha(g)=0.1g,qquad g\in[0.6,1.4].
\]

Do not save altered weights.  Sweep \(g\) with fixed common seeds and measure:

- \(\lambda_{\max}\);
- damage branching;
- susceptibility of movement and field entropy;
- spatial correlation length;
- temporal correlation time;
- active information storage and transfer entropy;
- source-feature divergence;
- controller temperature required to maintain activity.

Repeat dynamics-only sweeps at spatial widths such as 32, 48, 64, and 96 using the same convolution weights.  A credible critical point should show a stable crossing and finite-size scaling, not merely a peak at one field size.

### Phase D — strange-attractor tests

In a deterministic shadow mode, first disable optimizer updates and use one of:

- no stochastic forcing;
- a recorded forcing tape replayed exactly;
- common periodic forcing analyzed as a skew-product.

Then require multiple independent diagnostics:

1. positive maximal Lyapunov exponent;
2. bounded recurrence with no short period;
3. correlation-dimension saturation as embedding dimension increases;
4. a scaling region stable across sample lengths and Theiler windows;
5. rejection of phase-randomized and amplitude-adjusted surrogate time series;
6. robustness across observables and initial conditions.

Failure of dimension saturation is especially important: high-dimensional colored noise often produces an apparent slope that keeps increasing with embedding dimension.

### Phase E — basin and metastability analysis

Run at least 16 frozen initial worlds for long burn-in and observation periods.  Cluster their occupation measures rather than their final states.  Build a regime-transition matrix and test whether transition probabilities depend on the current field/memory state beyond controller action and noise amplitude.

This determines whether Titan has:

- one mixing statistical attractor;
- one attractor with metastable lobes;
- multiple basins;
- or a noise-driven cloud with no stable regime organization.

## 9. Operational acceptance criteria

The following are proposed research criteria, not immutable constants:

| Claim | Minimum evidence |
|---|---|
| bounded ecology | no rail growth, compact-state proof, finite DSP states |
| edge candidate | common-noise \(\lambda_{\max}\) crosses zero under a nearby gain sweep |
| self-organized edge | controller returns perturbed \(\lambda\) toward zero without a hard-coded direct Lyapunov target |
| computational edge | storage and transfer information peak near the crossing, not entropy alone |
| strange attractor | positive \(\lambda\), recurrence, stable finite correlation dimension, surrogate rejection |
| one statistical attractor | occupation measures from distinct worlds converge with run length |
| metastable lobes | recurrent clusters with state-dependent, reproducible transitions |
| source-truthful audio | generated final-renderer multiscale statistics approach the target corpus without sample passthrough |

No single console scalar should be allowed to satisfy more than one row.

## 10. Relationship to source learning and musical structure

Edge-of-chaos dynamics and source likeness can reinforce each other, but they are not equivalent.

For a contractive recurrent system with \(\lambda<0\), perturbation memory decays approximately as

\[
\|\delta z_t\|\sim e^{\lambda t}\|\delta z_0\|,
\]

giving a characteristic memory scale

\[
\tau\sim-\frac1\lambda.
\]

As \(\lambda\to0^-\), memory length grows.  This is one reason task performance can improve near the ordered side of an edge.  If \(\lambda>0\), however, source-conditioned details may be overwhelmed exponentially unless input forcing synchronizes the system.

v7.1 chooses a corpus family independently of current output once per
256-chunk episode (about 21.85 s), chooses one declared variant, and advances
contiguously. Its source objective compares 1,024-sample, 4,096-sample, and
eight-chunk full-band log spectra; band energy; chroma and pitch salience;
onset, modulation, and recurrence features; mid/side balance; relative level;
and cross-chunk seams. The CA/modal system remains the only sound generator;
target samples never enter the output. Robust Charbonnier distances bound the
loss derivative with respect to each residual, while normalized projector
floors bound gradients through near-null log magnitudes.

The dynamical target should therefore be task-conditioned:

\[
\text{maximize coherent memory and transfer}
\quad\text{subject to}\quad
\lambda_{\max}\lesssim0,
\]

while minimizing final-renderer divergence from source-derived spectral and modulation statistics.  “Pleasantness” need not be hard-coded.

## 11. Recommended experimental sequence for v7

1. Train the clean v7 generation until source losses and activity health reach
   a reproducible regime; do not compare its first few hundred chunks to a
   mature earlier checkpoint.
2. Freeze weights and add common-RNG shadow execution.
3. Measure \(\lambda_{\max}\), damage spreading, and information transfer.
4. Ablate the small Gaussian kick and sparse radiation independently to test
   whether complexity is endogenous or noise-supported.
5. Compare multiple seeds and fresh worlds using the source, seam, oscillator,
   topology, source-feature, development, validation, and clipping telemetry in schema v6.
6. Retrain or reset weights only if the frozen diagnostics show that no nearby controller/gain regime produces coherent marginal stability.

The 64-channel substrate and v7 objective are now the baseline worth preserving. The next changes should improve observability and experimental identifiability before another weight-generation break.

## 12. v7.1 source-identifiability and modal-decoder corrections

The previous nearest-of-\(K\) training rule selected

\[
j^*(\theta,z)=\arg\min_j d(G_\theta(z),x_j).
\]

Because the chosen target depends on the current generator, this objective has
a self-confirming fixed point: any narrow output mode can keep selecting the
corpus mode closest to itself. v7.1 instead samples a declared corpus family
uniformly, then a variant within that family. Thus target selection is
independent of \(G_\theta(z)\), and duplicate mastered/remix files do not alter
the probability mass of a musical family. Generated Titan files have zero
training probability by construction.

Let \(S(x)\) be the 20 Hz--20 kHz log spectrum, \(B(x)\) its twelve log-band
energies, \(C(x)\) centered log chroma, \(O(x)\) positive envelope increments,
and \(M(x_{t-63:t})\) the 0.5--12 Hz modulation spectrum of 64 chunks. The
source term is now

\[
\begin{aligned}
L_{src}={}&L_S+0.45L_B+0.25L_C+0.08L_{salience}
          +0.25L_O+0.20L_M\\
         &+0.25L_R+0.35L_{env}+0.50L_{fine}
          +0.50L_{low}+L_{stereo},
\end{aligned}
\]

where \(L_R\) matches cosine-recurrence geometry at lags 8, 32, and 64 chunks.
Here \(L_{low}\) compares log first-band power ratios. In v7.2.1,
\(L_{stereo}\) denotes the weighted stereo-geometry term derived below rather
than side/mid energy alone. These terms close empirically observed loopholes:
meeting global RMS with sub-bass and meeting channelwise spectra with
degenerate stereo.
Past generated and target features are detached, so memory is bounded and the
gradient is causal through the current state. This does not claim 64-chunk
backpropagation: it supplies a long-horizon statistical teaching signal while
the exact recurrent graph remains capped at eight chunks.

**Lemma 1 — normalized log projectors have finite input gradients.** Let
\(A\) be any finite DFT projection matrix and

\[
s(x)=\frac12\log(\|Ax\|_2^2+\varepsilon),\qquad \varepsilon>0.
\]

Then

\[
\|\nabla_xs\|
\le \|A\|\frac{r}{r^2+\varepsilon}
\le\frac{\|A\|}{2\sqrt\varepsilon},
\quad r=\|Ax\|.
\]

The implemented spectral DFT is normalized by \(2/n\) and uses positive
floors; chroma and modulation projectors do the same. Therefore spectral nulls
cannot create an infinite mathematical gradient. The bound may still be large,
so measured global norms and clip scales remain required diagnostics.
\(\square\)

The earlier sample-derivative penalty
\(\|\Delta y-\Delta x\|^2\) was ill-posed for an autonomous oscillator because
source phase is unavailable. It is replaced by a robust difference of log
derivative energies. This retains a target-relative roughness constraint but
does not reward phase-counterfeiting broadband noise.

For recurrent state \(h_t\), define the smooth lower rail

\[
\ell_{a,w}(x)=a+\frac12\left[(x-a)+\sqrt{(x-a)^2+w^2}\right]
\]

and the smooth two-sided rail

\[
R_{a,b}(x)=b-\ell_{0,64}\!\left(b-\ell_{a,8}(x)\right).
\]

The modal decoder uses

\[
f_{c,t}=R_{24,4000}\!\left(
R_{32,880}(e^{\beta_0})\exp(2\ln2\tanh(W_fh_t))+\epsilon_{CA}
\right),
\]

\[
r_{i,t}=r_i^{base}\exp(0.35\tanh(W_rh_t)),\qquad
a_{i,t}=a_i^{field}(0.2+1.6\sigma(W_ah_t))
\exp[-0.10\,r_{i,t}\sigma(W_dh_t)].
\]

The family gains and learned side gain are

\[
g_c=0.05+1.15\sigma(W_ch_t),\quad
g_{aux}=0.02+0.43\sigma(W_{aux}h_t),
\]

\[
g_{scan}=0.05+0.75\sigma(W_sh_t),\qquad
g_{width}=0.05+1.15\sigma(W_wh_t).
\]

**Proposition 3 — the soft frequency rail has no dead half-space.** For finite
\(x\) and \(w>0\),

\[
\ell'_{a,w}(x)=\frac12\left(1+
\frac{x-a}{\sqrt{(x-a)^2+w^2}}\right)\in(0,1).
\]

Both derivatives composing \(R_{a,b}\) are strictly positive; therefore
\(R'_{a,b}(x)>0\) for every finite \(x\). Also \(R_{a,b}(x)<b\), and its lower
limit is the finite value \(b-\ell_{0,64}(b-a)\), approximately \(a\) when
\(b-a\gg64\). Thus frequency is bounded but an oscillator below the nominal
rail retains a nonzero learning direction. This specifically removes the
zero-Jacobian absorbing region created by a hard `clamp`. \(\square\)

**Proposition 4 — renderer controls are bounded.** Sigmoid outputs lie in
\((0,1)\), hence \(g_c\in(0.05,1.20)\),
\(g_{aux}\in(0.02,0.45)\), \(g_{scan}\in(0.05,0.80)\), and
\(g_{width}\in(0.05,1.20)\). The learned width is multiplied by the bounded
controller/host factor in \([0.5,1.25]\), so total side gain remains bounded.
Modal damping is in \((0,1]\), the regional
amplitudes are normalized before damping, active depth and field values are
bounded, and final samples pass through `tanh`. Therefore adding learned gain
control does not invalidate frozen forward boundedness. \(\square\)

Every frequency and amplitude is continuous and bounded; oscillator phases are
carried across chunks. No pitch grid, source waveform, or post-loss resonator
enters the audible path. Consequently improved musical statistics must arise
through learned recurrent/modal dynamics rather than passthrough.

Finally, AdamW state is part of the dynamical training state:

\[
(\theta_t,m_t,v_t,k_t,z_t),
\]

not merely \((\theta_t,z_t)\). v7.1 checkpoints \(m_t,v_t,k_t\) and restores
them only when their global-step stamp equals the world stamp. This removes the
repeated 32-update transient that dominated short resumed runs.

## 13. v7.2--v7.2.1 controlled morphogenesis and identifiable stereo

### 13.1 Architecture selection needs three corpus roles

Let the family-disjoint corpus split be

\[
\mathcal D=\mathcal D_{train}\;\dot\cup\;
\mathcal D_{dev}\;\dot\cup\;\mathcal D_{test}.
\]

Gradient updates use only \(\mathcal D_{train}\). Morphic depth decisions may
observe the fixed \(\mathcal D_{dev}\) probes. Final validation telemetry uses
\(\mathcal D_{test}\), and neither the optimizer nor automatic morphic policy
reads it. Mastered/remix variants are assigned at the family boundary, so a
variant cannot leak across roles.

Define the development score

\[
q_t=L_{dev,spectral}(t)+0.35L_{dev,chroma}(t).
\]

For the current rolling window, let \(\bar q_E\) and \(\bar q_L\) be the means
of its first and last quarters and

\[
I_t=\frac{\bar q_E-\bar q_L}{\max(|\bar q_E|,10^{-4})}.
\]

The gate is ready after 192 samples, retains at most 256 samples, and declares
a plateau when \(I_t\le0.01\). Growth is now possible only when

\[
G_t=B_t\land P_t\land(I_t\le0.01)\land
(d_t<d_{max})\land\neg F,
\]

where \(B_t\) is the appropriate global-step boundary, \(P_t\) is capacity or
structural pressure, \(d_{max}\) is the run cap, and \(F\) is
`--freeze-morph`.

**Conditional Proposition 5 — measured improvement blocks morphic growth.**
Once the gate is ready, if the measured early-to-late development improvement
is greater than one percent, then \(I_t>0.01\), so the conjunction defining
\(G_t\) is false. Training-song difficulty alone is therefore insufficient to
grow the stack. This is an exact property of the implemented decision rule.
It does not prove that a declared plateau is caused by insufficient capacity;
optimizer failure, a poor decoder, or probe noise can also produce a plateau.
\(\square\)

**Conditional generalization statement.** If the family split was declared
before architecture selection and no human or automatic decision uses
\(\mathcal D_{test}\), then test telemetry remains untouched by optimizer and
architecture selection. Repeated human inspection can still leak test
information into later design decisions, so this is not an unconditional
unbiasedness theorem.

### 13.2 Why side energy admitted fake width

Consider a mono signal copied with unequal gains,

\[
L=ax,\qquad R=bx,\qquad ab>0.
\]

Its centered interchannel correlation is \(\rho=1\), yet

\[
\frac{E_{side}}{E_{mid}}
=\left(\frac{a-b}{a+b}\right)^2.
\]

Changing only \(a/b\) can therefore match many target side/mid ratios while
producing no independent stereo information. This exactly explains why the
old side-energy width reported a moderate image while rendered correlation
remained approximately `+0.99`.

v7.2 measures three differentiable coordinates from centered channels:

\[
r_{MS}=\log\frac{E_{side}+\epsilon}{E_{mid}+\epsilon},\qquad
\rho=\frac{E[LR]}{\sqrt{E[L^2]E[R^2]}},\qquad
\ell=\log\frac{E[L^2]+\epsilon}{E[R^2]+\epsilon}.
\]

The v7.2 term was

\[
L_{MS}=D(r_{MS},r^*_{MS})+0.65D(\rho,\rho^*)
       +0.20D(\ell,\ell^*),
\]

with the same finite-gradient robust distance used elsewhere. The reported
correlation-aware width is

\[
w_{truth}=\operatorname{clip}_{[0,1]}\left[
w_{side}\sqrt{\frac{1-\rho}{2}}\right],\qquad
w_{side}=\sqrt{\frac{\sum(L-R)^2}{\sum(L^2+R^2)}}.
\]

**Lemma 2 — panned mono has zero truthful width.** For \(L=ax,R=bx\) with
\(ab>0\), \(\rho=1\); hence \(w_{truth}=0\) regardless of the gain imbalance
or old side-energy width. If the target has \(\rho^*<1\), the correlation term
also assigns nonzero loss to this shortcut. The level term prevents arbitrary
panning from serving as its replacement. This removes one degeneracy; it does
not prove perceptually convincing spatialization. \(\square\)

### 13.3 Pre-intervention evidence and falsifier

Across the canonical 60 s, 30 s, and 60 s continuations ending at global step
2108, active depth grew at steps 1024 and 2048. During the final interval,
fixed test mean spectral loss moved approximately `0.421 -> 0.385`, mean chroma
`0.977 -> 0.438`, clipping occurred in about four percent of sampled rows,
health ended near `0.93`, and stagnation near `0.08`. At the same time,
rendered correlation remained about `+0.992`. These observations motivate
holding L03 and correcting stereo identifiability, but continued optimization
and changing training episodes confound any causal claim about L02 or L03.

The immediate falsifier is a frozen-L03 continuation in which fixed test
spectral/chroma scores regress persistently, gradients remain healthy, and
correlation fails to move toward the fixed target distribution. That outcome
would reject the claim that simple L03 maturation plus identifiable stereo is
sufficient and would motivate a slow hierarchical recurrent state.

That falsifier occurred. Two frozen-L03 canonical continuations covered global
steps 2108--3514 and AdamW updates 264--439. Mean decoder correlation was
`0.9918` and `0.9923`, while the selected-target means were `0.9290` and
`0.8774`. Mean decoder log power ratio moved from `1.069` to `1.115` rather
than toward the near-zero target means. Gradients stayed finite and the second
run did not clip. Source losses improved on average, but the fixed validation
probe regressed in the second run. This rejects more blind L03 continuation as
the strongest next experiment and provides no evidence for activating L04.

These validation results have now informed a human design decision. They must
therefore be treated as development evidence, not as untouched evidence for a
future final claim. A future confirmatory evaluation requires newly locked
families or an external corpus that is not inspected during development.

### 13.4 The hard-pan trap and the spatial map correction

The former global pan coordinate was

\[
p=\operatorname{clip}(\tanh z,-1/2,1/2).
\]

Whenever \(|\tanh z|>1/2\), \(\partial p/\partial z=0\). Equal-power gains were

\[
g_L^2=\tfrac12-\tfrac12p,\qquad
g_R^2=\tfrac12+\tfrac12p.
\]

At the observed negative endpoint, the resulting log power ratio is

\[
\ell=\log\frac{3/4}{1/4}=\log 3=1.0986,
\]

which quantitatively accounts for the measured `1.069` and `1.115`. The head
could not learn away from that shortcut because the clamp erased its gradient.
v7.2.1 instead uses

\[
p=0.25\tanh z,\qquad
\frac{\partial p}{\partial z}=0.25\operatorname{sech}^2z>0
\]

for every finite \(z\). Its maximum pan-only log power imbalance is
\(\log(5/3)=0.511\), and the derivative has no finite hard-dead region. The
global coordinate is deliberately residual: spatial structure should come
from the regional field, not a master channel-gain shortcut.

For region row \(r\) and column \(c\) in the 4x4 readout, v7.2.1 defines

\[
x_{r,c}=2\frac{c}{3}-1,\qquad
g_L(x)=\sqrt{\frac{1-x}{2}},\quad
g_R(x)=\sqrt{\frac{1+x}{2}}.
\]

Thus \(g_L^2+g_R^2=1\), every row has the same left-to-right map, and a
horizontally symmetric activity pattern has balanced aggregate pan energy.
The previous flat-index coordinate \(x_i=2i/15-1\) accidentally mapped row
number primarily onto stereo position. The corrected map is a declared
projection from the toroidal CA to a listening axis; because any line
projection introduces a seam into a torus, it is not claimed to be the unique
or topology-preserving map.

Finally, v7.2.1 reweights the source-conditioned stereo objective to

\[
L_{stereo}=0.25D(r_{MS},r^*_{MS})+0.85D(\rho,\rho^*)
             +0.45D(\ell,\ell^*).
\]

The per-target correlation term does not impose artificial width: mono-like
targets still request \(\rho^*\approx1\). It only makes the measured target
geometry harder to ignore. The three unweighted distances and \(p\) are logged
separately in telemetry schema v6. Whether this is sufficient to escape the
near-collinear attractor remains an empirical question; no proof of audible
stereo emergence is claimed.

### 13.5 First v7.2.1 intervention result

A 30-second canonical frozen-L03 continuation covered global steps 3514--3865
and AdamW updates 440--484. The immediate geometric prediction passed. The
rendered full-file log power ratio changed from `1.09994` to `0.50722`, just
below the derived $\log(5/3)=0.51083$ limit, and the old side/mid measure fell
from `0.2753` to `0.1405`. This confirms that the former apparent width was
largely the hard-pan shortcut.

The stronger claim did not pass: rendered correlation changed only from
`0.992354` to `0.992347`. Truthful width consequently fell from `0.0170` to
`0.0087`; the system became more honestly center-focused rather than genuinely
stereo. The combined stereo loss decreased across the short run, but target
episodes vary, so that trend is not a controlled generalization result. Four
of 36 sampled rows reported clipping during one contiguous target episode,
including one large pre-clip norm; later rows recovered to finite, unclipped
updates. This should be watched in the next run rather than used to justify
another immediate coefficient change.

The present result supports keeping the tested v7.2.1 binary and L03 weights as
a stable experimental checkpoint. It does not support L04 or a larger model as
the next intervention. A later branch should test a field-derived
antisymmetric side state or a slower hierarchical recurrent state against this
checkpoint, with locked evaluation data and without reintroducing an external
stereo effect.

## 14. Research references

- Chris G. Langton, “Computation at the edge of chaos: Phase transitions and emergent computation,” *Physica D* 42 (1990), 12–37. [DOI](https://doi.org/10.1016/0167-2789(90)90064-V)
- Joschka Boedecker, Oliver Obst, Joseph T. Lizier, N. Michael Mayer, and Minoru Asada, “Information processing in echo state networks at the edge of chaos,” *Theory in Biosciences* 131 (2012), 205–213. [DOI](https://doi.org/10.1007/s12064-011-0146-8)
- Joseph T. Lizier, Mikhail Prokopenko, and Albert Y. Zomaya, “Local information transfer as a spatiotemporal filter for complex systems.” [arXiv](https://arxiv.org/abs/0809.3275)
- Giancarlo Benettin, Luigi Galgani, Antonio Giorgilli, and Jean-Marie Strelcyn, “Lyapunov characteristic exponents for smooth dynamical systems and for Hamiltonian systems; a method for computing all of them,” *Meccanica* 15 (1980). [DOI](https://doi.org/10.1007/BF02128236)
- Peter Grassberger and Itamar Procaccia, “Characterization of Strange Attractors,” *Physical Review Letters* 50 (1983), 346–349. [DOI](https://doi.org/10.1103/PhysRevLett.50.346)
- James Theiler et al., “Testing for nonlinearity in time series: the method of surrogate data,” *Physica D* 58 (1992), 77–94. [Record](https://ndlsearch.ndl.go.jp/en/books/R100000136-I1571417125880748800)
- Matthew W. Choptuik, “Universality and scaling in gravitational collapse of a massless scalar field,” *Physical Review Letters* 70 (1993), 9–12. [DOI](https://doi.org/10.1103/PhysRevLett.70.9)
- S. B. Kuksin and A. R. Shirikyan, “On Random Attractors for Mixing Type Systems,” *Functional Analysis and Its Applications* 38 (2004), 28–37. [DOI](https://doi.org/10.4213/faa94)
