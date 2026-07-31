# Titan Audio Ecosystem: Theory, Mathematical Audit, and Research Roadmap

**Originally drafted:** 2026-05-16

**Rewritten after implementation review:** 2026-07-31

**Implementation reviewed:** `src/main.rs` at commit `8878981`

## Technical summary

Titan is best understood as a **controlled, stochastic, multiscale dynamical instrument** with three coupled learning loops:

1. gradient learning shapes the neural cellular automata, recurrent memory, and audio generator;
2. homeostatic control attempts to keep the evolving field alive, mobile, and bounded;
3. model-based control chooses interventions intended to improve future viability.

The central research hypothesis should not be that “chaos is good,” or that one scalar proves the system is at the edge of chaos. A more defensible and useful hypothesis is:

> Musically productive behavior occupies a metastable, controllable region in which perturbations neither vanish immediately nor grow without bound, several distinct futures remain reachable, and recurring structures survive long enough to become motifs.

This reframing matters. It replaces attractive metaphors with quantities that can be measured, perturbed, falsified, and optimized. The most promising next work is consequently not simply a larger network. It is better causal measurement of the dynamics, stronger actuator authority over those dynamics, parallel experimental worlds, temporally aware audio objectives, and calibrated uncertainty in the learned world model.

This document deliberately emphasizes theory, limitations, and future designs rather than repeating the operational README.

## 1. Scope and notation

Let

- \(x_t^\mu\) and \(x_t^M\) be the micro and macro cellular fields;
- \(h_t\) be recurrent memory;
- \(e_t\) be episodic memory readout;
- \(a_t\) be an ecological or synthesis intervention;
- \(y_t\) be the generated stereo audio chunk;
- \(o_t\) be the measured observation vector;
- \(r_t\) be an ecological reward;
- \(\theta\) be gradient-trained model parameters;
- \(\psi\) be the separately trained transition-model parameters.

The implemented observation vector contains 12 audio/ecology proxies. The controller state is a different 48-dimensional compressed state. The recurrent neural memory is 768-dimensional. These spaces should not be conflated: an observation, a planning state, and a learned hidden state answer different questions.

The following labels are used throughout:

- **Implemented:** a claim directly supported by the reviewed code.
- **Interpretation:** a plausible account of what the implemented mechanism does.
- **Hypothesis:** a prediction that still requires an experiment.

## 2. A compact mathematical model

### 2.1 Coupled stochastic neural fields

A useful abstraction of the micro update is

\[
x_{t+1}^{\mu}
= \operatorname{clip}\!\left[
(1-\eta_t)x_t^{\mu}
+\eta_t\left(x_t^{\mu}
+\alpha\,m_t\odot A\odot
F_{\theta}(x_t^{\mu},x_{t+1}^{M},h_t)
+b(x_t^{\mu})\right)
\right],
\]

where \(m_t\) is a stochastic update mask, \(A\) is a fixed anisotropic interference mask, \(b\) is a local anti-rail field, and \(\eta_t\) depends on macro activity. The macro field evolves by a related, slower stochastic rule.

This is a non-autonomous random dynamical system: its transition rule depends on learned parameters, recurrent state, random masks, and controller actions. Calling it a strange attractor is therefore a hypothesis about its long-run invariant behavior, not a fact implied by using a cellular automaton.

The toroidal grid removes boundary discontinuities and gives translation symmetry, but it also assumes periodic space. Any claimed spatial law is conditional on that topology.

### 2.2 Memory is finite, hierarchical, and lossy

The effective recurrent update can be written

\[
h_{t+1}=\operatorname{GRU}_{\theta}
\left(h_t,\;\operatorname{pool}(x_{t+1}^{\mu})\oplus e_t\right).
\]

The current system has a 768-value recurrent state and a small bank of episodic snapshots. This gives long-lived context, but not infinite context. Information survives only when compressed into \(h_t\), retained in the finite episodic bank, or embodied in the persistent CA fields.

That last category is important: the field itself is external recurrent memory. The relevant memory capacity is not just the GRU dimension; it is the joint state \((x_t^\mu,x_t^M,h_t,e_t)\). Measuring how long recoverable information survives in that joint state would be more meaningful than assigning an informal context length.

### 2.3 The audio nonlinearity is a conditional Fourier waveshaper

The layer currently called KAN is more precisely modeled as

\[
g(x\mid h)=\sum_{k=1}^{K}
\frac{w_k+0.15\tanh(d_k(h))}{\sqrt{k}}\sin(kx).
\]

This is a useful, memory-conditioned Fourier basis, but it is not a general Kolmogorov-Arnold Network in the usual architectural sense. The precise name matters because it makes the real research questions visible:

- Which partials can exceed Nyquist after modulation and folding?
- Is the basis expressive because it is harmonic, or merely because it is large?
- Would paired sine/cosine terms, learnable phase, splines, or rational bases improve conditioning?
- Does oversampling the nonlinear stage improve perceived quality enough to justify its cost?

Without band limiting or oversampling, high-order nonlinear terms can produce aliasing. Some aliasing may be aesthetically useful, but it should be a controlled texture, not an unmeasured side effect.

### 2.4 Training is multi-objective rather than a single biological drive

The differentiable objective is approximately

\[
\mathcal L_t =
w_m\mathcal L_{\text{mimic}}
+w_v\mathcal L_{\text{variance}}
+w_d\mathcal L_{\text{motion}}
+w_r\mathcal L_{\text{roughness}}
+w_g\mathcal L_{\text{scale}}
+w_s\mathcal L_{\text{self-model}}
+w_e\mathcal L_{\text{empowerment}}
+w_c\mathcal L_{\text{coupling}}
+w_n\mathcal L_{\text{novelty}}
+\mathcal R.
\]

The learned arbiter changes several weights from recent progress and uncertainty. This creates a non-stationary optimization problem: the target landscape changes while the model moves through it. That can support open-ended behavior, but it can also produce cycling objectives or reward hacking. Weight trajectories therefore need to be logged and evaluated as state variables, not treated as invisible optimizer details.

The min-of-\(K\) spectral target objective is a multiple-choice loss:

\[
\mathcal L_{\text{mimic}}
=\min_{j\in\{1,\ldots,K\}}
d_{\text{spectral}}(y_t,\tilde y_j).
\]

It avoids averaging incompatible target modes, but it encourages mode seeking and ignores much of phase and longer temporal organization. The WAV corpus acts as a distribution of local attractors, not as a conventional sequence target.

### 2.5 Viability is currently a weakest-link score

The world model uses a geometric aggregation resembling

\[
C_{\mathrm{viable}}
=\left(
C_{\mathrm{structured}}
C_{\mathrm{recoverable}}
C_{\mathrm{multiscale}}
C_{\mathrm{continuity}}
C_{\mathrm{critical}}
\right)^{1/5}.
\]

Here juxtaposition denotes multiplication, with small floors applied in code. The geometric mean is appropriate when a catastrophic weakness in any one factor should dominate. The floors, however, prevent a genuine zero and can conceal dead dimensions. Several factors are also computed from overlapping proxies, so this is not five independent pieces of evidence.

A future viability model should report both the vector of factors and its aggregate. The scalar is useful for control; the vector is necessary for diagnosis.

## 3. Mathematical audit: what the measurements establish

### 3.1 The reported sigma is a persistence index, not yet a branching ratio

The criticality estimator combines mean-centered autocorrelations of scalar movement at lags 1, 2, and 4. Schematically,

\[
\hat\sigma_t
=0.55\rho_1+0.28\sqrt{\rho_2}+0.17\rho_4^{1/4}.
\]

This is a reasonable **multi-lag persistence index**. It does not by itself estimate a branching process, a Jacobian spectral radius, a Lyapunov exponent, or susceptibility to intervention. A high value could arise from a smooth periodic process, while a genuinely excitable spatial process could have low scalar movement autocorrelation.

The name should remain provisional until calibrated against perturbation experiments. The most direct replacement is an intervention-derived gain:

\[
G(\tau)=
\mathbb E\left[
\frac{\|x_{t+\tau}'-x_{t+\tau}\|_2}
{\|x_t'-x_t\|_2+\epsilon}
\right],
\]

where paired worlds share random masks and differ only by a small perturbation at time \(t\). Productive metastability should show finite persistence over a range of \(\tau\): neither immediate contraction nor runaway divergence.

A complementary local measure is the dominant singular value of the transition Jacobian,

\[
s_{\max}\!\left(\frac{\partial F}{\partial x}\right),
\]

estimated periodically with Jacobian-vector products and power iteration rather than constructing the full Jacobian.

### 3.2 The current RG score is evidence at two resolutions, not an RG flow

The implementation compares regional structure after a 4-by-4 to 2-by-2 pooling step and includes temporal support. This is a useful multiscale consistency check. A renormalization-group claim would require several scales, an explicit coarse-graining operator, and evidence that statistics flow toward or remain near a fixed distribution.

A stronger experiment would compute, for scales \(\ell\in\{1,2,4,8,16\}\), quantities such as

\[
S_2(\ell)=\mathbb E\|x(r+\ell)-x(r)\|_2^2
\quad\text{and test}\quad
S_2(\ell)\propto \ell^{\zeta_2}.
\]

Stable exponents over a nontrivial range would be better evidence of scale organization than similarity at one pooling boundary. A learned coarse-grainer could be compared against average pooling, but the learned version must be constrained against simply erasing inconvenient detail.

### 3.3 The critical manifold is a designed target tube

Current critical health is an RBF-like distance to eight manually drifting target coordinates:

\[
H_{\mathrm{crit}}(s,t)
=\exp\left[-\beta
\|D^{-1}(s-\mu(t))\|_2^2\right].
\]

This is a sensible control prior, but the sinusoidally moving center \(\mu(t)\) is a composition rule written by the designer. It is not a discovered physical critical manifold.

The promising alternative is to learn a **viability boundary** from outcomes: states from which the organism can still reach several healthy futures under bounded action. This can be approximated by a classifier or value function trained on recovery success, then regularized to remain smooth and uncertainty-aware.

### 3.4 Potential control is Langevin-inspired, not full gradient flow

The controller defines a potential with amplitude wells, rail barriers, coupling and energy targets, a flatline ridge, and a sigma well. Only some state coordinates have a known direct gradient actuator; other terms mainly alter temperature, shear, kicks, or learning rate.

Therefore the closed loop is not literally

\[
\dot s=-\nabla V(s)+\sqrt{2T}\,\xi,
\]

even though that equation is a useful analogy. A more exact controller would learn a local action Jacobian

\[
B_t\approx\frac{\partial s_{t+1}}{\partial a_t}
\]

from small probes, then choose a bounded action

\[
a_t^*=\arg\min_a
\nabla V(s_t)^T B_ta
+\lambda\|a\|_2^2
+\kappa\,\mathrm{Risk}(a).
\]

That would distinguish an undesirable state from an undesirable state the available actuators can actually change.

### 3.5 Novelty is vulnerable to quantization noise

The planning state is hashed after quantizing selected dimensions into bins. “Unseen hash” is computationally cheap, but in a moderately high-dimensional state even small noise can create a new combination on nearly every step. Thus `state novel: 1.00` can coexist with visibly repetitive dynamics.

Prefer a continuous novelty estimate such as

\[
N(s)=\operatorname{median}_{j\in k\mathrm{NN}(s)}
\|W(s-s_j)\|_2,
\]

with normalized dimensions, or a density model with uncertainty. An adaptive random projection plus approximate nearest-neighbor index should be inexpensive at the current state size. Novelty must also be separated into:

- state novelty: a new measured configuration;
- transition novelty: a new change from one state to another;
- acoustic novelty: a perceptibly different result;
- useful novelty: novelty that remains viable or yields a reusable motif.

## 4. Architectural opportunities

The table is ordered by expected scientific and musical value, not implementation ease.

| Priority | Proposal | Why it is promising | Principal risk |
|---|---|---|---|
| P0 | Paired-world perturbation measurement | Turns “criticality” into a causal, calibrated property | Extra GPU memory and deterministic random-stream handling |
| P0 | Directly condition CA rules on controller actions | Gives the controller authority over field dynamics, not only downstream synthesis | The controller may learn shortcuts that flatten the ecology |
| P0 | Continuous transition/acoustic novelty | Prevents quantization noise from masquerading as exploration | Distance metrics can still reward meaningless noise |
| P1 | Bootstrap world-model ensemble with randomized priors | Makes disagreement a more credible epistemic uncertainty signal | More training cost and calibration work |
| P1 | Multi-step, distributional world model | Penalizes compounding error and models several possible futures | Can blur genuinely multimodal outcomes unless mixture-based |
| P1 | Gated, normalized morphic growth | Activates new capacity smoothly and makes depth reversible | Added gates can remain closed without a growth-specific objective |
| P1 | Multi-resolution temporal audio loss | Rewards rhythm, modulation, stereo motion, and phrase identity | More FFT work and risk of overconstraining timbre |
| P1 | Parallel populations of worlds | Improves GPU occupancy, counterfactual evaluation, and diversity | Changes training semantics if gradients are averaged carelessly |
| P2 | Depthwise spatial CA perception plus pointwise channel mixing | Separates spatial operators from channel computation; cheaper and interpretable | May reduce useful arbitrary cross-channel spatial filters |
| P2 | Differentiable or surrogate post-DSP path | Aligns the trained audio objective with what the ecology and listener hear | Long effect chains can make gradients unstable |
| P2 | Slow state-space memory beside the GRU | Adds explicit long timescales without quadratic dense depth | Redundant memory can be ignored by optimization |

### 4.1 Criticality-conditioned CA rules

At present, ecological actions have strong downstream effects, but the CA transition law itself should expose a small, structured action interface. A FiLM-style rule is sufficient:

\[
F_{\theta}(x;a)=
\gamma(a)\odot F_{\theta}(x)+\beta(a),
\]

with bounded \(\gamma\) and \(\beta\), separate micro/macro channels, and a penalty for excessive intervention. This lets EXPLORE, CONTRACT, TURBULENCE, and RECALL alter growth rules rather than merely decorating their acoustic projection.

The action interface should be low rank. Giving the planner unrestricted control of all CA channels would likely bypass emergence and turn the world model into a conventional synthesizer controller.

### 4.2 A causal, uncertainty-aware world model

The existing ensemble consists of small action-conditioned delta predictors. That is a good computational starting point, but identically structured members trained on largely shared replay can agree confidently outside their data.

A stronger ensemble should use:

- independent bootstrap masks over replay;
- a fixed randomized prior function per member;
- regime-balanced replay rather than only transition magnitude;
- heteroscedastic outputs or a mixture density over next states;
- losses at horizons 1, 2, 4, and 8;
- explicit calibration plots for predicted uncertainty versus realized error.

Planner value should be pessimistic when uncertainty is high:

\[
Q_{\mathrm{safe}}(s,a)
=\mathbb E[Q(s',a')]
-\kappa\sqrt{\operatorname{Var}[Q(s',a')]},
\]

except during deliberate information-gathering actions, where bounded uncertainty reduction receives an exploration bonus.

### 4.3 Gated morphogenesis rather than abrupt depth activation

The morphic stack holds up to 12 dense residual blocks of two 768-by-768 transforms. These blocks dominate parameter count; all are stored even when only a shallow prefix is active. Increasing this depth is therefore a costly and indirect way to add useful dynamical capacity.

Use pre-normalized gated residual blocks:

\[
z_{\ell+1}=z_\ell+\alpha_\ell
F_\ell(\operatorname{LN}(z_\ell)),
\]

where a newly activated block starts with \(\alpha_\ell=0\) and grows only when held-out predictive or audio utility improves. Pruning should close the gate before deleting a block from execution. A diversity loss on block outputs can discourage several layers from learning the same residual.

Growth should answer a measurable question: does the extra block expand reachable viable futures, reduce held-out spectral/temporal loss, or improve recovery? If none improves, depth is not capacity in a useful sense.

### 4.4 Explicit multirate fields

The micro/macro distinction is conceptually strong. It can be made more rigorous by assigning explicit clocks:

\[
x_{t+1}^{\mu}=F_\mu(x_t^\mu,Ux_t^M,h_t),
\]

\[
x_{t+q}^{M}=F_M(x_t^M,D\{x_t^\mu,\ldots,x_{t+q-1}^\mu\}),
\]

where \(D\) and \(U\) are defined down/up-sampling operators and \(q>1\). This turns the macro layer into a genuine slow field rather than a similarly sized companion updated less often. Three clocks—cellular, motif, and phrase—may be more musically productive than simply enlarging one state.

### 4.5 Long-timescale memory

A slow linear state-space branch can complement the GRU:

\[
z_{t+1}=\bar A z_t+\bar B u_t,\qquad
h_t^{\mathrm{slow}}=Cz_t,
\]

with stable parameterization of \(\bar A\). It can represent exponential traces across hundreds or thousands of chunks while the GRU handles fast nonlinear context. Episodic memory should be written on events—novel viable state, motif birth, regime transition, recovery—not merely on a fixed clock. Replacement should maximize memory coverage rather than use FIFO alone.

## 5. Audio-learning hypotheses

### 5.1 Temporal structure belongs in the objective

Local log-magnitude spectra reward timbral similarity but cannot distinguish many musically different sequences. Add a small hierarchy of differentiable terms:

1. multi-resolution STFT magnitude at short, medium, and long windows;
2. onset-strength or energy-envelope correlation;
3. modulation spectrum over several chunks;
4. stereo coherence and mid/side motion;
5. optional embedding distance from a frozen audio representation.

The embedding term should remain a weak semantic guide. A pretrained embedding can impose its dataset’s genre biases and may collapse unusual but compelling material into “error.”

For min-of-\(K\) targets, choose the target using a combined coarse temporal/timbral distance, then use that same target for all loss resolutions. Independently minimizing every resolution can assemble an impossible target from different examples.

### 5.2 Diversity needs quality-conditioned memory

Motifs should not be admitted because they are merely distant. A candidate \(m\) should maximize a quality-diversity acquisition score such as

\[
A(m)=q(m)
+\lambda d(m,\mathcal M)
+\eta u(m)
-\rho c(m),
\]

where \(q\) is acoustic/ecological quality, \(d\) is distance from stored motifs, \(u\) is future option value, and \(c\) is collapse risk. The 128-slot capacity is a ceiling, not a target occupancy. A full library of near-duplicates is worse than 40 well-separated, reusable motifs.

Use a two-stage comparison: a cheap spectral/structural descriptor for screening, then a richer temporal descriptor only near an admission or eviction boundary. This preserves throughput.

### 5.3 Human preference is part of the ground truth

Entropy, flatness, novelty, and prediction horizon are diagnostics, not a definition of musical value. The project needs periodic blinded A/B listening tests. Even a small set of pairwise preferences can train a low-weight reward model or reveal when a proxy is being gamed.

Keep this feedback separate from the core ecology at first. If incorporated too early or too strongly, the ecosystem may optimize a narrow listener model and lose the open-ended behavior being studied.

### 5.4 Data duration is not determined by parameter count alone

Because the target loss samples short windows, the number of training examples is nominally large even from one WAV; adjacent windows, however, are highly correlated. Doubling the neural parameter count does not imply exactly doubling WAV duration.

A practical research corpus should prioritize coverage:

- at least 30–60 minutes of varied, clean source material for early experiments;
- several hours if the aim is broad style and texture coverage;
- file-level train/validation separation, not adjacent-window separation;
- a held-out set containing wholly unseen recordings and source sessions;
- balanced sampling so one long file does not dominate the learned attractor distribution.

If validation diversity stops improving while training mimic loss continues improving, more heterogeneous data is more valuable than more parameters. If both training and validation losses plateau high, architecture or optimization is the likely bottleneck.

## 6. Scaling strategy for an 11 GB GPU

The first useful scaling axis is **parallel worlds**, not a single model twice as wide.

At batch size one, the current dense recurrent path and many small kernels leave GPU parallelism unused. Running 4–8 independent worlds with shared weights provides:

- better matrix utilization;
- paired counterfactuals for causal measurements;
- bootstrap data for the world model;
- more diverse motif candidates per optimizer update;
- more reliable population statistics.

Worlds should have independent field states, episodic memories, random streams, and controllers. Gradients may be averaged, but ecological rewards and motif stores should initially remain per-world so one dominant trajectory cannot homogenize the population.

After measurement and batching are sound, scale in this order:

1. increase CA hidden width or use more expressive perception kernels;
2. increase spatial resolution if multiscale tests show the current grid truncates structure;
3. add a slow memory branch;
4. increase recurrent width only if memory probes demonstrate a capacity bottleneck;
5. increase morphic depth only when gated growth passes a held-out utility test.

The morphic dense stack scales approximately as \(O(Ld^2)\). Doubling its 768-wide state roughly quadruples those weights and much of its compute. By contrast, increasing grid resolution raises CA activation memory and convolution work approximately with area. These are different costs and should answer different empirical bottlenecks.

Mixed precision should not be assumed to help equally on every GPU generation. Benchmark complete chunk throughput and numerical stability rather than relying on nominal FLOP claims.

## 7. A falsifiable experimental program

Every experiment should use matched target files, matched starting checkpoints or fresh seeds, and at least five seeds when variance is material. Report distributions, not only the best render.

### Experiment A: is sigma measuring causal criticality?

**Intervention:** clone a world, share future random masks, perturb 0.1–1% of cells at several amplitudes, and measure separation over 1, 2, 4, 8, 16, and 32 chunks.

**Compare:** current \(\hat\sigma\), Jacobian gain estimate, and measured perturbation gain.

**Pass condition:** the replacement measure predicts collapse, recovery time, and useful motif yield better than movement autocorrelation on held-out runs.

**Falsifier:** no relationship between any criticality measure and recovery or musical structure after controlling for movement amplitude.

### Experiment B: does direct CA action conditioning improve control?

**Intervention:** add bounded low-rank FiLM action conditioning to the CA update.

**Compare:** time to exit collapse, post-recovery viable complexity, action energy, and motif diversity.

**Pass condition:** faster recovery without lower long-run diversity or an increase in persistent controller saturation.

**Falsifier:** the conditioned model recovers only by forcing stereotyped states.

### Experiment C: is novelty real?

**Intervention:** replace hash novelty with normalized kNN state and transition novelty; retain the old metric for logging.

**Compare:** correlation with blinded human judgments of difference, motif eviction rate, and repeated acoustic fingerprints.

**Pass condition:** fewer false-new states and higher quality-diversity coverage at the same motif capacity.

### Experiment D: does the world-model ensemble know when it is wrong?

**Intervention:** bootstrap masks, randomized priors, and multi-step prediction.

**Compare:** reliability curves of ensemble disagreement versus realized prediction error, split by regime and action.

**Pass condition:** uncertainty ranks future errors and improves pessimistic planner outcomes.

**Falsifier:** disagreement remains low in novel or collapsing states, or planner performance does not improve.

### Experiment E: does temporal loss produce better music?

**Intervention:** add onset-envelope and modulation-spectrum losses at low weight.

**Compare:** spectral mimic, motif recurrence at 2–16 second lags, human pairwise preference, and ecology health.

**Pass condition:** stronger perceived phrase coherence without loss of timbral diversity or throughput beyond an agreed budget.

### Experiment F: should scale mean more worlds or one larger world?

Use the same memory/time budget for:

- one larger CA;
- four current-size worlds;
- eight smaller worlds.

Compare sample efficiency, steps per second, perturbation coverage, motif archive coverage, and listening preference. This experiment should precede a major size increase.

## 8. Metrics that should govern future tuning

No single scalar should determine tuning. Use a small scorecard:

| Dimension | Primary metric | Guardrail |
|---|---|---|
| Dynamical life | perturbation gain curve and recovery time | bounded state/amplitude |
| Controllability | reachable healthy successors under bounded actions | action saturation rate |
| Diversity | quality-diversity archive coverage | duplicate acoustic fingerprints |
| Prediction | multi-step error and uncertainty calibration | regime-balanced evaluation |
| Musical structure | recurrence/modulation measures | human pairwise preference |
| Training | held-out temporal/spectral loss | throughput and VRAM |
| Open-endedness | rate of new useful motifs over time | no monotonic collapse in quality |

Important derived metrics include:

- **recovery half-life:** chunks needed to remove half the collapse pressure;
- **viability volume:** fraction of sampled bounded actions leading to healthy successors;
- **effective motif count:** inverse concentration, \(1/\sum_i p_i^2\), not raw slots occupied;
- **novelty precision:** fraction of detector-labeled novel states judged acoustically or dynamically distinct;
- **model calibration error:** gap between predicted uncertainty quantiles and observed errors;
- **intervention efficiency:** viability improvement per unit action energy.

## 9. Specific inconsistencies and research risks

1. **Critical-health calibration remains provisional.** V8 now uses the same finite band centered near \(\sigma=1\) in adaptive dynamics and ecological reward, removing the older monotone reward inconsistency. The underlying \(\sigma\) measurement is still a persistence proxy until perturbation experiments calibrate it.

2. **Pre-DSP training and post-DSP evaluation can disagree.** The recursive ecology hears the processed audio while much of the differentiable loss trains the earlier signal. Either make important DSP stages differentiable, use a differentiable surrogate, or log the discrepancy as a domain gap.

3. **A horizon of one may mean trivial predictability.** Frozen behavior is easy to predict. Prediction horizon is valuable only when conditioned on nonzero activity, perturbation response, and alternative reachable futures.

4. **Raw confidence can be misleading.** Low model error in a collapsed regime does not imply useful knowledge. Effective confidence should continue to be gated by health, and ideally by epistemic calibration.

5. **Proxy multiplication can double-count evidence.** Movement, temporal continuity, recoverability, sigma, and critical health share inputs. Correlated proxies make a geometric mean appear more corroborated than it is.

6. **The controller and learner can chase one another.** Fast intervention changes the data distribution seen by gradient learning; gradient learning changes controller response. Separate timescales, freeze one loop during selected measurements, and log causal interventions.

7. **Open-ended claims require a non-saturating test.** A fixed 128-slot motif archive can fill even when invention stops. Track replacement quality, archive coverage, and useful novelty rate over long horizons.

## 10. Recommended implementation sequence

The highest-value sequence is:

1. calibrate the new viable-shell, radial/tangential, and modal-rotation measurements on real runs;
2. implement deterministic paired-world perturbation probes and rename sigma in telemetry until calibrated;
3. add healthy coarse-field anchors for guided re-entry after structured recovery fails;
4. replace binary hash novelty with continuous state and transition novelty;
5. add bootstrap/random-prior ensemble training and uncertainty calibration reports;
6. expose the actual continuous core-control magnitudes to the transition model;
7. add parallel training worlds, initially 4, with per-world ecology;
8. add low-cost temporal audio losses and blinded listening evaluation;
9. convert morphic blocks to gated pre-normalized growth;
10. only then run controlled larger-model experiments.

This ordering prioritizes observability before optimization and actuator authority before planning sophistication. A larger system whose criticality, novelty, and uncertainty are not trustworthy will produce more expensive ambiguity, not necessarily better audio.

## 11. Strong claims the project should try to falsify

The following are productive research claims precisely because experiments can prove them wrong:

1. Productive audio occurs in a band of finite perturbation persistence, not at maximal movement or maximal unpredictability.
2. The size of the reachable viable future set predicts motif yield better than scalar entropy.
3. Multirate micro/macro coupling produces longer musical organization than a single field with equal parameter count.
4. Quality-conditioned diversity preserves more reusable motifs than novelty-only admission.
5. Parallel worlds improve both GPU utilization and scientific validity more than equivalent parameter growth.
6. A calibrated uncertainty-aware planner outperforms reactive recovery without reducing diversity.
7. Temporal audio objectives improve perceived structure even when local spectral mimic scores change little.

If these claims fail, that is useful. Titan should evolve around reproducible findings, not protect its terminology from measurement.

## 12. V8 confinement implementation

The first confinement-control tranche implements a measurable approximation of the “suspended spinning yarn ball” model:

- a normalized shell learned only from sustained healthy reactor states;
- signed radial velocity and orthogonal tangential velocity in the 48-dimensional reactor state;
- complex low-frequency modes of the 4×4 micro and macro regional fields;
- modal rotation and micro/macro phase-lock measurements;
- staged `EDGE`, `ROTATE`, `GUIDED`, sparse `RESEED`, and `COOLDOWN` recovery;
- bounded CA update-drive and macro-coupling controls;
- approximately norm-preserving skew rotations across CA channel pairs;
- spatially coherent micro pulses derived from the persistent multiscale shear field;
- one recovery-neurogenesis event per recovery episode, re-armed only after sustained health;
- consistent band-shaped critical-health reward.

This is intentionally an interpretable controller before it is a learned one. Its telemetry should establish whether coherent tangential actuation increases movement, RG temporal activity, and recovery probability without pushing radius outward or homogenizing musical output.

Two major pieces remain experimental. First, a bank of compressed healthy field anchors could restore coarse organization without resetting recurrent memory. Second, active probes could learn a local actuator-response matrix and solve a small constrained control problem rather than following fixed phase amplitudes. Neither should be added until the V8 measurements show which actuator directions are actually useful.

## Closing perspective

The most original part of Titan is not any isolated component. It is the attempt to make a sound generator, a persistent spatial world, a memory system, and an ecological controller co-adapt over a long run. The project becomes scientifically stronger when its metaphors are treated as hypotheses and its gauges are calibrated by interventions.

The target is not “maximum chaos.” It is **controllable metastability with memory**: enough sensitivity to create alternatives, enough contraction to preserve identity, enough causal control to recover, and enough temporal memory for novelty to become form.
