# Decoder Equations

Let an edge `e=(s,d)` have event-time lists `A_e={a_i}` and `B_e={b_j}`.

## One-to-one timing kernel

For a temporal scale `sigma`, candidate event matches receive

`k_sigma(Δt) = exp(-0.5 * (Δt / sigma)^2)`

for `|Δt| <= 4 sigma`; events are matched one-to-one greedily by nearest distance.  Unmatched events remain in the union denominator.

`J_sigma(A_e,B_e) = matched_kernel_mass / (|A_e| + |B_e| - number_of_matches)`

The four required scales are

`Sigma = {0.50, 1.00, 2.00, 4.00} ms`

with weights

`w = {0.34, 0.29, 0.23, 0.14}`.

For each edge:

`M_e = Σ_sigma w_sigma J_sigma(A_e,B_e)`.

The episode multiscale score is the event-count-weighted average of `M_e` across recurrent edges.

## Inter-event rhythm

For an event list `A=[a_1,...,a_n]`, define

`ISI(A) = [a_2-a_1, a_3-a_2, ..., a_n-a_(n-1)]`.

The ISI sequences are compared by the same one-to-one Gaussian mechanism with `sigma=0.75 ms`.  This channel is translation-invariant to a uniform onset shift.

## Temporal combination

`T = 0.82 * M + 0.18 * I`

where `M` is multiscale recurrent timing similarity and `I` is ISI similarity.

## Full episode score

Let `S` be mean spike-frame Jaccard similarity and `D` the mean added/removed-state differential similarity.

`Score = 0.18*S + 0.12*D + 0.70*T`.

These are decoder equations only; they do not alter the neural simulation.

## 0.9.54 continuous-region discovery

Let frame activity be

`a(t) = 1` if at least one neuron spikes in frame `t`, otherwise `0`.

Let the ordered active times be `A = [t_1, t_2, ..., t_n]`. A new decoded region is started when

`t_i - t_(i-1) > G`, with `G = 32 ms` in the current baseline.

The encoder currently inserts a 96 ms quiet interval between text words, but that interval length is not stored in the episodic memory record and is not read by the region detector.

For exact deterministic translation of a recovered region, recurrent route times are made translation-invariant:

`tau_i = t_i - t_0`.

The ordered tuples `(source_i, destination_i, tau_i)` are hashed as a neural-evidence content identity. A hash hit is only an exact fast path. A miss is evaluated by the multiscale equations above.

## 0.9.55 compact engram coding

The neural decoder equations above are unchanged. 0.9.55 only changes representation.

For native recurrent route event `i`:

`Delta_t_i = t_i - t_(i-1)`

`edge_i = 128 * source_i + destination_i`

`Delta_t_i` is zig-zag encoded and both values are stored as unsigned variable-length integers before stream compression.

For one active frame, the 128-neuron state is represented by the bit vector

`B_t = sum_(n active at t) 2^n`.

Continuous memories persist only active frame masks; silent frames are implied by duration and reconstructed for a decoder region when required.

### Translation-invariant neural route digest

For one discovered region, let its recurrent route times be `t_i` and the first route time be `t_0`:

`tau_i = t_i - t_0`.

The ordered binary tuples `(source_i,destination_i,tau_i)` are SHA-256 hashed. This digest is a compact identity of the recurrent neural evidence. It contains no English token. The separate decoder dictionary supplies the learned label.

### Storage complexity

If raw route count is `R`, the old Python representation has very large per-object overhead in addition to `O(R)` values. MBE2 remains `O(R)` information but stores each event in a few delta-coded bytes before compression and keeps the compressed blob off-heap/on disk during ordinary recall.

Exact recall therefore requires approximately `O(W)` hot neural index entries for `W` recovered regions, while raw route evidence remains disk-backed.

---

## 0.9.57 optional semantic-state layer

These equations do **not** alter the neural-memory dynamics. They describe the deterministic bounded functional-state accumulator fed by the optional Qwen semantic sensor.

Let all sensor values except valence be clipped to `[0,1]`. For message `t`, define positivity `P_t`, negativity `N_t`, hostility `H_t`, praise `R_t`, urgency `U_t`, salience `S_t`, correction `C_t`, uncertainty `Q_t`, and novelty `V_t`.

The target valence is

```text
v*_t = clip(P_t - N_t, -1, 1)
```

and maintained valence uses inertia:

```text
v_{t+1} = clip(0.88 v_t + 0.12 v*_t, -1, 1)
```

Other maintained dimensions are:

```text
arousal_{t+1}     = clip(0.90 arousal_t     + 0.10 max(U_t,H_t,S_t))
frustration_{t+1} = clip(0.90 frustration_t + 0.07 H_t + 0.05 C_t)
warmth_{t+1}      = clip(0.94 warmth_t       + 0.04 R_t - 0.035 H_t)
confidence_{t+1}  = clip(0.96 confidence_t   - 0.035 C_t - 0.025 Q_t + 0.01 R_t)
curiosity_{t+1}   = clip(0.92 curiosity_t    + 0.08 max(V_t,S_t))
```

All `clip` operations above use `[0,1]` except valence. These values are deliberately low-bandwidth context with decay/inertia; they are not a claim of subjective emotion and they never replace the raw stored neural experience.
