# LAMb roadmap

v0.1 is a faithful, CPU-runnable scaffold of the full architecture. The
extensions below are ordered roughly by leverage. Consult arXiv for the latest
work before implementing each — several of these are active 2025–2026 areas.

## 1. GRPO-trained hypernetwork proposer (implemented)

Implemented in `lamb/selfplay/hyperproposer.py` (`proposer_kind="grpo_hyper"`): a
hypernetwork maps solver competence to task-distribution logits, trained by GRPO
with learnability rewards and a **KL anchor to the bandit** (the anti-collapse
mechanism). The bandit stays the recommended default; the hypernetwork is the
drop-in for large task spaces.

Remaining upgrades:

- **Heavier hypernetwork.** Emit the *weights* of a small task-generator network
  rather than the distribution parameters directly — needed once the task space
  is continuous/compositional rather than a fixed grid.
- **Richer context.** Feed recent success slope and memory-utilisation stats, not
  just per-cell success, so the policy anticipates the frontier.
- **Mixture annealing.** Anneal the coverage floor `eps` down as the policy
  earns trust.

Refs: GRPO (DeepSeekMath, [2402.03300](https://arxiv.org/abs/2402.03300));
HypeRL ([2501.04538](https://arxiv.org/abs/2501.04538));
Absolute Zero ([2505.03335](https://arxiv.org/abs/2505.03335)); R-Zero
([2508.05004](https://arxiv.org/pdf/2508.05004)).

## 2. GRPO / RLVR on the solver (implemented)

Implemented (`solver_algo="grpo"`): sampled-group answers scored by the exact
verifier, group-relative advantages, KL to a refreshed reference, added on top of
expert iteration (a warm start that avoids the RL cold-start). Includes DAPO
dynamic sampling and an optional Dr.GRPO no-std normalization.

Remaining: replace the SFT warm start with a scheduled hand-off; GSPO
sequence-level ratios; clip-higher (DAPO) once multiple epochs per rollout are
used. Refs: DAPO, Dr.GRPO, GSPO.

## 2b. Benchmark harness

Build out the battery in [`BENCHMARKS.md`](BENCHMARKS.md): a length-generalization
eval ships in `lamb.eval.length_generalization`; add long-context recall
(needle / passkey / BABILong-style) and a latent-reasoning task (ProntoQA/ProsQA)
adapter. General-LLM suites (MMLU) are out of scope until a language front-end
exists.

## 2c. Red Queen coevolution (open-endedness) — the active direction

The reactive autocurriculum saturates on a bounded task space (learnability -> 0
once everything is mastered). Unbounded self-improvement needs *relative* fitness:
the solver must keep dominating its own past on an ever-advancing frontier.

- **Step 1 (implemented, `red_queen=True`).** Solver league (historical
  self-play), a novelty/diversity term on task selection, and relative-fitness
  metrics (`dominance`, `forgetting`) in `lamb/selfplay/league.py`. On the bounded
  grid, `dominance` starts positive and decays to 0 as the space saturates — the
  measured signal that motivates Step 2.
- **Step 2 (implemented, `open_ended=True`).** A **generative grammar** of nested
  expressions (`lamb/selfplay/grammar.py`) with an append-only descriptor space
  grown by **minimal-criterion admission** (`openended.py`), and a **factored
  hypernetwork proposer** (`factoredhyper.py`) that generates *in* that space with
  fixed `D+G+O` outputs. Measured: the space grows and the frontier advances with
  zero forgetting; `dominance` is now bounded by solver capacity rather than
  task-space saturation (the ceiling moved from the curriculum to the model), so
  sustained positive dominance is a scale-up result. This subsumes item 5 below.
  Refs: POET ([1901.01753](https://arxiv.org/abs/1901.01753)); Minimal Criterion
  Coevolution (Brant & Stanley).
- **Step 3 (implemented, `python -m lamb.poet`).** A **POET-style population** of
  (environment, agent) pairs (`lamb/poet.py`): per-environment specialist solvers,
  agent **transfer** across environments, minimal-criterion + novelty
  **reproduction** of harder environments, and graduation of easy ones. This is
  the capacity-scaling mechanism -- specialists + transfer raise the ceiling Step 2
  hit -- so it doubles as roadmap item 7. Measured: the population grows, its
  attempted frontier advances, transfers fire, and it conquers base environments;
  the conquered frontier scales with per-agent budget and size. Refs: POET
  ([1901.01753](https://arxiv.org/abs/1901.01753)); Enhanced POET
  ([2003.08536](https://arxiv.org/pdf/2003.08536)); Transfer Dynamics
  ([2203.10941](https://arxiv.org/pdf/2203.10941)).
- **Shared-backbone POET (implemented, `python -m lamb.poet_shared`).** One shared
  backbone plus a tiny low-rank adapter per environment (`lamb/poet_shared.py`): a
  population of N environments costs one backbone + N adapters (~21% of the
  parameters of N full models at capacity 5), and the backbone accumulates
  cross-environment skill for implicit transfer. Measured: it grows, transfers, and
  conquers the base frontier, at lower absolute mastery than independent agents --
  the honest efficiency/capacity tradeoff of a final-layer adapter. **Deeper
  adapters** (`lamb/model/lora.py`) that adapt the core's attention linears are
  the default: **DoRA** (weight-decomposed, arXiv:2402.09353 -- magnitude/direction
  decoupling) and LoRA. At equal budget the best specialist reaches ~0.77 (DoRA)
  vs ~0.53 (LoRA) vs ~0.50 (final-hidden), for one magnitude vector per layer on
  top of LoRA -- DoRA is the clear winner, as in the paper. `--adapter-type
  {dora,lora,hidden}`.
- **Behavioural-novelty admission (implemented, `lamb/selfplay/novelty.py`).** Both
  POET variants gate reproduction on *behaviour*, not descriptor dedup: each
  candidate environment is characterised by its seed agent's competence across
  latent-step budgets, the answer length it demands, and how much extra thinking
  helps (a BC vector), and admitted only if it is far, in behaviour space, from an
  archive of admitted environments (novelty search). Measured: the population still
  grows and conquers while ~12 behaviourally-redundant candidates are rejected --
  real diversity maintenance. Refs: novelty search (Lehman & Stanley); POET.
  Remaining: GPU scale-up of the agents.

Refs: Digital Red Queen ([2601.03335](https://arxiv.org/abs/2601.03335));
PopuLoRA ([2605.16727](https://arxiv.org/pdf/2605.16727)); learnable information
gain ([2603.02218](https://arxiv.org/pdf/2603.02218)).

## 3. Continuous-thought decoding (Coconut) — Stage A implemented

Implemented in `lamb/coconut.py` (`python -m lamb.coconut`) and the Coconut path
on `LAMb` (`_roll_thoughts` / `coconut_logits` / `coconut_solve`). Where the
recurrent core thinks *vertically* (iterate a block at fixed positions), Coconut
adds *horizontal* latent thinking: insert `K` scratchpad positions between the
prompt and the answer whose input embedding is the model's own last hidden state,
fed straight back in continuous space (via `thought_norm` + a learned
`thought_marker`) and never decoded. Those thoughts join the attention context —
a working memory the answer reads. `K=0` reduces *exactly* to ordinary
answer-only teacher forcing (a regression test asserts it; RoPE is relative so the
left-padded prompt is unaffected).

Key result, and a departure from the literature. Coconut's published gains need a
curriculum that distils the thoughts from **language** CoT steps; its own
`w/o curriculum` ablation — our exact setting (feed the state back, supervise only
the answer) — underperforms even no-thoughts *in the language domain*, and
arXiv:2510.12167 traces that to geometrically homogeneous latents plus no reliable
way to **select** a good trajectory. Here, with a number-native substrate and
genuinely multi-step tasks, plain answer-NLL through a randomly sized thought
chain **does** make the scratchpad useful: on depth-2 nested expressions greedy
exact-match rises `K=0: 0.39 → K=3: 0.52` (a tiny 0.28M-param CPU model), and the
latent-collapse diagnostic (`collapse_metric`, the 2510.12167 homogeneity signal)
falls from cos≈0.88 to cos≈0.27 as the thoughts specialise.

The verifier is LAMb's asset the 2510.12167 line lacked. Perturbing the latent
phase with dropout draws diverse trajectories and the exact verifier **selects**
the winner (best-of-N), turning their monotone-but-unusable Pass@N into *realised*
accuracy: `N=1: 0.52 → N=8: 0.66` — a test-time self-improvement loop (search the
latent space, verify, keep the winner) that needs no labels at deploy. Hard
STaR-filtering to the verifier-solved subset was tried and **rejected**: on a tiny
model it starves the gradient and collapses the thoughts to a constant, and since
the grammar already supplies exact labels, teacher forcing already carries the
verifier's information — so the verifier's unique leverage is at inference.

Refs: Coconut ([2412.06769](https://arxiv.org/abs/2412.06769)); inference-time
scaling for continuous-space reasoning ([2510.12167](https://arxiv.org/abs/2510.12167)).

### 3a. Restructured for scale: parallel supervised latents (LOTUS) — implemented

The Coconut path above works at tiny scale, and that is precisely its documented
limit. Measured across backbones ([2606.31779](https://arxiv.org/abs/2606.31779)),
the **sequential** continuous-thought family's gap to explicit chain-of-thought
*widens* with scale (−0.1 pts at 124M, −2.3 at 1B, **−9.2 at 3B** — a performance
cliff), while the **looped, parallel-supervised** family stays flat (−1.5 at 3B).
`lamb/lotus.py` (`python -m lamb.lotus`) is the restructure:

- **Parallel, not autoregressive.** `n_latent` latent positions are appended after
  the prompt at once and refined by `loops` passes through the shared core. Cost is
  `O(loops)` forwards **regardless of the latent count** (asserted in tests: 4 and
  64 latents both cost `loops+1` passes), where Coconut needed one sequential
  forward per thought. The latent budget grows for free.
- **Every latent position is supervised** through the LM head. LOTUS uses gold CoT
  tokens; LAMb has no language, so it generates a gold **numeric** trace — the
  intermediate sub-expression values — exactly and for free from its own evaluator
  (`TaskGrammar.sample_with_trace`). No language, no external data, no annotation.
- **Still a blackbox.** Latent positions are never decoded; only the answer is
  emitted. The trace is a training signal, not an output.

Measured on depth-2 nested expressions, 5 seeds per arm
(`python -m lamb.study --task d2g1`). **The baseline that matters is `coconut-long`**:
the same Coconut given the extra steps that make its *wall clock* equal to LOTUS's,
since LOTUS costs 0.564 s/step against Coconut's 0.335 and equal step counts quietly
hand it 1.68x the compute.

| arm | steps | answer acc (mean, n=5) | sd | range |
| --- | --- | --- | --- | --- |
| Coconut, equal steps | 1000 | 0.504 | 0.138 | 0.316–0.703 |
| **Coconut, equal wall clock** | **1684** | **0.836** | **0.022** | 0.809–0.865 |
| LOTUS, answer-only | 1000 | 0.826 | 0.076 | 0.746–0.910 |
| LOTUS, + per-position trace | 1000 | 0.903 | 0.054 | 0.811–0.951 |

Paired against the wall-clock-matched control:

| comparison | mean | sign-flip *p* | permutation *p* |
| --- | --- | --- | --- |
| LOTUS answer-only − Coconut | **−0.010** | 0.688 | **0.802** |
| LOTUS +trace − Coconut | **+0.067** | 0.0625 (floor) | **0.040** |

- **The structural claim is not reduced, it is absent.** Parallel supervised latents
  buy **nothing** over the sequential loop once compute is equalised: −1.0 points at
  *p* = 0.80, negative on three of five seeds. An earlier version of this section
  claimed **+32.3 pts** and called it the one result a 5-seed study could establish.
  That number was measured against a Coconut arm that had simply not finished
  training, and it is withdrawn.
- **The variance claim is withdrawn too.** This section previously argued that the
  restructure made training *reliable*, from Coconut's 39-point seed swing. That
  swing was **undertraining, not instability**: given equal wall clock the same arm
  has sd **0.022**, the tightest in the study — tighter than either LOTUS arm.
- **The power figure was inflated by the same bug.** "61 seeds to resolve 5 points"
  came from the undertrained arm's spread. Corrected, it is 19 seeds for
  `lotus-answer`, 10 for `lotus-trace`, 2 for `coconut-long`. The constraint is real
  and much milder than stated.
- **What survives is the trace supervision, and only that.** +6.7 points over a
  properly trained, compute-matched baseline, positive on all five seeds,
  permutation *p* = 0.0397. It is also the claim tied to what this project actually
  has that others do not: the gold trace is free here because an exact verifier
  generates it. The honest statement is not "the parallel architecture wins" but
  "the free exact supervision is worth about seven points."

The trace-probe (a diagnostic, never decoded) confirms the latent block carries the
intermediates: 0.97 with supervision against 0.14 at chance without. The `+trace`
arm uses information the control does not, so the comparison is not like-for-like —
it prices the supervision, not the structure.

One methodological note, since this is the second time it has mattered: the single
run figures originally reported here (0.512 / 0.695 / 0.945) were wrong for two
independent reasons, a 53%-contaminated eval set (3a-iii) and a baseline that had
not converged. Each was found only by building the control that could expose it.
A caveat left standing in a document is not a control.

Refs: LOTUS ([2606.31779](https://arxiv.org/abs/2606.31779)); SIM-CoT
([2509.20317](https://arxiv.org/abs/2509.20317)); looped transformers
([2502.17416](https://arxiv.org/abs/2502.17416)).

### 3a-i. SWITCH boundary tokens — implemented (entry only)

RL is the known dead end for latent reasoning: on-policy methods move a latent
model essentially not at all (arXiv:2512.11816 — GRPO takes explicit CoT 62.6 →
72.6 while latent goes 22.6 → 21.8), because a continuous segment has no
well-defined policy ratio to optimise. SWITCH
([2606.13106](https://arxiv.org/abs/2606.13106)) fixes that by making entry into
the latent segment a **predicted token**. Two new ids (`BOT`, `EOT`) are appended
after the digits, so every pre-existing id is unchanged and neither can ever appear
in an answer.

`LotusReasoner` emits `[prompt] [BOT] [latent x L]`. The last prompt position
predicts `BOT`, which gives the latent block a real log-probability and a fixed position for
probes to attach to.

**The accuracy claim for this did not survive.** It was first measured as
0.875 → 0.934 (+5.9) on a contaminated eval set. On the clean held-out partition the
comparison is 0.945 without the boundary vs 0.934 with it — the sign flips. The
multi-seed study in 3a-iv then explained *why* both numbers were meaningless — and
was itself corrected once a compute-matched baseline existed. The sharp version: the
boundary was never compared against a converged baseline, and at the spread of the
arm it *was* compared against, a single run could not have detected a real +5.9 nor
ruled one out. The honest reading is **no measurable effect, and no measurement**. Together with 3a-ii (its RL rationale falsified), the entry
boundary is therefore **off by default**: it costs a sequence position and two
vocabulary ids for nothing demonstrated. It is kept opt-in (`use_boundaries=True`)
because it remains the only well-defined attachment point for a probe or an RL
policy.

**Negative result, and the reason the exit marker is off by default.** Wrapping the
segment on *both* sides is actively harmful: a static `EOT` embedding sits between
the refined latents and the answer readout, so the first answer token gets predicted
from a generic marker instead of the latent state. Controlled at 500 steps:

| variant | answer acc | trace-probe |
| --- | --- | --- |
| no boundaries (control) | 0.371 | 0.697 |
| `BOT`+`EOT`, switch loss off | 0.176 | 0.468 |
| **`BOT` only** | **0.500** | **0.821** |

So the cost is structural, not the switch loss, and entry-only both fixes it and
beats the control. This matches the paper's own finding that the computation
concentrates at the *entry* transition; with a fixed latent budget the exit is
deterministic and needs no marker. `--exit-boundary` re-enables it for ablation.

**Monitorability: not established.** The boundary gives probes an attachment point,
and `boundary_probe` fits a class-balanced linear probe on the entry state to
predict whether the answer will be right — the practical reply to the
CoT-monitorability objection ([2507.11473](https://arxiv.org/abs/2507.11473)). But
it does not yet measure anything: at depth 2 the model is 93% correct, leaving too
few errors to fit against; at depth 3 it only reaches 9%, and the balanced sample
(24 held-out points) returns 0.333 vs a 0.5 baseline — noise. A real number needs a
regime where the model is competent *and* fallible (~50-70%), which this tiny model
on these tasks does not provide. The interface ships; the claim does not.

### 3a-ii. Does on-policy RL move the latent block? No — premise falsified

The boundary tokens in 3a-i were justified by SWITCH's claim that they unblock RL
over latent recurrence. `lamb/latent_rl.py` (`python -m lamb.latent_rl`) tests that
claim instead of assuming it.

Testing it honestly requires being precise about what RL could shape. Sampling
answer tokens has a well-defined log-probability with or without boundaries, but it
only shapes the *answer head* — never the latent computation. So the boundary has
to buy a **sampled action that changes the latent computation itself**: a switch
head on the boundary hidden state picks the latent budget (how many latent
positions stay active, applied by masking). It is sampled, has an exact
log-probability, and demonstrably changes the computation (asserted in tests).
Three arms run from one shared supervised checkpoint, same step budget, same eval:

| arm | acc | Δ from SFT start |
| --- | --- | --- |
| SFT start | 0.266 | — |
| + 300 more **supervised** steps | **0.645** | **+0.379** |
| + GRPO, switch action (boundary) | 0.262 | −0.004 |
| + GRPO, switch + entropy bonus | 0.301 | +0.035 |
| + GRPO, answer-only (the 2512.11816 setting) | 0.254 | −0.012 |

**RL is inert.** At best +0.035 where the same step budget spent supervised gives
+0.379 — an order of magnitude worse. This reproduces arXiv:2512.11816 (GRPO moved
a latent model 22.6 → 21.8) and, more importantly, the boundary tokens did **not**
rescue it: the switch arm is indistinguishable from answer-only. (These numbers are
from the contaminated-eval era; contamination would if anything *favour* the
supervised arm, and the gap is an order of magnitude, so the conclusion stands.)
Combined with the clean re-measurement in 3a-i — where the boundary's apparent
accuracy gain vanished — the entry boundary delivered nothing it was built for.

Observed mechanism, and an honest limit of this test: the switch policy collapsed
to a single budget (max) within ~100 steps, entropy 0.09 → 0.01, and an entropy
bonus did not prevent it. That collapse is partly **by construction** — with
budgets (2, 4, 8) and no cost on compute, "always max" is genuinely optimal, so
there is no allocation policy to discover. For RL to shape latent compute
allocation the reward must *price* compute. Other uncontrolled factors: one seed,
RL lr (2e-4) and KL coefficient not swept. What is solid is the size of the gap:
RL ≈ 0 against SFT +0.379 is far outside any noise band here.

Next, if this is revisited: a reward that charges for latent compute
(`correct − λ·budget`), so allocation is a real tradeoff rather than a dominated
choice. Until then, **do not plan on RL as the self-improvement mechanism for the
latent path** — the verifier-selected best-of-N search in 3 remains the mechanism
that measurably works.

### 3a-iii. Eval contamination — found and fixed

Every accuracy number above was originally measured against a *seed-separated*
held-out set, which turned out not to be held out at all. Disjoint seeds are not
disjoint problems: the depth-2/1-digit space has ~80k expressions and a 1000-step
run at batch 64 draws 64k of them, so **53% of the "held-out" set had been trained
on** — and the contamination grew with training length, biasing precisely the
longer-vs-shorter comparisons the project relies on. The Stage B comm task was
worse: a 200-problem space, trained on in full, so its eval was **100%** seen.

`lamb/holdout.py` fixes this structurally. Membership is decided by a hash of the
problem string, so a problem is permanently either train or eval, independent of
seeds and of how long training runs; training samplers reject the eval partition and
eval sets draw only from it. `tests/test_holdout.py` asserts zero overlap for the
grammar, Coconut, LOTUS and comm streams.

Effect on the results: the LOTUS restructure held up almost unchanged
(0.520→0.512, 0.707→0.695) and the trace arm improved (0.875→0.945) — a 0.28M model
on 44k problems cannot memorise much, so it was largely generalising already. The
boundary claim did **not** hold up (3a-i).

**The first pass at this fix was incomplete**, which is worth recording because the
incompleteness was invisible from the tests that were written for it. Only the
Stage A and Stage B trainers were wired to the partition. A later sweep of every
problem-sampling call site found four more that still drew from the whole space:

- `lamb/comm_pop.py` — the population trainer *and* its eval set. It holds out
  *pairings* `(i, i)`, which is the zero-shot-coordination question, and that was
  mistaken for holding out problems. The partner-randomization result was measured
  on problems the population had trained on.
- `lamb/latent_rl.py` — the GRPO rollouts drew the arms' own evaluation problems.
  Harmless to the conclusion there (it was negative, and contamination biases
  *upward*), but wrong.
- `lamb/selfplay/loop.py` and `lamb/poet.py` / `lamb/poet_shared.py` — the
  self-play and POET training samplers, which back the self-improvement claims.
  POET's `_score` is worse than a misreported number: POET *selects* on it, so
  scoring on trained problems was steering the search toward memorisation. It now
  scores on held-out problems.
- `lamb/eval.py` — the benchmark harness itself, behind the length-generalization
  numbers, sampled from the whole space.

The scope of the lesson is worth stating too, because the obvious over-correction
is to distrust seed separation everywhere. It was checked: the needle-retrieval and
RULER generators draw episodes from a space of ~10<sup>92</sup>, so two
seed-separated streams have an expected collision count of ~10<sup>-85</sup>, and a
measured overlap of exactly zero (`tests/test_holdout.py`). Seed separation is fine
there and those benchmarks needed no change. The problem was never seed separation
as such — it was seed separation over an **80k-expression space that a single run
covers 80% of**.

`TaskGrammar.sample_heldout` is the mirror of `sample(..., exclude_heldout=True)`,
so training rejects a partition that evaluation draws only from, and the two can
never meet. The lesson is about the shape of the fix rather than the fix: a
partition is only as good as its *least* careful call site, and a test that asserts
"these two streams are disjoint" says nothing about the streams nobody thought of.
The sweep is `grep` over every `sample(` call site, and it is cheap to repeat.

One cost is worth being explicit about, because it is a real trade and not a free
win. Partitioning a *small* space buys cleanliness at the price of resolution. The
fixed grid's 1-digit `+` cell holds exactly 100 problems, so its evaluation
partition is **9**: that per-cell accuracy now has a granularity of ~11 points
however many samples are drawn. `evaluate()` therefore returns `per_cell_unique`
alongside `per_cell`, so the resolution is visible in the output rather than
something the reader has to reconstruct. The answer to a coarse cell is a wider
cell, not a dirtier split.

For Stage B the partition is real but small — exactly 22 of 200 problems at 1 digit
(`CommTask.unique_count`), so *any* comm accuracy there has a granularity of ~4.5
points regardless of how many samples are drawn. The comm accuracy is best read as a
*channel* test — the listener never sees `X`, so it cannot answer from memorisation
without the message — rather than a generalisation test. `--a-digits 2` gives 2431
held-out problems if a generalisation claim is wanted.

### 3a-iv. How many seeds a claim needs — the answer is not one

The contamination above was a bug. The deeper problem was that **every number in
this repo was a single run**, and at this scale a single run cannot separate an
effect from seed noise. `lamb/study.py` runs each arm at several seeds and reports
the paired difference with an exact permutation test rather than an interval, since
with 5 seeds a normality assumption does more work than the data can support.

What it found on the original setting (`--task d2g1`, 5 seeds, 1000 steps):

| arm | steps | mean | sd | range |
| --- | --- | --- | --- | --- |
| Coconut, equal steps | 1000 | 0.504 | 0.138 | 0.316–0.703 |
| **Coconut, equal wall clock** | **1684** | **0.836** | **0.022** | 0.809–0.865 |
| LOTUS answer-only | 1000 | 0.826 | 0.076 | 0.746–0.910 |
| LOTUS + trace | 1000 | 0.903 | 0.054 | 0.811–0.951 |

This section originally reported only the first, third and fourth rows and drew
three conclusions from them. The `coconut-long` control was listed at the bottom as
a known confound to be closed later. Running it retracted two of the three:

- **Structure: withdrawn.** Against the compute-matched control, LOTUS answer-only
  is **−0.010** at permutation *p* = **0.802** — no effect at all, negative on three
  of five seeds. It had been reported as **+32.3, established**, with disjoint seed
  ranges and *p* = 0.0079. All of that was an artefact of comparing against a
  Coconut arm that had not finished training.
- **"Variance falls with each change": withdrawn.** The 39-point Coconut swing that
  this section built its methodological argument on was **undertraining**. Given
  equal wall clock, sd drops from 0.138 to **0.022** — the tightest arm in the
  study, tighter than either LOTUS arm. The restructure does not make training more
  reliable; finishing training does.
- **The power figure was inflated by the same bug.** "~61 seeds to resolve 5 points"
  used the undertrained arm's spread. Corrected: 19 seeds for `lotus-answer`, 10 for
  `lotus-trace`, **2** for `coconut-long`. The constraint on small claims is real
  but far milder, and the sharper statement about the SWITCH boundary (3a-i) is not
  that it needed 61 seeds but that it was never compared against a converged
  baseline either.
- **Trace supervision: survives, and is now the only surviving claim.** +0.067 over
  the compute-matched control, positive on all five seeds, permutation *p* =
  **0.0397**. It uses information the control does not — the gold trace — so it
  prices *the supervision*, not the architecture. That is a narrower claim than the
  one this section used to make, and it is the one attached to what this project
  actually has that others do not.

The spread is training variance rather than measurement noise: each arm is scored on
512 held-out problems, so binomial standard error at these accuracies is ~1.8 points.
Arms at the same seed share an eval set, so that component cancels from the paired
differences entirely.

The lesson is not about latent reasoning. **A confound recorded honestly in a
document is not a control**, and the gap between writing "this is a known confound"
and running the 50 minutes of compute that closes it was two wrong headline claims
and a methodological argument built on a measurement artefact.

### 3a-v. Latent budget decoupled from trace length; space supervision added

Two design gaps that the multi-seed work made worth closing, both taken from
mid-2026 results rather than from this repo's own logic:

- **The latent block was a copy of the trace.** One latent position per trace token
  means the block has to grow with the trace, so a long trace is simply out of
  reach — and [2607.16972](https://arxiv.org/abs/2607.16972) finds *both*
  continuous-CoT training regimes collapsing to about a third of explicit-CoT
  accuracy precisely on long traces. `trace_compress = c` supervises `c` consecutive
  trace tokens per latent through multi-token-prediction heads
  ([2404.19737](https://arxiv.org/abs/2404.19737)), so capacity is `n_latent * c`
  and the two are decoupled. The compression is **generative, not geometric**: C-MTP
  compresses by averaging the token embeddings a latent stands for, and
  [2606.20075](https://arxiv.org/abs/2606.20075) finds that rigid geometric
  compression collapses the reasoning space while generative reconstruction
  preserves its capacity — so each head decodes its own token and nothing is
  averaged. `c = 1` builds no extra heads at all, so it is bit-for-bit the previous
  model.
- **There was trajectory supervision but no space supervision.**
  [2606.20075](https://arxiv.org/abs/2606.20075) decomposes process supervision into
  dense stepwise signal (which the trace loss supplies) and preservation of the
  latent manifold's structure (which nothing here supplied), and attributes latent
  drift to the missing second term. `space_coef` adds a supervised-contrastive term
  ([2004.11362](https://arxiv.org/abs/2004.11362)) over latent positions carrying
  the same intermediate value. It is deliberately *relational*: pinning each latent
  to a fixed embedding of its value would be exactly the rigid constraint that
  analysis warns collapses the space.

Both default to **off**. They are arms to be measured on the harness in 3a-iv, not
claims — which is the discipline the boundary saga and the RL arm should have had.

### 3a-vi. The latent ALU: the latents hold values and the algebra does the arithmetic

Every latent-reasoning method supervises latents against **tokens** — LOTUS against
gold CoT tokens, SIM-CoT against a decoder's, C-MTP against averaged embeddings, and
3a above against the digits of LAMb's own trace. A token target teaches a latent to
*name* a number; it says nothing about what numbers *are*. So composition — which is
what reasoning consists of — has to be relearned by the network at every magnitude
it meets. That is the arithmetic length-generalization wall.

Models that *do* extrapolate turn out to have found a **periodic** representation by
themselves: tokens as phases, addition as rotation
([2511.23443](https://arxiv.org/abs/2511.23443) proves the benefit for modular
addition; [2606.17399](https://arxiv.org/abs/2606.17399) finds multiplication reduced
to addition in discrete-log space by the same mechanism). The usual response is to
bake periodicity into the **positional** encoding — Abacus
([2405.17399](https://arxiv.org/abs/2405.17399), already in this model) and position
coupling ([2405.20671](https://arxiv.org/abs/2405.20671)) — which *helps the network
find* the algorithm. The network still has to run it.

LAMb is in a position a language model is not: an exact evaluator hands it the true
value of every intermediate of every problem, free and unannotated. That is enough to
skip the discovery and hand the latent space the algebra outright. `lamb/algebra.py`
and `lamb/alu.py`:

- a latent slot carries one **value**, coded as its residues modulo a set of coprime
  moduli (a residue number system);
- composition is exact and carry-free — `(a+b) mod p` is a cyclic convolution of the
  residue distributions, `(a−b) mod p` a cross-correlation, `(a·b) mod p` one small
  table — and differentiable, so a model *uncertain* about a residue composes its
  uncertainty rather than having to commit first;
- the answer is built **bottom-up from the leaf slots by the algebra**, not read out
  by the network. The network is only ever asked for a leaf: two literals, one
  operator, whatever the expression's depth.

RNS is classical, but in neural networks it has only ever been used for **hardware
efficiency** ([1712.04614](https://arxiv.org/abs/1712.04614),
[2306.09481](https://arxiv.org/abs/2306.09481),
[2408.05639](https://arxiv.org/abs/2408.05639)). Using it as the *reasoning*
representation, supervised from a free exact trace, is the new part.

**The moduli are chosen for the order of 10, not for size.** The network produces
`n mod p` from digits as `Σ dᵢ·(10ⁱ mod p)`, whose coefficients repeat every
`ord_p(10)` positions — and that period is exactly how many digit positions a run
must *see* before the modulus is learnable, with every wider number free afterwards.
Small primes are a trap: `ord(10)` is 16 mod 17, 18 mod 19, 22 mod 23. The default
`(2,5,9,11,7,13,37)` has max period 6 in 84 units, strictly dominating the obvious
`(7,11,13,17,19,23)` (90 units, period 22). Three of them are the schoolbook
divisibility rules — mod 9 the digit sum, mod 11 the alternating sum, mod 2 and 5 the
last digit — which extrapolate at any width immediately.

**What holds without any training** (`tests/test_algebra.py`, `tests/test_alu.py`):
exact round-trip over the whole signed ring; exact composition of `+`, `−`, `×` at
every magnitude in range; and, given correct leaf codes, an exact answer at **depth 6
— a 64-operand expression — with nothing learned about depth**.

**What this does *not* claim, stated plainly:**

- *"Depth is free" applies to the composition, not to end-to-end accuracy.* The leaf
  count grows as `2^(D−1)` and every leaf needs all its residues right, so accuracy
  falls roughly as `leaf_residue_acc ^ (n_moduli · n_leaves)`. CRT has no locality: one
  wrong residue is a wildly wrong answer, not a near one. Redundant moduli are the
  classical fix and would double as a second label-free error signal; not yet built.
- *The slot addressing does not extrapolate the way the algebra does.* Training at
  depth 2 only ever writes slots 0–1, so slot 6's embedding is untrained and a
  depth-4 evaluation asks the model to use slots it has never used. Depth transfer
  therefore needs mixed-depth training; it is not free.
- *The expression tree is read from the input.* That is legitimate here — the tree is
  the question, not the answer — but a natural-language problem does not come with
  one. The bridge needs the model to **emit** the structure, at which point the slots
  become registers and the selectors become instructions: a differentiable register
  machine over the algebra. The gold program is free from the same generator, so it
  can be supervised before being relaxed — which is how to avoid the instability that
  sank earlier neural program induction. (The *tree* is indeed absent from prose, but
  GSM8K's calculator annotations supply the evaluation *chain*, which is what the
  register machine actually needs — see 3c. The gap was narrower than this bullet
  assumed.)
- *The consistency check is not usable yet.* Measured at 400 steps, leaf-residue
  accuracy is **0.985** while the root slot sits at **0.229**, barely above chance:
  the model specialises on leaves exactly as intended, because composition is no
  longer its job, so it never learns to state the answer. A check that compares a
  near-chance direct prediction against a good composed one carries little signal.
  The term is implemented (`alu_consistency_coef`) and **off by default** — training
  on it invites the model to satisfy agreement by routing rather than by being right,
  which makes it worth more as a measurement than as a loss until it has been
  measured.

**Pre-registered test.** Train on operand widths 1–4, evaluate at width 5. The moduli
`(16,25,27,11,37)` have coefficient patterns fully determined by digit position 3, so
widths 1–4 see every pattern and width 5 is pure periodicity. Prediction: the ALU
holds and the token-trace baseline falls. If the ALU does not hold at width 5, the
residue representation is not buying what it claims and this is mostly dead.

**Result: the pre-registered test did not run, and that is the honest verdict.**
1200 steps, batch 64, 0.28M params, trained on widths 1-4:

| operand width | in training? | ALU (algebraic) | ALU leaf-residue | token-trace baseline |
| --- | --- | --- | --- | --- |
| 1 | yes | **0.992** | 0.998 | 0.133 |
| 2 | yes | 0.000 | 0.054 (chance) | 0.012 |
| 3 | yes | 0.000 | 0.053 | 0.000 |
| 4 | yes | 0.000 | 0.059 | 0.000 |
| 5 | **no** | 0.000 | 0.053 | 0.000 |

Widths 2-4 were *in the training distribution* and sit at chance, for **both** arms
— the baseline scored 0.133 at width 1 here against 0.903 in the single-width study
of 3a-iv, so mixed-width training at this budget broke it too. Extrapolation cannot
be measured from a model that failed in-distribution, so the pre-registration's
"then it is mostly dead" does not apply: its precondition failed. What does survive
is narrower and real — at the one width both arms could learn, on the same budget
and the same information, the ALU reached **0.992 against the baseline's 0.133**.

**The failure is a cliff, not a slope, and it exposes a design error.** At width 1
the leaf values are at most 18, so `n mod p` is nearly the identity for moduli
16/25/27/37 and almost no arithmetic is required; from width 2 the network must
actually compute `n mod p` from digits. But **that is a known fixed linear function**
— `Σ dᵢ·(10ⁱ mod p)` — and the digits are in the prompt. Asking a network to induce
an exact function it could simply be given is a waste of capacity, not a test of the
idea. Measured: a *fixed* digit→residue encoder plus algebraic composition is exact
at depth 2 width 8, depth 3 width 4, depth 4 width 3 and depth 5 width 2, with
nothing learned anywhere.

Confirmed by a follow-up: training on width 2 **alone**, with the whole budget on
one width, leaves leaf-residue accuracy flat at chance for 1200 steps (0.050, 0.055,
0.047, 0.047 at steps 300/600/900/1200, every modulus at chance). It does not learn
slowly; it never starts. So the fixed encoder is not an optimisation, it is required.

That reframes what the ALU is for, and the reframing is not a consolation. With a
fixed encoder and a known structure, LAMb+ALU on synthetic arithmetic **degenerates
to an exact calculator** — correct at any width and depth, and evidence of nothing,
because no learning is involved. The corollary has to be stated plainly: *synthetic
arithmetic is the wrong benchmark for this component.* Its value is that arithmetic
stops consuming capacity and stops being an error source, which only shows up where
arithmetic is incidental and comprehension is the bottleneck. Validating it
therefore has to wait for the bridge, which is a real deferral and not a result.

**One thing did land, and it replaces the broken part.** The root-slot consistency
check of 3a-vi is weak because the model never learns the root. Redundant residues
give the same signal without it: size the moduli so legitimate values occupy a
sub-range of the ring, and a single corrupted residue throws the CRT reconstruction
outside that range. Measured over 20,000 trials with core `(16,25,27,11)` and
redundant `(37,7)`: **100% of single-residue corruptions detected, 0 false alarms**
(theoretical rate 99.61%). No labels, no root slot, no training — a pure range check
on the reconstruction. That is the label-free correctness signal the project needs in
order to survive leaving the distribution its verifier was written for.


### 3a-vii. The program can be induced from the answer alone

The register machine (`lamb/regmachine.py`) turns the latents into registers and has
the model emit a *program* over them — an operation and two pointers per step — which
the residue algebra executes. Registers are append-only, so the dataflow is a DAG and
a pointer mask makes reading an unwritten register unrepresentable rather than merely
penalised.

The question that mattered, pre-registered before running: **is the differentiable
executor a capability or a convenience?** PAL and Program-of-Thought call an external
Python interpreter, so a wrong answer cannot tell a pointer which way to move and the
program can only be imitated or reinforced — and RL on this latent block was measured
inert (3a-ii). Here the executor is exact algebra and differentiable, so the answer
loss reaches the pointer heads *through the arithmetic*. Two arms, depth 2:

| arm | `program_coef` | answer acc | canonical acc |
| --- | --- | --- | --- |
| supervised | 1.0 | 1.000 | 1.000 |
| **answer-only** | **0.0** | **1.000** | **0.000** |

With the gold program removed entirely, the model reaches **perfect held-out answer
accuracy** while matching the generator's program on *zero* instructions. That is not
a contradiction, and the diagnostic that looked like a failure is the one that
explains it: **the program computing a function is not unique.** Exhaustively, 48
distinct three-instruction programs compute `(a+b)+(c+d)` exactly — commutativity and
re-association — of which the grammar emits one. A model inducing any of the other 47
scores 0.000 against the canonical form while being completely correct.

So `canonical_acc` measures *conformity to the generator's form*, and `answer_acc` is
the correctness measure. Held-out answer accuracy of 1.000 with a non-canonical
program is the strong version of the result: the program generalises, so it is
computing the right function rather than memorising a mapping.

**Gradients through exact arithmetic are sufficient to induce a correct program from
outcomes alone.** That is the one thing in this design that a non-differentiable
executor cannot do.

**Corrected in place:** this paragraph used to end *"and it is the mechanism the
language bridge depends on — a word problem does not come with a gold program
either."* The first clause stands; the second is **false for GSM8K**, whose train
solutions carry inline calculator annotations (`<<48/2=24>>`) that recover to a
program directly (3c). So the bridge does not have to rest on this result — it gets to
*test* it, with the supervised arm as a control. Which is better news for the bridge
and worse news for this section, because it removes the argument that an `n=1` result
had to be relied on rather than measured. The depth-3 criterion is pre-registered in
3a-xii.

Caveats, since the arms are n=1 at the smallest depth: three instructions over four
operands is a small program space, `op_acc` at 0.603 says the answer-only arm's
operator choices are partly canonical and partly not, and depth 3+ has not been run.
The claim established is that outcome-only induction *works here*, not that it scales.

### 3a-viii. Division and decimals: exact rationals in the ring

Checking what the algebra could actually express for a grade-school word problem
turned up two gaps, both fatal rather than inconvenient, and one of them silent.

**Division was absent and unavailable.** In a residue system you divide by
multiplying with a modular inverse, which exists only for divisors coprime to every
modulus and gives the true quotient only when the division is exact. Worse, the
moduli had been chosen for short digit-periods (3a-vi) — powers of 2 and 5, and
divisors of `10^k − 1` — and that is *precisely* the set that makes small divisors
non-invertible. Under `(2,5,9,11,7,13,37)`, **not one divisor from 2 to 12 has an
inverse**, and "half as many" is the most common operation in GSM8K. The
optimisation that made the encoder extrapolate is the one that made division
impossible.

**Mixed scales were silently wrong.** `extract_quantities` returns `3.25` as
`(325, scale=2)` and `7` as `(7, scale=0)`; composing those gives `332`, i.e. 3.32.
The parser tracked scale and the algebra did not, so any problem mixing decimals
with integers produced a confident wrong number — the worst failure mode available.

Both dissolve under one change (`lamb/rational.py`): carry a value as a **pair**
`(numerator, denominator)`, each an ordinary residue vector.

| | |
| --- | --- |
| `a/b + c/d` | `(ad + cb) / bd` |
| `a/b − c/d` | `(ad − cb) / bd` |
| `a/b × c/d` | `ac / bd` |
| `a/b ÷ c/d` | **`ad / bc`** |

Division becomes multiplication with the operands swapped: exact, closed, requiring
no modular inverse, and still differentiable because every step is `+`, `−` or `×`
on distributions. Decimals stop needing scale bookkeeping entirely — `3.25` *is*
`325/100`, and a scale is just a denominator.

Verified against Python's own `Fraction`: **0 errors over 1200 random operations**
across all four operations, plus chains, `3.25 + 7 = 41/4`, and `7 ÷ 2 = 7/2`.

The cost is that a fraction cannot be reduced in residue form, so denominators grow
multiplicatively and that, not the design, sets the ring size. `RATIONAL_MODULI`
`(64,125,27,11,7,13,37,101,41,271)` gives ±4.5e15 with every digit-period still ≤ 6;
the worst chain a grade-school problem produces (five two-decimal values, denominator
1e10) has five orders of magnitude of headroom. `denominator_magnitude` exists so a
long chain approaching the ring is *observed* rather than discovered from a wrong
answer.

Two things this does not solve, recorded rather than left implicit:

- **Division by zero is undetectable in the ring.** A zero denominator is a legal
  residue vector; decoding raises rather than returning a number, so guarding it is
  the program's job. That is the right place for it — the emitted program is what
  knows whether a divisor could be zero — but it is now a thing the program must do.
- **The batched-over-moduli path wastes more here.** Padding to the widest modulus
  costs ~7× the arithmetic at `P=271`, against ~2× at `P=37`. On a GPU that is still
  the right trade, since the path is launch-bound by four orders of magnitude; on CPU
  it is not, and the per-modulus loop remains available.

### 3a-x. Division in the register machine

The rational layer (3a-viii) made division exact; this puts it in the machine's
instruction set. `RegisterMachine(rational=True)` holds each register as a
`(numerator, denominator)` pair and widens `OPS` from `(+,−,*)` to `(+,−,*,/)`,
which costs one extra output on the operation head and nothing else — division is
multiplication with the operands swapped, so no new primitive is introduced.

Verified end to end: `(48 ÷ 2) + 3.25 → 109/4`, exactly, through emitted
instructions — a division *and* a decimal, neither expressible on plain residues.
Gradients still reach the pointer and operation heads through the rational
arithmetic, so the outcome-only induction of 3a-vii is not given up to gain
division.

The integer path is left byte-for-byte as it was, since it carries the depth-2
result; `rational=True` is opt-in.

**One caveat, and the first version of this paragraph got it wrong.** I wrote that a
soft pointer read produces a *blend* of the registers it mixes. A test written to
demonstrate that falsified it: a 50/50 read over `1/2` and `3/4` decoded to exactly
`1/2`. Investigating found something worse. Decoding takes an argmax **per modulus,
independently**, so the winning residues need not come from the same register; when
they disagree the CRT lands somewhere unrelated to either input. Measured over random
pairs, **~30% of soft reads decode to a value in neither register**, and the failure
is *incoherence*, not averaging.

It is also not specific to rationals — the integer register file has it too, for the
same reason, which the original framing obscured.

Two things contain it, one of which was already built:

- **Sharp pointers are exact**, and training drives pointers sharp: both depth-2 arms
  reached fully determined pointers and scored 1.000 answer accuracy.
- **Redundant moduli catch the rest.** An incoherent residue vector is not a
  legitimate value, so the range check of 3a-ix flags it without caring what made it
  inconsistent. Measured: **100% of incoherent reads detected, none silently wrong.**
  The redundancy added for the model's own residue errors covers this too.

Training is unaffected either way, since the losses read distributions rather than
the argmax.

**Corrected in place.** This paragraph used to read *"the grammar generates `+`,
`−` and `*`, so there is no division training data yet"*. That was true when it was
written and stopped being true one commit later: `ops_key=2` generates `+ − * /`
with divisions made exact by construction. The accurate statement is that the data
exists and **the consumer did not** — see 3a-xi, which is the more interesting
failure.

**Still open:** a zero divisor remains undetectable in the ring (3a-viii), so
guarding it belongs to the emitted program — and 3a-xi turns that from a note into
a cost, because the natural answer-side loss for rationals has a degenerate zero at
exactly that point.

### 3a-xi. Four seams, and an invariant broken where nobody was looking

Wiring the division work of 3a-x to the trainer of 3a-vii turned up four gaps
between components that were each correct on their own. Recorded because the shape
repeats: every one of them sat in the *join*, and the tests all passed because each
test constructed its own inputs rather than taking the previous stage's output.

**1. Out-of-range values were not masked in the register-machine trainer.** The
project's own invariant is that a wrapped value is a *different* number, not a large
one, so an out-of-ring value is masked and never clipped; `lamb/lotus.py` has done
that since the ALU landed. `RegMachineTrainer._losses` did not, and
`ResidueSystem.targets` is an unguarded `int(v) % p`, so an out-of-ring answer became
a perfectly legal cross-entropy target *for the wrong number*. It never bit because
the only configuration ever run — depth 2, one digit, `(+,−)` — cannot leave the ring.
Depth 3 and multiplication leave it immediately, which is to say the bug was waiting
precisely in the direction the next measurement goes. The drop rate is now reported
by `train_step` and `evaluate` rather than applied silently: an accuracy measured on
60% of a held-out set is not the same number as one measured on all of it.

**2. The grammar's division output was unparseable.** `alu.parse_expr` scanned
`"+-*"`, so the leaf `48/2` raised; and `gold_program` defaulted to the integer
`OPS`, so a `/` node raised from `ops.index`. Both sit inside
`RegMachineTrainer._prepare`. So `ops_key=2` — data that generates perfectly — crashed
the only program trainer in the repo, while the ROADMAP recorded the *opposite*
problem. Every rational test hand-fed `Fraction` values through hand-built pointers,
and a test that constructs its own inputs cannot fail on a parser.

**3. `execute` read the module-level `OPS` rather than the machine's.** Harmless
while every machine had three operations; silently wrong the moment the operation
head is four wide, since the fourth logit would be trainable, selectable, and never
composed.

**4. The bridge threw the scale away one line after parsing it.**
`registers_from_quantities` returned `[q.value for q in qs]`, so `3.25` entered the
register file as `325` and `7` as `7`. That is the mixed-scale failure 3a-viii was
written to kill, reintroduced at the one seam where nothing downstream can catch it,
with the correct join (`RationalAlgebra.encode_scaled`) sitting unused two modules
away. It now returns `(values, scales, counts)`.

**5. `lamb/study.py` was broken on any machine with a GPU, and had never been run on
one.** The first attempt to run the arms above died instantly: every worker raised
*"Cannot re-initialize CUDA in forked subprocess"* on its first optimiser step. The
chain is worth spelling out because each link looked harmless.

The file opened with a clamp — if `resolve_device("auto")` reported CUDA, drop to one
worker, since one CUDA context per worker is ~300–500 MB and four exhaust a small
card. That clamp calls `torch.cuda.is_available()`, which **initialises CUDA in the
parent**, and `ProcessPoolExecutor` then *forks*. Under the pinned torch 2.14,
`Adam.step` runs a health check through `torch.accelerator.current_stream()`, so every
child dies — even though every arm in `_run_one` pins `device="cpu"` and no worker
ever wanted a card.

So the clamp caused the failure it was written to prevent, and clamping to one worker
did not help, because the problem was the fork and not the context count. Fixed by
using a **spawn** context; the clamp is removed outright, because the arms are
CPU-pinned by construction and the guard was protecting against a situation that
cannot arise. On this box it also cost a 4× slowdown for nothing.

It went unseen because CI is CPU-only and the GPU work was done from notebooks — so
**the tool this project's measurement discipline rests on had never once executed on
the hardware its own scale-up section points at.** That is a worse gap than any of
the four above: a broken trainer produces a wrong number, and a broken harness
produces none, but it also means none of the discipline was available on the hardware
where the interesting runs happen.

**A design cost that came out of the fix, stated rather than buried.** The integer
answer loss is cross-entropy of the executed result against the answer's residues.
That does not transfer to rationals: a fraction cannot be reduced in residue form, so
the pair a program builds for `109/4` is some unreduced `(N, D)`, and supervising `N`
against `109` supervises the wrong number. The criterion that *is* correct is the
cross-product residual `N·q − D·p = 0` — one target, zero, in every modulus, built
from the `+ − *` the ring already has.

It has a degenerate zero, and **the first version of this paragraph got its shape
wrong** — in the direction that overstates the problem. I wrote that a program reaches
`(N, D) = (0, 0)` by dividing through a register holding `0`. The test written to
demonstrate that falsified it: `5 / 0` gives `(5, 0)`, whose residual is
`5·1 − 0·7 = 5`, so the criterion *rejects* it at full cost. A zero divisor is
detected, not hidden.

Since `a/b ÷ c/d → (ad, bc)`, reaching `(0, 0)` needs a zero **numerator** as well as
a zero divisor — `0 ÷ 0`. Still reachable: one-digit operands include `0`, and a
pointer pair may address the same register twice. So `den_zero_coef` is load-bearing
for that one case rather than a general necessity, and the residual does more of the
work than the first draft credited it with. 3a-viii's *"division by zero is
undetectable in the ring"* met from the loss side — where the ring turns out to be
better at it than expected.

### 3a-xiv. Pre-registered: can the answer loss *hold* a program it cannot *find*?

3a-xii establishes that outcome-only induction fails at 7 instructions (0.015 against a
supervised 1.000) while the supervised arm is perfect at both depths. Both arms are
constant extremes, and `RegMachineTrainer`'s own docstring has described the intended
curriculum since it was written -- *"supervise the program first, lean on the answer
after"* -- which **no arm had ever run**. The gap between those extremes is where the
answer lies, and it is also where the language bridge actually sits.

Two arms, both at **depth 3**, since that is where outcome-only failed:

`regmachine-anneal` -- supervised for the first 20% of steps, gold program withdrawn
linearly to zero by 60%, answer-only for the last 40%.

> **Criterion.** Precondition: the arm reaches ~1.000 during the supervised phase (the
> supervised arm is 1.000 on 5/5, so a failure here means the schedule broke something
> and the test did not run). Then:
> - **`answer_acc` >= 0.9 after withdrawal** => the answer loss can *hold* a committed
>   program it could not *find*. The depth-3 failure is then a cold-start / exploration
>   problem, not a gradient-quality one -- and that is an actionable difference, because
>   cold starts have known fixes and bad gradients do not.
> - **collapse toward 0.015** => the outcome gradient cannot even maintain a correct
>   program at 7 instructions. That is a *stronger* negative than 3a-xii: it would mean
>   the answer loss is actively destructive at this width rather than merely
>   insufficient, and the differentiable executor's practical value would be confined
>   to very short programs.

`regmachine-partial42` -- a gold program for a hash-selected **42%** of problems and
nothing for the other 58%. Membership is decided per *problem* rather than sampled per
step, because that is how a dataset behaves -- per-step dropout would hand every problem
supervision eventually and measure something else.

> **Criterion.** **`answer_acc` >= 0.9** => 42% supervision suffices at 7 instructions.
> Below that, partial supervision does not carry the unsupervised remainder at this
> width, and the bridge needs either more coverage or a different mechanism for it.

**Corrected before the arm reported, not after.** The 42% was chosen because it was
GSM8K's measured program coverage when this was pre-registered. It is no longer: 3c's
compound decomposition and unit constants moved coverage to **0.680 train / 0.691 test**.
So the arm no longer *mirrors* the bridge -- it is a **lower bound** on it, which is the
more useful direction to be wrong in. If 42% suffices then 68% does with margin; if 42%
fails, a 68% arm is the follow-up rather than the conclusion.

The criterion itself is untouched, and the arm was already running when the mismatch was
noticed. Rewriting a threshold after seeing an outcome is the failure this project logs
at 3a-i and 3a-vi; rewriting a *rationale* before any number exists is just bookkeeping,
and the distinction is only meaningful if the correction is timestamped by the run it
precedes.

Both criteria are fixed before running. Note what is deliberately *not* claimed: the
anneal schedule (20%/60%) and the 42% ratio are single points, not swept, so a negative
result at these settings does not establish that no schedule works -- only that the
obvious one does not. That distinction is the one 3a-ii failed to make about RL.

### 3a-xviii. Program induction on varied structure: both verdicts hold

3a-xv showed every prior induction measurement was taken on a task with exactly one
pointer pattern, so it measured constant recall. `d3g1s` (unbalanced trees, `shape=1`)
removes that: operand counts vary 5/7/9, instruction counts 3/5/7, 4 distinct pointer
patterns at depth 3 and 25 at depth 4. 5 seeds, 1000 steps.

| arm | mean | sd | `canonical_acc` | `ptr_acc` |
| --- | --- | --- | --- | --- |
| `regmachine-supervised` | **1.000** | 0.000 | **0.761** | 0.898 |
| `regmachine-answer-only` | 0.075 | 0.055 | 0.000 | 0.025 |

Paired difference -0.925, seed ranges disjoint, permutation *p* = 0.0079.

Two things this settles.

**Supervised program emission was not constant recall.** It reaches 1.000 on every seed
with varying operand counts and varying structure, and `canonical_acc` is 0.761 rather
than the 1.000 of the balanced task: the model reaches the right answer on problems whose
program it does not reproduce exactly. That is the first capability result in this repo
measured where the program actually varies.

**Outcome-only induction still fails, so 3a-xii was not an artifact of the degeneracy.**
0.075 soft, 0.034 when the argmax program is executed, against 1.000. It is higher than
the 0.015 on balanced trees, which is the wrong direction for a "the task was too easy"
explanation and consistent with the soft mixture flattering an uncommitted model.

What remains untested is the refusal signal, which needs a competent-and-fallible regime.
The varied task has one: supervised passes through intermediate accuracy over more steps
than the balanced task's five-step knife-edge, so `self_consistency` can finally be
measured somewhere it means something.

### 3a-xix. The answer's own sharpness is a calibrated refusal signal

3a-xvii proposed agreement across resampled programs as the refusal signal. On varied
structure it does not hold: separation swings from +0.38 to -0.22 across checkpoints with
no trend, and at step 140 *every* problem where the samples disagreed was answered
correctly. Two different programs compute the same function (the 48-programs point of
3a-vii), so program-level disagreement is expected even when the model is right.

The signal that does work reads the quantity the answer is actually decoded from: how
peaked the **output register's residue distribution** is. Mean max-probability across
moduli, no sampling, no labels, no verifier.

Measured across training on `d3g1s` (accuracy rising 0.139 -> 1.000):

| accuracy | `acc` given confident | coverage |
| --- | --- | --- |
| 0.139 | **1.000** | 0.006 |
| 0.338 | **1.000** | 0.127 |
| 0.652 | **1.000** | 0.488 |
| 0.738 | **1.000** | 0.521 |
| 0.879 | **1.000** | 0.545 |
| 0.908 | **1.000** | 0.844 |

**Precision is 1.000 at every checkpoint, and coverage tracks competence.** The signal
refuses almost everything when the model knows almost nothing and accepts 84% once it is
good, without being told which is which.

Threshold sweep at 260 steps, 3 seeds. Two reached 1.000 accuracy and so cannot test
precision; the informative seed sits at 0.871:

| threshold | precision | coverage |
| --- | --- | --- |
| 0.5 | 190/196 | 0.77 |
| 0.7 | **183/183** | 0.71 |
| 0.9 | **180/180** | 0.70 |

The threshold is load-bearing: 0.5 leaks six wrong answers, 0.7 leaks none.

**Why this works, and why a language model cannot have it.** The answer is *composed* by
exact arithmetic from the operands via the emitted program. Composition of distributions
is exact, so the output distribution's sharpness is a faithful readout of the program's
determinacy: a peaked result can essentially only arise from a coherent program over
coherent operands. The calibration is a property of the executor, not of a learned
confidence head, which is why it needs no calibration data and why it should survive
leaving the distribution a verifier was written for. A model that emits an answer as
tokens has no such quantity to read.

This is the one thing a design that refuses to expose its reasoning actually needs:
self-knowledge without readability. Unlike the verifier it needs no ground truth, and
unlike redundant residues (3a-xiii) it catches a *wrong* answer rather than only an
ill-formed one.

**The adversarial case, which is where it matters.** The answer-only arm answers 92.5% of
its errors with a confident wrong number, and redundant residues recover 1.3% of those
(3a-xiii). Three seeds at 600 steps, against three supervised seeds for contrast:

| arm | accuracy | coverage at t=0.9 | accepted |
| --- | --- | --- | --- |
| answer-only, seed 0 | 0.027 | **0.00** | nothing |
| answer-only, seed 1 | 0.023 | **0.00** | nothing |
| answer-only, seed 2 | 0.098 | 0.07 | **19/19** |
| supervised, 3 seeds | 1.000 | 0.71-0.77 | **570/570** |

**The signal discriminates models, not only problems.** It declines an untrustworthy
model almost wholesale and accepts three quarters of a competent one. Two of the three
answer-only seeds accept *nothing at all* -- which is the correct answer for a model at
0.02 accuracy, and the behaviour redundancy could not produce.

Tallying every measurement so far at threshold >= 0.7: **zero wrong answers accepted**,
across 7 training checkpoints, 3 answer-only seeds and 6 supervised seeds. The only
observed leak is at threshold 0.5, where six errors passed on an 0.871-accuracy model.

Limits, stated. Precision rests on the checkpoint sweep and the answer-only arm, because
every supervised seed trained to 1.000 and had no errors left to catch; getting supervised
false-accept evidence needs a task that plateaus below perfect. Coverage is the price
rather than the precision: at 0.871 accuracy it declines 30% of problems it would have
answered correctly, and at 0.027 it declines the 2.7% it would have got right too.

### 3a-xvii. Self-knowledge without readability: agreement under resampling

A design that never decodes its latents has spent its interpretability, so the only
safety property left is the model telling you when not to trust it. Redundant residues
cannot do that (3a-xiii: they catch an ill-formed code, and the failures are well-formed
codes for the wrong program), and the exact verifier cannot do it off-distribution,
which is the only regime that matters.

`self_consistency` draws `n_samples` **discrete** programs from the model's own
distribution through the Gumbel path, executes each exactly, and asks whether they agree.
No labels and no verifier, so it survives leaving the distribution the checker was
written for. It is the per-problem version of `ptr_sharp`, which at arm level already
separates the outcomes perfectly -- but it needs no access to the logits, so it transfers
to any emitter.

**The one valid measurement, depth 3 supervised at step 50 (`acc` 0.559):**

| | rows | accuracy |
| --- | --- | --- |
| 8 samples unanimous | 17 | **1.000** |
| samples disagree | 239 | 0.527 |

**Separation 0.473, and unanimous agreement is 17 for 17.** As a refusal rule it is
high-precision and low-recall: when it says trust this, you can, and it says so on 6.6%
of problems. Refusing on any disagreement also discards the 53% of disagreeing rows that
were right, so it is a usable *precision* signal and not yet a usable *coverage* one.

**Why there is exactly one valid row, and it is the interesting part.** The fallible
window is a knife-edge: 0.035 at step 45, 0.559 at 50, 1.000 at 55. And every harder
setting tried -- two-digit operands, multiplication, **depth 4 at 16 operands and 15
instructions** -- reaches 1.000 by step 200.

That is the constant-pointer degeneracy of 3a-xv showing up from a new direction. There
is no partial competence because there is nothing partial to learn: one address pattern,
memorised in a handful of steps. It also explains a failure recorded much earlier -- 3a-i
could not get a `boundary_probe` number because it needed a model that was "competent
*and* fallible (~50-70%), which this tiny model on these tasks does not provide." The
reason is not the model's size. It is that the task has no middle.

So the confidence signal cannot be properly evaluated until the program structure varies
(the `shape` axis, 3a-xv) or on the bridge, where difficulty is continuous. What is
established here is that the mechanism works where it can be measured at all, and that
its precision is the part worth having.

**A metric bug found on the way, and it is the same shape as the `nan` one.** The first
run reported `separation 1.000` at every step past 80, which reads as the signal working
perfectly. It was vacuous: `acc_split` was 0 because **no rows disagreed**, and the
`max(1, n_split)` denominator turns an empty set into a perfect score. `self_consistency`
now returns `n_agree`, `n_split` and a `valid` flag requiring at least eight of each,
because a measurement that cannot distinguish "perfect" from "not applicable" will report
the first.

### 3a-xvi. Forcing commitment: the pre-registered fork resolves against the hypothesis

3a-xii found the answer-only failure is *indecision*, not error -- every seed at
`ptr_sharp` 1.0 scored 1.000, every seed below it ~0.04, nothing between. 3a-xiv
pre-registered the fork: if commitment is the **cause**, forcing it should fix accuracy;
if pointers go sharp while accuracy stays near zero, commitment is a **symptom** and the
outcome gradient genuinely points the wrong way.

Depth 3, 5 seeds, `program_coef=0` throughout, GPU+CPU hybrid:

| arm | mean | sd | `ptr_sharp` | verdict |
| --- | --- | --- | --- | --- |
| `answer-only` (control) | 0.011 | 0.009 | 0.862 | — |
| `answer-only-gumbel` | 0.023 | 0.010 | **0.996** | commitment supplied |
| `answer-only-commit` (re-run, fixed) | 0.021 | 0.006 | 0.941 | commitment supplied |
| ~~`answer-only-commit` (original)~~ | ~~0.024~~ | — | **nan** | **void, see below** |

**Straight-through Gumbel made the pointers sharp -- 0.996 against the control's 0.862 --
and accuracy did not move: 0.023 against 0.011, permutation *p* = 0.087, CI crossing
zero.** The entropy arm, re-run correctly, lands in the same place: `ptr_sharp` 0.941,
accuracy 0.021. Both interventions supply commitment; neither supplies competence. The
~0.01 they add is at the edge of what 5 seeds resolves and is irrelevant against a gap of
0.985. The fork resolves to *symptom*. Commitment is not the missing ingredient, and
the outcome gradient does not locate a correct program at seven instructions even when
it is handed a decision.

That reinterprets 3a-xv's positive result. The warm start does not work by supplying
*commitment*; it works by supplying the **right program**. Which means the exact
generator has to stay in the loop permanently -- outcome-only learning will not bootstrap
structure, at any sharpness.

#### The entropy arm is void, and how it announced itself

`answer-only-commit` reported **+0.013 at permutation *p* = 0.0317, CI excluding zero**,
and that number is **withdrawn**: its weights were `nan` from **step 0**, so the accuracy
was decoded from a destroyed model.

The mechanism is worth writing down because every layer of it looked fine:

- A masked pointer slot has `lp = -inf`, so `p * lp` computes `0 * -inf` = `nan`.
- `nan_to_num(0.0)` repaired the **forward** value and does nothing to the **backward**,
  so a `nan` gradient reached `clip_grad_norm_`, which turned the whole gradient `nan`,
  which turned every weight `nan` on the first optimiser step.
- The printed loss stayed **finite** (3.22) because the forward was clean.
- The entropy term then reported **0.0000** -- because `nan_to_num` was also swallowing
  the evidence that anything was wrong.
- `evaluate` still returned an accuracy, and the permutation test still called it
  significant.

So a destroyed model produced a statistically significant result, and the only symptom
anywhere in the output was `ptr_sharp` reading `nan` -- a metric that exists because
3a-xii needed it, added for an unrelated reason a few hours earlier.

**And the accuracy could not have caught it.** Re-run with the fix, the arm gives
**0.021** against the void run's **0.024**. At near-zero accuracy a destroyed model and a
working one are indistinguishable, because both are decoding noise -- so the number that
would normally expose a broken run was, here, almost exactly right. That is the general
hazard: a metric only detects a fault when the fault would move it, and in a failing
regime almost nothing moves it.

Two fixes. The entropy is computed on `log_softmax(...).clamp_min(-30)` rather than
`nan_to_num`, so both passes stay finite and the masked terms contribute ~1e-13 x -30.
And `train_step` now **raises** on a non-finite loss or gradient, because a trainer that
silently continues on wreckage keeps printing numbers and the numbers pass significance
tests. A crash is cheaper than a void result that looks like a finding.

The arm is re-run with the fix; the entropy penalty is also floored at a target entropy,
since minimising entropy is unbounded below in logit space and would otherwise drive the
logits apart forever.

### 3a-xv. The curriculum arms: both criteria pass, and the task is easier than it looked

Depth 3, 5 seeds, everything else as 3a-xii:

| arm | mean | sd | `canonical_acc` | `mis_answered` |
| --- | --- | --- | --- | --- |
| `regmachine-supervised` | 1.000 | 0.000 | 1.000 | 0.000 |
| **`regmachine-anneal`** | **1.000** | 0.000 | 1.000 | 0.000 |
| **`regmachine-partial42`** | **1.000** | 0.000 | 1.000 | 0.000 |
| `regmachine-answer-only` | 0.015 | 0.013 | 0.000 | 0.985 |

Both pre-registered criteria (3a-xiv) pass: the anneal arm holds 1.000 through 400 steps
with the gold program fully withdrawn, and 42% problem-level supervision is as good as
100%. `answer_acc_hard` is 1.000 for both, so this is not a soft-readout artefact.

**And then the check that should have come first.** At fixed depth the grammar builds a
*balanced* tree, so the post-order traversal is identical for every problem — measured,
**exactly 1 distinct pointer pattern across 300 depth-3 problems**, with only the
operators varying (118 patterns of a possible 128). The program the model must emit is a
**constant** 7-instruction address structure plus seven binary operator reads.

That changes what these numbers mean, and mostly downward:

- **`partial42` says much less than it appears to.** Supervision on 42% of problems
  transfers to the other 58% — in a setting where the program is *the same program*.
  It shows 42% is enough to learn a constant. GSM8K's programs genuinely differ per
  problem in length, structure and operand addresses, so this arm does **not** license
  the inference its criterion was written to support. The criterion is satisfied
  literally; the conclusion I wanted from it does not follow. Recorded rather than
  reinterpreted.
- **`anneal` survives, but narrower.** What the answer loss holds without labels is a
  constant pattern, not a per-problem program. The gap it bridges is nonetheless real:
  the answer-only arm *fails to find this same constant*, so 0.015 → 1.000 is genuinely
  attributable to the warm start.
- **It also explains `sd` = 0.000 everywhere supervision is present.** A constant is
  memorisable, which is why every supervised variant is perfect on every seed at both
  depths. That is not evidence of a robust learner.

**The one result it strengthens is the negative one.** Outcome-only induction cannot
discover a pointer pattern that is *identical for every problem in the task* — 675
choices per instruction over seven instructions, with the same answer every time, and
it lands at 0.015. That is a worse verdict than 3a-xii stated, not a better one.

**What this makes necessary:** a task whose program structure actually varies. The
balanced-tree grammar cannot provide one, and `n_instr` was derived from depth until
3a-xvi, so nothing here could express a short program. Variable-length or unbalanced
expressions are the prerequisite for any claim about program *induction* as opposed to
constant-pattern recall — including everything 3a-vii ever said.

### 3a-xiii. Redundant residues on a trained model: the prediction fails

3a-ix measured `RedundantResidueSystem` standalone — 100% of single-residue
corruptions corrected, 0% mis-corrected — and nothing imported it. The depth-2 run in
3a-xii gave the reason to: the answer-only arm's `mis_answered` was **0.442**, i.e.
44% of held-out problems answered with a confident wrong number, with no checker in the
path at all.

So the arms were re-run with three redundant moduli (`(7,13,41)` on top of the core
`(16,25,27,11,37)`; legitimate values then occupy 2.198e6 of an 8.200e9 ring, 0.027%).
**Prediction, written into the code before the run: `mis_answered` collapses while
`refused` absorbs it.**

| arm | `answer_acc` | `refused` | `mis_answered` |
| --- | --- | --- | --- |
| `regmachine-supervised-redundant` | 1.000 (5/5) | 0.000 | 0.000 |
| `regmachine-answer-only-redundant` | 0.540 (sd 0.423) | **0.013** | **0.446** |

**The prediction is wrong.** Redundancy converted 1.3% of the error budget into
refusal and left 44.6% as silent wrong answers — statistically indistinguishable from
the 0.442 it was supposed to fix.

**Why, and this is the part worth keeping.** Redundant residues detect a *corrupted
codeword*: flip one residue and the reconstruction leaves the legitimate range. A
random incoherent vector should be caught ~99.97% of the time in this ring, and the
1.3% refusal rate says incoherent vectors are rare here. The failing seeds are not
emitting corrupted codes. At `ptr_sharp` 0.86–0.89 their pointers are sharp enough that
the program executes *coherently* — it is simply a **different program**, and its result
is a perfectly legal integer that happens to be wrong.

**Redundancy cannot detect a wrong answer, only an ill-formed one.** That distinction
was implicit in 3a-ix and 3a-x and is now explicit, because it is the difference between
the machinery being useful here and not. Both earlier measurements stand exactly as
stated — single-residue corruptions and incoherent soft reads *are* caught — they just
do not describe the error this model actually makes. The README's "the failure mode is
refusal, never a wrong answer" is true of the **code** and is not a guarantee about the
**computation**; it has been corrected to say so.

What the run does establish, and it is not nothing: **the wider code is free.** The
supervised arm is 1.000 on all 5 seeds with three extra residue blocks to predict (8
instead of 5) and `mis_answered` exactly 0. So redundancy costs no accuracy and can be
carried wherever an ill-formed code is the risk — a corrupted store, a soft read, a
transmission — which is a narrower and more honest claim than the one this project was
heading toward.

Open, and the obvious next question: a checker for a *coherent but wrong* answer cannot
come from the code space. It has to come from recomputation — the root-slot consistency
idea of 3a-vi, which failed for a different reason (the model never learned the root), or
from executing a second, independently-emitted program and comparing. Neither is built.

### 3a-xii. Pre-registered: does outcome-only induction survive depth 3?

3a-vii is the one learned claim in this project still standing, and the one the
language bridge depends on, and it is `n=1`: depth 2, four operands, three
instructions, `(+,−)`, one seed, run from a notebook cell. `lamb/study.py` exists
precisely to stop claims of that shape and had never been pointed at it. It is now:
`--arms regmachine-supervised regmachine-answer-only`.

**The criterion, fixed before running.** Outcome-only induction (`program_coef=0`)
reaches held-out `answer_acc` within noise of the supervised arm at **depth 3**, over
≥5 seeds, by the exact permutation test in `lamb/study.py`. Read `answer_acc`;
`canonical_acc` is conformity to the generator's form and is *expected* near zero for
the answer-only arm, because 48 distinct three-instruction programs compute
`(a+b)+(c+d)` and the grammar emits one of them. Low canonical agreement does not
imply a wrong program — that inference has never held here.

Two things that must be read alongside the number, or it means less than it looks:

- **`dropped`.** Depth 3 and multiplication leave the ring, and until 3a-xi this
  trainer trained on the wrapped value. Rows outside the ring are now masked and the
  share is reported; an accuracy measured on 60% of a held-out set is not the same
  number as one measured on all of it.
- **The search space.** Pointer choices per instruction grow as `|ops| · n_slots²` —
  3·7² at depth 2, 4·15² at depth 3 with division. This is not a small extrapolation
  and a negative result is a real one, to be reported with the same energy as a
  positive one.

Division (`ops_key=2`, task `d2g1d`) is a second arm rather than part of this one:
it needs rational registers, so the ring changes with it, and two changes in one
comparison is the confound this project keeps retracting claims over.

#### Depth 2 first, as a replication. It does not replicate.

Before the pre-registered depth-3 run, the same arms at the depth 3a-vii actually
measured. 5 seeds, 1000 steps, batch 64, `d_model=96`, moduli `(16,25,27,11,37)`:

| arm | n | mean | sd | per seed |
| --- | --- | --- | --- | --- |
| `regmachine-supervised` | 5 | **1.000** | 0.000 | 1.000 ×5 |
| `regmachine-answer-only` | 5 | **0.558** | **0.405** | 1.000, 1.000, 0.303, 0.283, 0.203 |

*(Read `answer_acc_hard` below for the stricter and more honest version of this row:
the arm mean is **0.422**, and the three failing seeds are at ~0.04 rather than ~0.25.)*

**The `n=1` answer-only result in 3a-vii was a lucky seed.** It reported 1.000 and
concluded that *"gradients through exact arithmetic are sufficient to induce a correct
program from outcomes alone."* At 5 seeds that outcome occurs **twice**; the other
three land at 0.203–0.303. The supervised arm, by contrast, is 1.000 on every seed
with sd exactly 0.

The honest claim is therefore **"outcome-only induction can find a correct program but
does so unreliably"**, not "is sufficient". Two of five is a real capability and a
poor foundation.

Diagnostics, which say what the failing seeds are doing:

| | supervised | answer-only |
| --- | --- | --- |
| `canonical_acc` | 1.000 | 0.000 |
| `op_acc` | 1.000 | 0.547 |
| `ptr_acc` | 1.000 | **0.037** |
| `ptr_sharp` | 1.000 | 0.886 |
| `mis_answered` | 0.000 | **0.442** |

`canonical_acc 0.000` is expected and not a failure (48 programs compute
`(a+b)+(c+d)`). With no redundant moduli configured, the 0.442 `mis_answered` are
silent wrong answers, which is precisely the case 3a-ix exists for and which no arm
here had switched on.

**Per seed, `ptr_sharp` separates the two basins exactly, and that is the whole
finding.** No extra compute; it was already in the run's own JSON:

| arm | seed | `answer_acc` | `ptr_sharp` | `op_acc` | `canonical_acc` |
| --- | --- | --- | --- | --- | --- |
| answer-only | 0 | **1.000** | **1.0000** | 0.510 | 0.000 |
| answer-only | 3 | **1.000** | **1.0000** | 0.655 | 0.000 |
| answer-only | 1 | 0.303 | 0.7663 | 0.520 | 0.000 |
| answer-only | 2 | 0.283 | 0.7744 | 0.456 | 0.000 |
| answer-only | 4 | 0.203 | 0.8913 | 0.596 | 0.000 |
| supervised | all 5 | 1.000 | 1.0000 | 1.000 | 1.000 |

Every seed that reached `ptr_sharp = 1.0000` scored exactly 1.000. Every seed that did
not scored below 0.31. There is no middle.

**So the failure mode is not a wrong program, it is an *uncommitted* one.** That is a
much sharper statement than "high variance", and it is mechanically consistent with
3a-x: a soft pointer read is not a blend, because decoding argmaxes each modulus
independently, so ~30% of blurred reads decode to a value in neither register. An
undecided program cannot be right even when its intent is right.

Note also that the two successful seeds have `canonical_acc 0.000`. They did not
recover the grammar's program; they found a *different* program that is exactly
correct — which is the 3a-vii non-uniqueness point holding up, and the reason
`answer_acc` is the only correctness measure here.

#### Pre-registered: is commitment the cause, or a symptom?

`RegisterMachine` has carried straight-through Gumbel-Softmax on the discrete choices
(`tau`, `hard`) since it was written, off by default, with the note that *"whether the
discreteness is needed is a question to measure, not to assume."* The table above is
the motivation for measuring it: straight-through keeps the forward pass discrete, so
the executor sees a real program rather than a blur of several, which is exactly what
the failing seeds lack.

The confound to be honest about first: `ptr_sharp = 1` may be a *consequence* of having
found a correct program rather than a cause of it — confidence following correctness.
Forcing discreteness is the intervention that separates the two, because it supplies
commitment without supplying any information about *which* program is right.

**Criterion, fixed before running.** Arm `regmachine-answer-only-gumbel`
(`program_coef=0`, `tau>0`, `hard=True`), 5 seeds, `d2g1`, everything else identical:

- **Commitment is the cause** if the arm reaches `answer_acc` ≥ 0.9 on at least 4 of 5
  seeds. The mechanism would then be that outcome-only induction finds correct
  programs routinely and loses them to indecision.
- **Commitment is a symptom** if pointers go sharp (`ptr_sharp` ≈ 1) while accuracy
  stays near the 0.2–0.3 band. The gradient would then genuinely be pointing the wrong
  way on those seeds, and outcome-only induction is weaker than even the corrected
  reading above suggests.
- **Neither, and the test did not run**, if `hard=True` destabilises training so that
  `ptr_sharp` does not reach ~1. A precondition failure is not a licence to
  reinterpret the outcome — that error is logged at 3a-vi and again two sections up.

This is a question about the *executor*, which is the exact part of the design that
distinguishes it from PAL/Program-of-Thought, so it is worth a clean answer either
way.

**On the statistics, including where my own criterion was badly written.** The paired
difference is −0.442 with a 95% CI of [−0.745, −0.139] that excludes zero, but the
exact permutation test gives *p* = **0.167** and the sign-flip test *p* = **0.250**.
By this project's own rule — permutation tests, not intervals — that is *not*
significant. The reason is the arm's own variance: at sd 0.405 the smallest paired
difference 5 seeds can resolve is **0.513**, and the study prints that it would take
**526 seeds** to resolve 0.05. Two of the five paired differences are exactly zero,
which weakens the paired test further.

So the criterion I pre-registered above — "within noise of the supervised arm ... by
the exact permutation test" — is **wrong as worded**, and I am not going to read a
pass out of it. A non-significant test is not evidence of equivalence, and a design
that could not have detected a 50-point difference cannot certify a 44-point one as
noise. That is the same error as 3a-i, where the boundary was never compared against a
converged baseline: a test whose precondition fails did not run.

What survives without any test is a description: three of five answer-only seeds
scored 0.203–0.303, and no supervised seed came within 0.7 of that. The claim to make
is about **reliability**, and reliability is what the n=1 result could not have seen.

**This makes the GSM8K annotation finding (3c) more valuable, not less.** The reliable
arm is the supervised one, and GSM8K supplies exactly that supervision for 42% of its
train split — so the bridge does not have to depend on the mechanism that just failed
to replicate.

#### The pre-registered depth-3 verdict: outcome-only induction does not scale.

5 seeds, 1000 steps, same settings, depth 3 (8 operands, **7 instructions**, 15
registers):

| arm | n | mean | sd | per seed |
| --- | --- | --- | --- | --- |
| `regmachine-supervised` | 5 | **1.000** | 0.000 | 1.000 ×5 |
| `regmachine-answer-only` | 5 | **0.015** | 0.013 | 0.029, 0.021, 0.021, 0.002, 0.000 |

Paired difference **−0.985**, 95% CI [−0.995, −0.975], **seed ranges disjoint**, exact
permutation *p* = **0.0079**, sign-flip *p* = 0.0625 (its floor at 5 seeds).

**The criterion fails, and unlike depth 2 this test was powered to say so.** The
smallest paired difference 5 seeds could resolve here is **0.017**; the observed one is
0.985, fifty-eight times that. Both arms need only 2 seeds for a 0.05 difference. There
is no underpowering to hide behind and no reinterpretation to make.

So: **gradients through exact arithmetic are not sufficient to induce a correct program
from outcomes alone.** They were sufficient on two of five seeds over *three*
instructions and on none over *seven*. The mechanism that ROADMAP 3a-vii called "the
one thing in this design that a non-differentiable executor cannot do" does work — it
just does not survive the step from a 3-instruction program to a 7-instruction one.

That is not surprising in hindsight and the pre-registration said so before the run:
pointer choices per instruction grow as `|ops| · n_slots²`, which is 3·7² = 147 at
depth 2 and 3·15² = 675 at depth 3, over seven instructions instead of three. What the
run establishes is that the answer signal alone does not navigate it.

Meanwhile **the supervised arm is 1.000 on every seed at both depths**, with `sd`
exactly 0 and `canonical_acc` 1.000. The executor, the algebra, the pointer masking and
the register file all scale to depth 3 without a wobble. It is specifically the
*outcome-only* learning signal that does not.

**The soft-eval caveat is closed at this depth too, and the verdict does not move.**
Executing the argmax program rather than the soft mixture gives `answer_acc_hard`
**1.000** for the supervised arm (identical — `ptr_sharp` is exactly 1, so soft *is*
argmax) and **0.025** for answer-only against its soft 0.015. At depth 2 hardening
*hurt* the failing seeds (0.558 → 0.422); here it helps them negligibly (0.015 →
0.025). Either way both numbers round to "nothing", so the readout is not what produced
this result — which is the only thing the check needed to establish.

The answer-only arm's `mis_answered` is **0.985** — essentially every held-out problem
answered with a confident wrong number, since no redundancy was configured. `op_acc`
sits at ~0.50 against a 0.33 chance baseline, so the operators are partly learned while
the program as a whole is not, and `ptr_sharp` is 0.87–0.95: as at depth 2, the failing
runs never commit.

**What this costs the project.** The language bridge was described in 3a-vii as
depending on this mechanism, because a word problem supplies no gold program. On this
evidence that dependency is not safe, and the bridge should not be built on it. Two
things carry the weight instead, both established tonight: GSM8K's calculator
annotations supply a real program for 42% of the train split (3c), and the *supervised*
arm is the one that is perfectly reliable. The honest plan is supervision where it
exists and an answer-only arm reported as a control, not as the mechanism.

#### The soft-eval caveat, tested: it went the other way, and the finding was understated

`answer_acc` executes the *soft* program, so a blurred-but-recoverable program would be
scored on a mixture it never computes — which would make the whole result an artefact of
the readout rather than a fact about the learning. The argmax program is now executed as
a real program and reported as `answer_acc_hard`. Depth 2, per seed:

| seed | `ptr_sharp` | `answer_acc` (soft) | `answer_acc_hard` |
| --- | --- | --- | --- |
| 0 | 1.0000 | 1.000 | **1.000** |
| 3 | 1.0000 | 1.000 | **1.000** |
| 1 | 0.7663 | 0.303 | **0.043** |
| 2 | 0.7744 | 0.283 | **0.041** |
| 4 | 0.8913 | 0.203 | **0.027** |

**Hardening does not rescue the failing seeds; it exposes them.** The committed seeds
are identical either way (`ptr_sharp` is exactly 1, so soft *is* argmax). The
uncommitted ones drop from 0.20–0.30 to **0.027–0.043** — the soft score was flattering
them by roughly sevenfold, because a mixture can land on the right answer by spreading
mass where the single program it would actually emit does not.

So the caveat resolves against itself and **the earlier reading was too generous, not
too harsh**. Two corrections to what is written above:

- The failing seeds do not "collapse to ~0.25". They collapse to **~0.04**. The 0.25
  was an artefact of scoring a mixture, and the arm mean is **0.422** on the stricter
  reading rather than 0.558.
- Depth 2 and depth 3 are therefore the *same* picture, not two different ones. At both
  depths the answer-only arm either finds the program (1.000) or essentially fails
  (~0.04 at depth 2, 0.015 at depth 3). **There is no partial credit in outcome-only
  induction** — the intermediate scores were measurement, not learning.

That also completes the case for `ptr_sharp` as the diagnostic: sharp gives 1.000, blurred
gives ~0.04, and nothing sits between. And it is a reminder in the other direction from
this project's usual one — the check was run expecting to weaken a negative result, and
it strengthened it.

### 3a-ix. Redundant residues: correcting errors, not just detecting them

The harshest constraint on the residue representation is that **CRT has no
locality**. One wrong residue does not give a nearby number, it gives an essentially
uniform one, so an answer's accuracy is roughly the per-residue accuracy raised to
the number of moduli. At the 0.985 per-residue accuracy measured in 3a-vi, seven
moduli give **0.900** — a 10% error rate produced entirely by the encoding, on top
of whatever the model gets wrong.

The classical fix costs width and nothing else. Size the **core** moduli so
legitimate values occupy only part of the ring and carry extra **redundant** ones.
A corrupted residue throws the reconstruction outside the legitimate range, so it is
detected; and because the survivors still over-determine the value, dropping each
modulus in turn identifies *which* residue was wrong and recovers the value exactly.

    q**K                      -> detection only
    q**K + K q**(K-1) (1-q)   -> single error corrected

At the measured `q = 0.985`, `K = 7`: **0.900 → 0.996**, a 25× reduction in error
rate. Nothing is learned and no label is required, which is what makes it usable at
inference on a benchmark with no verifier — the case an exact checker cannot reach.

Measured over 4,000 single-residue corruptions per row, core `(16,25,27,11)`:

| redundant moduli | units | corrected | refused | **mis-corrected** | 2-error silent |
| --- | --- | --- | --- | --- | --- |
| `(37,7)` | 123 | 84.3% | 15.7% | **0.00%** | 33.5% |
| `(37,7,41)` | 164 | **100.0%** | 0.0% | **0.00%** | 0.1% |
| `(37,7,41,101)` | 265 | 100.0% | 0.0% | **0.00%** | 0.0% |

Three redundant moduli is the sizing: full correction of single errors, and two
simultaneous errors go silent only 0.1% of the time.

**The column that matters is mis-correction, and it is zero throughout.** Under-
provisioned redundancy loses corrections — it never invents one. When the evidence
does not single out a culprit, `correct` returns `None` rather than picking the most
plausible candidate. That is deliberate: this representation is being used precisely
where nothing downstream can catch a wrong answer, so the failure mode has to be an
admitted failure rather than a confident number. It is the same reasoning that made
out-of-range values masked rather than clipped in 3a-vi.

## 3b. Latent inter-agent communication (Coconut) — Stage B implemented

Implemented in `lamb/comm.py` (`python -m lamb.comm`). The message passed between
two agents *is* a Coconut thought vector: the speaker rolls latent thoughts from
its half of a split problem, a differentiable `Channel` carries those continuous
vectors, and the listener consumes them as latent input positions (receiving =
thinking another agent's thought) and answers — reusing Stage A's exact
continuous-input path. No token is ever exchanged; the channel is a pure-latent
blackbox.

**Split task**: the speaker sees operand `X`, the listener sees `op` and `Y` and
must emit `X op Y` — impossible without the message. Training is one joint
objective (the listener's answer cross-entropy) back-propagated *through the
message* into the speaker: differentiable inter-agent learning (DIAL,
arXiv:1605.06676) with Coconut thoughts as the message — a combination the recent
latent-agent-communication papers (Interlat arXiv:2511.09149, DiffMAS
arXiv:2604.21794) approach but none train as the Coconut recurrence driven
end-to-end across the agent boundary.

Measured (tiny CPU pair, 1-digit `X,Y`, `+/−`, noiseless channel): exact-match
`comm 1.000` vs a zeroed-message ablation `0.098` (the guess-`X` prior) — a
`+0.90` causal channel gain (the positive-*listening* test of arXiv:1903.05168).
Two agents solve, purely in latent space, a task neither can solve alone. A
`channel_noise>0` (DIAL DRU) bandwidth sweep shows a sharp capacity threshold
(width `1/2/4` at the `~0.10` prior, `8/96 → 1.000`: a 1-digit operand needs `≥8`
noisy channel dimensions); message diagnostics (`signal_std`, `msg_cos`) guard
against the representational collapse of arXiv:2604.03809. `python -m lamb.comm --sweep`.

**Held-out-partner test** (`python -m lamb.comm_transfer`). Is the emergent code a
private co-adaptation or a shareable protocol (zero-shot coordination; Other-Play,
arXiv:2003.02979)? Two independent pairs each reach `comm 1.000`, but a zero-shot
speaker swap collapses to `0.004/0.041` — *below* the `0.076` blank prior (a
foreign code actively misleads); yet a freshly trained receiver learns a frozen
speaker's code to `1.000`. So the protocol is **private and strongly co-adapted
but learnable** — idiosyncratic, not canonical, exactly as Other-Play predicts.

**Partner randomization** (`python -m lamb.comm_pop`) is the fix, and it works. A
population of `P` speakers and `Q` listeners is trained with random pairing, the
diagonal `(i, i)` pairings held out of training and evaluated zero-shot. On a 3×3
CPU population the held-out (never-co-trained) pairings reach zero-shot `1.000`,
matching trained pairings and up from the single pair's `0.004` swap — the private
code becomes canonical. Other-Play realized in pure latent space.

**Re-measured on the clean split, and it holds.** `comm_pop.py` held out *pairings*
but not *problems* until the sweep in 3a-iii, so the original `1.000` was measured on
problems the population had trained on. Re-run with both held out (1600 steps, 3x3
population):

| | trained pairings | held-out zero-shot | blank |
| --- | --- | --- | --- |
| clean split | 0.941 | **0.935** (min 0.918, max 0.953) | 0.000–0.043 |

Zero-shot pairings perform **indistinguishably from trained ones** against a blank
prior at ~0. The headline number comes down from `1.000` to `0.935`, but the claim
it was making — partner randomization makes the latent code *canonical* rather than
private, so a never-co-trained pair coordinates as well as a trained one — is
exactly what survives, and this is now the only Stage A/B claim measured on an
uncontaminated split that did not shrink under scrutiny.

The 1-digit evaluation partition holds 22 problems, so this resolves to ~4.5 points;
`--a-digits 2` gives 2431 if a finer number is wanted.

Remaining: larger populations, heterogeneous agent sizes, and cross-architecture
transfer.

Refs: DIAL ([1605.06676](https://arxiv.org/abs/1605.06676)); pitfalls of measuring
emergent communication ([1903.05168](https://arxiv.org/abs/1903.05168));
zero-shot coordination / Other-Play ([2003.02979](https://arxiv.org/abs/2003.02979));
Interlat ([2511.09149](https://arxiv.org/abs/2511.09149)); DiffMAS
([2604.21794](https://arxiv.org/abs/2604.21794)).

## 3c. The language bridge (GSM8K) — implemented, unmeasured

`lamb/bridge.py` held a quantity parser, a resampler and a register loader, and no
path between them; `encode_dataset` was referenced twice in its own docstring and
defined nowhere. Nothing in the repo turned a sentence into a number.
`lamb/bridge_train.py` (`python -m lamb.bridge_train`, `lamb-bridge`) is that path.

    text -> frozen encoder -> cached embeddings     no
    text -> quantities -> registers                 no   (regex + fixed rational map)
    embeddings -> K latents (resampler)             yes
    latents -> program over registers               yes
    program -> answer                               no   (exact rational algebra)

The encoder runs **once** over the dataset and is cached to disk, so it never
participates in training and its size stops being a constraint. `transformers` is an
optional extra (`uv sync --extra bridge`) rather than a dependency: every arithmetic
number in this repo was produced under the pinned environment, torch moves accuracy
on a model this small between minor versions, and a text front-end should not be able
to shift the ground under a measurement it has nothing to do with.

**GSM8K ships gold programs, and that changes the shape of this.** The premise
recorded in 3a-vi and 3a-vii is that a natural-language problem does not come with a
program, so the bridge rested entirely on the `n=1` outcome-only result. That is true
of prose in general and **false of this dataset**: GSM8K's train solutions carry
inline calculator annotations of the form `<<48/2=24>>` — an operation, its operands
and its result, in evaluation order. So the bridge is a measurement with a control,
not a leap: the supervised arm recovers a program from the dataset's own annotations,
and `--program-coef 0` removes it entirely and asks 3a-vii's question on real text.

Three things are built into it because each one caps what any model on top could
reach, and all three are reported before training rather than inferred from a bad
number afterwards:

- **Coverage.** Only annotations that are a single binary operation over two
  literals are usable; compound ones (`<<48*3+5=149>>`) need a parser and more than
  one register write. Rejects are counted. So is `operand_miss` — the share of rows
  whose annotation names a number the extractor never found, which is the ceiling on
  the supervised arm and has nothing to do with the network.
- **Execution check.** A program recovered from someone else's annotations is a
  hypothesis. `verify_alignment` executes it in this ring and asks whether it
  reproduces the dataset's own answer; if it does not, training on it would teach the
  wrong program perfectly.
- **Constant registers.** "Half as many", "twice", "a third", "20% off" each name an
  operation whose other operand is never written down. Preloading `(1, 2, 3, 100)`
  keeps that decision in the program — `x / 2`, not a parser deciding that "half"
  means 0.5 — which is the same argument the percent flag already makes. By this
  repo's own note, "half as many" is the most common operation in GSM8K, so without
  the constants the alignment fails on problems the parser read perfectly.

**Measured: coverage, and one defect it caught.** Run before any training
(`python -m lamb.bridge_train --coverage`), on the official splits:

| | n | no_answer | answer_not_in_chain | operand_miss | program_coverage |
| --- | --- | --- | --- | --- | --- |
| train | 7473 | 0.000 | 0.054 | 0.316 | **0.614** |
| test | 1319 | 0.000 | 0.073 | 0.287 | **0.625** |

Recovered programs that execute to the dataset's own stated answer: **1.000**.

That last number was **0.824** on the first pass, and finding out why is the reason
the check exists. 16.6% of usable annotation chains **end somewhere other than the
answer** — the step that produced it was compound (`<<48*3+5=149>>`, which this
parser rejects) or was never annotated at all. Keeping those meant one recovered
program in six executed to the wrong number, which is worse than having no program:
it is supervision toward a wrong answer, and the model would learn it perfectly. The
chain is now truncated at the last step whose result *is* the answer, and a chain
that never reaches it is dropped. The execution rate went 0.824 → 1.000, and coverage
fell 0.533 → 0.420 to pay for it. **Less supervision that is correct beats more that is
not.** (Coverage later recovered to 0.614 for an unrelated reason — see the compound
decomposition below — and `answer_not_in_chain` fell 0.170 → 0.054 with it, because the
step that produced the answer was so often the compound one.)

**Ablated, because both of those components were added from reasoning rather than
evidence.** `--no-lexical` existed as the ablation and nobody had run it:

| constants | written numerals | `program_coverage` | Δ | what it buys |
| --- | --- | --- | --- | --- |
| `()` | no | 0.004 | — | |
| `(1,)` | no | 0.239 | **+23.5** | the identity register, not a fact about GSM8K |
| `(1,)` | yes | 0.318 | **+7.9** | "three", "a dozen" |
| `(1,2)` | yes | 0.397 | **+7.9** | "half", "twice", "double" |
| `(1,2,3)` | yes | 0.416 | +1.9 | "a third" |
| `(1,2,3,100)` | yes | **0.420** | **+0.4** | percent |

*(Measured before compound annotations were decomposed, so every row is ~19 points
below its current value. The ablation is a comparison between rows and the ranking is
unaffected; the absolute figures are not the headline coverage.)*

The first row is a confound in my own measurement and is called out rather than
reported: with no constants there is no register holding `1`, so the `x * 1` padding
instruction is unavailable and rows are rejected for a reason that has nothing to do
with the dataset. Isolating it costs one extra run and moves +23.5 points out of the
"constants help" column, where they did not belong.

With that removed, the ranking is: **the constant `2` and the written-numeral table are
worth about 8 points each**, and they are the two components that were guesses. `2` is
"half as many", which this repo's own note predicted would be the most common operation
in GSM8K, and the measurement agrees. `3` is worth 1.9.

**`100` is worth 0.4 and is nearly useless**, which is a negative result on a component
added by reasoning about percent: GSM8K's annotations write a percentage as a decimal
(`<<0.2*50=10>>`) rather than as `20/100`, so the constant is rarely what the chain
asks for. It is kept only because it still nets positive against the register slot it
occupies, and that is a thin margin rather than a justification.

**The largest single fix, found by asking what was actually in the failures.** The 39%
of usable chains that failed alignment break down as:

| cause | share |
| --- | --- |
| an operand produced by a **rejected compound step** | **52.1%** |
| integer operand absent from the text | 33.0% |
| non-integer operand absent from the text | 13.9% |
| register file full, a quantity pushed out | 0.9% |
| needed more than 8 instructions | 0.1% |

Half the gap was self-inflicted. A compound annotation (`<<48*3+5=149>>`) was rejected
for not being a single binary operation — but it is simply *two* instructions, and
rejecting it cascades: its result never reaches a register, so every later step that
reads it fails to align too. `_lhs_steps` now decomposes it with a recursive-descent
parser (precedence, left-associativity, parentheses, unary minus), and **coverage moved
0.420 → 0.614** with `exec_ok` still exactly 1.000.

Two things that breakdown settles, both of which were guesses in `BridgeConfig`: the
register file is essentially never the constraint (0.9%, measured at `n_operands=12`
with the four operation constants only -- it becomes 97.3% once unit constants are added,
which is the coupling below) and the instruction budget almost never is (0.1% at
`n_instr=8`). Chain length is median 3, p90
6, p99 9, max 15, so `n_instr=12` would add +1.0 point of coverage for 50% more
execution per step — measured and declined.

**Then the same question again, and the answer changed.** Re-categorising what *still*
failed after decomposition:

| cause | share |
| --- | --- |
| **integer operand absent from the text** | **59.6%** |
| non-integer operand absent from the text | 27.7% |
| operand from a still-rejected step | 5.7% |
| register file full | 3.8% |
| chain longer than 8 instructions | 3.2% |

The leader is not an extraction failure at all. *"Weng earns $12 an hour ... 50
minutes"* needs **60**, and no parser can extract a number the problem never writes.
That is unit knowledge, and it is the same shape as "half as many" needing a `2` — so
it gets the same answer: `DEFAULT_CONSTANTS` gains `60, 24, 7, 12, 52, 1000`, and
coverage moves **0.614 → 0.680** train, **0.691** test, `exec_ok` still 1.000.

**And the naive version of that change is actively harmful, which is the finding worth
keeping.** Every constant occupies a register slot. At `n_operands=12` those ten
constants fill **97.3%** of register files and push the problems' own quantities out:
coverage does not improve, it *halves*, 0.614 → **0.249**. Constants and file width are
one decision:

| constants | `n_operands` | coverage | file full |
| --- | --- | --- | --- |
| `(1,2,3,100)` | 12 | 0.614 | 0.044 |
| `+ 6 unit constants` | 12 | **0.249** | **0.973** |
| `+ 6 unit constants` | 16 | 0.657 | 0.162 |
| `+ 6 unit constants` | **20** | **0.680** | 0.012 |

`n_operands=20` is the default for that reason, and a test pins the pair together
because adding a constant without widening the file is a regression that reads as a
feature. The ring is unaffected (margin still 76.7×) since chain length is capped by
`n_instr`, not by operand count — checked rather than assumed, after the last time a
coverage change quietly ate the ring margin.

The dominant remaining loss is now `operand_miss` at ~0.25: an operand the annotation
names is not in the register file. Part of that is genuine extraction misses
(`1/2`, values the text never writes), and part is a cascade — when a compound
annotation is rejected, its result is never written to a register, so every later
step that reads it fails to align. **That 0.42 is the ceiling on the supervised arm
and it has nothing to do with the network**, which is exactly why it is measured
first. The answer-only arm is not bounded by it: it trains on all 7473.

**Measured: the ring GSM8K actually needs, which is far smaller than 3a-viii sized.**
Also computable with no model in the loop. What matters in residue form is the
*unreduced* numerator and denominator a program builds, since a fraction cannot be
reduced — so simulate that growth in plain integers over the 4590 recovered programs:

| percentile of worst \|num\|,\|den\| per program | value |
| --- | --- |
| p50 | **120** |
| p90 | 5.5e3 |
| p99 | 3.8e6 |
| p99.9 | 1.1e9 |
| p100 (worst of 4590) | **2.16e11** |

`RATIONAL_MODULI` gives ±4.49e15 — **six orders of magnitude more than the worst case
needs**. 3a-viii sized it from an argument ("five two-decimal values, denominator
1e10"); the argument was not wrong, it was answering a worst case the data does not
contain.

`GSM8K_MODULI = (64, 125, 27, 11, 7, 13, 37, 101, 41)` is the measured sizing:
±1.66e13, a **77× margin** on the observed worst case, every digit period still ≤ 6 —
and much cheaper, because the packed path pads every modulus to the widest. 9 moduli at
P=125 is `9·125² = 141k` against `10·271² = 734k`, so **5.2× less arithmetic per
composition**, and the head width drops 697 → 426 units. It is the bridge's default;
`RATIONAL_MODULI` is one constructor argument away.

**It was 8 moduli and a 139× margin until the chains got longer, and that is the part
worth remembering.** Decomposing compound annotations — a change about *coverage*,
with no apparent connection to the ring — roughly tripled the median chain length, and
since every operation multiplies denominators the worst case moved 2.91e9 → **2.16e11**
and the margin collapsed **139× → 1.9×**. No program exceeded the ring even then, so
nothing failed and **nothing would have failed visibly**: the next slightly longer chain
would simply have wrapped to a different number. It was caught only by re-measuring the
ring after a change that had nothing obviously to do with it.

The alternatives with a larger margin were rejected on principle rather than cost:
adding 73 or 137 instead of 41 buys a wider ring at similar width but pushes
`max_period` to 8, and a modulus whose digit-coefficient pattern needs 8 positions to
repeat is useless at the operand widths training actually sees. That constraint, which
exists for extrapolation, did real work here.

Two limits on that sizing, since it is a decision and not just an observation. It is
measured over the programs the annotations *describe*, so a model emitting a different
one — dividing repeatedly, say — is not bounded by it. And a soft, undecided program is
bounded by nothing at all, because its decoded denominator is an argmax over
incoherent residues rather than a value (see `ptr_sharp`). `denominator_magnitude`
remains the monitor.

**Measured: it runs end to end, and what it costs.** 7473 train and 1319 test
problems encoded once by `all-MiniLM-L6-v2` at `max_len=192`, with **1 problem
truncated** in 8792 — so the cache is not quietly cutting questions off. The trainer
then runs text -> cached embeddings -> resampler -> shared latent core -> program ->
exact rational execution, with gradients reaching the program heads through the
arithmetic.

Where the parameters are, which is the point of the design:

| | params | share |
| --- | --- | --- |
| resampler (learned) | 0.318M | 38.4% |
| shared latent core (learned) | 0.502M | 60.6% |
| **program heads (learned)** | **7.7k** | **0.93%** |
| frozen encoder | 22M, cached, never in the training graph | — |
| the arithmetic | 0 | — |

**The learned surface that decides *which computation to perform* is under eight
thousand parameters.** Everything else compresses the text or performs arithmetic
exactly. (It was 5k before `n_operands` went 12 → 20: the pointer heads are
`Linear(d_model, n_slots)`, so they scale with the register file.)

**Cost, and why no single number is given.** The same configuration measured **1.68
s/step** and **2.48 s/step** at batch 8 within an hour of each other, differing only in
what else the box was running. Earlier in this session a contended host turned a
19-minute study arm into a reported 36,915 s (3a-xi, item 5), so a wall-clock figure
from a shared machine is not a cost measurement and will not be quoted as one. The
honest statement is that this is a CPU-bound prototype at ~2 s/step at batch 8, that a
real run wants a bigger batch and a GPU, and that the profile in `CLAUDE.md` -- 92% of a
register-machine step is torch forward+backward, ~6.5% the exact execution -- is the
part that transfers between machines.

**No accuracy is claimed.** Nothing here has been trained to convergence or run on a
GPU, and this section will carry accuracy when there is accuracy. One number from the
smoke run is worth repeating as a warning rather than a result: `max_den_magnitude`
read **3.96e11** against a 4.04e11 ring after 10 steps, which looks like the ring is
exhausted and is not. It is an argmax over each modulus independently, so an
uncommitted program decodes to an essentially uniform ring value whatever the
arithmetic did. Read it with `ptr_sharp` or do not read it. And the contamination
claim stays withdrawn: the encoder is pretrained, it has seen these benchmarks, and
"frozen" means its weights do not move rather than that the information is absent.
What replaces the claim is an arm — `--encoder` swaps the tower, embeddings are
cached, so running the same core on a pretrained encoder and on one that never saw
the benchmark prices the encoder's prior directly. That arm is the only part of this
work that would be a *new* result rather than a port, and it has not been run either.

Known limits, recorded rather than left implicit:

- `extract_quantities` is a regex plus a table of written cardinals. It does not see
  `1/2`, and it emits spurious operands for dates, ordinals and item numbers, which
  can push a real quantity out of a fixed-width register file. Quantity *selection*
  is a missing component, not a tuning detail.
- The register file is a fixed shape, so a chain longer than `n_instr` is rejected
  rather than truncated.
- The answer-side loss for rationals is the cross-product residual. Its only blind
  spot is `0 ÷ 0`; a zero divisor with a live numerator is detected by the residual
  itself. See 3a-xi, including where I had that wrong.

**Trained, measured: 0.0159 exact match on GSM8K test (21 of 1319), against a ceiling of 0.691.**
The first real number. Getting to it meant fixing four faults that would each have made any number
meaningless, then two that stopped learning:

- *Wrong CLI default.* ``--n-operands`` defaulted to 12, overriding the config's 20; the config's own
  measurement is that 12 halves program coverage (0.68 -> 0.25). The CLI now reads the config.
- *NaN program loss from step 1.* Rows without a recovered program borrowed another row's program as a
  zero-weighted stand-in, whose pointers could address registers masked for *this* row: infinite
  cross-entropy times zero weight is nan (the weights survived only because the zero also zeroed the
  gradient). The stand-in is now all-zero pointers, legal everywhere, and a non-finite loss raises.
- *Soft evaluation.* Exact match was decoded from the soft program, which 3a-x measured lands in neither
  register ~30% of the time. It is now the argmax program executed exactly, over the full test set,
  checked by feeding it the gold programs (300/300).
- *Silent misalignment risk.* Examples are matched to the encoder cache by position; a dropped row would
  pair every later question with the wrong embedding. Now an error.
- *Collapse to the average program.* The model predicted ``*`` -- the padding operator, so the most
  common op overall -- at every instruction of every problem: op accuracy rose with position
  (0.11 -> 0.59) exactly as that predicts, and op logits varied across problems by 0.16. The soft-program
  answer loss was the cause; with it off, program loss falls (1.38 -> 0.97 at 400 steps), op logits spread
  (1.16), and argmax programs go from 54 to 210 distinct over 256 problems.
- *Index pointers need counting.* Operand registers hold the text's numbers in reading order, so an index
  pointer must say "the third number". The fix 3d found for the same problem is applied: a content
  pointer scores each operand register by the encoder state at the token where its quantity appears
  (anchors verified 1207/1209), constants get learned keys. Its contribution under program-only training
  was not separately ablated.

Then, with content pointers and program-only loss, 5000 steps: training program loss falls to 0.16 while
test exact match stays at 0.010-0.025 throughout. It memorises the 5080 annotated programs and does not
generalise. Whether the encoder can carry what is needed was probed directly: a linear probe on
mean-pooled MiniLM states predicts "the solution divides" at 0.682 against a 0.512 majority, so the
information is partly there; Qwen3-0.6B's layer 21 reaches 0.774. A larger language model was **not**
adopted as the encoder: if it reads the problem, it has largely solved it, and the core would be executing
its plan -- the opposite of the point. It remains only as the measuring arm 3c already pre-registers.
What would plausibly move the number, in order: far more program-annotated data than 5080 (self-generated
problems in the GSM8K style, where gold programs are free); the outcome-only expert iteration of 3e-ii,
which reached depth 3 on arithmetic, applied here so the 31% of problems without an annotation still train;
and alignment coverage (*corrected below: "22% missing a needed number" was wrong. Measured by
cause, about 2% of test problems lack a number the extractor could find. The rest of the
alignment failures are annotations that skip a step, or chains that never reach the answer*).

**Pre-registered: does generated data fix the memorisation?** ``lamb/gsm_synth.py`` writes
GSM8K-style problems (16 scenario families, phrasing variants, chains of 2-6 operations, implicit
constants, written numerals, distractor numbers) in GSM8K's own annotation format, so they pass
through the same parser and aligner as the real data (4000/4000 align and execute to their answer).
No language model writes any of it. The risk is the obvious one: a model can memorise the
*generator* instead of the 5080 real programs, and an early pilot shows it -- 0.92 on in-family
generated problems by step 200, 0.15 on families never trained on, 0.01 on GSM8K. So three numbers are
reported, never the first alone: real GSM8K test (the target), generated problems from four held-out
families (``ages``, ``recipe``, ``combined_rate``, ``dozens``: did it learn to read quantities and
relations, or the templates?), and in-family generated test. ``lamb/bridge_synth.py``, content pointer,
program-only loss, frozen MiniLM run on the fly, 20000 steps of batch 64, same seed
(*amended before any long-run result: 10000 steps of batch 128, the same 1.28M problems -- a
step is launch-bound GPU work, and batch 256 does not fit the 3.7 GB card*):

> Mixing generated problems with real GSM8K (half each) reaches GSM8K-test exact match >= 0.05 and
> above the real-only arm trained identically. A generated-only arm is reported beside both as the
> zero-shot transfer measurement.

**Result: not met.** The mixed arm ran to step 8000 of 10000 before the machine shut down (cause not
in the journal); the real-only and generated-only arms never started. Every evaluation it reached:

| step | GSM8K test | held-out families | in-family |
|---|---|---|---|
| 2000 | 0.0159 | 0.035 | 0.978 |
| 4000 | 0.0220 | 0.038 | 0.978 |
| 6000 | 0.0167 | 0.078 | 0.973 |
| 8000 | 0.0129 | 0.083 | 0.990 |

GSM8K never leaves the 0.016 the real-only bridge already had, and it is falling at the last
evaluation. In-family reaches 0.99 while families it never saw sit below 0.09. A third of the planned
compute was left, and the step-8000 number is lower than the step-4000 one. So the missing arms would
not turn this into a pass. The model learns the generator's templates. It does not learn to read
quantities and relations that carry over to new problems. More data of the same shape does not
address that, and the arms were not re-run.

The obvious next fix would be a program family that cannot memorise whole problems: fold the
quantities left to right in reading order, choosing each operation from local cue words. It was
measured against the annotations before anything was built. Of the 5080 aligned training programs,
0.409 are linear chains, and 0.259 are chains whose numbers appear in reading order. On test those
figures are 0.386 and 0.239, which is 0.165 of all 1319 problems. A reading-order fold therefore caps
out around a sixth of GSM8K. It is not the lever either.

**Pre-registered pilot: is the frozen encoder the cap?** The test is whether the bridge reads better
when it may change how it reads. Real GSM8K only (``--no-synth``), batch 64, 3000 steps, seed 0. Three
arms differ only in ``--unfreeze``: 0 (frozen), 2 (top two MiniLM layers) and 6 (all, 10.6M params).
Encoder lr is 3e-5 and dropout is off in every arm. The claim:

> An unfrozen arm reaches GSM8K-test exact match >= 0.03 at step 3000, and above the frozen arm at
> the same step.

This is n=1 and is a pilot. A pass earns a multi-seed study. A fail means reading is not what the
3.4M head is missing, at this data size.

**Result: not met.** GSM8K-test exact match:

| arm | step 1000 | step 2000 | step 3000 |
|---|---|---|---|
| frozen | 0.0121 | 0.0159 | 0.0129 |
| top 2 layers | 0.0190 | 0.0197 | 0.0174 |
| all 6 layers | 0.0235 | 0.0205 | 0.0182 |

The unfrozen arms lead the frozen one at every evaluation by 0.004-0.011, which is 5-15 problems
of 1319, from one seed. Neither comes near 0.03. Every arm reaches training loss ~0.03 by step 3000
while test accuracy falls from its step-1000 value. Unfreezing makes the model fit the 5080 training
programs faster; it does not make it read new problems. At this data size the cap is how few real
programs there are, and how the encoder reads is not the binding constraint. The watchdog stopped
nothing; the peak GPU temperature was 85C.

**Measured: does GSM8K decompose into sentence-local facts?** The proposal is a formal layer between
text and program. Each sentence becomes equations over named variables (`k = 50`, `n1 = k + k/2`),
and an exact solver composes them, in the line of verb categorisation (Hosseini et al. 2014) and
quantity-relation trees (Roy & Roth 2015). Before building it, 50 random test problems
(``random.Random(2026)``) were labelled by hand in a language fixed beforehand. The rule is that a
number may appear only in the equation of its own sentence, and a few named world constants are
allowed. A checker verifies every label mechanically: the locality of every literal, and that the
sympy-solved system gives gold. Files are in ``docs/factcheck/``; the labels are from one labeller.

* 47/50 (0.94) fit, and all 50 labels reproduce gold. 9 of the 47 use a world constant (dozen,
  days per week, coin values). The three that do not fit need an arithmetic series (2) or a count of
  named people (1).
* 0.66 of all operations sit in the question's `ANSWER` equation. 0.39 of equations read a variable
  defined in another sentence.

So the numbers are local, and the composition is not. Facts can be read sentence by sentence, but
two-thirds of the program ends up in the question sentence, assembled from variables named across
the whole problem. Naming those variables consistently is coreference, a hidden requirement. The
layer does make the hard step smaller and more structured: compose named facts, against today's
"point into an unlabelled quantity file". It does not make the hard step disappear. The 0.66 depends
on the labeller, since some operations could be pushed into fact sentences. The 0.94 depends on it
much less.

**Measured: is the trained bridge better than lookup?** No training is involved here.
``docs/factcheck/retrieval.py`` takes the k nearest training problems by MiniLM mean-pooled
similarity, runs their gold programs on the test problem's registers, and takes the majority answer.
The control runs k *random* training programs the same way.

| k | nearest: vote | random: vote | nearest: gold among k | random: gold among k |
|---|---|---|---|---|
| 25 | 0.0182 | 0.0152 | 0.168 | 0.121 |
| 100 | 0.0387 | 0.0167 | 0.399 | 0.349 |
| 300 | 0.0379 | 0.0265 | 0.646 | 0.614 |
| 1000 | 0.0349 | 0.0250 | 0.820 | 0.815 |

The trained bridge, at 0.013-0.018, scores the same as a majority vote over 100 *random* training
programs (0.0167). Whatever it learned, it is no more than a prior over common program shapes.
Plain retrieval beats it at 0.039, with no training. That the gold answer is "among the 100 nearest
candidates 40% of the time" looked like a shortlist to rerank. The control shows it is almost all
coincidence: 100 random programs reach 0.35, because GSM8K answers are small integers that many
wrong programs land on. Similarity adds 5 points of coverage over chance. A reranker over retrieved
programs would be choosing among coincidences it cannot tell apart without the answer, so that
route is closed.

**Sentence-level generated data: precondition failed, not built.** The plan was to train a per-number
role classifier on generated sentences and test it on real ones. Each number would be classed as a
fresh quantity, or as ADD, SUB, MUL, DIV or PART of something else. The roles were first derived
from the 50 labels (``docs/factcheck/roles.py``). Labeller choices are collapsed, since a listed sum,
a rate and a binding all introduce fresh quantities. Of the 148 numbers in real sentences, 122
(0.82) are fresh quantities and 26 carry a local relation. A ten-line keyword rule ("more", "fewer",
"times", "%" after the number) scores 0.905 overall against 0.824 for always guessing FRESH. Its
misses are verb semantics ("loses 8", "30 flew away", "sold for $8 each"), about ten numbers in 50
problems. The sentence-local layer is therefore mostly solved by rules. A learned model has at most
a tenth of the numbers to improve on, and that is not where GSM8K fails. It fails in composition:
0.66 of the operations sit in the question's equation. Generated sentences would teach the part that
is already nearly solved, so the pilot was not built.

**Pre-registered: does 75x more real-style programs lift the cap?** Every measurement above points
at one constraint: there are too few real programs. GSM8K-Aug (Deng et al. 2023, arXiv 2311.01460)
has about 385k problems, generated as variants of the GSM8K *train* questions and annotated in the
same ``<<a*b=c>>`` format. It goes through the same aligner. It is data, not a model: nothing in it
reads or reasons at test time. ``bridge_synth --no-synth --train-jsonl``, content pointer, frozen
MiniLM, program-only loss, run on an A100. Held-out: GSM8K test, as always. The aug rows derive
from train questions only, so the test split stays unseen.

> Trained on GSM8K train plus the aligned GSM8K-Aug rows, the bridge reaches GSM8K-test exact match
> >= 0.05, against 0.0129-0.0182 for every real-only arm above.

A pass means data quantity was the cap. A fail means that more programs of GSM8K's own shapes do not
teach composition either. The aug questions are variations on about 7k seeds, so the number of
distinct structures grows far less than the number of rows.

**Result: met, n=1.** 298,370 of the 385,620 aug rows align (0.774), on top of the 5080 real
programs. A100, batch 256, 8000 steps, 26 minutes:

| step | GSM8K test | held-out families | in-family | train loss |
|---|---|---|---|---|
| 1000 | 0.0356 | 0.178 | 0.210 | 0.48 |
| 2000 | 0.0371 | 0.183 | 0.245 | 0.48 |
| 3000 | 0.0485 | 0.257 | 0.323 | 0.42 |
| 4000 | 0.0516 | 0.308 | 0.360 | 0.41 |
| 5000 | 0.0660 | 0.245 | 0.413 | 0.38 |
| 6000 | 0.0599 | 0.278 | 0.413 | 0.34 |
| 7000 | 0.0682 | 0.245 | 0.373 | 0.35 |
| 8000 | 0.0675 | 0.272 | 0.440 | 0.30 |

GSM8K test ends at 0.0675, which is 89 of 1319 problems, against 17-24 for every real-only arm.
Two things changed in kind, not just in degree. Training loss stays at 0.3, where every real-only
arm went to 0.03, so the model is not memorising its programs. And it reaches 0.27 on the four
generated families it never saw, without training on a single generated problem. That is three
times the 0.083 of the arm that trained on the generator itself. The cap was data quantity. The
curve flattens over the last 3000 steps (0.066, 0.060, 0.068, 0.068), so this is near what this
model reaches on this data, not a point on a rising line. It is one seed; it earns a multi-seed study
before it is cited as a result.

**Pre-registered: the multi-seed study of the GSM8K-Aug result.** ``lamb/bridge_fast.py``:
the same model, aligner, anchors and exact evaluation. The frozen encoder runs once over every
problem instead of on every batch, and the loss step is compiled. The config is identical to the n=1
run (batch 256, 8000 steps, lr 3e-4, frozen MiniLM). The arms are paired by seed: ``aug`` (GSM8K
train plus aligned GSM8K-Aug) and ``real`` (GSM8K train only), with the same seed, initialisation
and sampler. There are six seeds, because the exact two-sided sign-flip test cannot go below
2/2^n, so five seeds could never reach 0.05.

> Over 6 paired seeds, ``aug`` beats ``real`` on final GSM8K-test exact match with exact paired
> permutation p <= 0.05, and mean ``aug`` >= 0.05.

Held-out families and in-family are reported beside it with the same test. If the precondition fails
(the fast trainer does not reproduce the n=1 run's regime on seed 0), the study did not run. That
is a different outcome from a failed result.

**Result: met.** Six paired seeds, final step-8000 exact match:

| seed | aug GSM8K | real GSM8K | aug held-out | real held-out |
|---|---|---|---|---|
| 0 | 0.0728 | 0.0114 | 0.308 | 0.035 |
| 1 | 0.0751 | 0.0174 | 0.295 | 0.022 |
| 2 | 0.0864 | 0.0159 | 0.363 | 0.047 |
| 3 | 0.0675 | 0.0136 | 0.225 | 0.022 |
| 4 | 0.0705 | 0.0197 | 0.313 | 0.032 |
| 5 | 0.0781 | 0.0106 | 0.170 | 0.015 |

GSM8K test: aug 0.0751 (sd 0.0066) against real 0.0148 (sd 0.0035). The paired difference is
+0.0603, aug wins 6/6, and the exact two-sided sign-flip test gives p = 0.0312, the smallest six
seeds can give. Held-out families: 0.279 against 0.029, also 6/6, p = 0.0312. The precondition
held, since seed 3 reproduced the n=1 run's 0.0675 exactly. Seeds 2 and 5 were run in a second
session on the same code and config. **Data quantity was the cap on the bridge, and that is now a
result, not a single run.** The absolute level is still 7.5% of GSM8K.

**Measured: what the alignment misses actually are.** The first unreadable annotation operand,
by cause, on test (train in brackets). 13.6% (15.1%) of problems use a value one unwritten step
from readable registers. 7.1% (5.2%) have chains that never reach the answer. 5.5% (5.2%) use a
percent as a fraction. About 2% lack a number the extractor could find: word operators 0.8%,
implicit constants 0.8%, register limit 0.2%, ``a/b`` literals 0.1%. The extractor is not the
cap. ``align_program(bridge=True)`` inserts the skipped step when it is unambiguous: division by
the constant 100 first, otherwise only a unique single-operation candidate. Coverage rises from
0.680 to 0.756 on train and from 0.691 to 0.763 on test, and every bridged program still executes
exactly to the dataset's answer. Of what is left on test, 163 problems are ambiguous (two or more
distinct steps give the value, often six or more) and 9 need more than one step. Scoring every
valid candidate as correct would recover the ambiguous ones without guessing; that is not built.

**Pre-registered: do bridged labels help?** ``bridge_fast --bridge``, the same six seeds and config as
the aug arm above. GSM8K train and GSM8K-Aug are aligned with bridging, so there are more supervised
programs; evaluation is unchanged (all 1319 test problems). Paired by seed against the unbridged aug
arm:

> Bridged aug beats unbridged aug on final GSM8K-test exact match, exact paired permutation p <= 0.05.

The expected effect is small against a seed sd of 0.0066, so a fail is the likely outcome and is
reported as it lands.

**Pre-registered pilot: can the head's weights be ternary?** The goal is a model that is cheaper to
*compute*, not just faster on one card. A matmul against weights in {-g, 0, +g} is additions plus
one scale, and it is integer, so it can be exact (BitNet b1.58, arXiv 2402.17764; arXiv
2406.02528). ``lamb/ternary.py`` quantises every 2-D weight in the head, including attention's
packed projection. Embeddings, norms and learned queries stay float, and so do activations; the
frozen encoder is not touched. Two runs, same data and seed as the aug arm (GSM8K train plus
GSM8K-Aug, seed 0, batch 256, 8000 steps, uncompiled): float, with int8 and ternary post-training
quantisation scored at the end; and ternary quantisation-aware training with the
straight-through estimator.

> Ternary QAT reaches GSM8K-test exact match within 0.01 of the float run at step 8000.

Expected beside it, and reported either way: int8 PTQ within 0.005 of float, and ternary PTQ
far below, since nothing in float training keeps the weights near three values. n=1: a pass earns
seeds, not a claim.

**Result: not met, by 0.002.** Step 8000, seed 0, 0.991 of the head's parameters quantised:

| model | GSM8K test | held-out families | in-family |
|---|---|---|---|
| float | 0.0720 | 0.267 | 0.490 |
| float, int8 after training | 0.0720 | 0.260 | 0.492 |
| float, ternary after training | 0.0053 | 0.002 | 0.002 |
| ternary-aware training | 0.0599 | 0.208 | 0.452 |

Ternary-aware training lands 0.0121 below float, against a 0.01 margin, so the criterion fails as
written. In size: it keeps 83% of float's GSM8K accuracy and 92% of in-family, and it is still
four times the real-data-only bridge, with every head matmul reduced to additions. Rounding a
float model to ternary after the fact destroys it (0.005), which is exactly why training under the
constraint matters. int8 is free: identical to float on GSM8K. One seed each, so the 0.012 gap
itself is not resolved; the seed-to-seed spread of the float aug arm is about 0.006.

**Pre-registered: a translator that cannot reason.** Natural-language output must not let a
language model do LAMb's thinking. ``lamb/translator.py`` makes that structural rather than
trained:
* every number and name is a typed slot (``<Q>`` quantity, ``<K>`` constant, ``<R>`` result,
  ``<N>`` name), added as single tokens, so the model never sees a value and values are copied
  in only after decoding;
* the model sees one instruction plus the question sentences its operands depend on, with every
  quantity and name slotted, and never the whole problem;
* there is one sentence per instruction, in program order;
* decoding bans every digit token and allows only the sentence's closed vocabulary: its input
  words, a fixed glue list, its slots, punctuation;
* every sentence is validated, and falls back to a deterministic renderer if it fails.

Training pairs come from GSM8K train and GSM8K-Aug-NL (sentences row-aligned with GSM8K-Aug's
annotated steps), and only sentences whose every word is in the closed vocabulary are used.
SmolLM2-135M, full fine-tune. On 600 GSM8K-test steps:

> (1) At least 0.90 of sentences are valid without fallback. (2) With each step's operation
> rotated to a different one, the sentence's wording follows the given operation at least 0.8
> times as often as with the true operation.

The cue lists that score (2) are crude keyword families, so (2) is a floor on faithfulness, not a
measure of quality. Numbers come from LAMb's registers in 100% of outputs by construction.

**Result: (1) met, (2) not met.** Training used 13,452 fully covered pairs, out of 73,543 built
(60,249 from 302,789 paired Aug-NL rows, and 13,294 from GSM8K train), for 2 epochs; 4.4 minutes
on the A100. On 600 GSM8K-test steps, 0.986 of sentences are valid without fallback. Keyword
agreement with the operation is 0.354 for the true operation and 0.220 with it rotated, against a
required 0.283, so (2) fails as written. The diagnostics below were *not* pre-registered.
Generated sentences overwhelmingly carry a slot formula (0.970). The formula's operator matches
the given operation 0.962 of the time with the true operation and 0.968 with it rotated, so the
structure follows LAMb's program and not the context. The keyword measure mostly failed because
the sentences use symbols instead of cue words. At most 5% of sentences pair ``+``/``-`` wording with the other operation (0.052 with the true
operation, 0.046 rotated). That is an upper bound. "<Q1>-<R1>=<R2> more than" is correct GSM8K
phrasing for a difference, and the cue lists cannot tell it from a contradiction. The equal rate
under rotation says the wording comes from the closed vocabulary letting the input's words
through, not from the model solving anything. A validation rule on these cue lists would reject
correct sentences, so none was added. Measuring word-level faithfulness properly needs a better
instrument than keyword families.

## 3d. infContext meets the register machine: the pillar nothing tests

The memory pillar and the reasoning pillar do not touch. `memory_bench` and
`ruler_bench` measure *retrieval* (needle, passkey, variable tracking) and neither
computes anything over what was retrieved; `regmachine` loads its operands exactly from
the prompt. So the claim that would matter for a latent frontier model, **reason over
what you remembered with O(1) state**, has no test anywhere in this repo.

The blocker is concrete and it is representational, not algorithmic. The register file is
loaded by a fixed digit->residue map because ROADMAP 3a-vi measured that asking a network
to induce that map leaves it at chance. That is exactly why operands have to be *in the
prompt*. An operand arriving from distant context has to get into a register some other
way, and there are two honest designs:

1. **Register file as bounded cache.** Every value seen enters a register; the query
   selects among them with pointers, so retrieval becomes pointer selection over a wide
   file. Clean, reuses the machine wholesale, and bounded: a file of R registers gives
   R-value context, so it is not infContext, it is a longer window.
2. **Neural memory gates the file.** The O(1) memory state decides which values are worth
   a register. This is the design that actually earns the name, and it is the harder one:
   the memory must emit something the fixed encoder can turn into a residue code, and a
   learned path from memory to residues is the thing 3a-vi found unlearnable.

Design 2 is the interesting one and it inherits 3a-vi's wall. The way through is probably
that the memory stores *the digits it saw* rather than a value embedding, so the fixed map
still does the encoding and nothing has to learn arithmetic. That has not been tried.

Also missing, and cheaper: a task representation. The tokenizer has 22 symbols and none of
them name a key, so a `(key, value)` context cannot currently be expressed at all. Any
version of this starts there.

**Built, measured, and it does not work.** `lamb/memreg.py`: a context of `@k = v`
bindings then a query `@i op @j =`, every value loaded into a register exactly by the
fixed map, so nothing about arithmetic is learned and the only thing to produce is a
pointer to the register a key named. 1500 steps, 8 bindings, 2-digit values, depth-1
programs.

| arm | accuracy | `ptr_acc` | by distance, 2 to 8 |
| --- | --- | --- | --- |
| `use_memory=True` | 0.176 | 0.117 | 0.54, 0.11, 0.03, 0.05, 0.03, 0.05, 0.00 |
| memoryless control | 0.172 | 0.098 | 0.45, 0.20, 0.12, 0.01, 0.02, 0.03, 0.05 |

**The neural memory makes no difference**, and both arms resolve only the nearest binding.
So the failure is not "memory needed and absent": attention alone over 45 tokens does not
do it either.

The cause is representational and it is not the memory's fault. A register's index is the
binding's *presentation order*, so resolving `@4` means knowing its binding was the third
one. Nothing in the embedding carries that: the value channel carries magnitude and Abacus
carries digit position *within a number*. The model is being asked to count bindings with
no counter.

**And the obvious fix dissolves the task.** Index registers by key id instead of
presentation order and `@4` points at register 4 directly, but then the loader has already
done the retrieval by rule and all that remains is a constant key->pointer mapping. Which
is the same shape as 3a-xv: a task that looks like retrieval and is really recall of a
fixed address.

That sharpens what the honest version has to be. Retrieval only exists where there are
**more bindings than registers**, so something must choose which values occupy the file.
That is design 2, and design 1 is now known to be either unlearnable (order-indexed) or
vacuous (key-indexed).

**Design 2 was built and it fails for the same reason.** 16 bindings, 4 register slots, a
`SelectiveLoader` emitting one distribution per slot over the candidates, selection over
*positions* so the fixed map still does all the encoding and 3a-vi's wall is untouched.
The gold selection plus gold program executes 8/8 exactly, so the task is well posed.
4000 steps, memory and memoryless:

| arm | accuracy | selection accuracy | `sel_loss` |
| --- | --- | --- | --- |
| `use_memory=True` | 0.008 | 0.004 | 2.774 |
| memoryless | 0.008 | 0.000 | 2.773 |

`ln(16) = 2.7726`. **The selection distribution is exactly uniform for the entire run**:
not slow learning, no learning. And the reason is the one design 1 died of. The head emits
a distribution over *candidate index*, index is presentation order, so selecting the
binding for `@4` still means knowing it was the third one. Moving the choice from the
pointer to the loader relocated the counting problem without solving it.

**The fix this identifies, built, and it works.** Selection must be a pointer into the
*sequence*, scored from the hidden state at each binding's position, rather than an index
into an abstract candidate list. Then "where was `@4` bound" is an attention operation and
the loader reads the digits found there, so the fixed map still does all the encoding.
`PointerLoader`, same task and same everything else, 3000 steps:

| selection | `sel_loss` | accuracy |
| --- | --- | --- |
| **pointer (content)** | 2.773 -> **0.836** | **0.754** |
| index (order) | 2.789, pinned at ln(16) | 0.008 |

**Accuracy by distance is flat**: 0.80, 0.78, 0.94, 0.71, 0.77, 0.79, 0.74, 0.61, 0.65,
0.87, 0.88, 0.70, 0.73 across distances 4 to 16. Retrieval does not decay with how far back
the binding was, which is the property the pillar is named for, and it is the first working
composition of retrieval with exact computation in this repo.

The anchor is the `SEP` closing each binding rather than the key token: attention is causal,
so at the key position the value has not been read and the hidden state cannot identify what
the binding holds.

Two readings to get right.

**`select_acc` of 0.211 against accuracy 0.754 is not a contradiction.** It requires the
argmax selection to match the *generator's* slot assignment, and register slots are
interchangeable: loading the right value into slot 2 rather than slot 1 is correct provided
the program points at slot 2. So it measures conformity and not competence, exactly as
`canonical_acc` does for programs (3a-vii). Answer accuracy is the correctness measure.

**The refusal signal's coverage collapses here, to 0.008 at precision 1.000.** Soft
selection makes each register a *mixture* over candidates, so the composed answer
distribution stays blurred even when its argmax is right, and the sharpness signal reads the
blur. Precision survives; coverage does not. Evaluating with a hard selection should recover
it, and that is untested.

Untested also: the memory. Both arms here ran `use_memory=True`, so nothing yet separates the
O(1) state from attention on this task, and across designs 1 and 2 the memory has made no
measurable difference. And this is one seed.

One flaw of mine in the above, which does not explain the result but should not stand: the
unqueried register slots were supervised toward a randomly drawn binding on every sample,
injecting irreducible noise. With two of four slots random the floor would be ~1.386, and
the observed 2.773 is uniform on *all* slots, so the noise was not what pinned it.

One thing did transfer. The refusal signal of 3a-xix works on a *retrieval* failure and
not only an arithmetic one: at 0.176 accuracy it accepted 0.2% of problems and was right
on all of them. A model that cannot do the task declines to answer it.

## 3e. Program writing by outcome, sampled not blended: GRPO with a curriculum bandit

Pre-registered before any run. This section carries the criterion and the design; the
numbers land in 3e-i once the study finishes.

**The question.** 3a-xii settled that outcome-only induction fails at depth 3 (0.015
against a supervised 1.000), and 3a-xvi found the failing seeds are *uncommitted* rather
than wrong: a blurred pointer decodes to a value in neither register ~30% of the time
(3a-x), so the dense answer gradient the answer-only arm follows is pointed at a soft
program with no executable meaning. That gradient is biased, not merely noisy. So the
question 3e asks is narrower than "can outcomes teach a program": it is whether a learning
rule that only ever scores *hard, valid* programs clears the depth-3 wall the dense-blend
rule fell off.

**The rule.** Group-relative policy optimisation (GRPO, arXiv:2402.03300), which fits this
repo for reasons that are structural rather than fashionable:

- The verifier is the exact executor. Reward is `1` iff the decoded answer equals the
  grammar's answer, and it cannot be hacked because the executor is ground truth, not a
  learned reward model. This is RLVR with an internal, exact verifier.
- No value network. The advantage is `(r_i - mean_g) / (std_g + eps)` within a group of
  `G` programs sampled for one problem, so there is nothing to learn a baseline with and
  nothing to go stale. This is the whole reason GRPO rather than PPO.
- Sampling is cheap. A program is tiny and its execution is ~6.5% of a step (`CLAUDE.md`),
  so `G` hard rollouts per problem cost roughly `G` executions, not `G` forward passes.
- Nothing differentiable flows through the algebra. The gradient is REINFORCE on the
  sampled program's log-probability, `-(adv.detach() * logp)`, so the soft-read pathology
  that biases the answer-only arm cannot arise: only valid programs are ever scored.

**The wall it inherits, and the bandit that is the answer to it.** At depth 3 a random
program's hit rate is `~(|ops| * n_slots^2)^-n_instr`, astronomically small, so every
group is all-wrong, `std_g` is zero, the advantage is zero, and GRPO stalls exactly where
the dense rule did (advantage collapse, arXiv:2605.21125; DAPO's dynamic sampling,
arXiv:2503.14476, is the standard patch and is applied: degenerate groups are dropped and
their rate reported as ACR). Two things create gradient where binary reward gives none,
and both are free here in a way they are not for a text model:

1. **Process reward from the gold trace.** The grammar hands every problem the exact value
   of every sub-expression (`sample_with_trace`). A sampled program's registers are decoded
   and scored by how many gold intermediate values they hit, order-free, so a program that
   computes a right sub-result gets credit before it ever gets the final answer right. It
   is order-free because 48 distinct programs compute `(a+b)+(c+d)` and conformity is not
   competence (3a-xii). The hazard is that an intermediate can be hit by coincidence, so
   `process_coef` is small and outcome dominates; it is a shaping term, not the objective.
2. **A curriculum bandit over difficulty.** The arms are the task descriptors (depth, and
   later `shape`/`ops_key`); the bandit keeps training where groups are still informative
   and grows from there. Its per-arm signal is learnability `p*(1-p)` (Bernoulli variance
   of the group success rate), which peaks at `p=0.5` — the frontier of learnability
   (arXiv:2502.12272) — and is tracked as a non-stationary EMA sampled Boltzmann with an
   epsilon floor (Self-Evolving Curriculum, arXiv:2505.14970). The machine is sized for the
   deepest arm and shorter problems are padded through the existing identity-register path
   (3a-xiv), so one machine spans the whole curriculum.

**The Jev borrow, optional and off by default.** Jev (TypeSafe, 2026) is a non-generative
model that emits a typed value with a calibrated confidence, trained by optimising the
probability against the outcome rather than against a rater (its "RLCD" is a proper scoring
rule on outcomes). LAMb already emits a typed value and already has a refusal signal
(3a-xix), so the only borrowable is calibration: a `calib_coef` term that trains an emitted
confidence toward the empirical `p(correct)`. It is a proxy on `ptr_sharp` for now, kept at
zero for the headline comparison so the core result is not entangled with a half-built head.

**Success criteria, pre-registered, honoured whatever the outcome.**

> **H1 (headline).** GRPO with hard sampling and exact verify, *no program supervision*,
> reaches held-out `answer_acc_hard` within noise of the supervised arm at depth 3, over
> >=5 seeds, by the exact permutation test in `lamb/study.py`. This is the same question as
> 3a-xii, asked of sampled-hard programs instead of dense-blended ones.
>
> **H2 (the bandit earns its place).** The curriculum bandit reaches H1's criterion in
> fewer environment steps, or at a lower advantage-collapse rate, than GRPO trained at a
> fixed depth 3. If H1 passes at fixed depth already, H2 is the only thing the bandit has
> to justify and it is reported on its own.

`answer_acc_hard` is the measure, not `canonical_acc` (conformity, expected low). A negative
H1 is a real result and is reported with the same energy as a positive one: it would say
the depth-3 wall is about *credit* rather than *gradient bias*, which the dense-vs-sampled
contrast alone cannot distinguish and this arm can. The trainer is `lamb/program_grpo.py`
(`GRPOTrainer`, `CurriculumBandit`); study.py arms are deferred until the ceiling in 3e-i is
cleared, since paired arms on an arm that does not reach its own criterion measure nothing.

### 3e-i. What GRPO-over-programs does and does not do yet (findings)

Built and tested (`lamb/program_grpo.py`, `tests/test_program_grpo.py`); the headline H1 is
**not** achieved and is not claimed. What holds:

- **The loop is sound and generalises where memorisation is blocked.** At 2-digit operands,
  depth 1, the sampled-hard GRPO reaches held-out `answer_acc_hard` 1.000 with the correct
  operand-invariant program. At 1-digit it reaches only ~0.08 held-out while fitting the
  training partition, because the ~180-problem space is memorisable and outcome reward has
  no pressure to generalise. This reframes 3a-xii, which was measured at 1-digit: part of
  the outcome-only failure there is a memorisation confound, separable from the depth cliff
  only at wider operands. It does not overturn 3a-xii (that was dense-blend backprop; this
  is sampled-hard), but it means the honest comparison must be at a memorisation-resistant
  width.
- **Exploration, not gradient, was the first wall, and it is closable.** A collapsed policy
  cannot sample the correct program, and REINFORCE cannot raise an action it never samples.
  An entropy floor could not un-collapse a saturated softmax, and a fixed sampling
  temperature was overwhelmed by logit growth; an epsilon-mixture floor on the sampling
  distribution, cooled per-arm on that arm's own step count, gives every legal action a
  probability floor and fixed it. Process reward from the free gold trace, scored only over
  the result slots a depth actually uses, and no std-normalisation of the advantage
  (Dr.GRPO, arXiv:2503.20783) were both needed to stop the argmax program oscillating.

What does **not** hold yet:

- **The large register file caps depth-1 consolidation.** In the depth-3-sized machine
  (`n_operands=9`, `n_slots=16`), even depth-1 training plateaus at ~0.4-0.6 held-out and
  the argmax program oscillates, though the *same* task in a depth-1-sized machine
  (`n_operands=3`) reaches 1.000. Attribution runs isolate it: it is not the model size
  (a `d_model=64` core caps identically to `d_model=96`), not the learning rate, not the
  process reward, not std-normalisation. It is the register-file width itself, and it blocks
  the curriculum from graduating past depth 1, so H1 at depth 3 is unreached. This is the
  open problem; likely candidates are the pointer head scaling with `n_slots` and the
  latent conditioning under a deeper machine, and it is where the next work goes before any
  study.py arm is worth running.

The value that transferred anyway: this work established that the emitted program is
operand-invariant when the core is competent, which is the property 3g turns into an exact
1:1 self-copy. The supervised core reaches that competence today, so 3g does not wait on
3e.

### 3e-ii. Pre-registered: expert iteration instead of policy gradient

The dispatch work (3g) found that REINFORCE on accept/refuse failed where the *same observations
used as cross-entropy targets* succeeded. The same move applies to programs: sample programs, keep
one whose executed answer equals the problem's answer, and train on it with cross-entropy -- the
supervised path the register machine already solves at 1.000 -- plus a per-depth replay buffer of
found programs (STaR arXiv:2203.14465, RFT arXiv:2308.01825, ReST arXiv:2308.08998). Nothing gold:
no gold program, no gold intermediate trace (``process_coef=0``), only the final answer. Same
machine, curriculum bandit and exploration as 3e-i (2-digit, depths 1-3, ``d_model=64``), with the
GRPO arm run on the identical configuration as the control.

> Outcome-only expert iteration reaches held-out ``answer_acc_hard >= 0.95`` at depth 3 on both of
> two seeds within 1500 steps. The GRPO control on the same configuration is reported beside it.

*Confirmation, pre-registered after the first result (below) and before running it:* the revised
method -- expert iteration plus the fresh-operand check plus self-composition, both designed after
seeing plain RFT fail -- is re-run on three fresh seeds (2, 3, 4), 1000 steps, same configuration.
Criterion: held-out ``answer_acc_hard >= 0.95`` at depth 3 on all three.

**Results: the pre-registered method fails, its revision passes the confirmation, and outcome-only
induction now reaches depth 3.** Same configuration throughout (2-digit, depths 1-3, ``d_model=64``,
1500 steps unless noted); held-out ``answer_acc_hard`` by depth:

| arm | depth 1 | depth 2 | depth 3 |
| --- | --- | --- | --- |
| GRPO control | 0.594 | 0.000 | 0.008 |
| plain RFT (the pre-registered method), 2 seeds | 1.000 / 1.000 | 0.203 / 0.008 | 0.000 / 0.000 |
| + fresh-operand check, most-probable verified sample | 1.000 / 1.000 | 0.891 / 1.000 | 0.000 / 0.000 |
| + self-composition, development seeds 0-1 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| + self-composition, **confirmation seeds 2-4**, 1000 steps | 1.000 x3 | 1.000 x3 | **1.000 x3** |

The pre-registered criterion for plain RFT is **not met**: zero at depth 3 on both seeds. But it
answers the 3e-i question outright. GRPO stuck at 0.594 on depth 1 in this machine; the *same*
samples used as cross-entropy targets reach 1.000. The "register-file ceiling" of 3e-i was the
policy-gradient estimator, not the register file -- the same lesson as dispatch (3g), where REINFORCE
on accept/refuse failed and the identical observations as targets worked.

Two further failures, each diagnosed and fixed without introducing anything gold. *Coincidences*: at
depth 2 a buffer of 1538 "verified" programs left accuracy at 0.20, because a wrong program can hit one
instance's answer (``a-b+c = a+b-c`` whenever ``b = c``). A real program is operand-invariant, so each
found program is re-run on its own problem with fresh operands; 70-130 coincidental programs per run
were rejected, and keeping the model's *most probable* verified sample (so one mode per structure, not a
splice of several) lifted depth 2 to 0.89-1.00. *Exploration*: depth 3 still found zero verified programs
in 1500 steps -- a 7-instruction program does not turn up by sampling. *Self-composition* writes one
instead: a depth-d problem is two depth-(d-1) problems and one operator, so the model runs itself on the
two halves, stitches its own programs into the full register layout (right half's pointers shifted), adds
the combining instruction, and verifies the candidate like any sample (answer, then fresh operands).
~6900-7200 composed programs per run passed; the core, trained on them, then emits the whole depth-3
program directly, which is what the held-out number measures. The structure comes from the input (as in
dispatch), the halves are the model's own programs, the check is the answer -- no gold program, no gold
trace. These two fixes were designed after seeing the failures, which is why the confirmation on three
fresh seeds was pre-registered before running; it passes on all three.

Against the project's history: outcome-only induction was 0.015 at depth 3 in 3a-xii, the sixth
retraction. With expert iteration, a fresh-operand check and self-composition it is 1.000 on five of five
seeds. What this does *not* show: depth beyond 3 (the machine here is sized for 3; composition is the
obvious way up), unbalanced trees, operators beyond + and -, or anything with language. Self-composition
needs the problem's structure, which arithmetic hands over in the input and prose does not.

## 3f. Verified self-distillation into a smaller student (lossy; a fallback)

Correction of framing, recorded because the distinction is the point. Distillation
produces a *smaller* student that approximates the teacher; it is lossy by construction
(bounded by the teacher's coverage, subject to mode collapse) and the pipeline is run by an
external harness, not by the model. It is therefore **not** the "1:1 cheaper copy the model
makes of itself" that the north star asks for -- that is 3g, self-compilation, which is
exact and whose content is the model's own emitted program. 3f is kept as the fallback for
regimes where 3g does not apply: task classes whose correct program is *not*
operand-invariant, where no single cached program covers the class and a learned smaller
core is the only compression available. Use 3g wherever the program is structural (all
arithmetic measured so far), 3f only where it is not.

Pre-registered before running. A working teacher (the supervised register machine, or a
GRPO teacher once 3e clears its ceiling) emits programs, and the ones the exact executor
verifies become supervised data for a *smaller* student. Two arxiv surveys (summarised
here) converged on the same recipe and the same holes.

**Why this fits LAMb and not a generic LLM.** The distillation family is rejection-
sampling fine-tuning: sample from the policy, keep only verifier-correct outputs, SFT on
them, iterate (STaR arXiv:2203.14465, ReST arXiv:2308.08998, ReST-EM arXiv:2312.06585,
RFT arXiv:2308.01825; DeepSeek-R1's headline is that SFT-distilling verified traces into
a small dense student beats running RL on the small student, arXiv:2501.12948). Every one
of these fights label noise from an imperfect verifier. LAMb's verifier is the exact
executor, so the kept data is label-noise-free by construction, and for any problem the
teacher *fails*, `gold_program` synthesises a correct program directly -- so the student
never suffers the coverage starvation (no correct depth-3 data) that is the standard
failure of this method and the shape of 3a-xii's collapse.

**The recipe, pre-registered.**
1. Teacher (frozen GRPO+bandit core) samples K programs per problem drawn with
   `exclude_heldout=True`; the exact executor keeps only those whose decoded answer and
   every intermediate are representable and correct (report the drop rate; mask, never
   clip). Problems the teacher never solves get a `gold_program` instead, so coverage is
   total.
2. Canonicalise to the shortest verified program (minimum instruction count, ties broken
   by a fixed order over `(op, ptr_a, ptr_b)` and earliest write), indexing subtrees by
   traversal position with `is`, never by value. Shortest-program/MDL bias is the
   regulariser the program-synthesis literature credits for out-of-distribution
   generalisation (MADIL arXiv:2505.01081, CompressARC arXiv:2512.06104). Keep one extra
   distinct verified path for a minority of problems, against mode collapse.
3. Depth-balance the dataset so the hard depths are represented regardless of the
   teacher's easy-skewed yield -- the coverage control for held-out difficulty.
4. Student: a lower-`d_model`/fewer-layer core, register file, moduli and token-id layout
   byte-identical to the teacher (token ids are positional; never renumber). Loss is the
   existing `gold_program` cross-entropy path in `RegMachineTrainer`, sourced from the
   canonical verified programs rather than the on-the-fly generator -- deliberately the
   same objective the exact path already trusts.

**Success criterion.**

> The student matches the teacher's held-out `answer_acc_hard` by depth, at a fraction of
> the parameters, over >=5 seeds by the permutation test, with **compute matched not
> steps** (the smaller student gets equal FLOPs, the coconut-long lesson). Report pass@1
> *and* pass@k>=8 by depth: pass@1 parity with pass@k regression is the mode-collapse
> signature and is not a pass. Headline is depth-3 held-out accuracy and its sd.

**The holes, recorded before building.** (i) This shares a genome with the retracted
outcome-induction claim; the defence is that supervision here is the full verified
*program*, not the outcome, and that difference is a hypothesis to test in `study.py`, not
an assumption. (ii) A student cannot exceed the teacher's coverage from hard-label SFT
alone; "student beats teacher" needs an on-policy phase (on-policy distillation survey
arXiv:2604.00626) and is a separate claim. (iii) Spurious verified programs (right answer,
wrong path, or a wrapped intermediate that became a legal target) are caught only by
verifying the whole program and masking every intermediate, and by preserving refusal end
to end so a wrong program cannot pass as a confident number. (iv) Iterated filter-then-SFT
saturates in 1-3 rounds and then overfits; cap the iterations. The mechanism is the point:
it generalises to domains with no gold generator (the language bridge), where the teacher's
own verified programs are the *only* supervision available.

## 3g. Self-compilation: the model makes an exact 1:1 cheaper copy of itself

Measured, not pre-registered, because the mechanism turned out to already hold and the job
was to verify it. This is the honest form of "the model creates a cheaper copy of itself":
the copy is exact (1:1, not a lossy student), and its content -- the algorithm -- is the
model's own emitted program, not something a harness designed.

**The mechanism.** The register machine's pointers index register *slots*, not values
(`regmachine.py`), so the program the neural core emits is operand-invariant: the same
`(op, ptr, ptr)` sequence that solves one problem of a structure solves every problem of
that structure (CLAUDE.md: one pointer pattern across 300 depth-3 problems, only operators
varying). The emitted program is thus a complete, exact specification of the core's
behaviour on the whole task class, executable by the algebra with the neural net removed at
~6.5% of a step. "The model understands how it works" is literal here rather than a
metaphor: its reasoning is an inspectable program it emits about its own computation, so
compiling itself is reading its own algorithm off itself -- run once per structure, record
the argmax program, keep the library. `lamb/selfcompile.py`: `compile_model` reads the
library off the core, `CompiledModel.solve` runs the executor-only copy, `verify` measures
fidelity against the core.

**Measured (supervised teacher, 2-digit, `d_model=64`).**

| structure | patterns | distinct programs / pattern | held-out coverage | 1:1 fidelity | compiled acc | speedup |
| --- | --- | --- | --- | --- | --- | --- |
| balanced depth 2 | 8 | 1 | 1.000 | **1.000** | 1.000 | 3.1x |
| balanced depth 3 | 128 | 1 | 1.000 | **1.000** | 1.000 | 3.7x |
| unbalanced (shape=1, depth<=3) | 200 | 1 | 1.000 | **1.000** | 1.000 | -- |

`fidelity` is the fraction of held-out problems where the compiled copy reproduces the
*core's own* decoded output, not merely the truth -- the copy claim. Every structure needed
exactly one program (`max_variants == 1`), so the class is fully compilable: 8 programs *are*
the depth-2 core, 128 *are* the depth-3 core, 200 *are* the variable-shape core, exactly. The
3-4x is a floor measured on a tiny core with an unoptimised Python executor; the asymptotic
is set by the 6.5%/93.5% execution/forward split, i.e. ~15x once the forward dominates.

**The key must encode shape, not just operators.** The first version keyed the library by a
flat post-order operator sequence, which is wrong for unbalanced trees: two shapes can share
an operator sequence but route differently (`((a+b)+c)+d` and `(a+b)+(c+d)` are both
`[+,+,+]`). Measured on the shape=1 core, the flat key collided on 28 patterns (up to 2
programs each) while the nested key `(op, left, right)` gave 200 keys with zero collisions and
fidelity 1.000. The fix matters because real problems are not balanced trees, so the general
self-copy needs a structural key -- `op_pattern` in `lamb/selfcompile.py` is now that.

**It holds for division.** With the rational register file (`+ - * /`, fractions decoded as
numerator/denominator), 1-digit operands: a depth-1 core compiles to 4 programs and a depth-2 core
to 64 (= 4^3), one program per structure, coverage 1.000, fidelity 1.000, compiled accuracy 1.000,
no refusals. The exact self-copy is not an artefact of integer `+/-`. One cost finding came with it:
the default `RATIONAL_MODULI` pad ten moduli to width 271, which made a rational training step 5.1 s
against 0.16 s for integer -- 32x, the padding trade `CLAUDE.md` warns about, here on the losing side.
A ring sized to the task (`(16, 25, 27, 11, 7, 13, 37)`, product 4.4e7, against a worst unreduced
magnitude of 9^4 = 6561 at depth 2) gives 0.38 s/step with no rows dropped out of range. Size the
rational ring to the workload, as GSM8K_MODULI already does, rather than defaulting to the widest.

**Multi-agent decomposition, measured.** `solve_hierarchical` (`lamb/selfcompile.py`) solves
a task deeper than any single copy using only cheaper copies: a subtree within a copy's
depth is solved by that copy, and two sub-results are combined by a *depth-1 copy* solving
``"v_left op v_right"`` -- not by Python arithmetic, so every step of the reasoning is a
cheap 1:1 copy. It works at any magnitude because the copies' programs are operand-invariant.
Measured: depth-4 problems, which a depth-2 core cannot solve monolithically at all, solved
by orchestrating a depth-1 copy (2 programs) and a depth-2 copy (8 programs) at **1.000
accuracy over 200 tasks, ~7 copy-calls each**. That is length generalisation by
decomposition, exact because the copies are exact.

**The boundary, kept honest.** The copy's algorithm is the model's; the caching, lookup, and
*the decomposition routing* are a harness. What is not yet the model's: the decision to
compile and to route. Full autonomy -- the model choosing at runtime to compile a structure
it keeps seeing and emitting a program that dispatches subtasks to its cheaper copies --
needs one architectural addition: a **dispatch instruction** the core can emit, whose
executor runs a named compiled copy on a sub-expression and writes the returned value into a
register.

**Phase A done: the decomposition is now an emittable object.** `gold_dispatch_program` and
`execute_dispatch_program` (`lamb/selfcompile.py`) make the decomposition an explicit
program -- a flat list of `("dispatch", subtree, depth)` and `("combine", op, i, j)` steps
over a working register list, every step a cheap-copy call, no Python arithmetic. This is the
form a model can *emit*, as opposed to `solve_hierarchical`'s harness control flow. The
executor is exact (each step is a fidelity-1.000 copy) and refuses on a read-ahead or
uncovered step rather than guessing; measured 1.000 on depth-4 tasks via depth-1/2 copies.
What remains is Phase B: the core *emitting* the dispatch program instead of
`gold_dispatch_program` writing it. The task space is variable-shape arithmetic, where the
choice is real -- the granularity of the carve (dispatch a depth-2 subtree whole vs split it)
trades copy-calls against available copy depths, so there is a cost-optimal program to learn
rather than a forced one. Pre-registration for that learned step:

> A depth-D task solved by a program that dispatches its depth-<=k subtrees to compiled
> copies reaches the same held-out accuracy as the monolithic core at strictly lower cost,
> over >=5 seeds. The copies must be exact (3g fidelity 1.000), so the only thing being
> tested is whether the core can emit a correct dispatch structure -- which is the same
> program-induction question as 3e, one level up.

**Phase B, measured: the core writes its own dispatch program** (`lamb/dispatch.py`). At
every internal node the core votes dispatch-or-split; leaves always dispatch; applied top
down the votes determine one program, which the exact executor runs. The copies cover every
subtree of depth <= 2 and only *balanced* depth-3 subtrees (the depth-3 copy was compiled on
balanced trees), so the cost-optimal program needs the core to know what its copies can do:
an unbalanced depth-3 subtree must be split, a balanced one should not be (on shape=1 trees
the depth-3 subtrees split 74 covered / 131 not, so the question is genuinely two-sided).
The core is causal, so each node is read at the end of its span, where the whole subtree has
been seen, concatenated with the state at its start. Trained on shape=1 depth <= 4, evaluated
held-out on depth 4 and on unseen depth 5, every program *executed*, two seeds per arm:

| signal | depth 4 acc | calls (gold 3.17, split 7.01) | depth 5 acc (unseen) | calls (gold 4.18, split 9.01) |
| --- | --- | --- | --- | --- |
| `probe`: query copies on every subtree | 1.000 / 1.000 | 3.17 / 3.17 | 0.940 / 0.980 | 3.87 / 4.11 |
| `outcome`: whole-program reward | 1.000 / 0.810 | 7.01 / 4.34 | 1.000 / 0.680 | 9.01 / 4.51 |
| `feedback`: per-attempt accept/refuse | 0.745 / 0.685 | 2.05 / 2.35 | 0.600 / 0.470 | 2.27 / 1.86 |

No arm uses an external label. `probe`'s target is `covering_depth` -- literally "would my
copy accept this subtree", answered by the copy's own library -- so it is the core querying
its copies about every subtree and learning to predict their competence. Applied top down that
prediction *is* the cost-optimal rule, and the core reproduces it exactly on held-out depth 4
(vote accuracy 1.000, calls equal to gold, 2.2x fewer than always-split) and mostly transfers
one level deeper, erring toward over-dispatch there (fewer calls than gold, 2-6% refusals).
The coverage oracle used during training agreed with real execution on every evaluated row.

**Learning the self-model from experience alone: it converges, once the observation is used
as a target rather than a reward.** Whole-program reward collapses to always-split --
splitting is always valid, early high dispatches fail, the dispatch probability runs to zero,
and every group becomes identical; an epsilon floor (the program_grpo fix) moves one seed off
it but into over-reaching. One bad vote zeroes the whole program, so the credit is unassignable.
Rewarding each attempted hand-off by whether its copy accepted it (`feedback`, REINFORCE, zero
baseline) fixes the credit and still fails. This was first put down here to the signal being
starved; that diagnosis was wrong. A per-category breakdown of the votes shows the failure is
specific: `feedback` learns to dispatch everything its copies take (depth 2, balanced depth 3:
1.00) but never learns to stop handing off what they refuse (unbalanced depth 3: 0.27, depth
>= 4: 0.32 still dispatched), and it plateaus there from step 1000 to 2000.

The same observations used as *prediction targets* fix it. `experience` mode trains the dispatch
head, by binary cross-entropy, to predict whether its copy will accept a subtree -- on exactly
the subtrees it chose to hand off while solving, sampled with an epsilon floor so uncertain ones
keep getting tried -- and dispatches where acceptance is predicted. A node it never tried
contributes nothing; `probe`, by contrast, asks the copies about every subtree of every problem.
Pre-registered criterion: both seeds reach held-out depth-4 accuracy >= 0.95 with vote accuracy
within 0.02 of `probe`. It passed, and was then run to five seeds against five of `probe`:

| | depth 4 (5 seeds each) | depth 5 unseen acc | depth 5 vote acc | queries per problem |
| --- | --- | --- | --- | --- |
| `probe` (asks about every subtree) | 1.000, 3.17 calls, sd 0 | 0.958 +- 0.020 | 0.988 +- 0.006 | ~3.0 |
| `experience` (only its own hand-offs) | 1.000, 3.17 calls, sd 0 | 0.952 +- 0.016 | 0.985 +- 0.004 | 1.6-2.0 |

Exact paired permutation tests: depth-5 accuracy difference -0.006, p = 0.69; vote accuracy
-0.003, p = 0.38. Indistinguishable -- with five seeds at this spread the design cannot resolve
differences below ~0.03, and none is claimed. The per-category votes converge by step 500 to
exactly 1 / 1 / 0 / 0. Run end to end against copies the cores compiled from themselves, it gives
the same result (depth 4: 1.000 at 3.17 calls; depth 5: 0.935). A plausible reason for the
feedback/experience gap, not separately tested: the cross-entropy gradient is proportional to the
prediction error, so correct confident predictions stop pushing and the remaining errors get the
signal, while the zero-baseline policy gradient keeps pushing on every accepted hand-off, and
through the shared representation that holds up similar-looking subtrees the copies refuse.
**End to end, no gold anywhere in the copies.** The table above used copies built from gold
programs for speed. Re-run with copies the cores compiled from themselves: depth-1/2/3 cores
trained to 1.000, each compiled to 2 / 8 / 128 programs at fidelity 1.000 with one program
per structure, then the `probe` dispatch policy trained against *those* copies: depth 4
accuracy 1.000 at 3.17 calls (= gold), unseen depth 5 0.940 at 3.87 calls, vote accuracy
0.983, oracle agreement 1.000 -- the same numbers as the gold-copy run. So the whole chain is
the model's: cores that compile themselves into exact copies, and a core that has learned what
those copies can do and writes the cheapest program routing a task across them.

What this establishes for the self-copy goal: the model can hold an accurate, learned model of
its own copies' competence and use it to write the cheapest program that routes a task across
them, and it can acquire that self-model from its own task experience alone -- the hand-offs it
made while solving and whether each was taken -- as well as it can by querying its copies
directly, using fewer interactions. Deciding *when* to compile a new copy was the remaining harness
step; Phase C below takes it over, with a fixed expected-value trigger reading only the model's
own experience.

**Phase C, pre-registered: the model decides when to compile a new copy of itself.** Until now
compilation was a harness step. Here the model runs a stream of deep tasks with copies that
cover depth <= 2 and *no* depth-3 structure. Work its copies refuse is split (a refused hand-off
falls back to splitting, so misrouting costs calls and never correctness). Each depth-3 subtree
it had to split is logged by structure key with the value it obtained the slow way. When a key
has recurred enough that a copy would pay for itself -- (times seen) x (calls saved each time)
>= (instances checked) x (cost of one core forward in copy-calls, 4, measured 3-4x in 3g) -- it
compiles one from its own neural core: runs the core on the logged instances and adopts the
program only if the core emits the *same* program on every instance and that program reproduces
the values it already computed through its trusted copies. The copy is verified against the
model's own decomposition; there is no external oracle. Routing (experience mode, online) must
then discover that the copy now accepts that work.

The boundary, stated in advance: the trigger is a fixed expected-value rule; what it decides on
-- which structures recur, what they cost, whether the core's program is consistent and right --
is entirely the model's own experience. Criteria, three seeds, same stream per seed:

> 1. By the end of the stream, greedy calls per task under self-directed compilation are within
>    10% of a copy precompiled up front over the core's own distribution.
> 2. Every adopted copy passes self-verification, and executed accuracy stays 1.000 throughout.
> 3. Self-directed compilation's total cost over the stream (copy calls including exploration,
>    plus 4 x core forwards spent compiling) is below never compiling.

Up-front compilation's total cost is reported beside it, with the caveat that it depends on how
the structure space is enumerated in advance, which the self-directed version never needs to know.

**Phase C, measured (`lamb/selfimprove.py`): all three criteria pass on all five seeds.** The
core is one variable-shape depth-3 register machine at accuracy 1.000; the stream is 6400
shape=1 depth<=5 tasks (200 rounds of 32); routing learns online in experience mode from
scratch; the same stream per seed for all three arms.

| arm | final greedy calls/task (ideal 2.86) | accuracy | core forwards | total cost/task over stream |
| --- | --- | --- | --- | --- |
| never compile | 5.36-5.37 | 1.000 | 0 | 6.16 +- 0.06 |
| compile up front | 2.86-2.92 | 1.000 | 3000 | 5.59 +- 0.08 |
| **model decides when** | **2.86-2.90** | **1.000** | **576** | **4.59 +- 0.08** |

Starting with no depth-3 copy at all, every seed discovered and compiled the entire depth-3
structure space of the workload -- 192 structures, three core forwards each -- purely from work it
kept doing the slow way: ~110-120 structures by round 50, all 192 by round 150. Every one passed
self-verification (192 adopted, 0 rejected per seed), executed accuracy was 1.000 at every
checkpoint, and routing to the new copies rose from 0 to 0.98-1.00 on its own, i.e. the
experience-mode self-model tracked a copy set that was changing under it. Criterion 1: final calls
within 2% of up-front and at the ideal. Criterion 3: total cost 25% below never compiling (paired
difference -1.57 per task) and also 18% below up-front (-0.99). Every seed agrees in direction; the
exact paired two-sided p is 0.0625 for both, which is the floor at five seeds -- consistency, not a
significance claim beyond that.

Where the advantage over up-front comes from, precisely: self-directed spends *more* copy calls on
the stream (26.4-27.9k against 23.1-24.5k), because it does the early work the slow way before it
has compiled anything. It wins on total only through compile cost: 576 forwards, exactly three per
structure it actually needed, found without knowing the structure space, against 3000 sampled
forwards to be sure of covering it. A targeted up-front enumeration would also cost 576 -- but only
for something that already knew the space. The holes: (i) the forward cost of 4 copy-calls was
measured on tiny cores; at scale it is larger (~15x), which delays the trigger and makes up-front
dearer, so the numbers move though the direction should not; (ii) the core here is perfect, so the
rejection path never fired in the stream -- it is exercised only by a unit test with an untrained
core, which it correctly refuses; (iii) the core's scope (depth <= 3) is given to the compiler, not
discovered; (iv) this workload uses its whole structure space, so "compile only what you need" is
not tested against a workload that needs a subset, where it should matter more.

**Phase D, pre-registered: the model discovers its core's reach, and the compile rule is
learned.** Two of Phase C's holes, closed together because they are one mechanism. The compiler
is no longer told its core handles depth <= 3: any work no copy takes is a candidate, at any
depth, and the core's reach is found out by trying (an instance the core cannot even load counts
as a failed attempt at full cost -- no free peek at its architecture). To make reach non-trivial,
the core is trained on *balanced* depth-3 trees only; whether it can back unbalanced depth-3 trees,
or skinny depth-4 trees that fit its register file, is for the model to find out. The fixed rule's
two baked-in assumptions -- every compile succeeds; a copy is worth it once past history alone paid
for it -- are replaced by learned predictions on the routing core's own representation of the
structure: a *reach head* (will my core back this? cross-entropy on its own verified attempt
outcomes, generalising to structures never tried) and a *rate head* (how often will this recur?
Poisson likelihood on logged occurrences). It attempts when p_backed x rate x calls_saved x
rounds_remaining > check cost, with a 5% exploration chance. Stated in advance: the decision's form
is expected value and the horizon is known; what is learned is its inputs. Arms on the same stream
per seed: fixed rule told its reach (Phase C), fixed rule told nothing ("blind"), learned rule told
nothing, and an oracle-reach reference in which failed attempts are free. Five seeds.
*Amended before the learned arm ran, under shared-machine limits: three seeds, 150 rounds, the
oracle arm dropped. One partial result was seen before amending -- seed 0's blind arm at 200
rounds (191 adopted: 128 depth-3 and 63 depth-4; 657 forwards wasted).*

> 1. Reach discovery: on structures it never attempted, the learned reach head separates what the
>    core backs from what it does not with AUC >= 0.9 (truth from an uncharged offline check).
> 2. Waste: the learned rule spends fewer core forwards on failed attempts than the blind fixed
>    rule, on every seed.
> 3. Cost: the learned rule's total cost is below the blind fixed rule's on every seed.
> 4. Executed accuracy 1.000 throughout; every adopted copy passes self-verification.

**Phase D, measured: discovery by trying works; a learned model of reach does not; the learned
rule's gain is small.** Three seeds, 150 rounds of 32 depth<=5 tasks, balanced-only depth-3 core
(accuracy 1.000), same stream per seed across arms:

| arm | total cost/task | forwards wasted on failed attempts | copies adopted | reach-head AUC, untried structures |
| --- | --- | --- | --- | --- |
| fixed rule, told reach | 6.64 / 6.65 / 6.50 | 192 | 128 depth-3 | -- |
| fixed rule, blind | 6.94 / 7.03 / 6.78 | 435 / 492 / 432 | 128 depth-3 + 34-39 depth-4 | -- |
| learned rule, blind | 6.60 / 6.96 / 6.67 | 405 / 423 / 396 | 127-128 depth-3 + 30-32 depth-4 | 0.608 / 0.911 / 0.415 |

Criterion 1 **fails**: the reach head reaches AUC >= 0.9 on unattempted structures on one seed of
three, and is below chance on another. The model does not learn a generalising model of what its
core can back. Criteria 2 and 3 pass, narrowly: on every seed the learned rule wastes 7-14% fewer
forwards and costs 0.07-0.34 per task less than the blind fixed rule -- but with the reach head not
generalising, the saving plausibly comes from the rate/horizon side (attempting less overall, not
attempting what will fail), which this design does not separate. Criterion 4 passes: accuracy 1.000
in every arm, every adopted copy verified, and this time the rejection path ran in earnest (64
unbalanced depth-3 structures refused in the told arm; 132-164 refusals in the blind arms).

The finding worth keeping is what trying discovered. A core trained only on balanced depth-3 trees
backs 30-39 *depth-4* structures that nobody specified: skinny trees small enough for its register
file whose +/- pattern makes them algebraically equal to a balanced program, found only by attempting
them and checking the core's program against values the model had already computed through its own
copies. It also correctly finds that it cannot back unbalanced depth-3 trees. So the model's reach is
not its training distribution, and only experience reveals it. But in this stream finding that reach
costs more than it saves -- the depth-4 structures are rare, mostly one-off keys -- so being *told*
the reach is still cheapest, and the blind arms pay for their discoveries.

Holes, stated plainly. (i) The reach head reads the *routing* core's representation, which is trained
to route, not to encode whether a sign pattern is associative-equivalent to a balanced program; a
reach model probably needs its own features or the compile outcomes fed back into that core. (ii)
Copies backed by algebraic equivalence are verified on three instances: for +/- programs, which are
linear with integer coefficients, agreement on three random 2-digit points makes a wrong program very
unlikely but is not a proof (the same holds for every copy compiled so far). (iii) Three seeds and 150
rounds, reduced from the pre-registered five on a shared machine; the amendment is recorded above.
(iv) The decision's form is expected value; only its inputs were learned, and only one of them (rate)
plausibly did anything. An engineering note: the learned compiler first re-featurised every structure
it had ever logged each round, so cost and memory grew with the log (~1.9 GB per process, slowing);
it now trains each round on tried and candidate structures plus a capped sample of the rest.

The holes to watch, from the same scars as everywhere else: an uncovered structure must be
reported, never silently run on the core (a copy that falls back to the original is not a
copy); the dispatch op's executor must be exact or it breaks the 1:1 property; and whether
the *core can learn to emit dispatch programs* is a learned claim and goes through
`lamb/study.py` like every other.

## 4. Deeper test-time memory (ATLAS)

Upgrade the linear delta-rule memory to a small non-linear MLP memory with a
momentum ("surprise") term and Muon/second-order-style updates, targeting the
10M-token regime. Wire the Rust `TopKStore` as an exact retrieval tier alongside
the compressive neural memory.

Refs: Titans; ATLAS ([2505.23735](https://arxiv.org/abs/2505.23735)).

## 5. Richer task space

Multi-term nested expressions, division/modulo, mixed radices, rationals, and
eventually symbolic/algebraic goals — the combinatorial space where the
hypernetwork proposer (item 1) earns its keep. The Rust verifier already parses
general `+ − * ( )`; extend its grammar and the tokenizer in lockstep.

## 6. Number representation experiments

A/B the value channel against BitTokens (IEEE-754 as one token) and Adelic
operation-preserving embeddings; sweep digit base and reversal.

Refs: *Numbers Already Carry Their Own Embeddings* ([2606.14108](https://arxiv.org/html/2606.14108v1)); BitTokens.

## 7. Scale-up path (GPU + CPU + RAM hybrid) — implemented

`lamb/device.py` makes every entry point device-agnostic and mixed-precision-ready
with no architecture change. `resolve_device` auto-selects `cuda` > `mps` > `cpu`
(`--device`, `LAMB_DEVICE`); `Amp` turns on bf16/fp16 autocast (+ gradient scaler)
on CUDA automatically and stays fp32 on CPU so the CPU path never regresses; the
exact Rust kernels keep running on CPU alongside the accelerator and RAM holds the
buffers/stores; `--threads` gives the CPU legs every core. `--scale
{tiny,small,base,large}` grows width/depth/batch together (0.28M → 1.98M → 7.90M →
31.5M params). Wired into `train`, `coconut`, `comm`, `comm_pop`; covered by
`tests/test_device.py`. The GPU path is unit-tested on CPU (forced bf16 autocast);
it has not been run on a physical GPU in this repo (CI is CPU-only), but
`--device auto` picks CUDA up with no code change.

Remaining: multi-GPU (FSDP / tensor-parallel) for the `large`+ regime, a serving
mode, and `torch.compile` once shapes are static.
