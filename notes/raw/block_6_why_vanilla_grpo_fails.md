# Block 6 — Why Vanilla GRPO Fails

## 1. The objective contains several hidden weighting choices

Vanilla GRPO approximately uses

\[
\hat A_i
=
\frac{R_i-\bar R}{\sigma_R}
\]

and

\[
J
=
\frac1G
\sum_i
\frac1{T_i}
\sum_t
\min
\left[
\rho_{i,t}\hat A_i,\,
\operatorname{clip}(\rho_{i,t})\hat A_i
\right].
\]

Important terms:
- \(\sigma_R\): reward normalization,
- \(T_i\): response-length normalization,
- \(\rho\): PPO probability ratio,
- \(R\): sparse verifier reward.

Each changes effective optimization pressure.

---

## 2. Failure mode 1: zero-variance groups

All wrong:

\[
[0,0,\ldots,0].
\]

All correct:

\[
[1,1,\ldots,1].
\]

Both have

\[
\sigma_R=0.
\]

There is no relative ranking information.

All-correct is often acceptable: the problem is already mastered.

All-wrong is dangerous:

\[
\boxed{
\text{the model knows everything failed,
but gets no direction toward success}.
}
\]

This is exploration starvation.

---

## 3. Mixed-group probability

If true success probability is \(p\),

\[
P_{\text{mixed}}
=
1-p^G-(1-p)^G.
\]

Thus:
- very hard prompts: mostly all fail,
- intermediate prompts: many mixed groups,
- very easy prompts: mostly all succeed.

Vanilla GRPO naturally focuses learning on partially solvable prompts.

---

## 4. DAPO dynamic sampling

DAPO filters/oversamples to keep informative prompts with both successes and failures.

This improves batch utilization.

But if a prompt remains all-failure under repeated sampling, dynamic sampling does not solve the fundamental exploration problem.

So:

\[
\boxed{
\text{dynamic sampling fixes wasted compute more directly than exploration starvation}.
}
\]

---

## 5. Failure mode 2: reward-standard-deviation normalization

Vanilla GRPO uses

\[
\hat A_i
=
\frac{R_i-\bar R}{\sigma_R}.
\]

For binary reward with \(G=8\):

- \(1/8\) correct: \(A_+\approx2.65\)
- \(4/8\) correct: \(A_+=1\)
- \(7/8\) correct: \(A_+\approx0.38\)

The same reward \(R=1\) receives different effective gradient weight.

Therefore

\[
\boxed{
1/\sigma_R
\text{ acts as implicit prompt weighting}.
}
\]

This is not merely variance reduction.

---

## 6. Why “std normalization is bad” is too simplistic

Standard-deviation normalization changes the effective optimization objective and weighting.

That may introduce undesirable difficulty bias, but it can also act as adaptive gradient rescaling.

The scientifically correct question is:

> What quantity is being reweighted, and does that weighting help the target objective?

---

## 7. Dr.GRPO response

Dr.GRPO removes per-group reward std normalization:

\[
\frac{R_i-\bar R}{\sigma_R}
\]

becomes approximately

\[
\boxed{
R_i-\bar R.
}
\]

This removes the explicit \(1/\sigma_R\) prompt-dependent amplification.

---

## 8. Failure mode 3: response-length normalization

Vanilla GRPO commonly uses

\[
\boxed{
\frac1{T_i}\sum_tL_{i,t}.
}
\]

This gives each response roughly equal total weight regardless of length.

If one response has 100 tokens and another 1000, each token in the short response has about 10x the per-token influence.

---

## 9. Length-bias consequence

For positive advantage:
- short correct responses receive stronger per-token reinforcement.

For negative advantage:
- long incorrect responses spread punishment across many tokens, weakening per-token suppression.

Therefore training can produce:

\[
\boxed{
\text{incorrect responses getting longer without becoming better}.
}
\]

This means increasing reasoning length is not itself evidence of deeper reasoning capability.

---

## 10. Dr.GRPO response to length bias

Dr.GRPO removes variable response-specific normalization and uses a constant denominator such as a fixed generation budget.

Conceptually:

\[
\frac1{T_i}\sum_tL_{i,t}
\]

becomes

\[
\boxed{
\frac1{T_{\max}}\sum_tL_{i,t}.
}
\]

The constant rescales gradients without making longer responses inherently weaker per token.

---

## 11. DAPO token-level loss

DAPO takes a token-level aggregation approach rather than equal-weighting each response first.

This allows long correct trajectories to contribute more total signal and long bad trajectories to receive stronger aggregate suppression.

Dr.GRPO and DAPO address related length issues in different ways.

---

## 12. Reasoning length as a confounder

If accuracy improves while average reasoning length rises sharply, the gain could come from:
- better reasoning,
- more inference compute,
- both.

Therefore matched token budgets are essential for clean interpretation.

---

## 13. Failure mode 4: symmetric PPO clipping can suppress rare-token growth

Standard PPO uses approximately

\[
\rho\in[1-\epsilon,1+\epsilon]
\]

inside the surrogate objective.

Suppose a rare useful token has

\[
\pi_{\text{old}}=0.001
\]

and \(\epsilon=.2\).

The positive-advantage incentive saturates around

\[
0.0012.
\]

That is only a tiny absolute probability increase.

By contrast, a common token with probability \(0.5\) can move to \(0.6\) before the same relative clip saturates.

Thus symmetric clipping can favor sharpening existing high-probability behavior over rapidly amplifying rare behavior.

---

## 14. DAPO Clip-Higher

DAPO uses asymmetric bounds:

\[
[1-\epsilon_{\rm low},1+\epsilon_{\rm high}]
\]

with

\[
\epsilon_{\rm high}>
\epsilon_{\rm low}.
\]

This gives positive-advantage rare tokens more room to grow, with the goal of preserving exploration.

---

## 15. Entropy collapse

Positive feedback can cause:

\[
P(A)\uparrow
\rightarrow
A\text{ sampled more}
\rightarrow
\text{more positive updates}
\rightarrow
P(A)\uparrow\uparrow.
\]

Alternative modes lose probability and token entropy falls.

If:
- pass@1 rises,
- entropy falls sharply,
- pass@64 coverage barely moves,

this is strong evidence for sharpening/exploitation.

But token entropy is only a diagnostic; it is not the same as semantic reasoning diversity.

---

## 16. Failure mode 5: sparse terminal reward and credit assignment

Suppose a 2,000-token solution is correct until one arithmetic slip near the end.

Binary verifier gives

\[
R=0.
\]

The whole trajectory can receive negative advantage.

Conversely, messy reasoning with a lucky correct final answer may receive

\[
R=1.
\]

Thus:

\[
\boxed{
\text{terminal RLVR reward gives coarse sequence-level credit}.
}
\]

Neither Dr.GRPO nor DAPO fully solves this fundamental issue.

---

## 17. Why terminal RLVR can still work

Possible reasons:
1. pretraining already provides useful local reasoning structure,
2. good patterns correlate statistically with successful outcomes,
3. repeated sampling reinforces recurring successful motifs,
4. math/code verifiers can provide clean outcome labels.

This supports the hypothesis that RLVR may often reweight pre-existing reasoning structures rather than learn reasoning from scratch.

---

## 18. Failure mode 6: truncation and overlong reward noise

If a rollout is truncated at \(T_{\max}\), assigning a hard negative reward may conflate:
- mathematical failure,
- excessive verbosity,
- unfinished but valid reasoning,
- formatting problems.

That injects noisy supervision.

DAPO uses overlong filtering/reward shaping to reduce this ambiguity.

---

## 19. Failure mode 7: reward hacking / verifier exploitation

RL optimizes the verifier, not an abstract notion of true reasoning.

If the verifier has exploitable shortcuts, reward can increase without genuine reasoning quality increasing.

Thus:

\[
\boxed{
\text{reward increase}
\neq
\text{reasoning improvement}.
}
\]

Verifier auditing is essential.

---

## 20. Optimizer bias versus exploration limitation

### Optimizer bias

Useful sampled information exists, but the objective weights it strangely.

Examples:
- \(1/T_i\) response normalization,
- \(1/\sigma_R\) reward normalization.

Dr.GRPO primarily targets these.

### Exploration limitation

Useful behavior is never sampled.

Example:

\[
p_{\text{success}}=10^{-8}
\]

and every group is all-failure.

No normalization fix can reward a trajectory that never appears.

This distinction is central to sharpen vs expand.

---

## 21. Failure map

| Component | Pathology | Observable signature | Typical response |
|---|---|---|---|
| Group-relative reward | all-0/all-1 groups | zero comparative signal | dynamic sampling |
| Divide by \(\sigma_R\) | prompt reweighting | rare mixed outcomes get large weight | Dr.GRPO removes std normalization |
| Divide by \(T_i\) | length bias | long failures weakly suppressed | Dr.GRPO constant normalization |
| Per-sequence loss | long-token underweighting | long patterns weakly trained | DAPO token-level loss |
| Symmetric clipping | rare-mode growth bottleneck | exploration tokens rise slowly | Clip-Higher |
| Positive feedback | entropy collapse | dominant modes take over | exploration-preserving changes |
| Sparse terminal reward | credit assignment | whole sequence rewarded/punished | still open |
| Hard truncation penalty | noisy supervision | unstable length/reward | overlong shaping |
| Imperfect verifier | reward hacking | reward ↑, true quality not ↑ | verifier auditing |

---

# Research Checkpoint — Questions and Answers

## Q1

Why is an all-failure group more concerning than an all-success group?

### Answer

Both have zero reward variance, but:
- all-success means the problem is already reliably solved,
- all-failure means the policy receives no directional signal about how to improve.

The latter is exploration starvation.

---

## Q2

Why is dividing by reward std implicit prompt weighting? Why is calling it “bad normalization” too simplistic?

### Answer

Because

\[
A_i=(R_i-\bar R)/\sigma_R
\]

multiplies the centered reward by a prompt-specific factor \(1/\sigma_R\).

Prompts with lower group variance can receive larger effective gradient weights.

Calling it simply bad is too simplistic because adaptive scaling can sometimes improve optimization; whether it is desirable depends on the target objective.

---

## Q3

Two incorrect trajectories have equal negative advantage, but one has 100 tokens and the other 2,000.

Why can the long failure receive weaker per-token suppression?

### Answer

With

\[
\frac1{T_i}\sum_tL_{i,t},
\]

the 2,000-token trajectory spreads the same response-level weight across 20x more tokens.

This can make long bad reasoning weakly penalized and create misleading “long reasoning” behavior without genuine capability gains.

---

## Q4

A successful token has old probability \(0.001\). Why can ordinary PPO clipping make its amplification slow, and why does this matter more for expansion?

### Answer

With \(\epsilon=.2\), the direct positive-advantage incentive saturates around a relative ratio of \(1.2\), corresponding to only about

\[
0.001\rightarrow0.0012.
\]

That is a tiny absolute increase.

Sharpening high-probability modes can produce larger absolute probability shifts under the same relative clip.

Expansion-like behavior often requires lifting very low-probability modes, so this matters more there.

---

## Q5

Why does dynamic sampling improve effective batch utilization without solving exploration starvation?

### Answer

It filters/oversamples prompts that already produce mixed groups, avoiding wasted all-correct/all-wrong batches.

But if a hard prompt never generates a success, repeated resampling still produces all-failure groups.

So it improves utilization of available signal rather than creating signal from nothing.

---

## Q6

GRPO and Dr.GRPO have the same pass@64 coverage, but Dr.GRPO has higher pass@1, shorter incorrect solutions, and better token efficiency.

What conclusion should we make?

### Answer

The evidence suggests Dr.GRPO fixed optimization/weighting pathologies—especially reward normalization and response-length bias—so it learned more efficiently from already accessible successful behavior.

Because high-\(k\) problem coverage did not expand, there is little evidence that the accessible capability set broadened.

This is improved sharpening/optimization, not evidence of capability expansion.

---

# Block 6 Summary

Vanilla GRPO can fail not because the core idea is wrong, but because several implementation/objective choices induce unintended weighting and exploration effects.

The most important distinction is:

\[
\boxed{
\text{optimizer bias}
\neq
\text{exploration failure}.
}
\]

Dr.GRPO mostly addresses how sampled behavior is weighted.

DAPO additionally targets usable sampling, token weighting, clipping asymmetry, and length/reward issues.

Neither can reward successful behavior that the policy never samples.
