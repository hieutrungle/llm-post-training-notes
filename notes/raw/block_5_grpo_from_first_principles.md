# Block 5 — GRPO from First Principles

## 1. GRPO's central substitution

PPO typically uses a critic/value function to estimate advantage.

GRPO replaces:

\[
\boxed{
\text{learned critic/value baseline}
\rightarrow
\text{group-relative reward baseline}.
}
\]

For one prompt \(q\), sample

\[
o_1,\ldots,o_G
\sim
\pi_{\text{old}}(\cdot|q)
\]

and score them:

\[
R_1,\ldots,R_G.
\]

---

## 2. Group-relative advantage

Define

\[
\bar R
=
\frac1G\sum_jR_j
\]

and group reward standard deviation \(s_R\).

Vanilla GRPO uses approximately

\[
\boxed{
\hat A_i
=
\frac{R_i-\bar R}{s_R}.
}
\]

For outcome-level supervision, the same sequence advantage is applied to every token of response \(i\).

---

## 3. Binary reward closed form

Suppose \(G\) rollouts contain \(k\) correct answers.

Then

\[
\hat p=\frac{k}{G}.
\]

For binary reward:

\[
\bar R=\hat p
\]

and

\[
s_R=
\sqrt{\hat p(1-\hat p)}.
\]

For a correct rollout:

\[
\boxed{
A_+
=
\sqrt{
\frac{1-\hat p}{\hat p}
}
=
\sqrt{
\frac{G-k}{k}
}.
}
\]

For a wrong rollout:

\[
\boxed{
A_-
=
-\sqrt{
\frac{\hat p}{1-\hat p}
}
=
-\sqrt{
\frac{k}{G-k}
}.
}
\]

---

## 4. Rare successes get large positive advantage

Example \(G=8\), \(k=1\):

\[
A_+
=
\sqrt7
\approx2.65
\]

and

\[
A_-
=
-\sqrt{\frac17}
\approx-0.38.
\]

Thus one rare success gets strong reinforcement.

If \(k=7\), the rare failure gets the strong negative advantage instead.

GRPO emphasizes outcomes that are surprising relative to the group.

---

## 5. Put the advantage into PPO

For response \(i\), token \(t\),

\[
\rho_{i,t}
=
\frac{
\pi_\theta(o_{i,t}|q,o_{i,<t})
}{
\pi_{\text{old}}(o_{i,t}|q,o_{i,<t})
}.
\]

Then use a PPO-style clipped term:

\[
\min
\left[
\rho_{i,t}A_i,
\operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)A_i
\right].
\]

Schematic objective:

\[
\boxed{
J_{\text{GRPO}}
\approx
\frac1G
\sum_i
\frac1{|o_i|}
\sum_t
L_{\text{clip}}(\rho_{i,t},A_i)
-
\beta KL.
}
\]

---

## 6. Why group comparison is useful

A reward of \(1\) on a very hard problem is more surprising than a reward of \(1\) on an easy problem.

Group normalization automatically changes effective gradient weight according to the group's performance.

This gives GRPO a critic-free estimate of “better or worse than expected for this prompt.”

---

## 7. Group mean as a critic approximation

If

\[
V(q)
=
E[R|q],
\]

then the group mean

\[
\bar R
=
\frac1G\sum_iR_i
\]

acts as a Monte Carlo estimate of \(V(q)\).

So

\[
R_i-\bar R
\]

is conceptually advantage-like.

---

## 8. Why this is not identical to a perfect critic

The group mean is noisy.

Each rollout contributes to its own baseline.

Ignoring std normalization,

\[
R_i-\bar R
=
\frac{G-1}{G}R_i
-
\frac1G\sum_{j\neq i}R_j.
\]

Thus GRPO has finite-group statistical structure distinct from simply inserting an exact value function.

---

## 9. Group size \(G\)

If true success probability is \(p\), the probability that a group contains at least one success is

\[
\boxed{
1-(1-p)^G.
}
\]

Example \(p=.01\):

- \(G=8\): \(\approx7.7\%\)
- \(G=64\): \(\approx47.4\%\)

Larger groups expose rare existing success more effectively.

But if

\[
p=10^{-8},
\]

even \(G=64\) is essentially useless.

Thus larger \(G\) helps discover rare accessible modes, not effectively unreachable ones.

---

## 10. Training-time pass@k analogy

Evaluation pass@k asks:

> Does repeated sampling reveal a successful mode?

GRPO group sampling asks:

> Does repeated sampling reveal a successful mode that can then be reinforced?

Thus rollout-group size acts like an exploration budget during training.

---

## 11. Positive feedback

If a rare successful strategy is sampled:

\[
\text{discovery}
\rightarrow
\text{positive advantage}
\rightarrow
P(\tau)\uparrow
\rightarrow
\text{more frequent sampling}
\rightarrow
\text{more positive updates}.
\]

This can rapidly turn a rare mode into a dominant mode.

If the mode already existed at nontrivial probability, this is sharpening.

---

## 12. Three group regimes for binary reward

### All wrong

\[
[0,\ldots,0]
\]

No relative distinction.

### Mixed

\[
[1,0,1,\ldots]
\]

Useful comparative signal.

### All correct

\[
[1,\ldots,1]
\]

No relative distinction.

Therefore GRPO learns primarily from mixed groups.

---

## 13. Probability of a mixed group

\[
\boxed{
P_{\text{mixed}}
=
1-p^G-(1-p)^G.
}
\]

So GRPO naturally has a learning frontier:
- \(p\approx0\): mostly all failures,
- intermediate \(p\): many mixed groups,
- \(p\approx1\): mostly all successes.

This suggests strongest learning on problems that are already partially solvable.

---

## 14. Difficulty is endogenous

A prompt's effective GRPO advantage changes as the policy improves.

At low success rate, a correct sample is rare and can receive large positive advantage.

At high success rate, the same reward becomes common and receives smaller relative advantage.

Thus:

\[
\boxed{
\text{reward may be fixed,
but optimization pressure is policy-dependent}.
}
\]

---

## 15. Reward and advantage are different

Two correct responses both have

\[
R=1
\]

but can receive very different advantages because their groups differ.

The verifier defines quality.

The group normalization defines optimization pressure.

---

## 16. GRPO versus PPO

| Feature | PPO | GRPO |
|---|---|---|
| Rollouts | policy | policy |
| Reward | yes | yes |
| PPO clipping | yes | yes |
| Old policy | yes | yes |
| Reference/KL | often | yes in original formulation |
| Learned critic | yes | no |
| Baseline | value model | group statistics |
| Advantage | critic/GAE based | group relative |

GRPO is best viewed as PPO-style policy optimization with critic-free group-relative advantage estimation.

---

## 17. GRPO versus DPO

DPO:

\[
(y_w,y_l)\in\mathcal D
\]

fixed preference data.

GRPO:

\[
o_1,\ldots,o_G
\]

generated by the current/old policy and evaluated on-policy.

Thus GRPO has a genuine:

\[
\boxed{
\text{sample}
\rightarrow
\text{discover}
\rightarrow
\text{verify}
\rightarrow
\text{reinforce}
}
\]

loop.

This gives it stronger exploration potential than offline DPO.

---

# Research Checkpoint — Questions and Answers

## Q1

For \(G=16\), exactly one rollout succeeds.

What are the normalized advantages?

### Answer

\[
A_+
=
\sqrt{15}
\approx3.87.
\]

Each failure:

\[
A_-
=
-\sqrt{\frac1{15}}
\approx-0.258.
\]

This strongly amplifies a rare success, which is appealing for RLVR.

The danger is high variance: one lucky, noisy, or verifier-exploiting sample can receive disproportionately large weight.

---

## Q2

Why do all-correct and all-wrong groups provide essentially no comparative signal?

### Answer

Both have

\[
s_R=0.
\]

Every rollout has identical reward, so there is no within-group ranking information.

All-correct is less concerning because the problem is already mastered.

All-wrong is concerning because the model gets no direction for how to improve.

---

## Q3

If \(p=.02\) and \(G=64\), what is the probability of at least one success?

### Answer

\[
1-(1-.02)^{64}
=
1-.98^{64}
\approx72.6\%.
\]

If GRPO later drives

\[
.02\rightarrow.50,
\]

the default interpretation is sharpening, because successful behavior was already measurably accessible.

---

## Q4

Why is \(R_i-\bar R\) not exactly equivalent to having a perfect critic?

### Answer

Statistically:
- \(\bar R\) is a noisy finite-sample estimate,
- the rollout contributes to its own baseline.

Conceptually:
- a perfect critic estimates expected return directly,
- GRPO uses peer outcomes from the same prompt group.

---

## Q5

Prompt A has \(1/8\) correct; Prompt B has \(4/8\) correct.

Why does the same reward \(R=1\) get different positive advantages?

### Answer

Prompt A:

\[
A_+=\sqrt7\approx2.65.
\]

Prompt B:

\[
A_+=1.
\]

Thus vanilla GRPO implicitly reweights prompts according to group reward statistics/difficulty.

---

## Q6

Suppose after GRPO:
- pass@1 improves strongly,
- gains concentrate on problems with nonzero SFT pass@64,
- problems with zero SFT success at \(k=64\) barely improve,
- entropy decreases.

What mechanism explains this?

### Answer

1. SFT already contains rare successful modes.
2. Group sampling finds them.
3. Rare successes receive positive relative advantage.
4. Their probability increases.
5. They are sampled more often, creating positive feedback.
6. Probability concentrates, reducing entropy.
7. Problems with no sampled success remain all-failure groups and receive little useful signal.
8. Therefore pass@1 rises strongly while high-\(k\) coverage expands little.

This is the canonical sharpening story for GRPO.

---

# Block 5 Summary

The central GRPO equations are:

\[
\boxed{
A_i
=
\frac{R_i-\bar R}{s_R}
}
\]

\[
\boxed{
\rho_{i,t}
=
\frac{\pi_\theta}{\pi_{\text{old}}}
}
\]

and a PPO-style clipped surrogate averaged across tokens and responses.

The deepest intuition is:

\[
\boxed{
\text{GRPO is a rare-mode amplifier if the rare successful mode is sampled at all}.
}
\]
