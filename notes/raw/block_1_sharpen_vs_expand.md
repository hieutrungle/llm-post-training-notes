# Block 1 — What Does It Mean for Post-Training to “Sharpen” vs “Expand” Capability?

## 1. The core question

For a problem \(x\), an LLM defines a probability distribution over complete reasoning trajectories

\[
\tau=(y_1,\ldots,y_T).
\]

The autoregressive policy is

\[
\pi_\theta(\tau\mid x)
=
\prod_{t=1}^{T}
\pi_\theta(y_t\mid x,y_{<t}).
\]

Let

\[
\mathcal T(x)=\mathcal S(x)\cup \mathcal F(x)
\]

where \(\mathcal S(x)\) is the set of successful trajectories and \(\mathcal F(x)\) is the set of failed trajectories.

The model's one-sample success probability is

\[
p_\theta(x)
=
P_{\tau\sim\pi_\theta}[\tau\in\mathcal S(x)]
=
\sum_{\tau\in\mathcal S(x)}
\pi_\theta(\tau\mid x).
\]

A more complete notation is

\[
p_\theta(x;c)
\]

where \(c\) denotes inference conditions such as prompt, temperature, top-p, max tokens, tools, and sampling procedure.

---

## 2. Why “nonzero support” is not a useful definition of capability

One could define capability as

\[
\exists \tau\in\mathcal S(x): \pi_\theta(\tau\mid x)>0.
\]

For a softmax language model, however, essentially every finite token sequence can have nonzero probability.

A successful trajectory with probability

\[
10^{-100}
\]

is technically in the support, but practically inaccessible.

This motivates the distinction between:

### Mathematical support

\[
\mathcal A_{\text{support}}
=
\{\tau:\pi(\tau)>0\}
\]

and an operational notion of accessibility:

\[
\mathcal A_\epsilon
=
\{\tau:\pi(\tau)\ge\epsilon\}.
\]

A better operational view is therefore:

> How likely is the model to produce a successful trajectory under a fixed inference budget?

---

## 3. pass@1

For one fixed problem with single-sample success probability \(p\),

\[
\boxed{\operatorname{pass@1}=p.}
\]

But the same value of pass@1 may arise from very different internal distributions.

For example, a model may:
- have one rare correct mode,
- have many distinct correct modes,
- have a mostly-correct trajectory with one recurring arithmetic mistake,
- or have no stable reasoning strategy at all.

So pass@1 alone does not tell us what kind of capability is present.

---

## 4. Derivation of pass@k

Suppose we take \(k\) independent samples.

The probability that a single sample fails is

\[
1-p.
\]

The probability all \(k\) samples fail is

\[
(1-p)^k.
\]

Therefore,

\[
\boxed{
\operatorname{pass@k}
=
1-(1-p)^k.
}
\]

Example: if \(p=0.1\),

\[
\operatorname{pass@8}
=
1-0.9^8
\approx0.570,
\]

and

\[
\operatorname{pass@32}
=
1-0.9^{32}
\approx0.966.
\]

### Mental visualization

```text
Low-probability success becomes visible as k increases

p = 0.01   pass@1  ≈ 1%    pass@64 ≈ 47%
p = 0.05   pass@1  ≈ 5%    pass@64 ≈ 96%
p = 0.10   pass@1  ≈10%    pass@64 ≈ ~100%
```

High-\(k\) sampling acts like a microscope for low-probability successful behavior.

---

## 5. Same true pass@1 for one problem implies the same pass@k

If two models have the same per-problem success probability

\[
p_A=p_B=0.4,
\]

then under i.i.d. sampling they also have the same

\[
\operatorname{pass@k}
=
1-(1-0.4)^k.
\]

This remains true even if one model has one correct strategy and the other has ten.

Ordinary pass@k sees total successful probability mass, not the internal semantic structure of the successful region.

---

## 6. Same benchmark-average pass@1 can hide different breadth

Consider two problems.

### Model A

\[
p_1=0.4,\qquad p_2=0.4.
\]

Average pass@1:

\[
0.4.
\]

### Model B

\[
p_1=0.8,\qquad p_2=0.
\]

Average pass@1:

\[
0.4.
\]

For \(k=2\),

\[
f_k(p)=1-(1-p)^k.
\]

Model A has average pass@2 of

\[
0.64,
\]

while Model B has

\[
\frac{0.96+0}{2}=0.48.
\]

Because \(f_k(p)\) is concave for \(k>1\),

\[
f_k''(p)<0,
\]

a more even spread of capability across problems can yield higher benchmark-level pass@k than concentrating success on a subset.

---

## 7. Sharpening

Suppose the SFT model has:

| Trajectory | Type | Probability |
|---|---|---:|
| \(\tau_{C1}\) | correct strategy A | 0.08 |
| \(\tau_{C2}\) | correct strategy B | 0.02 |
| failures | incorrect | 0.90 |

Then

\[
p_{\text{SFT}}=0.10.
\]

After post-training:

| Trajectory | Type | Probability |
|---|---|---:|
| \(\tau_{C1}\) | correct A | 0.45 |
| \(\tau_{C2}\) | correct B | 0.05 |
| failures | incorrect | 0.50 |

Then

\[
p_{\text{RL}}=0.50.
\]

This is a canonical sharpening story:

\[
\boxed{
\text{already-accessible successful trajectory}
\rightarrow
\text{much more probability mass}.
}
\]

---

## 8. Operational expansion

Literal support expansion is not the right notion for softmax LMs.

A more useful definition is:

> Post-training causes previously effectively inaccessible successful behavior to become reliably accessible under a fixed practical inference budget.

Two useful forms:

### Problem-level expansion

\[
p_{\text{before}}(x)\approx0
\]

becomes

\[
p_{\text{after}}(x)\gg0.
\]

### Strategy-level expansion

A semantic strategy family \(z=C\) is essentially unseen before training but becomes substantial afterward.

Important:

\[
\boxed{\text{newly observed}\neq\text{proven newly created}.}
\]

---

## 9. Why high pass@k still does not reveal an intrinsic capability boundary

Suppose

\[
p=0.001.
\]

Then

\[
\operatorname{pass@64}
=
1-(0.999)^{64}
\approx6.2\%.
\]

But

\[
\operatorname{pass@1000}
\approx63.2\%.
\]

So whether a problem looks “solvable” depends on inference budget.

There may be no sharp intrinsic boundary, only a probability landscape.

---

## 10. Zero successes does not imply zero capability

If we observe \(0\) successes in \(n\) trials,

\[
P(0\text{ successes}\mid p)
=
(1-p)^n.
\]

For \(n=64\), the one-sided 95% upper bound solves

\[
(1-p)^{64}=0.05,
\]

giving approximately

\[
p\approx4.6\%.
\]

Thus \(0/64\) is still compatible with a several-percent underlying success rate.

Even \(0/1024\) only constrains success probability to roughly below \(0.29\%\) at this one-sided level.

Therefore:

\[
\boxed{
\text{finite sampling almost never proves literal absence of capability}.
}
\]

---

## 11. Finite-sample estimator of pass@k

If \(n\) responses are sampled and \(c\) are correct, the standard estimator is

\[
\boxed{
\widehat{\operatorname{pass@k}}
=
1-
\frac{\binom{n-c}{k}}{\binom{n}{k}}
}
\]

for \(n\ge k\).

Reason:

- total subsets of size \(k\): \(\binom nk\)
- all-failure subsets: \(\binom{n-c}{k}\)

So one minus the all-failure fraction estimates pass@k.

---

## 12. Reasoning diversity

Suppose both models have total success probability

\[
p=0.4.
\]

### Model A

One correct strategy with probability \(0.4\).

### Model B

Four distinct correct strategies, each with probability \(0.1\).

Both have the same pass@1 and pass@k.

To study internal diversity, define the conditional distribution over successful trajectories:

\[
q(\tau\mid x,\text{success})
=
\frac{\pi(\tau\mid x)}{p(x)}.
\]

Then define successful-trajectory entropy

\[
H_{\text{correct}}
=
-\sum_{\tau\in\mathcal S}
q(\tau)\log q(\tau).
\]

A useful derived quantity is

\[
\boxed{
N_{\text{eff}}
=
e^{H_{\text{correct}}}.
}
\]

But semantic strategy diversity is more important than superficial token diversity.

---

## 13. Evidence hierarchy for sharpening vs expansion

Weakest to strongest:

1. pass@1 increases.
2. pass@k increases.
3. high-\(k\) problem coverage increases.
4. previously high-\(k\)-unsolved problems become repeatedly solvable.
5. new coherent strategy families appear.
6. effects replicate across seeds.
7. behavior generalizes to held-out perturbations/problem families.
8. high-volume pretraining/SFT sampling places a tight upper bound on previous accessibility.

Best scientific wording:

> “The results are consistent with operational capability expansion.”

Avoid:

> “RL proved that a completely new capability was created.”

---

# Research Checkpoint — Questions and Answers

## Q1

Suppose

\[
p_{\text{SFT}}=0.01
\]

and

\[
p_{\text{RL}}=0.30.
\]

Have we demonstrated sharpening, expansion, or neither?

### Answer

From these numbers alone, we have demonstrated improved accessibility, not expansion.

Because SFT already has measurable successful probability, the default explanation is sharpening:

\[
0.01\rightarrow0.30.
\]

To argue for expansion we would want evidence such as:
- previously unsolved problems becoming repeatedly solvable,
- new semantic strategies,
- high-volume SFT sampling failing to uncover them,
- replication across seeds,
- held-out generalization.

---

## Q2

Two models have average pass@1 \(=40\%\).

Model A has \(p_i=0.4\) on every problem.

Model B has \(p_i=0.8\) on half the problems and \(p_i=0\) on the other half.

Which has higher pass@16?

### Answer

Model A.

For Model A,

\[
1-0.6^{16}\approx99.97\%.
\]

For Model B, half the problems remain impossible under the assumed distribution, so average pass@16 is about 50%.

Model A has broader coverage.

---

## Q3

Model A has one correct strategy carrying probability \(0.3\).

Model B has ten correct strategies, each carrying probability \(0.03\).

Which has larger pass@1 and pass@64?

### Answer

They are identical on both because total successful mass is \(0.3\).

But Model B has richer successful strategy diversity.

Pass@k cannot capture the semantic structure of the successful region.

---

## Q4

SFT gives \(0/64\) correct answers. After DAPO, \(10/64\) are correct.

Why is “DAPO learned a novel capability that did not exist before” unjustified?

### Answer

Because \(0/64\) does not imply \(p_{\text{SFT}}=0\).

A nontrivial success probability can still produce no successes in 64 samples.

A defensible statement is:

> DAPO made previously rarely or unobservably accessible successful behavior reliably accessible.

With stronger evidence one may say the result is consistent with operational capability expansion.

---

## Q5

GRPO improves pass@1 but not pass@64. RS-GRPO improves both.

What hypothesis would you form?

### Answer

GRPO likely concentrates probability on already-accessible successful modes: sharpening.

RS-GRPO may preserve or create enough exploration to broaden problem-level accessibility, producing stronger expansion-like evidence.

This remains a mechanistic hypothesis until supported by transition analysis, matched budgets, multiple seeds, and diversity/generalization evidence.

---

# Block 1 Summary

The core mental model is:

\[
\boxed{
\text{Model}
=
\text{probability landscape over problems and reasoning trajectories}.
}
\]

Sharpening:

\[
\text{existing successful peaks get taller}.
\]

Operational expansion:

\[
\text{previously negligible successful regions become meaningfully accessible}.
\]

The key experimental challenge is distinguishing “did not exist” from “existed at extremely low probability.”
