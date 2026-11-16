# Block 2 — LLM Probability, Cross-Entropy, and Supervised Fine-Tuning

## 1. What SFT is doing to the Block 1 probability landscape

For problem \(x\), the pretrained model defines

\[
\pi_{\theta_0}(\tau\mid x)
\]

over reasoning trajectories.

SFT changes parameters

\[
\theta_0\rightarrow\theta_{\text{SFT}}
\]

so that demonstrated trajectories become more likely.

The core idea:

\[
\boxed{
\text{SFT performs conditional behavior cloning.}
}
\]

It says:

> When you see contexts like these, produce tokens like these.

It does not directly optimize mathematical correctness.

---

## 2. The SFT dataset

Let

\[
\mathcal D=\{(x_i,y_i)\}_{i=1}^N.
\]

Here:
- \(x_i\): prompt/problem
- \(y_i\): desired response

A reasoning response is

\[
y_i=(y_{i,1},\ldots,y_{i,T_i}).
\]

SFT treats the demonstrated solution sequence as the target to imitate.

---

## 3. Autoregressive likelihood

The response probability factorizes as

\[
\pi_\theta(y\mid x)
=
\prod_{t=1}^{T}
\pi_\theta(y_t\mid x,y_{<t}).
\]

Thus a trajectory's probability comes from a sequence of local next-token decisions.

---

## 4. Maximum likelihood objective

SFT maximizes demonstration likelihood:

\[
\theta^*
=
\arg\max_\theta
\sum_{(x,y)\in\mathcal D}
\log\pi_\theta(y\mid x).
\]

Using autoregressive factorization:

\[
\log\pi_\theta(y\mid x)
=
\sum_t
\log\pi_\theta(y_t\mid x,y_{<t}).
\]

Therefore the loss is

\[
\boxed{
\mathcal L_{\text{SFT}}
=
-\sum_{(x,y)}
\sum_t
\log\pi_\theta(y_t\mid x,y_{<t}).
}
\]

---

## 5. Cross-entropy and one-hot targets

For logits \(z_j\),

\[
p_j=
\frac{e^{z_j}}{\sum_k e^{z_k}}.
\]

If the target token has index \(c\), one-hot cross-entropy gives

\[
L_t
=
-\log p_c.
\]

The gradient is

\[
\boxed{
\frac{\partial L}{\partial z_j}
=
p_j-\mathbf 1[j=c].
}
\]

For the target token:

\[
p_c-1<0,
\]

so gradient descent raises \(z_c\).

For competitors:

\[
p_j>0,
\]

so gradient descent lowers their logits.

Thus SFT redistributes probability mass, not merely “adds probability” to the target.

---

## 6. Token sharpening can create large trajectory changes

Suppose a four-decision reasoning path has local probabilities

\[
0.5,\quad0.4,\quad0.5,\quad0.5.
\]

Then

\[
P(\tau)=0.05.
\]

After SFT these become

\[
0.8,\quad0.8,\quad0.9,\quad0.9,
\]

so

\[
P(\tau)=0.5184.
\]

Small local changes can therefore create huge sequence-level accessibility changes.

---

## 7. Local supervision versus sequence-level behavior

Training optimizes

\[
\sum_t-\log\pi(y_t\mid x,y_{<t}),
\]

but evaluation often uses

\[
V(x,\tau)
\]

on the whole trajectory.

This creates a fundamental mismatch:

\[
\boxed{
\text{token-level likelihood}
\neq
\text{sequence-level correctness}.
}
\]

---

## 8. Teacher forcing

During SFT, the model predicts token \(y_t\) conditioned on the ground-truth prefix

\[
x,y_1,\ldots,y_{t-1}.
\]

During inference, it conditions on its own generated history.

Training:

```text
correct prefix
   ↓
correct prefix
   ↓
predict next token
```

Inference:

```text
model output
   ↓
model output
   ↓
model output
```

So SFT is trained mainly on expert prefixes but deployed on model-generated prefixes.

---

## 9. Distribution shift / exposure bias

Let

\[
d_{\text{data}}(s)
\]

be the distribution of prefixes in demonstrations.

At deployment the model visits states according to

\[
d_{\pi_\theta}(s).
\]

There is no guarantee that

\[
d_{\text{data}}=d_{\pi_\theta}.
\]

An early mistake can send the model into states it rarely saw during SFT.

---

## 10. Population-level view: forward KL

Suppose the data distribution is

\[
q(y\mid x).
\]

Then the population SFT objective is

\[
\mathcal L
=
\mathbb E_{x,y\sim q}
[-\log\pi_\theta(y\mid x)].
\]

For fixed \(x\), this is cross-entropy

\[
H(q,\pi_\theta)
=
H(q)+D_{\mathrm{KL}}(q\|\pi_\theta).
\]

Since \(H(q)\) is constant w.r.t. \(\theta\),

\[
\boxed{
\arg\min_\theta\mathcal L
=
\arg\min_\theta
D_{\mathrm{KL}}(q_{\text{data}}\|\pi_\theta).
}
\]

This gives MLE a mode-covering tendency relative to modes represented in the data.

It does not mean SFT covers all valid reasoning modes.

---

## 11. SFT can reduce reasoning diversity

Suppose before SFT:

\[
P(A)=0.20,\qquad P(B)=0.20,
\]

and both are valid reasoning strategies.

If demonstrations always use A, post-SFT might become

\[
P(A)=0.70,\qquad P(B)=0.05.
\]

Accuracy may improve while semantic diversity decreases.

\[
\boxed{
\text{better SFT performance}
\not\Rightarrow
\text{richer reasoning distribution}.
}
\]

---

## 12. SFT is not mere memorization

Parameter sharing means updates on one example can generalize to unseen inputs.

Three possibilities:
1. memorization,
2. sharpening of an existing behavior,
3. operational expansion to new held-out behavior.

All can produce higher accuracy.

---

## 13. Lower SFT loss does not imply better reasoning

A model can improve average NLL by getting many easy tokens slightly better while making the crucial reasoning decision worse.

Therefore:

\[
\boxed{
\text{lower validation cross-entropy}
\not\Leftrightarrow
\text{higher reasoning accuracy}.
}
\]

---

## 14. Sequence length effects

Raw sequence probability decreases exponentially with length.

If

\[
p_t=0.9,
\]

then

\[
0.9^{10}\approx0.349
\]

but

\[
0.9^{100}\approx2.66\times10^{-5}.
\]

This motivates average NLL

\[
\bar L
=
-\frac1T\sum_t\log p_t
\]

and perplexity

\[
\operatorname{PPL}=e^{\bar L}.
\]

But these still measure predictive fit, not correctness.

---

## 15. Trajectory probability versus answer probability

If many distinct successful trajectories yield the same answer,

\[
P(\text{correct answer})
=
\sum_{\tau:\operatorname{answer}(\tau)=\text{correct}}
P(\tau).
\]

SFT typically trains on specific demonstrated trajectories, while evaluation cares about the full successful region.

---

## 16. SFT objective versus success objective

SFT optimizes

\[
\boxed{
-\log\pi_\theta(\tau_{\text{demo}}\mid x)
}
\]

while an ideal correctness objective is

\[
\boxed{
J(\theta)
=
\mathbb E_{\tau\sim\pi_\theta}[R(x,\tau)]
=
P_\theta(\text{success}\mid x)
}
\]

for binary rewards.

So:

> SFT says “imitate this solution.”

A correctness objective says:

> Put probability on any successful solution.

That distinction motivates RL-based post-training.

---

# Research Checkpoint — Questions and Answers

## Q1

Base:

\[
P(A)=0.20,\quad P(B)=0.10,\quad P(C)=0.05.
\]

After SFT:

\[
P(A)=0.70,\quad P(B)=0.03,\quad P(C)=0.01.
\]

What suggests sharpening, and what undesirable property changed?

### Answer

Strategy A was already accessible at \(0.20\), so the dominant change is strong sharpening:

\[
0.20\rightarrow0.70.
\]

The undesirable effect is reduced strategy diversity.

Total correct mass rises from \(0.35\) to \(0.74\), but conditional correct-mode probability becomes highly concentrated on A.

---

## Q2

Why does

\[
\frac{\partial L}{\partial z_j}
=
p_j-\mathbf1[j=c]
\]

mean SFT is not simply “adding probability” to the target?

### Answer

For the target token, the logit is increased; for every competitor, the logit is decreased.

Because softmax probabilities must sum to one, increasing one token necessarily redistributes mass away from others.

This can suppress even valid alternative continuations.

---

## Q3

How can a model have lower SFT validation loss but lower math pass@1?

### Answer

At least three reasons:

1. Most tokens are not decisive reasoning steps; easy-token improvements can dominate average loss.
2. SFT optimizes demonstration likelihood, not correctness.
3. Teacher-forced validation uses gold prefixes, while generation uses model-generated prefixes.
4. Small changes at critical decisions can have large multiplicative sequence-level effects.
5. Frequent easy patterns can dominate loss while rare reasoning behavior gets worse.

---

## Q4

A reasoning trajectory goes from probability \(10^{-8}\) before SFT to \(10^{-2}\) after SFT.

Did the model learn a new reasoning algorithm?

### Answer

Not necessarily.

The trajectory already had nonzero probability. SFT may simply have amplified an extremely rare mode by a factor of \(10^6\).

This is dramatic operational expansion in accessibility, but not proof that a new internal algorithm was created.

---

## Q5

What is the difference between optimizing

\[
-\log\pi_\theta(\tau^*|x)
\]

and

\[
E_{\tau\sim\pi_\theta}[R(x,\tau)]?
\]

### Answer

The first says:

> Make this demonstrated trajectory likely.

The second says:

> Increase probability of any sampled trajectory that receives reward.

Thus verifier-based RL can directly reinforce valid trajectories that were never present in the demonstration dataset, provided the policy samples them.

The catch is that RL cannot reward a successful behavior it never samples.

---

# Block 2 Summary

SFT is best viewed as:

\[
\boxed{
\text{demonstration}
\rightarrow
\text{teacher-forced token prediction}
\rightarrow
\text{cross-entropy gradient}
\rightarrow
\text{reshaped trajectory distribution}.
}
\]

The central scientific distinction is:

\[
\boxed{
-\log\pi(\tau_{\text{demo}})
\neq
-\log\sum_{\tau\in\mathcal S}\pi(\tau)
}
\]

or, equivalently:

\[
\boxed{
\text{imitate one demonstrated solution}
\neq
\text{maximize total successful probability mass}.
}
\]
