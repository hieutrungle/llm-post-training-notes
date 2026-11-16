# Block 3 — Preference Optimization and DPO

## 1. Why preference optimization?

SFT learns from targets:

\[
(x,y^*).
\]

Preference optimization instead learns from pairs:

\[
(x,y_w,y_l)
\]

where

\[
y_w\succ y_l.
\]

This is useful when we can say one response is better than another without assigning meaningful scalar scores.

---

## 2. Bradley–Terry preference model

Introduce a latent reward

\[
r(x,y).
\]

The Bradley–Terry model assumes

\[
\boxed{
P(y_w\succ y_l\mid x)
=
\sigma(r_w-r_l)
}
\]

where

\[
\sigma(z)=\frac{1}{1+e^{-z}}.
\]

Only reward differences matter.

---

## 3. KL-regularized RL starting point

Classical RLHF can be written as

\[
\boxed{
\max_\pi
E_{y\sim\pi}[r(x,y)]
-
\beta
D_{\mathrm{KL}}
(\pi(\cdot|x)\|\pi_{\mathrm{ref}}(\cdot|x)).
}
\]

The reward term encourages high-reward responses.

The KL term anchors the policy to a reference, often the SFT model.

---

## 4. Deriving the optimal KL-regularized policy

For one prompt \(x\), optimize

\[
\sum_y\pi(y)r(y)
-
\beta
\sum_y
\pi(y)
\log\frac{\pi(y)}{\pi_{\mathrm{ref}}(y)}
\]

subject to

\[
\sum_y\pi(y)=1.
\]

Using a Lagrange multiplier and solving gives

\[
\boxed{
\pi^*(y|x)
=
\frac{
\pi_{\mathrm{ref}}(y|x)e^{r(x,y)/\beta}
}{
Z(x)
}.
}
\]

Thus the optimal policy is:

\[
\boxed{
\text{reference probability}
\times
\text{reward-dependent reweighting}.
}
\]

For two responses,

\[
\frac{\pi^*(A)}{\pi^*(B)}
=
\frac{\pi_{\mathrm{ref}}(A)}
{\pi_{\mathrm{ref}}(B)}
e^{(r_A-r_B)/\beta}.
\]

---

## 5. Interpretation of \(\beta\)

Large \(\beta\):

\[
e^{r/\beta}\approx1
\]

so

\[
\pi^*\approx\pi_{\mathrm{ref}}.
\]

Small \(\beta\):

reward differences are magnified and the policy can move more aggressively.

Thus:

\[
\boxed{
\beta\uparrow \Rightarrow \text{stronger reference anchoring}
}
\]

\[
\boxed{
\beta\downarrow \Rightarrow \text{more aggressive reward-driven reshaping}.
}
\]

---

## 6. Reward as a log probability ratio

From

\[
\pi^*(y|x)
=
\frac{
\pi_{\mathrm{ref}}(y|x)e^{r(x,y)/\beta}
}{
Z(x)
},
\]

we obtain

\[
\boxed{
r(x,y)
=
\beta
\log
\frac{\pi^*(y|x)}
{\pi_{\mathrm{ref}}(y|x)}
+
\beta\log Z(x).
}
\]

This is the key DPO reparameterization.

---

## 7. Plug into Bradley–Terry

Because Bradley–Terry uses \(r_w-r_l\), the \(Z(x)\) term cancels:

\[
r_w-r_l
=
\beta
\left[
\log\frac{\pi^*(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)}
-
\log\frac{\pi^*(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}
\right].
\]

Replace \(\pi^*\) by trainable \(\pi_\theta\).

Define

\[
\Delta_\theta
=
\log\frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)}
-
\log\frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}.
\]

Then

\[
\boxed{
\mathcal L_{\mathrm{DPO}}
=
-
E
\left[
\log\sigma(\beta\Delta_\theta)
\right].
}
\]

---

## 8. Ratio-of-ratios interpretation

Equivalent form:

\[
\Delta_\theta
=
\log
\left[
\frac{
\pi_\theta(y_w)/\pi_{\mathrm{ref}}(y_w)
}{
\pi_\theta(y_l)/\pi_{\mathrm{ref}}(y_l)
}
\right].
\]

So DPO asks:

> Relative to the reference policy, did the chosen response improve more than the rejected response?

It does not simply require

\[
\pi_\theta(y_w)>\pi_\theta(y_l).
\]

---

## 9. Example

Suppose:

\[
\pi_{\mathrm{ref}}(y_w)=0.01,
\qquad
\pi_{\mathrm{ref}}(y_l)=0.20.
\]

After training:

\[
\pi_\theta(y_w)=0.02,
\qquad
\pi_\theta(y_l)=0.10.
\]

The loser is still more probable in absolute terms, but:

\[
\frac{0.02}{0.01}=2
\]

and

\[
\frac{0.10}{0.20}=0.5.
\]

So:

\[
\Delta
=
\log2-\log0.5
=
\log4>0.
\]

DPO views this as movement in the correct direction.

---

## 10. DPO gradient intuition

Let

\[
L=-\log\sigma(\beta\Delta).
\]

Then

\[
\boxed{
\frac{\partial L}{\partial\Delta}
=
-\beta\sigma(-\beta\Delta).
}
\]

Large negative \(\Delta\): strong gradient.

\(\Delta\approx0\): substantial gradient.

Large positive \(\Delta\): small gradient.

DPO focuses learning pressure on preference pairs that are not already well separated.

---

## 11. Sequence-level preference, token-level gradients

Because

\[
\log\pi_\theta(y|x)
=
\sum_t
\log\pi_\theta(y_t|x,y_{<t}),
\]

the sequence preference label affects token probabilities throughout the response.

This creates a credit-assignment issue:

> DPO knows the whole response is preferred, not which specific token or reasoning step caused the preference.

---

## 12. SFT versus DPO

### SFT

\[
-\log\pi_\theta(y_w|x).
\]

Interpretation:

> Make this sequence likely.

### DPO

\[
-\log\sigma(\beta\Delta_\theta).
\]

Interpretation:

> Shift the policy so the preferred response improves relative to the rejected response, measured against the reference.

---

## 13. Why DPO naturally leans toward sharpening

Offline DPO trains on a fixed preference dataset.

If a brilliant strategy \(Z\) never appears in the dataset, DPO receives no direct label saying it is good.

So DPO lacks an explicit:

\[
\text{sample}\rightarrow\text{discover}\rightarrow\text{reinforce}
\]

loop.

DPO can still generalize to unseen problems because parameters are shared.

Therefore a reasonable research hypothesis is:

\[
\boxed{
\text{offline DPO should often lean toward probability redistribution / sharpening}.
}
\]

This is a hypothesis, not a theorem.

---

# Research Checkpoint — Questions and Answers

## Q1

Why can DPO regard

\[
\pi_{\rm ref}(y_w)=0.01,\quad
\pi_{\rm ref}(y_l)=0.20
\]

and

\[
\pi_\theta(y_w)=0.02,\quad
\pi_\theta(y_l)=0.10
\]

as a successful preference shift?

### Answer

The winner doubled relative to reference while the loser halved.

Winner-to-loser odds changed from

\[
0.01/0.20=0.05
\]

to

\[
0.02/0.10=0.20.
\]

The odds improved by \(4\times\).

DPO cares about relative improvement, not absolute ordering alone.

---

## Q2

Why does KL-regularized RL produce

\[
\pi^*(y)\propto\pi_{\rm ref}(y)e^{r(y)/\beta}?
\]

What happens as \(\beta\to\infty\) and \(\beta\to0\)?

### Answer

The solution balances reward maximization and staying near the reference.

The reference supplies the starting probability; the exponential reward term tilts it.

As

\[
\beta\to\infty,
\]

the reward tilt disappears and

\[
\pi^*\to\pi_{\rm ref}.
\]

As

\[
\beta\to0^+,
\]

reward differences dominate and probability concentrates increasingly on maximum-reward behavior.

---

## Q3

DPO improves pass@1 from 30% to 50%, but pass@64 barely changes.

What likely happened?

### Answer

This is a classic sharpening signature.

DPO probably increased success probability on problems that were already accessible at high \(k\), turning rare/moderate successes into common successes without broadening the set of solvable problems much.

---

## Q4

Why is fixed offline DPO weaker for exploration than on-policy RLVR?

### Answer

DPO directly learns only from trajectories in its fixed preference dataset.

On-policy RLVR repeatedly generates new trajectories, evaluates them, and can reinforce a rare newly sampled success.

So RLVR has an explicit exploration-discovery-reinforcement loop that offline DPO lacks.

---

## Q5

Why does DPO have a credit-assignment problem when two 500-token responses differ in one crucial mathematical step?

### Answer

The preference label is sequence-level:

\[
y_w\succ y_l.
\]

But the log-probability decomposes across all 500 tokens.

DPO does not explicitly know that one specific token caused the preference. Shared gradients can affect many parts of the response, including incidental stylistic differences.

---

# Block 3 Summary

The DPO core is:

\[
\boxed{
\Delta_\theta
=
\log\frac{\pi_\theta(y_w)}{\pi_{\rm ref}(y_w)}
-
\log\frac{\pi_\theta(y_l)}{\pi_{\rm ref}(y_l)}
}
\]

and

\[
\boxed{
L_{\rm DPO}
=
-\log\sigma(\beta\Delta_\theta).
}
\]

The conceptual progression is:

```text
SFT
“imitate this”

DPO
“prefer this over that”

RLVR
“sample → test → reinforce”
```

DPO is best viewed as relative probability redistribution around a reference policy.
