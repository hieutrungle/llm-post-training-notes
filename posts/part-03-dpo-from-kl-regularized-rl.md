---
title: "DPO from KL-Regularized RL"
kind: Study notes
date: 2026-11-17 12:00:00 -0600
series: llm-post-training
part: 3
summary: "Direct preference optimization learns from pairs of responses, one preferred over the other. Part 3 derives it from KL-regularized reward maximization with a frozen reference policy, shows that its optimum only reweights the reference (with a 0/1 reward: sharpening), explains what β controls, and shows why training can still lower the probability of the preferred response."
math: true
anim: true
---

*Code, check scripts and references: [llm-post-training-notes](https://github.com/hieutrungle/llm-post-training-notes) · Experiments: [sharpen-or-expand](https://github.com/hieutrungle/sharpen-or-expand)*

<div class="goal" markdown="1">
<span class="label">Core idea</span>

DPO is KL-regularized reward maximization solved in closed form and fitted directly to comparisons. The optimum of $$\max_\pi \mathbb{E}_\pi[r] - \beta D_{\mathrm{KL}}(\pi \Vert \pi_{\text{ref}})$$ is the reference reweighted by $$e^{r/\beta}$$. With a 0/1 correctness reward, that multiplies each problem's odds of success by $$e^{1/\beta}$$, leaves the mix of correct solutions untouched, and adds nothing the reference cannot already produce: sharpening, in [Part 1](/blog/2026/11/capability-and-pass-at-k/)'s sense. The DPO loss itself constrains only a ratio of ratios, so training can lower the probability of the very response it prefers.
</div>

## The question

[Part 2](/blog/2026/11/sft-probability-reshaping/) showed what imitation does: supervised fine-tuning raises one demonstrated response and pays for it with every alternative. Often there is no demonstration to imitate, only a judgment that one response is better than another. This part asks how to learn from "this response is better than that one", and what that kind of signal does to the model's distribution over responses.

## Learning from comparisons

Preference data consist of triples $$(x, y_w, y_l)$$: a prompt, a preferred response $$y_w$$ (the "winner") and a dispreferred one $$y_l$$ (the "loser"), with $$y_w \succ y_l$$. Comparisons are often easier to obtain than demonstrations or scores. A person who cannot write the ideal answer, or put a meaningful number on one, can usually say which of two answers is better. For reasoning tasks a verifier gives comparisons for free: of two sampled solutions, the correct one is preferred to the incorrect one.

The signal is different from Part 2's in two ways. There is no target to copy: the data say which of two responses is better, not what the best response is. And the loser is part of the data. SFT lowers alternatives only as a side effect of normalization; a preference method pushes a specific bad response down on purpose.

## The Bradley–Terry model

To turn comparisons into something we can optimize, assume each response has a latent reward $$r(x, y)$$, and that the probability of preferring one response to another depends on the reward difference through the logistic function $$\sigma(z) = 1/(1 + e^{-z})$$:

$$
P(y_w \succ y_l \mid x) = \sigma\big(r(x, y_w) - r(x, y_l)\big).
$$

This is the Bradley–Terry model of paired comparisons [1]. A reward gap of 1 gives the better response a 73% chance of being preferred, and a gap of 2 gives 88%. Only differences matter: adding the same constant to every reward for a prompt changes nothing, so rewards of 11 and 10 give the same 73% as rewards of 1 and 0. The model can identify rewards from comparisons only up to such a per-prompt shift, and that fact is what makes DPO possible below.

Learning a reward model from comparisons this way is how reinforcement learning from human feedback (RLHF) starts. Christiano et al. [2] fitted a reward to pairwise human judgments with exactly this model, by maximum likelihood, and trained a policy on it with RL.

## The KL-regularized objective and its optimum

The second step of RLHF fine-tunes the policy to collect reward without drifting far from where it started. Ziegler et al. [3] and Ouyang et al. [4] penalize the KL divergence from a frozen reference policy $$\pi_{\text{ref}}$$, usually the model before this stage. For each prompt,

$$
\max_{\pi}\ \mathbb{E}_{y \sim \pi(\cdot \mid x)}\big[r(x, y)\big] - \beta\, D_{\mathrm{KL}}\big(\pi(\cdot \mid x) \,\Vert\, \pi_{\text{ref}}(\cdot \mid x)\big).
$$

The reward term pulls toward high-reward responses; the KL term charges for every response whose probability moves away from the reference. $$\beta \gt 0$$ sets the exchange rate between the two.

This objective has a closed-form maximizer. Fix the prompt and drop $$x$$ from the notation. Writing the KL term as a sum and collecting terms,

$$
\sum_y \pi(y)\Big[r(y) - \beta \log\frac{\pi(y)}{\pi_{\text{ref}}(y)}\Big]
= -\beta \sum_y \pi(y) \log\frac{\pi(y)}{\pi_{\text{ref}}(y)\, e^{r(y)/\beta}}.
$$

The denominator is almost a distribution. Normalize it with

$$
Z = \sum_y \pi_{\text{ref}}(y)\, e^{r(y)/\beta}, \qquad
\pi^*(y) = \frac{\pi_{\text{ref}}(y)\, e^{r(y)/\beta}}{Z},
$$

so that $$\pi_{\text{ref}}(y)\, e^{r(y)/\beta} = Z\, \pi^*(y)$$. The objective becomes

$$
\mathbb{E}_\pi[r] - \beta D_{\mathrm{KL}}(\pi \Vert \pi_{\text{ref}}) = \beta \log Z - \beta\, D_{\mathrm{KL}}(\pi \Vert \pi^*).
$$

$$Z$$ does not depend on $$\pi$$, and a KL divergence is zero only when the two distributions are equal and positive otherwise. So the unique maximizer is $$\pi^*$$, and the maximum value is $$\beta \log Z$$. This is the Gibbs (or Boltzmann) distribution of statistical physics, with $$\beta$$ playing the temperature.

The same answer comes out of a Lagrangian. Subtract $$\lambda(\sum_y \pi(y) - 1)$$ for normalization and set the derivative with respect to $$\pi(y)$$ to zero: $$r(y) - \beta\big(\log(\pi(y)/\pi_{\text{ref}}(y)) + 1\big) = \lambda$$, so $$\pi(y) \propto \pi_{\text{ref}}(y)\, e^{r(y)/\beta}$$. The objective is strictly concave in $$\pi$$, so this stationary point is the maximum. The [checks script](https://github.com/hieutrungle/llm-post-training-notes/blob/main/checks/part03_checks.py) confirms both on 300 random discrete problems: none of 12,000 alternative distributions, random or perturbed around $$\pi^*$$, scores higher.

The result is the most important formula in this part:

$$
\pi^*(y \mid x) = \frac{\pi_{\text{ref}}(y \mid x)\, e^{r(x, y)/\beta}}{Z(x)}.
$$

The optimal policy is the reference with every response reweighted by a factor that grows exponentially in its reward. It changes how much probability responses get. It cannot change which responses get any: if $$\pi_{\text{ref}}(y) = 0$$, then $$\pi^*(y) = 0$$, whatever the reward.

## What β controls

Comparing two responses $$A$$ and $$B$$ removes $$Z$$:

$$
\frac{\pi^*(A)}{\pi^*(B)} = \frac{\pi_{\text{ref}}(A)}{\pi_{\text{ref}}(B)}\, e^{(r_A - r_B)/\beta}.
$$

The reference ratio is the starting point and the reward gap, divided by $$\beta$$, tilts it. Take $$\pi_{\text{ref}}(A) = 0.1$$, $$\pi_{\text{ref}}(B) = 0.3$$ and $$r_A - r_B = 1$$. The ratio $$\pi^*(A)/\pi^*(B)$$ is 0.55 at $$\beta = 2$$, 0.91 at $$\beta = 1$$, 2.46 at $$\beta = 0.5$$ and 18.2 at $$\beta = 0.25$$. The better response overtakes the more familiar one once $$\beta \lt 1/\ln 3 \approx 0.91$$.

The two limits make the role of $$\beta$$ precise.

- **$$\beta \to \infty$$.** Every factor $$e^{r/\beta} \to 1$$ and $$\pi^* \to \pi_{\text{ref}}$$. The policy does not move.
- **$$\beta \to 0^+$$.** Divide numerator and denominator by $$e^{r_{\max}/\beta}$$, where $$r_{\max}$$ is the largest reward among responses that $$\pi_{\text{ref}}$$ can produce. Every response with a lower reward gets a factor $$e^{-(r_{\max} - r)/\beta} \to 0$$. The mass concentrates on the maximum-reward responses *that the reference supports*, and among them it is split in proportion to their reference probabilities.

Small $$\beta$$ means aggressive reward-driven reshaping; large $$\beta$$ means a strong anchor to the reference. Rafailov et al. [5] use $$\beta = 0.1$$ in most of their experiments. Note also that $$r$$ and $$\beta$$ enter only as $$r/\beta$$: doubling every reward has the same effect as halving $$\beta$$.

## A 0/1 reward: the optimum only sharpens

The connection to Part 1 is cleanest with a verifiable reward. Let $$R(x, \tau) = 1$$ if trajectory $$\tau$$ is correct and 0 otherwise, so that the success probability under the reference is $$p = \sum_{\tau \in \mathcal{S}(x)} \pi_{\text{ref}}(\tau \mid x)$$, with $$\mathcal{S}(x)$$ the set of correct trajectories. The optimum multiplies every correct trajectory by $$e^{1/\beta}$$ and every incorrect one by 1, so $$Z = p\, e^{1/\beta} + 1 - p$$ and the success probability at the optimum is

$$
p^* = \frac{p\, e^{1/\beta}}{p\, e^{1/\beta} + 1 - p},
\qquad\text{equivalently}\qquad
\frac{p^*}{1 - p^*} = e^{1/\beta}\, \frac{p}{1 - p}.
$$

The KL-regularized optimum multiplies the odds of success by $$e^{1/\beta}$$. At $$\beta = 1$$ the factor is 2.72: a problem with $$p = 0.30$$ (odds 0.43) moves to $$p^* = 0.538$$ (odds 1.16).

Three more facts follow directly, and the checks script confirms each on random problems.

1. **The mix of correct solutions does not change.** Within $$\mathcal{S}(x)$$, every trajectory is multiplied by the same factor, so Part 1's distribution of successful trajectories is unchanged: $$q^*(\tau \mid x) = \pi^*(\tau \mid x)/p^* = \pi_{\text{ref}}(\tau \mid x)/p = q(\tau \mid x)$$. The entropy of correct solutions, and $$N_{\text{eff}}$$, stay exactly where they were.
2. **Nothing new appears.** A correct trajectory the reference never produces stays at probability zero, and one it produces with negligible probability stays negligible relative to the other correct trajectories.
3. **Only the split between success and failure moves.** The whole effect of the optimum on a problem is the single number $$p \to p^*$$.

This is the cleanest statement of what KL-regularized preference learning does at its optimum: it reweights the reference. In Part 1's terms that is sharpening. The first panel of the animation further down shows it on a made-up problem with eight responses: as $$\beta$$ falls, the correct responses grow together and keep their proportions.

![Line plot of the success probability at the optimum, p star, against the success probability under the reference, p, on a log axis from 0.001 to 1, for beta equal to 2, 1, 0.5 and 0.25, with the diagonal for beta to infinity. All curves start near zero and end at 1. At beta = 1 two points are marked: p = 0.01 goes to 0.027 and p = 0.30 goes to 0.538. The beta = 0.25 curve rises steeply and reaches 0.355 at p = 0.01.]({{ '/assets/img/llm-post-training/part-03-tilt.png' | relative_url }})

The plot shows $$p^*$$ against $$p$$ for four values of $$\beta$$, with the reference success probability on a log axis so that rare problems are visible. The dashed diagonal is $$\beta \to \infty$$, where nothing changes.

Read the curves:

- **At $$\beta = 1$$** a problem at $$p = 0.30$$ rises to 0.538, but a rare problem at $$p = 0.01$$ only reaches 0.027. Multiplying small odds by 2.72 leaves them small.
- **Smaller $$\beta$$ tilts harder**: $$p = 0.30$$ reaches 0.414 at $$\beta = 2$$, 0.760 at $$\beta = 0.5$$ and 0.959 at $$\beta = 0.25$$. The rare problem reaches 0.069 at $$\beta = 0.5$$ and 0.355 at $$\beta = 0.25$$; at $$\beta = 0.25$$ even $$p = 0.001$$ becomes 0.052.
- **$$p = 0$$ stays at 0** for every $$\beta$$; the curves leave the origin only by multiplying something that is already there.

The takeaway connects to pass@k. The tilt raises $$p$$ on every problem the reference can solve, so it raises pass@k at every $$k$$ on those problems: at $$\beta = 1$$, pass@64 of the rare problem goes from 47% to 82%. On problems with $$p = 0$$ it does nothing at any $$k$$. Gains at large $$k$$ are therefore bounded by what the reference can reach, which is the sharpening signature of Part 1.

This statement is about the optimum, and it leaves three things out:

- **Training does not reach the optimum.** A finite number of gradient steps on finite data lands somewhere else, and a later section shows one way the path can differ sharply from the destination.
- **Parameters are shared across prompts.** The optimum is computed prompt by prompt, but a network updated on one prompt changes its behavior on others, including problems where $$p$$ was effectively zero.
- **On-policy RL samples new trajectories.** The derivation assumes the reward is known for every response. A method that samples from the current policy, scores the samples and reinforces them can find responses that were improbable at the start and build on them.

## From the optimum to DPO

RLHF fits a reward model and then runs RL against it. Rafailov et al. [5] noticed that the optimum above can be inverted. Taking logs of $$\pi^*$$ and solving for the reward,

$$
r(x, y) = \beta \log\frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x).
$$

Every reward function corresponds to an optimal policy, and the reward can be read off that policy as a scaled log-ratio to the reference, plus a term that depends only on the prompt. Plug this into Bradley–Terry. The difference $$r(x, y_w) - r(x, y_l)$$ is all the model needs, and the $$\beta \log Z(x)$$ terms cancel:

$$
P(y_w \succ y_l \mid x) = \sigma\Big(\beta \log\frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log\frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\Big).
$$

This cancellation is the reason the method works: $$Z(x)$$ is a sum over every possible response and could never be computed, and it never has to be. It is the same per-prompt shift that Bradley–Terry cannot see.

DPO now replaces $$\pi^*$$ by the trainable policy $$\pi_\theta$$ and fits the preference model by maximum likelihood. Define the *ratio of ratios*

$$
\Delta_\theta = \log\frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log\frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}
= \log\frac{\pi_\theta(y_w \mid x)/\pi_{\text{ref}}(y_w \mid x)}{\pi_\theta(y_l \mid x)/\pi_{\text{ref}}(y_l \mid x)},
$$

and the DPO loss is the negative log-likelihood of the observed preferences,

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\big[\log \sigma(\beta \Delta_\theta)\big].
$$

The policy is its own reward model: $$\beta \log(\pi_\theta/\pi_{\text{ref}})$$ is the *implicit reward*, and $$\beta\Delta_\theta$$ the implicit reward margin of a pair. If the policy class can represent $$\pi^*$$ for the reward that best explains the data, fitting the preferences and solving the KL-regularized problem are the same thing. No reward model is trained, and no response is sampled during training.

## What the loss sees: a ratio of ratios

DPO does not ask that the winner be more likely than the loser. It asks whether, *relative to the reference*, the winner gained more than the loser. A worked example: the reference gives $$\pi_{\text{ref}}(y_w) = 0.01$$ and $$\pi_{\text{ref}}(y_l) = 0.20$$, and after training $$\pi_\theta(y_w) = 0.02$$ and $$\pi_\theta(y_l) = 0.10$$. The loser is still 5 times more likely than the winner. But the winner's ratio to the reference is 2 and the loser's is 0.5, so

$$
\Delta = \log 2 - \log 0.5 = \log 4 \approx 1.386.
$$

Equivalently, the odds of winner to loser went from 0.05 to 0.20, a factor of 4. With $$\beta = 1$$, chosen to keep the numbers round, the loss falls from $$\log 2 = 0.693$$ at the start, where $$\Delta = 0$$, to 0.223, and the implied preference probability is $$\sigma(\log 4) = 0.80$$. With the more common $$\beta = 0.1$$, the implied reward margin is only 0.139 and the loss 0.626: at small $$\beta$$ the log-ratios have to move much further before the loss is satisfied.

## DPO can lower the probability of the preferred response

The example above shows half of something that is easy to miss. The loss depends on $$\pi_\theta(y_w)$$ and $$\pi_\theta(y_l)$$ only through $$\Delta_\theta$$, and $$\Delta_\theta$$ does not change if both log-probabilities move down by the same amount. Nothing in the loss holds $$\pi_\theta(y_w)$$ up.

Take the same reference, 0.01 and 0.20, and suppose training ends at $$\pi_\theta(y_w) = 0.008$$ and $$\pi_\theta(y_l) = 0.04$$. The ratios to the reference are 0.8 and 0.2, so $$\Delta = \log 0.8 - \log 0.2 = \log 4$$ again. The loss is identical to the first example, to every digit, yet the preferred response became *less* likely. The pair held 0.21 of the probability at the start, 0.12 in the first example and 0.048 here: 0.162 of the mass left the pair for responses the loss never looks at. (Even the first, "successful" example lost 0.09 from the pair.)

This is called *likelihood displacement*, and it is not a curiosity. Razin et al. [6] study it as a phenomenon that is often observed in DPO training, and show it can be catastrophic: the displaced mass can go to responses with the opposite meaning. Their example is training a model to prefer "No" over "Never", which can sharply increase the probability of "Yes". Pal et al. [7] show theoretically that the standard DPO loss can reduce the likelihood of preferred responses as long as the relative probability rises, and find it in practice, especially when the two responses in a pair differ in only a few tokens.

Why does the mass leave? Not because of the loss alone. With free logits, one DPO step on a single pair raises $$z_w$$ and lowers $$z_l$$ by the same amount (the softmax normalizer cancels in the difference of log-probabilities), and that always raises $$\pi(y_w)$$; the checks script confirms this on 500 random cases. Displacement comes from *shared parameters*. Here is a toy version of Razin et al.'s example. Three single-token responses, "No" (preferred), "Never" (dispreferred) and "Yes", have logits $$z_j = b_j + u_j^\top h$$, where $$h$$ is a shared hidden state and the $$u_j$$ are output vectors. "No" and "Never" are similar, $$u_{\text{No}} = (1, 0.3)$$ and $$u_{\text{Never}} = (1, -0.3)$$, and "Yes" is $$u_{\text{Yes}} = (0, 1)$$. The biases make the reference $$\pi_{\text{ref}} = (0.40, 0.30, 0.30)$$, and DPO with $$\beta = 1$$ trains only $$h$$. The gradient with respect to $$h$$ is proportional to $$u_{\text{No}} - u_{\text{Never}} = (0, 0.6)$$: the shared part of the two output vectors cancels, and what is left is a direction that "Yes" uses more than either of them. At the start, the log-probability of "No" changes at rate −0.03 per unit of movement along that direction, so it falls from the first step.

<div class="anim" data-anim="part-03-tilt-displacement" data-alt="Animation in two panels. Panel 1: one problem with eight responses, A, B, C correct with reference probabilities 0.15, 0.10, 0.05, D correct but with reference probability 0, and E to H incorrect with 0.30, 0.20, 0.12, 0.08. As beta falls from infinity through 2, 1 and 0.5 to 0.25, the correct bars grow together and the incorrect ones shrink; the success mass goes 0.300, 0.414, 0.538, 0.760, 0.959, the correct bars keep the proportions 0.50 to 0.33 to 0.17 with effective number 2.75, and D stays at 0. Panel 2: DPO on one pair, preferring No over Never, with Yes as a third response, reference probabilities 0.40, 0.30, 0.30. As training proceeds, Delta grows from 0 to log 4 and the loss falls from 0.693 to 0.223, while No falls to 0.20, Never falls to 0.038 and Yes rises to 0.76."><noscript><img src="{{ '/assets/img/llm-post-training/part-03-tilt-displacement.png' | relative_url }}" alt="Final frame of the animation. Panel 1: at beta = 0.25 the correct responses A, B, C have probabilities 0.48, 0.32, 0.16, success mass 0.959, N_eff 2.75, and the unsupported correct response D is still 0. Panel 2: after DPO training to Delta = log 4, No (preferred) is at 0.20 instead of 0.40, Never (dispreferred) at 0.038 instead of 0.30, and Yes at 0.76 instead of 0.30; the loss is 0.223."></noscript></div>

The animation contrasts where the KL-regularized objective says mass should go with where DPO training can actually send it. Both panels use illustrative numbers from the checks script; dashed outlines mark the reference, and Step jumps between the values of $$\beta$$ and then to the second panel.

- **The optimum (panel 1).** A made-up problem has eight responses: three correct ones, A, B and C, with reference probabilities 0.15, 0.10 and 0.05; a correct response D that the reference never produces; and four incorrect ones. As $$\beta$$ falls from $$\infty$$ to 0.25, the success mass goes from 0.300 to 0.959 and A, B and C grow to 0.48, 0.32 and 0.16. Their proportions stay 3 : 2 : 1, so $$N_{\text{eff}}$$ stays at 2.75, and D stays at zero.
- **Training (panel 2).** In the toy model, $$\Delta$$ grows to $$\log 4$$ and the loss falls from 0.693 to 0.223, the same values as in the worked examples. Meanwhile "No", the preferred response, falls from 0.40 to 0.20, "Never" falls from 0.30 to 0.038, and "Yes" rises from 0.30 to 0.76. The pair held 0.70 of the probability and ends with 0.24.

The takeaway is that the DPO loss measures a margin, not a likelihood. Whether the preferred response gains or loses probability depends on how the network's shared parameters couple the two responses to everything else. Razin et al. trace displacement to preferred and dispreferred responses with similar hidden representations, and propose filtering out such pairs. A common practical fix keeps a likelihood term on the winner: Pal et al.'s DPO-Positive adds a penalty whenever $$\pi_\theta(y_w)$$ falls below $$\pi_{\text{ref}}(y_w)$$, and Pang et al. [8] add a negative log-likelihood term on the winner when training on correct-versus-incorrect reasoning chains. They find it crucial, and observe on GSM8K that without it the log-probability of the chosen solutions falls during training.

## The gradient

Write $$L = -\log\sigma(\beta\Delta)$$ for one pair. Since $$\frac{d}{dz}\log\sigma(z) = \sigma(-z)$$,

$$
\frac{\partial L}{\partial \Delta} = -\beta\, \sigma(-\beta\Delta),
$$

and by the chain rule, with $$\pi_{\text{ref}}$$ frozen,

$$
\nabla_\theta L = -\beta\, \sigma(-\beta\Delta_\theta)\,\big[\nabla_\theta \log\pi_\theta(y_w \mid x) - \nabla_\theta \log\pi_\theta(y_l \mid x)\big].
$$

A gradient-descent step therefore moves $$\theta$$ along $$\nabla\log\pi_\theta(y_w) - \nabla\log\pi_\theta(y_l)$$: up on the winner, down on the loser, with one shared weight $$\beta\sigma(-\beta\Delta)$$. The checks script confirms both formulas against finite differences on a small autoregressive model.

The weight is largest for pairs the model currently gets wrong. At $$\beta = 1$$ it is 0.8 when $$\Delta = -\log 4$$ (the loser has gained relative to the winner), 0.5 at $$\Delta = 0$$, and 0.2 at $$\Delta = \log 4$$. Rafailov et al. [5] read it the same way: a pair gets more weight the more the implicit reward model ranks the loser above the winner. The weight depends on $$\beta\Delta$$, the implied reward margin, not on $$\Delta$$ alone. At $$\beta = 0.1$$ the same three pairs get 0.053, 0.050 and 0.047, almost the same; the weight falls to 0.01 only when $$\beta\Delta = \ln 9 \approx 2.2$$, that is, $$\Delta \approx 22$$. Small $$\beta$$ keeps pushing on pairs that are already separated by large log-ratio margins.

The gradient also shows why displacement is possible. The push on the winner and the push on the loser have the same size, and when the two responses share most of their structure (the same tokens, similar hidden states) the two pushes largely cancel on the shared part. What survives is the difference, and its effect on $$\log\pi_\theta(y_w)$$ can have either sign.

## One label, many tokens

A response's log-probability is a sum over its tokens, and so is its log-ratio to the reference:

$$
\log\frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} = \sum_{t=1}^{T} \log\frac{\pi_\theta(y_t \mid x, y_{\lt t})}{\pi_{\text{ref}}(y_t \mid x, y_{\lt t})}.
$$

The preference label is about whole responses, but the gradient reaches every token in both. DPO knows that $$y_w$$ is better than $$y_l$$, not which step made it better. This is a credit-assignment problem. One part of it resolves exactly: if the two responses share a prefix, the shared tokens appear with identical contexts in both log-ratios and cancel in $$\Delta$$, so they get no gradient at all from the pair (the checks script confirms this). Everything after the first difference is credited or blamed as a block, including incidental differences in wording.

Because log-ratios are sums, length scales them. If training shifts every token's log-ratio by 0.01 nats, a 100-token response's log-ratio moves by 1 and a 1,000-token response's by 10. At $$\beta = 0.1$$ that is the difference between an implied preference probability of 0.525 and one of 0.731. This is the *sum* normalization of Part 2's loss-normalization section: each token counts equally, so long responses carry more of the margin and more of the gradient. Length is a known weak point in practice: Park et al. [9] find that DPO produced answers about twice as long on average while the preference data favored longer responses only slightly, and propose a regularizer that controls it.

## SFT and DPO side by side

| | SFT (Part 2) | DPO |
|---|---|---|
| Data | prompt and demonstration $$(x, y)$$ | prompt and pair $$(x, y_w, y_l)$$ |
| Loss for one example | $$-\log\pi_\theta(y \mid x)$$ | $$-\log\sigma(\beta\Delta_\theta)$$ |
| What it asks | make this response likely | make the winner gain more than the loser, relative to $$\pi_{\text{ref}}$$ |
| Push on the chosen response | always up | up, weighted by $$\beta\sigma(-\beta\Delta_\theta)$$; its probability can still fall |
| Push on other responses | down, in proportion to their probability | down on the loser; others move through shared parameters |
| Reference policy | none | frozen $$\pi_{\text{ref}}$$, the anchor |
| Population optimum | $$\pi_\theta = q_{\text{data}}$$ (forward KL) | $$\pi_{\text{ref}}\, e^{r/\beta}/Z$$ for the reward that explains the preferences |
| Samples from the model during training | no | no (offline DPO) |

Both are supervised losses on a fixed dataset. SFT pulls toward the data distribution; DPO reweights the reference according to the preferences. Neither looks at a response the model generates during training.

## Offline DPO and exploration

The last row of the table matters for the question of this series. Offline DPO trains on a fixed set of pairs. If a strategy never appears in a pair, no label says it is good. There is no loop that samples a response, discovers that it works and reinforces it, the loop that reinforcement learning with verifiable rewards, as described in Part 0, is built on. And the optimum DPO aims at is a reweighting of the reference.

That makes a reasonable hypothesis: **offline DPO tends to sharpen, redistributing probability among responses the reference already produces, rather than expanding what the model can solve.** It is a hypothesis, not a theorem, for two reasons covered above. Training does not reach the optimum, and shared parameters mean DPO changes the probability of responses that are not in the data at all. Likelihood displacement is the clearest example; the same coupling could, in principle, also raise correct responses outside the data on other problems.

Two variations change the picture. *Iterative* or *online* DPO samples new pairs from the current policy, labels them, trains, and repeats, which restores a version of the sampling loop. Pang et al. [8] do this for reasoning: each iteration samples new chains of thought from the latest model, pairs correct against incorrect ones, and trains with DPO plus a likelihood term, using the previous iteration's model as the reference. They report accuracy rising over the iterations on GSM8K, MATH and ARC-Challenge.

The other variation concerns verifier labels themselves. A verifier prefers a correct response to an incorrect one every time. Under Bradley–Terry, a preference probability of exactly 1 requires an infinite reward gap, and plugged into the optimum that gives $$\pi^*(y_l) = 0$$ for every $$\beta$$. Azar et al. [10] point this out: the more deterministic the preferences, the weaker the KL regularization, and with finite data even preferences that are not deterministic can look deterministic. For correct-versus-incorrect pairs, the idealized target is therefore the $$\beta \to 0^+$$ limit of the binary tilt: the reference restricted to its correct responses, $$p^* = 1$$ on every problem with $$p \gt 0$$, and $$q(\tau \mid x)$$ unchanged. That is still pure sharpening, but $$\beta$$ no longer decides where training is headed. On a finite dataset many policies drive the loss toward zero, and what keeps the model near $$\pi_{\text{ref}}$$ is mostly that training stops.

<div class="note" markdown="1">
<span class="label">In my experiments</span>

The DPO arm of [sharpen-or-expand](https://github.com/hieutrungle/sharpen-or-expand) has no results yet; it runs after the RL screening runs, and its results will be added to this post when they exist. What is fixed is the design, and it follows from the question of the series.

- **The reference is the base model.** DPO starts from the same pinned Qwen3.5-4B-Base checkpoint as every RL run, so $$\pi_{\text{ref}}$$ is the base ([start decision](https://github.com/hieutrungle/sharpen-or-expand/blob/main/docs/decisions/031-rl-and-dpo-start-from-the-base.md)). Starting from the base rather than an SFT model is less conventional for DPO, but it keeps one starting point for every method compared.
- **Pairs come from the base policy, with the RL sampler.** Responses are sampled at temperature 1.0 and top-p 1.0, with the same prompt and the 16,384-token cap as RL rollouts ([sampling decision](https://github.com/hieutrungle/sharpen-or-expand/blob/main/docs/decisions/017-rl-rollout-sampling-1.0.md), including its amendment on preference data). The choice follows the closest published setup the decision record found, iterative DPO with rule-based rewards (Online-DPO-R1), which samples at temperature 1.0; it notes that other reasoning-DPO work uses different values, Pang et al. [8] for example 0.8 and later 1.3.
- **The verifier labels the pairs.** DPO here learns from correct-versus-incorrect pairs. The [milestone plan](https://github.com/hieutrungle/sharpen-or-expand/blob/main/docs/milestone-plan.md) sizes the preference data at 15K–30K prompts with 4–8 samples each, and the trained model is evaluated on the full suite with the same protocol as every other checkpoint.
- **Still open.** How pairs are built from the samples, including whether a response truncated at the token cap may serve as a loser, is decided with the DPO pipeline. The value of $$\beta$$ is not fixed yet either.

Why the design matters. Sampling pairs from the starting policy with the RL sampler means DPO sees the same distribution of responses the RL algorithms start from, so differences between them come from how each method learns, not from different data. Training on one fixed set of pairs from the starting policy is exactly the setting of the previous section: no discovery loop, a reweighting target, and, with verifier labels, deterministic preferences. The RL algorithms in the study train with [no KL term and no reference model](https://github.com/hieutrungle/sharpen-or-expand/blob/main/docs/decisions/013-rl-no-kl-step-checkpoints.md), so DPO is the only arm anchored to $$\pi_{\text{ref}}$$ by its objective. If the sharpening hypothesis is right, DPO should raise pass@1 on problems the base already solves and leave the hard problems the base never solves where they are.
</div>

## Check yourself

**1.** With $$\pi_{\text{ref}}(y_w) = 0.01$$ and $$\pi_{\text{ref}}(y_l) = 0.20$$, training ends at $$\pi_\theta(y_w) = 0.008$$ and $$\pi_\theta(y_l) = 0.04$$. Did DPO succeed?

<details markdown="1">
<summary>Answer</summary>

By its own measure, yes. The winner's ratio to the reference is 0.8 and the loser's 0.2, so $$\Delta = \log 4$$ and the odds of winner to loser rose from 0.05 to 0.20. The loss is exactly what it would be at 0.02 and 0.10. But the preferred response is less likely than before, and 0.162 of the probability left the pair for responses the loss never sees. Whether that is success depends on where the mass went, which the loss does not say. If you care about the winner's likelihood, monitor it, or add a likelihood term on the winner as DPO-Positive does.
</details>

**2.** Why does KL-regularized RL produce $$\pi^*(y) \propto \pi_{\text{ref}}(y)\, e^{r(y)/\beta}$$, and what happens as $$\beta \to \infty$$ and $$\beta \to 0^+$$?

<details markdown="1">
<summary>Answer</summary>

The objective equals $$\beta\log Z - \beta D_{\mathrm{KL}}(\pi \Vert \pi^*)$$, which is largest when the KL term is zero, that is, at $$\pi = \pi^*$$. The reference supplies the starting probabilities and the exponential reward factor tilts them. As $$\beta \to \infty$$ the tilt vanishes and $$\pi^* \to \pi_{\text{ref}}$$; at $$\beta = 1{,}000$$ a response with reference probability 0.2 keeps probability 0.200. As $$\beta \to 0^+$$ the mass concentrates on the maximum-reward responses that the reference supports, in proportion to their reference probabilities. If three such responses have reference probabilities 0.2, 0.1 and 0.05, the limit gives them 0.571, 0.286 and 0.143. A maximum-reward response with reference probability zero gets nothing.
</details>

**3.** After DPO, pass@1 on a benchmark goes from 30% to 50%, but pass@64 barely changes. What likely happened?

<details markdown="1">
<summary>Answer</summary>

The pattern is consistent with sharpening. pass@64 is close to 1 for any problem with a modest success probability: $$1 - 0.93^{64} = 0.99$$ already at $$p = 0.07$$, and at $$p = 0.30$$ pass@64 is >99.9%. So raising $$p$$ on problems the model already solved occasionally moves pass@1 a lot and pass@64 hardly at all. That is what the KL-regularized optimum would do: for a single problem, 0.30 → 0.50 is an odds factor of 2.33, the tilt at $$\beta = 1.18$$, which would lift a rare problem only from 0.01 to 0.023 and leave unsolvable ones at zero. But "consistent with" is the right wording, not "proves". By Part 1's ladder of evidence, a flat pass@64 is the weakest rung: it cannot rule out a few problems moving from unsolved to solved, and settling that needs problem-level transitions at a large budget.
</details>

**4.** Two 500-token solutions share their first 300 tokens and differ in one crucial step after that. What does DPO learn from the pair?

<details markdown="1">
<summary>Answer</summary>

The 300 shared tokens cancel exactly in $$\Delta$$ and receive no gradient from this pair. The remaining 200 tokens of each solution are pushed as a block: up for the winner, down for the loser, all with the same weight $$\beta\sigma(-\beta\Delta)$$. The crucial step is a few tokens among those 200, and DPO has no way to single it out. Everything after the divergence point, including later steps that are fine in both solutions but worded differently, gets the same credit or blame. Over many pairs the decisive differences are the consistent ones, so the signal can average out in their favor, but a single pair does not localize it. This is the same sequence-level-label problem that a verifiable reward has, and the reason step-level signals are studied.
</details>

## Takeaways

- Preference learning uses comparisons $$(x, y_w, y_l)$$. Under Bradley–Terry, $$P(y_w \succ y_l) = \sigma(r_w - r_l)$$: only reward differences matter.
- The KL-regularized objective $$\mathbb{E}_\pi[r] - \beta D_{\mathrm{KL}}(\pi \Vert \pi_{\text{ref}})$$ has the closed-form optimum $$\pi^* = \pi_{\text{ref}}\, e^{r/\beta}/Z$$: a reweighting of the reference that cannot create responses the reference never produces. $$\beta$$ trades reward against staying close; $$\beta \to 0^+$$ concentrates on the best supported responses.
- With a 0/1 reward the optimum multiplies the odds of success by $$e^{1/\beta}$$ and leaves $$q(\tau \mid x)$$ and $$N_{\text{eff}}$$ unchanged. At its optimum, KL-regularized preference learning is sharpening.
- DPO inverts the optimum, $$r = \beta\log(\pi/\pi_{\text{ref}}) + \beta\log Z$$, lets $$Z$$ cancel in Bradley–Terry, and fits $$-\log\sigma(\beta\Delta_\theta)$$ to the pairs: no reward model, no sampling.
- The loss sees only the ratio of ratios $$\Delta_\theta$$. Both responses can lose probability while $$\Delta_\theta$$ grows; through shared parameters the mass can go to responses outside the data (likelihood displacement).
- The gradient weight $$\beta\sigma(-\beta\Delta)$$ is largest on pairs the model gets wrong. One label is spread over every token after the first difference, and long responses carry larger log-ratios.
- Offline DPO has no sample–discover–reinforce loop, so it plausibly leans toward sharpening. With verifier labels the preferences are deterministic, and $$\beta$$ no longer fixes the end point.

## Next

Part 4 turns to that missing loop. Policy-gradient methods sample responses from the current policy, evaluate them, and reinforce the ones that score well. Part 4 derives the policy gradient from first principles, adds baselines and importance ratios, and builds up to PPO.

## References

1. R. A. Bradley and M. E. Terry, "Rank analysis of incomplete block designs: I. The method of paired comparisons," *Biometrika*, vol. 39, no. 3/4, pp. 324–345, 1952. [DOI](https://doi.org/10.2307/2334029)
2. P. Christiano, J. Leike, T. B. Brown, M. Martic, S. Legg, and D. Amodei, "Deep reinforcement learning from human preferences," arXiv:1706.03741, 2017. [arXiv](https://arxiv.org/abs/1706.03741)
3. D. M. Ziegler, N. Stiennon, J. Wu, T. B. Brown, A. Radford, D. Amodei, P. Christiano, and G. Irving, "Fine-tuning language models from human preferences," arXiv:1909.08593, 2019. [arXiv](https://arxiv.org/abs/1909.08593)
4. L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. L. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, J. Schulman, J. Hilton, F. Kelton, L. Miller, M. Simens, A. Askell, P. Welinder, P. Christiano, J. Leike, and R. Lowe, "Training language models to follow instructions with human feedback," in *Proc. NeurIPS*, 2022. [arXiv](https://arxiv.org/abs/2203.02155)
5. R. Rafailov, A. Sharma, E. Mitchell, S. Ermon, C. D. Manning, and C. Finn, "Direct preference optimization: Your language model is secretly a reward model," in *Proc. NeurIPS*, 2023. [arXiv](https://arxiv.org/abs/2305.18290)
6. N. Razin, S. Malladi, A. Bhaskar, D. Chen, S. Arora, and B. Hanin, "Unintentional unalignment: Likelihood displacement in direct preference optimization," in *Proc. ICLR*, 2025. [arXiv](https://arxiv.org/abs/2410.08847)
7. A. Pal, D. Karkhanis, S. Dooley, M. Roberts, S. Naidu, and C. White, "Smaug: Fixing failure modes of preference optimisation with DPO-Positive," arXiv:2402.13228, 2024. [arXiv](https://arxiv.org/abs/2402.13228)
8. R. Y. Pang, W. Yuan, K. Cho, H. He, S. Sukhbaatar, and J. Weston, "Iterative reasoning preference optimization," in *Proc. NeurIPS*, 2024. [arXiv](https://arxiv.org/abs/2404.19733)
9. R. Park, R. Rafailov, S. Ermon, and C. Finn, "Disentangling length from quality in direct preference optimization," in *Findings of ACL*, 2024. [arXiv](https://arxiv.org/abs/2403.19159)
10. M. G. Azar, M. Rowland, B. Piot, D. Guo, D. Calandriello, M. Valko, and R. Munos, "A general theoretical paradigm to understand learning from human preferences," arXiv:2310.12036, 2023. [arXiv](https://arxiv.org/abs/2310.12036)
