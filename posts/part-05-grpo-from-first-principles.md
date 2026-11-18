---
title: "GRPO from First Principles: An Advantage Without a Critic"
kind: Study notes
date: 2026-11-17 15:00:00 -0600
series: llm-post-training
part: 5
summary: "GRPO replaces PPO's learned critic with the statistics of a group of responses to the same prompt. Part 5 derives the group-relative advantage, works out its closed form for 0/1 rewards, shows what centering and dividing by the standard deviation do to the update, and explains why GRPO learns only from mixed groups and why the group size is a training-time exploration budget."
math: true
anim: true
---

*Code, check scripts and references: [llm-post-training-notes](https://github.com/hieutrungle/llm-post-training-notes) · Experiments: [sharpen-or-expand](https://github.com/hieutrungle/sharpen-or-expand)*

<div class="goal" markdown="1">
<span class="label">Core idea</span>

GRPO replaces PPO's learned critic with the statistics of a group of responses to the same prompt. Each response's advantage is how much better it scored than its group, measured in units of the group's spread. A group whose responses all score the same gives no signal, so GRPO learns only from mixed groups, and a success counts for more the rarer it is in its group. The group size sets how rare a success can be and still be found: it is the training-time exploration budget.
</div>

## The question

[Part 4](/blog/2026/11/policy-gradients-reinforce-to-ppo/) built the policy-gradient update and ended with PPO, which estimates each response's advantage with a learned critic, a second network about as large as the policy. Training and serving that network is expensive, and with a single reward at the end of a long response it is hard to train well. This part asks how to get an advantage without a critic, and what the replacement, a comparison within a group of samples, actually rewards.

## The picture to keep in mind

Here is Part 4's hard problem again, now trained with group-relative advantages instead of the exact baseline $$b = p$$. Each round samples a group of 8 responses, a verifier scores them, and each response is pushed up or down by how it compares with the rest of its group.

<div class="anim" data-anim="part-05-group-advantage" data-alt="Animation of eleven rounds of GRPO-style training on one problem. Left, bars give the policy's probability of each solution strategy: a rare correct strategy B at 0.02, a correct strategy Z with probability 0, and three wrong strategies. Right, each round shows a group of 8 sampled strategies as chips and, under each, a bar for its advantage above or below zero. Bottom, a strip records the advantage a correct sample earned in each round. In rounds 1 and 2 all samples are wrong and every advantage is 0, so nothing moves. In round 3 one sample is correct and gets advantage +2.65 while each wrong sample gets −0.38. Over the next rounds groups contain more correct samples and the advantage of a correct sample shrinks: 2.65, 2.65, 2.65, 1.73, 1.00, 0.77, 0.38. In round 9 the lone wrong sample gets −2.65. In rounds 10 and 11 all 8 samples are correct, every advantage is 0, and the success probability stays at 0.969. Z stays at 0."><noscript><img src="{{ '/assets/img/llm-post-training/part-05-group-advantage.png' | relative_url }}" alt="Final frame after eleven rounds: the rare correct strategy B holds probability 0.969, Z is still at 0, the last group has 8 of 8 correct with every advantage 0, and the strip of a correct sample's advantage by round reads –, –, 2.65, 2.65, 2.65, 1.73, 1.00, 0.77, 0.38, 0, 0."></noscript></div>

The animation shows one problem. The policy spreads its probability over five solution strategies, drawn as bars on the left: a rare correct strategy B (probability 0.02), a correct strategy Z that the policy never produces, and three wrong strategies. Each round draws a group of 8 samples, shown as chips on the right. A correct sample gets reward 1 and a wrong one 0. Each sample's advantage is its reward minus the group's mean reward, divided by the group's standard deviation, and is drawn as a bar under its chip. One policy-gradient step then pushes each sampled strategy by its advantage; every strategy has its own logit, and the step size is 2. The strip at the bottom records the advantage that a correct sample earned in each round. The run is simulated and the numbers are illustrative. The samples come from a seeded random generator, and the seed was chosen so that the first success comes early: at $$p = 0.02$$ a group of 8 contains a success with probability 0.149, so the wait is 6.7 rounds on average. The [checks script](https://github.com/hieutrungle/llm-post-training-notes/blob/main/checks/part05_checks.py) runs the same simulation.

Read the animation:

- **All-wrong groups teach nothing.** In rounds 1 and 2 every sample fails. The group mean is 0, every advantage is 0, and nothing moves, not even the sampled failures. The success probability stays at 0.020. (With Part 4's fixed baseline $$b = p$$, each failure got −0.02.)
- **A lone success gets a large advantage.** In round 3 one sample of 8 succeeds. Its advantage is $$\sqrt{7}$$ ≈ 2.65, and each of the 7 failures gets −0.38. After the update the success probability is 0.047, and the next two groups again contain one success each.
- **The same reward earns less as successes become common.** As $$p$$ rises to 0.226, 0.489, 0.788 and 0.930, groups contain 2, 4, 5 and then 7 successes, and the advantage of a correct sample falls from 2.65 to 1.73, 1.00, 0.77 and 0.38. In round 9 the roles flip: the lone failure is now the surprise and gets −2.65.
- **All-correct groups teach nothing either.** In rounds 10 and 11 all 8 samples succeed, every advantage is 0, and the policy stops moving at 0.969, with 3.1% of its probability still on wrong strategies. Z, never sampled, stays at 0.

That is the method in one sentence: **GRPO compares each response with the other responses to the same prompt, so it learns only from groups that contain both successes and failures, and a success is worth more the rarer it is in its group.** The rest of this part derives that advantage, works out what it does for 0/1 rewards, and puts it into the PPO-style objective.

## Replacing the critic with a group

PPO's advantage needs a value estimate $$V(x)$$, the expected reward from the prompt (Part 4). Shao et al. [1] introduced Group Relative Policy Optimization (GRPO) in DeepSeekMath to remove the network that provides it. They point out that the value function in PPO "is typically another model of comparable size as the policy model", a substantial memory and compute burden, and that with a reward only at the last token it is hard to train a value function that is accurate at every token. GRPO instead uses the average reward of several outputs sampled for the same question as the baseline.

For one prompt $$x$$, sample a group of $$G$$ responses from the rollout policy,

$$
o_1, \ldots, o_G \sim \pi_{\text{old}}(\cdot \mid x),
$$

where each $$o_i$$ is a trajectory $$\tau_i$$ in the notation of the earlier parts. Score them, $$R_i = R(x, o_i)$$, and compute the group mean

$$
\bar R = \frac{1}{G} \sum_{j=1}^{G} R_j .
$$

With a 0/1 verifier reward, the value of the prompt is its success probability, $$V(x) = \mathbb{E}[R \mid x] = p_\theta(x)$$, and $$\bar R$$ is a Monte Carlo estimate of it from $$G$$ samples. Subtracting it gives an advantage-like quantity, $$R_i - \bar R$$: positive for a response that did better than its group, negative for one that did worse.

Two facts about the setting are worth keeping straight. In DeepSeekMath the rewards came from a learned reward model, with 64 outputs per question and a KL coefficient of 0.04 [1]. DeepSeek-R1 later trained with GRPO on rule-based rewards that check the answer and the format; for DeepSeek-R1-Zero it sampled 16 outputs per question with a KL coefficient of 0.001 [2]. The algorithm does not care where the reward comes from; this series uses it with a 0/1 verifier.

The group mean is not an exact critic. It is noisy, since it is estimated from $$G$$ samples, and it includes the sample it is compared with. Part 4 showed what the second fact costs: with $$R_i - \bar R$$ as the advantage, the expected update is exactly $$(G - 1)/G$$ times the true policy gradient (0.875 at $$G = 8$$), a rescaling that does not change the direction. Leaving the sample out of its own baseline gives the leave-one-out estimator (RLOO) of Ahmadian et al. [4], whose advantages are $$G/(G - 1)$$ times the mean-centered ones. So mean-centering alone is RLOO up to a constant factor. GRPO also divides by the group's spread, and that changes more.

## The group-relative advantage

Define the group standard deviation as the population standard deviation of the $$G$$ rewards, dividing by $$G$$:

$$
\sigma_R = \sqrt{\frac{1}{G} \sum_{j=1}^{G} \big(R_j - \bar R\big)^2 } .
$$

GRPO's advantage for response $$i$$ is

$$
A_i = \frac{R_i - \bar R}{\sigma_R}.
$$

With one reward per response (outcome supervision), every token of response $$i$$ gets the same advantage $$A_i$$ [1]. This is Part 4's credit-assignment problem unchanged: the key step and the filler tokens of a response are pushed equally. What is new is only how the single number $$A_i$$ is computed.

## Binary rewards in closed form

With a 0/1 reward the advantage takes only two values. Suppose $$k$$ of the $$G$$ responses are correct, with $$0 \lt k \lt G$$. Then $$\bar R = k/G$$, and since every reward is 0 or 1,

$$
\sigma_R^2 = \frac{k}{G}\Big(1 - \frac{k}{G}\Big), \qquad \sigma_R = \frac{\sqrt{k (G - k)}}{G}.
$$

A correct response has $$R_i - \bar R = (G - k)/G$$ and a wrong one $$-k/G$$, so

$$
A_+ = \sqrt{\frac{G - k}{k}}, \qquad A_- = -\sqrt{\frac{k}{G - k}} .
$$

The two are tied by $$A_+ \cdot \lvert A_- \rvert = 1$$, and the advantages of a group always sum to zero: $$k A_+ + (G - k) A_- = 0$$. For $$G = 8$$:

| Correct in the group, $$k$$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| Advantage of a correct response, $$A_+$$ | 2.646 | 1.732 | 1.291 | 1.000 | 0.775 | 0.577 | 0.378 |
| Advantage of a wrong response, $$A_-$$ | −0.378 | −0.577 | −0.775 | −1.000 | −1.291 | −1.732 | −2.646 |

The table is symmetric. With one success in 8, the success gets $$\sqrt{7}$$ ≈ 2.65 and each failure −0.38; with seven successes, the lone failure gets −2.65 and each success 0.38. GRPO puts the large weight on whichever outcome is the surprise in its group. A larger group makes a lone success more surprising: at $$G = 16$$ and $$k = 1$$, $$A_+ = \sqrt{15}$$ ≈ 3.87 and each failure gets −0.258.

<div class="note" markdown="1">
<span class="label">Which standard deviation?</span>

The closed form assumes the population standard deviation. DeepSeekMath writes $$\mathrm{std}(\mathbf{r})$$ without saying which one [1]. Implementations commonly use the sample standard deviation, which divides by $$G - 1$$ (Bessel's correction), and add a small constant $$\delta$$ to the denominator. verl 0.9.1, the library the experiments use, does exactly this: its GRPO outcome-advantage function [6] takes `torch.std` of each group's scores, which is Bessel-corrected by default, and computes `(scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)` with `epsilon = 1e-6`, the constant $$\delta$$. At $$G = 8$$ and $$k = 1$$ that gives $$A_+$$ ≈ 2.475 and $$A_-$$ ≈ −0.354 instead of 2.646 and −0.378; at $$G = 16$$, 3.750 and −0.250 instead of 3.873 and −0.258.

The difference is a constant. For a fixed $$G$$, the sample standard deviation is $$\sqrt{G/(G - 1)}$$ times the population one, so every advantage is multiplied by $$\sqrt{(G - 1)/G}$$ (0.935 at $$G = 8$$), apart from $$\delta$$. That is a slightly smaller learning rate, not a different direction, so everything in this part holds for either convention. The constant $$\delta$$ is what makes an all-equal group safe: its advantages are $$0/(0 + \delta) = 0$$ rather than $$0/0$$.
</div>

## What dividing by the standard deviation does

Mean-centering alone keeps the update a rescaled policy gradient. Dividing by $$\sigma_R$$ does not, because $$\sigma_R$$ depends on the rewards, including the sample's own. To see what it does instead, compute the expected update for one prompt.

Take a 0/1 reward and split the expectation by the sample's reward. Given that a sample is correct, it is distributed as the policy restricted to correct responses, $$\pi_\theta(\tau)/p$$ on the successful set; Part 2 showed that the expected score under that distribution is $$\nabla_\theta \log p$$. So

$$
\mathbb{E}\big[\nabla_\theta \log \pi_\theta(o) \mid R = 1\big] = \frac{\nabla_\theta p}{p}, \qquad
\mathbb{E}\big[\nabla_\theta \log \pi_\theta(o) \mid R = 0\big] = -\frac{\nabla_\theta p}{1 - p},
$$

where the second follows from the same argument for the failures, whose probability is $$1 - p$$. GRPO's advantage depends on the group only through the rewards, and given the rewards the responses are independent. The expected update for one prompt is therefore a multiple of $$\nabla_\theta p$$:

$$
\mathbb{E}\Big[\frac{1}{G}\sum_{i=1}^{G} A_i\, \nabla_\theta \log \pi_\theta(o_i)\Big] = w_G(p)\, \nabla_\theta p,
\qquad
w_G(p) = \mathbb{E}\Big[A_1 \Big(\frac{R_1}{p} - \frac{1 - R_1}{1 - p}\Big)\Big].
$$

For one prompt, every choice of advantage points the update the same way, up the success probability. What the choice decides is the weight $$w_G(p)$$, and with many prompts in a batch the weights decide which prompts drive the update.

- **Mean-centering only**, $$A_i = R_i - \bar R$$: $$w_G = (G - 1)/G$$ for every $$p$$, Part 4's result. Every prompt keeps the weight the true gradient gives it.
- **With $$\sigma_R$$**: $$w_G$$ depends on $$p$$. The checks script sums it exactly over the number of correct responses. Summed over prompts, the update is then no longer the gradient of $$\sum_x p_\theta(x)$$ but of $$\sum_x h(p_\theta(x))$$ for a transform $$h$$ with $$h' = w_G$$.

For large $$G$$, $$\bar R \to p$$ and $$\sigma_R \to \sqrt{p(1 - p)}$$, so $$A_i \to (R_i - p)/\sqrt{p(1 - p)}$$ and

$$
w_G(p) \to \frac{1}{\sqrt{p(1 - p)}}, \qquad \frac{1}{\sqrt{p(1 - p)}}\, \nabla_\theta p = \nabla_\theta\, 2 \arcsin\sqrt{p} .
$$

In this limit GRPO climbs $$2 \arcsin \sqrt{p}$$ for each prompt instead of $$p$$. Davis and Recht [5] derive the same view: several RL algorithms for language models with binary rewards are stochastic gradient ascent on a monotone transform of the probability of a correct answer, and for GRPO the transform tends to the arcsine of the square root as the group grows and the constant in the denominator shrinks.

![Line plot of the weight w on a prompt's gradient, on a log axis from 0.5 to 40, against the per-sample success probability p on a logit axis from 0.001 to 0.999. Centering only at G = 8 is a flat line at 0.875. Dividing by the standard deviation at G = 8 gives a curve that is 1.86 at p = 0.5 and rises to about 2.65 at both ends. At G = 64 the curve follows the dashed large-G limit 1/sqrt(p(1 − p)) from about p = 0.03 to 0.97 and flattens near 8 at the ends. The dashed limit is 2 at p = 0.5 and 10.05 at p = 0.01; dots mark G = 8 at p = 0.01 (2.61) and p = 0.5 (1.86), open circles mark the limit at the same points.]({{ '/assets/img/llm-post-training/part-05-weight.png' | relative_url }})

The plot shows the weight $$w_G(p)$$ that one prompt's gradient $$\nabla_\theta p$$ receives in the expected update, against the prompt's success probability $$p$$, computed exactly. The orange line is mean-centering alone at $$G = 8$$. The blue curves divide by the population standard deviation, at $$G = 8$$ and $$G = 64$$, and the dashed curve is the large-$$G$$ limit $$1/\sqrt{p(1 - p)}$$. Both axes are stretched: the horizontal one so that both ends of $$[0, 1]$$ are visible, the vertical one logarithmically.

Read the curves:

- **Centering only is flat.** Every prompt gets weight 0.875 at $$G = 8$$, whatever its $$p$$.
- **Dividing by $$\sigma_R$$ favors the extremes.** At $$G = 8$$ the weight is 1.86 at $$p = 0.5$$ and 2.61 at $$p = 0.01$$: a prompt the model rarely solves, or almost always solves, gets more weight relative to its gradient than a prompt at even odds, and 2.12 to 2.98 times what centering alone gives.
- **The arcsine limit needs a large group.** At $$G = 8$$ the weight stays within 10% of the limit only for $$p$$ between 0.30 and 0.70, and it can never exceed $$\sqrt{G - 1}$$ ≈ 2.65, the advantage of a lone success, while the limit is 10.05 at $$p = 0.01$$. At $$G = 64$$ the curve is within 10% of the limit from $$p$$ ≈ 0.03 to 0.97, and 6.74 at $$p = 0.01$$.

The takeaway: dividing by $$\sigma_R$$ is a per-prompt reweighting. It does not change the direction of any single prompt's update; it changes how much each prompt counts, by an amount that depends on the prompt's current success probability and on $$G$$. The weight multiplies $$\nabla_\theta p$$, which itself shrinks near $$p = 0$$ and $$p = 1$$ (in Part 4's one-logit model its size is $$p(1 - p)$$), so for a single prompt the expected update is still largest at intermediate $$p$$. Whether this reweighting is a feature or a bias to correct is the subject of the next part.

## Reward is not advantage

Two correct responses with the same reward $$R = 1$$ can get very different advantages, because their groups differ. If prompt A's group has 1 correct response in 8, its success gets $$\sqrt{7}$$ ≈ 2.65. If prompt B's group has 4 in 8, each success gets 1.00. If prompt C's group has 8 in 8, they get 0.

The verifier defines what counts as correct. The group defines how hard the update pushes. And the group changes as the policy improves: in the opening animation, the same correct strategy B earned 2.65 in round 3 and nothing in round 10. Difficulty, as the update sees it, is not a fixed property of the prompt; it is the prompt's current success probability under the current policy. The reward is fixed, but the optimization pressure moves with the policy.

## Group regimes and the learning frontier

With a 0/1 reward a group falls into one of three regimes:

| Group | Example rewards | $$\sigma_R$$ | Advantages |
|---|---|---|---|
| All wrong | 0, 0, …, 0 | 0 | all 0 |
| Mixed | 1, 0, 1, 0, … | $$\gt 0$$ | $$A_+ \gt 0$$, $$A_- \lt 0$$ |
| All correct | 1, 1, …, 1 | 0 | all 0 |

All-wrong and all-correct groups have no spread: every response scored the same as its group, so there is nothing to compare, and the advantage is set to 0 (in verl, by the constant $$\delta$$ in the denominator). These are *zero-variance groups*. GRPO learns only from mixed groups, and for a prompt with success probability $$p$$ a group of $$G$$ independent samples is mixed with probability

$$
P_{\text{mixed}} = 1 - p^G - (1 - p)^G .
$$

At $$p = 0.5$$ that is 0.992 for $$G = 8$$; near $$p = 0$$ and $$p = 1$$ it vanishes. The prompts that drive learning at any moment sit on a frontier of intermediate difficulty, and the frontier moves as the policy improves. The two kinds of zero-variance group mean different things: an all-correct group is a prompt already mastered, while an all-wrong group is a prompt on which the update has no direction at all.

## Group size as an exploration budget

A prompt contributes a success to the update only when its group contains one. That happens with probability

$$
P(\text{at least one success}) = 1 - (1 - p)^G,
$$

the same formula as pass@k in [Part 1](/blog/2026/11/capability-and-pass-at-k/), with the group as the $$k$$ samples. At evaluation, pass@k asks whether repeated sampling reveals a successful response. During GRPO training the same number asks whether repeated sampling reveals a successful response that can then be reinforced.

| Success probability $$p$$ | $$G = 8$$ | $$G = 64$$ |
|---|---|---|
| 0.01 | 7.73% | 47.4% |
| 0.02 | 14.9% | 72.6% |
| $$10^{-8}$$ | $$8.0 \times 10^{-8}$$ | $$6.4 \times 10^{-7}$$ |

A larger group finds rare successes that a small one misses: at $$p = 0.01$$ a group of 64 contains one in almost half of all rounds, a group of 8 in fewer than one in twelve. But no practical group size helps a prompt with $$p = 10^{-8}$$. This is Part 4's conclusion in group form: GRPO can amplify a rare success only once a group contains one, and the group size sets how rare that success may be.

![Two panels against the per-sample success probability p on a logit axis from 0.0001 to 0.9999, for groups of 4, 8, 16 and 64 samples. Left: the probability that a group contains at least one success, an S-shaped curve that rises from 0 to 1, further left for larger groups; dots mark p = 0.01 at 7.7% for G = 8 and 47.4% for G = 64. Right: the probability that a group is mixed, a hump centered at p = 0.5 that is 0.875 high for G = 4 and essentially 1 for larger groups, and wider for larger groups.]({{ '/assets/img/llm-post-training/part-05-frontier.png' | relative_url }})

The figure plots both probabilities against $$p$$ for groups of 4, 8, 16 and 64, on a horizontal axis stretched so that both ends of $$[0, 1]$$ are visible. The thick line is $$G = 8$$, the group size of the experiments.

Read the curves:

- **Left, at least one success.** Each curve rises from 0 to 1 around $$p = 1/G$$, where it is about two thirds (0.656 for $$G = 8$$). At $$p = 0.01$$ the probability is 7.7% for $$G = 8$$ and 47.4% for $$G = 64$$. Below $$p = 0.001$$ even $$G = 64$$ rarely finds a success.
- **Right, a mixed group.** Each curve is a hump centered at $$p = 0.5$$, 0.875 high for $$G = 4$$ and essentially 1 for larger groups. A larger group widens the hump on both sides: it finds rare successes on hard prompts and rare failures on easy ones.

The takeaway: the group size moves the learning frontier toward rarer successes. It cannot reach prompts whose success probability is far below $$1/G$$.

## Positive feedback

Once a group contains a success, a loop starts. The success gets a positive advantage, its probability rises, it is sampled more often, more groups contain it, and it is reinforced again:

$$
\text{discovery} \;\to\; A_+ \gt 0 \;\to\; \pi_\theta(o)\uparrow \;\to\; \text{sampled more often} \;\to\; \text{reinforced again}.
$$

In the opening animation this loop took the rare strategy from 0.02 to 0.969 in seven rounds once it was found. The loop can turn a rare successful strategy into the dominant one quickly. If that strategy was already reachable before training, as B was at 0.02, the result is sharpening in [Part 1](/blog/2026/11/capability-and-pass-at-k/)'s sense: an existing successful mode became more probable. On Part 1's ladder of evidence, a rising pass@1 from such a loop is consistent with sharpening; expansion would need evidence higher up the ladder, such as problems unsolved at a large sampling budget becoming repeatedly solvable.

## The full objective

GRPO puts the group advantage into Part 4's clipped objective. With the per-token ratio $$\rho_{i,t} = \pi_\theta(o_{i,t} \mid x, o_{i,\lt t}) / \pi_{\text{old}}(o_{i,t} \mid x, o_{i,\lt t})$$ and clip range $$\epsilon$$, DeepSeekMath's objective [1] is

$$
J_{\text{GRPO}}(\theta) = \mathbb{E}_{x,\, \{o_i\} \sim \pi_{\text{old}}}\Bigg[\frac{1}{G}\sum_{i=1}^{G} \frac{1}{\lvert o_i \rvert} \sum_{t=1}^{\lvert o_i \rvert} \Big( \min\big(\rho_{i,t} A_i,\ \operatorname{clip}(\rho_{i,t}, 1 - \epsilon, 1 + \epsilon)\, A_i\big) - \lambda_{\text{KL}}\, D_{\mathrm{KL}}\big(\pi_\theta \Vert \pi_{\text{ref}}\big) \Big)\Bigg].
$$

Each piece comes from an earlier part:

- **The clipped term** is PPO's [3] (Part 4), with the group advantage $$A_i$$ in place of a critic-based one. On the first update after sampling every ratio is 1 and the term reduces to the policy gradient.
- **The KL term** is optional and sits in the loss rather than the reward, estimated per token with Part 4's k3 estimator. DeepSeekMath calls its coefficient $$\beta$$; in this series a loss-side KL coefficient is $$\lambda_{\text{KL}}$$. Many recent recipes, and the experiments here, set it to zero.
- **The averaging** first takes the mean over the tokens of each response, $$1/\lvert o_i \rvert$$, and then the mean over the $$G$$ responses. This is the per-sequence mean of [Part 2](/blog/2026/11/sft-probability-reshaping/)'s section on loss normalization: each response gets the same total weight, so each token of a short response counts more than each token of a long one. It is another weighting choice hidden in the objective, alongside the division by $$\sigma_R$$.

Papers write the objective in slightly different forms. The current version of the DeepSeek-R1 paper [2], for example, writes it with a ratio of whole-response probabilities and no $$1/\lvert o_i \rvert$$ factor. The group-relative advantage is the part they share.

## GRPO, PPO and DPO side by side

| | PPO (RLHF) | GRPO | Offline DPO |
|---|---|---|---|
| Data | fresh rollouts from the policy | fresh rollouts, $$G$$ per prompt | a fixed set of preference pairs |
| Signal | reward (model or verifier) | reward (model or verifier) | which of two responses is preferred |
| Learned critic | yes, about the size of the policy | none | none |
| Baseline | critic value $$V(s)$$, with GAE | group mean (and spread) | none needed; the pair is compared directly |
| Old policy and clipping | yes | yes | no |
| Reference and KL | KL penalty, usually in the reward | optional, in the loss ($$\lambda_{\text{KL}}$$) | built into the loss ($$\beta$$) |

GRPO is PPO-style policy optimization with a critic-free group baseline. What separates it from offline DPO ([Part 3](/blog/2026/11/dpo-from-kl-regularized-rl/)) is the loop: GRPO samples new responses from the current policy, verifies them and reinforces the successes, so it can find and amplify a success that no fixed dataset contains. That loop is what gives it more room to explore than offline DPO, within the limits the group size sets.

<div class="note" markdown="1">
<span class="label">In my experiments</span>

The RL arms of [sharpen-or-expand](https://github.com/hieutrungle/sharpen-or-expand) have no results on the public main branch yet. Three design choices fix how the group works in them.

- **The group.** Every 4B RL run uses 64 prompts with $$G = 8$$ responses each per training step (512 rollouts), 4 optimizer updates per step, clip range 0.2 unless Clip-Higher is on, and no KL term or reference model. The algorithms may differ only in their own switches and, in one ablation, in the group size ([shared 4B config](https://github.com/hieutrungle/sharpen-or-expand/blob/main/docs/decisions/025-g0-shared-4b-config.md)).
- **GRPO as implemented.** GRPO is verl's `grpo` advantage estimator with `seq-mean-token-mean` loss aggregation, the per-response token mean followed by the mean over responses, and a symmetric clip. Dr.GRPO [7] is the same with the division by the standard deviation switched off and a different loss aggregation, `seq-mean-token-sum-norm` ([GPU stack decision](https://github.com/hieutrungle/sharpen-or-expand/blob/main/docs/decisions/001-gpu-stack-verl.md)). The advantage uses verl 0.9.1's convention from the note above: Bessel-corrected standard deviation plus $$\delta = 10^{-6}$$.
- **The group-size ablation.** GRPO is also run at $$G$$ = 4, 8 and 16, with shorter runs matched by rollout tokens, and the result is to be reported as newly solved problems per million rollout tokens ([milestone plan](https://github.com/hieutrungle/sharpen-or-expand/blob/main/docs/milestone-plan.md), ablations).

What the starting policy's groups look like. Every RL run starts from the base model ([start decision](https://github.com/hieutrungle/sharpen-or-expand/blob/main/docs/decisions/031-rl-and-dpo-start-from-the-base.md)). A development run that trained from that base logged its rollouts, and its first step, sampled before any update, describes the base's own groups at $$G = 8$$ on the RL training set: of 64 groups, 40.6% were all correct, 21.9% all wrong and 37.5% mixed, and 92.9% of the rollouts in all-wrong groups hit the 16,384-token cap ([group audit](https://github.com/hieutrungle/sharpen-or-expand/blob/main/docs/reports/m2_base_start_audit.md), [data](https://github.com/hieutrungle/sharpen-or-expand/blob/main/docs/reports/m2_rollout_groups.json)). This describes one batch of the starting policy's rollouts, not a result. It says that a large share of the starting batch is zero-variance, and that much of the all-wrong share is responses that ran out of length rather than wrong answers.

Why it matters for the series question. With $$G = 8$$, a problem the base model solves with $$p = 0.01$$ contributes a success in about 7.7% of its groups, and one with $$p = 0.001$$ in fewer than 1%. The group size sets the training-time exploration budget per prompt, so the group-size ablation is the experiment's direct test of "more samples per prompt, more discovery".
</div>

## Check yourself

**1.** A group of $$G = 16$$ has exactly one correct response. What are the advantages, under the population standard deviation and under verl's convention? What is the risk?

<details markdown="1">
<summary>Answer</summary>

With the population standard deviation, the success gets $$\sqrt{15}$$ ≈ 3.87 and each failure $$-1/\sqrt{15}$$ ≈ −0.258. With the Bessel-corrected standard deviation and $$\delta = 10^{-6}$$, as in verl 0.9.1, they are 3.750 and −0.250: every advantage is scaled by $$\sqrt{15/16}$$. Either way the lone success carries the whole positive push, with 15 times the weight of each failure. That is what makes GRPO a rare-success amplifier, and also its risk: one lucky, noisy or verifier-exploiting response gets a very large update.
</details>

**2.** Prompt A's group has 1 correct response in 8, prompt B's has 4 in 8 and prompt C's has 8 in 8. Every correct response has reward 1. Why do they get different advantages?

<details markdown="1">
<summary>Answer</summary>

Because the advantage compares a response with its group, not with a fixed standard. A's success is the surprise in its group and gets $$\sqrt{7}$$ ≈ 2.65; B's successes get 1.00; C's group has no spread, so every advantage is 0 and C contributes nothing. The verifier decides what is correct; the group decides how hard each prompt pushes, and that changes as the policy's success probability on the prompt changes.
</details>

**3.** A problem has $$p = 0.02$$ under the starting model. What is the probability that a group of 64 contains a success? If GRPO later drives $$p$$ from 0.02 to 0.50, how would you describe the change?

<details markdown="1">
<summary>Answer</summary>

$$1 - 0.98^{64}$$ ≈ 72.6%. Successful behavior was already measurably reachable before training, so the change is consistent with sharpening: the positive-feedback loop amplified a mode the model already had. On Part 1's ladder of evidence a rising pass@1 is the first rung, which sharpening produces just as well. Claiming expansion would need evidence from higher rungs, such as problems unsolved at a large sampling budget becoming repeatedly solvable, or new strategy families.
</details>

**4.** Why is $$R_i - \bar R$$ not the same as having a perfect critic, and what changes when GRPO divides by $$\sigma_R$$?

<details markdown="1">
<summary>Answer</summary>

The group mean is a noisy estimate from $$G$$ samples, and it includes the response it is compared with. The second effect is exact and mild: with mean-centering only, $$\mathbb{E}[(R_i - \bar R)\nabla_\theta \log \pi_\theta(o_i)] = \frac{G - 1}{G}\, \mathbb{E}[R_i\, \nabla_\theta \log \pi_\theta(o_i)]$$, a rescaled policy gradient, equal to RLOO up to a constant factor. Dividing by $$\sigma_R$$ breaks this, because $$\sigma_R$$ depends on the response's own reward. For a 0/1 reward, each prompt's expected update becomes $$w_G(p)\, \nabla_\theta p$$ with a weight that depends on $$p$$, so across prompts GRPO climbs a transformed objective, $$2 \arcsin\sqrt{p}$$ per prompt in the large-$$G$$ limit, rather than the average success probability.
</details>

## Takeaways

- GRPO replaces PPO's learned critic with the statistics of $$G$$ responses to the same prompt: $$A_i = (R_i - \bar R)/\sigma_R$$, one number for every token of response $$i$$.
- The group mean estimates the prompt's value $$p_\theta(x)$$ and includes the sample itself; mean-centering alone gives $$(G - 1)/G$$ times the policy gradient, RLOO up to scale.
- For 0/1 rewards with $$k$$ of $$G$$ correct, $$A_+ = \sqrt{(G - k)/k}$$ and $$A_- = -\sqrt{k/(G - k)}$$ with the population standard deviation; a lone success in 8 gets $$\sqrt{7}$$ ≈ 2.65. The Bessel-corrected convention of verl scales every advantage by $$\sqrt{(G - 1)/G}$$.
- Dividing by $$\sigma_R$$ reweights prompts by their success probability. In the large-group limit GRPO climbs $$2\arcsin\sqrt{p}$$ per prompt; at $$G = 8$$ it is close to that only for $$p$$ between 0.30 and 0.70.
- Reward is not advantage: the same correct response is worth more when it is rare in its group, and difficulty moves with the policy.
- All-wrong and all-correct groups give zero advantage. GRPO learns only from mixed groups, which occur with probability $$1 - p^G - (1 - p)^G$$.
- The group size is a training-time exploration budget: a prompt contributes a success with probability $$1 - (1 - p)^G$$. GRPO amplifies a rare success once it is sampled, and that is consistent with sharpening when the success was already reachable.

## Conclusion

The opening animation is GRPO in miniature. Nothing happened while every group failed. The first success, alone in its group, got the largest advantage the group could give, and the positive-feedback loop took over. As the strategy became common, the same reward earned less and less, until all-correct groups stopped the update with some probability still on wrong answers. This part derived each of those behaviors. The group mean is a critic substitute equal to RLOO up to scale. Dividing by the group's standard deviation turns the objective into a per-prompt reweighting of the success probability, the arcsine of its square root in the large-group limit. And the group size decides how rare a success can be and still be found.

For the question of this series, GRPO inherits the policy gradient's strength and its blind spot, and makes both sharper. It can amplify a success that the starting model produces only rarely, and it rewards such a success more than a common one. But it gets nothing from prompts whose groups are all wrong, and the group size bounds how rare a success it can find. When the success it amplifies was already reachable, the result is consistent with sharpening. Whether GRPO in a real model also makes problems solvable that were not reachable before is a hypothesis to test, and the group-size ablation in the experiments is one direct way to test it.

## Next

[Part 6](/blog/2026/11/why-vanilla-grpo-fails/) asks why vanilla GRPO fails. It maps the failure modes that the choices of this part create: zero-variance groups that waste samples, the per-prompt weighting from the standard deviation, the length weighting from the per-response average, and what they do to training.

## References

1. Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. K. Li, Y. Wu, and D. Guo, "DeepSeekMath: Pushing the limits of mathematical reasoning in open language models," arXiv:2402.03300, 2024. [arXiv](https://arxiv.org/abs/2402.03300)
2. DeepSeek-AI (D. Guo, D. Yang, H. Zhang, et al.), "DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning," *Nature*, vol. 645, pp. 633–638, 2025. [DOI](https://doi.org/10.1038/s41586-025-09422-z) · [arXiv](https://arxiv.org/abs/2501.12948) (v2, "DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning")
3. J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal policy optimization algorithms," arXiv:1707.06347, 2017. [arXiv](https://arxiv.org/abs/1707.06347)
4. A. Ahmadian, C. Cremer, M. Gallé, M. Fadaee, J. Kreutzer, O. Pietquin, A. Üstün, and S. Hooker, "Back to basics: Revisiting REINFORCE-style optimization for learning from human feedback in LLMs," in *Proc. ACL*, 2024. [DOI](https://doi.org/10.18653/v1/2024.acl-long.662) · [arXiv](https://arxiv.org/abs/2402.14740)
5. D. Davis and B. Recht, "What is the objective of reasoning with reinforcement learning?," arXiv:2510.13651, 2025. [arXiv](https://arxiv.org/abs/2510.13651)
6. verl contributors, verl v0.9.1, `compute_grpo_outcome_advantage` in `verl/trainer/ppo/core_algos.py`, 2026. [Source](https://github.com/verl-project/verl/blob/v0.9.1/verl/trainer/ppo/core_algos.py#L266-L331)
7. Z. Liu, C. Chen, W. Li, P. Qi, T. Pang, C. Du, W. S. Lee, and M. Lin, "Understanding R1-Zero-like training: A critical perspective," arXiv:2503.20783, 2025. [arXiv](https://arxiv.org/abs/2503.20783)
