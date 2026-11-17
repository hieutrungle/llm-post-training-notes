---
title: "Policy Gradients for LLMs: REINFORCE to PPO"
kind: Study notes
date: 2026-11-17 14:00:00 -0600
series: llm-post-training
part: 4
summary: "Policy-gradient methods learn from the model's own samples: sample, score with a reward, reinforce. Part 4 derives the policy gradient from first principles, shows why it is noisy and how a baseline helps, builds up to PPO's ratios and clipping, and explains why on-policy RL can amplify a rare success but cannot reinforce one it never samples."
math: true
anim: true
---

*Code, check scripts and references: [llm-post-training-notes](https://github.com/hieutrungle/llm-post-training-notes) · Experiments: [sharpen-or-expand](https://github.com/hieutrungle/sharpen-or-expand)*

<div class="goal" markdown="1">
<span class="label">Core idea</span>

A policy gradient learns from the model's own samples. It raises the probability of what was sampled and scored above a baseline, lowers what was scored below it, and changes everything else only through normalization. Baselines reduce the noise without changing the expected direction; PPO's importance ratios and clipping let one batch of samples serve several updates without moving too far. None of this can reinforce a success the model never samples.
</div>

## The question

[Part 3](/blog/2026/11/dpo-from-kl-regularized-rl/) showed that offline DPO learns from a fixed set of pairs and has no loop that samples a new response, finds out that it works and reinforces it. This part is about that loop. How do we learn from the model's own samples and a reward that scores them, and what does that do to the distribution over responses?

## The picture to keep in mind

Before any derivation, here is the loop at work on two problems. Each round samples from the current policy, a verifier scores the samples, and one gradient step pushes each sampled strategy up or down.

<div class="anim" data-anim="part-04-pg-loop" data-alt="Animation of eight rounds of policy-gradient training on two problems, each a bar chart of solution strategies. Easy problem: a common correct strategy A at 0.30, a rare correct strategy B at 0.02, a correct strategy C with probability 0, and three wrong strategies. Hard problem: a rare correct strategy at 0.02, a correct strategy with probability 0, and three wrong strategies. Each round, eight sample dots land on the strategies they picked, correct ones in the accent colour; arrows push sampled correct strategies up and sampled wrong ones down. On the easy problem A grows to 0.941 and the success probability to 0.947, while B, sampled once in round 1, rises to 0.024 and then falls to 0.005. On the hard problem nothing moves for three rounds; the rare strategy is first sampled in round 4 and then grows to 0.593. The strategies with probability 0 stay at 0."><noscript><img src="{{ '/assets/img/llm-post-training/part-04-pg-loop.png' | relative_url }}" alt="Final frame after eight rounds: on the easy problem A is at 0.941, B at 0.005, C at 0, success probability 0.947; on the hard problem the rare correct strategy is at 0.593, the strategy with probability 0 still at 0, success probability 0.593."></noscript></div>

The animation shows two problems. On each, the policy spreads its probability over a handful of solution strategies, drawn as bars; a check mark labels the correct ones. The easy problem has a common correct strategy A (probability 0.30), a rare correct strategy B (0.02), a correct strategy C that the policy never produces, and three wrong strategies. On the hard problem the only reachable correct strategy is rare (0.02), another correct one has probability zero, and everything else is wrong. Each round draws 8 samples per problem. A sample that is correct gets reward 1 and a wrong one 0, and the update pushes each sampled strategy by its *advantage*: the reward minus the problem's current success probability $$p$$. Every strategy has its own logit, each update uses step size 3, and the samples come from a seeded random generator over 8 rounds. The run is simulated and the numbers are illustrative; the [checks script](https://github.com/hieutrungle/llm-post-training-notes/blob/main/checks/part04_checks.py) runs the same simulation.

Read the animation:

- **The easy problem sharpens.** The success probability goes from 0.32 to 0.947, and A from 0.30 to 0.941. B is sampled once, in round 1, and rises from 0.02 to 0.024. After that A is sampled almost every time, and B falls to 0.005 through normalization. B's share of the correct mass drops from 6.25% to 0.6%, and the effective number of correct strategies (Part 1's $$N_{\text{eff}}$$) from 1.26 to 1.04.
- **The hard problem waits, then takes off.** For three rounds every sample fails. Each failure gets advantage −0.02, so almost nothing moves; the rare strategy is still at 0.020 after round 3. In round 4 it is sampled once, with advantage 0.98. It rises to 0.033, comes up more often in later rounds, and ends at 0.593.
- **What is never sampled is never reinforced.** The two correct strategies with probability 0 stay at 0 on both problems.

That is the whole method in one sentence: **sample from the current policy, score the samples, and push each sampled response up or down according to whether it scored above or below what was expected.** The rest of this part derives that update, explains why it is noisy and how a baseline helps, and builds up to the importance ratios and clipping that PPO adds so that one batch of samples can serve several updates.

## The RL objective

A policy $$\pi_\theta$$ generates a trajectory $$\tau = (y_1, \ldots, y_T)$$ for a prompt $$x$$ token by token, with $$\pi_\theta(\tau \mid x) = \prod_t \pi_\theta(y_t \mid x, y_{\lt t})$$. A reward $$R(x, \tau)$$ scores the finished trajectory. Reinforcement learning maximizes the expected reward under the policy's own samples:

$$
J(\theta) = \mathbb{E}_{x \sim \mathcal{D}}\, \mathbb{E}_{\tau \sim \pi_\theta(\cdot \mid x)}\big[R(x, \tau)\big].
$$

With a 0/1 verifier reward, the inner expectation for one problem is the probability that a sample is correct: $$\sum_{\tau \in \mathcal{S}(x)} \pi_\theta(\tau \mid x) = p_\theta(x)$$, Part 1's success probability. Maximizing $$J$$ means raising pass@1, averaged over problems.

The data are what is new. [Part 2](/blog/2026/11/sft-probability-reshaping/)'s supervised fine-tuning imitates demonstrations someone else wrote. Part 3's DPO fits preferences on a fixed set of pairs. Here the training data are generated by the policy being trained, and they change every time the policy does. The reward does not say what a good response looks like; it only scores the responses the policy happened to produce.

## The log-derivative trick

Write the objective for one prompt as a sum over every possible trajectory (drop $$x$$ from the notation):

$$
J(\theta) = \sum_\tau \pi_\theta(\tau)\, R(\tau).
$$

The sum has more terms than could ever be enumerated, so it has to be estimated by sampling. That creates the difficulty. A sample is a discrete choice of tokens, and which token gets drawn does not change smoothly with $$\theta$$: a small change in the parameters either leaves the sampled token the same or flips it to another one. There is no useful derivative of the sample itself to backpropagate through. The reward is a black box too: a verifier that checks a final answer has no gradient.

The way out is to differentiate the probabilities rather than the samples. The reward does not depend on $$\theta$$, so

$$
\nabla_\theta J = \sum_\tau \nabla_\theta \pi_\theta(\tau)\, R(\tau).
$$

This is not yet an expectation under $$\pi_\theta$$, so it cannot be estimated by sampling. The identity $$\nabla_\theta \pi_\theta = \pi_\theta \nabla_\theta \log \pi_\theta$$ (the derivative of the logarithm, read backwards) makes it one:

$$
\nabla_\theta J = \sum_\tau \pi_\theta(\tau)\, R(\tau)\, \nabla_\theta \log \pi_\theta(\tau) = \mathbb{E}_{\tau \sim \pi_\theta}\big[R(\tau)\, \nabla_\theta \log \pi_\theta(\tau)\big].
$$

This is the *policy gradient*. Sample trajectories from the current policy, score them, and average $$R(\tau_i)\nabla_\theta\log\pi_\theta(\tau_i)$$: the average is an unbiased estimate of $$\nabla_\theta J$$. The derivative now acts only on the log-probability of what was sampled, which the model computes anyway. Williams [1] introduced this estimator as REINFORCE, and Sutton et al. [2] proved the more general policy gradient theorem for sequential decision problems.

For an autoregressive model the log-probability of a trajectory is a sum over its tokens, so

$$
\nabla_\theta J = \mathbb{E}_{\tau \sim \pi_\theta}\Big[R(\tau) \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(y_t \mid x, y_{\lt t})\Big].
$$

Each token's gradient is the gradient of the cross-entropy loss of Part 2 for that token, with the sign flipped and a weight $$R(\tau)$$. A REINFORCE step with a 0/1 reward is a supervised step on the model's own correct samples, and does nothing on the incorrect ones. The checks script confirms the identity on a small autoregressive model (3 tokens, length 3, 27 sequences): the exact expectation matches finite differences of $$J$$.

## The same direction as self-training on successes

Part 2 derived a related identity. For a 0/1 reward, let $$q(\tau \mid x) = \pi_\theta(\tau \mid x)/p_\theta(x)$$ on $$\mathcal{S}(x)$$ be the distribution of the model's own successful trajectories. Then $$\nabla_\theta \log p_\theta(x) = \mathbb{E}_{\tau \sim q}[\nabla_\theta \log \pi_\theta(\tau \mid x)]$$. Since $$J$$ for one problem is $$p_\theta(x)$$, and $$\nabla p = p\, \nabla \log p$$,

$$
\nabla_\theta J = p_\theta(x)\, \mathbb{E}_{\tau \sim q(\cdot \mid x)}\big[\nabla_\theta \log \pi_\theta(\tau \mid x)\big].
$$

The checks script confirms this on the same small model. REINFORCE with a 0/1 reward points in the same direction, problem by problem, as fine-tuning on the model's own correct samples, and the factor $$p_\theta(x)$$ sets how much each problem counts. Summed over problems, self-training that averages over each problem's successes climbs $$\sum_x \log p_\theta(x)$$ and gives every solved problem equal weight; REINFORCE climbs $$\sum_x p_\theta(x)$$ and gives problems the model rarely solves little weight. Both need the samples to come from the current model, and both get nothing from a problem on which no sample succeeds.

The identity also explains the easy problem in the opening animation. With one logit $$z_k$$ per strategy, $$\partial J / \partial z_k = \pi_k (R_k - p)$$ (the checks script confirms it): every correct strategy is pushed up in proportion to its current probability. At the start A is pushed 15 times harder than B. The expected update therefore raises the log-odds of A against B, and the mix of correct strategies narrows even before sampling luck plays a role. This is a contrast with Part 3, where the KL-regularized optimum multiplied every correct response by the same factor and left the mix unchanged.

## One reward, many tokens

The token form of the gradient shows the credit-assignment problem directly. Every token of a trajectory gets the same weight $$R(\tau)$$. A 2,000-token correct solution pushes up the token that contained the key step and the 1,999 that did not, all equally. This is the same problem as Part 3's "one label, many tokens", with one difference: DPO compares two responses, so shared prefixes cancel, while here each trajectory is pushed on its own.

The usual refinement in RL does not help with a reward given only at the end. In a task with rewards along the way, each action is weighted by the reward collected *after* it, because an action cannot affect what came before. With a single terminal reward, the reward collected after every token is $$R(\tau)$$, so nothing changes. Over many samples, tokens that consistently appear in correct solutions get pushed up more often than tokens that appear in both, so the signal can average out in their favor. A single trajectory does not localize it.

## Why REINFORCE is noisy

The estimate is unbiased but can be very noisy. Take a problem with $$p = 0.01$$ and draw 100 samples. The number of successes is binomial:

| Successes in 100 samples | 0 | 1 | 2 | 3 or more |
|---|---|---|---|---|
| Probability | 0.366 | 0.370 | 0.185 | 0.079 |

To see what that does to the gradient, lump all correct trajectories into one outcome and all incorrect ones into another, with a single logit $$\theta$$ and $$p = \sigma(\theta)$$. The true gradient of $$J$$ is $$p(1 - p)$$. The REINFORCE estimate from $$N$$ samples with $$k$$ successes is $$(k/N)(1 - p)$$, which is $$k/(Np)$$ times the true value: at $$N = 100$$ and $$p = 0.01$$, exactly $$k$$ times. So the estimate is exactly zero in 37% of batches, exactly right in another 37%, twice too large in 18%, and at least three times too large in 8%. In a real model the single successful trajectory also carries every incidental feature of that one sample into the update. The gradient of this problem is mostly zero, and occasionally a large push in the direction of one sample.

## Baselines

Subtract a number $$b$$ from the reward:

$$
\nabla_\theta J = \mathbb{E}_{\tau \sim \pi_\theta}\big[(R(\tau) - b)\, \nabla_\theta \log \pi_\theta(\tau)\big].
$$

This is still the exact gradient, because the subtracted term has expectation zero:

$$
\mathbb{E}_{\tau \sim \pi_\theta}\big[b\, \nabla_\theta \log \pi_\theta(\tau)\big] = b \sum_\tau \pi_\theta(\tau)\, \nabla_\theta \log \pi_\theta(\tau) = b \sum_\tau \nabla_\theta \pi_\theta(\tau) = b\, \nabla_\theta \sum_\tau \pi_\theta(\tau) = b\, \nabla_\theta 1 = 0.
$$

The proof pulls $$b$$ out of the sum, so it needs $$b$$ not to depend on the sampled trajectory. Precisely: a sequence-level baseline may depend on the prompt $$x$$; a token-level baseline for token $$t$$ may also depend on the prefix $$x, y_{\lt t}$$, because the same argument applies to the distribution of $$y_t$$ given that prefix. A baseline must not depend on the token it multiplies, on later tokens, or on the trajectory's own reward. The extreme case shows why: with $$b = R(\tau)$$ every advantage is zero and nothing is learned.

The rule matters for the most common practical baseline, the mean reward of a group of $$G$$ samples for the same prompt. The mean includes the sample itself, so it depends on that sample's reward. The bias is mild and exact: the mean-centered estimate has expectation $$(G - 1)/G$$ times the true gradient, a rescaled version of it pointing in the same direction (0.875 at $$G = 8$$). Leaving the sample out, and using the mean of the other $$G - 1$$ rewards as its baseline, gives an exactly unbiased estimate. This is the leave-one-out estimator (RLOO) that Ahmadian et al. [3] use for language models. Its advantages are $$G/(G - 1)$$ times the mean-centered ones, so the two differ only in scale. The checks script confirms both statements by enumerating every group of 4 samples on small examples.

A baseline changes how the update treats failures. With rewards 1 and 0 and no baseline, the success gets weight 1 and the failure weight 0: nothing pushes the failure down directly, and it loses probability only through normalization. With baseline 0.5, the success gets +0.5 and the failure −0.5. The signal becomes "better or worse than expected", which is how the opening animation, with $$b = p$$, pushes sampled failures down.

What a baseline does to the noise depends on the problem.

![Line plot of the relative noise of a gradient estimate from 8 samples, on a log axis from 0.3 to 30, against a constant baseline b from −0.5 to 1.5, for the two problems of the opening animation. The easy problem, with p = 0.32, has a U-shaped curve with its minimum of 0.45 near b = 0.30; open circle at b = 0 at 0.59, filled diamond at b = p at 0.45, and the curve crosses the dotted line at relative noise 1 near b = −0.45 and b = 1.0. The hard problem, with p = 0.02, lies far above: its minimum is 2.42 near b = 0.04, the b = 0 and b = p markers sit almost on top of each other at 2.47 and 2.44, and the curve rises steeply on both sides.]({{ '/assets/img/llm-post-training/part-04-baseline.png' | relative_url }})

The plot shows how noisy the gradient estimate from 8 samples is, as a function of a constant baseline $$b$$, for the two problems of the opening animation. Relative noise is the typical size of the estimation error divided by the size of the true gradient, both measured over all strategy logits and computed exactly; at 1 the noise is as large as the signal. The mean of the estimate is the true gradient for every $$b$$, so the curves show only what the baseline does to the spread. Open circles mark $$b = 0$$, no baseline, and diamonds $$b = p$$.

Read the curves:

- **Easy problem.** Without a baseline the relative noise is 0.59; with $$b = p$$ it falls to 0.45. The best constant baseline is 0.30, essentially the success probability. A reward offset is a baseline too: rewards of 10 and 11 instead of 0 and 1 act like $$b$$ = −10, and the relative noise rises to 12.9.
- **Hard problem.** The noise is 2.47 without a baseline, 2.44 with $$b = p$$, and 2.42 at the best constant, about 0.04. No choice of baseline brings the noise near the signal.

The takeaway: a baseline removes the noise that comes from the reward's offset, which grows quickly when the rewards are not centered. It cannot remove the noise that comes from rarity. When successes are rare, the estimate hinges on whether one appears in the batch at all, and only more samples or a higher success probability change that.

## The advantage

The quantity that multiplies the log-probability gradient is the *advantage*, $$A = R - b$$. Its sign sets the direction: $$A \gt 0$$ raises the sampled response, $$A \lt 0$$ lowers it, and $$A \approx 0$$ leaves it nearly alone.

In standard RL notation the ideal baseline is a value function. For a state $$s$$ (here, the prompt and the prefix so far) and an action $$a$$ (the next token),

$$
A^{\pi}(s, a) = Q^{\pi}(s, a) - V^{\pi}(s),
$$

where $$Q^\pi(s, a)$$ is the expected final reward after taking $$a$$ in $$s$$ and then following $$\pi$$, and $$V^\pi(s)$$ is the expected final reward from $$s$$. The advantage asks how much better this token is than the policy's average behavior from this prefix. The opening animation uses the exact value of the prompt, $$b = V^\pi(x) = p$$, which a real system does not know. Actor-critic methods train a second network, the *critic*, to estimate $$V^\pi$$, and generalized advantage estimation (GAE) [4] combines the critic's estimates over several steps to trade bias against variance. PPO-based RLHF for language models uses such a critic, typically a network of about the same size as the policy [9].

## Three policies

Sampling a batch of long responses is the expensive part of training. To get more out of each batch, practical methods split it into mini-batches and take several optimizer steps on it. After the first step, the policy being trained is no longer the one that generated the samples. Three policies are now in play.

- $$\pi_\theta$$, the policy being trained.
- $$\pi_{\text{old}}$$, the policy that generated the current batch. It is replaced by the current $$\pi_\theta$$ every time a new batch is sampled.
- $$\pi_{\text{ref}}$$, a frozen reference, usually the starting model. In RLHF it is often the supervised fine-tuned model. It appears only in a KL penalty, and many recent recipes have none.

Samples from $$\pi_{\text{old}}$$ can still estimate expectations under $$\pi_\theta$$ by reweighting them, $$\mathbb{E}_{\pi_\theta}[f] = \mathbb{E}_{\pi_{\text{old}}}[(\pi_\theta/\pi_{\text{old}})\, f]$$. PPO applies this token by token, with the per-token importance ratio

$$
\rho_{i,t} = \frac{\pi_\theta(y_{i,t} \mid x, y_{i,\lt t})}{\pi_{\text{old}}(y_{i,t} \mid x, y_{i,\lt t})}
$$

for token $$t$$ of response $$i$$. (The PPO paper writes the ratio as $$r_t(\theta)$$; this series uses $$\rho$$ because $$r$$ is already the reward.) The surrogate objective $$\mathbb{E}_{\pi_{\text{old}}}[\sum_t \rho_{i,t} A_i]$$ has exactly the policy gradient as its gradient at $$\pi_\theta = \pi_{\text{old}}$$, since there $$\nabla \rho = \nabla \log \pi_\theta$$. Away from that point it is an approximation. The ratio corrects the choice of each token given its prefix, but not the fact that a changed policy would also produce different prefixes. Trust-region policy optimization (TRPO) [5] uses this surrogate and bounds how far the policy may move so that the approximation stays good.

The two old-versus-reference roles are easy to confuse. $$\pi_{\text{old}}$$ controls the stability of each update: the ratio measures how far the current mini-batch has moved from the policy that sampled the data, and it resets every batch. $$\pi_{\text{ref}}$$ controls long-term drift: a KL penalty to it charges for every move away from the starting model, accumulated over the whole run.

## Why the ratio is per token

The importance weight of a whole trajectory is the product of its token ratios:

$$
\frac{\pi_\theta(\tau \mid x)}{\pi_{\text{old}}(\tau \mid x)} = \prod_{t=1}^{T} \rho_t.
$$

Small per-token changes compound. If every token of a 500-token response is 2% more likely than before, the trajectory weight is $$1.02^{500}$$ = 19,956.57, about 19,957; if every token is 2% less likely, $$0.98^{500}$$ = 0.000041. A batch weighted this way is dominated by one or two samples. Per-token ratios stay near 1 whenever each token's probability moved a little, which is why PPO-style methods use them, at the price of the approximation described above.

## PPO's clipped objective

Proximal policy optimization (PPO) [6] keeps each update close to $$\pi_{\text{old}}$$ by clipping the ratio inside the objective. For one token with advantage $$A$$,

$$
L^{\text{CLIP}} = \min\big(\rho A,\ \operatorname{clip}(\rho, 1 - \epsilon, 1 + \epsilon)\, A\big),
$$

averaged over the tokens of the batch; $$\epsilon = 0.2$$ is the common choice. The minimum of the clipped and unclipped terms makes the objective a pessimistic estimate. Working through the cases:

- **$$A \gt 0$$, $$\rho \gt 1 + \epsilon$$:** the clipped term $$(1 + \epsilon)A$$ is the smaller one. The objective is flat in $$\rho$$, so the token gets no gradient: the update has already raised it enough.
- **$$A \gt 0$$, $$\rho \lt 1 - \epsilon$$:** the unclipped term $$\rho A$$ is the smaller one, so the gradient is the full $$A\, \nabla \rho$$. The token was pushed the wrong way, and nothing stops the objective from undoing that.
- **$$A \lt 0$$, $$\rho \lt 1 - \epsilon$$:** the objective is flat at $$(1 - \epsilon)A$$: the token has been lowered enough.
- **$$A \lt 0$$, $$\rho \gt 1 + \epsilon$$:** the unclipped term $$\rho A$$ is the smaller one, and the full gradient pushes the token back down.
- **Inside $$[1 - \epsilon, 1 + \epsilon]$$** the objective is $$\rho A$$ for either sign.

With $$\epsilon = 0.2$$: at $$\rho = 1.3$$ and $$A = +1$$ the objective is 1.2 and the gradient zero; at $$\rho = 1.3$$ and $$A = -1$$ it is −1.3 with the full gradient; at $$\rho = 0.7$$ it is −0.8 and flat for $$A = -1$$, and 0.7 with the full gradient for $$A = +1$$.

![Two panels plotting the per-token objective against the ratio rho from 0 to 2, with dotted lines at 0.8 and 1.2. Left, A greater than 0: the clipped objective rises along rho times A up to 1.2 at rho = 1.2 and is flat at 1.2 after that, while the dashed unclipped line keeps rising to 2; the region above 1.2 is shaded as having no gradient, and a dot marks rho = 1 at objective 1. Right, A less than 0: the clipped objective is flat at −0.8 for rho below 0.8, in a shaded no-gradient region, and follows rho times A, falling to −2 at rho = 2, above that; a dot marks rho = 1 at −1.]({{ '/assets/img/llm-post-training/part-04-clip.png' | relative_url }})

The figure plots the clipped objective for one token against its ratio, with $$\epsilon = 0.2$$: on the left for a token with positive advantage, on the right for one with negative advantage. The dashed line is the unclipped surrogate $$\rho A$$, and the shaded regions are where the objective is flat and the token gets no gradient.

Read the curves:

- **Left ($$A \gt 0$$).** The objective rises with $$\rho$$ up to 1.2 and is flat after that. Below 0.8 it still follows $$\rho A$$: clipping never stops the objective from undoing a move in the wrong direction.
- **Right ($$A \lt 0$$).** The mirror image: flat below 0.8, and following $$\rho A$$ above 1.2.
- **The dot at $$\rho = 1$$** is where every token starts when the update uses fresh samples from the current policy. It lies inside the unclipped region in both panels.

The takeaway: clipping removes the incentive to move a token further in the direction its advantage asks for once its ratio has left $$[1 - \epsilon, 1 + \epsilon]$$. It never removes the incentive to undo a move in the wrong direction.

Two consequences are easy to miss.

**Clipping limits the incentive, not the ratio.** The objective is clipped; the parameters are not. A token whose ratio is past the bound gets no gradient of its own, but updates on other tokens move it anyway when they share parameters. The checks script builds the smallest example: one context with three tokens, $$\pi_{\text{old}} = (0.5, 0.3, 0.2)$$, and a current policy $$(0.65, 0.27, 0.08)$$. Token $$a$$ was sampled with $$A = +1$$ and has ratio 1.30, past the bound, so its term contributes nothing. Token $$b$$ was sampled with $$A = -1$$ and ratio 0.9, inside the range. One step on token $$b$$'s term lowers $$b$$, normalization raises $$a$$, and $$a$$'s ratio moves from 1.30 to 1.52. In a network, every token shares parameters with every other token in the batch, so this happens throughout.

**Clipping acts only when $$\pi_\theta \ne \pi_{\text{old}}$$.** On the first update after sampling, $$\pi_\theta = \pi_{\text{old}}$$, every ratio is exactly 1, and the gradient of the clipped objective is the plain policy gradient $$A\, \nabla \log \pi_\theta$$ (the checks script confirms this). Clipping and $$\epsilon$$ matter only on later updates that reuse the same batch. A method that makes one update per batch is on-policy policy gradient, whatever its loss is called. That was the setting of the original GRPO experiments in DeepSeekMath [9], where the policy model has only a single update following each exploration stage.

## Where the KL term goes

Part 3 maximized reward minus $$\beta$$ times the KL divergence from $$\pi_{\text{ref}}$$. In policy-gradient training the KL term appears in one of two places.

- **In the reward.** Ziegler et al. [7] train against the modified reward $$R(x, y) = r(x, y) - \beta \log\big(\pi_\theta(y \mid x)/\pi_{\text{ref}}(y \mid x)\big)$$, and InstructGPT [8] adds the same per-token KL penalty from the supervised model. The policy gradient then maximizes exactly Part 3's objective, whose optimum is the Gibbs policy $$\pi_{\text{ref}}\, e^{r/\beta}/Z$$.
- **In the loss.** GRPO [9] keeps the reward clean and adds $$\lambda_{\text{KL}}\, D_{\mathrm{KL}}(\pi_\theta \Vert \pi_{\text{ref}})$$ to the loss, estimated per token with $$\frac{\pi_{\text{ref}}}{\pi_\theta} - \log\frac{\pi_{\text{ref}}}{\pi_\theta} - 1$$. This estimator (often called k3) is unbiased for the KL divergence and never negative; the checks script confirms both. DeepSeekMath sets the coefficient to 0.04 (their paper calls it $$\beta$$; in this series it is $$\lambda_{\text{KL}}$$, because it multiplies a loss term rather than sitting in the reward).

Many recent recipes for reasoning drop the KL term entirely. DAPO [10] removes it, arguing that a long chain-of-thought model is expected to move far from its starting point, so the restriction is not necessary. Dr. GRPO [11] sets it to zero too: in RLHF the penalty keeps the policy where a learned reward model is accurate, and a rule-based verifier does not have that problem.

This is the stability–exploration tension. A KL anchor keeps training stable, preserves the broad behavior of the starting model and limits how far a policy can exploit a flawed reward. It also charges for moving probability toward responses the reference finds unlikely, which are exactly the moves that expansion would require: Part 3 showed that with the KL term in the objective, the optimum is a reweighting of $$\pi_{\text{ref}}$$. Without it, nothing in the objective keeps the policy near its start; only the optimizer does, through the learning rate, the number of updates per batch and clipping.

## Entropy

The entropy of the next-token distribution at a prefix $$s$$ is $$H(\pi(\cdot \mid s)) = -\sum_a \pi(a \mid s) \log \pi(a \mid s)$$. High entropy means the model is spreading its bets. PPO's objective can include an entropy bonus, a small multiple of $$H$$ added to the objective to keep exploration going [6].

Token entropy is not reasoning diversity. Take two models with one correct answer. Model A has one strategy but writes it with 20 independent choices between two equivalent phrasings: 1,048,576 equally likely trajectories, 13.9 nats of trajectory entropy, and $$N_{\text{eff}} = 1$$ strategy. Model B has four strategies, each written one way: 1.39 nats and $$N_{\text{eff}} = 4$$. Model A has 10 times the entropy and a quarter of the strategies. An entropy bonus rewards variation in wording as much as variation in method, and Part 1's $$N_{\text{eff}}$$ counts only the second.

Entropy still says something about training dynamics. Cui et al. [12] observe that in RL for reasoning models, without entropy intervention, policy entropy falls sharply early in training while performance saturates, and they fit validation performance as $$R = -a\, e^{H} + b$$. Their analysis of softmax policies explains the direction: under policy-gradient-like updates, entropy falls when high-probability actions get high advantage, and rises when rare actions do. The opening animation shows both cases. On the easy problem, where the common strategy is reinforced, the entropy over the six strategies falls from 1.41 to 0.30 nats. On the hard problem, where a rare strategy is reinforced, it rises from 1.09 to 1.36 by round 6, and only then falls, to 1.11, as that strategy becomes the common one. Cui et al. also report that a plain entropy loss is highly sensitive to its coefficient, and propose restricting the updates of the high-covariance tokens instead.

## Sparse rewards: what on-policy RL cannot see

Exploitation raises the probability of what already works; exploration keeps enough probability on alternatives to find something new. The policy gradient does exploitation by construction. Its only exploration is the randomness of sampling, and that is the crux of the sharpen-or-expand question.

Suppose a problem has $$p = 10^{-4}$$. One sample in 10,000 succeeds. A group of 64 samples contains a success with probability 0.0064, and even 1,024 samples find one only with probability 0.097. Almost every batch on this problem has reward 0 for every sample. Without a baseline, or with a group-mean baseline, every weight is then 0 and the problem contributes nothing; with a positive baseline such as $$b = p$$, the update can only push the sampled failures down and spread their probability over everything else.

The opening animation's hard problem is the favorable version of this. At $$p = 0.02$$ and 8 samples per round, the rare strategy appears in a given round with probability 0.15. Once it does, the policy gradient amplifies it, and every success makes the next one more likely. At $$p = 10^{-4}$$ the wait is far longer, and at $$p = 0$$ it never ends. **On-policy RL can amplify rare successes, but it cannot reinforce a success it never samples.**

Two things can change a problem's $$p$$ without sampling a success on it. Parameters are shared across problems, so learning on problems the model can solve may raise $$p$$ on ones it cannot. And the sampler matters: a higher temperature spreads probability toward unlikely continuations. Whether these effects are large enough to make previously unreachable problems solvable is an empirical question, and it is the question this series is built around.

<div class="note" markdown="1">
<span class="label">In my experiments</span>

The RL arms of [sharpen-or-expand](https://github.com/hieutrungle/sharpen-or-expand) have no results on the public main branch yet. What is fixed is the design, and three choices matter for this part.

- **No KL term and no reference model.** None of the RL algorithms (GRPO, Dr.GRPO, DAPO, RS-GRPO) has a KL penalty, in the reward or in the loss, and none keeps a reference model ([KL decision](https://github.com/hieutrungle/sharpen-or-expand/blob/main/docs/decisions/013-rl-no-kl-step-checkpoints.md)). Adding KL to GRPO alone would have made one more difference between the algorithms.
- **4 optimizer updates per batch of rollouts.** Every algorithm splits each training step's rollouts into four mini-batches and takes one optimizer step on each ([updates decision](https://github.com/hieutrungle/sharpen-or-expand/blob/main/docs/decisions/022-rl-updates-per-step.md)). On the first mini-batch $$\pi_\theta = \pi_{\text{old}}$$ and every ratio is exactly 1; the later ones are up to 3 updates stale, and only there can clipping and DAPO's asymmetric clip range act. Before this decision each step made a single update. A development smoke run from that time logged a clip fraction and an approximate KL to $$\pi_{\text{old}}$$ of exactly 0 on all three of its steps ([metrics check](https://github.com/hieutrungle/sharpen-or-expand/blob/main/docs/reports/m1_metrics_check.json), which notes that the run started before the decision): with one update per batch, clipping never acts.
- **The sampler is the policy.** Rollouts sample at temperature 1.0 and top-p 1.0 with top-k off, so the samples come from $$\pi_{\text{old}}$$ exactly and the importance ratio has no sampler-versus-policy mismatch beyond numerics; evaluation still samples at temperature 0.6 and top-p 0.95 ([sampling decision](https://github.com/hieutrungle/sharpen-or-expand/blob/main/docs/decisions/017-rl-rollout-sampling-1.0.md)). A rollout that reaches the 16,384-token cap gets reward 0 ([truncation rule](https://github.com/hieutrungle/sharpen-or-expand/blob/main/docs/decisions/014-rl-rollout-sampling-and-truncation.md), point 2).

Why the design matters. With no KL term, nothing in the objective holds the policy near the base model; the only anchors are the small number of updates per batch and clipping on the later mini-batches. With the ratio exactly 1 on the first mini-batch and only mildly off-policy after it, "PPO-style" here means mostly on-policy REINFORCE with a group baseline and a mild trust region. This is the setting the opening animation idealizes. Whatever expansion the experiments find will have to come from sampling, from parameters shared across problems, and from the fact that nothing pulls the policy back toward the base.
</div>

## Check yourself

**1.** A problem has $$p = 0.01$$. You draw 100 samples and none succeeds. What does one REINFORCE update do with no baseline, with the batch mean as baseline, and with the fixed baseline $$b = p$$?

<details markdown="1">
<summary>Answer</summary>

With no baseline every weight is $$R = 0$$, so nothing happens. With the batch mean the baseline is 0 and so is every advantage: nothing happens either. With $$b = p$$ every sample gets advantage −0.01, so each sampled failure is pushed down slightly, and normalization spreads the freed probability over everything that was not sampled, including the correct strategies, which gain a little. A batch with no success can only redistribute probability away from what was sampled; it cannot point toward a solution. That is the hard problem of the opening animation in rounds 1 to 3.
</details>

**2.** Two rollouts get rewards 1 and 0. Compare the update with raw rewards and with baseline 0.5.

<details markdown="1">
<summary>Answer</summary>

With raw rewards the success gets weight 1 and the failure weight 0: the success is pushed up, and the failure loses probability only through normalization. With baseline 0.5 the advantages are +0.5 and −0.5: the success goes up and the failure goes down, each by the same amount. The expected gradient is the same either way. The baseline turns the signal into "better or worse than expected". How much it reduces the noise depends on the problem, as the figure in the baselines section shows.
</details>

**3.** With $$\epsilon = 0.2$$, a token has ratio $$\rho = 1.3$$. What is its gradient if $$A = +1$$, and if $$A = -1$$? What changes if training makes only one update per batch of rollouts?

<details markdown="1">
<summary>Answer</summary>

With $$A = +1$$ the clipped term wins: the objective is 1.2 and flat, so the token gets no gradient of its own (updates on other tokens can still move it). With $$A = -1$$ the unclipped term $$\rho A$$ = −1.3 is the smaller one, and the full gradient pushes the token back down: clipping never blocks a correction. With one update per batch, $$\pi_\theta = \pi_{\text{old}}$$ at the update, every ratio is 1, neither case can occur, and the clipped objective reduces to the plain policy gradient.
</details>

**4.** After RL with a group baseline, pass@1 on a benchmark rises, token entropy falls, and pass@64 barely changes. What mechanism would you propose, and how strong is the evidence?

<details markdown="1">
<summary>Answer</summary>

The easy problem of the opening animation is a candidate mechanism. Successful strategies the model already samples get positive advantages; the expected push on each is proportional to its probability, so the most common one grows fastest, is sampled even more, and takes over. Pass@1 rises on problems the model could already solve, entropy falls because high-probability actions are being reinforced (the mechanism Cui et al. [12] describe), and pass@64, already close to 1 on those problems, barely moves. The pattern is consistent with sharpening. It is not proof. On Part 1's ladder of evidence a rising pass@1 is the first rung, which sharpening produces just as well, and a flat pass@64 means the third rung was not reached; a few newly solved problems can hide in an average. Falling token entropy is not evidence of lost strategies either, since token entropy is not $$N_{\text{eff}}$$. Settling it needs problem-level transitions at a large sampling budget.
</details>

## Takeaways

- The RL objective is the expected reward under the policy's own samples; with a 0/1 verifier reward it is the success probability $$p_\theta(x)$$, averaged over problems.
- The log-derivative trick turns its gradient into an expectation, $$\nabla J = \mathbb{E}[R\, \nabla \log \pi_\theta(\tau)]$$, a sum of per-token log-probability gradients that all share the same reward.
- With a 0/1 reward, $$\nabla J = p_\theta(x)\, \mathbb{E}_q[\nabla \log \pi_\theta]$$: the direction of self-training on the model's own successes, weighted by $$p_\theta(x)$$. With one logit per strategy, the expected push on each correct strategy is proportional to its probability.
- REINFORCE is unbiased but noisy. A baseline that does not depend on the sampled trajectory keeps it unbiased; a group mean that includes the sample rescales the gradient by $$(G - 1)/G$$, and leave-one-out is exact. Baselines remove offset noise, not the noise of rare successes.
- PPO reuses a batch for several updates with per-token ratios $$\rho = \pi_\theta/\pi_{\text{old}}$$ and clips the objective. Clipping removes the incentive to move further, not the ratio itself, and does nothing when $$\pi_\theta = \pi_{\text{old}}$$.
- The KL term sits in the reward (RLHF, $$\beta$$) or in the loss (GRPO, $$\lambda_{\text{KL}}$$); many reasoning recipes drop it, which removes the objective's anchor to the starting model.
- Token entropy is not reasoning diversity. On-policy RL can amplify rare successes but cannot reinforce one it never samples.

## Conclusion

The opening animation is the method in miniature: sample, score, and push each sampled strategy up or down by its advantage. Everything else in this part is about making that update usable. The log-derivative trick makes it computable from samples, baselines make it less noisy without changing its direction, and importance ratios with clipping let one expensive batch serve several updates without moving too far. Two facts about the update itself matter most for this series. Its expected push on a correct strategy is proportional to that strategy's current probability, which favors what the model already does well. And it acts only on what was sampled.

For the question of this series, the policy gradient sits between Part 3's two answers. Unlike offline DPO it has a loop that can find a success the starting model rarely produces and build on it, as the hard problem shows; that is the mechanism by which RL could expand what a model solves. Unlike that hopeful case, its default dynamics on problems with a common solution concentrate probability on that solution, as the easy problem shows, and it can do nothing for a problem whose successes it never samples. Without a KL anchor, as in the experiments, nothing in the objective pulls the policy back toward the base. Which of these effects dominates in a real model is not something this part can settle; whether RL with verifiable rewards mostly sharpens or also expands remains a hypothesis to test.

## Next

[Part 5](/blog/2026/11/grpo-from-first-principles/) derives GRPO from first principles. Instead of a learned critic, it estimates the advantage from a group of samples for the same prompt, and the choices hidden in that estimate (centering, scaling by the standard deviation, the group size) decide what the update rewards.

## References

1. R. J. Williams, "Simple statistical gradient-following algorithms for connectionist reinforcement learning," *Machine Learning*, vol. 8, no. 3–4, pp. 229–256, 1992. [DOI](https://doi.org/10.1007/BF00992696)
2. R. S. Sutton, D. McAllester, S. Singh, and Y. Mansour, "Policy gradient methods for reinforcement learning with function approximation," in *Advances in Neural Information Processing Systems 12 (NIPS 1999)*, pp. 1057–1063, 1999. [Proceedings](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation)
3. A. Ahmadian, C. Cremer, M. Gallé, M. Fadaee, J. Kreutzer, O. Pietquin, A. Üstün, and S. Hooker, "Back to basics: Revisiting REINFORCE-style optimization for learning from human feedback in LLMs," in *Proc. ACL*, 2024. [DOI](https://doi.org/10.18653/v1/2024.acl-long.662) · [arXiv](https://arxiv.org/abs/2402.14740)
4. J. Schulman, P. Moritz, S. Levine, M. Jordan, and P. Abbeel, "High-dimensional continuous control using generalized advantage estimation," arXiv:1506.02438, 2015. [arXiv](https://arxiv.org/abs/1506.02438)
5. J. Schulman, S. Levine, P. Moritz, M. I. Jordan, and P. Abbeel, "Trust region policy optimization," in *Proc. ICML*, 2015. [arXiv](https://arxiv.org/abs/1502.05477)
6. J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal policy optimization algorithms," arXiv:1707.06347, 2017. [arXiv](https://arxiv.org/abs/1707.06347)
7. D. M. Ziegler, N. Stiennon, J. Wu, T. B. Brown, A. Radford, D. Amodei, P. Christiano, and G. Irving, "Fine-tuning language models from human preferences," arXiv:1909.08593, 2019. [arXiv](https://arxiv.org/abs/1909.08593)
8. L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. L. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, J. Schulman, J. Hilton, F. Kelton, L. Miller, M. Simens, A. Askell, P. Welinder, P. Christiano, J. Leike, and R. Lowe, "Training language models to follow instructions with human feedback," in *Proc. NeurIPS*, 2022. [arXiv](https://arxiv.org/abs/2203.02155)
9. Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. K. Li, Y. Wu, and D. Guo, "DeepSeekMath: Pushing the limits of mathematical reasoning in open language models," arXiv:2402.03300, 2024. [arXiv](https://arxiv.org/abs/2402.03300)
10. Q. Yu, Z. Zhang, R. Zhu, Y. Yuan, X. Zuo, Y. Yue, et al., "DAPO: An open-source LLM reinforcement learning system at scale," arXiv:2503.14476, 2025. [arXiv](https://arxiv.org/abs/2503.14476)
11. Z. Liu, C. Chen, W. Li, P. Qi, T. Pang, C. Du, W. S. Lee, and M. Lin, "Understanding R1-Zero-like training: A critical perspective," arXiv:2503.20783, 2025. [arXiv](https://arxiv.org/abs/2503.20783)
12. G. Cui, Y. Zhang, J. Chen, L. Yuan, et al., "The entropy mechanism of reinforcement learning for reasoning language models," arXiv:2505.22617, 2025. [arXiv](https://arxiv.org/abs/2505.22617)
