---
title: "Why Vanilla GRPO Fails: A Failure Map"
kind: Study notes
date: 2026-11-17 18:00:00 -0600
series: llm-post-training
part: 6
summary: "Part 5's GRPO hides several weighting choices in its objective. Part 6 maps what each one does to training: zero-variance groups that waste samples and starve hard prompts, the per-prompt weight from the standard deviation, the per-token weight from the per-response average, symmetric clipping, truncation and positive feedback. Most of them bias how the signal GRPO gets is weighted; the one that matters most for expansion is the signal it never gets."
math: true
anim: true
---

*Code, check scripts and references: [llm-post-training-notes](https://github.com/hieutrungle/llm-post-training-notes) · Experiments: [sharpen-or-expand](https://github.com/hieutrungle/sharpen-or-expand)*

<div class="goal" markdown="1">
<span class="label">Core idea</span>

GRPO's failures come in two kinds. Most are optimizer biases: the objective weights the signal it does get in ways nobody chose on purpose, by the group's standard deviation, by response length, by a clip bound that is relative, and by a cap that turns every cut-off response into a wrong answer. A change of objective can fix those. The other kind is an exploration limit: a prompt whose groups are always wrong gives no signal at all, and no reweighting can create a success that is never sampled. For the question of this series, the second kind is the one that matters.
</div>

## The question

[Part 5](/blog/2026/11/grpo-from-first-principles/) built GRPO from first principles: a group of $$G$$ responses per prompt, an advantage measured against the group, and PPO's clipped objective around it. Every piece had a reason. This part asks what the pieces do once training runs, and which of them bias learning and which starve it. The answer is a failure map: for each design choice, the pathology it creates, what you would see in training logs, and the response that later algorithms adopted.

## The picture to keep in mind

Here is one training batch: twelve prompts, from never solved to almost always solved, trained with Part 5's group advantage for ten rounds.

<div class="anim" data-anim="part-06-failure-map" data-alt="Animation of ten rounds of GRPO-style training on a batch of twelve prompts, shown as a grid of cells. Each cell shows the prompt's starting success probability, its current probability p as a bar, and a group of 8 sampled responses as check or cross chips. The cell outline marks the group as all wrong (grey), mixed (blue) or all correct (gold); only mixed groups move. A counter shows how many groups and rollouts carried signal, and a strip of stacked bars records the number of all-wrong, mixed and all-correct groups per round. In round 1, 7 of 12 groups are mixed (56 of 96 rollouts carry signal) and 5 are all wrong. The prompt that started at 0.02 gets its first success in round 3 and climbs to 0.731. Easy prompts stop moving once their groups are all correct, at p between 0.955 and 0.996. In round 10, 2 groups are mixed (16 of 96 rollouts), 7 all correct and 3 all wrong. The three prompts that started at 10 to the minus 6, 10 to the minus 4 and 0.005 never produce a success and never move."><noscript><img src="{{ '/assets/img/llm-post-training/part-06-failure-map.png' | relative_url }}" alt="Final frame after ten rounds: 2 of 12 groups mixed, 7 all correct and 3 all wrong; the strip of groups per round shows the mixed count falling from 7 to 2; the prompt that started at 0.02 is at 0.731 and the three hardest prompts are unchanged."></noscript></div>

Each cell is one prompt. To keep the picture readable, every prompt is its own pass/fail model with one logit and no parameters shared with the others; their starting success probabilities are $$10^{-6}$$, $$10^{-4}$$, 0.005, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85 and 0.95. Each round samples a group of 8 responses per prompt, shown as chips. The cell's outline says which regime the group is in: all wrong (grey), mixed (blue) or all correct (gold). Each response gets Part 5's advantage, its reward minus the group mean divided by the group's standard deviation, and one policy-gradient step with step size 2 moves each prompt's logit. With $$k$$ of 8 correct that step is $$2\sqrt{k(8 - k)}/8$$, and it is zero when the group is all wrong or all correct. The counter at the top counts the groups and rollouts that carried any signal, and the strip at the bottom records the three regimes round by round.

The run is simulated and the numbers are illustrative. The samples come from a seeded random generator, and the seed was chosen by search for a run that shows the typical behavior: the prompt at 0.02 finds its first success in rounds 2 to 5, the three hardest prompts never succeed, and every prompt that started at 0.05 or above ends above 0.9 (48 of the 400 seeds tried qualify). The expected behavior is close to this: in 10 rounds, a prompt at $$p = 0.02$$ produces at least one success with probability 0.80, one at 0.005 with probability 0.33, and one at $$10^{-4}$$ with probability 0.008. The [checks script](https://github.com/hieutrungle/llm-post-training-notes/blob/main/checks/part06_checks.py) runs the same simulation and recomputes every number the animation prints.

Read the animation:

- **Half the batch is silent from the start.** In round 1, 7 of the 12 groups are mixed, so 56 of the 96 rollouts carry signal. The other 5 groups are all wrong, including the prompt at 0.1, whose group is all wrong with probability 0.43.
- **The frontier moves down when a success appears.** The prompt that started at 0.02 gets its first success in round 3. Alone in its group, that success earns the largest advantage, and the prompt climbs to 0.731 by the end.
- **Easy prompts freeze.** Once a group is all correct, its advantages are 0 and the prompt stops moving, at $$p$$ between 0.955 and 0.996, with a few percent of its probability still on wrong answers.
- **The signal shrinks as training succeeds.** By round 10 only 2 groups are mixed (16 of 96 rollouts), 7 are all correct and 3 are all wrong. In expectation over runs, the share of zero-variance groups grows from 50.9% in round 1 to 78.5% in round 10.
- **The hardest prompts never move.** The prompts that started at $$10^{-6}$$, $$10^{-4}$$ and 0.005 never produce a success, so their advantages are always 0.

That is the point of this part in one sentence: **most of vanilla GRPO's failures are about how it weights the signal in the mixed cells, which a change of objective can fix; the failure that matters most for expansion is the grey cells that never change, where there is no signal to weight.** The rest of this part goes through the weightings one at a time and ends with the map.

## The hidden weightings in the objective

Part 5 wrote DeepSeekMath's GRPO objective [1] (without the optional KL term) as

$$
J_{\text{GRPO}}(\theta) = \mathbb{E}\Bigg[\frac{1}{G}\sum_{i=1}^{G} \frac{1}{\lvert o_i \rvert} \sum_{t=1}^{\lvert o_i \rvert} \min\big(\rho_{i,t} A_i,\ \operatorname{clip}(\rho_{i,t}, 1 - \epsilon, 1 + \epsilon)\, A_i\big)\Bigg], \qquad A_i = \frac{R_i - \bar R}{\sigma_R}.
$$

Five choices in and around it decide how hard each token is pushed, and each is a weighting choice rather than a neutral normalization:

- **$$1/\sigma_R$$** weights each prompt by its group's spread (Part 5's per-prompt weight $$w_G(p)$$).
- **$$1/\lvert o_i \rvert$$** weights each token by the inverse length of its response.
- **The clip range $$\epsilon$$** caps each token's incentive at a fixed ratio to its old probability.
- **The 0/1 terminal reward** gives every token of a response the same number, and gives every all-equal group zero.
- **The length cap** decides that a response cut off at the cap is a wrong answer.

The sections below take them one at a time, starting with what decides whether a group has any signal at all.

## Zero-variance groups: wasted samples and starved prompts

Part 5 showed that a group of $$G$$ samples for a prompt with success probability $$p$$ is mixed with probability $$1 - p^G - (1 - p)^G$$, and that only mixed groups have nonzero advantages. A batch contains prompts at every difficulty, so the question is how much of a whole batch is silent.

For the opening batch the answer can be computed exactly: each prompt's logit can only take the values reached by sums of the four possible steps, so its distribution over rounds can be enumerated (at most 1,001 states per prompt). In round 1, an expected 50.9% of the batch's groups are zero-variance: 42.6% all wrong and 8.3% all correct. Since every group has 8 rollouts, that is also the share of generated rollouts that carry no gradient. The checks script confirms the exact numbers with a Monte Carlo run over 3,000 seeds.

![Two panels. Left: stacked bars for rounds 1 to 10 of the expected share of the batch's groups that are all wrong (grey, bottom), mixed (blue, middle) and all correct (gold, top). The zero-variance share is 50.9% in round 1 and 78.5% in round 10; the grey part shrinks slowly from about 43% to 29%, the gold part grows from 8% to 50%, and the blue part shrinks from 49% to 21%. Right: probability that a prompt produces at least one success after k groups of 8, on a log axis from 1 to 100,000 groups, for p = 0.01, 0.001, 0.0001 and 0.000001. Each curve is an S shape that reaches one half near k = 0.087/p; a dot marks p = 0.0001 after 100 groups at 7.7%.]({{ '/assets/img/llm-post-training/part-06-zero-variance.png' | relative_url }})

The left panel shows, for each round of the opening animation's batch, the expected share of groups in each regime, averaged over all possible runs. The right panel takes one prompt whose groups are all wrong and asks how likely it is to produce a success if it is simply sampled again and again, $$k$$ groups of 8 at a time.

Read the curves:

- **Left: the zero-variance share grows as training succeeds.** It is 50.9% in round 1 and 78.5% in round 10. The all-correct share grows from 8.3% to 49.7%, the mixed share falls from 49.1% to 21.5%, and the all-wrong share falls only slowly, to 28.9%, because the prompts near $$p = 0$$ never leave it.
- **Right: resampling a hard prompt follows $$1 - (1 - p)^{8k}$$.** All-wrong groups give no update, so $$p$$ stays where it is while the prompt is resampled. At $$p = 10^{-4}$$ the chance of at least one success is 0.08% after one group, 0.80% after 10 groups and 7.7% after 100; it takes 6,932 rollouts (867 groups) for an even chance.

The takeaway: the two kinds of zero-variance group are different problems. All-correct groups are prompts already mastered; they waste the compute spent sampling them, and a batch that fills with them makes the gradient smaller and noisier. All-wrong groups are prompts on which the update has no direction at all: the model learns nothing about how to succeed, not even which failures to avoid. This is exploration starvation.

DAPO [3] responds to the first problem with *dynamic sampling*: it over-samples and drops every group whose accuracy is 0 or 1 until the batch is full of mixed groups. In the opening batch that costs an expected 2.04 sampled groups per kept group in round 1 and 4.66 in round 10. It keeps the gradient batch informative, but it cannot help the second problem: a prompt whose groups stay all wrong is dropped every time, and resampling it only follows the curve on the right. Dynamic sampling recovers wasted compute; it does not create a success on a prompt that does not produce one. (Part 7 derives it.)

## The standard-deviation weighting: bias or a different objective?

Part 5 showed that dividing by $$\sigma_R$$ turns each prompt's expected update into $$w_G(p)\, \nabla_\theta p$$ with a weight that depends on $$p$$, where mean-centering alone gives the constant $$(G - 1)/G$$. What matters for a batch is how the weights compare across prompts. Divide the GRPO weight by the centering-only weight and normalize to 1 at $$p = 0.5$$; at $$G = 8$$:

| Success probability $$p$$ | 0.01 | 0.05 | 0.2 | 0.5 | 0.8 | 0.95 | 0.99 |
|---|---|---|---|---|---|---|---|
| GRPO's weight relative to centering, normalized at $$p = 0.5$$ | 1.41 | 1.33 | 1.14 | 1.00 | 1.14 | 1.33 | 1.41 |
| A pass@8 objective's weight, normalized at $$p = 0.5$$ | 119 | 89.4 | 26.8 | 1.00 | 0.0016 | $$1.0 \times 10^{-7}$$ | $$1.3 \times 10^{-12}$$ |

Relative to the true gradient of the average success probability, GRPO up-weights prompts at both ends: a prompt at $$p = 0.01$$ or $$p = 0.99$$ counts 1.41 times as much, per unit of $$\nabla_\theta p$$, as a prompt at even odds. The weight multiplies $$\nabla_\theta p$$, which itself shrinks near 0 and 1, so the largest updates still come from prompts in the middle; what changes is the balance.

There are two fair readings of this.

- **A bias.** Liu et al. [2] call it a *question-level difficulty bias*: questions with a low standard deviation, those that are too easy or too hard, get higher weights in the policy update. They note that advantage normalization is usually computed over a whole batch, and that normalizing per question gives different questions different weights in the objective. Their Dr. GRPO removes the division.
- **A different objective.** Davis and Recht [4] read the same algebra as optimizing a different monotone transform of $$p$$ for each prompt, the arcsine of its square root in the large-group limit (Part 5). On that reading nothing is broken; GRPO simply climbs a different objective than the average success probability.

Both readings are correct, and they lead to the right question: what is being reweighted, and does the reweighting serve the objective you care about? If the target is the average pass@1 over prompts, the right per-prompt weight is constant, and the division by $$\sigma_R$$ is a bias. If the target puts more value on prompts the model rarely solves, an up-weighting of low $$p$$ is what you want. The second row of the table shows what that looks like for one such target: the gradient of pass@8, $$1 - (1 - p)^8$$, is $$8(1 - p)^7\, \nabla_\theta p$$, which weights $$p = 0.01$$ 119 times as much as $$p = 0.5$$ and essentially ignores $$p = 0.99$$. GRPO's weighting points the same way at the hard end, far more weakly, and it also up-weights the easy end, which no coverage objective asks for. Calling the division simply "bad" misses this; calling it "an objective" without saying which one misses the rest.

## The length weighting

The second average in the objective is over tokens. With the per-response average, every response gets the same total weight, so a token of response $$i$$ is weighted by

$$
\frac{A_i}{G\, \lvert o_i \rvert}.
$$

[Part 2](/blog/2026/11/sft-probability-reshaping/) compared the three ways to normalize a loss on a batch with one 100-token and one 1,000-token response. Put advantages on them. If both are correct ($$A \gt 0$$), each token of the short response is pushed up 10 times as hard as each token of the long one. If both are wrong ($$A \lt 0$$), each token of the long response is pushed down a tenth as hard: a long failure spreads the same response-level penalty over ten times as many tokens.

verl 0.9.1, the library the experiments use, implements the aggregations in one function, `agg_loss` [7]. GRPO uses `seq-mean-token-mean` (the per-response average), DAPO uses `token-mean`, and Dr. GRPO uses `seq-mean-token-sum-norm`, a token sum divided by a constant. Reimplementing `agg_loss` on Part 2's two-response batch gives the weight each token's loss carries (to be multiplied by its advantage):

| Aggregation | Each token of the 100-token response | Each token of the 1,000-token response | Share of the total weight on the long response |
|---|---|---|---|
| `seq-mean-token-mean` (GRPO) | 0.005 | 0.0005 | 50% |
| `token-mean` (DAPO) | 0.00091 | 0.00091 | 90.9% |
| `seq-mean-token-sum-norm`, default, both in one micro-batch | 0.0005 | 0.0005 | 90.9% |
| `seq-mean-token-sum-norm`, default, each in its own micro-batch | 0.005 | 0.0005 | 50% |
| `seq-mean-token-sum-norm` with a constant 16,384 | 0.000031 | 0.000031 | 90.9% |

Three details of the implementation matter.

- **`token-mean` divides by the token count of the whole mini-batch.** The denominator is the number of valid response tokens in the mini-batch of one optimizer update, summed over all GPUs. Every token gets the same weight, so a pattern is pushed equally hard wherever it appears, and a long response carries weight in proportion to its length.
- **Dr. GRPO's "constant" is not the default.** Liu et al. replace the per-response length with a constant, such as the generation budget [2]. In verl, `seq-mean-token-sum-norm` divides by `loss_scale_factor` if it is set and otherwise by the padded width of the micro-batch it is computing. With dynamic batching that width changes from micro-batch to micro-batch, and the fourth row shows the result: the same short response gets 10 times the weight when it lands in a micro-batch padded to its own length. Only a fixed `loss_scale_factor` gives the per-token weight that Dr. GRPO intends.
- **The scale of the constant does not matter for the direction.** Rows three and five weight every token equally; they differ only by a constant factor, which acts like a learning rate.

### Does the per-response average make wrong answers longer?

Liu et al. [2] argue from these per-token weights that GRPO prefers shorter correct responses and longer incorrect ones, and that this may lead to progressively longer incorrect responses. They also report it in training: GRPO kept generating longer responses even when the reward improvement slowed down, and Dr. GRPO's incorrect responses on evaluation benchmarks were substantially shorter than GRPO's. The per-token weights are an identity; the lengthening is a claim about training dynamics. To see when the first implies the second, here is a toy where everything can be computed exactly.

Each response is one strategy token, correct with probability $$p = 0.5$$, followed by $$L \ge 1$$ tokens. After each token the response stops with probability $$s$$, so the length is geometric with mean $$1/s = 100$$ at the start. Length has nothing to do with correctness. Accuracy is held fixed, so the only thing that can change is the stop probability. Take $$G = 8$$ and Part 5's advantage. The score of a response's length tokens with respect to the stop logit is $$1 - Ls$$ (the stop token contributes $$1 - s$$ and each of the $$L - 1$$ continuations $$-s$$), and its expectation is 0. Then:

- **Constant denominator (or token sum).** The expected update of the stop logit is proportional to $$\mathbb{E}[A_i(1 - L_i s)]$$. Given the rewards, the length is independent of the advantage, so this is $$\mathbb{E}[A_i] \cdot 0 = 0$$: no drift.
- **Per-response average, with separate stop probabilities for correct and wrong responses.** The weight $$1/(1 + L_i)$$ is largest for short responses, and the expected update becomes $$\pm\,\mathbb{E}[\sigma_R]\; \mathbb{E}\big[(1 - Ls)/(1 + L)\big]$$, positive (shorter) for correct responses and negative (longer) for wrong ones. The second factor is positive whenever lengths vary, because $$1 - Ls$$ and $$1/(1 + L)$$ both fall as $$L$$ grows.
- **Per-response average, one stop probability shared by all responses.** The correct and wrong contributions now land on the same parameter. A group's advantages sum to 0, so the two drifts cancel exactly.

![Line plot of mean response length in tokens, on a log axis from 5 to 600, against the number of expected updates from 0 to 200, with accuracy held at 0.5. All lines start at 100. With the per-response average and separate stop probabilities, the wrong responses' length rises to 247 (solid) and the correct responses' length falls to 11 (dashed). A flat grey line at 100 marks both lengths with a constant denominator, or with one shared stop probability.]({{ '/assets/img/llm-post-training/part-06-length.png' | relative_url }})

The figure iterates the exact expected update with step size 0.5 from a mean length of 100 tokens. The blue lines are the per-response average when correct and wrong responses have their own stop probability; the grey line is every case without drift.

Read the curves:

- **Separate stop probabilities, per-response average.** After 200 expected updates, wrong responses average 247 tokens and correct ones 11. The rate is $$\mathbb{E}[\sigma_R] = 0.464$$ (at $$p = 0.5$$, $$G = 8$$) times $$\mathbb{E}[(1 - Ls)/(1 + L)] = 0.0273$$ at the start.
- **Constant denominator, or one shared stop probability.** No drift at all: lengths stay at 100 in expectation.
- **`token-mean` (not drawn).** Its denominator includes each response's own length, which leaves a small drift of the same sign. In a Monte Carlo estimate with mini-batches of 128 responses, it is under 1% of the per-response average's.

The takeaway: the per-response average lengthens wrong responses and shortens correct ones only to the extent that the model can tell, while it writes, that it is on a wrong path, so that the length of failures and the length of successes are controlled separately. In a toy where they share one parameter, nothing happens. A real model sits in between, and a response that loops or hedges is a plausible case of a wrong path the model can recognize. So the per-token weights are the solid argument; how much they lengthen wrong answers in a given model is an empirical question, and Liu et al.'s comparison is evidence that it can be large.

Whatever its cause, length growth confounds evaluation. If pass@1 rises while responses get much longer, the gain could come from better reasoning, from spending more inference compute per problem, or from both. Comparing methods at equal accuracy-per-token, or training them at equal rollout-token budgets, separates the two.

## Symmetric clipping and rare tokens

PPO's clipped objective [5], as [Part 4](/blog/2026/11/policy-gradients-reinforce-to-ppo/) built it, stops a token's own incentive once its ratio passes $$1 + \epsilon$$ (for $$A \gt 0$$). The bound is relative and identical for every token: whatever its old probability, one batch can raise a token's own term only until $$\pi_\theta = (1 + \epsilon)\, \pi_{\text{old}}$$. What differs between rare and common tokens is the absolute probability that this allows:

| $$\pi_{\text{old}}$$ of a token with $$A \gt 0$$ | Bound at $$\epsilon = 0.2$$ | Absolute gain | Bound at $$\epsilon_{\text{high}} = 0.28$$ | Absolute gain |
|---|---|---|---|---|
| 0.001 | 0.0012 | 0.0002 | 0.00128 | 0.00028 |
| 0.01 | 0.012 | 0.002 | 0.0128 | 0.0028 |
| 0.1 | 0.12 | 0.02 | 0.128 | 0.028 |
| 0.5 | 0.6 | 0.1 | 0.64 | 0.14 |
| 0.9 | 1.08 (no bound below 1) | up to 0.1 | 1.152 (no bound below 1) | up to 0.1 |

A rare useful token at 0.001 can gain 0.0002 from one batch's incentive; a common one at 0.5 can gain 0.1, 500 times as much. DAPO [3] makes the same point with 0.01 and 0.9: their upper bounds are 0.012 and 1.08, so a high-probability "exploitation" token is not held back at all, while a low-probability "exploration" token can barely rise. Symmetric clipping therefore favors sharpening behavior the model already prefers over quickly amplifying a rare one.

Three precisions keep this from being overstated.

- **The bound is per batch, not forever.** Across batches the ratio resets and a rare token can compound. If every batch raised it by the full clipped amount, a token at 0.001 would reach 0.5 after 35 batches at $$\epsilon = 0.2$$ and 26 at $$\epsilon = 0.28$$. The bound slows rare-token growth; it does not forbid it.
- **It bounds the token's own incentive, not its probability.** Part 4 showed that a token past its bound still moves when updates on other tokens change shared parameters.
- **It acts only when $$\pi_\theta \ne \pi_{\text{old}}$$.** On the first update after sampling every ratio is 1 and the objective is the plain policy gradient (Part 4). Clipping matters only on later updates that reuse the same batch.

DAPO's response is *Clip-Higher*: decouple the bounds into $$1 - \epsilon_{\text{low}}$$ and $$1 + \epsilon_{\text{high}}$$ and raise only the upper one, with $$\epsilon_{\text{low}} = 0.2$$ and $$\epsilon_{\text{high}} = 0.28$$ [3]. They keep $$\epsilon_{\text{low}}$$ as it is because raising it would push the probability of these tokens toward 0. The last two columns show what it buys: 40% more room per batch for every token, which matters most where the absolute room is smallest. (Part 7 derives it.)

## Positive feedback and entropy collapse

Part 4 showed that policy-gradient updates concentrate probability on what is already common, and cited Cui et al. [6]: without intervention, policy entropy falls sharply early in RL training for reasoning while performance saturates. Part 5 described the loop behind it: a success is reinforced, sampled more often, and reinforced again. What GRPO adds is speed at the start of the loop. A lone success gets the largest advantage its group can give, $$\sqrt{7}$$ ≈ 2.65 at $$G = 8$$, so a newly found mode is amplified hard; in the opening animation that is what moved the prompt at 0.02. The same loop on an easy prompt drives every sample to the dominant answer, until the group is all correct and the prompt freezes. DAPO reports exactly this collapse in its baseline runs and introduced Clip-Higher against it. Token entropy is a useful diagnostic of the collapse, but it is not reasoning diversity: Part 4's example had ten times the token entropy with a quarter of the strategies.

## Truncation: one label for three failures

RL training sets a maximum response length. A rollout that reaches it is cut off, has no final answer, and gets reward 0. That single label covers very different failures:

- a wrong answer that happened to be long;
- a sound derivation that was not finished in time;
- a loop or repetition that would never have ended.

The update treats all three the same way: every token of the cut-off response gets the negative advantage of a failure. For the first and third that is roughly right; for the second it punishes sound reasoning for its length. DAPO [3] calls this reward noise and responds in two steps: *overlong filtering*, which masks the loss of truncated samples, and *soft overlong punishment*, which adds a length penalty that grows linearly over a buffer before the cap (an expected maximum of 16,384 tokens plus a 4,096-token buffer, so 20,480 tokens in all) and is added to the correctness reward. (Part 7 derives both.) Truncation also interacts with the length weighting: a reward of 0 at the cap pushes toward shorter responses, while the per-response average can push failures toward longer ones.

## Sparse rewards, verifier errors and reward hacking

Three more failures come from the reward itself rather than from GRPO's weighting, and they are the subject of a later part. Each gets one row in the map:

- **Sparse terminal reward.** One number for a whole response: a derivation correct until a final slip gets the same 0 as nonsense, and a lucky answer gets the full 1. This is Part 4's "one reward, many tokens" problem, unchanged by GRPO.
- **Verifier errors.** A verifier that rejects a correct answer in an unusual format, or accepts a wrong one, mislabels the response, and the advantage amplifies the mislabel, most of all when it is the lone success or failure in its group.
- **Reward hacking.** RL optimizes the verifier, not correctness. If the verifier has a shortcut, the reward can rise without the reasoning improving.

## Optimizer bias versus exploration limitation

The opening animation can be read twice.

Read the blue cells. Their groups contain both successes and failures, so useful information was sampled. What the objective does with it is a choice: it weights each prompt by $$1/\sigma_R$$, each token by $$1/\lvert o_i \rvert$$, caps each token's incentive at a relative bound, and calls every cut-off response wrong. These are **optimizer biases**. They change which prompts and tokens drive the update, and they can change what training converges to, but each can be removed or reweighted by a change of objective, which is what the algorithms of the next part do.

Read the grey cells at the top left. Their groups were all wrong in every round. No advantage was ever nonzero, so there was nothing to weight. That is an **exploration limitation**: the useful behavior was never sampled. At $$p = 10^{-8}$$, a group of 8 contains a success with probability $$8.0 \times 10^{-8}$$, and even 10,000 rounds of such groups find one with probability $$8.0 \times 10^{-4}$$. No normalization, clip range or aggregation can reward a response that never appears; only something that changes what is sampled can, such as a larger group, a different sampler, or learning on other prompts that raises $$p$$ here through shared parameters.

This distinction is the series question in miniature. Fixing optimizer biases makes GRPO better at using the successes the model already produces, which is what sharpening needs. Expansion needs the grey cells to change color, and that is limited by exploration, not by weighting. Whether training on a real model, with parameters shared across prompts, reaches those cells is the empirical question; on [Part 1](/blog/2026/11/capability-and-pass-at-k/)'s ladder of evidence it is answered only by problems unsolved at a large sampling budget becoming repeatedly solvable.

## The failure map

| Component | Pathology | Observable signature | Typical response (Part 7) |
|---|---|---|---|
| Group-relative advantage, 0/1 reward | All-correct groups waste samples; all-wrong groups give no direction | Share of zero-variance groups rises during training; the batch's effective number of prompts falls | Dynamic sampling [3] keeps only mixed groups (recovers compute, not successes) |
| Division by $$\sigma_R$$ | Per-prompt reweighting: prompts near $$p = 0$$ or $$p = 1$$ get more weight per unit of $$\nabla_\theta p$$ | Not visible in the loss; per-prompt update sizes differ from those of a centered advantage | Dr. GRPO [2] removes it; or keep it as a chosen objective [4] |
| Per-response average, $$1/\lvert o_i \rvert$$ | Short correct tokens pushed harder; long wrong tokens suppressed less | Length of incorrect responses grows while accuracy stalls | Dr. GRPO's constant denominator [2]; DAPO's token-level loss [3] |
| Symmetric clip range | A rare token's own incentive allows only a small absolute gain per batch | Clipped tokens are mostly low-probability ones; entropy falls | Clip-Higher, $$\epsilon_{\text{high}} \gt \epsilon_{\text{low}}$$ [3] |
| Positive feedback | A found mode takes over; entropy collapses | Entropy falls early while pass@1 rises and pass@k flattens | Clip-Higher [3]; entropy controls such as Clip-Cov and KL-Cov [6] |
| Reward 0 at the length cap | Three failure types share one label | Many truncated rollouts, especially in all-wrong groups; length and reward unstable | Overlong filtering and soft overlong punishment [3] |
| Sparse terminal reward | One number for every token (credit assignment) | Whole responses rise or fall together | Still open (Part 4) |
| Verifier errors | Mislabels, amplified by group advantages | Rewards disagree with careful re-grading | Audit the verifier |
| Reward hacking | Reward rises without better reasoning | Reward goes up, independent checks do not | Audit the verifier; held-out evaluation |
| All-wrong prompts at tiny $$p$$ | Exploration limitation: no success is ever sampled | Prompts that stay at 0 of $$G$$ throughout training | None of the objective fixes; only changes to what is sampled |

The last row is different from the others: every response in the right-hand column above it changes how sampled information is weighted, and none of them reaches it.

<div class="note" markdown="1">
<span class="label">In my experiments</span>

The RL arms of [sharpen-or-expand](https://github.com/hieutrungle/sharpen-or-expand) have no results on the public main branch yet. What is fixed is how each failure in the map is handled, and which comparisons are designed to separate them.

- **The switches.** GRPO is verl's `grpo` advantage with `seq-mean-token-mean` and a symmetric clip. Dr.GRPO is GRPO with the division by the standard deviation switched off (`norm_adv_by_std_in_grpo=false`) and `seq-mean-token-sum-norm`. DAPO adds dynamic sampling, Clip-Higher, `token-mean` and the overlong buffer in its reward manager ([GPU stack decision](https://github.com/hieutrungle/sharpen-or-expand/blob/main/docs/decisions/001-gpu-stack-verl.md)). The algorithms' configs may differ only in these switches, RS-GRPO's risk parameter $$\beta_{\text{RS}}$$ and the group size of one ablation ([shared 4B config](https://github.com/hieutrungle/sharpen-or-expand/blob/main/docs/decisions/025-g0-shared-4b-config.md)).
- **Dr.GRPO's constant.** A test of verl's loss aggregation found the padded-width default described above ([loss tests](https://github.com/hieutrungle/sharpen-or-expand/blob/main/docs/reports/m1_rl_tests_mutations.md)). The project decided to set the constant to 16,384, the response cap; it is decided but not yet wired, because Dr.GRPO itself is not wired yet ([roadmap](https://github.com/hieutrungle/sharpen-or-expand/blob/main/docs/roadmap/M2.md)).
- **Truncation.** A rollout that reaches the 16,384-token cap gets reward 0, as in evaluation. DAPO's overlong shaping is applied on top of that rule, and the ablation that removes it keeps the base rule ([truncation rule](https://github.com/hieutrungle/sharpen-or-expand/blob/main/docs/decisions/014-rl-rollout-sampling-and-truncation.md), point 2).
- **Ablations.** Full DAPO minus dynamic sampling, minus Clip-Higher and minus the token-level loss; DAPO minus overlong handling only if truncation shows up in the main runs; and GRPO at $$G$$ = 4, 8 and 16 ([milestone plan](https://github.com/hieutrungle/sharpen-or-expand/blob/main/docs/milestone-plan.md), ablations).
- **Equal token budgets.** Runs are compared at equal rollout-token budgets ([budget stop](https://github.com/hieutrungle/sharpen-or-expand/blob/main/docs/decisions/020-rl-token-budget-stop.md), [checkpoints](https://github.com/hieutrungle/sharpen-or-expand/blob/main/docs/decisions/013-rl-no-kl-step-checkpoints.md)). Rollout tokens are the generated response tokens of training rollouts, a truncated one counted at its 16,384 tokens; prompt tokens are left out because each prompt repeats $$G$$ times and would make the count depend on the group size ([token count](https://github.com/hieutrungle/sharpen-or-expand/blob/main/docs/decisions/018-rl-checkpoint-export-and-token-count.md)). A method that makes responses longer pays for it in fewer training steps.

What the starting policy's failures are. Part 5 quoted the first step of a development run that trained from the base model, sampled before any update. Read for failures, the same step says more ([data](https://github.com/hieutrungle/sharpen-or-expand/blob/main/docs/reports/m2_rollout_groups.json), [group audit](https://github.com/hieutrungle/sharpen-or-expand/blob/main/docs/reports/m2_base_start_audit.md)): of its 512 rollouts, 185 hit the cap, and of the 327 that stopped, 312 were correct. So 185 of its 200 failures, 92.5%, were truncations, and 92.9% of the rollouts in all-wrong groups were truncated. This describes one batch of the starting policy's rollouts, not a result.

Why it matters for the series question. At the start, most of what the verifier calls wrong is a response that ran out of length, not a wrong answer. Early gains in pass@1 can therefore come from learning to stop rather than from reasoning better. For that reason every pass@1 change in the experiments is split into a stopping part and an accuracy-when-stopped part and shown with truncation, for every algorithm ([start decision](https://github.com/hieutrungle/sharpen-or-expand/blob/main/docs/decisions/031-rl-and-dpo-start-from-the-base.md), item 6).
</div>

## Check yourself

**1.** Two prompts in a batch both give zero-variance groups at $$G = 8$$: one all correct, one all wrong. Why is the second more worrying, and what does dynamic sampling do about each?

<details markdown="1">
<summary>Answer</summary>

Both get advantage 0 for every response, so both contribute nothing. The all-correct prompt is already mastered at the current sampling budget; spending rollouts on it wastes compute. The all-wrong prompt is starved: the update has no direction at all, not even which failures to avoid, and it stays that way until a group happens to contain a success. Dynamic sampling drops both kinds and fills the batch with mixed groups, which recovers the wasted compute. It cannot help the all-wrong prompt: resampling it produces a success only with probability $$1 - (1 - p)^{8k}$$ after $$k$$ groups, 7.7% after 100 groups at $$p = 10^{-4}$$.
</details>

**2.** In a group of 8, two wrong responses have the same advantage; one has 100 tokens and the other 2,000. How hard is each token pushed down under GRPO's aggregation, DAPO's and Dr. GRPO's? Does GRPO's aggregation make wrong answers longer?

<details markdown="1">
<summary>Answer</summary>

Under the per-response average each token is weighted by $$A/(8 \lvert o_i \rvert)$$: 0.00125 times $$A$$ per token for the short response and 0.0000625 times $$A$$ for the long one, so each token of the long failure is suppressed 20 times less. Under `token-mean`, and under a token sum with a constant denominator, every token gets the same weight. Whether this makes wrong answers longer depends on the model: in a toy where correct and wrong responses control their length separately, the per-response average lengthens wrong ones and shortens correct ones; when one stop probability is shared, the effects cancel exactly. Liu et al. report that in their runs GRPO's incorrect responses were substantially longer than Dr. GRPO's.
</details>

**3.** A token with positive advantage has $$\pi_{\text{old}} = 0.001$$, another $$\pi_{\text{old}} = 0.5$$, and $$\epsilon = 0.2$$. How much can one batch raise each through its own term? On which update does clipping not act at all?

<details markdown="1">
<summary>Answer</summary>

The bound is the same relative one for both, a ratio of 1.2, so the first can rise to 0.0012 (a gain of 0.0002) and the second to 0.6 (a gain of 0.1). The absolute room differs by a factor of 500, which is why symmetric clipping favors sharpening common behavior. Across batches the rare token can still compound, and other tokens' updates can move it through shared parameters. On the first update after sampling, $$\pi_\theta = \pi_{\text{old}}$$, every ratio is 1, and clipping does nothing; it acts only on later updates that reuse the batch.
</details>

**4.** GRPO and Dr. GRPO end with the same pass@64 coverage, but Dr. GRPO has higher pass@1, shorter incorrect responses and better accuracy per token. What do you conclude?

<details markdown="1">
<summary>Answer</summary>

That removing the optimizer biases let Dr. GRPO use the successes the model already produced more efficiently. The result is consistent with sharpening: pass@1 and token efficiency improved on problems that were already reachable. A flat pass@64 is weak evidence by itself, because pass@64 saturates and can hide problems gained and lost in equal numbers; on Part 1's ladder of evidence, expansion would need problems unsolved at a large sampling budget to become repeatedly solvable, checked problem by problem. Neither algorithm addresses the exploration limit, so neither result is evidence of expansion.
</details>

## Takeaways

- GRPO's objective hides five weighting choices: $$1/\sigma_R$$, $$1/\lvert o_i \rvert$$, a relative clip range, a 0/1 terminal reward and a length cap that scores a cut-off response as wrong.
- Zero-variance groups take over a batch as training succeeds (50.9% to 78.5% in the opening batch, in expectation). All-correct groups waste compute; all-wrong groups starve. Dynamic sampling fixes the first, not the second.
- Dividing by $$\sigma_R$$ up-weights prompts at both ends of $$p$$. It is a bias if the target is average pass@1, and a different objective otherwise; the useful question is whether the weighting serves the target.
- The per-response average weights a token by $$A_i/(G \lvert o_i \rvert)$$. That is an identity; that it lengthens wrong answers is a dynamics claim, which holds in a toy only when the length of failures is controlled separately from the length of successes.
- The clip bound is relative and the same for every token; a rare token gains little absolute probability per batch, but can compound across batches. Clipping acts only after the first update on a batch.
- A rollout cut off at the cap gets the same label as a wrong answer, whether it was wrong, unfinished or looping.
- Optimizer biases change how sampled signal is weighted and can be fixed by a change of objective. An exploration limitation is signal that is never sampled, and no reweighting reaches it.

## Conclusion

The opening batch shows both halves of this part at once. In the blue cells GRPO had something to learn from, and the objective decided how much each prompt and each token counted: more for prompts near the ends of $$p$$, more for the tokens of short responses, a bounded amount for rare tokens, and a zero for every response cut off at the cap. Each of those is a choice a different objective can make differently, and the next part takes the three main algorithms that do. The grey cells at the top left are a different kind of failure. They never contained a success, so nothing in the objective ever touched them, and as training mastered the easy prompts, the share of the batch that carried any signal shrank.

For the series question, this sharpens what each kind of fix can show. An algorithm that removes the optimizer biases should use the successes a model already has more efficiently, with less wasted length, and that is consistent with sharpening. Expansion asks for something else: that prompts which never produced a success start to. Whether a real model gets there through parameters shared across prompts, a larger group or a different sampler is a hypothesis to test. Yue et al. [8] report that RL-trained models lose to their base models at large $$k$$; Liu et al.'s ProRL [9] report that prolonged training with resets and diverse tasks makes problems solvable that the base model fails at any number of attempts; and Wu et al. [10] find, across the benchmarks they evaluate, that more questions lose a verified success under RL than gain one. The experiments are designed to measure which of these describes their runs, with the stopping and accuracy parts of every pass@1 change kept apart.

## Next

Part 7 derives the fixes from the failures they target: Dr. GRPO's removal of the standard-deviation and length weightings, DAPO's four components (dynamic sampling, Clip-Higher, the token-level loss and overlong reward shaping), and RS-GRPO's risk-sensitive objective, which reweights prompts on purpose to value rare successes.

## References

1. Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. K. Li, Y. Wu, and D. Guo, "DeepSeekMath: Pushing the limits of mathematical reasoning in open language models," arXiv:2402.03300, 2024. [arXiv](https://arxiv.org/abs/2402.03300)
2. Z. Liu, C. Chen, W. Li, P. Qi, T. Pang, C. Du, W. S. Lee, and M. Lin, "Understanding R1-Zero-like training: A critical perspective," in *Proc. Conference on Language Modeling (COLM)*, 2025. [arXiv](https://arxiv.org/abs/2503.20783)
3. Q. Yu, Z. Zhang, R. Zhu, Y. Yuan, X. Zuo, Y. Yue, et al., "DAPO: An open-source LLM reinforcement learning system at scale," arXiv:2503.14476, 2025. [arXiv](https://arxiv.org/abs/2503.14476)
4. D. Davis and B. Recht, "What is the objective of reasoning with reinforcement learning?," arXiv:2510.13651, 2025. [arXiv](https://arxiv.org/abs/2510.13651)
5. J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal policy optimization algorithms," arXiv:1707.06347, 2017. [arXiv](https://arxiv.org/abs/1707.06347)
6. G. Cui, Y. Zhang, J. Chen, L. Yuan, et al., "The entropy mechanism of reinforcement learning for reasoning language models," arXiv:2505.22617, 2025. [arXiv](https://arxiv.org/abs/2505.22617)
7. verl contributors, verl v0.9.1, `agg_loss` and `compute_policy_loss_vanilla` in `verl/trainer/ppo/core_algos.py`, 2026. [agg_loss](https://github.com/verl-project/verl/blob/v0.9.1/verl/trainer/ppo/core_algos.py#L1140-L1206) · [compute_policy_loss_vanilla](https://github.com/verl-project/verl/blob/v0.9.1/verl/trainer/ppo/core_algos.py#L1285-L1376)
8. Y. Yue et al., "Does reinforcement learning really incentivize reasoning capacity in LLMs beyond the base model?" arXiv:2504.13837, 2025. [arXiv](https://arxiv.org/abs/2504.13837)
9. M. Liu et al., "ProRL: Prolonged reinforcement learning expands reasoning boundaries in large language models," arXiv:2505.24864, 2025. [arXiv](https://arxiv.org/abs/2505.24864)
10. F. Wu et al., "The invisible leash: Why RLVR may or may not escape its origin," arXiv:2507.14843, 2025. [arXiv](https://arxiv.org/abs/2507.14843)
