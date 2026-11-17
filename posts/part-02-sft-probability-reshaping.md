---
title: "Supervised Fine-Tuning as Probability Reshaping"
kind: Study notes
date: 2026-11-16 12:00:00 -0600
series: llm-post-training
part: 2
summary: "Supervised fine-tuning makes demonstrated solutions more likely. Part 2 derives what that does to a model's trajectory distribution: the cross-entropy gradient pushes down every alternative, gains compound along a sequence, teacher forcing hides the model's own mistakes, and maximum likelihood is forward KL to the data, not to the set of correct answers."
math: true
anim: true
---

*Code, check scripts and references: [llm-post-training-notes](https://github.com/hieutrungle/llm-post-training-notes) · Experiments: [sharpen-or-expand](https://github.com/hieutrungle/sharpen-or-expand)*

<div class="goal" markdown="1">
<span class="label">Core idea</span>

Supervised fine-tuning (SFT) is conditional behavior cloning. It raises the probability of the demonstrated solution, and because probabilities sum to one, it lowers everything else, including other correct solutions. It minimizes $$-\log \pi_\theta(\tau_{\text{demo}} \mid x)$$, not the failure probability $$1 - p_\theta(x)$$, so it can sharpen, narrow or reshape the success landscape of [Part 1](/blog/2026/11/capability-and-pass-at-k/) without ever measuring correctness.
</div>

## The question

[Part 1](/blog/2026/11/capability-and-pass-at-k/) described a model on a problem $$x$$ as a distribution over trajectories $$\pi_\theta(\tau \mid x)$$, and its capability as the success mass $$p_\theta(x)$$ under fixed inference conditions. Supervised fine-tuning is usually the first post-training step applied to that distribution. This part asks what imitation does to it: which trajectories gain probability, which lose it, and why a lower training loss need not mean a model that solves more problems.

## SFT is conditional behavior cloning

SFT starts from a dataset of prompts and demonstrated responses,

$$
\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}, \qquad y_i = (y_{i,1}, \ldots, y_{i,T_i}).
$$

For a reasoning task a response is a full trajectory: the chain of thought and the final answer. I write $$\tau$$ for a trajectory the model generates and $$y$$ for a demonstrated one. The demonstrations come from some data distribution $$q_{\text{data}}(y \mid x)$$: human solutions, a stronger model, or, as in the experiment below, the model's own verified solutions. (Part 1's $$q(\tau \mid x)$$ is a different object, the model's own distribution restricted to correct trajectories.)

The model is autoregressive, so the probability of a response is a product of next-token probabilities,

$$
\pi_\theta(y \mid x) = \prod_{t=1}^{T} \pi_\theta(y_t \mid x, y_{\lt t}),
$$

and SFT maximizes the log-likelihood of the demonstrations. Taking the log turns the product into a sum, which gives the familiar token-level cross-entropy loss:

$$
\mathcal{L}_{\text{SFT}}(\theta) = -\sum_{(x, y) \in \mathcal{D}} \log \pi_\theta(y \mid x) = -\sum_{(x, y) \in \mathcal{D}} \sum_{t=1}^{T} \log \pi_\theta(y_t \mid x, y_{\lt t}).
$$

The name *behavior cloning* comes from imitation learning: learn a policy by supervised learning on an expert's actions in the states the expert visited. Here the state is the prefix $$(x, y_{\lt t})$$ and the action is the next token. The objective says: *in contexts like these, produce tokens like these.* Nothing in it refers to whether the final answer is correct.

## What one gradient step does

Look at a single position. The model produces logits $$z_1, \ldots, z_V$$ over the vocabulary, the softmax turns them into probabilities $$p_j = e^{z_j} / \sum_k e^{z_k}$$, and the demonstrated token has index $$c$$. The loss at this position is $$\ell = -\log p_c = -z_c + \log \sum_k e^{z_k}$$. Differentiating,

$$
\frac{\partial \ell}{\partial z_j} = -\mathbf{1}[j = c] + \frac{e^{z_j}}{\sum_k e^{z_k}} = p_j - \mathbf{1}[j = c].
$$

The gradient has a simple shape. For the target, $$p_c - 1 \lt 0$$, so a gradient-descent step raises $$z_c$$. For every other token, $$p_j \gt 0$$, so the step lowers $$z_j$$, and by an amount proportional to $$p_j$$. The components sum to zero. SFT does not add probability to the target out of nowhere: it moves mass from every competitor to the target, and it pushes hardest on the competitor the model currently likes most. The [checks script](https://github.com/hieutrungle/llm-post-training-notes/blob/main/checks/part02_checks.py) confirms the formula against finite differences on random logits.

A worked example with made-up numbers. The problem is "Solve $$x^2 - 5x + 6 = 0$$", and the next step could be *factor*, *use the quadratic formula*, *expand*, *divide by x* or *square both sides*. The demonstration factors. Factoring and the formula are both correct strategies, and the model gives each probability 0.30; the three unhelpful steps get 0.20, 0.12 and 0.08. The gradient on the logits is

$$
p - \mathbf{e}_c = (\,-0.70,\ 0.30,\ 0.20,\ 0.12,\ 0.08\,).
$$

The largest push down, 0.30, lands on the quadratic formula, a valid alternative, simply because it is the strongest competitor. Treating the logits as free parameters and taking one step of size 1 gives probabilities 0.52, 0.19, 0.14, 0.09 and 0.06: the formula loses 0.11, expand loses 0.06. After four such steps, factoring has 0.81 and the formula 0.07. In a real network the logits are not free parameters, and an update made on one prefix also moves the outputs on others through shared weights, but the direction of the per-position signal is exactly this one. The animation follows the same mass from one token to a whole trajectory and then to the mix of correct strategies.

<div class="anim" data-anim="part-02-reshaping" data-alt="Animation in three panels. Panel 1: five next-token bars for the step after 'Solve x squared minus 5x plus 6 equals 0'. Over four SFT gradient steps the demonstrated token 'factor' rises from 0.30 to 0.81 while every other token falls, including the also-correct 'formula' from 0.30 to 0.07. Panel 2: four decisions along the demonstrated solution rise from 0.5, 0.4, 0.5, 0.5 to 0.8, 0.8, 0.9, 0.9, and the trajectory probability on a log scale moves from 0.05 to 0.5184, 10.4 times higher. Panel 3: probability of two correct strategies, A demonstrated and B alternative, goes from 0.20 and 0.20 to 0.70 and 0.05; correct mass rises from 0.40 to 0.75 and the effective number of strategies falls from 2.00 to 1.28."><noscript><img src="{{ '/assets/img/llm-post-training/part-02-reshaping.png' | relative_url }}" alt="Final frame of the SFT reshaping animation: the demonstrated token at 0.81 and the valid alternative at 0.07; the demonstrated trajectory at 0.5184 instead of 0.05; strategy A at 0.70 and strategy B at 0.05, effective number of strategies 1.28 instead of 2.00."></noscript></div>

The three panels show one SFT story at three scales, with illustrative numbers. Dashed outlines mark where each bar started; use Step to jump between the panels.

- **One token.** At the step where the two correct strategies split, each gradient step raises *factor* and lowers all four other tokens. *Formula* falls fastest at first because it is the largest competitor: 0.30 → 0.19 after one step, 0.07 after four.
- **One trajectory.** Along the demonstrated solution, four decisions rise from 0.5, 0.4, 0.5, 0.5 to 0.8, 0.8, 0.9, 0.9. No single decision gains more than a factor of 2, yet the trajectory becomes 10.4 times more likely, as the next section derives.
- **All solutions.** Over whole solutions, the demonstrated strategy A rises from 0.20 to 0.70 and the alternative B falls from 0.20 to 0.05. The model is right more often (0.40 → 0.75), but it now has fewer ways of being right: the effective number of strategies drops from 2.00 to 1.28.

The takeaway is that SFT is a redistribution. It is easy to read the loss as "make the right answer more likely"; the gradient says "make *this* answer more likely, at the expense of every other one, correct or not."

## Small local gains compound along a trajectory

Because a trajectory's probability is a product, modest changes at each step multiply. Take a path of four decisions with local probabilities 0.5, 0.4, 0.5 and 0.5, so

$$
\pi(\tau) = 0.5 \times 0.4 \times 0.5 \times 0.5 = 0.05.
$$

After SFT the same decisions have probabilities 0.8, 0.8, 0.9 and 0.9, and

$$
\pi(\tau) = 0.8 \times 0.8 \times 0.9 \times 0.9 = 0.5184.
$$

The largest local gain is a factor of 2, but the trajectory is 10.4 times more likely, and it moved from "seen once in twenty samples" to "seen in half of them". The average negative log-likelihood per decision fell from 0.75 to 0.16 nats. This is the mechanism behind sharpening in Part 1's sense: a solution that was already accessible becomes the default.

The same arithmetic makes raw sequence probabilities shrink fast with length. With per-token probability 0.9,

$$
0.9^{10} \approx 0.349, \qquad 0.9^{100} \approx 2.66 \times 10^{-5}.
$$

A long response has a tiny probability even when every token is predicted well. That is why language models are scored per token: the average negative log-likelihood $$\bar\ell = -\frac{1}{T}\sum_t \log p_t$$ and the perplexity $$e^{\bar\ell}$$. Both responses above have $$\bar\ell = -\ln 0.9 \approx 0.105$$ and perplexity $$1/0.9 \approx 1.111$$. These normalized scores measure predictive fit. They say nothing about whether the response is correct.

Length enters training through a choice that is easy to overlook: how the loss is normalized. With per-token losses $$\ell_{i,t}$$ for response $$i$$ of length $$T_i$$ in a batch of $$N$$ responses, three common choices are

$$
\underbrace{\sum_{i}\sum_{t} \ell_{i,t}}_{\text{sum}}, \qquad
\underbrace{\frac{1}{N}\sum_{i}\frac{1}{T_i}\sum_{t} \ell_{i,t}}_{\text{per-sequence mean}}, \qquad
\underbrace{\frac{1}{\sum_i T_i}\sum_{i}\sum_{t} \ell_{i,t}}_{\text{token mean}}.
$$

They weight responses differently. Take a batch with one 100-token and one 1,000-token response. Under the sum, every token counts equally, so the long response carries 91% of the gradient. The token mean is the sum scaled by 1/1,100: within the batch it has the same proportions, but the scale changes with the total length of each batch. The per-sequence mean gives each response the same total weight, so each token of the short response gets weight 0.005 and each token of the long one 0.0005: a token in the short response counts 10 times as much. None of these is wrong, but they are different objectives, and the choice is easy to miss because it lives in a single line of training code.

## Local supervision, sequence-level correctness

Training minimizes a sum of local terms, one per token. Evaluation applies a verifier $$V(x, \tau)$$ to the finished trajectory and only then decides whether it was correct. These are different targets:

$$
\text{token-level likelihood of the demonstrations} \;\neq\; \text{sequence-level correctness}.
$$

Most tokens in a solution are routine: restating the problem, algebra that follows from the previous line, formatting. A few are decisive: which method to use, which case to check, when to stop. The loss weights them all the same, and the decisive ones are rare. A section further down shows how this lets the loss improve while the model gets worse at the decision that matters.

## Teacher forcing and exposure bias

During SFT, the model predicts token $$y_t$$ from the demonstrated prefix $$(x, y_1, \ldots, y_{t-1})$$, whatever it would have generated itself. This is called *teacher forcing*. During generation, the model conditions on its own previous tokens. Call the distribution of prefixes in the demonstrations $$d_{\text{data}}(s)$$ and the distribution of prefixes the model visits when it generates $$d_{\pi_\theta}(s)$$. SFT minimizes the loss in expectation over $$d_{\text{data}}$$, and nothing forces $$d_{\pi_\theta} = d_{\text{data}}$$. After one early mistake, the model is in a prefix that no demonstration contains, and SFT has given it no signal for what to do there.

Ranzato et al. [1] named this mismatch *exposure bias*: the model is exposed only to the data distribution, never to its own predictions, so errors accumulate along the generated sequence. Bengio et al. [2] had described the same discrepancy and proposed *scheduled sampling*, which gradually replaces the true previous token with one sampled from the model during training. In imitation learning the problem has a sharp form. Ross, Gordon and Bagnell [3] note that a policy which errs with probability $$\varepsilon$$ on the expert's states can make as many as $$T^2\varepsilon$$ mistakes over $$T$$ steps on the states it visits itself, and the bound is tight. Their algorithm, DAgger, collects states from the learner's own rollouts, has the expert label them, and retrains on the aggregate; its guarantee is back to linear in $$T$$.

A toy model shows where the quadratic comes from. At every step on the demonstrated path the model errs with probability $$\varepsilon = 0.001$$, a made-up number. Under teacher forcing the next prefix is the demonstration's again, so mistakes never propagate. Generating freely, a mistake moves the model off the path, and from there it either recovers with some probability per step or keeps making mistakes.

![Log–log plot of the expected number of mistakes in a response against its length T, from 10 to 4,000 tokens, all with per-step error rate 0.001. Teacher forcing gives the straight line T times epsilon, 1 mistake at T = 1,000. Free-running generation that recovers with probability 0.1 per step gives 9.8 mistakes at T = 1,000. Free-running generation that never recovers follows the dashed T squared epsilon over 2 line, 4.9 mistakes at T = 100 and 368 at T = 1,000.]({{ '/assets/img/llm-post-training/part-02-exposure.png' | relative_url }})

The plot shows the expected number of mistakes in a response of length $$T$$, on log–log axes, for the same per-step error rate in three settings. The blue line is what teacher-forced training and validation see: each step is scored from a correct prefix. The purple and orange lines are free-running generation, with and without a way back to the demonstrated path; the dashed line is $$T^2\varepsilon/2$$.

Read the curves:

- **Teacher forcing** grows linearly: 0.1 expected mistakes at $$T = 100$$, 1 at $$T = 1{,}000$$.
- **Free-running, never recovers** follows the quadratic while $$T\varepsilon$$ is small: 4.9 mistakes at $$T = 100$$ against $$T^2\varepsilon/2 = 5$$, and 368 at $$T = 1{,}000$$, where it starts to bend because a response cannot contain more mistakes than tokens.
- **Free-running, recovers with probability 0.1 per step** is linear again, but about ten times higher: 9.8 mistakes at $$T = 1{,}000$$.

The per-step error rate that validation measures is the same in all three cases. What differs is whether the model has learned to get back on track, and teacher-forced data contains no examples of that. Reasoning traces run to thousands of tokens, which is where this gap matters. Note also that DAgger's fix needs an expert to label the learner's own states. Reinforcement learning with verifiable rewards, as described in Part 0, also trains on $$d_{\pi_\theta}$$, the model's own states, but it replaces the expert's label with a reward for the finished trajectory.

## The population view: SFT minimizes forward KL

Write the SFT objective as an expectation over the data distribution. For a fixed prompt,

$$
\mathbb{E}_{y \sim q_{\text{data}}}\big[-\log \pi_\theta(y \mid x)\big]
= H(q_{\text{data}}, \pi_\theta)
= H(q_{\text{data}}) + D_{\mathrm{KL}}\big(q_{\text{data}} \,\Vert\, \pi_\theta\big),
$$

which follows from adding and subtracting $$\sum_y q_{\text{data}}(y) \log q_{\text{data}}(y)$$. The entropy $$H(q_{\text{data}})$$ does not depend on $$\theta$$, so

$$
\arg\min_\theta \mathcal{L}_{\text{SFT}} = \arg\min_\theta D_{\mathrm{KL}}\big(q_{\text{data}} \,\Vert\, \pi_\theta\big).
$$

Maximum likelihood is forward KL from the data to the model. Its term for each response is $$q_{\text{data}}(y)\log\big(q_{\text{data}}(y)/\pi_\theta(y)\big)$$, which blows up if the model gives near-zero probability to anything the data contains. So forward KL is *mode-covering*: the model must keep every mode that appears in the data. For example, if the demonstrations use strategy A 90% of the time and B 10%, a model that shrinks B from 0.1 to 0.01 pays 0.144 nats of forward KL. The reverse direction, $$D_{\mathrm{KL}}(\pi_\theta \Vert q_{\text{data}})$$, would charge only 0.071 for the same change.

The covering is relative to the data, not to the set of correct solutions. A response with $$q_{\text{data}}(y) = 0$$ contributes nothing to the KL, whether it is correct or not. Probability placed there is simply missing from the demonstrated responses: if a model moves 0.3 of its mass onto a valid but undemonstrated strategy C, the loss rises by $$-\ln 0.7 \approx 0.357$$ nats. A sufficiently flexible model therefore drives every undemonstrated solution toward zero. SFT covers the modes it is shown and erodes the ones it is not.

## SFT can reduce reasoning diversity

This is where the diversity measure of Part 1 comes in. Suppose two valid strategies have equal probability before SFT, $$P(A) = 0.20$$ and $$P(B) = 0.20$$, and the demonstrations always use A. After SFT the model might have $$P(A) = 0.70$$ and $$P(B) = 0.05$$, as in the third panel of the animation. Accuracy improves, from 0.40 to 0.75. Computed over the two strategy families, $$N_{\text{eff}} = e^{H_{\text{correct}}}$$ falls from 2 to 1.28: A now holds 93% of the correct mass.

Better accuracy and a richer reasoning distribution are different things, and SFT on single demonstrations tends to trade the second for the first. Whether that matters depends on what comes next. For RL that learns from the model's own samples, the strategies the model no longer produces are strategies it can no longer be rewarded for.

## Not memorization, and not proof of reasoning either

It is tempting to dismiss SFT as memorization. It is more than that: parameters are shared across inputs, so an update on one solution changes the model's behavior on problems it never saw. A gain in held-out accuracy after SFT can come from three different things:

1. memorization of training solutions, which helps only on near-duplicates;
2. sharpening of a behavior the model already had;
3. operational expansion, in Part 1's sense: a behavior that was inaccessible under the inference budget becomes reliably accessible on held-out problems.

All three raise accuracy, so accuracy alone cannot tell them apart. Part 1's ladder of evidence applies to SFT as much as to RL.

## A lower loss does not mean better reasoning

Because the loss averages over tokens, a model can improve it on many easy tokens while getting worse at the one decision that matters. A made-up example: a 100-token solution has 99 routine tokens, each predicted with probability 0.95, and one decisive step (say, which case to check) predicted with probability 0.6. After some training the routine tokens are at 0.97 and the decisive step has dropped to 0.4. Then

- the average negative log-likelihood falls from 0.0559 to 0.0393 nats per token, 30% lower;
- the probability of this exact demonstrated trajectory rises from 0.0037 to 0.0196;
- but if every wording of the routine tokens is acceptable, the probability of solving the problem is set by the decisive step, and it falls from 0.6 to 0.4.

Validation loss, the likelihood of the demonstration and the success rate can move in different directions. The experiment below shows a real case.

## Trajectory probability is not answer probability

A correct answer can be reached by many trajectories, so the probability of success is a sum,

$$
p_\theta(x) = \sum_{\tau \in \mathcal{S}(x)} \pi_\theta(\tau \mid x),
$$

while SFT works on one demonstrated trajectory at a time. The two objectives are

$$
\text{SFT:}\ \ -\log \pi_\theta(\tau_{\text{demo}} \mid x)
\qquad\text{vs.}\qquad
\text{success:}\ \ J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta(\cdot \mid x)}\big[R(x, \tau)\big] = p_\theta(x)
$$

for a binary verifier reward $$R$$. SFT says *imitate this solution*. The success objective says *put probability on any solution that is correct*. Moving mass from one correct trajectory to another changes the first and leaves the second unchanged.

The gradient of the success objective makes the difference concrete. Using $$\nabla \pi = \pi \nabla \log \pi$$,

$$
\nabla_\theta \log p_\theta(x) = \frac{1}{p_\theta(x)} \sum_{\tau \in \mathcal{S}(x)} \pi_\theta(\tau \mid x)\, \nabla_\theta \log \pi_\theta(\tau \mid x) = \mathbb{E}_{\tau \sim q(\cdot \mid x)}\big[\nabla_\theta \log \pi_\theta(\tau \mid x)\big],
$$

where $$q(\tau \mid x)$$ is Part 1's distribution of the model's own successful trajectories. So the log-likelihood gradient *on the model's own correct samples* is an unbiased estimate of the gradient of $$\log p_\theta(x)$$, the log of the success mass. The checks script verifies this identity on a small autoregressive model. Two conditions matter. The samples must come from the current model as it is, not from a sharpened version of it, and a problem contributes only if at least one sample succeeds. **RL cannot reward a success it never samples**, and neither can this kind of self-training.

<div class="note" markdown="1">
<span class="label">In my experiments</span>

In [sharpen-or-expand](https://github.com/hieutrungle/sharpen-or-expand), the SFT arm trains the base model on its own verified solutions. This is rejection-sampling fine-tuning, the recipe behind STaR [4] and Yuan et al. [5], in its simplest form: one round, and no rationalization step that shows the model the answer when it fails. Qwen3.5-4B-Base drew 4 samples on each of 16,000 training problems with the evaluation sampler (temperature 0.6), a verifier kept at most two correct solutions per problem, and 2,224 problems without a kept solution dropped out ([data report](https://github.com/hieutrungle/sharpen-or-expand/blob/main/docs/reports/m2_sft_data.json)). Training is [one epoch, with the loss on response tokens only](https://github.com/hieutrungle/sharpen-or-expand/blob/main/docs/decisions/029-sft-recipe-and-run2-rule.md). A first run on 25,551 examples failed two checks fixed before training: it truncated more often than the base model and lowered pass@1 by more than two standard errors. The pre-registered repair, a third of the learning rate and without the longest 10% of solutions (22,996 examples), failed the same two checks less badly, and its final weights are the [SFT checkpoint evaluated below](https://github.com/hieutrungle/sharpen-or-expand/blob/main/docs/decisions/030-frozen-sft-export.md).

The validation loss on held-out verified solutions kept falling during training: 0.2020, 0.1964 and 0.1959 at the three checkpoints ([report](https://github.com/hieutrungle/sharpen-or-expand/blob/main/docs/reports/m2_sft_run2_check_run2.json)). Generation did not improve. On 256 held-out problems with 16 samples each ([diagnostic report](https://github.com/hieutrungle/sharpen-or-expand/blob/main/docs/reports/m2_sft_diagnostic_run2.json)):

| | Base model | After SFT |
|---|---|---|
| pass@1 | 0.660 | 0.637 |
| pass@16 | 0.852 | 0.855 |
| responses that hit the 16,384-token limit | 32.1% | 34.5% |
| responses that end in a repetition loop | 5.1% | 10.4% |
| accuracy of finished responses | 97.2% | 97.2% |
| median length of finished responses, tokens | 3,884 | 3,348 |

A loop here is a response whose last 4,000 characters compress to less than 15% of their size, a [heuristic](https://github.com/hieutrungle/sharpen-or-expand/blob/main/scripts/sft_diagnostic.py) for repetition. Paired over problems, pass@1 fell by 0.023 (standard error 0.008), and the pass@16 change is well inside its standard error of 0.012. A truncated response scores zero, so pass@1 is the stop rate times the accuracy of finished responses, and its change splits exactly into two parts: −0.0230 from stopping less and +0.0003 from accuracy. The 500 held-out problems behind the validation loss show the same split: pass@1 0.868 → 0.849, truncation 7.8% → 10.4%, −0.025 from stopping, +0.005 from accuracy.

The full evaluation suite agrees ([comparison](https://github.com/hieutrungle/sharpen-or-expand/blob/main/docs/reports/sft_qwen35-4b_v1_compare_base_v4.json), [transitions](https://github.com/hieutrungle/sharpen-or-expand/blob/main/docs/reports/sft_qwen35-4b_v1_transitions_base_v4.json)). pass@1 went from 0.744 to 0.733 on 1,224 public benchmark problems and from 0.673 to 0.650 on 982 held-out problems from the RL training distribution, while pass@16 moved only from 0.856 to 0.853 and from 0.873 to 0.877, within their standard errors. Problem by problem, the changes at $$k = 16$$ are balanced: 19 lost and 15 gained on the benchmarks, 20 lost and 24 gained on the held-out problems, and on the 233 hard problems at $$k = 64$$, 16 lost and 15 gained.

What this shows:

- **Lower loss, worse generation.** The loss on held-out solutions went down while pass@1 on the same problems went down. The base model's own validation loss is not reported, so the comparison is with the loss early in training, not with the base.
- **The change is in stopping, not in solving.** Finished answers are exactly as accurate and shorter; more responses never finish. SFT neither narrowed nor widened what the model reaches in 16 samples.
- **Two explanations fit, and neither has been tested.** One is exposure bias. The training data contain only finished, correct solutions, so a prefix that has started to repeat never appears in $$d_{\text{data}}$$, and SFT teaches nothing about leaving it. Among truncated responses, the share that loops rose from 16% to 30%. The other is sharpening a distribution that was already sharpened. The solutions were sampled at temperature 0.6, so $$q_{\text{data}}$$ is sharper than the model's own distribution, and the derivation above no longer holds: maximum likelihood pulls $$\pi_\theta$$ toward the sharper distribution, and evaluation at temperature 0.6 sharpens it again. Decoding that favors high-probability text is known to get stuck in repetitive loops [6]. The first run, at three times the learning rate, had the larger effect (pass@1 down 0.052, loops 11.2%), but it also kept the longest solutions, so the comparison does not separate the causes.
- **In context.** STaR [4] and Yuan et al. [5] report gains from training on self-generated correct solutions, and Yuan et al. find that the gains shrink for stronger models. A strong base model, with pass@1 0.660 on these problems, gaining nothing in accuracy fits that pattern.
</div>

## Check yourself

**1.** Before SFT, three correct strategies have probabilities $$P(A) = 0.20$$, $$P(B) = 0.10$$ and $$P(C) = 0.05$$. After SFT they have 0.70, 0.03 and 0.01. What suggests sharpening, and what undesirable property changed?

<details markdown="1">
<summary>Answer</summary>

Strategy A was already accessible at 0.20 and became dominant at 0.70: that is sharpening, and it raised the correct mass from 0.35 to 0.74. The undesirable change is diversity. Over the three strategies, $$N_{\text{eff}}$$ falls from 2.60 to 1.27, and A now holds 95% of the correct mass. B and C, the fallbacks when A does not work, have almost disappeared, and a later RL stage can only reward strategies the model still samples.
</details>

**2.** How can a model have lower SFT validation loss but lower math pass@1?

<details markdown="1">
<summary>Answer</summary>

Five reasons:

1. Most tokens are not decisive reasoning steps, so gains on easy tokens can dominate the average loss.
2. SFT optimizes the likelihood of demonstrations, not correctness.
3. Teacher-forced validation conditions on demonstration prefixes; generation conditions on the model's own, so validation never sees the model's mistakes or how it recovers from them.
4. Small changes at a few critical decisions multiply along the sequence.
5. Frequent, easy patterns dominate the loss while rare behavior, such as when to stop, gets worse.

The SFT run in the box above is a real instance: validation loss fell from 0.2020 to 0.1959 during training while pass@1 fell from 0.660 to 0.637, all of it from responses that stopped less often.
</details>

**3.** A reasoning trajectory goes from probability $$10^{-8}$$ before SFT to $$10^{-2}$$ after. Did the model learn a new reasoning algorithm?

<details markdown="1">
<summary>Answer</summary>

Not necessarily, and the probability of one trajectory cannot settle the question. SFT amplified this trajectory by a factor of $$10^6$$, but three things are missing.

- **Was it ever reachable?** "It already had nonzero probability" is true only under untruncated softmax sampling. Under top-p 0.95, the evaluation sampler of the experiments, a trajectory of probability $$10^{-8}$$ may contain tokens outside the nucleus and then had probability exactly zero at evaluation.
- **One trajectory is not accessibility.** A solution can be worded in thousands of ways, so the strategy's total success mass may have been far larger than $$10^{-8}$$ before. What matters is the success mass $$p_\theta(x; c)$$ at a sampling budget. If the strategy's mass was really near $$10^{-8}$$, it was inaccessible: its pass@1024 is about $$1.02 \times 10^{-5}$$. If its mass is now near 0.01, pass@64 is 47%. That change would be operational expansion at the strategy level, but it has to be measured on the mass, not inferred from one trajectory.
- **Imitation is not invention.** The trajectory became likely because a demonstration showed it. Whether the model learned a reusable method needs held-out problems that require it, the upper rungs of Part 1's ladder.
</details>

**4.** What is the difference between minimizing $$-\log \pi_\theta(\tau^* \mid x)$$ and maximizing $$\mathbb{E}_{\tau \sim \pi_\theta}[R(x, \tau)]$$?

<details markdown="1">
<summary>Answer</summary>

The first says: make this demonstrated trajectory likely, and take the mass from everything else, correct alternatives included. The second says: increase the probability of any sampled trajectory that earns reward. It can reinforce correct solutions that never appeared in a demonstration, as long as the policy samples them. In expectation it raises every correct trajectory: for a softmax over whole trajectories with logits $$z_\tau$$, $$\partial J / \partial z_\tau = \pi_\theta(\tau \mid x)\,\big(R(x, \tau) - J\big)$$, which is positive for every correct $$\tau$$ when $$J \lt 1$$. The catch is the sampling: with finitely many samples, the correct trajectories that happen to be drawn go up and the ones that are not can go down, and a success the policy never produces gets no reward and no gradient.
</details>

## Takeaways

- SFT is conditional behavior cloning: maximum likelihood of demonstrations, or token-level cross-entropy. It never measures correctness.
- The gradient $$p_j - \mathbf{1}[j = c]$$ redistributes mass. Every competitor is pushed down in proportion to its probability, so the strongest alternative, often a valid one, is pushed hardest.
- Local gains compound: 0.05 → 0.5184 from per-step gains of at most 2×. How the loss is normalized (sum, per-sequence mean, token mean) decides how long and short responses are weighted.
- Teacher forcing trains on $$d_{\text{data}}$$, generation runs on $$d_{\pi_\theta}$$. Errors the model never learned to recover from can grow quadratically with length.
- SFT minimizes $$D_{\mathrm{KL}}(q_{\text{data}} \Vert \pi_\theta)$$. It covers the modes in the data and erodes valid modes outside it, so accuracy can rise while $$N_{\text{eff}}$$ falls, and validation loss can fall while pass@1 falls.
- Imitating one solution is not maximizing the success mass. Training on the model's own successes estimates $$\nabla \log p_\theta(x)$$, but only if the samples come from the model itself, and only on problems where it samples a success.

## Next

Part 3 turns from demonstrations to comparisons. Direct preference optimization (DPO) learns from pairs of responses, one preferred over the other, and it comes from a KL-regularized RL objective with a frozen reference policy $$\pi_{\text{ref}}$$. Part 3 derives it, explains what $$\beta$$ controls, and shows why DPO can lower the probability of the preferred response.

## References

1. M. Ranzato, S. Chopra, M. Auli, and W. Zaremba, "Sequence level training with recurrent neural networks," in *Proc. ICLR*, 2016. [arXiv](https://arxiv.org/abs/1511.06732)
2. S. Bengio, O. Vinyals, N. Jaitly, and N. Shazeer, "Scheduled sampling for sequence prediction with recurrent neural networks," in *Proc. NeurIPS*, 2015. [arXiv](https://arxiv.org/abs/1506.03099)
3. S. Ross, G. J. Gordon, and J. A. Bagnell, "A reduction of imitation learning and structured prediction to no-regret online learning," in *Proc. AISTATS*, 2011. [arXiv](https://arxiv.org/abs/1011.0686)
4. E. Zelikman, Y. Wu, J. Mu, and N. D. Goodman, "STaR: Bootstrapping reasoning with reasoning," in *Proc. NeurIPS*, 2022. [arXiv](https://arxiv.org/abs/2203.14465)
5. Z. Yuan, H. Yuan, C. Li, G. Dong, K. Lu, C. Tan, C. Zhou, and J. Zhou, "Scaling relationship on learning mathematical reasoning with large language models," arXiv:2308.01825, 2023. [arXiv](https://arxiv.org/abs/2308.01825)
6. A. Holtzman, J. Buys, L. Du, M. Forbes, and Y. Choi, "The curious case of neural text degeneration," in *Proc. ICLR*, 2020. [arXiv](https://arxiv.org/abs/1904.09751)
