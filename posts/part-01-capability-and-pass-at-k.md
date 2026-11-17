---
title: "What Does \"Capability\" Mean for an LLM?"
kind: Study notes
date: 2026-11-16 01:00:00 -0600
series: llm-post-training
part: 1
summary: "Before asking whether RL sharpens or expands what a language model can do, we need a definition of \"can do\". Part 1 treats capability as a success probability under a sampling budget, derives pass@k and its unbiased estimator, and shows why zero successes never prove zero capability."
math: true
anim: true
---

*Code, check scripts and references: [llm-post-training-notes](https://github.com/hieutrungle/llm-post-training-notes) · Experiments: [sharpen-or-expand](https://github.com/hieutrungle/sharpen-or-expand)*

<div class="goal" markdown="1">
<span class="label">Core idea</span>

A model does not simply have or lack a capability. On each problem it has a probability of success, $$p_\theta(x)$$, under fixed inference conditions, and what we can observe depends on how many samples we draw. **Sharpening** raises $$p$$ where it was already measurable. **Operational expansion** moves $$p$$ from below what our sampling budget can detect to reliably above it.
</div>

## The question

[Part 0](/blog/2026/11/llm-post-training-series/) asked whether reinforcement learning with verifiable rewards sharpens what a model can do or expands it. Both readings talk about what a model *can* solve, and that word hides a problem. A language model never answers "can I solve this?" with yes or no. It defines a distribution over responses, and the same model can solve a problem on one sample and fail it on the next.

So before comparing models, this part fixes what we measure. What does "can solve" mean for a stochastic model? How does pass@k turn it into a number, and how do we estimate that number from a finite set of samples? And what evidence would a careful claim of expansion need?

## A model is a distribution over trajectories

For a prompt $$x$$, an autoregressive model generates a response $$\tau = (y_1, \ldots, y_T)$$ one token at a time:

$$
\pi_\theta(\tau \mid x) = \prod_{t=1}^{T} \pi_\theta(y_t \mid x, y_{\lt t}).
$$

A verifier splits the complete trajectories into successes $$\mathcal{S}(x)$$, whose final answer is correct, and failures $$\mathcal{F}(x)$$. What matters for capability is the total probability on the success set:

$$
p_\theta(x) = \Pr_{\tau \sim \pi_\theta(\cdot \mid x)}\big[\tau \in \mathcal{S}(x)\big] = \sum_{\tau \in \mathcal{S}(x)} \pi_\theta(\tau \mid x).
$$

This single-sample success probability is the basic unit of the whole series. Every metric in this part is a function of $$p_\theta(x)$$, problem by problem.

One refinement matters later. $$p_\theta(x)$$ is not a property of the weights alone. The same weights give different success probabilities under a different prompt, temperature, top-p, length limit, tool set or sampling procedure. Collect these inference conditions in $$c$$ and write $$p_\theta(x; c)$$. A capability claim is always about a model *and* the conditions it was run under.

## Why "nonzero probability" is not a useful definition

A natural first definition says the model can solve $$x$$ if some successful trajectory has nonzero probability:

$$
\exists\, \tau \in \mathcal{S}(x): \ \pi_\theta(\tau \mid x) \gt 0.
$$

Under pure softmax sampling, at a positive temperature and with no truncation, this is true almost by construction. Every token has positive probability at every step, so every finite sequence of tokens has positive probability, including a correct solution produced by chance. A correct trajectory with probability $$10^{-100}$$ is in the support, yet no feasible amount of sampling will ever show it. The support cannot tell a model that solves a problem every time from one that never will in practice.

Truncated sampling breaks the definition in a different way. Top-k sampling keeps only the $$K$$ most likely tokens at each step [1], and top-p (nucleus) sampling keeps the smallest set of tokens whose cumulative probability reaches a threshold $$p_{\text{top}}$$ [2]. These are sampler settings, unrelated to the number of samples $$k$$ and the success probability $$p$$ used in the rest of this part. Both set every other token's probability to exactly zero, so the support becomes finite and depends on the sampler. A solution can be in the support at temperature 1.0 with no truncation and outside it at top-p 0.95. This matters because RL rollouts and evaluation often use different sampling settings, as the experiment below does. The set of solutions a model can reach during training is then not the same set it can reach at evaluation.

So the support is either too large (softmax) or a property of the sampler (truncation). Neither answers the question we actually care about, which is operational:

<div class="note" markdown="1">
How likely is the model to produce a successful trajectory under fixed inference conditions and a fixed sampling budget?
</div>

One could try to make "accessible" precise with a threshold on single trajectories, $$\mathcal{A}_\epsilon = \{\tau : \pi_\theta(\tau \mid x) \ge \epsilon\}$$. That misses the point. A long solution can be correct in thousands of wordings, each of them individually improbable, so a problem the model solves most of the time can have no single trajectory above the threshold. What makes a solution accessible is the total mass $$p_\theta(x; c)$$, together with the number of samples we are willing to draw. pass@k combines the two.

## pass@1 and pass@k

For one problem with success probability $$p$$, a single sample succeeds with probability $$p$$:

$$
\text{pass@}1 = p.
$$

Now draw $$k$$ samples independently, under the same conditions. One sample fails with probability $$1 - p$$. Because the samples are independent, all $$k$$ fail with probability $$(1-p)^k$$, and at least one succeeds with probability

$$
\text{pass@}k = 1 - (1 - p)^k.
$$

The formula assumes that the $$k$$ samples are independent draws from the same distribution, as in ordinary repeated sampling; beam search, or samples that continue a shared prefix, break that assumption. For example, a problem with $$p = 0.1$$ has $$\text{pass@}8 = 1 - 0.9^8 \approx 0.570$$ and $$\text{pass@}32 = 1 - 0.9^{32} \approx 0.966$$. On a benchmark, pass@k is this per-problem value averaged over problems. The next section shows why the averaging matters.

How many samples does a rare success need? Setting $$\text{pass@}k = 1/2$$ and solving gives $$k = \ln 2 \,/\, \big({-\ln(1-p)}\big) \approx 0.69/p$$ for small $$p$$. To see a success of probability $$p$$ in about half of all evaluation runs, draw about $$0.7/p$$ samples per problem: 2 samples for $$p = 0.4$$, 7 for $$p = 0.1$$, 14 for $$p = 0.05$$, 69 for $$p = 0.01$$ and 693 for $$p = 0.001$$.

The table gives the exact values for five problems. The animation below shows where those numbers come from.

| Success probability $$p$$ | pass@1 | pass@4 | pass@16 | pass@64 | pass@256 | pass@1024 |
|---|---|---|---|---|---|---|
| 0.4 | 40.0% | 87.0% | >99.9% | >99.9% | >99.9% | >99.9% |
| 0.1 | 10.0% | 34.4% | 81.5% | 99.9% | >99.9% | >99.9% |
| 0.05 | 5.0% | 18.5% | 56.0% | 96.2% | >99.9% | >99.9% |
| 0.01 | 1.0% | 3.9% | 14.9% | 47.4% | 92.4% | >99.9% |
| 0.001 | 0.1% | 0.4% | 1.6% | 6.2% | 22.6% | 64.1% |

<div class="anim" data-anim="part-01-microscope" data-alt="Animation of 20 simulated evaluation runs for each of five problems with success probability 0.4, 0.1, 0.05, 0.01 and 0.001. As the number of samples per problem k grows from 1 to 1024 on a log scale, each run's row turns from grey to blue at its first correct sample, and the blue share of each lane follows the exact pass@k curve. At k = 1024 the p = 0.001 problem is solved in 12 of 20 runs, and the exact pass@1024 is 64.1%."><noscript><img src="{{ '/assets/img/llm-post-training/part-01-microscope.png' | relative_url }}" alt="Final frame of the pass@k animation: at k = 1024 the four common problems are solved in every simulated run and the p = 0.001 problem in 12 of 20 runs; exact pass@1024 is 64.1%."></noscript></div>

Each lane is one problem, and each thin row in a lane is one simulated evaluation run that draws samples one at a time. A row stays grey until its first correct sample and turns blue from that point on. The rows are sorted by when they first succeed, so the blue share of a lane at the cursor is the share of runs that have found a solution within $$k$$ samples: a simulated pass@k. The dark line is the exact $$1-(1-p)^k$$. The runs use a fixed random seed, chosen so that the simulated shares stay within two runs of the exact values; use Step to jump between $$k$$ = 1, 16, 64 and 1024.

Read the lanes from left to right:

- **At k = 1** the lanes show pass@1. The problem with $$p = 0.4$$ has an exact pass@1 of 40.0%, and 9 of its 20 simulated runs succeed on the first sample. The rarer problems are almost never solved.
- **At k = 16**, $$p = 0.1$$ and $$p = 0.05$$ are solved in most runs (81.5% and 56.0%), while $$p = 0.01$$ is still rare (14.9%).
- **At k = 64**, $$p = 0.01$$ is solved in about half the runs (47.4%). The problem with $$p = 0.001$$ still looks unsolvable at 6.2%.
- **At k = 1024**, even $$p = 0.001$$ is solved in most runs (64.1%). In the simulation, 12 of its 20 runs found a correct answer and 8 saw none in 1,024 samples, although the problem's success probability is 0.001.

Large $$k$$ works like a microscope: it reveals successes too rare to appear in a few samples. But no value of $$k$$ draws a sharp line between "can" and "cannot". The curves have the same shape and slide to the right as $$p$$ falls. A problem with $$p = 0.001$$ has $$\text{pass@}64 \approx 6.2\%$$ and $$\text{pass@}1000 \approx 63.2\%$$: it looks unsolvable at one budget and mostly solvable at another. "Solvable" is a statement about a budget, and the model itself is better described as a probability landscape than as a boundary.

## Same pass@1, different breadth

For one problem, pass@k depends only on $$p$$. Two models with the same $$p = 0.4$$ on a problem have the same $$\text{pass@}k = 1 - 0.6^k$$ for every $$k$$, whether one of them always uses a single strategy and the other spreads its successes over ten. pass@k sees the total mass on the success set, not how that mass is organized. The section on reasoning diversity comes back to this.

On a benchmark the picture changes, because benchmark pass@k averages a nonlinear function of $$p$$ over problems. Take two problems and two models with the same average pass@1 of 0.4:

- **Model A** has $$p = 0.4$$ on both problems. Its pass@2 is $$1 - 0.6^2 = 0.64$$ on each, so the average is 0.64.
- **Model B** has $$p = 0.8$$ on the first problem and $$p = 0$$ on the second. Its pass@2 is $$1 - 0.2^2 = 0.96$$ and 0, so the average is 0.48.

The reason is the shape of $$f_k(p) = 1 - (1-p)^k$$. Its second derivative is

$$
f_k''(p) = -k(k-1)(1-p)^{k-2} \lt 0 \quad \text{for } k \ge 2,\ 0 \le p \lt 1,
$$

so $$f_k$$ is concave, and Jensen's inequality gives $$\frac{1}{N}\sum_i f_k(p_i) \le f_k\big(\frac{1}{N}\sum_i p_i\big)$$, with equality only when all $$p_i$$ are equal. For a fixed average pass@1, benchmark pass@k is largest when success is spread evenly across problems and smaller when it is concentrated on a subset.

![Left: the concave curve pass@2 = 1 − (1 − p)^2. Model A's two problems both sit at p = 0.4 on the curve, at pass@2 = 0.64. Model B's problems sit at p = 0 and p = 0.8; their average, 0.48, lies on the straight chord between them, below the curve. Right: benchmark pass@k against k from 1 to 64. Both models start at 0.4; Model A rises to nearly 1 by k = 16, Model B levels off at 0.5.]({{ '/assets/img/llm-post-training/part-01-breadth.png' | relative_url }})

The left panel plots pass@2 of a single problem against its success probability $$p$$, with the two toy models' problems marked on it. The right panel plots each model's benchmark pass@k, the average over its two problems, against $$k$$.

Read the curves:

- **Left:** Model A's point lies on the curve at 0.64. Model B's two problems lie at the ends of the dashed chord, and their average (open square, 0.48) lies on the chord, below the curve. The vertical gap between them is Jensen's inequality.
- **Right:** both models start at 0.4 at $$k = 1$$. By $$k = 8$$, Model A reaches 0.983 while Model B is at 0.500, and Model B can never pass 0.5, because one of its problems has $$p = 0$$.

So pass@1 measures average reliability, and pass@k at large $$k$$ measures breadth: on how many problems the model has a success probability that is not negligible. A model that concentrates its success on fewer problems can raise pass@1 and still lose at large $$k$$. That is the crossover in Part 0's figure, and it is why the sharpen-or-expand question needs both ends of the pass@k curve.

## Sharpening vs operational expansion

Here is sharpening on one problem, with made-up numbers. Before post-training, the model reaches correct answers through two strategies:

| Trajectories | Type | Probability before | Probability after |
|---|---|---|---|
| strategy A | correct | 0.08 | 0.45 |
| strategy B | correct | 0.02 | 0.05 |
| all others | incorrect | 0.90 | 0.50 |

The success probability rises from $$p = 0.10$$ to $$p = 0.50$$, and pass@8 from 0.570 to 0.996. Nothing new appeared: both successful strategies were already there, and training moved probability onto them, mostly onto strategy A. A's share of the successful mass grew from 0.80 to 0.90. This is sharpening: **existing successful peaks get taller.**

Expansion in the literal sense, adding trajectories to the support, is not a useful notion for softmax models, for the reasons in the section on support. The useful definition is operational:

<div class="note" markdown="1">
**Operational expansion:** post-training makes successful behavior that was effectively inaccessible under a fixed, practical inference budget reliably accessible under that budget.
</div>

It comes in two forms:

- **Problem level.** $$p_{\text{before}}(x)$$ is too small to detect at the budget, and $$p_{\text{after}}(x)$$ is large: a problem the model never solved in $$n$$ samples is now solved most of the time.
- **Strategy level.** A family of solutions $$z$$, for example a different method of proof, is essentially never seen before training and carries substantial probability afterwards, even on problems the model could already solve another way.

Both forms rest on an observation made with finitely many samples, and that has a consequence: **newly observed is not the same as newly created.** The model before training may have produced the behavior at a rate too low for our budget to notice. The next section puts a number on how low.

## Zero successes is not zero capability

Suppose a problem shows 0 successes in $$n$$ samples. If the true success probability is $$p$$, this happens with probability

$$
\Pr(0 \text{ successes} \mid p) = (1-p)^n.
$$

A one-sided 95% upper bound on $$p$$ is the largest $$p$$ for which this outcome is still not surprising at the 5% level, $$(1-p)^n = 0.05$$:

$$
p_{\text{upper}} = 1 - 0.05^{1/n}.
$$

For $$n = 64$$ this gives $$p_{\text{upper}} \approx 4.6\%$$. For $$n = 1024$$ it gives about 0.29%. So 0/64 is entirely compatible with a model that solves the problem several percent of the time. A problem with $$p = 0.02$$ shows 0 successes in 64 samples with probability $$0.98^{64} \approx 27\%$$.

For small $$p$$, $$\ln(1-p) \approx -p$$, so the bound becomes $$n\,p_{\text{upper}} \approx -\ln 0.05 = 2.996$$. This is the **rule of three** [3]: after 0 successes in $$n$$ trials, the 95% upper bound on the success probability is about $$3/n$$. It gives 4.7% for $$n = 64$$ and 0.29% for $$n = 1024$$.

![Log–log plot of the 95% upper bound on p after zero successes, against the number of samples n from 8 to 4096. The exact bound and the rule-of-three line 3/n almost coincide. Marked points: 0/16 gives 17%, 0/64 gives 4.6%, 0/1024 gives 0.29%. The shaded region below the curve holds the values of p still compatible with zero successes.]({{ '/assets/img/llm-post-training/part-01-zero-bound.png' | relative_url }})

The plot shows, for each number of samples $$n$$ that all failed, the largest success probability still compatible with that outcome at 95% confidence. The solid line is the exact bound and the dashed line the rule of three; the shaded region below them is what zero successes cannot rule out.

Read the curves:

- **0/16** still allows $$p$$ up to 17% (the rule of three says 19%). Sixteen samples, the main budget in my experiments, cannot reliably tell a problem with $$p = 0.1$$ from one with $$p = 0$$: the first shows 0/16 with probability $$0.9^{16} \approx 18.5\%$$.
- **0/64** allows up to 4.6% (rule of three: 4.7%).
- **0/1024** allows up to 0.29%. From $$n = 64$$ on, the rule of three is within 3% of the exact bound.

Finite sampling bounds $$p$$ from above. It never proves $$p = 0$$. "Unsolved at $$k = 64$$" means "$$p$$ is probably below about 5%". The 8 grey rows at the end of the animation are exactly this situation: 1,024 failures from a problem whose success probability is 0.001.

## Estimating pass@k from n samples

In practice $$p$$ is unknown. We draw $$n$$ samples per problem, count $$c$$ correct ones, and want pass@k for some $$k \le n$$. The estimator of Chen et al. [4] is

$$
\widehat{\text{pass@}k} = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}, \qquad n \ge k.
$$

It has a direct reading. Of the $$\binom{n}{k}$$ ways to choose $$k$$ of the $$n$$ samples, $$\binom{n-c}{k}$$ contain only failures, so the fraction is the chance that a random subset of size $$k$$ fails completely. One minus that fraction estimates pass@k.

The estimator is **unbiased**. Any fixed subset of $$k$$ samples consists of $$k$$ independent draws, so it fails completely with probability $$(1-p)^k$$. The fraction above is the average of the indicator "this subset failed" over all subsets, so its expectation is also $$(1-p)^k$$, and the expectation of the estimate is exactly $$1-(1-p)^k$$ for every $$p$$. It needs $$n \ge k$$, because with fewer samples there are no subsets of size $$k$$ to average over. When $$n - c \lt k$$, every subset contains a success and the estimate is 1. At $$k = n$$ the estimate is 1 if any sample succeeded and 0 otherwise.

A worked example: with $$n = 16$$, $$c = 2$$ and $$k = 8$$, $$\binom{14}{8} = 3003$$ and $$\binom{16}{8} = 12870$$, so the estimate is $$1 - 3003/12870 \approx 0.767$$.

The obvious alternative is to plug $$\hat p = c/n$$ into the formula. Here it gives $$1 - (1 - 2/16)^8 \approx 0.656$$, and it is biased low. $$\hat p$$ is an unbiased estimate of $$p$$, but $$f_k$$ is concave, so by Jensen's inequality again, $$\mathbb{E}[f_k(\hat p)] \le f_k(\mathbb{E}[\hat p]) = f_k(p)$$. For $$p = 0.1$$, $$n = 16$$ and $$k = 8$$, the plug-in averages 0.491, against a true pass@8 of 0.570, while the average of the unbiased estimator equals the true value exactly (0.570 to three decimals). The [checks script](https://github.com/hieutrungle/llm-post-training-notes/blob/main/checks/part01_checks.py) verifies unbiasedness by exact summation over all outcomes for several $$n$$, $$k$$ and $$p$$, and by simulation.

<div class="note" markdown="1">
<span class="label">In my experiments</span>

In [sharpen-or-expand](https://github.com/hieutrungle/sharpen-or-expand), every checkpoint is evaluated with the [same protocol](https://github.com/hieutrungle/sharpen-or-expand/blob/main/docs/decisions/004-evaluation-protocol.md), taken from Yue et al. [5]: temperature 0.6, top-p 0.95, up to 16,384 generated tokens, and the same zero-shot prompt for the base model and every trained model. Each problem gets 16 samples, and a frozen subset of hard problems gets 64. pass@k uses the unbiased estimator above ([`pass_at_k.py`](https://github.com/hieutrungle/sharpen-or-expand/blob/main/soe/utils/metric/pass_at_k.py)).

Changes are also tracked problem by problem ([`transitions.py`](https://github.com/hieutrungle/sharpen-or-expand/blob/main/soe/utils/metric/transitions.py)). A problem counts as solved at $$k$$ when its estimated pass@k is at least 0.5, and each problem is labeled gained, lost, kept or never solved between two checkpoints. At $$k = 1$$ "solved" means solved in at least half of the samples: reliable, the sharpening view. At the largest $$k$$ it means solved at least once: reachable, the expansion view.

The base model, Qwen3.5-4B-Base, shows why the bound in the previous section matters. The hard problems were drawn from those the base model solved at most 4 times in 16 samples, in a [separate selection run](https://github.com/hieutrungle/sharpen-or-expand/blob/main/docs/decisions/005-hard-subset.md) with its own random seed, so that selection does not bias the evaluated pass rates. On the 233 hard problems ([report](https://github.com/hieutrungle/sharpen-or-expand/blob/main/docs/reports/qwen35-4b-base_soe-eval-v4_report.json)), its pass@1 is 0.0695, pass@16 is 0.3621 and pass@64 is 0.4893: 114 problems were solved at least once in 64 samples and 119 never were. Each of those 119 can still have $$p$$ up to about 4.6%. "Unsolved by the base model" means "below about 5% at this budget", and that is the baseline a claim of expansion has to beat.

The inference conditions matter here too. On these problems, 90% of the base model's responses reach the 16,384-token limit and count as incorrect, so part of "unsolved" is a length budget. And RL rollouts [sample at temperature 1.0 and top-p 1.0](https://github.com/hieutrungle/sharpen-or-expand/blob/main/docs/decisions/017-rl-rollout-sampling-1.0.md), a different sampler from evaluation, so the distribution the policy explores in training is not the one that is measured.
</div>

## Reasoning diversity

pass@k cannot see how successes are organized. Take two models with the same $$p = 0.4$$ on a problem. Model A reaches its correct answers through one strategy, with probability 0.4. Model B uses four distinct strategies with probability 0.1 each. Their pass@k is identical for every $$k$$.

To see the difference, look at the distribution over successful trajectories alone:

$$
q(\tau \mid x) = \frac{\pi_\theta(\tau \mid x)}{p_\theta(x)}, \qquad \tau \in \mathcal{S}(x),
$$

and its entropy,

$$
H_{\text{correct}} = -\sum_{\tau \in \mathcal{S}(x)} q(\tau \mid x) \log q(\tau \mid x).
$$

Over raw trajectories this entropy counts every wording separately. To count strategies, group the successful trajectories into strategy families $$z$$ and use the family masses instead:

$$
q(z \mid x) = \sum_{\tau \in z} q(\tau \mid x), \qquad H_{\text{correct}} = -\sum_{z} q(z \mid x) \log q(z \mid x).
$$

The exponential $$N_{\text{eff}} = e^{H_{\text{correct}}}$$ is easier to read: it is the number of equally likely strategies that would have the same entropy. Ecologists know it as the Hill number of order 1, the "effective number of species" [6]. Model A has $$q = 1$$ on its one strategy and $$N_{\text{eff}} = 1$$. Model B has $$q = 0.25$$ on each of four, $$H_{\text{correct}} = \ln 4 \approx 1.39$$ and $$N_{\text{eff}} = 4$$. An uneven model with correct mass 0.3 and 0.1 on two strategies has $$q = (0.75, 0.25)$$ and $$N_{\text{eff}} \approx 1.75$$: two strategies, but one dominates.

The unit matters. Computed over raw token sequences, the entropy counts every rewording of a solution as a new trajectory, so it is large for any model and says little. What matters is semantic diversity: how many genuinely different ways the model has of solving the problem. Measuring it needs a way to group solutions into strategy families, by hand, with embeddings or with a judge model, and that grouping is a measurement choice in its own right. Diversity is relevant to the main question because it is what large-$$k$$ sampling and RL exploration draw on. A model that collapses onto one strategy can keep its pass@1 and have less to fall back on when that strategy fails. That is a plausible mechanism, not an established fact, and it has to be measured.

## How strong is the evidence?

Putting the pieces together gives a ladder of evidence for expansion, from weakest to strongest:

1. pass@1 increases.
2. pass@k increases at moderate $$k$$.
3. Coverage at large $$k$$ increases: more problems are solved at least once.
4. Problems that were unsolved at a large budget become repeatedly solvable.
5. New, coherent strategy families appear.
6. The effects replicate across training seeds.
7. The behavior generalizes to held-out problem families and perturbations.
8. High-volume sampling of the starting model puts a tight upper bound on how accessible the behavior was before.

The first two steps are equally compatible with sharpening. The later steps narrow the gap between "newly observed" and "newly created", but no finite experiment closes it, because of the bound in the section on zero successes. The defensible wording is "the results are consistent with operational capability expansion", not "RL created a capability that did not exist".

The published evidence sits on different rungs of this ladder, which is part of why it seems to disagree. Yue et al. [5] evaluate base and RLVR models at large $$k$$, find that the base models overtake at large $$k$$, and conclude that the reasoning they observe originates from and is bounded by the base model. ProRL [7] trains much longer, with KL control, reference-policy resets and a diverse task mix, and reports problems on which the base model fails regardless of the number of attempts while the trained model succeeds. By the logic of this part, "fails at every budget tried" is an upper bound on the base model's $$p$$, and it is strong evidence when the budget is large. Wu et al. [8] match verified successes question by question and find that, at their sampling budgets, more questions lose a verified success than gain one, alongside gains in pass@1. The question is open, and answering it for a given training run means climbing as far up the ladder as the budget allows.

## Check yourself

**1.** A problem has $$p_{\text{SFT}} = 0.01$$ before RL and $$p_{\text{RL}} = 0.30$$ after. Have we shown sharpening, expansion or neither?

<details markdown="1">
<summary>Answer</summary>

Improved accessibility, and by default sharpening. A success probability of 0.01 was already measurable: at $$k = 64$$ the SFT model solves the problem in about 47% of evaluation runs. Arguing for expansion would need evidence higher up the ladder: problems unsolved at a large budget becoming repeatedly solvable, new strategy families, replication across seeds, held-out generalization.
</details>

**2.** Two models have average pass@1 = 40%. Model A has $$p_i = 0.4$$ on every problem. Model B has $$p_i = 0.8$$ on half of the problems and $$p_i = 0$$ on the other half. Which has the higher pass@16?

<details markdown="1">
<summary>Answer</summary>

Model A. Its pass@16 is $$1 - 0.6^{16} \approx 99.97\%$$ on every problem. Model B solves its good half almost surely and its other half never, so its pass@16 is about 50.00%. With the same average pass@1, the model whose success is spread across problems has far broader coverage (concavity and Jensen's inequality).
</details>

**3.** Model A has one correct strategy with probability 0.3. Model B has ten correct strategies with probability 0.03 each. Which has the larger pass@1 and pass@64?

<details markdown="1">
<summary>Answer</summary>

Neither. Both have total success mass 0.3, so both have pass@1 = 0.3 and pass@64 = $$1 - 0.7^{64}$$, which is 1.000000 to six decimal places. They differ in diversity: $$N_{\text{eff}}$$ is 1 for Model A and 10 for Model B. pass@k cannot see the structure of the success set.
</details>

**4.** The SFT model gets 0 of 64 samples right on a problem. After DAPO, 10 of 64 are right. Why is "DAPO learned a novel capability that did not exist before" unjustified?

<details markdown="1">
<summary>Answer</summary>

Because 0/64 does not mean $$p_{\text{SFT}} = 0$$. It is compatible with $$p_{\text{SFT}}$$ up to about 4.6%; a problem with $$p = 0.03$$ shows 0/64 with probability $$0.97^{64} \approx 14\%$$. After DAPO, $$\hat p \approx 10/64 \approx 0.16$$. A defensible statement is that DAPO made a rarely or unobservably accessible behavior reliably accessible. With evidence from higher up the ladder, one could say the result is consistent with operational capability expansion.
</details>

## Takeaways

- Capability is a success probability $$p_\theta(x; c)$$ under fixed inference conditions, observed through a finite sampling budget. Nonzero support is not a useful definition: under softmax it is almost everything, and under top-p or top-k it depends on the sampler.
- $$\text{pass@}k = 1 - (1-p)^k$$. Large $$k$$ is a microscope for rare successes, and a success of probability $$p$$ needs about $$0.7/p$$ samples to show up in half of all runs.
- Benchmark pass@k averages a concave function, so for the same pass@1 it rewards breadth. Sharpening can raise pass@1 and lower pass@k at large $$k$$.
- Zero successes in $$n$$ samples bound $$p$$ below about $$3/n$$; they never prove $$p = 0$$. Newly observed is not newly created.
- Estimate pass@k with the unbiased estimator $$1 - \binom{n-c}{k}/\binom{n}{k}$$ and $$n \ge k$$; plugging in $$c/n$$ is biased low.

## Next

[Part 2](/blog/2026/11/sft-probability-reshaping/) turns to the first post-training method in the chain, supervised fine-tuning. It asks what imitation does to the trajectory distribution $$\pi_\theta(\tau \mid x)$$ defined here: the autoregressive likelihood, the cross-entropy gradient, teacher forcing, and why SFT behaves like forward-KL minimization.

## References

1. A. Fan, M. Lewis, and Y. Dauphin, "Hierarchical neural story generation," in *Proc. ACL*, 2018. [arXiv](https://arxiv.org/abs/1805.04833)
2. A. Holtzman, J. Buys, L. Du, M. Forbes, and Y. Choi, "The curious case of neural text degeneration," in *Proc. ICLR*, 2020. [arXiv](https://arxiv.org/abs/1904.09751)
3. J. A. Hanley and A. Lippman-Hand, "If nothing goes wrong, is everything all right? Interpreting zero numerators," *JAMA*, vol. 249, no. 13, pp. 1743–1745, 1983. [DOI](https://doi.org/10.1001/jama.1983.03330370053031)
4. M. Chen et al., "Evaluating large language models trained on code," arXiv:2107.03374, 2021. [arXiv](https://arxiv.org/abs/2107.03374)
5. Y. Yue et al., "Does reinforcement learning really incentivize reasoning capacity in LLMs beyond the base model?" arXiv:2504.13837, 2025. [arXiv](https://arxiv.org/abs/2504.13837)
6. M. O. Hill, "Diversity and evenness: A unifying notation and its consequences," *Ecology*, vol. 54, no. 2, pp. 427–432, 1973. [DOI](https://doi.org/10.2307/1934352)
7. M. Liu et al., "ProRL: Prolonged reinforcement learning expands reasoning boundaries in large language models," arXiv:2505.24864, 2025. [arXiv](https://arxiv.org/abs/2505.24864)
8. F. Wu et al., "The invisible leash: Why RLVR may or may not escape its origin," arXiv:2507.14843, 2025. [arXiv](https://arxiv.org/abs/2507.14843)
