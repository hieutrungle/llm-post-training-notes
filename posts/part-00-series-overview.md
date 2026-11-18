---
title: "Sharpen or Expand? A Study Series on LLM Post-Training"
kind: Study notes
date: 2026-11-16 00:00:00 -0600
series: llm-post-training
part: 0
short_title: "Overview and roadmap"
summary: "Does reinforcement learning teach a language model to reason in new ways, or make it more reliable at what it could already do? Part 0 of a 12-part series that builds LLM post-training from first principles and tests it in a real experiment."
math: true
---

*Code, check scripts and references: [llm-post-training-notes](https://github.com/hieutrungle/llm-post-training-notes) · Experiments: [sharpen-or-expand](https://github.com/hieutrungle/sharpen-or-expand)*

<div class="goal" markdown="1">
<span class="label">Goal of the series</span>

Understand LLM post-training from first principles, well enough to answer one question with evidence: **does reinforcement learning with verifiable rewards sharpen what a model can already do, or expand what it can solve?** Each part derives one piece of the theory, checks every number it quotes, and connects the theory to measurements from a real experiment.
</div>

## The question

Reinforcement learning with verifiable rewards (RLVR) has become a standard step in training reasoning models. The recipe is simple: sample several solutions to a math or coding problem, check each one with a program, and push the model toward the solutions that pass. DeepSeekMath introduced GRPO for this setting [1], and DeepSeek-R1 showed how far the recipe can go [2].

What the recipe actually changes inside the model is still argued about. There are two readings:

- **Sharpening.** RL moves probability onto solutions the model could already produce. The model is right more often on its first try (pass@1 rises), but the set of problems it can solve at all stays roughly the same.
- **Expansion.** RL makes the model reliably reach solutions that were effectively out of reach before, so the set of solvable problems grows.

The evidence points both ways. Yue et al. [3] sampled base models and their RLVR-trained versions many times per problem. The RL models win when only a few samples are allowed, but the base models solve more problems when many samples are allowed. The authors read this as RL working inside the reasoning the base model already has. ProRL [4] reports the opposite under prolonged training with KL control, reference-policy resets and a diverse task mix: its RL-trained model solves problems on which the base model fails at every sampling budget tried. Wu et al. [5] follow problems one by one and find that, at their sampling budgets, more problems lose a verified solution than gain one, even while pass@1 improves. Risk-sensitive GRPO [6] changes the training objective specifically to push exploration toward hard prompts.

The two readings make different predictions, and those predictions can be measured. The figure below shows the signature with made-up numbers.

![Benchmark pass@k against the number of samples per problem k, on a log scale from 1 to 1024, for three illustrative models. The sharpened model beats the base model at k = 1 (0.47 vs 0.30) but falls below it from k = 7 and ends at 0.71 vs 0.85 at k = 1024. The expanded model stays above both at every k and ends at 0.91.]({{ '/assets/img/llm-post-training/part-00-sharpen-vs-expand.png' | relative_url }})

The three curves are three hypothetical models evaluated on the same 1,000 problems. "Sharpened" makes already-common solutions more likely and lets rare ones fade; "Expanded" makes the same gain on common solutions and also makes 40% of the rarely or never solved problems reliably solvable. pass@k is the share of problems solved at least once in k samples.

Read the curves from left to right:

- **At k = 1**, both trained models beat the base model (0.47 and 0.53 vs 0.30). Single-sample accuracy cannot tell sharpening from expansion.
- **As k grows**, the base model keeps finding its rare solutions, and from k = 7 on it is ahead of the sharpened model. At k = 1024 the base model solves 85% of the problems and the sharpened model 71%: the sharpened model traded its rare solutions for reliability on common ones.
- **The expanded model** stays above the base model at every k and ends at 91%, because it now solves problems the base model rarely or never solved.

So sharpening shows up as a *crossover* at large k, and expansion as a gap that *persists* at large k. Real models are harder to read than this cartoon. A problem with no success in 1,024 samples can still have a small nonzero success probability, so "never solved" is always a statement about a sampling budget. Part 1 makes that precise, and most of this series is about making the comparison rigorous.

## Why a series

Answering the question well needs a chain of ideas, each built on the one before:

1. What are we measuring? (capability, pass@k)
2. Supervised fine-tuning: imitate demonstrations.
3. DPO: prefer one response over another.
4. Policy gradients: sample, evaluate, reinforce.
5. GRPO: reinforce relative to a group, without a critic.
6. Why GRPO breaks.
7. How newer algorithms try to fix it.
8. What does the reward actually teach?
9. How do we know an observed gain is real?
10. How do we find out why it happened?

Parts 11 and 12 then turn this into systems awareness and a final synthesis.

## Roadmap

| Part | Question | Topics |
|---|---|---|
| 0 | Why this series? | The sharpen-or-expand question, roadmap, notation (this post) |
| 1 | [What does "capability" mean for an LLM?](/blog/2026/11/capability-and-pass-at-k/) | Trajectory distributions, pass@1 and pass@k, accessibility, reasoning diversity, sharpening vs expansion |
| 2 | [What does supervised fine-tuning do to the distribution?](/blog/2026/11/sft-probability-reshaping/) | Autoregressive likelihood, cross-entropy gradients, teacher forcing, forward KL |
| 3 | [How do we learn from "this is better than that"?](/blog/2026/11/dpo-from-kl-regularized-rl/) | Bradley–Terry, KL-regularized RL, DPO derivation, the reference policy |
| 4 | [How do we sample, evaluate and reinforce?](/blog/2026/11/policy-gradients-reinforce-to-ppo/) | REINFORCE, baselines, advantages, importance sampling, PPO clipping, KL, entropy |
| 5 | [How do we get an advantage without a critic?](/blog/2026/11/grpo-from-first-principles/) | GRPO, group-relative advantages, group size, mixed groups |
| 6 | [Why does vanilla GRPO fail?](/blog/2026/11/why-vanilla-grpo-fails/) | Zero-variance groups, normalization biases, length bias, clipping, entropy collapse |
| 7 | How do newer algorithms fix it? | Dr.GRPO, DAPO, RS-GRPO, each derived from the failure it targets |
| 8 | What does a verifiable reward teach? | Sparse outcome rewards, verifier errors, reward hacking, credit assignment |
| 9 | Is the gain real? | pass@k estimation, bootstrap intervals, paired tests, seeds, capability transitions |
| 10 | Why did it happen? | Matched rollout-token budgets, one-variable ablations, causal controls |
| 11 | Which systems choices change results? | LoRA vs full fine-tuning, BF16, FSDP, vLLM rollouts, policy staleness |
| 12 | Did RL sharpen or expand? | Reading the experiment's final results, alternative explanations |

Links appear here as parts are published; the [series page](/blog/series/llm-post-training/) lists every part.

## How each part is built

I am working through this material from first principles and writing it down as I go. Each part follows the same pattern:

- **One question**, linked to the part before it.
- **A derivation from first principles**, with worked numbers.
- **Every number is checked.** A script in the companion repository recomputes each value the post quotes.
- **In my experiments.** Where it helps, a short box links a real result or design choice from the [sharpen-or-expand](https://github.com/hieutrungle/sharpen-or-expand) experiment.
- **Check yourself.** A few questions, with answers folded underneath.
- **References** to the original papers.

If you find an error, please open an issue on the [companion repository](https://github.com/hieutrungle/llm-post-training-notes/issues). Corrections are logged in its errata file.

## The experiment behind the series

[sharpen-or-expand](https://github.com/hieutrungle/sharpen-or-expand) tests the question directly on Qwen3.5-4B-Base. It compares DPO [7], GRPO [1], Dr.GRPO [8], DAPO [9] and RS-GRPO [6] at equal rollout-token budgets, with every method starting from the same base model. Supervised fine-tuning on the base model's own verified solutions is evaluated alongside them. Every checkpoint is scored with pass@k from small to large k and with problem-by-problem transitions: which problems become solvable, and which stop being solvable.

The experiment is in progress. Results appear in the parts they illustrate, and Part 12 reads the final ones.

## Notation used across the series

Several papers in this series use the same letter for different things. The table fixes one meaning per symbol.

| Symbol | Meaning |
|---|---|
| $$x$$ | A prompt (one problem) |
| $$\tau = (y_1, \ldots, y_T)$$ | A complete response, or trajectory, of $$T$$ tokens (DPO writes a response as $$y$$; GRPO writes the $$i$$-th response in a group as $$o_i$$) |
| $$\pi_\theta(\tau \mid x) = \prod_{t=1}^{T} \pi_\theta(y_t \mid x, y_{\lt t})$$ | The policy being trained, an autoregressive language model with parameters $$\theta$$ |
| $$\pi_{\text{old}}$$ | The policy that generated the current batch of rollouts |
| $$\pi_{\text{ref}}$$ | A frozen reference policy, usually the starting model |
| $$R(x, \tau) \in \{0, 1\}$$ | Verifier reward: 1 if the final answer is correct |
| $$\mathcal{S}(x)$$ | The set of successful trajectories for $$x$$ |
| $$p_\theta(x) = \sum_{\tau \in \mathcal{S}(x)} \pi_\theta(\tau \mid x)$$ | Single-sample success probability on $$x$$ |
| $$\text{pass@}k = 1 - (1 - p_\theta(x))^k$$ | Probability that at least one of $$k$$ independent samples succeeds [10] |
| $$G$$ | Group size: responses sampled per prompt in GRPO |
| $$A_i$$ | Advantage of response $$i$$ |
| $$\rho_{i,t} = \pi_\theta(y_{i,t} \mid \cdot) \,/\, \pi_{\text{old}}(y_{i,t} \mid \cdot)$$ | Per-token importance ratio |
| $$\epsilon$$, $$\epsilon_{\text{low}}$$, $$\epsilon_{\text{high}}$$ | PPO clip range [11], symmetric or asymmetric |
| $$\beta$$ | KL-regularization strength in DPO and RLHF: $$\pi^\ast \propto \pi_{\text{ref}}\, e^{r/\beta}$$ |
| $$\lambda_{\text{KL}}$$ | Coefficient of the KL penalty in a GRPO-style loss on $$D_{\mathrm{KL}}(\pi_\theta \Vert \pi_{\text{ref}})$$ |
| $$\beta_{\text{RS}}$$ | Risk-sensitivity of RS-GRPO, in $$\frac{1}{\beta_{\text{RS}}} \log \mathbb{E}\big[e^{\beta_{\text{RS}} R}\big]$$ |
| $$H$$ | Entropy |

**Three β's.** DPO's paper [7] uses $$\beta$$ for the KL strength. DeepSeekMath [1] also uses $$\beta$$, for the KL coefficient in the GRPO loss. RS-GRPO [6] uses $$\beta$$ for risk sensitivity. They are different quantities, so this series writes them as $$\beta$$, $$\lambda_{\text{KL}}$$ and $$\beta_{\text{RS}}$$. The first and third have a family resemblance: both tilt probabilities by an exponential of the reward, but the conventions run opposite. A *small* $$\beta$$ lets the policy move far from the reference, while a *large* $$\beta_{\text{RS}}$$ pushes the objective toward the best sample.

## Next

[Part 1](/blog/2026/11/capability-and-pass-at-k/) starts at the bottom of the chain: what it means for a model to "be able to" solve a problem, why nonzero probability is not a useful definition, and how pass@k turns capability into something we can measure.

## References

1. Z. Shao et al., "DeepSeekMath: Pushing the limits of mathematical reasoning in open language models," arXiv:2402.03300, 2024. [arXiv](https://arxiv.org/abs/2402.03300)
2. DeepSeek-AI, "DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning," arXiv:2501.12948, 2025. [arXiv](https://arxiv.org/abs/2501.12948)
3. Y. Yue et al., "Does reinforcement learning really incentivize reasoning capacity in LLMs beyond the base model?" arXiv:2504.13837, 2025. [arXiv](https://arxiv.org/abs/2504.13837)
4. M. Liu et al., "ProRL: Prolonged reinforcement learning expands reasoning boundaries in large language models," arXiv:2505.24864, 2025. [arXiv](https://arxiv.org/abs/2505.24864)
5. F. Wu et al., "The invisible leash: Why RLVR may or may not escape its origin," arXiv:2507.14843, 2025. [arXiv](https://arxiv.org/abs/2507.14843)
6. Y. Jiang et al., "Risk-sensitive RL for alleviating exploration dilemmas in large language models," arXiv:2509.24261, 2025 (ICLR 2026). [arXiv](https://arxiv.org/abs/2509.24261)
7. R. Rafailov et al., "Direct preference optimization: Your language model is secretly a reward model," in *Proc. NeurIPS*, 2023. [arXiv](https://arxiv.org/abs/2305.18290)
8. Z. Liu et al., "Understanding R1-Zero-like training: A critical perspective," arXiv:2503.20783, 2025. [arXiv](https://arxiv.org/abs/2503.20783)
9. Q. Yu et al., "DAPO: An open-source LLM reinforcement learning system at scale," arXiv:2503.14476, 2025. [arXiv](https://arxiv.org/abs/2503.14476)
10. M. Chen et al., "Evaluating large language models trained on code," arXiv:2107.03374, 2021. [arXiv](https://arxiv.org/abs/2107.03374)
11. J. Schulman et al., "Proximal policy optimization algorithms," arXiv:1707.06347, 2017. [arXiv](https://arxiv.org/abs/1707.06347)
