# Errata

Corrections made after a post was published, newest first. Each entry gives the part, the section, what changed and why.

### 2026-11-17 · Part 3, "Conclusion" (new section)

Added a closing section between "Takeaways" and "Next" that returns to the opening picture and states what the part means for the series question: the target of KL-regularized preference learning is sharpening, while the training path, through shared parameters, is not constrained the same way, so the sharpening expected from offline DPO stays a hypothesis. No derivation or number changed.

### 2026-11-17 · Part 3, "The picture to keep in mind" (new section)

Added an opening animation and section, before the derivation, that shows what DPO does to one training pair: the preferred response goes up, the dispreferred one goes down, both measured as ratios to the frozen reference, and the push fades as the margin grows (illustrative numbers with free logits, recomputed in `checks/part03_checks.py`). Two sentences that pointed to "the animation" now say which of the two animations they mean. No derivation or existing number changed.

### 2026-11-17 · Parts 0–2, editorial

Each part now reads on its own, using only earlier parts. Passages that cited the raw study notes were rewritten as self-contained explanations (Part 1: accessibility as a threshold on single trajectories, the sharpening example; Part 2: the four-decision path), and pointers to specific later parts were removed or replaced with references to earlier ones (Part 2: loss normalization, exposure bias and RL, reverse KL, the policy-gradient identity). The byline now reads "Code, check scripts and references". No derivation or number changed.

### 2026-11-16 · Part 1, "What does capability mean for an LLM?"

A review after publication; no derivation or number changed.

- "Same pass@1, different breadth": "Section 9 comes back to this" pointed nowhere, because the blog headings are not numbered. It now names the section on reasoning diversity.
- "Why nonzero probability is not a useful definition": top-k and top-p were described with $k$ and $p$, which collide with pass@k's $k$ and the success probability $p$. The sampler settings are now $K$ and $p_{\text{top}}$, with a sentence that they are unrelated.
- Same section: "no amount of sampling will ever show it" became "no feasible amount".
- "pass@1 and pass@k": added the assumption behind $1-(1-p)^k$ (independent samples from one distribution).
- Animation reading notes: the $k = 1$ bullet gave the exact 40.0% only; it now also gives the simulated 9 of 20 runs the lane shows.
- "Estimating pass@k": "the unbiased estimator averages exactly 0.570" now says its average equals the true value (0.570 to three decimals).
- "In my experiments": added how the hard problems were selected (solved at most 4 times in 16 samples by the base model, in a separate run with its own seed).
- "Reasoning diversity": $H_{\text{correct}}$ was defined over trajectories but applied to strategies. The post now defines the strategy-level masses $q(z \mid x)$ and the entropy over them.
- Blog layout: the seven-column pass@k table overflowed on phones; wide tables now scroll inside their own box on narrow screens.

## Corrections to the study notes

The raw notes in `notes/raw/` stay as written. Where a post corrects or sharpens a statement in them, the change is listed here.

### Block 1 (Part 1)

- §2, "every finite token sequence has nonzero probability": true only for pure softmax sampling. With top-k or top-p truncation the support is finite and depends on the sampler, so it can differ between RL rollouts and evaluation. Part 1, "Why nonzero probability is not a useful definition".
- §2, accessibility as $\mathcal{A}_\epsilon = \{\tau : \pi(\tau) \ge \epsilon\}$: a threshold on single trajectories does not measure accessibility, because a correct solution can be written in many wordings that are each improbable, so a problem with high success probability can have an empty $\mathcal{A}_\epsilon$. Part 1 uses the total success mass $p_\theta(x; c)$ together with the sampling budget instead.
- §10: added the rule of three (95% upper bound about $3/n$ after 0 successes in $n$ trials; Hanley and Lippman-Hand, 1983) next to the exact bound.
- §11: the estimator $1 - \binom{n-c}{k}/\binom{n}{k}$ is the unbiased estimator of Chen et al. (2021), not just "standard"; it requires $n \ge k$. Part 1 also shows why the plug-in $1 - (1 - c/n)^k$ is biased low.
- §12: $N_{\text{eff}} = e^{H}$ is the Hill number of order 1 (Hill, 1973), and it is meaningful over strategy families, not raw token sequences.
- §13: the sharpen-versus-expand evidence is presented with both sides (Yue et al., 2025; ProRL, Liu et al., 2025; Wu et al., 2025, v5).

### Block 2 (Part 2)

- Q3, "at least three reasons" introduced a list of five; the post gives the five reasons as five.
- §8–9 (teacher forcing, exposure bias): added the standard references. Ranzato et al. (2016) named exposure bias; Bengio et al. (2015) proposed scheduled sampling; Ross, Gordon and Bagnell (2011) give the $T^2\varepsilon$ compounding-error bound for behavior cloning (citing Ross and Bagnell, 2010) and DAgger, whose guarantee is linear in $T$.
- §10: the data distribution is written $q_{\text{data}}(y \mid x)$, so it does not collide with Part 1's $q(\tau \mid x)$, the model's distribution over its own successful trajectories. The post also states the other half of "mode-covering relative to the data": responses with zero data probability contribute nothing to the KL, so maximum likelihood drives valid but undemonstrated solutions toward zero.
- §11: $N_{\text{eff}}$ is computed over strategy families, as in Part 1 (2 before, 1.28 after for the A/B example).
- §14: added that loss normalization (sum over tokens, mean per sequence, mean over all tokens in a batch) changes how long and short responses are weighted, with a forward pointer to the length bias in Parts 6–7.
- §16: added the derivation $\nabla \log p_\theta(x) = \mathbb{E}_{\tau \sim q}[\nabla \log \pi_\theta(\tau \mid x)]$: likelihood training on the model's own successful samples estimates the gradient of the log success mass, provided the samples come from the current model unmodified (not, for example, at a lower temperature).
- Q4 ($10^{-8} \to 10^{-2}$): "the trajectory already had nonzero probability" holds only for untruncated sampling; under top-p it may have had probability exactly zero. Accessibility is the total success mass $p_\theta(x; c)$ at a sampling budget, not one trajectory's probability, so a $10^6$ gain on one trajectory neither shows nor rules out operational expansion. Part 2, check-yourself question 3.
- Q5: added that the expected-reward gradient raises every correct trajectory only in expectation; with finite samples, correct trajectories that are not drawn can go down.
- New context for the experiment: the SFT arm is rejection-sampling fine-tuning on the base model's own verified solutions, placed next to STaR (Zelikman et al., 2022) and Yuan et al. (2023).

### Block 3 (Part 3)

- §8–9 and Q1 (ratio of ratios, the 0.01/0.20 → 0.02/0.10 example): the example shows only half of the story. The DPO loss depends on the pair only through $\Delta_\theta$, which is unchanged when both log-probabilities fall by the same amount, so DPO can lower the absolute probability of the preferred response (likelihood displacement; Razin et al., 2025; Pal et al., 2024). Part 3 adds a worked case with the same loss in which both probabilities fall (0.01/0.20 → 0.008/0.04), a toy model showing that the effect needs shared parameters, and the likelihood-term fixes (DPO-Positive; Pang et al., 2024). Part 3, "DPO can lower the probability of the preferred response" and check-yourself question 1.
- §5 and Q2, "as β → 0, probability concentrates on maximum-reward behavior": stated precisely, the mass concentrates on the maximum-reward responses that $\pi_{\text{ref}}$ supports, in proportion to their reference probabilities; a maximum-reward response with zero reference probability gets nothing.
- Q3, "a classic sharpening signature": the conclusion is now "consistent with sharpening", in Part 1's evidence-ladder language. A flat pass@64 is weak evidence: pass@64 is near 1 for any problem with modest $p$, and it cannot rule out a few problems moving from unsolved to solved.
- Q4, "DPO directly learns only from trajectories in its fixed preference dataset": true of the labels, but not of the effect. Because parameters are shared, DPO changes the probability of responses outside the dataset as well; likelihood displacement is the clearest case. Part 3, "Offline DPO and exploration".
- §10 (gradient intuition: large for negative $\Delta$, small for large positive $\Delta$): the weight $\beta\sigma(-\beta\Delta)$ depends on $\beta\Delta$. At $\beta = 0.1$ it is nearly flat over moderate margins (0.053, 0.050, 0.047 for $\Delta = -\log 4, 0, \log 4$) and falls to 0.01 only at $\Delta \approx 22$.
- §7, "replace $\pi^*$ by trainable $\pi_\theta$": made precise as fitting the Bradley–Terry model by maximum likelihood with the implicit reward $\beta\log(\pi_\theta/\pi_{\text{ref}})$; the equivalence with the KL-regularized optimum needs a policy class that can represent $\pi^*$.
- §13 (offline DPO leans toward sharpening): kept as a hypothesis, as in the notes. Part 3 adds the result behind it, that with a 0/1 reward the KL-regularized optimum multiplies the odds of success by $e^{1/\beta}$ and leaves $q(\tau \mid x)$ and $N_{\text{eff}}$ unchanged, and the three reasons training need not follow it (the optimum is not reached, parameters are shared, on-policy RL samples new trajectories). It also adds that verifier labels are deterministic preferences, for which the Bradley–Terry reward gap is infinite and $\beta$ no longer fixes the target (Azar et al., 2023), and that iterative DPO re-samples pairs (Pang et al., 2024).
- §11 (credit assignment): added that tokens of a shared prefix cancel exactly in $\Delta$, and that log-ratios are sums over tokens, so response length scales them (the sum normalization of Part 2; length exploitation in DPO, Park et al., 2024).
- Standard background now cited: Bradley and Terry (1952); Christiano et al. (2017); Ziegler et al. (2019); Ouyang et al. (2022).

### Block 4 (Part 4)

- §5 (variance): made numeric. With $p = 0.01$ and 100 samples, P(0 successes) = 0.366, P(1) = 0.370, P(2) = 0.185, P(≥3) = 0.079; in a one-logit model the REINFORCE estimate is then exactly $k$ times the true gradient for $k$ successes. Part 4 adds a figure showing that a baseline removes the noise from the reward's offset but not the noise from rare successes.
- §6 and Q2 (baseline): "does not depend improperly on the sampled action" replaced by the precise condition. A sequence-level baseline may depend on the prompt; a token-level baseline also on the prefix; neither on the token it multiplies, later tokens or the trajectory's reward. A group mean that includes the sample gives $(G-1)/G$ times the gradient (a rescaling, not a change of direction); the leave-one-out mean (RLOO; Ahmadian et al., 2024) is exactly unbiased.
- §8 ("this corrects for policy change between rollout generation and optimization"): the per-token ratio corrects the choice of each token given its prefix, not the change in which prefixes occur, so PPO's surrogate is a local approximation that is exact in gradient only at $\pi_\theta = \pi_{\text{old}}$ (TRPO's surrogate). The notes' $r_t(\theta)$ is $\rho_{i,t}$ in the series, since $r$ is the reward.
- §9: $1.02^{500} = 19{,}956.57$, so ≈ 19,957, not 19,956.
- §10 and Q5 (clipping): added the condition that clipping acts only when $\pi_\theta \ne \pi_{\text{old}}$. With one optimizer update per batch of rollouts every ratio is exactly 1 and the clipped objective is the plain policy gradient. Also made explicit that the min keeps the unclipped term when a token moved the wrong way, so clipping never blocks a correction.
- §11: $\pi_{\text{ref}}$ is "often the SFT model" in RLHF; many recent reasoning recipes, and the RL algorithms in sharpen-or-expand, have no reference model at all.
- §12 (KL): the notes write the RLHF-style objective with $\lambda_{\text{KL}}$. By the series notation the coefficient of a KL term in the objective or reward is $\beta$; $\lambda_{\text{KL}}$ is only for a KL term added to a GRPO-style loss. Part 4 adds that the term sits in one of two places (in the reward: Ziegler et al., 2019; Ouyang et al., 2022; in the loss with the k3 estimator: Shao et al., 2024) and that DAPO, Dr. GRPO and the experiments drop it.
- Q6 ("a strong sharpening/exploitation signature"): the conclusion is now "consistent with sharpening", in Part 1's evidence-ladder language, and entropy collapse is cited to Cui et al. (2025). The proposed mechanism also gains a step the notes do not state: with one logit per strategy, the expected push on a correct strategy is $\pi_k(R_k - p)$, proportional to its probability, so the mix of correct strategies narrows even before sampling feedback.
- Standard background now cited: Williams (1992); Sutton et al. (1999); Schulman et al. (TRPO 2015, GAE 2015, PPO 2017).

### Block 5 (Part 5)

- §3 (and Q1, Q5), the closed forms $A_+ = \sqrt{(G-k)/k}$, $A_- = -\sqrt{k/(G-k)}$: they assume the population standard deviation (divide by $G$). `torch.std` divides by $G - 1$, and implementations add a small constant to the denominator. verl 0.9.1, which the experiments use, does both (`torch.std` plus `epsilon = 1e-6`), so at $G = 8$, $k = 1$ it gives $A_+ \approx 2.475$ and $A_- \approx -0.354$ instead of 2.646 and −0.378, and at $G = 16$, $k = 1$ it gives 3.750 and −0.250 instead of 3.873 and −0.258. At fixed $G$ the difference is the constant factor $\sqrt{(G-1)/G}$. Part 5 uses the population form in the text and gives verl's values in a note.
- §8 and Q4, "the rollout contributes to its own baseline": made precise. With mean-centering only, $\mathbb{E}[(R_i - \bar R)\nabla \log \pi_i] = \frac{G-1}{G}\mathbb{E}[R_i \nabla \log \pi_i]$, a rescaled policy gradient equal to RLOO (Ahmadian et al., 2024) up to scale. Dividing by $\sigma_R$ breaks this because $\sigma_R$ depends on $R_i$: for a 0/1 reward each prompt's expected update becomes $w_G(p)\,\nabla p$ with a $p$-dependent weight, which tends to $1/\sqrt{p(1-p)}$, the gradient of $2\arcsin\sqrt{p}$, as $G$ grows (Davis and Recht, 2025). At $G = 8$ the weight is within 10% of that limit only for $p$ between 0.30 and 0.70.
- §5, "$-\beta \mathrm{KL}$" in the GRPO objective: by the series notation a KL term in a GRPO-style loss has coefficient $\lambda_{\text{KL}}$ (DeepSeekMath calls it $\beta$). The experiments have no KL term.
- Notation: the notes write the prompt as $q$ and the group standard deviation as $s_R$. The post uses $x$ (Part 1's $q(\tau \mid x)$ is the distribution of successful trajectories) and $\sigma_R$, defined as the population standard deviation.
- Q3 ("the default interpretation is sharpening") and Q6 ("the canonical sharpening story for GRPO"): the conclusion is now "consistent with sharpening", in Part 1's evidence-ladder language.
- §9: the probabilities are given to the precision the checks print, 7.73% ($p = 0.01$, $G = 8$) and 47.4% ($G = 64$).
- §1, context added: DeepSeekMath's GRPO scored outputs with a learned reward model (64 outputs per question, KL coefficient 0.04); rule-based rewards came with DeepSeek-R1 (16 outputs per question and KL coefficient 0.001 for R1-Zero). The current DeepSeek-R1 paper also writes the objective with whole-response ratios and no $1/\lvert o_i \rvert$ factor; the post presents DeepSeekMath's per-token form, as the notes do.
- §13, "strongest learning on problems that are already partially solvable": true of the expected update size for one prompt, which peaks at $p = 0.5$, but the standard-deviation weight on a prompt's gradient is largest near $p = 0$ and $p = 1$ (at $G = 8$ it is 2.61 at $p = 0.01$ and 1.86 at $p = 0.5$). Part 5 states both.
- Standard references now cited: Davis and Recht (2025) for the arcsine view; verl 0.9.1's `compute_grpo_outcome_advantage` for the standard-deviation convention.

### Block 6 (Part 6)

- §5, the values 2.65 / 1 / 0.38 for $A_+$ at 1, 4 and 7 correct of 8: they assume the population standard deviation. verl's Bessel-corrected standard deviation plus $\delta$ gives values smaller by the factor $\sqrt{7/8}$ (2.475 instead of 2.646 at $k = 1$); Part 5's note "Which standard deviation?" gives both. Part 6 points back to it.
- §9, "training can produce incorrect responses getting longer without becoming better": the per-token weights $A_i/(G\lvert o_i \rvert)$ are an identity; the lengthening is a claim about training dynamics. **Flagged, it narrows the note's meaning.** In a toy where correctness does not depend on length (checks/part06_checks.py, figures/part06_length.py), the per-response average lengthens wrong responses and shortens correct ones only when the two have separate stop probabilities; with one shared stop probability the drifts cancel exactly, and with a constant denominator there is no drift at all. Part 6 attributes the lengthening to Liu et al. only as their current version (v2) states it: the per-token argument, "may lead to progressively longer incorrect responses", and their measurement that Dr. GRPO's incorrect responses are substantially shorter.
- §10, Dr.GRPO's "constant denominator such as a fixed generation budget": it is not verl's default. Without `loss_scale_factor`, `seq-mean-token-sum-norm` divides by the padded width of each micro-batch, which varies with dynamic batching (the same response's per-token weight can change tenfold). The experiments decided on a constant 16,384, the response cap; it is decided but not yet wired, because Dr.GRPO is not wired yet (sharpen-or-expand docs/roadmap/M2.md, D9; docs/reports/m1_rl_tests_mutations.md).
- §11, DAPO's token-level loss: verl's `token-mean` divides by the number of valid response tokens in the whole mini-batch of one optimizer update, summed over all data-parallel ranks.
- §13 and Q4, symmetric clipping: two conditions added. (1) Clipping acts only when $\pi_\theta \ne \pi_{\text{old}}$: on the first update after sampling every ratio is 1 (Part 4; the experiments' single-update smoke run logged a clip fraction of 0 on every step, before they moved to four updates per step). (2) **New precision, flagged:** the clip bound is relative and identical for every token, $\times(1 + \epsilon)$ per batch on the token's own incentive only; what differs between rare and common tokens is the absolute probability one batch can move (0.0002 at $\pi_{\text{old}} = 0.001$, 0.1 at 0.5). Across batches a rare token can still compound (0.001 to 0.5 in at least 35 batches at $\epsilon = 0.2$), and other tokens' updates can move it through shared parameters. DAPO's own example (0.01 → 0.012, 0.9 → 1.08) and $\epsilon_{\text{low}} = 0.2$, $\epsilon_{\text{high}} = 0.28$ are cited.
- §15 and Q6, "strong evidence for sharpening" and "improved sharpening/optimization, not evidence of expansion": now "consistent with sharpening", in Part 1's evidence-ladder language; a flat pass@64 is weak evidence by itself (see the Block 3 entry, Q3).
- §17, "RLVR may often reweight pre-existing reasoning structures": kept as a hypothesis and balanced with Yue et al. (2025), ProRL (Liu et al., 2025) and The Invisible Leash (Wu et al., 2025).
- §4 and Q5, dynamic sampling: DAPO drops groups whose accuracy is 0 or 1 and keeps sampling until the batch is full of mixed groups. In Part 6's opening batch that costs an expected 2.04 sampled groups per kept group in round 1 and 4.66 in round 10; resampling an all-wrong prompt finds a success only with probability $1 - (1 - p)^{Gk}$.
- §18, truncation: DAPO's response is two steps, overlong filtering (masking the loss of truncated samples) and soft overlong punishment (a linear length penalty over a 4,096-token buffer after an expected maximum of 16,384, added to the correctness reward). In the experiments a rollout at the 16,384 cap gets reward 0 and DAPO's shaping is applied on top.
- §21, the failure map: the "per-sequence loss" row duplicated the $1/T_i$ row and is merged into it; the $1/\sigma_R$ row now says what is reweighted (prompts near $p = 0$ or $1$ get more weight per unit of $\nabla_\theta p$, while the largest updates still come from intermediate $p$); rows added for verifier errors and for the exploration limitation (prompts that stay at 0 of $G$).
- Notation: $T_i \to \lvert o_i \rvert$; $\sigma_R$ is the population standard deviation as in Part 5.
- Standard references now cited: Cui et al. (2025) for entropy collapse; the DAPO paper for the Clip-Higher example; verl 0.9.1's `agg_loss` and `compute_policy_loss_vanilla`.
