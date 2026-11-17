# Errata

Corrections made after a post was published, newest first. Each entry gives the part, the section, what changed and why.

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
