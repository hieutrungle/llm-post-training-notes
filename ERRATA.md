# Errata

Corrections made after a post was published, newest first. Each entry gives the part, the section, what changed and why.

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
