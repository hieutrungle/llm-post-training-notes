# Errata

Corrections made after a post was published, newest first. Each entry gives the part, the section, what changed and why.

_None yet._

## Corrections to the study notes

The raw notes in `notes/raw/` stay as written. Where a post corrects or sharpens a statement in them, the change is listed here.

### Block 1 (Part 1)

- §2, "every finite token sequence has nonzero probability": true only for pure softmax sampling. With top-k or top-p truncation the support is finite and depends on the sampler, so it can differ between RL rollouts and evaluation. Part 1, "Why nonzero probability is not a useful definition".
- §2, accessibility as $\mathcal{A}_\epsilon = \{\tau : \pi(\tau) \ge \epsilon\}$: a threshold on single trajectories does not measure accessibility, because a correct solution can be written in many wordings that are each improbable, so a problem with high success probability can have an empty $\mathcal{A}_\epsilon$. Part 1 uses the total success mass $p_\theta(x; c)$ together with the sampling budget instead.
- §10: added the rule of three (95% upper bound about $3/n$ after 0 successes in $n$ trials; Hanley and Lippman-Hand, 1983) next to the exact bound.
- §11: the estimator $1 - \binom{n-c}{k}/\binom{n}{k}$ is the unbiased estimator of Chen et al. (2021), not just "standard"; it requires $n \ge k$. Part 1 also shows why the plug-in $1 - (1 - c/n)^k$ is biased low.
- §12: $N_{\text{eff}} = e^{H}$ is the Hill number of order 1 (Hill, 1973), and it is meaningful over strategy families, not raw token sequences.
- §13: the sharpen-versus-expand evidence is presented with both sides (Yue et al., 2025; ProRL, Liu et al., 2025; Wu et al., 2025, v5).
