# Checks

One script per part, `partNN_checks.py`, that recomputes every number the post quotes and fails loudly if any differs from the text.

Part 0 quotes no derived numbers except the figure's end points, which `figures/part00_sharpen_vs_expand.py` prints when it runs.

`part01_checks.py` recomputes every number in Part 1, including the worked examples, the check-yourself answers, the seeded simulation behind the pass@k animation, and an exact and Monte Carlo check that the pass@k estimator is unbiased. It re-reads the experiment numbers from a pinned commit of sharpen-or-expand and fails if any value differs from the text.

`part02_checks.py` recomputes every number in Part 2: the worked gradient step and the numbers shown in the reshaping animation, the compounding and length examples, the loss-normalization weights, the toy exposure-bias model behind the figure, and the check-yourself answers. It checks the softmax cross-entropy gradient against finite differences, the identity $H(q, \pi) = H(q) + \mathrm{KL}(q \Vert \pi)$ on random distributions, and $\nabla \log p_\theta(x) = \mathbb{E}_{q}[\nabla \log \pi_\theta]$ on a small autoregressive model. It confirms which SFT checkpoint the experiment reports evaluate and re-reads every experiment number from a pinned commit of sharpen-or-expand.
