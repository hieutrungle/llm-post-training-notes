# Checks

One script per part, `partNN_checks.py`, that recomputes every number the post quotes and fails loudly if any differs from the text.

Part 0 quotes no derived numbers except the figure's end points, which `figures/part00_sharpen_vs_expand.py` prints when it runs.

`part01_checks.py` recomputes every number in Part 1, including the worked examples, the check-yourself answers, the seeded simulation behind the pass@k animation, and an exact and Monte Carlo check that the pass@k estimator is unbiased. It re-reads the experiment numbers from a pinned commit of sharpen-or-expand and fails if any value differs from the text.
