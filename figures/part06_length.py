"""Part 6 figure: does the per-response average make wrong responses longer? A toy, in expectation.

Each response is one strategy token (correct with probability p = 0.5, held fixed) followed by L >= 1
tokens, where each token is the last with stop probability s (geometric length, mean 1/s = 100 at the
start). Correctness does not depend on length. With G = 8 and Part 5's advantage (population standard
deviation), the expected update of a stop logit under each loss aggregation is:
  per-response average (seq-mean-token-mean):  +/- E[sigma_R] * E[(1 - L s)/(1 + L)]  (+ correct, - wrong)
  token sum with a constant denominator (seq-mean-token-sum-norm with a fixed factor):  0, since E[1 - L s] = 0
  one stop probability shared by correct and wrong responses, any aggregation:  0, since sum_i A_i = 0.
The figure iterates the expected update (step size 0.5, 200 steps) for separate stop logits.
Token-mean is not drawn: its drift is below 1% of the per-response average's (Monte Carlo in
checks/part06_checks.py). The time scale is arbitrary; only the directions matter.
Run: python -B figures/part06_length.py  ->  assets/part-06/length.png
"""
from math import comb, exp, log, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
G, P, S0, LR, STEPS = 8, 0.5, 0.01, 0.5, 200


def sig(z):
    return 1 / (1 + exp(-z))


def e_sigma(p, g=G):
    """E[population sd of the group's 0/1 rewards] = E[(1/G) sum_i A_i R_i]."""
    return sum(comb(g, k) * p**k * (1 - p) ** (g - k) * sqrt(k * (g - k)) / g for k in range(g + 1))


def c_len(s):
    """E[(1 - L s) / (1 + L)] for L ~ Geometric(s) on {1, 2, ...} (closed form)."""
    q = 1 - s
    e_inv = s * (-log(s) - q) / q**2          # E[1/(1+L)]
    return e_inv - s * (1 - e_inv)


def trajectories():
    tc = tw = log(S0 / (1 - S0))
    e = e_sigma(P)
    out = [(1 / S0, 1 / S0)]
    for _ in range(STEPS):
        tc += LR * e * c_len(sig(tc))
        tw -= LR * e * c_len(sig(tw))
        out.append((1 / sig(tc), 1 / sig(tw)))
    return np.array(out)


def main(out: Path = Path(__file__).resolve().parents[1] / "assets/part-06/length.png"):
    tr = trajectories()
    x = np.arange(STEPS + 1)
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.plot(x, tr[:, 1], color="#1f4f8f", lw=2.4)
    ax.plot(x, tr[:, 0], color="#1f4f8f", lw=2.0, ls="--")
    ax.plot(x, np.full_like(x, 1 / S0, dtype=float), color="#8c8680", lw=2.0)
    ax.text(STEPS + 3, tr[-1, 1], f"wrong responses: {tr[-1, 1]:.0f}", color="#1f4f8f", va="center", fontsize=9)
    ax.text(STEPS + 3, tr[-1, 0], f"correct responses: {tr[-1, 0]:.0f}", color="#1f4f8f", va="center", fontsize=9)
    ax.text(STEPS + 3, 100, "both, with a constant\ndenominator or one shared\nstop probability: 100",
            color="#5f5a54", va="center", fontsize=9)
    ax.text(6, 330, "per-response average,\nseparate stop probabilities", color="#1f4f8f", fontsize=9, va="center")
    ax.set_yscale("log")
    ax.set_ylim(5, 600)
    ax.set_yticks([10, 30, 100, 300], ["10", "30", "100", "300"])
    ax.minorticks_off()
    ax.set_xlim(0, STEPS)
    ax.set_xlabel("Expected update (accuracy held at 0.5)")
    ax.set_ylabel("Mean response length (tokens)")
    ax.set_title("The per-response average lengthens wrong responses only when their length is separable",
                 loc="left")
    ax.grid(axis="y", color="#dddddd", lw=0.6)
    print(f"after {STEPS} steps: correct {tr[-1, 0]:.1f}, wrong {tr[-1, 1]:.1f}; "
          f"E[sigma_R] {e_sigma(P):.4f}, c(0.01) {c_len(S0):.5f}")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    main()
