"""Part 6 figure: zero-variance groups take over the batch, and resampling cannot create a success.

Left: the opening animation's batch (12 prompts, each with its own logit, G = 8, Part 5's group
advantage with the population standard deviation, step size 2). For each round, the expected share of
the batch's groups (equivalently, rollouts) that are all wrong, mixed or all correct, computed exactly
by enumerating every prompt's distribution over logits (at most 1,001 states per prompt).
Right: a prompt with success probability p whose groups are all wrong gets no update, so p stays fixed
while it is resampled; after k groups of 8 it has produced at least one success with probability
1 - (1 - p)^(8k). checks/part06_checks.py recomputes the quoted values.
Run: python -B figures/part06_zero_variance.py  ->  assets/part-06/zero-variance.png
"""
from collections import defaultdict
from math import comb, exp, log, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
P0 = [1e-6, 1e-4, 0.005, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 0.95]
G, ETA, ROUNDS = 8, 2.0, 10
WRONG, MIXED, CORRECT = "#a8a29b", "#1f4f8f", "#b7862c"


def sig(z):
    return 1 / (1 + exp(-z))


def expected_regimes():
    """Expected number of all-wrong, mixed and all-correct groups in each round (exact)."""
    dists = [{round(log(p / (1 - p)), 12): 1.0} for p in P0]
    out = []
    for _ in range(ROUNDS):
        ew = em = ec = 0.0
        new = []
        for d in dists:
            nd = defaultdict(float)
            for z, pr in d.items():
                p = sig(z)
                for k in range(G + 1):
                    pk = pr * comb(G, k) * p**k * (1 - p) ** (G - k)
                    if k == 0:
                        ew += pk
                        nd[z] += pk
                    elif k == G:
                        ec += pk
                        nd[z] += pk
                    else:
                        em += pk
                        nd[round(z + ETA * sqrt(k * (G - k)) / G, 12)] += pk
            new.append(nd)
        dists = new
        out.append((ew, em, ec))
    return np.array(out) / len(P0)


def main(out: Path = Path(__file__).resolve().parents[1] / "assets/part-06/zero-variance.png"):
    sh = expected_regimes()
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9), gridspec_kw={"wspace": 0.28, "width_ratios": [1, 1.05]})
    ax = axes[0]
    r = np.arange(1, ROUNDS + 1)
    ax.bar(r, sh[:, 0], color=WRONG, width=0.72, label="all wrong")
    ax.bar(r, sh[:, 1], bottom=sh[:, 0], color=MIXED, width=0.72, label="mixed (the only groups with signal)")
    ax.bar(r, sh[:, 2], bottom=sh[:, 0] + sh[:, 1], color=CORRECT, width=0.72, label="all correct")
    for i in (0, ROUNDS - 1):
        zv = sh[i, 0] + sh[i, 2]
        ax.text(r[i], 1.02, f"{zv:.1%}\nzero-variance", ha="center", va="bottom", fontsize=9, color="#333333")
    ax.set_xticks(r)
    ax.set_xlabel("Round of training")
    ax.set_ylabel("Expected share of the batch's groups")
    ax.set_ylim(0, 1.22)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25%", "50%", "75%", "100%"])
    ax.set_title("Zero-variance groups take over the batch", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False, handlelength=1.2,
              columnspacing=1.0)

    ax = axes[1]
    k = np.logspace(0, 5, 400)
    for p, col, lab in ((1e-2, "#08306b", "p = 0.01"), (1e-3, "#2171b5", "p = 0.001"),
                        (1e-4, "#6baed6", "p = 0.0001"), (1e-6, "#a8a29b", "p = 0.000001")):
        y = 1 - (1 - p) ** (G * k)
        ax.plot(k, y, color=col, lw=2.2 if p == 1e-4 else 1.6)
        if p > 1e-5:                                 # label just right of where the curve crosses 3/4
            ax.text(log(4) / (G * p) * 1.3, 0.75, lab, color=col, fontsize=9, va="center", ha="left")
        else:                                        # the slowest curve: label above its right end
            ax.text(9e4, 0.66, lab, color=col, fontsize=9, va="center", ha="right")
    y100 = 1 - (1 - 1e-4) ** (G * 100)
    ax.plot([100], [y100], "o", ms=5.5, color="#6baed6", zorder=5)
    ax.text(100 * 1.25, y100 - 0.03, f"{y100:.1%}", fontsize=9, va="center", ha="left", color="#333333")
    ax.set_xscale("log")
    ax.set_xlim(1, 1e5)
    ax.set_xticks([1, 10, 100, 1000, 10000, 100000], ["1", "10", "100", "1k", "10k", "100k"])
    ax.minorticks_off()
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("Groups of 8 sampled for one prompt, k")
    ax.set_ylabel("P(at least one success)")
    ax.set_title("Resampling a hard prompt rarely finds a success", loc="left")
    ax.grid(axis="y", color="#dddddd", lw=0.6)
    print("zero-variance share by round:", " ".join(f"{a + c:.4f}" for a, _, c in sh),
          f"| p=1e-4 after 100 groups: {y100:.4f}")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    main()
