"""Part 1 figure: how large can p still be after 0 successes in n samples?

One-sided 95% upper bound, the solution of (1 - p)^n = 0.05, against the rule of three (3 / n).
Run: python figures/part01_zero_bound.py  ->  assets/part-01/zero_bound.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
EXACT, RULE = "#1f77b4", "#d95f02"


def upper95(n):
    return 1.0 - 0.05 ** (1.0 / n)


def main(out: Path = Path(__file__).resolve().parents[1] / "assets/part-01/zero_bound.png") -> None:
    n = np.unique(np.logspace(3, 12, 200, base=2).round().astype(int))
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.fill_between(n, 1e-4, upper95(n), color=EXACT, alpha=0.10, lw=0)
    ax.plot(n, upper95(n), color=EXACT, lw=2.0, label=r"exact bound: solve $(1-p)^n = 0.05$")
    ax.plot(n, 3 / n, color=RULE, lw=1.6, ls="--", label=r"rule of three: $3/n$")
    ax.legend(frameon=False, loc="upper right")
    for m, lab in [(16, "0/16: 17%"), (64, "0/64: 4.6%"), (1024, "0/1024: 0.29%")]:
        ax.plot([m], [upper95(m)], "o", color=EXACT, ms=6, zorder=3)
        ax.annotate(lab, (m, upper95(m)), xytext=(7, 5), textcoords="offset points", ha="left", va="bottom")
    ax.text(10, 0.0016, "values of p still compatible\nwith zero successes", color=EXACT, va="bottom")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([8, 16, 64, 256, 1024, 4096], ["8", "16", "64", "256", "1024", "4096"])
    ax.set_yticks([0.001, 0.01, 0.1, 0.3], ["0.1%", "1%", "10%", "30%"])
    ax.set_xlim(8, 4096)
    ax.set_ylim(5e-4, 0.45)
    ax.set_xlabel("Samples drawn, n (all of them failed)")
    ax.set_ylabel("95% upper bound on p")
    ax.set_title("Zero successes in n samples still allows p up to about 3/n", loc="left")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    for m in (16, 64, 1024):
        print(f"n={m}: exact {upper95(m):.5f}, 3/n {3 / m:.5f}")


if __name__ == "__main__":
    main()
