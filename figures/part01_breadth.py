"""Part 1 figure: the same average pass@1 can give different pass@k (concavity / Jensen).

Two toy models on two problems, from the study notes: Model A has p = (0.4, 0.4), Model B has
p = (0.8, 0). Both average pass@1 = 0.4.
Run: python figures/part01_breadth.py  ->  assets/part-01/breadth.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
COLOR = {"A": "#1f77b4", "B": "#d95f02"}
MODELS = {"A": np.array([0.4, 0.4]), "B": np.array([0.8, 0.0])}


def f(p, k):
    return 1.0 - (1.0 - p) ** k


def main(out: Path = Path(__file__).resolve().parents[1] / "assets/part-01/breadth.png") -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.5), gridspec_kw={"wspace": 0.32})

    # Left: the concave curve f_2 and the two models' problem points
    p = np.linspace(0, 1, 200)
    ax1.plot(p, f(p, 2), color="#555555", lw=1.6)
    ax1.text(1.0, 0.78, r"$1-(1-p)^2$", ha="right", color="#555555")
    pa = MODELS["A"][0]
    ax1.plot([pa], [f(pa, 2)], "o", ms=8, color=COLOR["A"], zorder=3)
    ax1.annotate("Model A: p = 0.4 and p = 0.4\naverage pass@2 = 0.64", (pa, f(pa, 2)),
                 xytext=(0.0, 0.93), textcoords="data", ha="left", va="center", color=COLOR["A"],
                 arrowprops=dict(arrowstyle="-", color=COLOR["A"], lw=0.8, shrinkB=6))
    b0, b1 = MODELS["B"]
    ax1.plot([b0, b1], [f(b0, 2), f(b1, 2)], color=COLOR["B"], lw=1.2, ls="--")
    ax1.plot([b0, b1], [f(b0, 2), f(b1, 2)], "s", ms=7, color=COLOR["B"], zorder=3)
    mid = (f(b0, 2) + f(b1, 2)) / 2
    ax1.plot([0.4], [mid], "s", ms=7, mfc="white", mec=COLOR["B"], mew=1.6, zorder=3)
    ax1.annotate("Model B: p = 0.8 and p = 0\naverage pass@2 = 0.48", (0.4, mid),
                 xytext=(10, -34), textcoords="offset points", ha="left", color=COLOR["B"])
    ax1.annotate("", xy=(0.4, mid + 0.01), xytext=(0.4, f(pa, 2) - 0.01),
                 arrowprops=dict(arrowstyle="<->", color="#333333", lw=1))
    ax1.set_xlabel("Single-sample success probability p")
    ax1.set_ylabel("pass@2 of one problem")
    ax1.set_xlim(-0.03, 1.03)
    ax1.set_ylim(-0.04, 1.06)
    ax1.set_title("Same average p, but the chord lies below the curve", loc="left")

    # Right: benchmark pass@k vs k for both models
    k = np.unique(np.logspace(0, 6, 120, base=2).round().astype(int))
    for name, ps in MODELS.items():
        y = np.mean([f(pi, k) for pi in ps], axis=0)
        ax2.plot(k, y, color=COLOR[name], lw=2.0, marker=None)
        ax2.annotate(f"Model {name}", (k[-1], y[-1]), xytext=(5, 0), textcoords="offset points",
                     va="center", color=COLOR[name])
    ax2.set_xscale("log", base=2)
    ax2.set_xticks([1, 2, 4, 8, 16, 32, 64], ["1", "2", "4", "8", "16", "32", "64"])
    ax2.set_xlim(1, 64)
    ax2.set_ylim(0.35, 1.03)
    ax2.set_xlabel("Samples per problem, k")
    ax2.set_ylabel("Benchmark pass@k (2 problems)")
    ax2.set_title("Equal at k = 1, A pulls ahead as k grows", loc="left")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    for name, ps in MODELS.items():
        print(name, "pass@1", ps.mean(), "pass@2", round(np.mean(f(ps, 2)), 4),
              "pass@8", round(np.mean(f(ps, 8)), 4), "pass@64", round(np.mean(f(ps, 64)), 4))


if __name__ == "__main__":
    main()
