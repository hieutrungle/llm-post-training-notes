"""Part 5 figure: group size as an exploration budget, and the learning frontier.

For a problem with per-sample success probability p and a group of G independent samples,
  left:  P(at least one success)      = 1 - (1 - p)^G
  right: P(mixed group: some right, some wrong) = 1 - p^G - (1 - p)^G,
for G = 4, 8, 16 and 64, on a logit p-axis so that both ends of [0, 1] are visible.
Only mixed groups give GRPO a nonzero advantage. checks/part05_checks.py recomputes the quoted values.
Run: python -B figures/part05_frontier.py  ->  assets/part-05/frontier.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
GROUPS = [(4, "#9ecae1", 1.4), (8, "#08519c", 2.6), (16, "#4292c6", 1.4), (64, "#08306b", 1.4)]
TICKS = [1e-4, 1e-3, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 0.9999]
TICKL = ["0.0001", "0.001", "0.01", "0.1", "0.5", "0.9", "0.99", "0.999", "0.9999"]


def p_any(p, G):
    return 1 - (1 - p) ** G


def p_mixed(p, G):
    return 1 - p**G - (1 - p) ** G


def main(out: Path = Path(__file__).resolve().parents[1] / "assets/part-05/frontier.png"):
    p = np.logspace(-4, 0, 4001)[:-1]
    p = np.concatenate([p[p < 0.5], 1 - p[p < 0.5][::-1]])
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True, gridspec_kw={"wspace": 0.14})
    for ax, f, title in ((axes[0], p_any, "A group contains a success only once p is near 1/G"),
                         (axes[1], p_mixed, "Mixed groups, the only ones with signal, need intermediate p")):
        for G, col, lw in GROUPS:
            ax.plot(p, f(p, G), color=col, lw=lw, zorder=3 if G == 8 else 2)
        ax.set_xscale("logit")
        ax.set_xlim(1e-4, 1 - 1e-4)
        ax.set_xticks(TICKS, TICKL, rotation=45)
        ax.minorticks_off()
        ax.set_xlabel("Per-sample success probability p")
        ax.set_title(title, loc="left")
        ax.set_ylim(-0.03, 1.05)
        ax.grid(axis="y", color="#dddddd", lw=0.6)
    axes[0].set_ylabel("Probability per group")
    # series identity: a frameless legend in the empty lower right of the left panel
    for G, col, lw in GROUPS:
        axes[0].plot([], [], color=col, lw=lw, label=f"G = {G}" + (" (the experiments)" if G == 8 else ""))
    axes[0].legend(loc="lower right", frameon=False, handlelength=1.6, borderaxespad=0.6)
    for G, mk in ((8, "o"), (64, "o")):
        y = p_any(0.01, G)
        axes[0].plot([0.01], [y], mk, ms=5.5, color=dict((g, c) for g, c, _ in GROUPS)[G], zorder=5)
        if G == 64:
            axes[0].text(0.0085, y, f"{y:.1%}", fontsize=9, va="center", ha="right", color="#333333")
        else:
            axes[0].text(0.0125, y - 0.015, f"{y:.1%}", fontsize=9, va="top", ha="left", color="#333333")
    print(f"P(>=1) at p=0.01: G=8 {p_any(0.01, 8):.4f}, G=64 {p_any(0.01, 64):.4f}; "
          f"P(mixed) at p=0.5: " + ", ".join(f"G={G} {p_mixed(0.5, G):.4f}" for G, _, _ in GROUPS))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    main()
