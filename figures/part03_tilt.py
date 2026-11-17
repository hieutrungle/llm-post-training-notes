"""Part 3 figure: what the KL-regularized optimum does to the success probability under a 0/1 reward.

With reward R = 1 for a correct trajectory and 0 otherwise, the optimum pi* = pi_ref e^{R/beta} / Z
multiplies the odds of success by e^{1/beta}:

    p* = p e^{1/beta} / (p e^{1/beta} + 1 - p).

The plot shows p* against the reference success probability p (log scale) for several beta. p = 0 stays
at 0 for every beta. The marked points are the numbers quoted in the post; checks/part03_checks.py
recomputes them.
Run: python figures/part03_tilt.py  ->  assets/part-03/tilt.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
# one hue, darker = smaller beta = stronger tilt
BETAS = [(2.0, "#9ecae1"), (1.0, "#4292c6"), (0.5, "#08519c"), (0.25, "#081d58")]
GUIDE = "#888888"


def p_star(p: np.ndarray, beta: float) -> np.ndarray:
    g = np.exp(1.0 / beta)
    return p * g / (p * g + 1.0 - p)


def main(out: Path = Path(__file__).resolve().parents[1] / "assets/part-03/tilt.png") -> None:
    p = np.logspace(-3, 0, 600)
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    for beta, col in BETAS[::-1]:
        ax.plot(p, p_star(p, beta), color=col, lw=2.0, label=f"β = {beta:g}")
    ax.plot(p, p, color=GUIDE, lw=1.2, ls="--", label="β → ∞ (the reference)")
    ax.legend(loc="upper left", frameon=False, handlelength=1.8, borderaxespad=0.2)
    # the two points quoted in the post, at beta = 1
    box = dict(fc="white", ec="none", pad=1.0)
    for p0, off, ha in ((0.01, (-7, 9), "right"), (0.30, (7, -2), "left")):
        y0 = float(p_star(np.array(p0), 1.0))
        ax.plot([p0, p0], [p0, y0], color="#4292c6", lw=1.0, ls=":")
        ax.plot([p0], [p0], "o", ms=4.5, mfc="white", mec=GUIDE, zorder=3)
        ax.plot([p0], [y0], "o", ms=5, color="#4292c6", zorder=3)
        ax.annotate(f"{p0:.2f} → {y0:.3f}", (p0, y0), xytext=off, textcoords="offset points",
                    ha=ha, va="bottom" if ha == "right" else "top", color="#2b6ca3", fontsize=9, bbox=box)
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 1.0)
    ax.set_ylim(0, 1.03)
    ax.set_xticks([0.001, 0.01, 0.1, 1], ["0.001", "0.01", "0.1", "1"])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.5", "0.75", "1"])
    ax.set_xlabel("Success probability under the reference, p")
    ax.set_ylabel("Success probability at the optimum, p*")
    ax.set_title("Smaller β tilts harder, but rare successes stay rare unless β is small", loc="left")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    for beta, _ in BETAS:
        print(f"beta={beta:g}: p*(0.01)={p_star(np.array(0.01), beta):.3f}, p*(0.30)={p_star(np.array(0.30), beta):.3f}")
    return fig


if __name__ == "__main__":
    main()
