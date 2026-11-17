"""Part 5 figure: what dividing by the group standard deviation does to one prompt's update.

Binary reward, group of G samples, advantage A_i = R_i - mean (centering only) or
A_i = (R_i - mean) / sd with the population standard deviation (0 when every reward is equal).
Given its reward, a sample's score has E[grad log pi | R = 1] = grad p / p and E[grad log pi | R = 0] =
-grad p / (1 - p), so the expected update for one prompt is w_G(p) * grad p, with
  w_G(p) = E[ A_1 (R_1 / p - (1 - R_1) / (1 - p)) ],
computed here exactly by summing over the number k of correct samples. Centering alone gives the
constant (G - 1)/G; with the standard deviation w depends on p, and for large G it tends to
1 / sqrt(p (1 - p)), the derivative of 2 arcsin(sqrt(p)).
checks/part05_checks.py recomputes the quoted values.
Run: python -B figures/part05_weight.py  ->  assets/part-05/weight.png
"""
from math import comb
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
TICKS = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999]
TICKL = ["0.001", "0.01", "0.1", "0.5", "0.9", "0.99", "0.999"]


def weight(G: int, p: np.ndarray, std: bool = True) -> np.ndarray:
    out = np.zeros_like(p)
    for k in range(1, G):
        m = k / G
        sd = np.sqrt(m * (1 - m)) if std else 1.0
        a_pos, a_neg = (1 - m) / sd, -m / sd
        pk = comb(G, k) * p**k * (1 - p) ** (G - k)
        out += pk * ((k / G) * a_pos / p - ((G - k) / G) * a_neg / (1 - p))
    return out


def main(out: Path = Path(__file__).resolve().parents[1] / "assets/part-05/weight.png"):
    p = np.logspace(-3, np.log10(0.5), 1500)
    p = np.concatenate([p, 1 - p[:-1][::-1]])
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    lim = 1 / np.sqrt(p * (1 - p))
    ax.plot(p, lim, color="#222222", lw=1.3, ls=(0, (4, 3)), zorder=2)
    ax.plot(p, weight(64, p), color="#6baed6", lw=1.6, zorder=3)
    ax.plot(p, weight(8, p), color="#08519c", lw=2.6, zorder=4)
    ax.plot(p, weight(8, p, std=False), color="#d95f02", lw=2.0, zorder=3)
    ax.set_xscale("logit")
    ax.set_yscale("log")
    ax.set_xlim(1e-3, 1 - 1e-3)
    ax.set_ylim(0.5, 40)
    ax.set_xticks(TICKS, TICKL)
    ax.minorticks_off()
    ax.set_yticks([0.5, 1, 2, 5, 10, 20], ["0.5", "1", "2", "5", "10", "20"])
    ax.set_xlabel("Per-sample success probability p", labelpad=8)
    ax.set_ylabel("Weight w on the prompt's gradient")
    ax.set_title("Dividing by the standard deviation reweights prompts by p", loc="left")
    ax.text(0.0012, 0.875 * 1.08, "centering only, G = 8: (G − 1)/G = 0.875", color="#d95f02", fontsize=9, va="bottom")
    ax.text(0.0012, 2.65 * 0.86, r"÷ $\sigma_R$, G = 8", color="#08519c", fontsize=9, va="top", fontweight="bold")
    ax.text(0.0012, 6.0, r"÷ $\sigma_R$, G = 64", color="#3182bd", fontsize=9, va="top")
    ax.text(0.0035, 22, "large-G limit 1/√(p(1 − p))", color="#222222", fontsize=9, va="bottom")
    for pp in (0.01, 0.5):
        ax.plot([pp], weight(8, np.array([pp])), "o", ms=5, color="#08519c", zorder=5)
        ax.plot([pp], [1 / np.sqrt(pp * (1 - pp))], "o", ms=5, mfc="white", mec="#222222", mew=1.3, zorder=5)
    print("w_8(0.01) {:.3f}, limit {:.2f}; w_8(0.5) {:.3f}, limit {:.3f}; w_64(0.01) {:.3f}".format(
        weight(8, np.array([0.01]))[0], 1 / np.sqrt(0.0099), weight(8, np.array([0.5]))[0], 2.0,
        weight(64, np.array([0.01]))[0]))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    main()
