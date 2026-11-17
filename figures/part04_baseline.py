"""Part 4 figure: how noisy is a REINFORCE gradient estimate, as a function of a constant baseline b?

Two problems from the opening animation, one logit per solution strategy (the strategy with probability 0
is left out; it contributes nothing):
  easy: strategy probabilities (0.30, 0.02 | 0.33, 0.20, 0.15), the first two correct, p = 0.32;
  hard: (0.02 | 0.50, 0.30, 0.18), only the first correct, p = 0.02.
The estimate from a group of G = 8 samples is (1/G) sum_i (R_i - b)(e_{s_i} - pi), the gradient with
respect to the logits. Its mean is the true gradient pi_k (R_k - p) for every constant b. The y-axis is
its relative noise, sqrt(trace Cov) / |true gradient|, computed exactly. A reward offset is a baseline:
rewards 10 and 11 behave like rewards 0 and 1 with b = -10.
checks/part04_checks.py recomputes the values quoted in the post.
Run: python figures/part04_baseline.py  ->  assets/part-04/baseline.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
PROBLEMS = [("easy problem, p = 0.32", np.array([0.30, 0.02, 0.33, 0.20, 0.15]), np.array([1, 1, 0, 0, 0.0]), "#08519c"),
            ("hard problem, p = 0.02", np.array([0.02, 0.50, 0.30, 0.18]), np.array([1, 0, 0, 0.0]), "#d95f02")]
G = 8


def noise(pi: np.ndarray, R: np.ndarray, b: np.ndarray) -> np.ndarray:
    K = len(pi)
    S = np.eye(K) - pi                          # row s: e_s - pi
    p = pi @ R
    true = pi * (R - p)
    n2 = (S**2).sum(1)
    # E|(R-b) s|^2 - |true|^2, vectorized over b
    second = ((pi * n2)[None, :] * (R[None, :] - b[:, None]) ** 2).sum(1)
    return np.sqrt((second - true @ true) / G) / np.linalg.norm(true)


def main(out: Path = Path(__file__).resolve().parents[1] / "assets/part-04/baseline.png") -> None:
    b = np.linspace(-0.5, 1.5, 801)
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    for name, pi, R, col in PROBLEMS:
        y = noise(pi, R, b)
        ax.plot(b, y, color=col, lw=2.0)
        p = float(pi @ R)
        for bb, mk in ((0.0, "o"), (p, "D")):
            ax.plot([bb], noise(pi, R, np.array([bb])), mk, ms=5.5, color=col, mfc="white" if mk == "o" else col,
                    mew=1.5, zorder=4)
        ax.text(1.53, y[-1], name, color=col, va="center", ha="left", fontsize=9)
        print(f"{name}: noise b=0 {noise(pi, R, np.array([0.0]))[0]:.3f}, b=p {noise(pi, R, np.array([p]))[0]:.3f}, "
              f"min {y.min():.3f} at b={b[y.argmin()]:.3f}, b=-10 {noise(pi, R, np.array([-10.0]))[0]:.1f}")
    ax.plot([], [], "o", ms=5.5, color="#555555", mfc="white", mew=1.5, label="b = 0 (no baseline)")
    ax.plot([], [], "D", ms=5.5, color="#555555", label="b = p (success probability)")
    ax.axhline(1.0, color="#888888", lw=0.8, ls=":")
    ax.text(0.0, 1.07, "noise as large as the signal", fontsize=9, color="#555555", va="bottom")
    ax.set_yscale("log")
    ax.set_ylim(0.3, 30)
    ax.set_yticks([0.3, 1, 3, 10], ["0.3", "1", "3", "10"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_xlabel("Constant baseline b")
    ax.set_ylabel("Relative noise of the gradient\nestimate from 8 samples")
    ax.set_title("A baseline near p removes noise on the easy problem, not on the hard one", loc="left")
    ax.legend(loc="upper left", frameon=False, handlelength=1.2, borderaxespad=0.2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    main()
