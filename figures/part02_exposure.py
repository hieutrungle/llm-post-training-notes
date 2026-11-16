"""Part 2 figure: teacher forcing vs free-running generation (an illustration with a made-up error rate).

A toy model of exposure bias. At every step on the demonstrated path the model makes a mistake with
probability eps = 0.001. Under teacher forcing the next prefix is the demonstration's again, so mistakes
never propagate: the expected count is T * eps. When the model generates freely, a mistake moves it to
a prefix it never saw in training. From there it returns to a good path with probability r per step
and otherwise keeps making mistakes. r = 0 is the worst case behind the T^2 eps bound of Ross and
Bagnell (cited in the DAgger paper).
Run: python figures/part02_exposure.py  ->  assets/part-02/exposure.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
TF, REC, NOREC, GUIDE = "#1f77b4", "#7570b3", "#d95f02", "#888888"
EPS = 1e-3


def mistakes(T: int, eps: float, r: float) -> np.ndarray:
    """Expected cumulative mistakes after t = 1..T free-running steps (same model as checks/part02_checks.py)."""
    on, tot, out = 1.0, 0.0, np.empty(T)
    for t in range(T):
        off = 1.0 - on
        tot += on * eps + off * (1.0 - r)
        on = on * (1.0 - eps) + off * r
        out[t] = tot
    return out


def main(out: Path = Path(__file__).resolve().parents[1] / "assets/part-02/exposure.png") -> None:
    T = 4000
    t = np.arange(1, T + 1)
    free0, free1 = mistakes(T, EPS, 0.0), mistakes(T, EPS, 0.1)
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    ax.plot(t, t * EPS, color=TF, lw=2.0)
    ax.plot(t, free1, color=REC, lw=2.0)
    ax.plot(t, free0, color=NOREC, lw=2.0)
    ax.plot(t, t**2 * EPS / 2, color=GUIDE, lw=1.2, ls="--")
    # direct labels at the right end of each line
    ax.text(T * 1.08, T * EPS, "teacher forcing: $T\\varepsilon$", color=TF, va="center")
    ax.text(T * 1.08, free1[-1], "free-running, recovers\nwith prob. 0.1 per step", color=REC, va="center")
    ax.text(T * 1.08, free0[-1] * 1.25, "free-running,\nnever recovers", color=NOREC, va="center")
    ax.text(330, 330**2 * EPS / 2 * 2.2, "$T^2\\varepsilon/2$", color=GUIDE, ha="right", va="bottom")
    for T0, lab, dy in [(100, "4.9", 1.5), (1000, "368", 1.6)]:
        ax.plot([T0], [free0[T0 - 1]], "o", color=NOREC, ms=5, zorder=3)
        ax.annotate(lab, (T0, free0[T0 - 1]), xytext=(-6, 6), textcoords="offset points", ha="right", color=NOREC)
    ax.plot([1000], [free1[999]], "o", color=REC, ms=5, zorder=3)
    ax.annotate("9.8", (1000, free1[999]), xytext=(6, -12), textcoords="offset points", ha="left", color=REC)
    ax.plot([1000], [1.0], "o", color=TF, ms=5, zorder=3)
    ax.annotate("1", (1000, 1.0), xytext=(6, -12), textcoords="offset points", ha="left", color=TF)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(10, T * 1.05)
    ax.set_ylim(5e-3, 1e4)
    ax.set_xticks([10, 100, 1000], ["10", "100", "1,000"])
    ax.set_yticks([0.01, 0.1, 1, 10, 100, 1000], ["0.01", "0.1", "1", "10", "100", "1,000"])
    ax.set_xlabel("Response length T (tokens)")
    ax.set_ylabel("Expected mistakes in the response")
    ax.set_title("Same per-step error rate (0.001), very different number of mistakes", loc="left")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    for T0 in (100, 1000):
        print(f"T={T0}: teacher forcing {T0 * EPS:.2f}, recovers {free1[T0 - 1]:.2f}, never recovers {free0[T0 - 1]:.2f}")


if __name__ == "__main__":
    main()
