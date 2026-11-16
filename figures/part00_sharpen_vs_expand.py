"""Part 0 illustration: sharpening vs expansion seen through pass@k.

Made-up per-problem success probabilities, not experimental data.
Run: python figures/part00_sharpen_vs_expand.py  ->  assets/part-00/sharpen_vs_expand.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

N_PROBLEMS = 1000
RARE = 0.03  # below this single-sample success rate a problem counts as "rarely solved"


def pass_at_k(p: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Benchmark-average pass@k for per-problem success probabilities p."""
    return 1.0 - np.mean((1.0 - p[None, :]) ** k[:, None], axis=1)


def make_models(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_easy, n_rare = int(0.55 * N_PROBLEMS), int(0.30 * N_PROBLEMS)
    n_out = N_PROBLEMS - n_easy - n_rare
    base = np.concatenate([
        rng.beta(2.0, 2.0, n_easy),          # often solved
        rng.uniform(0.002, RARE, n_rare),    # rarely solved: visible only at large k
        np.zeros(n_out),                     # never solved within any budget we sample
    ])
    logit = np.log(base.clip(1e-9) / (1 - base).clip(1e-9))
    common = base >= RARE
    # Sharpening: common solutions become more likely, rare ones lose mass.
    sharpened = np.where(common, 1 / (1 + np.exp(-(logit + 2.0))), base * 0.05)
    # Expansion: the same sharpening of common solutions, no loss on rare ones,
    # and 40% of the rarely/never-solved problems become reliably solvable.
    lifted = (~common) & (rng.random(N_PROBLEMS) < 0.4)
    expanded = np.where(lifted, 0.3, np.where(common, sharpened, base))
    return {"Base model": base, "Sharpened": sharpened, "Expanded": expanded}


def main(out: Path = Path(__file__).resolve().parents[1] / "assets/part-00/sharpen_vs_expand.png") -> None:
    models = make_models()
    k = np.unique(np.logspace(0, 10, 200, base=2).round().astype(int))
    colors = {"Base model": "#6b6b6b", "Sharpened": "#1f77b4", "Expanded": "#d95f02"}
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for name, p in models.items():
        y = pass_at_k(p, k)
        ax.plot(k, y, color=colors[name], lw=2.2 if name != "Base model" else 1.8,
                ls="--" if name == "Base model" else "-")
        ax.annotate(name, (k[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=colors[name])
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 16, 64, 256, 1024], ["1", "4", "16", "64", "256", "1024"])
    ax.set_xlabel("Samples per problem, k")
    ax.set_ylabel("pass@k (share of problems solved)")
    ax.set_ylim(0.25, 1.0)
    ax.set_xlim(1, 1024)
    ax.margins(x=0.02)
    ax.set_title("Sharpening wins at k = 1 but falls below the base model at large k", loc="left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    for name, p in models.items():
        print(f"{name:11s} pass@1={pass_at_k(p, np.array([1]))[0]:.3f} "
              f"pass@1024={pass_at_k(p, np.array([1024]))[0]:.3f}")


if __name__ == "__main__":
    main()
