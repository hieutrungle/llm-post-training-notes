"""Part 1 checks: recompute every number the post quotes.

Run:  python checks/part01_checks.py

Each entry in QUOTES pairs a computed value with the exact string the post prints. The script fails
if the computed value, rounded as the post rounds it, differs from that string, or (once the post
exists) if the string does not appear in posts/part-01-capability-and-pass-at-k.md.

It also
  * checks numerically that the pass@k estimator 1 - C(n-c,k)/C(n,k) is unbiased (exact expectation
    over c ~ Binomial(n, p), plus a Monte Carlo run), and that the plug-in 1 - (1 - c/n)^k is not;
  * re-runs the seeded simulation behind the "pass@k microscope" animation
    (blog assets/js/anim/part-01-microscope.js) and checks the counts it shows;
  * re-reads the experiment numbers from sharpen-or-expand's public main branch, pinned to a commit.
"""

from __future__ import annotations

import json
import math
import random
import sys
import urllib.request
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POST = ROOT / "posts" / "part-01-capability-and-pass-at-k.md"

# sharpen-or-expand, public main, the commit the post links to
SOE_COMMIT = "62b86c98a37916977e9048dbacfed2c7deaf68e9"
SOE_RAW = f"https://raw.githubusercontent.com/hieutrungle/sharpen-or-expand/{SOE_COMMIT}/"

QUOTES: list[tuple[str, float, str, str]] = []  # (label, value, format, string in the post)


def quote(label: str, value: float, fmt: str, text: str) -> None:
    QUOTES.append((label, value, fmt, text))


def pct(x: float) -> str:
    """Percent as the post and the animation print it: never 100% for a probability below 1."""
    return ">99.9%" if x >= 0.9995 else f"{100 * x:.1f}%"


def pass_k(p: float, k: int) -> float:
    """pass@k for one problem with single-sample success probability p."""
    return 1.0 - (1.0 - p) ** k


def pass_k_hat(n: int, c: int, k: int) -> float:
    """Unbiased estimator of Chen et al. (2021); needs n >= k."""
    if k > n:
        raise ValueError("needs n >= k")
    return 1.0 - comb(n - c, k) / comb(n, k)


def plug_in(n: int, c: int, k: int) -> float:
    return 1.0 - (1.0 - c / n) ** k


def upper95(n: int) -> float:
    """One-sided 95% upper bound on p after 0 successes in n trials: (1-p)^n = 0.05."""
    return 1.0 - 0.05 ** (1.0 / n)


def k_half(p: float) -> int:
    """Smallest k with pass@k >= 0.5."""
    return math.ceil(math.log(0.5) / math.log(1.0 - p))


def entropy(q: list[float]) -> float:
    return -sum(x * math.log(x) for x in q if x > 0)


# --- Section: pass@k worked numbers --------------------------------------------------------------
quote("p=0.1 pass@8", pass_k(0.1, 8), "{:.3f}", "0.570")
quote("p=0.1 pass@32", pass_k(0.1, 32), "{:.3f}", "0.966")
quote("p=0.001 pass@64 (%)", 100 * pass_k(0.001, 64), "{:.1f}", "6.2")
quote("p=0.001 pass@1000 (%)", 100 * pass_k(0.001, 1000), "{:.1f}", "63.2")

# Microscope table (animation stops and the table in the post), percent
MICRO_P = [0.4, 0.1, 0.05, 0.01, 0.001]
MICRO_K = [1, 4, 16, 64, 256, 1024]
for p in MICRO_P:
    for k in MICRO_K:
        v = 100 * pass_k(p, k)
        quote(f"micro p={p} k={k} (%)", v, "{:.1f}", f"{v:.1f}")  # table row; self-consistent
quote("k for pass@k = 0.5 at p=0.4", k_half(0.4), "{:.0f}", "2")
quote("k for pass@k = 0.5 at p=0.1", k_half(0.1), "{:.0f}", "7")
quote("k for pass@k = 0.5 at p=0.05", k_half(0.05), "{:.0f}", "14")
quote("k for pass@k = 0.5 at p=0.01", k_half(0.01), "{:.0f}", "69")
quote("k for pass@k = 0.5 at p=0.001", k_half(0.001), "{:.0f}", "693")
# rule of thumb: k_half ~ ln 2 / p
quote("ln 2", math.log(2), "{:.2f}", "0.69")

# --- Section: same p, different breadth (Jensen) -------------------------------------------------
quote("A avg pass@1", (0.4 + 0.4) / 2, "{:.1f}", "0.4")
quote("A avg pass@2", (pass_k(0.4, 2) + pass_k(0.4, 2)) / 2, "{:.2f}", "0.64")
quote("B p1=0.8 pass@2", pass_k(0.8, 2), "{:.2f}", "0.96")
quote("B avg pass@2", (pass_k(0.8, 2) + pass_k(0.0, 2)) / 2, "{:.2f}", "0.48")
# Benchmark pass@k at large k for the figure text
quote("A pass@8", pass_k(0.4, 8), "{:.3f}", "0.983")
quote("B pass@8", (pass_k(0.8, 8) + 0) / 2, "{:.3f}", "0.500")
# concavity: f_k''(p) = -k(k-1)(1-p)^(k-2) < 0 for k >= 2, p in [0, 1)
# Jensen check: f(mean) >= mean of f for random pairs, strictly when the pair differs
rng_j = random.Random(1)
for k in (2, 4, 16, 64):
    for _ in range(10_000):
        a, b = rng_j.random(), rng_j.random()
        assert pass_k((a + b) / 2, k) >= (pass_k(a, k) + pass_k(b, k)) / 2 - 1e-15, (k, a, b)
    assert -k * (k - 1) * (1 - 0.5) ** (k - 2) < 0  # second derivative at an interior point

# --- Section: sharpening example -----------------------------------------------------------------
quote("p before", 0.08 + 0.02, "{:.2f}", "0.10")
quote("p after", 0.45 + 0.05, "{:.2f}", "0.50")
quote("before: share of correct mass on strategy A", 0.08 / 0.10, "{:.2f}", "0.80")
quote("after: share of correct mass on strategy A", 0.45 / 0.50, "{:.2f}", "0.90")
quote("before pass@8", pass_k(0.10, 8), "{:.3f}", "0.570")
quote("after pass@8", pass_k(0.50, 8), "{:.3f}", "0.996")

# --- Section: zero successes ---------------------------------------------------------------------
quote("0/64 exact upper bound (%)", 100 * upper95(64), "{:.1f}", "4.6")
quote("0/64 rule of three (%)", 100 * 3 / 64, "{:.1f}", "4.7")
quote("0/1024 exact upper bound (%)", 100 * upper95(1024), "{:.2f}", "0.29")
quote("0/1024 rule of three (%)", 100 * 3 / 1024, "{:.2f}", "0.29")
quote("0/16 exact upper bound (%)", 100 * upper95(16), "{:.0f}", "17")
quote("0/16 rule of three (%)", 100 * 3 / 16, "{:.0f}", "19")
quote("-ln 0.05", -math.log(0.05), "{:.3f}", "2.996")
quote("P(0/16 | p=0.1) (%)", 100 * 0.9**16, "{:.1f}", "18.5")
rel = max((3 / n - upper95(n)) / upper95(n) for n in range(64, 100_000))
assert 0 < rel < 0.03, rel  # rule of three within 3% of the exact bound for n >= 64
print(f"rule of three vs exact bound, n >= 64: max relative gap {rel:.4f}")
# chance of 0/64 when p equals the bound is 5% by construction; at p = 0.02 it is
quote("P(0/64 | p=0.02) (%)", 100 * (0.98**64), "{:.0f}", "27")

# --- Section: the estimator ----------------------------------------------------------------------
quote("C(14,8)", comb(14, 8), "{:.0f}", "3003")
quote("C(16,8)", comb(16, 8), "{:.0f}", "12870")
quote("estimate n=16 c=2 k=8", pass_k_hat(16, 2, 8), "{:.3f}", "0.767")
quote("plug-in n=16 c=2 k=8", plug_in(16, 2, 8), "{:.3f}", "0.656")


def expect(f, n: int, k: int, p: float) -> float:
    return sum(comb(n, c) * p**c * (1 - p) ** (n - c) * f(n, c, k) for c in range(n + 1))


worst_unbiased = 0.0
for n in (8, 16, 64):
    for k in range(1, n + 1):
        for p in (0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.9):
            worst_unbiased = max(worst_unbiased, abs(expect(pass_k_hat, n, k, p) - pass_k(p, k)))
assert worst_unbiased < 1e-12, worst_unbiased
print(f"unbiasedness: max |E[estimate] - pass@k| over n in (8,16,64), all k <= n, 7 p values = {worst_unbiased:.1e}")

# Plug-in bias at the worked example (n = 16, k = 8, p = 0.1): truth 0.570
quote("E[plug-in] n=16 k=8 p=0.1", expect(plug_in, 16, 8, 0.1), "{:.3f}", "0.491")
quote("E[estimate] n=16 k=8 p=0.1", expect(pass_k_hat, 16, 8, 0.1), "{:.3f}", "0.570")

# Monte Carlo sanity check of the same example
rng_mc = random.Random(0)
T = 200_000
est = plug = 0.0
for _ in range(T):
    c = sum(rng_mc.random() < 0.1 for _ in range(16))
    est += pass_k_hat(16, c, 8)
    plug += plug_in(16, c, 8)
print(f"Monte Carlo ({T} draws, n=16, k=8, p=0.1): mean estimate {est / T:.4f}, mean plug-in {plug / T:.4f}, "
      f"true {pass_k(0.1, 8):.4f}")
assert abs(est / T - pass_k(0.1, 8)) < 0.003

# --- Section: reasoning diversity ----------------------------------------------------------------
qA = [1.0]
qB = [0.1 / 0.4] * 4
quote("model B q per strategy", qB[0], "{:.2f}", "0.25")
quote("N_eff A", math.exp(entropy(qA)), "{:.0f}", "1")
quote("N_eff B", math.exp(entropy(qB)), "{:.0f}", "4")
quote("H_correct B = ln 4", entropy(qB), "{:.2f}", "1.39")
# uneven example: correct mass 0.3 on one strategy and 0.1 on another (total 0.4)
qU = [0.75, 0.25]
quote("uneven N_eff", math.exp(entropy(qU)), "{:.2f}", "1.75")
# Q3: one strategy at 0.3 vs ten at 0.03
quote("Q3 total B", 10 * 0.03, "{:.1f}", "0.3")
quote("Q3 pass@64", pass_k(0.3, 64), "{:.6f}", "1.000000")
quote("Q3 N_eff B", math.exp(entropy([0.1] * 10)), "{:.0f}", "10")

# --- Check yourself ------------------------------------------------------------------------------
quote("Q1 SFT pass@64 at p=0.01 (%)", 100 * pass_k(0.01, 64), "{:.0f}", "47")
quote("Q2 A pass@16 (%)", 100 * pass_k(0.4, 16), "{:.2f}", "99.97")
quote("Q2 B pass@16 (%)", 100 * (pass_k(0.8, 16) + 0) / 2, "{:.2f}", "50.00")
quote("Q4 10/64", 10 / 64, "{:.2f}", "0.16")
quote("Q4 P(0/64 | p=0.03) (%)", 100 * 0.97**64, "{:.0f}", "14")

# --- Animation: seeded simulation (must match part-01-microscope.js) ------------------------------
ANIM_SEED = 20261242  # chosen so the simulated fractions stay within 2 runs of the exact values
ANIM_RUNS = 20
M32 = 0xFFFFFFFF


def xorshift32(seed: int):
    s = (seed & M32) or 1

    def r() -> float:
        nonlocal s
        s = (s ^ (s << 13)) & M32
        s ^= s >> 17
        s = (s ^ (s << 5)) & M32
        return s / 4294967296

    return r


def first_hits() -> list[list[int]]:
    """Per lane, the index of the first correct sample in each simulated run (geometric draw)."""
    r = xorshift32(ANIM_SEED)
    lanes = []
    for p in MICRO_P:
        lanes.append(sorted(math.floor(math.log(1 - r()) / math.log(1 - p)) + 1 for _ in range(ANIM_RUNS)))
    return lanes


hits = first_hits()
print("animation runs solved by k (out of 20):")
for p, h in zip(MICRO_P, hits):
    print(f"  p={p:<6}", {k: sum(x <= k for x in h) for k in MICRO_K})
worst = max(abs(sum(x <= k for x in h) / ANIM_RUNS - pass_k(p, k))
            for p, h in zip(MICRO_P, hits) for k in [2**i for i in range(11)])
assert worst <= 0.1 + 1e-12, worst
quote("anim p=0.001 runs solved by 1024", sum(x <= 1024 for x in hits[4]), "{:.0f}", "12")
quote("anim p=0.001 runs with 0/1024", sum(x > 1024 for x in hits[4]), "{:.0f}", "8")
quote("anim p=0.01 runs solved by 64", sum(x <= 64 for x in hits[3]), "{:.0f}", "9")

# --- In my experiments: re-read from the committed report -----------------------------------------


def fetch(path: str) -> bytes:
    with urllib.request.urlopen(SOE_RAW + path, timeout=60) as f:
        return f.read()


report = json.loads(fetch("docs/reports/qwen35-4b-base_soe-eval-v4_report.json"))
hard = report["hard"]["all"]
quote("hard subset problems", hard["problems"], "{:.0f}", "233")
quote("hard pass@1", hard["pass@1"], "{:.4f}", "0.0695")
quote("hard pass@16", hard["pass@16"], "{:.4f}", "0.3621")
quote("hard pass@64", hard["pass@64"], "{:.4f}", "0.4893")
solved64 = round(hard["pass@64"] * hard["problems"])
assert abs(solved64 / hard["problems"] - hard["pass@64"]) < 5e-5  # pass@64 with n = 64 is a count share
quote("hard problems with >= 1 success in 64", solved64, "{:.0f}", "114")
quote("hard problems with 0 of 64", hard["problems"] - solved64, "{:.0f}", "119")
quote("hard truncation rate (%)", 100 * hard["truncation_rate"], "{:.0f}", "90")

protocol = fetch("docs/decisions/004-evaluation-protocol.md").decode()
for s in ("0.6 / 0.95", "16,384", "16 (main); 64 (hard subset", "at least\n0.5"):
    assert s in protocol, f"protocol text changed: {s!r}"
rollout = fetch("docs/decisions/017-rl-rollout-sampling-1.0.md").decode()
assert "temperature 1.0, top-p 1.0" in rollout

# --- Compare with the text -----------------------------------------------------------------------
post = POST.read_text() if POST.exists() else None
bad = []
for label, value, fmt, text in QUOTES:
    got = fmt.format(value)
    status = "ok" if got == text else "MISMATCH"
    if got != text:
        bad.append(label)
    if post is not None and text not in post and not label.startswith("micro "):
        status += " (not in post)"
        bad.append(label + " [not in post]")
    print(f"{status:<18} {label:<44} computed {got:<12} text {text}")

if post is not None:
    # the microscope table rows in the post must equal the computed values
    for p in MICRO_P:
        row = " | ".join(pct(pass_k(p, k)) for k in MICRO_K)
        if row not in post:
            bad.append(f"microscope row p={p} [not in post]: {row}")

if bad:
    print("\nFAILED:", *bad, sep="\n  ")
    sys.exit(1)
print(f"\nall {len(QUOTES)} quoted values match" + ("" if post is not None else " (post not written yet)"))
