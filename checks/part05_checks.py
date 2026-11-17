"""Part 5 checks: recompute every number the post quotes.

Run:  python checks/part05_checks.py

Each entry in QUOTES pairs a computed value with the exact string the post prints. The script fails
if the computed value, formatted as the post formats it, differs from that string, or (once the post
exists) if the string does not appear in posts/part-05-grpo-from-first-principles.md. Entries in ANIM
are numbers the opening animation prints (blog assets/js/anim/part-05-group-advantage.js); they are
recomputed but not required in the post. The animation runs the same seeded simulation as section 0.

It also checks numerically that
  * the closed forms A+ = sqrt((G-k)/k), A- = -sqrt(k/(G-k)) match a direct computation for every k at
    G = 4, 8, 16 (population standard deviation), and that verl's convention (Bessel-corrected standard
    deviation plus a constant 1e-6) matches its own closed form and rescales by sqrt((G-1)/G);
  * by exact enumeration of every group, mean-centering gives E[(R_i - mean) grad log pi_i] =
    ((G-1)/G) grad J, while dividing by the group standard deviation gives w_G(p) grad J with a weight
    that depends on p (so across prompts the update is not proportional to the summed gradient);
  * E[grad log pi | R = 1] = grad p / p and E[grad log pi | R = 0] = -grad p / (1 - p);
  * the closed-form weight w_G(p), its large-G limit 1/sqrt(p(1-p)) = d/dp 2 arcsin sqrt(p), and its cap
    sqrt(G-1) at finite G;
  * P(mixed group) and P(at least one success), by formula and by enumeration.
The verl 0.9.1 source and the experiment design statements are re-read from pinned commits.
"""

from __future__ import annotations

import json
import math
import random
import sys
import urllib.request
from itertools import product
from math import comb, sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POST = ROOT / "posts" / "part-05-grpo-from-first-principles.md"

# sharpen-or-expand, public main, the commit the post links to
SOE_COMMIT = "62b86c98a37916977e9048dbacfed2c7deaf68e9"
SOE_RAW = f"https://raw.githubusercontent.com/hieutrungle/sharpen-or-expand/{SOE_COMMIT}/"
# verl v0.9.1 (tag commit; the repository moved from volcengine/verl to verl-project/verl)
VERL_COMMIT = "1876b06d0a3e4e71e06230be10af14492ca8a75b"
VERL_RAW = f"https://raw.githubusercontent.com/verl-project/verl/{VERL_COMMIT}/"

QUOTES: list[tuple[str, float, str, str]] = []  # (label, value, format, string in the post)
ANIM: list[tuple[str, float, str, str]] = []    # numbers the animation prints (not required in the post)


def quote(label: str, value: float, fmt: str, text: str) -> None:
    QUOTES.append((label, value, fmt, text))


def anim(label: str, value: float, fmt: str, text: str) -> None:
    ANIM.append(("anim " + label, value, fmt, text))


def signed(x: float, nd: int) -> str:
    """Signed number with a typographic minus, as the post prints it."""
    return f"{x:+.{nd}f}".replace("-", "\u2212")


def fmt_adv(a: float) -> str:
    """The animation's format for an advantage: '0', '+2.65' or '\u22120.38'."""
    return "0" if abs(a) < 0.005 else ("+" if a > 0 else "\u2212") + f"{abs(a):.2f}"


def softmax(z: list[float]) -> list[float]:
    m = max(z)
    e = [math.exp(v - m) if v > -math.inf else 0.0 for v in z]
    s = sum(e)
    return [v / s for v in e]


def group_adv(rs: list[float], ddof: int = 0, delta: float = 0.0) -> list[float]:
    """GRPO advantages of one group, computed directly. ddof=0: population sd; ddof=1: Bessel."""
    G = len(rs)
    m = sum(rs) / G
    sd = math.sqrt(sum((r - m) ** 2 for r in rs) / (G - ddof))
    if sd + delta == 0:
        return [0.0] * G
    return [(r - m) / (sd + delta) for r in rs]


def a_pm(G: int, k: int) -> tuple[float, float]:
    """Closed form, population standard deviation, 0 < k < G."""
    return sqrt((G - k) / k), -sqrt(k / (G - k))


def weight(G: int, p: float, std: bool = True, ddof: int = 0, delta: float = 0.0) -> float:
    """w_G(p) = E[A_1 (R_1/p - (1-R_1)/(1-p))], exact sum over the number k of correct responses."""
    tot = 0.0
    for k in range(G + 1):
        pk = math.exp(math.lgamma(G + 1) - math.lgamma(k + 1) - math.lgamma(G - k + 1)
                      + k * math.log(p) + (G - k) * math.log1p(-p))
        rs = [1.0] * k + [0.0] * (G - k)
        A = group_adv(rs, ddof, delta) if std else [r - k / G for r in rs]
        a_pos = A[0] if k > 0 else 0.0
        a_neg = A[-1] if k < G else 0.0
        tot += pk * ((k / G) * a_pos / p - ((G - k) / G) * a_neg / (1 - p))
    return tot


# --- 0. The opening picture (blog assets/js/anim/part-05-group-advantage.js) --------------------------
# Part 4's hard problem: strategies B (rare, correct), Z (correct, probability 0), D, E, F (wrong); one
# logit per strategy. Each round draws G = 8 samples with xorshift32 (the same generator as AL.rng in
# the blog's lib.js), computes population-sd group advantages (0 when the sd is 0), and takes one step
#     z_k += 2 * (1/G) sum_i A_i (1[k = s_i] - pi_k).
# The seed was chosen so that the first success comes early (round 3). All numbers are illustrative.
SEED, ETA, G, ROUNDS = 20264905, 2.0, 8, 11
P0 = [0.02, 0.0, 0.50, 0.30, 0.18]
REW = [1, 1, 0, 0, 0]
NAMES = "BZDEF"


def xorshift32(seed: int):
    s = (seed & 0xFFFFFFFF) or 1
    while True:
        s ^= (s << 13) & 0xFFFFFFFF
        s ^= s >> 17
        s ^= (s << 5) & 0xFFFFFFFF
        s &= 0xFFFFFFFF
        yield s / 4294967296


def draw(pi: list[float], u: float) -> int:
    c = 0.0
    for k, v in enumerate(pi):
        c += v
        if u < c:
            return k
    return max(k for k in range(len(pi)) if pi[k] > 0)


def simulate():
    rng = xorshift32(SEED)
    z = [math.log(v) if v > 0 else -math.inf for v in P0]
    hist = []
    for _ in range(ROUNDS):
        pi = softmax(z)
        s = [draw(pi, next(rng)) for _ in range(G)]
        rs = [float(REW[k]) for k in s]
        A = group_adv(rs)
        m = sum(rs) / G
        sd = math.sqrt(sum((r - m) ** 2 for r in rs) / G)
        hist.append(dict(pi=pi, s=s, rs=rs, k=int(sum(rs)), m=m, sd=sd, A=A))
        g = [0.0] * len(z)
        for a, sk in zip(A, s):
            for j in range(len(z)):
                g[j] += a * ((1.0 if j == sk else 0.0) - pi[j]) / G
        z = [z[j] + ETA * g[j] if z[j] > -math.inf else z[j] for j in range(len(z))]
    hist.append(dict(pi=softmax(z)))
    return hist


SIM = simulate()
print("Opening picture samples:", " ".join("".join(NAMES[k] for k in h["s"]) for h in SIM[:-1]))
ks = [h["k"] for h in SIM[:-1]]
assert ks == [0, 0, 1, 1, 1, 2, 4, 5, 7, 8, 8], ks
for r, h in enumerate(SIM[:-1]):
    assert h["pi"][1] == 0.0, "Z moved"
    anim(f"round {r + 1} p", h["pi"][0], "{:.3f}", f"{h['pi'][0]:.3f}")
    anim(f"round {r + 1} correct", h["k"], "{:.0f}", str(h["k"]))
    anim(f"round {r + 1} mean", h["m"], "{:.3f}", f"{h['m']:.3f}")
    anim(f"round {r + 1} std", h["sd"], "{:.2f}", f"{h['sd']:.2f}")
    for i, a in enumerate(h["A"]):
        anim(f"round {r + 1} advantage {i}", a, "adv", fmt_adv(a))
    assert abs(sum(h["A"])) < 1e-12                       # a group's advantages sum to zero
    if h["k"] == 0 or h["k"] == G:
        assert all(a == 0.0 for a in h["A"])            # zero-variance groups give no signal
        if r > 0:
            assert SIM[r + 1]["pi"] == h["pi"]          # ... and the policy does not move
    else:
        a_pos, a_neg = a_pm(G, h["k"])
        assert all(abs(a - (a_pos if rr else a_neg)) < 1e-12 for a, rr in zip(h["A"], h["rs"]))
    i = h["rs"].index(1.0) if h["k"] else None
    if i is not None:
        anim(f"strip round {r + 1}", h["A"][i], "{:.2f}" if h["A"][i] > 0.005 else "zero",
             f"{h['A'][i]:.2f}" if h["A"][i] > 0.005 else "0")
anim("final p", SIM[-1]["pi"][0], "{:.3f}", "0.969")
anim("wrong mass at the freeze (%)", 100 * (1 - SIM[9]["pi"][0]), "{:.1f}", "3.1")

first = ks.index(1)
assert first == 2 and SIM[0]["pi"] == SIM[1]["pi"] == SIM[2]["pi"] and abs(SIM[0]["pi"][0] - 0.02) < 1e-15
quote("pic group", G, "group of {:.0f}", "group of 8")
quote("pic rounds", ROUNDS, "eleven rounds", "eleven rounds")
quote("pic step size", ETA, "step size is {:.0f}", "step size is 2")
quote("pic P(success in a group of 8 at 0.02)", 1 - 0.98**8, "probability {:.3f}", "probability 0.149")
quote("pic mean wait", 1 / (1 - 0.98**8), "{:.1f} rounds on average", "6.7 rounds on average")
quote("pic p while all wrong", SIM[2]["pi"][0], "stays at {:.3f}", "stays at 0.020")
quote("pic Part 4 failure advantage", -0.02, "signed2", "\u22120.02")
quote("pic lone success", SIM[2]["A"][SIM[2]["rs"].index(1.0)], "\u2248 {:.2f}", "\u2248 2.65")
quote("pic failure in round 3", min(SIM[2]["A"]), "signed2", "\u22120.38")
quote("pic p after round 3", SIM[3]["pi"][0], "{:.3f}", "0.047")
assert ks[3] == ks[4] == 1
quote("pic p rounds 6-9", 0, "", "0.226, 0.489, 0.788 and 0.930")
assert ", ".join(f"{SIM[r]['pi'][0]:.3f}" for r in (5, 6, 7)) + f" and {SIM[8]['pi'][0]:.3f}" == "0.226, 0.489, 0.788 and 0.930"
assert ks[5:9] == [2, 4, 5, 7]
quote("pic A+ rounds 6-9", 0, "", "1.73, 1.00, 0.77 and 0.38")
assert ", ".join(f"{max(SIM[r]['A']):.2f}" for r in (5, 6, 7)) + f" and {max(SIM[8]['A']):.2f}" == "1.73, 1.00, 0.77 and 0.38"
quote("pic lone failure", min(SIM[8]["A"]), "signed2", "\u22122.65")
quote("pic freeze", SIM[9]["pi"][0], "{:.3f}", "0.969")
quote("pic wrong mass at the freeze", 1 - SIM[9]["pi"][0], "{:.1%}", "3.1%")
assert ks[9] == ks[10] == 8 and SIM[-1]["pi"][0] == SIM[9]["pi"][0]
# positive feedback: from the first success (round 3) to the freeze (after round 9): seven rounds
assert sum(1 for r in range(ROUNDS) if SIM[r + 1]["pi"][0] != SIM[r]["pi"][0]) == 7
quote("feedback", SIM[9]["pi"][0], "0.02 to {:.3f} in seven rounds", "0.02 to 0.969 in seven rounds")
print(f"Opening picture: k per round {ks}; p {SIM[0]['pi'][0]:.3f} -> {SIM[-1]['pi'][0]:.4f}")

# --- 1. Closed forms --------------------------------------------------------------------------------
worst = worst_b = 0.0
for Gc in (4, 8, 16):
    for k in range(1, Gc):
        rs = [1.0] * k + [0.0] * (Gc - k)
        A = group_adv(rs)
        a_pos, a_neg = a_pm(Gc, k)
        worst = max(worst, abs(A[0] - a_pos), abs(A[-1] - a_neg))
        assert abs(a_pos * abs(a_neg) - 1) < 1e-12 and abs(k * a_pos + (Gc - k) * a_neg) < 1e-12
        # verl: Bessel-corrected sd plus delta = 1e-6
        Ab = group_adv(rs, ddof=1, delta=1e-6)
        sd_b = sqrt(k * (Gc - k) / (Gc * (Gc - 1)))
        worst_b = max(worst_b, abs(Ab[0] - (1 - k / Gc) / (sd_b + 1e-6)), abs(Ab[-1] - (-k / Gc) / (sd_b + 1e-6)))
        assert abs(Ab[0] / A[0] - sqrt((Gc - 1) / Gc)) < 1e-5     # a constant factor, apart from delta
    for k in (0, Gc):
        assert group_adv([1.0] * k + [0.0] * (Gc - k), ddof=1, delta=1e-6) == [0.0] * Gc
assert worst < 1e-12 and worst_b < 1e-12
print(f"Closed forms match direct computation for every k at G = 4, 8, 16 (err {worst:.1e}, verl {worst_b:.1e})")

row_pos = "| Advantage of a correct response, $$A_+$$ | " + " | ".join(f"{a_pm(8, k)[0]:.3f}" for k in range(1, 8)) + " |"
row_neg = "| Advantage of a wrong response, $$A_-$$ | " + " | ".join(signed(a_pm(8, k)[1], 3) for k in range(1, 8)) + " |"
quote("table A+ row", 0, "", row_pos)
quote("table A- row", 0, "", row_neg)
quote("G=8 k=1 A+ (sqrt 7)", sqrt(7), "\u2248 {:.2f}", "\u2248 2.65")
quote("G=8 k=1 A-", a_pm(8, 1)[1], "signed2", "\u22120.38")
quote("G=8 k=7 lone failure", a_pm(8, 7)[1], "signed2", "\u22122.65")
quote("G=8 k=7 success", a_pm(8, 7)[0], "{:.2f}", "0.38")
quote("G=16 k=1 A+", sqrt(15), "\u2248 {:.2f}", "\u2248 3.87")
quote("G=16 k=1 A-", a_pm(16, 1)[1], "signed3", "\u22120.258")
vb8 = group_adv([1.0] + [0.0] * 7, ddof=1, delta=1e-6)
vb16 = group_adv([1.0] + [0.0] * 15, ddof=1, delta=1e-6)
quote("verl G=8 k=1 A+", vb8[0], "\u2248 {:.3f}", "\u2248 2.475")
quote("verl G=8 k=1 A-", vb8[-1], "signed3", "\u22120.354")
quote("pop G=8 k=1 A+ (3 d.p.)", a_pm(8, 1)[0], "instead of {:.3f}", "instead of 2.646")
quote("verl G=16 k=1 A+", vb16[0], "{:.3f}", "3.750")
quote("verl G=16 k=1 A-", vb16[-1], "signed3", "\u22120.250")
quote("pop G=16 k=1 A+ (3 d.p.)", sqrt(15), "{:.3f}", "3.873")
quote("Bessel factor G=8", sqrt(7 / 8), "{:.3f} at", "0.935 at")
quote("Q1 ratio of a lone success to a failure, G=16", a_pm(16, 1)[0] / abs(a_pm(16, 1)[1]), "{:.0f} times", "15 times")
quote("Q2 prompt B (4 of 8)", a_pm(8, 4)[0], "get {:.2f}", "get 1.00")

# --- 2. Exact enumeration: mean-centering vs dividing by the standard deviation -------------------------
rng = random.Random(5)


def enum_update(pi: list[float], rew: list[int], Gc: int, std: bool) -> list[float]:
    """E[(1/G) sum_i A_i grad_z log pi(s_i)] over every group of Gc samples, free logits z."""
    K = len(pi)
    out = [0.0] * K
    for grp in product(range(K), repeat=Gc):
        wgt = math.prod(pi[s] for s in grp)
        rs = [float(rew[s]) for s in grp]
        A = group_adv(rs) if std else [r - sum(rs) / Gc for r in rs]
        for a, s in zip(A, grp):
            for j in range(K):
                out[j] += wgt * a * ((1.0 if j == s else 0.0) - pi[j]) / Gc
    return out


Ge = 4
worst_mean = worst_std = worst_cond = 0.0
ws_seen = []
for trial in range(25):
    K = rng.randint(3, 5)
    pi = softmax([rng.gauss(0, 1.5) for _ in range(K)])
    rew = [1, 0] + [int(rng.random() < 0.4) for _ in range(K - 2)]
    p = sum(a for a, r in zip(pi, rew) if r)
    grad = [pi[j] * (rew[j] - p) for j in range(K)]          # grad_z p for free logits
    # E[grad log pi | R = 1] = grad p / p, E[grad log pi | R = 0] = -grad p / (1 - p)
    for r_val, scale in ((1, 1 / p), (0, -1 / (1 - p))):
        cond = [sum(pi[s] * ((1.0 if j == s else 0.0) - pi[j]) for s in range(K) if rew[s] == r_val)
                / (p if r_val else 1 - p) for j in range(K)]
        worst_cond = max(worst_cond, max(abs(c - scale * g) for c, g in zip(cond, grad)))
    um = enum_update(pi, rew, Ge, std=False)
    us = enum_update(pi, rew, Ge, std=True)
    worst_mean = max(worst_mean, max(abs(a - (Ge - 1) / Ge * g) for a, g in zip(um, grad)))
    wG = weight(Ge, p)
    worst_std = max(worst_std, max(abs(a - wG * g) for a, g in zip(us, grad)))
    ws_seen.append((p, wG))
assert worst_cond < 1e-12 and worst_mean < 1e-12 and worst_std < 1e-12
# the weight is not a constant: dividing by sd changes the update beyond a constant factor
ratios = [wv / ((Ge - 1) / Ge) for _, wv in ws_seen]
assert max(ratios) - min(ratios) > 0.1, ratios
# two prompts with different p: with sd the two prompts' gradients get different weights, so the summed
# update is not a multiple of grad (J1 + J2); with centering only both get (G-1)/G
pa, ra = [0.05, 0.6, 0.35], [1, 0, 0]
pb, rb = [0.5, 0.3, 0.2], [1, 0, 0]
ga = [pa[j] * (ra[j] - 0.05) for j in range(3)]
gb = [pb[j] * (rb[j] - 0.5) for j in range(3)]
eff = lambda u, g: sum(x * y for x, y in zip(u, g)) / sum(y * y for y in g)
wa, wb = eff(enum_update(pa, ra, Ge, True), ga), eff(enum_update(pb, rb, Ge, True), gb)
ma, mb = eff(enum_update(pa, ra, Ge, False), ga), eff(enum_update(pb, rb, Ge, False), gb)
assert abs(wa / wb - 1) > 0.04 and abs(ma - mb) < 1e-12 and abs(ma - (Ge - 1) / Ge) < 1e-12
print(f"Exact enumeration, groups of {Ge}: centering gives (G-1)/G x gradient (err {worst_mean:.1e}); "
      f"with sd the update is w_G(p) grad p (err {worst_std:.1e}), w/(G-1)/G from {min(ratios):.2f} to {max(ratios):.2f}; "
      f"two prompts at p = 0.05 and 0.5: weights {wa:.3f} vs {wb:.3f} with sd, {ma:.3f} vs {mb:.3f} centering only")
quote("centering factor G=8", 7 / 8, "{:.3f}", "0.875")
quote("RLOO factor", 8 / 7, "", "G/(G - 1)")

# --- 3. The weight w_G(p) and its large-G limit --------------------------------------------------------
lim = lambda p: 1 / sqrt(p * (1 - p))
# the limit is the derivative of 2 arcsin sqrt(p)
for p in (0.01, 0.1, 0.3, 0.5, 0.8):
    h = 1e-6
    fd = (2 * math.asin(sqrt(p + h)) - 2 * math.asin(sqrt(p - h))) / (2 * h)
    assert abs(fd - lim(p)) < 1e-6 * lim(p)
# large G approaches the limit (exact sum at G = 2000)
for p in (0.2, 0.5):
    assert abs(weight(2000, p) / lim(p) - 1) < 0.005, (p, weight(2000, p))
for Gc in (8, 16, 64):
    assert abs(weight(Gc, 0.3, std=False) - (Gc - 1) / Gc) < 1e-12
grid = [i / 10000 for i in range(1, 10000)]
w8 = [weight(8, p) for p in grid]
assert max(w8) <= sqrt(7) and abs(weight(8, 1e-7) - sqrt(7)) < 1e-5     # capped by sqrt(G-1)
within = [p for p, wv in zip(grid, w8) if wv >= 0.9 * lim(p)]
quote("G=8 within 10% low", min(within), "between {:.2f}", "between 0.30")
quote("G=8 within 10% high", max(within), "and {:.2f},", "and 0.70,")
assert all(0.30 <= p <= 0.70 for p in within) and len(within) == len([p for p in grid if min(within) <= p <= max(within)])
w64 = [weight(64, p) for p in grid]
within64 = [p for p, wv in zip(grid, w64) if wv >= 0.9 * lim(p)]
quote("G=64 within 10%", 0, "", "0.03 to 0.97")
assert f"{min(within64):.2f} to {max(within64):.2f}" == "0.03 to 0.97"
quote("w_8(0.5)", weight(8, 0.5), "{:.2f} at $$p = 0.5$$", "1.86 at $$p = 0.5$$")
quote("w_8(0.01)", weight(8, 0.01), "{:.2f} at $$p = 0.01$$", "2.61 at $$p = 0.01$$")
quote("ratio to centering", 0, "", "2.12 to 2.98")
assert f"{weight(8, 0.5) / 0.875:.2f} to {weight(8, 0.01) / 0.875:.2f}" == "2.12 to 2.98"
quote("cap sqrt(G-1)", sqrt(7), "\u2248 {:.2f}, the advantage", "\u2248 2.65, the advantage")
quote("limit at 0.01", lim(0.01), "{:.2f} at $$p = 0.01$$. At", "10.05 at $$p = 0.01$$. At")
quote("w_64(0.01)", weight(64, 0.01), "{:.2f} at $$p = 0.01$$", "6.74 at $$p = 0.01$$")
# for one prompt, the expected update size w * p(1-p) (one-logit model) is largest at p = 0.5
upd = [wv * p * (1 - p) for p, wv in zip(grid, w8)]
assert abs(grid[upd.index(max(upd))] - 0.5) < 1e-9
# figure legend values (the figure script prints the same numbers)
quote("fig limit at 0.5 (alt text)", lim(0.5), "limit is {:.0f} at p = 0.5", "limit is 2 at p = 0.5")

# --- 4. Group regimes and exploration budget ------------------------------------------------------------
p_any = lambda p, Gc: 1 - (1 - p) ** Gc
p_mixed = lambda p, Gc: 1 - p**Gc - (1 - p) ** Gc
# enumeration check
for p in (0.03, 0.4):
    for Gc in (4, 6):
        e_any = e_mix = 0.0
        for rs in product((0, 1), repeat=Gc):
            pr = math.prod(p if r else 1 - p for r in rs)
            e_any += pr * (sum(rs) > 0)
            e_mix += pr * (0 < sum(rs) < Gc)
        assert abs(e_any - p_any(p, Gc)) < 1e-12 and abs(e_mix - p_mixed(p, Gc)) < 1e-12
quote("P(mixed) p=0.5 G=8", p_mixed(0.5, 8), "{:.3f} for $$G = 8$$", "0.992 for $$G = 8$$")
quote("P(mixed) p=0.5 G=4", p_mixed(0.5, 4), "{:.3f} high for $$G = 4$$", "0.875 high for $$G = 4$$")
assert p_mixed(0.5, 16) > 0.9999
quote("table 0.01 G=8", p_any(0.01, 8), "{:.2%}", "7.73%")
quote("table 0.01 G=64", p_any(0.01, 64), "{:.1%}", "47.4%")
quote("table 0.02 G=8", p_any(0.02, 8), "{:.1%}", "14.9%")
quote("table 0.02 G=64", p_any(0.02, 64), "{:.1%}", "72.6%")
quote("table 1e-8 G=8", p_any(1e-8, 8), "", "8.0 \\times 10^{-8}")
assert f"{p_any(1e-8, 8):.1e}" == "8.0e-08"
quote("table 1e-8 G=64", p_any(1e-8, 64), "", "6.4 \\times 10^{-7}")
assert f"{p_any(1e-8, 64):.1e}" == "6.4e-07"
assert p_any(0.01, 8) < 1 / 12                                    # "fewer than one in twelve"
quote("fig 7.7%", p_any(0.01, 8), "{:.1%} for $$G = 8$$", "7.7% for $$G = 8$$")
quote("fig 1/G point", p_any(1 / 8, 8), "{:.3f} for $$G = 8$$", "0.656 for $$G = 8$$")
assert 0.63 < p_any(1 / 64, 64) < p_any(1 / 4, 4) < 0.69                # "about two thirds"
assert p_any(0.001, 64) < 0.07                                     # "even G = 64 rarely finds a success"
quote("Q3", p_any(0.02, 64), "\u2248 {:.1%}", "\u2248 72.6%")

# --- 5. Literature values (re-read from the papers' current arXiv versions when written) ---------------
quote("DeepSeekMath outputs per question", 64, "{:.0f} outputs per question", "64 outputs per question")
quote("DeepSeekMath KL coefficient", 0.04, "{:.2f}", "0.04")
quote("R1-Zero outputs per question", 16, "{:.0f} outputs per question", "16 outputs per question")
quote("R1-Zero KL coefficient", 0.001, "{:.3f}", "0.001")

# --- 6. verl 0.9.1: the standard-deviation convention -------------------------------------------------


def fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as fh:
        return fh.read().decode()


core = fetch(VERL_RAW + "verl/trainer/ppo/core_algos.py")
lines = core.splitlines()
assert lines[266] == "@register_adv_est(AdvantageEstimator.GRPO)  # or simply: @register_adv_est(\"grpo\")"
assert lines[267] == "def compute_grpo_outcome_advantage("
assert lines[330] == "    return scores, scores"                       # the post links lines 266-331
grpo_fn = "\n".join(lines[265:331])
assert "epsilon: float = 1e-6," in grpo_fn
assert "id2std[idx] = torch.std(scores_tensor)" in grpo_fn          # torch.std: Bessel-corrected by default
assert "scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)" in grpo_fn
assert "scores[i] = scores[i] - id2mean[index[i]]" in grpo_fn        # norm_adv_by_std_in_grpo=False (Dr.GRPO)
assert "seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / (seq_mask + 1e-8)  # token-mean" in core
trainer = fetch(VERL_RAW + "verl/trainer/ppo/ray_trainer.py")
call = trainer[trainer.index("advantages, returns = core_algos.compute_grpo_outcome_advantage("):][:400]
assert "epsilon" not in call.split(")")[0]                           # the default 1e-6 is used
quote("verl delta", 1e-6, "", "epsilon = 1e-6")
quote("verl delta (math)", 1e-6, "", "\\delta = 10^{-6}")
print("verl 0.9.1 (", VERL_COMMIT[:7], "): GRPO advantage = (score - mean) / (torch.std + 1e-6)")

# --- 7. In my experiments: design statements re-read from sharpen-or-expand main ---------------------------
flat = lambda s: " ".join(s.split())
d025 = flat(fetch(SOE_RAW + "docs/decisions/025-g0-shared-4b-config.md"))
assert "| Batch | 64 prompts × G = 8, 4 updates per step (decision 022), 2,048 prompt tokens |" in d025
assert "no KL term, no reference model (013)" in d025 and "clip 0.2 unless Clip-Higher" in d025
assert "Algorithm configs may differ only in the decision-001 switches" in d025
assert "in `name`, and in M7's group size." in d025
assert "verl 0.9.1" in d025
d001 = flat(fetch(SOE_RAW + "docs/decisions/001-gpu-stack-verl.md"))
assert "| GRPO | `algorithm.adv_estimator=grpo`, `actor.loss_agg_mode=seq-mean-token-mean`, symmetric clip |" in d001
assert ("| Dr.GRPO | GRPO + `algorithm.norm_adv_by_std_in_grpo=false`, "
        "`actor.loss_agg_mode=seq-mean-token-sum-norm` |") in d001
plan = flat(fetch(SOE_RAW + "docs/milestone-plan.md"))
assert "GRPO at G = 4, 8, 16 with shorter runs, matched by rollout tokens." in plan
assert "Group-size result stated as newly solved problems per million rollout tokens" in plan
d031 = flat(fetch(SOE_RAW + "docs/decisions/031-rl-and-dpo-start-from-the-base.md"))
assert ("Every non-development 4B RL run (GRPO, Dr.GRPO, DAPO, RS-GRPO, every seed and ablation) "
        "and DPO starts from `Qwen/Qwen3.5-4B-Base`") in d031
audit = flat(fetch(SOE_RAW + "docs/reports/m2_base_start_audit.md"))
assert "(job 3255480, the M1 4B throughput rollout dumps)" in audit
assert "untrained base at step 1 (512 rollouts)" in audit
assert "step 1 alone 41% / 22% / 37%" in audit
groups = json.loads(fetch(SOE_RAW + "docs/reports/m2_rollout_groups.json"))["full_bf16"]
s1 = groups["step_1"]
assert s1["groups"] == 64 and s1["group_sizes"] == [8] and s1["rollouts"] == 512
assert "m1_throughput_bf0_qwen35-4b" in groups["file"]
assert abs(s1["groups_all_correct"] + s1["groups_all_wrong"] + s1["groups_mixed"] - 1) < 1e-12
quote("prompts per step", 64, "{:.0f} prompts", "64 prompts")
quote("rollouts per step", 64 * 8, "({:.0f} rollouts)", "(512 rollouts)")
quote("updates per step", 4, "{:.0f} optimizer updates", "4 optimizer updates")
quote("clip", 0.2, "clip range {:.1f}", "clip range 0.2")
quote("ablation group sizes", 0, "", "$$G$$ = 4, 8 and 16")
quote("step 1 groups", s1["groups"], "of {:.0f} groups", "of 64 groups")
quote("step 1 all correct", s1["groups_all_correct"], "{:.1%} were all correct", "40.6% were all correct")
quote("step 1 all wrong", s1["groups_all_wrong"], "{:.1%} all wrong", "21.9% all wrong")
quote("step 1 mixed", s1["groups_mixed"], "{:.1%} mixed", "37.5% mixed")
quote("step 1 truncated in all-wrong groups", s1["truncated_in_all_wrong_groups"], "{:.1%} of the rollouts", "92.9% of the rollouts")
quote("token cap", 16384, "{:,}", "16,384")
quote("exploration budget p=0.01 G=8", p_any(0.01, 8), "about {:.1%}", "about 7.7%")
assert p_any(0.001, 8) < 0.01                                      # "fewer than 1%"
print("Experiment design statements found in decisions 001, 025, 031, the milestone plan and the M2 group "
      "audit at", SOE_COMMIT[:7])

# --- Compare with the text ---------------------------------------------------------------------------
post = POST.read_text() if POST.exists() else None
bad = []
for label, value, fmt, text in QUOTES + ANIM:
    if fmt == "":
        got = text                                               # checked by an assert above
    elif fmt.startswith("signed"):
        got = signed(value, int(fmt[-1]))
    elif fmt == "adv":
        got = fmt_adv(value)
    elif fmt == "zero":
        got = "0"
    else:
        got = fmt.format(value)
    status = "ok" if got == text else "MISMATCH"
    if got != text:
        bad.append(label)
    if post is not None and not label.startswith("anim ") and text not in post:
        status += " (not in post)"
        bad.append(label + " [not in post]")
    print(f"{status:<18} {label:<48} computed {got[:40]:<40} text {text[:40]}")

if bad:
    print("\nFAILED:", *bad, sep="\n  ")
    sys.exit(1)
print(f"\nall {len(QUOTES)} quoted values and {len(ANIM)} animation values match"
      + ("" if post is not None else " (post not written yet)"))
