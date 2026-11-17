"""Part 3 checks: recompute every number the post quotes.

Run:  python checks/part03_checks.py

Each entry in QUOTES pairs a computed value with the exact string the post prints. The script fails
if the computed value, formatted as the post formats it, differs from that string, or (once the post
exists) if the string does not appear in posts/part-03-dpo-from-kl-regularized-rl.md. Entries in ANIM
are numbers the animation prints (blog assets/js/anim/part-03-tilt-displacement.js); they are
recomputed but not required in the post.

It also checks numerically that
  * the Gibbs policy pi* = pi_ref e^{r/beta} / Z maximizes E_pi[r] - beta KL(pi || pi_ref) on random
    discrete examples, against random and local perturbations, with optimal value beta log Z;
  * the Lagrangian stationarity condition holds at pi*;
  * the reparameterization r = beta log(pi*/pi_ref) + beta log Z recovers r, and Z cancels in
    Bradley-Terry: sigma(r_w - r_l) = sigma(beta * Delta*) for any per-prompt shift of r;
  * dL/dDelta = -beta sigma(-beta Delta) and the full parameter gradient
    beta sigma(-beta Delta) [grad log pi(y_l) - grad log pi(y_w)] match central finite differences on a
    small autoregressive model, and tokens of a shared prefix cancel exactly in Delta;
  * with a 0/1 reward the optimum multiplies the odds of success by e^{1/beta}, leaves q(tau | x) and
    N_eff unchanged, and keeps zero-probability responses at zero;
  * with free logits one DPO step on a single pair always raises pi(y_w), while with a shared hidden
    state (the toy behind the animation) both pi(y_w) and pi(y_l) fall.
Experiment design statements are re-read from sharpen-or-expand's public main branch, pinned to a commit.
"""

from __future__ import annotations

import math
import random
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POST = ROOT / "posts" / "part-03-dpo-from-kl-regularized-rl.md"

# sharpen-or-expand, public main, the commit the post links to
SOE_COMMIT = "62b86c98a37916977e9048dbacfed2c7deaf68e9"
SOE_RAW = f"https://raw.githubusercontent.com/hieutrungle/sharpen-or-expand/{SOE_COMMIT}/"

QUOTES: list[tuple[str, float, str, str]] = []  # (label, value, format, string in the post)
ANIM: list[tuple[str, float, str, str]] = []    # numbers the animation prints (not required in the post)


def quote(label: str, value: float, fmt: str, text: str) -> None:
    QUOTES.append((label, value, fmt, text))


def anim(label: str, value: float, fmt: str, text: str) -> None:
    ANIM.append(("anim " + label, value, fmt, text))


def signed(x: float, nd: int) -> str:
    """Signed number with a typographic minus, as the post prints it."""
    return f"{x:+.{nd}f}".replace("-", "\u2212")


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))


def softmax(z: list[float]) -> list[float]:
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


def kl(p: list[float], q: list[float]) -> float:
    return sum(a * math.log(a / b) for a, b in zip(p, q) if a > 0)


def n_eff(masses: list[float]) -> float:
    """Hill number of order 1 of the distribution over correct responses (Part 1)."""
    tot = sum(masses)
    q = [m / tot for m in masses if m > 0]
    return math.exp(-sum(x * math.log(x) for x in q))


def gibbs(ref: list[float], r: list[float], beta: float) -> list[float]:
    m = max(r)
    w = [a * math.exp((x - m) / beta) for a, x in zip(ref, r)]
    z = sum(w)
    return [v / z for v in w]


def objective(pi: list[float], ref: list[float], r: list[float], beta: float) -> float:
    return sum(a * x for a, x in zip(pi, r)) - beta * kl(pi, ref)


def p_star(p: float, beta: float) -> float:
    """Success mass at the KL-regularized optimum with a 0/1 reward."""
    g = math.exp(1.0 / beta)
    return p * g / (p * g + 1.0 - p)


# --- 0. The opening picture (blog assets/js/anim/part-03-dpo-picture.js) ---------------------------
# three outcomes with free logits (y_w, y_l, everything else), pi_ref = (0.20, 0.30, 0.50), beta = 1,
# step size 1: each step adds the push beta*sigma(-beta*Delta) to z_w and subtracts it from z_l.
PIC_REF = [0.20, 0.30, 0.50]


def pic_probs(cshift: float) -> list[float]:
    return softmax([math.log(PIC_REF[0]) + cshift, math.log(PIC_REF[1]) - cshift, math.log(PIC_REF[2])])


cs = [0.0]
for _ in range(6):
    d = math.log(pic_probs(cs[-1])[0] / PIC_REF[0]) - math.log(pic_probs(cs[-1])[1] / PIC_REF[1])
    cs.append(cs[-1] + 1.0 * sigmoid(-1.0 * d))
for kstep, cshift in enumerate(cs):
    pp = pic_probs(cshift)
    d = math.log(pp[0] / PIC_REF[0]) - math.log(pp[1] / PIC_REF[1])
    assert abs(d - 2 * cshift) < 1e-12
    anim(f"picture step {kstep} y_w", pp[0], "{:.3f}", f"{pp[0]:.3f}")
    anim(f"picture step {kstep} margin", d, "{:.2f}", f"{2 * cshift:.2f}")
    anim(f"picture step {kstep} push", sigmoid(-d), "{:.3f}", f"{1 / (1 + math.exp(2 * cshift)):.3f}")
# every step raises y_w and lowers y_l (free logits)
for a, b in zip(cs, cs[1:]):
    assert pic_probs(b)[0] > pic_probs(a)[0] and pic_probs(b)[1] < pic_probs(a)[1]
pf = pic_probs(cs[-1])
df = math.log(pf[0] / PIC_REF[0]) - math.log(pf[1] / PIC_REF[1])
quote("picture y_w after 6 steps", pf[0], "{:.3f}", "0.548")
quote("picture y_l after 6 steps", pf[1], "{:.3f}", "0.065")
quote("picture others after 6 steps", pf[2], "{:.2f}", "0.39")
quote("picture ratio y_w", pf[0] / PIC_REF[0], "{:.2f} times", "2.74 times")
quote("picture ratio y_l", pf[1] / PIC_REF[1], "{:.2f} times", "0.22 times")
quote("picture margin", df, "{:.2f}", "2.53")
quote("picture push start", sigmoid(0.0), "{:.2f} to 0.074", "0.50 to 0.074")
quote("picture push end", sigmoid(-df), "0.50 to {:.3f}", "0.50 to 0.074")
anim("picture step 6 y_l", pf[1], "{:.3f}", "0.065")
anim("picture step 6 ratio y_w", pf[0] / PIC_REF[0], "{:.2f}", "2.74")
anim("picture step 6 ratio y_l", pf[1] / PIC_REF[1], "{:.2f}", "0.22")

# --- 1. The Gibbs optimum ------------------------------------------------------------------------
rng = random.Random(3)
worst_gap, worst_val, worst_stat = float("inf"), 0.0, 0.0
n_alt = n_pos = 0
for trial in range(300):
    K = rng.randint(2, 9)
    ref = [rng.random() + 1e-3 for _ in range(K)]
    s = sum(ref)
    ref = [v / s for v in ref]
    r = [rng.gauss(0, 2) for _ in range(K)]
    beta = math.exp(rng.uniform(math.log(0.05), math.log(5)))
    ps = gibbs(ref, r, beta)
    best = objective(ps, ref, r, beta)
    Z = sum(a * math.exp(x / beta) for a, x in zip(ref, r))
    worst_val = max(worst_val, abs(best - beta * math.log(Z)))
    # identity: objective(pi) = beta log Z - beta KL(pi || pi*), so every other pi is lower
    for _ in range(40):
        if rng.random() < 0.5:   # random distribution
            alt = [rng.random() for _ in range(K)]
        else:                    # small perturbation of pi*
            alt = [max(1e-12, v * math.exp(rng.gauss(0, 0.05))) for v in ps]
        sa = sum(alt)
        alt = [v / sa for v in alt]
        gap = best - objective(alt, ref, r, beta)
        assert abs(gap - beta * kl(alt, ps)) < 1e-9, "objective != beta log Z - beta KL(pi || pi*)"
        worst_gap = min(worst_gap, gap)
        n_alt += 1
        n_pos += gap > 0
    # Lagrangian: r(y) - beta (log(pi/pi_ref) + 1) is the same constant (-lambda) for every y
    stat = [x - beta * (math.log(a / b) + 1) for x, a, b in zip(r, ps, ref)]
    worst_stat = max(worst_stat, max(stat) - min(stat))
assert worst_gap > -1e-12, "a perturbed policy beat the Gibbs policy"
assert worst_val < 1e-9 and worst_stat < 1e-9
print(f"Gibbs optimum: {n_alt} alternatives on 300 random problems, none better ({n_pos} strictly worse, "
      f"min gap {worst_gap:.1e}); "
      f"value = beta log Z (max err {worst_val:.1e}); Lagrangian stationarity spread {worst_stat:.1e}")

# --- 2. Reparameterization and the Z cancellation -------------------------------------------------
worst = 0.0
for trial in range(200):
    K = rng.randint(2, 8)
    ref = [rng.random() + 1e-3 for _ in range(K)]
    s = sum(ref)
    ref = [v / s for v in ref]
    r = [rng.gauss(0, 2) for _ in range(K)]
    beta = math.exp(rng.uniform(math.log(0.05), math.log(5)))
    ps = gibbs(ref, r, beta)
    Z = sum(a * math.exp(x / beta) for a, x in zip(ref, r))
    rec = [beta * math.log(a / b) + beta * math.log(Z) for a, b in zip(ps, ref)]
    worst = max(worst, max(abs(a - b) for a, b in zip(rec, r)))
    # a per-prompt shift c(x) changes Z but not pi*, and not the Bradley-Terry probability
    c = rng.gauss(0, 5)
    ps2 = gibbs(ref, [x + c for x in r], beta)
    worst = max(worst, max(abs(a - b) for a, b in zip(ps, ps2)))
    w, l = rng.sample(range(K), 2)
    delta_star = math.log(ps[w] / ref[w]) - math.log(ps[l] / ref[l])
    worst = max(worst, abs(sigmoid(r[w] - r[l]) - sigmoid(beta * delta_star)))
assert worst < 1e-9
print(f"Reparameterization recovers r; Z cancels in Bradley-Terry (max err {worst:.1e})")

# --- 3. Bradley-Terry numbers ---------------------------------------------------------------------
quote("Gibbs alternatives tested", n_alt, "{:,}", "12,000")
quote("Gibbs random problems", 300, "{:.0f} random", "300 random")
quote("free-logit cases", 500, "{:.0f} random", "500 random")
quote("BT gap 1", sigmoid(1.0), "{:.0%}", "73%")
quote("BT gap 2", sigmoid(2.0), "{:.0%}", "88%")
quote("BT gap 1 shifted by 10 (same)", sigmoid(11.0 - 10.0), "{:.0%}", "73%")

# --- 4. What beta controls: the ratio form ----------------------------------------------------------
ratio = lambda beta: (0.1 / 0.3) * math.exp(1.0 / beta)
quote("ratio beta=2", ratio(2.0), "{:.2f}", "0.55")
quote("ratio beta=1", ratio(1.0), "{:.2f}", "0.91")
quote("ratio beta=0.5", ratio(0.5), "{:.2f}", "2.46")
quote("ratio beta=0.25", ratio(0.25), "{:.1f}", "18.2")
quote("beta at which A overtakes B", 1.0 / math.log(3.0), "{:.2f}", "0.91")
# beta -> 0+: mass on the max-reward responses in proportion to pi_ref (Q2)
top = [0.2, 0.1, 0.05]
quote("Q2 limit share 1", top[0] / sum(top), "{:.3f}", "0.571")
quote("Q2 limit share 2", top[1] / sum(top), "{:.3f}", "0.286")
quote("Q2 limit share 3", top[2] / sum(top), "{:.3f}", "0.143")
lim = gibbs([0.2, 0.1, 0.05, 0.4, 0.25], [1, 1, 1, 0.5, 0], 1e-3)
assert max(abs(a - b) for a, b in zip(lim[:3], [v / sum(top) for v in top])) < 1e-9
quote("Q2 large beta", gibbs([0.2, 0.1, 0.05, 0.4, 0.25], [1, 1, 1, 0.5, 0], 1e3)[0], "{:.3f}", "0.200")

# --- 5. Binary reward: the odds tilt ----------------------------------------------------------------
# derivation check on random problems: p* formula, odds factor, q(tau|x) and N_eff unchanged, zeros stay zero
for trial in range(300):
    K = rng.randint(3, 12)
    ref = [rng.random() if rng.random() > 0.2 else 0.0 for _ in range(K)]
    ref[0] = max(ref[0], 0.01)
    s = sum(ref)
    ref = [v / s for v in ref]
    R = [1 if rng.random() < 0.4 else 0 for _ in range(K)]
    R[0], R[-1] = 1, 0
    ref[-1] = max(ref[-1], 0.0)
    beta = math.exp(rng.uniform(math.log(0.1), math.log(10)))
    ps = gibbs(ref, R, beta)
    p = sum(a for a, x in zip(ref, R) if x)
    pstar = sum(a for a, x in zip(ps, R) if x)
    assert abs(pstar - p_star(p, beta)) < 1e-12
    if p < 1:
        assert abs((pstar / (1 - pstar)) / (p / (1 - p)) - math.exp(1 / beta)) < 1e-9 * math.exp(1 / beta)
    q_ref = [a / p for a, x in zip(ref, R) if x]
    q_new = [a / pstar for a, x in zip(ps, R) if x]
    assert max(abs(a - b) for a, b in zip(q_ref, q_new)) < 1e-12
    assert abs(n_eff([a for a, x in zip(ref, R) if x]) - n_eff([a for a, x in zip(ps, R) if x])) < 1e-9
    assert all(b == 0.0 for a, b in zip(ref, ps) if a == 0.0)
print("Binary-reward tilt: p* formula, odds x e^(1/beta), q(tau|x) and N_eff unchanged, zeros stay zero "
      "(300 random problems)")

quote("e^(1/beta) at beta=1", math.e, "{:.2f}", "2.72")
quote("p*: p=0.30, beta=2", p_star(0.30, 2.0), "{:.3f}", "0.414")
quote("p*: p=0.30, beta=1", p_star(0.30, 1.0), "{:.3f}", "0.538")
quote("p*: p=0.30, beta=0.5", p_star(0.30, 0.5), "{:.3f}", "0.760")
quote("p*: p=0.30, beta=0.25", p_star(0.30, 0.25), "{:.3f}", "0.959")
quote("p*: p=0.01, beta=1", p_star(0.01, 1.0), "{:.3f}", "0.027")
quote("p*: p=0.01, beta=0.5", p_star(0.01, 0.5), "{:.3f}", "0.069")
quote("p*: p=0.01, beta=0.25", p_star(0.01, 0.25), "{:.3f}", "0.355")
quote("p*: p=0.001, beta=0.25", p_star(0.001, 0.25), "{:.3f}", "0.052")
quote("odds before p=0.30", 0.30 / 0.70, "{:.2f}", "0.43")
quote("odds after beta=1", p_star(0.30, 1.0) / (1 - p_star(0.30, 1.0)), "{:.2f}", "1.16")
quote("pass@64 at p=0.01", 1 - 0.99**64, "{:.0%}", "47%")
quote("pass@64 at p*=0.027", 1 - (1 - p_star(0.01, 1.0)) ** 64, "{:.0%}", "82%")
# the toy distribution behind the animation's first panel (and the post's example)
REF8 = [0.15, 0.10, 0.05, 0.0, 0.30, 0.20, 0.12, 0.08]   # A, B, C correct; D correct but pi_ref = 0; E-H wrong
R8 = [1, 1, 1, 1, 0, 0, 0, 0]
assert abs(sum(REF8) - 1) < 1e-12
quote("toy N_eff", n_eff(REF8[:4]), "{:.2f}", "2.75")
for inv_beta, lab in [(0.5, "2"), (1, "1"), (2, "0.5"), (4, "0.25")]:
    ps = gibbs(REF8, R8, 1 / inv_beta)
    anim(f"tilt p* beta={lab}", sum(ps[:4]), "{:.3f}", f"{p_star(0.30, 1 / inv_beta):.3f}")
    anim(f"tilt N_eff beta={lab}", n_eff(ps[:4]), "{:.2f}", "2.75")
    anim(f"tilt D beta={lab}", ps[3], "{:.2f}", "0.00")
    for j, name in enumerate("ABCEFGH"):
        k = j if j < 3 else j + 1
        anim(f"tilt {name} beta={lab}", ps[k], "{:.2f}", f"{REF8[k] * math.exp(R8[k] * inv_beta) / (1 - 0.30 + 0.30 * math.exp(inv_beta)):.2f}")
    for j, (name, txt) in enumerate(zip("ABC", ["0.50", "0.33", "0.17"])):
        anim(f"tilt q({name}) beta={lab}", ps[j] / sum(ps[:4]), "{:.2f}", txt)
ps4 = gibbs(REF8, R8, 0.25)
quote("toy A at beta=0.25", ps4[0], "{:.2f}", "0.48")
quote("toy B at beta=0.25", ps4[1], "{:.2f}", "0.32")
quote("toy C at beta=0.25", ps4[2], "{:.2f}", "0.16")

# --- 6. From the optimum to DPO: the worked example -----------------------------------------------
odds_ref, odds_new = 0.01 / 0.20, 0.02 / 0.10
quote("loser five times the winner", 0.10 / 0.02, "{:.0f} times", "5 times")
quote("odds ref", odds_ref, "{:.2f}", "0.05")
quote("odds new", odds_new, "{:.2f}", "0.20")
quote("odds factor", odds_new / odds_ref, "{:.0f}", "4")
D4 = math.log(0.02 / 0.01) - math.log(0.10 / 0.20)
quote("Delta = log 4", D4, "{:.3f}", "1.386")
assert abs(D4 - math.log(4)) < 1e-12
L = lambda d, beta: -math.log(sigmoid(beta * d))
quote("loss at Delta=0", L(0.0, 1.0), "{:.3f}", "0.693")
quote("loss beta=1 Delta=log4", L(D4, 1.0), "{:.3f}", "0.223")
quote("BT prob beta=1 Delta=log4", sigmoid(D4), "{:.2f}", "0.80")
quote("implied margin beta=0.1", 0.1 * D4, "{:.3f}", "0.139")
quote("loss beta=0.1 Delta=log4", L(D4, 0.1), "{:.3f}", "0.626")

# --- 7. Likelihood displacement: the arithmetic -----------------------------------------------------
Dd = math.log(0.008 / 0.01) - math.log(0.04 / 0.20)
assert abs(Dd - D4) < 1e-12
quote("displaced ratio w", 0.008 / 0.01, "{:.1f}", "0.8")
quote("displaced ratio l", 0.04 / 0.20, "{:.1f}", "0.2")
quote("pair mass before", 0.01 + 0.20, "{:.2f}", "0.21")
quote("pair mass, first example", 0.02 + 0.10, "{:.2f}", "0.12")
quote("pair mass, displaced example", 0.008 + 0.04, "{:.3f}", "0.048")
quote("mass leaving pair, displaced", 0.21 - 0.048, "{:.3f}", "0.162")
quote("mass leaving pair, first example", 0.21 - 0.12, "{:.2f}", "0.09")

# free logits: a DPO step on one pair raises z_w and lowers z_l by the same amount -> pi(y_w) rises
for trial in range(500):
    K = rng.randint(2, 10)
    z = [rng.gauss(0, 2) for _ in range(K)]
    w, l = rng.sample(range(K), 2)
    c = rng.uniform(1e-3, 5)
    z2 = list(z)
    z2[w] += c
    z2[l] -= c
    assert softmax(z2)[w] > softmax(z)[w]
print("Free logits: one DPO step on one pair always raises pi(y_w) (500 random cases)")

# shared hidden state (the toy behind the animation's second panel): logits z_j = b_j + u_j . h, h trained,
# b fixed so that pi_ref = softmax(b). The DPO gradient in h is proportional to u_w - u_l (the softmax
# normalizer cancels), so every gradient step moves h along the same line h = s (u_w - u_l)/|u_w - u_l|.
U = {"w": (1.0, 0.3), "l": (1.0, -0.3), "o": (0.0, 1.0)}   # "No" (preferred), "Never" (dispreferred), "Yes"
REF3 = [0.40, 0.30, 0.30]
B3 = [math.log(v) for v in REF3]


def toy(s: float) -> list[float]:
    """Probabilities after moving h a distance s along (u_w - u_l)."""
    return softmax([B3[0] + U["w"][1] * s, B3[1] + U["l"][1] * s, B3[2] + U["o"][1] * s])


def toy_delta(s: float) -> float:
    p = toy(s)
    return math.log(p[0] / REF3[0]) - math.log(p[1] / REF3[1])


# gradient direction check: dL/dh by finite differences is parallel to u_w - u_l
h0 = (0.0, 0.7)
f = lambda h: L(math.log(softmax([B3[j] + U[k][0] * h[0] + U[k][1] * h[1] for j, k in enumerate("wlo")])[0] / REF3[0])
                - math.log(softmax([B3[j] + U[k][0] * h[0] + U[k][1] * h[1] for j, k in enumerate("wlo")])[1] / REF3[1]), 1.0)
eps = 1e-6
g = [(f((h0[0] + eps, h0[1])) - f((h0[0] - eps, h0[1]))) / (2 * eps),
     (f((h0[0], h0[1] + eps)) - f((h0[0], h0[1] - eps))) / (2 * eps)]
assert abs(g[0]) < 1e-8 and g[1] < 0, g
S_END = math.log(4) / 0.6           # Delta(s) = 0.6 s, so Delta = log 4 at s = log(4)/0.6
assert abs(toy_delta(S_END) - math.log(4)) < 1e-12
ss = [S_END * i / 400 for i in range(401)]
pw = [toy(s)[0] for s in ss]
pl = [toy(s)[1] for s in ss]
assert all(b < a for a, b in zip(pw, pw[1:])) and all(b < a for a, b in zip(pl, pl[1:])), "not both falling"
pe = toy(S_END)
quote("toy ref w", REF3[0], "{:.2f}", "0.40")
quote("toy ref l", REF3[1], "{:.2f}", "0.30")
quote("toy ref other", REF3[2], "{:.2f}", "0.30")
quote("toy end w", pe[0], "{:.2f}", "0.20")
quote("toy end l", pe[1], "{:.3f}", "0.038")
quote("toy end other", pe[2], "{:.2f}", "0.76")
quote("toy pair mass end", pe[0] + pe[1], "{:.2f}", "0.24")
quote("toy pair mass start", REF3[0] + REF3[1], "{:.2f}", "0.70")
quote("toy initial slope of log pi_w", 0.3 - (REF3[0] * 0.3 - REF3[1] * 0.3 + REF3[2] * 1.0), "signed2", "\u22120.03")
for frac, lab in [(0.0, "0"), (0.5, "half"), (1.0, "end")]:
    p = toy(S_END * frac)
    anim(f"disp w {lab}", p[0], "{:.2f}", f"{p[0]:.2f}")
    anim(f"disp l {lab}", p[1], "{:.3f}", f"{p[1]:.3f}")
    anim(f"disp o {lab}", p[2], "{:.2f}", f"{p[2]:.2f}")
    anim(f"disp Delta {lab}", toy_delta(S_END * frac), "{:.2f}", f"{math.log(4) * frac:.2f}")
    anim(f"disp loss {lab}", L(toy_delta(S_END * frac), 1.0), "{:.3f}", f"{L(math.log(4) * frac, 1.0):.3f}")
anim("disp logpw end", math.log(pe[0]), "signed2", "\u22121.60")
anim("disp logpl end", math.log(pe[1]), "signed2", "\u22123.28")
anim("disp logpw start", math.log(0.40), "signed2", "\u22120.92")
anim("disp logpl start", math.log(0.30), "signed2", "\u22121.20")

# --- 8. The gradient --------------------------------------------------------------------------------
worst = 0.0
for trial in range(300):
    beta = math.exp(rng.uniform(math.log(0.05), math.log(5)))
    d = rng.gauss(0, 4)
    fd = (L(d + 1e-6, beta) - L(d - 1e-6, beta)) / 2e-6
    worst = max(worst, abs(fd - (-beta * sigmoid(-beta * d))))
assert worst < 1e-6
print(f"dL/dDelta = -beta sigma(-beta Delta) (max FD err {worst:.1e})")
wgt = lambda d, beta: beta * sigmoid(-beta * d)
quote("weight beta=1 Delta=-log4", wgt(-D4, 1.0), "{:.1f}", "0.8")
quote("weight beta=1 Delta=0", wgt(0.0, 1.0), "{:.1f}", "0.5")
quote("weight beta=1 Delta=log4", wgt(D4, 1.0), "{:.1f}", "0.2")
quote("weight beta=0.1 Delta=-log4", wgt(-D4, 0.1), "{:.3f}", "0.053")
quote("weight beta=0.1 Delta=0", wgt(0.0, 0.1), "{:.3f}", "0.050")
quote("weight beta=0.1 Delta=log4", wgt(D4, 0.1), "{:.3f}", "0.047")
quote("Delta for weight 0.01 at beta=0.1 (beta Delta)", math.log(0.1 / 0.01 - 1), "{:.1f}", "2.2")
quote("Delta for weight 0.01 at beta=0.1", math.log(0.1 / 0.01 - 1) / 0.1, "{:.0f}", "22")

# full parameter gradient on a small autoregressive model: theta[prefix] = logits over the vocabulary
VOC, T = 3, 4


def seq_logprob(th: dict, seq: tuple) -> float:
    return sum(math.log(softmax(th[seq[:t]])[seq[t]]) for t in range(len(seq)))


def dpo_loss(th: dict, ref: dict, yw: tuple, yl: tuple, beta: float) -> float:
    d = (seq_logprob(th, yw) - seq_logprob(ref, yw)) - (seq_logprob(th, yl) - seq_logprob(ref, yl))
    return L(d, beta)


def grad_logprob(th: dict, seq: tuple) -> dict:
    g = {k: [0.0] * VOC for k in th}
    for t in range(len(seq)):
        p = softmax(th[seq[:t]])
        for j in range(VOC):
            g[seq[:t]][j] += (1.0 if j == seq[t] else 0.0) - p[j]
    return g


prefixes = [()]
for t in range(1, T):
    prefixes += [p + (v,) for p in prefixes if len(p) == t - 1 for v in range(VOC)]
worst = 0.0
worst_cancel = 0.0
for trial in range(30):
    th = {p: [rng.gauss(0, 1) for _ in range(VOC)] for p in prefixes}
    ref = {p: [rng.gauss(0, 1) for _ in range(VOC)] for p in prefixes}
    beta = rng.choice([0.1, 0.5, 1.0, 2.0])
    k = rng.randint(0, T - 1)                       # shared prefix length
    pre = tuple(rng.randrange(VOC) for _ in range(k))
    a = rng.randrange(VOC)
    b = (a + 1 + rng.randrange(VOC - 1)) % VOC
    yw = pre + (a,) + tuple(rng.randrange(VOC) for _ in range(T - k - 1))
    yl = pre + (b,) + tuple(rng.randrange(VOC) for _ in range(T - k - 1))
    d = (seq_logprob(th, yw) - seq_logprob(ref, yw)) - (seq_logprob(th, yl) - seq_logprob(ref, yl))
    gw, gl = grad_logprob(th, yw), grad_logprob(th, yl)
    coef = beta * sigmoid(-beta * d)
    for p in prefixes:
        for j in range(VOC):
            th[p][j] += 1e-6
            up = dpo_loss(th, ref, yw, yl, beta)
            th[p][j] -= 2e-6
            dn = dpo_loss(th, ref, yw, yl, beta)
            th[p][j] += 1e-6
            fd = (up - dn) / 2e-6
            formula = -coef * (gw[p][j] - gl[p][j])
            worst = max(worst, abs(fd - formula))
            # the shared prefix contributes nothing: its parameters get zero gradient from the pair
            if len(p) < k and p == pre[: len(p)]:
                worst_cancel = max(worst_cancel, abs(formula))
    # and Delta itself does not depend on the shared-prefix tokens
    dprefix = sum(math.log(softmax(th[pre[:t]])[pre[t]]) - math.log(softmax(ref[pre[:t]])[pre[t]]) for t in range(k))
    d_suffix = d  # Delta computed on full sequences
    d_wo = ((seq_logprob(th, yw) - seq_logprob(ref, yw)) - dprefix) - ((seq_logprob(th, yl) - seq_logprob(ref, yl)) - dprefix)
    worst_cancel = max(worst_cancel, abs(d_suffix - d_wo))
assert worst < 1e-6 and worst_cancel < 1e-12
print(f"DPO parameter gradient matches finite differences (max err {worst:.1e}); "
      f"shared-prefix tokens cancel in Delta and get zero gradient")

# --- 9. Length: log-ratios are sums over tokens ------------------------------------------------------
quote("100 tokens x 0.01 nats", 100 * 0.01, "{:.0f}", "1")
quote("1,000 tokens x 0.01 nats", 1000 * 0.01, "{:.0f}", "10")
quote("BT prob, margin 1 at beta=0.1", sigmoid(0.1 * 1), "{:.3f}", "0.525")
quote("BT prob, margin 10 at beta=0.1", sigmoid(0.1 * 10), "{:.3f}", "0.731")
quote("Q4 shared prefix", 300, "{:.0f}", "300")
quote("Q4 divergent tokens", 500 - 300, "{:.0f}", "200")

# --- 10. Check-yourself numbers -----------------------------------------------------------------------
quote("Q3 pass@64 at p=0.07", 1 - 0.93**64, "{:.2f}", "0.99")
p64_03 = 1 - 0.7**64
quote("Q3 pass@64 at p=0.30", p64_03, "pct", ">99.9%")
quote("Q3 odds factor 0.3->0.5", (0.5 / 0.5) / (0.3 / 0.7), "{:.2f}", "2.33")
quote("Q3 beta for that factor", 1 / math.log(7 / 3), "{:.2f}", "1.18")
quote("Q3 rare problem under that tilt", 0.01 * (7 / 3) / (0.01 * (7 / 3) + 0.99), "{:.3f}", "0.023")

# --- 11. In my experiments: design statements re-read from sharpen-or-expand main ---------------------


def fetch(path: str) -> str:
    with urllib.request.urlopen(SOE_RAW + path, timeout=60) as fh:
        return fh.read().decode()


d031 = fetch("docs/decisions/031-rl-and-dpo-start-from-the-base.md")
assert "and DPO starts from `Qwen/Qwen3.5-4B-Base`" in d031
assert "**Preference pairs** (DPO, M3/M4) are sampled from the base policy, at 1.0 / 1.0" in d031
assert "Pair construction (truncated\n  losers) is still M4's decision." in d031
d017 = fetch("docs/decisions/017-rl-rollout-sampling-1.0.md")
assert "Preference pairs therefore use 1.0 / 1.0, top-k off, the 16,384-token cap" in d017
assert "whether truncated responses can be losers) is decided with the DPO pipeline in M4" in d017
assert "Online-DPO-R1 (Zhang et al. 2025, RLHFlow)" in d017
assert "Iterative RPO (Pang et al. 2024) samples at 0.8, then 1.3" in d017
mp = fetch("docs/milestone-plan.md")
assert "preference generation from the base policy (decision 031; 15K–30K prompts × 4–8 samples, verifier-labelled pairs)" in mp
assert "DPO trained from the frozen starting checkpoint (the base, decision 031) and evaluated on the full suite" in mp
# the only beta sweep in the plan is RS-GRPO's risk parameter, not DPO's beta
assert "RS-GRPO β ∈ {1, 2, 4} on 2B" in mp
assert "DPO β" not in mp and "beta for DPO" not in mp.lower()
d013 = fetch("docs/decisions/013-rl-no-kl-step-checkpoints.md")
assert "**No KL term and no reference model for any RL algorithm** (GRPO, Dr.GRPO, DAPO, RS-GRPO)" in d013
quote("token cap", 16384, "{:,}", "16,384")
print("Experiment design statements found in decisions 031, 017, 013 and the milestone plan at", SOE_COMMIT[:7])

# --- Compare with the text ---------------------------------------------------------------------------


def pct(x: float) -> str:
    return ">99.9%" if x >= 0.9995 else f"{x:.1%}"


post = POST.read_text() if POST.exists() else None
bad = []
for label, value, fmt, text in QUOTES + ANIM:
    if fmt.startswith("signed"):
        got = signed(value, int(fmt[-1]))
    elif fmt == "pct":
        got = pct(value)
    else:
        got = fmt.format(value)
    status = "ok" if got == text else "MISMATCH"
    if got != text:
        bad.append(label)
    if post is not None and not label.startswith("anim ") and text not in post:
        status += " (not in post)"
        bad.append(label + " [not in post]")
    print(f"{status:<18} {label:<52} computed {got:<10} text {text}")

if bad:
    print("\nFAILED:", *bad, sep="\n  ")
    sys.exit(1)
print(f"\nall {len(QUOTES)} quoted values and {len(ANIM)} animation values match"
      + ("" if post is not None else " (post not written yet)"))
