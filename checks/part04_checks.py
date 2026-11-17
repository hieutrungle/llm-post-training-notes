"""Part 4 checks: recompute every number the post quotes.

Run:  python checks/part04_checks.py

Each entry in QUOTES pairs a computed value with the exact string the post prints. The script fails
if the computed value, formatted as the post formats it, differs from that string, or (once the post
exists) if the string does not appear in posts/part-04-policy-gradients-reinforce-to-ppo.md. Entries in
ANIM are numbers the opening animation prints (blog assets/js/anim/part-04-pg-loop.js); they are
recomputed but not required in the post. The animation runs the same seeded simulation as section 0.

It also checks numerically that
  * the policy gradient grad J = E[R grad log pi(tau)] = E[R sum_t grad log pi(y_t | x, y_<t)] holds on a
    small autoregressive model, by exact enumeration and against central finite differences;
  * with a 0/1 reward grad J = p(x) E_q[grad log pi(tau)], q the distribution of successful trajectories;
  * with free logits per strategy, dJ/dz_k = pi_k (R_k - J);
  * E[b grad log pi] = 0 for a constant baseline; with a group of G samples, the batch mean that includes
    the sample gives (G-1)/G times the gradient and the leave-one-out mean gives it exactly;
  * the gradient noise of a group estimate as a function of a constant baseline (the figure), and the
    variance-minimizing constant baseline b* = E[R |s|^2] / E[|s|^2];
  * PPO's clipped objective and its derivative in all four cases (sign of A, ratio above or below the
    clip range), that at ratio 1 its gradient equals the policy gradient, and that a ratio already
    clipped can still be pushed further by an update on another token through shared parameters;
  * the k3 KL estimator is unbiased and non-negative;
  * the variance, sequence-ratio, entropy and sparse-reward numbers.
Experiment design statements are re-read from sharpen-or-expand's public main branch, pinned to a commit.
"""

from __future__ import annotations

import json
import math
import random
import sys
import urllib.request
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POST = ROOT / "posts" / "part-04-policy-gradients-reinforce-to-ppo.md"

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


def softmax(z: list[float]) -> list[float]:
    m = max(z)
    e = [math.exp(v - m) if v > -math.inf else 0.0 for v in z]
    s = sum(e)
    return [v / s for v in e]


def entropy(p: list[float]) -> float:
    return -sum(v * math.log(v) for v in p if v > 0)


def n_eff(masses: list[float]) -> float:
    """Hill number of order 1 of the distribution over correct strategies (Part 1)."""
    tot = sum(masses)
    return math.exp(entropy([m / tot for m in masses if m > 0]))


def binom_pmf(n: int, k: int, p: float) -> float:
    return math.comb(n, k) * p**k * (1 - p) ** (n - k)


# --- 0. The opening picture (blog assets/js/anim/part-04-pg-loop.js) ------------------------------
# Two problems, one logit per solution strategy (free parameters), reward 1 for a correct strategy.
# Each round draws G = 8 samples from the current policy with a xorshift32 generator (the same as
# AL.rng in the blog's lib.js; easy problem first, then hard), scores them, and takes one REINFORCE
# step on the logits with baseline b = p (the current success probability) and step size 3:
#     z_k += 3 * (1/G) sum_i (R_i - p) (1[k = s_i] - pi_k).
# A strategy with probability 0 has logit -inf and stays there. All numbers are illustrative.
ANIM_SEED, ETA, G, ROUNDS = 21726, 3.0, 8, 8
EASY_P = [0.30, 0.02, 0.0, 0.33, 0.20, 0.15]   # A common correct, B rare correct, C correct (p = 0), D-F wrong
EASY_R = [1, 1, 1, 0, 0, 0]
HARD_P = [0.02, 0.0, 0.50, 0.30, 0.18]          # rare correct, correct (p = 0), three wrong
HARD_R = [1, 1, 0, 0, 0]


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


def reinforce_step(z: list[float], rew: list[int], samples: list[int]) -> list[float]:
    pi = softmax(z)
    p = sum(pi[k] for k in range(len(z)) if rew[k])
    g = [0.0] * len(z)
    for s in samples:
        a = rew[s] - p
        for k in range(len(z)):
            g[k] += a * ((1.0 if k == s else 0.0) - pi[k]) / G
    return [z[k] + ETA * g[k] if z[k] > -math.inf else z[k] for k in range(len(z))]


def simulate():
    rng = xorshift32(ANIM_SEED)
    ze = [math.log(v) if v > 0 else -math.inf for v in EASY_P]
    zh = [math.log(v) if v > 0 else -math.inf for v in HARD_P]
    hist = []
    for _ in range(ROUNDS):
        pe, ph = softmax(ze), softmax(zh)
        se = [draw(pe, next(rng)) for _ in range(G)]
        sh = [draw(ph, next(rng)) for _ in range(G)]
        hist.append((pe, ph, se, sh))
        ze, zh = reinforce_step(ze, EASY_R, se), reinforce_step(zh, HARD_R, sh)
    hist.append((softmax(ze), softmax(zh), None, None))
    return hist


SIM = simulate()
succ = lambda pi, rew: sum(v for v, r in zip(pi, rew) if r)
for r, (pe, ph, se, sh) in enumerate(SIM):
    anim(f"round {r} easy p", succ(pe, EASY_R), "{:.3f}", f"{succ(pe, EASY_R):.3f}")
    anim(f"round {r} easy A", pe[0], "{:.3f}", f"{pe[0]:.3f}")
    anim(f"round {r} easy B", pe[1], "{:.3f}", f"{pe[1]:.3f}")
    anim(f"round {r} hard p", succ(ph, HARD_R), "{:.3f}", f"{succ(ph, HARD_R):.3f}")
    assert pe[2] == 0.0 and ph[1] == 0.0, "a zero-probability strategy moved"
    if se is not None:
        anim(f"round {r} easy correct samples", sum(EASY_R[s] for s in se), "{:.0f}", str(sum(EASY_R[s] for s in se)))
        anim(f"round {r} hard correct samples", sum(HARD_R[s] for s in sh), "{:.0f}", str(sum(HARD_R[s] for s in sh)))
        for nm, pi, rew in (("easy", pe, EASY_R), ("hard", ph, HARD_R)):
            p = succ(pi, rew)
            anim(f"round {r} {nm} advantage of a success", 1 - p, "{:+.2f}", f"+{1 - p:.2f}")
            anim(f"round {r} {nm} advantage of a failure", -p, "signed2", f"\u2212{p:.2f}")
        # every sampled success is pushed up, every sampled failure down (its own logit, before normalization)
        for zrew, pi, samples in ((EASY_R, pe, se), (HARD_R, ph, sh)):
            p = succ(pi, zrew)
            for s in set(samples):
                assert (zrew[s] - p) > 0 if zrew[s] else (zrew[s] - p) < 0
# the printed sample sequences, so the JavaScript port can be compared draw by draw
SAMPLES_EASY = ["".join("ABCDEF"[s] for s in h[2]) for h in SIM[:-1]]
SAMPLES_HARD = ["".join("BZDEF"[s] for s in h[3]) for h in SIM[:-1]]
print("Opening picture samples (easy):", " ".join(SAMPLES_EASY))
print("Opening picture samples (hard):", " ".join(SAMPLES_HARD))

easy_b_rounds = [r + 1 for r, h in enumerate(SIM[:-1]) if 1 in h[2]]
hard_first = min(r + 1 for r, h in enumerate(SIM[:-1]) if 0 in h[3])
assert easy_b_rounds == [1] and SIM[0][2].count(1) == 1
assert hard_first == 4 and SIM[3][3].count(0) == 1
pe0, pe1, peN = SIM[0][0], SIM[1][0], SIM[-1][0]
ph0, phN = SIM[0][1], SIM[-1][1]
quote("pic rounds", ROUNDS, "{:.0f} rounds", "8 rounds")
quote("pic group", G, "{:.0f} samples", "8 samples")
quote("pic step size", ETA, "step size {:.0f}", "step size 3")
quote("pic easy A start", pe0[0], "{:.2f}", "0.30")
quote("pic easy A end", peN[0], "{:.3f}", "0.941")
quote("pic easy p start", succ(pe0, EASY_R), "{:.2f}", "0.32")
quote("pic easy p end", succ(peN, EASY_R), "{:.3f}", "0.947")
quote("pic easy B after round 1", pe1[1], "{:.3f}", "0.024")
quote("pic easy B end", peN[1], "{:.3f}", "0.005")
quote("pic easy B share start", pe0[1] / succ(pe0, EASY_R), "{:.2%}", "6.25%")
quote("pic easy B share end", peN[1] / succ(peN, EASY_R), "{:.1%}", "0.6%")
quote("pic easy N_eff start", n_eff(pe0[:3]), "{:.2f}", "1.26")
quote("pic easy N_eff end", n_eff(peN[:3]), "{:.2f}", "1.04")
quote("pic hard B start", ph0[0], "{:.2f}", "0.02")
quote("pic hard B after rounds 1-3", SIM[3][1][0], "{:.3f}", "0.020")
quote("pic hard advantage of failures", -ph0[0], "signed2", "\u22120.02")
quote("pic hard advantage of the first success", 1 - SIM[3][1][0], "{:.2f}", "0.98")
quote("pic hard B after round 4", SIM[4][1][0], "{:.3f}", "0.033")
quote("pic hard B end", phN[0], "{:.3f}", "0.593")
quote("pic P(rare sampled in a round)", 1 - 0.98**8, "{:.2f}", "0.15")
for r in range(ROUNDS + 1):
    assert abs(SIM[r][1][0] - succ(SIM[r][1], HARD_R)) < 1e-15
print(f"Opening picture: easy A {pe0[0]:.2f} -> {peN[0]:.3f}, B {pe0[1]:.3f} -> {pe1[1]:.3f} -> {peN[1]:.4f}; "
      f"hard rare {ph0[0]:.3f} -> {SIM[4][1][0]:.4f} (round 4) -> {phN[0]:.3f}")

# entropy of the strategy distributions (entropy section)
quote("pic easy entropy start", entropy(pe0), "{:.2f}", "1.41")
quote("pic easy entropy end", entropy(peN), "{:.2f}", "0.30")
hard_H = [entropy(h[1]) for h in SIM]
quote("pic hard entropy start", hard_H[0], "{:.2f}", "1.09")
quote("pic hard entropy peak", max(hard_H), "{:.2f}", "1.36")
assert hard_H.index(max(hard_H)) == 6, "peak after round 6"
quote("pic hard entropy peak round", hard_H.index(max(hard_H)), "round {:.0f}", "round 6")
quote("pic hard entropy end", hard_H[-1], "{:.2f}", "1.11")

# --- 1. The policy gradient on a small autoregressive model ----------------------------------------
VOC, T = 3, 3
rng = random.Random(4)
prefixes = [()]
for t in range(1, T):
    prefixes += [p + (v,) for p in prefixes if len(p) == t - 1 for v in range(VOC)]
SEQS = list(product(range(VOC), repeat=T))


def seq_prob(th: dict, seq: tuple) -> float:
    return math.exp(sum(math.log(softmax(th[seq[:t]])[seq[t]]) for t in range(T)))


def grad_logprob(th: dict, seq: tuple) -> dict:
    """grad log pi(tau) = sum over tokens of grad log pi(y_t | x, y_<t)."""
    g = {k: [0.0] * VOC for k in th}
    for t in range(T):
        p = softmax(th[seq[:t]])
        for j in range(VOC):
            g[seq[:t]][j] += (1.0 if j == seq[t] else 0.0) - p[j]
    return g


def J(th: dict, R: dict) -> float:
    return sum(seq_prob(th, s) * R[s] for s in SEQS)


worst_pg = worst_q = 0.0
for trial in range(20):
    th = {p: [rng.gauss(0, 1) for _ in range(VOC)] for p in prefixes}
    R = {s: float(rng.random() < 0.3) for s in SEQS}
    R[SEQS[rng.randrange(len(SEQS))]] = 1.0
    # exact E[R grad log pi] by enumeration
    pg = {k: [0.0] * VOC for k in prefixes}
    eq = {k: [0.0] * VOC for k in prefixes}
    p_succ = J(th, R)
    for s in SEQS:
        ps, gs = seq_prob(th, s), grad_logprob(th, s)
        for k in prefixes:
            for j in range(VOC):
                pg[k][j] += ps * R[s] * gs[k][j]
                if R[s]:
                    eq[k][j] += ps / p_succ * gs[k][j]    # E_q[grad log pi], q = pi restricted to successes
    for k in prefixes:
        for j in range(VOC):
            th[k][j] += 1e-6
            up = J(th, R)
            th[k][j] -= 2e-6
            dn = J(th, R)
            th[k][j] += 1e-6
            worst_pg = max(worst_pg, abs((up - dn) / 2e-6 - pg[k][j]))
            worst_q = max(worst_q, abs(pg[k][j] - p_succ * eq[k][j]))
assert worst_pg < 1e-8 and worst_q < 1e-12
quote("sequences of the small model", len(SEQS), "{:.0f} sequences", "27 sequences")
print(f"Policy gradient = finite differences on a {VOC}-token, length-{T} model "
      f"(max err {worst_pg:.1e}); grad J = p E_q[grad log pi] (max err {worst_q:.1e})")

# free logits per strategy: dJ/dz_k = pi_k (R_k - J)
worst = 0.0
for trial in range(200):
    K = rng.randint(2, 9)
    z = [rng.gauss(0, 2) for _ in range(K)]
    Rk = [float(rng.random() < 0.4) for _ in range(K)]
    pi = softmax(z)
    Jz = sum(a * b for a, b in zip(pi, Rk))
    for k in range(K):
        z2 = list(z); z2[k] += 1e-6
        z3 = list(z); z3[k] -= 1e-6
        fd = (sum(a * b for a, b in zip(softmax(z2), Rk)) - sum(a * b for a, b in zip(softmax(z3), Rk))) / 2e-6
        worst = max(worst, abs(fd - pi[k] * (Rk[k] - Jz)))
assert worst < 1e-8
print(f"Free logits: dJ/dz_k = pi_k (R_k - J) (max err {worst:.1e})")
quote("expected push A vs B", (0.30 * (1 - 0.32)) / (0.02 * (1 - 0.32)), "{:.0f} times", "15 times")

# --- 2. Variance: p = 0.01, 100 samples ---------------------------------------------------------------
quote("P(0 successes)", binom_pmf(100, 0, 0.01), "{:.3f}", "0.366")
quote("P(1 success)", binom_pmf(100, 1, 0.01), "{:.3f}", "0.370")
quote("P(2 successes)", binom_pmf(100, 2, 0.01), "{:.3f}", "0.185")
quote("P(>=3 successes)", 1 - sum(binom_pmf(100, k, 0.01) for k in range(3)), "{:.3f}", "0.079")
quote("share zero", binom_pmf(100, 0, 0.01), "zero in {:.0%}", "zero in 37%")
quote("share exact", binom_pmf(100, 1, 0.01), "another {:.0%}", "another 37%")
quote("share double", binom_pmf(100, 2, 0.01), "large in {:.0%}", "large in 18%")
quote("share >= triple", 1 - sum(binom_pmf(100, k, 0.01) for k in range(3)), "large in {:.0%}.", "large in 8%.")
# lumped two-outcome model: logit theta, p = sigma(theta); the batch estimate of dJ/dtheta = p(1-p) is
# (k/N)(1-p) for k successes, i.e. k/(N p) times the true value -- k times at N = 100, p = 0.01.
for k in range(4):
    est = (k / 100) * (1 - 0.01)
    assert abs(est / (0.01 * 0.99) - k) < 1e-12

# --- 3. Baselines ------------------------------------------------------------------------------------
worst = 0.0
for trial in range(200):
    K = rng.randint(2, 9)
    pi = softmax([rng.gauss(0, 2) for _ in range(K)])
    b = rng.gauss(0, 3)
    # E_a[b grad_z log pi(a)] = b sum_a pi_a (e_a - pi) = 0
    for j in range(K):
        worst = max(worst, abs(sum(pi[a] * b * ((1.0 if a == j else 0.0) - pi[j]) for a in range(K))))
assert worst < 1e-12
print(f"E[b grad log pi] = 0 for a constant baseline (max |.| {worst:.1e})")

# groups of G samples, enumerated exactly: batch mean (with self) vs leave-one-out
Gb = 4
worst_mean = worst_loo = 0.0
for trial in range(30):
    K = rng.randint(2, 4)
    pi = softmax([rng.gauss(0, 1.5) for _ in range(K)])
    Rk = [float(rng.random() < 0.5) for _ in range(K)]
    Rk[0], Rk[-1] = 1.0, 0.0
    true = [pi[j] * (Rk[j] - sum(a * b for a, b in zip(pi, Rk))) for j in range(K)]
    e_mean = [0.0] * K
    e_loo = [0.0] * K
    for grp in product(range(K), repeat=Gb):
        w = math.prod(pi[s] for s in grp)
        rs = [Rk[s] for s in grp]
        for i, s in enumerate(grp):
            a_mean = rs[i] - sum(rs) / Gb
            a_loo = rs[i] - (sum(rs) - rs[i]) / (Gb - 1)
            assert abs(a_loo - Gb / (Gb - 1) * a_mean) < 1e-12
            for j in range(K):
                sc = (1.0 if j == s else 0.0) - pi[j]
                e_mean[j] += w * a_mean * sc / Gb
                e_loo[j] += w * a_loo * sc / Gb
    worst_mean = max(worst_mean, max(abs(a - (Gb - 1) / Gb * b) for a, b in zip(e_mean, true)))
    worst_loo = max(worst_loo, max(abs(a - b) for a, b in zip(e_loo, true)))
assert worst_mean < 1e-12 and worst_loo < 1e-12
print(f"Groups of {Gb}, exact: batch-mean baseline gives (G-1)/G x gradient (err {worst_mean:.1e}); "
      f"leave-one-out is unbiased (err {worst_loo:.1e})")
quote("enumerated group size", Gb, "every group of {} samples", "every group of 4 samples")
quote("batch-mean factor G=8", (8 - 1) / 8, "{:.3f}", "0.875")
quote("baseline example success", 1 - 0.5, "{:+.1f}", "+0.5")
quote("baseline example failure", 0 - 0.5, "signed1", "\u22120.5")

# the baseline figure: noise of a group estimate (G = 8) as a function of a constant baseline b
def noise(pi: list[float], rew: list[float], b: float, n: int = 8) -> float:
    """sqrt(trace Cov / n) / |true gradient| for the estimate (1/n) sum (R_i - b)(e_{s_i} - pi)."""
    K = len(pi)
    sc = [[(1.0 if j == s else 0.0) - pi[j] for j in range(K)] for s in range(K)]
    g = [[(rew[s] - b) * v for v in sc[s]] for s in range(K)]
    mean = [sum(pi[s] * g[s][j] for s in range(K)) for j in range(K)]
    p = sum(a * r for a, r in zip(pi, rew))
    true = [pi[j] * (rew[j] - p) for j in range(K)]
    assert max(abs(a - c) for a, c in zip(mean, true)) < 1e-12      # unbiased for every constant b
    var = sum(pi[s] * sum((g[s][j] - mean[j]) ** 2 for j in range(K)) for s in range(K))
    return math.sqrt(var / n) / math.sqrt(sum(v * v for v in true))


def b_star(pi: list[float], rew: list[float]) -> float:
    n2 = [sum(((1.0 if j == s else 0.0) - pi[j]) ** 2 for j in range(len(pi))) for s in range(len(pi))]
    return sum(pi[s] * rew[s] * n2[s] for s in range(len(pi))) / sum(pi[s] * n2[s] for s in range(len(pi)))


FIG_EASY = ([0.30, 0.02, 0.33, 0.20, 0.15], [1, 1, 0, 0, 0])   # the opening picture's problems (p = 0 dropped)
FIG_HARD = ([0.02, 0.50, 0.30, 0.18], [1, 0, 0, 0])
for pi, rew in (FIG_EASY, FIG_HARD):
    bs = b_star(pi, rew)
    grid = [-1 + 3 * i / 3000 for i in range(3001)]
    assert min(noise(pi, rew, b) for b in grid) >= noise(pi, rew, bs) - 1e-12
quote("fig easy noise b=0", noise(*FIG_EASY, 0.0), "{:.2f}", "0.59")
quote("fig easy noise b=p", noise(*FIG_EASY, 0.32), "{:.2f}", "0.45")
quote("fig easy b*", b_star(*FIG_EASY), "{:.2f}", "0.30")
quote("fig easy noise offset 10", noise(*FIG_EASY, -10.0), "{:.1f}", "12.9")
quote("fig hard noise b=0", noise(*FIG_HARD, 0.0), "{:.2f}", "2.47")
quote("fig hard noise b=p", noise(*FIG_HARD, 0.02), "{:.2f}", "2.44")
quote("fig hard noise b*", noise(*FIG_HARD, b_star(*FIG_HARD)), "{:.2f}", "2.42")
quote("fig hard b*", b_star(*FIG_HARD), "{:.2f}", "0.04")

# --- 4. Ratios: per token vs per sequence ------------------------------------------------------------
quote("1.02^500", 1.02**500, "{:,.0f}", "19,957")
quote("1.02^500 exact", 1.02**500, "{:,.2f}", "19,956.57")
quote("0.98^500", 0.98**500, "{:.6f}", "0.000041")

# --- 5. PPO's clipped objective ---------------------------------------------------------------------
EPS = 0.2


def clip_obj(rho: float, A: float, eps: float = EPS) -> float:
    return min(rho * A, max(1 - eps, min(1 + eps, rho)) * A)


def clip_grad(rho: float, A: float, eps: float = EPS) -> float:
    """d/drho of the clipped objective (away from the kinks)."""
    if A > 0:
        return A if rho < 1 + eps else 0.0
    return A if rho > 1 - eps else 0.0


for A in (1.0, -1.0, 0.37, -2.5):
    for rho in [0.05 + 0.01 * i for i in range(196)]:
        if min(abs(rho - (1 - EPS)), abs(rho - (1 + EPS))) < 1e-3:
            continue
        fd = (clip_obj(rho + 1e-7, A) - clip_obj(rho - 1e-7, A)) / 2e-7
        assert abs(fd - clip_grad(rho, A)) < 1e-6, (A, rho)
        assert clip_obj(rho, A) <= rho * A + 1e-12            # a pessimistic (lower) bound
# the four cases quoted in the post (A = +1 and A = -1, ratio 1.3 and 0.7)
quote("A>0 rho=1.3 objective", clip_obj(1.3, 1.0), "{:.1f}", "1.2")
assert clip_grad(1.3, 1.0) == 0.0 and clip_grad(0.7, 1.0) == 1.0
assert clip_grad(0.7, -1.0) == 0.0 and clip_grad(1.3, -1.0) == -1.0
quote("A<0 rho=1.3 objective", clip_obj(1.3, -1.0), "signed1", "\u22121.3")
quote("A<0 rho=0.7 objective", clip_obj(0.7, -1.0), "signed1", "\u22120.8")
quote("A>0 rho=0.7 objective", clip_obj(0.7, 1.0), "{:.1f}", "0.7")

# at pi_theta = pi_old (ratio 1) the clipped objective's gradient is the policy gradient A grad log pi
worst = 0.0
for trial in range(200):
    K = rng.randint(2, 8)
    z_old = [rng.gauss(0, 1.5) for _ in range(K)]
    a = rng.randrange(K)
    A = rng.gauss(0, 1)
    pold = softmax(z_old)[a]
    for j in range(K):
        zp = list(z_old); zp[j] += 1e-6
        zm = list(z_old); zm[j] -= 1e-6
        fd = (clip_obj(softmax(zp)[a] / pold, A) - clip_obj(softmax(zm)[a] / pold, A)) / 2e-6
        pg = A * ((1.0 if j == a else 0.0) - softmax(z_old)[j])
        worst = max(worst, abs(fd - pg))
assert worst < 1e-8
print(f"At ratio 1, the clipped objective's gradient equals A grad log pi (max err {worst:.1e})")

# clipping limits the incentive, not the ratio: one context, free logits shared by two sampled tokens.
# pi_old = (0.5, 0.3, 0.2); the current policy has moved to (0.65, 0.27, 0.08). Token a was sampled with
# A = +1 and its ratio is 1.3 (clipped: no gradient); token b was sampled with A = -1, ratio 0.9 (inside).
P_OLD = [0.5, 0.3, 0.2]
P_CUR = [0.65, 0.27, 0.08]
z = [math.log(v) for v in P_CUR]
rho = lambda zz, k: softmax(zz)[k] / P_OLD[k]
assert abs(rho(z, 0) - 1.3) < 1e-12 and abs(rho(z, 1) - 0.9) < 1e-12
obj = lambda zz: 0.5 * (clip_obj(rho(zz, 0), 1.0) + clip_obj(rho(zz, 1), -1.0))
grad = []
for j in range(3):
    zp = list(z); zp[j] += 1e-6
    zm = list(z); zm[j] -= 1e-6
    grad.append((obj(zp) - obj(zm)) / 2e-6)
# token a's own term contributes nothing: its gradient comes only through token b's term
ga = []
for j in range(3):
    zp = list(z); zp[j] += 1e-6
    zm = list(z); zm[j] -= 1e-6
    ga.append((clip_obj(rho(zp, 0), 1.0) - clip_obj(rho(zm, 0), 1.0)) / 2e-6)
assert max(abs(v) for v in ga) < 1e-9
z_new = [a + 1.0 * g for a, g in zip(z, grad)]          # one ascent step, step size 1
quote("shared toy rho_a before", rho(z, 0), "{:.2f}", "1.30")
quote("shared toy rho_a after", rho(z_new, 0), "{:.2f}", "1.52")
quote("shared toy rho_b", rho(z, 1), "{:.1f}", "0.9")
print(f"Clipped token a: ratio {rho(z, 0):.3f} -> {rho(z_new, 0):.3f} after a step driven by token b alone")

# --- 6. KL: the k3 estimator used in GRPO's loss ------------------------------------------------------
worst = 0.0
for trial in range(300):
    K = rng.randint(2, 10)
    pt = softmax([rng.gauss(0, 2) for _ in range(K)])
    pr = softmax([rng.gauss(0, 2) for _ in range(K)])
    kl = sum(a * math.log(a / b) for a, b in zip(pt, pr))
    k3 = [pr[j] / pt[j] - math.log(pr[j] / pt[j]) - 1 for j in range(K)]
    assert min(k3) >= -1e-15
    worst = max(worst, abs(sum(a * v for a, v in zip(pt, k3)) - kl))
assert worst < 1e-12
print(f"k3 = r - log r - 1 (r = pi_ref/pi_theta) is unbiased for KL(pi_theta || pi_ref) and >= 0 (err {worst:.1e})")
quote("GRPO KL coefficient (DeepSeekMath)", 0.04, "{:.2f}", "0.04")

# --- 7. Entropy is not reasoning diversity -----------------------------------------------------------
quote("model A trajectory entropy (20 binary choices)", 20 * math.log(2), "{:.1f} nats", "13.9 nats")
quote("model B trajectory entropy (4 strategies)", math.log(4), "{:.2f} nats", "1.39 nats")
quote("ratio of the two", 20 * math.log(2) / math.log(4), "{:.0f} times", "10 times")
quote("model A outcomes", 2**20, "{:,}", "1,048,576")

# --- 8. Sparse rewards -------------------------------------------------------------------------------
quote("pass@64 at 1e-4", 1 - (1 - 1e-4) ** 64, "{:.4f}", "0.0064")
quote("pass@1024 at 1e-4", 1 - (1 - 1e-4) ** 1024, "{:.3f}", "0.097")
quote("samples per success at 1e-4", 1 / 1e-4, "{:,.0f}", "10,000")

# --- 9. Check-yourself numbers -----------------------------------------------------------------------
quote("Q1 advantage of a failure with b = p", 0 - 0.01, "signed2", "\u22120.01")
quote("Q1 group size", 100, "{:.0f}", "100")

# --- 10. In my experiments: design statements re-read from sharpen-or-expand main ---------------------


def fetch(path: str) -> str:
    with urllib.request.urlopen(SOE_RAW + path, timeout=60) as fh:
        return fh.read().decode()


flat = lambda s: " ".join(s.split())
d013 = flat(fetch("docs/decisions/013-rl-no-kl-step-checkpoints.md"))
assert "**No KL term and no reference model for any RL algorithm** (GRPO, Dr.GRPO, DAPO, RS-GRPO)" in d013
assert "`use_kl_loss=false` and `use_kl_in_reward=false`" in d013
d022 = flat(fetch("docs/decisions/022-rl-updates-per-step.md"))
assert "**4 updates per step** for GRPO, Dr.GRPO, DAPO and RS-GRPO" in d022
assert "so the staleness of the last mini-batch (3 updates), is the same in every comparison" in d022
assert "so the PPO ratio pi_theta / pi_old is 1 for every token" in d022
assert "the clip ratios (and Clip-Higher) act on the later mini-batches" in d022
assert "Runs before this decision (M1.3–M1.6 development runs) made one update per step; none counts toward conclusions." in d022
d017 = flat(fetch("docs/decisions/017-rl-rollout-sampling-1.0.md"))
assert "**Rollouts** sample at temperature 1.0, top-p 1.0, top-k off, for every algorithm" in d017
assert "so PPO's importance ratio has no sampler/policy mismatch beyond numerics" in d017
assert "Evaluation** is unchanged (decision 004: 0.6 / 0.95, 16,384 tokens)" in d017
assert "Point 2 of 014, reward 0 for a rollout that reaches the 16,384-token cap, stands." in d017
d014 = flat(fetch("docs/decisions/014-rl-rollout-sampling-and-truncation.md"))
assert "**A rollout that reaches the 16,384-token cap gets reward 0**" in d014
mc = json.loads(fetch("docs/reports/m1_metrics_check.json"))
assert mc["note"] == "run started before decision 022 (one optimizer update per step)"
assert mc["updates_per_step"] == 1
for step in ("1", "2", "3"):
    assert mc["cross_checks"][step]["actor/pg_clipfrac"] == 0.0 and mc["cross_checks"][step]["actor/ppo_kl"] == 0.0
assert sorted(mc["cross_checks"]) == ["1", "2", "3"]
quote("updates per step", 4, "{:.0f} optimizer updates", "4 optimizer updates")
quote("staleness of the last mini-batch", 3, "up to {:.0f} updates", "up to 3 updates")
quote("token cap", 16384, "{:,}", "16,384")
print("Experiment design statements found in decisions 013, 014, 017, 022 and reports/m1_metrics_check.json at",
      SOE_COMMIT[:7])

# --- Compare with the text ---------------------------------------------------------------------------
post = POST.read_text() if POST.exists() else None
bad = []
for label, value, fmt, text in QUOTES + ANIM:
    if fmt.startswith("signed"):
        got = signed(value, int(fmt[-1]))
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
