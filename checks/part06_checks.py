"""Checks for Part 6, "Why vanilla GRPO fails: a failure map".

Recomputes every number the post quotes and every number the opening animation prints, and fails if any
differs from the text. Sections:
  0. the opening picture: a seeded simulation of 12 one-logit prompts under GRPO (the same xorshift32
     generator as the blog's AL.rng; the animation reproduces it draw by draw)
  1. the expected regime shares of that batch, exactly (enumeration) and by Monte Carlo
  2. zero-variance groups: dynamic-sampling cost, and resampling an all-wrong prompt
  3. the standard-deviation weighting at G = 8, against centering and against a pass@8 objective
  4. per-token weights of verl 0.9.1's three aggregations (agg_loss reimplemented; source asserted)
  5. the length toy: exact expected drifts, the trajectory in the figure, Monte Carlo checks
  6. clipping: absolute gains under a relative bound, Clip-Higher, compounding, the first update
  7. optimizer bias versus exploration limitation
  8. the experiments box: design statements re-read from sharpen-or-expand main at a pinned commit
Run: python checks/part06_checks.py
"""
import json
import math
import random
import sys
import urllib.request
from collections import defaultdict
from math import comb, exp, log, sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POST = ROOT / "posts" / "part-06-why-vanilla-grpo-fails.md"

SOE_COMMIT = "62b86c98a37916977e9048dbacfed2c7deaf68e9"
SOE_RAW = f"https://raw.githubusercontent.com/hieutrungle/sharpen-or-expand/{SOE_COMMIT}/"
VERL_COMMIT = "1876b06d0a3e4e71e06230be10af14492ca8a75b"
VERL_RAW = f"https://raw.githubusercontent.com/verl-project/verl/{VERL_COMMIT}/"

QUOTES: list[tuple[str, float, str, str]] = []
ANIM: list[tuple[str, float, str, str]] = []


def quote(label, value, fmt, text):
    QUOTES.append((label, value, fmt, text))


def anim(label, value, fmt, text):
    ANIM.append(("anim " + label, value, fmt, text))


def sig(z):
    return 1 / (1 + exp(-z))


def xorshift32(seed):
    s = (seed & 0xFFFFFFFF) or 1
    while True:
        s ^= (s << 13) & 0xFFFFFFFF
        s ^= s >> 17
        s ^= (s << 5) & 0xFFFFFFFF
        s &= 0xFFFFFFFF
        yield s / 4294967296


def group_adv(rs):
    G = len(rs)
    m = sum(rs) / G
    sd = sqrt(sum((r - m) ** 2 for r in rs) / G)
    return [0.0] * G if sd == 0 else [(r - m) / sd for r in rs]


# --- 0. The opening picture (blog assets/js/anim/part-06-failure-map.js) --------------------------------
SEED, ETA, G, ROUNDS = 20261120, 2.0, 8, 10
P0 = [1e-6, 1e-4, 0.005, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 0.95]
N = len(P0)


def simulate(seed=SEED, rng=None):
    rng = rng or xorshift32(seed)
    z = [log(p / (1 - p)) for p in P0]
    hist = []
    for _ in range(ROUNDS):
        p = [sig(v) for v in z]
        o = [[1 if next(rng) < p[i] else 0 for _ in range(G)] for i in range(N)]
        k = [sum(oi) for oi in o]
        reg = ["W" if v == 0 else "C" if v == G else "M" for v in k]
        hist.append(dict(p=p, o=o, k=k, reg=reg, nW=reg.count("W"), nM=reg.count("M"), nC=reg.count("C")))
        for i in range(N):
            if 0 < k[i] < G:
                # one policy-gradient step on the prompt's logit: (1/G) sum_j A_j (R_j - p) = sqrt(k(G-k))/G
                A = group_adv([float(r) for r in o[i]])
                g = sum(a * (r - p[i]) for a, r in zip(A, o[i])) / G
                assert abs(g - sqrt(k[i] * (G - k[i])) / G) < 1e-12
                z[i] += ETA * sqrt(k[i] * (G - k[i])) / G
    hist.append(dict(p=[sig(v) for v in z]))
    return hist


SIM = simulate()
fmtP = lambda p: "p < 0.001" if p < 0.0005 else f"p = {p:.3f}"
for r, h in enumerate(SIM[:-1]):
    for i in range(N):
        anim(f"round {r + 1} prompt {i} p", h["p"][i], "", fmtP(h["p"][i]))
        anim(f"round {r + 1} prompt {i} k", h["k"][i], "{:.0f}", str(h["k"][i]))
    anim(f"round {r + 1} mixed groups", h["nM"], "{:.0f}", str(h["nM"]))
    anim(f"round {r + 1} rollouts with signal", 8 * h["nM"], "{:.0f}", str(8 * h["nM"]))
    anim(f"round {r + 1} all wrong", h["nW"], "{:.0f}", str(h["nW"]))
    anim(f"round {r + 1} all correct", h["nC"], "{:.0f}", str(h["nC"]))
for i in range(N):
    anim(f"final prompt {i} p", SIM[-1]["p"][i], "", fmtP(SIM[-1]["p"][i]))

regs = [(h["nW"], h["nM"], h["nC"]) for h in SIM[:-1]]
print("Opening picture, (all wrong, mixed, all correct) per round:", regs)
# the seed criteria stated in the post
first = [r for r in range(ROUNDS) if SIM[r]["k"][3] > 0][0]
assert 1 <= first <= 4                                                # first success at 0.02 in rounds 2-5
assert all(SIM[r]["k"][i] == 0 for r in range(ROUNDS) for i in (0, 1, 2))
assert all(SIM[-1]["p"][i] == P0[i] or abs(SIM[-1]["p"][i] - P0[i]) < 1e-15 for i in (0, 1, 2))
assert min(SIM[-1]["p"][4:]) > 0.9


def criteria(hist):
    f = [r for r in range(ROUNDS) if hist[r]["k"][3] > 0]
    return (bool(f) and 1 <= f[0] <= 4 and all(hist[r]["k"][i] == 0 for r in range(ROUNDS) for i in (0, 1, 2))
            and min(hist[-1]["p"][4:]) > 0.9)


qual = [s for s in range(20261118, 20261118 + 400) if criteria(simulate(s))]
assert SEED == qual[0]
quote("seeds qualifying", len(qual), "({:.0f} of the 400 seeds tried qualify)", "(48 of the 400 seeds tried qualify)")
quote("P(success in 10 rounds) p=0.02", 1 - 0.98**80, "probability {:.2f}, one at 0.005", "probability 0.80, one at 0.005")
quote("P(success in 10 rounds) p=0.005", 1 - 0.995**80, "with probability {:.2f}, and one", "with probability 0.33, and one")
quote("P(success in 10 rounds) p=1e-4", 1 - (1 - 1e-4) ** 80, "with probability {:.3f}.", "with probability 0.008.")
quote("pic step", 0, "", "$$2\\sqrt{k(8 - k)}/8$$")
quote("pic step size", ETA, "step size {:.0f} moves", "step size 2 moves")
h1, h10 = SIM[0], SIM[ROUNDS - 1]
assert (h1["nM"], h1["nW"], h1["nC"]) == (7, 5, 0) and h1["reg"][5] == "W"
quote("pic round 1 mixed", h1["nM"], "In round 1, {:.0f} of the 12 groups are mixed", "In round 1, 7 of the 12 groups are mixed")
quote("pic round 1 rollouts", 8 * h1["nM"], "so {:.0f} of the 96 rollouts carry signal", "so 56 of the 96 rollouts carry signal")
quote("pic round 1 all wrong", h1["nW"], "The other {:.0f} groups are all wrong", "The other 5 groups are all wrong")
quote("P(all wrong) at p=0.1", 0.9**8, "all wrong with probability {:.2f}", "all wrong with probability 0.43")
quote("pic first success round", first + 1, "first success in round {:.0f}. Alone", "first success in round 3. Alone")
assert SIM[first]["k"][3] == 1
quote("pic 0.02 final", SIM[-1]["p"][3], "climbs to {:.3f} by the end", "climbs to 0.731 by the end")
frozen = [SIM[-1]["p"][i] for i in range(N) if h10["reg"][i] == "C"]
assert len(frozen) == 7 and all(SIM[-1]["p"][i] == h10["p"][i] for i in range(N) if h10["reg"][i] == "C")
quote("pic frozen range", 0, "", f"between {min(frozen):.3f} and {max(frozen):.3f}")
assert f"between {min(frozen):.3f} and {max(frozen):.3f}" == "between 0.955 and 0.996"
assert (h10["nM"], h10["nC"], h10["nW"]) == (2, 7, 3)
quote("pic round 10", 0, "", "By round 10 only 2 groups are mixed (16 of 96 rollouts), 7 are all correct and 3 are all wrong")
quote("alt text 0.731", SIM[-1]["p"][3], "climbs to {:.3f}. Easy", "climbs to 0.731. Easy")

# --- 1. Expected regime shares of the batch: exact enumeration and Monte Carlo ----------------------------


def expected_regimes():
    dists = [{round(log(p / (1 - p)), 12): 1.0} for p in P0]
    out, nstates = [], 0
    for _ in range(ROUNDS):
        ew = em = ec = 0.0
        new = []
        for d in dists:
            nd = defaultdict(float)
            for z, pr in d.items():
                p = sig(z)
                for k in range(G + 1):
                    pk = pr * comb(G, k) * p**k * (1 - p) ** (G - k)
                    if k == 0:
                        ew += pk
                        nd[z] += pk
                    elif k == G:
                        ec += pk
                        nd[z] += pk
                    else:
                        em += pk
                        nd[round(z + ETA * sqrt(k * (G - k)) / G, 12)] += pk
            new.append(nd)
        dists = new
        out.append((ew / N, em / N, ec / N))
        nstates = max(len(d) for d in dists)
    return out, nstates


EXP, NSTATES = expected_regimes()
assert NSTATES == comb(ROUNDS + 4, 4) == 1001   # 4 distinct step sizes, at most 10 steps: multisets of size <= 10
# round 1 by the closed form P(all wrong) = (1-p)^G, P(all correct) = p^G
assert abs(EXP[0][0] - sum((1 - p) ** G for p in P0) / N) < 1e-12
assert abs(EXP[0][2] - sum(p**G for p in P0) / N) < 1e-12
zv = [w + c for w, _, c in EXP]
quote("states per prompt", NSTATES, "at most {:,} states per prompt", "at most 1,001 states per prompt")
quote("zero-var round 1", zv[0], "{:.1%} of the batch's groups are zero-variance", "50.9% of the batch's groups are zero-variance")
quote("all wrong round 1", EXP[0][0], "{:.1%} all wrong", "42.6% all wrong")
quote("all correct round 1", EXP[0][2], "and {:.1%} all correct. Since", "and 8.3% all correct. Since")
quote("zero-var r1 (pic)", zv[0], "grows from {:.1%} in round 1", "grows from 50.9% in round 1")
quote("zero-var r10 (pic)", zv[-1], "to {:.1%} in round 10.", "to 78.5% in round 10.")
quote("fig zero-var r1", zv[0], "It is {:.1%} in round 1", "It is 50.9% in round 1")
quote("fig zero-var r10", zv[-1], "and {:.1%} in round 10. The", "and 78.5% in round 10. The")
quote("fig all-correct", 0, "", f"grows from {EXP[0][2]:.1%} to {EXP[-1][2]:.1%}")
assert f"grows from {EXP[0][2]:.1%} to {EXP[-1][2]:.1%}" == "grows from 8.3% to 49.7%"
quote("fig mixed", 0, "", f"falls from {EXP[0][1]:.1%} to {EXP[-1][1]:.1%}")
assert f"falls from {EXP[0][1]:.1%} to {EXP[-1][1]:.1%}" == "falls from 49.1% to 21.5%"
quote("fig all wrong r10", EXP[-1][0], "only slowly, to {:.1%}", "only slowly, to 28.9%")
quote("takeaway zero-var", 0, "", f"({zv[0]:.1%} to {zv[-1]:.1%} in the opening batch")
assert f"({zv[0]:.1%} to {zv[-1]:.1%} in the opening batch" == "(50.9% to 78.5% in the opening batch"

# Monte Carlo over 3,000 seeds of the same simulation
MC = 3000
acc = [[0, 0, 0] for _ in range(ROUNDS)]
for s in range(MC):
    hist = simulate(rng=xorshift32(1000003 + 7919 * s))
    for r in range(ROUNDS):
        acc[r][0] += hist[r]["nW"]
        acc[r][1] += hist[r]["nM"]
        acc[r][2] += hist[r]["nC"]
worst = max(abs(acc[r][j] / (MC * N) - EXP[r][j]) for r in range(ROUNDS) for j in range(3))
assert worst < 0.01, worst
quote("MC seeds", MC, "over {:,} seeds", "over 3,000 seeds")
print(f"Expected zero-variance share {zv[0]:.4f} -> {zv[-1]:.4f}; Monte Carlo over {MC} seeds within {worst:.4f}")

# --- 2. Dynamic sampling and resampling an all-wrong prompt ------------------------------------------------
quote("DS cost round 1", 1 / EXP[0][1], "an expected {:.2f} sampled groups", "an expected 2.04 sampled groups")
quote("DS cost round 10", 1 / EXP[-1][1], "and {:.2f} in round 10. It", "and 4.66 in round 10. It")
p_any = lambda p, n: 1 - (1 - p) ** n
quote("resample k=1", p_any(1e-4, 8), "{:.2%} after one group", "0.08% after one group")
quote("resample k=10", p_any(1e-4, 80), "{:.2%} after 10 groups", "0.80% after 10 groups")
quote("resample k=100", p_any(1e-4, 800), "{:.1%} after 100;", "7.7% after 100;")
n_half = math.ceil(log(0.5) / log(1 - 1e-4))
assert p_any(1e-4, n_half) >= 0.5 > p_any(1e-4, n_half - 1)
quote("rollouts for an even chance", n_half, "it takes {:,} rollouts", "it takes 6,932 rollouts")
quote("groups for an even chance", math.ceil(n_half / 8), "({:.0f} groups)", "(867 groups)")
quote("Q1 resample", p_any(1e-4, 800), "{:.1%} after 100 groups", "7.7% after 100 groups")
# an all-wrong group leaves p unchanged, so resampling k groups finds a success with exactly 1-(1-p)^{Gk};
# simulate the GRPO loop on one prompt that stops at its first success
rng = random.Random(11)
for p, kk in ((0.01, 30), (0.002, 60)):
    hits, T = 0, 20000
    for _ in range(T):
        z = log(p / (1 - p))
        for _ in range(kk):
            o = [1 if rng.random() < sig(z) else 0 for _ in range(G)]
            if sum(o) == 0:
                assert all(a == 0 for a in group_adv([0.0] * G))      # no advantage, no step: z unchanged
                continue
            hits += 1
            break
    se = sqrt(p_any(p, G * kk) * (1 - p_any(p, G * kk)) / T)
    assert abs(hits / T - p_any(p, G * kk)) < 4 * se, (p, hits / T, p_any(p, G * kk))
print("Resampling an all-wrong prompt: simulated success rate matches 1-(1-p)^(Gk)")

# --- 3. The standard-deviation weighting -----------------------------------------------------------------


def w_std(p, Gc=8):
    tot = 0.0
    for k in range(1, Gc):
        pk = comb(Gc, k) * p**k * (1 - p) ** (Gc - k)
        tot += pk * ((k / Gc) * sqrt((Gc - k) / k) / p + ((Gc - k) / Gc) * sqrt(k / (Gc - k)) / (1 - p))
    return tot


def w_center(p, Gc=8):
    tot = 0.0
    for k in range(0, Gc + 1):
        pk = comb(Gc, k) * p**k * (1 - p) ** (Gc - k)
        tot += pk * ((k / Gc) * (1 - k / Gc) / p + ((Gc - k) / Gc) * (k / Gc) / (1 - p))
    return tot


assert all(abs(w_center(p) - 7 / 8) < 1e-12 for p in (0.01, 0.3, 0.9))
PS = [0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99]
rel = [(w_std(p) / w_center(p)) / (w_std(0.5) / w_center(0.5)) for p in PS]
row1 = "| GRPO's weight relative to centering, normalized at $$p = 0.5$$ | " + " | ".join(f"{v:.2f}" for v in rel) + " |"
quote("table std row", 0, "", row1)
assert row1 == "| GRPO's weight relative to centering, normalized at $$p = 0.5$$ | 1.41 | 1.33 | 1.14 | 1.00 | 1.14 | 1.33 | 1.41 |"
pk8 = [(1 - p) ** 7 / 0.5**7 for p in PS]


def fmt_pk(v):
    if v >= 100:
        return f"{v:.0f}"
    if v >= 10:
        return f"{v:.1f}"
    if v >= 0.001:
        return f"{v:.2f}" if v >= 0.1 else f"{v:.4f}".rstrip("0")
    m, e = f"{v:.1e}".split("e")
    return f"$${m} \\times 10^{{{int(e)}}}$$"


row2 = "| A pass@8 objective's weight, normalized at $$p = 0.5$$ | " + " | ".join(fmt_pk(v) for v in pk8) + " |"
quote("table pass@8 row", 0, "", row2)
assert row2 == ("| A pass@8 objective's weight, normalized at $$p = 0.5$$ | 119 | 89.4 | 26.8 | 1.00 | 0.0016 | "
                "$$1.0 \\times 10^{-7}$$ | $$1.3 \\times 10^{-12}$$ |"), row2
quote("std 1.41 text", rel[0], "counts {:.2f} times as much", "counts 1.41 times as much")
quote("pass@8 119 text", pk8[0], "weights $$p = 0.01$$ {:.0f} times", "weights $$p = 0.01$$ 119 times")

# --- 4. Per-token weights of verl 0.9.1's aggregations ---------------------------------------------------


def agg_loss(loss_mat, mask, mode, dp_size=1, batch_num_tokens=None, global_batch_size=None, loss_scale_factor=None):
    """A line-by-line reimplementation of verl 0.9.1's agg_loss on lists of rows."""
    tsum = lambda rows: sum(sum(a * b for a, b in zip(r, m)) for r, m in zip(rows, mask))
    if mode == "token-mean":
        n = sum(map(sum, mask)) if batch_num_tokens is None else batch_num_tokens
        return tsum(loss_mat) / n * dp_size
    if mode in ("seq-mean-token-sum", "seq-mean-token-sum-norm"):
        seq = [sum(a * b for a, b in zip(r, m)) for r, m in zip(loss_mat, mask)]
        smask = [1.0 if sum(m) > 0 else 0.0 for m in mask]
        gbs = sum(smask) if global_batch_size is None else global_batch_size
        loss = sum(a * b for a, b in zip(seq, smask)) / gbs * dp_size
        if mode == "seq-mean-token-sum-norm":
            loss /= len(mask[0]) if loss_scale_factor is None else loss_scale_factor   # loss_mask.shape[-1]
        return loss
    if mode == "seq-mean-token-mean":
        cnt = [sum(m) for m in mask]
        seq = [sum(a * b for a, b in zip(r, m)) / (c + 1e-8) for r, m, c in zip(loss_mat, mask, cnt)]
        smask = [1.0 if c > 0 else 0.0 for c in cnt]
        gbs = sum(smask) if global_batch_size is None else global_batch_size
        return sum(a * b for a, b in zip(seq, smask)) / gbs * dp_size
    raise ValueError(mode)


LENS = (100, 1000)


def token_weights(mode, layout, lsf=None):
    """d(loss)/d(loss_mat[i, 0]) for the 100- and 1,000-token responses (the loss is linear in loss_mat)."""
    out = []
    for i, L in enumerate(LENS):
        if layout == "one":                      # both responses in one micro-batch, padded to 1,000
            mask = [[1.0] * LENS[j] + [0.0] * (1000 - LENS[j]) for j in range(2)]
            lm = [[0.0] * 1000 for _ in range(2)]
            lm[i][0] = 1.0
        else:                                    # each response in its own micro-batch, padded to its own length
            mask, lm = [[1.0] * L], [[1.0] + [0.0] * (L - 1)]
        out.append(agg_loss(lm, mask, mode, batch_num_tokens=1100, global_batch_size=2, loss_scale_factor=lsf))
    return out


rows = [("`seq-mean-token-mean` (GRPO)", "seq-mean-token-mean", "one", None),
        ("`token-mean` (DAPO)", "token-mean", "one", None),
        ("`seq-mean-token-sum-norm`, default, both in one micro-batch", "seq-mean-token-sum-norm", "one", None),
        ("`seq-mean-token-sum-norm`, default, each in its own micro-batch", "seq-mean-token-sum-norm", "sep", None),
        ("`seq-mean-token-sum-norm` with a constant 16,384", "seq-mean-token-sum-norm", "one", 16384)]
expect = ["| `seq-mean-token-mean` (GRPO) | 0.005 | 0.0005 | 50% |",
          "| `token-mean` (DAPO) | 0.00091 | 0.00091 | 90.9% |",
          "| `seq-mean-token-sum-norm`, default, both in one micro-batch | 0.0005 | 0.0005 | 90.9% |",
          "| `seq-mean-token-sum-norm`, default, each in its own micro-batch | 0.005 | 0.0005 | 50% |",
          "| `seq-mean-token-sum-norm` with a constant 16,384 | 0.000031 | 0.000031 | 90.9% |"]


def two_sig(v):
    """At most two significant figures, without trailing zeros: 0.005, 0.0005, 0.00091, 0.000031."""
    d = -int(math.floor(math.log10(abs(v)))) + 1
    return f"{round(v, d):.{d}f}".rstrip("0").rstrip(".")


for (name, mode, layout, lsf), exp_row in zip(rows, expect):
    w = token_weights(mode, layout, lsf)
    if mode == "seq-mean-token-mean":
        assert token_weights(mode, "sep") == w or all(abs(a - b) < 1e-15 for a, b in zip(token_weights(mode, "sep"), w))
    share = w[1] * 1000 / (w[0] * 100 + w[1] * 1000)
    sh = f"{share:.0%}" if abs(share - 0.5) < 1e-9 else f"{share:.1%}"
    row = f"| {name} | {two_sig(w[0])} | {two_sig(w[1])} | {sh} |"
    assert row == exp_row, (row, exp_row)
    quote("agg table " + name, 0, "", row)
# Q2: 100 and 2,000 tokens in a group of 8 under the per-response average
quote("Q2 short", 1 / 800, "{:.5f} times $$A$$ per token", "0.00125 times $$A$$ per token")
quote("Q2 long", 1 / 16000, "{:.7f} times $$A$$ for the long", "0.0000625 times $$A$$ for the long")
quote("Q2 ratio", (1 / 800) / (1 / 16000), "suppressed {:.0f} times less", "suppressed 20 times less")
quote("Part 2 ratio", 10, "pushed up {:.0f} times as hard", "pushed up 10 times as hard")

core = urllib.request.urlopen(VERL_RAW + "verl/trainer/ppo/core_algos.py", timeout=60).read().decode()
L = core.splitlines()
assert L[1139] == "def agg_loss(" and L[1205] == "    return loss"                       # linked L1140-L1206
assert L[1284] == '@register_policy_loss("vanilla")  # type: ignore[arg-type]'            # linked L1285-L1376
assert L[1285] == "def compute_policy_loss_vanilla(" and L[1375] == "    return pg_loss, pg_metrics"
agg_src = "\n".join(L[1139:1206])
for line in ['        loss = verl_F.masked_sum(loss_mat, loss_mask) / batch_num_tokens * dp_size',
             '                horizon = loss_mask.shape[-1]',
             '                loss_scale_factor = horizon',
             '            loss /= loss_scale_factor',
             '        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / (seq_mask + 1e-8)  # token-mean',
             '        loss = verl_F.masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size  # seq-mean']:
    assert line in agg_src, line
pol_src = "\n".join(L[1284:1376])
for line in ["    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio",
             "        ratio, 1 - cliprange_low, 1 + cliprange_high",
             "    clip_pg_losses1 = torch.maximum(", "    pg_losses1 = -advantages * ratio"]:
    assert line in pol_src, line
eng = urllib.request.urlopen(VERL_RAW + "verl/workers/engine/fsdp/transformer_impl.py", timeout=60).read().decode()
assert "# compute num_tokens in global batch for loss normalization" in eng
assert 'batch_num_tokens = data["loss_mask"].sum().to(get_device_id())' in eng
assert "batch_num_tokens, op=torch.distributed.ReduceOp.SUM, group=self.get_data_parallel_group()" in eng
print("verl 0.9.1 (", VERL_COMMIT[:7], "): agg_loss and compute_policy_loss_vanilla asserted; token-mean "
      "divides by the all-reduced token count of the mini-batch")

# --- 5. The length toy -----------------------------------------------------------------------------------


def e_sigma(p, g=G):
    return sum(comb(g, k) * p**k * (1 - p) ** (g - k) * sqrt(k * (g - k)) / g for k in range(g + 1))


def c_len(s):
    q = 1 - s
    e_inv = s * (-log(s) - q) / q**2
    return e_inv - s * (1 - e_inv)


def c_len_sum(s, Lmax=400000):
    tot = mean0 = 0.0
    for Lq in range(1, Lmax):
        pm = s * (1 - s) ** (Lq - 1)
        tot += pm * (1 - Lq * s) / (1 + Lq)
        mean0 += pm * (1 - Lq * s)
    return tot, mean0


cs, m0 = c_len_sum(0.01)
assert abs(cs - c_len(0.01)) < 1e-12 and abs(m0) < 1e-10                 # E[1 - L s] = 0: no drift with a constant
# E[(1/G) sum_i A_i R_i] = E[sigma_R]
for p in (0.2, 0.5):
    lhs = sum(comb(G, k) * p**k * (1 - p) ** (G - k) * ((k / G) * sqrt((G - k) / k) if 0 < k < G else 0) for k in range(G + 1))
    assert abs(lhs - e_sigma(p)) < 1e-12
quote("E sigma", e_sigma(0.5), "$$\\mathbb{{E}}[\\sigma_R] = {:.3f}$$", "$$\\mathbb{E}[\\sigma_R] = 0.464$$")
quote("c(0.01) text", c_len(0.01), "$$\\mathbb{{E}}[(1 - Ls)/(1 + L)] = {:.4f}$$", "$$\\mathbb{E}[(1 - Ls)/(1 + L)] = 0.0273$$")
tc = tw = log(0.01 / 0.99)
for _ in range(200):
    tc += 0.5 * e_sigma(0.5) * c_len(sig(tc))
    tw -= 0.5 * e_sigma(0.5) * c_len(sig(tw))
quote("toy wrong", 1 / sig(tw), "wrong responses average {:.0f} tokens", "wrong responses average 247 tokens")
quote("toy correct", 1 / sig(tc), "and correct ones {:.0f}.", "and correct ones 11.")
quote("toy step", 0.5, "step size {:.1f} from", "step size 0.5 from")
quote("toy mean", 100, "mean $$1/s = {:.0f}$$", "mean $$1/s = 100$$")

# Monte Carlo of the expected stop-logit updates at the start (mini-batches of 16 groups x 8 = 128 responses)
rng = random.Random(2026)


def drift(mode, shared, nb=3000, B=16, p=0.5, s=0.01):
    tot = [0.0, 0.0]
    lq = log(1 - s)
    for _ in range(nb):
        recs = []
        for _ in range(B):
            wrong = [rng.random() >= p for _ in range(G)]
            A = group_adv([0.0 if w else 1.0 for w in wrong])
            for w, a in zip(wrong, A):
                Lq = int(log(1 - rng.random()) / lq) + 1
                recs.append((w, a, Lq))
        ntok = sum(1 + r[2] for r in recs)
        for w, a, Lq in recs:
            wt = a / (B * G * (1 + Lq)) if mode == "smtm" else a / ntok if mode == "tm" else a / (B * G * 101)
            tot[0 if shared else int(w)] += wt * (1 - Lq * s)
    return [t / nb for t in tot]


d_smtm, d_tm = drift("smtm", False), drift("tm", False)
exact = e_sigma(0.5) * c_len(0.01)
assert abs(d_smtm[0] - exact) < 0.1 * exact and abs(d_smtm[1] + exact) < 0.1 * exact
assert abs(d_tm[0]) < 0.01 * exact and abs(d_tm[1]) < 0.01 * exact
quote("token-mean drift", 0, "", "it is under 1% of the per-response average's")
quote("mini-batch size", 16 * 8, "mini-batches of {:.0f} responses", "mini-batches of 128 responses")
print(f"Length toy: exact drift {exact:.5f}; Monte Carlo per-response average {d_smtm[0]:+.5f}/{d_smtm[1]:+.5f}, "
      f"token-mean {d_tm[0]:+.6f}/{d_tm[1]:+.6f}; after 200 updates correct {1 / sig(tc):.1f}, wrong {1 / sig(tw):.1f}")
# shared stop probability: the per-group sum of advantages is 0, so the expected drift is exactly 0
for k in range(G + 1):
    assert abs(sum(group_adv([1.0] * k + [0.0] * (G - k)))) < 1e-12

# --- 6. Clipping -----------------------------------------------------------------------------------------
OLD = [0.001, 0.01, 0.1, 0.5, 0.9]


def fnum(v):
    return f"{v:.5f}".rstrip("0").rstrip(".")


for po in OLD:
    for eps in (0.2, 0.28):
        bound = po * (1 + eps)
        # the clipped term min(rho A, clip(rho) A) with A > 0 is flat beyond rho = 1 + eps, for every po
        A = 1.0
        f = lambda pn: min(pn / po * A, min(max(pn / po, 1 - 0.2), 1 + eps) * A)
        h = 1e-9 * po
        assert abs((f(bound * 1.01 + h) - f(bound * 1.01)) / h) < 1e-6           # no gradient past the bound
        assert abs((f(bound * 0.99 + h) - f(bound * 0.99)) / h - 1 / po) < 1e-3 / po
for po, b2, g2, b28, g28 in ((0.001, "0.0012", "0.0002", "0.00128", "0.00028"), (0.01, "0.012", "0.002", "0.0128", "0.0028"),
                             (0.1, "0.12", "0.02", "0.128", "0.028"), (0.5, "0.6", "0.1", "0.64", "0.14")):
    assert (fnum(po * 1.2), fnum(po * 0.2), fnum(po * 1.28), fnum(po * 0.28)) == (b2, g2, b28, g28)
    row = f"| {po:g} | {b2} | {g2} | {b28} | {g28} |"
    quote(f"clip row {po}", 0, "", row)
assert fnum(0.9 * 1.2) == "1.08" and fnum(0.9 * 1.28) == "1.152" and abs((1 - 0.9) - 0.1) < 1e-12
quote("clip row 0.9", 0, "", "| 0.9 | 1.08 (no bound below 1) | up to 0.1 | 1.152 (no bound below 1) | up to 0.1 |")
quote("rare vs common", 0.1 / 0.0002, "{:.0f} times as much", "500 times as much")
quote("DAPO example", 0, "", "their upper bounds are 0.012 and 1.08")
assert fnum(0.01 * 1.2) == "0.012" and fnum(0.9 * 1.2) == "1.08"
n02 = math.ceil(log(0.5 / 0.001) / log(1.2))
n028 = math.ceil(log(0.5 / 0.001) / log(1.28))
assert 0.001 * 1.2 ** (n02 - 1) < 0.5 <= 0.001 * 1.2**n02
quote("compound 0.2", n02, "0.5 after {:.0f} batches", "0.5 after 35 batches")
quote("compound 0.28", n028, "and {:.0f} at $$\\epsilon = 0.28$$", "and 26 at $$\\epsilon = 0.28$$")
quote("Clip-Higher room", 0.28 / 0.2 - 1, "{:.0%} more room", "40% more room")
quote("Q3 factor", 0.1 / 0.0002, "differs by a factor of {:.0f}", "differs by a factor of 500")
quote("DAPO eps", 0, "", "$$\\epsilon_{\\text{low}} = 0.2$$ and $$\\epsilon_{\\text{high}} = 0.28$$")
quote("DAPO overlong", 16384 + 4096, "so {:,} tokens in all", "so 20,480 tokens in all")
# on the first update the ratio is 1 and the clipped objective's gradient is the policy gradient
po = 0.3
fa = lambda pn: min(pn / po, min(max(pn / po, 0.8), 1.2))
assert abs((fa(po + 1e-9) - fa(po)) / 1e-9 - 1 / po) < 1e-3
quote("lone success", sqrt(7), "\\sqrt{{7}}$$ ≈ {:.2f}", "\\sqrt{7}$$ ≈ 2.65")

# --- 7. Optimizer bias versus exploration limitation -------------------------------------------------------
v8 = p_any(1e-8, 8)
quote("1e-8 group", v8, "", "$$8.0 \\times 10^{-8}$$")
assert f"{v8:.1e}" == "8.0e-08"
v10k = p_any(1e-8, 80000)
quote("1e-8 10,000 rounds", v10k, "", "$$8.0 \\times 10^{-4}$$")
assert f"{v10k:.1e}" == "8.0e-04"

# --- 8. In my experiments: design statements re-read from sharpen-or-expand main ----------------------------
fetch = lambda path: urllib.request.urlopen(SOE_RAW + path, timeout=60).read().decode()
flat = lambda s: " ".join(s.split())
d001 = flat(fetch("docs/decisions/001-gpu-stack-verl.md"))
assert "| GRPO | `algorithm.adv_estimator=grpo`, `actor.loss_agg_mode=seq-mean-token-mean`, symmetric clip |" in d001
assert ("| Dr.GRPO | GRPO + `algorithm.norm_adv_by_std_in_grpo=false`, "
        "`actor.loss_agg_mode=seq-mean-token-sum-norm` |") in d001
assert ("| DAPO | `algorithm.filter_groups.enable` (dynamic sampling), `actor.clip_ratio_high > clip_ratio_low` "
        "(Clip-Higher), `actor.loss_agg_mode=token-mean`, DAPO reward manager `overlong_buffer` |") in d001
d025 = flat(fetch("docs/decisions/025-g0-shared-4b-config.md"))
assert ("Algorithm configs may differ only in the decision-001 switches (advantage estimator, loss aggregation, "
        "Clip-Higher, dynamic sampling, overlong shaping, RS-GRPO's β), in `name`, and in M7's group size.") in d025
assert "clip 0.2 unless Clip-Higher" in d025
mut = flat(fetch("docs/reports/m1_rl_tests_mutations.md"))
assert ("Without `loss_scale_factor`, `seq-mean-token-sum-norm` (Dr.GRPO's aggregation) divides by the padded "
        "width of each micro-batch, which varies with dynamic batching.") in mut
m2 = flat(fetch("docs/roadmap/M2.md"))
assert ("**Dr.GRPO constant `loss_scale_factor` (D9):** placed in M3's wiring of Dr.GRPO (the algorithm is not "
        "wired yet). Proposal: 16,384, the response cap.") in m2
assert "| D9 | **Answered: (a)** (user, 2026-11-03). Dr.GRPO `loss_scale_factor` | (a) constant 16,384 (the cap), wired in M3" in m2
d014 = flat(fetch("docs/decisions/014-rl-rollout-sampling-and-truncation.md"))
assert ("2. **A rollout that reaches the 16,384-token cap gets reward 0**, whatever its last box says, as in "
        "evaluation (truncated = incorrect). DAPO's overlong shaping (M3) is applied on top of this, and its "
        "ablation keeps the base rule.") in d014
plan = flat(fetch("docs/milestone-plan.md"))
for s_ in ("Full DAPO minus dynamic sampling, minus Clip-Higher, minus token-level loss.",
           "DAPO minus overlong handling only if truncation showed up in M3/M5.",
           "GRPO at G = 4, 8, 16 with shorter runs, matched by rollout tokens."):
    assert s_ in plan, s_
d020 = flat(fetch("docs/decisions/020-rl-token-budget-stop.md"))
assert "Runs are compared at equal rollout-token budgets (decision 013)." in d020
d013 = flat(fetch("docs/decisions/013-rl-no-kl-step-checkpoints.md"))
assert "Runs are compared at equal rollout-token budgets by taking, for each token mark, the checkpoint nearest to it." in d013
d018 = flat(fetch("docs/decisions/018-rl-checkpoint-export-and-token-count.md"))
assert ("**Rollout tokens are generated tokens of training rollouts:** the response tokens of every training "
        "rollout, a truncated one counted at its 16,384 tokens.") in d018
assert ("Prompt tokens are left out because each prompt repeats G times per group, so they would make the count "
        "depend on the group size the G = 4 / 8 / 16 comparison varies.") in d018
d031 = flat(fetch("docs/decisions/031-rl-and-dpo-start-from-the-base.md"))
assert ("Every pass@1 change is split into a stopping part and an accuracy-when-stopped part (the "
        "`scripts/sft_diagnostic.py` method) and shown with truncation, for every algorithm.") in d031
audit = flat(fetch("docs/reports/m2_base_start_audit.md"))
assert "untrained base at step 1 (512 rollouts)" in audit
s1 = json.loads(fetch("docs/reports/m2_rollout_groups.json"))["full_bf16"]["step_1"]
assert s1["rollouts"] == 512 and s1["groups"] == 64 and s1["group_sizes"] == [8]
n_trunc = s1["truncated"] * 512
n_stop = 512 - n_trunc
n_ok = s1["score_when_stopped"] * n_stop
assert abs(n_trunc - round(n_trunc)) < 1e-9 and abs(n_ok - round(n_ok)) < 1e-6
n_trunc, n_stop, n_ok = round(n_trunc), round(n_stop), round(n_ok)
n_fail = 512 - n_ok
# group shares are consistent with the counts: every rollout of an all-correct group is correct
assert s1["groups_all_correct"] * 64 * 8 <= n_ok
quote("step 1 rollouts", 512, "of its {:.0f} rollouts", "of its 512 rollouts")
quote("step 1 truncated", n_trunc, "{:.0f} hit the cap", "185 hit the cap")
quote("step 1 stopped", n_stop, "of the {:.0f} that stopped", "of the 327 that stopped")
quote("step 1 correct", n_ok, "{:.0f} were correct", "312 were correct")
quote("step 1 failures", 0, "", f"So {n_trunc} of its {n_fail} failures, {n_trunc / n_fail:.1%}, were truncations")
assert f"So {n_trunc} of its {n_fail} failures, {n_trunc / n_fail:.1%}, were truncations" == "So 185 of its 200 failures, 92.5%, were truncations"
quote("step 1 truncated in all-wrong", s1["truncated_in_all_wrong_groups"], "{:.1%} of the rollouts in all-wrong", "92.9% of the rollouts in all-wrong")
quote("cap", 16384, "{:,}-token cap", "16,384-token cap")
print("Experiment design statements found in decisions 001, 013, 014, 018, 020, 025, 031, the milestone plan, the "
      "M2 roadmap, the loss-test report and the step-1 rollout data at", SOE_COMMIT[:7])

# --- Compare with the text ---------------------------------------------------------------------------------
post = POST.read_text() if POST.exists() else None
bad = []
for label, value, fmt, text in QUOTES + ANIM:
    got = text if fmt == "" else fmt.format(value)
    status = "ok" if got == text else "MISMATCH"
    if got != text:
        bad.append(label)
    if post is not None and not label.startswith("anim ") and text not in post:
        status += " (not in post)"
        bad.append(label + " [not in post]")
    if status != "ok" or not label.startswith("anim "):
        print(f"{status:<18} {label:<44} computed {got[:44]:<44} text {text[:44]}")
if bad:
    print("\nFAILED:", *bad, sep="\n  ")
    sys.exit(1)
print(f"\nall {len(QUOTES)} quoted values and {len(ANIM)} animation values match"
      + ("" if post is not None else " (post not written yet)"))
