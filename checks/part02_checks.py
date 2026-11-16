"""Part 2 checks: recompute every number the post quotes.

Run:  python checks/part02_checks.py

Each entry in QUOTES pairs a computed value with the exact string the post prints. The script fails
if the computed value, formatted as the post formats it, differs from that string, or (once the post
exists) if the string does not appear in posts/part-02-sft-probability-reshaping.md.

It also
  * checks the softmax cross-entropy gradient dL/dz_j = p_j - 1[j = c] against central finite
    differences on random logits;
  * checks H(q, pi) = H(q) + KL(q || pi) on random distributions;
  * recomputes the numbers shown in the animation (blog assets/js/anim/part-02-reshaping.js):
    the logit-space gradient steps on the branching token, the trajectory 0.05 -> 0.5184 and the
    strategy masses with N_eff;
  * recomputes the toy exposure-bias model behind figures/part02_exposure.py;
  * re-reads the experiment numbers from sharpen-or-expand's public main branch, pinned to a commit.
"""

from __future__ import annotations

import json
import math
import random
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POST = ROOT / "posts" / "part-02-sft-probability-reshaping.md"

# sharpen-or-expand, public main, the commit the post links to
SOE_COMMIT = "62b86c98a37916977e9048dbacfed2c7deaf68e9"
SOE_RAW = f"https://raw.githubusercontent.com/hieutrungle/sharpen-or-expand/{SOE_COMMIT}/"

QUOTES: list[tuple[str, float, str, str]] = []  # (label, value, format, string in the post)
ANIM: list[tuple[str, float, str, str]] = []    # numbers the animation prints (not required in the post)


def quote(label: str, value: float, fmt: str, text: str) -> None:
    QUOTES.append((label, value, fmt, text))


def anim(label: str, value: float, fmt: str, text: str) -> None:
    ANIM.append((label, value, fmt, text))


def signed(x: float, nd: int) -> str:
    """Signed number with a typographic minus, as the post prints it."""
    s = f"{x:+.{nd}f}"
    return s.replace("-", "\u2212")


def softmax(z: list[float]) -> list[float]:
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


def ce(z: list[float], c: int) -> float:
    return -math.log(softmax(z)[c])


def entropy(q: list[float]) -> float:
    return -sum(x * math.log(x) for x in q if x > 0)


def n_eff(masses: list[float]) -> float:
    """Hill number of order 1 of the distribution over correct strategies (Part 1)."""
    tot = sum(masses)
    return math.exp(entropy([m / tot for m in masses]))


# --- Gradient of softmax cross-entropy: finite differences ----------------------------------------
rng = random.Random(2)
worst = 0.0
for _ in range(200):
    V = rng.randint(2, 12)
    z = [rng.gauss(0, 3) for _ in range(V)]
    c = rng.randrange(V)
    p = softmax(z)
    for j in range(V):
        h = 1e-6
        zp, zm = z[:], z[:]
        zp[j] += h
        zm[j] -= h
        fd = (ce(zp, c) - ce(zm, c)) / (2 * h)
        worst = max(worst, abs(fd - (p[j] - (1.0 if j == c else 0.0))))
assert worst < 1e-6, worst
print(f"softmax cross-entropy gradient vs finite differences: max abs error {worst:.2e}")

# --- Worked gradient step on the branching token (also the animation's first phase) ---------------
TOKENS = ["factor", "formula", "expand", "divide", "square"]  # target first, then the valid alternative
P0 = [0.30, 0.30, 0.20, 0.12, 0.08]
C = 0
grad = [P0[j] - (1.0 if j == C else 0.0) for j in range(5)]
quote("grad target", grad[0], "{:.2f}", "-0.70")
quote("grad valid alternative", grad[1], "{:.2f}", "0.30")
quote("grad expand", grad[2], "{:.2f}", "0.20")
quote("grad divide", grad[3], "{:.2f}", "0.12")
quote("grad square", grad[4], "{:.2f}", "0.08")
quote("sum of gradient", abs(sum(grad)), "{:.0f}", "0")


def steps(n: int, eta: float = 1.0) -> list[list[float]]:
    z = [math.log(v) for v in P0]
    out = [softmax(z)]
    for _ in range(n):
        p = softmax(z)
        z = [z[j] - eta * (p[j] - (1.0 if j == C else 0.0)) for j in range(5)]
        out.append(softmax(z))
    return out


ST = steps(4)
quote("after 1 step: target", ST[1][0], "{:.2f}", "0.52")
quote("after 1 step: valid alternative", ST[1][1], "{:.2f}", "0.19")
quote("after 1 step: expand", ST[1][2], "{:.2f}", "0.14")
quote("after 1 step: divide", ST[1][3], "{:.2f}", "0.09")
quote("after 1 step: square", ST[1][4], "{:.2f}", "0.06")
quote("after 1 step: valid alternative drop", P0[1] - ST[1][1], "{:.2f}", "0.11")
quote("after 1 step: expand drop", P0[2] - ST[1][2], "{:.2f}", "0.06")
quote("after 4 steps: target", ST[4][0], "{:.2f}", "0.81")
quote("after 4 steps: valid alternative", ST[4][1], "{:.2f}", "0.07")
for s in range(5):  # every value the animation's bars print
    for j, name in enumerate(TOKENS):
        anim(f"anim step {s} {name}", ST[s][j], "{:.2f}", f"{ST[s][j]:.2f}")
assert all(abs(sum(p) - 1) < 1e-12 for p in ST)
# every competitor falls at every step, the valid alternative fastest in absolute terms on step 1
for s in range(4):
    assert ST[s + 1][0] > ST[s][0]
    assert all(ST[s + 1][j] < ST[s][j] for j in range(1, 5))
assert max(range(1, 5), key=lambda j: P0[j] - ST[1][j]) == 1

# --- Compounding along a trajectory ---------------------------------------------------------------
BEFORE, AFTER = [0.5, 0.4, 0.5, 0.5], [0.8, 0.8, 0.9, 0.9]
pb, pa = math.prod(BEFORE), math.prod(AFTER)
quote("trajectory before", pb, "{:.2f}", "0.05")
quote("trajectory after", pa, "{:.4f}", "0.5184")
quote("trajectory ratio", pa / pb, "{:.1f}", "10.4")
quote("mean NLL before", -sum(map(math.log, BEFORE)) / 4, "{:.2f}", "0.75")
quote("mean NLL after", -sum(map(math.log, AFTER)) / 4, "{:.2f}", "0.16")
quote("largest local factor", max(a / b for a, b in zip(AFTER, BEFORE)), "{:.0f}", "2")
anim("anim trajectory before", pb, "{:.2f}", "0.05")
anim("anim trajectory after", pa, "{:.4f}", "0.5184")
anim("anim trajectory ratio", pa / pb, "{:.1f}", "10.4")

# --- Length, average NLL, perplexity --------------------------------------------------------------
quote("0.9^10", 0.9**10, "{:.3f}", "0.349")
quote("0.9^100 mantissa", 0.9**100 / 1e-5, "{:.2f}", "2.66")
quote("mean NLL at 0.9", -math.log(0.9), "{:.3f}", "0.105")
quote("perplexity at 0.9", math.exp(-math.log(0.9)), "{:.3f}", "1.111")

# Loss normalization: one short (100 tokens) and one long (1,000 tokens) response in a batch
TS, TL = 100, 1000
quote("long share of weight, sum", TL / (TS + TL), "{:.0%}", "91%")
quote("per-token weight short, per-sequence mean", 1 / (2 * TS), "{:.3f}", "0.005")
quote("per-token weight long, per-sequence mean", 1 / (2 * TL), "{:.4f}", "0.0005")
quote("short/long token weight ratio, per-sequence mean", (1 / (2 * TS)) / (1 / (2 * TL)), "{:.0f}", "10")
quote("token-mean weight", TS + TL, "1/{:,.0f}", "1/1,100")

# --- Exposure bias toy model (figures/part02_exposure.py) -----------------------------------------


def mistakes(T: int, eps: float, r: float) -> float:
    """Expected mistakes in T free-running steps. On track: a mistake with probability eps, which
    leaves the track. Off track: back on track with probability r (no mistake), else another mistake."""
    on, tot = 1.0, 0.0
    for _ in range(T):
        off = 1.0 - on
        tot += on * eps + off * (1.0 - r)
        on = on * (1.0 - eps) + off * r
    return tot


EPS = 1e-3
quote("teacher-forced mistakes, T=1000", 1000 * EPS, "{:.0f}", "1")
quote("teacher-forced mistakes, T=100", 100 * EPS, "{:.1f}", "0.1")
quote("no recovery, T=100", mistakes(100, EPS, 0.0), "{:.1f}", "4.9")
quote("T^2 eps / 2 at T=100", 100**2 * EPS / 2, "{:.0f}", "5")
quote("no recovery, T=1000", mistakes(1000, EPS, 0.0), "{:.0f}", "368")
quote("recovery 0.1, T=1000", mistakes(1000, EPS, 0.1), "{:.1f}", "9.8")
# the no-recovery case is the quadratic regime while T * eps is small
assert abs(mistakes(100, EPS, 0.0) / (100**2 * EPS / 2) - 1) < 0.03
# with recovery the count is linear in T: slope approx eps / (eps + r) * ... ~ eps (1 + (1 - r)/r)
slope = (mistakes(4000, EPS, 0.1) - mistakes(2000, EPS, 0.1)) / 2000
assert abs(slope / EPS - 10) < 0.2, slope / EPS  # about ten times the teacher-forced rate

# --- Cross-entropy decomposition ------------------------------------------------------------------
rng = random.Random(3)
worst = 0.0
for _ in range(1000):
    V = rng.randint(2, 20)
    q = [rng.random() ** 3 for _ in range(V)]
    q = [v / sum(q) for v in q]
    pi = [rng.random() + 1e-9 for _ in range(V)]
    pi = [v / sum(pi) for v in pi]
    cross = -sum(a * math.log(b) for a, b in zip(q, pi) if a > 0)
    kl = sum(a * math.log(a / b) for a, b in zip(q, pi) if a > 0)
    worst = max(worst, abs(cross - (entropy(q) + kl)))
    assert kl >= -1e-12
assert worst < 1e-12, worst
print(f"H(q, pi) = H(q) + KL(q || pi): max abs error {worst:.2e}")
# forward vs reverse KL when a minority mode of the data shrinks from 0.1 to 0.01
q, m = [0.9, 0.1], [0.99, 0.01]
fwd = sum(a * math.log(a / b) for a, b in zip(q, m))
rev = sum(b * math.log(b / a) for a, b in zip(q, m))
quote("forward KL, minority mode shrinks", fwd, "{:.3f}", "0.144")
quote("reverse KL, minority mode shrinks", rev, "{:.3f}", "0.071")
# mass on an undemonstrated strategy C only lowers likelihood: optimum puts 0 there
for pc in (0.0, 0.1, 0.3):
    loss = -(0.9 * math.log(0.9 * (1 - pc)) + 0.1 * math.log(0.1 * (1 - pc)))
    if pc == 0.0:
        base_loss = loss
    else:
        assert loss > base_loss
quote("loss increase when 0.3 moves to an undemonstrated strategy", -math.log(0.7), "{:.3f}", "0.357")

# --- Success objective: grad log p = E_q[grad log pi] and dJ/dz_tau = pi (R - J) --------------------
# A small autoregressive model: 3 tokens per step, 3 steps, one free logit vector per prefix.
rng = random.Random(4)
VOC, LEN = 3, 3
prefixes = [()]
for t_ in range(1, LEN):
    prefixes += [p_ for p_ in __import__("itertools").product(range(VOC), repeat=t_)]
theta = {pre: [rng.gauss(0, 1) for _ in range(VOC)] for pre in prefixes}
seqs = list(__import__("itertools").product(range(VOC), repeat=LEN))
correct = {sq for sq in seqs if (sum(sq) % 3 == 0)}  # an arbitrary verifier


def seq_logprob(th: dict, sq: tuple) -> float:
    return sum(math.log(softmax(th[sq[:t_]])[sq[t_]]) for t_ in range(LEN))


def log_success(th: dict) -> float:
    return math.log(sum(math.exp(seq_logprob(th, sq)) for sq in correct))


def grad_fd(f, th: dict) -> dict:
    g = {}
    for pre, vec in th.items():
        g[pre] = []
        for j in range(VOC):
            hi = {k: v[:] for k, v in th.items()}
            lo = {k: v[:] for k, v in th.items()}
            hi[pre][j] += 1e-6
            lo[pre][j] -= 1e-6
            g[pre].append((f(hi) - f(lo)) / 2e-6)
    return g


lhs = grad_fd(log_success, theta)
psucc = math.exp(log_success(theta))
rhs = {pre: [0.0] * VOC for pre in prefixes}
for sq in correct:
    w = math.exp(seq_logprob(theta, sq)) / psucc  # q(tau | x)
    g = grad_fd(lambda th: seq_logprob(th, sq), theta)
    for pre in prefixes:
        for j in range(VOC):
            rhs[pre][j] += w * g[pre][j]
worst = max(abs(lhs[pre][j] - rhs[pre][j]) for pre in prefixes for j in range(VOC))
assert worst < 1e-6, worst
print(f"grad log p = E_q[grad log pi] on a 3-step model: max abs error {worst:.2e}")
# softmax over whole trajectories: dJ/dz_tau = pi(tau) (R(tau) - J), positive on every correct tau
z = [rng.gauss(0, 1) for _ in range(12)]
R = [1.0 if i % 3 == 0 else 0.0 for i in range(12)]
J = lambda zz: sum(a * b for a, b in zip(softmax(zz), R))
pz, Jz = softmax(z), J(z)
for i in range(12):
    zp, zm = z[:], z[:]
    zp[i] += 1e-6
    zm[i] -= 1e-6
    fd = (J(zp) - J(zm)) / 2e-6
    assert abs(fd - pz[i] * (R[i] - Jz)) < 1e-8
    assert (fd > 0) == (R[i] == 1.0)

# --- Diversity (also the animation's last phase) ---------------------------------------------------
quote("A+B before", 0.20 + 0.20, "{:.2f}", "0.40")
quote("A+B after", 0.70 + 0.05, "{:.2f}", "0.75")
quote("N_eff before", n_eff([0.20, 0.20]), "{:.0f}", "2")
quote("N_eff after", n_eff([0.70, 0.05]), "{:.2f}", "1.28")
quote("A share of correct mass after", 0.70 / 0.75, "{:.0%}", "93%")
anim("anim correct mass before", 0.40, "{:.2f}", "0.40")
anim("anim correct mass after", 0.75, "{:.2f}", "0.75")
anim("anim N_eff before", n_eff([0.20, 0.20]), "{:.2f}", "2.00")
anim("anim N_eff after", n_eff([0.70, 0.05]), "{:.2f}", "1.28")

# --- Lower loss, worse decision ---------------------------------------------------------------------
easy_b, easy_a, dec_b, dec_a = 0.95, 0.97, 0.6, 0.4
nll_b = (99 * -math.log(easy_b) - math.log(dec_b)) / 100
nll_a = (99 * -math.log(easy_a) - math.log(dec_a)) / 100
quote("avg NLL before", nll_b, "{:.4f}", "0.0559")
quote("avg NLL after", nll_a, "{:.4f}", "0.0393")
quote("relative loss drop", (nll_b - nll_a) / nll_b, "{:.0%}", "30%")
quote("demo trajectory prob before", easy_b**99 * dec_b, "{:.4f}", "0.0037")
quote("demo trajectory prob after", easy_a**99 * dec_a, "{:.4f}", "0.0196")

# --- Check yourself -------------------------------------------------------------------------------
quote("Q1 correct mass before", 0.20 + 0.10 + 0.05, "{:.2f}", "0.35")
quote("Q1 correct mass after", 0.70 + 0.03 + 0.01, "{:.2f}", "0.74")
quote("Q1 N_eff before", n_eff([0.20, 0.10, 0.05]), "{:.2f}", "2.60")
quote("Q1 N_eff after", n_eff([0.70, 0.03, 0.01]), "{:.2f}", "1.27")
quote("Q1 A share after", 0.70 / 0.74, "{:.0%}", "95%")
quote("Q3 amplification", math.log10(1e-2 / 1e-8), "10^{:.0f}", "10^6")
quote("Q3 pass@1024 at p=1e-8", (1 - (1 - 1e-8) ** 1024) / 1e-5, "{:.2f}", "1.02")
quote("Q3 pass@64 at p=0.01 (%)", 100 * (1 - 0.99**64), "{:.0f}", "47")

# --- In my experiments: re-read from the committed reports -----------------------------------------


def fetch(path: str) -> bytes:
    with urllib.request.urlopen(SOE_RAW + path, timeout=60) as f:
        return f.read()


def rj(path: str) -> dict:
    return json.loads(fetch(path))


# which checkpoint was evaluated: the frozen SFT export is run #2's final step
frozen = rj("docs/reports/m2_sft_frozen.json")
assert frozen["frozen"].endswith("frozen/sft_qwen35-4b_v1/hf")
assert frozen["run"]["run_id"] == "20261109-185607-sft_m2_run2_qwen35-4b-s1-j3237928"
assert frozen["run"]["data"]["train_file"].endswith("train_run2.jsonl")
assert frozen["soe_export"]["step"] == 719 and frozen["soe_export"]["training"] == "full"
evalrep = rj("docs/reports/sft_qwen35-4b_v1_soe-eval-v4_report.json")
assert evalrep["plan"]["checkpoint"] == frozen["frozen"]
compare = rj("docs/reports/sft_qwen35-4b_v1_compare_base_v4.json")
trans = rj("docs/reports/sft_qwen35-4b_v1_transitions_base_v4.json")
assert compare["b"].endswith("sft_qwen35-4b_v1_soe-eval-v4") and trans["after"].endswith("sft_qwen35-4b_v1_soe-eval-v4")
diag = rj("docs/reports/m2_sft_diagnostic_run2.json")
assert diag["rl_val"]["sft_checkpoint"] == frozen["source"] and diag["sft_val"]["sft_checkpoint"] == frozen["source"]
check2 = rj("docs/reports/m2_sft_run2_check_run2.json")
assert check2["sft"]["checkpoint"] == frozen["source"]
print("evaluated checkpoint = SFT run #2, step 719 (compare, transitions and diagnostic reports agree)")

# the data: the base model's own verified solutions
data = rj("docs/reports/m2_sft_data.json")
quote("problems drawn", data["problems_drawn"], "{:,}", "16,000")
quote("samples per problem", data["samples_total"] / data["problems_drawn"], "{:.0f}", "4")
kept = data["kept_per_problem_counts"]
assert max(int(k) for k in kept) == 2
quote("problems with no verified solution", kept["0"], "{:,}", "2,224")
assert data["drops"]["problems_without_a_kept_trajectory"] == kept["0"]
# the validation loss and the 500-problem diagnostic use the same held-out problems
assert len(set(data["val_problem_ids"])) == data["files"]["val"]["problems"] == 500
quote("examples, first run", data["files"]["train"]["examples"], "{:,}", "25,551")
run2 = rj("docs/reports/m2_sft_run2_data.json")
quote("examples, second run", run2["examples"]["kept"], "{:,}", "22,996")
assert run2["percentile"] == 90.0
d029 = fetch("docs/decisions/029-sft-recipe-and-run2-rule.md").decode()
d030 = fetch("docs/decisions/030-frozen-sft-export.md").decode()
for s in ("SFT on the base model's own correct traces", "rejection-sampling fine-tuning", "learning rate 1e-5",
          "One epoch", "Liger loss on completion tokens only", "learning rate 1e-5 / 3 (3.33e-6)"):
    assert s in d029.replace("\n", " ").replace("  ", " ") or s in d029, f"decision 029 text changed: {s!r}"
for s in ("sampled at the suite's temperature 0.6", "Likely cause, untested", "a 3× lower lr halved the effect",
          "The frozen SFT checkpoint is run #2's final export"):
    assert s in d030.replace("\n", " ").replace("   ", " "), f"decision 030 text changed: {s!r}"

# validation loss during training (held-out verified solutions)
vl = check2["criteria"]["1"]["validation_loss"]
quote("val loss step 290", vl["290"], "{:.4f}", "0.2020")
quote("val loss step 580", vl["580"], "{:.4f}", "0.1964")
quote("val loss step 719", vl["719"], "{:.4f}", "0.1959")

# RL validation split, 256 problems x 16 samples
rv = diag["rl_val"]
b, s = rv["base"], rv["sft"]
assert b["problems"] == 256 and b["samples_per_problem"] == 16
quote("rl-val pass@1 base", b["pass@1"], "{:.3f}", "0.660")
quote("rl-val pass@1 sft", s["pass@1"], "{:.3f}", "0.637")
quote("rl-val pass@1 drop", -rv["paired"]["pass@1"]["sft_minus_base"], "{:.3f}", "0.023")
quote("rl-val pass@1 drop SE", rv["paired"]["pass@1"]["se"], "{:.3f}", "0.008")
quote("rl-val pass@16 base", b["pass@16"], "{:.3f}", "0.852")
quote("rl-val pass@16 sft", s["pass@16"], "{:.3f}", "0.855")
quote("rl-val pass@16 SE", rv["paired"]["pass@n"]["se"], "{:.3f}", "0.012")
quote("rl-val truncation base", b["truncation_rate"], "{:.1%}", "32.1%")
quote("rl-val truncation sft", s["truncation_rate"], "{:.1%}", "34.5%")
quote("rl-val loops base", b["loop_rate"], "{:.1%}", "5.1%")
quote("rl-val loops sft", s["loop_rate"], "{:.1%}", "10.4%")
quote("rl-val accuracy when stopped base", b["accuracy_when_stopped"], "{:.1%}", "97.2%")
quote("rl-val accuracy when stopped sft", s["accuracy_when_stopped"], "{:.1%}", "97.2%")
quote("rl-val median finished length base", b["median_stopped_tokens"], "{:,.0f}", "3,884")
quote("rl-val median finished length sft", s["median_stopped_tokens"], "{:,.0f}", "3,348")
dec = rv["pass@1_change"]
# pass@1 = stop rate x accuracy when stopped; the split is exact
assert abs(b["stop_rate"] * b["accuracy_when_stopped"] - b["pass@1"]) < 1e-12
assert abs(s["stop_rate"] * s["accuracy_when_stopped"] - s["pass@1"]) < 1e-12
stopping = (s["stop_rate"] - b["stop_rate"]) * b["accuracy_when_stopped"]
accuracy = s["stop_rate"] * (s["accuracy_when_stopped"] - b["accuracy_when_stopped"])
assert abs(stopping - dec["stopping"]) < 1e-12 and abs(accuracy - dec["accuracy"]) < 1e-12
assert abs(stopping + accuracy - dec["total"]) < 1e-12
quote("rl-val stopping part", stopping, "signed4", signed(stopping, 4))
quote("rl-val accuracy part", accuracy, "signed4", signed(accuracy, 4))
assert signed(stopping, 4) == "\u22120.0230" and signed(accuracy, 4) == "+0.0003"
quote("loops among truncated base", b["loops_among_truncated"], "{:.0%}", "16%")
quote("loops among truncated sft", s["loops_among_truncated"], "{:.0%}", "30%")
diag_src = fetch("scripts/sft_diagnostic.py").decode()
assert "TAIL_CHARS, LOOP_RATIO = 4000, 0.15" in diag_src
quote("loop heuristic tail", 4000, "{:,} characters", "4,000 characters")
quote("loop heuristic ratio", 0.15, "less than {:.0%}", "less than 15%")
assert "base pass@1 − SFT pass@1 > 2 SE" in d029
check1 = rj("docs/reports/m2_sft_run2_check.json")
assert check1["triggered"] == [3, 4] and check2["triggered"] == [3, 4]  # both runs failed the same two checks

# held-out problems of the SFT validation set, 500 x 4
sv = diag["sft_val"]
assert sv["base"]["problems"] == 500 and sv["base"]["samples_per_problem"] == 4
quote("sft-val pass@1 base", sv["base"]["pass@1"], "{:.3f}", "0.868")
quote("sft-val pass@1 sft", sv["sft"]["pass@1"], "{:.3f}", "0.849")
quote("sft-val truncation base", sv["base"]["truncation_rate"], "{:.1%}", "7.8%")
quote("sft-val truncation sft", sv["sft"]["truncation_rate"], "{:.1%}", "10.4%")
quote("sft-val stopping part", sv["pass@1_change"]["stopping"], "signed3", signed(sv["pass@1_change"]["stopping"], 3))
quote("sft-val accuracy part", sv["pass@1_change"]["accuracy"], "signed3", signed(sv["pass@1_change"]["accuracy"], 3))
assert signed(sv["pass@1_change"]["stopping"], 3) == "\u22120.025" and signed(sv["pass@1_change"]["accuracy"], 3) == "+0.005"

# first run (higher learning rate), for the learning-rate remark
r1 = rj("docs/reports/m2_sft_diagnostic.json")["rl_val"]
quote("run 1 pass@1 drop", -r1["paired"]["pass@1"]["sft_minus_base"], "{:.3f}", "0.052")
quote("run 1 loops", r1["sft"]["loop_rate"], "{:.1%}", "11.2%")

# full evaluation suite
A, B = compare["level A"], compare["level B"]
quote("level A problems", A["pass@1"]["problems"], "{:,}", "1,224")
quote("level B problems", B["pass@1"]["problems"], "{:,}", "982")
quote("level A pass@1 base", A["pass@1"]["a"], "{:.3f}", "0.744")
quote("level A pass@1 sft", A["pass@1"]["b"], "{:.3f}", "0.733")
quote("level B pass@1 base", B["pass@1"]["a"], "{:.3f}", "0.673")
quote("level B pass@1 sft", B["pass@1"]["b"], "{:.3f}", "0.650")
quote("level A pass@16 base", A["pass@16"]["a"], "{:.3f}", "0.856")
quote("level A pass@16 sft", A["pass@16"]["b"], "{:.3f}", "0.853")
quote("level B pass@16 base", B["pass@16"]["a"], "{:.3f}", "0.873")
quote("level B pass@16 sft", B["pass@16"]["b"], "{:.3f}", "0.877")
for lev in (A, B):
    assert abs(lev["pass@16"]["diff"]) < 1.0 * lev["pass@16"]["se"]  # within one SE
    assert lev["pass@1"]["diff"] < 0 and abs(lev["pass@1"]["diff"]) > 2 * lev["pass@1"]["se"]
    assert lev["truncation_rate"]["diff"] > 0
TA, TB, TH = trans["main"]["A"]["16"], trans["main"]["B"]["16"], trans["hard"]["64"]
quote("A k=16 lost", TA["solved_to_lost"], "{:.0f}", "19")
quote("A k=16 gained", TA["unsolved_to_solved"], "{:.0f}", "15")
quote("B k=16 lost", TB["solved_to_lost"], "{:.0f}", "20")
quote("B k=16 gained", TB["unsolved_to_solved"], "{:.0f}", "24")
quote("hard k=64 lost", TH["solved_to_lost"], "{:.0f}", "16")
quote("hard k=64 gained", TH["unsolved_to_solved"], "{:.0f}", "15")
quote("hard problems", trans["hard"]["problems"], "{:.0f}", "233")
assert trans["threshold"] == 0.5

# --- Compare with the text -----------------------------------------------------------------------
post = POST.read_text() if POST.exists() else None
bad = []
for label, value, fmt, text in QUOTES + ANIM:
    if fmt.startswith("signed"):
        got = signed(value, int(fmt[-1]))
    else:
        got = fmt.format(value)
    want = text
    status = "ok" if got == want else "MISMATCH"
    if got != want:
        bad.append(label)
    in_post_required = not label.startswith("anim ")
    if post is not None and in_post_required and want not in post:
        status += " (not in post)"
        bad.append(label + " [not in post]")
    print(f"{status:<18} {label:<58} computed {got:<12} text {want}")

if bad:
    print("\nFAILED:", *bad, sep="\n  ")
    sys.exit(1)
print(f"\nall {len(QUOTES)} quoted values and {len(ANIM)} animation values match"
      + ("" if post is not None else " (post not written yet)"))
