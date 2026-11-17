# LLM post-training: study notes

Companion repository for the blog series **[Sharpen or Expand? A Study Series on LLM Post-Training](https://hieutrungle.github.io/blog/2026/11/llm-post-training-series/)**.

The series builds LLM post-training from first principles (supervised fine-tuning, DPO, policy gradients, PPO, GRPO and its modern variants) around one question: does reinforcement learning with verifiable rewards **sharpen** what a model can already do, or **expand** what it can solve at all? The theory is tested in a real experiment, [sharpen-or-expand](https://github.com/hieutrungle/sharpen-or-expand).

This repository holds everything behind the posts:

- the post sources (blog-ready Markdown),
- the raw study notes they were written from,
- scripts that recompute every number a post quotes, and every figure,
- the references, with arXiv/DOI links checked.

## Parts

| Part | Topic | Post | Checks |
|---|---|---|---|
| 0 | Series overview, roadmap, notation | [blog](https://hieutrungle.github.io/blog/2026/11/llm-post-training-series/) · [source](posts/part-00-series-overview.md) | figure script only |
| 1 | What does "capability" mean? pass@k, its unbiased estimator, zero-success bounds | [blog](https://hieutrungle.github.io/blog/2026/11/capability-and-pass-at-k/) · [source](posts/part-01-capability-and-pass-at-k.md) | [part01_checks.py](checks/part01_checks.py) |
| 2 | Supervised fine-tuning as probability reshaping: the cross-entropy gradient, compounding, exposure bias, forward KL | [blog](https://hieutrungle.github.io/blog/2026/11/sft-probability-reshaping/) · [source](posts/part-02-sft-probability-reshaping.md) | [part02_checks.py](checks/part02_checks.py) |
| 3 | DPO from KL-regularized RL: Bradley–Terry, the Gibbs optimum, the binary-reward tilt, the DPO loss and gradient, likelihood displacement | [blog](https://hieutrungle.github.io/blog/2026/11/dpo-from-kl-regularized-rl/) · [source](posts/part-03-dpo-from-kl-regularized-rl.md) | [part03_checks.py](checks/part03_checks.py) |
| 4 | Policy gradients for LLMs: REINFORCE to PPO: the log-derivative trick, baselines and RLOO, importance ratios, PPO clipping, KL placement, entropy, sparse rewards | [blog](https://hieutrungle.github.io/blog/2026/11/policy-gradients-reinforce-to-ppo/) · [source](posts/part-04-policy-gradients-reinforce-to-ppo.md) | [part04_checks.py](checks/part04_checks.py) |
| 5 | GRPO from first principles | planned | — |
| 6 | Why vanilla GRPO fails | planned | — |
| 7–12 | Modern GRPO family, reward theory, evaluation statistics, ablations, systems, synthesis | after the corresponding study block | — |

The series page on the blog lists every part: https://hieutrungle.github.io/blog/series/llm-post-training/

## Layout

```
posts/      blog-ready post sources (Jekyll front matter, kramdown + MathJax)
notes/raw/  study notes as written during each study session (unedited; corrections live in the posts and ERRATA.md)
checks/     one script per part that recomputes the numbers quoted in the post
figures/    one script per figure; outputs go to assets/
assets/     rendered figures, one folder per part
references.bib
ERRATA.md
```

## Reproduce

```bash
python -m pip install -r requirements.txt
python figures/part00_sharpen_vs_expand.py
python checks/part01_checks.py   # recomputes every number in Part 1 (reads one report from sharpen-or-expand)
python figures/part01_breadth.py
python figures/part01_zero_bound.py
python checks/part02_checks.py   # recomputes every number in Part 2 (reads its reports from sharpen-or-expand)
python figures/part02_exposure.py
python checks/part03_checks.py   # recomputes every number in Part 3 and checks the derivations numerically
python figures/part03_tilt.py
python checks/part04_checks.py   # recomputes every number in Part 4, including the simulation behind its animation
python figures/part04_clip.py
python figures/part04_baseline.py
```

## Writing conventions

- Math is written for kramdown + MathJax: `$$…$$` for both inline and display math, `\mid` for conditioning bars and `\Vert` for norm bars, so a `|` never breaks a Markdown table, and `\lt` / `\gt` instead of `<` / `>`, which kramdown passes through raw into the HTML.
- Notation is fixed once in Part 0. In particular β (DPO/RLHF KL strength), λ_KL (KL coefficient in a GRPO-style loss) and β_RS (RS-GRPO risk sensitivity) are kept apart.
- Experimental results are quoted only from reports committed to the public `main` branch of sharpen-or-expand, and each number is re-read from its report file before it is quoted.

## License

Text and figures: [CC BY 4.0](LICENSE). Code: [MIT](LICENSE-CODE).
