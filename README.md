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
| 2 | Supervised fine-tuning as probability reshaping | planned | — |
| 3 | DPO from KL-regularized RL | planned | — |
| 4 | Policy gradients: REINFORCE to PPO | planned | — |
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
```

## Writing conventions

- Math is written for kramdown + MathJax: `$$…$$` for both inline and display math, `\mid` for conditioning bars and `\Vert` for norm bars, so a `|` never breaks a Markdown table, and `\lt` / `\gt` instead of `<` / `>`, which kramdown passes through raw into the HTML.
- Notation is fixed once in Part 0. In particular β (DPO/RLHF KL strength), λ_KL (KL coefficient in a GRPO-style loss) and β_RS (RS-GRPO risk sensitivity) are kept apart.
- Experimental results are quoted only from reports committed to the public `main` branch of sharpen-or-expand, and each number is re-read from its report file before it is quoted.

## License

Text and figures: [CC BY 4.0](LICENSE). Code: [MIT](LICENSE-CODE).
