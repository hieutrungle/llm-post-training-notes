You’re close — the roadmap we originally established has **12 blocks**, although Blocks 11–12 are somewhat different from the core algorithm sequence. We have completed **Blocks 1–6** so far.

| Block | Study session | Main content | Status |
|---|---|---|---|
| **1** | **What does “capability” mean?** | Sharpening vs expansion, trajectory distributions, pass@1/pass@k, effective accessibility, reasoning diversity, problem-level transitions | ✅ Done |
| **2** | **LLM probability & SFT** | Autoregressive likelihood, cross-entropy, SFT gradients, teacher forcing, forward KL, trajectory vs token objectives, SFT as probability reshaping | ✅ Done |
| **3** | **Preference optimization & DPO** | Bradley–Terry model, KL-regularized RL, DPO derivation, reference policy, \(\beta\), implicit reward, ratio-of-ratios interpretation | ✅ Done |
| **4** | **Policy-gradient foundations** | Expected return, REINFORCE, log-derivative trick, baselines, advantages, importance sampling, PPO clipping, KL, entropy | ✅ Done |
| **5** | **GRPO from first principles** | Critic-free group baseline, group-relative advantages, reward normalization, binary RLVR, group size, mixed groups, GRPO vs PPO/DPO | ✅ Done |
| **6** | **Why vanilla GRPO fails** | Zero-variance groups, reward-normalization bias, length bias, clipping, entropy collapse, credit assignment, exploration starvation | ✅ Done |
| **7** | **Modern GRPO family** | **Dr.GRPO → DAPO → RS-GRPO**, then GSPO/GMPO/GFPO only when useful; derive each method from the pathology it addresses | ⏭ Next |
| **8** | **RLVR & reward theory** | Verifiable rewards, sparse outcome reward, reward hacking, verifier errors, credit assignment, exploration, outcome vs process supervision | Upcoming |
| **9** | **Evaluation & research statistics** | pass@k estimation, bootstrap CIs, paired tests, effect sizes, multiple seeds, uncertainty, contamination, transition matrices | Upcoming |
| **10** | **Ablation & experimental design** | Matched rollout-token budgets, causal controls, one-variable-at-a-time ablations, stopping rules, compute efficiency | Upcoming |
| **11** | **Training dynamics & systems theory** | LoRA/full FT, BF16, FSDP, rollout generation, vLLM, KV cache, policy staleness, throughput and how systems choices affect experiments | Upcoming |
| **12** | **Research synthesis** | Interpret final results, distinguish sharpening/expansion, alternative explanations, cross-model validation, paper/interview-level research story | Upcoming |

### The main intellectual progression

You can think of the first ten theoretical blocks as:

\[
\boxed{
\begin{aligned}
&\textbf{1. What are we measuring?}\\
&\downarrow\\
&\textbf{2. SFT: imitate demonstrations}\\
&\downarrow\\
&\textbf{3. DPO: prefer one trajectory over another}\\
&\downarrow\\
&\textbf{4. RL: sample and reinforce}\\
&\downarrow\\
&\textbf{5. GRPO: relative reinforcement without a critic}\\
&\downarrow\\
&\textbf{6. Why GRPO breaks}\\
&\downarrow\\
&\textbf{7. How modern algorithms try to fix it}\\
&\downarrow\\
&\textbf{8. What does the reward actually teach?}\\
&\downarrow\\
&\textbf{9. How do we know an observed gain is real?}\\
&\downarrow\\
&\textbf{10. How do we causally determine why it happened?}
\end{aligned}
}
\]

Then Blocks **11–12** turn that understanding into systems awareness and research synthesis.

For `sharpen-or-expand`, **Blocks 7–10 are probably the most important remaining research blocks**. Block 7 in particular is where we stop learning algorithms as names and start deriving:

\[
\text{GRPO pathology}
\rightarrow
\text{Dr.GRPO/DAPO/RS-GRPO modification}
\rightarrow
\text{predicted measurable effect}.
\]

So our **next session is Block 7: Modern GRPO Family — Dr.GRPO, DAPO, and RS-GRPO from their failure modes**.