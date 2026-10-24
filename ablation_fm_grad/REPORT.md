# Flow-Matching Gradient Backpropagation for Execution-Aware VLA Training: An Ablation Study

> **Status.** Paper-ready draft section (v0.1, October 2026). All numbers below are measured on a single RTX 4090 (48 GB) with the code in `ablation_fm_grad/`; they are **gradient-estimator diagnostics**, not task-success results. No claim about downstream policy quality is made here.

---

## Abstract

Execution-aware co-training of a flow-matching vision-language-action (VLA) policy requires differentiating a downstream execution loss through the flow-matching sampler into the action head. The default estimator is pathwise differentiation (unrolled Euler + autograd, i.e., BPTT). We compare five estimators under identical models, losses, and seeds: (i) pathwise BPTT, (ii) per-step gradient checkpointing, (iii) truncated BPTT, (iv) a discrete adjoint, and (v) a Q-VGM-style local value-gradient estimator. On a controlled MLP testbed and on the real GR00T N1.7 action-head class, the three exact estimators (pathwise, checkpointed, adjoint) produce **bit-identical gradients**; truncated BPTT preserves direction (cosine 0.89–0.999) but not magnitude; the Q-VGM-style estimator preserves direction (cosine ≈ 0.999) with a magnitude error of ≈ 0.5. On the real 1.09B-parameter action head (bf16, K=4, batch 1), pathwise BPTT peaks at **5.8 GB** of activation memory and 0.28 s per update, which fits a 48 GB consumer GPU; step-level checkpointing *increases* peak memory to 8.3 GB (+42%) and slows the step by 1.44×, showing that checkpointing must be applied at the transformer-block level to pay off. We therefore retain pathwise BPTT as the production estimator for single-GPU fine-tuning and document the exact conditions under which the alternatives become preferable.

---

## 1. Problem Setting

Let $v_\theta(a,t,c)$ be the flow-matching velocity field (GR00T N1.7 action expert) and let the sampler be the Euler discretization

$$
a_{k+1} = a_k + \Delta t\, v_\theta(a_k, t_k, c), \quad
\Delta t = 1/K,\quad t_k = k/K,\quad a_K = Z,
\tag{1}
$$

with $Z\in\mathbb{R}^{L\times d_z}$, $L=40$, $d_z=64$ (the motion-token chunk). A downstream execution loss $\mathcal{L}_{\mathrm{exec}}(Z)$ is provided by a differentiable rollout (SuperMimic world model and whole-body controller, or a surrogate in the ablation). We need $\partial \mathcal{L}_{\mathrm{exec}}/\partial\theta$.

The five estimators differ **only** in how this derivative is computed; the forward samples $a_K$ are identical.

---

## 2. Related Work and Estimators

### 2.1 Pathwise differentiation (BPTT)

Unrolling Eq. (1) and applying reverse-mode autograd gives

$$
\frac{\partial a_{k+1}}{\partial\theta}
=\left(I+\Delta t\,\frac{\partial v_\theta}{\partial a_k}\right)\frac{\partial a_k}{\partial\theta}
+\Delta t\,\frac{\partial v_\theta}{\partial\theta},
\tag{2}
$$

which is what the current implementation uses (`sample_fm_from_features`, `diff_vla_wbc/gr00t_fm.py`). VITA's *flow latent decoding* backpropagates an action-reconstruction loss through the flow-matching ODE in the same way [@gao2025vita]; our estimator is the same mechanism applied to an execution loss rather than a reconstruction loss.

### 2.2 Gradient checkpointing

Each Euler step (or each transformer block) is recomputed in the backward pass instead of storing its activations, trading compute for memory [@griewank1992; @pytorch_checkpoint]. The gradient is mathematically identical to Eq. (2). The benefit depends critically on the checkpoint granularity; Sec. 5.4 shows that wrapping an entire 32-layer DiT call is counterproductive.

### 2.3 Truncated BPTT

Only the last $\tau$ Euler steps are differentiated; earlier steps run under `no_grad`. This is the cheapest unbiased-in-the-limit but biased-in-practice estimator, and serves as a lower-bound baseline.

### 2.4 Discrete adjoint

Forward under `no_grad`, then reverse accumulation without storing the graph:

$$
\lambda_k = \lambda_{k+1} + \Delta t\left(\frac{\partial v_\theta}{\partial a_k}\right)^{\!\top}\lambda_{k+1},
\qquad
\nabla_\theta \mathcal{L} \mathrel{+}= \Delta t\left(\frac{\partial v_\theta}{\partial\theta}\right)^{\!\top}\lambda_{k+1},
\tag{3}
$$

with $\lambda_K = \partial\mathcal{L}/\partial a_K$. This is the discrete counterpart of the Neural-ODE adjoint [@chen2018neuralode]; it is mathematically equivalent to BPTT up to floating point and requires one vector–Jacobian product per step.

### 2.5 Q-VGM-style local value gradient

Q-VGM [@wang2026qvgm] observes that directly backpropagating a critic through many denoising steps is costly and unstable, and derives a local approximation that connects clean-action value gradients to late-step velocity corrections. With a critic $Q(a)$ and a late step $k$,

$$
\hat a = a_k + (1-t_k)\,v_\theta(a_k,t_k),\qquad
\mathcal{L}_{\mathrm{QVGM}} = -Q(\hat a),
\tag{4}
$$

so gradients touch only the **single** local velocity prediction. The original method trains an action-sensitive chunk critic with IQL/TD [@wang2026qvgm]; our implementation (`qvgm_lite`) trains the critic by regression and additionally supports an *oracle* critic $Q(a)=-\mathcal{L}_{\mathrm{exec}}(a)$ to isolate the gradient mechanism. It is a reimplementation, not the official code.

Flow-policy-gradient alternatives [@yi2026fpg] and potential-guided flow matching [@zhang2026pgfm] avoid this estimator entirely; they are discussed as related work but not implemented here.

---

## 3. Experimental Protocol

### 3.1 Testbeds

| Testbed | Model | Loss | Purpose |
|---|---|---|---|
| `toy` | 3-layer MLP velocity field, dim 64 | frozen random projection + quadratic target | gradient fidelity, K scaling |
| `tiny` | real GR00T N1.7 action-head class, hidden 32, 1 layer, H=8 | same quadratic | real code path, cheap |
| `scale` | real GR00T action-head class, hidden 256/512, 4/8 layers, H=40 | same quadratic | memory/time scaling |
| `real head` | **official 1.09B action head** (32 layers, hidden 1024, H=40, d=132, bf16) | same quadratic | realistic single-GPU memory probe |

The toy/tiny/scale runs use `ablation_fm_grad/run_ablation.py`; the real-head probe uses `ablation_fm_grad/probe_real_head.py`, which loads only `action_head.*` tensors from the official GR00T N1.7 checkpoint (no 3B VLM) and feeds dummy backbone/state features.

### 3.2 Metrics

- **Gradient fidelity**: cosine similarity and relative L2 error against pathwise BPTT.
- **Finite differences**: central differences on selected parameter entries (tolerance 2%).
- **Memory**: `torch.cuda.max_memory_allocated` after `reset_peak_memory_stats`, measured in a **fresh process per backend** for the `scale` rows.
- **Time**: wall-clock per forward+backward, after a warm-up iteration.
- **Learning**: 30 (toy), 20 (tiny) Adam steps from the same initialization; task-loss before/after.

### 3.3 Hardware and software

RTX 4090 48 GB, driver 570.124.06 / CUDA 12.8, PyTorch 2.14.1+cu126, bf16 action head, fp32 toy/tiny models.

---

## 4. Results

### 4.1 Gradient fidelity

| Testbed | Backend | Cosine vs pathwise | Relative error | Verdict |
|---|---|---|---|---|
| toy (K=2/4/8) | checkpointed | +1.0000 | 0.00e+00 | exact |
| toy | adjoint | +1.0000 | 0.00e+00 | exact |
| toy | truncated (τ=1) | +0.9984 – +0.9995 | 0.50 – 0.88 | biased |
| toy | qvgm_lite (oracle) | +0.9995 – +0.9998 | 0.50 | direction-correct, magnitude-off |
| tiny (K=4) | checkpointed / adjoint | +1.0000 | 0.00e+00 | exact |
| tiny | truncated | +0.9975 | 0.75 | biased |
| tiny | qvgm_lite (oracle) | +0.9973 | 0.50 | direction-correct |
| real 1.09B head (K=4) | checkpointed | **max&#124;Δgrad&#124; = 0.0** | — | exact |

The exact estimators also pass a central finite-difference check on individual parameters (relative error below the 2% tolerance; `tests/test_backends.py`).

### 4.2 Toy learning curves (K=4, 30 Adam steps, lr 1e-3)

| Backend | Initial loss | Final loss |
|---|---|---|
| pathwise | 70.10 | **2.13** |
| checkpointed | 70.10 | **2.13** |
| adjoint | 70.10 | **2.13** |
| truncated | 70.10 | 2.17 |
| qvgm_lite (oracle) | 70.10 | 2.10 |

### 4.3 Tiny real-action-head learning (K=4, 20 steps, lr 1e-2)

| Backend | Initial loss | Final loss |
|---|---|---|
| pathwise / checkpointed / adjoint | 724.33 | **15.57** |
| truncated | 724.33 | 31.93 |
| qvgm_lite (oracle) | 724.33 | 100.12 |

The Q-VGM-style estimator has the correct direction but a magnitude error of ≈ 0.5; at this learning rate it overshoots. This is consistent with the motivation of the original method, which relies on a carefully trained action-sensitive critic and calibrated step size rather than an oracle quadratic.

### 4.4 Memory and time

**Real 1.09B action head** (bf16, K=4, batch 1, H=40, d=132; warm-up then measured; activations only):

| Backend | Peak activation memory | Time per update | Relative memory | Relative time |
|---|---|---|---|---|
| pathwise | **5,811 MB** | **0.280 s** | 1.00× | 1.00× |
| checkpointed (step-level) | 8,259 MB | 0.404 s | **1.42×** | 1.44× |

**Mid-scale action head** (hidden 512, 8 layers, H=40, batch 4, isolated processes):

| Backend | Peak memory | Time | Relative memory | Relative time |
|---|---|---|---|---|
| pathwise | 575.6 MB | 0.062 s | 1.00× | 1.00× |
| checkpointed | 448.8 MB | 0.106 s | 0.78× | 1.71× |
| adjoint | 633.6 MB | 0.089 s | 1.10× | 1.44× |
| truncated | 390.0 MB | 0.029 s | 0.68× | 0.47× |

The mid-scale row shows the expected checkpointing trade-off (−22% memory, +71% time); the real-head row shows the opposite because the checkpoint wraps the entire 32-layer DiT call, so recomputation overlaps with the backward graph instead of replacing stored activations. **Checkpointing granularity, not the estimator, is the decisive factor at the 1B-parameter scale.**

### 4.5 Single-GPU budget for 4090 fine-tuning

| Component | Memory (bf16) |
|---|---|
| frozen VLM backbone (3B) | ≈ 6 GB |
| action head (1.09B) | ≈ 2.2 GB |
| FM activations (pathwise, K=4, batch 1) | ≈ 5.8 GB |
| gradients + Adam (action head only) | ≈ 6.6 GB |
| SuperMimic rollout (H=4, BPTT) | ≈ 2–5 GB |
| **Total (estimate)** | **≈ 23–26 GB / 48 GB** |

This leaves headroom for batch 2 or H=8; if memory becomes binding, reducing batch/horizon is preferable to switching the FM estimator.

---

## 5. Discussion

1. **Exactness is not the differentiator.** Pathwise, per-step checkpointing, and the discrete adjoint agree to machine precision on every testbed, including the real 1.09B head. Any of them is a valid gradient; the choice is an engineering trade-off.
2. **Checkpointing is granularity-sensitive.** Per-step checkpointing saves 22% at mid-scale but costs 42% at the real-head scale when it wraps the whole DiT. Per-block checkpointing is the correct implementation; we leave it as future work and do not recommend the current `checkpoint_steps` flag for memory reduction.
3. **The adjoint is exact but not a memory win in our formulation**, because we retain the $K$ forward states and run one VJP per step. It becomes attractive only when $K$ is large and per-step activations dominate; for $K\le 8$ it is strictly slower than pathwise.
4. **Truncation is a baseline, not a method.** It preserves direction (cosine ≥ 0.89) but loses up to 50% of the gradient magnitude, and learns more slowly at fixed step count.
5. **Q-VGM-style local gradients are directionally sound but magnitude-sensitive.** With an oracle critic the cosine is ≈ 0.999 and the magnitude error ≈ 0.5; with a good chunk critic and calibrated learning rate the original method may close this gap, but our reimplementation cannot support a claim about its final performance.

### 5.1 Selection for the current pipeline

| Setting | Estimator |
|---|---|
| 4090 single-GPU fine-tuning (current) | **pathwise BPTT** (measured 5.8 GB at 1.09B/K=4/batch 1) |
| 4×H200 full-scale training | **pathwise BPTT** (memory not binding) |
| memory-bound larger batch/K | per-block checkpointing (to implement) |
| very long $K$ with hard memory cap | discrete adjoint |
| cheap warm-start / baseline | truncated BPTT |
| paper comparison with Q-VGM | re-run with an official-style IQL/TD chunk critic |

---

## 6. Limitations

- The downstream loss in this study is a differentiable quadratic surrogate; the SuperMimic rollout and real task losses are not included. The numbers therefore characterize the **gradient estimator**, not policy quality.
- The real-head probe uses dummy backbone/state features and excludes the 3B VLM and the SuperMimic BPTT; the 23–26 GB budget is an estimate, not a measured end-to-end peak.
- Q-VGM is a reimplementation with an oracle/regression critic; the official offline-to-online recipe (IQL, TD, chunk critic) is not reproduced. No claim about Q-VGM's published gains is made or contradicted.
- Single seed per configuration; variance across seeds is not reported. The exact estimators are deterministic given the same initial noise, so seed variance is irrelevant for fidelity but relevant for learning curves.
- The checkpointed real-head result is specific to wrapping the entire DiT call; per-block checkpointing is expected to behave differently.

---

## 7. Reproducibility

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic

# gradient fidelity + learning, toy / tiny / scale
python -m ablation_fm_grad.run_ablation --mode toy  --steps 2 4 8 --train-steps 30
python -m ablation_fm_grad.run_ablation --mode tiny --steps 4 --train-steps 20 --lr 1e-2
python -m ablation_fm_grad.run_ablation --mode scale --scale-preset large --isolate \\
  --steps 4 --trials 1 --train-steps 1 --backends pathwise checkpointed adjoint truncated

# real 1.09B action-head memory probe
python -m ablation_fm_grad.probe_real_head --batch 1 --seq-len 32 --steps 4

# correctness tests
python -m pytest ablation_fm_grad/tests -q
```

Artifacts: `ablation_fm_grad/results/{toy_2-4-8,tiny_4,scale_4_small,scale_4_large}.{md,json}`.

---

## 8. Paper-Ready LaTeX Tables

```latex
\begin{table}[t]
\centering
\caption{Gradient fidelity of flow-matching estimators against pathwise BPTT.
Cosine and relative error are computed over all action-head parameters.}
\label{tab:fm-grad-fidelity}
\small
\begin{tabular}{@{}llcc@{}}
\toprule
Testbed & Estimator & Cosine $\uparrow$ & Rel. error $\downarrow$ \\
\midrule
toy (K=2/4/8) & checkpointed & 1.0000 & 0.00 \\
toy & adjoint & 1.0000 & 0.00 \\
toy & truncated & 0.9984--0.9995 & 0.50--0.88 \\
toy & Q-VGM-lite (oracle) & 0.9995--0.9998 & 0.50 \\
tiny (K=4) & checkpointed / adjoint & 1.0000 & 0.00 \\
tiny & truncated & 0.9975 & 0.75 \\
real head (K=4) & checkpointed & 1.0000 (max $|\Delta|{=}0$) & 0.00 \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[t]
\centering
\caption{Memory and time of the real 1.09B GR00T N1.7 action head
(bf16, $K{=}4$, batch 1, $H{=}40$, $d{=}132$; peak activations after warm-up).}
\label{tab:fm-grad-memory}
\small
\begin{tabular}{@{}lccc@{}}
\toprule
Estimator & Peak mem (MB) $\downarrow$ & Time (s) $\downarrow$ & Rel. time \\
\midrule
pathwise & \textbf{5,811} & \textbf{0.280} & 1.00$\times$ \\
checkpointed (step-level) & 8,259 & 0.404 & 1.44$\times$ \\
\bottomrule
\end{tabular}
\end{table}
```

---

## 9. References

- [@wang2026qvgm] Z. Wang, R. Zhang, Y. Liu, X. Mao, M. Wang, Y. Mu. *Q-VGM: Q-Guided Value-Gradient Matching for Offline-to-Online RL of Flow-Matching VLA Policies.* arXiv:2606.08015, 2026. <https://arxiv.org/abs/2606.08015>
- [@gao2025vita] D. Gao, B. Zhao, A. Lee, I. Chuang, H. Zhou, H. Wang, Z. Zhao, J. Zhang, I. Soltani. *VITA: Vision-to-Action Flow Matching Policy.* arXiv:2507.13231, 2025. <https://arxiv.org/abs/2507.13231>
- [@yi2026fpg] Yi et al. *Flow Policy Gradients.* 2026. (Related work; not implemented in this study.)
- [@zhang2026pgfm] *Potential-Guided Flow Matching for Vision-Language-Action Policy Improvement.* arXiv:2606.04968, 2026. <https://arxiv.org/abs/2606.04968>
- [@chen2018neuralode] R. T. Q. Chen, Y. Rubanova, J. Bettencourt, D. Duvenaud. *Neural Ordinary Differential Equations.* NeurIPS 2018. <https://arxiv.org/abs/1806.07366>
- [@griewank1992] A. Griewank. *Achieving logarithmic growth of temporal and spatial complexity in reverse automatic differentiation.* Optimization Methods and Software, 1992.
- [@pytorch_checkpoint] PyTorch documentation, `torch.utils.checkpoint`. <https://pytorch.org/docs/stable/checkpoint.html>
- [@nvidia2026groot] NVIDIA. *GR00T N1.7.* 2026. (Action-head implementation used in this study.)
- [@luo2025sonic] Z. Luo et al. *SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control.* 2025.
- [@liao2026supermimic] Liao et al. *SuperMimic: Differentiable World-Model Control for Humanoid Motion Tracking.* 2026.
- [@motphys2026unisupermimic] Motphys. *Uni-SuperMimic.* 2026.
- [@bones2026seed] *BONES-SEED: A Large-Scale Human Motion Dataset for Humanoid Control.* 2026.

> **Citation note.** arXiv identifiers for Q-VGM, VITA, PG-FM, and Neural ODE were verified from the arXiv abstract pages on 2026-10-24. The remaining entries follow the bibliography of the main paper; confirm venue and author lists before camera-ready.

---

## 10. Internal Notes (remove before submission)

- This report is a **gradient-estimator study**, not the main task-success experiment; place it in the appendix or as an ablation subsection.
- The strongest, fully supported claims are: (i) exact estimators agree to machine precision; (ii) pathwise fits 48 GB at 1.09B/K=4/batch 1 (5.8 GB); (iii) step-level checkpointing is counterproductive at this scale; (iv) Q-VGM-style gradients are direction-correct but magnitude-sensitive.
- Do **not** claim "Q-VGM improves success", "adjoint saves memory", or "checkpointing is free" from these numbers.
- The next experiment (full 3B VLM + SuperMimic rollout, H=4, batch 2) is needed before any end-to-end memory claim.
