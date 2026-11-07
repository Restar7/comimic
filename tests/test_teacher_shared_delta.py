"""Teacher 梯度聚合的**解析**验证（不依赖 WBC/WM 真权重，毫秒级）。

验证的恒等式
------------
部署口径是 ``token'_t = token_t + Δ``（Δ 在 chunk 上共享）。于是::

    g_A = dJ(token + eps)/d eps        # eps 是 (B,1,D) 共享变量
    g_B = Σ_t dJ(token_t)/d token_t    # 对 (B,T,D) 叶子求导再沿 T 求和

在同一个 ``J`` 定义下两者必须**逐元素相等**（不是"余弦接近"，而是 ``max|g_A−g_B|`` 小）。
第三个独立对照是中心差分方向导数 ``(J(ε+hv) − J(ε−hv))/2h`` 与 ``g·v``。

为什么用玩具 bundle 而不直接上真 WM
---------------------------------
真 WM 里包含 huber / exp / clamp 等**非光滑**点，测试时容易把"差分数值误差"和"解析错误"混在一起。
这里用**平滑**的解析模型把链路（广播 → WBC → WM → reward → 回报）验证到位；
真 WBC/WM 上的同一组检查由 ``scripts/harness02_gradient_consistency.py`` 负责。

运行：``python3 tests/test_teacher_shared_delta.py`` 或 ``pytest tests/test_teacher_shared_delta.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diff_vla_wbc.isaac_joint import FrameRef, flatten_term_major  # noqa: E402
from diff_vla_wbc.teacher_probe import (  # noqa: E402
    cosine,
    finite_difference_directional,
    shared_delta_reward_grad,
    tokenwise_reward_grad,
    unit_labels,
)

D = 64
A = 29
S = 71
OBS = 29
HIST = 10


class ToyPolicy(torch.nn.Module):
    def __init__(self, dtype=torch.float32) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.token_dim = D
        self.dtype = dtype
        self.w = torch.nn.Parameter((torch.randn(D, A) * 0.1).to(dtype))
        self.v = torch.nn.Parameter((torch.randn(OBS, A) * 0.02).to(dtype))

    def forward(self, obs, cond):
        return torch.tanh(cond @ self.w + obs @ self.v)


class ToyBundle:
    """平滑、完全可微的 WBC + WM + reward；接口与 ``imagine_with_tokens`` 约定一致。"""

    def __init__(self, seed: int = 0, dtype=torch.float32) -> None:
        torch.manual_seed(seed)
        self.dtype = dtype
        self.policy = ToyPolicy(dtype)
        self.obs_width = OBS
        self.term_dims = (OBS,)
        self.hist = HIST
        self.no_termination = 1.0
        self.A = (torch.randn(A, S) * 0.05).to(dtype)
        self.R = (torch.randn(S, S) * 0.1).to(dtype)
        self.sref = (torch.randn(S) * 0.3).to(dtype)
        self.rw = (torch.randn(S) * 0.05).to(dtype)

    # --- rollout 需要的最小接口 -------------------------------------------------
    def act(self, obs_hist_flat, cond, ref=None):
        obs = obs_hist_flat.reshape(obs_hist_flat.shape[0], HIST, OBS)[:, -1]
        return self.policy(obs, cond)

    def step(self, state, action, xi, step_idx=None):
        return torch.tanh(state @ self.R * 0.1 + action @ self.A + self.sref * 0.1)

    def robot_bodies(self, state):
        return None

    def reward(self, state, action, prev_action, reference, bodies=None):
        target = reference.f
        e = (state - target).pow(2).mean(-1)
        r = torch.exp(-e) - 0.01 * action.pow(2).mean(-1) - 0.02 * (action - prev_action).pow(2).mean(-1)
        return {"total": r, "track": r, "act": torch.zeros_like(r)}

    def observation(self, next_state, action, xi, ref=None):
        return torch.tanh(next_state[:, :OBS] + 0.01 * action[:, :OBS])

    def flatten_history(self, history):
        return flatten_term_major(history, self.term_dims)

    def termination_logits(self, state, reference):
        return torch.zeros(state.shape[0], device=state.device)


def _ctx(bundle, B, H, seed):
    dt = getattr(bundle, "dtype", torch.float32)
    g = torch.Generator().manual_seed(seed)
    state = (torch.randn(B, S, generator=g) * 0.3).to(dt)
    refs = [FrameRef((torch.randn(B, S, generator=g) * 0.3).to(dt)) for _ in range(H)]
    hist = (torch.randn(HIST, B, OBS, generator=g) * 0.2).to(dt)
    xi = torch.zeros(B, 33, dtype=dt)
    xi[:, 0] = 0.75
    prev = (torch.randn(B, A, generator=g) * 0.1).to(dt)
    return dict(state=state, references=refs, history=hist, xi=xi, prev_action=prev, obs_hist=None)


def _report(name, ok, **kw):
    status = "PASS" if ok else "FAIL"
    extra = " ".join(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}" for k, v in kw.items())
    print(f"[{status}] {name} {extra}")
    return ok


def check_consistency():
    ok_all = True
    bundle = ToyBundle()
    torch.manual_seed(0)
    for T in (1, 4, 40):
        for H in (1, 2, 5, 20):
            for eps_scale in (0.0, 1e-3):
                for seed in (0, 1):
                    B = 3
                    ctx = _ctx(bundle, B, H, seed)
                    tok = (torch.randn(B, T, D) * 0.2).to(bundle.dtype)
                    eps0 = (None if eps_scale == 0
                            else (torch.randn(B, 1, D) * eps_scale).to(bundle.dtype))
                    ga = shared_delta_reward_grad(bundle, tok, **ctx, horizon=H, gamma=0.99,
                                                  reward_steps=0, eps0=eps0)
                    gb = tokenwise_reward_grad(bundle, tok, **ctx, horizon=H, gamma=0.99,
                                               reward_steps=0, eps0=eps0)
                    diff = (ga.g - gb.g).abs().max().item()
                    rel = ((ga.g - gb.g).norm() / gb.g.norm().clamp_min(1e-12)).item()
                    cos = cosine(ga.g, gb.g).min().item()
                    ok = diff < 2e-5 and rel < 2e-5 and cos > 0.999999
                    ok_all &= _report(
                        f"gA==gB T={T} H={H} eps={eps_scale:g} seed={seed}", ok,
                        cos=cos, max_abs=diff, rel_l2=rel, gA_norm=ga.g.norm().item(),
                        gB_norm=gb.g.norm().item())
    return ok_all


def check_finite_difference():
    """float64 玩具：FD 误差应随 h 先降（截断 O(h²)）后升（舍入 ~eps/h），有清晰稳定区。"""
    ok_all = True
    bundle = ToyBundle(dtype=torch.float64)
    torch.manual_seed(1)
    B, T, H = 4, 8, 5
    ctx = _ctx(bundle, B, H, 3)
    tok = (torch.randn(B, T, D) * 0.2).double()
    ga = shared_delta_reward_grad(bundle, tok, **ctx, horizon=H, gamma=0.99)
    # 方向 = 归一化梯度（信噪比最高：g·v = |g|），避免随机方向的导数落在 FP 噪声下
    v = ga.g / ga.g.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    v = v[:, None, :]
    gv = (ga.g * v[:, 0]).sum(-1)
    errs = {}
    for h in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5):
        fd = finite_difference_directional(bundle, tok, direction=v, h=h, **ctx,
                                          horizon=H, gamma=0.99)
        rel = ((fd - gv).norm() / gv.norm().clamp_min(1e-12)).item()
        errs[h] = rel
        print(f"       h={h:g}  fd_err_rel={rel:.3e}  gv_mean={gv.mean().item():+.4e}")
    best = min(errs.values())
    ok_all &= _report("finite-difference (smooth toy fp64): min rel err < 1e-8", best < 1e-8,
                      best=best)
    return ok_all


def check_labels():
    g = torch.tensor([[3.0, 4.0], [0.0, 0.0], [float("nan"), 1.0], [0.0, 0.0]])
    tgt, n, valid = unit_labels(g, 0.08)
    ok = (abs(tgt[0, 0].item() - 0.048) < 1e-6 and abs(tgt[0, 1].item() - 0.064) < 1e-6
          and float(tgt[1].abs().sum()) == 0.0 and bool(valid[0]) and not bool(valid[1])
          and not bool(valid[2]) and n[3] == 0.0)
    return _report("unit_labels validity + zero-row masking", ok,
                   n0=n[0].item(), n2=n[2].item())


def main() -> int:
    ok = True
    print("== shared-Δ gradient aggregation (analytic toy) ==")
    ok &= check_consistency()
    print("== finite difference directional derivative ==")
    ok &= check_finite_difference()
    print("== label helper ==")
    ok &= check_labels()
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
