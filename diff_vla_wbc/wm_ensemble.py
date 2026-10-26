"""WM 集成不确定性：用**已有的多个 WM**当 ensemble，分歧大的地方就是外推区。

为什么要它：冻结 WM 在策略漂出去之后会瞎编；只靠奖励最大化会让 token 往"WM 觉得好"的地方跑
（我们实测过：无约束 200 步 token 漂 30×、手塌 0，而想象回报是 GT 的 5.8×）。
SuperMimic 靠"真数据保鲜"压住它；在**纯想象**里，最便宜的替代品就是**集成分歧**：

    u(s,a) = Var_models[ 预测的下一帧状态 ]

u 大的地方不采信奖励：``r_t ← r_t · exp(−λ_u · u_t)``（或直接 early-stop 想象）。
我们在仓库里已经有多个同构 WM（C6h / A@20000 / physint10500），**零训练成本**。
"""

from __future__ import annotations

from pathlib import Path

import torch

from .isaac_joint import default_xi, flat_state_tuple, normalize_xi


class WMEnsemble:
    """加载若干 ``{pi,wm}`` ckpt 的 WM，做无梯度分歧估计。"""

    def __init__(self, ckpt_paths, *, device="cuda", exclude: str | None = None):
        from trackerLab.algorithms.supermimic.networks import WorldModel  # noqa: PLC0415

        self.device = torch.device(device)
        self.models = []
        self.names = []
        for p in ckpt_paths:
            p = Path(p)
            if not p.exists():
                continue
            if exclude and p.name == exclude:
                continue
            ck = torch.load(str(p), map_location="cpu", weights_only=False)
            if "wm" not in ck:
                continue
            wm = WorldModel(num_joints=29)
            wm.load_state_dict(ck["wm"])
            integ = str(ck.get("wm_integration") or "").lower()
            wm.explicit_integration = integ.startswith("explicit")
            if wm.explicit_integration and "dt" in integ:
                try:
                    wm.dt = float(integ.split("dt", 1)[1])
                except ValueError:
                    pass
            wm.to(self.device).eval()
            for q in wm.parameters():
                q.requires_grad_(False)
            self.models.append(wm)
            self.names.append(p.name)
        if not self.models:
            raise RuntimeError("WMEnsemble: 没有可用的 WM（检查 ckpt 路径里有 'wm'）")

    @torch.no_grad()
    def disagreement(self, states: torch.Tensor, actions: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """``states (B,71), actions (B,29), xi (B,33)`` → ``u (B,)``（分歧，单位 rad²/m²）。"""
        S = flat_state_tuple(states.to(self.device))
        a = actions.to(self.device)
        x = normalize_xi(xi.to(self.device))
        ps, qs = [], []
        for wm in self.models:
            p1, _R1, _v1, _w1, q1, _qd1 = wm.step(S, a, x)
            ps.append(p1)
            qs.append(q1)
        P = torch.stack(ps, 0)          # (M,B,3)
        Q = torch.stack(qs, 0)          # (M,B,29)
        u = Q.var(0, unbiased=False).mean(-1) + 0.1 * P.var(0, unbiased=False).mean(-1)
        return u

    def audit(self) -> str:
        return f"[AUDIT][wm-ensemble] n={len(self.models)} models={self.names}"


__all__ = ["WMEnsemble"]
