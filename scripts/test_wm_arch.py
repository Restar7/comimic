#!/usr/bin/env python3
"""CPU 自检：WM 的两种结构 ``--arch {mlp,gru}``（纯 python，不需要 GPU / Isaac）。

覆盖：
  1. 参数量：mlp 必须 == 8,804,352（ckpt ``net.0/3/6/9`` 的 2D 矩阵口径）；
     gru 与 MLP 基线相差 ≤ 20%（h=1024 实测 ≈ +13.5%）。
  2. 前向形状：两种结构 ``step`` 都返回同一套 6-tuple 形状（B,3)/(B,3,3)/(B,3)/(B,3)/(B,29)/(B,29)。
  3. 记忆是否真的接线：给 GRU 换一个隐状态 ⇒ 输出必须变（不是"假 GRU"）。
  4. 多步 BPTT：K=3 步自回归后 backward，梯度必须回到**第 1 步**的动作输入
     （只能穿过 h_1..h_3 传回去 ⇒ 证明隐状态跨步传递且梯度穿过全部 K 步）。
  5. K=1 单步也能跑；``step``（无粘性隐状态）与 ``step_h(h=0)`` 等价。
  6. 参数量守卫：超出 ±20% 时 ``report_wm_arch`` 必须 SystemExit；未知 arch 必须 SystemExit。

用法：
    python3 scripts/test_wm_arch.py
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "trackerlab-supermimic/source/trackerLab"):
    sys.path.insert(0, str(_p))

from trackerLab.kinematics.supermimic.rotations import so3_exp  # noqa: E402
from trackerLab.tracker_env.mdp.supermimic import NUM_J  # noqa: E402

HERE = Path(__file__).resolve().parent


def _load_finetune():
    """按路径 import 训练脚本（复用**同一份** GRU 定义 / 参数口径，避免自检与训练漂移）。"""
    path = HERE / "finetune_wm_velocity_policy.py"
    spec = importlib.util.spec_from_file_location("finetune_wm_velocity_policy", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _state(batch: int):
    """一个形状合法的 flat 状态 6-tuple（p, R, v, w, q, qd）。"""
    g = torch.Generator().manual_seed(0)
    return (
        torch.randn(batch, 3, generator=g),
        so3_exp(torch.randn(batch, 3, generator=g) * 0.05),
        torch.randn(batch, 3, generator=g),
        torch.randn(batch, 3, generator=g),
        torch.randn(batch, NUM_J, generator=g) * 0.3,
        torch.randn(batch, NUM_J, generator=g) * 0.3,
    )


def _make_live(wm):
    """把最后一层从 0 初始化改成随机值。

    ``WorldModel`` / ``GRUWorldModel`` 出厂时都 ``zeros_(net[-1])``（残差从 0 起步，这是对的），
    但那样输出恒等于积分外推、与输入无关，测不出"记忆是否接线"。训练过的 ckpt 最后一层非零，
    所以自检里先随机化它。
    """
    last = wm.head[-1] if hasattr(wm, "head") else wm.net[-1]
    with torch.no_grad():
        last.weight.normal_(0.0, 0.01)
        last.bias.normal_(0.0, 0.01)
    return wm


def main() -> int:
    torch.manual_seed(0)
    fm = _load_finetune()

    # ---------------------------------------------------------------- 1. 造模型 + 参数量
    mlp = fm.build_wm_arch("mlp", num_joints=NUM_J)
    gru = fm.build_wm_arch("gru", num_joints=NUM_J)
    n_mlp = fm.report_wm_arch("mlp", mlp)
    n_gru = fm.report_wm_arch("gru", gru)

    assert n_mlp == fm.WM_MLP_BASELINE_PARAMS == 8_804_352, (
        f"mlp 参数量 {n_mlp} != 8,804,352（2D 矩阵口径和 wm_d3.pt 不一致）")
    rel = (n_gru - fm.WM_MLP_BASELINE_PARAMS) / float(fm.WM_MLP_BASELINE_PARAMS)
    assert abs(rel) <= fm.WM_PARAM_TOL, f"gru 参数量偏差 {rel:+.2%} 超过 ±{fm.WM_PARAM_TOL:.0%}"
    assert rel > 0.0, f"h=1024 的 GRU 应该比 MLP 略大，实际 {rel:+.2%}"

    # 结构断言：单层 GRU(133->1024) + 头 Linear(1024,2048)/(2048,2048)/(70,2048)
    assert isinstance(gru.gru, torch.nn.GRU), "gru 分支必须是 nn.GRU"
    assert gru.gru.num_layers == 1 and gru.gru.hidden_size == 1024
    assert gru.gru.input_size == 133, f"GRU 输入维 {gru.gru.input_size} != 133"
    head_linears = [m for m in gru.head if isinstance(m, torch.nn.Linear)]
    assert [tuple(m.weight.shape) for m in head_linears] == [
        (2048, 1024), (2048, 2048), (70, 2048)], \
        f"GRU 头形状 {[tuple(m.weight.shape) for m in head_linears]}"
    mlp_linears = [m for m in mlp.net if isinstance(m, torch.nn.Linear)]
    assert [tuple(m.weight.shape) for m in mlp_linears] == [
        (2048, 133), (2048, 2048), (2048, 2048), (70, 2048)], \
        f"MLP 形状 {[tuple(m.weight.shape) for m in mlp_linears]}"

    # ---------------------------------------------------------------- 2. 前向形状一致
    B = 5
    S, a, xi = _state(B), torch.randn(B, NUM_J), torch.randn(B, 33)
    mlp.eval()
    gru.eval()
    _make_live(gru)
    with torch.no_grad():
        out_mlp = mlp.step(S, a, xi)
        out_gru = gru.step(S, a, xi)
        out_gru_h, h1 = gru.step_h(S, a, xi, gru.init_hidden(B))
    assert isinstance(out_mlp, tuple) and isinstance(out_gru, tuple) and len(out_gru) == 6
    for j, (t_mlp, t_gru) in enumerate(zip(out_mlp, out_gru)):
        assert t_mlp.shape == t_gru.shape, f"输出 #{j} 形状不一致 {t_mlp.shape} vs {t_gru.shape}"
        assert t_mlp.shape[0] == B
    assert all(torch.isfinite(t).all() for t in out_gru), "gru 前向出现 NaN/Inf"
    for j in range(6):
        assert torch.allclose(out_gru[j], out_gru_h[j], atol=1e-6), \
            f"step() 应等价于 step_h(h=0)（输出 #{j}）"
    assert h1.shape == (1, B, 1024) and torch.isfinite(h1).all()

    # ---------------------------------------------------------------- 3. 换隐状态 ⇒ 输出必须变
    h_alt = torch.randn_like(h1)
    with torch.no_grad():
        out_alt, _ = gru.step_h(S, a, xi, h_alt)
    assert not torch.allclose(out_gru[5], out_alt[5]), \
        "换一个隐状态输出没变 ⇒ GRU 的记忆没接到输出上（假 GRU）"
    # 粘性隐状态（验证/想象路径）也要真的带记忆
    with torch.no_grad():
        gru.begin_unroll(B)
        _ = gru.step(S, a, xi)
        out_sticky = gru.step(S, a, xi)
        gru.end_unroll()
        out_reset = gru.step(S, a, xi)
    assert not torch.allclose(out_sticky[5], out_reset[5]), \
        "begin_unroll 之后 step 没有延续隐状态"
    assert torch.allclose(out_reset[5], out_gru[5], atol=1e-6), "end_unroll 之后必须回到零隐状态"

    # ---------------------------------------------------------------- 4. K 步 BPTT（记忆 + 梯度穿全步）
    # 每步都喂**同一个** S（不做自回归）：这样第 K 步输出对第 1 步动作的唯一通路就是
    # 隐状态 h_1..h_K。有梯度 ⇒ 记忆真的跨步传递、BPTT 真的穿过了全部 K 步。
    K = 3
    acts = [torch.randn(B, NUM_J) for _ in range(K)]
    acts[0] = acts[0].clone().requires_grad_(True)
    h = gru.init_hidden(B)
    for k in range(K):
        out, h = gru.step_h(S, acts[k], xi, h)
    (out[5] - torch.randn_like(out[5])).pow(2).mean().backward()
    g = acts[0].grad
    assert g is not None and float(g.abs().sum()) > 0.0, \
        "第 K 步的 loss 对第 1 步动作没有梯度 ⇒ 隐状态没跨步传递 / BPTT 断了"
    assert float(g.abs().max()) > 1e-8

    # 对照：每步把 h 清零（无记忆）⇒ 第 1 步动作对第 K 步输出**没有任何**梯度
    acts0 = [torch.randn(B, NUM_J) for _ in range(K)]
    acts0[0] = acts0[0].clone().requires_grad_(True)
    for k in range(K):
        out, _ = gru.step_h(S, acts0[k], xi, gru.init_hidden(B))
    (out[5] - torch.randn_like(out[5])).pow(2).mean().backward()
    assert acts0[0].grad is None or float(acts0[0].grad.abs().sum()) == 0.0, \
        "无记忆 rollout 里第 1 步动作竟然影响第 K 步输出（测试本身有问题）"

    # 自回归版（训练里真正的用法）：K 步展开 + backward 必须能跑通
    s, h = S, gru.init_hidden(B)
    for k in range(K):
        s, h = gru.step_h(s, acts[k].detach(), xi, h)
    assert s[5].shape == (B, NUM_J) and h.shape == (1, B, 1024)

    # ---------------------------------------------------------------- 5. K=1
    h1s, h_next = gru.step_h(S, acts[0].detach(), xi, gru.init_hidden(B))
    assert h1s[5].shape == (B, NUM_J) and h_next.shape == (1, B, 1024)

    # ================================================================ 5b. 接管审查新增
    # 5b-1 积分等价：**同一个 70 维残差 y** ⇒ MLP 与 GRU 必须给出逐位相同的下一状态。
    #      （GRU 版把 ``WorldModel.step`` 的下半段抄了一遍；抄错一个矩阵转置就会在这里现形。）
    class _ConstOut(torch.nn.Module):
        def __init__(self, y):
            super().__init__()
            self.register_buffer("y", y)

        def forward(self, x):
            return self.y.expand(int(x.shape[0]), -1)

    g = torch.Generator().manual_seed(7)
    for explicit in (True, False):
        y7 = torch.randn(B, 70, generator=g) * 0.5
        m2 = fm.build_wm_arch("mlp", num_joints=NUM_J)
        g2 = fm.build_wm_arch("gru", num_joints=NUM_J)
        m2.net = _ConstOut(y7)
        g2.head = _ConstOut(y7)
        for w_ in (m2, g2):
            w_.explicit_integration = explicit
            w_.dt = 0.02
        with torch.no_grad():
            o_m = m2.step(S, a, xi)
            o_g = g2.step(S, a, xi)
        for j in range(6):
            assert torch.allclose(o_m[j], o_g[j], atol=1e-6), (
                f"explicit={explicit} 时输出 #{j} 不等价：GRU 的积分与 MLP 不一致"
                f"（max|Δ|={(o_m[j] - o_g[j]).abs().max():.3e}）")
    print("[test] 积分等价 OK（同 y ⇒ MLP 与 GRU 的下一状态逐位相同，explicit/residual 两套）")

    # 5b-2 ``IsaacJointBundle.step`` 的隐状态生命周期（这是**评估/想象**真正走的那条路）：
    #      step_idx=1 清零 → step_idx=2,3 延续 → step_idx=None 无记忆（MLP 行为）
    from diff_vla_wbc.isaac_joint import IsaacJointBundle, mat_to_quat  # noqa: PLC0415

    class _FakeBundle:                     # 只喂 step() 需要的那两个属性
        def __init__(self, wm):
            self.wm = wm

        def _w_calib_alpha(self, step_idx):
            return None

    flat = torch.cat((S[0], mat_to_quat(S[1]), S[4], S[2], S[3], S[5]), -1)  # (B,71)
    fb = _FakeBundle(_make_live(fm.build_wm_arch("gru", num_joints=NUM_J)))
    fb.wm.explicit_integration = True
    fb.wm.dt = 0.02
    xi0 = torch.zeros(B, 33)
    with torch.no_grad():
        s1 = IsaacJointBundle.step(fb, flat, a, xi0, 1)
        s2 = IsaacJointBundle.step(fb, s1, a, xi0, 2)
        s2b = IsaacJointBundle.step(fb, flat, a, xi0, 1)      # 回到 step_idx=1 ⇒ 必须清零
        s1_cold = IsaacJointBundle.step(_FakeBundle(fb.wm), flat, a, xi0, 1)
        s_none = IsaacJointBundle.step(fb, s2, a, xi0, None)  # 单步 ⇒ 无记忆
        s_none2 = IsaacJointBundle.step(fb, s2, a, xi0, None)  # 再调一次必须逐位相同
    assert torch.allclose(s1, s2b, atol=1e-6), "step_idx=1 没有把隐状态清零（会串味到下一次 rollout）"
    assert torch.allclose(s1, s1_cold, atol=1e-6), "第一个 rollout 的第一步不等于零隐状态"
    assert not torch.allclose(s2, s1, atol=1e-7) or not torch.allclose(s2, s2b, atol=1e-7), \
        "step_idx=2 与 step_idx=1 结果相同 ⇒ 记忆没生效"
    assert torch.allclose(s_none, s_none2, atol=1e-6), \
        "step_idx=None 必须是**无状态**的单步（否则旧调用方会静默串味）"
    assert fb.wm._roll_h is None, "step_idx=None 之后必须清掉粘性隐状态"
    # MLP 版对 step_idx 完全不敏感（逐位不变）
    fbm = _FakeBundle(_make_live(fm.build_wm_arch("mlp", num_joints=NUM_J)))
    fbm.wm.explicit_integration = True
    fbm.wm.dt = 0.02
    with torch.no_grad():
        m_a = IsaacJointBundle.step(fbm, flat, a, xi0, 1)
        m_b = IsaacJointBundle.step(fbm, flat, a, xi0, 3)
    assert torch.allclose(m_a, m_b, atol=0.0), "MLP 路径被 step_idx 影响了（基线不再逐位不变）"
    print("[test] bundle.step 隐状态生命周期 OK（1=清零 / >1=记忆 / None=无状态；MLP 不受影响）")

    # 5b-3 存档往返：GRU ckpt 必须能被**评估侧**（load_wm_weights + detect_wm_arch）严格读回
    import tempfile  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "wm_gru.pt"
        torch.save({"wm": gru.state_dict(), "wm_arch": "gru", "pi": {}}, p)
        assert fm.detect_wm_arch(torch.load(p, map_location="cpu", weights_only=False)) == "gru"
        g3 = fm.build_wm_arch("gru", num_joints=NUM_J)
        fm.load_wm_weights(g3, "gru", p)
        for k, v in gru.state_dict().items():
            assert torch.equal(v, g3.state_dict()[k]), f"GRU 权重往返丢精度/丢键：{k}"
        # MLP ckpt（wm_d3 那种键名）必须被判成 mlp，不能被误认成 GRU
        torch.save({"wm": mlp.state_dict(), "pi": {}}, Path(td) / "wm_mlp.pt")
        assert fm.detect_wm_arch(torch.load(Path(td) / "wm_mlp.pt",
                                            map_location="cpu", weights_only=False)) == "mlp"
    print("[test] ckpt 往返 OK（GRU strict 读回逐位相等；MLP 不会被误判成 GRU）")

    # 5b-4 记录一个**配方事实**（不是 bug，但决定了 GRU 必须多跑一版，见 docs 交接）：
    #      K=1 时 h≡0 ⇒ 循环矩阵 weight_hh 拿不到任何梯度 ⇒ 阶段1/2（K=1）训不到记忆。
    g4 = fm.build_wm_arch("gru", num_joints=NUM_J)
    s_, h_ = S, g4.init_hidden(B)
    out_, h_ = g4.step_h(s_, torch.randn(B, NUM_J), xi, g4.init_hidden(B))
    out_[5].pow(2).mean().backward()
    whh = g4.gru.weight_hh_l0.grad
    assert whh is None or float(whh.abs().sum()) == 0.0, (
        "K=1 时 weight_hh 竟然有梯度 —— 说明 h=0 那条路不是真零，与我们的配方分析不符")
    print("[test] 配方事实 OK（K=1 ⇒ weight_hh 零梯度 ⇒ 阶段1/2 训不到记忆，需多步配方）")

    # ---------------------------------------------------------------- 6. 守卫 + 非法 arch
    assert fm.wm_param_count(_make_live(mlp)) == n_mlp
    try:
        fm.build_wm_arch("transformer")
    except SystemExit:
        pass
    else:
        raise AssertionError("未知 --arch 必须 SystemExit")
    with torch.device("meta"):
        _big = torch.nn.Linear(3400, 3400)          # 11.56M 2D 参数 ≈ +31%，必须被拦
    _buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(_buf):
            fm.report_wm_arch("gru", _big)
    except SystemExit as e:
        assert "20%" in str(e), f"守卫报错信息不清晰: {e}"
    else:
        raise AssertionError("参数量超出 ±20% 必须 SystemExit")
    print("[test] 参数守卫 OK（未知 arch / 超限参数量均被拦截）")

    print(f"[test] PASS  mlp params={n_mlp}  gru params={n_gru} ({rel:+.2%})  "
          f"hidden({h1.shape})  BPTT K={K} 梯度回传 OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
