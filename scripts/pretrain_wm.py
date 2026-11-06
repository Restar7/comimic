#!/usr/bin/env python3
"""在**联合微调之前**把 WM 训稳（三阶段），存成一个新的 joint ckpt。

为什么单独一步：门要测"实际将要被训练用的那个 WM"；critic 的 GAE 目标来自 WM 的想象 rollout
⇒ WM 必须先训稳；联合微调期间 WM 不再变动 ⇒ reward 是固定靶。

三阶段各自对应一次实测故障：
  * 阶段 1（完整场景采集 102 万行、有物体/接触）：速度尺度从"平地 0.433"拉到"场景 0.357"，
    修掉 qd_ratio 2.95 里"没见过接触"那部分；
  * 阶段 2（mix：场景全量 + B 线加权到 45%）：适配 VLA 在环的部署分布（qd_corr 0.638 的主因）；
  * 阶段 3（只用 B 线，小 lr + 幅值比正则）：两阶段后 bline qd_ratio 仍 1.10–1.12，诊断显示是
    **全局高估**（各 root_err 桶 1.09–1.13）⇒ MSE 对"整体高估 10%"不敏感，必须显式加
    ((mean|pred|−mean|ref|)/mean|ref|)^2。
    ⚠️ 阶段 3 必须是**最后一次训练**，否则会被后面的 mix 覆盖回 1.10（今天踩过）。

用法：
    python3 scripts/pretrain_wm.py --joint-ckpt <上一轮 ckpt> \
        --transitions <round_r>/wm_mix.pt --steps 6000 --out <round_r>/wm_pre.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))


def _stats(out: dict) -> str:
    keep = ("steps", "loss_first", "loss_last", "qd_corr")
    return json.dumps({k: v for k, v in out.items() if k in keep})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint-ckpt", type=Path, required=True, help="起点（上一轮累积的 policy+wm）")
    ap.add_argument("--transitions", type=Path, required=True, help="wm_mix.pt（阶段 2 用）")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=6000, help="阶段 2（mix）步数")
    ap.add_argument("--stage1-transitions", type=Path, default=None,
                    help="阶段 1 数据；默认取 --transitions 同目录的 transitions.pt（完整场景采集）")
    ap.add_argument("--stage1-steps", type=int, default=3000, help="阶段 1 步数")
    ap.add_argument("--stage3-steps", type=int, default=0,
                    help="阶段 3（纯 B 线）步数；默认 0=关（2026-10-29 实测：纯 B 线 10000 步会把 sim 分布毁掉，"
                         "corr 0.989→0.581、ratio 1.03→1.33，属灾难性遗忘）")
    ap.add_argument("--mix-mag-weight", type=float, default=30.0,
                    help="阶段 2（mix）上的幅值比正则权重：在保持 sim 分布的同时压 mean|pred|/mean|ref|")
    ap.add_argument("--stage3-mag-weight", type=float, default=100.0,
                    help="阶段 3 幅值比正则权重（直接压 mean|pred|/mean|ref| → 1）")
    ap.add_argument("--no-stage1", action="store_true")
    # ---- 阶段 2b：k 步自回归（2026-10-29 新增，默认开）----
    # 单步损失（阶段 1/2）里 state 永远是真实状态 ⇒ 累积误差不受约束：k=1 的 qd_ratio
    # 0.977（门禁只量这一步，永远 PASS）而 k=20 是 1.329。用 B 线连续窗口滚 k 步、
    # 每步喂模型自己的输出即可修掉：held-out 上 k=20 qd_ratio 1.279→1.095、
    # qd_corr 0.771→0.817、joint_mae 1.14°→0.80°，三项同时变好。
    ap.add_argument("--roll-steps", type=int, default=1500, help="阶段 2b 步数；0=关")
    ap.add_argument("--roll-k", type=int, default=20,
                    help="自回归步数。实测 2026-10-29：k=20 优于 k=8（held-out k=20 的 qd_corr "
                         "0.861 vs 0.817，joint 0.74° vs 0.80°），且 k=8 处不退化。"
                         "另：B 线每 20 tick 重规划一次，k=20 正好覆盖一整段开环")
    ap.add_argument("--roll-lr", type=float, default=3e-5)
    ap.add_argument("--roll-batch", type=int, default=64)
    ap.add_argument("--roll-mag-weight", type=float, default=30.0)
    ap.add_argument("--roll-sim-weight", type=float, default=0.3,
                    help="阶段2b 每步交织一条真实场景单步样本的权重；只跑多步会把 sim 门打坏")
    ap.add_argument("--roll-dumps", type=str, default=None,
                    help="窗口来源 glob；默认取 --transitions 同目录的 bline/b_*.pt")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--vel-weight", type=float, default=5.0)
    ap.add_argument("--kl-anchor", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    # ★ 2026-11-03：refresh_wm 的 minibatch 采样用的是**没播种**的 torch.Generator，
    #   同一份数据每跑一次结果都不一样（实测 qd_corr 0.6735 vs 0.5089/0.512，
    #   幅度比"改进"本身还大）。这里显式播种，让刷新可复现、可 best-of-N 选。
    # ★ 2026-11-04（用户口径）：**WM 从头训练**。
    #   理由：旧 ckpt 的 WM 是在 **SuperMimic 动作空间**上训的，而我们现在用 SONIC decoder
    #   的 raw action（不同执行器）。接着微调会带着"动作尺度/敏感性"的先验偏差，而这个
    #   偏差恰好是我们最关心的量（∂s'/∂a）。既然 A' 已经有 120 万条转移，直接重训更干净。
    # ★ 2026-11-04：**阶段2 中间落盘 + 可续跑**（原来只在最后 save，阶段2b 一慢就白跑 6 分钟）
    ap.add_argument("--stage2-ckpt", type=Path, default=None,
                    help="阶段2 结束立刻存这里（默认 <out>.stage2.pt）")
    ap.add_argument("--resume-stage2", action="store_true",
                    help="若 <out>.stage2.pt 存在则跳过阶段2（只跑阶段2b）")
    ap.add_argument("--wm-scratch", action="store_true",
                    help="加载 bundle 后把 WM 权重重新初始化（只用结构/spec/奖励/FK）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-tag", type=str, default="",
                    help="产物名后缀（多 seed 扫时区分，例如 seed1 ⇒ wm_pre_s1.pt）")
    a = ap.parse_args()
    _wenv = None
    def _wgen():
        """每次调用返回同一个播种 generator（refresh_wm 会持续消费它）。"""
        nonlocal _wenv
        if _wenv is None:
            import torch as _t
            _wenv = _t.Generator().manual_seed(int(a.seed))
        return _wenv


    import torch

    from diff_vla_wbc.wm_refresh import refresh_wm
    from diff_vla_wbc.isaac_joint import is_isaac_checkpoint, load_isaac_bundle

    dev = torch.device(a.device)
    if is_isaac_checkpoint(a.joint_ckpt):
        # 2026-11-03：新线 WBC 是 trackerlab(Isaac) ckpt；旧的 pretrain_wm 只认 mjlab。
        _npz = sorted((ROOT / "data/npz/ha_tokens").glob("*.npz"))
        bundle = load_isaac_bundle(
            a.joint_ckpt, device=dev, npz=_npz,
            packed_motion=ROOT / "runs/isaac_motion/ha_cmd_exec/packed.pt",
            obs_mode=None, hist=None, token_dim=64,
        )
        print(f"[wm-pre] backend=isaac ckpt={a.joint_ckpt}", flush=True)
    else:
        from diff_vla_wbc.mjlab_joint import load_mjlab_bundle  # noqa: PLC0415
        bundle = load_mjlab_bundle(a.joint_ckpt, device=dev)
        print(f"[wm-pre] backend=mjlab ckpt={a.joint_ckpt}", flush=True)
    if bool(getattr(a, "wm_scratch", False)):
        import torch.nn as _nn  # noqa: PLC0415
        _n = 0
        for _m in bundle.wm.modules():
            if isinstance(_m, _nn.Linear):
                _nn.init.orthogonal_(_m.weight)
                if _m.bias is not None:
                    _nn.init.zeros_(_m.bias)
                _n += 1
        print(f"[wm-pre] **WM 从头训练**：重新初始化 {_n} 个 Linear 层（丢弃旧动作空间先验）",
              flush=True)
    mix = torch.load(str(a.transitions), map_location="cpu", weights_only=False)
    print(f"[wm-pre] ckpt={a.joint_ckpt} | mix rows={int(mix['states'].shape[0])}", flush=True)
    t_all = time.time()

    # ---------------- 阶段 1：完整场景采集（有物体/接触）----------------
    stage1 = None if a.no_stage1 else a.stage1_transitions
    if stage1 is None and not a.no_stage1:
        cand = Path(a.transitions).parent / "transitions.pt"
        if cand.exists() and cand.resolve() != Path(a.transitions).resolve():
            stage1 = cand
    if stage1 is not None and Path(stage1).exists():
        d1 = torch.load(str(stage1), map_location="cpu", weights_only=False)
        print(f"[wm-pre] 阶段1（完整场景）：{stage1} rows={int(d1['states'].shape[0])} "
              f"steps={a.stage1_steps}", flush=True)
        o1 = refresh_wm(bundle.wm, d1, steps=int(a.stage1_steps), batch=int(a.batch), lr=float(a.lr),
                        vel_weight=float(a.vel_weight), kl_anchor=float(a.kl_anchor),
                        device=dev, log=print, step_fn=bundle.step, generator=_wgen())
        print("[wm-pre] 阶段1 完成 " + _stats(o1), flush=True)

    # ---------------- 阶段 2：mix（场景全量 + B 线加权）----------------
    _st2 = Path(a.stage2_ckpt) if a.stage2_ckpt else Path(str(a.out) + ".stage2.pt")
    _skip2 = bool(getattr(a, "resume_stage2", False)) and _st2.exists()
    print(f"[wm-pre] 阶段2（mix）：rows={int(mix['states'].shape[0])} steps={a.steps}"
          f"{'（续跑：跳过）' if _skip2 else ''}", flush=True)
    t0 = time.time()
    if _skip2:
        _ck2 = torch.load(str(_st2), map_location="cpu", weights_only=False)
        bundle.wm.load_state_dict(_ck2.get("wm", _ck2), strict=True)
        o2 = {"steps": 0, "resumed_from": str(_st2)}
    else:
        o2 = refresh_wm(bundle.wm, mix, steps=int(a.steps), batch=int(a.batch), lr=float(a.lr),
                        vel_weight=float(a.vel_weight), kl_anchor=float(a.kl_anchor),
                        device=dev, log=print, step_fn=bundle.step,
                        mag_weight=float(a.mix_mag_weight), generator=_wgen())
        bundle.save_checkpoint(_st2)
        print(f"[wm-pre] 阶段2 中间产物 -> {_st2}（阶段2b 崩了可 --resume-stage2 续跑）", flush=True)
    print(f"[wm-pre] 阶段2 完成 {_stats(o2)} | {time.time()-t0:.1f}s", flush=True)

    # ---------------- 阶段 3：纯 B 线 + 幅值比正则（**必须最后**）----------------
    if int(a.stage3_steps) > 0:
        cand3 = Path(a.transitions).parent / "bline_transitions.pt"
        if cand3.exists():
            d3 = torch.load(str(cand3), map_location="cpu", weights_only=False)
            print(f"[wm-pre] 阶段3（纯 B 线）：{cand3} rows={int(d3['states'].shape[0])} "
                  f"steps={a.stage3_steps} lr={a.lr*0.5:.1e} mag={a.stage3_mag_weight:g}", flush=True)
            o3 = refresh_wm(bundle.wm, d3, steps=int(a.stage3_steps), batch=int(a.batch),
                            lr=float(a.lr) * 0.5, vel_weight=float(a.vel_weight),
                            kl_anchor=float(a.kl_anchor), device=dev, log=print,
                            step_fn=bundle.step, mag_weight=float(a.stage3_mag_weight),
                            generator=_wgen())
            print("[wm-pre] 阶段3 完成 " + _stats(o3), flush=True)
        else:
            print(f"[wm-pre][warn] 没有 {cand3} ⇒ 跳过阶段 3", flush=True)

    # ---------------- 阶段 2b：k 步自回归（必须最后，且不改阶段 1/2 的语义）----------------
    if int(a.roll_steps) > 0:
        import glob as _glob  # noqa: PLC0415
        from diff_vla_wbc.wm_refresh import train_multistep  # noqa: PLC0415
        pat = a.roll_dumps or str(Path(a.transitions).parent / "bline" / "b_*.pt")
        fs = sorted(_glob.glob(pat))
        if not fs:
            print(f"[wm-pre][warn] 阶段2b 没有窗口来源（{pat}）⇒ 跳过", flush=True)
        else:
            S, A = [], []
            _prebuilt = False
            for f in fs:
                d = torch.load(f, map_location="cpu", weights_only=False)
                st, ac = d["states"].float(), d["actions"].float()
                # ★ 2026-11-04：支持**预构建的三维窗口文件**（build_wm_roll_windows.py 的产物）：
                #   states (N,k+1,71) / actions (N,k,29)。A' 的行是 env 交错 + 段边界瞬移，
                #   直接 `st[t:t+k+1]` 会切出"21 个不同 env"的假窗口 ⇒ 必须离线按
                #   (clip, 连续帧, 不跨段) 抽好。
                if st.ndim == 3:
                    S.append(st); A.append(ac); _prebuilt = True
                    continue
                for t in range(0, int(st.shape[0]) - int(a.roll_k), 8):
                    S.append(st[t:t + int(a.roll_k) + 1]); A.append(ac[t:t + int(a.roll_k)])
            if _prebuilt:
                S, A = torch.cat(S, 0), torch.cat(A, 0)
            else:
                S, A = torch.stack(S), torch.stack(A)
            # ★ 2026-11-04 **性能修复**：把窗口整块搬到 GPU。
            #   否则 train_multistep 内层循环每步都做 `A[idx,t].to(dev)` / `S[idx,t+1].to(dev)`
            #   （6000 步 × 20 × 2 = 24 万次同步小拷贝）⇒ 实测 16 分钟还没跑完 6000 步。
            #   18661×21×71 只有 111 MB，显存完全放得下。
            _mb = (S.numel() + A.numel()) * 4 / 1024 ** 2
            print(f"[wm-pre] 阶段2b 窗口搬到 {dev}（{_mb:.0f} MB）", flush=True)
            S, A = S.to(dev), A.to(dev)
            _sim = None
            for _c in (Path(a.transitions).parent / "transitions.pt", stage1):
                if _c is not None and Path(_c).exists():
                    _sim = torch.load(str(_c), map_location="cpu", weights_only=False)
                    break
            print(f"[wm-pre] 阶段2b（k={a.roll_k} 自回归）：{len(fs)} 集 → 窗口 {tuple(S.shape)} "
                  f"steps={a.roll_steps} | 交织 sim={0 if _sim is None else int(_sim['states'].shape[0])} 行",
                  flush=True)
            o4 = train_multistep(bundle.wm, S, A, steps=int(a.roll_steps), batch=int(a.roll_batch),
                                 lr=float(a.roll_lr), k=int(a.roll_k),
                                 vel_weight=float(a.vel_weight), mag_weight=float(a.roll_mag_weight),
                                 device=dev, step_fn=bundle.step,
                                 sim_data=_sim, sim_weight=float(a.roll_sim_weight),
                                 xi_fn=(lambda b: bundle.default_xi(b, dev))
                                 if hasattr(bundle, "default_xi") else None, log=print)
            print(f"[wm-pre] 阶段2b 完成 {o4}", flush=True)

    bundle.save_checkpoint(a.out)
    # ★ 2026-11-04：把**刷新配方**写进产物（seed 尤其重要）。以前 `args` 里没有 seed，
    #   事后无法判断两份 ckpt 是不是同一次随机采样，只能重跑对照 —— 这正是
    #   "同一份数据 qd_corr 0.67 vs 0.51" 那次事故查不清的原因。
    try:
        import torch as _t  # noqa: PLC0415
        _ck = _t.load(str(a.out), map_location="cpu", weights_only=False)
        _ck["refresh_recipe"] = {
            "seed": int(a.seed), "steps": int(a.steps), "batch": int(a.batch),
            "lr": float(a.lr), "vel_weight": float(a.vel_weight),
            "kl_anchor": float(a.kl_anchor), "mix_mag_weight": float(a.mix_mag_weight),
            "transitions": str(a.transitions),
        }
        _t.save(_ck, str(a.out))
    except Exception as _e:  # noqa: BLE001
        print(f"[wm-pre][warn] 写入 refresh_recipe 失败: {_e}", flush=True)
    print(f"[wm-pre] 总耗时 {time.time()-t_all:.1f}s → {a.out} (seed={a.seed})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
