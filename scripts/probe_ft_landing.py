#!/usr/bin/env python3
"""**动作落地探针**：FT 到底把动作挪了多少、方向对不对、有没有盖过采样噪声？

为什么需要它（2026-11-05 深夜）
--------------------------------
"联合微调降漂移"卡在一句话上：**优化压力有没有落到正确方向**。已有的 35–52% 同号率是
拿"两次**不同**闭环的帧匹配平均位移"去比"语料窗口上的 WM 梯度"—— 两边状态分布不同，
这个数既可能低估也可能高估。这里改成**同状态配对**，而且不经 WM：

    dump 里同一条 (图像 state64, obs994)
        ├─ base 头 → token → **部署用的 ONNX decoder** → a_base(29)
        └─ FT 头   → token → 同一个 decoder      → a_ft(29)
    Δa = mean_{前 20 帧}(a_ft − a_base)

`obs994` 取自真实闭环 dump（`steps.jsonl`），token 槽位置零后由 decoder 拼入 —— 这就是
部署时的同一条通路（`SonicActionProvider` 的 ONNX decoder），所以 Δa 的单位/顺序与
真机动作一致。

三个要回答的问题
----------------
1. **落地量**：‖Δa‖ 有多大？和训练目标位移（η=0.15 SONIC 单位）比是多少？
   ≪ 目标 ⇒ 压力主要被"刹车/信任域"吃掉了（M3）。
2. **方向一致性**：三个独立训练的臂（v10/v11/v13）都朝同一片 ĝ 场优化，
   如果它们的 Δa **互相不相关**，说明每个臂的 Δa 都被噪声主导（M1）。
3. **信噪比**：同一臂、同一输入、换采样噪声，Δa 的波动有多大？
   效应/噪声 <1 ⇒ 闭环里根本测不到（这也解释了 8 对配对为什么永远给不出结论）。

用法::

    python3 scripts/probe_ft_landing.py --dump runs/closedloop_r3/base_s0 \\
        --arms base= v10=runs/joint_v10/vla_final.pt v11=runs/joint_v11/vla_final.pt \\
               v13=runs/joint_v13/vla_000100.pt --n-noise 3 --max-req 24
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="runs/closedloop_r3/base_s0", help="闭环 run 目录（有 dump/）")
    ap.add_argument("--arms", nargs="+", required=True,
                    help="名字=head ckpt；base= 表示不回填（LoRA 零初始化 = 底模）")
    ap.add_argument("--ckpt", default="runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-noise", type=int, default=3, help="每个请求采几次噪声（测信噪比）")
    ap.add_argument("--max-req", type=int, default=24, help="用多少个请求（均匀取）")
    ap.add_argument("--horizon-used", type=int, default=20, help="部署实际执行的帧数")
    a = ap.parse_args()

    import torch  # noqa: PLC0415

    import ha_vla_latent64_server as SRV  # noqa: PLC0415
    from diff_vla_wbc.lora import apply_lora  # noqa: PLC0415
    from diff_vla_wbc.sonic_decoder import TOKEN_DIM, SonicDecoderTorch  # noqa: PLC0415
    from diff_vla_wbc.sonic_mjlab import DEFAULT_DECODER_ONNX  # noqa: PLC0415

    dump = ROOT / a.dump / "dump"
    steps = [json.loads(l) for l in (dump / "steps.jsonl").open() if l.strip()]
    chunks = [json.loads(l) for l in (dump / "chunks.jsonl").open() if l.strip()]
    n_req = len(chunks)
    idx = np.unique(np.linspace(0, n_req - 1, min(a.max_req, n_req)).astype(int))
    print(f"[land] dump={a.dump} 请求 {n_req}，取样 {len(idx)}；每请求 {a.n_noise} 个噪声", flush=True)

    # ---- 任务指令（与闭环一致）----
    task = "Pick up the object from the double desk and place it into the basket."

    # ---- 1) 底模 + LoRA（三臂的 LoRA 组成完全一致：attn1 288 键 + ff 144 键）----
    ns = argparse.Namespace(
        ckpt=Path(a.ckpt), vlm_path=ROOT / "groot_repro/nvidia/Cosmos-Reason2-2B-weights",
        device=a.device, bf16=False, horizon=40, action_dim=78, fm_steps=4,
        token_quant="none", sample_avg=1, decode="codec", state_order="ha_v31",
        state_layout="legacy", hand_from_base=False, hand_source="feedback",
        dry_run=False, log_dir="", head_ckpt="", lora_rank=32,
        lora_targets="attn1,ff", lora_also_train="", lead_beta=0.0, seed=-1, verbose=False)
    eng = SRV.Engine(ns)
    n = apply_lora(eng.model.action_head, rank=32, alpha=None,
                   targets=("attn1", "ff"), also_train=())
    print(f"[land] LoRA 挂载：{n} 层（targets=attn1,ff）", flush=True)
    # ★ ckpt 里的键带 `action_head.` 前缀（= 整个 model 的 state_dict），所以快照/装载都在
    #   `eng.model` 上做；只在 action_head 上做会 unexpected=432（2026-11-05 实测）。
    base_sd = {k: v.detach().clone() for k, v in eng.model.state_dict().items()
               if "lora_" in k}
    print(f"[land] 底模快照 {len(base_sd)} 个 LoRA 张量（B 全零 ⇒ 等价底模）", flush=True)

    # ---- 2) 部署用的 decoder（ONNX → torch）----
    dec = SonicDecoderTorch.from_onnx(DEFAULT_DECODER_ONNX, dtype=torch.float32).to(a.device).eval()
    print(f"[land] decoder: {Path(DEFAULT_DECODER_ONNX).name}", flush=True)

    arm_names, arm_sds = [], {}
    for spec in a.arms:
        name, _, path = spec.partition("=")
        arm_names.append(name)
        arm_sds[name] = None if not path else torch.load(str(ROOT / path), map_location="cpu",
                                                        weights_only=False)

    def set_arm(name: str) -> None:
        sd = base_sd if arm_sds[name] is None else (arm_sds[name].get("fm", arm_sds[name])
                                                    if isinstance(arm_sds[name], dict) else arm_sds[name])
        missing, unexpected = eng.model.load_state_dict(sd, strict=False)
        if unexpected:
            raise SystemExit(f"[land] {name}: unexpected={len(unexpected)}（LoRA 配置不一致）")
        eng._head_loaded = True

    # ---- 3) 逐请求、逐噪声：三臂用**同一个噪声种子** ⇒ Δa 是干净的配对差 ----
    disp = {k: [] for k in arm_names}          # 每个元素 (29,) = 20 帧平均动作
    disp_by_req = {k: {} for k in arm_names}   # (req_idx, noise_idx) -> (29,)，供方向探针复用
    eff = {k: [] for k in arm_names}           # |Δa|（相对 base 臂）
    per_req = []
    for r_i, ci in enumerate(idx):
        rgb = np.load(dump / f"req_{ci:03d}_rgb.npy")
        st = np.asarray(chunks[ci]["state64"], np.float32)
        obs = np.asarray(steps[min(ci * 20, len(steps) - 1)]["obs994"], np.float32)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=a.device).reshape(1, -1)
        for k in range(a.n_noise):
            seed = 1234 + 97 * r_i + k
            for name in arm_names:
                set_arm(name)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                c64, _ = eng.infer(rgb, st, task)
                tok = torch.as_tensor(c64[:a.horizon_used, :TOKEN_DIM], dtype=torch.float32,
                                     device=a.device)
                with torch.no_grad():
                    acts = dec.act(tok, obs_t.expand(tok.shape[0], -1))     # (H,29)
                _m = acts.mean(0).cpu().numpy()
                disp[name].append(_m)
                disp_by_req[name][(int(ci), int(k))] = _m
        per_req.append(ci)
        if (r_i + 1) % 4 == 0:
            print(f"[land] {r_i + 1}/{len(idx)} 请求完成", flush=True)

    # ---- 3b) 落盘：per-request 平均动作（方向探针要用同一批状态）----
    np.savez_compressed(ROOT / "runs/_landing_delta.npz",
                        req_idx=np.asarray(idx),
                        arms=np.asarray(arm_names),
                        **{f"a_{k}": np.stack([disp_by_req[k][(int(c), kk)]
                                               for c in idx for kk in range(a.n_noise)], 0)
                           for k in arm_names if disp_by_req[k]})
    print(f"[land] 落盘 runs/_landing_delta.npz（{len(idx)} 请求 × {a.n_noise} 噪声）", flush=True)

    # ---- 4) 统计 ----
    print("\n[land] === 落地量（同状态配对，相对 base 臂）===")
    print(f"{'arm':<8}{'|Δa| RMS/关节':>14}{'‖Δa‖':>9}{'/η(0.15)':>10}{'噪声内 sd':>11}{'效应/噪声':>10}")
    base_arr = np.stack(disp["base"], 0) if "base" in disp else None
    summary = {}
    for name in arm_names:
        X = np.stack(disp[name], 0)                      # (M,29)
        if name == "base" or base_arr is None:
            print(f"{name:<8}{0.0:>14.4f}{0.0:>9.4f}{0.0:>10.2f}{float(X.std(0).mean()):>11.4f}"
                  f"{0.0:>10.2f}")
            continue
        D = X - base_arr                                # (M,29) 配对差（M = 请求×噪声）
        rms = float(np.sqrt((D ** 2).mean()))
        nrm = float(np.linalg.norm(D.mean(0)))
        sd = float(D.std(0).mean())
        print(f"{name:<8}{rms:>14.4f}{nrm:>9.4f}{nrm / 0.15:>10.2f}{sd:>11.4f}"
              f"{(np.abs(D.mean(0)) / (D.std(0) + 1e-9)).mean():>10.2f}")
        summary[name] = D.mean(0)

    # ---- 5) 跨臂方向一致性：三个独立训练都朝同一片 ĝ 场 ⇒ 应该正相关 ----
    arms_ft = [k for k in arm_names if k != "base" and k in summary]
    if len(arms_ft) >= 2:
        print("\n[land] === 跨臂方向一致性（cos(Δa_i, Δa_j)；同一片 ĝ 场 ⇒ 应 > 0.5）===")
        for x, y in itertools.combinations(arms_ft, 2):
            u, v = summary[x], summary[y]
            cos = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
            print(f"  cos({x:<6},{y:<6}) = {cos:+.3f}")

    # ---- 6) 判读 ----
    print("\n[land] === 判读 ===")
    if arms_ft:
        ratio = {k: float(np.linalg.norm(summary[k]) / 0.15) for k in arms_ft}
        worst = max(ratio.values())
        if worst < 0.5:
            print(f"  ⚠ 落地量普遍 < 目标位移的一半（最大 {worst:.2f}×）"
                  "⇒ 压力主要被刹车/信任域吃掉（M3），闭环 8 对不可能看出来")
        else:
            print(f"  ✅ 落地量达到目标的 {worst:.2f}×，量级够；问题不在'挪不动'")
        if len(arms_ft) >= 2:
            cs = []
            for x, y in itertools.combinations(arms_ft, 2):
                u, v = summary[x], summary[y]
                cs.append(float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)))
            if np.mean(cs) < 0.3:
                print(f"  ⚠ 跨臂 cos 均值 {np.mean(cs):+.2f} < 0.3 ⇒ 各臂的 Δa **互不相关**："
                      "方向没有被共同学到（M1，噪声/路径主导）")
            else:
                print(f"  ✅ 跨臂 cos 均值 {np.mean(cs):+.2f} ⇒ 方向被一致地实现，问题在别处"
                      "（起点分布迁移 M2 或模式 M4）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
