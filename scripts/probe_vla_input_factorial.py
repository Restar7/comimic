#!/usr/bin/env python3
"""VLA 闭环"站桩"的输入通道定位：把 (图像, 状态) 拆成 2×2，看**谁在决定命令**。

为什么需要这个
--------------
2026-11-05 晚间的发现：所有 base 闭环都会在 ~1 m / prog≈0.4 处停住不再前进
（`scripts/analyze_closedloop_stall.py`）。要修之前必须先知道"停住"是**谁**造成的：

  * 如果换**图像**（站桩时的图像 ← 出发时的图像）能救回来 ⇒ 视觉是主通道，
    问题在"看不到进度/看不到纹理"；
  * 如果换**状态**能救回来 ⇒ 状态通道有信息，问题在状态里没有进度量；
  * 如果两个都救不回来 ⇒ 模型的条件分布本身就塌到"站"这个模态（条件均值塌缩），
    不是输入缺失，而是**这个 (图, 状态) 的可观测信息不足以唯一确定命令**。

做法：直接从闭环 dump 里取真实的服务端输入（`dump/chunks.jsonl` 的 `state64` +
`dump/req_NNN_rgb.npy`），四种组合各采 N 次，用**命令语料**把输出 token 翻译成
"它命令机器人以多快走"（最近邻检索 `g1_motion_token_64`，取该帧的参考速度），
再比较四种组合的 walkingness。

用法::

    python3 scripts/probe_vla_input_factorial.py --run runs/closedloop_r3/base_s0 \\
        --req-a 0 --req-b 40 --n-noise 4 --fm-steps 4 20
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402


def build_bank(task_frag: str = "double_desk"):
    """命令语料 → (token[N,64], 参考速度[N], 弧长进度[N])。"""
    fs = [f for f in sorted(glob.glob(str(ROOT / "data/npz/ha_tokens/*.npz"))) if task_frag in f]
    tok, spd, ph = [], [], []
    for f in fs:
        z = np.load(f, allow_pickle=True)
        T = z["g1_motion_token_64"].astype(np.float32)
        bp = z["body_pos_w"][:, 0, :].astype(np.float32)
        fps = float(z["fps"])
        v = np.linalg.norm(np.diff(bp[:, :2], axis=0), axis=1) * fps
        v = np.concatenate([v, v[-1:]]).astype(np.float32)
        seg = np.linalg.norm(np.diff(bp[:, :2], axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)]).astype(np.float32)
        tok.append(T)
        spd.append(v)
        ph.append(cum / max(float(cum[-1]), 1e-9))
    return np.concatenate(tok), np.concatenate(spd), np.concatenate(ph)


def walkingness(q: np.ndarray, bank, k: int = 1) -> tuple[float, float]:
    """把 (n,64) 的 token 检索到命令语料，返回 (参考速度均值, 最近邻距离均值)。"""
    TOK, SPD, _ = bank
    d = np.linalg.norm(q[:, None, :] - TOK[None, :, :], axis=2)
    nn = d.argmin(1)
    return float(SPD[nn].mean()), float(d.min(1).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/closedloop_r3/base_s0")
    ap.add_argument("--req-a", type=int, default=0, help="出发时的请求序号（会走）")
    ap.add_argument("--req-b", type=int, default=40, help="站桩时的请求序号（不动）")
    ap.add_argument("--n-noise", type=int, default=4)
    ap.add_argument("--fm-steps", type=int, nargs="+", default=[4])
    ap.add_argument("--fr", type=int, nargs="+", default=[0, 19], help="取 chunk 的哪几帧")
    ap.add_argument("--ckpt", default="runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    run = ROOT / a.run
    rows = [json.loads(l) for l in (run / "dump/chunks.jsonl").open() if l.strip()]
    task = "Pick up the object from the double desk and place it into the basket."
    print(f"[fac] {a.run}: {len(rows)} 请求；A=req{a.req_a}（走）B=req{a.req_b}（站）", flush=True)
    for r in rows[:1]:
        print(f"[fac] 起点 state64[0:8]={np.round(np.asarray(r['state64'])[:8], 3)}", flush=True)

    import torch  # noqa: PLC0415

    import ha_vla_latent64_server as SRV  # noqa: PLC0415

    bank = build_bank()
    print(f"[fac] 命令语料 {bank[0].shape[0]} 帧", flush=True)

    def mk_args(fm_steps: int):
        ns = argparse.Namespace(
            ckpt=Path(a.ckpt), vlm_path=ROOT / "groot_repro/nvidia/Cosmos-Reason2-2B-weights",
            device=a.device, bf16=False, horizon=40, action_dim=78, fm_steps=int(fm_steps),
            token_quant="none", sample_avg=1, decode="codec", state_order="ha_v31",
            state_layout="legacy", hand_from_base=False, hand_source="feedback",
            dry_run=False, log_dir="", head_ckpt="", lora_rank=32,
            lora_targets="attn1,attn2,ff", lora_also_train="proj_out", lead_beta=0.0,
            seed=-1, verbose=False,
        )
        return ns

    for fm in a.fm_steps:
        eng = SRV.Engine(mk_args(fm))
        print(f"\n[fac] ===== fm_steps={fm}", flush=True)
        res = {}
        for tag, (ia, ib) in {"AA": (a.req_a, a.req_a), "AB": (a.req_a, a.req_b),
                              "BA": (a.req_b, a.req_a), "BB": (a.req_b, a.req_b)}.items():
            rgb = np.load(run / "dump" / f"req_{ia:03d}_rgb.npy")
            st = np.asarray(rows[ib]["state64"], np.float32)
            qs = []
            for k in range(a.n_noise):
                torch.manual_seed(1000 + k)
                torch.cuda.manual_seed_all(1000 + k)
                c64, _ = eng.infer(rgb, st, task)
                qs.append(c64)
            Q = np.stack(qs, 0)                      # (n_noise, H, 64)
            print(f"  [{tag}] 图像=req{ia} 状态=req{ib}  |chunk|={np.abs(Q).mean():.4f} "
                  f"帧内波动={np.abs(np.diff(Q, axis=1)).mean():.5f}", flush=True)
            for fr in a.fr:
                w, dist = walkingness(Q[:, fr, :64], bank)
                # 噪声内离散度：不同噪声采到的 chunk 差多少（越大=分布越宽）
                spread = float(np.linalg.norm(Q[:, fr, :64] - Q[:, fr, :64].mean(0), axis=1).mean())
                res[(tag, fr)] = (w, dist, spread)
                print(f"     帧{fr:2d}: 命令速度={w:.3f} m/s  最近邻距离={dist:.3f}  噪声内离散={spread:.3f}",
                      flush=True)
        # 判读
        for fr in a.fr:
            wAA, wAB, wBA, wBB = (res[(t, fr)][0] for t in ("AA", "AB", "BA", "BB"))
            print(f"  [判读 帧{fr}] 只换图像 A→B: {wAA:.3f}→{wAB:.3f}  "
                  f"只换状态 A→B: {wAA:.3f}→{wBA:.3f}  "
                  f"全换成站桩: {wBB:.3f}", flush=True)
        del eng
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
