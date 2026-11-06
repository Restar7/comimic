#!/usr/bin/env python3
"""**从零训练 WM**（4 卡 DDP，2026-11-05 用户要求）

与 `scripts/finetune_wm_multistep.py` 的区别
--------------------------------------------
* **从零**：网络权重全部重新随机初始化（`WorldModel.__init__` 里最后一层是零初始化，
  所以"从零"= 纯物理积分器起步，学到的全是数据里的修正项），
  **不继承**老数据训出来的动力学。`in_norm/out_norm` 只是输入/输出的数据尺度统计，保留继承。
* **DDP**：4 卡数据并行。注意本训练是 **launch/latency 主导**（实测 batch 1024→8192 每步耗时
  都是 ~200 ms）⇒ 卡数只在该"每卡 batch 也大"时才有增益；这里每卡 batch 默认 4096。
* **两阶段**：先单步（`--steps-single`），再多步自回归 k（`--steps-multi`）。
  单步学"一步动力学"，多步学"自己的误差怎么累积"（只做单步会让 20 步 rollout 发散）。

用法（4 卡）::

    torchrun --nproc_per_node=4 scripts/pretrain_wm_scratch_ddp.py \
        --windows runs/a_prime_sonic_exc/roll_windows.pt \
        --out runs/a_prime_sonic_exc/wm_scratch_ddp.pt \
        --steps-single 12000 --steps-multi 8000 --batch 4096 --lr 3e-4

判据（跑完自己测）：`probe_wm_multistep.py`（ratio ∈[0.8,1.2]）+ `probe_wm_headroom.py --gate`。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402


def reset_wm_net(wm) -> int:
    """把 WM 网络权重重新随机初始化（保留 in_norm/out_norm 的数据尺度统计）。

    `WorldModel.__init__` 把最后一层置零 ⇒ 从零开始时模型先是"纯物理积分器"，
    学习目标就是数据里的残差修正。这比继承老权重干净。
    """
    n = 0
    for m in wm.net.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
            n += 1
    # 复刻 __init__ 里的零初始化
    torch.nn.init.zeros_(wm.net[-1].weight)
    torch.nn.init.zeros_(wm.net[-1].bias)
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=Path,
                    default=ROOT / "runs/a_prime_sonic_exc/roll_windows.pt")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "runs/a_prime_sonic_exc/wm_scratch_ddp.pt")
    ap.add_argument("--transitions", type=str,
                    default=str(ROOT / "runs/a_prime_sonic_exc/*_off*.pt"),
                    help="glob；给了就用**全量转移**（约 52.9 万条）做单步阶段，"
                         "否则只用窗口的首帧（只有 2.3 万个不同起点，太少）")
    ap.add_argument("--steps-single", type=int, default=12000)
    ap.add_argument("--steps-multi", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=4096, help="**每卡** batch")
    ap.add_argument("--lr", type=float, default=3e-4, help="单步阶段 lr（从零要偏大）")
    ap.add_argument("--lr-multi", type=float, default=3e-5, help="多步阶段 lr")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--inc-weight", type=float, default=2.0)
    ap.add_argument("--mag-weight", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=500)
    a = ap.parse_args()

    dist.init_process_group("nccl")
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")

    from diff_vla_wbc.isaac_joint import flat_state_tuple, load_isaac_bundle  # noqa: PLC0415
    from diff_vla_wbc.wm_refresh import _split_parts, ssm_wm_loss  # noqa: PLC0415
    from trackerLab.kinematics.supermimic.rotations import mat_to_quat  # noqa: PLC0415
    from trackerLab.tracker_env.mdp.supermimic import normalize_xi  # noqa: PLC0415

    npz = sorted((ROOT / "data/npz/ha_tokens").glob("*.npz"))
    bundle = load_isaac_bundle(ROOT / "runs/S1_handover/WBC_A24000_wmvel_cmd.pt", device=dev,
                               npz=npz, packed_motion=ROOT / "runs/isaac_motion/ha_cmd_exec/packed.pt",
                               obs_mode=None, hist=None, token_dim=64)
    wm = bundle.wm
    n_reset = reset_wm_net(wm)
    if rank == 0:
        print(f"[scratch] 网络权重已重置（{n_reset} 个模块 + 末层置零）；world_size={world} "
              f"每卡 batch={a.batch} 等效 batch={a.batch * world}", flush=True)

    w = torch.load(a.windows, map_location="cpu", weights_only=False)
    S = w["states"].float()          # (N, k+1, 71)
    A = w["actions"].float()         # (N, k, 29)
    N, K = int(S.shape[0]), int(S.shape[1]) - 1
    if rank == 0:
        print(f"[scratch] 窗口 {tuple(S.shape)}", flush=True)

    # ---- 单步阶段的数据：优先用全量转移（从零训必须有足够的状态多样性）----
    import glob as _glob  # noqa: PLC0415
    _tf = sorted(_glob.glob(str(a.transitions))) if a.transitions else []
    if _tf:
        _st, _ac, _ns = [], [], []
        for _f in _tf:
            _z = torch.load(_f, map_location="cpu", weights_only=False)
            _st.append(_z["states"]); _ac.append(_z["actions"]); _ns.append(_z["next_states"])
        S1 = torch.cat(_st).float()
        A1 = torch.cat(_ac).float()
        NS1 = torch.cat(_ns).float()
        if rank == 0:
            print(f"[scratch] 单步数据：{len(_tf)} 个 shard，{S1.shape[0]} 条真实转移", flush=True)
    else:
        S1, A1, NS1 = S[:, 0], A[:, 0], S[:, 1]
        if rank == 0:
            print(f"[scratch] 单步数据：窗口首帧 {S1.shape[0]} 条（建议给 --transitions）", flush=True)
    N1 = int(S1.shape[0])

    wm.requires_grad_(True)
    wm = DDP(wm, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    params = list(wm.parameters())

    def step_flat(x: torch.Tensor, act: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """与 `bundle.step` 逐位一致，但走 DDP 包装后的模块。"""
        Sx = flat_state_tuple(x)
        p1, R1, v1, w1_, q1, qd1 = wm.module.step(Sx, act, normalize_xi(xi))
        return torch.cat((p1, mat_to_quat(R1), q1, v1, w1_, qd1), dim=-1)

    g = torch.Generator().manual_seed(int(a.seed) + rank)

    # ---------------- 阶段 1：单步 ----------------
    opt = torch.optim.Adam(params, lr=float(a.lr))
    t0 = time.time()
    for it in range(int(a.steps_single)):
        idx = torch.randint(N1, (int(a.batch),), generator=g)
        s0 = S1[idx].to(dev, non_blocking=True)
        a0 = A1[idx].to(dev, non_blocking=True)
        s1 = NS1[idx].to(dev, non_blocking=True)
        xi = bundle.default_xi(s0.shape[0], dev)
        pred = step_flat(s0, a0, xi)
        loss = ssm_wm_loss(wm.module, pred, s1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if rank == 0 and (it % int(a.log_every) == 0 or it == int(a.steps_single) - 1):
            print(f"  [single {it}/{a.steps_single}] loss={float(loss):.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # ---------------- 阶段 2：多步（自回归 k） ----------------
    opt = torch.optim.Adam(params, lr=float(a.lr_multi))
    t1 = time.time()
    for it in range(int(a.steps_multi)):
        idx = torch.randint(N, (int(a.batch),), generator=g)
        x = S[idx, 0].to(dev, non_blocking=True)
        acts = A[idx, :K].to(dev, non_blocking=True)
        tgts = S[idx, 1:].to(dev, non_blocking=True)
        loss = torch.zeros((), device=dev)
        for t in range(K):
            x_prev = x
            xi = bundle.default_xi(x.shape[0], dev)
            x = step_flat(x, acts[:, t], xi)
            loss = loss + ssm_wm_loss(wm.module, x, tgts[:, t])
            if float(a.inc_weight) > 0:
                from diff_vla_wbc.wm_refresh import increment_loss  # noqa: PLC0415
                loss = loss + float(a.inc_weight) * increment_loss(x, x_prev, tgts[:, t])
        loss = loss / K
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if rank == 0 and (it % int(a.log_every) == 0 or it == int(a.steps_multi) - 1):
            print(f"  [multi {it}/{a.steps_multi}] loss={float(loss):.4f} "
                  f"({time.time()-t1:.0f}s, k={K})", flush=True)

    if rank == 0:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"wm": wm.module.state_dict(),
                    "scratch": True, "steps_single": int(a.steps_single),
                    "steps_multi": int(a.steps_multi), "batch_per_rank": int(a.batch),
                    "world_size": world, "k": K}, a.out)
        print(f"[scratch] -> {a.out}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
