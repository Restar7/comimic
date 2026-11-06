#!/usr/bin/env python3
"""**B 分布 + A 引擎**的反事实采集：给"策略真实下发的 token"加扰动，在 Isaac 真物理里走。

为什么需要它（2026-10-31）：
  WM 的误差随步数爆掉（k=1 根位置 0.1 cm → k=8 ~2 cm → k=20 7.6 cm），而策略在 8 步里
  能改变的位置只有 1–5 mm ⇒ 想象梯度信噪比 1:13，方向是噪声。
  根因不是模型结构，是**训练数据里没有反事实样本**：现有 B 线全是"策略正常执行"，
  WM 从没见过"token 稍微不同会怎样"，而这正是梯度每天问它的问题。

做法（分布 = B 线，引擎 = A 线）：
  * 起点状态、基础 token、参考帧号 **全部取自 B 线 dump**（策略真实所处 + 真实下发）
  * 扰动加在 **token** 上（θ 控制的是 token），σ 分档扫 {0.03…0.30}（实测 token_shift≈0.107）
  * 用 A 线那套 **512+ 并行 Isaac** 执行（不需要 VLA ⇒ 吞吐是物理速度）
  * **每 seg_len=20 tick 重置到一个新的 (起点, σ)**，与部署的 `HA_VLA_EXEC_STEPS=20` 同构，
    避免"跑久了必倒"浪费样本

输出格式 **与 B dump 完全一致**（states/next_states/actions/q_target_applied/tokens/frames）：
  * `states`/`next_states` 关节 = **HA_V31_29 序**（与 B dump 一致；训练侧会对它做 ha_v31→standard）
  * `q_target_applied` = **STANDARD 序**（与 default_q/action 同序；训练侧直接拿它减 default_q 当动作）
  * `tokens` = 该段实际喂给 WBC 的（已扰动）token，段内重复

⚠️ 产物只给 **WM 训练**用（`--pert-dump`），**绝不能进 B_GLOB**：
   B 池同时是 VLA 的自模仿监督（self_targets = 策略真正下发的 token），人为扰动混进去会污染监督。

用法：
  # 1) 只验证候选起点（不开 Isaac，几秒）
  python3 scripts/isaac_bw_pert_collect.py --dry-run --b-dump 'runs/B_*/vla_*_f*_r*/dump/*.pt'
  # 2) Isaac 真采（由 scripts/run_bw_pert_batch.sh 调用）
  ./third_party/isaacsim_env/bin/python -u scripts/isaac_bw_pert_collect.py \\
      --num-envs 2048 --steps 300 --out runs/Bw_pert/bw_w0.pt
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIST = 10
DEFAULT_USD = ("third_party/HumanoidArena/isaaclab_twist2_g1/assets/robots/"
               "g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0.usd")


# --------------------------------------------------------------------------- 起点候选
def load_starts(pattern: str, e0_max: float, far_e0: float, far_frac: float,
                seed: int, log=print):
    """从 B dump 里抽 (起点状态, 真实 token, 参考帧号, e0)。

    只保留"能切出完整 H=20 窗"的时刻（与 b_start 的 stride=20 / history=10 对齐），
    并按瞬时误差分层：主桶 e0 ≤ e0_max（策略工作点），远桶 e0_max < e0 ≤ far_e0。
    """
    lib = torch.load(str(ROOT / "runs/isaac_motion/ha_cmd/packed.pt"),
                     map_location="cpu", weights_only=False)["ref"].float()
    files = [f for f in sorted(glob.glob(pattern)) if not f.endswith("_r.pt")]
    if not files:
        raise SystemExit(f"[bw-pert] 没有匹配到 B dump：{pattern}")
    rng = np.random.RandomState(int(seed))
    near, far = [], []
    ha_names = None
    n_skip = 0
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        if not all(k in d for k in ("states", "tokens", "frames")):
            n_skip += 1
            continue
        S = d["states"].float()
        Tk = d["tokens"].float().reshape(S.shape[0], -1)
        F = d["frames"].long().clamp(0, lib.shape[0] - 1)
        jn = list(d.get("joint_names") or [])
        if ha_names is None:
            ha_names = jn
        elif jn and jn != ha_names:
            raise SystemExit(f"[bw-pert] joint_names 不一致：{Path(f).name}")
        for t in range(HIST, S.shape[0] - 20, 20):
            e0 = float((S[t, :3] - lib[int(F[t]), :3]).norm())
            if e0 <= float(e0_max):
                bucket = near
            elif e0 <= float(far_e0):
                bucket = far
            else:
                continue
            bucket.append((S[t].clone(), Tk[t].clone(), int(F[t]), e0, Path(f).name))
    if not near and not far:
        raise SystemExit("[bw-pert] 没有任何可用起点（检查 --e0-max / dump 是否含 states/frames）")
    # 远桶按比例降采样（默认只占 15%）
    n_tot = len(near) + len(far)
    want_far = int(round(n_tot * float(far_frac) / max(1e-9, 1.0 - float(far_frac))))
    if len(far) > want_far:
        sel = rng.choice(len(far), size=want_far, replace=False)
        far = [far[i] for i in sel]
        n_tot = len(near) + len(far)
    cands = near + far
    rng.shuffle(cands)
    log(f"[bw-pert] dump={len(files)} skip={n_skip}  起点: near(e0≤{e0_max})={len(near)} "
        f"far({e0_max}<e0≤{far_e0})={len(far)} ⇒ {len(cands)} 个候选")
    if ha_names is None:
        raise SystemExit("[bw-pert] dump 里没有 joint_names")
    return cands, ha_names


def _fmt_q(v, n=6):
    return "[" + ", ".join(f"{float(x):+.3f}" for x in v[:n]) + "]"


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b-dump", default="runs/B_*/vla_*_f*_r*/dump/*.pt")
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=300, help="总 tick 数（= (steps/seg_len) 段/env）")
    ap.add_argument("--seg-len", type=int, default=20, help="每段长度；与 exec_steps 对齐")
    ap.add_argument("--sigmas", nargs="*", type=float,
                    default=[0.03, 0.06, 0.10, 0.20, 0.30], help="token 扰动 σ 分档")
    ap.add_argument("--action-sigmas", nargs="*", type=float, default=None,
                    help="在**动作空间**加探索噪声的 σ 分档（归一化动作单位，≈0.01–0.05 rad）。"
                         "给了它就**不再扰 token**——这才是 SuperMimic 式 on-manifold 数据")
    ap.add_argument("--e0-max", type=float, default=0.30)
    ap.add_argument("--far-e0", type=float, default=0.60)
    ap.add_argument("--far-frac", type=float, default=0.15)
    ap.add_argument("--wbc-ckpt", type=Path,
                    default=ROOT / "runs/S1_handover/WBC_A24000_wmvel_v6.pt")
    ap.add_argument("--packed", type=Path, default=ROOT / "runs/isaac_motion/ha_cmd/packed.pt")
    ap.add_argument("--out", type=Path, required=False)
    ap.add_argument("--seed", type=int, default=20261031)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--headless", action="store_true", default=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    args_cli = ap.parse_args()

    # ★ 2026-10-31：**必须把设备解析成带序号的**（cuda:0）。裸 `torch.device("cuda")`
    #   的 index 是 None ⇒ `.to(DEV)` 会跟随"当前设备"，而 Isaac 启动后会把当前设备切到
    #   它自己选的那张卡 ⇒ 出现 "found at least two devices, cuda:0 and cuda:1"。
    global DEV
    _dev = str(args_cli.device)
    if ":" not in _dev and torch.cuda.is_available():
        _dev = _dev + ":0"
    DEV = torch.device(_dev)
    if DEV.type == "cuda":
        torch.cuda.set_device(DEV)
    print(f"[bw-pert] DEV={DEV}", flush=True)

    torch.manual_seed(int(args_cli.seed))
    np.random.seed(int(args_cli.seed))
    log = lambda *a: print(*a, flush=True)  # noqa: E731

    cands, ha_names = load_starts(args_cli.b_dump, args_cli.e0_max, args_cli.far_e0,
                                 args_cli.far_frac, args_cli.seed, log=log)
    sig_list = [float(s) for s in (args_cli.action_sigmas or args_cli.sigmas)]
    seg_len = int(args_cli.seg_len)
    n = int(args_cli.num_envs)
    n_seg = max(1, int(args_cli.steps) // seg_len)
    log(f"[bw-pert] envs={n} steps={args_cli.steps} seg_len={seg_len} ⇒ {n_seg} 段/env, "
        f"预计 {n * n_seg * seg_len} 条转移, σ={sig_list}")

    if args_cli.dry_run:
        rng = np.random.RandomState(int(args_cli.seed) + 7)
        e0s = [cands[i][3] for i in rng.randint(0, len(cands), size=min(500, len(cands)))]
        sig = rng.choice(sig_list, size=min(500, len(cands)))
        log(f"[bw-pert] dry-run OK：cand={len(cands)} e0 mean={np.mean(e0s):.3f} "
            f"p90={np.percentile(e0s, 90):.3f}  σ 覆盖={sorted(set(np.round(sig, 2)))}")
        log(f"[bw-pert] 关节名（HA_V31，与 B dump 一致）: {ha_names[:6]} ...")
        log(f"[bw-pert] 基础 token 范数: mean={float(np.mean([c[1].norm().item() for c in cands[:200]])):.3f}")
        return 0
    if not args_cli.out:
        raise SystemExit("[bw-pert] 真采必须给 --out")

    # ------------------------------------------------------------ WBC / spec
    ck = torch.load(str(args_cli.wbc_ckpt), map_location="cpu", weights_only=False)
    rs = ck["robot_spec"]
    joint_names = list(rs["joint_names"])                 # STANDARD_29（策略/动作序）
    default_q = torch.tensor(rs["default_q"], dtype=torch.float32, device=DEV)
    scale = torch.tensor(rs["action_scale"], dtype=torch.float32, device=DEV)
    nj = len(joint_names)
    if len(ha_names) != nj:
        raise SystemExit(f"[bw-pert] HA 关节数 {len(ha_names)} != robot_spec {nj}")
    usd_path = (rs.get("usd") or os.environ.get("SUPERMIMIC_ROBOT_USD") or str(ROOT / DEFAULT_USD))
    log(f"[bw-pert] USD = {usd_path}")

    # ------------------------------------------------------------ Isaac
    # 铁律：先启动 Isaac Sim，再 import isaaclab.*（否则 pxr/warp 找不到）
    from isaaclab.app import AppLauncher  # noqa: PLC0415

    app = AppLauncher(args_cli).app  # noqa: F841
    import isaaclab.sim as sim_utils  # noqa: PLC0415
    from isaaclab.actuators import ImplicitActuatorCfg  # noqa: PLC0415
    from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg  # noqa: PLC0415
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: PLC0415
    from isaaclab.sim import SimulationContext  # noqa: PLC0415
    from isaaclab.utils import configclass  # noqa: PLC0415
    from trackerLab.algorithms.supermimic.networks import Policy  # noqa: PLC0415

    pol = Policy(obs_dim=93, goal_dim=0, hist=HIST, token_dim=64, num_joints=nj, action_limit=None)
    miss, unexp = pol.load_state_dict(ck["pi"], strict=False)
    if miss or unexp:
        raise SystemExit(f"[bw-pert] policy 权重不匹配 missing={miss[:4]} unexpected={unexp[:4]}")
    pol = pol.to(DEV).eval()

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
        robot: ArticulationCfg = None

    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 200.0, device=str(DEV)))
    kp = torch.as_tensor(rs["kp"], dtype=torch.float32)
    kd = torch.as_tensor(rs["kd"], dtype=torch.float32)
    ef = torch.as_tensor(rs.get("effort_limit", [0.0] * nj), dtype=torch.float32)
    acts = {f"j{i}": ImplicitActuatorCfg(joint_names_expr=[joint_names[i]],
                                         stiffness=float(kp[i]), damping=float(kd[i]),
                                         effort_limit=float(ef[i]) if float(ef[i]) > 0 else None)
            for i in range(nj)}
    robot_cfg = ArticulationCfg(prim_path="/World/envs/env_.*/Robot",
                               spawn=sim_utils.UsdFileCfg(usd_path=usd_path),
                               init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.80)),
                               actuators=acts)
    cfg = SceneCfg(num_envs=n, env_spacing=2.5)
    cfg.robot = robot_cfg
    scene = InteractiveScene(cfg)
    robot: Articulation = scene["robot"]
    sim.reset()
    scene.update(sim.get_physics_dt())

    isaac_names = list(robot.joint_names)
    miss_j = [j for j in joint_names if j not in isaac_names]
    miss_h = [j for j in ha_names if j not in isaac_names]
    if miss_j or miss_h:
        raise SystemExit(f"[bw-pert] USD 缺关节 std={miss_j[:4]} ha={miss_h[:4]}")
    idx_std = torch.tensor([isaac_names.index(j) for j in joint_names],
                           dtype=torch.long, device=DEV)      # 策略/动作序
    idx_ha = torch.tensor([isaac_names.index(j) for j in ha_names],
                          dtype=torch.long, device=DEV)       # dump 存储序
    log(f"[bw-pert] scene ready num_envs={n}  关节映射 std/ha 都 OK")

    # ------------------------------------------------------------ 采样 + 重置
    starts = torch.stack([c[0] for c in cands]).to(DEV)        # (C,71) HA 序
    toks = torch.stack([c[1] for c in cands]).to(DEV)          # (C,64)
    frames = torch.tensor([c[2] for c in cands], dtype=torch.long, device=DEV)
    e0s = torch.tensor([c[3] for c in cands], dtype=torch.float32, device=DEV)
    srcs = [c[4] for c in cands]
    C = starts.shape[0]
    rng = np.random.RandomState(int(args_cli.seed) + 13)

    _last_a = torch.zeros(n, nj, device=DEV)
    _hist = torch.zeros(n, HIST, 93, device=DEV)
    _tok = torch.zeros(n, 64, device=DEV)
    _q_tgt = torch.zeros(n, nj, device=DEV)
    _act_sig = torch.zeros(n, device=DEV)

    def obs_now() -> torch.Tensor:
        q = robot.data.joint_pos[:, idx_std]
        dq = robot.data.joint_vel[:, idx_std]
        return torch.cat((q - default_q, dq, robot.data.root_ang_vel_b,
                          robot.data.projected_gravity_b, _last_a), dim=-1)

    def reset_envs(sel_idx: np.ndarray, sigmas: np.ndarray) -> None:
        """把选中 env 重置到新的 (B 起点, 扰动 token)。"""
        nonlocal _last_a, _hist, _tok
        sel = torch.as_tensor(sel_idx, dtype=torch.long, device=DEV)
        k = int(sel.numel())
        row = starts[sel]                                     # (k,71) HA 序
        # ★ 关键区分（2026-10-31 修，CUDA device-side assert 的根因）：
        #   `sel` 是**候选**索引（0..C-1，C 七千多），只能索引 starts/toks/frames；
        #   `eid` 是**env** 索引（0..n-1），用来写 _tok/_hist/_last_a 这些形状为 n 的缓冲区。
        if k != n:
            raise SystemExit(f"[bw-pert] reset_envs 期望覆盖全部 {n} 个 env，收到 {k}")
        eid = torch.arange(n, dtype=torch.long, device=DEV)
        pose = torch.zeros(k, 7, device=DEV)
        pose[:, :3] = row[:, :3]
        pose[:, 3:] = row[:, 3:7]
        vel = torch.zeros(k, 6, device=DEV)
        vel[:, :3] = row[:, 36:39]
        vel[:, 3:] = row[:, 39:42]
        q0 = torch.zeros(k, robot.num_joints, device=DEV)
        qd0 = torch.zeros_like(q0)
        q0[:, idx_ha] = row[:, 7:7 + nj]
        qd0[:, idx_ha] = row[:, 42:42 + nj]
        # ★ 2026-10-31：**不要传 env_ids**（传 CUDA 张量进去会在 IsaacLab 内部 scatter 时
        #   触发 device-side assert；A 线采集器也是全量写、不传 env_ids，已验证可跑）。
        #   本采集器每段重置**全部** env，所以全量写等价。
        robot.write_root_pose_to_sim(pose)
        robot.write_root_velocity_to_sim(vel)
        robot.write_joint_state_to_sim(q0, qd0)
        scene.update(sim.get_physics_dt())
        # 扰动 token（θ 控制的是 token；σ 分档）
        base = toks[sel]
        noise = torch.as_tensor(
            np.stack([rng.randn(64) for _ in range(k)]).astype(np.float32), device=DEV)
        if args_cli.action_sigmas:
            _tok[eid] = base                        # token 保持真实（不再扰）
            _act_sig[eid] = torch.as_tensor(sigmas, dtype=torch.float32, device=DEV)
        else:
            _tok[eid] = base + torch.as_tensor(sigmas, dtype=torch.float32, device=DEV)[:, None] * noise
            _act_sig[eid] = 0.0
        _last_a[eid] = 0.0
        o = obs_now()
        _hist[eid] = o[eid].unsqueeze(1).repeat(1, HIST, 1)

    def step_once() -> torch.Tensor:
        nonlocal _last_a
        obs = obs_now()
        _hist[:, :-1] = _hist[:, 1:].clone()
        _hist[:, -1] = obs
        with torch.no_grad():
            a = pol(_hist, _tok)
            if float(_act_sig.abs().max()) > 0.0:
                a = a + _act_sig[:, None] * torch.randn_like(a)   # ★ 动作空间探索噪声
        _q_tgt[:] = default_q.unsqueeze(0) + a * scale.unsqueeze(0)
        # ★ _q_tgt 已经是 (n,29) 的 **STANDARD 序**；joint_ids=idx_std 把它写进对应 isaac 关节。
        #   2026-10-31 修：原来写成 `_q_tgt[:, idx_std]`（拿 isaac 索引去切 29 宽张量）⇒ 越界，
        #   CUDA 报 device-side assert，堆栈还指在无关行上。
        robot.set_joint_position_target(_q_tgt, joint_ids=[int(i) for i in idx_std])
        robot.write_data_to_sim()
        for _ in range(4):
            sim.step()
        scene.update(sim.get_physics_dt())
        _last_a = a.detach()
        return a

    S, A, Qt, S2, TK, FR, SG, E0, SID = [], [], [], [], [], [], [], [], []
    t0 = time.time()
    for seg in range(n_seg):
        sel_idx = rng.randint(0, C, size=n)
        sig = rng.choice(sig_list, size=n)
        reset_envs(sel_idx, sig)
        SG.append(torch.as_tensor(sig, dtype=torch.float32))
        E0.append(e0s[torch.as_tensor(sel_idx, dtype=torch.long)])
        SID.append(torch.as_tensor(sel_idx, dtype=torch.long))
        for _ in range(seg_len):
            a = step_once()
            st = torch.cat((robot.data.root_pos_w, robot.data.root_quat_w,
                            robot.data.joint_pos[:, idx_ha], robot.data.root_lin_vel_w,
                            robot.data.root_ang_vel_w,
                            robot.data.joint_vel[:, idx_ha]), dim=-1)
            S.append(st.cpu()); A.append(a.cpu()); Qt.append(_q_tgt.cpu())
            TK.append(_tok.cpu())
            FR.append(frames[torch.as_tensor(sel_idx, dtype=torch.long)].cpu())
            S2.append(torch.cat((robot.data.root_pos_w, robot.data.root_quat_w,
                                 robot.data.joint_pos[:, idx_ha], robot.data.root_lin_vel_w,
                                 robot.data.root_ang_vel_w,
                                 robot.data.joint_vel[:, idx_ha]), dim=-1).cpu())
        if (seg % max(1, int(args_cli.log_every)) == 0) or seg == n_seg - 1:
            up = 1 - 2 * (robot.data.root_quat_w[:, 1] ** 2 + robot.data.root_quat_w[:, 2] ** 2)
            log(f"[bw-pert] seg {seg + 1}/{n_seg} z={float(robot.data.root_pos_w[:, 2].mean()):.3f} "
                f"up={float(up.mean()):.3f} |a|={float(a.abs().mean()):.3f} "
                f"{time.time() - t0:.0f}s")

    St = torch.cat(S)
    nrow = St.shape[0]
    out = {
        "states": St, "next_states": torch.cat(S2), "actions": torch.cat(A),
        "q_target_applied": torch.cat(Qt), "tokens": torch.cat(TK),
        "frames": torch.cat(FR),
        "joint_names": ha_names,
        "meta": {
            "source": "bw_perturb", "perturbed": True,
            "base": "B-line dump start states + recorded tokens (+ Gaussian token noise)",
            "seg_len": seg_len, "num_envs": n, "n_seg_per_env": n_seg,
            "sigmas": sig_list, "e0_max": float(args_cli.e0_max),
            "far_e0": float(args_cli.far_e0), "far_frac": float(args_cli.far_frac),
            "wbc_ckpt": str(args_cli.wbc_ckpt), "seed": int(args_cli.seed),
            "states_joint_order": "ha_v31", "q_target_applied_order": "standard",
            "n_rows": int(nrow),
            "note": "只给 WM 训练（--pert-dump）；不得进 B_GLOB",
        },
        "sigma": torch.cat(SG), "e0": torch.cat(E0), "cand_id": torch.cat(SID),
        "src_dump": [srcs[int(i)] for i in torch.cat(SID).tolist()],
    }
    args_cli.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, str(args_cli.out))
    log(f"[bw-pert] saved {args_cli.out} rows={nrow} "
        f"({torch.cat(SG).numel()} 段 × {seg_len}) {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
