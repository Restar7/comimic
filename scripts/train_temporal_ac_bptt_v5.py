#!/usr/bin/env python3
"""v5 主线训练器：**per-frame Δtoken** + 可微 WM 的 H 步 BPTT（默认 β=0，不做 Critic）。

唯一规范来源：`docs/TEMPORAL_AC_BPTT_V5_MAINLINE_SPEC.md`（下称"规范"）。
本脚本只做**一条主线**：逐帧 Δ + Chunk-Attention/History-GRU Actor + H20 想象回报，
**不做** Constant / 共享 Δ / 位置模板对照臂，**默认不启用 Critic**（§5.5.3 / §6.1）。

锁定口径（§0-A，执行期不允许自行改动）
------------------------------------
* 数据：**on-the-fly 均匀抽帧**（`multinomial(w, B)` over 合格帧）+ **现场 burn-in(10)**
  + **现场 VLA（每步新噪声）**；不建训练池（评测集是固定窗口，只用于 held-out 评测）。
* Actor：`TemporalDeltaActorV5`（`delta_token_temporal_v5.py`，1,744,832 参数），
  输入整段 40 帧 chunk，输出 `(B,T,64)`，**t ≥ K 恒 0**；`cap=0.08`（前 100 updates 从 0.04 线性升）。
* 时间轴：`T=40`、`K=20`（= 客户端 `HA_VLA_EXEC_STEPS`）、`H=20`、`ℓ(k)=k`。
* 损失：`L = -mean(J20) + 0.010·act_trust(29D rad, τ=0.020) + 0.002·smooth + 0.001·anchor`。
* Critic：`critic.enabled=false, beta=0`，日志写 `BOOTSTRAP_NOT_IDENTIFIABLE (K==H)`；
  代码路径（TDλ + EMA target + Value Readiness Gate）在 `imagined_value_v5.py`，
  由 `--critic-enabled 1 --beta …` 打开（只在 `K > H` 时允许 β>0）。
* 预算：`updates=1000`（上限 1200）、`batch=256`、val 每 50 步、早停=连续 6 次无改善。

`data_mode` 必须显式记录（§11）：`on-the-fly`（主训练）与 `cached_pool`（只用于 pilot /
单测 / 调代码；报告不得与主训练混表）在 `config.json` / `train.jsonl` / `final_report.md`
里都带这个字段。

用法（主训练，规范 §8 M2）::

    source scripts/harness03_env.sh
    CUDA_VISIBLE_DEVICES=0 python3 scripts/train_temporal_ac_bptt_v5.py \
        --out $OUT/main/seed0 --updates 1000 --batch 256 --horizon 20 --beta 0 --seed 0 \
        --on-the-fly 1 --eval-set $OUT/eval_set --device cuda:0
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path(os.environ.get("COMIMIC_MAIN", str(ROOT)))
for _p in (str(MAIN), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("SUPERMIMIC_HA_ALIGN", "1")
os.environ.setdefault(
    "SUPERMIMIC_ROBOT_USD",
    str(MAIN / "trackerlab-supermimic/data/assetslib/unitree/unitree_g1/usd/g1_29dof_rev_1_0.usd"))

import diag_bptt_learnability as d03                        # noqa: E402
from diff_vla_wbc.compat import g1_default_joint_pos        # noqa: E402
from diff_vla_wbc.action_trust_v5 import (                  # noqa: E402
    action_budget_stats, anchor_penalty, joint_error_rad, physical_action_trust,
    q_target_from_action, smoothness_penalty)
from diff_vla_wbc.delta_token_io_v5 import (                # noqa: E402
    DeltaTokenSpecV5, hist10_from_hist64, save_temporal_actor_v5,
    spec_v5_from_actor, with_meta)
from diff_vla_wbc.delta_token_temporal_v5 import (          # noqa: E402
    HIST_DIM, PositionTemplateActor, TemporalActorCfg, TemporalDeltaActorV5)
from diff_vla_wbc.imagined_value_v5 import (                # noqa: E402
    CriticBatch, CriticCfgV5, CriticFeatureAdapter, ValueReadinessGate,
    bootstrap_identifiable, build_value_net, ema_update, mc_returns, td_lambda_targets)
from diff_vla_wbc.isaac_joint import FrameRef               # noqa: E402

GAMMA = 0.99
CRITIC_LEGACY_AUDIT = """# critic legacy audit（规范 §1.1 / §8 M0）

| 项 | 值 | 证据 | 判定 |
|---|---|---|---|
| `bptt1` 的 `critic_grad_through_state` | `False` | `runs/bptt1/v3_config.json`；`diff_vla_wbc/v3_step.py:1332` | FOUND |
| ⇒ 旧 Critic 对 Actor 的贡献 | **恒为 0**（走 `with torch.no_grad()`，`V(x_H)` 是常数） | `v3_step.py:1332-1345` | FOUND |
| 旧 Critic 权重 | `critic_ckpt=None`，从未保存 | `runs/bptt1/args.json` + `ls runs/bptt1` | **NOT_FOUND ⇒ 不可复用** |
| Critic 输入 | `(state 71, ref 71, xi 33)`，无 plan token / 无 history | `diff_vla_wbc/token_critic.py:24-46` | FOUND |
| 时间轴 | `T=40`（VLA chunk）、`H=20`（WM 步）、`K=20`（客户端 `HA_VLA_EXEC_STEPS`） | `rollout.py:366`、`action_provider_supermimic.py:911` | FOUND |
| `K == H` ⇒ 尾部区间 | `K − H = 0` ⇒ `V(y_H)` **不可识别** | 规范 §6.1 | **BLOCKED(BOOTSTRAP_NOT_IDENTIFIABLE)** |

**结论**：v5 的 Critic 是**首次真正接入梯度**，不是"修复旧 bug"；旧 Critic 权重不存在，
**禁止**在文档/日志里写"复用旧 critic / 旧 v_ev 一直很好"。
"""


# --------------------------------------------------------------------------- 工具
def _git_revision() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "UNKNOWN"
    except Exception:  # noqa: BLE001
        return "UNKNOWN"


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, torch.Tensor):
        return float(o.detach().cpu().mean())
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not jsonable: {type(o)}")


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=_json_default))


def _append_jsonl(path: Path, row: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row, default=_json_default) + "\n")


class FrozenParamCtx:
    """冻结参数、**保留** ∂V/∂y（规范 §7.2：不是 `no_grad()`）。"""

    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.flags: list[bool] = []

    def __enter__(self):
        self.flags = [p.requires_grad for p in self.module.parameters()]
        for p in self.module.parameters():
            p.requires_grad_(False)
        return self.module

    def __exit__(self, *exc):
        for p, f in zip(self.module.parameters(), self.flags):
            p.requires_grad_(bool(f))
        return False


def pooled_rms(e: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """``(B,H,29)`` 误差在 mask 上的**池化** RMS（rad）。"""
    m = m.float()
    num = (e.pow(2).sum(-1) * m).sum()
    den = (29.0 * m.sum()).clamp_min(1e-8)
    return (num / den).clamp_min(0.0).sqrt()


def masked_return(env, batch: dict, H: int):
    """`J`（折扣 WM 想象回报）+ rollout（与 `diag.window_returns` 同一实现）。"""
    return d03.window_returns(env.bundle, batch, H, GAMMA, want_actions=True)


# --------------------------------------------------------------------------- 数据
class OnTheFlySource:
    """**每个 optimizer step**：合格帧上均匀抽 B 个起点 → 现场 burn-in → 现场 VLA。"""

    def __init__(self, env, args, seed: int):
        self.env = env
        self.args = args
        self.H = int(args.horizon)
        self.dev = env.device
        # ★ 防泄漏：训练采样只允许评测集的 **train clips**（val/test 的 clip 一帧都不取）
        allowed = None
        if args.eval_set is not None:
            _meta = json.loads((Path(args.eval_set) / "windows_meta.json").read_text())
            _train_names = list(_meta.get("train_clips") or [])
            if not _train_names:
                raise SystemExit("[v5][fail-closed] eval_set 的 windows_meta 没有 train_clips，"
                                 "无法保证训练不泄漏 val/test clip")
            _cid, _names = d03.clip_groups(env.lib)
            _name2cid = {str(nm): i for i, nm in enumerate(_names)}
            missing = [nm for nm in _train_names if nm not in _name2cid]
            if missing:
                raise SystemExit(f"[v5][fail-closed] eval_set 的 train_clips 有 {len(missing)} 个"
                                 f"对不上 motion library 的 clip 名（例：{missing[:3]}）")
            allowed = {_name2cid[nm] for nm in _train_names}
            if len(allowed) != len(_train_names):
                raise SystemExit("[v5][fail-closed] clip 名映射后数量不一致")
        self.w, self.info = d03.eligible_start_mask(
            env, horizon=self.H, drift=float(args.start_max_drift),
            h_tail=int(args.beta_h_tail), min_frame=10, allowed_clips=allowed)
        if allowed is not None and int(self.info["n_eligible"]) <= 0:
            raise SystemExit("[v5] train-clip 过滤后没有合格起点")
        if int(self.info["n_eligible"]) <= 0:
            raise SystemExit("[v5] 合格起点为 0，检查 ref_drift_lib / real_transitions")
        self.gen = torch.Generator(device="cpu").manual_seed(int(seed) + 7919)
        self.vla = d03.FrozenVLASampler(args, env)
        self.n_calls = 0

    def sample(self, batch: int, *, seed: int) -> dict:
        from diff_vla_wbc.burnin import sample_burnin

        idx = torch.multinomial(self.w, int(batch), replacement=True, generator=self.gen)
        self.n_calls += 1
        fr_dev = idx.to(self.dev)
        bi = sample_burnin(
            self.env.bundle, self.env.lib, fr_dev,
            port_tokens=lambda f: self.env.frame_src({"frames": f})[:, 0],
            token_fn=self.env.online.__call__,
            recorded_states=self.env.recorded_states,
            horizon=self.H, k=int(self.args.burnin_steps),
            generator=torch.Generator(device="cpu").manual_seed(int(seed) + self.n_calls),
            perturb=not bool(self.args.no_perturb),
            xy=float(self.args.perturb_xy), yaw=float(self.args.perturb_yaw),
            dq=float(self.args.perturb_dq), dqd=float(self.args.perturb_dqd),
        )
        # 现场 VLA：同一批帧、**每次新 flow-matching 初噪**
        tokens = self.vla.sample(idx, seed=int(seed) * 1_000_003 + self.n_calls * 7919)
        state64 = d03.state64_from(self.env.bundle, bi.state, fr_dev, self.dev,
                                   use_full=bool(self.args.delta_token_fullstate))
        hist = bi.history.permute(1, 0, 2).contiguous()          # (B,10,93)
        return dict(
            idx=idx, frames=idx, tokens=tokens, state=bi.state,
            history=bi.history, xi=bi.xi, prev_action=bi.prev_action,
            refs=bi.references, refs_f=torch.stack([r.f for r in bi.references], 1),
            state64=state64, hist10=hist[:, :, :HIST_DIM].contiguous(),
            burnin_reward=bi.burnin_reward,
        )


def load_eval_set(env, args, H: int) -> dict:
    """固定评测集（`diag build --uniform-frames 1` 的产物）：val / test 各 96 窗口。"""
    es = Path(args.eval_set)
    if not (es / "windows.pt").exists():
        raise SystemExit(f"[v5] 缺 {es/'windows.pt'}：先跑 §8 M2 ① 建评测集")
    shim = SimpleNamespace(windows=es / "windows.pt", vla_tokens=es / "vla_tokens.pt",
                           out=es, tokens="vla")
    ws = d03.load_windows(env, shim, H)
    sp = d03.split_sets(ws)
    meta = dict(getattr(ws, "meta", {}))
    return dict(sets=sp, meta=meta, root=str(es))


class CachedPoolSource:
    """**只用于 pilot / 单测**：从固定窗口池里按索引抽（`data_mode=cached_pool`）。"""

    def __init__(self, env, args, H: int):
        shim = SimpleNamespace(windows=Path(args.windows), vla_tokens=args.vla_tokens,
                               out=Path(args.windows).parent, tokens="vla")
        ws = d03.load_windows(env, shim, H)
        self.sp = d03.split_sets(ws)
        self.pool = self.sp["train"]
        self.dev = env.device
        self.H = int(H)
        self.gen = torch.Generator(device="cpu").manual_seed(int(args.seed) + 7919)
        self.reps = [self.pool.tokens]
        if args.vla_tokens and int(args.noise_cycling):
            vt = torch.load(str(args.vla_tokens), map_location="cpu", weights_only=False)
            if "tokens_vla_rep2" in vt and vt["tokens_vla_rep2"].shape == self.pool.tokens.shape:
                self.reps.append(vt["tokens_vla_rep2"].float())
        self.info = dict(n_pool=len(self.pool), uniform_over_eligible=False,
                         data_mode="cached_pool")

    def sample(self, batch: int, *, seed: int) -> dict:
        idx = torch.randint(0, len(self.pool), (int(batch),), generator=self.gen)
        rep = self.reps[int(torch.randint(len(self.reps), (1,), generator=self.gen))]
        w = self.pool.index(idx)
        b = w.to(self.dev, self.H)
        b["tokens"] = rep[idx].to(self.dev)
        b["hist10"] = w.history[:, :, :HIST_DIM].contiguous().to(self.dev)
        b["refs"] = [FrameRef(b["refs_f"][:, t]) for t in range(self.H)]
        b["idx"] = idx
        b["frames"] = w.frames
        return b


# --------------------------------------------------------------------------- 条件化验收
@torch.no_grad()
def conditional_check(env, ws, actor, *, H: int, cap: float, use_history: bool,
                      mask_full: torch.Tensor, batch: int = 48) -> dict:
    """规范 §11 "学成常量 ⇒ FAIL" 的**在线门禁**：Δ 的窗口/帧间差异 + 输入 shuffle 敏感性。

    判据（建议，写入报告）：`cos(shuffle_z) < 0.9` 且 `delta_std_across_windows > 1e-4`
    且 `frame_to_frame_std > 1e-4`。history 的 cosine 不强求（历史可能确实没有额外信息）。
    """
    dev = env.device
    w = ws.index(slice(0, min(int(batch), len(ws))))
    b = w.to(dev, H)
    B = len(w)
    hist = w.history[:, :, :HIST_DIM].contiguous().to(dev) if use_history else None
    m = mask_full[None, :].expand(B, -1)
    d = actor(b["tokens"], b["state64"], hist, m, cap=float(cap))
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(B, generator=g)
    res = dict(
        delta_std_across_windows=float(d.std(0).mean()),
        delta_mean_norm=float(d.norm(dim=-1).mean()),
        frame_to_frame_std=float((d - d.mean(1, keepdim=True)).std(1).mean()),
    )
    variants = {
        "shuffle_z": (b["tokens"][perm], b["state64"], hist),
        "shuffle_state": (b["tokens"], b["state64"][perm], hist),
        "shuffle_hist": (b["tokens"], b["state64"],
                         None if hist is None else hist[perm]),
        "zero_all": (torch.zeros_like(b["tokens"]), torch.zeros_like(b["state64"]),
                     None if hist is None else torch.zeros_like(hist)),
    }
    for name, (z_, s_, h_) in variants.items():
        d2 = actor(z_, s_, h_, m, cap=float(cap))
        res[f"cos_{name}"] = float(torch.nn.functional.cosine_similarity(
            d2.flatten(1), d.flatten(1), dim=1).mean())
        res[f"maxdiff_{name}"] = float((d2 - d).abs().max())
    res["passed"] = bool(res["cos_shuffle_z"] < 0.9
                         and res["delta_std_across_windows"] > 1e-4
                         and res["frame_to_frame_std"] > 1e-4)
    return res


# --------------------------------------------------------------------------- 评测
@torch.no_grad()
def eval_split(env, ws, actor, *, H: int, cap: float, use_history: bool,
               mask_full: torch.Tensor, batch: int = 48, want_base: bool = False,
               q_base_cache: torch.Tensor | None = None) -> dict:
    """固定窗口评测：`J`、`ΔJ vs Base`、29D rad 动作预算（规范 §0 验收 / §5.5.4）。"""
    dev = env.device
    Js, J0s, dn, stats = [], [], [], []
    q_bases = []
    for s in range(0, len(ws), int(batch)):
        w = ws.index(slice(s, s + int(batch)))
        b = w.to(dev, H)
        hist = w.history[:, :, :HIST_DIM].contiguous().to(dev) if use_history else None
        m = mask_full[None, :].expand(len(w), -1)
        delta = actor(b["tokens"], b["state64"], hist, m, cap=float(cap))
        # 同观测动作预算：记录修正 rollout 的每步 obs93，再用同一 obs + 基础 token 重解
        _rec: list = []
        _oe = env.bundle.executor.act

        def _esc(token, obs93):
            _rec.append(obs93)
            return _oe(token, obs93)

        env.bundle.executor.act = _esc
        try:
            J, extra = masked_return(env, {**b, "tokens": b["tokens"] + delta}, H)
        finally:
            env.bundle.executor.act = _oe
        Js.append(J.detach().cpu())
        dn.append(float(delta.norm(dim=-1)[:, : int(mask_full.sum())].mean()))
        q_corr = q_target_from_action(env.bundle.default_joint_pos,
                                      env.bundle.action_scale,
                                      extra["actions"].float().permute(1, 0, 2))
        if len(_rec) >= H:
            _rows = [q_target_from_action(
                env.bundle.default_joint_pos, env.bundle.action_scale,
                _oe(b["tokens"][:, _ti], _ob.detach()).float())
                for _ti, _ob in enumerate(_rec[:H])]
            q_base_same = torch.stack(_rows, dim=1)                 # (B,H,29)
            e_same = joint_error_rad(q_corr, q_base_same.detach())
            st_same = action_budget_stats(e_same, torch.ones(len(w), H, device=dev))
        else:
            st_same = dict(rms_mrad=float("nan"), peak_mrad=float("nan"),
                           p99_mrad=float("nan"), steps=0.0, samples=0.0)
        msk = torch.ones(len(w), H, device=dev)
        if want_base:
            J0, extra0 = masked_return(env, b, H)
            J0s.append(J0.detach().cpu())
            q_base_dev = q_target_from_action(
                env.bundle.default_joint_pos, env.bundle.action_scale,
                extra0["actions"].float().permute(1, 0, 2))
            q_bases.append(q_base_dev.cpu())
            q_base = q_base_dev
        else:
            if q_base_cache is None:
                raise SystemExit("[v5] eval 需要 base 动作缓存（want_base=True 先跑一次）")
            q_base = q_base_cache[s:s + len(w)].to(dev)
        st_traj = action_budget_stats(joint_error_rad(q_corr, q_base.detach()), msk)
        st_traj.update({f"same_obs_{k}": v for k, v in st_same.items()})
        stats.append(st_traj)
    out = dict(J=torch.cat(Js) if Js else torch.zeros(0),
               delta_norm=float(np.mean(dn)) if dn else 0.0,
               stats={k: float(np.mean([s[k] for s in stats])) for k in stats[0]} if stats else {})
    if want_base:
        out["J_base"] = torch.cat(J0s) if J0s else torch.zeros(0)
        out["q_base"] = torch.cat(q_bases) if q_bases else None
    return out


# --------------------------------------------------------------------------- 主流程
def main() -> int:  # noqa: C901
    ap = argparse.ArgumentParser("train_temporal_ac_bptt_v5")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    # 数据
    ap.add_argument("--on-the-fly", type=int, default=0, help="1=主训练口径（规范 §3.3）")
    ap.add_argument("--eval-set", type=Path, default=None, help="固定评测集目录（build 产物）")
    ap.add_argument("--windows", type=Path, default=None, help="cached_pool：windows.pt")
    ap.add_argument("--vla-tokens", type=Path, default=None, help="cached_pool：vla_tokens.pt")
    ap.add_argument("--bptt-args", type=Path, default=MAIN / "runs/bptt1/args.json")
    ap.add_argument("--motion", type=Path, default=d03._abs(d03.DEFAULTS["motion"]))
    ap.add_argument("--vlm-ckpt", type=Path, default=d03._abs(d03.DEFAULTS["vlm_ckpt"]))
    ap.add_argument("--vlm-path", type=Path, default=d03._abs(d03.DEFAULTS["vlm_path"]))
    ap.add_argument("--vla-batch", type=int, default=64)
    ap.add_argument("--fm-steps", type=int, default=4)
    ap.add_argument("--burnin-steps", type=int, default=10)
    ap.add_argument("--start-max-drift", type=float, default=0.10)
    ap.add_argument("--no-perturb", type=int, default=0)
    ap.add_argument("--perturb-xy", type=float, default=0.05)
    ap.add_argument("--perturb-yaw", type=float, default=0.10)
    ap.add_argument("--perturb-dq", type=float, default=0.05)
    ap.add_argument("--perturb-dqd", type=float, default=0.30)
    ap.add_argument("--noise-cycling", type=int, default=0,
                    help="cached_pool 专用：在 rep1/rep2 间轮换")
    ap.add_argument("--delta-token-fullstate", type=int, default=1,
                    help="state64 = rotlocal6+q+qd（bptt1 DELTA_TOKEN_FULLSTATE=1）")
    # 时间轴
    ap.add_argument("--horizon", type=int, default=20, help="H：可微 WM 步数")
    ap.add_argument("--exec-k", type=int, default=20, help="K：每次 chunk 实际执行的 token 数")
    ap.add_argument("--beta-h-tail", type=int, default=0,
                    help="尾部预留步数（仅 β>0 / S1 用；默认 0）")
    # Actor / 优化
    ap.add_argument("--cap", type=float, default=0.08)
    ap.add_argument("--cap-warmup-updates", type=int, default=100)
    ap.add_argument("--cap-warmup-from", type=float, default=0.04)
    ap.add_argument("--head-lr", type=float, default=3e-4)
    ap.add_argument("--enc-lr", type=float, default=1e-4)
    ap.add_argument("--min-lr-head", type=float, default=3e-5)
    ap.add_argument("--min-lr-enc", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--warmup-updates", type=int, default=50)
    ap.add_argument("--use-history", type=int, default=1)
    ap.add_argument("--init-actor", default="",
                    help="★本实验唯一与 v5 文档不同的地方：从 v5 格式 ckpt 初始化 Actor"
                         "（监督蒸馏预训练），其余超参/损失/结构/流程与文档完全一致")
    ap.add_argument("--actor-kind", default="temporal", choices=["temporal", "pos_template"],
                    help="temporal=v5 条件化 Actor；pos_template=逐帧常量模板（防伪对照 A2）")
    ap.add_argument("--lambda-raw", type=float, default=0.0,
                    help="未 cap 范数 hinge 权重 L_raw=E[max(0,‖u‖/(κ·cap)-1)²]（防 tanh 饱和）")
    ap.add_argument("--raw-norm-kappa", type=float, default=2.0, help="κ：目标 ‖u‖≤κ·cap")
    ap.add_argument("--cond-check-every", type=int, default=0,
                    help="每 N 步做一次条件化检查（Δ 方差 + shuffle 敏感性）；0=关")
    ap.add_argument("--trust-mode", default="same_obs", choices=["same_obs", "traj"],
                    help="动作 trust 的参照：same_obs=同一 SONIC obs 只换 token（规范原意）；"
                         "traj=旧口径（两条 rollout 的动作差，混入轨迹差异）")
    # 损失
    ap.add_argument("--lambda-action", type=float, default=0.010)
    ap.add_argument("--tau-action", type=float, default=0.020, help="rad")
    ap.add_argument("--lambda-smooth", type=float, default=0.002)
    ap.add_argument("--lambda-anchor", type=float, default=0.001)
    # Critic（默认关闭；§5.5.3 / §6.1）
    ap.add_argument("--critic-enabled", type=int, default=0)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--critic-lr", type=float, default=3e-4)
    ap.add_argument("--critic-wd", type=float, default=1e-4)
    ap.add_argument("--critic-clip", type=float, default=1.0)
    ap.add_argument("--critic-updates", type=int, default=4)
    ap.add_argument("--critic-batch", type=int, default=2048)
    ap.add_argument("--beta-warmup-updates", type=int, default=100)
    # 预算
    ap.add_argument("--updates", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-batch", type=int, default=48)
    ap.add_argument("--early-stop-patience", type=int, default=6)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--profile", type=int, default=0)
    a = ap.parse_args()

    if int(a.updates) > 1200:
        raise SystemExit("[v5] updates 上限 1200（规范 §0-A）")
    data_mode = "on-the-fly" if int(a.on_the_fly) else "cached_pool"
    if data_mode == "on-the-fly" and a.eval_set is None:
        raise SystemExit("[v5] --on-the-fly 1 必须给 --eval-set（固定 held-out 窗口）")
    if data_mode == "cached_pool" and a.windows is None:
        raise SystemExit("[v5] cached_pool 模式必须给 --windows")

    a.out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    torch.manual_seed(int(a.seed))
    np.random.seed(int(a.seed) % (1 << 31))
    T0 = time.time()
    d03.log(f"[v5] data_mode={data_mode} seed={a.seed} out={a.out}")
    _write_json(a.out / "config.json", vars(a))
    (a.out / "git_revision.txt").write_text(_git_revision() + "\n")
    (a.out / "critic_legacy_audit.md").write_text(CRITIC_LEGACY_AUDIT)

    # ---------------------------------------------------------------- 环境
    env = d03.load_env(a.device, a.bptt_args)
    H = int(a.horizon)
    sp_eval = load_eval_set(env, a, H) if a.eval_set is not None else None
    pool_meta: dict = {}
    split_manifest: dict = {}
    if data_mode == "on-the-fly":
        src = OnTheFlySource(env, a, int(a.seed))
        pool_meta = dict(src.info, data_mode=data_mode,
                         n_frames=int(env.lib.num_frames),
                         burnin_steps=int(a.burnin_steps), perturb=True,
                         perturb_xy=float(a.perturb_xy), perturb_yaw=float(a.perturb_yaw),
                         perturb_dq=float(a.perturb_dq), perturb_dqd=float(a.perturb_dqd),
                         sampling="torch.multinomial(w, B, replacement=True) over eligible frames",
                         seed=int(a.seed), horizon=H, bptt_args=str(a.bptt_args),
                         vlm_ckpt=str(a.vlm_ckpt),
                         wm_ckpt=str(env.bptt_args.get("wm_ckpt", d03.DEFAULTS["wm_ckpt"])))
        split_manifest = dict(eligible=src.info, eval_set=sp_eval["meta"],
                              data_mode=data_mode,
                              eval_windows=dict(n_val=int(len(sp_eval["sets"]["val"])),
                                                n_test=int(len(sp_eval["sets"]["test"]))),
                              uniform_over_eligible=bool(src.info["uniform_over_eligible"]))
        _append_jsonl(a.out / "train.jsonl", dict(event="eligible", data_mode=data_mode,
                                                  **src.info))
    else:
        src = CachedPoolSource(env, a, H)
        pool_meta = dict(src.info, data_mode=data_mode, windows=str(a.windows),
                         vla_tokens=str(a.vla_tokens))
        split_manifest = dict(data_mode=data_mode, windows=str(a.windows))
        if sp_eval is None:
            sp_eval = dict(sets=src.sp, meta={}, root=str(Path(a.windows).parent))
    _write_json(a.out / "split_manifest.json", split_manifest)
    _write_json(a.out / "windows_meta.json", pool_meta)
    if sp_eval is not None:
        _write_json(a.out / "vla_token_manifest.json", dict(
            eval_set=sp_eval["root"],
            available_manifests=sorted(str(p) for p in Path(sp_eval["root"]).glob("*manifest*.json")),
            on_the_fly=(data_mode == "on-the-fly"),
            note=("on-the-fly：每步用冻结 VLA 现场采样（每次新 FM 初噪）；评测集用缓存 token"
                  if data_mode == "on-the-fly" else "cached_pool：tokens 来自 vla_tokens.pt"),
            sampler=(getattr(src, "vla", None).meta if hasattr(src, "vla") else None)))

    # ---------------------------------------------------------------- Actor
    T_chunk = 40
    K = int(min(a.exec_k, T_chunk))
    mask_full = torch.zeros(T_chunk, device=dev)
    mask_full[:K] = 1.0
    cfg = TemporalActorCfg(chunk_len=T_chunk, cap=float(a.cap),
                           use_history=bool(a.use_history))
    if str(a.actor_kind) == "pos_template":
        actor = PositionTemplateActor(chunk_len=T_chunk, cap=float(a.cap),
                                      init_std=float(cfg.init_std)).to(dev)
    else:
        actor = TemporalDeltaActorV5(cfg).to(dev)
    d03.log(f"[v5] Actor {actor.n_params:,} 参数 d_in={actor.d_in} | T={T_chunk} K={K} H={H} "
            f"| history={'ON' if a.use_history else 'OFF'}")
    # ★ 初始化（唯一偏离：不随机初始化，而是加载监督蒸馏的 Actor）
    if str(getattr(a, "init_actor", "") or ""):
        _ck = torch.load(str(a.init_actor), map_location="cpu", weights_only=False)
        _sd = (_ck.get("delta_token_v5_state") or _ck.get("state_dict") or _ck)
        _res = actor.load_state_dict(_sd, strict=True)
        with torch.no_grad():
            _z = torch.zeros(2, T_chunk, 64, device=dev)
            _s = torch.zeros(2, 64, device=dev)
            _hh = torch.zeros(2, 10, actor.cfg.hist_dim, device=dev)
            _d = actor(_z, _s, _hh, mask_full[None].expand(2, -1), cap=float(a.cap))
        d03.log(f"[v5][init] Actor 从 {a.init_actor} 初始化（strict=True OK）"
                f" ‖Δ‖(零输入)={float(_d.norm(dim=-1).mean()):.5f}"
                f" spec={_ck.get('delta_token_v5_spec', {}) if isinstance(_ck, dict) else {}}")

    enc_params, head_params = [], []
    _enc_prefix = ("token_proj", "pe_chunk", "chunk_encoder", "chunk_norm", "hist_gru", "pe_pos")
    for n_, p_ in actor.named_parameters():
        (enc_params if n_.split(".")[0] in _enc_prefix else head_params).append((n_, p_))
    groups = []
    for name, params, lr, minlr in (("enc", enc_params, float(a.enc_lr), float(a.min_lr_enc)),
                                    ("head", head_params, float(a.head_lr), float(a.min_lr_head))):
        dec = [p for n_, p in params if p.ndim >= 2]
        nod = [p for n_, p in params if p.ndim < 2]
        if dec:
            groups.append(dict(name=f"{name}_decay", params=dec, lr=lr, base_lr=lr,
                               min_lr=minlr, weight_decay=float(a.weight_decay)))
        if nod:
            groups.append(dict(name=f"{name}_nodecay", params=nod, lr=lr, base_lr=lr,
                               min_lr=minlr, weight_decay=0.0))
    opt = torch.optim.AdamW(groups)

    def lr_scale(it: int) -> float:
        w = int(a.warmup_updates)
        if it <= w:
            return max(1e-8, it / max(1, w))
        prog = min(1.0, (it - w) / max(1, int(a.updates) - w))
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    def set_lr(it: int) -> None:
        sc = lr_scale(it)
        for g in opt.param_groups:
            g["lr"] = float(g["min_lr"]) + (float(g["base_lr"]) - float(g["min_lr"])) * sc

    def cap_at(it: int) -> float:
        w = int(a.cap_warmup_updates)
        if w <= 0 or it >= w:
            return float(a.cap)
        f = it / max(1, w)
        return float(a.cap_warmup_from) + f * (float(a.cap) - float(a.cap_warmup_from))

    # ---------------------------------------------------------------- Critic
    critic_enabled = bool(int(a.critic_enabled)) and float(a.beta) > 0
    adapter = online_critic = target_critic = critic_opt = None
    gate = ValueReadinessGate(K=K, H=H, ramp_updates=int(a.beta_warmup_updates))
    beta_cfg = float(a.beta)
    if not bootstrap_identifiable(K, H) and float(a.beta) > 0:
        d03.log(f"[v5][critic] BOOTSTRAP_NOT_IDENTIFIABLE (K==H=={K}) ⇒ β 强制为 0"
                f"（规范 §5.5.3 / §6.1）")
        beta_cfg = 0.0
        critic_enabled = False
    if critic_enabled:
        cc = CriticCfgV5(lr=float(a.critic_lr), weight_decay=float(a.critic_wd),
                         grad_clip=float(a.critic_clip),
                         updates_per_actor=int(a.critic_updates),
                         hist_frames=10, hist_dim=HIST_DIM)
        adapter = CriticFeatureAdapter(cc)
        online_critic = build_value_net(adapter, cc, device=dev)
        target_critic = copy.deepcopy(online_critic).to(dev)
        for p in target_critic.parameters():
            p.requires_grad_(False)
        critic_opt = torch.optim.AdamW(online_critic.parameters(), lr=float(a.critic_lr),
                                       weight_decay=float(a.critic_wd))
        d03.log(f"[v5][critic] enabled beta={beta_cfg} in_dim={adapter.in_dim} "
                f"params={online_critic.n_params:,} K={K} H={H}")
    else:
        d03.log("[v5][critic] OFF（critic.enabled=false, beta=0）"
                + ("" if bootstrap_identifiable(K, H)
                   else f"；BOOTSTRAP_NOT_IDENTIFIABLE(K==H=={K})"))
    _write_json(a.out / "critic_gate.json", dict(
        gate.as_dict(), beta_cfg=beta_cfg, critic_enabled=bool(critic_enabled),
        adapter=(adapter.as_dict() if adapter else None)))

    # ---------------------------------------------------------------- 评测基线
    hist_rows: list[dict] = []
    budget_rows: list[dict] = []
    base_J: dict[str, float] = {}
    base_q: dict[str, torch.Tensor | None] = {}
    val_set = sp_eval["sets"]["val"] if sp_eval is not None else None
    test_set = sp_eval["sets"]["test"] if sp_eval is not None else None
    if val_set is not None:
        for name, ws_e in (("val", val_set), ("test", test_set)):
            r = eval_split(env, ws_e, actor, H=H, cap=float(a.cap),
                           use_history=bool(a.use_history), mask_full=mask_full,
                           batch=int(a.eval_batch), want_base=True)
            base_J[name] = float(r["J_base"].mean())
            base_q[name] = r["q_base"]
            d03.log(f"[v5][base] {name} J={base_J[name]:.5f}（n={len(r['J_base'])}）")

    # ---------------------------------------------------------------- 历史口径对拍（§4.5）
    parity: dict = dict(checked=False)
    if val_set is not None and bool(a.use_history):
        try:
            n_p = min(32, len(val_set))
            w = val_set.index(slice(0, n_p))
            hist10_train = w.history[:, :, :58].contiguous()
            spec_hist = _hist_spec(env, K, H)
            # ★ hist10 的 q 块用 **obs/decoder** 的 default（g1_default_joint_pos），
            #   不是 MDP/SONIC 的 bundle.default_joint_pos（两者差 ≈0.27 rad）
            dq_cpu = g1_default_joint_pos().detach().cpu()
            # 部署路径：把 (q_abs, qd) 换到 **部署序（HA_V31）**，拼成 state64 序列，
            # 再走 `hist10_from_hist64`（内部换回 standard 并减 standard 序 default）
            from diff_vla_wbc.delta_token_io_v5 import (DEPLOY_STATE_JOINT_ORDER,
                                                        joint_permutation)
            _q_abs_std = (hist10_train[:, :, :29] + dq_cpu[None, None]).reshape(-1, 29).numpy()
            _qd_std = hist10_train[:, :, 29:].reshape(-1, 29).numpy()
            _perm = joint_permutation("standard", DEPLOY_STATE_JOINT_ORDER)
            hist64 = torch.zeros(n_p, 10, 64)
            hist64[:, :, 6:35] = torch.from_numpy(_q_abs_std[:, _perm].reshape(n_p, 10, 29))
            hist64[:, :, 35:64] = torch.from_numpy(_qd_std[:, _perm].reshape(n_p, 10, 29))
            hist10_deploy = torch.stack(
                [torch.from_numpy(hist10_from_hist64(hist64[i].numpy(), spec_hist))
                 for i in range(n_p)], 0)
            d_hist = float((hist10_train - hist10_deploy).abs().max())
            state64 = w.state64.cpu()
            d_q = float((hist10_train[:, -1, :29] - (state64[:, 6:35] - dq_cpu[None])).abs().max())
            d_qd = float((hist10_train[:, -1, 29:] - state64[:, 35:64]).abs().max())
            parity = dict(checked=True, n=int(n_p), max_abs_hist10_train_vs_deploy=d_hist,
                          max_abs_q_block_vs_state64=d_q, max_abs_qd_block_vs_state64=d_qd,
                          block_order="[q-default(29) | qd(29)]",
                          frame_order="oldest→newest; [-1] = 决策时刻",
                          passed=bool(max(d_hist, d_q, d_qd) <= 1e-6))
            d03.log(f"[v5][hist] parity max|Δ| train-vs-deploy={d_hist:.2e} "
                    f"q-vs-state64={d_q:.2e} qd-vs-state64={d_qd:.2e} "
                    f"=> {'PASS' if parity['passed'] else 'FAIL'}")
            if not parity["passed"]:
                raise SystemExit("[v5][fail-closed] history10 与部署口径不一致（规范 §4.7）")
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001
            parity = dict(checked=False, error=str(exc))
            d03.log(f"[v5][hist][warn] parity 检查未完成：{exc}")
    _write_json(a.out / "history_parity.json", parity)
    _write_json(a.out / "history_contract.json", dict(
        hist_frames=10, hist_dim=HIST_DIM, block_order="[q-default(29) | qd(29)]",
        frame_order="oldest→newest（[-1] = 决策时刻）",
        train_source="burn-in: bundle.observation 的 q/qd 块",
        deploy_source="客户端 decoder 的 (q-default)/qd 两块 → state_hist_10",
        joint_order="ha_v31（训练侧 obs93 与 state64 逐位同序，规范 §1.3/§4.2）",
        fail_closed="缺 hist10 / 形状不符 / float32 不符 / 关节序不符 ⇒ 抛异常",
        parity=parity))

    # ---------------------------------------------------------------- 训练循环
    best = dict(val_dJ=-1e9, it=-1)
    bad = 0
    beta = 0.0
    t0 = time.time()
    it = 0
    for it in range(1, int(a.updates) + 1):
        set_lr(it)
        cap = cap_at(it)
        _t = [time.time()]
        batch = src.sample(int(a.batch), seed=int(a.seed) + it * 1009)
        _t.append(time.time())
        tokens = batch["tokens"].detach()
        hist10 = batch["hist10"] if a.use_history else None
        m = mask_full[None, :].expand(int(tokens.shape[0]), -1)
        delta, raw_u = actor(tokens, batch["state64"], hist10, m, cap=cap, return_raw=True)
        _t.append(time.time())
        # Δ=0 的动作参照：**同起点、同 SONIC obs**，每步现算（规范 §7.1 注）
        with torch.no_grad():
            _, extra0 = masked_return(env, dict(batch, tokens=tokens), H)
            q_base = q_target_from_action(env.bundle.default_joint_pos,
                                          env.bundle.action_scale,
                                          extra0["actions"].float().permute(1, 0, 2))
        _t.append(time.time())
        # ★ 同观测动作 trust（规范原意）：记录修正 rollout 每一步的 SONIC obs，
        #   再用**同一 obs + 基础 token** 解一次 action 作为参照 —— 只隔离"token 差异"的
        #   直接动作影响，不混入两条 WM 轨迹未来状态不同造成的差异。
        #   ⚠️ SonicIsaacBundle.act 是**类方法**（__setattr__ 会把赋值转发给 base，但读取仍走类），
        #   所以只能拦**执行器级**的 `executor.act(token, obs93)` —— 它正好就是
        #   "token + 观测 → 动作" 的接口，拦截后可以用同一份 obs93 换回基础 token 重解。
        _recorded: list = []
        _orig_exec_act = env.bundle.executor.act
        if str(a.trust_mode) == "same_obs":
            def _rec_act(token, obs93):
                _recorded.append((token.detach(), obs93))
                return _orig_exec_act(token, obs93)

            env.bundle.executor.act = _rec_act
        try:
            J, extra = masked_return(env, dict(batch, tokens=tokens + delta), H)
        finally:
            if str(a.trust_mode) == "same_obs":
                env.bundle.executor.act = _orig_exec_act
        q_corr = q_target_from_action(env.bundle.default_joint_pos,
                                      env.bundle.action_scale,
                                      extra["actions"].float().permute(1, 0, 2))
        if str(a.trust_mode) == "same_obs":
            if not _recorded:
                raise SystemExit("[v5][fail-closed] same_obs trust 没拦到 executor.act 调用")
            with torch.no_grad():
                _base_rows = []
                for _step_i, (_tok, _obs) in enumerate(_recorded[:H]):
                    _a0 = _orig_exec_act(tokens[:, _step_i], _obs.detach())
                    _base_rows.append(q_target_from_action(
                        env.bundle.default_joint_pos, env.bundle.action_scale, _a0.float()))
                q_base_same = torch.stack(_base_rows, dim=1)   # (B,H,29)
            e = joint_error_rad(q_corr, q_base_same.detach())     # (B,H,29) rad（同观测）
            e_traj = joint_error_rad(q_corr, q_base.detach())     # 旧口径，仅诊断
            e_loss = e
        else:
            e = joint_error_rad(q_corr, q_base.detach())          # 旧口径（两轨迹）
            e_traj = e
            e_loss = e
        M_surv = extra["rollout"].survival[:H].permute(1, 0).detach()
        L_act, act_rms = physical_action_trust(e_loss, M_surv, tau=float(a.tau_action))
        L_sm = smoothness_penalty(e_loss, M_surv, tau=float(a.tau_action))
        L_anc = anchor_penalty(delta, m, cap=float(cap))
        # 未 cap 范数 hinge：把 ‖u‖ 按在 κ·cap 附近（避免 tanh 饱和后输入梯度 ~ cap/‖u‖）
        _un = raw_u.norm(dim=-1)
        if float(a.lambda_raw) > 0:
            L_raw = (torch.relu(_un / (float(a.raw_norm_kappa) * float(cap)) - 1.0).pow(2)
                     * m).sum() / m.sum()
        else:
            L_raw = delta.sum() * 0.0
        J_ac = J
        tail_val = 0.0
        if critic_enabled and beta > 0.0:
            y_H = adapter.build(extra["rollout"], CriticBatch(
                ref_states=batch["refs_f"].to(dev), xi=batch["xi"],
                z=tokens + delta.detach(), hist10=hist10), H, K=K, T=T_chunk)
            with FrozenParamCtx(target_critic):
                v_H = target_critic(y_H)
            J_ac = (J + float(beta) * (GAMMA ** H)
                    * extra["rollout"].survival[H].detach() * v_H)
            tail_val = float(v_H.detach().mean())
        loss = (-J_ac.mean() + float(a.lambda_action) * L_act
                + float(a.lambda_smooth) * L_sm + float(a.lambda_anchor) * L_anc
                + float(a.lambda_raw) * L_raw)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gsum = 0.0
        for p in actor.parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                raise SystemExit(f"[v5][fail] update {it}: 梯度非有限（参数 {tuple(p.shape)}）")
            gsum += float(p.grad.norm())
        if not math.isfinite(gsum) or gsum == 0.0:
            raise SystemExit(f"[v5][fail] update {it}: 梯度全为 0 或非有限（{gsum}）")
        _t.append(time.time())
        gn = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), float(a.grad_clip)))
        opt.step()
        _t.append(time.time())

        c_metrics: dict = {}
        if critic_enabled:
            with torch.no_grad():
                ys = [adapter.build(extra["rollout"], CriticBatch(
                    ref_states=batch["refs_f"].to(dev), xi=batch["xi"],
                    z=tokens + delta.detach(), hist10=hist10), t, K=K, T=T_chunk)
                    for t in range(H)]
                rs = [extra["rollout"].rewards[t].detach() for t in range(H)]
                ds = [1.0 - extra["rollout"].survival[t + 1].detach() for t in range(H)]
                y_end = adapter.build(extra["rollout"], CriticBatch(
                    ref_states=batch["refs_f"].to(dev), xi=batch["xi"],
                    z=tokens + delta.detach(), hist10=hist10), H, K=K, T=T_chunk)
                y_all = torch.stack(ys + [y_end], 0)
                v_bar = target_critic(y_all.reshape(-1, adapter.in_dim)).reshape(H + 1, -1)
                G = td_lambda_targets(torch.stack(rs, 0), torch.stack(ds, 0), v_bar,
                                      gamma=GAMMA, lam=0.95)
            for _ in range(int(a.critic_updates)):
                idxc = torch.randint(0, int(G.numel()), (int(a.critic_batch),))
                yt = y_all.reshape(-1, adapter.in_dim)[idxc]
                Gt = G.reshape(-1)[idxc]
                v = online_critic(yt)
                lv = torch.nn.functional.smooth_l1_loss(v, Gt, beta=1.0)
                critic_opt.zero_grad(set_to_none=True)
                lv.backward()
                torch.nn.utils.clip_grad_norm_(online_critic.parameters(), float(a.critic_clip))
                critic_opt.step()
                ema_update(target_critic, online_critic, 0.995)
            c_metrics = dict(v_loss=float(lv.detach()), v_mean=float(v.detach().mean()))

        if int(a.log_every) > 0 and (it % int(a.log_every) == 0 or it == 1):
            row = dict(update=it, data_mode=data_mode, J=float(J.mean()), loss=float(loss),
                       J_ac=float(J_ac.mean()), tail=tail_val, beta=beta,
                       L_act=float(L_act), act_rms_mrad=float(act_rms.mean()) * 1e3,
                       L_smooth=float(L_sm), L_anchor=float(L_anc),
                       L_raw=float(L_raw),
                       act_rms_traj_mrad=float(pooled_rms(e_traj, M_surv)) * 1e3,
                       u_norm_over_cap_p50=float((_un[:, :K] / float(cap)).median()),
                       u_norm_over_cap_p95=float(torch.quantile(
                           (_un[:, :K] / float(cap)).flatten(), 0.95)),
                       frac_u_gt_2c=float((_un[:, :K] > 2.0 * float(cap)).float().mean()),
                       frac_u_gt_3c=float((_un[:, :K] > 3.0 * float(cap)).float().mean()),
                       delta_norm=float(delta.norm(dim=-1)[:, :K].mean()),
                       burnin_reward=(float(batch["burnin_reward"].mean())
                                      if batch.get("burnin_reward") is not None else None),
                       cap=cap, grad_norm=gn, s_per_step=(time.time() - t0) / it,
                       t_data=_t[1] - _t[0], t_base_roll=_t[2] - _t[1],
                       t_roll=_t[3] - _t[2], t_opt=_t[4] - _t[3],
                       **c_metrics)
            hist_rows.append(row)
            _append_jsonl(a.out / "train.jsonl", row)
            d03.log(f"[v5] it={it:4d} J={row['J']:.4f} loss={row['loss']:+.4f} "
                    f"|Δ|={row['delta_norm']:.4f} actRMS={row['act_rms_mrad']:.1f}mrad "
                    f"Lact={row['L_act']:.3f} gn={gn:.2f} cap={cap:.3f} "
                    f"beta={beta:.2f} ({row['s_per_step']:.1f}s/step) "
                    f"[data {row['t_data']:.1f}s base {row['t_base_roll']:.1f}s "
                    f"roll {row['t_roll']:.1f}s opt {row['t_opt']:.1f}s]")

        if (val_set is not None and int(a.cond_check_every) > 0
                and it % int(a.cond_check_every) == 0):
            cc = conditional_check(env, val_set, actor, H=H, cap=float(a.cap),
                                   use_history=bool(a.use_history), mask_full=mask_full,
                                   batch=int(a.eval_batch))
            _append_jsonl(a.out / "cond_check.jsonl", dict(update=it, **cc))
            d03.log(f"[v5][cond] it={it} std_win={cc['delta_std_across_windows']:.2e} "
                    f"std_t={cc['frame_to_frame_std']:.2e} "
                    f"cosZ={cc['cos_shuffle_z']:.4f} cosS={cc['cos_shuffle_state']:.4f} "
                    f"cosH={cc['cos_shuffle_hist']:.4f} "
                    f"=> {'PASS' if cc['passed'] else 'FAIL_CONSTANT?'}")

        if val_set is not None and int(a.eval_every) > 0 and it % int(a.eval_every) == 0:
            v = eval_split(env, val_set, actor, H=H, cap=float(a.cap),
                           use_history=bool(a.use_history), mask_full=mask_full,
                           batch=int(a.eval_batch), q_base_cache=base_q["val"])
            dJ = float(v["J"].mean()) - base_J["val"]
            row = dict(update=it, eval="val", data_mode=data_mode,
                       J=float(v["J"].mean()), J_base=base_J["val"], dJ=dJ,
                       delta_norm=v["delta_norm"], **v["stats"])
            _append_jsonl(a.out / "val.jsonl", row)
            budget_rows.append(dict(update=it, split="val", dJ=dJ, **v["stats"],
                                    delta_norm=v["delta_norm"]))
            d03.log(f"[v5][eval] it={it} val J={row['J']:.5f} dJ={dJ:+.5f} "
                    f"actRMS={v['stats'].get('rms_mrad', 0):.2f}mrad "
                    f"peak={v['stats'].get('peak_mrad', 0):.2f}mrad")
            if dJ > best["val_dJ"] + 1e-7:
                best = dict(val_dJ=dJ, it=it)
                bad = 0
                _save_ckpt(a.out / "checkpoint_actor_best.pt", actor, env, a, K, H, T_chunk,
                           extra=dict(it=it, dJ_val=dJ, cap=float(a.cap), data_mode=data_mode))
            else:
                bad += 1
                if bad >= int(a.early_stop_patience):
                    d03.log(f"[v5] 早停：连续 {bad} 次 val 无改善（it={it}）")
                    break
        if critic_enabled and val_set is not None and it % 50 == 0:
            metrics = _critic_heldout_ev(env, val_set, actor, adapter, target_critic,
                                         H=H, cap=float(a.cap), K=K, T=T_chunk,
                                         use_history=bool(a.use_history), mask_full=mask_full)
            res = gate.update(update=it, **metrics)
            _write_json(a.out / "critic_gate.json", dict(gate.as_dict(), beta_cfg=beta_cfg,
                                                         adapter=adapter.as_dict()))
            if res.status == "PASS" and beta == 0.0:
                d03.log(f"[v5][critic] VALUE_GATE_PASS ⇒ β 开始 0→{beta_cfg} 线性提升")
            elif res.status in ("FAIL", "INCONCLUSIVE"):
                d03.log(f"[v5][critic] VALUE_GATE_{res.status} ⇒ β=0（{res.reasons}）")
            beta = beta_cfg * gate.beta_at(it)

    # ---------------------------------------------------------------- 收尾
    _save_ckpt(a.out / "checkpoint_actor_last.pt", actor, env, a, K, H, T_chunk,
               extra=dict(it=it, data_mode=data_mode))
    test_once: dict = {}
    if test_set is not None:
        te = eval_split(env, test_set, actor, H=H, cap=float(a.cap),
                        use_history=bool(a.use_history), mask_full=mask_full,
                        batch=int(a.eval_batch), q_base_cache=base_q["test"])
        dJ_test = float(te["J"].mean()) - base_J["test"]
        test_once = dict(J=float(te["J"].mean()), J_base=base_J["test"], dJ=dJ_test,
                         delta_norm=te["delta_norm"], **te["stats"],
                         data_mode=data_mode, n=int(len(te["J"])))
        budget_rows.append(dict(update="final", split="test", dJ=dJ_test, **te["stats"],
                                delta_norm=te["delta_norm"]))
        d03.log(f"[v5][test] J={test_once['J']:.5f} (base {base_J['test']:.5f}) "
                f"dJ={dJ_test:+.5f}")
        _write_json(a.out / "test_once.json", test_once)
    if budget_rows:
        keys = sorted({k for r in budget_rows for k in r})
        with (a.out / "action_budget.csv").open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            wr.writerows(budget_rows)
    for cand in (Path(a.out).parent / "wm_horizon_error.csv",
                 Path(a.out).parent.parent / "wm_horizon_error.csv"):
        if cand.exists():
            (a.out / "wm_horizon_error.csv").write_text(cand.read_text())
            break
    for cand in (Path(a.out).parent / "grad_fd.json",
                 Path(a.out).parent.parent / "grad_fd.json"):
        if cand.exists():
            (a.out / "grad_fd.json").write_text(cand.read_text())
            break

    summary = dict(
        data_mode=data_mode, seed=int(a.seed), updates=int(it), best_val=best,
        test_once=test_once, base_J=base_J, T=T_chunk, K=K, H=H, params=int(actor.n_params),
        uniform_over_eligible=bool(pool_meta.get("uniform_over_eligible", False)),
        critic=dict(enabled=bool(critic_enabled), beta_cfg=beta_cfg,
                    status=gate.status,
                    bootstrap_identifiable=bootstrap_identifiable(K, H)),
        history_parity=parity, cap=float(a.cap), wall_clock_s=(time.time() - T0),
    )
    _write_json(a.out / "summary.json", summary)
    _final_report(a, summary, hist_rows, budget_rows)
    d03.log(f"[v5] 完成 {a.out} | best_val={best} | dJ_test={test_once.get('dJ')} "
            f"| {(time.time() - T0)/60:.1f} min")
    return 0


# --------------------------------------------------------------------------- 保存
def _hist_spec(env, K: int, H: int, T: int = 40) -> DeltaTokenSpecV5:
    """hist10 口径用的最小 spec（关节序 + **obs/decoder** 的 default）。

    注意：hist 的 q 块默认角是 `g1_default_joint_pos()`（mjlab KNEES_BENT），
    **不是** `bundle.default_joint_pos`（SONIC/MDP 动作路径），见 `_save_ckpt`。
    """
    return DeltaTokenSpecV5(
        T=int(T), K=int(K), H=int(H), use_history=True,
        meta=dict(hist_default_joint_pos=np.asarray(
            g1_default_joint_pos().numpy(), np.float32).tolist()))


def _save_ckpt(path: Path, actor, env, a, K: int, H: int, T: int, *, extra: dict) -> None:
    if str(getattr(a, "actor_kind", "temporal")) == "pos_template":
        import torch as _t  # noqa: PLC0415
        _payload = dict(actor_kind="pos_template", state=actor.state_dict(),
                        chunk_len=int(T))
        _payload.setdefault("cap", float(a.cap))
        _payload.update(extra)
        _t.save(_payload, str(path))
        return
    spec = spec_v5_from_actor(
        actor, K=int(K), H=int(H),
        meta=dict(
            # hist10 的 q 块偏移（obs/decoder 口径）
            hist_default_joint_pos=np.asarray(
                g1_default_joint_pos().numpy(), np.float32).tolist(),
            # 动作换算 q_target = default + scale·a（MDP/SONIC 口径）
            default_joint_pos=np.asarray(
                env.bundle.default_joint_pos.detach().cpu().numpy(), np.float32).tolist(),
            action_scale=np.asarray(
                env.bundle.action_scale.detach().cpu().numpy(), np.float32).tolist(),
            T=int(T), K=int(K), H=int(H),
            state_joint_order="ha_v31", hist_joint_order="ha_v31",
            hist_clock="control_tick_50hz", hist_frames=10, hist_dim=HIST_DIM,
            data_mode=("on-the-fly" if int(a.on_the_fly) else "cached_pool"),
            seed=int(a.seed), train_commit=_git_revision()))
    spec = with_meta(spec, eval_set=(str(a.eval_set) if a.eval_set else ""))
    payload = save_temporal_actor_v5(path, actor, spec, extra=extra)
    path.with_suffix(path.suffix + ".sha256").write_text(payload["ckpt_sha256"] + "\n")


@torch.no_grad()
def _critic_heldout_ev(env, ws, actor, adapter, target_critic, *, H: int, cap: float,
                       K: int, T: int, use_history: bool, mask_full, batch: int = 32) -> dict:
    """held-out（独立 clip）价值校验：`v_ev` / MAE / 方向一致性 vs **MC 回报**。"""
    dev = env.device
    vals, mcs = [], []
    for s in range(0, len(ws), int(batch)):
        w = ws.index(slice(s, s + int(batch)))
        b = w.to(dev, H)
        hist = w.history[:, :, :HIST_DIM].contiguous().to(dev) if use_history else None
        m = mask_full[None, :].expand(len(w), -1)
        delta = actor(b["tokens"], b["state64"], hist, m, cap=float(cap))
        _, extra = masked_return(env, {**b, "tokens": b["tokens"] + delta}, H)
        ro = extra["rollout"]
        G = mc_returns(ro.rewards[:H].detach(), None, gamma=GAMMA)      # (H,B) 无自举 MC
        for t in range(H):
            y = adapter.build(ro, CriticBatch(
                ref_states=b["refs_f"].to(dev), xi=b["xi"],
                z=b["tokens"] + delta.detach(), hist10=hist), t, K=K, T=T)
            vals.append(target_critic(y).detach().cpu())
            mcs.append(G[t].cpu())
    v = torch.cat(vals)
    g = torch.cat(mcs)
    ss = float(((v - g) ** 2).mean())
    v_ev = 1.0 - ss / max(float(g.var(unbiased=False)), 1e-9)
    return dict(v_ev_heldout=v_ev, mae=float((v - g).abs().mean()),
                mae_const_baseline=float((g - g.mean()).abs().mean()),
                dir_consistency=float(((v - v.mean()) * (g - g.mean()) > 0).float().mean()))


def _final_report(a, summary, hist_rows, budget_rows) -> None:
    t = summary.get("test_once") or {}
    lines = [
        "# v5 主线 run 报告（自动生成）",
        "",
        f"- data_mode: **{summary['data_mode']}**"
        + ("（主训练口径：on-the-fly 均匀抽帧 + 现场 burn-in + 现场 VLA）"
           if summary["data_mode"] == "on-the-fly"
           else "（**只用于 pilot，不得与 on-the-fly 数字混表**，规范 §3.3/§11）"),
        f"- seed: {summary['seed']} | updates: {summary['updates']} | "
        f"T/K/H = {summary['T']}/{summary['K']}/{summary['H']} | 参数 {summary['params']:,}",
        f"- uniform_over_eligible: {summary['uniform_over_eligible']}",
        f"- critic: enabled={summary['critic']['enabled']} status={summary['critic']['status']} "
        f"BOOTSTRAP_IDENTIFIABLE={summary['critic']['bootstrap_identifiable']}",
        f"- history parity: {json.dumps(summary.get('history_parity'), ensure_ascii=False)}",
        "", "## held-out（独立 clip）test", "",
    ]
    if t:
        lines += [f"- J(test) = {t['J']:.5f}，Base = {t['J_base']:.5f}，**ΔJ = {t['dJ']:+.5f}**",
                  f"- 29D 动作增量预算：RMS={t.get('rms_mrad', 0):.2f} mrad，"
                  f"peak={t.get('peak_mrad', 0):.2f} mrad，p99={t.get('p99_mrad', 0):.2f} mrad",
                  f"- ‖Δtoken‖(t<K) mean = {t.get('delta_norm', 0):.4f}"]
    else:
        lines.append("- test 未评测（eval-set 缺失）")
    lines += ["", "## 训练里的 J（WM 想象回报）", ""]
    jrows = [r for r in hist_rows if "J" in r]
    if jrows:
        first, last = jrows[:10], jrows[-10:]
        j0 = float(np.mean([r["J"] for r in first]))
        j1 = float(np.mean([r["J"] for r in last]))
        lines += [f"- 首 10 次 log：J={j0:.5f}；末 10 次：J={j1:.5f}（{(j1/j0-1)*100:+.2f}%）",
                  f"- 末 10 次 actRMS={np.mean([r['act_rms_mrad'] for r in last]):.2f} mrad，"
                  f"|Δ|={np.mean([r['delta_norm'] for r in last]):.4f}，"
                  f"secs/step={np.mean([r['s_per_step'] for r in last]):.1f}"]
    lines += ["", "## 口径与边界", "",
              "- Critic 默认关闭：`K==H==20` ⇒ 尾部区间 0 ⇒ `V(y_H)` 不可识别（规范 §6.1）。",
              "- `t≥K` 的 Δ 恒 0（单测 + 部署契约 `apply_delta_per_frame` 双保险）。",
              "- 起点 = 合格帧上均匀（`multinomial(w,B)`，含 bptt1 的 `w==0` 前一帧回填原坑）。",
              "- 动作 trust 的参照 = 同起点、同 SONIC obs、Δ=0 的 rollout（每步现算）。",
              "", f"wall_clock: {summary.get('wall_clock_s', 0)/60:.1f} min", ""]
    (a.out / "final_report.md").write_text("\n".join(lines))
    (a.out / "reproduce.sh").write_text(
        "#!/usr/bin/env bash\n# 由 train_temporal_ac_bptt_v5.py 自动生成\nset -euo pipefail\n"
        f"WT={ROOT}\nMAIN=${{COMIMIC_MAIN:-{MAIN}}}\ncd \"$WT\"\nsource scripts/harness03_env.sh\n"
        + " ".join(sys.argv) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
