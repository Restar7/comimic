"""训练起点 = **B 线采集里真实发生过的重规划状态**（评审 §3 / S3 闭环的前提）。

为什么要它
----------
录制状态上 VLA 本来就执行得很好（lockdown §5：达到 GT 的 97.2%），梯度几乎没有信号；
真正的漂移只发生在**闭环**里：执行一段 → 状态偏了 → 下一次决策看到的是偏了的状态。
这些"偏了的状态"只存在于 B 线 dump 里，录制数据里没有。

所以想象训练的起点必须是 B 线里真实发生过的重规划时刻：

* ``state``       = 那一刻机器人的真实状态（71 维 standard 序）
* ``history``     = 那一刻 WBC 看到的 10 帧观测（用真实状态 + **真实下发动作** 重建）
* ``frames``      = 那一刻的真实**库行号**（奖励参考从这一帧开始，不偏移）
* ``references``  = 从该帧起 H 个参考（与训练/评测同一个 bundle）

与 ``burnin.sample_burnin`` 的区别：那边是"从参考状态出发、让冻结 WBC 自己走 K 步"来伪造
一个起点；这里是**直接用真实起点**，不做 burn-in，也不再叠加 DR 扰动（漂移已经是真的）。

数据来源（Online 模式）
-----------------------
``SM_DUMP_STATES`` 落盘的 dump：``states``(T,71, **asset/HA 序**) / ``actions``(T,29, standard,
即策略原始输出 a_norm) / ``q_target_applied`` / ``tokens`` / ``frames``(T, **真实库行号**)。
**必须是 online 模式**采的 dump（npy 回放模式的 ``frames`` 是步号，不能当库行号用）——
默认严格校验 ``meta["SM_TOKEN_SOURCE"] == "online"``。

已知局限
--------
VLA 的**图像**输入仍然是该库行对应的录制 ego 帧（B 线当时实时渲染的画面没有落盘，
server 的 ``--log-dir`` 那次没开）。本体感觉通道（A1）用的是真实状态，所以
"看不到偏差"的问题在本体感觉上已经解决；图像通道是文档里记录的 ① 未解项。
"""

from __future__ import annotations

import glob
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch

HISTORY = 10
NUM_JOINTS = 29
XI_DIM = 33
STATE_DIM = 71


TOKEN_DIM = 64


def glob_match_path(path: str, pattern: str) -> bool:
    """按**相对或绝对** pattern 匹配一个（通常是 resolve() 后的**绝对**）路径。

    2026-10-31 的坑：`--b-fresh-glob runs/B_*/vla_*_f*_r*/dump/*.pt` 是相对仓库根的，
    而 `episode["path"]` 是 `Path(f).resolve()` 的**绝对**路径 ⇒ `fnmatch(绝对, 相对)` 恒 False，
    "新一轮数据 ×N" 在 d1–d3 全部**静默退化**成均匀采样（只在日志里 warn 一行，没人看）。
    这里对相对 pattern 补一条 `*/` 前缀（fnmatch 的 `*` 可以跨 `/`），再兜底匹配 basename。
    """
    import fnmatch as _fn  # noqa: PLC0415

    p = (pattern or "").strip()
    if not p or not path:
        return False
    cands = [p]
    if not p.startswith("/"):
        _rel = p[2:] if p.startswith("./") else p
        cands.append("*/" + _rel)
    if any(_fn.fnmatch(path, c) for c in cands):
        return True
    return _fn.fnmatch(Path(path).name, p)


@dataclass
class BStartBatch:
    frames: torch.Tensor            # (B,) 真实库行号（重规划时刻）
    state: torch.Tensor             # (B,71) standard
    history: torch.Tensor           # (H_in,B,W) oldest→newest
    obs_hist: torch.Tensor          # (B, H_in*W) term-major
    prev_action: torch.Tensor       # (B,29) 该时刻之前**最后一次下发的原始动作**
    xi: torch.Tensor                # (B,33)
    references: list = field(default_factory=list)
    teacher_states: torch.Tensor | None = None
    source_rows: torch.Tensor | None = None   # (B,) 诊断：取自哪个 dump 的哪一行
    # 2026-10-29：采集时的真值（策略自己下发的 token / 接下来 horizon 步的实际漂移）
    self_tokens: torch.Tensor | None = None   # (B,64)
    self_tokens_has: torch.Tensor | None = None  # (B,) 1=有真标签 0=零占位（2026-10-30）
    realized_err: torch.Tensor | None = None
    # rotlocal6 用：每个样本所属回合的**起始根姿态**（wxyz, (B,4)）
    start_quat0: torch.Tensor | None = None  # (B,)


def state_to_standard(s: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """dump 的 71 维（q/qd 为 asset 序）→ standard 序；root 各块与顺序无关。"""
    q = s[..., 7:36][..., perm]
    qd = s[..., 42:71][..., perm]
    return torch.cat([s[..., :7], q, s[..., 36:42], qd], dim=-1)


class BStartSource:
    """把一批 B 线 dump 变成"重规划时刻"起点采样器。"""

    def __init__(
        self,
        pattern: str | Path,
        bundle,
        library,
        *,
        stride: int = 20,
        horizon: int = 20,
        history: int = HISTORY,
        device=None,
        order: str = "ha_v31",
        strict_online: bool = True,
        split: str = "all",          # all / train / eval（按行号取模 5 切开）
        log: Callable[[str], None] = print,
        # 2026-10-29：用采集时的**真实闭环结果**做监督。
        # self_targets = build_b_self_targets.py 的产物：{dump 文件名: {tok,err,frames}}
        # max_err = 只保留 "接下来 horizon 步实际漂移 ≤ max_err" 的重规划时刻（None=不筛）
        self_targets: dict | None = None,
        max_err: float | None = None,
        self_span: int = 40,          # token BC 要 (B,T,64) 的 chunk，T=40（head 的 action_horizon）
        max_e_inst: float | None = None,   # 只保留"瞬时 root 误差 ≤ 该值"的起点（奖励核有梯度的区间）
        # ★ 2026-10-31：**新一轮数据加权**。混全池训练时，本轮刚采的（on-policy、和当前策略同代）
        #   数据应该被更频繁地采样。fresh_glob 匹配到的 episode 权重 = fresh_weight，其余 = 1。
        fresh_glob: str | None = None,
        fresh_weight: float = 1.0,
    ) -> None:
        self.self_targets = self_targets or None
        self.max_err = None if max_err is None else float(max_err)
        self.self_span = int(self_span)
        self.max_e_inst = None if max_e_inst is None else float(max_e_inst)
        self.fresh_glob = fresh_glob
        self.fresh_weight = float(fresh_weight)
        self._n_inst_filtered = 0
        # 2026-10-29：直接拿 library 的参考张量算瞬时误差，不依赖任何 sidecar 文件
        self._ref_lib = None
        if self.max_e_inst is not None:
            _r = getattr(library, "ref", None)
            if torch.is_tensor(_r):
                self._ref_lib = _r.detach().float().cpu()
            else:
                log("[b-start][warn] library 没有 .ref ⇒ --b-start-max-err 筛选失效")
        self._n_filtered = 0
        self.bundle = bundle
        self.library = library
        self.history = int(history)
        self.horizon = int(horizon)
        self.stride = max(1, int(stride))
        self.device = torch.device(device or bundle.device)
        self.log = log
        self.episodes: list[dict] = []

        import joint_contract as JC  # noqa: PLC0415  (仓库根在 sys.path 上)

        perm = torch.as_tensor(JC.permutation(order, "standard"), dtype=torch.long)
        self.order = str(order)
        self.split = str(split)
        if self.split not in ("all", "train", "eval"):
            raise SystemExit(f"[b-start] split 只能是 all/train/eval，收到 {self.split!r}")

        pats = str(pattern)
        files = sorted(glob.glob(pats))
        if not files:
            raise SystemExit(f"[b-start] 没有匹配到 dump：{pats}")
        n_rows = 0
        for f in files:
            d = torch.load(f, map_location="cpu", weights_only=False)
            S = d.get("states")
            A = d.get("actions")
            F = d.get("frames")
            if S is None or A is None or F is None:
                log(f"[b-start] SKIP {Path(f).name}: 缺 states/actions/frames")
                continue
            T = int(S.shape[0])
            if T <= self.history:
                log(f"[b-start] SKIP {Path(f).name}: 只有 {T} 步（< history {self.history}）")
                continue
            meta = d.get("meta") or {}
            cfg = meta.get("cfg") if isinstance(meta, dict) else {}
            src = str((cfg or {}).get("SM_TOKEN_SOURCE", ""))
            fsrc = str(meta.get("frames_source", "")) if isinstance(meta, dict) else ""
            if strict_online:
                if src == "npy":
                    raise SystemExit(
                        f"[b-start] {Path(f).name} 是 npy 回放模式：frames 是步号、不是库行号，"
                        "不能当训练起点。"
                    )
                if src not in ("online", "http"):
                    raise SystemExit(
                        f"[b-start] {Path(f).name} 的 token 源 {src!r} 无法确认 frames 是库行号；"
                        "确认后再用 --b-start-nonstrict 放行。"
                    )
                if fsrc != "library_row":
                    log(f"[b-start] WARN {Path(f).name}: 没有 frames_source 标记（旧 dump）；"
                        "假定 frames 是库行号（仅当该集从 clip 起点开始且不超长时成立）")
            fmax = int(F.max())
            n_lib = int(getattr(library, "num_frames", 0) or 0)
            if n_lib and fmax >= n_lib:
                raise SystemExit(f"[b-start] {Path(f).name}: frames.max={fmax} 超出库 {n_lib} 行")
            S = S.float()
            S = state_to_standard(S, perm)
            if int(F.numel()) != T:
                # vla 臂只在**重规划时刻**记 frames（execute_steps=20 tick 一个，600 tick ⇒ 30 个），
                # 而 states/actions 是每 tick 的（600）⇒ 按段展开到 600 个 tick。
                # 2026-10-29：不展开就在 `F.long().reshape(T)` 上炸 `shape '[600]' is invalid for input of size 30`。
                n_f = int(F.numel())
                _stride_f = max(1, (T + n_f - 1) // n_f)
                _plan_of_tick = torch.clamp(torch.arange(T) // _stride_f, max=n_f - 1)
                F = F[_plan_of_tick]
            ep = {
                "name": Path(f).name,
                "path": str(Path(f).resolve()),   # 完整路径（fresh 加权要按轮次目录匹配）
                "states": S,                                  # (T,71) standard
                "acts": A.float().reshape(T, -1)[:, :NUM_JOINTS],  # (T,29) 策略原始输出
                "frames": F.long().reshape(T),
                "n": T,
                # 2026-10-31（A5/rotlocal6）：**本回合第一帧的根姿态**。
                #   线上协议里 state64[:6] = rot6d(conj(heading(q0)) * q_now)，
                #   训练侧必须用同一个 q0（每集自己的起点），否则就是训练/部署错配。
                "q0": S[0, 3:7].float().clone(),
            }
            self.episodes.append(ep)
            # 重规划时刻：t % stride == 0 且 t >= history（需要 10 帧观测 + 10 个 last_action）
            t0 = self.history
            ts = [t for t in range(t0, T) if (t % self.stride) == 0]
            if self.split != "all":
                # 每 5 个重规划时刻留 1 个给评测（与 stride 无关、与 t 的具体取值无关）
                _is_eval = [((t // self.stride) % 5 == 4) for t in ts]
                ts = [t for t, ev in zip(ts, _is_eval) if (ev == (self.split == "eval"))]
            # ---- 2026-10-29：挂上"采集时的真值"（策略自己的 token + 实际漂移）----
            if self.self_targets is not None:
                _nm = str(Path(f).resolve())   # 与 build_b_self_targets.py 的 key 一致
                _st = self.self_targets.get(_nm)
                if _st is not None:
                    ep["self_tok"] = _st["tok"].float()      # (T,64)
                    ep["realized_err"] = _st["err"].float()  # (T,)
                    ep["e_inst"] = _st.get("e_inst")
                else:
                    self.log(f"[b-start] WARN {_nm}: 没有真值标签，按不筛处理")
            if self.max_e_inst is not None and self._ref_lib is not None:
                _lib = self._ref_lib
                _n = int(_lib.shape[0])
                _st = ep["states"]
                _fr = ep["frames"]
                _keep = []
                for t in ts:
                    f0 = int(_fr[t].reshape(-1)[0])
                    f0 = 0 if f0 < 0 else (_n - 1 if f0 >= _n else f0)
                    e_inst = float((_st[t][:3].float() - _lib[f0][:3]).norm())
                    if e_inst <= self.max_e_inst:
                        _keep.append(t)
                self._n_inst_filtered += len(ts) - len(_keep)
                ts = _keep
            if self.max_err is not None and ep.get("realized_err") is not None:
                _e = ep["realized_err"]
                keep = [t for t in ts if float(_e[t]) <= self.max_err]
                self._n_filtered += len(ts) - len(keep)
                ts = keep
                if not ts:
                    self.log(f"[b-start] WARN {Path(f).name}: 全部时刻都超过 max_err={self.max_err}，跳过该集")
            ep["ts"] = ts
            n_rows += len(ts)
        if n_rows == 0:
            raise SystemExit(f"[b-start] 匹配到 {len(files)} 个文件，但没有可用的重规划时刻")
        self.n_rows = n_rows
        # ★ 2026-10-29 修：这个清理原来被 `if self.max_err is not None` 包着，
        #   于是「没设 max_err」或「空集是别的原因造成的（太短/stride/horizon）」时
        #   空 episode 会留在 self.episodes 里，sample() 撞上 torch.randint(0,0) 直接崩：
        #   random_ expects 'from' to be less than 'to', but got from=0 >= to=0
        #   现在无条件清理（没空集时是 no-op）。
        _empty = [e for e in self.episodes if not e["ts"]]
        if _empty:
            self.episodes = [e for e in self.episodes if e["ts"]]
            n_rows = sum(len(e["ts"]) for e in self.episodes)
            log(f"[b-start] 丢弃 {len(_empty)} 条空 episode（没有任何可用重规划时刻）")
        if not self.episodes or n_rows == 0:
            raise SystemExit("[b-start] 清理后一条 episode 都不剩；检查 B_GLOB / 阈值 / stride")
        if self.max_err is not None:
            log(f"[b-start] 按真实结果筛选：max_err={self.max_err} m，"
                f"筛掉 {self._n_filtered} 个时刻，剩 {n_rows} 个")
            if n_rows == 0:
                raise SystemExit(
                    f"[b-start] max_err={self.max_err} 把所有时刻都筛掉了；"
                    "把阈值放宽（看 build_b_self_targets.py 打印的分位数）")
        log(f"[b-start] {len(self.episodes)} 条 episode / {n_rows} 个重规划时刻 "
            f"(stride={self.stride}, history={self.history}, order={self.order}, "
            f"split={self.split}, horizon={self.horizon})")

        # ★ 2026-10-31：新一轮数据加权在 **init 时算一次**（原来放在 sample() 里，用
        #   fnmatch(绝对 path, 相对 glob) ⇒ 恒不匹配 ⇒ d1/d2/d2b/d3 的 "×3 新数据" 全部静默失效）。
        #   匹配为 0 且是训练侧 ⇒ 直接失败，不再允许"只 warn 一行"混过去。
        self._fresh_idx: list[int] = []
        if self.fresh_glob and abs(float(self.fresh_weight) - 1.0) > 1e-9:
            self._fresh_idx = [
                i for i, e in enumerate(self.episodes)
                if glob_match_path(str(e.get("path") or e.get("name", "")), self.fresh_glob)
            ]
            log(f"[b-start] fresh 加权 {self.fresh_glob} ⇒ "
                f"{len(self._fresh_idx)}/{len(self.episodes)} 条 episode ×{self.fresh_weight:g}"
                f"（split={self.split}）")
            if not self._fresh_idx:
                _eg = [str(e.get("path", "")) for e in self.episodes[:2]]
                _msg = (f"[b-start] fresh_glob={self.fresh_glob} 一条 episode 都没匹配到；"
                        f"示例 path={_eg}")
                if self.split == "train":
                    raise SystemExit(_msg + " —— 训练侧不允许静默丢掉新数据加权")
                log(_msg)

    # ------------------------------------------------------------------ 采样
    def sample(self, batch: int, generator: torch.Generator | None = None) -> BStartBatch:
        picks: list[tuple[int, int]] = []
        # 兜底：只在**有可用时刻**的 episode 里抽（空集的权重为 0）
        valid = [i for i, e in enumerate(self.episodes) if e["ts"]]
        if not valid:
            raise RuntimeError("[b-start] 所有 episode 都没有可用重规划时刻，无法采样")
        w = torch.zeros(len(self.episodes), dtype=torch.double)
        w[valid] = 1.0
        # 新一轮数据加权：init 时已算好（self._fresh_idx），这里只查表。
        for i in self._fresh_idx:
            w[i] = self.fresh_weight
        for _ in range(int(batch)):
            ei = int(torch.multinomial(w, 1, generator=generator).item())
            ts = self.episodes[ei]["ts"]
            ti = int(torch.randint(0, len(ts), (1,), generator=generator).item())
            picks.append((ei, ts[ti]))

        st_hist, a_hist, f_hist, frame0, state, prev_a = [], [], [], [], [], []
        q0_list = []
        _self_tok, _self_err, _self_has = [], [], []
        for ei, t in picks:
            ep = self.episodes[ei]
            lo = t - self.history + 1
            st_hist.append(ep["states"][lo: t + 1])          # (H,71)
            a_hist.append(ep["acts"][lo - 1: t])             # (H,29) 每帧的 last_action
            f_hist.append(ep["frames"][lo: t + 1])           # (H,)
            state.append(ep["states"][t])
            prev_a.append(ep["acts"][t - 1])
            frame0.append(ep["frames"][t])
            q0_list.append(ep.get("q0", ep["states"][0, 3:7]))
            # ★ 2026-10-30 修：**每个窗口都必须占一行**，缺标签时用 0 占位。
            #   原来只在有 self_tok 时才 append ⇒ 长度可能小于 B（实测 B=16 而 self_tokens=15），
            #   于是 v3_step 里 `tokens - prior_t` 尺寸不匹配直接崩：
            #   "The size of tensor a (16) must match the size of tensor b (15)"。
            #   触发条件很常见：**本轮刚采的 episode 还没进 self_targets**。
            if ep.get("self_tok") is not None:
                # BC 目标要 (T,64) 的 chunk：取 t 起 self_span 步，尾部用最后一帧补齐
                _st = ep["self_tok"]
                _L = min(self.self_span, int(_st.shape[0]) - t)
                _ck = _st[t: t + _L]
                if _L < self.self_span:
                    _pad = _ck[-1:].expand(self.self_span - _L, -1)
                    _ck = torch.cat([_ck, _pad], 0)
                _self_tok.append(_ck)
                _self_err.append(ep["realized_err"][t])
                _self_has.append(1.0)
            else:
                _self_tok.append(torch.zeros(self.self_span, TOKEN_DIM))
                _self_err.append(torch.tensor(float("inf")))
                _self_has.append(0.0)
        st = torch.stack(st_hist).float().to(self.device)     # (B,H,71)
        ac = torch.stack(a_hist).float().to(self.device)      # (B,H,29)
        fh = torch.stack(f_hist).long().to(self.device)      # (B,H)
        frames = torch.stack(frame0).long()                  # (B,)
        state = torch.stack(state).float().to(self.device)   # (B,71)
        prev_a = torch.stack(prev_a).float().to(self.device) # (B,29)
        b = int(st.shape[0])

        from .burnin import _flat_tuple  # noqa: PLC0415
        from .isaac_joint import default_xi  # noqa: PLC0415

        xi = default_xi(b, self.device)
        xi_rep = xi[:, None].expand(b, self.history, XI_DIM).reshape(-1, XI_DIM)
        try:
            # **后端无关**的入口：两个 bundle 都实现了 obs_from_states(flat, a, xi, frames)。
            # 不要用 `mdp.obs(flat_state_tuple(...))` —— 那个 tuple 里 R 是 3×3 矩阵，
            # mjlab 侧无法从 tuple 还原扁平 state（2026-10-28 实测 RuntimeError:
            # "Tensors must have same number of dimensions: got 2 and 3"）。
            if hasattr(self.bundle, "obs_from_states"):
                obs = self.bundle.obs_from_states(
                    st.reshape(-1, STATE_DIM), ac.reshape(-1, NUM_JOINTS),
                    xi_rep, fh.reshape(-1),
                ).reshape(b, self.history, -1)
            else:
                obs = self.bundle.mdp.obs(
                    _flat_tuple(st.reshape(-1, STATE_DIM)),
                    ac.reshape(-1, NUM_JOINTS),
                    xi_rep,
                    fh.reshape(-1),
                ).reshape(b, self.history, -1)
        except AttributeError:
            # 没有 mdp 的 bundle：退回逐帧 observation（只支持 93 维，ref=None）
            rows = []
            for i in range(b):
                rows.append(torch.stack([
                    self.bundle.observation(st[i, k], ac[i, k], xi[i])
                    for k in range(self.history)
                ]))
            obs = torch.stack(rows)
        history = obs.permute(1, 0, 2).contiguous()          # (H,B,W)
        obs_hist = self.bundle.flatten_history(history)

        motion = self.library
        # 后端无关：Isaac/trackerlab 的 `motion.ref` 是张量（有 .device）；
        # mjlab 的 `MotionLibrary.ref` 是**方法**，设备在 `motion.device` 上。
        _dev = getattr(motion, "device", None)
        if _dev is None:
            _ref = getattr(motion, "ref", None)
            _dev = _ref.device if torch.is_tensor(_ref) else torch.device("cpu")
        frames_dev = frames.to(_dev)
        refs = self.bundle.teacher_references(motion, frames_dev, self.horizon)
        teacher = self.bundle.teacher_states(motion, frames_dev, self.horizon)
        return BStartBatch(
            frames=frames_dev,
            state=state.to(self.device),
            history=history.to(self.device),
            obs_hist=obs_hist.to(self.device),
            prev_action=prev_a.to(self.device),
            xi=xi,
            references=refs,
            teacher_states=teacher,
            source_rows=torch.tensor([ei * 1000000 + t for ei, t in picks], dtype=torch.long),
            self_tokens=(torch.stack(_self_tok).float().to(self.device)
                         if _self_tok else None),
            # 1=有真标签，0=占位；v3 的信任域只在这些位置上算 hinge
            self_tokens_has=(torch.tensor(_self_has, dtype=torch.float32).to(self.device)
                             if _self_has else None),
            realized_err=(torch.stack(_self_err).float().to(self.device)
                          if _self_err else None),
            start_quat0=(torch.stack(q0_list).float().to(self.device)
                         if q0_list else None),
        )


__all__ = ["BStartBatch", "BStartSource", "state_to_standard"]
