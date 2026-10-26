#!/usr/bin/env python3
"""S3 联合微调「统一口径」审计（启动前必跑）。

一次把**训练侧**和**部署侧**的所有口径逐项对拍，任何一项 FAIL 就不要起训练——
历史上翻过的车（bc/wbc token 空间不一致、H 与重规划周期对不上、回放 consume 快放 3.3×、
WBC 不是 hold 窗口、codec 没接、评测 WM 在漂）都是"两边各说各话"造成的。

用法：
    python3 scripts/s3_audit_config.py
    python3 scripts/s3_audit_config.py --horizon 20 --lambda-r 0.1        # 带即将启动的参数
    python3 scripts/s3_audit_config.py --json                             # 机器可读

退出码：0 = 全部 PASS；1 = 有 FAIL（或 --strict 下有 WARN）。
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path("/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic")
TRAIN_ENTRY = ROOT / "scripts/train_joint_v2.py"
LAUNCHER = ROOT / "scripts/launch_joint_ha_v2.sh"
QUEUE = ROOT / "scripts/s3_queue_lamr_sweep.sh"
REPLAY = ROOT / "scripts/ha_replay_token_server.py"
SONIC_AP = ROOT / "third_party/HumanoidArena/isaaclab_twist2_g1/action_provider/action_provider_sonic.py"
SMM_AP = ROOT / "third_party/HumanoidArena/isaaclab_twist2_g1/action_provider/action_provider_supermimic.py"
TRAINER_V2 = ROOT / "diff_vla_wbc/train_v2.py"
V3_STEP = ROOT / "diff_vla_wbc/v3_step.py"

WBC_C6H = ROOT / "runs/S1_handover/WBC_A20000_wmvel_cmd.pt"
WBC_C6H_MD5 = "e0596acb98957d11931a7854bbe043d3"
LIB_CMD = ROOT / "runs/isaac_motion/ha_cmd/packed.pt"
REAL_TRANSITIONS = ROOT / "runs/wm_real/ha_transitions.pt"
VLM_CKPT = ROOT / "runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000"

CHUNK = 40          # 服务端一次返回的行数（HA 固定）
REPLAN = 20         # 部署重规划周期 = HA_VLA_EXEC_STEPS 默认值

ROWS: list[tuple[str, str, str, str, str]] = []   # (id, 口径项, 期望, 实测, PASS/WARN/FAIL)


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def _arg_call(src: str, flag: str) -> str | None:
    """截出 ``add_argument("flag", ...)`` 的完整实参串（括号匹配，能越过 choices=(...)）。

    只认「flag 是第一个位置参数」的那种调用，避免命中别的参数 help 里提到的同名字符串。
    """
    for m in re.finditer(r"add_argument\(", src):
        j = m.end() - 1
        depth = 0
        for k in range(j, len(src)):
            c = src[k]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    call = src[j + 1:k]
                    head = call.lstrip()
                    if head.startswith('"' + flag + '"') or head.startswith("'" + flag + "'"):
                        return call
                    break
    return None


def _arg_default(src: str, flag: str) -> str | None:
    """取某个 flag 的 default 字面量（先做括号匹配拿到整段实参）。"""
    call = _arg_call(src, flag)
    if call is None:
        return None
    m = re.search(r"default\s*=\s*([^,\n]+)", call)
    if not m:
        return None
    return m.group(1).strip()


def _sh_default(src: str, var: str) -> str | None:
    m = re.search(rf'^{var}=\$\{{{var}:-([^}}]*)\}}', src, re.M)
    return m.group(1).strip() if m else None


def add(cid, item, expect, got, ok, warn=False):
    verdict = "PASS" if ok else ("WARN" if warn else "FAIL")
    ROWS.append((cid, item, expect, str(got), verdict))


# ---------------------------------------------------------------- 1. 想象步数 H
def check_horizon(cli_h: int | None, allow_dev: bool = False) -> int:
    got_entry = _arg_default(_read(TRAIN_ENTRY), "--horizon")
    ok = got_entry is not None and int(got_entry) == REPLAN
    add("H0", "入口 --horizon 默认", f"{REPLAN} (= 部署重规划周期)", got_entry, ok)

    got_launch = _sh_default(_read(LAUNCHER), "H")
    add("H1", "启动器 H 默认", str(REPLAN), got_launch,
        got_launch is not None and int(got_launch) == REPLAN)
    got_queue = _sh_default(_read(QUEUE), "H")
    add("H2", "扫描队列 H 默认", str(REPLAN), got_queue,
        got_queue is not None and int(got_queue) == REPLAN)

    h = REPLAN if cli_h is None else int(cli_h)
    # H 阶梯实验会刻意偏离部署口径（只为测"裁判几步之内可信"），显式放行成 WARN
    add("H3", "本次启动 H", str(REPLAN), h, h == REPLAN, warn=bool(allow_dev))
    return h


# ---------------------------------------------------------------- 2. chunk / 重规划
def check_replan():
    sonic = _read(SONIC_AP)
    m = re.search(r'HA_VLA_EXEC_STEPS",\s*"(\d+)"', sonic)
    got = m.group(1) if m else None
    add("C0", "sonic provider 重规划 (HA_VLA_EXEC_STEPS)", str(REPLAN), got,
        got is not None and int(got) == REPLAN)

    smm = _read(SMM_AP)
    add("C1", "supermimic provider 支持 receding horizon", "读 HA_VLA_EXEC_STEPS", 
        "有" if "HA_VLA_EXEC_STEPS" in smm else "无", "HA_VLA_EXEC_STEPS" in smm)

    train = _read(TRAIN_ENTRY)
    hs = set(re.findall(r"token_horizon=(\d+)", train))
    # 允许 token_horizon=1 的「单帧源」(port_frame_src)；其余必须都是 40
    add("C2", "训练侧读入的 chunk 长度", f"{CHUNK}（单帧源=1 允许）", sorted(hs),
        hs == {str(CHUNK), "1"})

    replay = _read(REPLAY)
    got_consume = _arg_default(replay, "--consume")
    add("C3", "回放服务 --consume 默认", f"{REPLAN} (与部署同口径)", got_consume,
        got_consume is not None and int(got_consume) == REPLAN)


# ---------------------------------------------------------------- 3. 采样 / decode
def check_forward():
    train = _read(TRAIN_ENTRY)
    k = _arg_default(train, "--fm-steps")
    add("F0", "FM 采样步数 K", "4", k, k is not None and int(k) == 4)

    v2 = _read(TRAINER_V2)
    add("F1", "token→动作走 codec.decode（可微反归一化）", "train_v2 里 codec.decode",
        "有" if "self.codec.decode(x_norm)" in v2 else "无",
        "self.codec.decode(x_norm)" in v2)

    add("F2", "summary 的 decode 口径", '"codec"', '"codec"' if '"decode": "codec"' in train else "其它",
        '"decode": "codec"' in train)

    add("F3", "WBC 冻结（J5）", "assert_wbc_frozen + requires_grad_(False)",
        "有" if "def assert_wbc_frozen" in v2 else "无", "def assert_wbc_frozen" in v2)


# ---------------------------------------------------------------- 4. token 前端（S0.4）
def check_token_space():
    train = _read(TRAIN_ENTRY)
    add("T0", "BC 标签 token 空间", "label (= 录制 encoder_latent)", _arg_default(train, "--bc-token-space"),
        _arg_default(train, "--bc-token-space") == '"label"')
    add("T1", "WBC token 窗口", "hold", _arg_default(train, "--wbc-token-window"),
        _arg_default(train, "--wbc-token-window") == '"hold"')

    motion = _arg_default(train, "--motion-packed")
    add("T2", "参考库 = 命令流 ha_cmd", "PACKED_CMD_MOTION", motion, motion is not None and "PACKED_CMD" in motion)
    try:
        sys.path.insert(0, str(ROOT))
        from diff_vla_wbc.sonic_frontend import library_kind  # noqa: PLC0415
        kind = library_kind(LIB_CMD)
    except Exception as e:                                   # noqa: BLE001
        kind = f"<err {type(e).__name__}>"
    add("T3", "库 kind", "command", kind, kind == "command")

    win = lib = "?"
    try:
        import torch  # noqa: PLC0415
        ck = torch.load(str(WBC_C6H), map_location="cpu", weights_only=False)
        prov, a = (ck.get("provenance") or {}), (ck.get("args") or {})
        lib = str(prov.get("library") or a.get("motion_packed") or "")
        win = str(prov.get("token_window") or a.get("token_window") or "").lower() or "?"
    except Exception as e:                                   # noqa: BLE001
        win = f"<err {type(e).__name__}>"
    add("T4", "WBC ckpt 的 token_window", "hold", win, win == "hold")


# ---------------------------------------------------------------- 5. 执行器（两侧同一个）
def check_executor():
    ok_file = WBC_C6H.exists()
    md5 = _md5(WBC_C6H) if ok_file else "-"
    add("E0", "C6h WBC 存在 + md5", WBC_C6H_MD5, md5, ok_file and md5 == WBC_C6H_MD5)

    # 训练/队列/闭环三处必须指同一个 ckpt
    queue = _read(QUEUE)
    launch = _read(LAUNCHER)
    ref = "runs/S1_handover/WBC_A20000_wmvel_cmd.pt"
    add("E1", "队列默认 WBC", ref, "有" if ref in queue else "有(变量)", ref in queue)
    add("E2", "启动器默认 WBC（环境变量覆盖队列的）", "由 WBC_CKPT 传入",
        "有" if "WBC_CKPT=${WBC_CKPT:-" in launch else "无", "WBC_CKPT=${WBC_CKPT:-" in launch)

    add("E3", "起始 VLA", "checkpoint-8000", VLM_CKPT.exists(), VLM_CKPT.exists())

    ref_win = ref.split("/")[-1]
    cl_scripts = list((ROOT / "scripts").glob("*closedloop*.sh")) + list((ROOT / "scripts").glob("*oracle*.sh"))
    hits = [p.name for p in cl_scripts if "WBC_A20000_wmvel_cmd" in _read(p) or ref_win in _read(p)]
    add("E4", "闭环脚本与训练同一个 WBC", ref_win, hits or "未引用 C6h", bool(hits), warn=not hits)


# ---------------------------------------------------------------- 6. 训练配方
def check_recipe(h: int, lam_r: float | None, allow_dev: bool = False):
    train = _read(TRAIN_ENTRY)
    v2 = _read(TRAINER_V2)
    v3 = _read(V3_STEP)

    burn = _arg_default(train, "--burnin-steps")
    add("R0", "burn-in 步数", "10", burn, burn is not None and int(burn) == 10)

    add("R1", "DR 默认开启", "True", _arg_default(train, "--dr"),
        (_arg_default(train, "--dr") or "").lower() in ("true", "1"))
    pn, pm = _arg_default(train, "--dr-p-nominal"), _arg_default(train, "--dr-p-mid")
    rest = None
    try:
        rest = round(1 - float(pn) - float(pm), 2)
    except Exception:                                        # noqa: BLE001
        pass
    add("R2", "DR 分桶 nominal/mid/hard", "0.25 / 0.45 / 0.30",
        f"{pn} / {pm} / {rest}", pn == "0.25" and pm == "0.45" and rest == 0.30)
    xy = _arg_default(train, "--dr-xy-max")
    add("R3", "DR 位移上限", "0.20 m", xy, xy == "0.20")

    add("R4", "critic 默认开启", "True", _arg_default(train, "--critic"),
        "action=\"store_true\", default=True" in train.replace("'", '"'))
    gam = _arg_default(train, "--gamma")
    lam = _arg_default(train, "--lam")
    boot = None
    try:
        boot = round(float(gam) ** h, 4)
    except Exception:                                        # noqa: BLE001
        pass
    # gamma^H 与 H 绑定：H 阶梯实验里它必然不等于 0.8179，降级成 WARN
    add("R5", "critic bootstrap gamma^H", "0.99^20 ≈ 0.8179", f"gamma={gam} lam={lam} → {boot}",
        gam == "0.99" and boot is not None and abs(boot - 0.8179) < 2e-3, warn=bool(allow_dev))

    add("R6", "token BC 关闭（v3）", "hand-only mask[...,64:78]",
        "有" if "hand_mask[..., 64:78]" in v3 else "无", "hand_mask[..., 64:78]" in v3)

    add("R7", "评测用固定 WM 快照", "_eval_wm = copy.deepcopy", 
        "有" if "_eval_wm = _copy.deepcopy" in train else "无", "_eval_wm = _copy.deepcopy" in train)

    add("R8", "真实转移（WM 保鲜 / recorded 起点）", "ha_transitions.pt",
        f"{REAL_TRANSITIONS.exists()} ({REAL_TRANSITIONS.stat().st_size/2**20:.0f} MB)"
        if REAL_TRANSITIONS.exists() else "缺失", REAL_TRANSITIONS.exists())

    add("R9", "手部监督权重 lambda_hand", "0.2", _arg_default(train, "--lambda-hand"),
        _arg_default(train, "--lambda-hand") == "0.2")

    if lam_r is not None:
        add("R10", "本次启动 lambda_r", "0(A0) / 0.1(A1)", lam_r, lam_r in (0.0, 0.1))


def check_v4(launcher_src: str) -> None:
    """v4 口径：奖励去速度 / grounded critic / 动作空间约束。"""
    import re as _re

    def _sh(var):
        m = _re.search(rf'^{var}=\$\{{{var}:-([^}}]*)\}}', launcher_src, _re.M)
        return m.group(1).strip() if m else None

    add("V0", "奖励去速度通道 (reward_vel_scale)", "0",
        _sh("REWARD_VEL_SCALE"), _sh("REWARD_VEL_SCALE") == "0")
    add("V1", "critic 监督来源", "real（真录制 return-to-go）",
        _sh("CRITIC_SOURCE"), _sh("CRITIC_SOURCE") == "real")
    add("V2", "动作空间约束 eps", "0.30（≈1.4× 底模自身动作偏差 0.21）",
        _sh("ACT_EPS"), _sh("ACT_EPS") == "0.30")
    add("V3", "动作空间约束权重", "1.0", _sh("ACT_WEIGHT"), _sh("ACT_WEIGHT") == "1.0")
    add("V4", "token 空间旧约束已关", "trust 0/0",
        f'{_sh("TRUST_EPS")}/{_sh("TRUST_WEIGHT")}',
        _sh("TRUST_EPS") == "0" and _sh("TRUST_WEIGHT") == "0")
    rr = ROOT / "runs/wm_real/real_returns_vs0.pt"
    add("V5", "真回报文件（vel_scale=0 版）", "real_returns_vs0.pt",
        f'{rr.exists()} ({rr.stat().st_size/2**20:.0f} MB)' if rr.exists() else "缺失", rr.exists())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--lambda-r", type=float, default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--allow-horizon-deviation", action="store_true",
                    help="H 阶梯实验：允许 --horizon != 20（只降级成 WARN）")
    ap.add_argument("--strict", action="store_true", help="WARN 也算失败")
    a = ap.parse_args()

    h = check_horizon(a.horizon, allow_dev=a.allow_horizon_deviation)
    check_replan()
    check_forward()
    check_token_space()
    check_executor()
    check_recipe(h, a.lambda_r, allow_dev=a.allow_horizon_deviation)
    check_v4(_read(ROOT / "scripts/s3_launch_one.sh"))

    n_fail = sum(r[4] == "FAIL" for r in ROWS)
    n_warn = sum(r[4] == "WARN" for r in ROWS)
    if a.json:
        print(json.dumps({"rows": ROWS, "fail": n_fail, "warn": n_warn}, ensure_ascii=False, indent=1))
    else:
        w = [max(len(str(r[i])) for r in ROWS + [("ID", "口径项", "期望", "实测", "判定")]) for i in range(5)]
        w[1] = min(w[1], 34)
        sep = "+".join("-" * (x + 2) for x in w)
        print("\n" + "=" * (sum(w) + 14))
        print(" S3 联合微调 · 统一口径审计")
        print("=" * (sum(w) + 14))
        print("| " + " | ".join(f"{t:<{w[i]}}" for i, t in
                                enumerate(("ID", "口径项", "期望", "实测", "判定"))) + " |")
        print(sep)
        for r in ROWS:
            mark = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌"}[r[4]]
            print("| " + " | ".join(f"{str(v)[:w[i]]:<{w[i]}}" for i, v in enumerate(r[:4]))
                  + f" | {mark} {r[4]:<4} |")
        print(sep)
        print(f" 合计 {len(ROWS)} 项：PASS {len(ROWS)-n_fail-n_warn} / WARN {n_warn} / FAIL {n_fail}")
        print("=" * (sum(w) + 14) + "\n")
        if n_fail:
            print("❌ 有口径不一致，**不要启动训练**。先修上面 FAIL 的项：")
            for r in ROWS:
                if r[4] == "FAIL":
                    print(f"   - [{r[0]}] {r[1]}：期望 {r[2]}，实测 {r[3]}")
            print()

    bad = n_fail + (n_warn if a.strict else 0)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
