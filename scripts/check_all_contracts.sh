#!/usr/bin/env bash
# **一把梭契约门**（2026-11-04）：采集 / 训练 / 评估之前先跑这个，任何一条 FAIL 就别往下走。
#
#   bash scripts/check_all_contracts.sh
#   bash scripts/check_all_contracts.sh runs/a_prime_sonic_exc    # 指定分片目录（默认：现役两份语料）
#   SKIP_PIN=1 bash scripts/check_all_contracts.sh                  # 跳过 pin 契约（要 Isaac 环境）
#
# 锁死的历史 bug（每一条都真实发生过，详见 scripts/verify_contracts.py 头部表）：
#   1. 数据集里没有转移（states == next_states）
#   2. 关节顺序：standard 与 SONIC 是两个不同的 29 序，用错 = 动作打到错关节
#   3. 段重置把机器人瞬移到错位姿态（"读回偏差 0.0000" 是循环论证）
#   4. SONIC decoder 的 obs q/qd 块用错序 ⇒ 闭环 50 步内发散
#   5. ref 布局 [p,quat,v,w,q,qd] vs 扁平 [p,quat,q,v,w,qd]，混用 ⇒ 假漂移 0.38 rad
#   6. WM 刷新没播种 ⇒ 同数据跑两次 qd_corr 差 0.16
#   7. 联合微调 act hinge 饱和（frac_out=1.0）⇒ 300 步白跑
#   8. 联合微调没加载刷新过的 WM（WM_CKPT 默认空）
#   9. 并行 Isaac 撞共享内存 ⇒ rc=0 但没产物（只用 rc 判断就会永久漏采）
set -uo pipefail
C=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SHARDS=${1:-$C/runs/a_prime_sonic_exc}
# ★ 2026-11-05：默认分片目录改成**当前在用的**语料。
#   原默认 `runs/a_prime_sonic` 是历史目录（「没有摔倒」门 z_min=0.238 < 0.25，里面有倒地帧），
#   它又没被 round-5 任何一步用到，结果只是让整把梭门永远红着——而"永远红的门"等于没有门。
#   已移入 `runs/_quarantine/`（含 README）。传参时只查传入目录；用默认时同时查现役两份。
ACTIVE_CORPORA=("$SHARDS")
if [[ -z "${1:-}" ]]; then
  ACTIVE_CORPORA+=("$C/runs/a_prime_sonic_leash")
fi
FAIL=0

say() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }

say "1/4 离线契约（录制数据 ↔ 库 ↔ 常量 ↔ 关节序）"
for _corp in "${ACTIVE_CORPORA[@]}"; do
  echo "-- 分片目录: $_corp"
  python3 "$C/scripts/verify_contracts.py" --shards "$_corp" || FAIL=1
done

if [[ "${SKIP_PIN:-0}" != "1" ]]; then
  say "2/4 闭环 pin 契约（场景/资产/初始位姿/编解码器 sha256）"
  python3 "$C/scripts/pin_ha_closedloop.py" --check || FAIL=1
else
  say "2/4 闭环 pin 契约：跳过（SKIP_PIN=1）"
fi

say "3/4 微调数据/配置卫生"
python3 - "$C" "$SHARDS" <<'PY'
import sys
from pathlib import Path
import torch
ROOT = Path(sys.argv[1]); shard_dir = Path(sys.argv[2])
bad = []
# 只对**当前在用的**数据目录做硬门禁；历史目录只做信息展示（它们可能本来就该被淘汰）
cand = [shard_dir, Path("runs/a_prime_sonic"), Path("runs/a_prime_seg20")]
seen = set()
for d in cand:
    d = d if d.is_absolute() else ROOT / d
    if str(d) in seen:
        continue
    seen.add(str(d))
    hard = (d == shard_dir)
    p = d / "filtered.pt"
    tag = "HARD" if hard else "info"
    if not p.exists():
        print(f"  [{tag}] {d.name}/filtered.pt 不存在")
        continue
    t = torch.load(p, map_location="cpu", weights_only=False)
    n = int(t["states"].shape[0])
    gap = float((t["states"] - t["next_states"]).abs().max())
    zmin = float(t["states"][:, 2].min())
    ok = (gap > 1e-2) and (zmin > 0.20)
    print(f"  [{'PASS' if ok else ('FAIL' if hard else 'warn')}] {tag} {d.name}: "
          f"rows={n} max|s-s'|={gap:.2f} z_min={zmin:.3f}")
    if hard and not ok:
        bad.append(f"{d.name}: gap={gap:.2f} z_min={zmin:.3f}")
for f in ("runs/S1_handover/WBC_A24000_wmvel_cmd.pt",):
    if not (ROOT / f).exists():
        bad.append(f"缺 {f}")
if bad:
    print("  ❌ " + "; ".join(bad)); sys.exit(1)
PY
[[ $? -ne 0 ]] && FAIL=1

# ---- 部署侧 state46 布局（2026-11-04 闭环"原地站桩"的根因，见
#      docs/CLOSEDLOOP_INFERENCE_FIX_20261104.md §2.1）--------------------------
# 用训练数据 g1_state_46 反推：服务端拼出来的 46 维必须与它逐位一致，
# 否则 right_arm 会被 left_hand 顶掉（21/46 维错），VLA 只吐常量 token、机器人不动。
say "3b/4 部署 state46 布局对拍训练数据"
python3 - <<'PY'
import sys
from pathlib import Path
ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT))
from scripts.ha_vla_latent64_server import assert_state_layout_matches_training_data as chk
for layout in ("legacy",):
    try:
        chk(layout)
    except SystemExit as exc:
        print(f"  [FAIL] {layout}: {exc}")
        sys.exit(1)
# 反向断言：错布局必须**被拒**（否则校验本身失效，等于没锁）
try:
    chk("train_v1")
except SystemExit:
    print("  [PASS] 反向断言：train_v1 被拒绝（校验有效）")
else:
    print("  [FAIL] train_v1 竟然通过了 —— 布局校验已失效")
    sys.exit(1)
PY
[[ $? -ne 0 ]] && FAIL=1

# ---- 启动脚本在 `set -e` 下不许"静默退出" ----------------------------------
# 事故 2026-11-05：`$([[ "$TERM_OFF" == "1" ]] && echo --term-off)` 在 TERM_OFF≠1 时
# 返回 1；launch 脚本是 `set -euo pipefail`，而它出现在 `_ARGS=(... $(...))` 里 ⇒
# 整条赋值失败 ⇒ **进程零报错退出 rc=1**，日志停在一行无关的 [deps_env] 上。
# 那次连吃 4 个对照臂全军覆没，还被误当成"跑得很快"。静态锁死：所有这种替换必须 `|| true`。
say "3c/4 启动脚本的命令替换在 set -e 下安全"
bash -n scripts/launch_joint_ha_v2.sh || { echo "  [FAIL] launch_joint_ha_v2.sh 语法错"; FAIL=1; }
python3 - <<'PY'
import re
import sys
from pathlib import Path

bad = []
for f in ("scripts/launch_joint_ha_v2.sh", "scripts/run_joint_ft_round5.sh"):
    src = Path(f).read_text()
    for m in re.finditer(r"\$\([^)]*\[\[[^)]*\)", src):
        line = src[:m.start()].count("\n") + 1
        if "|| true" not in m.group(0):
            bad.append(f"{f}:{line}: {m.group(0)[:70]}")
if bad:
    print("  [FAIL] 这些命令替换没加 `|| true`（set -e 下会让启动静默退出）:")
    for b in bad:
        print("    " + b)
    sys.exit(1)
print("  [PASS] 命令替换都带 `|| true`（TERM_OFF=0 的启动也安全）")
PY
[[ $? -ne 0 ]] && FAIL=1

say "4/4 结论"
if [[ "$FAIL" != "0" ]]; then
  echo "❌ 契约门没过 —— 不要采集 / 训练 / 评估 / 交付"
  exit 1
fi
echo "✅ 全部契约成立，可以继续"
