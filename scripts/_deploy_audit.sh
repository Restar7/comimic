#!/usr/bin/env bash
# 部署口径审计的**共享**小工具（会话 01）。用法：`source scripts/_deploy_audit.sh`。
#
# 为什么需要：评审 §5.6/§7 记录了两类假失败 —— ①叶子脚本单独跑时 SONIC 环境变量没人
# 导出，静默落到 SuperMimic provider（正式 eval 是由 r3_sr_queue.sh 等**调用方**导出的，
# 所以历史上大部分结果确实是 SONIC；出问题的是"直接调叶子脚本"这条路）；
# ②参考库/checkpoint 路径是默认值，事后无法从产物反查。这里把"这次到底跑的是什么"写进**日志和 JSON**，
# 并在期望与实际不符时**非零退出**（不静默 fallback）。
#
# 提供：
#   audit_init <dir>                    创建审计目录
#   audit_kv <key> <value>              打印并写键值
#   audit_file <key> <path>             记录路径 + 字节数 + sha256（缺失记 MISSING）
#   audit_python                        python 解释器/版本/关键依赖
#   audit_provider_check <log> <expect> 跑完后核对 provider（sonic|supermimic|any）
#   audit_delta_required <head_ckpt>    Δtoken checkpoint ⇒ 必须 SONIC（否则 FAIL）

AUDIT_DIR="${AUDIT_DIR:-}"
_AUDIT_TSV=""

audit_init() {
  AUDIT_DIR="${1:-$AUDIT_DIR}"
  [[ -n "$AUDIT_DIR" ]] || { AUDIT_DIR="$PWD/runs/harness_01_deploy/audit"; }
  mkdir -p "$AUDIT_DIR" || return 1
  _AUDIT_TSV="$AUDIT_DIR/deploy_audit.tsv"
  : > "$_AUDIT_TSV"
  echo "[audit] dir=$AUDIT_DIR"
}

audit_kv() {
  local k="$1"; shift
  local v="$*"
  printf '[audit] %s=%s\n' "$k" "$v"
  [[ -n "$_AUDIT_TSV" ]] && printf '%s\t%s\n' "$k" "$v" >> "$_AUDIT_TSV"
  return 0
}

audit_file() {
  local k="$1" p="$2"
  if [[ -z "$p" ]]; then audit_kv "$k" "EMPTY"; return 0; fi
  if [[ ! -e "$p" ]]; then audit_kv "$k" "MISSING:$p"; return 0; fi
  local n s
  n=$(stat -c%s "$p" 2>/dev/null || echo "?")
  if [[ -d "$p" ]]; then
    s="(dir)"
  else
    s=$(sha256sum "$p" 2>/dev/null | cut -d' ' -f1)
  fi
  audit_kv "$k" "path=$p bytes=$n sha256=${s:-ERR}"
}

audit_python() {
  local py="${1:-python3}"
  audit_kv "python" "$(command -v "$py" 2>/dev/null) -> $(readlink -f "$(command -v "$py" 2>/dev/null)" 2>/dev/null)"
  "$py" - <<'PY' 2>/dev/null || true
import platform, sys
print(f"[audit] python_version={platform.python_version()} exe={sys.executable}")
for m in ("torch", "numpy", "onnxruntime", "onnx", "transformers"):
    try:
        mod = __import__(m)
        print(f"[audit] dep_{m}={getattr(mod, '__version__', 'unknown')}")
    except Exception as exc:  # noqa: BLE001
        print(f"[audit] dep_{m}=MISSING({type(exc).__name__})")
PY
}

#: 从日志判定实际 provider：数标记行，只有一种出现才算确定。
audit_provider_check() {
  local log="$1" expect="${2:-any}" actual="unknown" ns nm
  if [[ ! -f "$log" ]]; then
    actual="no_log"
  else
    ns=$(grep -c "\[SonicActionProvider\]" "$log" 2>/dev/null || true)
    nm=$(grep -c "\[SuperMimic\]" "$log" 2>/dev/null || true)
    ns=${ns:-0}; nm=${nm:-0}
    if [[ "$ns" -gt 0 && "$nm" -eq 0 ]]; then actual="sonic"
    elif [[ "$nm" -gt 0 && "$ns" -eq 0 ]]; then actual="supermimic"
    elif [[ "$ns" -eq 0 && "$nm" -eq 0 ]]; then actual="unknown"
    else actual="mixed(sonic=$ns,supermimic=$nm)"; fi
  fi
  audit_kv "provider_actual" "$actual"
  audit_kv "provider_expected" "$expect"
  if [[ "$expect" == "any" ]]; then
    [[ "$actual" == "unknown" || "$actual" == "no_log" ]] && {
      echo "[audit][FAIL] 日志里没有任何 provider 标记（$log）—— 不能判定执行器，结论不可用" >&2
      return 2
    }
    return 0
  fi
  if [[ "$actual" != "$expect" ]]; then
    echo "[audit][FAIL] 期望 provider=$expect，实际=$actual（log=$log）" >&2
    echo "[audit][FAIL] 这正是评审 §7 记录的假失败来源：没人导出 HA_ALLOW_SONIC 就会落到 SuperMimic" >&2
    echo "[audit][FAIL] （正式 eval 由 r3_sr_queue.sh 等**调用方**导出；直接调叶子脚本则没有兜底）。" >&2
    echo "[audit][FAIL] 修法：SONIC=1 跑（r3_sr_cell.sh / ha_isaac_closedloop.sh 会导出三件套）" >&2
    return 2
  fi
  echo "[audit][OK] provider=$actual"
  return 0
}

#: 如果 head checkpoint 里有 delta_token_state ⇒ Δtoken 已启用 ⇒ 必须用 SONIC decoder，
#: 因为 Δ 是在 SONIC decoder 的 token→动作映射上训出来的（另一个执行器下方向不成立）。
audit_delta_required() {
  local head="${1:-}"
  [[ -z "$head" || "$head" == "none" || ! -f "$head" ]] && return 0
  local has
  has=$(python3 - "$head" <<'PY' 2>/dev/null || echo "?"
import sys, torch
d = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(1 if isinstance(d, dict) and d.get("delta_token_state") is not None else 0)
PY
)
  audit_kv "head_has_delta_token" "${has:-?}"
  if [[ "$has" == "1" && "${HA_ALLOW_SONIC:-0}" != "1" && "${ALLOW_DELTA_WITH_SUPERMIMIC:-0}" != "1" ]]; then
    echo "[audit][FAIL] $head 含 delta_token_state（Δtoken 已启用），但 HA_ALLOW_SONIC != 1" >&2
    echo "[audit][FAIL] ⇒ 这次会用 SuperMimic 执行器跑，Δ 的语义在它上面不成立，成功率没有可比性。" >&2
    echo "[audit][FAIL] 用 SONIC=1 重跑；只做对照实验时显式 ALLOW_DELTA_WITH_SUPERMIMIC=1。" >&2
    return 2
  fi
  return 0
}

#: 参考库 clip↔行号一致性（评审批次里 pp_box 从 246502 行开始，而脚本默认 0）。
audit_lib_clip_row() {
  local lib="$1" clip_pat="$2" start="${3:-0}"
  [[ -n "$lib" && -f "$lib" ]] || { audit_kv "lib_clip_check" "SKIP(no_lib)"; return 0; }
  python3 - "$lib" "$clip_pat" "$start" <<'PY'
import sys, torch
lib, pat, start = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = torch.load(lib, map_location="cpu", weights_only=False)
names, starts = list(d["names"]), list(d["starts"])
hit = [(n, s) for n, s in zip(names, starts) if pat in n]
if not hit:
    print(f"[audit][WARN] lib={lib} 里没有任何 clip 名包含 {pat!r}")
    sys.exit(0)
n0, s0 = hit[0]
ok = (s0 == start)
print(f"[audit] lib_clip_first={n0} first_start_row={s0} requested_start_row={start} "
      f"n_clips={len(hit)} match={'YES' if ok else 'NO'}")
if not ok:
    print(f"[audit][FAIL] SM_TOKEN_START={start} 但 {n0} 从第 {s0} 行开始 ⇒ "
          f"参考运动与手的 token 会错位（评审 §5.6 的 'frame 索引对不上'）。"
          f" 修法：SM_TOKEN_START={s0}", file=sys.stderr)
    sys.exit(3)
PY
}
