#!/usr/bin/env bash
# 下载 BONES-SEED 的 G1 动作包（23.5 GB）到工作区，并解压校验 142,220 条 CSV。
#
# 前置（缺一不可，否则 403）：
#   1) HF 账号已在 https://huggingface.co/datasets/bones-studio/seed 点过
#      "Agree and access repository"（要填姓名/单位/用途 + 勾选 license）；
#   2) token 具备 gated 仓库读权限
#      （classic 的 Read token 自带；fine-grained 要勾 "Read access to contents of all
#        public gated repos you can access"）。
#
# 用法：
#   HF_TOKEN=hf_xxx bash scripts/fetch_bones_seed.sh [目标目录]
#   不传 HF_TOKEN 时退回到 $HF_HOME/token（或 ~/.cache/huggingface/token）。
set -euo pipefail

DEST="${1:-/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/datasets/bones-seed/v1}"
REPO="bones-studio/seed"
FILE="g1.tar.gz"
URL="https://huggingface.co/datasets/${REPO}/resolve/main/${FILE}"
EXPECT_BYTES=23499973647
EXPECT_CSV=142220

die() { echo "ERROR: $*" >&2; exit 1; }

mkdir -p "$DEST"
cd "$DEST"

if [[ -z "${HF_TOKEN:-}" ]]; then
  for f in "${HF_HOME:-$HOME/.cache/huggingface}/token" "$HOME/.cache/huggingface/token"; do
    [[ -s "$f" ]] && HF_TOKEN="$(cat "$f")" && break
  done
fi
[[ -n "${HF_TOKEN:-}" ]] || die "没有 HF token：export HF_TOKEN=hf_xxx 后重跑"

# 0) 权限自检：403 = 没同意协议，或 token 没有 gated 读权限
code=$(curl -s -o /dev/null -w '%{http_code}' -I -L -H "Authorization: Bearer ${HF_TOKEN}" "$URL" || true)
echo "HEAD ${URL} -> ${code}"
case "$code" in
  200|302) ;;
  401|403) die "无权访问 gated 仓库：① 去数据集页点 Agree；② 换成带 gated 读权限的 token" ;;
  *) die "网络异常（HTTP ${code}）" ;;
esac

# 1) 下载（可断点续传；已存在同大小文件会秒过）
if [[ -f "$FILE" && "$(stat -c%s "$FILE")" -eq "$EXPECT_BYTES" ]]; then
  echo "已存在完整 $FILE（$EXPECT_BYTES B），跳过下载"
else
  hf download "$REPO" "$FILE" --repo-type dataset --local-dir "$DEST"
fi

size=$(stat -c%s "$FILE" || echo 0)
[[ "$size" -eq "$EXPECT_BYTES" ]] || die "$FILE 大小不对：$size != $EXPECT_BYTES（可能被截断，重跑本脚本续传）"

# 2) 解压（顶层应是 g1/csv/{date}/{motion}.csv）
echo "解压前 3 个条目："; tar -tzf "$FILE" | head -3
tar -xzf "$FILE"

# 3) 校验
n=$(find "$DEST/g1" -name '*.csv' 2>/dev/null | wc -l)
echo "g1 CSV 数量：$n（期望 $EXPECT_CSV）"
[[ "$n" -eq "$EXPECT_CSV" ]] || die "CSV 数量不对，检查解压目录布局"
du -sh "$DEST"
echo "OK: $DEST/g1"
echo "下一步（数据管线）见 docs/BONES_SEED_H200X2.md §3"
