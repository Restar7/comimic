#!/usr/bin/env bash
# 共享依赖环境（P0-1 修复，2026-10-28 换机后）。
#
# 背景：`third_party/av_py311` / `third_party/skimage_py311` 是给 **python3.11** 准备的副本，
# 里面的编译扩展是 `*.cpython-311-*.so`。而本机的训练 python 是 `/usr/bin/python3` = **3.12**
# ⇒ 用 PYTHONPATH 把这两个目录放在最前面会**遮蔽**系统里装好的版本，然后 `_ccomp` 之类的
# cp311 扩展加载不了：
#
#   ModuleNotFoundError: No module named 'skimage.measure._ccomp'
#
# 现象就是训练在 `load_gr00t_processor` 就崩（albumentations -> skimage.measure.label）。
#
# 正确做法：先**探测**当前 python3 是否已经有可用的 av + skimage（含编译扩展）；
# 有就完全不加 vendored 目录，没有才退回（那时跑的应该是 3.11）。
#
# 用法：脚本里 `source scripts/deps_env.sh` 之后直接用 $PYTHONPATH。
#   COMIMIC 必须已经设置。

if [[ -z "${COMIMIC:-}" ]]; then
  echo "[deps_env] 需要先设置 COMIMIC" >&2
  return 1 2>/dev/null || exit 1
fi

# 两个包**分别**探测：av 的 wheel 是 abi3（3.11/3.12 都能用），skimage 的编译扩展是
# 按版本绑死的（cp311 ≠ 3.12）。合在一起探测会因为其中一个缺失而把两个都退回，等于白修。
_DEPS_VENDOR=""
if ! python3 -c "import av" >/dev/null 2>&1; then
  _DEPS_VENDOR="${_DEPS_VENDOR}$COMIMIC/third_party/av_py311:"
  echo "[deps_env] av 不在系统 python3 ⇒ 用 third_party/av_py311（abi3，版本无关）" >&2
fi
if ! python3 -c "from skimage.measure import label" >/dev/null 2>&1; then
  _DEPS_VENDOR="${_DEPS_VENDOR}$COMIMIC/third_party/skimage_py311:"
  echo "[deps_env][WARN] 系统 python3 缺 skimage 编译扩展 ⇒ 退回 third_party/skimage_py311" \
       "（只在 python3.11 下可用；修复：pip install scikit-image imageio tifffile）" >&2
fi
# 2026-11-03：本机 python3 = conda isaaclab **3.11**；而 third_party/gr00t_deps/av 是
# 3.12 构建（_core.abi3.so 依赖 PyErr_GetRaisedException）。deps_env 最后会把 gr00t_deps
# 放进 PYTHONPATH 且排在 site-packages 之前 ⇒ 必须把 py311 的 av 前置，否则训练在
# EgoViewImages 读图时 ImportError。旧日志里同一台机器就是走 av_py311 的。
if python3 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)"; then
  case "$_DEPS_VENDOR" in
    *av_py311*) ;;
    *) _DEPS_VENDOR="${_DEPS_VENDOR}$COMIMIC/third_party/av_py311:" ;;
  esac
  echo "[deps_env] py3.11 ⇒ 前置 third_party/av_py311（gr00t_deps/av 是 3.12 构建）" >&2
fi
if [[ -z "$_DEPS_VENDOR" ]]; then
  echo "[deps_env] av+skimage 都来自系统 python3（含编译扩展）✓" >&2
fi

# 2026-11-06：`third_party/gr00t_deps/skimage` 被换成了 **cp311** 构建（05:06），
# 而本机 python3 = 3.12 ⇒ 一旦 gr00t_deps 进了 PYTHONPATH，就会用那份坏的 skimage，
# 报 `ModuleNotFoundError: No module named 'skimage.measure._ccomp'`
# （训练在 load_gr00t_processor 就崩：albumentations → skimage.measure.label）。
# 上面的探测是在**没有** gr00t_deps 的路径下做的，探不出来 ⇒ 这里单独兜一层：
#   py3.12 时把 py312_shims（里面 skimage -> 系统 0.26.0 的 cp312 构建）排在 gr00t_deps 前面。
_DEPS_SHIMS=""
if python3 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)"; then
  if [[ -e "$COMIMIC/third_party/py312_shims/skimage" ]]; then
    _DEPS_SHIMS="$COMIMIC/third_party/py312_shims:"
    echo "[deps_env] py3.12 ⇒ 前置 third_party/py312_shims（遮住 gr00t_deps 里 cp311 的 skimage）" >&2
  fi
fi

export PYTHONPATH="${_DEPS_VENDOR}$COMIMIC:$COMIMIC/Isaac-GR00T:${_DEPS_SHIMS}$COMIMIC/third_party/gr00t_deps:$COMIMIC/third_party/gr00t_deps/_noflash${PYTHONPATH:+:$PYTHONPATH}"
export NO_ALBUMENTATIONS_UPDATE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
