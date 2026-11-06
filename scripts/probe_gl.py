#!/usr/bin/env python3
"""Probe which MuJoCo GL backend can actually render on this box (CPU only, ~20 s).

``closed_loop_vla_supermimic.py`` now tries ``egl`` then ``osmesa`` automatically, but when
neither works you want to know *why* on this specific machine, without loading the 3B VLA.
Run this on the target host:

    python3 scripts/probe_gl.py

It prints one ``[ok]``/``[fail]`` line per backend and, at the end, the line to paste into
the round script, e.g.::

    export MUJOCO_GL=osmesa            # 或者 SM_GL_BACKEND=osmesa

Exit code is 0 when at least one backend renders.
"""
from __future__ import annotations

import os
import sys

# 必须探测**闭环真正会用的那个 mujoco**：闭环通过 PYTHONPATH 拿到 gr00t_deps 里 vendored 的
# 那一份（和系统 python 里的 3.16 行为不同）。这里手动把它插到最前，和 run_j1_rounds.sh 一致。
_HERE = os.path.dirname(os.path.abspath(__file__))
_VENDORED = os.path.join(os.path.dirname(_HERE), "third_party", "gr00t_deps")
if os.path.isdir(os.path.join(_VENDORED, "mujoco")) and _VENDORED not in sys.path:
    sys.path.insert(0, _VENDORED)

BACKENDS = ("egl", "osmesa", "glfw")
XML = "<mujoco><worldbody><body><geom size='.1'/></body></worldbody></mujoco>"


def purge() -> None:
    for name in [m for m in list(sys.modules) if m.split(".")[0] in ("mujoco", "OpenGL")]:
        del sys.modules[name]


def probe(backend: str) -> str | None:
    """Return None on success, else the error string."""
    os.environ["MUJOCO_GL"] = backend
    if backend == "egl":
        os.environ["PYOPENGL_PLATFORM"] = "egl"
    else:
        os.environ.pop("PYOPENGL_PLATFORM", None)
    purge()
    try:
        import mujoco  # noqa: PLC0415

        model = mujoco.MjModel.from_xml_string(XML)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        renderer = mujoco.Renderer(model, 64, 64)
        renderer.update_scene(data)
        pixels = renderer.render()
        renderer.close()
        print(f"[ok]   {backend:7s} render={pixels.shape} dtype={pixels.dtype} "
              f"mujoco={mujoco.__file__}")
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"[fail] {backend:7s} {type(exc).__name__}: {exc}")
        return f"{backend}: {type(exc).__name__}: {exc}"


def main() -> int:
    print(f"host={os.uname().nodename} DISPLAY={os.environ.get('DISPLAY', '-')} "
          f"MUJOCO_GL={os.environ.get('MUJOCO_GL', '-')} "
          f"PYOPENGL_PLATFORM={os.environ.get('PYOPENGL_PLATFORM', '-')}")
    ok = [b for b in BACKENDS if probe(b) is None]
    print()
    if not ok:
        print("[fail] 三个后端都不行：装 libOSMesa（apt install libosmesa6）或修 EGL 驱动后重试")
        return 1
    print(f"[ok] 可用后端: {', '.join(ok)}")
    print(f"     → 跑闭环时用: SM_GL_BACKEND={ok[0]}（closed_loop 会自己按 egl→osmesa 试，"
          f"只有想强制指定时才需要）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
