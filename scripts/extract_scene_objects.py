#!/usr/bin/env python3
"""把 SONIC 场景里的**物体**拆成 mjlab 能单独加载的 MJCF（B 方案第一步）。

背景（2026-10-29）：WM 只在"平地、无物体"的仿真数据上训过，而部署场景有桌子/罐子/抽屉 ⇒
WM 学不到"接触会让速度变小"，在部署分布上 `qd_ratio` 高估 26%。要让 mjlab 的向量化采集器
也能在**有物体的场景**里采数据，就得把场景物体搬进 mjlab 的 ``SceneCfg``。

mjlab 的 ``SceneCfg(entities={...})`` 里每个 entity 是一份独立 MJCF：
  * 静态物件（桌子/柜子/托盘）→ 无 joint 的实体，可合成一个 entity
  * 自由体（罐子/香蕉/砖）→ 各带 ``freejoint``，一个 entity 一个
  * 抽屉 → 带 ``slide`` joint 的 entity

实现用 **MjSpec**（而不是文本切标签）：``<include file="g1_29dof_with_hand.xml"/>`` 会把机器人
展开进 spec，直接 ``spec.delete(body)`` 就能精确剔除，避免文本配对出错。

用法：
    python3 scripts/extract_scene_objects.py --task task1 --check
    # 产出 assets/scene_objs/task1/{static,<obj>}.xml
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENES = Path(
    "/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/SONIC_DATA/deployment/"
    "gear_sonic/data/robot_model/model_data/g1"
)
TASKS = {
    "task1": "scene_43dof_two_table.xml",
    "task2": "scene_43dof.xml",
    "task3": "scene_43dof_handover.xml",
    "task4": "scene_43dof_bend_and_pick.xml",
}
ROBOT_ROOT = "pelvis"


def extract_body(text: str, start: int) -> str | None:
    """从 ``<body name=...>`` 起做标签配对，返回完整（含嵌套）的 body 块。

    mujoco 3.2.3 的 MjSpec 没有 ``bodies`` 列表/``delete``（只有 ``detach_body``/``find_body``），
    所以这里走文本配对 —— 场景里的物体确实有嵌套 body（桌腿等），必须按 depth 配对。
    """
    i, depth = start + 5, 1
    while i < len(text):
        nxt_open = text.find("<body", i)
        nxt_close = text.find("</body>", i)
        if nxt_close < 0:
            return None
        if 0 <= nxt_open < nxt_close:
            depth += 1
            i = nxt_open + 5
        else:
            depth -= 1
            i = nxt_close + 7
            if depth == 0:
                return text[start:nxt_close + 7]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="task1", choices=sorted(TASKS))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    import mujoco

    import re

    src = SCENES / TASKS[a.task]
    text = src.read_text()
    head = text[: text.index("<worldbody>")]
    _wb_end = text.index("</worldbody>")
    tail = text[_wb_end + len("</worldbody>"):]   # 注意别把 </worldbody> 重复带上
    head = re.sub(r"\s*<include[^>]*/>", "", head)          # 去掉机器人 include
    # ★ compiler 是从 include 的 robot xml 里带进来的（`<compiler meshdir="meshes"/>`）：
    #   删掉 include 后 meshdir 会丢，`ycb/...` 就解析到生成目录下（2026-10-29 实测报错）。
    #   这里显式补回，且必须指 SCENES/meshes（ycb 在 meshes/ycb 下）。
    if "<compiler" in head:
        head = re.sub(r'<compiler([^>]*?)meshdir="[^"]*"', rf'<compiler\1meshdir="{SCENES}/meshes"', head)
        if f'meshdir="{SCENES}/meshes"' not in head:
            head = head.replace("<compiler", f'<compiler meshdir="{SCENES}/meshes"', 1)
    else:
        head = head.replace("<worldbody>", f'<compiler angle="radian" meshdir="{SCENES}/meshes"/>\n<worldbody>', 1)
    # texture 的 file 走 texturedir（不是 meshdir）：不改成绝对路径就会找 assets/scene_objs/<task>/meshes/ycb/…
    if 'texturedir="' in head:
        head = re.sub(r'texturedir="[^"]*"', f'texturedir="{SCENES}/meshes"', head)
    else:
        head = re.sub(r'<compiler([^>]*)/>', rf'<compiler\1 texturedir="{SCENES}/meshes"/>', head, count=1)

    objs, chunks, kind = [], {}, {}
    for m in re.finditer(r'<body\s+name="([^"]+)"', text):
        name = m.group(1)
        if name in chunks:
            continue
        chunk = extract_body(text, m.start())
        if not chunk:
            continue
        if chunk.count("<body") != chunk.count("</body>"):
            raise SystemExit(f"[fail] {name} 的 body 标签不配对")
        chunks[name] = chunk
        objs.append(name)
        kind[name] = ("free" if "freejoint" in chunk
                      else ("slide" if 'type="slide"' in chunk else "static"))
    print(f"[split] {src.name}: 物体 {len(objs)} 个 → " +
          ", ".join(f"{n}({kind[n]})" for n in objs))

    outdir = ROOT / "assets/scene_objs" / a.task
    outdir.mkdir(parents=True, exist_ok=True)

    static = [n for n in objs if kind[n] == "static"]
    if static:
        xml = head + "<worldbody>\n" + "\n".join(chunks[n] for n in static) + "\n</worldbody>\n" + tail
        (outdir / "static.xml").write_text(xml)
        print(f"[ok] static.xml ← {static}")

    for n in objs:
        if kind[n] == "static":
            continue
        xml = head + "<worldbody>\n" + chunks[n] + "\n</worldbody>\n" + tail
        (outdir / f"{n}.xml").write_text(xml)
        print(f"[ok] {n}.xml ← ({kind[n]})")

    if a.check:
        for f in sorted(outdir.glob("*.xml")):
            try:
                m = mujoco.MjModel.from_xml_path(str(f))
                free = sum(1 for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE)
                slide = sum(1 for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_SLIDE)
                print(f"[check] {f.name:24s} nq={m.nq:3d} nu={m.nu} nbody={m.nbody} free={free} slide={slide}")
            except Exception as exc:
                print(f"[check][fail] {f.name}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
