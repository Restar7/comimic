# -*- coding: utf-8 -*-
"""pptx 结构自检：XML 合法、rels 完整、CT 覆盖、id 唯一、坐标在画布内、pPr 子元素顺序。"""
from __future__ import annotations

import re
import sys
import zipfile
import xml.etree.ElementTree as ET

A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
P = "{http://schemas.openxmlformats.org/presentationml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
W, H = 12192000, 6858000
PORDER = ["lnSpc", "spcBef", "spcAft", "buClr", "buClrTx", "buSz", "buSzTx", "buFont",
          "buFontTx", "buNone", "buAutoNum", "buChar", "buBlip", "tabLst", "defRPr", "extLst"]


def main(path: str) -> int:
    errs, warns = [], []
    zf = zipfile.ZipFile(path)
    names = set(zf.namelist())
    # 1) 全 XML 可解析
    for n in names:
        if n.endswith((".xml", ".rels")):
            try:
                ET.fromstring(zf.read(n))
            except Exception as exc:
                errs.append(f"XML 解析失败 {n}: {exc}")
    # 2) content types 覆盖
    ct = zf.read("[Content_Types].xml").decode()
    overrides = set(re.findall(r'PartName="([^"]+)"', ct))
    defaults = set(re.findall(r'Extension="([^"]+)"', ct))
    for n in names:
        if n == "[Content_Types].xml":
            continue
        if "/" + n in overrides:
            continue
        ext = n.rsplit(".", 1)[-1]
        if ext not in defaults:
            errs.append(f"CT 未覆盖: {n}")
    for o in overrides:
        if o.lstrip("/") not in names:
            errs.append(f"CT 指向不存在的 part: {o}")
    # 3) 每个 .rels 的 target 存在
    for n in [x for x in names if x.endswith(".rels")]:
        base = n.replace("_rels/", "").rsplit(".rels", 1)[0]
        base_dir = base.rsplit("/", 1)[0] if "/" in base else ""
        for rel in ET.fromstring(zf.read(n)):
            t = rel.get("Target")
            if rel.get("TargetMode") == "External":
                continue
            tgt = (t.lstrip("/") if t.startswith("/") else (base_dir + "/" + t if base_dir else t)).lstrip("/")
            tgt = re.sub(r"/\./", "/", tgt)
            while "/../" in tgt:
                tgt = re.sub(r"[^/]+/\.\./", "", tgt)
            if tgt not in names:
                errs.append(f"rels 指向不存在的 part: {n} -> {tgt}")
    # 4) 幻灯片列表一致性
    pres = ET.fromstring(zf.read("ppt/presentation.xml"))
    ids = pres.findall(f"{P}sldIdLst/{P}sldId")
    slides = sorted([n for n in names if re.match(r"ppt/slides/slide\d+\.xml$", n)],
                    key=lambda s: int(re.findall(r"\d+", s)[0]))
    if len(ids) != len(slides):
        errs.append(f"sldIdLst {len(ids)} != slides {len(slides)}")
    pres_rels = {r.get("Id"): r.get("Target") for r in ET.fromstring(zf.read("ppt/_rels/presentation.xml.rels"))}
    for i, sld in enumerate(ids, 1):
        rid = sld.get(f"{R}id")
        if rid not in pres_rels:
            errs.append(f"sldId#{i} r:id {rid} 不在 presentation rels")
        elif pres_rels[rid] != f"slides/slide{i}.xml":
            errs.append(f"sldId#{i} 指向 {pres_rels[rid]}")
    # 5) 逐页检查
    for i in range(1, len(slides) + 1):
        xml = zf.read(f"ppt/slides/slide{i}.xml")
        root = ET.fromstring(xml)
        rels = {r.get("Id") for r in ET.fromstring(zf.read(f"ppt/slides/_rels/slide{i}.xml.rels"))}
        used = set(re.findall(rb'r:(?:id|embed|link)="([^"]+)"', xml))
        for u in used:
            if u.decode() not in rels:
                errs.append(f"slide{i} 引用了不存在的 rel {u.decode()}")
        spids = [e.get("id") for e in root.iter() if e.tag in (f"{P}cNvPr",)]
        if len(spids) != len(set(spids)):
            errs.append(f"slide{i} shape id 重复")
        for sp in list(root.iter(f"{P}sp")):
            xfrm = sp.find(f"{P}spPr/{A}xfrm")
            if xfrm is None:
                continue
            off, ext = xfrm.find(f"{A}off"), xfrm.find(f"{A}ext")
            x, y = int(off.get("x")), int(off.get("y"))
            w, h = int(ext.get("cx")), int(ext.get("cy"))
            if x < 0 or y < 0 or x + w > W + 2 or y + h > H + 2:
                errs.append(f"slide{i} 越界: x={x} y={y} w={w} h={h}")
        for p in root.iter(f"{A}p"):
            ppr = p.find(f"{A}pPr")
            if ppr is None:
                continue
            order = [c.tag.split("}")[1] for c in list(ppr)]
            idx = [PORDER.index(t) for t in order if t in PORDER]
            if idx != sorted(idx):
                errs.append(f"slide{i} pPr 子元素顺序错误: {order}")
    print(f"[validate] parts={len(names)} slides={len(slides)} errors={len(errs)}")
    for e in errs:
        print("   ERR", e)
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
