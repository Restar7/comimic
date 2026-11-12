# -*- coding: utf-8 -*-
"""真实字形预览：解析生成的 pptx，用 Noto Sans SC / DejaVu 画成 PNG（用于版式自检）。"""
from __future__ import annotations

import os
import re
import sys
import zipfile
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pptxlib import wrap_text_lines, CJK_RE, NOTO_R, NOTO_B, DEJA_R, DEJA_B

A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
P = "{http://schemas.openxmlformats.org/presentationml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
DPI = 96
EMU = 914400
_fc: dict = {}


def font(size_px: float, bold: bool, latin: str = ""):
    kind = "dejavu" if latin == "DejaVu Sans" else "noto"
    key = (kind, round(size_px), bold)
    if key not in _fc:
        path = (DEJA_B if bold else DEJA_R) if kind == "dejavu" else (NOTO_B if bold else NOTO_R)
        _fc[key] = ImageFont.truetype(path, max(6, int(round(size_px))))
    return _fc[key]


def px(v: int) -> float:
    return v / EMU * DPI


def color_of(el):
    if el is None:
        return None
    c = el.find(f"{A}srgbClr")
    return "#" + c.get("val") if c is not None else None


def iter_shapes(slide_xml):
    root = ET.fromstring(slide_xml)
    spTree = root.find(f"{P}cSld/{P}spTree")
    for child in list(spTree):
        yield child


def draw_text(dr, x, y, w, h, body, anchor, default_size):
    paras = []
    for p in body.findall(f"{A}p"):
        ppr = p.find(f"{A}pPr")
        align = ppr.get("algn", "l") if ppr is not None else "l"
        spcb = 0.0
        lhp = 1.22
        if ppr is not None:
            sb = ppr.find(f"{A}spcBef/{A}spcPts")
            if sb is not None:
                spcb = float(sb.get("val")) / 100.0
            lh = ppr.find(f"{A}lnSpc/{A}spcPct")
            if lh is not None:
                lhp = float(lh.get("val")) / 100000.0
        segs = []      # (text, size, bold, color, latin)
        for r in p.findall(f"{A}r"):
            t = r.find(f"{A}t")
            txt = (t.text or "") if t is not None else ""
            rpr = r.find(f"{A}rPr")
            sz = float(rpr.get("sz")) / 100.0 if rpr is not None and rpr.get("sz") else default_size
            bold = rpr is not None and rpr.get("b") == "1"
            col = color_of(rpr.find(f"{A}solidFill")) if rpr is not None else None
            latin = ""
            if rpr is not None:
                lt = rpr.find(f"{A}latin")
                if lt is not None:
                    latin = lt.get("typeface", "")
            segs.append((txt, sz, bold, col or "#000000", latin))
        if segs:
            paras.append((segs, align, spcb, lhp))

    laid = []   # 行：(segs, size, lhp)
    for segs, align, spcb, lhp in paras:
        size = segs[0][1]
        latin = segs[0][4]
        text = "".join(s[0] for s in segs)
        lines = wrap_text_lines(text, w / DPI, size, segs[0][2], latin)
        # 把 text 按行首尾切回 segs（顺序消费）
        pos = 0
        nl = len(lines)
        for li, line in enumerate(lines):
            lsize = size
            laid.append(([(line, size, segs[0][2], segs[0][3], latin)], align, spcb if li == 0 else 0.0, lhp))
        pos += len(text)
    # 垂直锚点
    total = sum((s[0][1] * lhp) / 72.0 * DPI for s, a, sb, lhp in laid) + sum(sb / 72.0 * DPI for s, a, sb, lhp in laid)
    ty = y + (h - total) / 2 if anchor == "ctr" else y
    for segs, align, spcb, lhp in laid:
        ty += spcb / 72.0 * DPI
        text, size, bold, col, latin = segs[0]
        px_size = size / 72.0 * DPI
        adv = 0.0
        for ch in text:
            adv += dr.textlength(ch, font=font(px_size, bold, "" if CJK_RE.match(ch) else latin))
        cx = x
        if align == "c":
            cx = x + (w - adv) / 2
        elif align == "r":
            cx = x + w - adv
        for ch in text:
            fnt = font(px_size, bold, "" if CJK_RE.match(ch) else latin)
            dr.text((cx, ty), ch, font=fnt, fill=col)
            cx += dr.textlength(ch, font=fnt)
        ty += size * lhp / 72.0 * DPI


def render(pptx_path: str, outdir: str) -> list[str]:
    os.makedirs(outdir, exist_ok=True)
    zf = zipfile.ZipFile(pptx_path)
    outs = []
    names = sorted([n for n in zf.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)],
                   key=lambda s: int(re.findall(r"\d+", s)[0]))
    for name in names:
        idx = int(re.findall(r"\d+", name)[0])
        img = Image.new("RGB", (1280, 720), "white")
        dr = ImageDraw.Draw(img)
        rels = ET.fromstring(zf.read(f"ppt/slides/_rels/slide{idx}.xml.rels"))
        relmap = {r.get("Id"): r.get("Target") for r in rels}
        for sh in iter_shapes(zf.read(name)):
            tag = sh.tag.split("}")[1]
            if tag == "pic":
                blip = sh.find(f"{P}blipFill/{A}blip")
                tgt = relmap[blip.get(f"{R}embed")].replace("../", "ppt/")
                xfrm = sh.find(f"{P}spPr/{A}xfrm")
                off, ext = xfrm.find(f"{A}off"), xfrm.find(f"{A}ext")
                x, y = px(int(off.get("x"))), px(int(off.get("y")))
                w, h = px(int(ext.get("cx"))), px(int(ext.get("cy")))
                img.paste(Image.open(zf.open(tgt)).convert("RGB").resize((int(w), int(h))),
                          (int(x), int(y)))
                continue
            if tag != "sp":
                continue
            sp = sh.find(f"{P}spPr")
            xfrm = sp.find(f"{A}xfrm") if sp is not None else None
            if xfrm is None:
                continue
            off, ext = xfrm.find(f"{A}off"), xfrm.find(f"{A}ext")
            x, y = px(int(off.get("x"))), px(int(off.get("y")))
            w, h = px(int(ext.get("cx"))), px(int(ext.get("cy")))
            geom = sp.find(f"{A}prstGeom")
            prst = geom.get("prst") if geom is not None else "rect"
            fill = color_of(sp.find(f"{A}solidFill"))
            ln = sp.find(f"{A}ln")
            lcol = color_of(ln.find(f"{A}solidFill")) if ln is not None else None
            if prst == "line":
                if lcol:
                    dr.line([x, y, x + w, y + h], fill=lcol, width=1)
                continue
            if fill or lcol:
                box = [x, y, x + max(w, 1), y + max(h, 1)]
                if prst == "roundRect":
                    dr.rounded_rectangle(box, radius=max(2, min(w, h) * 0.10),
                                         fill=fill, outline=lcol, width=1)
                elif prst == "chevron":
                    dr.polygon([(x, y), (x + w * 0.6, y), (x + w, y + h / 2), (x + w * 0.6, y + h),
                                (x, y + h), (x + w * 0.4, y + h / 2)], fill=fill, outline=lcol)
                else:
                    dr.rectangle(box, fill=fill, outline=lcol, width=1)
            body = sh.find(f"{P}txBody")
            if body is None:
                continue
            bpr = body.find(f"{A}bodyPr")
            anchor = bpr.get("anchor", "t") if bpr is not None else "t"
            draw_text(dr, x, y, w, h, body, anchor, 13.2)
        p = os.path.join(outdir, f"p{idx:02d}.png")
        img.save(p)
        outs.append(p)
    return outs


if __name__ == "__main__":
    for p in render(sys.argv[1], sys.argv[2]):
        print(p)
