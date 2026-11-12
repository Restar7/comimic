"""极简 OOXML (pptx) 生成器：只依赖标准库。

设计目标
--------
* 与 COMIMIC 课程设计汇报原稿同一套视觉语言（navy/teal/gray、Noto Sans CJK SC）；
* 每个文本块在**生成时**就用粗略字宽模型估算行数与高度，超框直接报错（防止投影时溢字）；
* 输出一个可直接被 PowerPoint / WPS 打开的 .pptx（复用原稿的 master/layout/theme）。

坐标系：英寸（1 in = 914400 EMU）。画布 13.333 x 7.5 in。
"""
from __future__ import annotations

import math
import os
import re
import shutil
import zipfile
from xml.sax.saxutils import escape

EMU = 914400
SLIDE_W = 13.3333
SLIDE_H = 7.5

# ---- 调色板（取自原稿） -----------------------------------------------------
NAVY = "133057"
TEAL = "0C778E"
GRAY = "607589"
BORDER = "D8E5EE"
DIVIDER = "DBE7EF"
FOOTLINE = "DCE7F0"
FILL_SOFT = "EAF3F8"
FILL_FAINT = "F5F9FC"
SLATE = "243C56"
AMBER = "C98B37"
RED = "AD5950"
WARM = "FFF4E4"
BLUE_D = "1E4974"
BLUE_M = "396C96"
CYAN_L = "E7F5FB"
CYAN_M = "B4DEE7"
WHITE = "FFFFFF"

CJK_RE = re.compile(r"[\u2e80-\u9fff\uff00-\uffef\u3000-\u303f]")

NS = (
    'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
    'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"'
)


def emu(v: float) -> int:
    return int(round(v * EMU))


# --------------------------------------------------------------------------- 字宽估算
_FONT_CACHE: dict = {}
_REF_SIZE = 64.0
NOTO_R = "/root/.fonts/NotoSansSC-Regular.otf"
NOTO_B = "/root/.fonts/NotoSansSC-Bold.otf"
DEJA_R = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
DEJA_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _pil_font(kind: str, bold: bool):
    key = (kind, bold)
    if key not in _FONT_CACHE:
        from PIL import ImageFont
        if kind == "dejavu":
            path = DEJA_B if bold else DEJA_R
        else:
            path = NOTO_B if bold else NOTO_R
        try:
            _FONT_CACHE[key] = ImageFont.truetype(path, int(_REF_SIZE))
        except Exception:
            try:
                _FONT_CACHE[key] = ImageFont.truetype(DEJA_B if bold else DEJA_R, int(_REF_SIZE))
            except Exception:
                _FONT_CACHE[key] = None
    return _FONT_CACHE[key]


def _char_w(ch: str, size: float, bold: bool = False, latin: str = "") -> float:
    """单字符宽度（pt）。CJK 用 Noto Sans SC 实测；ASCII 按 latin 字体族实测。"""
    if CJK_RE.match(ch):
        f = _pil_font("noto", bold)
        if f is not None:
            try:
                return f.getlength(ch) * size / _REF_SIZE
            except Exception:
                pass
        return size * 1.0
    f = _pil_font("dejavu" if latin == "DejaVu Sans" else "noto", bold)
    if f is not None:
        try:
            return f.getlength(ch) * size / _REF_SIZE
        except Exception:
            pass
    return size * 0.6


def measure(text: str, size: float, bold: bool = False, latin: str = "") -> float:
    return sum(_char_w(c, size, bold, latin) for c in text)


def _atoms(text: str):
    """切成可断行的原子：CJK 单字 / ASCII 词 / 空格。"""
    cur = ""
    for ch in text:
        if CJK_RE.match(ch) or ch == " ":
            if cur:
                yield cur
                cur = ""
            yield ch
        else:
            cur += ch
    if cur:
        yield cur


def wrap_text_lines(text: str, width_in: float, size: float, bold: bool = False,
                    latin: str = "") -> list[str]:
    """贪心断行（CJK 任意断、ASCII 按词断），返回真实行列表。"""
    avail = width_in * 72.0
    lines: list[str] = []
    cur = ""
    curw = 0.0
    for atom in _atoms(text):
        w = measure(atom, size, bold, latin)
        if curw + w > avail + 1e-6 and cur.strip():
            lines.append(cur.rstrip())
            cur, curw = "", 0.0
            if atom == " ":
                continue
        if atom != " " or cur:
            cur += atom
            curw += w
    if cur.strip() or not lines:
        lines.append(cur.rstrip())
    return lines


def wrap_lines(text: str, width_in: float, size: float, bold: bool = False,
               latin: str = "") -> int:
    return len(wrap_text_lines(text, width_in, size, bold, latin))


# --------------------------------------------------------------------------- 形状
class Slide:
    def __init__(self) -> None:
        self.body: list[str] = []
        self.rels: list[tuple[str, str, str]] = []   # (rId, type, target)
        self._id = 1
        self.warnings: list[str] = []
        self.fill: list[tuple[float, str]] = []

    # -- 基础 -----------------------------------------------------------------
    def next_id(self) -> int:
        self._id += 1
        return self._id

    def _xfrm(self, x, y, w, h) -> str:
        return (f'<a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/>'
                f'<a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>')

    def rect(self, x, y, w, h, fill=None, line=None, lw=1.0, radius=None,
             dash="solid") -> None:
        sid = self.next_id()
        geom = ('<a:prstGeom prst="roundRect"><a:avLst><a:gd name="adj" '
                f'fmla="val {int(round((radius or 0) / min(w, h) * 100000)) if radius else 5000}"/>'
                '</a:avLst></a:prstGeom>') if radius else \
               '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        fill_xml = f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>' if fill else "<a:noFill/>"
        line_xml = (f'<a:ln w="{int(lw * 12700)}"><a:solidFill><a:srgbClr val="{line}"/>'
                    f'</a:solidFill><a:prstDash val="{dash}"/></a:ln>') if line else "<a:ln><a:noFill/></a:ln>"
        self.body.append(
            f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="Shape {sid}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr>{self._xfrm(x, y, w, h)}{geom}{fill_xml}{line_xml}</p:spPr></p:sp>')

    def hline(self, x, y, w, color=DIVIDER, lw=1.0, dash="solid") -> None:
        sid = self.next_id()
        self.body.append(
            f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="Line {sid}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr>{self._xfrm(x, y, w, 0)}<a:prstGeom prst="line"><a:avLst/></a:prstGeom>'
            f'<a:noFill/><a:ln w="{int(lw * 12700)}"><a:solidFill><a:srgbClr val="{color}"/>'
            f'</a:solidFill><a:prstDash val="{dash}"/></a:ln></p:spPr></p:sp>')

    def vline(self, x, y, h, color=BORDER, lw=1.0, dash="solid") -> None:
        sid = self.next_id()
        self.body.append(
            f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="VLine {sid}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr>{self._xfrm(x, y, 0, h)}<a:prstGeom prst="line"><a:avLst/></a:prstGeom>'
            f'<a:noFill/><a:ln w="{int(lw * 12700)}"><a:solidFill><a:srgbClr val="{color}"/>'
            f'</a:solidFill><a:prstDash val="{dash}"/></a:ln></p:spPr></p:sp>')

    def chevron(self, x, y, w=0.26, h=0.34, color=TEAL) -> None:
        sid = self.next_id()
        self.body.append(
            f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="Chev {sid}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr>{self._xfrm(x, y, w, h)}<a:prstGeom prst="chevron"><a:avLst/></a:prstGeom>'
            f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
            f'<a:ln w="3175"><a:solidFill><a:srgbClr val="{color}"/></a:solidFill></a:ln></p:spPr></p:sp>')

    def pic(self, x, y, w, h, target, descr="figure") -> None:
        rid = f"rId{len(self.rels) + 2}"
        self.rels.append((rid, "image", target))
        sid = self.next_id()
        self.body.append(
            f'<p:pic><p:nvPicPr><p:cNvPr id="{sid}" name="Image {sid}" descr="{escape(descr)}"/>'
            '<p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>'
            f'<p:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>'
            f'<p:spPr>{self._xfrm(x, y, w, h)}<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
            '</p:spPr></p:pic>')

    # -- 文本 -----------------------------------------------------------------
    def text(self, x, y, w, h, blocks, anchor="t", align="l", check=True,
             name=None) -> None:
        """blocks: str | list[dict]；dict 字段 t/size/color/b/italic/spc/sb/ind/latin。

        check=True 时估算高度，超过 h 记 warning（不抛异常，统一在最后汇总）。
        """
        if isinstance(blocks, str):
            blocks = [{"t": blocks}]
        # 文本里出现的 \n 一律拆成独立段落（a:t 内不放裸换行，避免各家渲染器行为不一致）
        expanded = []
        for blk in blocks:
            parts = str(blk["t"]).split("\n")
            for k, part in enumerate(parts):
                nb = dict(blk)
                nb["t"] = part
                if k > 0:
                    nb["sb"] = 0
                expanded.append(nb)
        blocks = expanded
        sid = self.next_id()
        size_ref = float(blocks[0].get("size", 13.2))
        total_pt = 0.0
        max_lines = 0
        for blk in blocks:
            size = float(blk.get("size", size_ref))
            ind = float(blk.get("ind", 0.0))
            n = wrap_lines(str(blk["t"]), max(0.2, w - ind), size, bool(blk.get("b")),
                           str(blk.get("latin", "")))
            max_lines = max(max_lines, n)
            total_pt += n * size * float(blk.get("lh", 1.22)) + float(blk.get("sb", 0))
        total_in = total_pt / 72.0
        self.fill.append((total_in / max(h, 1e-6), _short(blocks)))
        if check and total_in > h + 0.02:
            self.warnings.append(
                f"[overflow] '{_short(blocks)}' 需 {total_in:.2f}in > 框 {h:.2f}in "
                f"({max_lines} 行)")
        runs = []
        for blk in blocks:
            size = float(blk.get("size", size_ref))
            color = blk.get("color", GRAY)
            b = ' b="1"' if blk.get("b") else ""
            i = ' i="1"' if blk.get("italic") else ""
            spc = f' spc="{int(blk["spc"])}"' if blk.get("spc") is not None else ""
            latin = blk.get("latin", "Noto Sans CJK SC")
            ind = float(blk.get("ind", 0.0))
            ppr = '<a:pPr marL="%d" indent="0"%s>' % (
                int(ind * EMU), ' algn="%s"' % blk["align"] if blk.get("align") else "")
            if blk.get("lh"):                       # 顺序必须是 lnSpc -> spcBef -> spcAft -> buNone
                ppr += f'<a:lnSpc><a:spcPct val="{int(float(blk["lh"]) * 100000)}"/></a:lnSpc>'
            if blk.get("sb"):
                ppr += f'<a:spcBef><a:spcPts val="{int(float(blk["sb"]) * 100)}"/></a:spcBef>'
            if blk.get("sa"):
                ppr += f'<a:spcAft><a:spcPts val="{int(float(blk["sa"]) * 100)}"/></a:spcAft>'
            ppr += "<a:buNone/></a:pPr>"
            rpr = (f'<a:rPr lang="en-US" altLang="zh-CN" sz="{int(round(size * 100))}"{b}{i}'
                   f' kern="0"{spc} dirty="0"><a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
                   f'<a:latin typeface="{latin}"/><a:ea typeface="Noto Sans CJK SC"/>'
                   '<a:cs typeface="Noto Sans CJK SC"/></a:rPr>')
            runs.append(f'<a:p>{ppr}<a:r>{rpr}<a:t>{escape(str(blk["t"]))}</a:t></a:r>'
                        f'<a:endParaRPr lang="en-US" altLang="zh-CN" sz="{int(round(size * 100))}" dirty="0"/>'
                        '</a:p>')
        self.body.append(
            f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="{escape(name or ("Text " + str(sid)))}"/>'
            '<p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr>{self._xfrm(x, y, w, h)}<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
            '<a:noFill/></p:spPr>'
            f'<p:txBody><a:bodyPr wrap="square" lIns="0" tIns="0" rIns="0" bIns="0" rtlCol="0" '
            f'anchor="{anchor}"><a:normAutofit/></a:bodyPr><a:lstStyle/>'
            + "".join(runs) + "</p:txBody></p:sp>")

    # -- 复合组件 -------------------------------------------------------------
    def header(self, eyebrow: str, title: str) -> None:
        self.rect(0, 0, SLIDE_W, 0.115, fill=NAVY, line=NAVY, lw=1.0)
        self.text(0.78, 0.36, 10.6, 0.24,
                  [{"t": eyebrow, "size": 11.2, "color": TEAL, "b": True, "spc": 100}],
                  anchor="ctr")
        self.text(0.77, 0.76, 11.75, 0.56, [{"t": title, "size": 27.5, "color": NAVY, "b": True}],
                  anchor="ctr")
        self.hline(0.78, 1.48, 11.8, color=DIVIDER, lw=1.1)

    def footer(self, page: int, note: str = "COMIMIC  /  GR00T N1.7 + SONIC + A′ World Model + BPTT   ｜   可微联合微调（训练侧）") -> None:
        self.hline(0.69, 7.10, 11.9, color=FOOTLINE, lw=0.6)
        self.text(0.72, 7.16, 10.4, 0.17, [{"t": note, "size": 8.5, "color": GRAY}], anchor="ctr")
        self.text(12.1, 7.13, 0.45, 0.23, [{"t": f"{page:02d}", "size": 8.5, "color": GRAY}],
                  anchor="ctr", align="r")

    def card(self, x, y, w, h, title, body, accent=TEAL, fill=WHITE, tsize=17.0,
             bsize=12.8, tag=None, tcolor=NAVY, border=BORDER, foot=None,
             fsize=11.4, lh=1.30, foot_color=None) -> None:
        self.rect(x, y, w, h, fill=fill, line=border, lw=0.9, radius=0.07)
        self.rect(x, y, 0.065, h, fill=accent, line=accent, lw=0.5)
        ty = y + 0.20
        if tag:
            self.text(x + 0.30, ty, w - 0.5, 0.20,
                      [{"t": tag, "size": 9.6, "color": accent, "b": True, "spc": 60}], anchor="ctr")
            ty += 0.27
        self.text(x + 0.30, ty, w - 0.52, 0.34, [{"t": title, "size": tsize, "color": tcolor, "b": True}],
                  anchor="ctr")
        bh = h - (ty - y) - 0.56 - (0.52 if foot else 0.0)
        self.text(x + 0.30, ty + 0.44, w - 0.52, bh,
                  [{"t": t, "size": bsize, "color": GRAY, "lh": lh} for t in body.split("\n")])
        if foot:
            self.text(x + 0.30, y + h - 0.60, w - 0.52, 0.40,
                      [{"t": foot, "size": fsize, "color": foot_color or accent, "b": True, "lh": 1.22}],
                      anchor="ctr")

    def kpi(self, x, y, w, value, label, color=TEAL, vsize=30.0, lsize=10.5, h=0.62) -> None:
        self.text(x, y, w, h, [{"t": value, "size": vsize, "color": color, "b": True}], anchor="ctr")
        self.text(x, y + h, w, 0.30, [{"t": label, "size": lsize, "color": GRAY}], anchor="t")

    def band(self, x, y, w, h, text, fill=FILL_SOFT, color=NAVY, size=13.2, bold=True,
             accent=TEAL, lw=0.9) -> None:
        self.rect(x, y, w, h, fill=fill, line=fill if not accent else fill, lw=lw, radius=0.05)
        self.rect(x, y, 0.055, h, fill=accent, line=accent, lw=0.5)
        lines = 1 + text.count("\n")
        self.text(x + 0.24, y, w - 0.42, h,
                  [{"t": t, "size": size, "color": color, "b": bold, "lh": 1.26}
                   for t in text.split("\n")], anchor="ctr")

    def step(self, x, y, w, h, num, title, body, accent=TEAL) -> None:
        self.rect(x, y, w, h, fill=WHITE, line=BORDER, lw=0.9, radius=0.06)
        self.text(x + 0.26, y + 0.16, 0.5, 0.42, [{"t": f"{num}", "size": 22.0, "color": accent, "b": True}],
                  anchor="ctr")
        self.text(x + 0.80, y + 0.17, w - 1.0, 0.38, [{"t": title, "size": 14.6, "color": NAVY, "b": True}],
                  anchor="ctr")
        self.text(x + 0.26, y + 0.64, w - 0.50, h - 0.76, [{"t": body, "size": 12.3, "color": GRAY, "lh": 1.30}])

    def table(self, x, y, w, cols, rows, rh=0.34, header=True, size=12.2, hsize=11.4,
              aligns=None, col_fills=None, bold_first=False, zebra=True,
              first_color=NAVY) -> float:
        """画一个表格（矩形+线+文本），返回总高度。cols 为列宽比例。"""
        total = sum(cols)
        xs, acc = [], 0.0
        for c in cols:
            xs.append(acc / total * w)
            acc += c
        aligns = aligns or ["l"] * len(cols)
        n = len(rows)
        self.rect(x, y, w, rh * n, fill=WHITE, line=BORDER, lw=0.8)
        for i, row in enumerate(rows):
            ry = y + i * rh
            if i > 0:
                self.hline(x, ry, w, color=BORDER, lw=0.6)
            fill = None
            if header and i == 0:
                fill = FILL_SOFT
            elif zebra and (i % 2 == 0):
                fill = FILL_FAINT
            if fill:
                self.rect(x, ry, w, rh, fill=fill, line=None)
            for j, cell in enumerate(row):
                cx = x + xs[j]
                cw = xs[j + 1] - xs[j] if j + 1 < len(xs) else w - xs[j]
                bold = (header and i == 0) or (bold_first and j == 0)
                color = first_color if j == 0 and bold_first else (NAVY if (header and i == 0) else GRAY)
                s = hsize if (header and i == 0) else size
                if isinstance(cell, dict):
                    s = cell.get("size", s); color = cell.get("color", color); bold = cell.get("b", bold)
                    cell = cell["t"]
                self.text(cx + 0.10, ry, cw - 0.16, rh,
                          [{"t": str(cell), "size": s, "color": color, "b": bold}],
                          anchor="ctr", align=aligns[j])
        return rh * n

    def arrow_flow(self, items, x, y, w, h, gap=0.22, chev=True, fill=WHITE, tsize=14.5,
                   bsize=11.6, accent=TEAL) -> None:
        n = len(items)
        cw = (w - gap * (n - 1)) / n
        for i, (title, body) in enumerate(items):
            cx = x + i * (cw + gap)
            self.rect(cx, y, cw, h, fill=fill, line=BORDER, lw=0.9, radius=0.05)
            self.text(cx + 0.14, y + 0.10, cw - 0.28, 0.30,
                      [{"t": title, "size": tsize, "color": NAVY, "b": True}], anchor="ctr")
            if body:
                self.text(cx + 0.14, y + 0.44, cw - 0.28, h - 0.52,
                          [{"t": body, "size": bsize, "color": GRAY, "lh": 1.24}], anchor="t")
            if chev and i < n - 1:
                self.chevron(cx + cw + gap / 2 - 0.13, y + h / 2 - 0.17)

    # -- 输出 -----------------------------------------------------------------
    def xml(self) -> str:
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            f'<p:sld {NS}><p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="FFFFFF"/>'
            '</a:solidFill><a:effectLst/></p:bgPr></p:bg><p:spTree>'
            '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
            '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
            '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
            + "".join(self.body) +
            '</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>')

    def rels_xml(self) -> str:
        items = [('rId1', "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout",
                  "../slideLayouts/slideLayout1.xml")]
        items += self.rels
        body = "".join(
            f'<Relationship Id="{rid}" Type="{t}" Target="{tgt}"/>' for rid, t, tgt in items)
        return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                + body + "</Relationships>")


def _short(blocks) -> str:
    t = " ".join(str(b["t"]) for b in blocks)
    return t[:26] + ("…" if len(t) > 26 else "")


# --------------------------------------------------------------------------- 打包
CT_SLIDE = "application/vnd.openxmlformats-officedocument.presentationml.slide+xml"
REL_SLIDE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide"
REL_IMG = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
REL_LAYOUT = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout"
REL_MASTER = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster"
REL_THEME = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme"
REL_PRESPROPS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/presProps"
REL_VIEWPROPS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/viewProps"
REL_TABLESTYLES = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/tableStyles"
REL_NOTESMASTER = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesMaster"
REL_NOTES_SLIDE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide"
CT_NOTESMASTER = "application/vnd.openxmlformats-officedocument.presentationml.notesMaster+xml"
CT_NOTES = "application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml"


def _notes_xml(text: str) -> bytes:
    body = "".join(
        '<a:p><a:r><a:rPr lang="zh-CN" sz="1200" dirty="0"/><a:t>'
        + escape(para) + "</a:t></a:r></a:p>"
        for para in text.split("\n"))
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        f'<p:notes {NS}><p:cSld><p:spTree>'
        '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
        '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
        '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
        '<p:sp><p:nvSpPr><p:cNvPr id="2" name="Slide Image Placeholder 1"/><p:cNvSpPr>'
        '<a:spLocks noGrp="1" noRot="1" noChangeAspect="1"/></p:cNvSpPr>'
        '<p:nvPr><p:ph type="sldImg"/></p:nvPr></p:nvSpPr><p:spPr/></p:sp>'
        '<p:sp><p:nvSpPr><p:cNvPr id="3" name="Notes Placeholder 2"/><p:cNvSpPr>'
        '<a:spLocks noGrp="1"/></p:cNvSpPr><p:nvPr><p:ph type="body" idx="1"/></p:nvPr>'
        '</p:nvSpPr><p:spPr/><p:txBody><a:bodyPr/><a:lstStyle/>'
        + body + '</p:txBody></p:sp>'
        '</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:notes>'
    ).encode("utf-8")


def _notes_rels(i: int) -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'<Relationship Id="rId1" Type="{REL_SLIDE}" Target="../slides/slide{i}.xml"/>'
        f'<Relationship Id="rId2" Type="{REL_NOTESMASTER}" Target="../notesMasters/notesMaster1.xml"/>'
        "</Relationships>").encode("utf-8")


def build(slides: list[Slide], out_path: str, skeleton: str, assets: dict[str, str],
          title: str = "COMIMIC 全身 VLA 可微联合优化",
          subject: str = "GR00T N1.7 + SONIC + A-prime world model + BPTT (C baseline -> p21)",
          notes: list[str] | None = None) -> list[str]:
    """用 skeleton 目录（原稿解包）里的 master/layout/theme 组装新 pptx。"""
    warns: list[str] = []
    n = len(slides)
    notes = list(notes or [])
    has_notes = any(str(x).strip() for x in notes)
    tmp = out_path + ".tmpdir"
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    parts: dict[str, bytes] = {}

    def read(rel: str) -> bytes:
        with open(os.path.join(skeleton, rel), "rb") as fh:
            return fh.read()

    # ---- 静态部件（来自原稿，去掉 notes）----
    parts["docProps/custom.xml"] = read("docProps/custom.xml")
    if has_notes:
        parts["ppt/notesMasters/notesMaster1.xml"] = read("ppt/notesMasters/notesMaster1.xml")
        parts["ppt/notesMasters/_rels/notesMaster1.xml.rels"] = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f'<Relationship Id="rId1" Type="{REL_THEME}" Target="../theme/theme1.xml"/>'
            "</Relationships>").encode("utf-8")
    for rel in ("ppt/slideMasters/slideMaster1.xml",
                "ppt/slideMasters/_rels/slideMaster1.xml.rels",
                "ppt/slideLayouts/slideLayout1.xml",
                "ppt/slideLayouts/_rels/slideLayout1.xml.rels",
                "ppt/theme/theme1.xml", "ppt/presProps.xml", "ppt/viewProps.xml",
                "ppt/tableStyles.xml", "_rels/.rels"):
        parts[rel] = read(rel)

    # ---- slides + rels ----
    sld_ids, pres_rels = [], []
    for i, s in enumerate(slides, start=1):
        parts[f"ppt/slides/slide{i}.xml"] = s.xml().encode("utf-8")
        parts[f"ppt/slides/_rels/slide{i}.xml.rels"] = s.rels_xml().encode("utf-8")
        sld_ids.append(f'<p:sldId id="{255 + i}" r:id="rId{i + 10}"/>')
        pres_rels.append(f'<Relationship Id="rId{i + 10}" Type="{REL_SLIDE}" '
                         f'Target="slides/slide{i}.xml"/>')
        warns += [f"slide{i}: {w}" for w in s.warnings]
        txt = str(notes[i - 1]) if i - 1 < len(notes) else ""
        if txt.strip():
            parts[f"ppt/notesSlides/notesSlide{i}.xml"] = _notes_xml(txt)
            parts[f"ppt/notesSlides/_rels/notesSlide{i}.xml.rels"] = _notes_rels(i)
            s.rels.append((f"rId{100 + i}", REL_NOTES_SLIDE, f"../notesSlides/notesSlide{i}.xml"))
            parts[f"ppt/slides/_rels/slide{i}.xml.rels"] = s.rels_xml().encode("utf-8")

    # ---- 图片 ----
    for name, path in assets.items():
        with open(path, "rb") as fh:
            parts[f"ppt/media/{name}"] = fh.read()

    # ---- presentation.xml ----
    pres = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        f'<p:presentation {NS} autoCompressPictures="0" saveSubsetFonts="1">'
        '<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>'
        + ('<p:notesMasterIdLst><p:notesMasterId r:id="rId6"/></p:notesMasterIdLst>'
           if has_notes else "")
        + f'<p:sldIdLst>{"".join(sld_ids)}</p:sldIdLst>'
        f'<p:sldSz cx="{emu(SLIDE_W)}" cy="{emu(SLIDE_H)}"/><p:notesSz cx="6858000" cy="12192000"/>'
        '<p:defaultTextStyle><a:lvl1pPr marL="0" algn="l" defTabSz="914400" rtl="0" eaLnBrk="1" '
        'latinLnBrk="0" hangingPunct="1"><a:defRPr sz="1800" kern="1200"><a:solidFill>'
        '<a:schemeClr val="tx1"/></a:solidFill><a:latin typeface="+mn-lt"/><a:ea typeface="+mn-ea"/>'
        '<a:cs typeface="+mn-cs"/></a:defRPr></a:lvl1pPr></p:defaultTextStyle></p:presentation>')
    parts["ppt/presentation.xml"] = pres.encode("utf-8")

    pres_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'<Relationship Id="rId1" Type="{REL_MASTER}" Target="slideMasters/slideMaster1.xml"/>'
        f'<Relationship Id="rId2" Type="{REL_THEME}" Target="theme/theme1.xml"/>'
        f'<Relationship Id="rId3" Type="{REL_PRESPROPS}" Target="presProps.xml"/>'
        f'<Relationship Id="rId4" Type="{REL_VIEWPROPS}" Target="viewProps.xml"/>'
        f'<Relationship Id="rId5" Type="{REL_TABLESTYLES}" Target="tableStyles.xml"/>'
        + (f'<Relationship Id="rId6" Type="{REL_NOTESMASTER}" '
           'Target="notesMasters/notesMaster1.xml"/>' if has_notes else "")
        + "".join(pres_rels) + "</Relationships>")
    parts["ppt/_rels/presentation.xml.rels"] = pres_rels_xml.encode("utf-8")

    # ---- content types ----
    overrides = [
        ("/ppt/presentation.xml",
         "application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"),
        ("/ppt/slideMasters/slideMaster1.xml",
         "application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"),
        ("/ppt/slideLayouts/slideLayout1.xml",
         "application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"),
        ("/ppt/theme/theme1.xml", "application/vnd.openxmlformats-officedocument.theme+xml"),
        ("/ppt/presProps.xml",
         "application/vnd.openxmlformats-officedocument.presentationml.presProps+xml"),
        ("/ppt/viewProps.xml",
         "application/vnd.openxmlformats-officedocument.presentationml.viewProps+xml"),
        ("/ppt/tableStyles.xml",
         "application/vnd.openxmlformats-officedocument.presentationml.tableStyles+xml"),
        ("/docProps/core.xml", "application/vnd.openxmlformats-package.core-properties+xml"),
        ("/docProps/app.xml", "application/vnd.openxmlformats-officedocument.extended-properties+xml"),
        ("/docProps/custom.xml",
         "application/vnd.openxmlformats-officedocument.custom-properties+xml"),
    ]
    for i in range(1, n + 1):
        overrides.append((f"/ppt/slides/slide{i}.xml", CT_SLIDE))
    if has_notes:
        overrides.append(("/ppt/notesMasters/notesMaster1.xml", CT_NOTESMASTER))
        for i in range(1, n + 1):
            if f"ppt/notesSlides/notesSlide{i}.xml" in parts:
                overrides.append((f"/ppt/notesSlides/notesSlide{i}.xml", CT_NOTES))
    ct = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
          '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
          '<Default Extension="png" ContentType="image/png"/>'
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
          '<Default Extension="xml" ContentType="application/xml"/>'
          + "".join(f'<Override PartName="{p}" ContentType="{c}"/>' for p, c in overrides)
          + "</Types>")
    parts["[Content_Types].xml"] = ct.encode("utf-8")

    # ---- doc props ----
    parts["docProps/core.xml"] = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        f'<dc:title>{escape(title)}</dc:title><dc:creator>COMIMIC Course Design</dc:creator>'
        f'<dc:subject>{escape(subject)}</dc:subject><cp:lastModifiedBy>COMIMIC</cp:lastModifiedBy>'
        '<cp:revision>4</cp:revision>'
        '<dcterms:created xsi:type="dcterms:W3CDTF">2026-11-11T12:45:43Z</dcterms:created>'
        '<dcterms:modified xsi:type="dcterms:W3CDTF">2026-11-11T13:10:00Z</dcterms:modified>'
        '</cp:coreProperties>').encode("utf-8")
    parts["docProps/app.xml"] = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        f'<Slides>{n}</Slides><Notes>{sum(1 for i in range(1, n + 1) if f"ppt/notesSlides/notesSlide{i}.xml" in parts)}</Notes><HiddenSlides>0</HiddenSlides>'
        '<PresentationFormat>On-screen Show (16:9)</PresentationFormat>'
        '<Company>COMIMIC</Company><AppVersion>16.0000</AppVersion></Properties>').encode("utf-8")

    # ---- zip ----
    order = ["[Content_Types].xml", "_rels/.rels", "docProps/core.xml", "docProps/app.xml",
             "docProps/custom.xml",
             "ppt/presentation.xml", "ppt/_rels/presentation.xml.rels",
             "ppt/presProps.xml", "ppt/viewProps.xml", "ppt/tableStyles.xml",
             "ppt/slideMasters/slideMaster1.xml", "ppt/slideMasters/_rels/slideMaster1.xml.rels",
             "ppt/slideLayouts/slideLayout1.xml", "ppt/slideLayouts/_rels/slideLayout1.xml.rels",
             "ppt/theme/theme1.xml"]
    for i in range(1, n + 1):
        order += [f"ppt/slides/slide{i}.xml", f"ppt/slides/_rels/slide{i}.xml.rels"]
    if has_notes:
        order += ["ppt/notesMasters/notesMaster1.xml", "ppt/notesMasters/_rels/notesMaster1.xml.rels"]
        for i in range(1, n + 1):
            if f"ppt/notesSlides/notesSlide{i}.xml" in parts:
                order += [f"ppt/notesSlides/notesSlide{i}.xml",
                          f"ppt/notesSlides/_rels/notesSlide{i}.xml.rels"]
    order += [f"ppt/media/{k}" for k in assets]
    missing = [k for k in order if k not in parts]
    if missing:
        raise SystemExit(f"[build] 缺少 part: {missing}")
    extra = [k for k in parts if k not in order]
    order += extra
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in order:
            zf.writestr(name, parts[name])
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    return warns
