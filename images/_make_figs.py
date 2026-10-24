# -*- coding: utf-8 -*-
"""生成 04 笔记里的 7 张示意图。

Pillow 手绘，画布按 2 倍超采样再缩回，保证边缘不发虚。
用法:  python _make_figs.py            # 全画
       python _make_figs.py 3 5       # 只画第 3、5 张
注意:  修改本文件请用 UTF-8 编辑器保存，别用 PowerShell 的 Set-Content，
       否则中文会被按 GBK 重编码而乱码。
"""
import colorsys
import sys

from PIL import Image, ImageDraw, ImageFont

S = 2  # 超采样倍数

INK, MUT, FAINT = "#1F2933", "#5B6B7B", "#8A97A6"
LINE, GRID, PANEL = "#C9D4E0", "#DCE3EC", "#F5F8FC"
BLUE, BLUE_L = "#2563EB", "#DCE8FE"
ORANGE, ORANGE_L = "#EA580C", "#FFE6D2"
GREEN, GREEN_L = "#059669", "#D3F5E7"
PURPLE, PURPLE_L = "#7C3AED", "#EADFFE"
RED, RED_L = "#DC2626", "#FDE2E2"
SLATE_L = "#EEF2F7"
SLATE = "#64748B"

F_REG = "C:/Windows/Fonts/msyh.ttc"
F_BLD = "C:/Windows/Fonts/msyhbd.ttc"
F_MONO = "C:/Windows/Fonts/consola.ttf"
_fonts = {}


def has_cjk(s):
    return any("\u2e80" <= ch <= "\u9fff" or "\uff00" <= ch <= "\uffef" for ch in s)


def font(sz, bold=False, mono=False):
    key = (sz, bold, mono)
    if key not in _fonts:
        path = F_MONO if mono else (F_BLD if bold else F_REG)
        _fonts[key] = ImageFont.truetype(path, int(sz * S))
    return _fonts[key]


class Fig:
    def __init__(self, w, h, bg="#FFFFFF"):
        self.w, self.h = w, h
        self.img = Image.new("RGB", (w * S, h * S), bg)
        self.d = ImageDraw.Draw(self.img)

    # ---------- 基础图元（坐标都用逻辑像素，内部乘 S） ----------
    def rect(self, box, fill=None, outline=None, width=1, radius=0):
        x0, y0, x1, y1 = [v * S for v in box]
        if radius:
            self.d.rounded_rectangle([x0, y0, x1, y1], radius=radius * S, fill=fill,
                                     outline=outline,
                                     width=int(width * S) if outline else 0)
        else:
            self.d.rectangle([x0, y0, x1, y1], fill=fill, outline=outline,
                             width=int(width * S) if outline else 0)

    def dashed_rect(self, box, outline=LINE, width=1.5, dash=(7, 5), fill=None):
        if fill:
            self.d.rectangle([v * S for v in box], fill=fill)
        x0, y0, x1, y1 = box
        for a in [(x0, y0, x1, y0), (x1, y0, x1, y1),
                  (x1, y1, x0, y1), (x0, y1, x0, y0)]:
            self.dashed(*a, outline, width, dash)

    def ellipse(self, box, fill=None, outline=None, width=1.5):
        self.d.ellipse([v * S for v in box], fill=fill, outline=outline,
                       width=int(width * S) if outline else 0)

    def line(self, x0, y0, x1, y1, fill=INK, width=1):
        self.d.line([x0 * S, y0 * S, x1 * S, y1 * S], fill=fill,
                    width=max(1, int(width * S)))

    def dashed(self, x0, y0, x1, y1, fill=LINE, width=1, dash=(6, 5)):
        import math
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length == 0:
            return
        ux, uy = dx / length, dy / length
        on, off = dash
        t = 0.0
        while t < length:
            b = min(t + on, length)
            self.line(x0 + ux * t, y0 + uy * t, x0 + ux * b, y0 + uy * b, fill, width)
            t = b + off

    def arrow(self, x0, y0, x1, y1, fill=MUT, width=2, head=7):
        import math
        self.line(x0, y0, x1, y1, fill, width)
        ang = math.atan2(y1 - y0, x1 - x0)
        for s in (+1, -1):
            a = ang + s * math.radians(26)
            self.line(x1, y1, x1 - head * math.cos(a), y1 - head * math.sin(a),
                      fill, width)

    def elbow(self, pts, fill=MUT, width=2, head=7):
        for i in range(len(pts) - 2):
            self.line(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], fill, width)
        self.arrow(pts[-2][0], pts[-2][1], pts[-1][0], pts[-1][1], fill, width, head)

    def text(self, x, y, s, size=14, fill=INK, anchor="lt", bold=False, mono=False):
        # 中文字符在等宽字体里会变豆腐块，这里自动退回雅黑
        use_mono = mono and not has_cjk(s)
        self.d.text((x * S, y * S), s, font=font(size, bold, use_mono),
                    fill=fill, anchor=anchor)

    def text_rot(self, x, y, s, size=12, fill=MUT, angle=90):
        fnt = font(size, False, False)
        box = self.d.textbbox((0, 0), s, font=fnt)
        tw, th = box[2] - box[0], box[3] - box[1]
        tmp = Image.new("RGBA", (tw + 8 * S, th + 8 * S), (0, 0, 0, 0))
        ImageDraw.Draw(tmp).text((4 * S - box[0], 4 * S - box[1]), s, font=fnt, fill=fill)
        tmp = tmp.rotate(angle, expand=True)
        self.img.paste(tmp, (int(x * S - tmp.width / 2),
                             int(y * S - tmp.height / 2)), tmp)

    def matrix(self, box, fill=SLATE_L, outline=LINE, rows=6, cols=8,
               grid=GRID, width=2, radius=4):
        self.rect(box, fill=fill, outline=outline, width=width, radius=radius)
        x0, y0, x1, y1 = box
        for i in range(1, rows):
            yy = y0 + (y1 - y0) * i / rows
            self.line(x0 + 1, yy, x1 - 1, yy, grid, 1)
        for j in range(1, cols):
            xx = x0 + (x1 - x0) * j / cols
            self.line(xx, y0 + 1, xx, y1 - 1, grid, 1)

    def dim_h(self, x0, x1, y, label, size=12, fill=MUT):
        self.arrow(x0, y, x1, y, fill, 1, 5)
        self.arrow(x1, y, x0, y, fill, 1, 5)
        self.text((x0 + x1) / 2, y - 7, label, size, fill, "mb", mono=True)

    def dim_v(self, y0, y1, x, label, size=12, fill=MUT):
        self.arrow(x, y0, x, y1, fill, 1, 5)
        self.arrow(x, y1, x, y0, fill, 1, 5)
        self.text_rot(x - 8, (y0 + y1) / 2, label, size, fill)

    def pill(self, x, y, s, fill=RED_L, edge=RED, size=12, pad=10, h=26, bold=True):
        fnt = font(size, bold, mono=not has_cjk(s))
        tw = self.d.textlength(s, font=fnt) / S
        w = tw + pad * 2
        self.rect([x, y, x + w, y + h], fill=fill, outline=edge, width=1.5, radius=h / 2)
        self.text(x + w / 2, y + h / 2 - 0.5, s, size, edge, "mm", bold,
                  mono=not has_cjk(s))
        return w

    def chip(self, x, y, w, h, s, fill, edge, size=12, tc=None, bold=True):
        self.rect([x, y, x + w, y + h], fill=fill, outline=edge, width=1.5, radius=6)
        self.text(x + w / 2, y + h / 2, s, size, tc or edge, "mm", bold)

    def title(self, s, sub=None):
        self.text(46, 26, s, 21, INK, bold=True)
        if sub:
            self.text(46, 60, sub, 13, MUT)

    def save(self, path):
        self.img.resize((self.w, self.h), Image.LANCZOS).save(path)
        print("saved", path)


RANKS = [(BLUE, BLUE_L), (ORANGE, ORANGE_L), (GREEN, GREEN_L), (PURPLE, PURPLE_L)]


# ============================================================ 图1 全景
def fig01():
    f = Fig(1400, 900)
    f.title("① Qwen3-0.6B 在 TP=4 下，单张卡到底看见什么",
            "同一个 28 层模型，每张卡只装 1/4 的参数、只算 1/4 的 head，靠 58 次集合通信拼起来")

    X0, X1 = 470, 900
    C = (X0 + X1) / 2

    def box(y0, y1, main, sub=None, fill="#FFFFFF", edge=SLATE, tc=INK, size=13.5):
        f.rect([X0, y0, X1, y1], fill=fill, outline=edge, width=1.8, radius=7)
        if sub:
            f.text(C, y0 + 9, main, size, tc, "mt", bold=True)
            f.text(C, y1 - 22, sub, 10.5, MUT, "mt")
        else:
            f.text(C, (y0 + y1) / 2, main, size, tc, "mm", bold=True)

    box(96, 136, "input_ids   [T]", fill=SLATE_L, edge=SLATE)

    box(152, 216, "Embedding 分片", "[37984, 1024] · 每卡 78 MB",
        fill=BLUE_L, edge=BLUE, tc=BLUE)
    f.arrow(C, 140, C, 148, MUT, 1.8, 6)
    f.arrow(C, 220, C, 244, MUT, 1.8, 6)
    f.pill(926, 172, "all-reduce ×1", RED_L, RED, 11.5)

    f.dashed_rect([446, 244, 924, 748], outline="#B8C4D4", width=1.6, dash=(8, 6),
                  fill="#FBFCFE")
    f.text(458, 226, "× 28 层（每层都是同一套切法）", 10.5, MUT)

    rows = [
        (262, 300, "RMSNorm", None, "#FFFFFF", SLATE),
        (310, 366, "QKV 分片", "q 4 个 head · kv 2 个 head（16/8 里的 1/4）", BLUE_L, BLUE),
        (376, 436, "Attention（本卡 heads）", "KV cache 只存这 2 个 kv head 的 page",
         "#FFFFFF", SLATE),
        (446, 502, "O 投影", "输入 512 列 → 输出 1024", ORANGE_L, ORANGE),
        (512, 550, "RMSNorm", None, "#FFFFFF", SLATE),
        (560, 616, "gate_up 分片", "1024 → 2 × 768（原 2 × 3072 的 1/4）", ORANGE_L, ORANGE),
        (626, 664, "silu_and_mul", None, "#FFFFFF", SLATE),
        (674, 730, "down 分片", "输入 768 列 → 输出 1024", ORANGE_L, ORANGE),
    ]
    for (y0, y1, m, s, fl, ed) in rows:
        box(y0, y1, m, s, fill=fl, edge=ed, size=12.5)

    f.pill(926, 460, "all-reduce ×1", RED_L, RED, 11.5)
    f.pill(926, 688, "all-reduce ×1", RED_L, RED, 11.5)

    f.arrow(C, 754, C, 762, MUT, 1.8, 6)
    box(766, 806, "最终 RMSNorm", None, fill="#FFFFFF", edge=SLATE)
    f.arrow(C, 810, C, 818, MUT, 1.8, 6)
    box(822, 872, "LM head 分片（与 Embedding 共享权重）",
        "logits [T, 37984] → 截断到真实 vocab", ORANGE_L, ORANGE)
    f.pill(926, 836, "all-gather ×1", RED_L, RED, 11.5)

    left = [
        (176, "只读自己那段 vocab"),
        (338, "16 个 q head 里拿 4 个"),
        (406, "8 个 kv head 里拿 2 个"),
        (474, "4 个 head 的输出拼回 1024"),
        (588, "3072 中间维里拿 768"),
        (702, "再乘回 1024"),
    ]
    for y, s in left:
        f.text(434, y, s, 11.5, MUT, "rt")
        f.rect([438, y + 5, 442, y + 9], fill=BLUE, radius=2)

    f.text(926, 262, "整卡占用", 12.5, INK, bold=True)
    stats = [("≈188M", "参数 / 卡（总 751.6M ÷ 4）"),
             ("1/4", "KV cache 显存（kv head 只留 1/4）"),
             ("58 次", "集合通信 / forward")]
    for i, (big, small) in enumerate(stats):
        y = 288 + i * 74
        f.rect([926, y, 1354, y + 60], fill=PANEL, outline=LINE, width=1, radius=8)
        f.text(946, y + 7, big, 18, BLUE, bold=True)
        f.text(946, y + 34, small, 11, MUT)

    f.text(926, 540, "通信为什么洗不掉：", 12, INK, bold=True)
    for i, s in enumerate(["层内每次 all-reduce 都是", "「两张卡各算一半，必须相加」；",
                           "embedding 是「各认一段 vocab」；", "lm_head 是「各算一段 logits」。"]) :
        f.text(926, 564 + i * 24, s, 11, MUT)
    f.save("fig01-tp-overview.png")


# ============================================================ 图2 列/行并行
def fig02():
    f = Fig(1400, 680)
    f.title("② 列并行 vs 行并行：一个不需要通信，一个必须求和",
            "TP 的所有通信都来自行并行 —— 列并行的输出本身就是最终结果的一段")

    # ---------- A 列并行 ----------
    f.text(46, 106, "A · 列并行 = 切输出维 o", 15, BLUE, bold=True)
    f.text(300, 108, "（qkv_proj / gate_up_proj 就是这种）", 12, MUT)

    f.chip(46, 163, 62, 30, "卡 0", BLUE_L, BLUE, 12)
    f.chip(46, 233, 62, 30, "卡 1", ORANGE_L, ORANGE, 12)

    f.rect([128, 148, 282, 274], fill=SLATE_L, outline=SLATE, width=1.8, radius=6)
    f.text(205, 178, "X", 14, INK, "mm", bold=True, mono=True)
    f.text(205, 200, "[T, 1024]", 11.5, MUT, "mm", mono=True)
    f.text(205, 236, "两卡都完整", 10.5, FAINT, "mm")

    f.arrow(290, 178, 330, 178, MUT, 2, 7)
    f.arrow(290, 248, 330, 248, MUT, 2, 7)

    f.rect([336, 148, 486, 208], fill=BLUE_L, outline=BLUE, width=2, radius=6)
    f.text(411, 166, "W[:, :o/2]", 12.5, BLUE, "mm", bold=True, mono=True)
    f.text(411, 188, "输出的前半列", 10.5, MUT, "mm")

    f.rect([336, 218, 486, 278], fill=ORANGE_L, outline=ORANGE, width=2, radius=6)
    f.text(411, 236, "W[:, o/2:]", 12.5, ORANGE, "mm", bold=True, mono=True)
    f.text(411, 258, "输出的后半列", 10.5, MUT, "mm")

    f.arrow(492, 178, 530, 178, MUT, 2, 7)
    f.arrow(492, 248, 530, 248, MUT, 2, 7)

    f.rect([536, 148, 700, 208], fill=BLUE_L, outline=BLUE, width=2, radius=6)
    f.text(618, 164, "Y0 = X @ W0", 12.5, BLUE, "mm", bold=True, mono=True)
    f.text(618, 186, "[T, o/2]", 11, MUT, "mm", mono=True)

    f.rect([536, 218, 700, 278], fill=ORANGE_L, outline=ORANGE, width=2, radius=6)
    f.text(618, 234, "Y1 = X @ W1", 12.5, ORANGE, "mm", bold=True, mono=True)
    f.text(618, 256, "[T, o/2]", 11, MUT, "mm", mono=True)

    f.text(730, 158, "按列拼起来 = 完整的 [T, o]", 13.5, INK, bold=True)
    f.text(730, 186, "每张卡的输出天生就是结果的一段，", 11.5, MUT)
    f.text(730, 208, "不需要跟任何人交换 → 直接进下一层", 11.5, MUT)
    f.pill(730, 236, "0 次通信", GREEN_L, GREEN, 12)

    f.dashed(60, 320, 1344, 320, LINE, 1, (8, 6))

    # ---------- B 行并行 ----------
    f.text(46, 348, "B · 行并行 = 切输入维 i", 15, ORANGE, bold=True)
    f.text(300, 350, "（o_proj / down_proj 就是这种）", 12, MUT)

    f.chip(46, 405, 62, 30, "卡 0", BLUE_L, BLUE, 12)
    f.chip(46, 485, 62, 30, "卡 1", ORANGE_L, ORANGE, 12)

    f.rect([128, 392, 262, 448], fill=BLUE_L, outline=BLUE, width=2, radius=6)
    f.text(195, 406, "X[:, :i/2]", 12, BLUE, "mm", bold=True, mono=True)
    f.text(195, 428, "[T, i/2]", 10.5, MUT, "mm", mono=True)

    f.rect([128, 472, 262, 528], fill=ORANGE_L, outline=ORANGE, width=2, radius=6)
    f.text(195, 486, "X[:, i/2:]", 12, ORANGE, "mm", bold=True, mono=True)
    f.text(195, 508, "[T, i/2]", 10.5, MUT, "mm", mono=True)

    f.arrow(268, 420, 306, 420, MUT, 2, 7)
    f.arrow(268, 500, 306, 500, MUT, 2, 7)

    f.rect([312, 392, 452, 448], fill=BLUE_L, outline=BLUE, width=2, radius=6)
    f.text(382, 406, "W[:i/2, :]", 12, BLUE, "mm", bold=True, mono=True)
    f.text(382, 428, "[i/2, o]", 10.5, MUT, "mm", mono=True)

    f.rect([312, 472, 452, 528], fill=ORANGE_L, outline=ORANGE, width=2, radius=6)
    f.text(382, 486, "W[i/2:, :]", 12, ORANGE, "mm", bold=True, mono=True)
    f.text(382, 508, "[i/2, o]", 10.5, MUT, "mm", mono=True)

    f.arrow(458, 420, 496, 420, MUT, 2, 7)
    f.arrow(458, 500, 496, 500, MUT, 2, 7)

    f.rect([502, 392, 664, 448], fill=BLUE_L, outline=BLUE, width=2, radius=6)
    f.text(583, 406, "Y0 部分和", 12, BLUE, "mm", bold=True)
    f.text(583, 428, "[T, o]", 10.5, MUT, "mm", mono=True)

    f.rect([502, 472, 664, 528], fill=ORANGE_L, outline=ORANGE, width=2, radius=6)
    f.text(583, 486, "Y1 部分和", 12, ORANGE, "mm", bold=True)
    f.text(583, 508, "[T, o]", 10.5, MUT, "mm", mono=True)

    f.elbow([(670, 420), (700, 420), (700, 458), (712, 458)], MUT, 2, 7)
    f.elbow([(670, 500), (700, 500), (700, 458), (712, 458)], MUT, 2, 7)

    f.ellipse([716, 428, 776, 488], fill=SLATE_L, outline=INK, width=2)
    f.text(746, 458, "Σ", 19, INK, "mm", bold=True)

    f.arrow(782, 458, 826, 458, RED, 2.5, 9)
    f.rect([832, 424, 966, 492], fill=RED_L, outline=RED, width=2.2, radius=6)
    f.text(899, 442, "Y = Y0 + Y1", 12.5, RED, "mm", bold=True, mono=True)
    f.text(899, 466, "[T, o]", 11, MUT, "mm", mono=True)

    f.pill(996, 428, "每层 1 次 all-reduce", RED_L, RED, 12)
    f.text(996, 468, "两个部分和必须相加 → 这是 TP 唯一的", 11.5, MUT)
    f.text(996, 490, "通信来源（embedding 用 all-reduce，", 11.5, MUT)
    f.text(996, 512, "lm_head 用 all-gather，同一个道理）", 11.5, MUT)

    f.rect([46, 596, 60, 610], fill=BLUE_L, outline=BLUE, width=1.5, radius=3)
    f.text(66, 594, "卡 0 的切片", 11.5, MUT)
    f.rect([176, 596, 190, 610], fill=ORANGE_L, outline=ORANGE, width=1.5, radius=3)
    f.text(196, 594, "卡 1 的切片", 11.5, MUT)
    f.rect([306, 596, 320, 610], fill=SLATE_L, outline=SLATE, width=1.5, radius=3)
    f.text(326, 594, "两卡都完整的一份", 11.5, MUT)
    f.text(470, 594, "矩阵上的细线只是示意网格，不代表真实维度。", 11, FAINT)
    f.save("fig02-col-row-parallel.png")


# ============================================================ 图3 embedding 切分
def fig03():
    f = Fig(1400, 680)
    f.title("③ Embedding 权重：沿 vocab 维切成 4 份（TP=4）",
            "Qwen3-0.6B · vocab 151936 × hidden 1024 · 一行 = 一个 token 的 1024 维向量")

    box = (130, 186, 330, 516)
    f.text(130, 118, "单卡（TP=1）时有这么大一块", 12.5, MUT, bold=True)
    f.matrix(box, rows=16, cols=12)
    for i in (1, 2, 3):
        y = box[1] + (box[3] - box[1]) * i / 4
        f.dashed(122, y, 338, y, RED, 1.6, (5, 4))
    f.dim_v(box[1], box[3], 104, "vocab = 151936")
    f.dim_h(box[0], box[2], 170, "hidden = 1024")
    f.text(230, 528, "W  [151936, 1024]", 14.5, INK, "mt", bold=True, mono=True)

    f.arrow(352, 351, 744, 351, RED, 2.5, 10)
    f.text(548, 322, "按行切 / 4", 14.5, RED, "mb", bold=True)
    f.text(548, 366, "每段 37984 行", 12, MUT, "mt")

    for r in range(4):
        y = 186 + r * 86
        c, cl = RANKS[r]
        f.rect([778, y, 1050, y + 72], fill=cl, outline=c, width=2, radius=8)
        f.rect([778, y, 784, y + 72], fill=c)
        f.text(798, y + 11, f"rank {r}", 14, c, bold=True)
        f.text(798, y + 37, "[37984, 1024]", 12, INK, mono=True)
        f.text(1038, y + 11, "78 MB", 12, MUT, "rt")
        f.text(1038, y + 37, f"vocab 行 [{r*37984}, {(r+1)*37984})", 10.5, FAINT, "rt")

    f.rect([96, 566, 700, 650], fill=PANEL, outline=LINE, width=1, radius=8)
    for i, (c, s) in enumerate([
        (BLUE, "151936 × 1024 = 155.6M 参数"),
        (ORANGE, "bf16 → 311 MB"),
        (GREEN, "= 全模型 751.6M 的 20.7%"),
        (PURPLE, "tie=True → lm_head 复用这块"),
    ]):
        x = 112 if i % 2 == 0 else 400
        y = 584 + (i // 2) * 30
        f.rect([x, y + 3, x + 8, y + 11], fill=c, radius=2)
        f.text(x + 18, y, s, 12.5, INK)

    f.rect([778, 566, 1354, 650], fill=SLATE_L, outline=LINE, width=1, radius=8)
    f.text(796, 580, "显存 ÷ TP，代价是每次 forward 多 1 次 all-reduce", 13.5, INK, bold=True)
    f.text(796, 608, "为什么值得切：这一块占 20.7% 权重；回传 lm_head 时本来就要按 vocab 切",
           12, MUT)
    f.save("fig03-embedding-shard.png")


# ============================================================ 图4 查表 + all-reduce
def fig04():
    f = Fig(1400, 660)
    f.title("④ 查表：id 只命中一个 rank，其余输出 0 —— all-reduce 求和等于「取出来」",
            "4 个 rank 各自只认得自己那段 vocab；一次 forward 只通信 1 次")
    rows_y = [168, 264, 360, 456]
    for r, y in enumerate(rows_y):
        c, cl = RANKS[r]
        hit = (r == 1)
        f.rect([70, y, 344, y + 54], fill=(c if hit else "#FFFFFF"),
               outline=c, width=(2.5 if hit else 1.5), radius=8)
        f.text(86, y + 10, f"rank {r}", 13.5, ("#FFFFFF" if hit else c), bold=True)
        f.text(86, y + 30, f"vocab [{r*37984}, {(r+1)*37984})", 10.5,
               ("#FFFFFF" if hit else MUT))

        if hit:
            f.rect([386, y, 744, y + 54], fill=cl, outline=c, width=2.5, radius=8)
            f.text(404, y + 9, "W[12016] → 非零的 1024 维向量", 12.5, c, bold=True)
            f.text(404, y + 31, "（50000 − 37984 = 12016，本卡第 12016 行）", 10.5, MUT)
        else:
            f.rect([386, y, 744, y + 54], fill="#F7F9FC", outline="#DDE5EE",
                   width=1.5, radius=8)
            f.text(404, y + 12, "0   0   0   …   0", 13, FAINT, mono=True)
            f.text(560, y + 14, "（1024 维全零，等长占位）", 11, FAINT)

    f.pill(70, 112, "token id = 50000 落在 rank 1 的区间里", RED_L, RED, 11.5)
    f.arrow(200, 146, 200, 164, RED, 1.6, 6)

    bus = 800
    for r, y in enumerate(rows_y):
        c = RANKS[r][0] if r == 1 else "#C9D4E0"
        f.line(752, y + 27, bus, y + 27, c, 2.2 if r == 1 else 1.4)
    f.line(bus, rows_y[0] + 27, bus, 318, "#C9D4E0", 1.4)
    f.arrow(bus, 318, 852, 318, RED, 2.5, 9)

    f.ellipse([858, 288, 922, 352], fill=SLATE_L, outline=INK, width=2)
    f.text(890, 320, "Σ", 22, INK, "mm", bold=True)
    f.text(890, 362, "all-reduce", 11, MUT, "mt", mono=True)

    f.arrow(928, 320, 992, 320, RED, 2.5, 9)
    f.rect([998, 282, 1344, 358], fill=BLUE_L, outline=BLUE, width=2.5, radius=8)
    f.text(1016, 292, "x[id] : 完整 [1024] 向量", 14, BLUE, bold=True)
    f.text(1016, 320, "每张卡都拿到同一份，喂给下一层", 11.5, MUT)

    f.rect([70, 540, 1344, 624], fill="#FFFBF5", outline="#F3D6AE", width=1.5, radius=8)
    f.text(88, 554, "为什么不会丢精度：一个 id 只属于一个 rank，其余恒为 0，"
                    "0 + x 在 bf16 下精确无损。", 12.5, INK)
    f.text(88, 582, "成本：1 次 all-reduce / forward，数据量 T × 1024 × 2B —— "
                    "相对 28 层里的 56 次，可忽略。", 12.5, MUT)
    f.save("fig04-vocab-lookup.png")


# ============================================================ 图5 GQA
def fig05():
    f = Fig(1400, 700)
    f.title("⑤ GQA 的 8 个 KV head，在 5 种 TP 下分别落在哪张卡上",
            "方块颜色 = 拥有它的 rank；TP=16 时 16 张卡分 8 个 head，必然有两卡算同一份")

    def owner_color(r, n):
        h = (0.58 + (r / max(n, 1)) * 0.85) % 1.0
        rr, gg, bb = colorsys.hsv_to_rgb(h, 0.62, 0.92)
        r2, g2, b2 = colorsys.hsv_to_rgb(h, 0.26, 1.0)
        return (f"#{int(rr*255):02X}{int(gg*255):02X}{int(bb*255):02X}",
                f"#{int(r2*255):02X}{int(g2*255):02X}{int(b2*255):02X}")

    f.text(180, 108, "8 个 KV head，每个 128 维", 11.5, MUT, "mt")
    tps = [(1, 8, "每卡 8 个（全都要）"),
           (2, 4, "每卡 4 个"),
           (4, 2, "每卡 2 个"),
           (8, 1, "每卡 1 个，正好"),
           (16, 1, "每卡 1 个，但【复制】—— 8 个变成 16 份")]
    y = 140
    for tp, per, note in tps:
        f.text(60, y + 14, f"TP = {tp}", 13, INK, bold=True)
        f.text(60, y + 36, f"每卡 {per} 个", 10.5, MUT)
        dup = tp > 8
        for h in range(8):
            x = 180 + h * 62
            owner = (h * tp) // 8
            c, cl = owner_color(owner, min(tp, 8))
            f.rect([x, y, x + 52, y + 52], fill=cl, outline=c, width=2, radius=6)
            f.text(x + 26, y + 16, f"kv{h}", 10.5, c, "mm", bold=True, mono=True)
            f.text(x + 26, y + 34, f"r{owner}", 9.5, c, "mm", mono=True)
            if dup:
                f.dashed_rect([x - 4, y - 4, x + 56, y + 56], outline=RED,
                              width=1.4, dash=(4, 3))
        f.text(700, y + 8, note, 11.5, (RED if dup else MUT), bold=dup)
        if dup:
            f.text(700, y + 30, "虚线框 = 这个 head 被 2 张卡各存了一份", 10.5, FAINT)
        if tp == 8:
            f.text(700, y + 30, "q head 是 16 个，所以 q 还能继续切", 10.5, FAINT)
        y += 96

    f.rect([60, 620, 1340, 676], fill=PANEL, outline=LINE, width=1.2, radius=8)
    f.text(78, 632, "规则（utils/misc.py:20-26 div_even）：TP ≤ num_kv_heads 时整除切分；"
                    "TP > num_kv_heads 时 allow_replicate → 每卡 1 个、重复放置。", 12, INK)
    f.text(78, 658, "所以 KV cache 的显存不是永远 ÷TP：TP 超过 8 之后，"
                    "每卡仍要存那 1 个 head 的 page。", 12, MUT)
    f.save("fig05-gqa-head-split.png")


# ============================================================ 图6 通信账本
def fig06():
    f = Fig(1400, 520)
    f.title("⑥ 一次 forward 的通信账：58 次，56 次都埋在层里",
            "prefill T=2048 时每次 all-reduce 约 4 MB —— 大头在层内，不在 embedding")

    y0, y1 = 128, 200
    f.rect([100, y0, 300, y1], fill=BLUE_L, outline=BLUE, width=2, radius=8)
    f.text(200, y0 + 14, "Embedding", 13, BLUE, "mt", bold=True)
    f.text(200, y0 + 38, "all-reduce ×1", 11, BLUE, "mt")

    f.rect([312, y0, 1090, y1], fill=RED_L, outline=RED, width=2, radius=8)
    for i in range(28):
        x = 320 + i * 27.4
        f.line(x, y0 + 4, x, y0 + 20, RED, 1.2)
        f.line(x + 10, y0 + 4, x + 10, y0 + 20, RED, 1.2)
    f.text(701, y0 + 26, "28 层 × 2 次 = 56 次（o_proj + down_proj）", 12.5, RED, "mt", bold=True)
    f.text(701, y0 + 48, "每层：QKV(不通信) → Attention(不通信) → O(1 次) → MLP(1 次)",
           10.5, MUT, "mt")

    f.rect([1102, y0, 1302, y1], fill=ORANGE_L, outline=ORANGE, width=2, radius=8)
    f.text(1202, y0 + 14, "LM head", 13, ORANGE, "mt", bold=True)
    f.text(1202, y0 + 38, "all-gather ×1", 11, ORANGE, "mt")

    f.text(100, 242, "次数占比", 12, MUT, bold=True)
    total = 58.0
    w, x = 1200, 100
    for (n, c) in [(1, BLUE), (56, RED), (1, ORANGE)]:
        seg = w * n / total
        f.rect([x, 266, x + seg, 300], fill=c, radius=3)
        x += seg
    f.rect([100, 266, 1300, 300], outline=LINE, width=1, radius=3)
    f.text(102, 308, "embedding 1 次（1.7%）", 10.5, BLUE)
    f.text(700, 308, "层内 56 次（96.6%）", 10.5, RED, "mt")
    f.text(1298, 308, "lm_head 1 次（1.7%）", 10.5, ORANGE, "rt")

    f.rect([100, 356, 1300, 462], fill="#FFFBF5", outline="#F3D6AE", width=1.5, radius=8)
    f.text(118, 370, "所以：优化 embedding 的那次 all-reduce 没有意义，要动就动层内那 56 次"
                     "（融合、或干脆减少 TP）。", 12.5, INK)
    f.text(118, 398, "lm_head 的 all-gather 也不大：prefill 只对每条序列最后一个 token 算 "
                     "logits（embedding.py:92-95），bs=1 时只有 37984×2B ≈ 76 KB。", 12.5, MUT)
    f.text(118, 426, "注：TP=1 时这 58 次全部短路，一次都不发生"
                     "（每个类里都有 tp_size > 1 判断）。", 12.5, RED)
    f.save("fig06-comm-ledger.png")


# ============================================================ 图7 权重加载
def fig07():
    f = Fig(1400, 680)
    f.title("⑦ 权重是「切着加载」的：checkpoint 张量 → 每卡分片（TP=4）",
            "BaseOP.load_state_dict 对形状做严格断言（base.py:45），"
            "所以必须在加载时切，不能加载完再切")

    def group(y0, y1, title, color, cl, items):
        f.dashed_rect([60, y0, 470, y1], outline=color, width=1.6, dash=(7, 5),
                      fill="#FCFDFF")
        f.text(76, y0 + 8, title, 12.5, color, bold=True)
        for i, (nm, shp) in enumerate(items):
            yy = y0 + 34 + i * 30
            f.rect([84, yy, 446, yy + 24], fill=cl, outline=color, width=1, radius=4)
            f.text(96, yy + 4, nm, 10.5, INK, mono=True)
            f.text(432, yy + 5, shp, 10, MUT, "rt", mono=True)

    group(96, 300, "① 切第 0 维：chunk(dim=0)[r]", BLUE, BLUE_L, [
        ("q_proj.weight", "[4096, 1024]"),
        ("k_proj.weight", "[1024, 1024]"),
        ("v_proj.weight", "[1024, 1024]"),
        ("gate_proj.weight", "[3072, 1024]"),
        ("up_proj.weight", "[3072, 1024]"),
    ])
    group(334, 448, "② 切第 1 维：chunk(dim=1)[r]", ORANGE, ORANGE_L, [
        ("o_proj.weight", "[1024, 2048]"),
        ("down_proj.weight", "[1024, 3072]"),
    ])
    group(462, 552, "③ 按 vocab 区间切片", GREEN, GREEN_L, [
        ("embed_tokens.weight", "[151936, 1024]"),
    ])

    f.arrow(474, 160, 546, 160, BLUE, 2, 8)
    f.text(510, 136, "行", 11, BLUE, "mb", bold=True)
    f.arrow(474, 390, 546, 390, ORANGE, 2, 8)
    f.text(510, 366, "列", 11, ORANGE, "mb", bold=True)
    f.arrow(474, 506, 546, 506, GREEN, 2, 8)
    f.text(510, 482, "vocab", 10.5, GREEN, "mb", bold=True)

    f.text(560, 96, "加载后每张卡持有的（TP=4）", 12.5, INK, bold=True)
    rt = [("qkv_proj", "[1536, 1024]", "q 1024 + k 256 + v 256 行拼成", BLUE, BLUE_L),
          ("o_proj", "[1024, 512]", "输入维被切掉 3/4", ORANGE, ORANGE_L),
          ("gate_up_proj", "[1536, 1024]", "gate 768 + up 768 行拼成", BLUE, BLUE_L),
          ("down_proj", "[1024, 768]", "输入维被切掉 3/4", ORANGE, ORANGE_L),
          ("embed_tokens", "[37984, 1024]", "vocab 只留自己那段", GREEN, GREEN_L)]
    y = 128
    for nm, shp, note, c, cl in rt:
        f.rect([560, y, 900, y + 62], fill=cl, outline=c, width=1.8, radius=7)
        f.text(576, y + 8, nm, 12, c, bold=True, mono=True)
        f.text(576, y + 28, shp, 10.5, INK, mono=True)
        f.text(576, y + 44, note, 10, MUT)
        y += 74

    f.rect([926, 128, 1358, 250], fill="#FFFBF5", outline="#F3D6AE", width=1.5, radius=8)
    f.text(944, 142, "为什么要「拼」", 12.5, INK, bold=True)
    f.text(944, 168, "checkpoint 里是 q/k/v 三个独立矩阵，", 11.5, MUT)
    f.text(944, 190, "运行时是 qkv_proj 一个融合矩阵。", 11.5, MUT)
    f.text(944, 212, "加载时用 merge_buf 凑齐三个 slot 再 cat", 11.5, MUT)
    f.text(944, 232, "（weight.py:100-109）。", 10.5, FAINT, mono=True)

    f.rect([926, 266, 1358, 400], fill=PANEL, outline=LINE, width=1.2, radius=8)
    f.text(944, 280, "不切会怎样", 12.5, INK, bold=True)
    f.text(944, 306, "base.py:45 断言 param.shape == item.shape，", 11, MUT)
    f.text(944, 328, "形状不一致立刻 assert 炸，不会静默算错。", 11, MUT)
    f.text(944, 356, "所以模型的 TP 必须整除：", 11, MUT)
    f.text(944, 376, "num_qo_heads=16、intermediate=3072。", 10.5, INK, mono=True)

    f.text(60, 590, "补充：k/v 在 TP > num_kv_heads 时不是除以 TP，而是按 head 复制"
                    "（weight.py:38-41，head_idx = r × num_kv_heads // TP）。", 11.5, MUT)
    f.text(60, 618, "norm / bias / router gate 这些一维或小张量不切，每卡都存一份完整拷贝"
                    "（weight.py:51-52 的 else 分支）。", 11.5, MUT)
    f.save("fig07-weight-sharding.png")


# ============================================================ 图8 显存预算
def fig08():
    f = Fig(1400, 780)
    f.title("⑧ 8 GB 显存怎么分：KV cache 是从「余量」里抢出来的",
            "Qwen3-0.6B / bf16 / RTX 3070 8 GB —— 同一张卡，三档 --num-pages 的账")

    # 公式条
    f.rect([100, 92, 1300, 176], fill=PANEL, outline=LINE, width=1.2, radius=8)
    f.text(118, 104, "源码里的两步（engine.py::_determine_num_pages）", 11.5, INK, bold=True)
    f.text(118, 126, "cache_per_page = 2(K,V) × 128(head_dim) × 8(kv_heads) × 1(page) × 2B × 28(层)"
                     " = 114,688 B = 112 KiB / token", 11.5, MUT, mono=False)
    f.text(118, 150, "num_pages = (int(memory_ratio × 加载前的空闲显存) − 权重占用) // cache_per_page",
           11.5, MUT, mono=False)

    # 坐标
    y0, top = 590, 210          # 0 GiB 与 8 GiB
    per = (y0 - top) / 8.0      # 每 GiB 的像素
    for g in range(0, 9, 2):
        yy = y0 - g * per
        f.line(150, yy, 1290, yy, GRID if g else LINE, 1)
        f.text(136, yy, f"{g} GiB", 10.5, FAINT, "rm", mono=True)
    f.dashed(150, y0 - 8 * per, 1290, y0 - 8 * per, RED, 1.4, (6, 4))
    f.text(1290, y0 - 8 * per - 4, "8 GiB 上限", 10.5, RED, "rb", bold=True)

    data = [("默认\n46932 pages", 5.01, "ratio 0.9"),
            ("40960", 4.38, "--num-pages 40960"),
            ("推荐 32768", 3.50, "--num-pages 32768"),
            ("16384", 1.75, "--num-pages 16384"),
            ("8192", 0.88, "--num-pages 8192")]
    w = 150
    for i, (name, kv, sub) in enumerate(data):
        x = 210 + i * 218
        h_kv = kv * per
        h_w = 1.5 * per
        h_rest = 8 * per - h_kv - h_w
        # 余量
        f.rect([x, top, x + w, top + h_rest], fill=GREEN_L, outline=GREEN, width=1, radius=2)
        # KV
        f.rect([x, top + h_rest, x + w, y0 - h_w], fill=ORANGE_L, outline=ORANGE,
               width=1.6, radius=2)
        # 权重
        f.rect([x, y0 - h_w, x + w, y0], fill=SLATE_L, outline=SLATE, width=1, radius=2)

        f.text(x + w / 2, top + h_rest / 2, f"{8 - 1.5 - kv:.2f}", 10.5, GREEN, "mm",
               bold=True)
        if h_kv > 40:
            f.text(x + w / 2, top + h_rest + h_kv / 2, f"KV {kv:.2f}", 12, ORANGE, "mm",
                   bold=True)
        for li, seg in enumerate(name.split("\n")):
            f.text(x + w / 2, y0 + 10 + li * 15, seg, 10.5, INK, "mt", bold=True)
        f.text(x + w / 2, y0 + 44, sub, 9.5, FAINT, "mt")

    # 图例（横向放在图下方，避免压住柱子）
    lx = 120
    for (c, cl, s) in [(SLATE, SLATE_L, "权重 1.5 GiB（固定）"),
                       (ORANGE, ORANGE_L, "KV cache（随 --num-pages 变化）"),
                       (GREEN, GREEN_L, "余量：context / graph / 激活 / 碎片")]:
        f.rect([lx, 652, lx + 14, 664], fill=cl, outline=c, width=1.2, radius=2)
        f.text(lx + 20, 651, s, 10.5, MUT)
        lx += 30 + f.d.textlength(s, font=font(10.5)) / S

    # 结论
    f.rect([100, 690, 1300, 760], fill="#FFFBF5", outline="#F3D6AE", width=1.5, radius=8)
    f.text(118, 702, "两条硬约束：① 余量不能压到 0 —— CUDA context、CUDA graph 缓冲、"
                     "激活值、显存碎片都要钱，压没了就是 OOM 或被 kill；", 12.5, INK)
    f.text(118, 728, "② KV 池变小会同时压低单条请求的上限：max_seq_len = min(40960, num_tokens)，"
                     "所以 --num-pages 8192 时 8K 以上的请求根本发不出去。", 12.5, MUT)
    f.save("fig08-vram-budget.png")


# ============================================================ 图9 KV 池共享
def fig09():
    f = Fig(1400, 620)
    f.title("⑨ KV 池是「所有并发请求共享」的一个池子，不是每条请求各给一个",
            "以 --num-pages 32768 为例：池子 32768 token，装满就拒绝新请求")

    x0, x1, y0, y1 = 120, 1280, 180, 240
    total = 32768
    f.rect([x0, y0, x1, y1], fill="#F7F9FC", outline=SLATE, width=2, radius=6)

    reqs = [("请求 A", 8192, BLUE, BLUE_L), ("B", 4096, ORANGE, ORANGE_L),
            ("C", 2048, GREEN, GREEN_L), ("D", 16384, PURPLE, PURPLE_L)]
    x = x0
    used = 0
    for name, n, c, cl in reqs:
        wseg = (x1 - x0) * n / total
        f.rect([x, y0, x + wseg, y1], fill=cl, outline=c, width=1.6, radius=3)
        f.text(x + wseg / 2, (y0 + y1) / 2 - 8, name, 12, c, "mm", bold=True)
        f.text(x + wseg / 2, (y0 + y1) / 2 + 12, f"{n} tok", 10, MUT, "mm", mono=True)
        x += wseg
        used += n

    left = total - used
    f.rect([x, y0, x1, y1], fill=GREEN_L, outline=GREEN, width=1.4, radius=3)
    f.text((x + x1) / 2, (y0 + y1) / 2, f"空 {left}", 10, GREEN, "mm", bold=True)

    f.text(x0, 152, "KV 池  32768 token（--num-pages 32768）", 12, INK, bold=True)
    f.text(x1, 152, f"已用 {used} / {total}", 11, MUT, "rt", mono=True)

    # 第 5 个请求
    f.rect([120, 300, 700, 420], fill=RED_L, outline=RED, width=2, radius=8)
    f.text(140, 314, "请求 E：prompt 30000 + 生成 10960 = 40960 token", 12.5, RED, bold=True)
    f.text(140, 344, "池子只剩 %d，装不下 → 排队，或直接拒绝" % left, 12, INK)
    f.text(140, 372, "就算池子空着也不行：单条上限 = min(40960, 32768)", 12, INK)
    f.text(140, 394, "（engine 里 max_seq_len = min(config.max_seq_len, num_tokens)）", 10.5, MUT)

    # 右侧说明
    f.rect([730, 300, 1280, 420], fill=PANEL, outline=LINE, width=1.2, radius=8)
    f.text(748, 314, "所以调参时要想清楚两件事", 12.5, INK, bold=True)
    for i, s in enumerate([
        "并发高、序列短 → 池子小一点也能跑满，比如 16384",
        "要跑长上下文（>32K） → 池子必须 ≥ 该长度，且并发只能 1~2",
        "池子是独占的：一条 30K 的请求会把整个池子吃掉 90%",
    ]):
        f.text(748, 344 + i * 26, "· " + s, 11.5, MUT)

    f.rect([120, 460, 1290, 540], fill="#FFFBF5", outline="#F3D6AE", width=1.5, radius=8)
    f.text(138, 474, "一个常见误解：以为 --max-running-requests 256 就意味着「256 条请求各有自己的 KV」。"
                     "实际上 page 表是共享的，256 只是队列上限。", 12.5, INK)
    f.text(138, 504, "池子一满，新请求要么排队、要么报错；这也是长 prompt 打到短池子上会「莫名失败」的原因。",
           12.5, MUT)
    f.save("fig09-kv-pool-shared.png")


# ============================================================ 图10 进程拓扑
def fig10():
    f = Fig(1400, 780)
    f.title("⑩ 进程拓扑：一个 main + 每卡一个 scheduler + 一个 detokenizer",
            "默认 --tp-size 1、--num-tokenizer 0（分词与反分词共用同一个进程）")

    def box(x0, y0, x1, y1, title, lines, color, cl, mono_first=False):
        f.rect([x0, y0, x1, y1], fill=cl, outline=color, width=2, radius=10)
        f.text((x0 + x1) / 2, y0 + 10, title, 13.5, color, "mt", bold=True)
        for i, s in enumerate(lines):
            f.text((x0 + x1) / 2, y0 + 38 + i * 20, s, 10.5, MUT, "mt")

    # 客户端
    box(60, 128, 250, 208, "客户端", ["curl / OpenAI SDK"], SLATE, SLATE_L)
    # main
    box(560, 108, 940, 214, "main 进程", ["python -m minisgl", "FastAPI · 0.0.0.0:8000"],
        BLUE, BLUE_L)
    # detokenizer
    box(120, 356, 520, 480, "detokenizer", ["分词 + 反分词（num_tokenizer=0 时共用一个）",
                                            "minisgl-detokenizer-0"], ORANGE, ORANGE_L)
    # scheduler
    box(880, 356, 1300, 480, "scheduler TP0", ["Scheduler.run_forever()",
                                               "占 GPU · 持有 KV cache", "minisgl-TP0-scheduler"],
        GREEN, GREEN_L)

    # 客户端 <-> main
    f.arrow(258, 168, 552, 168, MUT, 2, 8)
    f.text(405, 148, "HTTP / SSE", 11, MUT, "mb", mono=True)
    f.arrow(552, 190, 258, 190, MUT, 2, 8)

    # main <-> detok
    f.arrow(620, 222, 420, 348, ORANGE, 2.2, 8)
    f.text(486, 296, "TokenizeMsg（prompt 文本）", 11, ORANGE, "rt")
    f.arrow(470, 348, 700, 222, ORANGE, 2.2, 8)
    f.text(710, 238, "UserReply（增量文本）", 11, ORANGE, "lt")

    # detok <-> scheduler
    f.arrow(528, 400, 872, 400, GREEN, 2.2, 8)
    f.text(700, 384, "token ids", 11, GREEN, "mb", mono=True)
    f.arrow(872, 440, 528, 440, GREEN, 2.2, 8)
    f.text(700, 448, "BatchTokenizerMsg（新 token）", 11, GREEN, "lt")

    # ipc 说明
    f.rect([120, 520, 1300, 590], fill=PANEL, outline=LINE, width=1.2, radius=8)
    f.text(138, 532, "四个 ZMQ 通道都在本机 ipc：", 11.5, INK, bold=True)
    f.text(138, 556, "zmq_frontend_addr（main ← detok 回文本）  ·  zmq_tokenizer_addr"
                     "（main → detok 送 prompt / scheduler → detok 送新 token）", 11, MUT)
    f.text(138, 574, "zmq_backend_addr（detok → scheduler 送 ids）  ·  "
                     "distributed_addr（TP 之间的 NCCL 引导，tcp:port+1）", 11, MUT)

    f.rect([120, 610, 1300, 700], fill="#FFFBF5", outline="#F3D6AE", width=1.5, radius=8)
    f.text(138, 622, "关键点：GPU 只在 scheduler 进程里被使用；main 只做 HTTP 和路由，"
                     "detokenizer 是纯 CPU 的 tokenize/detokenize。", 12.5, INK)
    f.text(138, 650, "所以 tp_size=2 时会有 2 个 scheduler、2 份权重、2 份 KV cache，"
                     "而 main 和 detokenizer 仍然只有一个。", 12.5, MUT)
    f.text(138, 676, "代码：server/launch.py:49-119（spawn 与 ack）、server/args.py:21-47（地址与共享开关）。",
           11, FAINT)
    f.save("fig10-process-topology.png")


# ============================================================ 图11 启动时序 + 进程名
def fig11():
    f = Fig(1400, 700)
    f.title("11 · 启动都发生了什么：7 步，以及 ps 里应该看到什么",
            "左边是 launch_server() 的执行顺序（launch.py），右边是同一时刻 ps 的样子")

    steps = [
        ("1", "终端", "python -m minisgl --model ... --memory-ratio 0.8", SLATE, SLATE_L),
        ("2", "main", "parse_args() → ServerArgs(tp_size=1, num_tokenizer=0)", BLUE, BLUE_L),
        ("3", "main", "起 FastAPI（:8000），把 start_subprocess 当回调传进去", BLUE, BLUE_L),
        ("4", "main", "start_subprocess()：set_start_method(\"spawn\")；"
                      "patch_resource_tracker_title()", BLUE, BLUE_L),
        ("5", "spawn", "scheduler × tp_size（每卡一个，各自建 CUDA context、加载权重）", GREEN, GREEN_L),
        ("6", "spawn", "detokenizer × 1（+ tokenizer × num_tokenizer，默认 0 个）", ORANGE, ORANGE_L),
        ("7", "main", "阻塞等 num_tokenizer + 2 个 ack，全部到齐才打印 ready", BLUE, BLUE_L),
    ]
    y = 104
    for num, who, text, c, cl in steps:
        f.rect([60, y, 96, y + 28], fill=c, outline=c, width=1.5, radius=14)
        f.text(78, y + 14, num, 12, "#FFFFFF", "mm", bold=True)
        f.rect([108, y - 6, 690, y + 46], fill=cl, outline=c, width=1.4, radius=8)
        f.text(122, y + 1, who, 11, c, bold=True)
        f.text(122, y + 20, text, 11.5, INK)
        if num != "7":
            f.line(78, y + 28, 78, y + 56, "#B8C4D4", 1.4)
        y += 56

    # 右侧 ps 对照
    f.text(740, 104, "对应时刻 ps 里看到的（procname.py 改名前后）", 12, INK, bold=True)

    f.rect([740, 130, 1350, 250], fill="#FDF5F5", outline=RED, width=1.4, radius=8)
    f.text(756, 142, "改名前：认不出来", 11.5, RED, bold=True)
    for i, s in enumerate([
        "python -m minisgl --model /home/mzygd/models/Qwen3-0.6B ...",
        "python -c from multiprocessing.spawn import spawn_main; spawn_main(...)",
        "python -c from multiprocessing.spawn import spawn_main; spawn_main(...)",
        "python -c from multiprocessing.resource_tracker import main;main(18)",
    ]):
        f.text(756, 166 + i * 20, s, 9.5, MUT, mono=True)

    f.rect([740, 268, 1350, 388], fill="#F3FBF7", outline=GREEN, width=1.4, radius=8)
    f.text(756, 280, "改名后：一眼看懂", 11.5, GREEN, bold=True)
    for i, s in enumerate([
        "python -m minisgl --model /home/mzygd/models/Qwen3-0.6B ...",
        "minisgl-TP0-scheduler",
        "minisgl-detokenizer-0",
        "minisgl-resource-tracker",
    ]):
        f.text(756, 304 + i * 20, s, 9.5, MUT, mono=True)

    f.rect([740, 408, 1350, 500], fill=PANEL, outline=LINE, width=1.2, radius=8)
    f.text(756, 420, "三个启动期的坑（都踩过）", 11.5, INK, bold=True)
    for i, s in enumerate([
        "spawn 不像 fork 那样继承父进程状态 → 环境变量要在 .bashrc 里备好",
        "等 ack 是阻塞的 → 某个子进程起不来，main 会卡在 ack_queue.get()（launch.py:118）",
        "daemon=False → Ctrl-C 后子进程可能残留，用 stop.sh 清",
    ]):
        f.text(756, 446 + i * 20, "· " + s, 10.5, MUT)

    f.rect([60, 560, 1350, 640], fill="#FFFBF5", outline="#F3D6AE", width=1.5, radius=8)
    f.text(78, 572, "调试入口：VSCode Remote-WSL + debugpy（见 4.1）；要跟子进程就在 "
                    "_run_scheduler 第一行下断点。", 12.5, INK)
    f.text(78, 600, "注意 spawn 出的子进程不继承父进程的 debugpy attach —— 想断进去要子进程自己 "
                    "listen（debugpy.listen + wait_for_attachment）。", 12.5, MUT)
    f.text(78, 626, "本页时序据 launch.py 源码整理，逐步断点未实测 → 标「待验证」。", 11, RED)
    f.save("fig11-startup-sequence.png")


if __name__ == "__main__":
    want = sys.argv[1:] or ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
    for n in want:
        globals()[f"fig{n.zfill(2)}"]()
