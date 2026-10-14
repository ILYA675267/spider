"""Ролик №7: «Почему пальцы морщатся в воде?».

python tools/fingers_mg.py work/fingers/voice_clean.wav output/fingers.mp4 [--preview] [--jobs 4]
Общий движок, персонаж и концовка с Telegram — из memory_mg.py; параллельный рендер — как в money_mg.py.
Новое в монтаже: камера падает сверху вместе с каплей, по полу расходятся круги; заставка всплывает из пола;
смена сцен «водой» — кадр идёт рябью, под ней склейка; светящийся нервный сигнал; разрез пальца с сосудами;
подошва, по бороздкам которой уходит вода; 3D-шарики перелетают из миски в миску + «гонка» пальцев.
Звуки: только тихий «вжух» на смене ракурса (правило пользователя).
"""
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memory_mg as base  # noqa: E402
import mg  # noqa: E402
from memory_mg import (BLUE_RGB, GRAY_RGB, RED_RGB, TG_RGB, basis, card, chip, face, place, pop,  # noqa: E402,F401
                       rounded, stamp)
from mg import H, W, clamp, out_back, smooth, track, wobble  # noqa: E402

FPS = 30
ROOT = mg.ROOT
OUTRO = float(open(os.path.join(ROOT, "work", "fingers", "outro_start.txt")).read())  # 37.25
DUR = 45.9
AQUA_RGB, SKIN_RGB, GOLD_RGB, GREEN_RGB = (60, 200, 230), (226, 180, 150), (240, 190, 60), (46, 196, 110)
AQUA = AQUA_RGB[::-1]

# ---------------------------------------------------------------- субтитры
T_, A_, G_ = TG_RGB, AQUA_RGB, GREEN_RGB
SUBS = [
    (0.228, "ПОЧЕМУ ПАЛЬЦЫ МОРЩАТСЯ", ("МОРЩАТСЯ",), A_), (1.408, "В ВОДЕ?", ("ВОДЕ?",), A_), (2.2, "", ()),
    (3.528, "КАЖЕТСЯ, ЧТО КОЖА", ()), (4.448, "ПРОСТО НАМОКЛА И НАБУХЛА", ("НАБУХЛА",)),
    (6.008, "НО НЕТ", ("НЕТ",)), (6.968, "ВРАЧИ ЗАМЕТИЛИ:", ()), (8.308, "ЕСЛИ НЕРВ В ПАЛЬЦЕ ПОВРЕЖДЁН,", ("НЕРВ",), A_),
    (10.088, "ЭТОТ ПАЛЕЦ В ВОДЕ", ()), (10.968, "НЕ МОРЩИТСЯ", ("НЕ", "МОРЩИТСЯ")),
    (11.888, "ЗНАЧИТ, МОРЩИНЫ ВКЛЮЧАЕТ", ("ВКЛЮЧАЕТ",)), (13.168, "НЕРВНАЯ СИСТЕМА", ("НЕРВНАЯ", "СИСТЕМА"), A_),
    (14.598, "НЕРВЫ СЖИМАЮТ СОСУДЫ", ("СЖИМАЮТ",), A_), (15.838, "ПОД КОЖЕЙ,", ()),
    (16.378, "И ПОДУШЕЧКИ ПАЛЬЦЕВ", ()), (17.498, "НЕМНОГО СДУВАЮТСЯ", ("СДУВАЮТСЯ",)),
    (18.798, "ТАК ПОЯВЛЯЮТСЯ БОРОЗДКИ,", ("БОРОЗДКИ",), A_), (20.258, "КАК РИСУНОК НА ПОДОШВЕ", ("ПОДОШВЕ",), A_),
    (21.258, "КРОССОВОК", ()), (22.278, "ВОДА УХОДИТ ПО НИМ,", ("ВОДА",), A_),
    (23.298, "И ПАЛЬЦЫ НЕ СКОЛЬЗЯТ", ("НЕ", "СКОЛЬЗЯТ")), (24.943, "УЧЁНЫЕ ИЗ НЬЮКАСЛА", ("НЬЮКАСЛА",), BLUE_RGB),
    (26.003, "ЭТО ПРОВЕРИЛИ", ()), (26.903, "ЛЮДИ ПЕРЕКЛАДЫВАЛИ", ()), (27.843, "МОКРЫЕ ШАРИКИ", ("МОКРЫЕ",), A_),
    (28.623, "И СО СМОРЩЕННЫМИ ПАЛЬЦАМИ", ("СМОРЩЕННЫМИ",)), (29.903, "СПРАВЛЯЛИСЬ БЫСТРЕЕ", ("БЫСТРЕЕ",), G_),
    (31.063, "А С СУХИМИ ШАРИКАМИ", ("СУХИМИ",)), (32.063, "РАЗНИЦЫ НЕ БЫЛО", ("НЕ", "БЫЛО")),
    (33.423, "ПОЛУЧАЕТСЯ, ТЕЛО САМО", ()), (34.703, "НАДЕВАЕТ ТЕБЕ", ()), (35.343, "НЕСКОЛЬЗЯЩИЕ ПЕРЧАТКИ", ("ПЕРЧАТКИ",), A_),
    (37.22, "А ЕЩЁ БОЛЬШЕ ИНТЕРЕСНОГО —", (), T_), (38.58, "В МОЁМ ТЕЛЕГРАМ-КАНАЛЕ", ("ТЕЛЕГРАМ-КАНАЛЕ",), T_),
    (39.91, "«ПАУЧЬЯ НОРКА»", ("«ПАУЧЬЯ", "НОРКА»"), T_), (41.03, "ТАМ ЭКСКЛЮЗИВ,", ("ЭКСКЛЮЗИВ",), T_),
    (41.73, "КОТОРОГО НЕТ В РОЛИКАХ", (), T_), (43.03, "ССЫЛКА В ПРОФИЛЕ —", ("ПРОФИЛЕ",), T_),
    (44.05, "ЗАЛЕТАЙ!", ("ЗАЛЕТАЙ!",), T_),
]

# ---------------------------------------------------------------- камера и сцены
RIPPLES = (3.25, 14.1, 24.55)            # «водяная» рябь по кадру; под ней — склейка
C1, C2, C3 = (r + 0.25 for r in RIPPLES)
CAMERA = [
    (0.0, (-8.0, 2.4, 3.6, 0.5, 54.0)),                     # сверху, вместе с каплей…
    (1.3, (-8.0, 3.6, 1.15, 0.84, 50.0), "inout"),          # …падаем к полу
    (2.0, (-6.0, 3.6, 1.15, 0.86, 50.0)),
    (2.5, (0.0, 3.9, 1.3, 0.98, 50.0), "inout"),            # заставка всплывает из пола
    (C1 - 0.01, (2.0, 3.9, 1.3, 0.98, 50.0)),
    (C1 + 0.01, (95.0, 3.6, 1.2, 0.84, 50.0), "lin"),       # не набухла — включают нервы
    (C2 - 0.01, (102.0, 3.5, 1.2, 0.86, 50.0)),
    (C2 + 0.01, (190.0, 3.6, 1.1, 0.88, 50.0), "lin"),       # сосуды и подошва — медленный наезд
    (C3 - 0.01, (194.0, 3.5, 1.15, 0.88, 50.0)),
    (C3 + 0.01, (285.0, 3.6, 1.25, 0.84, 50.0), "lin"),     # опыт с шариками — медленный облёт
    (36.9, (300.0, 3.7, 1.3, 0.86, 50.0), "lin"),
    (37.3, (370.0, 4.3, 1.3, 1.12, 50.0), "expo"), (DUR, (371.0, 4.0, 1.3, 1.15, 50.0)),
]
OFFSET = [(0.0, -0.5), (2.0, -0.5), (2.5, 0.0, "inout"), (C1 - 0.01, 0.0), (C1 + 0.01, 0.55, "lin"),
          (C2 - 0.01, 0.55), (C2 + 0.01, -0.55, "lin"), (C3 - 0.01, -0.55), (C3 + 0.01, 0.55, "lin"),
          (36.9, 0.55), (37.3, 0.0, "expo"), (DUR, 0.0)]
TINT = [(0.0, (1.16, 1.06, 0.84)), (C1 - 0.01, (1.16, 1.06, 0.84)), (C1 + 0.01, (1.1, 0.98, 0.92), "lin"),
        (C2 - 0.01, (1.1, 0.98, 0.92)), (C2 + 0.01, (0.96, 1.0, 1.06), "lin"), (C3 - 0.01, (0.96, 1.0, 1.06)),
        (C3 + 0.01, (1.14, 1.08, 0.84), "lin"), (37.0, (1.14, 1.08, 0.84)), (37.4, (1.0, 1.0, 1.0)), (DUR, (1.0, 1.0, 1.0))]
SEC = {"hook": -8, "nerve": 98, "skin": 192, "test": 290, "tg": 370.5}
SHAKES = ()
FLASHES = ((2.25, 0.12), (37.05, 0.12))
JUMPS = []
ALARM = None
SPOT = None

G = lambda *k: list(k)  # noqa: E731
ARM_L = G((0, 62), (0.6, 62), (0.85, -25, "back"), (1.9, -22), (2.25, -50, "back"), (3.2, -45), (3.4, 62, "back"),
          (3.6, 36, "back"), (5.6, 38), (5.9, 62), (C2 + 0.05, 62), (14.6, -30, "back"), (18.6, -28),
          (18.85, -15, "back"), (24.4, -12), (24.6, 62), (44.16, 62), (44.31, -50, "back"), (44.91, -45),
          (45.31, 62, "back"), (DUR, 62))
ARM_R = G((0, 62), (2.0, 62), (2.25, -50, "back"), (3.2, -45), (3.4, 62, "back"), (3.6, 36, "back"), (5.6, 38),
          (5.95, -20, "back"), (6.15, -5), (6.35, -20), (6.55, -8), (6.9, -28, "back"), (11.6, -26),
          (11.9, -45, "back"), (13.9, -43), (14.2, 62), (C3 + 0.05, 62), (25.0, -28, "back"), (26.8, -26),
          (27.0, 22, "back"), (33.1, 24), (33.4, -35, "back"), (36.6, -33), (36.9, 62), (37.01, 62),
          (37.47, -48, "back"), (39.82, -45), (40.12, -55, "back"), (42.82, -52), (43.07, -40, "back"), (44.02, -42),
          (44.17, -50, "back"), (44.91, -45), (45.06, -58), (45.21, -32), (45.36, -58), (45.51, -32), (DUR, 62))
HEAD = G((0, 0), (0.8, -6), (1.4, 7), (2.2, 0), (3.6, 7), (5.3, 6), (6.0, -8), (6.4, 8), (6.8, -5), (9.0, -6),
         (11.0, 6), (11.9, 9), (13.9, 4), (14.6, -6), (17.5, 7), (18.8, -6), (22.3, -5), (23.3, 7), (25.0, -6),
         (27.0, 8), (29.9, 9), (31.1, -6), (33.4, 7), (36.5, 4), (37.47, -6), (44.2, 0), (DUR, 0))


# ---------------------------------------------------------------- картинки вставок
def header(d, text, col, dark=True):
    """Плашка-заголовок карточки, ширина по тексту."""
    f = mg.font(36)
    tw = d.textbbox((0, 0), text, font=f)[2]
    d.rounded_rectangle([36, 30, 36 + tw + 44, 92], 30, fill=col + (255,))
    d.text((58, 38), text, font=f, fill=(10, 30, 40) if dark else "white")


def draw_finger(d, x, y, w, h, wr, nerve=None):
    """Палец (вид на подушечку): wr — сколько морщин (0..1); nerve: None / 'ok' / 'cut'."""
    d.rounded_rectangle([x, y, x + w, y + h], w // 2, fill=SKIN_RGB + (255,), outline=(150, 110, 90, 255), width=4)
    d.rounded_rectangle([x + w * 0.22, y + 16, x + w * 0.78, y + h * 0.22], 18, fill=(245, 225, 215, 255))  # ноготь
    n = int(round(7 * wr))
    for i in range(n):
        yy = y + h * 0.32 + i * h * 0.085
        d.arc([x + 8, yy - 18, x + w - 8, yy + 18], 200, 340, fill=(150, 100, 80, 255), width=5)
    if nerve:
        d.line([(x + w // 2, y + h * 0.3), (x + w // 2, y + h + 40)], fill=(255, 220, 90, 255), width=8)
        if nerve == "cut":
            cx, cy = x + w // 2, y + h * 0.75
            d.line([(cx - 28, cy - 28), (cx + 28, cy + 28)], fill=RED_RGB + (255,), width=12)
            d.line([(cx - 28, cy + 28), (cx + 28, cy - 28)], fill=RED_RGB + (255,), width=12)


def finger_tex(wr):
    im = rounded((520, 640), 34, (16, 22, 28, 255), AQUA_RGB + (255,), 4)
    d = ImageDraw.Draw(im)
    header(d, "ПАЛЕЦ В ВОДЕ", AQUA_RGB)
    draw_finger(d, 170, 130, 180, 440, wr)
    for k in range(5):  # волны воды внизу
        y = 560 + k * 14
        d.arc([20 + k * 30, y - 20, 500 - k * 30, y + 20], 200, 340, fill=(90, 170, 220, 160), width=4)
    return mg.premul(im)


def nerve_tex():
    im = rounded((780, 600), 34, (16, 18, 26, 255), (72, 72, 92, 255), 3)
    d = ImageDraw.Draw(im)
    header(d, "НАБЛЮДЕНИЕ", AQUA_RGB)
    draw_finger(d, 120, 130, 170, 340, 1.0, "ok")
    draw_finger(d, 490, 130, 170, 340, 0.0, "cut")
    d.text((96, 520), "нерв цел", font=mg.font(40), fill=(255, 220, 120))
    d.text((430, 520), "нерв повреждён", font=mg.font(40), fill=(255, 130, 130))
    return mg.premul(im)


def section_tex(v, g):
    """Разрез подушечки: сосуды сжимаются (v), поверхность покрывается бороздками (g)."""
    Wc, Hc = 780, 560
    im = rounded((Wc, Hc), 34, (16, 18, 26, 255), (72, 72, 92, 255), 3)
    d = ImageDraw.Draw(im)
    header(d, "ПОД КОЖЕЙ", AQUA_RGB)
    xs = np.arange(40, Wc - 40, 6)
    top = 190 + g * 22 * np.sin((xs - 40) / 30.0) + g * 14
    poly = [(40, 470)] + list(zip(xs, top)) + [(Wc - 40, 470)]
    d.polygon(poly, fill=SKIN_RGB + (255,))
    d.line(list(zip(xs, top)), fill=(150, 100, 80, 255), width=5)
    d.rounded_rectangle([40, 300, Wc - 40, 470], 10, fill=(206, 150, 126, 255))
    for i in range(5):
        cx, cy = 110 + i * 140, 385
        r = int(48 * (1 - 0.55 * v))
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(200, 50, 60, 255), outline=(120, 20, 30, 255), width=4)
    if v > 0.1:
        d.text((40, 488), "сосуды сжимаются", font=mg.font(38), fill=(255, 150, 150))
    if g > 0.1:
        d.text((440, 488), "бороздки", font=mg.font(38), fill=(150, 220, 255))
    return mg.premul(im)


def sole_tex(ph, flow):
    """Подошва кроссовка с рисунком; капли воды уходят по бороздкам (flow — включено ли)."""
    im = rounded((560, 760), 34, (16, 18, 26, 255), (72, 72, 92, 255), 3)
    d = ImageDraw.Draw(im)
    header(d, "КАК ПОДОШВА", GRAY_RGB, dark=False)
    d.rounded_rectangle([110, 110, 450, 730], 160, fill=(50, 52, 62, 255), outline=(110, 110, 130, 255), width=6)
    grooves = []
    for i in range(11):
        y = 170 + i * 50
        a, b = (140, y), (420, y + 26) if i % 2 else (420, y - 26)
        d.line([a, b], fill=(24, 24, 30, 255), width=16)
        grooves.append((a, b))
    if flow:
        for j, (a, b) in enumerate(grooves):
            for k in range(2):
                q = (ph + k * 0.5 + j * 0.13) % 1
                x, y = a[0] + (b[0] - a[0]) * q, a[1] + (b[1] - a[1]) * q
                d.ellipse([x - 9, y - 9, x + 9, y + 9], fill=(90, 180, 255, 255))
    return mg.premul(im)


def race_tex(pa, pb, dry):
    """Гонка: сколько шариков переложили гладкими (pa) и сморщенными (pb) пальцами."""
    im = rounded((780, 470), 34, (16, 18, 26, 255), (72, 72, 92, 255), 3)
    d = ImageDraw.Draw(im)
    head = "СУХИЕ ШАРИКИ" if dry else "МОКРЫЕ ШАРИКИ"
    col = GRAY_RGB if dry else AQUA_RGB
    tw = d.textbbox((0, 0), head, font=mg.font(36))[2]
    d.rounded_rectangle([36, 30, 36 + tw + 44, 92], 30, fill=col + (255,))
    d.text((58, 38), head, font=mg.font(36), fill="white" if dry else (10, 30, 40))
    for i, (txt, v, c) in enumerate((("гладкие пальцы", pa, (150, 150, 170)), ("сморщенные", pb, GREEN_RGB))):
        y = 130 + i * 150
        d.text((40, y), txt, font=mg.font(40), fill=(210, 210, 225))
        n = int(round(10 * v))
        for k in range(10):
            x = 50 + k * 68
            d.ellipse([x, y + 60, x + 50, y + 110], fill=c + (255,) if k < n else (40, 42, 54, 255))
    return mg.premul(im)


def drop_tex():
    im = Image.new("RGBA", (160, 220), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([20, 80, 140, 200], fill=(90, 180, 240, 255))
    d.polygon([(28, 120), (132, 120), (80, 0)], fill=(90, 180, 240, 255))
    d.ellipse([50, 110, 80, 150], fill=(200, 235, 255, 255))
    return mg.premul(im)


def ball_tex():
    im = Image.new("RGBA", (120, 120), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([4, 4, 116, 116], fill=(40, 170, 210, 255))
    d.ellipse([28, 22, 60, 50], fill=(200, 240, 255, 255))
    return mg.premul(im)


def brain_tex():
    im = Image.new("RGBA", (260, 220), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for cx, cy, r in ((80, 90, 70), (150, 70, 72), (195, 120, 58), (110, 140, 62), (60, 140, 45)):
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(235, 150, 180, 255))
    d.line([(130, 20), (130, 190)], fill=(190, 100, 140), width=6)
    return mg.premul(im)


# ---------------------------------------------------------------- сцена
class Scene(base.Scene):
    cfg = sys.modules[__name__]

    def __init__(self, voice):
        super().__init__(voice)
        T = self.T
        T["drop"], T["ball"], T["brain"] = drop_tex(), ball_tex(), brain_tex()
        T["swell"] = card("ВЕРСИЯ", "КОЖА НАБУХЛА?", "просто впитала воду", GRAY_RGB)
        T["no"] = stamp("НЕТ!", RED_RGB)
        T["nerve"] = nerve_tex()
        T["ns"] = chip("НЕРВНАЯ СИСТЕМА", AQUA_RGB)
        T["grip"] = stamp("НЕ СКОЛЬЗИТ", GREEN_RGB)
        T["study"] = card("ОПЫТ", "УНИВЕРСИТЕТ НЬЮКАСЛА", "Великобритания, 2013", BLUE_RGB)
        T["gloves"] = card("ВЫВОД", "ПЕРЧАТКИ ОТ ПРИРОДЫ", "тело надевает их само", AQUA_RGB)
        T["faster"] = stamp("БЫСТРЕЕ!", GREEN_RGB)
        T["same"] = stamp("ОДИНАКОВО", GRAY_RGB)
        self.cache = {}
        orig = self.room.render
        self.room.render = lambda cam: self.grade(orig(cam))
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        self.xx, self.yy = xx, yy
        self.rr = np.sqrt((xx - W / 2) ** 2 + (yy - H * 0.45) ** 2) + 1e-3

    def cached(self, key, fn, *a):
        k = (key,) + a
        if k not in self.cache:
            self.cache[k] = fn(*a)
        return self.cache[k]

    def grade(self, frame):
        m = np.array(track(TINT, self.t), np.float32)
        if np.abs(m - 1).max() < 1e-3:
            return frame
        return np.clip(frame.astype(np.float32) * m, 0, 255).astype(np.uint8)

    def draw(self, t):
        self.t = t
        return super().draw(t)

    def cam(self, t):
        yaw, dist, hgt, ty, fov = track(CAMERA, t)
        off = track(OFFSET, t)
        a = math.radians(yaw)
        return mg.Cam(yaw + 0.5 * wobble(t, 1), dist, hgt + 0.02 * wobble(t, 2), ty + 0.015 * wobble(t, 3), fov=fov,
                      roll=0.5 * wobble(t * 0.8, 4), tx=off * math.cos(a), tz=off * math.sin(a))

    def finish(self, frame, t):
        fr = super().finish(frame, t)
        for r0 in RIPPLES:  # «вода»: круги ряби по всему кадру и лёгкое осветление
            if r0 < t < r0 + 0.5:
                s = math.sin(math.pi * (t - r0) / 0.5)
                d = 26 * s * np.sin(self.rr / 34.0 - (t - r0) * 26)
                mx = self.xx + d * (self.xx - W / 2) / self.rr
                my = self.yy + d * (self.yy - H * 0.45) / self.rr
                fr = cv2.remap(fr, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                fr = np.clip(fr.astype(np.float32) * (1 + 0.25 * s) + np.array(AQUA, np.float32) * 0.18 * s,
                             0, 255).astype(np.uint8)
        return fr

    def ripples(self, it, cam, cx, cz, t0, t, n=3, big=1.0):
        """Круги по воде на полу, расходятся от точки (cx, cz)."""
        def fn():
            lay = np.zeros_like(self.frame)
            for i in range(n):
                q = (t - t0 - i * 0.28) / 1.3
                if not 0 < q < 1:
                    continue
                r = (0.08 + 0.9 * q) * big
                a = np.linspace(0, 2 * math.pi, 48)
                P = np.stack([cx + r * np.cos(a), np.full_like(a, 0.015), cz + r * np.sin(a)], -1)
                uv, z = cam.proj(P)
                if (z > 0.3).all():
                    c = tuple(int(v * (1 - q)) for v in AQUA)
                    mg.lines(lay, uv, c, 5)
            cv2.add(self.frame, cv2.GaussianBlur(lay, (0, 0), 4), self.frame)
            cv2.add(self.frame, lay, self.frame)
        it.append((cam.depth((cx, 0.0, cz)) + 3.0, fn))

    def items_for(self, t, cam, it):
        T = self.T
        P = lambda y0, *a, **k: place(y0, *a, **k)  # noqa: E731

        # 1. Крючок (персонаж справа): капля падает, круги по полу; палец в воде морщится
        y0 = SEC["hook"]
        r_, c_ = basis(y0)
        dp = r_ * -0.75 + c_ * -0.6
        if t < 1.05:
            p = clamp(t / 1.05)
            self.add_bb(it, cam, dp + [0, 2.6 * (1 - p * p) + 0.05, 0], 0.13, T["drop"], anchor=(0.5, 0.5))
        if 1.0 < t < 3.4:
            self.ripples(it, cam, dp[0], dp[2], 1.0, t, 3, 1.2)
        k = pop(t, 0.9, 2.3, 0.35, 0.25)
        if k > 0:
            wr = round(smooth((t - 0.9) / 1.0) * 7) / 7
            pos = P(y0, -0.74, 1.6) + [0, 0.02 * math.sin(t * 3), 0]
            self.add_panel(it, cam, pos, face(y0, -12), 0.52 * k, 0.64 * k, 0.05 * k, self.cached("fing", finger_tex, wr),
                           pitch=-5, side=(30, 60, 70), edge=(190, 240, 255))

        # 2. Заставка всплывает из пола, по полу круги
        k = clamp((t - 2.2) / 0.45)
        if 2.2 < t < C1:
            e = out_back(k)
            kk = 1 - smooth((t - (C1 - 0.3)) / 0.3)
            kk *= clamp(k * 1.5)
            self.add_panel(it, cam, place(0, 0.0, 1.55 + 0.4 * e, 0.35), face(0, -6), 1.2 * kk, 0.564 * kk, 0.11 * kk,
                           T["stinger"], pitch=-6, side=(30, 24, 120), edge=(200, 190, 255))
            self.ripples(it, cam, 0.0, -0.5, 2.2, t, 3, 1.4)

        # 3. Не набухла — нерв: палец с повреждённым нервом не морщится; сигнал от мозга
        y0 = SEC["nerve"]
        r_, c_ = basis(y0)
        k = pop(t, C1 + 0.05, 6.9, 0.4, 0.3)
        if k > 0:
            pos = P(y0, 0.72, 1.55) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 12), 0.9 * k, 0.48 * k, 0.06 * k, T["swell"], pitch=-6,
                           side=(50, 50, 60), edge=(220, 220, 235))
            if t > 6.0:
                ks = clamp((t - 6.0) / 0.15)
                self.add_bb(it, cam, pos + c_ * 0.12, 0.2 * (1 + 0.6 * (1 - ks)) * k, T["no"], anchor=(0.5, 0.5))
        k = pop(t, 6.95, 11.8, 0.4, 0.3)
        if k > 0:
            pos = P(y0, 0.72, 1.55) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 12), 0.9 * k, 0.692 * k, 0.06 * k, T["nerve"], pitch=-5,
                           side=(30, 60, 70), edge=(190, 240, 255))
        k = pop(t, 11.85, C2, 0.4, 0.3)
        if k > 0:
            bpos = P(y0, 0.5, 2.05) + [0, 0.02 * math.sin(t * 3), 0]
            fpos = P(y0, 1.05, 1.05)
            self.add_bb(it, cam, bpos, 0.24 * k, T["brain"], anchor=(0.5, 0.5))
            self.add_panel(it, cam, fpos, face(y0, 12), 0.3 * k, 0.37 * k, 0.03, self.cached("fing", finger_tex, 1.0),
                           side=(30, 60, 70), radius=0.04)
            it.append((cam.depth(fpos) + 0.05, lambda b=bpos, f=fpos, k=k: self.signal(cam, b, f, t, k)))
            kc = pop(t, 13.1, C2, 0.3, 0.3)
            if kc > 0:
                self.add_bb(it, cam, P(y0, 0.95, 1.6), 0.11 * kc, T["ns"], anchor=(0.5, 0.5))

        # 4. Под кожей (персонаж справа): сосуды сжимаются, бороздки; подошва, вода уходит
        y0 = SEC["skin"]
        k = pop(t, C2 + 0.05, 18.85, 0.4, 0.3)
        if k > 0:
            v = round(smooth((t - 15.1) / 0.7) * 12) / 12
            g = round(smooth((t - 17.6) / 0.7) * 12) / 12
            pos = P(y0, -0.72, 1.55) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, -12), 0.9 * k, 0.646 * k, 0.06 * k, self.cached("sec", section_tex, v, g),
                           pitch=-5, side=(70, 50, 50), edge=(255, 210, 200))
        k = pop(t, 18.8, C3, 0.4, 0.3)
        if k > 0:
            flow = t > 22.2
            ph = round((t * 1.1) % 1 * 12) / 12 if flow else 0.0
            pos = P(y0, -0.72, 1.5) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, -12), 0.56 * k, 0.76 * k, 0.05 * k, self.cached("sole", sole_tex, ph, flow),
                           pitch=-5, side=(40, 40, 50), edge=(200, 200, 220))
            kg = pop(t, 23.6, C3, 0.15, 0.25)
            if kg > 0:
                ks = clamp((t - 23.6) / 0.15)
                self.add_bb(it, cam, P(y0, -0.8, 0.95, 0.1), 0.15 * (1 + 0.6 * (1 - ks)) * kg, T["grip"], anchor=(0.5, 0.5))

        # 5. Опыт (персонаж слева): карточка, гонка пальцев, 3D-шарики из миски в миску; «перчатки от природы»
        y0 = SEC["test"]
        r_, c_ = basis(y0)
        k = pop(t, C3 + 0.05, 27.0, 0.4, 0.3)
        if k > 0:
            pos = P(y0, 0.72, 1.6) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 14), 0.9 * k, 0.48 * k, 0.06 * k, T["study"], pitch=-6,
                           side=(40, 50, 90), edge=(200, 210, 255))
        k = pop(t, 26.95, 33.3, 0.4, 0.3)
        if k > 0:
            dry = t > 31.1
            if not dry:
                pa = round(smooth((t - 27.3) / 3.4) * 0.6 * 10) / 10
                pb = round(smooth((t - 27.3) / 3.0) * 10) / 10
            else:
                pa = pb = round(smooth((t - 31.3) / 1.4) * 10) / 10
            pos = P(y0, 0.72, 1.62) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 12), 0.9 * k, 0.542 * k, 0.06 * k, self.cached("race", race_tex, pa, pb, dry),
                           pitch=-5, side=(30, 60, 70), edge=(190, 240, 255))
            if 30.4 < t < 31.1:
                ks = clamp((t - 30.4) / 0.15)
                self.add_bb(it, cam, pos + c_ * 0.12 + [0, -0.42, 0], 0.13 * (1 + 0.6 * (1 - ks)), T["faster"], anchor=(0.5, 0.5))
            if 32.2 < t < 33.3:
                ks = clamp((t - 32.2) / 0.15)
                self.add_bb(it, cam, pos + c_ * 0.12 + [0, -0.42, 0], 0.13 * (1 + 0.6 * (1 - ks)) * k, T["same"], anchor=(0.5, 0.5))
            a = r_ * 0.7 + c_ * -0.7
            b = r_ * 1.2 + c_ * -0.7
            for q in (a, b):
                self.add_box(it, cam, (q[0], q[2]), (0.3, 0.3), 0.1 * k, (60, 70, 80), radius=0.14, glow=(200, 170, 60))
            for i in range(14):  # шарики перелетают дугой
                s0 = 27.2 + i * 0.4
                p = (t - s0) / 0.45
                if 0 < p < 1:
                    pos3 = a * (1 - p) + b * p + [0, 0.16 + 0.38 * math.sin(math.pi * p), 0]
                    self.add_bb(it, cam, pos3, 0.07 * k, T["ball"], anchor=(0.5, 0.5))
        k = pop(t, 33.4, 37.0, 0.4, 0.3)
        if k > 0:
            pos = P(y0, 0.72, 1.6) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 12), 0.9 * k, 0.48 * k, 0.06 * k, T["gloves"], pitch=-6,
                           side=(30, 60, 70), edge=(190, 240, 255))

        # 6. Концовка с Telegram — общий шаблон
        self.outro_items(t, cam, it, OUTRO, SEC["tg"])

    def signal(self, cam, b, f, t, k):
        """Светящаяся дорожка от мозга к пальцу, по ней бегут импульсы."""
        mid = (b + f) / 2 + [0.15, 0.25, 0]
        ss = np.linspace(0, 1, 30)
        pts = np.array([(1 - s) ** 2 * b + 2 * (1 - s) * s * mid + s * s * f for s in ss])
        uv, z = cam.proj(pts)
        if (z < 0.3).any():
            return
        prog = smooth((t - 11.95) / 0.5)
        n = max(2, int(len(uv) * prog))
        lay = np.zeros_like(self.frame)
        mg.lines(lay, uv[:n], tuple(int(c * k) for c in (90, 220, 255)), 6, closed=False)
        for j in range(3):
            q = ((t - 12.0) * 1.2 + j / 3) % 1
            if q < prog:
                p = uv[int(q * (len(uv) - 1))]
                cv2.circle(lay, (int(p[0]), int(p[1])), 14, (200, 250, 255), -1, cv2.LINE_AA)
        cv2.add(self.frame, cv2.GaussianBlur(lay, (0, 0), 7), self.frame)
        cv2.add(self.frame, lay, self.frame)


SFX = [(t, "whoosh") for t in (0.0, 2.2) + RIPPLES + (37.0,)]


def render(voice, out, frames):
    sc = Scene(voice)
    wr = mg.Writer(out, FPS)
    cuts = (C1, C2, C3)
    for n, i in enumerate(frames):
        t = i / FPS
        speed = abs(sc.cam(t + 0.5 / FPS).yaw - sc.cam(t - 0.5 / FPS).yaw) * FPS
        cut = any(abs(t - c) < 1.0 / FPS for c in cuts)
        sub = 5 if speed > math.radians(60) and not cut else 1
        if sub == 1:
            fr = sc.draw(t)
        else:
            acc = np.zeros((H, W, 3), np.float32)
            for k in range(sub):
                acc += sc.draw(t + (k / sub - 0.5) / FPS)
            fr = (acc / sub).astype(np.uint8)
        wr.write(sc.finish(fr, t))
        if n % 100 == 0:
            print(f"{out}: {n}/{len(frames)}", flush=True)
    wr.close()


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    voice, out = args[0], args[1]
    n = int(DUR * FPS)
    if "--chunk" in sys.argv:  # служебный режим: один кусок кадров (для параллельного рендера)
        cv2.setNumThreads(1)
        render(voice, out, range(int(args[2]), int(args[3])))
        return
    if "--preview" in sys.argv:
        sc = Scene(voice)
        for t in (0.15, 0.7, 1.4, 2.0, 2.6, 3.4, 4.5, 6.2, 8.5, 10.8, 12.6, 13.6, 14.3, 15.5, 17.2, 18.4, 20.2, 22.9,
                  24.1, 25.6, 28.0, 30.6, 32.5, 34.5, 39.0, 44.0):
            fr = sc.finish(sc.draw(t), t)
            cv2.imwrite(f"{out}_{int(t * 100):05d}.jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 4
    cuts = [round(n * j / jobs) for j in range(jobs + 1)]
    parts = [f"{out}.part{j}.mp4" for j in range(jobs)]
    procs = [subprocess.Popen([sys.executable, os.path.abspath(__file__), voice, parts[j], str(cuts[j]), str(cuts[j + 1]),
                               "--chunk"]) for j in range(jobs)]
    if any(p.wait() for p in procs):
        sys.exit("рендер куска упал")
    lst = out + ".list.txt"
    open(lst, "w").write("".join(f"file '{os.path.abspath(p)}'\n" for p in parts))
    tmp = out + ".video.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", tmp],
                   check=True)
    for p in parts + [lst]:
        os.remove(p)
    base.mix_audio(tmp, voice, out, music=os.path.join(ROOT, "work", "fingers", "music.wav"), sfx_list=SFX, dur=DUR,
                   sfx_vol=0.15)  # «вжух» лёгкий и тихий
    os.remove(tmp)


if __name__ == "__main__":
    main()
