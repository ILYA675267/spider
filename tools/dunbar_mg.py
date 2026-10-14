"""Ролик №9: «Почему у тебя не может быть тысячи настоящих друзей?» (число Данбара).

python tools/dunbar_mg.py work/dunbar/voice_clean.wav output/dunbar.mp4 [--preview] [--jobs 4]
Общий движок, персонаж и концовка с Telegram — из memory_mg.py; параллельный рендер — как в money_mg.py.
Новое в монтаже: «диафрагма» (круг сжимается в точку и раскрывает новую сцену); подъём камеры по спирали;
светящиеся круги друзей на полу вокруг паучка (150 / 15 / 5); открытки разлетаются по кругу;
ночь в комнате на «позвонишь ночью»; точечный график «мозг — размер стаи»; круговая диаграмма времени.
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
OUTRO = float(open(os.path.join(ROOT, "work", "dunbar", "outro_start.txt")).read())  # 35.37
DUR = 44.0
GOLD_RGB, GREEN_RGB, PINK_RGB, MOON_RGB = (240, 186, 60), (46, 196, 110), (240, 110, 160), (180, 200, 255)
GOLD, PINK = GOLD_RGB[::-1], PINK_RGB[::-1]

# ---------------------------------------------------------------- субтитры
T_, Go_, Pi_, G_ = TG_RGB, GOLD_RGB, PINK_RGB, GREEN_RGB
SUBS = [
    (0.213, "ПОЧЕМУ У ТЕБЯ НЕ МОЖЕТ БЫТЬ", ()), (1.413, "ТЫСЯЧИ НАСТОЯЩИХ ДРУЗЕЙ?", ("НАСТОЯЩИХ",), Pi_),
    (3.05, "", ()),
    (4.653, "УЧЁНЫЙ ИЗ ОКСФОРДА", ("ОКСФОРДА",), BLUE_RGB), (5.373, "РОБИН ДАНБАР", ("ДАНБАР",), BLUE_RGB),
    (6.233, "СРАВНИЛ МОЗГ РАЗНЫХ ОБЕЗЬЯН", ("МОЗГ",)), (7.873, "И РАЗМЕР ИХ СТАЙ", ("СТАЙ",)),
    (9.293, "И ПОСЧИТАЛ, ЧТО ЧЕЛОВЕК", ()), (10.573, "МОЖЕТ ПОДДЕРЖИВАТЬ ОТНОШЕНИЯ", ()),
    (12.033, "ПРИМЕРНО СО 150 ЛЮДЬМИ", ("150",), Go_), (14.407, "ЭТО ЧИСЛО НАЗВАЛИ", ()),
    (15.387, "ЧИСЛОМ ДАНБАРА", ("ДАНБАРА",), Go_), (16.607, "ЕГО ДАЖЕ ПРОВЕРИЛИ", ()),
    (17.487, "ПО РОЖДЕСТВЕНСКИМ", ()), (18.227, "ОТКРЫТКАМ:", ("ОТКРЫТКАМ:",), Pi_), (19.027, "ЛЮДИ В СРЕДНЕМ", ()), (19.727, "ПОЗДРАВЛЯЛИ", ()),
    (20.307, "ОКОЛО 150 ЧЕЛОВЕК", ("150",), Go_), (22.147, "А САМЫХ БЛИЗКИХ,", ("БЛИЗКИХ,",), Pi_),
    (23.247, "КОМУ ПОЗВОНИШЬ НОЧЬЮ, —", ("НОЧЬЮ,",)), (24.667, "ВСЕГО ОКОЛО ПЯТИ", ("ПЯТИ",), Pi_),
    (26.114, "ПОЧЕМУ ТАК МАЛО?", ("МАЛО?",)), (27.454, "ДРУЖБА ТРЕБУЕТ ВРЕМЕНИ,", ("ВРЕМЕНИ,",), Go_),
    (28.834, "А ВРЕМЯ У НАС НЕ РЕЗИНОВОЕ", ("НЕ", "РЕЗИНОВОЕ")), (30.314, "ТАК ЧТО НЕ ПЕРЕЖИВАЙ,", ()),
    (31.194, "ЕСЛИ ДРУЗЕЙ НЕМНОГО", ("НЕМНОГО",), G_), (32.414, "ЛУЧШЕ ЧАЩЕ ПИШИ", ()), (33.334, "ТЕМ ПЯТЕРЫМ,", ("ПЯТЕРЫМ,",), Pi_),
    (33.954, "КТО ПРАВДА РЯДОМ", ("РЯДОМ",), Pi_),
    (35.335, "А ЕЩЁ БОЛЬШЕ ИНТЕРЕСНОГО —", (), T_), (36.695, "В МОЁМ ТЕЛЕГРАМ-КАНАЛЕ", ("ТЕЛЕГРАМ-КАНАЛЕ",), T_),
    (38.025, "«ПАУЧЬЯ НОРКА»", ("«ПАУЧЬЯ", "НОРКА»"), T_), (39.145, "ТАМ ЭКСКЛЮЗИВ,", ("ЭКСКЛЮЗИВ",), T_),
    (39.845, "КОТОРОГО НЕТ В РОЛИКАХ", (), T_), (41.145, "ССЫЛКА В ПРОФИЛЕ —", ("ПРОФИЛЕ",), T_),
    (42.165, "ЗАЛЕТАЙ!", ("ЗАЛЕТАЙ!",), T_),
]

# ---------------------------------------------------------------- камера и сцены
IRIS = (2.85, 25.75)                 # «диафрагма»: круг сжимается в точку, под ней склейка, раскрывается
C1, C2 = (i + 0.25 for i in IRIS)
CAMERA = [
    (0.0, (-10.0, 3.9, 1.3, 0.9, 50.0)), (0.8, (-8.0, 3.6, 1.15, 0.84, 50.0), "out"),
    (C1 - 0.01, (-6.0, 3.6, 1.15, 0.84, 50.0)),
    (C1 + 0.01, (0.0, 3.9, 1.3, 0.98, 50.0), "lin"),       # заставка
    (4.3, (1.0, 3.9, 1.3, 0.98, 50.0)),
    (4.7, (95.0, 3.6, 1.2, 0.84, 50.0), "expo"),           # Данбар и обезьяны
    (14.0, (100.0, 3.5, 1.2, 0.86, 50.0)),
    (15.1, (200.0, 4.8, 2.9, 0.45, 52.0), "inout"),        # подъём по спирали: круги друзей сверху
    (C2 - 0.01, (238.0, 4.6, 2.6, 0.5, 52.0), "lin"),
    (C2 + 0.01, (300.0, 3.6, 1.2, 0.84, 50.0), "lin"),     # почему так мало
    (34.95, (312.0, 3.7, 1.25, 0.86, 50.0), "lin"),
    (35.35, (380.0, 4.3, 1.3, 1.12, 50.0), "expo"), (DUR, (381.0, 4.0, 1.3, 1.15, 50.0)),
]
OFFSET = [(0.0, -0.5), (C1 - 0.01, -0.5), (C1 + 0.01, 0.0, "lin"), (4.3, 0.0), (4.7, 0.55, "expo"), (14.0, 0.55),
          (15.1, 0.0, "inout"), (C2 - 0.01, 0.0), (C2 + 0.01, 0.55, "lin"), (34.95, 0.55), (35.35, 0.0, "expo"),
          (DUR, 0.0)]
TINT = [(0.0, (1.02, 0.94, 1.1)), (4.3, (1.02, 0.94, 1.1)), (4.7, (1.0, 1.0, 1.0)), (14.2, (1.0, 1.0, 1.0)),
        (15.0, (1.0, 0.96, 1.08)), (22.0, (1.0, 0.96, 1.08)), (22.6, (1.1, 0.58, 0.42), "inout"),
        (C2 - 0.01, (1.1, 0.58, 0.42)), (C2 + 0.01, (0.94, 1.02, 1.1), "lin"), (35.0, (0.94, 1.02, 1.1)),
        (35.4, (1.0, 1.0, 1.0)), (DUR, (1.0, 1.0, 1.0))]
SEC = {"hook": -8, "dunbar": 98, "rings": 220, "fix": 305, "tg": 380.5}
SHAKES = ()
FLASHES = ((4.65, 0.08), (35.4, 0.12))
JUMPS = []
ALARM = None
SPOT = None

G = lambda *k: list(k)  # noqa: E731
ARM_L = G((0, 62), (0.5, 62), (0.75, -25, "back"), (1.7, -22), (1.85, 36, "back"), (2.8, 38), (C1 + 0.05, 62),
          (3.3, -50, "back"), (4.2, -45), (4.5, 62, "back"), (14.3, 62), (14.6, 30, "back"), (16.2, 32),
          (16.6, -20, "back"), (21.5, -18), (21.9, 40, "back"), (25.5, 42), (C2 + 0.05, 62), (26.2, 36, "back"),
          (27.3, 38), (27.5, 62), (30.4, 62), (30.6, 30, "back"), (32.2, 32), (32.4, 62), (42.275, 62),
          (42.425, -50, "back"), (43.025, -45), (43.425, 62, "back"), (DUR, 62))
ARM_R = G((0, 62), (1.85, 62), (2.0, 36, "back"), (2.8, 38), (C1 + 0.05, 62), (3.3, -50, "back"), (4.2, -45),
          (4.5, 62, "back"), (4.8, -30, "back"), (7.0, -28), (7.2, -15, "back"), (9.2, -12), (9.4, -35, "back"),
          (13.9, -33), (14.3, 62), (14.6, 30, "back"), (16.2, 32), (16.6, -20, "back"), (21.5, -18),
          (21.9, 40, "back"), (25.5, 42), (C2 + 0.05, 62), (26.2, 36, "back"), (27.3, 38), (27.5, -30, "back"),
          (30.1, -28), (30.6, 30, "back"), (32.2, 32), (32.4, -35, "back"), (34.8, -33), (35.125, 62),
          (35.585, -48, "back"), (37.935, -45), (38.235, -55, "back"), (40.935, -52), (41.185, -40, "back"),
          (42.135, -42), (42.285, -50, "back"), (43.025, -45), (43.175, -58), (43.325, -32), (43.475, -58),
          (43.625, -32), (DUR, 62))
HEAD = G((0, 0), (0.6, -6), (1.8, 8), (2.8, 6), (3.3, 0), (4.8, -6), (7.2, -5), (9.4, -8), (12.6, 7), (14.3, 0),
         (16.6, 6), (19.0, 5), (22.1, -4), (24.6, 7), (26.1, 9), (27.5, -6), (29.3, -7), (30.6, 6), (32.4, -6),
         (34.0, 5), (35.585, -6), (42.3, 0), (DUR, 0))


# ---------------------------------------------------------------- картинки вставок
def header(d, text, col, dark=False):
    f = mg.font(36)
    tw = d.textbbox((0, 0), text, font=f)[2]
    d.rounded_rectangle([36, 30, 36 + tw + 44, 92], 30, fill=col + (255,))
    d.text((58, 38), text, font=f, fill=(30, 24, 10) if dark else "white")


def friends_tex(n):
    """Профиль в соцсети: счётчик «друзей» бежит вверх."""
    im = rounded((560, 400), 40, (18, 18, 26, 255), (80, 80, 100, 255), 4)
    d = ImageDraw.Draw(im)
    d.ellipse([40, 40, 150, 150], fill=(60, 64, 90, 255))
    d.ellipse([75, 60, 115, 100], fill=(140, 140, 170, 255))
    d.pieslice([55, 100, 135, 180], 180, 360, fill=(140, 140, 170, 255))
    d.text((180, 54), "мой профиль", font=mg.font(40), fill="white")
    d.text((180, 104), "друзья", font=mg.font(34), fill=(150, 150, 170))
    s = f"{n:,}".replace(",", " ")
    d.text((40, 170), s, font=mg.font(130, tall=True), fill=PINK_RGB + (255,))
    return mg.premul(im)


def scatter_tex(p, human):
    """Точечный график: размер мозга (кора) → размер стаи. human — точка «человек ≈ 150»."""
    Wc, Hc = 780, 600
    im = rounded((Wc, Hc), 34, (18, 18, 26, 255), (72, 72, 92, 255), 3)
    d = ImageDraw.Draw(im)
    header(d, "МОЗГ И РАЗМЕР СТАИ", BLUE_RGB)
    x0, y0, x1, y1 = 90, 520, 730, 130
    d.line([(x0, y0), (x1, y0)], fill=(90, 90, 110), width=4)
    d.line([(x0, y0), (x0, y1)], fill=(90, 90, 110), width=4)
    d.text((x1 - 170, y0 + 14), "мозг больше", font=mg.font(30), fill=(150, 150, 170))
    d.text((x0 + 10, y1 - 30), "стая больше ↑", font=mg.font(30), fill=(150, 150, 170))
    pts = [(0.08, 0.06), (0.17, 0.1), (0.26, 0.12), (0.33, 0.18), (0.42, 0.2), (0.5, 0.26), (0.58, 0.3), (0.66, 0.36)]
    n = int(round(len(pts) * p))
    for i, (u, v) in enumerate(pts[:n]):
        x, y = x0 + (x1 - x0) * u, y0 - (y0 - y1) * v
        d.ellipse([x - 14, y - 14, x + 14, y + 14], fill=(120, 150, 220, 255))
    if n >= 3:
        d.line([(x0 + 20, y0 - 10), (x0 + (x1 - x0) * 0.92, y0 - (y0 - y1) * 0.82)], fill=(80, 100, 150), width=4)
    if human:
        x, y = x0 + (x1 - x0) * 0.9, y0 - (y0 - y1) * 0.8
        d.ellipse([x - 24, y - 24, x + 24, y + 24], fill=GOLD_RGB + (255,))
        d.rounded_rectangle([x - 290, y - 30, x - 40, y + 30], 26, fill=GOLD_RGB + (255,))
        d.text((x - 270, y - 22), "человек ~150", font=mg.font(32), fill=(30, 24, 10))
    return mg.premul(im)


def envelope_tex():
    im = rounded((240, 160), 16, (240, 230, 220, 255), (200, 120, 140, 255), 4)
    d = ImageDraw.Draw(im)
    d.polygon([(4, 6), (236, 6), (120, 92)], fill=(225, 205, 200, 255), outline=(200, 120, 140, 255))
    d.ellipse([104, 76, 136, 108], fill=PINK_RGB + (255,))
    return mg.premul(im)


def counter_tex(n):
    im = rounded((520, 240), 34, (26, 16, 22, 255), PINK_RGB + (255,), 5)
    d = ImageDraw.Draw(im)
    d.text((36, 26), "открыток отправлено", font=mg.font(38), fill=(255, 190, 220))
    d.text((36, 90), str(n), font=mg.font(120, tall=True), fill="white")
    return mg.premul(im)


def moon_tex():
    im = Image.new("RGBA", (240, 240), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([10, 10, 230, 230], fill=MOON_RGB + (255,))
    d.ellipse([70, -10, 270, 190], fill=(0, 0, 0, 0))
    return mg.premul(im)


def pie_tex(p):
    """Куда уходит время на общение: около 40% — пятерым самым близким, ~20% — следующим десяти."""
    im = rounded((760, 520), 34, (18, 18, 26, 255), (72, 72, 92, 255), 3)
    d = ImageDraw.Draw(im)
    header(d, "ВРЕМЯ НА ОБЩЕНИЕ", GOLD_RGB, dark=True)
    cx, cy, r = 200, 300, 160
    parts = [(0.4, PINK_RGB, "5 близких — ~40%"), (0.2, GOLD_RGB, "ещё 10 — ~20%"), (0.4, (90, 90, 110), "все остальные")]
    a0 = -90
    for i, (v, col, lab) in enumerate(parts):
        a1 = a0 + 360 * v * p
        if a1 > a0 + 0.5:
            d.pieslice([cx - r, cy - r, cx + r, cy + r], a0, a1, fill=col + (255,))
        a0 = a0 + 360 * v
        y = 190 + i * 80
        d.rounded_rectangle([386, y, 412, y + 26], 8, fill=col + (255,))
        d.text((424, y - 6), lab, font=mg.font(30), fill=(220, 220, 232))
    return mg.premul(im)


def five_tex(lit, sent):
    """Пять самых близких: аватарки загораются, уходит «Привет! Как ты?»."""
    im = rounded((760, 420), 34, (26, 16, 22, 255), PINK_RGB + (255,), 5)
    d = ImageDraw.Draw(im)
    header(d, "ТВОИ ПЯТЬ", PINK_RGB)
    for i in range(5):
        x = 60 + i * 136
        on = i < lit
        d.ellipse([x, 130, x + 110, 240], fill=PINK_RGB + (255,) if on else (60, 50, 60, 255))
        d.ellipse([x + 35, 150, x + 75, 190], fill=(255, 230, 240, 255) if on else (100, 90, 100, 255))
        d.pieslice([x + 20, 185, x + 90, 255], 180, 360, fill=(255, 230, 240, 255) if on else (100, 90, 100, 255))
    if sent:
        d.rounded_rectangle([200, 290, 720, 380], 40, fill=TG_RGB + (255,))
        d.text((230, 310), "Привет! Как ты?", font=mg.font(44), fill="white")
    return mg.premul(im)


# ---------------------------------------------------------------- сцена
class Scene(base.Scene):
    cfg = sys.modules[__name__]

    def __init__(self, voice):
        super().__init__(voice)
        T = self.T
        T["real"] = stamp("НАСТОЯЩИХ?", PINK_RGB)
        T["oxford"] = card("ОКСФОРД", "РОБИН ДАНБАР", "антрополог, 1990-е", BLUE_RGB)
        T["dn"] = chip("ЧИСЛО ДАНБАРА ~150", GOLD_RGB)
        T["env"], T["moon"] = envelope_tex(), moon_tex()
        T["c5"] = chip("5 САМЫХ БЛИЗКИХ", PINK_RGB)
        T["ok"] = chip("ЭТО НОРМАЛЬНО", GREEN_RGB)
        T["why"] = chip("ПОЧЕМУ ТАК МАЛО?", GOLD_RGB)
        self.cache = {}
        orig = self.room.render
        self.room.render = lambda cam: self.grade(orig(cam))
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        self.rr = np.sqrt((xx - W / 2) ** 2 + (yy - H * 0.42) ** 2)

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
        for i0 in IRIS:  # «диафрагма»: чёрный круг сжимается до точки и раскрывается уже в новой сцене
            if i0 < t < i0 + 0.5:
                p = (t - i0) / 0.5
                q = 1 - smooth(p * 2) if p < 0.5 else smooth((p - 0.5) * 2)
                R = 1300 * q
                m = np.clip((R - self.rr) / 6.0, 0, 1)[..., None]
                ring = np.clip(1 - np.abs(self.rr - R) / 8.0, 0, 1)[..., None] * (1 if 0.02 < q < 0.98 else 0)
                fr = (fr.astype(np.float32) * m + ring * np.array(PINK, np.float32)).clip(0, 255).astype(np.uint8)
        return fr

    def over_char(self, cam, pos):
        """Попадает ли точка на экране в область паучка (с запасом) — тогда вставку перед ним не рисуем."""
        uv, z = cam.proj(np.array([pos, [0, 0.0, 0], [0, 1.3, 0]]))
        hw = 0.45 * cam.focal / max(0.3, z[1]) + 60
        return abs(uv[0][0] - uv[1][0]) < hw and uv[2][1] - 60 < uv[0][1] < uv[1][1] + 30

    def rings(self, cam, t):
        """Круги друзей на полу: 150 точек, 15 и 5 самых близких (тёплые, светятся ночью)."""
        lay = np.zeros_like(self.frame)
        spin = t * 0.15
        specs = ((1.75, 150, 14.6, (150, 140, 160), 5), (1.05, 15, 15.5, GOLD, 9), (0.55, 5, 22.5, PINK, 14))
        for R, n, t0, col, size in specs:
            k = smooth((t - t0) / 0.9)
            if k <= 0:
                continue
            a = np.linspace(0, 2 * math.pi, 96)
            P = np.stack([R * np.cos(a), np.full_like(a, 0.01), R * np.sin(a)], -1)
            uv, z = cam.proj(P)
            if (z > 0.3).all():
                mg.lines(lay, uv, tuple(int(c * 0.45 * k) for c in col), 3)
            m = int(n * k)
            ang = spin * (1 if R > 1 else -1.5) + np.arange(m) * 2 * math.pi / n
            Q = np.stack([R * np.cos(ang), np.full_like(ang, 0.02), R * np.sin(ang)], -1)
            if m:
                uv, z = cam.proj(Q)
                for (x, y), zz in zip(uv, z):
                    if zz > 0.3:
                        s = int(size * 4.0 / zz)
                        cv2.circle(lay, (int(x), int(y)), max(2, s), col, -1, cv2.LINE_AA)
        cv2.add(self.frame, cv2.GaussianBlur(lay, (0, 0), 6), self.frame)
        cv2.add(self.frame, lay, self.frame)

    def items_for(self, t, cam, it):
        T = self.T
        P = lambda y0, *a, **k: place(y0, *a, **k)  # noqa: E731

        # 1. Крючок (персонаж справа): счётчик «друзей» до тысячи, «настоящих?»
        y0 = SEC["hook"]
        k = pop(t, 0.35, C1, 0.35, 0.2)
        if k > 0:
            n = int(round(1000 * smooth((t - 0.5) / 1.3) / 10)) * 10
            pos = P(y0, -0.74, 1.62) + [0, 0.02 * math.sin(t * 3), 0]
            self.add_panel(it, cam, pos, face(y0, -12), 0.62 * k, 0.443 * k, 0.05 * k, self.cached("fr", friends_tex, n),
                           pitch=-5, side=(60, 40, 60), edge=(255, 200, 230))
        k = pop(t, 1.85, C1, 0.15, 0.2)
        if k > 0:
            ks = clamp((t - 1.85) / 0.15)
            self.add_bb(it, cam, P(y0, -0.62, 1.15, 0.1), 0.15 * (1 + 0.6 * (1 - ks)) * k, T["real"], anchor=(0.5, 0.5))

        # 2. Заставка
        k = pop(t, C1 + 0.05, 4.45, 0.35, 0.3)
        if k > 0:
            self.add_panel(it, cam, place(0, 0.0, 1.95, 0.35), face(0, -6), 1.2 * k, 0.564 * k, 0.11 * k,
                           T["stinger"], pitch=-6, side=(30, 24, 120), edge=(200, 190, 255))

        # 3. Данбар (персонаж слева): карточка → график «мозг — стая», точка «человек ≈ 150»
        y0 = SEC["dunbar"]
        k = pop(t, 4.75, 7.1, 0.4, 0.3)
        if k > 0:
            pos = P(y0, 0.72, 1.55) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 14), 0.9 * k, 0.48 * k, 0.06 * k, T["oxford"], pitch=-6,
                           side=(40, 50, 90), edge=(200, 210, 255))
        k = pop(t, 7.05, 14.4, 0.4, 0.3)
        if k > 0:
            p = round(smooth((t - 7.3) / 2.4) * 8) / 8
            pos = P(y0, 0.72, 1.62) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 12), 0.9 * k, 0.692 * k, 0.06 * k,
                           self.cached("sc", scatter_tex, p, t > 12.6), pitch=-5, side=(40, 50, 90), edge=(200, 210, 255))

        # 4. Круги друзей (вид сверху): 150 → открытки → ночь, 5 близких
        if 14.3 < t < C2 + 0.05:
            it.append((1e6, lambda: self.rings(cam, t)))
        y0 = SEC["rings"]
        r_, c_ = basis(y0)
        cam_yaw = math.degrees(cam.yaw)
        k = pop(t, 15.3, 16.5, 0.35, 0.3)
        if k > 0:
            self.add_bb(it, cam, place(cam_yaw, 0.0, 2.25), 0.14 * k, T["dn"], anchor=(0.5, 0.5))
        if 17.4 < t < 21.9:
            for i in range(10):  # открытки разлетаются от паучка к большому кругу
                s0 = 17.5 + i * 0.35
                p = (t - s0) / 0.9
                if 0 < p < 1:
                    a = i * 2.3
                    end = np.array([1.75 * math.cos(a), 0.05, 1.75 * math.sin(a)])
                    pos = np.array([0, 1.4, 0]) * (1 - p) + end * p + [0, 0.5 * math.sin(math.pi * p), 0]
                    if cam.depth(pos) > cam.depth((0, 0.6, 0)) + 0.1 or not self.over_char(cam, pos):
                        self.add_bb(it, cam, pos, 0.1 * (1 - 0.3 * p), T["env"], anchor=(0.5, 0.5))
        k = pop(t, 18.2, 21.95, 0.4, 0.3)
        if k > 0:
            n = int(round(150 * smooth((t - 18.4) / 2.4)))
            self.add_bb(it, cam, place(cam_yaw, 0.0, 2.4), 0.32 * k, self.cached("cnt", counter_tex, n), anchor=(0.5, 0.5))
        k = pop(t, 22.4, C2, 0.5, 0.2)
        if k > 0:
            self.add_bb(it, cam, place(cam_yaw, 0.85, 2.45, -1.5), 0.3 * k, T["moon"], anchor=(0.5, 0.5))
            kc = pop(t, 24.7, C2, 0.3, 0.2)
            if kc > 0:
                self.add_bb(it, cam, place(cam_yaw, 0.0, 2.15), 0.13 * kc, T["c5"], anchor=(0.5, 0.5))

        # 5. Почему так мало (персонаж слева): время на общение → «это нормально» → пятерым — «как ты?»
        y0 = SEC["fix"]
        k = pop(t, C2 + 0.05, 27.4, 0.3, 0.25)
        if k > 0:
            self.add_bb(it, cam, P(y0, 0.78, 1.95), 0.13 * k, T["why"], anchor=(0.5, 0.5))
        k = pop(t, 27.4, 30.35, 0.4, 0.3)
        if k > 0:
            p = round(smooth((t - 27.6) / 1.2) * 16) / 16
            pos = P(y0, 0.72, 1.6) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 12), 0.9 * k, 0.616 * k, 0.06 * k, self.cached("pie", pie_tex, p),
                           pitch=-5, side=(70, 60, 30), edge=(255, 230, 170))
        k = pop(t, 30.5, 32.4, 0.3, 0.25)
        if k > 0:
            self.add_bb(it, cam, P(y0, 0.78, 1.9), 0.13 * k, T["ok"], anchor=(0.5, 0.5))
        k = pop(t, 32.35, 35.2, 0.4, 0.3)
        if k > 0:
            lit = min(5, int((t - 32.5) / 0.25) + 1) if t > 32.5 else 0
            pos = P(y0, 0.72, 1.6) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 12), 0.9 * k, 0.497 * k, 0.06 * k,
                           self.cached("five", five_tex, lit, t > 33.9), pitch=-5, side=(60, 40, 60), edge=(255, 200, 230))

        # 6. Концовка с Telegram — общий шаблон
        self.outro_items(t, cam, it, OUTRO, SEC["tg"])


SFX = [(t, "whoosh") for t in (0.0, IRIS[0], 4.3, 14.2, IRIS[1], 34.95)]


def render(voice, out, frames):
    sc = Scene(voice)
    wr = mg.Writer(out, FPS)
    for n, i in enumerate(frames):
        t = i / FPS
        speed = abs(sc.cam(t + 0.5 / FPS).yaw - sc.cam(t - 0.5 / FPS).yaw) * FPS
        cut = any(abs(t - c) < 1.0 / FPS for c in (C1, C2))
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
        for t in (0.6, 1.5, 2.4, 2.95, 3.6, 5.5, 8.5, 12.9, 14.6, 15.8, 17.9, 19.5, 21.2, 23.0, 24.9, 25.9, 26.5,
                  28.6, 31.0, 33.0, 34.4, 37.0, 42.5):
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
    base.mix_audio(tmp, voice, out, music=os.path.join(ROOT, "work", "dunbar", "music.wav"), sfx_list=SFX, dur=DUR,
                   sfx_vol=0.15)  # «вжух» лёгкий и тихий
    os.remove(tmp)


if __name__ == "__main__":
    main()
