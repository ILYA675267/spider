"""Ролик №8: «Почему чужая зарплата портит настроение?».

python tools/salary_mg.py work/salary/voice_clean.wav output/salary.mp4 [--preview] [--jobs 4]
Общий движок, персонаж и концовка с Telegram — из memory_mg.py; параллельный рендер — как в money_mg.py.
Новое в монтаже: купюры падают сверху; вопрос учёных — как телевикторина (прожектор, варианты А/Б, голоса);
переход «влёт в карточку» (кадр резко приближается к вставке, под этим склейка); 3D-весы «ты / коллега»;
окно браузера с таблицей зарплат, которая прокручивается; полупрозрачный «призрак» паучка — «ты год назад».
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
from money_mg import rub_width, text_rub  # noqa: E402
from mg import H, W, clamp, out_back, smooth, track, wobble  # noqa: E402

FPS = 30
ROOT = mg.ROOT
OUTRO = float(open(os.path.join(ROOT, "work", "salary", "outro_start.txt")).read())  # 38.47
DUR = 47.15
GOLD_RGB, GREEN_RGB, PURPLE_RGB = (240, 186, 60), (46, 196, 110), (150, 90, 230)
GOLD, GREEN = GOLD_RGB[::-1], GREEN_RGB[::-1]

# ---------------------------------------------------------------- субтитры
T_, Go_, G_ = TG_RGB, GOLD_RGB, GREEN_RGB
SUBS = [
    (0.228, "ПОЧЕМУ ЧУЖАЯ ЗАРПЛАТА", ("ЧУЖАЯ",), Go_), (1.388, "ПОРТИТ НАСТРОЕНИЕ?", ("НАСТРОЕНИЕ?",)),
    (2.55, "", ()),
    (4.268, "УЧЁНЫЕ ИЗ ГАРВАРДА", ("ГАРВАРДА",), BLUE_RGB), (5.328, "СПРОСИЛИ ЛЮДЕЙ:", ()),
    (6.448, "ЧТО ЛУЧШЕ —", ()), (6.928, "ПОЛУЧАТЬ 50 ТЫСЯЧ,", ("50",), Go_), (8.248, "КОГДА У ВСЕХ 25,", ("25",)),
    (9.328, "ИЛИ 100 ТЫСЯЧ,", ("100",), Go_), (10.388, "КОГДА У ВСЕХ 200?", ("200?",)),
    (12.048, "ОКОЛО ПОЛОВИНЫ ВЫБРАЛИ", ("ПОЛОВИНЫ",), Go_), (13.228, "МЕНЬШЕ ДЕНЕГ,", ("МЕНЬШЕ",)),
    (13.928, "НО БОЛЬШЕ, ЧЕМ У ДРУГИХ", ("БОЛЬШЕ,",), Go_),
    (15.471, "МОЗГУ ВАЖНЫ НЕ ДЕНЬГИ,", ("НЕ", "ДЕНЬГИ,")), (16.771, "А СРАВНЕНИЕ", ("СРАВНЕНИЕ",), Go_),
    (17.911, "В КАЛИФОРНИИ ЗАРПЛАТЫ", ("КАЛИФОРНИИ",), BLUE_RGB), (19.171, "СОТРУДНИКОВ УНИВЕРСИТЕТА", ()),
    (20.371, "ВЫЛОЖИЛИ В ИНТЕРНЕТ", ("ИНТЕРНЕТ",), BLUE_RGB), (21.771, "ТЕ, КТО УЗНАЛ,", ()),
    (22.571, "ЧТО ПОЛУЧАЕТ МЕНЬШЕ КОЛЛЕГ,", ("МЕНЬШЕ",)), (24.051, "СТАЛИ МЕНЬШЕ ДОВОЛЬНЫ", ("МЕНЬШЕ",)),
    (25.131, "РАБОТОЙ", ()), (26.111, "А ТЕМ, КТО ПОЛУЧАЛ БОЛЬШЕ,", ("БОЛЬШЕ,",), G_),
    (27.451, "РАДОСТИ ЭТО ПОЧТИ", ()), (28.271, "НЕ ДОБАВИЛО", ("НЕ", "ДОБАВИЛО")),
    (29.471, "ЧТО ДЕЛАТЬ?", ("ДЕЛАТЬ?",), G_), (30.491, "СРАВНИВАЙ СЕБЯ НЕ С СОСЕДОМ,", ("СОСЕДОМ",)),
    (31.971, "А С СОБОЙ ГОД НАЗАД", ("СОБОЙ",), G_), (33.211, "ТОГДА ВИДНО, ЧТО ТЫ", ()),
    (34.291, "НА САМОМ ДЕЛЕ РАСТЁШЬ", ("РАСТЁШЬ",), G_), (35.611, "А ЧУЖУЮ ЗАРПЛАТУ", ()),
    (36.591, "ПУСТЬ СЧИТАЕТ БУХГАЛТЕРИЯ", ("БУХГАЛТЕРИЯ",), Go_),
    (38.44, "А ЕЩЁ БОЛЬШЕ ИНТЕРЕСНОГО —", (), T_), (39.8, "В МОЁМ ТЕЛЕГРАМ-КАНАЛЕ", ("ТЕЛЕГРАМ-КАНАЛЕ",), T_),
    (41.13, "«ПАУЧЬЯ НОРКА»", ("«ПАУЧЬЯ", "НОРКА»"), T_), (42.25, "ТАМ ЭКСКЛЮЗИВ,", ("ЭКСКЛЮЗИВ",), T_),
    (42.95, "КОТОРОГО НЕТ В РОЛИКАХ", (), T_), (44.25, "ССЫЛКА В ПРОФИЛЕ —", ("ПРОФИЛЕ",), T_),
    (45.27, "ЗАЛЕТАЙ!", ("ЗАЛЕТАЙ!",), T_),
]

# ---------------------------------------------------------------- камера и сцены
ZOOMS = ((15.0, (760, 640)), (29.1, (330, 640)))  # «влёт в карточку»: (начало, точка на экране)
Z1, Z2 = (z[0] + 0.25 for z in ZOOMS)
CAMERA = [
    (0.0, (-12.0, 4.6, 1.5, 0.9, 50.0)), (1.0, (-8.0, 3.6, 1.15, 0.84, 50.0), "out"),
    (2.55, (-6.0, 3.6, 1.15, 0.84, 50.0)),
    (3.0, (0.0, 3.9, 1.3, 0.98, 50.0), "inout"),            # заставка въезжает сбоку
    (3.85, (1.0, 3.9, 1.3, 0.98, 50.0)),
    (4.3, (95.0, 3.6, 1.2, 0.84, 50.0), "expo"),            # викторина — студия под прожектором
    (Z1 - 0.01, (100.0, 3.3, 1.15, 0.86, 50.0)),
    (Z1 + 0.01, (190.0, 3.6, 1.2, 0.86, 50.0), "lin"),      # весы и таблица — медленный облёт
    (Z2 - 0.01, (198.0, 3.6, 1.25, 0.86, 50.0)),
    (Z2 + 0.01, (285.0, 3.6, 1.2, 0.84, 50.0), "lin"),      # сравни с собой
    (38.05, (300.0, 3.7, 1.25, 0.86, 50.0), "lin"),
    (38.45, (375.0, 4.3, 1.3, 1.12, 50.0), "expo"), (DUR, (376.0, 4.0, 1.3, 1.15, 50.0)),
]
OFFSET = [(0.0, -0.5), (2.55, -0.5), (3.0, 0.0, "inout"), (3.85, 0.0), (4.3, 0.55, "expo"), (Z1 - 0.01, 0.55),
          (Z1 + 0.01, -0.55, "lin"), (Z2 - 0.01, -0.55), (Z2 + 0.01, 0.55, "lin"), (38.05, 0.55),
          (38.45, 0.0, "expo"), (DUR, 0.0)]
TINT = [(0.0, (0.88, 1.04, 1.14)), (3.85, (0.88, 1.04, 1.14)), (4.3, (1.0, 1.0, 1.0)), (Z1 - 0.01, (1.0, 1.0, 1.0)),
        (Z1 + 0.01, (1.14, 1.0, 0.88), "lin"), (Z2 - 0.01, (1.14, 1.0, 0.88)), (Z2 + 0.01, (0.9, 1.1, 0.94), "lin"),
        (38.1, (0.9, 1.1, 0.94)), (38.5, (1.0, 1.0, 1.0)), (DUR, (1.0, 1.0, 1.0))]
SEC = {"hook": -8, "quiz": 98, "scale": 194, "self": 292, "tg": 375.5}
SHAKES = ()
FLASHES = ((2.6, 0.12), (3.9, 0.08), (38.1, 0.12))
JUMPS = []
ALARM = None
SPOT = (4.3, Z1)  # студия: комната темнее, прожектор на паучке

G = lambda *k: list(k)  # noqa: E731
ARM_L = G((0, 62), (0.4, 62), (0.65, -25, "back"), (1.6, -22), (1.75, 36, "back"), (2.45, 38), (2.7, -50, "back"),
          (3.6, -45), (3.95, 62, "back"), (Z1 + 0.05, 62), (15.5, -25, "back"), (17.7, -22), (17.95, -12, "back"),
          (22.0, -10), (22.3, -28, "back"), (28.9, -26), (29.2, 62), (29.5, 30, "back"), (30.3, 32), (30.5, 62),
          (35.6, 62), (35.8, 36, "back"), (37.8, 38), (38.1, 62), (45.38, 62), (45.53, -50, "back"), (46.13, -45),
          (46.53, 62, "back"), (DUR, 62))
ARM_R = G((0, 62), (1.75, 62), (1.9, 36, "back"), (2.45, 38), (2.7, -50, "back"), (3.6, -45), (3.95, 62, "back"),
          (4.4, 62), (6.4, -30, "back"), (9.5, -28), (9.75, -12, "back"), (11.7, -10), (12.0, -30, "back"),
          (14.9, -28), (15.15, 62), (Z2 + 0.05, 62), (29.5, 30, "back"), (30.3, 32), (30.5, -15, "back"),
          (31.8, -12), (31.95, -12, "back"), (33.0, -10), (33.2, -35, "back"), (35.4, -33), (35.8, 36, "back"),
          (37.8, 38), (38.1, 62), (38.23, 62), (38.69, -48, "back"), (41.04, -45), (41.34, -55, "back"), (44.04, -52),
          (44.29, -40, "back"), (45.24, -42), (45.39, -50, "back"), (46.13, -45), (46.28, -58), (46.43, -32),
          (46.58, -58), (46.73, -32), (DUR, 62))
HEAD = G((0, 0), (0.6, -6), (1.75, 8), (2.1, -6), (2.45, 4), (2.7, 0), (4.4, -5), (6.4, -6), (9.8, 6), (12.0, -5),
         (13.3, 8), (15.0, 0), (15.6, 7), (18.0, 6), (22.3, 5), (24.0, -8), (26.1, 4), (28.3, -6), (29.5, 0),
         (30.5, -6), (32.0, -9), (33.2, -6), (34.3, 7), (35.8, -8), (37.4, 8), (38.69, -6), (45.4, 0), (DUR, 0))


# ---------------------------------------------------------------- картинки вставок
def header(d, text, col, dark=False):
    f = mg.font(36)
    tw = d.textbbox((0, 0), text, font=f)[2]
    d.rounded_rectangle([36, 30, 36 + tw + 44, 92], 30, fill=col + (255,))
    d.text((58, 38), text, font=f, fill=(30, 24, 10) if dark else "white")


def bill_tex():
    im = rounded((360, 180), 18, (60, 120, 80, 255), (150, 210, 160, 255), 6)
    d = ImageDraw.Draw(im)
    d.ellipse([130, 40, 230, 140], outline=(170, 230, 180, 255), width=6)
    text_rub(d, (152, 52), "₽", mg.font(70, tall=True), (200, 245, 210))
    return mg.premul(im)


def mine_tex():
    im = rounded((620, 330), 34, (18, 18, 26, 255), (72, 72, 92, 255), 3)
    d = ImageDraw.Draw(im)
    header(d, "ЗАРПЛАТА", GOLD_RGB, dark=True)
    f = mg.font(58, tall=True)
    d.text((40, 120), "моя", font=mg.font(40), fill=(170, 170, 190))
    text_rub(d, (250, 108), "80 000 ₽", f, (255, 255, 255))
    d.text((40, 220), "у коллеги", font=mg.font(40), fill=(255, 150, 150))
    text_rub(d, (250, 208), "95 000 ₽", f, (255, 130, 130))
    return mg.premul(im)


def opt_tex(letter, top, bottom, state):
    """Вариант викторины. state: 0 — обычный, 1 — выбран (половина людей)."""
    col = GOLD_RGB if state else (90, 90, 110)
    im = rounded((760, 230), 50, (24, 22, 30, 255) if not state else (44, 36, 18, 255), col + (255,), 6)
    d = ImageDraw.Draw(im)
    d.ellipse([28, 45, 168, 185], fill=col + (255,))
    f = mg.font(90, tall=True)
    l, t_, r, b = d.textbbox((0, 0), letter, font=f)
    d.text((98 - (r - l) / 2 - l, 115 - (b - t_) / 2 - t_), letter, font=f, fill=(20, 18, 24))
    text_rub(d, (200, 30), top, mg.font(74, tall=True), (255, 255, 255))
    text_rub(d, (204, 140), bottom, mg.font(40), (170, 170, 190))
    if state:
        d.line([(660, 115), (690, 145), (735, 80)], fill=GREEN_RGB + (255,), width=14)
    return mg.premul(im)


def vote_tex(p):
    im = rounded((760, 300), 34, (18, 18, 26, 255), (72, 72, 92, 255), 3)
    d = ImageDraw.Draw(im)
    header(d, "КТО ЧТО ВЫБРАЛ", GOLD_RGB, dark=True)
    d.rounded_rectangle([40, 130, 720, 200], 35, fill=(60, 60, 76, 255))
    w = int(680 * 0.5 * p)
    if w > 70:
        d.rounded_rectangle([40, 130, 40 + w, 200], 35, fill=GOLD_RGB + (255,))
        d.text((64, 144), "A", font=mg.font(44), fill=(30, 24, 10))
    d.text((672, 144), "Б", font=mg.font(44), fill=(200, 200, 215))
    d.text((40, 222), "около половины — вариант А", font=mg.font(38), fill=(240, 210, 140))
    return mg.premul(im)


def table_tex(scroll, hl):
    """Окно браузера с таблицей зарплат; scroll — прокрутка (0..1), hl — подсветить строки «ниже среднего»."""
    Wt, Ht = 760, 620
    im = rounded((Wt, Ht), 30, (18, 18, 24, 255), (80, 80, 100, 255), 4)
    d = ImageDraw.Draw(im)
    for i, c in enumerate(((255, 95, 86), (255, 189, 46), (39, 201, 63))):
        d.ellipse([26 + i * 36, 26, 50 + i * 36, 50], fill=c + (255,))
    d.rounded_rectangle([150, 18, Wt - 26, 58], 20, fill=(36, 36, 46, 255))
    d.text((172, 22), "зарплаты сотрудников", font=mg.font(30), fill=(160, 160, 180))
    rows = []
    rng = np.random.default_rng(3)
    for i in range(18):
        v = int(rng.integers(52, 140)) * 1000
        rows.append((f"Сотрудник {i + 1}", v))
    avg = sum(v for _, v in rows) / len(rows)
    y0 = 90 - int(scroll * 18 * 62 * 0.55)
    for i, (n, v) in enumerate(rows):
        y = y0 + i * 62
        if y < 76 or y > Ht - 70:
            continue
        low = v < avg
        fill = (60, 26, 30, 255) if (hl and low) else (30, 30, 40, 255)
        d.rounded_rectangle([24, y, Wt - 24, y + 52], 16, fill=fill)
        d.text((44, y + 8), n, font=mg.font(34), fill=(210, 210, 225))
        s = f"{v:,} ₽".replace(",", " ")
        f = mg.font(34)
        text_rub(d, (Wt - 50 - rub_width(d, s, f), y + 8), s, f, (255, 140, 140) if (hl and low) else (230, 230, 240))
    d.rounded_rectangle([24, Ht - 62, Wt - 24, Ht - 18], 16, fill=(40, 40, 52, 255))
    text_rub(d, (44, Ht - 58), f"средняя: {int(avg):,} ₽".replace(",", " "), mg.font(32), (240, 210, 140))
    return mg.premul(im)


def sat_tex(a, b):
    """Довольство работой: у тех, кто ниже среднего (a), и выше среднего (b)."""
    im = rounded((760, 440), 34, (18, 18, 26, 255), (72, 72, 92, 255), 3)
    d = ImageDraw.Draw(im)
    header(d, "ДОВОЛЬСТВО РАБОТОЙ", BLUE_RGB)
    for i, (txt, v, col) in enumerate((("получают меньше коллег", a, RED_RGB), ("получают больше", b, GRAY_RGB))):
        y = 130 + i * 150
        d.text((40, y), txt, font=mg.font(40), fill=(210, 210, 225))
        d.rounded_rectangle([40, y + 60, 720, y + 104], 22, fill=(40, 40, 52, 255))
        d.rounded_rectangle([40, y + 60, 40 + int(680 * v), y + 104], 22, fill=col + (255,))
    return mg.premul(im)


def grow_tex(p):
    im = rounded((700, 520), 34, (16, 26, 20, 255), GREEN_RGB + (255,), 5)
    d = ImageDraw.Draw(im)
    header(d, "СРАВНИ С СОБОЙ", GREEN_RGB)
    for i, (lab, h, col) in enumerate((("год назад", 0.42, (80, 110, 100)), ("сейчас", 0.42 + 0.4 * p, GREEN_RGB))):
        x = 110 + i * 280
        hh = int(300 * h)
        d.rounded_rectangle([x, 440 - hh, x + 180, 440], 26, fill=col + (255,))
        d.text((x + 10, 450), lab, font=mg.font(40), fill=(200, 230, 210))
    if p > 0.3:
        d.line([(230, 250), (430, 160)], fill=(240, 240, 240, 255), width=10)
        d.polygon([(430, 160), (390, 158), (420, 190)], fill=(240, 240, 240, 255))
    return mg.premul(im)


# ---------------------------------------------------------------- сцена
class Scene(base.Scene):
    cfg = sys.modules[__name__]

    def __init__(self, voice):
        super().__init__(voice)
        T = self.T
        T["bill"], T["mine"] = bill_tex(), mine_tex()
        T["hurt"] = stamp("ОБИДНО!", RED_RGB)
        T["q"] = card("ВОПРОС УЧЁНЫХ", "ЧТО ЛУЧШЕ?", "Гарвард, 1998", BLUE_RGB, h=300)
        T["optA"], T["optA1"] = opt_tex("A", "50 000 ₽", "а у всех — 25 000 ₽", 0), opt_tex("A", "50 000 ₽", "а у всех — 25 000 ₽", 1)
        T["optB"] = opt_tex("Б", "100 000 ₽", "а у всех — 200 000 ₽", 0)
        T["you"], T["peer"] = chip("ТЫ", GRAY_RGB), chip("КОЛЛЕГА", GOLD_RGB)
        T["cmp"] = chip("ВАЖНО СРАВНЕНИЕ", GOLD_RGB)
        T["coin"] = mg.premul(Image.new("RGBA", (10, 10)))
        T["past"] = chip("ТЫ ГОД НАЗАД", GREEN_RGB)
        T["acc"] = stamp("БУХГАЛТЕРИЯ", GOLD_RGB)
        T["what"] = chip("ЧТО ДЕЛАТЬ?", GREEN_RGB)
        self.cache = {}
        orig = self.room.render
        self.room.render = lambda cam: self.grade(orig(cam))

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
        for z0, (cx, cy) in ZOOMS:  # «влёт в карточку»: резкое приближение к вставке, под ним склейка
            if z0 < t < z0 + 0.5:
                p = (t - z0) / 0.5
                s = 1 + 1.8 * (smooth(p * 2) if p < 0.5 else 1 - smooth((p - 0.5) * 2))
                if p >= 0.5:
                    cx, cy = W / 2, H * 0.45
                M = np.array([[s, 0, cx * (1 - s)], [0, s, cy * (1 - s)]], np.float32)
                fr = cv2.warpAffine(fr, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                fl = math.sin(math.pi * p) * 0.35
                fr = np.clip(fr.astype(np.float32) * (1 + fl), 0, 255).astype(np.uint8)
        return fr

    def ghost(self, cam, pos, k):
        """Полупрозрачная зеленоватая копия паучка — «ты год назад»."""
        if not hasattr(self, "char_args"):
            return
        c = self.char_args[2]
        a = c[..., 3:4]
        g = c.copy()
        g[..., :3] = c[..., :3] * 0.45 + a * np.array([0.55, 0.75, 0.35], np.float32) * 0.55
        g *= 0.5 * k
        mg.billboard(self.frame, cam, pos, self.char_args[1] * 0.9, g, anchor=self.anchor)

    def strings(self, cam, end, pan, r_):
        """Нити от коромысла весов к чаше."""
        for sgn in (-1, 1):
            uv, z = cam.proj(np.array([end, pan + r_ * sgn * 0.17]))
            if (z > 0.3).all():
                mg.lines(self.frame, uv, (150, 140, 120), 3, closed=False)

    def items_for(self, t, cam, it):
        T = self.T
        P = lambda y0, *a, **k: place(y0, *a, **k)  # noqa: E731

        # 1. Крючок (персонаж справа): купюры падают, «моя / у коллеги», «обидно!»
        y0 = SEC["hook"]
        r_, c_ = basis(y0)
        if t < 2.6:
            rng = np.random.default_rng(9)
            for i in range(12):
                s0 = rng.uniform(-0.3, 1.6)
                p = (t - s0) / 1.4
                if not 0 < p < 1:
                    continue
                x, z = rng.uniform(-1.5, 0.6), rng.uniform(-1.6, -0.7)
                pos = r_ * x + c_ * z + [0.05 * math.sin(t * 5 + i), 2.8 - 2.8 * p, 0]
                self.add_bb(it, cam, pos, 0.09, T["bill"], anchor=(0.5, 0.5))
        k = pop(t, 0.5, 2.55, 0.35, 0.3)
        if k > 0:
            pos = P(y0, -0.74, 1.62) + [0, 0.02 * math.sin(t * 3), 0]
            self.add_panel(it, cam, pos, face(y0, -12), 0.7 * k, 0.372 * k, 0.05 * k, T["mine"], pitch=-5,
                           side=(70, 60, 30), edge=(255, 230, 170))
        k = pop(t, 1.75, 2.55, 0.15, 0.25)
        if k > 0:
            ks = clamp((t - 1.75) / 0.15)
            self.add_bb(it, cam, P(y0, -0.6, 1.15, 0.1), 0.15 * (1 + 0.6 * (1 - ks)) * k, T["hurt"], anchor=(0.5, 0.5))

        # 2. Заставка въезжает сбоку
        if 2.6 < t < 4.0:
            e = 1 - (1 - clamp((t - 2.6) / 0.45)) ** 3
            kk = 1 - smooth((t - 3.7) / 0.3)
            self.add_panel(it, cam, place(0, 1.6 * (1 - e), 1.95, 0.35), face(0, -6 + 40 * (1 - e)), 1.2 * kk,
                           0.564 * kk, 0.11 * kk, T["stinger"], pitch=-6, side=(30, 24, 120), edge=(200, 190, 255))

        # 3. Викторина (персонаж слева, прожектор): вопрос → варианты А и Б → голоса
        y0 = SEC["quiz"]
        k = pop(t, 4.35, 6.5, 0.4, 0.3)
        if k > 0:
            pos = P(y0, 0.72, 1.6) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 12), 0.8 * k, 0.31 * k, 0.05 * k, T["q"], pitch=-6,
                           side=(40, 50, 90), edge=(200, 210, 255))
        for key, t0, hy in (("A", 6.45, 1.92), ("B", 9.35, 1.58)):
            k = pop(t, t0, Z1, 0.4, 0.2)
            if k > 0:
                tex = T["optB"] if key == "B" else (T["optA1"] if t > 13.2 else T["optA"])
                s = 1 + (0.06 * smooth((t - 13.2) / 0.2) if key == "A" else 0)
                pos = P(y0, 0.74, hy) + [0, 0.015 * math.sin(t * 2 + hy), 0]
                self.add_panel(it, cam, pos, face(y0, 12), 0.76 * k * s, 0.23 * k * s, 0.05 * k, tex, pitch=-5,
                               side=(70, 60, 30) if key == "A" and t > 13.2 else (50, 50, 60), radius=0.11 * k)
        k = pop(t, 12.0, Z1, 0.4, 0.2)
        if k > 0:
            p = round(smooth((t - 12.1) / 1.0) * 16) / 16
            pos = P(y0, 0.74, 1.24) + [0, 0.02 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 12), 0.76 * k, 0.3 * k, 0.05 * k, self.cached("vote", vote_tex, p),
                           pitch=-5, side=(70, 60, 30), edge=(255, 230, 170))

        # 4. Сравнение (персонаж справа): весы «ты / коллега»; таблица зарплат в интернете; довольство
        y0 = SEC["scale"]
        r_, c_ = basis(y0)
        k = pop(t, Z1 + 0.05, 18.0, 0.45, 0.3)
        if k > 0:
            base_p = r_ * -1.02 + c_ * -0.45
            top = base_p + [0, 1.05 * k, 0]
            self.add_box(it, cam, (base_p[0], base_p[2]), (0.3, 0.3), 0.04, (70, 64, 50), radius=0.12)
            self.add_box(it, cam, (base_p[0], base_p[2]), (0.06, 0.06), 1.05 * k, (110, 100, 70), radius=0.02)
            th = -14 * smooth((t - 16.0) / 0.8) + 2 * math.sin(t * 3) * smooth((t - 16.8) / 0.3)
            ct, st = math.cos(math.radians(th)), math.sin(math.radians(th))
            self.add_panel(it, cam, top + c_ * 0.02, face(y0, 0), 0.56 * k, 0.04 * k, 0.04, self.cached("beam", beam_tex),
                           roll=th, side=(150, 130, 80), radius=0.02)
            for sgn, key in ((-1, "you"), (1, "peer")):
                end = top + r_ * sgn * 0.35 * ct + np.array([0, 1.0, 0]) * sgn * 0.35 * st
                pan = end + [0, -0.28, 0]
                it.append((cam.depth(end) + 0.01, lambda e=end, q=pan: self.strings(cam, e, q, r_)))
                self.add_panel(it, cam, pan + c_ * 0.02, face(y0, 0), 0.3 * k, 0.05 * k, 0.05, self.cached("beam", beam_tex),
                               side=(150, 130, 80), radius=0.02)
                self.add_bb(it, cam, pan + [0, 0.04, 0], 0.08 * k, T[key], anchor=(0.5, 1.0))
            kc = pop(t, 16.8, 18.0, 0.3, 0.25)
            if kc > 0:
                self.add_bb(it, cam, P(y0, -0.8, 2.05), 0.12 * kc, T["cmp"], anchor=(0.5, 0.5))
        k = pop(t, 17.95, 23.2, 0.4, 0.3)
        if k > 0:
            sc = round(smooth((t - 19.0) / 2.6) * 30) / 30
            pos = P(y0, -0.72, 1.6) + [0, 0.02 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, -12), 0.8 * k, 0.653 * k, 0.05 * k,
                           self.cached("table", table_tex, sc, t > 22.6), pitch=-5, side=(40, 40, 52), radius=0.04)
        k = pop(t, 23.15, Z2, 0.4, 0.2)
        if k > 0:
            a = 0.75 - 0.35 * smooth((t - 24.1) / 0.9)
            b = 0.75 + 0.04 * smooth((t - 27.4) / 0.8)
            a, b = round(a * 40) / 40, round(b * 40) / 40
            pos = P(y0, -0.72, 1.55) + [0, 0.02 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, -12), 0.8 * k, 0.463 * k, 0.05 * k, self.cached("sat", sat_tex, a, b),
                           pitch=-5, side=(40, 50, 90), edge=(200, 210, 255))

        # 5. Сравни с собой (персонаж слева): «призрак» — ты год назад, рост; «бухгалтерия»
        y0 = SEC["self"]
        r_, c_ = basis(y0)
        k = pop(t, Z2 + 0.05, 30.6, 0.3, 0.25)
        if k > 0:
            self.add_bb(it, cam, P(y0, 0.78, 1.95), 0.13 * k, T["what"], anchor=(0.5, 0.5))
        k = pop(t, 31.95, 38.1, 0.6, 0.35)
        if k > 0:
            gp = r_ * 0.8 + c_ * -0.6
            it.append((cam.depth(gp + [0, 0.6, 0]), lambda gp=gp, k=k: self.ghost(cam, gp, k)))
            self.add_bb(it, cam, gp + [0, 1.2, 0], 0.1 * k, T["past"], anchor=(0.5, 0.5))
        k = pop(t, 33.15, 36.0, 0.4, 0.3)
        if k > 0:
            p = round(smooth((t - 34.3) / 0.8) * 16) / 16
            pos = P(y0, 0.72, 1.85) + [0, 0.02 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 12), 0.6 * k, 0.446 * k, 0.05 * k, self.cached("grow", grow_tex, p),
                           pitch=-5, side=(40, 70, 50), edge=(200, 255, 220))
        k = pop(t, 36.55, 38.1, 0.15, 0.3)
        if k > 0:
            ks = clamp((t - 36.55) / 0.15)
            self.add_bb(it, cam, P(y0, 0.75, 1.85, 0.1), 0.15 * (1 + 0.6 * (1 - ks)) * k, T["acc"], anchor=(0.5, 0.5))

        # 6. Концовка с Telegram — общий шаблон
        self.outro_items(t, cam, it, OUTRO, SEC["tg"])


def beam_tex():
    return mg.premul(rounded((400, 40), 18, (200, 170, 100, 255)))


SFX = [(t, "whoosh") for t in (0.0, 2.55, 3.85, ZOOMS[0][0], ZOOMS[1][0], 38.05)]


def render(voice, out, frames):
    sc = Scene(voice)
    wr = mg.Writer(out, FPS)
    for n, i in enumerate(frames):
        t = i / FPS
        speed = abs(sc.cam(t + 0.5 / FPS).yaw - sc.cam(t - 0.5 / FPS).yaw) * FPS
        cut = any(abs(t - c) < 1.0 / FPS for c in (Z1, Z2))
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
        for t in (0.6, 1.3, 2.2, 2.9, 3.5, 5.2, 7.5, 10.4, 12.8, 13.9, 15.1, 15.4, 16.6, 17.5, 19.0, 21.5, 23.0,
                  24.8, 27.9, 29.2, 30.0, 32.6, 34.8, 37.2, 40.0, 45.5):
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
    base.mix_audio(tmp, voice, out, music=os.path.join(ROOT, "work", "salary", "music.wav"), sfx_list=SFX, dur=DUR,
                   sfx_vol=0.15)  # «вжух» лёгкий и тихий
    os.remove(tmp)


if __name__ == "__main__":
    main()
