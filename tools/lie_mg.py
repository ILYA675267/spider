"""Ролик №16: «Что будет, если человек перестанет врать совсем?» (+ рекламный баннер в середине).

python tools/lie_mg.py work/lie/voice_clean.wav output/lie.mp4 [--preview] [--jobs 4]
затем: python tools/ad_banner.py insert output/lie.mp4 27.72 output/lie_ad.mp4 lie_mg
Общий движок, персонаж и концовка с Telegram — из memory_mg.py; параллельный рендер — как в money_mg.py.
Новое в монтаже: «детектор лжи» — живая линия-самописец с красной лампой; 10 человечков, 6 краснеют, и таймер
«10 минут» по кругу; камера раскачивается, как маятник, на «мелкой лжи»; пузыри-реплики и «пузыри правды»
с неловкими рожицами; смена сцены «выключением телевизора» (кадр схлопывается в полоску и точку);
календарь 10 недель; итоги стрелками вверх/вниз; финал — кран вверх.
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
OUTRO = float(open(os.path.join(ROOT, "work", "lie", "outro_start.txt")).read())  # 42.19
DUR = 50.75
AD_CUT = 27.72                   # здесь вклеивается баннер (камера стоит спокойно)
GOLD_RGB, GREEN_RGB, PINK_RGB, PURPLE_RGB = (245, 190, 50), (46, 196, 110), (240, 70, 110), (150, 90, 230)

# ---------------------------------------------------------------- субтитры
T_, Go_, G_, Pi_, Pu_ = TG_RGB, GOLD_RGB, GREEN_RGB, PINK_RGB, PURPLE_RGB
SUBS = [
    (0.162, "ЧТО БУДЕТ, ЕСЛИ ЧЕЛОВЕК", ()), (1.464, "ПЕРЕСТАНЕТ ВРАТЬ СОВСЕМ?", ("ВРАТЬ",)), (3.2, "", ()),
    (4.748, "МЫ ВРЁМ ГОРАЗДО ЧАЩЕ,", ("ЧАЩЕ,",)), (6.422, "ЧЕМ ДУМАЕМ", ()),
    (7.408, "В ОПЫТЕ ПСИХОЛОГА", ()), (8.376, "РОБЕРТА ФЕЛЬДМАНА", ("ФЕЛЬДМАНА",), T_),
    (9.306, "60% ЛЮДЕЙ СОВРАЛИ", ("60%",)), (11.501, "ХОТЯ БЫ РАЗ", ()),
    (12.19, "ЗА 10 МИНУТ РАЗГОВОРА", ("10", "МИНУТ"), Go_), (13.715, "С НЕЗНАКОМЦЕМ", ()),
    (14.781, "ЧАЩЕ ВСЕГО ЭТО МЕЛОЧИ:", ("МЕЛОЧИ:",), Pi_), (16.446, "«ВСЁ ХОРОШО»", ()),
    (17.487, "«МНЕ НРАВИТСЯ ПОДАРОК»", ()), (18.901, "«УЖЕ ВЫХОЖУ»", ()),
    (19.879, "ТАКАЯ ЛОЖЬ БЕРЕЖЁТ", ("БЕРЕЖЁТ",), G_), (21.013, "ЧУВСТВА ДРУГИХ", ("ЧУВСТВА",), G_),
    (22.148, "БЕЗ НЕЁ ТЫ БЫ СКАЗАЛ", ()), (23.134, "ДРУГУ, ЧТО ЕГО", ()), (23.86, "СТРИЖКА УЖАСНАЯ", ("УЖАСНАЯ",)),
    (25.283, "А НАЧАЛЬНИКУ — ЧТО", ()), (26.287, "СОВЕЩАНИЕ БЕСПОЛЕЗНОЕ", ("БЕСПОЛЕЗНОЕ",)),
    (AD_CUT - 0.02, "", ()),
    (27.86, "НО ЕСТЬ И ХОРОШАЯ НОВОСТЬ", ("ХОРОШАЯ",), G_), (29.62, "В ДРУГОМ ОПЫТЕ ЛЮДИ", ()),
    (30.811, "10 НЕДЕЛЬ СТАРАЛИСЬ", ("10", "НЕДЕЛЬ"), Go_), (31.946, "ПОЧТИ НЕ ВРАТЬ", ("НЕ", "ВРАТЬ"), G_),
    (32.709, "И СТАЛИ РЕЖЕ ЖАЛОВАТЬСЯ", ("РЕЖЕ",), G_), (34.188, "НА СТРЕСС", ("СТРЕСС",)),
    (34.672, "И ГОЛОВНУЮ БОЛЬ,", ("ГОЛОВНУЮ", "БОЛЬ,")), (35.937, "А ОТНОШЕНИЯ У НИХ", ()),
    (36.886, "СТАЛИ ЛУЧШЕ", ("ЛУЧШЕ",), G_), (37.918, "ТАК ЧТО ВРАТЬ МЕНЬШЕ", ()),
    (38.811, "ПОЛЕЗНО,", ("ПОЛЕЗНО,",), G_), (39.76, "А ПРАВДУ МОЖНО ГОВОРИТЬ", ("ПРАВДУ",), G_),
    (40.913, "ПО-ДОБРОМУ", ("ПО-ДОБРОМУ",), Pi_),
    (42.154, "А ЕЩЁ БОЛЬШЕ ИНТЕРЕСНОГО —", (), T_), (43.514, "В МОЁМ ТЕЛЕГРАМ-КАНАЛЕ", ("ТЕЛЕГРАМ-КАНАЛЕ",), T_),
    (44.844, "«ПАУЧЬЯ НОРКА»", ("«ПАУЧЬЯ", "НОРКА»"), T_), (45.964, "ТАМ ЭКСКЛЮЗИВ,", ("ЭКСКЛЮЗИВ",), T_),
    (46.664, "КОТОРОГО НЕТ В РОЛИКАХ", (), T_), (47.964, "ССЫЛКА В ПРОФИЛЕ —", ("ПРОФИЛЕ",), T_),
    (48.984, "ЗАЛЕТАЙ!", ("ЗАЛЕТАЙ!",), T_),
]

# ---------------------------------------------------------------- камера и сцены
TV = 27.75                       # «выключение телевизора»; склейка — на его середине
C2 = TV + 0.15
SWAY = (15.0, 22.0)              # маятник камеры на «мелкой лжи»
CAMERA = [
    (0.0, (14.0, 4.2, 1.45, 0.88, 50.0)), (0.8, (8.0, 3.6, 1.15, 0.84, 50.0), "out"),
    (3.0, (6.0, 3.6, 1.15, 0.84, 50.0)),
    (3.3, (0.0, 3.9, 1.3, 0.98, 50.0), "inout"),              # заставка
    (4.6, (-1.0, 3.9, 1.3, 0.98, 50.0)),
    (4.95, (-95.0, 3.6, 1.2, 0.84, 50.0), "expo"),            # опыт Фельдмана — медленный наезд
    (14.6, (-100.0, 3.4, 1.15, 0.86, 49.0), "lin"),
    (15.0, (-188.0, 3.7, 1.3, 0.86, 50.0), "expo"),           # мелкая ложь — маятник
    (22.0, (-188.0, 3.7, 1.3, 0.86, 50.0)),
    (22.25, (-190.0, 3.4, 1.25, 0.88, 48.0), "back"),         # «без неё…» — резкий наезд
    (C2 - 0.01, (-192.0, 3.35, 1.25, 0.88, 48.0), "lin"),
    (C2 + 0.01, (-290.0, 3.6, 1.25, 0.84, 50.0), "lin"),      # хорошая новость
    (37.6, (-297.0, 3.5, 1.25, 0.84, 50.0), "lin"),
    (38.0, (-372.0, 3.6, 2.5, 1.15, 50.0), "expo"),           # финал — кран вверх
    (41.9, (-376.0, 3.9, 2.9, 1.2, 50.0), "lin"),
    (42.15, (-460.0, 4.3, 1.3, 1.12, 50.0), "expo"), (DUR, (-461.0, 4.0, 1.3, 1.15, 50.0)),
]
OFFSET = [(0.0, 0.55), (3.0, 0.55), (3.3, 0.0, "inout"), (4.6, 0.0), (4.95, -0.55, "expo"), (14.6, -0.55),
          (15.0, 0.55, "expo"), (C2 - 0.01, 0.55), (C2 + 0.01, -0.55, "lin"), (37.6, -0.55), (38.0, 0.0, "expo"),
          (DUR, 0.0)]
# окраска комнаты по сценам (B, G, R)
TINT = [(0.0, (0.9, 0.95, 1.12)), (3.0, (0.9, 0.95, 1.12)), (3.3, (1.0, 1.0, 1.0)), (4.6, (1.0, 1.0, 1.0)),
        (4.95, (1.1, 1.05, 0.9)), (14.6, (1.1, 1.05, 0.9)), (15.0, (1.08, 0.92, 1.1)), (C2 - 0.01, (1.08, 0.92, 1.1)),
        (C2 + 0.01, (0.92, 1.1, 0.95), "lin"), (37.6, (0.92, 1.1, 0.95)), (38.0, (0.88, 1.02, 1.12)),
        (41.9, (0.88, 1.02, 1.12)), (42.3, (1.0, 1.0, 1.0)), (DUR, (1.0, 1.0, 1.0))]
SEC = {"hook": 10, "feld": -97, "lies": -188, "good": -293, "final": -374, "tg": -460.5}
SHAKES = (9.31,)
FLASHES = ((2.1, 0.08), (42.2, 0.12))
JUMPS = [28.0]
ALARM = None
SPOT = None

G = lambda *k: list(k)  # noqa: E731
O = OUTRO - 48.035      # жесты концовки — как в monet_mg, со сдвигом
ARM_L = G((0, 62), (3.0, 62), (3.3, -50, "back"), (4.3, -45), (4.6, 62, "back"), (4.95, -30, "back"), (8.9, -28),
          (9.3, -48, "back"), (11.2, -45), (11.5, -25, "back"), (14.4, -22), (14.7, 62), (19.7, 62), (19.9, 30, "back"),
          (21.8, 32), (22.0, 62), (23.7, 62), (23.9, 20, "back"), (25.0, 22), (25.2, 62), (C2 + 0.05, 62),
          (28.0, -30, "back"), (32.5, -28), (32.8, -45, "back"), (33.3, -42), (33.5, -25, "back"), (37.5, -22),
          (37.8, 62), (38.0, 25, "back"), (39.6, 28), (39.8, -20, "back"), (41.6, -18), (41.9, 62),
          (54.94 + O, 62), (55.09 + O, -50, "back"), (55.69 + O, -45), (56.09 + O, 62, "back"), (DUR, 62))
ARM_R = G((0, 62), (0.3, -28, "back"), (2.9, -26), (3.1, 62), (3.3, -50, "back"), (4.3, -45), (4.6, 62, "back"),
          (14.7, 62), (15.0, -30, "back"), (16.3, -28), (16.45, -38, "back"), (17.4, -36), (17.5, -26, "back"),
          (18.8, -24), (18.9, -38, "back"), (19.7, -36), (19.9, 30, "back"), (21.8, 32), (22.1, -40, "back"),
          (23.7, -38), (23.9, 20, "back"), (25.0, 22), (25.3, -35, "back"), (27.2, -33), (27.5, 62),
          (32.8, 62), (33.0, -45, "back"), (33.3, -42), (33.5, 62), (37.8, 62), (38.0, 25, "back"), (39.6, 28),
          (39.8, -50, "back"), (41.6, -48), (41.9, 62),
          (47.79 + O, 62), (48.25 + O, -48, "back"), (50.6 + O, -45), (50.9 + O, -55, "back"), (53.6 + O, -52),
          (53.85 + O, -40, "back"), (54.8 + O, -42), (54.95 + O, -50, "back"), (55.69 + O, -45), (55.84 + O, -58),
          (55.99 + O, -32), (56.14 + O, -58), (56.29 + O, -32), (DUR, 62))
HEAD = G((0, 0), (0.5, 6), (1.5, -7), (2.3, 8), (3.3, 0), (4.8, -6), (6.4, 7), (7.4, -5), (9.3, 9), (11.5, -6),
         (12.2, 6), (13.7, -5), (14.8, 7), (16.4, -8), (17.5, 8), (18.9, -8), (19.9, 6), (21.0, -4), (22.1, 7),
         (23.9, -9), (25.3, 7), (26.3, -8), (27.6, 0), (28.0, 7), (29.6, -6), (30.8, 6), (32.7, -7), (34.7, 6),
         (35.9, -5), (36.9, 8), (37.9, -5), (39.8, 6), (40.9, -6), (42.4, -6), (49.2, 0), (DUR, 0))


# ---------------------------------------------------------------- картинки вставок
def poly_tex(t):
    """«Детектор лжи»: бегущая линия самописца, на «врать» и «совсем» — всплески, лампа «ЛОЖЬ?»."""
    im = rounded((780, 440), 34, (16, 16, 22, 255), (72, 72, 92, 255), 3)
    d = ImageDraw.Draw(im)
    d.text((40, 30), "ДЕТЕКТОР ЛЖИ", font=mg.font(56, tall=True), fill="white")
    hot = 2.0 < t < 3.2
    lamp = RED_RGB if hot and int(t * 8) % 2 == 0 else (70, 40, 44)
    d.ellipse([640, 30, 720, 110], fill=lamp + (255,))
    d.text((560, 120), "ЛОЖЬ?", font=mg.font(40, tall=True), fill=RED_RGB + (255,) if hot else (90, 90, 104))
    d.rounded_rectangle([40, 180, 740, 400], 20, fill=(24, 24, 32, 255))
    for y in range(200, 400, 40):
        d.line([(50, y), (730, y)], fill=(40, 40, 52), width=2)
    pts = []
    for i in range(140):
        x = 50 + i * 5
        tt = t - (139 - i) * 0.02            # история сигнала: правый край — «сейчас»
        a = 12 * math.sin(tt * 9) + 6 * math.sin(tt * 23)
        for s0 in (2.08, 2.3):
            if s0 < tt < s0 + 0.5:
                a += 70 * math.sin((tt - s0) * 40) * (1 - (tt - s0) / 0.5)
        pts.append((x, 290 + a))
    d.line(pts, fill=(RED_RGB if hot else GREEN_RGB) + (255,), width=5, joint="curve")
    return mg.premul(im)


def people_tex(nred):
    im = rounded((780, 420), 34, (18, 18, 26, 255), (72, 72, 92, 255), 3)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([36, 30, 330, 92], 30, fill=RED_RGB + (255,))
    d.text((58, 38), "СОВРАЛИ: 60%", font=mg.font(36), fill="white")
    for i in range(10):
        r, c = divmod(i, 5)
        x, y = 70 + c * 135, 120 + r * 150
        col = RED_RGB if i < nred else (110, 110, 125)
        d.ellipse([x + 25, y, x + 75, y + 50], fill=col + (255,))
        d.rounded_rectangle([x + 10, y + 56, x + 90, y + 130], 26, fill=col + (255,))
    return mg.premul(im)


def ring_tex(p):
    """Таймер «10 минут»: кольцо убывает, в центре оставшееся время."""
    S = 300
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([10, 10, S - 10, S - 10], fill=(18, 18, 26, 255))
    d.arc([22, 22, S - 22, S - 22], 0, 360, fill=(50, 50, 62, 255), width=20)
    if p < 1:
        d.arc([22, 22, S - 22, S - 22], -90 + 360 * p, 270, fill=GOLD_RGB + (255,), width=20)
    m = int(round(10 * (1 - p)))
    txt = f"{m:02d}:00"
    f = mg.font(64, tall=True)
    tw = d.textbbox((0, 0), txt, font=f)[2]
    d.text((S / 2 - tw / 2, 100), txt, font=f, fill="white")
    d.text((S / 2 - 40, 186), "минут", font=mg.font(30), fill=(150, 150, 170))
    return mg.premul(im)


def bubble_tex(text, col, truth=False):
    f = mg.font(52)
    tmp = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    tw = tmp.textbbox((0, 0), text, font=f)[2]
    w = tw + 90 + (130 if truth else 0)
    im = Image.new("RGBA", (w, 170), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, w - 1, 128], 46, fill=(22, 20, 30, 255), outline=col + (255,), width=6)
    d.polygon([(46, 122), (96, 122), (40, 168)], fill=col + (255,))
    d.text((45, 30), text, font=f, fill="white")
    if truth:  # неловкая рожица
        cx, cy = w - 80, 64
        d.ellipse([cx - 44, cy - 44, cx + 44, cy + 44], fill=GOLD_RGB + (255,))
        d.ellipse([cx - 22, cy - 18, cx - 10, cy - 4], fill=(40, 30, 10))
        d.ellipse([cx + 10, cy - 18, cx + 22, cy - 4], fill=(40, 30, 10))
        d.line([(cx - 22, cy + 18), (cx - 8, cy + 12), (cx + 6, cy + 20), (cx + 22, cy + 14)], fill=(40, 30, 10), width=5)
    return mg.premul(im)


def heart_chip(text, col):
    f = mg.font(54)
    tmp = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    tw = tmp.textbbox((0, 0), text, font=f)[2]
    im = rounded((tw + 170, 110), 30, (20, 20, 28, 255), col + (255,), 5)
    d = ImageDraw.Draw(im)
    d.ellipse([30, 32, 62, 64], fill=PINK_RGB + (255,))
    d.ellipse([56, 32, 88, 64], fill=PINK_RGB + (255,))
    d.polygon([(32, 54), (86, 54), (59, 86)], fill=PINK_RGB + (255,))
    d.text((115, 22), text, font=f, fill="white")
    return mg.premul(im)


def weeks_tex(n):
    im = rounded((780, 200), 34, (18, 18, 26, 255), (72, 72, 92, 255), 3)
    d = ImageDraw.Draw(im)
    d.text((40, 24), "10 НЕДЕЛЬ ПОЧТИ БЕЗ ЛЖИ", font=mg.font(40, tall=True), fill="white")
    for i in range(10):
        x = 40 + i * 70
        on = i < n
        d.rounded_rectangle([x, 96, x + 56, 160], 14, fill=(GREEN_RGB if on else (44, 44, 56)) + (255,))
        if on:
            d.line([(x + 14, 128), (x + 25, 141), (x + 44, 112)], fill="white", width=7)
    return mg.premul(im)


def result_tex(rows):
    """Итоги опыта: стресс ↓, головная боль ↓, отношения ↑ (стрелки нарисованы, в шрифте их нет)."""
    im = rounded((780, 420), 34, (16, 24, 20, 255), GREEN_RGB + (255,), 4)
    d = ImageDraw.Draw(im)
    tw = d.textbbox((0, 0), "ИТОГ ОПЫТА", font=mg.font(36))[2]
    d.rounded_rectangle([36, 30, 36 + tw + 44, 92], 30, fill=GREEN_RGB + (255,))
    d.text((58, 38), "ИТОГ ОПЫТА", font=mg.font(36), fill="white")
    items = (("стресс", False), ("головная боль", False), ("отношения", True))
    for i, (txt, up) in enumerate(items[:rows]):
        y = 120 + i * 96
        d.text((50, y + 8), txt, font=mg.font(52), fill="white")
        col = GREEN_RGB
        cx = 680
        if up:
            d.polygon([(cx - 34, y + 50), (cx, y + 6), (cx + 34, y + 50)], fill=col + (255,))
            d.rectangle([cx - 12, y + 48, cx + 12, y + 76], fill=col + (255,))
        else:
            d.polygon([(cx - 34, y + 32), (cx, y + 76), (cx + 34, y + 32)], fill=col + (255,))
            d.rectangle([cx - 12, y + 6, cx + 12, y + 34], fill=col + (255,))
        d.text((480, y + 14), "меньше" if not up else "лучше", font=mg.font(40), fill=(170, 220, 190))
    return mg.premul(im)


def final_tex():
    im = rounded((860, 400), 40, (22, 16, 22, 255), PINK_RGB + (255,), 5)
    d = ImageDraw.Draw(im)
    f = mg.font(96, tall=True)
    d.text((44, 40), "ПРАВДА", font=f, fill="white")
    d.text((64 + d.textbbox((44, 40), "ПРАВДА", font=f)[2], 40), "+", font=f, fill=PINK_RGB + (255,))
    d.text((44, 160), "ДОБРОТА", font=f, fill=PINK_RGB + (255,))
    for k, (x, y) in enumerate(((680, 210), )):
        d.ellipse([x, y, x + 60, y + 60], fill=PINK_RGB + (255,))
        d.ellipse([x + 48, y, x + 108, y + 60], fill=PINK_RGB + (255,))
        d.polygon([(x + 4, y + 40), (x + 104, y + 40), (x + 54, y + 100)], fill=PINK_RGB + (255,))
    d.text((48, 300), "говори честно, но мягко", font=mg.font(44), fill=(190, 170, 190))
    return mg.premul(im)


# ---------------------------------------------------------------- сцена
class Scene(base.Scene):
    cfg = sys.modules[__name__]

    def __init__(self, voice):
        super().__init__(voice)
        T = self.T
        T["feld"] = card("ОПЫТ", "ФЕЛЬДМАН, 2002", "Массачусетский университет, США", TG_RGB)
        T["b1"] = bubble_tex("Всё хорошо!", GRAY_RGB)
        T["b2"] = bubble_tex("Мне нравится подарок!", GRAY_RGB)
        T["b3"] = bubble_tex("Уже выхожу!", GRAY_RGB)
        T["small"] = stamp("МЕЛКАЯ ЛОЖЬ", PINK_RGB)
        T["care"] = heart_chip("БЕРЕЖЁТ ЧУВСТВА", GREEN_RGB)
        T["t1"] = bubble_tex("Стрижка ужасная", RED_RGB, truth=True)
        T["t2"] = bubble_tex("Совещание бесполезное", RED_RGB, truth=True)
        T["truth"] = chip("ПРАВДА БЕЗ ЛЖИ:", RED_RGB)
        T["good"] = chip("ХОРОШАЯ НОВОСТЬ", GREEN_RGB)
        T["kelly"] = card("ОПЫТ", "АНИТА КЕЛЛИ, 2012", "университет Нотр-Дам, США", GREEN_RGB)
        T["final"] = final_tex()
        self.cache = {}
        orig = self.room.render
        self.room.render = lambda cam: self.grade(orig(cam))

    def cached(self, key, fn, *a):
        k = (key,) + a
        if k not in self.cache:
            if len(self.cache) > 400:
                self.cache.clear()
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
        # маятник: камера плавно качается влево-вправо, пока звучит «мелкая ложь»
        sw = pop(t, SWAY[0] + 0.3, SWAY[1], 0.6, 0.6)
        yaw += 7.0 * sw * math.sin((t - SWAY[0]) * 2.4)
        a = math.radians(yaw)
        shake = 0.0
        for s0 in SHAKES:
            if s0 < t < s0 + 0.35:
                shake = (1 - (t - s0) / 0.35) * math.sin((t - s0) * 70)
        return mg.Cam(yaw + 0.5 * wobble(t, 1) + 1.2 * shake, dist, hgt + 0.02 * wobble(t, 2) + 0.03 * shake,
                      ty + 0.015 * wobble(t, 3), fov=fov, roll=0.5 * wobble(t * 0.8, 4) + 1.5 * shake + 2.5 * sw *
                      math.sin((t - SWAY[0]) * 2.4), tx=off * math.cos(a), tz=off * math.sin(a))

    def finish(self, frame, t):
        fr = super().finish(frame, t)
        p = (t - TV) / 0.3
        if 0 < p < 1:  # «выключение телевизора»: кадр схлопывается в полоску и точку, новый — разворачивается
            q = p / 0.5 if p < 0.5 else (1 - p) / 0.5       # 0 → 1 (схлопнут) → 0
            sy = max(0.006, 1 - smooth(min(1.0, q / 0.7)))
            sx = 1 - 0.92 * smooth(max(0.0, (q - 0.7) / 0.3))
            h, w = max(2, int(H * sy)), max(2, int(W * sx))
            small = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
            out = np.zeros_like(fr)
            y0, x0 = (H - h) // 2, (W - w) // 2
            glow = 1 + 1.5 * smooth(q)
            out[y0:y0 + h, x0:x0 + w] = np.clip(small.astype(np.float32) * glow + 60 * smooth(q), 0, 255)
            fr = out
        return fr

    def items_for(self, t, cam, it):
        T = self.T
        P = lambda y0, *a, **k: place(y0, *a, **k)  # noqa: E731

        # 1. Крючок (персонаж слева): детектор лжи с живой линией
        y0 = SEC["hook"]
        k = pop(t, 0.2, 3.15, 0.45, 0.25)
        if k > 0:
            pos = P(y0, 0.72, 1.62) + [0, 0.02 * math.sin(t * 3), 0]
            self.add_panel(it, cam, pos, face(y0, 12), 0.9 * k, 0.508 * k, 0.06 * k,
                           self.cached("poly", poly_tex, round(t * 15) / 15), pitch=-6, side=(70, 40, 44),
                           edge=(255, 200, 200))

        # 2. Заставка
        k = pop(t, 3.3, 4.6, 0.35, 0.3)
        if k > 0:
            self.add_panel(it, cam, place(0, 0.0, 1.95, 0.35), face(0, -6), 1.2 * k, 0.564 * k, 0.11 * k,
                           T["stinger"], pitch=-6, side=(30, 24, 120), edge=(200, 190, 255))

        # 3. Опыт Фельдмана (персонаж справа): карточка → 10 человечков, 6 краснеют → таймер 10 минут
        y0 = SEC["feld"]
        k = pop(t, 7.35, 9.25, 0.4, 0.25)
        if k > 0:
            pos = P(y0, -0.72, 1.6) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, -14), 0.9 * k, 0.427 * k, 0.06 * k, T["feld"], pitch=-6,
                           side=(30, 60, 80), edge=(190, 230, 255))
        k = pop(t, 9.25, 14.65, 0.4, 0.25)
        if k > 0:
            n = int(clamp((t - 9.4) / 1.2) * 6 + 0.999) if t > 9.4 else 0
            pos = P(y0, -0.72, 1.45) + [0, 0.02 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, -14), 0.9 * k, 0.485 * k, 0.06 * k, self.cached("ppl", people_tex, n),
                           pitch=-6, side=(80, 40, 40), edge=(255, 200, 200))
            kr = pop(t, 12.1, 14.65, 0.35, 0.25)
            if kr > 0:
                pr = round(clamp((t - 12.4) / 2.0) * 20) / 20
                self.add_bb(it, cam, P(y0, -0.22, 1.85), 0.28 * kr, self.cached("ring", ring_tex, pr), anchor=(0.5, 0.5))

        # 4. Мелкая ложь (персонаж слева, камера-маятник): реплики → «бережёт чувства» → «правда без лжи»
        y0 = SEC["lies"]
        r_, c_ = basis(y0)
        for i, (t0, key, up) in enumerate(((16.4, "b1", 2.15), (17.45, "b2", 1.8), (18.85, "b3", 1.45))):
            kb = pop(t, t0, 21.95, 0.35, 0.25)
            if kb > 0:
                self.add_bb(it, cam, P(y0, 0.74 + 0.04 * i, up), 0.17 * kb, T[key], anchor=(0.5, 0.5))
        k = pop(t, 14.8, 16.3, 0.35, 0.25)
        if k > 0:
            self.add_bb(it, cam, P(y0, 0.78, 1.8), 0.22 * k * (1 + 0.5 * (1 - clamp((t - 14.8) / 0.15))), T["small"],
                        anchor=(0.5, 0.5))
        k = pop(t, 19.85, 21.95, 0.35, 0.25)
        if k > 0:
            self.add_bb(it, cam, P(y0, 0.78, 1.1), 0.11 * k, T["care"], anchor=(0.5, 0.5))
        k = pop(t, 22.15, AD_CUT + 0.3, 0.35, 0.2)
        if k > 0:
            self.add_bb(it, cam, P(y0, 0.75, 2.05), 0.1 * k, T["truth"], anchor=(0.5, 0.5))
            k1 = pop(t, 23.85, AD_CUT + 0.3, 0.35, 0.2)
            if k1 > 0:
                self.add_bb(it, cam, P(y0, 0.78, 1.85), 0.16 * k1, T["t1"], anchor=(0.5, 0.5))
            k2 = pop(t, 26.25, AD_CUT + 0.3, 0.35, 0.2)
            if k2 > 0:
                self.add_bb(it, cam, P(y0, 0.66, 1.45), 0.16 * k2, T["t2"], anchor=(0.5, 0.5))

        # 5. Хорошая новость (персонаж справа): опыт Келли → 10 недель → итоги стрелками
        y0 = SEC["good"]
        k = pop(t, C2 + 0.05, 29.55, 0.35, 0.25)
        if k > 0:
            self.add_bb(it, cam, P(y0, -0.7, 1.95), 0.14 * k * (1 + 0.4 * (1 - clamp((t - C2) / 0.2))), T["good"],
                        anchor=(0.5, 0.5))
        k = pop(t, 29.6, 32.65, 0.4, 0.25)
        if k > 0:
            pos = P(y0, -0.72, 1.85) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, -14), 0.9 * k, 0.427 * k, 0.06 * k, T["kelly"], pitch=-6,
                           side=(30, 70, 40), edge=(190, 255, 210))
            kw = pop(t, 30.75, 32.65, 0.35, 0.25)
            if kw > 0:
                n = int(clamp((t - 30.9) / 1.4) * 10 + 0.999) if t > 30.9 else 0
                self.add_panel(it, cam, P(y0, -0.72, 1.25), face(y0, -14), 0.9 * kw, 0.231 * kw, 0.04 * kw,
                               self.cached("wk", weeks_tex, n), pitch=-6, side=(30, 70, 40), edge=(190, 255, 210))
        k = pop(t, 32.7, 37.85, 0.4, 0.25)
        if k > 0:
            rows = 1 + (t > 34.65) + (t > 36.05)
            pos = P(y0, -0.72, 1.6) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, -14), 0.9 * k, 0.485 * k, 0.06 * k, self.cached("res", result_tex, rows),
                           pitch=-6, side=(30, 70, 40), edge=(190, 255, 210))

        # 6. Финал (персонаж в центре, кран вверх): «ПРАВДА + ДОБРОТА»
        y0 = SEC["final"]
        k = pop(t, 38.1, 41.95, 0.45, 0.25)
        if k > 0:
            pos = place(y0, 0.0, 2.15, 0.2) + [0, 0.02 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 0), 0.95 * k, 0.442 * k, 0.06 * k, T["final"], pitch=-18,
                           side=(70, 30, 50), edge=(255, 200, 220))

        # 7. Концовка с Telegram — общий шаблон
        self.outro_items(t, cam, it, OUTRO, SEC["tg"])


SFX = [(t, "whoosh") for t in (0.0, 3.25, 4.9, 14.95, TV, 37.95, 42.0)]


def render(voice, out, frames):
    sc = Scene(voice)
    wr = mg.Writer(out, FPS)
    for n, i in enumerate(frames):
        t = i / FPS
        speed = abs(sc.cam(t + 0.5 / FPS).yaw - sc.cam(t - 0.5 / FPS).yaw) * FPS
        cut = abs(t - C2) < 1.0 / FPS
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
        for t in (0.6, 2.2, 4.0, 6.0, 8.2, 10.5, 13.0, 15.5, 17.0, 19.3, 21.0, 23.0, 25.0, 27.5, 27.82, 28.6, 31.5,
                  32.4, 36.5, 39.0, 41.0, 45.0):
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
    base.mix_audio(tmp, voice, out, music=os.path.join(ROOT, "work", "lie", "music.wav"), sfx_list=SFX, dur=DUR,
                   sfx_vol=0.15)  # «вжух» лёгкий и тихий
    os.remove(tmp)


if __name__ == "__main__":
    main()
