"""Ролик №21: «Что будет, если месяц не пользоваться телефоном?» (+ рекламный баннер в середине).

python tools/nophone_mg.py work/nophone/voice_clean.wav output/nophone.mp4 [--preview] [--jobs 4]
затем: python tools/ad_banner.py insert output/nophone.mp4 27.12 output/nophone_ad.mp4 nophone_mg
Общий движок, персонаж и концовка с Telegram — из memory_mg.py; параллельный рендер — как в money_mg.py.
Новое в монтаже: огромный телефон лежит на полу, камера смотрит сверху и опускается; уведомления сыплются
и счётчик «60 раз в день»; телефон дрожит от «фантомной вибрации», 9 человечков из 10 подсвечены; смена сцены
«поворотом телефона» — кадр поворачивается на 90°, как экран, и выпрямляется уже новым; мозг ждёт сигнала;
два стола — с телефоном и без, столбики результатов; календарь «30 дней без соцсетей» и «+1 час»;
настроение-шкала; телефон уезжает в ящик.
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
OUTRO = float(open(os.path.join(ROOT, "work", "nophone", "outro_start.txt")).read())  # 41.31
DUR = 49.95
AD_CUT = 27.12                   # здесь вклеивается баннер (камера стоит спокойно)
GOLD_RGB, GREEN_RGB, PINK_RGB, PURPLE_RGB = (245, 190, 50), (46, 196, 110), (240, 70, 110), (150, 90, 230)

# ---------------------------------------------------------------- субтитры
T_, Go_, G_, Pi_, Pu_ = TG_RGB, GOLD_RGB, GREEN_RGB, PINK_RGB, PURPLE_RGB
SUBS = [
    (0.179, "ЧТО БУДЕТ, ЕСЛИ МЕСЯЦ", ()), (1.559, "НЕ ПОЛЬЗОВАТЬСЯ ТЕЛЕФОНОМ?", ("ТЕЛЕФОНОМ?",), T_), (3.0, "", ()),
    (4.859, "В СРЕДНЕМ ЧЕЛОВЕК", ()), (5.619, "БЕРЁТ ТЕЛЕФОН", ()), (6.479, "ОКОЛО 60 РАЗ В ДЕНЬ", ("60",), Go_),
    (8.71, "ПОЭТОМУ ПЕРВЫЕ ДНИ", ()), (9.829, "БУДЕТ КАЗАТЬСЯ, ЧТО ОН", ()),
    (10.929, "ВИБРИРУЕТ В КАРМАНЕ", ("ВИБРИРУЕТ",), Pu_), (12.389, "ТАК БЫВАЕТ ПОЧТИ", ()),
    (13.269, "У 9 ИЗ 10 СТУДЕНТОВ", ("9", "ИЗ", "10"), Go_),
    (15.63, "ЭТО МОЗГ ПО ПРИВЫЧКЕ", ()), (16.803, "ЖДЁТ СИГНАЛА", ("СИГНАЛА",), Pu_),
    (17.95, "УЧЁНЫЕ ВЫЯСНИЛИ:", ()), (19.423, "ДАЖЕ ВЫКЛЮЧЕННЫЙ ТЕЛЕФОН", ("ВЫКЛЮЧЕННЫЙ",)),
    (20.723, "НА СТОЛЕ МЕШАЕТ ДУМАТЬ", ("МЕШАЕТ",)), (22.65, "ЛЮДИ ХУЖЕ РЕШАЛИ ЗАДАЧИ,", ("ХУЖЕ",)),
    (24.343, "ЧЕМ ТЕ, У КОГО ТЕЛЕФОН", ()), (25.443, "ЛЕЖАЛ В ДРУГОЙ КОМНАТЕ", ("ДРУГОЙ", "КОМНАТЕ"), G_),
    (AD_CUT - 0.02, "", ()),
    (27.33, "А КОГДА ЛЮДЕЙ НА МЕСЯЦ", ()), (28.575, "ОТКЛЮЧАЛИ ОТ СОЦСЕТЕЙ,", ("СОЦСЕТЕЙ,",), T_),
    (30.155, "У НИХ ОСВОБОЖДАЛСЯ", ()), (31.115, "ПОЧТИ ЧАС В ДЕНЬ", ("ЧАС",), G_),
    (32.195, "И УЛУЧШАЛОСЬ НАСТРОЕНИЕ", ("НАСТРОЕНИЕ",), G_), (34.38, "СОВЕТ:", ("СОВЕТ:",), Go_),
    (35.345, "НЕ ОБЯЗАТЕЛЬНО", ()), (35.865, "БРОСАТЬ ТЕЛЕФОН —", ()), (37.245, "ПРОСТО УБИРАЙ ЕГО ПОДАЛЬШЕ,", ("ПОДАЛЬШЕ,",), G_),
    (39.025, "КОГДА РАБОТАЕШЬ ИЛИ СПИШЬ", ()),
    (41.273, "А ЕЩЁ БОЛЬШЕ ИНТЕРЕСНОГО —", (), T_), (42.633, "В МОЁМ ТЕЛЕГРАМ-КАНАЛЕ", ("ТЕЛЕГРАМ-КАНАЛЕ",), T_),
    (43.963, "«ПАУЧЬЯ НОРКА»", ("«ПАУЧЬЯ", "НОРКА»"), T_), (45.083, "ТАМ ЭКСКЛЮЗИВ,", ("ЭКСКЛЮЗИВ",), T_),
    (45.783, "КОТОРОГО НЕТ В РОЛИКАХ", (), T_), (47.083, "ССЫЛКА В ПРОФИЛЕ —", ("ПРОФИЛЕ",), T_),
    (48.103, "ЗАЛЕТАЙ!", ("ЗАЛЕТАЙ!",), T_),
]

# ---------------------------------------------------------------- камера и сцены
TURNS = (15.25, 27.15)           # «поворот телефона»; склейка — на середине
C1, C2 = (r + 0.2 for r in TURNS)
CAMERA = [
    (0.0, (-8.0, 2.6, 3.4, 0.2, 52.0)), (2.4, (-8.0, 3.6, 1.2, 0.84, 50.0), "inout"),   # сверху на телефон → вниз
    (2.85, (-7.0, 3.6, 1.2, 0.84, 50.0)),
    (3.15, (0.0, 3.9, 1.3, 0.98, 50.0), "inout"),             # заставка
    (4.6, (-1.0, 3.9, 1.3, 0.98, 50.0)),
    (4.9, (-92.0, 3.6, 1.25, 0.86, 50.0), "expo"),            # 60 раз в день, вибрация
    (C1 - 0.01, (-99.0, 3.4, 1.25, 0.86, 50.0), "lin"),
    (C1 + 0.01, (-182.0, 3.7, 1.6, 0.86, 50.0), "lin"),       # мозг и опыт с телефоном на столе
    (C2 - 0.01, (-189.0, 3.6, 1.55, 0.86, 50.0), "lin"),
    (C2 + 0.01, (-272.0, 3.6, 1.2, 0.86, 50.0), "lin"),       # месяц без соцсетей
    (34.1, (-277.0, 3.5, 1.2, 0.86, 50.0), "lin"),
    (34.45, (-366.0, 3.6, 1.1, 0.9, 50.0), "expo"),           # совет
    (40.9, (-372.0, 3.4, 1.1, 0.9, 49.0), "lin"),
    (41.25, (-460.0, 4.3, 1.3, 1.12, 50.0), "expo"), (DUR, (-461.0, 4.0, 1.3, 1.15, 50.0)),
]
OFFSET = [(0.0, -0.4), (2.85, -0.5), (3.15, 0.0, "inout"), (4.6, 0.0), (4.9, 0.55, "expo"), (C1 - 0.01, 0.55),
          (C1 + 0.01, -0.55, "lin"), (C2 - 0.01, -0.55), (C2 + 0.01, 0.55, "lin"), (34.1, 0.55),
          (34.45, -0.55, "expo"), (40.9, -0.55), (41.25, 0.0, "expo"), (DUR, 0.0)]
TINT = [(0.0, (1.12, 1.0, 0.9)), (2.85, (1.12, 1.0, 0.9)), (3.15, (1.0, 1.0, 1.0)), (4.6, (1.0, 1.0, 1.0)),
        (4.9, (1.12, 0.94, 1.0)), (C1 - 0.01, (1.12, 0.94, 1.0)), (C1 + 0.01, (1.05, 1.05, 0.92), "lin"),
        (C2 - 0.01, (1.05, 1.05, 0.92)), (C2 + 0.01, (0.9, 1.1, 0.95), "lin"), (34.1, (0.9, 1.1, 0.95)),
        (34.45, (0.88, 1.0, 1.12)), (40.9, (0.88, 1.0, 1.12)), (41.3, (1.0, 1.0, 1.0)), (DUR, (1.0, 1.0, 1.0))]
SEC = {"hook": -8, "vibe": -95, "brain": -185, "social": -274, "tip": -369, "tg": -460.5}
SHAKES = ()
FLASHES = ((41.3, 0.12),)
JUMPS = [31.2]
ALARM = None
SPOT = None

G = lambda *k: list(k)  # noqa: E731
O = OUTRO - 48.035      # жесты концовки — как в monet_mg, со сдвигом
ARM_L = G((0, 62), (0.4, -10, "back"), (2.7, -8), (2.95, 62), (3.15, -50, "back"), (4.3, -45), (4.6, 62, "back"),
          (C1 + 0.05, 62), (15.7, -30, "back"), (17.8, -28), (18.0, -40, "back"), (22.4, -38), (22.6, -28, "back"),
          (26.8, -25), (27.0, 62), (34.4, 62), (34.6, -30, "back"), (37.1, -28), (37.25, -45, "back"), (40.7, -42),
          (41.0, 62), (54.94 + O, 62), (55.09 + O, -50, "back"), (55.69 + O, -45), (56.09 + O, 62, "back"), (DUR, 62))
ARM_R = G((0, 62), (2.95, 62), (3.15, -50, "back"), (4.3, -45), (4.6, 62, "back"), (5.0, -30, "back"), (6.4, -28),
          (6.5, -48, "back"), (8.4, -45), (8.7, 35, "back"), (10.8, 38), (10.95, -30, "back"), (15.0, -28),
          (15.2, 62), (C2 + 0.05, 62), (27.5, -30, "back"), (31.0, -28), (31.15, -50, "back"), (32.1, -48),
          (32.3, -35, "back"), (34.1, -32), (34.3, 62),
          (47.79 + O, 62), (48.25 + O, -48, "back"), (50.6 + O, -45), (50.9 + O, -55, "back"), (53.6 + O, -52),
          (53.85 + O, -40, "back"), (54.8 + O, -42), (54.95 + O, -50, "back"), (55.69 + O, -45), (55.84 + O, -58),
          (55.99 + O, -32), (56.14 + O, -58), (56.29 + O, -32), (DUR, 62))
HEAD = G((0, 0), (0.5, 8), (1.6, -6), (2.6, 6), (3.2, 0), (4.9, -6), (6.5, 7), (8.0, -6), (9.8, 6), (10.9, -8),
         (12.4, 6), (13.3, -6), (14.2, 7), (15.6, -6), (16.8, 8), (18.0, -6), (19.4, 6), (20.7, -7), (22.6, 6),
         (24.3, -5), (25.4, 7), (27.0, 0), (27.4, 6), (28.6, -6), (30.2, 6), (31.1, -7), (32.2, 8), (34.4, -6),
         (35.9, 6), (37.2, -6), (39.0, 7), (40.7, 0), (41.5, -6), (48.4, 0), (DUR, 0))


# ---------------------------------------------------------------- картинки вставок
def phone_tex(n, buzz=False):
    """Телефон: n уведомлений на экране; buzz — дуги вибрации по бокам."""
    im = Image.new("RGBA", (520, 860), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([60, 20, 460, 840], 60, fill=(16, 16, 22, 255), outline=(150, 150, 165, 255), width=10)
    d.rounded_rectangle([90, 60, 430, 800], 40, fill=(30, 34, 52, 255))
    d.rounded_rectangle([210, 76, 310, 100], 12, fill=(16, 16, 22, 255))
    d.text((200, 130), "12:47", font=mg.font(64, tall=True), fill="white")
    cols = [TG_RGB, PINK_RGB, GREEN_RGB, GOLD_RGB, PURPLE_RGB, RED_RGB]
    for i in range(min(n, 6)):
        y = 240 + i * 88
        d.rounded_rectangle([108, y, 412, y + 76], 20, fill=(50, 54, 74, 255))
        d.rounded_rectangle([122, y + 14, 170, y + 62], 12, fill=cols[i % 6] + (255,))
        d.rounded_rectangle([186, y + 18, 380, y + 32], 7, fill=(170, 175, 195, 255))
        d.rounded_rectangle([186, y + 44, 330, y + 56], 6, fill=(110, 115, 135, 255))
    if buzz:
        for s in (-1, 1):
            cx = 260 + s * 230
            for r in (40, 70):
                d.arc([cx - r, 380 - r * 2, cx + r, 380 + r * 2], 90 - 50 if s > 0 else 270 - 50,
                      90 + 50 if s > 0 else 270 + 50, fill=GOLD_RGB + (255,), width=10)
    return mg.premul(im)


def counter_tex(v):
    im = rounded((640, 220), 34, (18, 18, 26, 255), GOLD_RGB + (255,), 5)
    d = ImageDraw.Draw(im)
    f = mg.font(110, tall=True)
    d.text((40, 40), f"{v}", font=f, fill=GOLD_RGB + (255,))
    d.text((40 + d.textbbox((0, 0), f"{v}", font=f)[2] + 24, 70), "РАЗ В ДЕНЬ", font=mg.font(60, tall=True), fill="white")
    return mg.premul(im)


def people_tex(n):
    im = rounded((780, 360), 34, (18, 18, 26, 255), (72, 72, 92, 255), 3)
    d = ImageDraw.Draw(im)
    d.text((40, 24), "ЧУВСТВУЮТ «ВИБРАЦИЮ»", font=mg.font(44, tall=True), fill="white")
    for i in range(10):
        r, c = divmod(i, 5)
        x, y = 70 + c * 135, 100 + r * 128
        col = PURPLE_RGB if i < n else (90, 90, 105)
        d.ellipse([x + 30, y, x + 70, y + 40], fill=col + (255,))
        d.rounded_rectangle([x + 16, y + 46, x + 84, y + 110], 22, fill=col + (255,))
    return mg.premul(im)


def brain_tex(a):
    """Мозг со значком «ожидания сигнала» (крутящееся колесо загрузки)."""
    im = Image.new("RGBA", (520, 460), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    pink = (240, 150, 170)
    for (x, y, r) in ((170, 170, 110), (300, 150, 120), (360, 260, 100), (200, 280, 110), (280, 300, 90)):
        d.ellipse([x - r, y - r, x + r, y + r], fill=pink + (255,))
    for k in range(5):
        d.arc([120 + k * 40, 120, 220 + k * 40, 260], 200, 330, fill=(200, 110, 130), width=8)
    cx, cy = 400, 380
    for k in range(10):
        ang = math.radians(a + k * 36)
        col = int(80 + 17 * k)
        d.ellipse([cx + 52 * math.cos(ang) - 10, cy + 52 * math.sin(ang) - 10, cx + 52 * math.cos(ang) + 10,
                   cy + 52 * math.sin(ang) + 10], fill=(col, col, min(255, col + 40)))
    return mg.premul(im)


def desk_label(text, col):
    return mg.premul(mg.text_rgba(text, 60, col, tall=True))


def month_tex(n):
    im = rounded((780, 300), 34, (18, 18, 26, 255), TG_RGB + (255,), 4)
    d = ImageDraw.Draw(im)
    d.text((40, 24), "30 ДНЕЙ БЕЗ СОЦСЕТЕЙ", font=mg.font(48, tall=True), fill="white")
    for i in range(30):
        r, c = divmod(i, 10)
        x, y = 40 + c * 70, 100 + r * 62
        on = i < n
        d.rounded_rectangle([x, y, x + 56, y + 50], 12, fill=(TG_RGB if on else (44, 44, 56)) + (255,))
        if on:
            d.line([(x + 14, y + 26), (x + 25, y + 37), (x + 44, y + 14)], fill="white", width=6)
    return mg.premul(im)


def mood_tex(v):
    """Шкала настроения от грустной рожицы к весёлой; v — положение бегунка 0..1."""
    im = rounded((780, 220), 34, (18, 18, 26, 255), GREEN_RGB + (255,), 4)
    d = ImageDraw.Draw(im)
    d.text((40, 20), "НАСТРОЕНИЕ", font=mg.font(44, tall=True), fill="white")
    d.rounded_rectangle([140, 120, 640, 150], 15, fill=(44, 44, 56))
    d.rounded_rectangle([140, 120, 140 + int(500 * v), 150], 15, fill=GREEN_RGB + (255,))
    for cx, happy in ((80, False), (700, True)):
        d.ellipse([cx - 40, 95, cx + 40, 175], fill=GOLD_RGB + (255,))
        d.ellipse([cx - 18, 118, cx - 8, 130], fill=(40, 30, 10))
        d.ellipse([cx + 8, 118, cx + 18, 130], fill=(40, 30, 10))
        d.arc([cx - 20, 128 if happy else 148, cx + 20, 160 if happy else 180], 20 if happy else 200,
              160 if happy else 340, fill=(40, 30, 10), width=5)
    x = 140 + int(500 * v)
    d.ellipse([x - 22, 113, x + 22, 157], fill="white")
    return mg.premul(im)


def drawer_tex(p):
    """Ящик стола, телефон опускается в него (p от 0 до 1)."""
    im = Image.new("RGBA", (560, 520), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    y = int(20 + 230 * p)
    d.rounded_rectangle([200, y, 360, y + 270], 26, fill=(16, 16, 22, 255), outline=(150, 150, 165, 255), width=6)
    d.rounded_rectangle([214, y + 18, 346, y + 252], 18, fill=(30, 34, 52, 255))
    d.rounded_rectangle([40, 260, 520, 500], 30, fill=(110, 78, 56, 255))
    d.rounded_rectangle([60, 280, 500, 300], 10, fill=(80, 56, 40, 255))
    d.rounded_rectangle([230, 380, 330, 404], 12, fill=(200, 180, 140, 255))
    return mg.premul(im)


# ---------------------------------------------------------------- сцена
class Scene(base.Scene):
    cfg = sys.modules[__name__]

    def __init__(self, voice):
        super().__init__(voice)
        T = self.T
        T["phantom"] = chip("ФАНТОМНАЯ ВИБРАЦИЯ", PURPLE_RGB)
        T["exp"] = card("ОПЫТ", "ТЕЛЕФОН НА СТОЛЕ", "даже выключенный мешает", RED_RGB)
        T["l_on"] = desk_label("НА СТОЛЕ", (255, 140, 140))
        T["l_off"] = desk_label("В ДРУГОЙ КОМНАТЕ", (150, 240, 180))
        T["hour"] = chip("+1 ЧАС В ДЕНЬ", GREEN_RGB)
        T["away"] = chip("УБЕРИ ПОДАЛЬШЕ", GREEN_RGB)
        T["when"] = chip("РАБОТА · СОН", TG_RGB)
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
        a = math.radians(yaw)
        return mg.Cam(yaw + 0.5 * wobble(t, 1), dist, hgt + 0.02 * wobble(t, 2), ty + 0.015 * wobble(t, 3), fov=fov,
                      roll=0.5 * wobble(t * 0.8, 4), tx=off * math.cos(a), tz=off * math.sin(a))

    def finish(self, frame, t):
        fr = super().finish(frame, t)
        for r0 in TURNS:  # «поворот телефона»: кадр поворачивается на 90° и уменьшается, под ним склейка
            p = (t - r0) / 0.4
            if 0 < p < 1:
                q = smooth(p / 0.5) if p < 0.5 else smooth((1 - p) / 0.5)   # 0 → 1 → 0
                ang = 90 * q * (1 if p < 0.5 else -1)
                sc = 1 - (1 - W / H) * q
                M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, sc)
                rot = cv2.warpAffine(fr, M, (W, H), borderValue=(20, 16, 24))
                fr = (rot.astype(np.float32) * (1 - 0.4 * q)).astype(np.uint8)
        return fr

    def items_for(self, t, cam, it):
        T = self.T
        P = lambda y0, *a, **k: place(y0, *a, **k)  # noqa: E731

        # 1. Крючок (персонаж справа, камера сверху): телефон лежит на полу, сыплются уведомления
        y0 = SEC["hook"]
        k = pop(t, 0.1, 2.95, 0.45, 0.25)
        if k > 0:
            n = int(clamp((t - 0.4) / 2.0) * 6 + 0.999) if t > 0.4 else 0
            q = P(y0, -0.8, 0.03, 0.0)
            self.add_panel(it, cam, q, face(y0, -8), 0.62 * k, 1.03 * k, 0.04 * k, self.cached("ph", phone_tex, n),
                           pitch=88, side=(40, 40, 50), radius=0.07)

        # 2. Заставка
        k = pop(t, 3.15, 4.6, 0.35, 0.3)
        if k > 0:
            self.add_panel(it, cam, place(0, 0.0, 1.95, 0.35), face(0, -6), 1.2 * k, 0.564 * k, 0.11 * k,
                           T["stinger"], pitch=-6, side=(30, 24, 120), edge=(200, 190, 255))

        # 3. 60 раз в день и вибрация (персонаж слева): телефон стоит → счётчик → дрожит → 9 из 10
        y0 = SEC["vibe"]
        r_, c_ = basis(y0)
        k = pop(t, 4.95, 12.45, 0.45, 0.25)
        if k > 0:
            buzz = 10.9 < t < 12.4
            sh = 0.015 * math.sin(t * 90) if buzz else 0.0
            pos = P(y0, 0.75, 1.45) + r_ * sh + [0, 0.02 * math.sin(t * 2), 0]
            self.add_bb(it, cam, pos, 0.85 * k, self.cached("ph2", phone_tex, 3 if t < 8.6 else 1, buzz),
                        anchor=(0.5, 0.5))
            kc = pop(t, 6.45, 8.65, 0.35, 0.25)
            if kc > 0:
                v = int(round(60 * smooth((t - 6.6) / 1.0)))
                self.add_bb(it, cam, P(y0, 0.75, 2.05), 0.16 * kc, self.cached("cnt", counter_tex, v), anchor=(0.5, 0.5))
            kp = pop(t, 10.95, 12.35, 0.3, 0.25)
            if kp > 0:
                self.add_bb(it, cam, P(y0, 0.75, 2.05), 0.1 * kp, T["phantom"], anchor=(0.5, 0.5))
        kq = pop(t, 12.45, C1, 0.4, 0.2)
        if kq > 0:
            n = int(clamp((t - 13.3) / 1.0) * 9 + 0.999) if t > 13.3 else 0
            self.add_panel(it, cam, P(y0, 0.72, 1.7), face(y0, 14), 0.9 * kq, 0.369 * kq, 0.05 * kq,
                           self.cached("ppl", people_tex, n), pitch=-6, side=(50, 40, 70), edge=(220, 200, 255))

        # 4. Мозг и опыт (персонаж справа): мозг ждёт сигнала → два стола, столбики результатов
        y0 = SEC["brain"]
        r_, c_ = basis(y0)
        k = pop(t, C1 + 0.05, 17.9, 0.4, 0.25)
        if k > 0:
            self.add_bb(it, cam, P(y0, -0.72, 1.6) + [0, 0.02 * math.sin(t * 2), 0], 0.55 * k,
                        self.cached("brain", brain_tex, int(t * 12) % 10 * 36), anchor=(0.5, 0.5))
        k = pop(t, 17.95, 22.6, 0.4, 0.25)
        if k > 0:
            self.add_panel(it, cam, P(y0, -0.72, 1.85) + [0, 0.02 * math.sin(t * 2), 0], face(y0, -14), 0.9 * k,
                           0.427 * k, 0.06 * k, T["exp"], pitch=-6, side=(80, 40, 40), edge=(255, 200, 200))
        k = pop(t, 22.65, AD_CUT + 0.3, 0.4, 0.2)
        if k > 0:
            for i, (sd, v, key, col) in enumerate(((-1.3, 0.55, "l_on", (60, 60, 170)),
                                                   (-0.75, 0.95, "l_off", (70, 150, 80)))):
                kb = pop(t, 22.8 + 2.4 * i, AD_CUT + 0.3, 0.45, 0.2)
                if kb <= 0:
                    continue
                q = P(y0, sd, 0.0, -0.7)
                hh = 1.1 * v * kb
                self.add_box(it, cam, (q[0], q[2]), (0.36, 0.3), hh, col, radius=0.08, edge=(220, 220, 230))
                self.add_bb(it, cam, q + [0, hh + 0.12, 0], 0.075 * kb, T[key], anchor=(0.5, 0.5))

        # 5. Месяц без соцсетей (персонаж слева): календарь → «+1 час» → шкала настроения
        y0 = SEC["social"]
        k = pop(t, C2 + 0.05, 34.15, 0.4, 0.25)
        if k > 0:
            n = int(clamp((t - 28.6) / 1.4) * 30)
            self.add_panel(it, cam, P(y0, 0.72, 2.0) + [0, 0.02 * math.sin(t * 2), 0], face(y0, 14), 0.9 * k,
                           0.346 * k, 0.05 * k, self.cached("mon", month_tex, n), pitch=-6, side=(30, 60, 80),
                           edge=(190, 230, 255))
            kh = pop(t, 31.1, 34.15, 0.35, 0.25)
            if kh > 0:
                self.add_bb(it, cam, P(y0, 0.72, 1.55), 0.12 * kh * (1 + 0.4 * (1 - clamp((t - 31.1) / 0.2))),
                            T["hour"], anchor=(0.5, 0.5))
            km = pop(t, 32.2, 34.15, 0.35, 0.25)
            if km > 0:
                v = round((0.3 + 0.6 * smooth((t - 32.5) / 1.0)) * 20) / 20
                self.add_panel(it, cam, P(y0, 0.72, 1.15), face(y0, 14), 0.8 * km, 0.226 * km, 0.04 * km,
                               self.cached("mood", mood_tex, v), pitch=-6, side=(30, 70, 40), edge=(190, 255, 210))

        # 6. Совет (персонаж справа): телефон уезжает в ящик → «убери подальше» → «работа · сон»
        y0 = SEC["tip"]
        k = pop(t, 34.5, 40.95, 0.4, 0.25)
        if k > 0:
            p = round(smooth((t - 37.4) / 1.0) * 20) / 20
            self.add_bb(it, cam, P(y0, -0.72, 1.3) + [0, 0.02 * math.sin(t * 2), 0], 0.6 * k,
                        self.cached("dr", drawer_tex, p), anchor=(0.5, 0.5))
            ka = pop(t, 37.3, 40.95, 0.35, 0.25)
            if ka > 0:
                self.add_bb(it, cam, P(y0, -0.72, 2.1), 0.12 * ka, T["away"], anchor=(0.5, 0.5))
            kw = pop(t, 39.0, 40.95, 0.35, 0.25)
            if kw > 0:
                self.add_bb(it, cam, P(y0, -0.72, 1.82), 0.1 * kw, T["when"], anchor=(0.5, 0.5))

        # 7. Концовка с Telegram — общий шаблон
        self.outro_items(t, cam, it, OUTRO, SEC["tg"])


SFX = [(t, "whoosh") for t in (0.0, 3.1, 4.8) + TURNS + (34.35, 41.15)]


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
        for t in (0.6, 2.0, 3.8, 5.6, 7.5, 11.5, 14.0, 15.45, 16.8, 19.5, 23.5, 26.0, 27.35, 29.5, 31.8, 33.5,
                  36.0, 38.5, 40.0, 45.0):
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
    base.mix_audio(tmp, voice, out, music=os.path.join(ROOT, "work", "nophone", "music.wav"), sfx_list=SFX, dur=DUR,
                   sfx_vol=0.15)  # «вжух» лёгкий и тихий
    os.remove(tmp)


if __name__ == "__main__":
    main()
