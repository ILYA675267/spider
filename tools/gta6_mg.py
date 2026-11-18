"""Ролик №23: «Что будет, когда выйдет GTA 6?» (+ рекламный баннер в середине).

python tools/gta6_mg.py work/gta6/voice_clean.wav output/gta6.mp4 [--preview] [--jobs 4]
затем: python tools/ad_banner.py insert output/gta6.mp4 34.15 output/gta6_ad.mp4 gta6_mg
Общий движок, персонаж и концовка с Telegram — из memory_mg.py; параллельный рендер — как в money_mg.py.
Новое в монтаже: неоновая «майамская» гамма Вайс-Сити (розовый и фиолетовый); настоящий логотип GTA VI со
свечением; календарь «19 НОЯБРЯ» и лента переносов с зачёркнутыми датами; смена сцены «перемоткой VHS» —
полосы, сдвиг цветов и дрожание кадра; счётчик просмотров и столбики «GTA VI против Дэдпула»; ретро-закат
с пальмами; карточки героев; ценник; логотипы PlayStation и Xbox, зачёркнутый «ПК».
Логотипы — в assets/logos (Wikimedia Commons). Звуки: только тихий «вжух» на смене ракурса.
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
OUTRO = float(open(os.path.join(ROOT, "work", "gta6", "outro_start.txt")).read())  # 51.39
DUR = 60.05
AD_CUT = 34.15                   # здесь вклеивается баннер (середина ролика с баннером ±5 с)
PINK_RGB, PURPLE_RGB, CYAN_RGB, GOLD_RGB, GREEN_RGB = (255, 70, 170), (150, 90, 230), (60, 220, 240), (245, 190, 50), (46, 196, 110)


def logo(name, height=None):
    im = Image.open(os.path.join(ROOT, "assets", "logos", name)).convert("RGBA")
    if height:
        im = im.resize((int(im.width * height / im.height), height), Image.LANCZOS)
    return mg.premul(im)


# ---------------------------------------------------------------- субтитры
T_, Pk_, Pu_, C_, Go_, G_ = TG_RGB, PINK_RGB, PURPLE_RGB, CYAN_RGB, GOLD_RGB, GREEN_RGB
SUBS = [
    (0.18, "ЧТО БУДЕТ, КОГДА", ()), (1.02, "ВЫЙДЕТ GTA 6?", ("GTA", "6?"), Pk_), (2.7, "", ()),
    (4.46, "ИГРУ, КОТОРУЮ ЖДАЛИ", ()), (5.7, "БОЛЬШЕ 10 ЛЕТ,", ("10", "ЛЕТ,"), Pk_),
    (7.08, "ВЫПУСКАЮТ 19 НОЯБРЯ", ("19", "НОЯБРЯ"), C_), (9.49, "ВЫХОД ПЕРЕНОСИЛИ ДВАЖДЫ,", ("ДВАЖДЫ,",)),
    (11.41, "И ТЕПЕРЬ ЭТО ГЛАВНАЯ", ()), (12.27, "ИГРА ГОДА", ("ИГРА", "ГОДА"), Pk_),
    (14.13, "ЕЁ ВТОРОЙ ТРЕЙЛЕР", ()), (15.321, "ЗА СУТКИ ПОСМОТРЕЛИ", ()),
    (16.421, "ОКОЛО 475 МИЛЛИОНОВ РАЗ", ("475",), C_), (19.581, "БОЛЬШЕ, ЧЕМ ТРЕЙЛЕР", ()),
    (20.821, "«ДЭДПУЛА И РОСОМАХИ»", ()), (22.6, "ЭТО РЕКОРД ДЛЯ ЛЮБОГО", ("РЕКОРД",), Go_),
    (23.551, "ВИДЕО В ИСТОРИИ", ()), (25.41, "ДЕЙСТВИЕ — В ВАЙС-СИТИ,", ("ВАЙС-СИТИ,",), Pk_),
    (27.172, "В ВЫМЫШЛЕННОМ ШТАТЕ", ()), (28.292, "ЛЕОНИДА", ("ЛЕОНИДА",), Pk_),
    (29.172, "ГЕРОЕВ ДВОЕ:", ()), (30.512, "ЛЮСИЯ И ДЖЕЙСОН", ("ЛЮСИЯ", "ДЖЕЙСОН"), C_),
    (32.132, "ПАРА В ДУХЕ", ()), (32.892, "БОННИ И КЛАЙДА", ("БОННИ", "КЛАЙДА"), Pu_),
    (AD_CUT - 0.02, "", ()),
    (34.342, "СТОИТ ИГРА 80 ДОЛЛАРОВ —", ("80",), Go_), (36.662, "ДОРОЖЕ ПОЧТИ ВСЕХ", ("ДОРОЖЕ",)),
    (37.862, "ИГР НА РЫНКЕ", ()), (39.15, "НО ЕСТЬ НЮАНС:", ("НЮАНС:",), Pk_),
    (40.301, "НА СТАРТЕ ОНА ВЫЙДЕТ", ()), (41.361, "ТОЛЬКО НА PLAYSTATION 5", ("PLAYSTATION", "5"), T_),
    (42.521, "И XBOX", ("XBOX",), G_), (43.701, "ВЕРСИЮ ДЛЯ КОМПЬЮТЕРА", ("КОМПЬЮТЕРА",)),
    (44.861, "ЖДУТ НЕ РАНЬШЕ", ()), (45.501, "СЛЕДУЮЩЕГО ГОДА", ()),
    (46.981, "А ТЫ БУДЕШЬ ИГРАТЬ", ()), (47.941, "В ПЕРВЫЙ ДЕНЬ", ("ПЕРВЫЙ", "ДЕНЬ"), Pk_),
    (48.541, "ИЛИ ПОДОЖДЁШЬ?", ("ПОДОЖДЁШЬ?",), C_), (49.69, "ПИШИ В КОММЕНТАРИЯХ", ("КОММЕНТАРИЯХ",), Pk_),
    (51.354, "А ЕЩЁ БОЛЬШЕ ИНТЕРЕСНОГО —", (), T_), (52.714, "В МОЁМ ТЕЛЕГРАМ-КАНАЛЕ", ("ТЕЛЕГРАМ-КАНАЛЕ",), T_),
    (54.044, "«ПАУЧЬЯ НОРКА»", ("«ПАУЧЬЯ", "НОРКА»"), T_), (55.164, "ТАМ ЭКСКЛЮЗИВ,", ("ЭКСКЛЮЗИВ",), T_),
    (55.864, "КОТОРОГО НЕТ В РОЛИКАХ", (), T_), (57.164, "ССЫЛКА В ПРОФИЛЕ —", ("ПРОФИЛЕ",), T_),
    (58.184, "ЗАЛЕТАЙ!", ("ЗАЛЕТАЙ!",), T_),
]

# ---------------------------------------------------------------- камера и сцены
VHS = (13.8, 25.2, 34.2)         # «перемотка VHS»; склейка — на середине
C1, C2, C3 = (v + 0.15 for v in VHS)
CAMERA = [
    (0.0, (-14.0, 3.0, 0.6, 1.3, 50.0)), (2.4, (-6.0, 3.7, 1.2, 0.9, 50.0), "out"),   # снизу вверх к логотипу
    (2.55, (-5.0, 3.7, 1.2, 0.9, 50.0)),
    (2.8, (0.0, 3.9, 1.3, 0.98, 50.0), "inout"),              # заставка
    (4.2, (-1.0, 3.9, 1.3, 0.98, 50.0)),
    (4.5, (88.0, 3.6, 1.25, 0.86, 50.0), "expo"),             # дата выхода и переносы
    (C1 - 0.01, (95.0, 3.4, 1.25, 0.86, 50.0), "lin"),
    (C1 + 0.01, (180.0, 3.8, 1.5, 0.9, 50.0), "lin"),         # трейлер-рекорд
    (C2 - 0.01, (187.0, 3.6, 1.5, 0.9, 50.0), "lin"),
    (C2 + 0.01, (262.0, 3.6, 0.8, 1.0, 50.0), "lin"),         # Вайс-Сити — снизу, закат
    (C3 - 0.01, (262.0, 3.6, 0.8, 1.0, 50.0)),                # (камера стоит — склейка под баннер)
    (C3 + 0.01, (350.0, 3.6, 1.25, 0.86, 50.0), "lin"),       # цена и платформы
    (51.0, (358.0, 3.4, 1.25, 0.86, 49.0), "lin"),
    (51.35, (460.0, 4.3, 1.3, 1.12, 50.0), "expo"), (DUR, (461.0, 4.0, 1.3, 1.15, 50.0)),
]
OFFSET = [(0.0, -0.5), (2.55, -0.5), (2.8, 0.0, "inout"), (4.2, 0.0), (4.5, 0.55, "expo"), (C1 - 0.01, 0.55),
          (C1 + 0.01, -0.55, "lin"), (C2 - 0.01, -0.55), (C2 + 0.01, 0.55, "lin"), (C3 - 0.01, 0.55),
          (C3 + 0.01, -0.55, "lin"), (51.0, -0.55), (51.35, 0.0, "expo"), (DUR, 0.0)]
# неоновые цвета сцен (B, G, R)
TINT = [(0.0, (1.18, 0.9, 1.15)), (2.55, (1.18, 0.9, 1.15)), (2.8, (1.0, 1.0, 1.0)), (4.2, (1.0, 1.0, 1.0)),
        (4.5, (1.2, 1.0, 0.85)), (C1 - 0.01, (1.2, 1.0, 0.85)), (C1 + 0.01, (1.15, 0.88, 1.12), "lin"),
        (C2 - 0.01, (1.15, 0.88, 1.12)), (C2 + 0.01, (0.95, 0.85, 1.25), "lin"), (C3 - 0.01, (0.95, 0.85, 1.25)),
        (C3 + 0.01, (1.18, 0.95, 0.95), "lin"), (51.0, (1.18, 0.95, 0.95)), (51.4, (1.0, 1.0, 1.0)),
        (DUR, (1.0, 1.0, 1.0))]
SEC = {"hook": -6, "date": 91, "trailer": 183, "vice": 262, "price": 354, "tg": 460.5}
SHAKES = (22.6,)
FLASHES = ((1.6, 0.12), (51.4, 0.12))
JUMPS = [7.8]
ALARM = None
SPOT = None

G = lambda *k: list(k)  # noqa: E731
O = OUTRO - 48.035      # жесты концовки — как в monet_mg, со сдвигом
ARM_L = G((0, 62), (0.3, -30, "back"), (2.4, -28), (2.6, 62), (2.8, -50, "back"), (3.9, -45), (4.2, 62, "back"),
          (C1 + 0.05, 62), (14.3, -30, "back"), (16.3, -28), (16.4, -50, "back"), (19.4, -48), (19.6, -30, "back"),
          (22.5, -28), (22.6, -55, "back"), (25.0, -52), (25.2, 62), (C3 + 0.05, 62), (34.5, -30, "back"),
          (39.0, -28), (39.2, 30, "back"), (40.2, 32), (40.4, -40, "back"), (46.8, -38), (47.0, -25, "back"),
          (51.0, -22), (51.3, 62),
          (54.94 + O, 62), (55.09 + O, -50, "back"), (55.69 + O, -45), (56.09 + O, 62, "back"), (DUR, 62))
ARM_R = G((0, 62), (2.6, 62), (2.8, -50, "back"), (3.9, -45), (4.2, 62, "back"), (4.6, -30, "back"), (7.0, -28),
          (7.1, -50, "back"), (9.3, -48), (9.5, -30, "back"), (13.5, -28), (13.7, 62), (C2 + 0.05, 62),
          (25.6, -30, "back"), (29.0, -28), (29.2, -45, "back"), (31.9, -42), (32.1, -30, "back"), (33.8, -28),
          (34.0, 62), (39.2, 30, "back"), (40.2, 32), (40.4, 62),
          (47.79 + O, 62), (48.25 + O, -48, "back"), (50.6 + O, -45), (50.9 + O, -55, "back"), (53.6 + O, -52),
          (53.85 + O, -40, "back"), (54.8 + O, -42), (54.95 + O, -50, "back"), (55.69 + O, -45), (55.84 + O, -58),
          (55.99 + O, -32), (56.14 + O, -58), (56.29 + O, -32), (DUR, 62))
HEAD = G((0, 0), (0.6, 7), (1.6, -8), (2.6, 6), (2.8, 0), (4.5, -6), (5.7, 6), (7.1, -7), (8.3, 6), (9.5, -6),
         (10.5, 7), (12.3, -6), (13.6, 5), (14.2, -6), (15.8, 6), (16.7, -8), (18.0, 7), (19.6, -6), (21.0, 7),
         (22.6, 9), (23.6, -5), (25.0, 0), (25.4, 6), (27.2, -6), (28.3, 7), (29.2, -6), (30.5, 6), (31.2, -6),
         (32.9, 7), (34.0, 0), (34.4, -6), (35.6, 8), (37.0, -6), (39.2, 9), (40.3, -5), (41.7, 6), (43.7, -6),
         (45.2, 6), (47.0, -6), (48.5, 8), (49.7, -5), (51.0, 0), (51.6, -6), (58.5, 0), (DUR, 0))


# ---------------------------------------------------------------- картинки вставок
def calendar_tex():
    im = rounded((420, 460), 34, (240, 236, 228, 255), (180, 175, 165, 255), 3)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, 419, 120], 34, fill=PINK_RGB + (255,))
    d.rectangle([0, 70, 419, 120], fill=PINK_RGB + (255,))
    f = mg.font(60, tall=True)
    tw = d.textbbox((0, 0), "НОЯБРЬ", font=f)[2]
    d.text((210 - tw / 2, 28), "НОЯБРЬ", font=f, fill="white")
    f2 = mg.font(230, tall=True)
    l, t_, r, b = d.textbbox((0, 0), "19", font=f2)
    d.text((210 - (r + l) / 2, 290 - (b + t_) / 2), "19", font=f2, fill=(30, 30, 40))
    d.text((150, 400), "2026", font=mg.font(44, tall=True), fill=(120, 120, 130))
    return mg.premul(im)


def delays_tex(n):
    """Лента переносов: 2025 → 26 мая 2026 → 19 ноября 2026; n — сколько дат уже зачёркнуто."""
    im = rounded((820, 380), 34, (18, 14, 26, 255), PURPLE_RGB + (255,), 4)
    d = ImageDraw.Draw(im)
    d.text((40, 26), "ДАТА ВЫХОДА", font=mg.font(52, tall=True), fill="white")
    dates = ("ОСЕНЬ 2025", "26 МАЯ 2026", "19 НОЯБРЯ 2026")
    for i, s in enumerate(dates):
        y = 110 + i * 86
        col = (120, 110, 140) if i < n else CYAN_RGB
        f = mg.font(56, tall=True)
        d.text((60, y), s, font=f, fill=col + (255,))
        if i < n:
            tw = d.textbbox((0, 0), s, font=f)[2]
            d.line([(50, y + 36), (70 + tw, y + 30)], fill=RED_RGB + (255,), width=10)
        elif i == 2:
            d.ellipse([700, y + 8, 750, y + 58], fill=GREEN_RGB + (255,))
            d.line([(712, y + 34), (722, y + 45), (740, y + 22)], fill="white", width=7)
    return mg.premul(im)


def views_tex(v):
    im = rounded((820, 300), 34, (16, 14, 22, 255), CYAN_RGB + (255,), 5)
    d = ImageDraw.Draw(im)
    d.text((40, 24), "ТРЕЙЛЕР 2 · ЗА СУТКИ", font=mg.font(44, tall=True), fill=(190, 220, 230))
    txt = f"{v} МЛН"
    f = mg.font(150, tall=True)
    tw = d.textbbox((0, 0), txt, font=f)[2]
    d.text((410 - tw / 2, 90), txt, font=f, fill=CYAN_RGB + (255,))
    return mg.premul(im)


def bar_label(text, col, size=60):
    return mg.premul(mg.text_rgba(text, size, col, tall=True))


def sunset_tex():
    """Ретро-закат: полосатое солнце и пальмы-силуэты."""
    S = 700
    im = Image.new("RGBA", (S, 560), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for y in range(40, 240):
        c = (255, int(210 - (y - 40) * 0.7), int(90 + (y - 40) * 0.6), 255)
        half = math.sqrt(max(0, 200 ** 2 - (240 - y) ** 2))
        if (y // 12) % 3 == 2 and y > 140:
            continue
        d.line([(350 - half, y), (350 + half, y)], fill=c)
    for px, h, lean in ((90, 400, -1), (600, 360, 1), (520, 300, 1)):
        top = (px + lean * 30, 540 - h)
        d.line([(px, 540), top], fill=(30, 10, 40), width=16)
        for a in range(6):
            ang = math.radians(200 + a * 28)
            d.line([top, (top[0] + 110 * math.cos(ang), top[1] + 60 * math.sin(ang) + 20)], fill=(30, 10, 40), width=14)
    return mg.premul(im)


def hero_tex(name, col):
    im = rounded((380, 460), 34, (20, 14, 28, 255), col + (255,), 6)
    d = ImageDraw.Draw(im)
    d.ellipse([130, 60, 250, 180], fill=col + (255,))
    d.pieslice([70, 190, 310, 430], 180, 360, fill=col + (255,))
    f = mg.font(64, tall=True)
    tw = d.textbbox((0, 0), name, font=f)[2]
    d.rounded_rectangle([190 - tw / 2 - 24, 330, 190 + tw / 2 + 24, 420], 30, fill=(20, 14, 28, 255),
                        outline=col + (255,), width=4)
    d.text((190 - tw / 2, 340), name, font=f, fill="white")
    return mg.premul(im)


def price_tex():
    im = Image.new("RGBA", (520, 280), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.polygon([(0, 140), (90, 0), (519, 0), (519, 279), (90, 279)], fill=(26, 18, 32, 255))
    d.line([(0, 140), (90, 0), (519, 0), (519, 279), (90, 279), (0, 140)], fill=GOLD_RGB + (255,), width=8)
    d.ellipse([44, 122, 80, 158], fill=(80, 70, 96))
    d.text((130, 14), "$80", font=mg.font(150, tall=True), fill=GOLD_RGB + (255,))
    d.text((134, 222), "Ultimate — $100", font=mg.font(34), fill=(190, 180, 200))
    return mg.premul(im)


def pc_tex():
    im = rounded((460, 260), 34, (18, 18, 26, 255), (110, 110, 125, 255), 4)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([60, 40, 240, 160], 14, outline=(170, 170, 185), width=10)
    d.rectangle([130, 160, 170, 196], fill=(170, 170, 185))
    d.rounded_rectangle([90, 196, 210, 214], 8, fill=(170, 170, 185))
    d.text((270, 60), "ПК", font=mg.font(90, tall=True), fill="white")
    d.text((270, 170), "2027?", font=mg.font(52, tall=True), fill=GOLD_RGB + (255,))
    d.line([(40, 230), (420, 30)], fill=RED_RGB + (255,), width=14)
    return mg.premul(im)


# ---------------------------------------------------------------- сцена
class Scene(base.Scene):
    cfg = sys.modules[__name__]

    def __init__(self, voice):
        super().__init__(voice)
        T = self.T
        T["gta"], T["rockstar"] = logo("gta6.png", 600), logo("rockstar.png", 300)
        T["ps"], T["xbox"] = logo("playstation.png", 300), logo("xbox_series.png", 300)
        T["cal"] = calendar_tex()
        T["tomorrow"] = chip("10+ ЛЕТ ОЖИДАНИЯ", PINK_RGB)
        T["goty"] = chip("ГЛАВНАЯ ИГРА ГОДА", PINK_RGB)
        T["l_gta"], T["l_dp"] = bar_label("GTA VI", PINK_RGB), bar_label("ДЭДПУЛ И РОСОМАХА", (190, 190, 205), 46)
        T["n_gta"], T["n_dp"] = bar_label("475 млн", (255, 255, 255)), bar_label("365 млн", (200, 200, 210))
        T["record"] = stamp("РЕКОРД", GOLD_RGB)
        T["sunset"] = sunset_tex()
        T["leonida"] = card("ШТАТ", "ЛЕОНИДА", "город — Вайс-Сити", PINK_RGB)
        T["lucia"], T["jason"] = hero_tex("ЛЮСИЯ", PINK_RGB), hero_tex("ДЖЕЙСОН", CYAN_RGB)
        T["bonnie"] = chip("КАК БОННИ И КЛАЙД", PURPLE_RGB)
        T["price"] = price_tex()
        T["pc"] = pc_tex()
        T["ask"] = chip("ПЕРВЫЙ ДЕНЬ ИЛИ ЖДЁШЬ ПК?", PINK_RGB)
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
        shake = 0.0
        for s0 in SHAKES:
            if s0 < t < s0 + 0.35:
                shake = (1 - (t - s0) / 0.35) * math.sin((t - s0) * 70)
        return mg.Cam(yaw + 0.5 * wobble(t, 1) + 1.2 * shake, dist, hgt + 0.02 * wobble(t, 2) + 0.03 * shake,
                      ty + 0.015 * wobble(t, 3), fov=fov, roll=0.5 * wobble(t * 0.8, 4) + 1.5 * shake,
                      tx=off * math.cos(a), tz=off * math.sin(a))

    def finish(self, frame, t):
        fr = super().finish(frame, t)
        for v0 in VHS:  # «перемотка VHS»: полосы, сдвиг цветов и дрожание; под ними склейка
            p = (t - v0) / 0.3
            if 0 < p < 1:
                k = 1 - abs(2 * p - 1)
                rng = np.random.default_rng(int(t * 1000))
                out = fr.copy()
                sh = int(40 * k)
                out[..., 2] = np.roll(fr[..., 2], sh, axis=1)
                out[..., 0] = np.roll(fr[..., 0], -sh, axis=1)
                for _ in range(int(14 * k) + 1):
                    y = int(rng.integers(0, H - 60))
                    h = int(rng.integers(8, 60))
                    out[y:y + h] = np.roll(out[y:y + h], int(rng.integers(-160, 160)), axis=1)
                band = int((p * 2.5 % 1) * H)
                out[band:band + 40] = np.clip(out[band:band + 40].astype(np.int16) + 90, 0, 255).astype(np.uint8)
                fr = (out.astype(np.float32) * (1 - 0.35 * k)).astype(np.uint8)
        return fr

    def items_for(self, t, cam, it):
        T = self.T
        P = lambda y0, *a, **k: place(y0, *a, **k)  # noqa: E731

        # 1. Крючок (персонаж справа, камера снизу вверх): логотип GTA VI со свечением
        y0 = SEC["hook"]
        k = pop(t, 0.15, 2.7, 0.5, 0.25)
        if k > 0:
            pul = 1 + 0.03 * math.sin(t * 6)
            self.add_bb(it, cam, P(y0, -0.6, 1.8) + [0, 0.02 * math.sin(t * 2), 0], 0.68 * k * pul, T["gta"],
                        anchor=(0.5, 0.5))

        # 2. Заставка
        k = pop(t, 2.8, 4.2, 0.35, 0.3)
        if k > 0:
            self.add_panel(it, cam, place(0, 0.0, 1.95, 0.35), face(0, -6), 1.2 * k, 0.564 * k, 0.11 * k,
                           T["stinger"], pitch=-6, side=(30, 24, 120), edge=(200, 190, 255))

        # 3. Дата выхода (персонаж слева): «10+ лет» → календарь 19 ноября → лента переносов → «главная игра года»
        y0 = SEC["date"]
        k = pop(t, 4.5, 9.4, 0.4, 0.25)
        if k > 0:
            self.add_bb(it, cam, P(y0, 0.72, 2.1), 0.11 * k, T["tomorrow"], anchor=(0.5, 0.5))
            kc = pop(t, 7.0, 9.4, 0.4, 0.25)
            if kc > 0:
                self.add_panel(it, cam, P(y0, 0.72, 1.45) + [0, 0.02 * math.sin(t * 2), 0], face(y0, 12), 0.48 * kc,
                               0.526 * kc, 0.05 * kc, T["cal"], side=(150, 145, 135))
        k = pop(t, 9.45, C1, 0.4, 0.2)
        if k > 0:
            n = 0 if t < 10.0 else 1 if t < 10.7 else 2
            pos = P(y0, 0.72, 1.6) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 14), 0.9 * k, 0.417 * k, 0.06 * k, self.cached("dl", delays_tex, n),
                           pitch=-6, side=(50, 30, 70), edge=(220, 200, 255))
            kg = pop(t, 12.2, C1, 0.35, 0.2)
            if kg > 0:
                self.add_bb(it, cam, P(y0, 0.72, 1.12), 0.11 * kg, T["goty"], anchor=(0.5, 0.5))

        # 4. Трейлер (персонаж справа): счётчик просмотров → столбики GTA VI против Дэдпула → «РЕКОРД»
        y0 = SEC["trailer"]
        k = pop(t, C1 + 0.05, 19.5, 0.4, 0.25)
        if k > 0:
            v = int(round(475 * smooth((t - 16.4) / 1.6)))
            pos = P(y0, -0.72, 1.7) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, -14), 0.9 * k, 0.329 * k, 0.06 * k, self.cached("vw", views_tex, v),
                           pitch=-6, side=(30, 60, 70), edge=(190, 240, 255))
            self.add_bb(it, cam, P(y0, -0.72, 2.25), 0.24 * k, T["rockstar"], anchor=(0.5, 0.5))
        k = pop(t, 19.55, C2, 0.4, 0.2)
        if k > 0:
            for i, (sd, v, lab, num, col) in enumerate(((-1.2, 0.77, "l_dp", "n_dp", (110, 110, 125)),
                                                         (-0.7, 1.0, "l_gta", "n_gta", (170, 70, 255)))):
                kb = pop(t, 19.7 + 0.5 * i, C2, 0.5, 0.2)
                q = P(y0, sd, 0.0, -0.6)
                hh = 1.5 * v * kb
                self.add_box(it, cam, (q[0], q[2]), (0.34, 0.3), hh, col, radius=0.08, edge=(240, 220, 255),
                             glow=(255, 80, 200) if i == 1 else None)
                if kb > 0.05:
                    self.add_bb(it, cam, q + [0, hh + 0.12, 0], 0.08 * kb, T[num], anchor=(0.5, 0.5))
                    self.add_bb(it, cam, q + basis(y0)[1] * 0.5, 0.075 * kb, T[lab], anchor=(0.5, 1.0))
            if t > 22.6:
                ks = clamp((t - 22.6) / 0.15)
                self.add_bb(it, cam, P(y0, -0.9, 2.15, 0.1), 0.18 * (1 + 0.6 * (1 - ks)) * k, T["record"],
                            anchor=(0.5, 0.5))

        # 5. Вайс-Сити (персонаж слева, снизу): ретро-закат → штат Леонида → Люсия и Джейсон → Бонни и Клайд
        y0 = SEC["vice"]
        k = pop(t, C2 + 0.05, AD_CUT + 0.3, 0.5, 0.2)
        if k > 0:
            self.add_bb(it, cam, P(y0, 0.75, 1.3, -1.6), 1.2 * k, T["sunset"], anchor=(0.5, 0.5))
            kl = pop(t, 25.5, 29.1, 0.4, 0.25)
            if kl > 0:
                self.add_panel(it, cam, P(y0, 0.72, 1.85), face(y0, 14), 0.9 * kl, 0.427 * kl, 0.06 * kl, T["leonida"],
                               pitch=8, side=(70, 30, 60), edge=(255, 200, 240))
            for i, (key, t0, sd) in enumerate((("lucia", 30.4, 0.5), ("jason", 31.1, 1.0))):
                kh = pop(t, t0, AD_CUT + 0.3, 0.4, 0.2)
                if kh > 0:
                    self.add_panel(it, cam, P(y0, sd, 1.75), face(y0, 10 + 6 * i), 0.4 * kh, 0.484 * kh, 0.04 * kh,
                                   T[key], pitch=8, side=(50, 30, 60))
            kb = pop(t, 32.6, AD_CUT + 0.3, 0.35, 0.2)
            if kb > 0:
                self.add_bb(it, cam, P(y0, 0.75, 2.3), 0.1 * kb, T["bonnie"], anchor=(0.5, 0.5))

        # 6. Цена и платформы (персонаж справа): ценник → логотипы PlayStation и Xbox → зачёркнутый ПК → вопрос
        y0 = SEC["price"]
        k = pop(t, C3 + 0.05, 39.05, 0.4, 0.25)
        if k > 0:
            self.add_bb(it, cam, P(y0, -0.72, 1.7) + [0, 0.02 * math.sin(t * 2), 0], 0.38 * k, T["price"],
                        anchor=(0.5, 0.5))
        k = pop(t, 40.3, 46.9, 0.4, 0.25)
        if k > 0:
            self.add_bb(it, cam, P(y0, -0.95, 1.85), 0.26 * k, T["ps"], anchor=(0.5, 0.5))
            kx = pop(t, 42.5, 46.9, 0.35, 0.25)
            if kx > 0:
                self.add_bb(it, cam, P(y0, -0.42, 1.85), 0.2 * kx, T["xbox"], anchor=(0.5, 0.5))
            kp = pop(t, 43.7, 46.9, 0.35, 0.25)
            if kp > 0:
                self.add_panel(it, cam, P(y0, -0.72, 1.3), face(y0, -12), 0.55 * kp, 0.311 * kp, 0.04 * kp, T["pc"],
                               side=(50, 50, 60))
        k = pop(t, 46.95, 51.2, 0.4, 0.25)
        if k > 0:
            self.add_bb(it, cam, P(y0, -0.72, 1.95), 0.36 * k, T["gta"], anchor=(0.5, 0.5))
            self.add_bb(it, cam, P(y0, -0.72, 1.4), 0.1 * k, T["ask"], anchor=(0.5, 0.5))

        # 7. Концовка с Telegram — общий шаблон
        self.outro_items(t, cam, it, OUTRO, SEC["tg"])


SFX = [(t, "whoosh") for t in (0.0, 2.75, 4.4) + VHS + (51.2,)]


def render(voice, out, frames):
    sc = Scene(voice)
    wr = mg.Writer(out, FPS)
    for n, i in enumerate(frames):
        t = i / FPS
        speed = abs(sc.cam(t + 0.5 / FPS).yaw - sc.cam(t - 0.5 / FPS).yaw) * FPS
        cut = any(abs(t - c) < 1.0 / FPS for c in (C1, C2, C3))
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
        for t in (0.6, 1.8, 3.6, 5.5, 8.2, 10.9, 12.8, 13.95, 17.5, 21.0, 23.0, 25.35, 27.0, 30.0, 33.5, 35.5,
                  41.0, 43.0, 45.0, 48.0, 50.5, 55.0):
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
    base.mix_audio(tmp, voice, out, music=os.path.join(ROOT, "work", "gta6", "music.wav"), sfx_list=SFX, dur=DUR,
                   sfx_vol=0.15)  # «вжух» лёгкий и тихий
    os.remove(tmp)


if __name__ == "__main__":
    main()
