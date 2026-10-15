"""Ролик №10: «Почему товару с оценкой 4,8 мы верим больше, чем идеальной пятёрке?».

python tools/reviews_mg.py work/reviews/voice_clean.wav output/reviews.mp4 [--preview] [--jobs 4]
Общий движок, персонаж и концовка с Telegram — из memory_mg.py; параллельный рендер — как в money_mg.py.
Новое в монтаже: смена сцен «жалюзи» (кадр закрывается и открывается полосками); звёзды кружат вокруг заставки;
3D-график из столбиков на полу «как часто покупают при разной оценке» (пик золотом, у 5,0 — проседает);
лупа над подозрительной пятёркой; лента отзывов, где отзыв на три звезды выезжает вперёд.
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
OUTRO = float(open(os.path.join(ROOT, "work", "reviews", "outro_start.txt")).read())  # 36.81
DUR = 45.5
GOLD_RGB, GREEN_RGB, VIOLET_RGB = (245, 190, 50), (46, 196, 110), (150, 90, 230)
GOLD, RED, GRAYB = GOLD_RGB[::-1], RED_RGB[::-1], (110, 104, 100)

# ---------------------------------------------------------------- субтитры
T_, Go_, V_, G_ = TG_RGB, GOLD_RGB, VIOLET_RGB, GREEN_RGB
SUBS = [
    (0.228, "ПОЧЕМУ ТОВАРУ С ОЦЕНКОЙ 4,8", ("4,8",), Go_), (2.188, "МЫ ВЕРИМ БОЛЬШЕ,", ("ВЕРИМ",)),
    (3.048, "ЧЕМ ИДЕАЛЬНОЙ ПЯТЁРКЕ?", ("ПЯТЁРКЕ?",), Go_), (4.3, "", ()),
    (5.908, "УЧЁНЫЕ ИЗ СЕВЕРО-ЗАПАДНОГО", ()), (7.328, "УНИВЕРСИТЕТА В США", ("США",), BLUE_RGB),
    (8.288, "ИЗУЧИЛИ ПОКУПКИ В ИНТЕРНЕТЕ", ("ПОКУПКИ",)), (10.308, "ЧАЩЕ ВСЕГО ЛЮДИ ПОКУПАЛИ", ("ЧАЩЕ",)),
    (11.628, "ТОВАРЫ С ОЦЕНКОЙ", ()), (12.488, "ОТ 4 ДО 4,7", ("4", "4,7"), Go_),
    (14.569, "А КОГДА ОЦЕНКА", ()), (15.409, "ПОДБИРАЛАСЬ К ПЯТИ,", ("ПЯТИ,",), Go_),
    (16.549, "ПОКУПАЛИ УЖЕ РЕЖЕ", ("РЕЖЕ",)), (17.849, "МОЗГ ДУМАЕТ:", ()), (18.849, "ТАК НЕ БЫВАЕТ", ("НЕ", "БЫВАЕТ")),
    (20.009, "ИДЕАЛЬНАЯ ПЯТЁРКА", ()), (21.049, "КАЖЕТСЯ НАКРУТКОЙ,", ("НАКРУТКОЙ,",)),
    (22.229, "А ПАРА ЧЕСТНЫХ МИНУСОВ", ("МИНУСОВ",), V_), (23.409, "НАОБОРОТ, ВЫЗЫВАЕТ ДОВЕРИЕ", ("ДОВЕРИЕ",), G_),
    (25.493, "КАК ЭТИМ ПОЛЬЗОВАТЬСЯ?", ("ПОЛЬЗОВАТЬСЯ?",), G_), (26.793, "НЕ СМОТРИ ТОЛЬКО НА ЗВЁЗДЫ", ("ЗВЁЗДЫ",), Go_),
    (28.493, "ОТКРОЙ ОТЗЫВЫ", ("ОТЗЫВЫ",)), (29.213, "НА ТРИ ЗВЕЗДЫ", ("ТРИ",), Go_),
    (30.273, "ТАМ ОБЫЧНО ПИШУТ ЧЕСТНО,", ("ЧЕСТНО,",), G_), (31.633, "БЕЗ ВОСТОРГОВ И БЕЗ ЗЛОСТИ", ()),
    (33.433, "И СРАЗУ ВИДНО,", ()), (34.413, "ПОДХОДИТ ЛИ ТОВАР", ()), (35.393, "ИМЕННО ТЕБЕ", ("ТЕБЕ",), G_),
    (36.78, "А ЕЩЁ БОЛЬШЕ ИНТЕРЕСНОГО —", (), T_), (38.14, "В МОЁМ ТЕЛЕГРАМ-КАНАЛЕ", ("ТЕЛЕГРАМ-КАНАЛЕ",), T_),
    (39.47, "«ПАУЧЬЯ НОРКА»", ("«ПАУЧЬЯ", "НОРКА»"), T_), (40.59, "ТАМ ЭКСКЛЮЗИВ,", ("ЭКСКЛЮЗИВ",), T_),
    (41.29, "КОТОРОГО НЕТ В РОЛИКАХ", (), T_), (42.59, "ССЫЛКА В ПРОФИЛЕ —", ("ПРОФИЛЕ",), T_),
    (43.61, "ЗАЛЕТАЙ!", ("ЗАЛЕТАЙ!",), T_),
]

# ---------------------------------------------------------------- камера и сцены
BLINDS = (4.15, 25.2)               # «жалюзи»: кадр закрывается полосками, под ними склейка, открывается
C1, C2 = (b + 0.25 for b in BLINDS)
CAMERA = [
    (0.0, (-14.0, 4.2, 1.4, 0.88, 50.0)), (0.9, (-8.0, 3.6, 1.15, 0.84, 50.0), "out"),
    (C1 - 0.01, (-5.0, 3.6, 1.15, 0.84, 50.0)),
    (C1 + 0.01, (0.0, 3.9, 1.3, 0.98, 50.0), "lin"),       # заставка со звёздами
    (5.5, (2.0, 3.9, 1.3, 0.98, 50.0)),
    (5.9, (95.0, 3.6, 1.2, 0.84, 50.0), "expo"),           # исследование и 3D-график
    (17.3, (110.0, 3.9, 1.6, 0.72, 50.0), "lin"),
    (17.75, (190.0, 3.6, 1.2, 0.86, 50.0), "expo"),        # подозрительная пятёрка
    (C2 - 0.01, (196.0, 3.5, 1.2, 0.86, 50.0)),
    (C2 + 0.01, (285.0, 3.6, 1.2, 0.84, 50.0), "lin"),     # как пользоваться
    (36.4, (298.0, 3.7, 1.25, 0.86, 50.0), "lin"),
    (36.8, (380.0, 4.3, 1.3, 1.12, 50.0), "expo"), (DUR, (381.0, 4.0, 1.3, 1.15, 50.0)),
]
OFFSET = [(0.0, -0.5), (C1 - 0.01, -0.5), (C1 + 0.01, 0.0, "lin"), (5.5, 0.0), (5.9, 0.55, "expo"), (17.3, 0.55),
          (17.75, -0.55, "expo"), (C2 - 0.01, -0.55), (C2 + 0.01, 0.55, "lin"), (36.4, 0.55), (36.8, 0.0, "expo"),
          (DUR, 0.0)]
TINT = [(0.0, (1.14, 0.9, 1.06)), (C1 - 0.01, (1.14, 0.9, 1.06)), (C1 + 0.01, (1.0, 1.0, 1.0), "lin"),
        (5.5, (1.0, 1.0, 1.0)), (6.0, (0.9, 1.02, 1.12)), (17.4, (0.9, 1.02, 1.12)), (17.9, (1.04, 0.92, 1.1)),
        (C2 - 0.01, (1.04, 0.92, 1.1)), (C2 + 0.01, (0.92, 1.08, 0.96), "lin"), (36.5, (0.92, 1.08, 0.96)),
        (36.9, (1.0, 1.0, 1.0)), (DUR, (1.0, 1.0, 1.0))]
SEC = {"hook": -8, "study": 100, "sus": 193, "tips": 290, "tg": 380.5}
SHAKES = ()
FLASHES = ((5.85, 0.08), (36.85, 0.12))
JUMPS = []
ALARM = None
SPOT = None

G = lambda *k: list(k)  # noqa: E731
ARM_L = G((0, 62), (1.3, 62), (1.5, -25, "back"), (2.9, -22), (3.1, -10, "back"), (4.0, -8), (C1 + 0.05, 62),
          (4.6, -50, "back"), (5.4, -45), (5.75, 62, "back"), (17.9, 62), (18.2, 36, "back"), (19.5, 38),
          (19.8, 62), (20.0, -25, "back"), (21.9, -22), (22.2, -12, "back"), (24.9, -10), (25.15, 62),
          (43.721, 62), (43.871, -50, "back"), (44.471, -45), (44.871, 62, "back"), (DUR, 62))
ARM_R = G((0, 62), (C1 + 0.05, 62), (4.6, -50, "back"), (5.4, -45), (5.75, 62, "back"), (6.0, -30, "back"),
          (9.9, -28), (10.3, 20, "back"), (13.9, 22), (14.6, 25, "back"), (17.2, 26), (17.5, 62), (18.2, 36, "back"),
          (19.5, 38), (19.8, 62), (C2 + 0.05, 62), (25.6, -30, "back"), (28.2, -28), (28.5, -15, "back"),
          (33.2, -12), (33.4, -35, "back"), (35.9, -33), (36.571, 62), (37.031, -48, "back"), (39.381, -45),
          (39.681, -55, "back"), (42.381, -52), (42.631, -40, "back"), (43.581, -42), (43.731, -50, "back"),
          (44.471, -45), (44.621, -58), (44.771, -32), (44.921, -58), (45.071, -32), (DUR, 62))
HEAD = G((0, 0), (1.5, -6), (2.4, 7), (3.1, -6), (4.3, 0), (6.0, -6), (8.3, -5), (10.3, 7), (12.5, 8), (14.6, 4),
         (16.5, -6), (18.2, 9), (19.5, 6), (20.0, -6), (22.2, -5), (23.4, 7), (25.6, -7), (28.5, -5), (30.3, 6),
         (33.4, -6), (35.6, 6), (37.03, -6), (43.7, 0), (DUR, 0))


# ---------------------------------------------------------------- картинки вставок
def star_poly(cx, cy, r):
    pts = []
    for k in range(10):
        a = -math.pi / 2 + k * math.pi / 5
        rr = r if k % 2 == 0 else r * 0.45
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return pts


def stars_row(d, x, y, r, value, gap=None):
    gap = gap or r * 2.3
    for i in range(5):
        cx = x + r + i * gap
        f = clamp(value - i)
        d.polygon(star_poly(cx, y, r), fill=(60, 60, 74, 255))
        if f > 0:
            # частично закрашенная звезда — рисуем золотую и обрезаем по ширине
            sub = Image.new("RGBA", (int(2 * r + 2), int(2 * r + 2)), (0, 0, 0, 0))
            ImageDraw.Draw(sub).polygon(star_poly(r + 1, r + 1, r), fill=GOLD_RGB + (255,))
            sub = sub.crop((0, 0, max(1, int((2 * r + 2) * f)), sub.height))
            d._image.paste(sub, (int(cx - r - 1), int(y - r - 1)), sub)


def header(d, text, col, dark=False):
    f = mg.font(36)
    tw = d.textbbox((0, 0), text, font=f)[2]
    d.rounded_rectangle([36, 30, 36 + tw + 44, 92], 30, fill=col + (255,))
    d.text((58, 38), text, font=f, fill=(30, 24, 10) if dark else "white")


def product_tex(rating, reviews, good):
    """Карточка товара в магазине: картинка, название, звёзды, оценка, число отзывов; good — галочка «верю»."""
    im = rounded((720, 300), 34, (20, 18, 28, 255), (90, 80, 110, 255), 4)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([30, 30, 240, 270], 26, fill=(44, 40, 58, 255))
    d.ellipse([75, 80, 195, 200], outline=(150, 140, 180, 255), width=12)  # «наушники»
    d.rounded_rectangle([60, 150, 95, 215], 12, fill=(150, 140, 180, 255))
    d.rounded_rectangle([175, 150, 210, 215], 12, fill=(150, 140, 180, 255))
    d.text((270, 36), "Наушники", font=mg.font(46), fill="white")
    stars_row(d, 270, 140, 22, rating, gap=46)
    d.text((540, 112), f"{rating:.1f}".replace(".", ","), font=mg.font(56, tall=True), fill=GOLD_RGB + (255,))
    d.text((270, 196), reviews, font=mg.font(36), fill=(160, 160, 180))
    if good:
        d.ellipse([620, 200, 690, 270], fill=GREEN_RGB + (255,))
        d.line([(637, 236), (652, 251), (676, 220)], fill="white", width=9)
    return mg.premul(im)


def star_tex(size=160):
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(im).polygon(star_poly(size / 2, size / 2 + 6, size / 2 - 4), fill=GOLD_RGB + (255,))
    return mg.premul(im)


def label_tex(text, col):
    return base.label(text, col, 46)


def magnifier_tex():
    im = Image.new("RGBA", (300, 300), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([20, 20, 200, 200], fill=(200, 220, 255, 60), outline=(230, 230, 240, 255), width=16)
    d.line([(180, 180), (280, 280)], fill=(200, 160, 90, 255), width=34)
    return mg.premul(im)


def pros_tex(trust):
    im = rounded((760, 470), 34, (18, 18, 26, 255), (72, 72, 92, 255), 3)
    d = ImageDraw.Draw(im)
    header(d, "ОТЗЫВЫ", VIOLET_RGB)
    rows = [("+", "звук отличный", GREEN_RGB), ("+", "быстро доставили", GREEN_RGB), ("−", "коробка помята", RED_RGB)]
    for i, (s, txt, col) in enumerate(rows):
        y = 120 + i * 70
        d.ellipse([40, y, 90, y + 50], fill=col + (255,))
        d.text((54 if s == "+" else 57, y - 4), s, font=mg.font(46), fill="white")
        d.text((110, y + 2), txt, font=mg.font(40), fill=(220, 220, 232))
    d.text((40, 340), "доверие", font=mg.font(36), fill=(200, 200, 215))
    d.rounded_rectangle([210, 345, 720, 385], 20, fill=(40, 40, 52, 255))
    if trust > 0.03:
        d.rounded_rectangle([210, 345, 210 + int(510 * trust), 385], 20, fill=GREEN_RGB + (255,))
    return mg.premul(im)


def review_tex(stars, text, tag, tag_col, hot):
    im = rounded((760, 210), 30, (34, 30, 20, 255) if hot else (20, 20, 28, 255),
                 GOLD_RGB + (255,) if hot else (72, 72, 92, 255), 5 if hot else 3)
    d = ImageDraw.Draw(im)
    stars_row(d, 30, 50, 22, stars, gap=50)
    f = mg.font(36)
    tw = d.textbbox((0, 0), tag, font=f)[2]
    d.rounded_rectangle([740 - tw - 50, 22, 740, 80], 28, fill=tag_col + (255,))
    d.text((740 - tw - 26, 28), tag, font=f, fill="white")
    d.text((30, 104), text, font=mg.font(38), fill=(230, 230, 240) if hot or stars != 3 else (230, 230, 240))
    return mg.premul(im)


# ---------------------------------------------------------------- сцена
BARS = [(3.5, 0.22), (3.8, 0.32), (4.0, 0.5), (4.2, 0.64), (4.4, 0.74), (4.6, 0.78), (4.7, 0.74), (4.9, 0.6),
        (5.0, 0.48)]


class Scene(base.Scene):
    cfg = sys.modules[__name__]

    def __init__(self, voice):
        super().__init__(voice)
        T = self.T
        T["p48"], T["p48ok"] = product_tex(4.8, "2 341 отзыв", False), product_tex(4.8, "2 341 отзыв", True)
        T["p50"] = product_tex(5.0, "12 отзывов", False)
        T["star"] = star_tex()
        T["nwu"] = card("ИССЛЕДОВАНИЕ", "СЕВЕРО-ЗАПАДНЫЙ УН-Т", "США · покупки в интернете", BLUE_RGB)
        T["often"] = chip("ПОКУПАЮТ ЧАЩЕ", GOLD_RGB)
        T["rare"] = chip("У 5,0 — РЕЖЕ", RED_RGB)
        T["labels"] = [label_tex(f"{r:.1f}".replace(".", ","), (220, 210, 190)) for r, _ in BARS]
        T["nope"] = stamp("ТАК НЕ БЫВАЕТ", GRAY_RGB)
        T["mag"] = magnifier_tex()
        T["fake"] = stamp("НАКРУТКА?", RED_RGB)
        T["how"] = chip("КАК ПОЛЬЗОВАТЬСЯ?", GREEN_RGB)
        T["notstars"] = card("СОВЕТ", "НЕ ТОЛЬКО ЗВЁЗДЫ", "читай, что пишут люди", GREEN_RGB, h=300)
        T["r5"] = review_tex(5, "Лучшее в жизни!!! Всем брать!!!", "ВОСТОРГ", (120, 110, 140), False)
        T["r3"], T["r3h"] = (review_tex(3, "Звук хороший, но через час давят", "ЧЕСТНО", GREEN_RGB, h) for h in (False, True))
        T["r1"] = review_tex(1, "Ужас!!! Не покупайте никогда!!!", "ЗЛОСТЬ", (120, 110, 140), False)
        T["fit"] = card("ИТОГ", "ПОДХОДИТ ЛИ ТЕБЕ?", "решай по честным отзывам", GREEN_RGB)
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
        n = 12
        hb = H / n
        for b0 in BLINDS:  # «жалюзи»: полоски закрываются сверху вниз с задержкой, потом открываются
            if b0 < t < b0 + 0.5:
                p = (t - b0) / 0.5
                for i in range(n):
                    d = i * 0.012
                    q = smooth((p - d) / 0.48 * 2) if p < 0.5 else 1 - smooth((p - 0.5 - d) / 0.48 * 2)
                    hh = int(hb * q)
                    if hh <= 0:
                        continue
                    y0 = int(i * hb)
                    fr[y0:y0 + hh] = (fr[y0:y0 + hh] * 0.04).astype(np.uint8)
                    if hh < hb - 2:
                        fr[y0 + hh:y0 + hh + 3] = GOLD
        return fr

    def items_for(self, t, cam, it):
        T = self.T
        P = lambda y0, *a, **k: place(y0, *a, **k)  # noqa: E731

        # 1. Крючок (персонаж справа): 4,8 с тысячами отзывов против «идеальной» 5,0
        y0 = SEC["hook"]
        k = pop(t, 1.4, C1, 0.35, 0.2)
        if k > 0:
            tex = T["p48ok"] if t > 2.45 else T["p48"]
            pos = P(y0, -0.74, 1.8) + [0, 0.02 * math.sin(t * 3), 0]
            self.add_panel(it, cam, pos, face(y0, -12), 0.72 * k, 0.3 * k, 0.05 * k, tex, pitch=-5,
                           side=(60, 50, 70), edge=(230, 210, 255))
        k = pop(t, 3.15, C1, 0.35, 0.2)
        if k > 0:
            pos = P(y0, -0.74, 1.36) + [0, 0.02 * math.sin(t * 3 + 1), 0]
            self.add_panel(it, cam, pos, face(y0, -12), 0.72 * k, 0.3 * k, 0.05 * k, T["p50"], pitch=-5,
                           side=(60, 50, 70), edge=(230, 210, 255))

        # 2. Заставка: вокруг карточки кружат звёзды
        k = pop(t, C1 + 0.05, 5.75, 0.35, 0.3)
        if k > 0:
            c = place(0, 0.0, 1.95, 0.35)
            self.add_panel(it, cam, c, face(0, -6), 1.2 * k, 0.564 * k, 0.11 * k, T["stinger"], pitch=-6,
                           side=(30, 24, 120), edge=(200, 190, 255))
            r_, c_ = basis(0)
            for i in range(6):
                a = t * 2.2 + i * math.pi / 3
                pos = c + r_ * 0.95 * math.cos(a) * k + np.array([0, 0.5 * math.sin(a), 0]) * k + c_ * (0.3 * math.sin(a) + 0.1)
                self.add_bb(it, cam, pos, 0.16 * k, T["star"], anchor=(0.5, 0.5))

        # 3. Исследование (персонаж слева): карточка → 3D-график на полу (пик золотом, у 5,0 — проседает)
        y0 = SEC["study"]
        r_, c_ = basis(y0)
        k = pop(t, 6.0, 10.2, 0.4, 0.3)
        if k > 0:
            pos = P(y0, 0.72, 1.6) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 14), 0.9 * k, 0.48 * k, 0.06 * k, T["nwu"], pitch=-6,
                           side=(40, 50, 90), edge=(200, 210, 255))
        kb = pop(t, 10.2, 17.6, 0.5, 0.3)
        if kb > 0:
            for i, (r, h) in enumerate(BARS):
                g = smooth((t - 10.4 - 0.08 * i) / 0.7)
                if r >= 4.9:
                    h *= 1 - 0.45 * smooth((t - 15.9) / 0.6)
                hh = max(0.01, h * g * kb)
                pk = 4.0 <= r <= 4.7 and t > 12.5
                col = GOLD if pk else (RED if r >= 4.9 and t > 15.9 else GRAYB)
                p = r_ * (0.42 + i * 0.12) + c_ * -0.55
                self.add_box(it, cam, (p[0], p[2]), (0.085, 0.085), hh, col, radius=0.03)
                self.add_bb(it, cam, p + [0, hh + 0.02, 0], 0.045 * g * kb, T["labels"][i], anchor=(0.5, 1.0))
            kc = pop(t, 12.5, 15.6, 0.3, 0.25)
            if kc > 0:
                self.add_bb(it, cam, P(y0, 0.85, 1.75), 0.12 * kc, T["often"], anchor=(0.5, 0.5))
            kr = pop(t, 16.1, 17.6, 0.3, 0.25)
            if kr > 0:
                self.add_bb(it, cam, P(y0, 0.95, 1.6), 0.12 * kr, T["rare"], anchor=(0.5, 0.5))

        # 4. Подозрительная пятёрка (персонаж справа): «так не бывает», лупа, «накрутка?» → честные минусы, доверие
        y0 = SEC["sus"]
        r_, c_ = basis(y0)
        k = pop(t, 18.85, 19.9, 0.15, 0.25)
        if k > 0:
            ks = clamp((t - 18.85) / 0.15)
            self.add_bb(it, cam, P(y0, -0.72, 1.9, 0.1), 0.15 * (1 + 0.6 * (1 - ks)) * k, T["nope"], anchor=(0.5, 0.5))
        k = pop(t, 19.9, 22.3, 0.4, 0.3)
        if k > 0:
            pos = P(y0, -0.74, 1.62) + [0, 0.02 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, -12), 0.72 * k, 0.3 * k, 0.05 * k, T["p50"], pitch=-5,
                           side=(60, 50, 70), edge=(230, 210, 255))
            sweep = math.sin((t - 20.2) * 2.6)
            self.add_bb(it, cam, pos + r_ * 0.25 * sweep + c_ * 0.12 + [0, -0.02, 0], 0.26 * k, T["mag"], anchor=(0.35, 0.35))
            if t > 21.3:
                ks = clamp((t - 21.3) / 0.15)
                self.add_bb(it, cam, pos + c_ * 0.15 + [0, -0.35, 0], 0.14 * (1 + 0.6 * (1 - ks)) * k, T["fake"],
                            anchor=(0.5, 0.5))
        k = pop(t, 22.25, C2, 0.4, 0.2)
        if k > 0:
            tr = round(smooth((t - 23.6) / 1.0) * 16) / 16
            pos = P(y0, -0.72, 1.55) + [0, 0.02 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, -12), 0.9 * k, 0.556 * k, 0.06 * k, self.cached("pros", pros_tex, tr),
                           pitch=-5, side=(60, 40, 90), edge=(220, 200, 255))

        # 5. Как пользоваться (персонаж слева): «не только звёзды» → лента отзывов, тройка выезжает вперёд → итог
        y0 = SEC["tips"]
        r_, c_ = basis(y0)
        k = pop(t, C2 + 0.05, 26.7, 0.3, 0.25)
        if k > 0:
            self.add_bb(it, cam, P(y0, 0.78, 1.95), 0.13 * k, T["how"], anchor=(0.5, 0.5))
        k = pop(t, 26.7, 28.5, 0.4, 0.3)
        if k > 0:
            pos = P(y0, 0.72, 1.6) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 12), 0.8 * k, 0.31 * k, 0.05 * k, T["notstars"], pitch=-6,
                           side=(40, 70, 50), edge=(200, 255, 220))
        if 28.45 < t < 33.5:
            out = smooth((t - 29.4) / 0.5)
            for i, (key, hy) in enumerate((("r5", 2.06), ("r3", 1.74), ("r1", 1.42))):
                k = pop(t, 28.45 + 0.1 * i, 33.5, 0.35, 0.3)
                if k <= 0:
                    continue
                hot = key == "r3"
                dim = 1 - (0.5 * smooth((t - 31.6) / 0.4) if not hot else 0)
                tex = T["r3h"] if hot and t > 29.4 else T[key]
                s = 1 + (0.12 * out if hot else 0)
                pos = P(y0, 0.72 + (-0.04 * out if hot else 0), hy, 0.1 * out if hot else 0)
                self.add_panel(it, cam, pos, face(y0, 12), 0.78 * k * s, 0.216 * k * s, 0.04, tex * dim, pitch=-5,
                               side=(70, 60, 30) if hot else (40, 40, 52), radius=0.04)
        k = pop(t, 33.45, 36.6, 0.4, 0.3)
        if k > 0:
            pos = P(y0, 0.72, 1.6) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 12), 0.9 * k, 0.48 * k, 0.06 * k, T["fit"], pitch=-6,
                           side=(40, 70, 50), edge=(200, 255, 220))

        # 6. Концовка с Telegram — общий шаблон
        self.outro_items(t, cam, it, OUTRO, SEC["tg"])


SFX = [(t, "whoosh") for t in (0.0, BLINDS[0], 5.5, 17.3, BLINDS[1], 36.4)]


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
        for t in (1.0, 2.6, 3.8, 4.3, 5.0, 7.5, 11.5, 13.2, 16.8, 19.2, 20.6, 21.8, 23.9, 25.3, 26.0, 27.6, 29.0,
                  30.0, 32.0, 34.5, 38.0, 44.0):
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
    base.mix_audio(tmp, voice, out, music=os.path.join(ROOT, "work", "reviews", "music.wav"), sfx_list=SFX, dur=DUR,
                   sfx_vol=0.15)  # «вжух» лёгкий и тихий
    os.remove(tmp)


if __name__ == "__main__":
    main()
