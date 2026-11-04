"""Ролик №20: «Что будет, если всегда говорить "да"?» (+ рекламный баннер в середине).

python tools/yes_mg.py work/yes/voice_clean.wav output/yes.mp4 [--preview] [--jobs 4]
затем: python tools/ad_banner.py insert output/yes.mp4 27.75 output/yes_ad.mp4 yes_mg
Общий движок, персонаж и концовка с Telegram — из memory_mg.py; параллельный рендер — как в money_mg.py.
Новое в монтаже: две объёмные кнопки «ДА» / «НЕТ», «ДА» нажимается снова и снова; камера начинает с пола и
поднимает взгляд; год из 365 клеточек заполняется галочками; хлопушка кино; смена сцены «переворотом» — кадр
сжимается в полоску, как переворачиваемая карточка, и раскрывается уже новым; три находки — друзья, работа,
любовь; свеча догорает («выгорает»); «я не могу» против «я такое не делаю» столбиками 39% / 64%.
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
OUTRO = float(open(os.path.join(ROOT, "work", "yes", "outro_start.txt")).read())  # 41.76
DUR = 50.4
AD_CUT = 27.75                   # здесь вклеивается баннер (камера стоит спокойно)
GOLD_RGB, GREEN_RGB, PINK_RGB, PURPLE_RGB = (245, 190, 50), (46, 196, 110), (240, 70, 110), (150, 90, 230)

# ---------------------------------------------------------------- субтитры
T_, Go_, G_, Pi_, Pu_ = TG_RGB, GOLD_RGB, GREEN_RGB, PINK_RGB, PURPLE_RGB
SUBS = [
    (0.153, "ЧТО БУДЕТ, ЕСЛИ ВСЕГДА", ()), (1.553, "ГОВОРИТЬ «ДА»?", ("«ДА»?",), G_), (2.6, "", ()),
    (4.123, "ЖУРНАЛИСТ ДЭННИ УОЛЛЕС", ("УОЛЛЕС",), T_), (5.563, "ПРОВЕРИЛ ЭТО НА СЕБЕ:", ()),
    (7.073, "ЦЕЛЫЙ ГОД ОН СОГЛАШАЛСЯ", ("ГОД",), Go_), (8.513, "НА ВСЁ, ЧТО ЕМУ ПРЕДЛАГАЛИ", ("ВСЁ,",), G_),
    (10.34, "ПО ЕГО ИСТОРИИ ДАЖЕ", ()), (11.353, "СНЯЛИ ФИЛЬМ", ("ФИЛЬМ",), Go_), (11.993, "С ДЖИМОМ КЕРРИ", ("КЕРРИ",), Go_),
    (13.25, "ОН НАШЁЛ НОВЫХ ДРУЗЕЙ,", ("ДРУЗЕЙ,",), G_), (14.782, "РАБОТУ И ДАЖЕ ЛЮБОВЬ", ("ЛЮБОВЬ",), Pi_),
    (16.502, "НО ИНОГДА СОГЛАШАЛСЯ", ()), (17.602, "И НА ГЛУПОСТИ", ("ГЛУПОСТИ",)),
    (18.69, "ПСИХОЛОГИ ГОВОРЯТ:", ()), (20.052, "КТО НЕ УМЕЕТ ОТКАЗЫВАТЬ,", ("ОТКАЗЫВАТЬ,",), Pu_),
    (21.692, "ЧАЩЕ УСТАЁТ И ВЫГОРАЕТ", ("ВЫГОРАЕТ",)), (23.62, "А СОГЛАШАЕМСЯ МЫ ЧАСТО", ()),
    (24.762, "ПРОСТО ПОТОМУ, ЧТО", ()), (26.082, "СКАЗАТЬ «НЕТ» НЕЛОВКО", ("«НЕТ»",)),
    (AD_CUT - 0.02, "", ()),
    (27.98, "КАК ОТКАЗЫВАТЬ ПРОЩЕ?", ("ПРОЩЕ?",), G_), (29.596, "В ОПЫТЕ УЧЁНЫХ ЛЮДИ,", ()),
    (30.956, "КОТОРЫЕ ГОВОРИЛИ", ()), (31.836, "«Я ТАКОЕ НЕ ДЕЛАЮ»,", ("«Я", "ТАКОЕ", "НЕ", "ДЕЛАЮ»,"), G_),
    (33.336, "А НЕ «Я НЕ МОГУ»,", ("«Я", "НЕ", "МОГУ»,")), (34.536, "ГОРАЗДО ЧАЩЕ ДЕРЖАЛИСЬ", ("ЧАЩЕ",), G_),
    (35.856, "СВОЕГО РЕШЕНИЯ", ()), (37.136, "ТАК ЧТО «ДА»", ("«ДА»",), G_), (37.616, "«ДА» ОТКРЫВАЕТ ДВЕРИ,", ("«ДА»",), G_),
    (39.506, "А «НЕТ» БЕРЕЖЁТ СИЛЫ", ("«НЕТ»",), Pu_),
    (41.721, "А ЕЩЁ БОЛЬШЕ ИНТЕРЕСНОГО —", (), T_), (43.081, "В МОЁМ ТЕЛЕГРАМ-КАНАЛЕ", ("ТЕЛЕГРАМ-КАНАЛЕ",), T_),
    (44.411, "«ПАУЧЬЯ НОРКА»", ("«ПАУЧЬЯ", "НОРКА»"), T_), (45.531, "ТАМ ЭКСКЛЮЗИВ,", ("ЭКСКЛЮЗИВ",), T_),
    (46.231, "КОТОРОГО НЕТ В РОЛИКАХ", (), T_), (47.531, "ССЫЛКА В ПРОФИЛЕ —", ("ПРОФИЛЕ",), T_),
    (48.551, "ЗАЛЕТАЙ!", ("ЗАЛЕТАЙ!",), T_),
]

# ---------------------------------------------------------------- камера и сцены
FLIPS = (12.95, 27.78)           # «переворот карточки»; склейка — на середине
C1, C2 = (f + 0.15 for f in FLIPS)
CAMERA = [
    (0.0, (-6.0, 3.6, 1.0, 0.15, 50.0)), (1.2, (-6.0, 3.6, 1.15, 0.84, 50.0), "inout"),   # взгляд с пола вверх
    (2.4, (-5.0, 3.6, 1.15, 0.84, 50.0)),
    (2.65, (0.0, 3.9, 1.3, 0.98, 50.0), "inout"),             # заставка
    (3.9, (-1.0, 3.9, 1.3, 0.98, 50.0)),
    (4.2, (88.0, 3.6, 1.25, 0.86, 50.0), "expo"),             # Дэнни Уоллес — год «да»
    (C1 - 0.01, (95.0, 3.4, 1.25, 0.86, 50.0), "lin"),
    (C1 + 0.01, (180.0, 3.7, 1.35, 0.88, 50.0), "lin"),       # находки и выгорание
    (20.9, (184.0, 3.7, 1.35, 0.88, 50.0), "lin"),
    (21.4, (210.0, 3.5, 1.3, 0.88, 50.0), "expo"),
    (C2 - 0.01, (214.0, 3.4, 1.3, 0.88, 50.0), "lin"),
    (C2 + 0.01, (300.0, 3.6, 0.9, 0.95, 50.0), "lin"),        # как отказывать — чуть снизу
    (36.9, (306.0, 3.5, 0.9, 0.95, 50.0), "lin"),
    (37.2, (372.0, 3.9, 1.4, 0.95, 50.0), "expo"),            # финал
    (41.3, (376.0, 3.6, 1.4, 0.95, 49.0), "lin"),
    (41.7, (460.0, 4.3, 1.3, 1.12, 50.0), "expo"), (DUR, (461.0, 4.0, 1.3, 1.15, 50.0)),
]
OFFSET = [(0.0, -0.5), (2.4, -0.5), (2.65, 0.0, "inout"), (3.9, 0.0), (4.2, 0.55, "expo"), (C1 - 0.01, 0.55),
          (C1 + 0.01, -0.55, "lin"), (C2 - 0.01, -0.55), (C2 + 0.01, 0.55, "lin"), (36.9, 0.55), (37.2, 0.0, "expo"),
          (DUR, 0.0)]
TINT = [(0.0, (0.95, 1.08, 0.95)), (2.4, (0.95, 1.08, 0.95)), (2.65, (1.0, 1.0, 1.0)), (3.9, (1.0, 1.0, 1.0)),
        (4.2, (0.9, 1.0, 1.12)), (C1 - 0.01, (0.9, 1.0, 1.12)), (C1 + 0.01, (1.05, 0.95, 1.1), "lin"),
        (20.9, (1.05, 0.95, 1.1)), (21.4, (0.88, 0.95, 1.18)), (C2 - 0.01, (0.88, 0.95, 1.18)),
        (C2 + 0.01, (1.1, 1.05, 0.9), "lin"), (36.9, (1.1, 1.05, 0.9)), (37.2, (1.06, 0.95, 1.06)),
        (41.3, (1.06, 0.95, 1.06)), (41.75, (1.0, 1.0, 1.0)), (DUR, (1.0, 1.0, 1.0))]
SEC = {"hook": -6, "danny": 91, "found": 182, "burn": 212, "how": 303, "final": 374, "tg": 460.5}
SHAKES = (17.6,)
FLASHES = ((41.75, 0.12),)
JUMPS = [13.3]
ALARM = None
SPOT = None

G = lambda *k: list(k)  # noqa: E731
O = OUTRO - 48.035      # жесты концовки — как в monet_mg, со сдвигом
ARM_L = G((0, 62), (0.4, -25, "back"), (2.3, -22), (2.5, 62), (2.7, -50, "back"), (3.6, -45), (3.9, 62, "back"),
          (13.0, 62), (13.3, -45, "back"), (14.0, -42), (14.2, -28, "back"), (16.3, -25), (16.5, 30, "back"),
          (18.4, 32), (18.7, -30, "back"), (21.0, -28), (21.4, -35, "back"), (25.9, -32), (26.1, 25, "back"),
          (27.3, 28), (27.55, 62), (37.1, 62), (37.3, -45, "back"), (39.3, -42), (39.5, 62),
          (54.94 + O, 62), (55.09 + O, -50, "back"), (55.69 + O, -45), (56.09 + O, 62, "back"), (DUR, 62))
ARM_R = G((0, 62), (2.5, 62), (2.7, -50, "back"), (3.6, -45), (3.9, 62, "back"), (4.3, -30, "back"), (7.0, -28),
          (7.1, -45, "back"), (10.2, -42), (10.4, -28, "back"), (12.7, -25), (12.9, 62), (13.3, -45, "back"),
          (14.0, -42), (14.2, 62), (26.1, 25, "back"), (27.3, 28), (27.55, 62), (C2 + 0.05, 62), (28.1, -30, "back"),
          (31.7, -28), (31.85, -45, "back"), (33.2, -42), (33.4, -20, "back"), (36.8, -18), (37.1, 62),
          (39.5, -45, "back"), (41.2, -42), (41.5, 62),
          (47.79 + O, 62), (48.25 + O, -48, "back"), (50.6 + O, -45), (50.9 + O, -55, "back"), (53.6 + O, -52),
          (53.85 + O, -40, "back"), (54.8 + O, -42), (54.95 + O, -50, "back"), (55.69 + O, -45), (55.84 + O, -58),
          (55.99 + O, -32), (56.14 + O, -58), (56.29 + O, -32), (DUR, 62))
HEAD = G((0, 0), (0.6, 7), (1.5, -6), (2.4, 6), (2.7, 0), (4.2, -6), (5.6, 6), (7.1, -6), (8.5, 7), (10.3, -5),
         (11.4, 6), (12.5, -6), (13.3, 8), (14.8, -6), (15.7, 7), (16.5, -6), (17.6, 9), (18.7, -5), (20.1, 6),
         (21.7, -8), (23.6, 6), (24.8, -5), (26.1, 8), (27.5, 0), (28.0, -6), (29.6, 6), (31.2, -6), (32.5, 7),
         (33.4, -7), (34.5, 6), (35.9, -5), (37.1, 7), (38.1, -6), (39.5, 6), (41.2, 0), (41.9, -6), (48.8, 0),
         (DUR, 0))


# ---------------------------------------------------------------- картинки вставок
def btn_label(text, col):
    return mg.premul(mg.text_rgba(text, 110, col, tall=True))


def year_tex(n):
    """Год «да»: 365 клеточек, n заполнено."""
    im = rounded((780, 500), 34, (18, 18, 26, 255), (72, 72, 92, 255), 3)
    d = ImageDraw.Draw(im)
    d.text((40, 26), "ГОД «ДА»", font=mg.font(56, tall=True), fill="white")
    txt = f"{n} / 365"
    f = mg.font(48, tall=True)
    tw = d.textbbox((0, 0), txt, font=f)[2]
    d.text((740 - tw, 32), txt, font=f, fill=GREEN_RGB + (255,))
    for i in range(365):
        r, c = divmod(i, 27)
        x, y = 40 + c * 26, 110 + r * 26
        d.rounded_rectangle([x, y, x + 20, y + 20], 5, fill=(GREEN_RGB if i < n else (44, 44, 56)) + (255,))
    return mg.premul(im)


def clapper_tex(ang):
    """Хлопушка кино; ang — угол открытой верхней планки."""
    im = Image.new("RGBA", (520, 460), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([20, 160, 500, 440], 20, fill=(24, 24, 30, 255), outline=(220, 220, 230, 255), width=6)
    d.text((50, 190), "ФИЛЬМ", font=mg.font(64, tall=True), fill="white")
    d.text((50, 280), "«Всегда говори", font=mg.font(42), fill=(200, 200, 215))
    d.text((50, 336), '"да"», 2008', font=mg.font(42), fill=(200, 200, 215))
    top = Image.new("RGBA", (480, 70), (0, 0, 0, 0))
    td = ImageDraw.Draw(top)
    td.rounded_rectangle([0, 0, 479, 69], 14, fill=(230, 230, 236, 255))
    for x in range(0, 480, 80):
        td.polygon([(x, 0), (x + 40, 0), (x + 70, 69), (x + 30, 69)], fill=(24, 24, 30, 255))
    top = top.rotate(ang, expand=True, center=(0, 69), resample=Image.BICUBIC)
    im.alpha_composite(top, (20, max(0, 150 - top.height + 70)))
    return mg.premul(im)


def icon_tex(kind, col):
    S = 260
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([0, 0, S - 1, S - 1], fill=(22, 22, 30, 255), outline=col + (255,), width=10)
    c = S // 2
    if kind == "friends":
        for dx in (-40, 40):
            d.ellipse([c + dx - 26, c - 60, c + dx + 26, c - 8], fill=col + (255,))
            d.pieslice([c + dx - 46, c, c + dx + 46, c + 92], 180, 360, fill=col + (255,))
    elif kind == "job":
        d.rounded_rectangle([c - 70, c - 30, c + 70, c + 60], 16, fill=col + (255,))
        d.rounded_rectangle([c - 28, c - 62, c + 28, c - 26], 10, outline=col + (255,), width=12)
    else:
        d.ellipse([c - 64, c - 50, c, c + 14], fill=col + (255,))
        d.ellipse([c, c - 50, c + 64, c + 14], fill=col + (255,))
        d.polygon([(c - 60, c - 4), (c + 60, c - 4), (c, c + 70)], fill=col + (255,))
    return mg.premul(im)


def candle_tex(h, t):
    """Свеча: h — оставшаяся высота (0..1), пламя дрожит."""
    im = Image.new("RGBA", (260, 560), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    top = 540 - int(400 * h)
    d.rounded_rectangle([70, top, 190, 540], 18, fill=(235, 225, 205, 255))
    d.ellipse([60, 520, 200, 556], fill=(200, 190, 170, 255))
    d.line([(130, top), (130, top - 20)], fill=(40, 40, 40), width=6)
    fl = 1 + 0.12 * math.sin(t * 17)
    w, hh = int(34 * h ** 0.3 * fl) + 6, int(80 * h ** 0.3 * fl) + 10
    d.ellipse([130 - w, top - 20 - hh, 130 + w, top - 14], fill=(255, 170, 50, 255))
    d.ellipse([130 - w // 2, top - 20 - hh // 2, 130 + w // 2, top - 16], fill=(255, 240, 160, 255))
    return mg.premul(im)


def bars_tex(k):
    """Опыт про отказы: «я не могу» — 39%, «я такое не делаю» — 64%."""
    im = rounded((820, 520), 34, (18, 18, 26, 255), (72, 72, 92, 255), 3)
    d = ImageDraw.Draw(im)
    d.text((40, 26), "УСТОЯЛИ ПЕРЕД СОБЛАЗНОМ", font=mg.font(44, tall=True), fill="white")
    for x, v, lab, col in ((90, 0.39, "«Я НЕ МОГУ»", (110, 110, 125)), (470, 0.64, "«Я ТАКОЕ\nНЕ ДЕЛАЮ»", GREEN_RGB)):
        h = int(330 * v * k)
        d.rounded_rectangle([x, 420 - h, x + 240, 420], 24, fill=col + (255,))
        if k > 0.3:
            f = mg.font(64, tall=True)
            txt = f"{int(round(100 * v * k))}%"
            tw = d.textbbox((0, 0), txt, font=f)[2]
            d.text((x + 120 - tw / 2, 420 - h - 76), txt, font=f, fill="white")
        f2 = mg.font(34, tall=True)
        for j, line in enumerate(lab.split("\n")):
            tw = d.textbbox((0, 0), line, font=f2)[2]
            d.text((x + 120 - tw / 2, 432 + j * 38 - (19 if "\n" in lab else 0)), line, font=f2, fill=(200, 200, 215))
    return mg.premul(im)


def final_tex():
    im = rounded((860, 330), 40, (18, 22, 20, 255), GREEN_RGB + (255,), 5)
    d = ImageDraw.Draw(im)
    f = mg.font(80, tall=True)
    for y, word, rest, col in ((36, "ДА", "— ДВЕРИ", GREEN_RGB), (180, "НЕТ", "— СИЛЫ", PURPLE_RGB)):
        l, t_, r, b = d.textbbox((0, 0), word, font=f)
        d.rounded_rectangle([36, y, 36 + r - l + 48, y + 112], 30, fill=col + (255,))
        d.text((60 - l, y + 56 - (b + t_) / 2), word, font=f, fill="white")
        d.text((36 + r - l + 72 - l, y + 56 - (b + t_) / 2), rest, font=f, fill="white")
    return mg.premul(im)


# ---------------------------------------------------------------- сцена
class Scene(base.Scene):
    cfg = sys.modules[__name__]

    def __init__(self, voice):
        super().__init__(voice)
        T = self.T
        T["l_yes"], T["l_no"] = btn_label("ДА", (255, 255, 255)), btn_label("НЕТ", (255, 255, 255))
        T["danny"] = card("ЖУРНАЛИСТ", "ДЭННИ УОЛЛЕС", "год говорил «да» на всё", TG_RGB)
        T["i_friends"], T["i_job"], T["i_love"] = (icon_tex(k, c) for k, c in (("friends", TG_RGB),
                                                                                ("job", GOLD_RGB), ("love", PINK_RGB)))
        T["l_friends"], T["l_job"], T["l_love"] = (btn_label(s, (220, 220, 235)) for s in ("ДРУЗЬЯ", "РАБОТА", "ЛЮБОВЬ"))
        T["silly"] = stamp("И ГЛУПОСТИ", RED_RGB)
        T["psy"] = chip("НЕ УМЕЕШЬ ОТКАЗЫВАТЬ = ВЫГОРАНИЕ", PURPLE_RGB)
        T["awk"] = chip("«НЕТ» — НЕЛОВКО…", RED_RGB)
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
        for f0 in FLIPS:  # «переворот карточки»: кадр сжимается по ширине в полоску и раскрывается новым
            p = (t - f0) / 0.3
            if 0 < p < 1:
                q = abs(2 * p - 1)                       # 1 → 0 (полоска, склейка) → 1
                w = max(4, int(W * smooth(q)))
                small = cv2.resize(fr, (w, H), interpolation=cv2.INTER_AREA)
                shade = 0.45 + 0.55 * q
                out = np.full_like(fr, (20, 16, 24))
                x0 = (W - w) // 2
                out[:, x0:x0 + w] = (small.astype(np.float32) * shade).astype(np.uint8)
                fr = out
        return fr

    def items_for(self, t, cam, it):
        T = self.T
        P = lambda y0, *a, **k: place(y0, *a, **k)  # noqa: E731

        # 1. Крючок (персонаж справа, взгляд с пола вверх): кнопки «ДА» / «НЕТ», «ДА» нажимается снова и снова
        y0 = SEC["hook"]
        k = pop(t, 0.1, 2.55, 0.45, 0.25)
        if k > 0:
            for sd, key, col, press in ((-1.0, "l_yes", (60, 170, 90), True), (-0.45, "l_no", (60, 50, 190), False)):
                q = P(y0, sd, 0.0, -0.1)
                hp = 0.06 if press and (t * 2.6) % 1 < 0.35 else 0.22
                self.add_box(it, cam, (q[0], q[2]), (0.42, 0.42), hp * k, col, radius=0.2, edge=(230, 240, 230))
                self.add_bb(it, cam, q + [0, hp * k + 0.25, 0], 0.18 * k, T[key], anchor=(0.5, 0.5))

        # 2. Заставка
        k = pop(t, 2.65, 3.9, 0.35, 0.3)
        if k > 0:
            self.add_panel(it, cam, place(0, 0.0, 1.95, 0.35), face(0, -6), 1.2 * k, 0.564 * k, 0.11 * k,
                           T["stinger"], pitch=-6, side=(30, 24, 120), edge=(200, 190, 255))

        # 3. Дэнни Уоллес (персонаж слева): карточка → год из 365 клеточек → хлопушка
        y0 = SEC["danny"]
        k = pop(t, 4.2, 7.0, 0.4, 0.25)
        if k > 0:
            pos = P(y0, 0.72, 1.6) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 14), 0.9 * k, 0.427 * k, 0.06 * k, T["danny"], pitch=-6,
                           side=(30, 60, 80), edge=(190, 230, 255))
        k = pop(t, 7.05, 10.3, 0.4, 0.25)
        if k > 0:
            n = int(365 * smooth((t - 7.3) / 2.4))
            pos = P(y0, 0.72, 1.6) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 14), 0.9 * k, 0.577 * k, 0.06 * k,
                           self.cached("yr", year_tex, n // 5 * 5), pitch=-6, side=(30, 70, 40), edge=(190, 255, 210))
        k = pop(t, 10.35, C1, 0.4, 0.2)
        if k > 0:
            ang = track([(10.35, 0.0), (11.2, 28.0, "out"), (11.33, 0.0, "lin"), (C1, 0.0)], t)
            self.add_bb(it, cam, P(y0, 0.72, 1.65) + [0, 0.02 * math.sin(t * 2), 0], 0.5 * k,
                        self.cached("clap", clapper_tex, round(ang / 2) * 2), anchor=(0.5, 0.5))

        # 4. Находки (персонаж справа): друзья, работа, любовь → «и глупости»
        y0 = SEC["found"]
        for i, (key, t0) in enumerate((("friends", 14.0), ("job", 14.75), ("love", 15.65))):
            kk = pop(t, t0, 20.9, 0.35, 0.25)
            if kk > 0:
                q = P(y0, -1.15 + 0.42 * i, 2.1)
                self.add_bb(it, cam, q, 0.3 * kk, T["i_" + key], anchor=(0.5, 0.5))
                self.add_bb(it, cam, q + [0, -0.24, 0], 0.065 * kk, T["l_" + key], anchor=(0.5, 0.5))
        if t > 17.6:
            k = pop(t, 17.6, 20.9, 0.15, 0.25)
            ks = clamp((t - 17.6) / 0.15)
            self.add_bb(it, cam, P(y0, -0.75, 1.5, 0.1), 0.16 * (1 + 0.6 * (1 - ks)) * k, T["silly"], anchor=(0.5, 0.5))

        # 5. Выгорание (персонаж справа, другой ракурс): свеча догорает → «нет — неловко»
        y0 = SEC["burn"]
        k = pop(t, 21.45, AD_CUT + 0.3, 0.4, 0.2)
        if k > 0:
            h = 1 - 0.6 * smooth((t - 21.7) / 2.5)
            self.add_bb(it, cam, P(y0, -0.75, 1.15), 0.62 * k, self.cached("cd", candle_tex, round(h * 30) / 30,
                                                                            round(t * 10) / 10), anchor=(0.5, 0.5))
            self.add_bb(it, cam, P(y0, -0.55, 2.0), 0.08 * k, T["psy"], anchor=(0.5, 0.5))
            ka = pop(t, 26.05, AD_CUT + 0.3, 0.35, 0.2)
            if ka > 0:
                self.add_bb(it, cam, P(y0, -0.6, 1.75), 0.11 * ka, T["awk"], anchor=(0.5, 0.5))

        # 6. Как отказывать (персонаж слева, чуть снизу): столбики «не могу» 39% / «не делаю» 64%
        y0 = SEC["how"]
        k = pop(t, 29.55, 37.0, 0.45, 0.25)
        if k > 0:
            kb = round(smooth((t - 34.5) / 1.0) * 20) / 20
            pos = P(y0, 0.72, 1.6) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 14), 0.95 * k, 0.602 * k, 0.06 * k, self.cached("bars", bars_tex, kb),
                           pitch=6, side=(30, 70, 40), edge=(190, 255, 210))

        # 7. Финал (персонаж в центре): «ДА — ДВЕРИ, НЕТ — СИЛЫ»
        y0 = SEC["final"]
        k = pop(t, 37.25, 41.35, 0.45, 0.25)
        if k > 0:
            pos = place(y0, 0.0, 1.9, 0.2) + [0, 0.02 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 0), 0.9 * k, 0.345 * k, 0.06 * k, T["final"], pitch=-6,
                           side=(30, 70, 40), edge=(190, 255, 210))

        # 8. Концовка с Telegram — общий шаблон
        self.outro_items(t, cam, it, OUTRO, SEC["tg"])


SFX = [(t, "whoosh") for t in (0.0, 2.6, 4.1) + FLIPS + (21.35, 37.15, 41.6)]


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
        for t in (0.5, 1.8, 3.2, 5.5, 8.5, 11.3, 12.5, 13.1, 15.0, 16.3, 18.0, 22.0, 24.5, 26.8, 27.95, 30.5,
                  35.5, 36.5, 39.0, 45.0):
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
    base.mix_audio(tmp, voice, out, music=os.path.join(ROOT, "work", "yes", "music.wav"), sfx_list=SFX, dur=DUR,
                   sfx_vol=0.15)  # «вжух» лёгкий и тихий
    os.remove(tmp)


if __name__ == "__main__":
    main()
