"""Ролик №22: «Что будет, если каждый день откладывать по 100 рублей?» (+ рекламный баннер в середине).

python tools/save_mg.py work/save/voice_clean.wav output/save.mp4 [--preview] [--jobs 4]
затем: python tools/ad_banner.py insert output/save.mp4 27.1 output/save_ad.mp4 save_mg
Общий движок, персонаж и концовка с Telegram — из memory_mg.py; параллельный рендер — как в money_mg.py.
Новое в монтаже: копилка, в которую падают монеты «100 ₽»; камера входит в сцену по спирали; три столбика из
монет — месяц, год, 10 лет, и над ними суммы; смена сцены «сдвигом» — старый кадр уезжает влево, новый въезжает
справа; стаканчик кофе и коробка доставки с ценниками, монетки бегут в банку, и она наполняется; вклад
добавляет сверху; шестерёнка «автоматики»; опыт Талера столбиками 3,5% → 13,6%; экран телефона «автоперевод».
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
OUTRO = float(open(os.path.join(ROOT, "work", "save", "outro_start.txt")).read())  # 41.84
DUR = 50.5
AD_CUT = 27.1                    # здесь вклеивается баннер (камера стоит спокойно)
GOLD_RGB, GREEN_RGB, PINK_RGB, PURPLE_RGB = (245, 190, 50), (46, 196, 110), (240, 70, 110), (150, 90, 230)

# ---------------------------------------------------------------- субтитры
T_, Go_, G_, Pi_, Pu_ = TG_RGB, GOLD_RGB, GREEN_RGB, PINK_RGB, PURPLE_RGB
SUBS = [
    (0.204, "ЧТО БУДЕТ, ЕСЛИ КАЖДЫЙ ДЕНЬ", ()), (1.66, "ОТКЛАДЫВАТЬ ПО 100 РУБЛЕЙ?", ("100", "РУБЛЕЙ?"), Go_),
    (3.0, "", ()),
    (4.699, "ЗА МЕСЯЦ ВЫЙДЕТ 3 000,", ("3 000,",), Go_), (6.641, "ЗА ГОД — 36 С ПОЛОВИНОЙ", ("36",), Go_),
    (8.097, "36 С ПОЛОВИНОЙ ТЫСЯЧ,", ("36",), Go_), (9.301, "А ЗА 10 ЛЕТ —", ("10", "ЛЕТ"), G_),
    (10.058, "365 ТЫСЯЧ РУБЛЕЙ", ("365",), G_),
    (13.049, "100 РУБЛЕЙ — ЭТО КОФЕ", ("КОФЕ",), Pi_), (14.276, "НАВЫНОС ИЛИ ДОСТАВКА", ("ДОСТАВКА",), Pi_),
    (15.46, "ДО ДВЕРИ", ()), (16.353, "ФИНАНСИСТЫ НАЗЫВАЮТ ЭТО", ()), (17.635, "«ЭФФЕКТОМ ЛАТТЕ»:", ("«ЭФФЕКТОМ", "ЛАТТЕ»:"), Pi_),
    (19.014, "МЕЛКИЕ ТРАТЫ НЕЗАМЕТНЫ,", ("НЕЗАМЕТНЫ,",)), (20.645, "НО ЗА ГОД СКЛАДЫВАЮТСЯ", ()),
    (21.946, "В БОЛЬШИЕ ДЕНЬГИ", ("БОЛЬШИЕ",), Go_), (23.363, "А ЕСЛИ КЛАСТЬ ИХ НА ВКЛАД,", ("ВКЛАД,",), G_),
    (25.068, "ПРОЦЕНТЫ ДОБАВЯТ ЕЩЁ СВЕРХУ", ("СВЕРХУ",), G_),
    (AD_CUT - 0.02, "", ()),
    (27.311, "ГЛАВНЫЙ СЕКРЕТ —", ()), (28.475, "НЕ СИЛА ВОЛИ,", ("НЕ",)), (29.582, "А АВТОМАТИКА", ("АВТОМАТИКА",), T_),
    (30.66, "В ОПЫТЕ ЭКОНОМИСТА", ()), (31.737, "РИЧАРДА ТАЛЕРА ЛЮДИ,", ("ТАЛЕРА",), T_),
    (33.193, "У КОТОРЫХ ДЕНЬГИ", ()), (33.873, "ОТКЛАДЫВАЛИСЬ САМИ,", ("САМИ,",), T_),
    (35.29, "ЗА НЕСКОЛЬКО ЛЕТ", ()), (36.086, "СТАЛИ КОПИТЬ ПОЧТИ", ()),
    (36.805, "В 4 РАЗА БОЛЬШЕ", ("4", "РАЗА"), G_), (38.942, "ТАК ЧТО НАСТРОЙ", ()),
    (39.65, "АВТОПЕРЕВОД — И ЗАБУДЬ", ("АВТОПЕРЕВОД",), T_),
    (41.804, "А ЕЩЁ БОЛЬШЕ ИНТЕРЕСНОГО —", (), T_), (43.164, "В МОЁМ ТЕЛЕГРАМ-КАНАЛЕ", ("ТЕЛЕГРАМ-КАНАЛЕ",), T_),
    (44.494, "«ПАУЧЬЯ НОРКА»", ("«ПАУЧЬЯ", "НОРКА»"), T_), (45.614, "ТАМ ЭКСКЛЮЗИВ,", ("ЭКСКЛЮЗИВ",), T_),
    (46.314, "КОТОРОГО НЕТ В РОЛИКАХ", (), T_), (47.614, "ССЫЛКА В ПРОФИЛЕ —", ("ПРОФИЛЕ",), T_),
    (48.634, "ЗАЛЕТАЙ!", ("ЗАЛЕТАЙ!",), T_),
]

# ---------------------------------------------------------------- камера и сцены
PUSHES = (12.75, 27.15)          # смена сцены «сдвигом»; склейка — на середине
C1, C2 = (p + 0.15 for p in PUSHES)
CAMERA = [
    (0.0, (-60.0, 5.0, 2.2, 0.9, 50.0)), (2.6, (6.0, 3.6, 1.15, 0.84, 50.0), "out"),   # вход по спирали
    (2.85, (6.0, 3.6, 1.15, 0.84, 50.0)),
    (3.1, (0.0, 3.9, 1.3, 0.98, 50.0), "inout"),              # заставка
    (4.4, (-1.0, 3.9, 1.3, 0.98, 50.0)),
    (4.7, (-90.0, 3.8, 1.4, 1.0, 50.0), "expo"),              # столбики из монет
    (C1 - 0.01, (-96.0, 4.2, 1.6, 1.05, 50.0), "lin"),
    (C1 + 0.01, (-180.0, 3.6, 1.25, 0.86, 50.0), "lin"),      # эффект латте
    (C2 - 0.01, (-186.0, 3.5, 1.25, 0.86, 50.0), "lin"),
    (C2 + 0.01, (-270.0, 3.6, 0.85, 0.95, 50.0), "lin"),      # автоматика — чуть снизу
    (38.7, (-276.0, 3.5, 0.85, 0.95, 50.0), "lin"),
    (39.0, (-366.0, 3.7, 1.3, 0.92, 50.0), "expo"),           # финал
    (41.4, (-370.0, 3.5, 1.3, 0.92, 49.0), "lin"),
    (41.8, (-460.0, 4.3, 1.3, 1.12, 50.0), "expo"), (DUR, (-461.0, 4.0, 1.3, 1.15, 50.0)),
]
OFFSET = [(0.0, 0.5), (2.85, 0.5), (3.1, 0.0, "inout"), (4.4, 0.0), (4.7, -0.55, "expo"), (C1 - 0.01, -0.55),
          (C1 + 0.01, 0.55, "lin"), (C2 - 0.01, 0.55), (C2 + 0.01, -0.55, "lin"), (38.7, -0.55), (39.0, 0.0, "expo"),
          (DUR, 0.0)]
TINT = [(0.0, (0.88, 1.04, 1.12)), (2.85, (0.88, 1.04, 1.12)), (3.1, (1.0, 1.0, 1.0)), (4.4, (1.0, 1.0, 1.0)),
        (4.7, (0.85, 1.0, 1.15)), (C1 - 0.01, (0.85, 1.0, 1.15)), (C1 + 0.01, (0.95, 0.98, 1.12), "lin"),
        (C2 - 0.01, (0.95, 0.98, 1.12)), (C2 + 0.01, (1.12, 1.02, 0.9), "lin"), (38.7, (1.12, 1.02, 0.9)),
        (39.0, (0.92, 1.1, 0.95)), (41.4, (0.92, 1.1, 0.95)), (41.85, (1.0, 1.0, 1.0)), (DUR, (1.0, 1.0, 1.0))]
SEC = {"hook": 4, "stacks": -93, "latte": -183, "auto": -273, "final": -368, "tg": -460.5}
SHAKES = (36.82,)
FLASHES = ((41.85, 0.12),)
JUMPS = [10.1]
ALARM = None
SPOT = None

G = lambda *k: list(k)  # noqa: E731
O = OUTRO - 48.035      # жесты концовки — как в monet_mg, со сдвигом
ARM_L = G((0, 62), (2.85, 62), (3.1, -50, "back"), (4.1, -45), (4.4, 62, "back"), (4.8, -30, "back"), (6.5, -28),
          (6.65, -38, "back"), (9.1, -36), (9.3, -50, "back"), (12.4, -48), (12.6, 62), (C2 + 0.05, 62),
          (27.5, -30, "back"), (30.5, -28), (30.7, -38, "back"), (35.1, -36), (35.3, -25, "back"), (36.7, -22),
          (36.82, -55, "back"), (38.5, -52), (38.8, 62), (38.95, -40, "back"), (41.3, -38), (41.6, 62),
          (54.94 + O, 62), (55.09 + O, -50, "back"), (55.69 + O, -45), (56.09 + O, 62, "back"), (DUR, 62))
ARM_R = G((0, 62), (0.4, -25, "back"), (2.6, -22), (2.9, 62), (3.1, -50, "back"), (4.1, -45), (4.4, 62, "back"),
          (9.3, -50, "back"), (12.4, -48), (12.6, 62), (13.1, -30, "back"), (16.2, -28), (16.4, -40, "back"),
          (18.9, -38), (19.0, 30, "back"), (20.5, 32), (20.7, -45, "back"), (23.2, -42), (23.4, -30, "back"),
          (26.7, -28), (26.9, 62), (36.82, -55, "back"), (38.5, -52), (38.8, 62),
          (47.79 + O, 62), (48.25 + O, -48, "back"), (50.6 + O, -45), (50.9 + O, -55, "back"), (53.6 + O, -52),
          (53.85 + O, -40, "back"), (54.8 + O, -42), (54.95 + O, -50, "back"), (55.69 + O, -45), (55.84 + O, -58),
          (55.99 + O, -32), (56.14 + O, -58), (56.29 + O, -32), (DUR, 62))
HEAD = G((0, 0), (0.6, -6), (1.7, 7), (2.6, -5), (3.1, 0), (4.7, 6), (5.6, -6), (6.6, 7), (8.1, -5), (9.3, 8),
         (10.1, -6), (11.2, 6), (12.6, 0), (13.1, -6), (14.3, 7), (15.5, -5), (16.4, 6), (17.6, -7), (19.0, 6),
         (20.6, -6), (21.9, 7), (23.4, -6), (25.1, 7), (26.8, 0), (27.4, -6), (28.5, 6), (29.6, -7), (30.7, 6),
         (31.7, -6), (33.2, 6), (34.6, -6), (35.3, 7), (36.8, 9), (38.0, -5), (39.0, 6), (40.4, -6), (41.5, 0),
         (42.0, -6), (48.9, 0), (DUR, 0))


# ---------------------------------------------------------------- картинки вставок
def rub_tex(s, size, col, pad=12):
    f = mg.font(size, tall=True)
    tmp = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    w = int(rub_width(tmp, s, f)) + 2 * pad
    l, t_, r, b = f.getbbox("0Р")
    im = Image.new("RGBA", (w, b + 2 * pad), (0, 0, 0, 0))
    text_rub(ImageDraw.Draw(im), (pad, pad), s, f, col)
    return mg.premul(im)


def pig_tex():
    im = Image.new("RGBA", (560, 440), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    pink = (240, 140, 170)
    d.ellipse([60, 90, 480, 380], fill=pink + (255,))
    d.ellipse([430, 180, 540, 290], fill=(250, 160, 185, 255))
    d.ellipse([470, 215, 488, 240], fill=(170, 80, 100))
    d.ellipse([500, 215, 518, 240], fill=(170, 80, 100))
    d.polygon([(330, 110), (380, 30), (410, 120)], fill=pink + (255,))
    for x in (130, 200, 330, 400):
        d.rounded_rectangle([x, 330, x + 50, 420], 18, fill=(220, 120, 150, 255))
    d.ellipse([370, 170, 396, 196], fill=(40, 20, 30))
    d.rounded_rectangle([200, 96, 320, 116], 10, fill=(150, 60, 90))
    d.arc([10, 160, 90, 240], 60, 300, fill=pink + (255,), width=12)
    return mg.premul(im)


def coin_tex():
    S = 200
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([0, 0, S - 1, S - 1], fill=(196, 140, 30, 255))
    d.ellipse([12, 12, S - 13, S - 13], fill=GOLD_RGB + (255,))
    f = mg.font(70, tall=True)
    d.text((44, 50), "100", font=f, fill=(130, 90, 10))
    return mg.premul(im)


def cup_tex():
    im = Image.new("RGBA", (360, 460), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.polygon([(60, 120), (300, 120), (270, 440), (90, 440)], fill=(235, 225, 210, 255))
    d.rounded_rectangle([40, 90, 320, 130], 16, fill=(90, 60, 45, 255))
    d.rectangle([80, 220, 280, 320], fill=(150, 100, 70, 255))
    for k in range(3):
        d.arc([110 + k * 50, 0, 150 + k * 50, 80], 200, 340, fill=(200, 200, 210), width=6)
    return mg.premul(im)


def box_tex():
    im = Image.new("RGBA", (420, 380), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.polygon([(30, 120), (210, 60), (390, 120), (210, 180)], fill=(200, 150, 90, 255))
    d.polygon([(30, 120), (210, 180), (210, 370), (30, 300)], fill=(170, 120, 70, 255))
    d.polygon([(390, 120), (210, 180), (210, 370), (390, 300)], fill=(150, 100, 60, 255))
    d.line([(120, 90), (300, 150)], fill=(230, 200, 150), width=14)
    return mg.premul(im)


def tag_tex():
    im = Image.new("RGBA", (300, 150), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.polygon([(0, 75), (50, 0), (299, 0), (299, 149), (50, 149)], fill=(28, 26, 34, 255))
    d.line([(0, 75), (50, 0), (299, 0), (299, 149), (50, 149), (0, 75)], fill=PINK_RGB + (255,), width=6)
    text_rub(d, (74, 22), "100 ₽", mg.font(84, tall=True), (255, 255, 255))
    return mg.premul(im)


def jar_tex(v):
    """Банка, которая наполняется монетами (v от 0 до 1)."""
    im = Image.new("RGBA", (380, 480), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([40, 70, 340, 460], 50, fill=(40, 50, 60, 200), outline=(170, 200, 220, 255), width=8)
    d.rounded_rectangle([70, 30, 310, 80], 16, fill=(150, 160, 175, 255))
    h = int(360 * v)
    if h > 10:
        d.rounded_rectangle([56, 448 - h, 324, 448], 40, fill=GOLD_RGB + (255,))
        for y in range(448 - h + 16, 440, 34):
            d.line([(70, y), (310, y)], fill=(200, 145, 30), width=4)
    return mg.premul(im)


def gear_tex(a):
    S = 360
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c = S / 2
    for k in range(10):
        ang = math.radians(a + k * 36)
        pts = []
        for da, r in ((-9, 120), (-6, 160), (6, 160), (9, 120)):
            aa = ang + math.radians(da)
            pts.append((c + r * math.cos(aa), c + r * math.sin(aa)))
        d.polygon(pts, fill=TG_RGB + (255,))
    d.ellipse([c - 128, c - 128, c + 128, c + 128], fill=TG_RGB + (255,))
    d.ellipse([c - 52, c - 52, c + 52, c + 52], fill=(18, 18, 26, 255))
    return mg.premul(im)


def thaler_tex(k):
    """Опыт Талера: доля сбережений 3,5% → 13,6%."""
    im = rounded((820, 520), 34, (18, 18, 26, 255), (72, 72, 92, 255), 3)
    d = ImageDraw.Draw(im)
    d.text((40, 26), "СКОЛЬКО ОТКЛАДЫВАЛИ", font=mg.font(48, tall=True), fill="white")
    for x, v, lab, col in ((110, 3.5, "ДО", (110, 110, 125)), (470, 13.6, "С АВТОМАТИКОЙ", GREEN_RGB)):
        vv = v * (1 if x == 110 else k)
        h = int(18 * vv)
        d.rounded_rectangle([x, 420 - max(h, 8), x + 240, 420], 24, fill=col + (255,))
        f = mg.font(64, tall=True)
        txt = f"{vv:.1f}%".replace(".", ",")
        tw = d.textbbox((0, 0), txt, font=f)[2]
        d.text((x + 120 - tw / 2, 420 - max(h, 8) - 78), txt, font=f, fill="white")
        f2 = mg.font(36, tall=True)
        tw = d.textbbox((0, 0), lab, font=f2)[2]
        d.text((x + 120 - tw / 2, 436), lab, font=f2, fill=(200, 200, 215))
    return mg.premul(im)


def phone_tex(on):
    im = Image.new("RGBA", (520, 860), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([60, 20, 460, 840], 60, fill=(16, 16, 22, 255), outline=(150, 150, 165, 255), width=10)
    d.rounded_rectangle([90, 60, 430, 800], 40, fill=(26, 30, 44, 255))
    d.text((120, 120), "АВТОПЕРЕВОД", font=mg.font(52, tall=True), fill="white")
    d.rounded_rectangle([110, 220, 410, 380], 26, fill=(40, 46, 66, 255))
    text_rub(d, (140, 240), "100 ₽", mg.font(80, tall=True), GOLD_RGB)
    d.text((140, 330), "каждый день", font=mg.font(32), fill=(170, 175, 195))
    col = GREEN_RGB if on else (80, 80, 96)
    d.rounded_rectangle([140, 440, 380, 540], 50, fill=col + (255,))
    x = 330 if on else 190
    d.ellipse([x - 42, 448, x + 42, 532], fill="white")
    if on:
        d.ellipse([200, 600, 320, 720], fill=GREEN_RGB + (255,))
        d.line([(228, 662), (252, 688), (296, 634)], fill="white", width=14)
    return mg.premul(im)


# ---------------------------------------------------------------- сцена
class Scene(base.Scene):
    cfg = sys.modules[__name__]

    def __init__(self, voice):
        super().__init__(voice)
        T = self.T
        T["pig"], T["coin"] = pig_tex(), coin_tex()
        T["sum"] = [rub_tex(s, 90, GOLD_RGB) for s in ("3 000 ₽", "36 500 ₽", "365 000 ₽")]
        T["per"] = [mg.premul(mg.text_rgba(s, 60, (200, 200, 215), tall=True)) for s in ("МЕСЯЦ", "ГОД", "10 ЛЕТ")]
        T["cup"], T["box"], T["tag"] = cup_tex(), box_tex(), tag_tex()
        T["latte"] = card("ФИНАНСЫ", "ЭФФЕКТ ЛАТТЕ", "мелкие траты складываются", PINK_RGB)
        T["dep"] = chip("ВКЛАД: + ПРОЦЕНТЫ", GREEN_RGB)
        T["will"] = chip("НЕ СИЛА ВОЛИ", RED_RGB)
        T["auto"] = chip("А АВТОМАТИКА", TG_RGB)
        T["thaler"] = card("ЭКОНОМИСТ", "РИЧАРД ТАЛЕР", "лауреат Нобелевской премии", TG_RGB)
        T["x4"] = stamp("×4", GREEN_RGB)
        T["forget"] = chip("…И ЗАБУДЬ", GREEN_RGB)
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
        for p0 in PUSHES:  # «сдвиг»: старый кадр уезжает влево, новый въезжает справа
            p = (t - p0) / 0.3
            if 0 < p < 1:
                out = np.full_like(fr, (20, 16, 24))
                if p < 0.5:
                    dx = -int(W * smooth(p / 0.5))
                else:
                    dx = int(W * (1 - smooth((p - 0.5) / 0.5)))
                if dx < 0:
                    out[:, :W + dx] = fr[:, -dx:]
                elif dx > 0:
                    out[:, dx:] = fr[:, :W - dx]
                else:
                    out = fr
                fr = out
        return fr

    def items_for(self, t, cam, it):
        T = self.T
        P = lambda y0, *a, **k: place(y0, *a, **k)  # noqa: E731

        # 1. Крючок (персонаж слева, камера по спирали): копилка, в неё падают монеты «100»
        y0 = SEC["hook"]
        k = pop(t, 0.2, 2.95, 0.45, 0.25)
        if k > 0:
            pig = P(y0, 0.75, 1.0)
            self.add_bb(it, cam, pig, 0.5 * k, T["pig"], anchor=(0.5, 0.5))
            for i in range(4):
                p = ((t - 0.6 - 0.55 * i) % 2.2) / 0.6
                if t > 0.6 + 0.55 * i and p < 1:
                    self.add_bb(it, cam, pig + [0.05, 1.0 - 0.75 * p * p, 0], 0.13 * k, T["coin"], anchor=(0.5, 0.5))

        # 2. Заставка
        k = pop(t, 3.1, 4.4, 0.35, 0.3)
        if k > 0:
            self.add_panel(it, cam, place(0, 0.0, 1.95, 0.35), face(0, -6), 1.2 * k, 0.564 * k, 0.11 * k,
                           T["stinger"], pitch=-6, side=(30, 24, 120), edge=(200, 190, 255))

        # 3. Столбики из монет (персонаж справа): месяц → год → 10 лет
        y0 = SEC["stacks"]
        for i, (sd, hh, t0) in enumerate(((-1.45, 0.25, 4.75), (-1.05, 0.8, 6.65), (-0.62, 1.6, 9.35))):
            kb = pop(t, t0, C1, 0.6, 0.2)
            if kb <= 0:
                continue
            q = P(y0, sd, 0.0, -0.5)
            h = hh * smooth(min(1.0, (t - t0) / 0.9))
            self.add_box(it, cam, (q[0], q[2]), (0.28, 0.28), max(0.02, h), (40, 160, 225), radius=0.14,
                         edge=(120, 220, 255))
            self.add_bb(it, cam, q + [0, h + 0.16, 0], 0.11 * kb, T["sum"][i], anchor=(0.5, 0.5))
            self.add_bb(it, cam, q + [0, -0.02, 0] + basis(y0)[1] * 0.25, 0.08 * kb, T["per"][i], anchor=(0.5, 1.0))

        # 4. Эффект латте (персонаж слева): кофе и доставка с ценниками → карточка → банка наполняется → вклад
        y0 = SEC["latte"]
        r_, c_ = basis(y0)
        k = pop(t, C1 + 0.05, 16.3, 0.4, 0.25)
        if k > 0:
            self.add_bb(it, cam, P(y0, 0.5, 1.5), 0.45 * k, T["cup"], anchor=(0.5, 0.5))
            self.add_bb(it, cam, P(y0, 0.5, 1.95), 0.12 * k, T["tag"], anchor=(0.5, 0.5))
            kb = pop(t, 14.9, 16.3, 0.35, 0.25)
            if kb > 0:
                self.add_bb(it, cam, P(y0, 1.0, 1.4), 0.4 * kb, T["box"], anchor=(0.5, 0.5))
                self.add_bb(it, cam, P(y0, 1.0, 1.85), 0.12 * kb, T["tag"], anchor=(0.5, 0.5))
        k = pop(t, 16.35, 18.95, 0.4, 0.25)
        if k > 0:
            pos = P(y0, 0.72, 1.6) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 14), 0.9 * k, 0.427 * k, 0.06 * k, T["latte"], pitch=-6,
                           side=(80, 40, 60), edge=(255, 200, 220))
        k = pop(t, 19.0, AD_CUT + 0.3, 0.4, 0.2)
        if k > 0:
            v = round(clamp((t - 20.6) / 2.0) * 0.8 * 20) / 20 + (0.15 if t > 25.0 else 0.0)
            jar = P(y0, 0.75, 1.3)
            self.add_bb(it, cam, jar, 0.6 * k, self.cached("jar", jar_tex, min(1.0, v)), anchor=(0.5, 0.5))
            for i in range(10):
                s0 = 20.6 + 0.2 * i
                p = (t - s0) / 0.6
                if 0 < p < 1:
                    q = jar + r_ * (-0.55 * (1 - p)) + np.array([0, 0.6 * (1 - p) * p * 2 + 0.25 * (1 - p), 0])
                    self.add_bb(it, cam, q, 0.08, T["coin"], anchor=(0.5, 0.5))
            kd = pop(t, 23.4, AD_CUT + 0.3, 0.35, 0.2)
            if kd > 0:
                self.add_bb(it, cam, P(y0, 0.75, 2.05), 0.11 * kd, T["dep"], anchor=(0.5, 0.5))

        # 5. Автоматика (персонаж справа, чуть снизу): «не сила воли» → шестерёнка → Талер → столбики → ×4
        y0 = SEC["auto"]
        k = pop(t, C2 + 0.05, 30.6, 0.4, 0.25)
        if k > 0:
            self.add_bb(it, cam, P(y0, -0.72, 2.0), 0.11 * k, T["will"], anchor=(0.5, 0.5))
            kg = pop(t, 29.55, 30.6, 0.35, 0.25)
            if kg > 0:
                self.add_bb(it, cam, P(y0, -0.72, 1.4), 0.4 * kg, self.cached("gear", gear_tex, int(t * 60) % 36 * 10),
                            anchor=(0.5, 0.5))
                self.add_bb(it, cam, P(y0, -0.72, 1.75), 0.11 * kg, T["auto"], anchor=(0.5, 0.5))
        k = pop(t, 30.65, 35.25, 0.4, 0.25)
        if k > 0:
            pos = P(y0, -0.72, 1.6) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, -14), 0.9 * k, 0.427 * k, 0.06 * k, T["thaler"], pitch=6,
                           side=(30, 60, 80), edge=(190, 230, 255))
        k = pop(t, 35.3, 38.85, 0.4, 0.25)
        if k > 0:
            kb = round(smooth((t - 35.6) / 1.1) * 20) / 20
            pos = P(y0, -0.72, 1.6) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, -14), 0.95 * k, 0.602 * k, 0.06 * k, self.cached("th", thaler_tex, kb),
                           pitch=6, side=(30, 70, 40), edge=(190, 255, 210))
            if t > 36.82:
                ks = clamp((t - 36.82) / 0.15)
                self.add_bb(it, cam, pos + basis(y0)[1] * 0.12 + basis(y0)[0] * 0.3 + [0.0, -0.45, 0.0], 0.16 * (1 + 0.6 * (1 - ks)) * k,
                            T["x4"], anchor=(0.5, 0.5))

        # 6. Финал (персонаж в центре): экран телефона «автоперевод», переключатель щёлкает
        y0 = SEC["final"]
        k = pop(t, 39.05, 41.45, 0.45, 0.25)
        if k > 0:
            self.add_bb(it, cam, place(y0, -0.45, 1.5, 0.1) + [0, 0.02 * math.sin(t * 2), 0], 0.75 * k,
                        self.cached("ph", phone_tex, t > 39.8), anchor=(0.5, 0.5))
            kf = pop(t, 40.4, 41.45, 0.3, 0.25)
            if kf > 0:
                self.add_bb(it, cam, place(y0, 0.4, 2.05, 0.1), 0.1 * kf, T["forget"], anchor=(0.5, 0.5))

        # 7. Концовка с Telegram — общий шаблон
        self.outro_items(t, cam, it, OUTRO, SEC["tg"])


SFX = [(t, "whoosh") for t in (0.0, 3.05, 4.6) + PUSHES + (38.9, 41.7)]


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
        for t in (0.6, 1.8, 3.7, 5.8, 8.2, 11.5, 12.9, 14.0, 15.5, 17.5, 21.5, 24.0, 26.5, 27.3, 28.5, 30.0, 33.0,
                  37.2, 40.2, 45.0):
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
    base.mix_audio(tmp, voice, out, music=os.path.join(ROOT, "work", "save", "music.wav"), sfx_list=SFX, dur=DUR,
                   sfx_vol=0.15)  # «вжух» лёгкий и тихий
    os.remove(tmp)


if __name__ == "__main__":
    main()
