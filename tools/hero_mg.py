"""Ролик №15: «Что будет, если в мире появится настоящий супергерой?» (+ рекламный баннер в середине).

python tools/hero_mg.py work/hero/voice_clean.wav output/hero.mp4 [--preview] [--jobs 4]
затем: python tools/ad_banner.py insert output/hero.mp4 24.97 output/hero_ad.mp4 hero_mg
Общий движок, персонаж и концовка с Telegram — из memory_mg.py; параллельный рендер — как в money_mg.py.
Новое в монтаже: газета крутится и влетает в кадр; 3D-молоток судьи бьёт с толчком камеры; «комиксная» смена
сцены — диагональная шторка с точками-растром; город из зданий, которые проседают после «битвы», и чек с
бегущей суммой; съёмка снизу; «эффект свидетеля» — человечки и столбики «помогли: 85% / 62% / 31%»; толпа
смотрит в небо «ГДЕ ГЕРОЙ?»; финал сверху с опусканием камеры к персонажу.
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
OUTRO = float(open(os.path.join(ROOT, "work", "hero", "outro_start.txt")).read())  # 41.36
DUR = 50.06
AD_CUT = 24.97                   # здесь вклеивается баннер (камера стоит спокойно)
GOLD_RGB, GREEN_RGB, PURPLE_RGB = (245, 190, 50), (46, 196, 110), (150, 90, 230)

# ---------------------------------------------------------------- субтитры
T_, Go_, G_, Pu_ = TG_RGB, GOLD_RGB, GREEN_RGB, PURPLE_RGB
SUBS = [
    (0.174, "ЧТО БУДЕТ, ЕСЛИ В МИРЕ", ()), (1.457, "ПОЯВИТСЯ НАСТОЯЩИЙ", ()),
    (2.626, "СУПЕРГЕРОЙ?", ("СУПЕРГЕРОЙ?",), Go_), (3.45, "", ()),
    (5.155, "СНАЧАЛА ЕГО ЖДЁТ", ()), (6.155, "НЕ СЛАВА, А СУД", ("СУД",)),
    (7.683, "ЛОВИТЬ ПРЕСТУПНИКОВ", ()), (8.702, "БЕЗ РАЗРЕШЕНИЯ —", ("БЕЗ", "РАЗРЕШЕНИЯ")),
    (9.872, "ЭТО САМОУПРАВСТВО", ("САМОУПРАВСТВО",), Pu_), (11.155, "ЗА НЕГО НАКАЗЫВАЮТ,", ("НАКАЗЫВАЮТ,",)),
    (12.4, "ДАЖЕ ЕСЛИ", ()), (13.042, "ТЫ ХОТЕЛ ДОБРА", ("ДОБРА",), G_),
    (14.255, "ПОТОМ ПРИДУТ СЧЕТА", ("СЧЕТА",), Go_), (15.736, "КОМПАНИЯ, КОТОРАЯ", ()),
    (16.639, "ОЦЕНИВАЕТ УЩЕРБ", ("УЩЕРБ",)), (17.488, "ОТ КАТАСТРОФ,", ()), (18.318, "ПОДСЧИТАЛА:", ()),
    (19.299, "БИТВА ИЗ ФИЛЬМА", ()), (20.186, "«МСТИТЕЛИ»", ("«МСТИТЕЛИ»",)),
    (20.827, "ОБОШЛАСЬ БЫ НЬЮ-ЙОРКУ", ()), (21.865, "ПРИМЕРНО В 160", ("160",), Go_),
    (22.808, "МИЛЛИАРДОВ ДОЛЛАРОВ", ("МИЛЛИАРДОВ",), Go_), (AD_CUT - 0.02, "", ()),
    (25.17, "НО САМОЕ СТРАННОЕ —", ()), (26.611, "ЛЮДИ ПЕРЕСТАНУТ", ()),
    (27.347, "ПОМОГАТЬ ДРУГ ДРУГУ", ("ПОМОГАТЬ",)), (28.621, "ПСИХОЛОГИ НАЗЫВАЮТ ЭТО", ()),
    (29.904, "ЭФФЕКТОМ СВИДЕТЕЛЯ", ("СВИДЕТЕЛЯ",), Pu_), (31.375, "ЧЕМ БОЛЬШЕ ВОКРУГ ТЕХ,", ()),
    (32.602, "КТО МОЖЕТ ПОМОЧЬ,", ()), (33.734, "ТЕМ РЕЖЕ ПОМОГАЕТ", ("РЕЖЕ",)),
    (34.64, "КАЖДЫЙ", ("КАЖДЫЙ",)), (35.526, "А С СУПЕРГЕРОЕМ", ()), (36.489, "ВСЕ БУДУТ ПРОСТО", ()),
    (37.375, "ЖДАТЬ ЕГО", ("ЖДАТЬ",), Go_), (38.574, "НАСТОЯЩИЙ ГЕРОЙ —", ("ГЕРОЙ",), G_),
    (39.875, "ТОТ, КТО НЕ ЖДЁТ", ("НЕ", "ЖДЁТ"), G_),
    (41.328, "А ЕЩЁ БОЛЬШЕ ИНТЕРЕСНОГО —", (), T_), (42.688, "В МОЁМ ТЕЛЕГРАМ-КАНАЛЕ", ("ТЕЛЕГРАМ-КАНАЛЕ",), T_),
    (44.018, "«ПАУЧЬЯ НОРКА»", ("«ПАУЧЬЯ", "НОРКА»"), T_), (45.138, "ТАМ ЭКСКЛЮЗИВ,", ("ЭКСКЛЮЗИВ",), T_),
    (45.838, "КОТОРОГО НЕТ В РОЛИКАХ", (), T_), (47.138, "ССЫЛКА В ПРОФИЛЕ —", ("ПРОФИЛЕ",), T_),
    (48.158, "ЗАЛЕТАЙ!", ("ЗАЛЕТАЙ!",), T_),
]

# ---------------------------------------------------------------- камера и сцены
SLASH = (14.0, 25.0)             # «комиксная» шторка; склейка — на её середине
C1, C2 = (s + 0.15 for s in SLASH)
CAMERA = [
    (0.0, (-14.0, 4.2, 1.45, 0.88, 50.0)), (0.8, (-8.0, 3.6, 1.15, 0.84, 50.0), "out"),
    (3.3, (-5.0, 3.6, 1.15, 0.84, 50.0)),
    (3.55, (0.0, 3.9, 1.3, 0.98, 50.0), "inout"),             # заставка
    (4.95, (1.0, 3.9, 1.3, 0.98, 50.0)),
    (5.3, (92.0, 3.8, 2.1, 0.92, 50.0), "expo"),              # суд — чуть сверху
    (C1 - 0.01, (101.0, 3.4, 1.8, 0.92, 50.0), "lin"),
    (C1 + 0.01, (200.0, 3.4, 0.55, 1.05, 50.0), "lin"),       # счета — снизу, город над головой
    (AD_CUT + 0.1, (205.0, 3.55, 0.6, 1.05, 50.0), "lin"),
    (C2 - 0.01, (205.1, 3.55, 0.6, 1.05, 50.0), "lin"),
    (C2 + 0.01, (300.0, 3.8, 1.35, 0.86, 50.0), "lin"),       # эффект свидетеля
    (31.2, (304.0, 3.7, 1.3, 0.86, 50.0), "lin"),
    (31.6, (330.0, 3.7, 1.3, 0.86, 50.0), "expo"),
    (38.3, (336.0, 3.6, 1.3, 0.86, 50.0), "lin"),
    (38.65, (372.0, 3.0, 3.3, 0.55, 52.0), "expo"),           # финал — сверху…
    (40.9, (378.0, 3.9, 1.5, 0.95, 50.0), "inout"),           # …камера опускается к персонажу
    (41.33, (460.0, 4.3, 1.3, 1.12, 50.0), "expo"), (DUR, (461.0, 4.0, 1.3, 1.15, 50.0)),
]
OFFSET = [(0.0, -0.5), (3.3, -0.5), (3.55, 0.0, "inout"), (4.95, 0.0), (5.3, 0.55, "expo"), (C1 - 0.01, 0.55),
          (C1 + 0.01, -0.55, "lin"), (C2 - 0.01, -0.55), (C2 + 0.01, 0.55, "lin"), (38.3, 0.55),
          (38.65, 0.0, "expo"), (DUR, 0.0)]
# окраска комнаты по сценам (B, G, R)
TINT = [(0.0, (1.12, 1.02, 0.9)), (3.3, (1.12, 1.02, 0.9)), (3.55, (1.0, 1.0, 1.0)), (4.95, (1.0, 1.0, 1.0)),
        (5.3, (1.12, 0.92, 1.04)), (C1 - 0.01, (1.12, 0.92, 1.04)), (C1 + 0.01, (0.85, 0.97, 1.15), "lin"),
        (C2 - 0.01, (0.85, 0.97, 1.15)), (C2 + 0.01, (1.1, 1.06, 0.88), "lin"), (38.3, (1.1, 1.06, 0.88)),
        (38.65, (0.9, 1.02, 1.12)), (41.0, (0.9, 1.02, 1.12)), (41.4, (1.0, 1.0, 1.0)), (DUR, (1.0, 1.0, 1.0))]
SEC = {"hook": -8, "court": 96, "bills": 203, "crowd": 302, "crowd2": 333, "hero": 375, "tg": 460.5}
SHAKES = (7.0,)
FLASHES = ((7.0, 0.12), (38.65, 0.1), (41.37, 0.12))
JUMPS = [38.7]
ALARM = None
SPOT = None

G = lambda *k: list(k)  # noqa: E731
O = OUTRO - 48.035      # жесты концовки — как в monet_mg, со сдвигом
ARM_L = G((0, 62), (0.3, -25, "back"), (3.2, -22), (3.4, 62), (3.6, -50, "back"), (4.6, -45), (4.9, 62, "back"),
          (9.8, 62), (10.0, -20, "back"), (11.0, -18), (11.3, 25, "back"), (13.0, 28), (13.3, 62),
          (C1 + 0.05, 62), (14.4, -30, "back"), (19.1, -28), (19.4, -48, "back"), (20.7, -45), (21.0, 62, "back"),
          (22.4, 62), (22.6, -38, "back"), (24.5, -36), (24.8, 62), (28.5, 62), (28.7, -15, "back"), (30.8, -12),
          (31.1, 62), (35.4, 62), (35.6, -50, "back"), (37.2, -48), (37.5, 62), (39.9, 62), (40.0, -40, "back"),
          (40.9, -38), (41.2, 62), (54.94 + O, 62), (55.09 + O, -50, "back"), (55.69 + O, -45),
          (56.09 + O, 62, "back"), (DUR, 62))
ARM_R = G((0, 62), (3.4, 62), (3.6, -50, "back"), (4.6, -45), (4.9, 62, "back"), (5.3, -30, "back"), (6.6, -28),
          (6.8, -62, "back"), (6.98, 25), (7.3, 20), (7.7, -32, "back"), (9.6, -30), (11.3, 25, "back"), (13.0, 28),
          (13.3, -40, "back"), (13.9, -38), (C1 - 0.02, 62), (19.1, 62), (19.4, -48, "back"), (20.7, -45),
          (21.0, 62, "back"), (C2 + 0.05, 62), (25.4, -30, "back"), (28.4, -28), (28.7, -15, "back"), (30.8, -12),
          (31.7, -28, "back"), (33.6, -26), (33.9, 40, "back"), (35.2, 42), (35.6, -50, "back"), (37.2, -48),
          (37.5, 62), (38.6, 62), (38.7, -60, "back"), (39.8, -58), (40.0, -40, "back"), (40.9, -38),
          (47.79 + O, 62), (48.25 + O, -48, "back"), (50.6 + O, -45), (50.9 + O, -55, "back"), (53.6 + O, -52),
          (53.85 + O, -40, "back"), (54.8 + O, -42), (54.95 + O, -50, "back"), (55.69 + O, -45), (55.84 + O, -58),
          (55.99 + O, -32), (56.14 + O, -58), (56.29 + O, -32), (DUR, 62))
HEAD = G((0, 0), (0.6, -6), (1.5, 7), (2.6, -4), (3.6, 0), (5.2, 6), (6.2, -6), (7.0, 9), (7.7, -5), (9.9, 7),
         (11.2, -6), (13.0, 6), (14.3, -6), (16.6, 6), (19.3, -7), (20.8, 8), (22.5, -5), (24.2, 4), (24.9, 0),
         (25.3, 6), (27.3, -6), (28.6, 5), (29.9, -7), (31.4, 6), (33.7, -8), (34.6, 4), (35.5, -9), (37.4, 6),
         (38.6, -5), (39.9, 7), (41.6, -6), (48.4, 0), (DUR, 0))


# ---------------------------------------------------------------- картинки вставок
def news_tex():
    im = rounded((760, 560), 30, (34, 33, 38, 255), (90, 88, 100, 255), 3)
    d = ImageDraw.Draw(im)
    d.text((36, 26), "ГОРОДСКИЕ НОВОСТИ", font=mg.font(40, tall=True), fill=(200, 196, 186))
    d.text((520, 34), "№ 1  · сегодня", font=mg.font(26), fill=(140, 138, 150))
    d.rectangle([36, 86, 724, 90], fill=(140, 138, 150))
    d.text((36, 104), "ГЕРОЙ", font=mg.font(130, tall=True), fill=(245, 242, 232))
    d.text((36, 234), "СРЕДИ НАС?!", font=mg.font(96, tall=True), fill=GOLD_RGB + (255,))
    d.rounded_rectangle([36, 360, 300, 520], 20, fill=(54, 54, 64))
    d.ellipse([128, 380, 208, 460], fill=(90, 90, 104))
    d.text((150, 384), "?", font=mg.font(64, tall=True), fill=(220, 220, 230))
    d.pieslice([98, 450, 238, 590], 180, 360, fill=(90, 90, 104))
    for i in range(6):
        y = 372 + i * 26
        d.rounded_rectangle([330, y, 724 - (90 if i == 5 else 0), y + 10], 5, fill=(84, 84, 96))
    return mg.premul(im)


def gavel_base():
    """Молоток судьи; точка вращения (рука) — в центре картинки."""
    im = Image.new("RGBA", (600, 600), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([287, 120, 313, 300], 12, fill=(128, 84, 52, 255))
    d.rounded_rectangle([205, 50, 395, 140], 26, fill=(96, 62, 40, 255))
    for x in (238, 352):
        d.rounded_rectangle([x, 50, x + 12, 140], 6, fill=(210, 170, 70, 255))
    return im


def gavel_tex(gb, ang):
    return mg.premul(gb.rotate(ang, resample=Image.BICUBIC))


def slava_tex(crossed):
    im = rounded((420, 140), 40, (20, 20, 28, 255), GOLD_RGB + (255,), 5)
    d = ImageDraw.Draw(im)
    d.text((66, 22), "СЛАВА", font=mg.font(84, tall=True), fill=(255, 230, 160))
    if crossed:
        d.line([(40, 110), (380, 30)], fill=RED_RGB + (255,), width=16)
    return mg.premul(im)


def receipt_tex(v):
    im = rounded((560, 760), 30, (24, 24, 30, 255), (100, 96, 80, 255), 3)
    d = ImageDraw.Draw(im)
    d.text((40, 34), "СЧЁТ ЗА БИТВУ", font=mg.font(56, tall=True), fill="white")
    d.text((40, 108), "Нью-Йорк, оценка ущерба", font=mg.font(28), fill=(150, 150, 170))
    rows = [("небоскрёбы", "█████"), ("дороги и мосты", "███"), ("машины", "██"), ("уборка", "█"),
            ("простой бизнеса", "████")]
    for i, (a, b) in enumerate(rows):
        y = 180 + i * 70
        d.text((40, y), a, font=mg.font(34), fill=(200, 200, 215))
        d.text((380, y), b, font=mg.font(34), fill=(110, 110, 125))
        for x in range(40, 520, 22):
            d.line([(x, y + 54), (x + 10, y + 54)], fill=(60, 60, 72), width=2)
    d.text((40, 560), "ИТОГО:", font=mg.font(44, tall=True), fill=(200, 200, 215))
    txt = f"${v} МЛРД"
    f = mg.font(104, tall=True)
    tw = d.textbbox((0, 0), txt, font=f)[2]
    d.text((520 - tw, 610), txt, font=f, fill=GOLD_RGB + (255,))
    return mg.premul(im)


def person_tex(col, mark=None):
    im = Image.new("RGBA", (200, 330), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([60, 110, 140, 190], fill=col + (255,))
    d.rounded_rectangle([40, 200, 160, 330], 40, fill=col + (255,))
    if mark:
        d.rounded_rectangle([50, 0, 150, 90], 30, fill=(20, 20, 28, 255), outline=col + (255,), width=5)
        d.text((86 if mark == "!" else 80, 6), mark, font=mg.font(66, tall=True), fill="white")
    return mg.premul(im)


def bar_label(text, col):
    return mg.premul(mg.text_rgba(text, 80, col, tall=True))


def hero_card():
    im = rounded((860, 420), 40, (16, 22, 18, 255), GREEN_RGB + (255,), 5)
    d = ImageDraw.Draw(im)
    f = mg.font(84, tall=True)
    d.text((44, 40), "НАСТОЯЩИЙ ГЕРОЙ —", font=f, fill="white")
    d.text((44, 160), "ТОТ, КТО", font=f, fill="white")
    l, t_, r, b = d.textbbox((0, 0), "НЕ ЖДЁТ", font=f)
    d.rounded_rectangle([40, 280, 40 + r - l + 52, 280 + b - t_ + 36], 30, fill=GREEN_RGB + (255,))
    d.text((66 - l, 298 - t_), "НЕ ЖДЁТ", font=f, fill="white")
    return mg.premul(im)


def halftone():
    """Комиксный растр: тёмно-красный фон с точками."""
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (34, 24, 110)
    for y in range(0, H, 28):
        for x in range((y // 28) % 2 * 14, W, 28):
            cv2.circle(img, (x, y), 7, (60, 50, 190), -1, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------- сцена
class Scene(base.Scene):
    cfg = sys.modules[__name__]

    def __init__(self, voice):
        super().__init__(voice)
        T = self.T
        T["news"] = news_tex()
        T["gb"] = gavel_base()
        T["slava"], T["slava_x"] = slava_tex(False), slava_tex(True)
        T["sud"] = stamp("СУД!", RED_RGB)
        T["law"] = card("ЗАКОН", "САМОУПРАВСТВО", "ловить преступников самому", PURPLE_RGB)
        T["punish"] = card("ЗА ЭТО", "НАКАЗЫВАЮТ", "намерения не важны", RED_RGB)
        T["good"] = stamp("ДАЖЕ ЗА ДОБРО", GREEN_RGB)
        T["kac"] = card("ОЦЕНКА УЩЕРБА", "KINETIC ANALYSIS", "считает ущерб от катастроф", BLUE_RGB)
        T["p_gray"] = person_tex((120, 120, 136))
        T["p_need"] = person_tex(RED_RGB, "!")
        T["p_q"] = person_tex((120, 120, 136), "?")
        T["nohelp"] = chip("НИКТО НЕ ПОМОГАЕТ", RED_RGB)
        T["bys"] = card("ПСИХОЛОГИЯ", "ЭФФЕКТ СВИДЕТЕЛЯ", "опыт Дарли и Латане, 1968", PURPLE_RGB)
        T["helped"] = chip("ПОМОГЛИ:", GREEN_RGB)
        T["pct"] = [bar_label(s, (255, 255, 255)) for s in ("85%", "62%", "31%")]
        T["cnt"] = [bar_label(s, (190, 190, 205)) for s in ("ОДИН", "ДВОЕ", "ПЯТЕРО")]
        T["where"] = chip("ГДЕ ГЕРОЙ?", GOLD_RGB)
        T["hero"] = hero_card()
        T["halftone"] = halftone()
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        self.diag = (xx * 0.8 + yy * 0.45) / (W * 0.8 + H * 0.45)   # 0 — левый верх, 1 — правый низ
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
        shake = 0.0
        for s0 in SHAKES:
            if s0 < t < s0 + 0.35:
                shake = (1 - (t - s0) / 0.35) * math.sin((t - s0) * 70)
        return mg.Cam(yaw + 0.5 * wobble(t, 1) + 1.2 * shake, dist, hgt + 0.02 * wobble(t, 2) + 0.03 * shake,
                      ty + 0.015 * wobble(t, 3), fov=fov, roll=0.5 * wobble(t * 0.8, 4) + 1.5 * shake,
                      tx=off * math.cos(a), tz=off * math.sin(a))

    def finish(self, frame, t):
        fr = super().finish(frame, t)
        for s in SLASH:  # комиксная шторка: растр наезжает по диагонали, под ним склейка, потом уезжает
            p = (t - s) / 0.3
            if 0 < p < 1:
                if p < 0.5:
                    e = smooth(p / 0.5) * 1.15
                    m = self.diag < e
                    edge = np.abs(self.diag - e) < 0.012
                else:
                    e = smooth((p - 0.5) / 0.5) * 1.15
                    m = self.diag > e
                    edge = np.abs(self.diag - e) < 0.012
                fr = fr.copy()
                fr[m] = self.T["halftone"][m]
                fr[edge] = (235, 235, 245)
        return fr

    def items_for(self, t, cam, it):
        T = self.T
        P = lambda y0, *a, **k: place(y0, *a, **k)  # noqa: E731

        # 1. Крючок (персонаж справа): газета крутится и влетает — «ГЕРОЙ СРЕДИ НАС?!»
        y0 = SEC["hook"]
        k = pop(t, 0.2, 3.4, 0.6, 0.25)
        if k > 0:
            spin = (1 - smooth(clamp((t - 0.2) / 0.6))) * 720
            pos = P(y0, -0.7, 1.62) + [0, 0.02 * math.sin(t * 3), 0]
            self.add_panel(it, cam, pos, face(y0, -12), 0.82 * k, 0.604 * k, 0.04 * k, T["news"], roll=spin,
                           pitch=-5, side=(50, 48, 56), edge=(220, 220, 230))

        # 2. Заставка
        k = pop(t, 3.6, 4.95, 0.35, 0.3)
        if k > 0:
            self.add_panel(it, cam, place(0, 0.0, 1.95, 0.35), face(0, -6), 1.2 * k, 0.564 * k, 0.11 * k,
                           T["stinger"], pitch=-6, side=(30, 24, 120), edge=(200, 190, 255))

        # 3. Суд (персонаж слева): «СЛАВА» зачёркивается → молоток бьёт → закон → наказание
        y0 = SEC["court"]
        r_, c_ = basis(y0)
        k = pop(t, 5.6, 7.6, 0.35, 0.3)
        if k > 0:
            self.add_bb(it, cam, P(y0, 0.75, 2.05), 0.13 * k, T["slava_x"] if t > 6.9 else T["slava"],
                        anchor=(0.5, 0.5))
        k = pop(t, 6.1, 7.6, 0.3, 0.3)
        if k > 0:
            ang = track([(6.1, 40.0), (6.75, 55.0, "out"), (7.0, -62.0, "lin"), (7.15, -55.0, "out"), (7.6, -58.0)], t)
            self.add_bb(it, cam, P(y0, 0.8, 1.2, 0.3), 0.95 * k,
                        self.cached("gav", lambda a: gavel_tex(T["gb"], a), round(ang / 3) * 3), anchor=(0.5, 0.5))
            if t > 7.0:
                ks = clamp((t - 7.0) / 0.15)
                self.add_bb(it, cam, P(y0, 0.75, 1.6, 0.35), 0.2 * (1 + 0.6 * (1 - ks)) * k, T["sud"],
                            anchor=(0.5, 0.5))
        k = pop(t, 7.65, 11.1, 0.4, 0.3)
        if k > 0:
            pos = P(y0, 0.72, 1.6) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 14), 0.9 * k, 0.427 * k, 0.06 * k, T["law"], pitch=-6,
                           side=(60, 40, 80), edge=(220, 200, 255))
        k = pop(t, 11.15, C1, 0.4, 0.2)
        if k > 0:
            pos = P(y0, 0.72, 1.6) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 14), 0.9 * k, 0.427 * k, 0.06 * k, T["punish"], pitch=-6,
                           side=(80, 40, 40), edge=(255, 200, 200))
            if t > 13.05:
                ks = clamp((t - 13.05) / 0.15)
                self.add_bb(it, cam, pos + c_ * 0.12 + [0, -0.4, 0], 0.15 * (1 + 0.6 * (1 - ks)) * k, T["good"],
                            anchor=(0.5, 0.5))

        # 4. Счета (персонаж справа, снизу): город за спиной → «битва» — здания проседают → чек с суммой
        y0 = SEC["bills"]
        r_, c_ = basis(y0)
        if C1 < t < C2:
            for i, (sd, fw, hh, w) in enumerate(((-1.25, -1.6, 1.9, 0.42), (-0.75, -2.1, 2.6, 0.5),
                                                  (-0.2, -2.4, 2.2, 0.46), (0.35, -2.2, 2.9, 0.44),
                                                  (0.9, -1.9, 2.0, 0.5), (1.4, -1.5, 1.6, 0.4))):
                kb = pop(t, C1 + 0.05 + 0.08 * i, C2 + 1, 0.5)
                fall = smooth((t - 20.85 - 0.12 * i) / 0.45) * (0.45 + 0.1 * (i % 3))
                q = P(y0, sd, 0.0, fw)
                self.add_box(it, cam, (q[0], q[2]), (w, w * 0.8), hh * kb * (1 - fall), (64, 58, 70),
                             radius=0.05, edge=(150, 140, 160), glow=(40, 90, 230) if fall > 0.05 else None)
        k = pop(t, 15.7, 19.2, 0.4, 0.3)
        if k > 0:
            pos = P(y0, -0.7, 1.6) + [0, 0.02 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, -12), 0.9 * k, 0.427 * k, 0.06 * k, T["kac"], pitch=8,
                           side=(40, 50, 90), edge=(200, 210, 255))
        k = pop(t, 19.3, C2, 0.45, 0.2)
        if k > 0:
            v = int(round(160 * smooth((t - 22.3) / 1.2)))
            pos = P(y0, -0.66, 1.5) + [0, 0.02 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, -12), 0.62 * k, 0.841 * k, 0.05 * k,
                           self.cached("rc", receipt_tex, v), pitch=8, side=(70, 60, 40), edge=(255, 230, 170))

        # 5. Эффект свидетеля (персонаж слева): человечки → карточка → столбики «помогли» → толпа ждёт героя
        y0 = SEC["crowd"]
        r_, c_ = basis(y0)
        if C2 < t < 31.7:
            kk = pop(t, C2 + 0.05, 31.7, 0.4, 0.3)
            self.add_bb(it, cam, P(y0, 1.05, 0.0, -0.3), 0.42 * kk, T["p_need"])
            for i, (sd, fw) in enumerate(((0.62, 0.1), (0.75, -0.5), (1.4, -0.4), (1.3, 0.2), (0.95, -0.9))):
                kp = pop(t, C2 + 0.15 + 0.1 * i, 31.7, 0.35, 0.3)
                self.add_bb(it, cam, P(y0, sd, 0.0, fw), 0.36 * kp, T["p_gray"])
            k = pop(t, 26.6, 28.55, 0.35, 0.3)
            if k > 0:
                self.add_bb(it, cam, P(y0, 0.85, 1.95), 0.11 * k, T["nohelp"], anchor=(0.5, 0.5))
            k = pop(t, 28.6, 31.7, 0.4, 0.3)
            if k > 0:
                pos = P(y0, 0.72, 1.85) + [0, 0.03 * math.sin(t * 2), 0]
                self.add_panel(it, cam, pos, face(y0, 14), 0.9 * k, 0.427 * k, 0.06 * k, T["bys"], pitch=-6,
                               side=(60, 40, 80), edge=(220, 200, 255))
        y0 = SEC["crowd2"]
        r_, c_ = basis(y0)
        k = pop(t, 31.75, 35.45, 0.4, 0.3)
        if k > 0:
            self.add_bb(it, cam, P(y0, 0.8, 2.15), 0.1 * k, T["helped"], anchor=(0.5, 0.5))
            for i, (sd, v) in enumerate(((0.55, 0.85), (0.9, 0.62), (1.25, 0.31))):
                kb = pop(t, 32.0 + 0.5 * i, 35.45, 0.5, 0.3)
                q = P(y0, sd, 0.0, -0.75)
                hh = 1.25 * v * kb
                self.add_box(it, cam, (q[0], q[2]), (0.26, 0.26), hh, (70, 150, 60) if i == 0 else
                             (60, 120, 70) if i == 1 else (60, 70, 160), radius=0.06, edge=(200, 255, 210))
                if kb > 0.05:
                    self.add_bb(it, cam, q + [0, hh + 0.12, 0], 0.11 * kb, T["pct"][i], anchor=(0.5, 0.5))
                    self.add_bb(it, cam, q + [0, 0.02, 0.0] + c_ * 0.2, 0.075 * kb, T["cnt"][i], anchor=(0.5, 1.0))
        if 35.45 < t < 38.6:
            for i, (sd, fw) in enumerate(((0.6, 0.1), (0.85, -0.4), (1.1, 0.0), (1.4, -0.5), (1.25, 0.35))):
                kp = pop(t, 35.5 + 0.08 * i, 38.6, 0.35, 0.25)
                self.add_bb(it, cam, P(y0, sd, 0.0, fw), 0.4 * kp, T["p_q"])
            k = pop(t, 36.4, 38.6, 0.35, 0.25)
            if k > 0:
                self.add_bb(it, cam, P(y0, 0.85, 2.1) + [0, 0.04 * math.sin(t * 3), 0], 0.12 * k, T["where"],
                            anchor=(0.5, 0.5))

        # 6. Финал (персонаж в центре, камера сверху опускается): «НАСТОЯЩИЙ ГЕРОЙ — ТОТ, КТО НЕ ЖДЁТ»
        y0 = SEC["hero"]
        k = pop(t, 38.75, 41.25, 0.45, 0.25)
        if k > 0:
            pos = place(y0, 0.0, 2.05, 0.2) + [0, 0.02 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 0), 0.95 * k, 0.464 * k, 0.06 * k, T["hero"], pitch=-12,
                           side=(30, 70, 40), edge=(190, 255, 210))

        # 7. Концовка с Telegram — общий шаблон
        self.outro_items(t, cam, it, OUTRO, SEC["tg"])


SFX = [(t, "whoosh") for t in (0.0, 3.5, 5.0) + SLASH + (31.4, 38.5, 41.0)]


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
        for t in (0.5, 1.5, 3.0, 4.2, 6.5, 7.1, 9.0, 12.0, 13.6, 14.1, 17.0, 20.5, 21.6, 24.0, 24.9, 25.08, 27.5,
                  30.0, 33.5, 35.0, 37.0, 39.5, 40.6, 44.0):
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
    base.mix_audio(tmp, voice, out, music=os.path.join(ROOT, "work", "hero", "music.wav"), sfx_list=SFX, dur=DUR,
                   sfx_vol=0.15)  # «вжух» лёгкий и тихий
    os.remove(tmp)


if __name__ == "__main__":
    main()
