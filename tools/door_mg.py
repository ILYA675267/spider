"""Ролик №5: «Почему ты заходишь на кухню и забываешь, зачем пришёл?» (эффект двери).

python tools/door_mg.py work/door/voice_clean.wav output/door.mp4 [--preview] [--jobs 4]
Общий движок, персонаж и концовка с Telegram — из memory_mg.py; параллельный рендер — как в money_mg.py.
Новое в монтаже: камера в начале пролетает сквозь открывающуюся 3D-дверь; заставка с поворотом кадра
вокруг оси; смена сцен «дверью» — шторка-дверь закрывает экран и открывает уже новую сцену (жёсткая склейка);
кинокадр с чёрными полосами на «сценах, как в кино»; кинолента, хлопушка, архив, мысль улетает в архив;
облачко со словами пролетает сквозь дверь.
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
OUTRO = float(open(os.path.join(ROOT, "work", "door", "outro_start.txt")).read())  # 39.09
DUR = 47.75
WARM_RGB, TEAL_RGB = (240, 160, 70), (40, 190, 170)
WARM = WARM_RGB[::-1]

# ---------------------------------------------------------------- субтитры
T_, W_, C_ = TG_RGB, WARM_RGB, TEAL_RGB
SUBS = [
    (0.224, "ПОЧЕМУ ТЫ ЗАХОДИШЬ НА КУХНЮ", ("КУХНЮ",), W_), (1.604, "И ЗАБЫВАЕШЬ,", ("ЗАБЫВАЕШЬ",)),
    (2.424, "ЗАЧЕМ ПРИШЁЛ?", ("ЗАЧЕМ",)), (3.3, "", ()),
    (4.764, "ЭТО НЕ СКЛЕРОЗ", ("СКЛЕРОЗ",)), (5.484, "И НЕ УСТАЛОСТЬ", ("УСТАЛОСТЬ",)),
    (6.384, "ВИНОВАТА ДВЕРЬ", ("ДВЕРЬ",), W_), (7.664, "ПСИХОЛОГИ ИЗ УНИВЕРСИТЕТА", ()),
    (9.064, "НОТР-ДАМ ПРОВЕРИЛИ:", ("НОТР-ДАМ",), BLUE_RGB), (10.304, "ЛЮДИ НОСИЛИ ПРЕДМЕТЫ", ()),
    (11.404, "ИЗ КОМНАТЫ В КОМНАТУ", ()), (12.364, "И ЗАБЫВАЛИ БОЛЬШЕ,", ("БОЛЬШЕ",)),
    (13.504, "КОГДА ПРОХОДИЛИ", ()), (14.184, "ЧЕРЕЗ ДВЕРНОЙ ПРОЁМ", ("ДВЕРНОЙ",), W_),
    (15.752, "МОЗГ ДЕЛИТ ЖИЗНЬ НА СЦЕНЫ,", ("СЦЕНЫ",), W_), (17.412, "КАК В КИНО", ("КИНО",), W_),
    (18.112, "ДВЕРЬ ДЛЯ НЕГО —", ("ДВЕРЬ",)), (18.992, "ЭТО СМЕНА СЦЕНЫ", ("СМЕНА",), W_),
    (20.232, "ОН СОХРАНЯЕТ ПРОШЛУЮ СЦЕНУ", ()), (21.752, "В АРХИВ", ("АРХИВ",), W_),
    (22.152, "И ОСВОБОЖДАЕТ МЕСТО", ()), (23.272, "ДЛЯ НОВОЙ", ("НОВОЙ",)),
    (24.492, "А ВМЕСТЕ С НЕЙ В АРХИВ", ("АРХИВ",), W_), (25.592, "УЛЕТАЕТ И ТВОЯ МЫСЛЬ", ("МЫСЛЬ",)),
    (27.308, "КАК ВСПОМНИТЬ?", ("ВСПОМНИТЬ?",), C_), (28.588, "ВЕРНИСЬ ТУДА, ГДЕ БЫЛА МЫСЛЬ", ("ВЕРНИСЬ",), C_),
    (30.508, "ОБСТАНОВКА ПОДСКАЖЕТ", ("ОБСТАНОВКА",), C_), (31.928, "А ЛУЧШЕ ПЕРЕД ДВЕРЬЮ", ()),
    (32.868, "СКАЖИ ВСЛУХ, ЗАЧЕМ ИДЁШЬ", ("ВСЛУХ",), C_), (34.828, "ТОГДА МЫСЛЬ ПРОЙДЁТ", ("МЫСЛЬ",), C_),
    (35.788, "ЧЕРЕЗ ДВЕРЬ ВМЕСТЕ С ТОБОЙ", ("ДВЕРЬ",), C_), (37.328, "РАЗ УЖ ТЫ ЗАБЫВАКА", ("ЗАБЫВАКА",)),
    (39.06, "А ЕЩЁ БОЛЬШЕ ИНТЕРЕСНОГО —", (), T_), (40.42, "В МОЁМ ТЕЛЕГРАМ-КАНАЛЕ", ("ТЕЛЕГРАМ-КАНАЛЕ",), T_),
    (41.75, "«ПАУЧЬЯ НОРКА»", ("«ПАУЧЬЯ", "НОРКА»"), T_), (42.87, "ТАМ ЭКСКЛЮЗИВ,", ("ЭКСКЛЮЗИВ",), T_),
    (43.57, "КОТОРОГО НЕТ В РОЛИКАХ", (), T_), (44.87, "ССЫЛКА В ПРОФИЛЕ —", ("ПРОФИЛЕ",), T_),
    (45.89, "ЗАЛЕТАЙ!", ("ЗАЛЕТАЙ!",), T_),
]

# ---------------------------------------------------------------- камера и сцены
WIPES = (15.45, 26.95)  # шторка-дверь: закрывает экран, под ней жёсткая склейка, открывает новую сцену
CUT1, CUT2 = WIPES[0] + 0.25, WIPES[1] + 0.25
CAMERA = [
    (0.0, (0.0, 6.2, 1.15, 1.0, 50.0)),                    # из коридора…
    (1.45, (0.0, 3.6, 1.2, 0.86, 50.0), "inout"),          # …сквозь дверь на кухню
    (3.3, (-4.0, 3.6, 1.2, 0.86, 50.0)),
    (4.05, (0.0, 3.9, 1.3, 0.98, 50.0), "inout"),          # заставка: кадр поворачивается вокруг оси
    (4.6, (0.0, 3.9, 1.3, 0.98, 50.0)),
    (4.95, (95.0, 3.6, 1.2, 0.84, 50.0), "expo"),          # не склероз — виновата дверь, опыт
    (CUT1 - 0.01, (103.0, 3.5, 1.2, 0.86, 50.0)),
    (CUT1 + 0.01, (190.0, 3.4, 0.75, 0.95, 48.0), "lin"),  # склейка: сцены, как в кино (снизу)
    (CUT2 - 0.01, (197.0, 3.25, 0.8, 0.95, 48.0)),
    (CUT2 + 0.01, (280.0, 3.6, 1.2, 0.84, 50.0), "lin"),   # склейка: как вспомнить
    (38.6, (300.0, 3.7, 1.25, 0.86, 50.0), "lin"),
    (39.0, (365.0, 4.3, 1.3, 1.12, 50.0), "expo"), (DUR, (366.0, 4.0, 1.3, 1.15, 50.0)),
]
# >0 — персонаж слева, вставки справа; <0 — наоборот
OFFSET = [(0.0, -0.5), (3.35, -0.5), (4.05, 0.0, "inout"), (4.6, 0.0), (4.95, 0.55, "expo"), (CUT1 - 0.01, 0.55),
          (CUT1 + 0.01, -0.55, "lin"), (CUT2 - 0.01, -0.55), (CUT2 + 0.01, 0.55, "lin"), (38.6, 0.55),
          (39.0, 0.0, "expo"), (DUR, 0.0)]
ROLL = [(0.0, 0.0), (3.35, 0.0), (4.05, 360.0, "expo"), (DUR, 360.0)]
TINT = [(0.0, (0.92, 1.0, 1.1)), (4.6, (0.92, 1.0, 1.1)), (5.0, (1.08, 1.0, 0.92)), (CUT1 - 0.01, (1.08, 1.0, 0.92)),
        (CUT1 + 0.01, (0.84, 0.98, 1.14), "lin"), (CUT2 - 0.01, (0.84, 0.98, 1.14)),
        (CUT2 + 0.01, (1.04, 1.1, 0.86), "lin"), (38.7, (1.04, 1.1, 0.86)), (39.1, (1.0, 1.0, 1.0)), (DUR, (1.0, 1.0, 1.0))]
LETTERBOX = (CUT1, CUT2)  # чёрные полосы кино
SEC = {"hook": 0, "door": 99, "film": 193, "fix": 288, "tg": 365.5}
SHAKES = (6.9,)
FLASHES = ((4.0, 0.15), (4.85, 0.08), (6.85, 0.12), (19.15, 0.1), (38.85, 0.12))
JUMPS = [37.35]
ALARM = None
SPOT = None

G = lambda *k: list(k)  # noqa: E731
ARM_L = G((0, 62), (0.9, 62), (1.15, -25, "back"), (1.6, -22), (1.75, 36, "back"), (3.1, 38), (3.35, -50, "back"),
          (4.0, -45), (4.4, 62, "back"), (15.6, 62), (CUT1 + 0.05, 62), (15.9, -25, "back"), (17.9, -22),
          (18.1, -10, "back"), (20.1, -8), (20.3, -38, "back"), (22.3, -36), (22.5, -15, "back"), (24.4, -12),
          (24.6, -30, "back"), (26.7, -28), (26.95, 62), (46.0, 62), (46.15, -50, "back"), (46.75, -45),
          (47.15, 62, "back"), (DUR, 62))
ARM_R = G((0, 62), (1.75, 62), (1.9, 36, "back"), (3.1, 38), (3.35, -50, "back"), (4.0, -45), (4.4, 62, "back"),
          (4.75, 30, "back"), (6.1, 32), (6.35, -35, "back"), (7.5, -32), (7.7, -15, "back"), (10.2, -12),
          (10.4, -30, "back"), (15.2, -28), (15.5, 62), (CUT2 + 0.05, 62), (27.4, -40, "back"), (28.4, -38),
          (28.6, -10, "back"), (31.7, -8), (31.95, -30, "back"), (34.6, -28), (34.85, -12, "back"), (37.1, -10),
          (37.3, 36, "back"), (38.6, 38), (38.85, 62), (39.31, -48, "back"), (41.66, -45), (41.96, -55, "back"),
          (44.66, -52), (44.91, -40, "back"), (45.86, -42), (46.01, -50, "back"), (46.75, -45), (46.9, -58),
          (47.05, -32), (47.2, -58), (47.35, -32), (DUR, 62))
HEAD = G((0, 0), (1.1, -6), (1.8, 9), (3.1, 8), (3.4, 0), (4.8, -8), (6.1, 6), (6.4, -6), (7.7, -5), (10.4, -7),
         (12.4, 5), (15.2, 0), (15.9, 7), (18.1, 6), (20.3, 8), (22.5, 5), (24.6, 7), (26.6, 4), (27.4, -6),
         (28.6, -5), (31.95, -7), (34.85, -6), (37.3, 9), (38.7, 6), (39.31, -6), (46.0, 0), (DUR, 0))


# ---------------------------------------------------------------- картинки вставок
def door_tex():
    """Тёмная филёнчатая дверь с круглой ручкой."""
    im = rounded((400, 760), 18, (46, 38, 40, 255), (90, 76, 70, 255), 6)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([50, 50, 350, 330], 16, outline=(78, 64, 60, 255), width=8)
    d.rounded_rectangle([50, 400, 350, 710], 16, outline=(78, 64, 60, 255), width=8)
    d.ellipse([300, 360, 350, 410], fill=(200, 160, 90, 255))
    return mg.premul(im)


def thought_tex(top, big):
    im = Image.new("RGBA", (700, 300), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([60, 0, 699, 210], 60, fill=(24, 24, 32, 255), outline=(110, 100, 90, 255), width=4)
    d.ellipse([20, 210, 80, 260], fill=(24, 24, 32, 255), outline=(110, 100, 90, 255), width=4)
    d.ellipse([0, 262, 30, 292], fill=(24, 24, 32, 255), outline=(110, 100, 90, 255), width=4)
    d.text((110, 26), top, font=mg.font(50), fill=(170, 170, 190))
    d.text((110, 92), big, font=mg.font(84, tall=True), fill="white")
    return mg.premul(im)


def crossed(text):
    """Плашка с зачёркнутым словом («не склероз»)."""
    f = mg.font(54)
    tw = ImageDraw.Draw(Image.new("RGBA", (10, 10))).textbbox((0, 0), text, font=f)[2]
    im = rounded((tw + 80, 110), 30, (20, 20, 28, 255), GRAY_RGB + (255,), 5)
    d = ImageDraw.Draw(im)
    d.text((40, 22), text, font=f, fill=(180, 180, 195))
    d.line([(26, 70), (tw + 54, 40)], fill=RED_RGB + (255,), width=10)
    return mg.premul(im)


def exp_tex(p, bars):
    """Схема опыта: две комнаты с дверью; человечек несёт коробку; bars — сравнение «забыли»."""
    im = rounded((780, 600), 34, (18, 18, 26, 255), (72, 72, 92, 255), 3)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([36, 30, 280, 92], 30, fill=BLUE_RGB + (255,))
    d.text((58, 38), "ОПЫТ, 2011", font=mg.font(36), fill="white")
    x0, y0, x1, y1 = 60, 120, 720, 330
    d.rounded_rectangle([x0, y0, x1, y1], 20, outline=(120, 120, 140, 255), width=6)
    xm = (x0 + x1) // 2
    d.line([(xm, y0), (xm, 190)], fill=(120, 120, 140), width=6)
    d.line([(xm, 262), (xm, y1)], fill=(120, 120, 140), width=6)
    d.rounded_rectangle([xm - 8, 190, xm + 8, 262], 4, fill=WARM_RGB + (255,))  # проём
    x = int(120 + (x1 - 180) * p)
    d.ellipse([x - 22, 204, x + 22, 248], fill=TG_RGB + (255,))
    d.rounded_rectangle([x + 18, 196, x + 50, 228], 6, fill=(200, 160, 90, 255))
    if bars > 0:
        for i, (txt, v, col) in enumerate((("в одной комнате", 0.38, GRAY_RGB), ("через дверь", 0.78, RED_RGB))):
            y = 360 + i * 110
            d.text((60, y), txt, font=mg.font(38), fill=(200, 200, 215))
            d.rounded_rectangle([60, y + 52, 720, y + 90], 19, fill=(40, 40, 52, 255))
            w = int(660 * v * bars)
            if w > 40:
                d.rounded_rectangle([60, y + 52, 60 + w, y + 90], 19, fill=col + (255,))
        d.text((470, 360), "ЗАБЫЛИ", font=mg.font(38), fill=(255, 140, 140))
    return mg.premul(im)


def film_tex(hl):
    """Кинолента из трёх кадров-сцен; hl — какой кадр подсвечен (0..3)."""
    Wf, Hf = 960, 360
    im = rounded((Wf, Hf), 30, (14, 14, 18, 255), (70, 70, 84, 255), 3)
    d = ImageDraw.Draw(im)
    for x in range(30, Wf - 30, 60):
        d.rounded_rectangle([x, 16, x + 30, 40], 6, fill=(60, 60, 72, 255))
        d.rounded_rectangle([x, Hf - 40, x + 30, Hf - 16], 6, fill=(60, 60, 72, 255))
    names = [("СЦЕНА 1", "комната"), ("СЦЕНА 2", "дверь"), ("СЦЕНА 3", "кухня")]
    for i, (a, b) in enumerate(names):
        x = 30 + i * 305
        on = i < hl
        d.rounded_rectangle([x, 60, x + 290, Hf - 60], 18, fill=(52, 44, 36, 255) if on else (30, 30, 38, 255),
                            outline=WARM_RGB + (255,) if on else (60, 60, 72, 255), width=4)
        d.text((x + 24, 90), a, font=mg.font(44, tall=True), fill="white" if on else (110, 110, 125))
        d.text((x + 24, 170), b, font=mg.font(40), fill=(240, 200, 150) if on else (100, 100, 115))
    return mg.premul(im)


def clap_tex():
    im = rounded((620, 380), 24, (24, 24, 30, 255), (90, 90, 104, 255), 4)
    d = ImageDraw.Draw(im)
    d.text((40, 40), "СЦЕНА 2", font=mg.font(100, tall=True), fill="white")
    d.text((44, 190), "смена сцены", font=mg.font(48), fill=(200, 200, 215))
    d.line([(40, 280), (580, 280)], fill=(70, 70, 84), width=4)
    d.text((44, 300), "ДУБЛЬ 1", font=mg.font(44), fill=(160, 160, 175))
    return mg.premul(im)


def stick_tex():
    im = rounded((620, 90), 16, (230, 230, 236, 255))
    d = ImageDraw.Draw(im)
    for i in range(-1, 9):
        x = i * 80
        d.polygon([(x, 0), (x + 40, 0), (x + 80, 90), (x + 40, 90)], fill=(26, 26, 32, 255))
    return mg.premul(im)


def archive_tex():
    im = rounded((420, 150), 24, (40, 34, 30, 255), WARM_RGB + (255,), 5)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([150, 22, 270, 44], 10, fill=(90, 76, 64, 255))  # ручка ящика
    d.text((86, 62), "АРХИВ", font=mg.font(70, tall=True), fill=(240, 200, 150))
    return mg.premul(im)


def frame_tex(text, sub, on=True):
    im = rounded((520, 330), 24, (52, 44, 36, 255) if on else (26, 26, 34, 255),
                 WARM_RGB + (255,) if on else (120, 120, 140, 255), 5)
    d = ImageDraw.Draw(im)
    d.text((36, 40), text, font=mg.font(84, tall=True), fill="white")
    d.text((40, 180), sub, font=mg.font(44), fill=(240, 200, 150) if on else (150, 150, 170))
    return mg.premul(im)


def pin_tex():
    im = Image.new("RGBA", (200, 260), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([10, 0, 190, 180], fill=TEAL_RGB + (255,))
    d.polygon([(40, 140), (160, 140), (100, 255)], fill=TEAL_RGB + (255,))
    d.ellipse([60, 50, 140, 130], fill=(16, 40, 36, 255))
    return mg.premul(im)


def speech_tex(text):
    f = mg.font(56)
    tw = ImageDraw.Draw(Image.new("RGBA", (10, 10))).textbbox((0, 0), text, font=f)[2]
    im = Image.new("RGBA", (tw + 90, 170), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, tw + 89, 120], 50, fill=TEAL_RGB + (255,))
    d.polygon([(40, 112), (90, 112), (30, 168)], fill=TEAL_RGB + (255,))
    d.text((45, 26), text, font=f, fill="white")
    return mg.premul(im)


# ---------------------------------------------------------------- сцена
class Scene(base.Scene):
    cfg = sys.modules[__name__]

    def __init__(self, voice):
        super().__init__(voice)
        T = self.T
        T["door"] = door_tex()
        T["why"] = thought_tex("а зачем я…", "ПРИШЁЛ?")
        T["kitchen"] = chip("КУХНЯ", WARM_RGB)
        T["no1"], T["no2"] = crossed("СКЛЕРОЗ"), crossed("УСТАЛОСТЬ")
        T["guilty"] = stamp("ВИНОВАТА ДВЕРЬ", WARM_RGB)
        T["study"] = card("ИССЛЕДОВАНИЕ", "УНИВЕРСИТЕТ НОТР-ДАМ", "США, психолог Габриэль Радванский", BLUE_RGB)
        T["clap"], T["stick"] = clap_tex(), stick_tex()
        T["archive"] = archive_tex()
        T["old"] = frame_tex("СЦЕНА 1", "комната")
        T["new"] = frame_tex("СЦЕНА 2", "новая", on=False)
        T["idea"] = thought_tex("я иду, чтобы…", "ВЗЯТЬ ЗАРЯДКУ")
        T["lost"] = stamp("МЫСЛЬ В АРХИВЕ", WARM_RGB)
        T["how"] = chip("КАК ВСПОМНИТЬ?", TEAL_RGB)
        T["pin"] = pin_tex()
        T["place"] = card("СОВЕТ 1", "ВЕРНИСЬ НАЗАД", "обстановка подскажет", TEAL_RGB)
        T["say"] = speech_tex("Иду за зарядкой!")
        T["forget"] = stamp("ЗАБЫВАКА", TEAL_RGB)
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
                      ty + 0.015 * wobble(t, 3), fov=fov, roll=track(ROLL, t) + 0.5 * wobble(t * 0.8, 4) + 1.5 * shake,
                      tx=off * math.cos(a), tz=off * math.sin(a))

    def finish(self, frame, t):
        fr = super().finish(frame, t)
        k = pop(t, LETTERBOX[0] + 0.1, LETTERBOX[1], 0.5, 0.35)  # кино: чёрные полосы сверху и снизу
        if k > 0:
            hb = int(190 * min(1.0, k))
            fr[:hb] = (fr[:hb] * 0.08).astype(np.uint8)
            fr[H - hb:] = (fr[H - hb:] * 0.08).astype(np.uint8)
        for w0 in WIPES:  # шторка-дверь
            if w0 < t < w0 + 0.5:
                p = (t - w0) / 0.5
                x = int(W * (smooth(p * 2) - 1)) if p < 0.5 else int(W * smooth(p * 2 - 1))
                self.wipe(fr, x)
        return fr

    def wipe(self, fr, x):
        """Дверь во весь экран, левый край в x (сдвигается слева направо)."""
        a, b = max(0, x), min(W, x + W)
        if a >= b:
            return
        lay = np.zeros((H, W, 3), np.uint8)
        lay[:] = (40, 36, 46)
        for (y0, y1) in ((140, 820), (980, 1780)):
            cv2.rectangle(lay, (110, y0), (W - 110, y1), (64, 56, 62), 14, cv2.LINE_AA)
        cv2.circle(lay, (W - 150, 900), 34, (90, 160, 200), -1, cv2.LINE_AA)
        cv2.line(lay, (0, 0), (0, H), (120, 110, 120), 12)
        fr[:, a:b] = lay[:, a - x:b - x]

    # --- 3D-дверь: косяки, перекладина и полотно на петлях
    def door(self, it, cam, cx, cz, u, width, height, ang, k=1.0, leaf=True):
        if k <= 0.01:
            return
        u = np.asarray(u, float) / np.linalg.norm(u)
        c = np.array([cx, 0.0, cz])
        hh = height * k
        yaw = math.degrees(math.atan2(u[2], u[0]))
        for s in (-1, 1):
            p = c + u * s * width / 2
            self.add_box(it, cam, (p[0], p[2]), (0.08, 0.08), hh, (70, 60, 64), radius=0.02)
        self.add_box(it, cam, (cx, cz), (width + 0.16, 0.08), 0.09 * k, (70, 60, 64), yaw=yaw, base_y=hh, radius=0.02)
        if leaf:
            th = math.radians(ang)
            dvec = np.array([u[0] * math.cos(th) - u[2] * math.sin(th), 0.0, u[0] * math.sin(th) + u[2] * math.cos(th)])
            hinge = c - u * width / 2
            cen = hinge + dvec * width * 0.48 + [0, hh * 0.49, 0]
            pyaw = math.degrees(math.atan2(-dvec[2], dvec[0]))
            w_, h_ = width * 0.96 / 1.25, hh * 0.98 / 1.25  # add_panel увеличивает в 1,25 раза
            self.add_panel(it, cam, cen, pyaw, w_, h_, 0.03, self.T["door"], back_tex=self.T["door"],
                           side=(60, 50, 52), radius=0.02)

    def light(self, it, cam, x, z, s):
        it.append((cam.depth((x, 0.0, z)) + 0.02, lambda: mg.floor_blob(self.frame, cam, (x, z), 0.5, 0.35, 0.5 * s,
                                                                        color=(120, 190, 255), blur=81, mode="light")))

    def items_for(self, t, cam, it):
        T = self.T
        P = lambda y0, *a, **k: place(y0, *a, **k)  # noqa: E731

        # 1. Крючок: камера пролетает сквозь дверь на кухню; холодильник; «а зачем я… пришёл?»
        y0 = SEC["hook"]
        r_, c_ = basis(y0)
        if t < 2.0:
            dc = r_ * -0.5 + c_ * 2.3
            ang = -100 * smooth((t - 0.1) / 0.6)
            self.door(it, cam, dc[0], dc[2], r_, 1.0, 1.75, ang)
        k = pop(t, 0.6, 3.4, 0.5, 0.3)
        if k > 0:
            p = r_ * -1.05 + c_ * -0.45
            self.add_box(it, cam, (p[0], p[2]), (0.5, 0.45), 1.35 * k, (78, 72, 80), radius=0.08)
            self.add_bb(it, cam, p + c_ * 0.25 + [0, 1.35 * k + 0.03, 0], 0.1 * k, T["kitchen"], anchor=(0.5, 1.0))
        k = pop(t, 1.7, 3.4, 0.35, 0.3)
        if k > 0:
            pos = P(y0, -0.72, 1.75) + [0, 0.02 * math.sin(t * 3), 0]
            self.add_panel(it, cam, pos, face(y0, -10), 0.75 * k, 0.32 * k, 0.03 * k, T["why"], side=(40, 36, 44),
                           radius=0.06)

        # 2. Заставка «Паучок объясняет» (после поворота кадра)
        k = pop(t, 3.8, 4.85, 0.35, 0.3)
        if k > 0:
            self.add_panel(it, cam, place(0, 0.0, 1.95, 0.35), face(0, -6), 1.2 * k, 0.564 * k, 0.11 * k,
                           T["stinger"], pitch=-6, side=(30, 24, 120), edge=(200, 190, 255))

        # 3. Не склероз, не усталость — виновата дверь; опыт Нотр-Дам
        y0 = SEC["door"]
        r_, c_ = basis(y0)
        for key, t0, x0, hy in (("no1", 4.95, 0.62, 2.0), ("no2", 5.55, 0.95, 1.72)):
            k = pop(t, t0, 6.4, 0.3, 0.25)
            if k > 0:
                self.add_bb(it, cam, P(y0, x0, hy), 0.12 * k, T[key], anchor=(0.5, 0.5))
        k = pop(t, 6.3, 10.3, 0.5, 0.35)
        if k > 0:
            dc = r_ * 0.95 + c_ * -0.6
            self.door(it, cam, dc[0], dc[2], r_, 0.75, 1.3, -95 * smooth((t - 6.75) / 0.25), k)
            if t > 6.8:
                self.light(it, cam, dc[0] + c_[0] * 0.3, dc[2] + c_[2] * 0.3, k * smooth((t - 6.8) / 0.3))
        k = pop(t, 6.85, 7.65, 0.15, 0.25)
        if k > 0:
            ks = clamp((t - 6.85) / 0.15)
            self.add_bb(it, cam, P(y0, 0.85, 1.65, 0.1), 0.2 * (1 + 0.6 * (1 - ks)) * k, T["guilty"], anchor=(0.5, 0.5))
        k = pop(t, 7.65, 10.3, 0.4, 0.3)
        if k > 0:
            pos = P(y0, 0.72, 1.82) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 14), 0.9 * k, 0.48 * k, 0.06 * k, T["study"], pitch=-6,
                           side=(40, 50, 90), edge=(200, 210, 255))
        k = pop(t, 10.25, 15.45, 0.4, 0.2)
        if k > 0:
            p = round(smooth((t - 10.6) / 1.8) * 24) / 24
            bars = round(smooth((t - 12.6) / 0.8) * 16) / 16
            pos = P(y0, 0.72, 1.55) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 12), 0.9 * k, 0.692 * k, 0.06 * k, self.cached("exp", exp_tex, p, bars),
                           pitch=-5, side=(40, 50, 90), edge=(200, 210, 255))

        # 4. Сцены, как в кино (персонаж справа, съёмка снизу): кинолента → хлопушка → архив, мысль улетает туда же
        y0 = SEC["film"]
        r_, c_ = basis(y0)
        k = pop(t, CUT1 + 0.1, 18.95, 0.4, 0.3)
        if k > 0:
            hl = 0 + (t > 16.3) + (t > 16.9) + (t > 17.5)
            pos = P(y0, -0.72, 1.62) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, -12), 0.95 * k, 0.356 * k, 0.05 * k, self.cached("film", film_tex, hl),
                           pitch=6, side=(60, 50, 40), edge=(255, 220, 180))
        k = pop(t, 18.85, 20.3, 0.35, 0.3)
        if k > 0:
            pos = P(y0, -0.72, 1.5) + [0, 0.02 * math.sin(t * 2), 0]
            bw, bh = 0.62 * k, 0.38 * k
            self.add_panel(it, cam, pos, face(y0, -12), bw, bh, 0.04, T["clap"], pitch=6, side=(50, 50, 60), radius=0.03)
            rr = np.array([math.cos(math.radians(y0 + 12)), 0.0, math.sin(math.radians(y0 + 12))])
            # палка хлопушки: открыта, хлопает на «смена» и на «сцены»
            phi = 28 * (1 - smooth((t - 19.0) / 0.12)) + 28 * smooth((t - 19.2) / 0.15) * (1 - smooth((t - 19.42) / 0.1))
            hinge = pos - rr * bw * 1.25 / 2 + [0, bh * 1.25 / 2 + 0.06 * k, 0]
            ph = math.radians(phi)
            cen = hinge + (rr * math.cos(ph) + np.array([0, 1.0, 0]) * math.sin(ph)) * bw * 1.25 / 2
            self.add_panel(it, cam, cen, face(y0, -12), bw, 0.09 * k, 0.04, T["stick"], roll=phi, side=(200, 200, 205),
                           radius=0.02)
        ka = pop(t, 20.2, 27.0, 0.45, 0.2)
        box = r_ * -1.05 + c_ * -0.95
        if ka > 0:
            self.add_box(it, cam, (box[0], box[2]), (0.55, 0.4), 0.42 * ka, (60, 50, 44), radius=0.06, edge=(120, 100, 80))
            self.add_panel(it, cam, box + c_ * 0.205 + [0, 0.21 * ka, 0], face(y0, 0), 0.4 * ka, 0.143 * ka, 0.02,
                           T["archive"], side=(60, 50, 44), radius=0.03)
        top = box + [0, 0.45, 0]
        if 20.3 < t < 22.4:  # старая сцена улетает в архив
            f = smooth((t - 21.6) / 0.6)
            k = out_back(clamp((t - 20.3) / 0.4))
            pos = P(y0, -0.72, 1.6) * (1 - f) + top * f
            if f < 0.97:
                self.add_panel(it, cam, pos, face(y0, -12 + 30 * f), 0.5 * k * (1 - 0.8 * f), 0.317 * k * (1 - 0.8 * f),
                               0.03, T["old"], side=(60, 50, 40), radius=0.03)
        k = pop(t, 22.45, 24.45, 0.4, 0.3)
        if k > 0:
            pos = P(y0, -0.72, 1.6) + [0, 0.02 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, -12), 0.5 * k, 0.317 * k, 0.03, T["new"], side=(40, 40, 50), radius=0.03)
        if 24.5 < t < 26.4:  # и мысль туда же
            f = smooth((t - 25.6) / 0.6)
            k = out_back(clamp((t - 24.5) / 0.4))
            pos = P(y0, -0.75, 1.65) * (1 - f) + top * f
            if f < 0.97:
                self.add_panel(it, cam, pos, face(y0, -10 + 40 * f), 0.75 * k * (1 - 0.85 * f), 0.32 * k * (1 - 0.85 * f),
                               0.03, T["idea"], side=(40, 36, 44), radius=0.06)
        k = pop(t, 26.25, 27.0, 0.15, 0.15)
        if k > 0:
            ks = clamp((t - 26.25) / 0.15)
            self.add_bb(it, cam, P(y0, -0.8, 1.2, 0.1), 0.13 * (1 + 0.6 * (1 - ks)) * k, T["lost"], anchor=(0.5, 0.5))

        # 5. Как вспомнить: путь назад по полу → «скажи вслух» → облачко пролетает сквозь дверь → «забывака»
        y0 = SEC["fix"]
        r_, c_ = basis(y0)
        k = pop(t, CUT2 + 0.1, 28.6, 0.3, 0.25)
        if k > 0:
            self.add_bb(it, cam, P(y0, 0.78, 1.95), 0.13 * k, T["how"], anchor=(0.5, 0.5))
        k = pop(t, 28.6, 31.8, 0.4, 0.3)
        if k > 0:
            goal = r_ * 1.05 + c_ * -1.3
            it.append((cam.depth(goal) + 0.5, lambda k=k, goal=goal: self.path(cam, goal, t, k)))
            self.add_bb(it, cam, goal + [0, 0.02 * math.sin(t * 4), 0], 0.28 * k, T["pin"], anchor=(0.5, 1.0))
        k = pop(t, 30.4, 31.85, 0.35, 0.3)
        if k > 0:
            pos = P(y0, 0.72, 1.85) + [0, 0.02 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 12), 0.9 * k, 0.48 * k, 0.06 * k, T["place"], pitch=-6,
                           side=(30, 70, 60), edge=(190, 255, 235))
        kd = pop(t, 32.0, 37.3, 0.45, 0.3)
        u = r_ * 0.5 + c_ * 0.866
        n = r_ * 0.866 - c_ * 0.5
        dc = r_ * 1.0 + c_ * -0.35
        if kd > 0:
            self.door(it, cam, dc[0], dc[2], u, 0.7, 1.25, -80 * smooth((t - 32.4) / 0.4), kd)
        if 32.9 < t < 37.3:
            k = out_back(clamp((t - 32.9) / 0.35)) * (1 - smooth((t - 36.9) / 0.35))
            a = P(y0, 0.62, 1.68)
            b = dc - n * 0.5 + [0, 0.8, 0]
            c = dc + n * 0.3 + [0, 0.8, 0]
            f1, f2 = smooth((t - 34.8) / 0.6), smooth((t - 35.6) / 0.8)
            pos = a * (1 - f1) + b * f1
            pos = pos * (1 - f2) + c * f2
            self.add_bb(it, cam, pos, 0.15 * k, T["say"], anchor=(0.5, 0.5))
        k = pop(t, 37.4, 38.85, 0.2, 0.3)
        if k > 0:
            ks = clamp((t - 37.4) / 0.15)
            self.add_bb(it, cam, P(y0, 0.85, 1.75, 0.1), 0.16 * (1 + 0.6 * (1 - ks)) * k, T["forget"], anchor=(0.5, 0.5))

        # 6. Концовка с Telegram — общий шаблон
        self.outro_items(t, cam, it, OUTRO, SEC["tg"])

    def path(self, cam, goal, t, k):
        """Светящаяся пунктирная дорожка по полу от паучка к месту, где была мысль."""
        g = np.asarray(goal)
        ctrl = g * 0.5 + np.array([g[2], 0, -g[0]]) * 0.25
        prog = smooth((t - 28.7) / 0.8)
        lay = np.zeros_like(self.frame)
        n = 24
        for i in range(n):
            s0, s1 = i / n, (i + 0.5) / n
            if s0 > prog:
                break
            pts = []
            for s in np.linspace(s0, min(s1, prog), 4):
                q = (1 - s) ** 2 * np.array([0.0, 0, 0]) + 2 * (1 - s) * s * ctrl + s * s * g
                pts.append([q[0], 0.02, q[2]])
            uv, z = cam.proj(np.array(pts))
            if (z > 0.3).all():
                mg.lines(lay, uv, tuple(int(c * k) for c in TEAL_RGB[::-1]), 7, closed=False)
        cv2.add(self.frame, cv2.GaussianBlur(lay, (0, 0), 5), self.frame)
        cv2.add(self.frame, lay, self.frame)


SFX = [(t, "whoosh") for t in (0.2, 3.35, 4.8, WIPES[0], WIPES[1], 38.85)]


def render(voice, out, frames):
    sc = Scene(voice)
    wr = mg.Writer(out, FPS)
    for n, i in enumerate(frames):
        t = i / FPS
        speed = abs(sc.cam(t + 0.5 / FPS).yaw - sc.cam(t - 0.5 / FPS).yaw) * FPS
        rspeed = abs(track(ROLL, t + 0.5 / FPS) - track(ROLL, t - 0.5 / FPS)) * FPS
        cut = any(abs(t - c) < 1.0 / FPS for c in (CUT1, CUT2))  # склейка спрятана под шторкой — без смешивания
        sub = 5 if (speed > math.radians(60) or rspeed > 60) and not cut else 1
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
        for t in (0.2, 0.6, 1.0, 2.6, 3.6, 4.3, 5.8, 7.1, 9.0, 11.5, 13.6, 15.6, 16.6, 18.4, 19.1, 19.3, 21.0, 22.0,
                  23.5, 25.2, 26.5, 27.1, 28.0, 30.0, 31.0, 33.5, 35.5, 36.2, 38.0, 41.0, 46.0):
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
    base.mix_audio(tmp, voice, out, music=os.path.join(ROOT, "work", "door", "music.wav"), sfx_list=SFX, dur=DUR,
                   sfx_vol=0.15)  # «вжух» лёгкий и тихий
    os.remove(tmp)


if __name__ == "__main__":
    main()
