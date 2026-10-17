"""Ролик №11: «Почему наши с тобой ролики не набирают просмотры?».

python tools/views_mg.py work/views/voice_clean.wav output/views.mp4 [--preview] [--jobs 4]
Общий движок, персонаж и концовка с Telegram — из memory_mg.py; параллельный рендер — как в money_mg.py.
Новое в монтаже: в начале — интерфейс ленты Shorts поверх кадра и камера «с рук»; смена сцен «свайпом вверх»
(кадр уезжает вверх, снизу приходит новая сцена — как в ленте); «тестовая группа» зрителей; график удержания;
счётчик «посмотрели / смахнули»; дождь из превью видео; 3D-указатель «сдаться / продолжать»;
топ-комментарий с растущими лайками; кнопки «досмотри / лайк / отправь другу».
Звуки: только тихий «вжух» на смене ракурса (правило пользователя).
"""
import math
import os
import subprocess
import sys
import types

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
OUTRO = float(open(os.path.join(ROOT, "work", "views", "outro_start.txt")).read())  # 43.16
DUR = 51.85
GREEN_RGB, GOLD_RGB, PINK_RGB = (46, 196, 110), (245, 190, 50), (240, 70, 110)
GREEN, RED = GREEN_RGB[::-1], RED_RGB[::-1]

# ---------------------------------------------------------------- субтитры
T_, G_, Go_, Pi_ = TG_RGB, GREEN_RGB, GOLD_RGB, PINK_RGB
SUBS = [
    (0.224, "ПОЧЕМУ НАШИ С ТОБОЙ РОЛИКИ", ()), (1.424, "НЕ НАБИРАЮТ ПРОСМОТРЫ?", ("ПРОСМОТРЫ?",), Pi_),
    (2.75, "", ()),
    (4.124, "КОГДА ВЫХОДИТ НОВОЕ ВИДЕО,", ("НОВОЕ",)), (5.564, "ПРИЛОЖЕНИЕ СНАЧАЛА ПОКАЗЫВАЕТ ЕГО", ()),
    (7.164, "МАЛЕНЬКОЙ ГРУППЕ ЛЮДЕЙ", ("МАЛЕНЬКОЙ",), Go_), (8.684, "И СМОТРИТ, ЧТО ОНИ СДЕЛАЮТ:", ()),
    (10.404, "ДОСМОТРЯТ", ("ДОСМОТРЯТ",), G_), (11.124, "ИЛИ ПРОЛИСТАЮТ", ("ПРОЛИСТАЮТ",)),
    (12.244, "ОТ ЭТОГО ЗАВИСИТ МНОГОЕ", ("МНОГОЕ",), Go_), (14.0, "ЕСЛИ ПРОЛИСТАЛИ", ("ПРОЛИСТАЛИ",)),
    (15.06, "В ПЕРВЫЕ СЕКУНДЫ,", ("ПЕРВЫЕ",)), (16.02, "ВИДЕО ДАЛЬШЕ НЕ ИДЁТ", ("НЕ", "ИДЁТ")),
    (17.52, "В YOUTUBE ДАЖЕ ЕСТЬ", ()), (18.44, "ТАКОЙ СЧЁТЧИК:", ("СЧЁТЧИК:",), Go_),
    (19.4, "СКОЛЬКО ЛЮДЕЙ ПОСМОТРЕЛИ,", ("ПОСМОТРЕЛИ,",), G_), (20.8, "А СКОЛЬКО СМАХНУЛИ", ("СМАХНУЛИ",)),
    (22.32, "А КОНКУРЕНЦИЯ ОГРОМНАЯ", ("ОГРОМНАЯ",), Go_), (24.0, "КАЖДУЮ МИНУТУ НА YOUTUBE", ()),
    (25.06, "И ДРУГИХ ПЛОЩАДКАХ", ()), (26.34, "ЗАГРУЖАЮТ", ()), (27.06, "СОТНИ ЧАСОВ ВИДЕО", ("СОТНИ", "ЧАСОВ"), Go_),
    (28.719, "ТАК ЧТО У НАС ДВА ПУТИ:", ("ДВА",)), (30.119, "СДАТЬСЯ ИЛИ ПРОДОЛЖАТЬ", ("ПРОДОЛЖАТЬ",), G_),
    (31.739, "МЫ ПРОДОЛЖАЕМ!", ("ПРОДОЛЖАЕМ!",), G_), (32.839, "ЕСЛИ ЭТО ВИДЕО ЗАЛЕТИТ,", ("ЗАЛЕТИТ,",), Pi_),
    (34.239, "МЫ СНИМЕМ ВИДЕО НА ТЕМУ", ()), (35.339, "САМОГО ЗАЛАЙКАННОГО", ("ЗАЛАЙКАННОГО",), Pi_),
    (36.299, "КОММЕНТАРИЯ,", ("КОММЕНТАРИЯ,",), Pi_), (37.079, "ТОЛЬКО БЕЗ БАЛОВСТВА", ("БАЛОВСТВА",)),
    (38.439, "А ТЫ МОЖЕШЬ ПОМОЧЬ:", ("ПОМОЧЬ:",), G_), (39.759, "ДОСМОТРИ ДО КОНЦА,", ("ДОСМОТРИ",), G_),
    (40.839, "ПОСТАВЬ ЛАЙК", ("ЛАЙК",), Pi_), (41.599, "И ОТПРАВЬ ДРУГУ", ("ДРУГУ",), T_),
    (43.13, "А ЕЩЁ БОЛЬШЕ ИНТЕРЕСНОГО —", (), T_), (44.49, "В МОЁМ ТЕЛЕГРАМ-КАНАЛЕ", ("ТЕЛЕГРАМ-КАНАЛЕ",), T_),
    (45.82, "«ПАУЧЬЯ НОРКА»", ("«ПАУЧЬЯ", "НОРКА»"), T_), (46.94, "ТАМ ЭКСКЛЮЗИВ,", ("ЭКСКЛЮЗИВ",), T_),
    (47.64, "КОТОРОГО НЕТ В РОЛИКАХ", (), T_), (48.94, "ССЫЛКА В ПРОФИЛЕ —", ("ПРОФИЛЕ",), T_),
    (49.96, "ЗАЛЕТАЙ!", ("ЗАЛЕТАЙ!",), T_),
]
NOSUBS = [(0.0, "", ())]

# ---------------------------------------------------------------- камера и сцены
SWIPES = (2.6, 13.6, 28.4)       # «свайп вверх»: старая сцена уезжает вверх, снизу приходит новая
SW = 0.4
CAMERA = [
    (0.0, (-6.0, 3.6, 1.15, 0.84, 50.0)), (SWIPES[0], (-4.0, 3.6, 1.15, 0.84, 50.0)),
    (SWIPES[0] + SW, (0.0, 3.9, 1.3, 0.98, 50.0), "lin"),   # заставка
    (3.85, (2.0, 3.9, 1.3, 0.98, 50.0)),
    (4.25, (95.0, 3.6, 1.2, 0.84, 50.0), "expo"),           # тестовая группа
    (SWIPES[1], (101.0, 3.5, 1.2, 0.86, 50.0)),
    (SWIPES[1] + SW, (190.0, 3.6, 1.2, 0.86, 50.0), "lin"),  # удержание, счётчик, конкуренция
    (SWIPES[2], (198.0, 3.6, 1.25, 0.86, 50.0)),
    (SWIPES[2] + SW, (285.0, 3.6, 1.2, 0.84, 50.0), "lin"),  # два пути и просьба
    (42.75, (298.0, 3.7, 1.25, 0.86, 50.0), "lin"),
    (43.15, (380.0, 4.3, 1.3, 1.12, 50.0), "expo"), (DUR, (381.0, 4.0, 1.3, 1.15, 50.0)),
]
OFFSET = [(0.0, 0.3), (SWIPES[0], 0.3), (SWIPES[0] + SW, 0.0, "lin"), (3.85, 0.0), (4.25, 0.55, "expo"),
          (SWIPES[1], 0.55), (SWIPES[1] + SW, -0.55, "lin"), (SWIPES[2], -0.55), (SWIPES[2] + SW, 0.55, "lin"),
          (42.75, 0.55), (43.15, 0.0, "expo"), (DUR, 0.0)]
HAND = [(0.0, 3.5), (SWIPES[0], 3.5), (SWIPES[0] + SW, 1.0, "lin"), (DUR, 1.0)]  # камера «с рук» в начале
TINT = [(0.0, (1.06, 0.94, 1.1)), (SWIPES[0], (1.06, 0.94, 1.1)), (SWIPES[0] + SW, (1.0, 1.0, 1.0), "lin"),
        (3.85, (1.0, 1.0, 1.0)), (4.3, (0.92, 1.04, 1.1)), (SWIPES[1], (0.92, 1.04, 1.1)),
        (SWIPES[1] + SW, (1.1, 0.96, 0.9), "lin"), (SWIPES[2], (1.1, 0.96, 0.9)),
        (SWIPES[2] + SW, (0.9, 1.1, 0.95), "lin"), (42.9, (0.9, 1.1, 0.95)), (43.3, (1.0, 1.0, 1.0)), (DUR, (1.0, 1.0, 1.0))]
SEC = {"hook": -5, "test": 98, "ret": 194, "fix": 290, "tg": 380.5}
SHAKES = ()
FLASHES = ((4.2, 0.08), (43.2, 0.12))
JUMPS = [31.8]
ALARM = None
SPOT = None

G = lambda *k: list(k)  # noqa: E731
ARM_L = G((0, 62), (2.0, 62), (2.15, 36, "back"), (2.55, 38), (SWIPES[0] + SW, 62), (3.05, -50, "back"), (3.8, -45),
          (4.1, 62, "back"), (SWIPES[1] + SW, 62), (14.2, -25, "back"), (17.2, -22), (17.5, -12, "back"), (22.0, -10),
          (22.3, -35, "back"), (28.1, -33), (28.35, 62), (28.9, 30, "back"), (30.9, 32), (31.2, 62), (31.75, -50, "back"),
          (32.4, -45), (32.7, 62, "back"), (50.07, 62), (50.22, -50, "back"), (50.82, -45), (51.22, 62, "back"), (DUR, 62))
ARM_R = G((0, 62), (0.4, 62), (0.6, -30, "back"), (1.9, -28), (2.15, 36, "back"), (2.55, 38), (SWIPES[0] + SW, 62),
          (3.05, -50, "back"), (3.8, -45), (4.1, 62, "back"), (4.3, -30, "back"), (8.3, -28), (8.6, -15, "back"),
          (12.0, -12), (12.2, -40, "back"), (13.4, -38), (13.55, 62), (28.9, 30, "back"), (30.9, 32), (31.2, 62),
          (31.75, -50, "back"), (32.4, -45), (32.7, -15, "back"), (38.0, -12), (38.4, -35, "back"), (42.6, -33),
          (42.92, 62), (43.38, -48, "back"), (45.73, -45), (46.03, -55, "back"), (48.73, -52), (48.98, -40, "back"),
          (49.93, -42), (50.08, -50, "back"), (50.82, -45), (50.97, -58), (51.12, -32), (51.27, -58), (51.42, -32),
          (DUR, 62))
HEAD = G((0, 0), (0.6, -6), (1.6, 6), (2.15, 9), (2.6, 4), (3.0, 0), (4.3, -6), (7.2, -5), (8.7, 6), (10.4, 7),
         (11.1, -7), (12.3, 5), (14.2, 6), (16.0, -7), (17.5, 5), (20.8, -6), (22.3, 8), (24.0, 6), (28.8, -5),
         (30.1, 6), (31.7, 0), (32.8, -6), (35.3, 7), (37.1, -8), (38.4, 6), (40.8, -5), (41.6, 7), (43.38, -6),
         (50.1, 0), (DUR, 0))


# ---------------------------------------------------------------- картинки вставок
def header(d, text, col, dark=False):
    f = mg.font(36)
    tw = d.textbbox((0, 0), text, font=f)[2]
    d.rounded_rectangle([36, 30, 36 + tw + 44, 92], 30, fill=col + (255,))
    d.text((58, 38), text, font=f, fill=(30, 24, 10) if dark else "white")


def heart(d, cx, cy, r, col):
    d.ellipse([cx - r, cy - r * 0.8, cx, cy + r * 0.2], fill=col)
    d.ellipse([cx, cy - r * 0.8, cx + r, cy + r * 0.2], fill=col)
    d.polygon([(cx - r * 0.96, cy - r * 0.1), (cx + r * 0.96, cy - r * 0.1), (cx, cy + r)], fill=col)


def share_arrow(d, cx, cy, r, col):
    d.polygon([(cx + r, cy - r * 0.2), (cx + r * 0.1, cy - r), (cx + r * 0.1, cy - r * 0.5), (cx - r, cy - r * 0.4),
               (cx - r * 0.9, cy + r * 0.8), (cx + r * 0.1, cy + r * 0.1), (cx + r * 0.1, cy + r * 0.6)], fill=col)


def feed_ui():
    """Интерфейс ленты коротких видео поверх кадра: кнопки справа, подпись и полоска снизу."""
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    x = W - 95
    for i, (kind, txt) in enumerate((("like", "12"), ("comm", "0"), ("share", "1"))):
        y = 760 + i * 170
        d.ellipse([x - 52, y - 52, x + 52, y + 52], fill=(30, 30, 38, 150))
        if kind == "like":
            heart(d, x, y - 4, 30, (240, 240, 245, 255))
        elif kind == "comm":
            d.rounded_rectangle([x - 30, y - 26, x + 30, y + 16], 14, fill=(240, 240, 245, 255))
            d.polygon([(x - 14, y + 12), (x + 4, y + 12), (x - 18, y + 30)], fill=(240, 240, 245, 255))
        else:
            share_arrow(d, x, y, 30, (240, 240, 245, 255))
        f = mg.font(34)
        tw = d.textbbox((0, 0), txt, font=f)[2]
        d.text((x - tw / 2, y + 58), txt, font=f, fill=(240, 240, 245, 255))
    d.text((40, H - 170), "@Паучья Норка", font=mg.font(40), fill=(240, 240, 245, 255))
    d.rounded_rectangle([0, H - 14, W, H], 0, fill=(255, 255, 255, 70))
    return mg.premul(im)


def views_tex(n):
    im = rounded((560, 300), 34, (20, 18, 26, 255), (90, 80, 110, 255), 4)
    d = ImageDraw.Draw(im)
    d.polygon([(46, 50), (46, 110), (96, 80)], fill=(200, 200, 215, 255))
    d.text((120, 46), "просмотры", font=mg.font(44), fill=(200, 200, 215))
    d.text((44, 130), str(n), font=mg.font(130, tall=True), fill=PINK_RGB + (255,))
    return mg.premul(im)


def person(d, x, y, s, col):
    d.ellipse([x - 16 * s, y, x + 16 * s, y + 32 * s], fill=col)
    d.rounded_rectangle([x - 26 * s, y + 36 * s, x + 26 * s, y + 90 * s], int(22 * s), fill=col)


def group_tex(shown, decided):
    """Тестовая группа: 10 человечков; shown — скольким показали; decided — сколько уже «решили»."""
    im = rounded((780, 470), 34, (18, 18, 26, 255), (72, 72, 92, 255), 3)
    d = ImageDraw.Draw(im)
    header(d, "ТЕСТОВАЯ ГРУППА", GOLD_RGB, dark=True)
    verdict = [1, 0, 0, 1, 0, 1, 0, 0, 1, 0]  # 1 — досмотрел, 0 — пролистал
    for i in range(10):
        r, c = divmod(i, 5)
        x, y = 100 + c * 145, 130 + r * 150
        if i >= shown:
            col = (50, 50, 62, 255)
        elif i < decided:
            col = GREEN_RGB + (255,) if verdict[i] else RED_RGB + (255,)
        else:
            col = (200, 200, 215, 255)
        person(d, x, y, 1.2, col)
    if decided >= 10:
        d.text((40, 420), "досмотрели", font=mg.font(30), fill=(150, 230, 180))
        d.text((300, 420), "пролистали", font=mg.font(30), fill=(255, 140, 140))
    return mg.premul(im)


def retention_tex(p):
    """График удержания: обрыв в первые секунды (иллюстрация, без цифр)."""
    im = rounded((780, 520), 34, (18, 18, 26, 255), (72, 72, 92, 255), 3)
    d = ImageDraw.Draw(im)
    header(d, "КТО ОСТАЛСЯ СМОТРЕТЬ", RED_RGB)
    x0, y0, x1, y1 = 70, 440, 730, 140
    d.rounded_rectangle([x0, y1, x0 + 120, y0], 14, fill=(70, 26, 32, 255))
    d.text((x0 + 4, y0 + 10), "первые сек.", font=mg.font(28), fill=(255, 140, 140))
    d.line([(x0, y0), (x1, y0)], fill=(90, 90, 110), width=4)
    xs = np.linspace(0, 1, 60)
    ys = 0.35 + 0.65 * np.exp(-xs * 14) - 0.08 * xs
    n = max(2, int(len(xs) * p))
    pts = [(x0 + (x1 - x0) * x, y0 - (y0 - y1) * v) for x, v in zip(xs[:n], ys[:n])]
    d.line(pts, fill=(255, 120, 120, 255), width=10, joint="curve")
    return mg.premul(im)


def donut_tex(p):
    im = rounded((720, 440), 34, (18, 18, 26, 255), (72, 72, 92, 255), 3)
    d = ImageDraw.Draw(im)
    header(d, "СЧЁТЧИК В СТУДИИ", BLUE_RGB)
    cx, cy, r = 200, 270, 130
    a0 = -90
    for v, col in ((0.42, GREEN_RGB), (0.58, RED_RGB)):
        a1 = a0 + 360 * v * p
        if a1 > a0 + 0.5:
            d.pieslice([cx - r, cy - r, cx + r, cy + r], a0, a1, fill=col + (255,))
        a0 += 360 * v
    d.ellipse([cx - 70, cy - 70, cx + 70, cy + 70], fill=(18, 18, 26, 255))
    d.rounded_rectangle([380, 200, 410, 230], 8, fill=GREEN_RGB + (255,))
    d.text((426, 192), "посмотрели", font=mg.font(38), fill=(220, 230, 225))
    d.rounded_rectangle([380, 290, 410, 320], 8, fill=RED_RGB + (255,))
    d.text((426, 282), "смахнули", font=mg.font(38), fill=(255, 190, 190))
    return mg.premul(im)


def thumb_tex(seed):
    rng = np.random.default_rng(seed)
    col = tuple(int(v) for v in rng.integers(60, 200, 3))
    im = rounded((180, 320), 18, col + (255,), (230, 230, 240, 255), 4)
    d = ImageDraw.Draw(im)
    d.polygon([(70, 130), (70, 190), (120, 160)], fill=(255, 255, 255, 230))
    return mg.premul(im)


def sign_tex(text, col, right):
    f = mg.font(54, tall=True)
    tw = ImageDraw.Draw(Image.new("RGBA", (10, 10))).textbbox((0, 0), text, font=f)[2]
    w, h = tw + 150, 120
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    if right:
        d.rounded_rectangle([0, 0, w - 60, h - 1], 24, fill=col + (255,))
        d.polygon([(w - 70, 0), (w - 1, h // 2), (w - 70, h - 1)], fill=col + (255,))
        d.text((30, 20), text, font=f, fill="white")
    else:
        d.rounded_rectangle([60, 0, w - 1, h - 1], 24, fill=col + (255,))
        d.polygon([(70, 0), (0, h // 2), (70, h - 1)], fill=col + (255,))
        d.text((100, 20), text, font=f, fill="white")
    return mg.premul(im)


def comment_tex(likes):
    im = rounded((760, 330), 34, (18, 18, 26, 255), PINK_RGB + (255,), 4)
    d = ImageDraw.Draw(im)
    header(d, "ТОП-КОММЕНТАРИЙ", PINK_RGB)
    d.ellipse([40, 120, 120, 200], fill=(70, 80, 120, 255))
    d.text((140, 116), "зритель", font=mg.font(34), fill=(160, 160, 180))
    d.text((140, 160), "Сделайте ролик про…", font=mg.font(42), fill="white")
    heart(d, 520, 262, 24, PINK_RGB + (255,))
    d.text((560, 240), f"{likes:,}".replace(",", " "), font=mg.font(40), fill="white")
    return mg.premul(im)


def button_tex(kind, on):
    S = 260
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    base_col = {"watch": GREEN_RGB, "like": PINK_RGB, "share": TG_RGB}[kind]
    d.ellipse([0, 0, S - 1, S - 1], fill=base_col + (255,) if on else (40, 40, 52, 255),
              outline=base_col + (255,), width=10)
    c = (255, 255, 255, 255)
    if kind == "watch":
        d.polygon([(100, 80), (100, 180), (185, 130)], fill=c)
    elif kind == "like":
        heart(d, 130, 120, 70, c)
    else:
        share_arrow(d, 130, 130, 70, c)
    return mg.premul(im)


# ---------------------------------------------------------------- сцена
class Scene(base.Scene):
    cfg = sys.modules[__name__]

    def __init__(self, voice):
        super().__init__(voice)
        T = self.T
        T["ui"] = feed_ui()
        T["why"] = stamp("ПОЧЕМУ?", PINK_RGB)
        T["matter"] = chip("ОТ ЭТОГО ЗАВИСИТ МНОГОЕ", GOLD_RGB)
        T["stop"] = stamp("ДАЛЬШЕ НЕ ИДЁТ", RED_RGB)
        T["comp"] = card("КАЖДУЮ МИНУТУ", "СОТНИ ЧАСОВ ВИДЕО", "только на YouTube — около 500 часов", GOLD_RGB)
        T["thumbs"] = [thumb_tex(s) for s in range(8)]
        T["s_give"], T["s_go"] = sign_tex("СДАТЬСЯ", (90, 90, 104), False), sign_tex("ПРОДОЛЖАТЬ", GREEN_RGB, True)
        T["go_on"] = stamp("ПРОДОЛЖАЕМ!", GREEN_RGB)
        T["noplay"] = stamp("БЕЗ БАЛОВСТВА", GOLD_RGB)
        T["help"] = chip("ТЫ МОЖЕШЬ ПОМОЧЬ", GREEN_RGB)
        T["btn"] = {(k, o): button_tex(k, o) for k in ("watch", "like", "share") for o in (False, True)}
        T["blabel"] = {k: base.label(t, (220, 220, 232), 50) for k, t in (("watch", "досмотри"), ("like", "лайк"),
                                                                           ("share", "другу"))}
        self.cache = {}
        self.swipe_cache = {}
        orig = self.room.render
        self.room.render = lambda cam: self.grade(orig(cam))
        self.nosub = types.SimpleNamespace(**{k: getattr(self.cfg, k) for k in dir(self.cfg) if k.isupper()})
        self.nosub.SUBS = NOSUBS

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
        for s in SWIPES:
            if s <= t < s + SW:
                if s not in self.swipe_cache:  # оба кадра замораживаем на время свайпа
                    self.cfg = self.nosub
                    self.t = s - 1.0 / FPS
                    a = super().draw(self.t).copy()
                    self.t = s + SW
                    b = super().draw(self.t).copy()
                    del self.cfg
                    self.swipe_cache = {s: (a, b)}
                a, b = self.swipe_cache[s]
                off = int(H * smooth((t - s) / SW))
                out = np.empty_like(a)
                out[:H - off] = a[off:]
                out[H - off:] = b[:off]
                if 0 < off < H:
                    cv2.line(out, (0, H - off), (W, H - off), (20, 20, 24), 6)
                mg.subtitles(out, t, SUBS)
                self.frame = out
                return out
        self.t = t
        return super().draw(t)

    def cam(self, t):
        yaw, dist, hgt, ty, fov = track(CAMERA, t)
        off = track(OFFSET, t)
        hand = track(HAND, t)
        a = math.radians(yaw)
        return mg.Cam(yaw + 0.5 * hand * wobble(t * hand ** 0.5, 1), dist, hgt + 0.02 * hand * wobble(t * 1.6, 2),
                      ty + 0.015 * hand * wobble(t * 1.4, 3), fov=fov, roll=0.5 * hand * wobble(t * 0.8 * hand ** 0.5, 4),
                      tx=off * math.cos(a), tz=off * math.sin(a))

    def finish(self, frame, t):
        fr = super().finish(frame, t)
        k = 1 - smooth((t - (SWIPES[0] - 0.2)) / 0.2) if t < SWIPES[0] else 0.0
        if k > 0:  # интерфейс ленты поверх кадра
            mg.over(fr, self.T["ui"] * min(1.0, t / 0.3) * k, 0, 0)
        return fr

    def items_for(self, t, cam, it):
        T = self.T
        P = lambda y0, *a, **k: place(y0, *a, **k)  # noqa: E731

        # 1. Крючок (лента): счётчик просмотров еле растёт, «почему?»
        y0 = SEC["hook"]
        k = pop(t, 0.35, SWIPES[0] + 0.1, 0.35, 0.2)
        if k > 0:
            n = 12 + int(25 * smooth((t - 0.5) / 1.8))
            pos = P(y0, 0.55, 1.88) + [0, 0.02 * math.sin(t * 3), 0]
            self.add_panel(it, cam, pos, face(y0, 10), 0.5 * k, 0.268 * k, 0.04 * k, self.cached("v", views_tex, n),
                           pitch=-5, side=(60, 40, 60), edge=(255, 200, 230))
        k = pop(t, 2.0, SWIPES[0] + 0.1, 0.15, 0.2)
        if k > 0:
            ks = clamp((t - 2.0) / 0.15)
            self.add_bb(it, cam, P(y0, 0.5, 1.4, 0.1), 0.14 * (1 + 0.6 * (1 - ks)) * k, T["why"], anchor=(0.5, 0.5))

        # 2. Заставка
        k = pop(t, SWIPES[0] + 0.3, 4.2, 0.35, 0.3)
        if k > 0:
            self.add_panel(it, cam, place(0, 0.0, 1.95, 0.35), face(0, -6), 1.2 * k, 0.564 * k, 0.11 * k,
                           T["stinger"], pitch=-6, side=(30, 24, 120), edge=(200, 190, 255))

        # 3. Тестовая группа (персонаж слева): показывают 10 людям → кто досмотрел, кто пролистал
        y0 = SEC["test"]
        k = pop(t, 4.3, SWIPES[1] + 0.1, 0.4, 0.2)
        if k > 0:
            shown = min(10, max(0, int((t - 6.3) / 0.18) + 1)) if t > 6.3 else 0
            decided = min(10, max(0, int((t - 10.3) / 0.16) + 1)) if t > 10.3 else 0
            pos = P(y0, 0.72, 1.62) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 12), 0.9 * k, 0.542 * k, 0.06 * k,
                           self.cached("grp", group_tex, shown, decided), pitch=-5, side=(70, 60, 30), edge=(255, 230, 170))
        k = pop(t, 12.25, SWIPES[1] + 0.1, 0.3, 0.2)
        if k > 0:
            self.add_bb(it, cam, P(y0, 0.78, 1.08), 0.11 * k, T["matter"], anchor=(0.5, 0.5))

        # 4. Удержание, счётчик в студии, конкуренция (персонаж справа)
        y0 = SEC["ret"]
        r_, c_ = basis(y0)
        k = pop(t, SWIPES[1] + SW, 17.5, 0.4, 0.3)
        if k > 0:
            p = round(smooth((t - 14.3) / 1.4) * 20) / 20
            pos = P(y0, -0.72, 1.6) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, -12), 0.9 * k, 0.6 * k, 0.06 * k, self.cached("ret", retention_tex, p),
                           pitch=-5, side=(80, 40, 40), edge=(255, 200, 200))
            if t > 16.6:
                ks = clamp((t - 16.6) / 0.15)
                self.add_bb(it, cam, pos + c_ * 0.12 + [0, -0.45, 0], 0.13 * (1 + 0.6 * (1 - ks)) * k, T["stop"],
                            anchor=(0.5, 0.5))
        k = pop(t, 17.5, 22.4, 0.4, 0.3)
        if k > 0:
            p = round(smooth((t - 19.4) / 1.6) * 20) / 20
            pos = P(y0, -0.72, 1.6) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, -12), 0.84 * k, 0.513 * k, 0.06 * k, self.cached("don", donut_tex, p),
                           pitch=-5, side=(40, 50, 90), edge=(200, 210, 255))
        if 22.3 < t < SWIPES[2] + 0.1:  # дождь из превью видео за спиной
            rng = np.random.default_rng(4)
            for i in range(26):
                s0 = 22.3 + rng.uniform(0, 5.2)
                p = (t - s0) / 1.3
                if not 0 < p < 1:
                    continue
                x, z = rng.uniform(-1.9, 0.4), rng.uniform(-1.8, -0.9)
                pos = r_ * x + c_ * z + [0, 2.9 - 2.9 * p, 0]
                self.add_bb(it, cam, pos, 0.2, T["thumbs"][i % 8], anchor=(0.5, 0.5))
        k = pop(t, 23.9, SWIPES[2] + 0.1, 0.4, 0.2)
        if k > 0:
            pos = P(y0, -0.64, 1.8) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, -12), 0.8 * k, 0.427 * k, 0.06 * k, T["comp"], pitch=-6,
                           side=(70, 60, 30), edge=(255, 230, 170))

        # 5. Два пути → топ-комментарий → кнопки «досмотри / лайк / другу» (персонаж слева)
        y0 = SEC["fix"]
        r_, c_ = basis(y0)
        k = pop(t, 28.75, 32.75, 0.45, 0.3)
        if k > 0:
            sp = r_ * 0.95 + c_ * -0.45
            self.add_box(it, cam, (sp[0], sp[2]), (0.07, 0.07), 1.5 * k, (90, 80, 70), radius=0.02)
            go = smooth((t - 31.75) / 0.25)
            self.add_bb(it, cam, sp + [-0.0, 1.38 * k, 0] + r_ * 0.3, 0.13 * k * (1 + 0.15 * go), T["s_go"], anchor=(0.5, 0.5))
            self.add_bb(it, cam, sp + [0, 1.12 * k, 0] - r_ * 0.25, 0.13 * k, T["s_give"] * (1 - 0.6 * go), anchor=(0.5, 0.5))
            if t > 31.75:
                ks = clamp((t - 31.75) / 0.15)
                self.add_bb(it, cam, P(y0, 0.75, 1.95, 0.1), 0.15 * (1 + 0.6 * (1 - ks)) * k, T["go_on"], anchor=(0.5, 0.5))
        k = pop(t, 32.85, 38.3, 0.4, 0.3)
        if k > 0:
            likes = int(round(1254 * smooth((t - 35.3) / 1.6)))
            pos = P(y0, 0.72, 1.62) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 12), 0.9 * k, 0.39 * k, 0.06 * k, self.cached("cm", comment_tex, likes),
                           pitch=-5, side=(70, 30, 50), edge=(255, 200, 220))
            if t > 37.3:
                ks = clamp((t - 37.3) / 0.15)
                self.add_bb(it, cam, pos + c_ * 0.12 + [0, -0.38, 0], 0.13 * (1 + 0.6 * (1 - ks)) * k, T["noplay"],
                            anchor=(0.5, 0.5))
        k = pop(t, 38.4, 42.8, 0.3, 0.3)
        if k > 0:
            self.add_bb(it, cam, P(y0, 0.78, 2.12), 0.12 * k, T["help"], anchor=(0.5, 0.5))
            for i, (key, t0, hy) in enumerate((("watch", 39.76, 1.78), ("like", 41.0, 1.44), ("share", 41.9, 1.1))):
                kk = pop(t, 38.6 + 0.12 * i, 42.8, 0.35, 0.3)
                if kk <= 0:
                    continue
                on = t > t0
                press = math.sin(math.pi * clamp((t - t0) / 0.25)) if t0 < t < t0 + 0.25 else 0.0
                pos = P(y0, 0.95, hy, 0.05)
                self.add_bb(it, cam, pos, 0.22 * kk * (1 - 0.15 * press + 0.08 * on), T["btn"][(key, on)], anchor=(0.5, 0.5))
                self.add_bb(it, cam, pos - r_ * 0.32, 0.07 * kk, T["blabel"][key], anchor=(1.0, 0.5))

        # 6. Концовка с Telegram — общий шаблон
        self.outro_items(t, cam, it, OUTRO, SEC["tg"])


SFX = [(t, "whoosh") for t in SWIPES + (3.85, 43.0)]


def render(voice, out, frames):
    sc = Scene(voice)
    wr = mg.Writer(out, FPS)
    for n, i in enumerate(frames):
        t = i / FPS
        speed = abs(sc.cam(t + 0.5 / FPS).yaw - sc.cam(t - 0.5 / FPS).yaw) * FPS
        inside = any(s - 1.0 / FPS <= t < s + SW + 1.0 / FPS for s in SWIPES)
        sub = 5 if speed > math.radians(60) and not inside else 1
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
        for t in (0.8, 2.2, 2.75, 3.4, 5.5, 8.0, 11.5, 12.8, 13.8, 15.5, 16.9, 20.5, 24.5, 27.0, 28.6, 30.5, 32.0,
                  36.0, 37.6, 40.2, 41.3, 42.2, 45.0, 50.5):
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
    base.mix_audio(tmp, voice, out, music=os.path.join(ROOT, "work", "views", "music.wav"), sfx_list=SFX, dur=DUR,
                   sfx_vol=0.15)  # «вжух» лёгкий и тихий
    os.remove(tmp)


if __name__ == "__main__":
    main()
