"""Ролик №13: «Как правильно стримить?» (по самому залайканному комментарию, как обещали в ролике №11).

python tools/stream_mg.py work/stream/voice_clean.wav output/stream.mp4 [--preview] [--jobs 4]
Общий движок, персонаж и концовка с Telegram — из memory_mg.py; параллельный рендер — как в money_mg.py.
Голос укорочен до минуты: tools/tighten.py (паузы до 0,32 с + ускорение 5%, концовка не тронута).
Новое в монтаже: «прямой эфир» — резкие склейки между камерами с плашкой «● LIVE · КАМЕРА N»; каждый совет
открывается большой золотой цифрой, падающей сверху; скриншот топ-комментария; чистый и шумный звук; расписание
стримов; пустой чат и первый зритель; ножницы режут запись на короткие ролики; кнопка «В ЭФИР»; живой чат в конце.
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
OUTRO = float(open(os.path.join(ROOT, "work", "stream", "outro_start.txt")).read())  # 51.08
DUR = 59.7
PURPLE_RGB, GOLD_RGB, GREEN_RGB, PINK_RGB = (145, 70, 255), (245, 190, 50), (46, 196, 110), (240, 70, 110)
PURPLE, GOLD = PURPLE_RGB[::-1], GOLD_RGB[::-1]

# ---------------------------------------------------------------- субтитры
T_, Pu_, Go_, G_, Pi_ = TG_RGB, PURPLE_RGB, GOLD_RGB, GREEN_RGB, PINK_RGB
SUBS = [
    (0.146, "КАК ПРАВИЛЬНО СТРИМИТЬ?", ("СТРИМИТЬ?",), Pu_), (1.6, "", ()),
    (3.06, "ДЕЛАЮ ВИДЕО НА ТЕМУ", ()), (4.146, "САМОГО ЗАЛАЙКАННОГО", ("ЗАЛАЙКАННОГО",), Pi_),
    (5.155, "КОММЕНТАРИЯ, КАК И ОБЕЩАЛ", ("ОБЕЩАЛ",), Pi_), (6.965, "СОВЕТ ПЕРВЫЙ:", ("ПЕРВЫЙ:",), Go_),
    (8.298, "ЗВУК ВАЖНЕЕ КАРТИНКИ", ("ЗВУК",), Pu_), (10.203, "УЧЁНЫЕ ПРОВЕРИЛИ:", ()),
    (11.593, "КОГДА ЗВУК ПЛОХОЙ,", ("ПЛОХОЙ,",)), (13.022, "ДАЖЕ УМНЫЕ СЛОВА КАЖУТСЯ", ()),
    (14.946, "МЕНЕЕ УБЕДИТЕЛЬНЫМИ", ("УБЕДИТЕЛЬНЫМИ",)), (16.429, "СОВЕТ ВТОРОЙ:", ("ВТОРОЙ:",), Go_),
    (17.522, "СТРИМЬ ПО РАСПИСАНИЮ", ("РАСПИСАНИЮ",), Pu_), (18.952, "ЗРИТЕЛЬ ДОЛЖЕН ЗНАТЬ,", ()),
    (20.16, "КОГДА ТЕБЯ ЖДАТЬ,", ()), (21.208, "КАК ЛЮБИМЫЙ СЕРИАЛ", ("СЕРИАЛ",), Pu_),
    (22.629, "СОВЕТ ТРЕТИЙ:", ("ТРЕТИЙ:",), Go_), (23.798, "ГОВОРИ БЕЗ ОСТАНОВКИ,", ("ГОВОРИ",), Pu_),
    (25.227, "ДАЖЕ ЕСЛИ В ЧАТЕ НИКОГО НЕТ", ("НИКОГО",)), (27.255, "НОВЫЙ ЗРИТЕЛЬ ЗАХОДИТ", ("НОВЫЙ",), G_),
    (28.817, "И СРАЗУ ВИДИТ ЖИВОГО ЧЕЛОВЕКА,", ("ЖИВОГО",), G_), (30.836, "А НЕ ТИШИНУ", ("ТИШИНУ",)),
    (31.8, "СОВЕТ ЧЕТВЁРТЫЙ:", ("ЧЕТВЁРТЫЙ:",), Go_), (32.785, "ЗДОРОВАЙСЯ С КАЖДЫМ", ()), (34.042, "ПО ИМЕНИ", ("ИМЕНИ",), Pu_),
    (34.632, "И НАРЕЗАЙ ЛУЧШИЕ МОМЕНТЫ", ("НАРЕЗАЙ",), Pu_), (36.232, "В КОРОТКИЕ РОЛИКИ —", ("КОРОТКИЕ",)),
    (37.09, "ТАК ТЕБЯ НАЙДУТ НОВЫЕ ЛЮДИ", ("НОВЫЕ",), G_), (38.918, "А ПЕРВЫЙ СТРИМ", ()),
    (40.118, "ВСЕГДА СТРАШНЫЙ", ("СТРАШНЫЙ",)), (41.204, "ПРОСТО НАЧНИ!", ("НАЧНИ!",), G_),
    (42.61, "РЕБЯТА, ВСЕМ СПАСИБО", ("СПАСИБО",), Pi_), (44.298, "ЗА АКТИВ, ВСЕХ ВИЖУ!", ("АКТИВ,",), Pi_),
    (46.07, "ПИШИТЕ ВАШИ ИДЕИ", ("ИДЕИ",), Pu_), (47.06, "В КОММЕНТАРИИ —", ("КОММЕНТАРИИ",), Pu_),
    (47.955, "ТАК БОЛЬШЕ ШАНС,", ()), (49.136, "ЧТО ИМЕННО ВАС Я ЗАМЕЧУ", ("ЗАМЕЧУ",), Pi_),
    (51.05, "А ЕЩЁ БОЛЬШЕ ИНТЕРЕСНОГО —", (), T_), (52.41, "В МОЁМ ТЕЛЕГРАМ-КАНАЛЕ", ("ТЕЛЕГРАМ-КАНАЛЕ",), T_),
    (53.74, "«ПАУЧЬЯ НОРКА»", ("«ПАУЧЬЯ", "НОРКА»"), T_), (54.86, "ТАМ ЭКСКЛЮЗИВ,", ("ЭКСКЛЮЗИВ",), T_),
    (55.56, "КОТОРОГО НЕТ В РОЛИКАХ", (), T_), (56.86, "ССЫЛКА В ПРОФИЛЕ —", ("ПРОФИЛЕ",), T_),
    (57.88, "ЗАЛЕТАЙ!", ("ЗАЛЕТАЙ!",), T_),
]

# ---------------------------------------------------------------- «камеры» прямого эфира (резкие склейки)
# (время склейки, (yaw, dist, height, target_y, fov), offset)
CAMS = [
    (0.0, (-8.0, 3.6, 1.15, 0.84, 50.0), -0.5),       # 1 крючок
    (1.65, (0.0, 3.9, 1.3, 0.98, 50.0), 0.0),         # 2 заставка
    (2.95, (95.0, 3.6, 1.2, 0.84, 50.0), 0.55),       # 3 комментарий
    (6.9, (182.0, 3.5, 0.8, 0.95, 48.0), -0.55),      # 4 совет 1 — снизу
    (16.38, (272.0, 3.6, 1.6, 0.8, 50.0), 0.55),      # 5 совет 2 — чуть сверху
    (22.58, (362.0, 3.5, 1.2, 0.86, 50.0), -0.55),    # 6 совет 3
    (31.74, (452.0, 3.6, 1.2, 0.84, 50.0), 0.55),     # 7 совет 4 и «просто начни»
    (42.55, (540.0, 4.1, 1.5, 0.9, 50.0), 0.0),       # 8 спасибо, живой чат
]
CUTS = tuple(c[0] for c in CAMS[1:])
CAMERA, OFFSET = [], []
for i, (t0, cam, off) in enumerate(CAMS):
    t1 = CAMS[i + 1][0] if i + 1 < len(CAMS) else 50.7
    drift = (cam[0] + 5.0,) + cam[1:]          # внутри «камеры» — медленный дрейф
    CAMERA += [(t0 + 0.01, cam, "lin"), (t1 - 0.01, drift, "lin")]
    OFFSET += [(t0 + 0.01, off, "lin"), (t1 - 0.01, off, "lin")]
CAMERA += [(51.1, (625.0, 4.3, 1.3, 1.12, 50.0), "expo"), (DUR, (626.0, 4.0, 1.3, 1.15, 50.0))]
OFFSET += [(51.1, 0.0, "expo"), (DUR, 0.0)]
TINT = [(0.0, (1.12, 0.9, 1.06)), (50.7, (1.12, 0.9, 1.06)), (51.1, (1.0, 1.0, 1.0)), (DUR, (1.0, 1.0, 1.0))]
SEC = {"hook": -8, "cm": 98, "t1": 185, "t2": 275, "t3": 365, "t4": 455, "thx": 543, "tg": 625.5}
NUMS = ((6.97, 1, "t1", -0.75), (16.43, 2, "t2", 0.75), (22.63, 3, "t3", -0.75), (31.8, 4, "t4", 0.75))
SHAKES = tuple(n[0] + 0.25 for n in NUMS)
FLASHES = ((41.25, 0.18), (51.1, 0.12))
JUMPS = [41.3]
ALARM = None
SPOT = None

G = lambda *k: list(k)  # noqa: E731
ARM_L = G((0, 62), (0.3, -25, "back"), (1.55, -22), (1.7, 62), (1.8, -50, "back"), (2.8, -45), (2.95, 62),
          (6.9, 62), (7.05, -45, "back"), (8.1, -42), (8.3, -25, "back"), (16.2, -22), (16.38, 62), (22.58, 62),
          (22.7, -45, "back"), (23.6, -42), (23.8, -20, "back"), (31.5, -18), (31.74, 62), (38.9, 62),
          (39.05, 36, "back"), (40.9, 38), (41.1, 62), (42.55, 62), (42.7, 30, "back"), (44.6, 32), (44.8, 62),
          (57.99, 62), (58.14, -50, "back"), (58.74, -45), (59.14, 62, "back"), (DUR, 62))
ARM_R = G((0, 62), (1.7, 62), (1.8, -50, "back"), (2.8, -45), (2.95, 62), (3.1, -30, "back"), (6.8, -28), (6.9, 62),
          (16.38, 62), (16.5, -45, "back"), (17.4, -42), (17.5, -25, "back"), (22.4, -22), (22.58, 62), (31.74, 62),
          (31.85, -45, "back"), (32.7, -42), (32.8, -25, "back"), (38.7, -22), (39.05, 36, "back"), (40.9, 38),
          (41.05, -10, "back"), (42.3, -8), (42.55, 62), (42.7, 30, "back"), (44.6, 32), (44.75, -45, "back"),
          (45.0, -60), (45.25, -40), (45.5, -60), (45.8, -40, "back"), (46.0, -20, "back"), (50.7, -18), (50.84, 62),
          (51.3, -48, "back"), (53.65, -45), (53.95, -55, "back"), (56.65, -52), (56.9, -40, "back"), (57.85, -42),
          (58.0, -50, "back"), (58.74, -45), (58.89, -58), (59.04, -32), (59.19, -58), (59.34, -32), (DUR, 62))
HEAD = G((0, 0), (0.4, -6), (0.9, 7), (1.6, 0), (3.1, -6), (5.2, 6), (6.05, 9), (7.0, -4), (8.3, -6), (11.6, 6),
         (14.9, -7), (16.4, 0), (17.5, 6), (20.2, -5), (21.4, 7), (22.6, 0), (23.8, -6), (25.3, 7), (27.3, -5),
         (29.5, 6), (31.8, 0), (32.8, -6), (34.6, 5), (37.1, -6), (39.0, 8), (40.5, -7), (41.3, 0), (42.6, 7),
         (44.3, -6), (46.1, 5), (49.1, -6), (51.3, -6), (58.0, 0), (DUR, 0))


# ---------------------------------------------------------------- картинки вставок
def header(d, text, col, dark=False):
    f = mg.font(36)
    tw = d.textbbox((0, 0), text, font=f)[2]
    d.rounded_rectangle([36, 30, 36 + tw + 44, 92], 30, fill=col + (255,))
    d.text((58, 38), text, font=f, fill=(30, 24, 10) if dark else "white")


def hud_tex(cam):
    """Плашка прямого эфира в углу: «● LIVE · КАМЕРА N»."""
    f = mg.font(36)
    txt = f"КАМЕРА {cam}"
    tw = ImageDraw.Draw(Image.new("RGBA", (10, 10))).textbbox((0, 0), txt, font=f)[2]
    im = Image.new("RGBA", (190 + tw + 50, 70), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, 170, 69], 20, fill=RED_RGB + (255,))
    d.ellipse([22, 24, 44, 46], fill="white")
    d.text((56, 14), "LIVE", font=f, fill="white")
    d.rounded_rectangle([180, 0, 180 + tw + 50, 69], 20, fill=(20, 20, 26, 210))
    d.text((205, 14), txt, font=f, fill=(230, 230, 240))
    return mg.premul(im)


def eye(d, x, y, col):
    d.ellipse([x - 22, y - 13, x + 22, y + 13], outline=col, width=5)
    d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=col)


def stream_tex(viewers):
    im = rounded((760, 470), 30, (16, 14, 22, 255), PURPLE_RGB + (255,), 5)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([24, 24, 520, 300], 20, fill=(34, 30, 48, 255))
    d.polygon([(240, 120), (240, 200), (310, 160)], fill=(120, 110, 150, 255))
    d.rounded_rectangle([40, 40, 150, 84], 14, fill=RED_RGB + (255,))
    d.text((58, 44), "LIVE", font=mg.font(32), fill="white")
    eye(d, 200, 62, (220, 220, 235, 255))
    d.text((232, 42), str(viewers), font=mg.font(34), fill="white")
    d.rounded_rectangle([540, 24, 736, 446], 20, fill=(26, 24, 34, 255))
    d.text((560, 40), "ЧАТ", font=mg.font(30), fill=(150, 140, 180))
    d.text((556, 210), "пусто…", font=mg.font(30), fill=(100, 96, 120))
    d.text((28, 330), "Мой первый стрим", font=mg.font(46), fill="white")
    d.text((28, 395), "Паучок · игры и болтовня", font=mg.font(32), fill=(150, 140, 180))
    return mg.premul(im)


def comment_tex():
    shot = Image.open(os.path.join(ROOT, "assets", "viewer_comment.jpg")).convert("RGB")
    sw = 900
    sh = int(shot.height * sw / shot.width)
    shot = shot.resize((sw, sh), Image.LANCZOS)
    im = rounded((sw + 40, sh + 130), 36, (250, 250, 252, 255), PINK_RGB + (255,), 6)
    d = ImageDraw.Draw(im)
    tw = d.textbbox((0, 0), "ТОП-КОММЕНТАРИЙ", font=mg.font(36))[2]
    d.rounded_rectangle([24, 22, 24 + tw + 50, 84], 30, fill=PINK_RGB + (255,))
    d.text((48, 30), "ТОП-КОММЕНТАРИЙ", font=mg.font(36), fill="white")
    im.paste(shot, (20, 110))
    return mg.premul(im)


def number_tex(n):
    S = 420
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([10, 10, S - 11, S - 11], 90, fill=(26, 20, 10, 235), outline=GOLD_RGB + (255,), width=12)
    d.text((40 + 0, 30), "СОВЕТ", font=mg.font(48), fill=(240, 210, 140))
    f = mg.font(280, tall=True)
    l, t_, r, b = d.textbbox((0, 0), str(n), font=f)
    d.text(((S - (r - l)) / 2 - l, 255 - (b - t_) / 2 - t_), str(n), font=f, fill=GOLD_RGB + (255,))
    return mg.premul(im)


def sound_tex(ph):
    """Два звука: чистый и шумный; ниже — «насколько убедительно» (иллюстрация к опыту 2018 года)."""
    im = rounded((780, 600), 34, (18, 18, 26, 255), (72, 72, 92, 255), 3)
    d = ImageDraw.Draw(im)
    header(d, "ОПЫТ, 2018", BLUE_RGB)
    rng = np.random.default_rng(int(ph * 100))
    for i, (lab, col, noisy) in enumerate((("хороший звук", GREEN_RGB, False), ("плохой звук", RED_RGB, True))):
        y0 = 160 + i * 150
        d.text((40, y0 - 50), lab, font=mg.font(36), fill=(210, 210, 225))
        pts = []
        for k in range(120):
            x = 40 + k * 5.8
            v = math.sin(k * 0.35 + ph * 8) * math.sin(k * 0.06 + 1) * 40
            if noisy:
                v = v * 0.7 + rng.normal(0, 16)
            pts.append((x, y0 + 40 + v))
        d.line(pts, fill=col + (255,), width=6)
    d.text((40, 470), "убедительность выступления", font=mg.font(34), fill=(200, 200, 215))
    d.rounded_rectangle([40, 520, 400, 556], 18, fill=GREEN_RGB + (255,))
    d.rounded_rectangle([420, 520, 620, 556], 18, fill=RED_RGB + (255,))
    return mg.premul(im)


def schedule_tex(n):
    im = rounded((780, 420), 34, (18, 16, 26, 255), PURPLE_RGB + (255,), 4)
    d = ImageDraw.Draw(im)
    header(d, "РАСПИСАНИЕ СТРИМОВ", PURPLE_RGB)
    days = ["ПН", "ВТ", "СР", "ЧТ", "ПТ", "СБ", "ВС"]
    lit = [0, 2, 4]
    for c, dn in enumerate(days):
        x = 40 + c * 102
        on = c in lit[:n]
        d.rounded_rectangle([x, 130, x + 88, 360], 22, fill=PURPLE_RGB + (255,) if on else (36, 32, 48, 255))
        d.text((x + 20, 150), dn, font=mg.font(36), fill="white" if on else (120, 115, 140))
        if on:
            d.text((x + 6, 270), "19:00", font=mg.font(30), fill="white")
            d.ellipse([x + 32, 220, x + 56, 244], fill=RED_RGB + (255,))
    return mg.premul(im)


def chat_tex(viewers, joined, talk):
    """Пустой чат; streamer «говорит»; потом заходит новый зритель."""
    im = rounded((760, 520), 30, (16, 14, 22, 255), PURPLE_RGB + (255,), 5)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([24, 24, 300, 84], 16, fill=(34, 30, 48, 255))
    eye(d, 60, 54, (220, 220, 235, 255))
    d.text((92, 34), f"зрителей: {viewers}", font=mg.font(34), fill="white")
    for i in range(talk):
        y = 110 + i * 92
        d.rounded_rectangle([24, y, 520, y + 76], 30, fill=(60, 44, 100, 255))
        d.text((48, y + 18), ["…и вот что дальше", "…а сейчас смотрите", "…вот этот момент"][i % 3],
               font=mg.font(34), fill="white")
    if joined:
        d.rounded_rectangle([24, 400, 736, 490], 30, fill=(30, 60, 44, 255))
        d.text((48, 420), "+1  новый зритель зашёл!", font=mg.font(40), fill=(150, 240, 190))
    return mg.premul(im)


def greet_tex(n):
    im = rounded((780, 440), 34, (18, 16, 26, 255), PURPLE_RGB + (255,), 4)
    d = ImageDraw.Draw(im)
    header(d, "ЧАТ", PURPLE_RGB)
    rows = [("Дима", "привет!"), ("Паучок", "Привет, Дима!"), ("Аня", "я тут впервые"), ("Паучок", "Аня, добро пожаловать!")]
    for i, (who, txt) in enumerate(rows[:n]):
        y = 120 + i * 76
        me = who == "Паучок"
        d.text((40, y), who + ":", font=mg.font(36), fill=(255, 160, 190) if me else (150, 200, 255))
        d.text((40 + (190 if me else 120), y), txt, font=mg.font(36), fill="white")
    return mg.premul(im)


def cut_tex(p):
    """Запись стрима и ножницы: лучшие моменты вырезаются в короткие ролики."""
    im = rounded((780, 420), 34, (18, 16, 26, 255), (72, 72, 92, 255), 3)
    d = ImageDraw.Draw(im)
    header(d, "НАРЕЗКА", GOLD_RGB, dark=True)
    d.rounded_rectangle([40, 130, 740, 180], 20, fill=(50, 46, 66, 255))
    for x0 in (150, 380, 600):
        d.rounded_rectangle([x0, 130, x0 + 70, 180], 12, fill=GOLD_RGB + (255,))
    for i, x0 in enumerate((150, 380, 600)):
        q = clamp(p * 3 - i)
        if q <= 0:
            continue
        y = int(200 + 30 * q)
        d.rounded_rectangle([x0 - 20, y, x0 + 90, y + 170], 18, fill=(40, 30, 10, 255), outline=GOLD_RGB + (255,), width=5)
        d.polygon([(x0 + 22, y + 60), (x0 + 22, y + 110), (x0 + 62, y + 85)], fill=GOLD_RGB + (255,))
    return mg.premul(im)


def live_button_tex(pressed):
    S = 420
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([0, 0, S - 1, S - 1], fill=(120, 20, 30, 255))
    d.ellipse([20, 20 + (14 if pressed else 0), S - 21, S - 21], fill=(255, 60, 70, 255) if pressed else RED_RGB + (255,))
    f = mg.font(80, tall=True)
    l, t_, r, b = d.textbbox((0, 0), "В ЭФИР", font=f)
    d.text(((S - (r - l)) / 2 - l, (S - (b - t_)) / 2 - t_ + (8 if pressed else 0)), "В ЭФИР", font=f, fill="white")
    return mg.premul(im)


def ideas_tex(text):
    im = rounded((700, 230), 40, (18, 16, 26, 255), PINK_RGB + (255,), 4)
    d = ImageDraw.Draw(im)
    header(d, "ТВОЯ ИДЕЯ", PINK_RGB)
    d.rounded_rectangle([36, 116, 664, 196], 36, fill=(34, 32, 46, 255))
    d.text((66, 132), (text + "|") if text else "Напиши тему для ролика…", font=mg.font(38),
           fill="white" if text else (120, 118, 135))
    return mg.premul(im)


CHAT_WORDS = ["имба!", "лайк", "ещё!", "топ", "привет!", "сними про сон", "сними про деньги", "спасибо!", "огонь",
              "паучок топ"]


# ---------------------------------------------------------------- сцена
class Scene(base.Scene):
    cfg = sys.modules[__name__]

    def __init__(self, voice):
        super().__init__(voice)
        T = self.T
        T["hud"] = [hud_tex(i + 1) for i in range(len(CAMS))]
        T["comment"] = comment_tex()
        T["promise"] = stamp("КАК И ОБЕЩАЛ!", PINK_RGB)
        T["nums"] = {n: number_tex(n) for n in range(1, 5)}
        T["btn"], T["btn2"] = live_button_tex(False), live_button_tex(True)
        T["fear"] = stamp("СТРАШНО?", GRAY_RGB)
        T["chips"] = [chip(w, PURPLE_RGB if i % 2 else PINK_RGB) for i, w in enumerate(CHAT_WORDS)]
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
        if t < 50.8:  # плашка прямого эфира: номер камеры после каждой склейки
            idx = max(i for i, c in enumerate(CAMS) if c[0] <= t + 1e-6)
            since = t - CAMS[idx][0]
            k = clamp(since / 0.12) if idx else clamp(t / 0.3)
            hud = self.T["hud"][idx]
            mg.over(fr, hud * k, 40, 50)
        return fr

    def drop_number(self, it, cam, t, t0, n, y0, side):
        if not t0 < t < t0 + 1.3:
            return
        p = clamp((t - t0) / 0.35)
        y = 2.45 - 0.75 * (p * p)
        k = 1 - smooth((t - t0 - 1.0) / 0.3)
        sq = 1 - 0.12 * math.sin(math.pi * clamp((t - t0 - 0.35) / 0.15))
        self.add_bb(it, cam, place(y0, side, y, 0.1), 0.3 * k * sq, self.T["nums"][n], anchor=(0.5, 0.5))

    def items_for(self, t, cam, it):
        T = self.T
        P = lambda y0, *a, **k: place(y0, *a, **k)  # noqa: E731
        for t0, n, key, side in NUMS:
            self.drop_number(it, cam, t, t0, n, SEC[key], side)

        # 1. Крючок (персонаж справа): окно стрима, ноль зрителей
        y0 = SEC["hook"]
        k = pop(t, 0.2, CUTS[0], 0.35, 0.1)
        if k > 0:
            pos = P(y0, -0.74, 1.62) + [0, 0.02 * math.sin(t * 3), 0]
            self.add_panel(it, cam, pos, face(y0, -12), 0.9 * k, 0.556 * k, 0.05 * k, T.setdefault("st0", stream_tex(0)),
                           pitch=-5, side=(50, 30, 80), edge=(220, 190, 255))

        # 2. Заставка
        k = pop(t, CUTS[0] + 0.05, CUTS[1], 0.3, 0.15)
        if k > 0:
            self.add_panel(it, cam, place(0, 0.0, 1.95, 0.35), face(0, -6), 1.2 * k, 0.564 * k, 0.11 * k,
                           T["stinger"], pitch=-6, side=(30, 24, 120), edge=(200, 190, 255))

        # 3. Топ-комментарий (персонаж слева) — «как и обещал»
        y0 = SEC["cm"]
        r_, c_ = basis(y0)
        k = pop(t, CUTS[1] + 0.05, CUTS[2], 0.45, 0.1)
        if k > 0:
            tex = T["comment"]
            th, tw = tex.shape[:2]
            ww = 1.0
            pos = P(y0, 0.64, 1.72) + [0, 0.02 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 12) + 200 * (1 - smooth((t - CUTS[1]) / 0.55)), ww * k, ww * th / tw * k,
                           0.04 * k, tex, side=(70, 30, 50), radius=0.05)
            if t > 6.05:
                ks = clamp((t - 6.05) / 0.15)
                self.add_bb(it, cam, pos + c_ * 0.12 + [0, -0.42, 0], 0.15 * (1 + 0.6 * (1 - ks)) * k, T["promise"],
                            anchor=(0.5, 0.5))

        # 4. Совет 1 (персонаж справа, съёмка снизу): звук важнее картинки
        y0 = SEC["t1"]
        k = pop(t, 8.2, CUTS[3], 0.4, 0.1)
        if k > 0:
            ph = round((t * 1.2) % 1 * 10) / 10
            pos = P(y0, -0.72, 1.62) + [0, 0.02 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, -12), 0.9 * k, 0.692 * k, 0.06 * k, self.cached("snd", sound_tex, ph),
                           pitch=6, side=(40, 50, 90), edge=(200, 210, 255))

        # 5. Совет 2 (персонаж слева, чуть сверху): расписание
        y0 = SEC["t2"]
        k = pop(t, 17.45, CUTS[4], 0.4, 0.1)
        if k > 0:
            n = min(3, max(0, int((t - 18.9) / 0.35) + 1)) if t > 18.9 else 0
            pos = P(y0, 0.72, 1.62) + [0, 0.02 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 12), 0.9 * k, 0.485 * k, 0.06 * k, self.cached("sch", schedule_tex, n),
                           pitch=-8, side=(50, 30, 80), edge=(220, 190, 255))

        # 6. Совет 3 (персонаж справа): говори без остановки — пустой чат, потом заходит зритель
        y0 = SEC["t3"]
        k = pop(t, 23.75, CUTS[5], 0.4, 0.1)
        if k > 0:
            talk = min(3, max(0, int((t - 24.0) / 0.9) + 1)) if t > 24.0 else 0
            joined = t > 27.3
            pos = P(y0, -0.72, 1.62) + [0, 0.02 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, -12), 0.88 * k, 0.602 * k, 0.06 * k,
                           self.cached("chat", chat_tex, 1 if joined else 0, joined, talk), pitch=-5, side=(50, 30, 80),
                           edge=(220, 190, 255))

        # 7. Совет 4 (персонаж слева): здоровайся по имени → нарезка → «страшно?» → кнопка «В ЭФИР»
        y0 = SEC["t4"]
        k = pop(t, 32.75, 34.75, 0.4, 0.3)
        if k > 0:
            n = min(4, max(0, int((t - 33.0) / 0.4) + 1)) if t > 33.0 else 0
            pos = P(y0, 0.72, 1.62) + [0, 0.02 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 12), 0.9 * k, 0.508 * k, 0.06 * k, self.cached("gr", greet_tex, n),
                           pitch=-5, side=(50, 30, 80), edge=(220, 190, 255))
        k = pop(t, 34.7, 38.9, 0.4, 0.3)
        if k > 0:
            p = round(smooth((t - 35.0) / 2.2) * 24) / 24
            pos = P(y0, 0.72, 1.62) + [0, 0.02 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 12), 0.9 * k, 0.485 * k, 0.06 * k, self.cached("cut", cut_tex, p),
                           pitch=-5, side=(70, 60, 30), edge=(255, 230, 170))
        k = pop(t, 40.0, 41.2, 0.15, 0.2)
        if k > 0:
            ks = clamp((t - 40.0) / 0.15)
            self.add_bb(it, cam, P(y0, 0.8, 1.9, 0.1), 0.15 * (1 + 0.6 * (1 - ks)) * k, T["fear"], anchor=(0.5, 0.5))
        k = pop(t, 38.95, CUTS[6], 0.4, 0.1)
        if k > 0:
            pressed = 41.2 < t < 41.5
            on = t > 41.2
            s = 1 - (0.08 if pressed else 0)
            self.add_bb(it, cam, P(y0, 0.95, 1.3, 0.05), 0.3 * k * s, T["btn2"] if on else T["btn"], anchor=(0.5, 0.5))

        # 8. Спасибо (персонаж в центре): живой чат летит вверх; «напиши идею»
        y0 = SEC["thx"]
        r_, c_ = basis(y0)
        if CUTS[6] < t < 50.8:
            rng = np.random.default_rng(8)
            for i in range(34):
                s0 = 42.6 + rng.uniform(0, 7.4)
                p = (t - s0) / 1.8
                if not 0 < p < 1:
                    continue
                side = rng.choice([-1, 1]) * rng.uniform(0.55, 0.82)
                pos = r_ * side + c_ * -0.2 + [0, 0.4 + 1.6 * p, 0]
                kk = out_back(clamp(p / 0.15)) * (1 - smooth((p - 0.8) / 0.2))
                self.add_bb(it, cam, pos, 0.12 * kk, T["chips"][i % len(T["chips"])], anchor=(0.5, 0.5))
        k = pop(t, 46.0, 50.8, 0.4, 0.3)
        if k > 0:
            msg = "Сними про сон!"
            n = int(len(msg) * clamp((t - 46.9) / 1.4))
            pos = P(y0, 0.0, 2.2) + [0, 0.02 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 0), 0.75 * k, 0.246 * k, 0.04 * k, self.cached("idea", ideas_tex, msg[:n]),
                           pitch=-6, side=(70, 30, 50), edge=(255, 200, 220))

        # 9. Концовка с Telegram — общий шаблон
        self.outro_items(t, cam, it, OUTRO, SEC["tg"])


SFX = [(t, "whoosh") for t in CUTS + (50.7,)]


def render(voice, out, frames):
    sc = Scene(voice)
    wr = mg.Writer(out, FPS)
    for n, i in enumerate(frames):
        t = i / FPS
        speed = abs(sc.cam(t + 0.5 / FPS).yaw - sc.cam(t - 0.5 / FPS).yaw) * FPS
        cut = any(abs(t - c) < 1.0 / FPS for c in CUTS)
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
        for t in (1.0, 2.3, 4.0, 6.4, 7.3, 10.5, 14.0, 16.7, 19.8, 23.0, 25.5, 28.5, 32.1, 33.8, 36.8, 40.3, 41.35,
                  43.5, 45.2, 48.3, 54.0, 59.0):
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
    base.mix_audio(tmp, voice, out, music=os.path.join(ROOT, "work", "stream", "music.wav"), sfx_list=SFX, dur=DUR,
                   sfx_vol=0.15)  # «вжух» лёгкий и тихий
    os.remove(tmp)


if __name__ == "__main__":
    main()
