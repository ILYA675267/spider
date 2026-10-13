"""Ролик №6: «Почему свой голос на записи звучит так противно?».

python tools/voice_mg.py work/voice/voice_clean.wav output/voice.mp4 [--preview] [--jobs 4]
Общий движок, персонаж и концовка с Telegram — из memory_mg.py; параллельный рендер — как в money_mg.py.
Новое в монтаже: резкий наезд в начале; эквалайзер вокруг паучка прыгает в такт голосу; короткие «рывки»
камеры на ударных словах; разделённый экран «как слышишь ты / как слышат все» (два ракурса сразу);
схема головы с двумя путями звука; микрофон, который ловит волны; красная вспышка на «пугается».
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
OUTRO = float(open(os.path.join(ROOT, "work", "voice", "outro_start.txt")).read())  # 38.86
DUR = 47.5
WARM_RGB, GOLD_RGB, GREEN_RGB = (240, 140, 60), (240, 190, 60), (46, 196, 110)
WARM, BLUE = WARM_RGB[::-1], BLUE_RGB[::-1]

# ---------------------------------------------------------------- субтитры
T_, W_, B_, G_ = TG_RGB, WARM_RGB, BLUE_RGB, GREEN_RGB
SUBS = [
    (0.229, "ПОЧЕМУ СВОЙ ГОЛОС НА ЗАПИСИ", ("ГОЛОС",)), (1.629, "ЗВУЧИТ ТАК ПРОТИВНО?", ("ПРОТИВНО?",)),
    (2.95, "", ()),
    (4.429, "КОГДА ТЫ ГОВОРИШЬ,", ()), (5.409, "ЗВУК ДОХОДИТ ДО УШЕЙ", ("ЗВУК",)),
    (6.489, "ДВУМЯ ПУТЯМИ:", ("ДВУМЯ",)), (7.489, "ПО ВОЗДУХУ", ("ВОЗДУХУ",), B_),
    (8.129, "И ЧЕРЕЗ КОСТИ ЧЕРЕПА", ("КОСТИ",), W_), (9.609, "КОСТИ ЛУЧШЕ ПЕРЕДАЮТ", ("КОСТИ",), W_),
    (10.609, "НИЗКИЕ ЗВУКИ,", ("НИЗКИЕ",), W_), (11.649, "ПОЭТОМУ ИЗНУТРИ ГОЛОС", ()),
    (12.689, "КАЖЕТСЯ ТЕБЕ НИЖЕ", ("НИЖЕ",), W_), (13.649, "И ПРИЯТНЕЕ", ("ПРИЯТНЕЕ",), W_),
    (14.841, "А МИКРОФОН ЛОВИТ", ("МИКРОФОН",), B_), (15.781, "ТОЛЬКО ВОЗДУХ", ("ВОЗДУХ",), B_),
    (16.961, "НА ЗАПИСИ ТЫ СЛЫШИШЬ", ()), (18.061, "СВОЙ ГОЛОС ТАК,", ()),
    (18.841, "КАК ЕГО СЛЫШАТ ВСЕ", ("ВСЕ",), B_), (20.401, "МОЗГ НЕ УЗНАЁТ", ("МОЗГ",)),
    (21.401, "ПРИВЫЧНЫЙ ЗВУК", ()), (22.181, "И ПУГАЕТСЯ", ("ПУГАЕТСЯ",)),
    (23.281, "У ПСИХОЛОГОВ ДАЖЕ", ()), (24.241, "ЕСТЬ НАЗВАНИЕ:", ()),
    (25.481, "ВСТРЕЧА С СОБСТВЕННЫМ", ("ВСТРЕЧА",), B_), (26.461, "ГОЛОСОМ", ("ГОЛОСОМ",), B_), (27.416, "ХОРОШАЯ НОВОСТЬ:", ("ХОРОШАЯ",), G_),
    (28.616, "В ОДНОМ ОПЫТЕ ЛЮДИ,", ()), (29.996, "НЕ УЗНАВ СВОЙ ГОЛОС", ()),
    (30.936, "НА ЗАПИСИ,", ()), (31.776, "ОЦЕНИВАЛИ ЕГО", ()), (32.456, "ВПОЛНЕ ПРИЯТНЫМ", ("ПРИЯТНЫМ",), G_),
    (33.656, "ТАК ЧТО ЗВУЧИШЬ ТЫ", ()), (34.436, "НОРМАЛЬНО", ("НОРМАЛЬНО",), G_),
    (35.416, "ПРОСТО ПОСЛУШАЙ СЕБЯ", ()), (36.496, "ЕЩЁ ПАРУ РАЗ", ("ПАРУ",), G_),
    (37.396, "И МОЗГ ПРИВЫКНЕТ", ("ПРИВЫКНЕТ",), G_),
    (38.83, "А ЕЩЁ БОЛЬШЕ ИНТЕРЕСНОГО —", (), T_), (40.19, "В МОЁМ ТЕЛЕГРАМ-КАНАЛЕ", ("ТЕЛЕГРАМ-КАНАЛЕ",), T_),
    (41.52, "«ПАУЧЬЯ НОРКА»", ("«ПАУЧЬЯ", "НОРКА»"), T_), (42.64, "ТАМ ЭКСКЛЮЗИВ,", ("ЭКСКЛЮЗИВ",), T_),
    (43.34, "КОТОРОГО НЕТ В РОЛИКАХ", (), T_), (44.64, "ССЫЛКА В ПРОФИЛЕ —", ("ПРОФИЛЕ",), T_),
    (45.66, "ЗАЛЕТАЙ!", ("ЗАЛЕТАЙ!",), T_),
]
NOSUBS = [(0.0, "", ())]

# ---------------------------------------------------------------- камера и сцены
SPLIT = (16.9, 20.3)   # разделённый экран: слева «как слышишь ты», справа «как слышат все»
SPLIT_CAMS = ((150.0, 3.3, 1.1, 0.84, 50.0), (222.0, 3.3, 1.1, 0.84, 50.0))
CAMERA = [
    (0.0, (-10.0, 5.2, 1.7, 0.9, 70.0)),                   # резкий наезд
    (0.45, (-8.0, 3.6, 1.15, 0.84, 50.0), "out"),
    (2.85, (-4.0, 3.6, 1.15, 0.84, 50.0)),
    (3.2, (0.0, 3.9, 1.3, 0.98, 50.0), "inout"),           # заставка
    (4.2, (2.0, 3.9, 1.3, 0.98, 50.0)),
    (4.6, (95.0, 3.6, 1.2, 0.84, 50.0), "expo"),           # два пути звука
    (14.3, (101.0, 3.5, 1.2, 0.86, 50.0)),
    (14.7, (185.0, 3.6, 1.2, 0.86, 50.0), "expo"),         # микрофон
    (SPLIT[1] - 0.01, (190.0, 3.6, 1.2, 0.86, 50.0)),
    (SPLIT[1] + 0.01, (275.0, 3.6, 1.2, 0.84, 50.0), "lin"),  # мозг пугается
    (27.0, (281.0, 3.35, 1.2, 0.86, 50.0)),
    (27.4, (365.0, 3.6, 1.2, 0.84, 50.0), "expo"),         # хорошая новость
    (38.4, (378.0, 3.7, 1.25, 0.86, 50.0), "lin"),
    (38.8, (450.0, 4.3, 1.3, 1.12, 50.0), "expo"), (DUR, (451.0, 4.0, 1.3, 1.15, 50.0)),
]
OFFSET = [(0.0, -0.5), (2.85, -0.5), (3.2, 0.0, "inout"), (4.2, 0.0), (4.6, 0.55, "expo"), (14.3, 0.55),
          (14.7, -0.55, "expo"), (SPLIT[1] - 0.01, -0.55), (SPLIT[1] + 0.01, 0.55, "lin"), (27.0, 0.55),
          (27.4, -0.55, "expo"), (38.4, -0.55), (38.8, 0.0, "expo"), (DUR, 0.0)]
PUNCH = (2.11, 8.35, 13.29, 15.94, 22.24, 34.44)  # короткий «рывок» камеры на ударных словах
TINT = [(0.0, (1.12, 0.92, 1.08)), (4.2, (1.12, 0.92, 1.08)), (4.6, (1.0, 1.0, 1.0)), (14.4, (1.0, 1.0, 1.0)),
        (14.8, (1.16, 1.0, 0.86)), (SPLIT[1] - 0.01, (1.16, 1.0, 0.86)), (SPLIT[1] + 0.01, (1.0, 0.95, 1.05), "lin"),
        (27.1, (1.0, 0.95, 1.05)), (27.5, (0.88, 1.08, 0.94)), (38.5, (0.88, 1.08, 0.94)), (38.9, (1.0, 1.0, 1.0)),
        (DUR, (1.0, 1.0, 1.0))]
SPLIT_TINT = ((0.84, 0.98, 1.2), (1.22, 1.0, 0.82))   # тёплый «изнутри» / холодный «запись»
SEC = {"hook": -8, "paths": 99, "mic": 187, "brain": 278, "good": 371, "tg": 450.5}
SHAKES = (22.24,)
FLASHES = ((2.95, 0.15), (4.45, 0.08), (14.55, 0.08), (16.88, 0.15), (20.28, 0.15), (27.25, 0.08), (38.65, 0.12))
JUMPS = [22.2]
ALARM = (22.15, 22.95)
SPOT = None

G = lambda *k: list(k)  # noqa: E731
ARM_L = G((0, 62), (0.45, 62), (0.7, -25, "back"), (1.9, -22), (2.1, 36, "back"), (2.8, 38), (3.05, -50, "back"),
          (4.0, -45), (4.4, 62, "back"), (14.8, 62), (15.0, -20, "back"), (16.7, -18), (16.9, 30, "back"),
          (20.2, 32), (20.35, 62), (22.15, 62), (22.25, -50, "back"), (22.9, -45), (23.2, 62, "back"), (27.3, 62),
          (27.55, -30, "back"), (31.6, -28), (31.8, -12, "back"), (35.3, -10), (35.5, -35, "back"), (38.2, -33),
          (38.5, 62), (45.77, 62), (45.92, -50, "back"), (46.52, -45), (46.92, 62, "back"), (DUR, 62))
ARM_R = G((0, 62), (2.0, 62), (2.15, 36, "back"), (2.8, 38), (3.05, -50, "back"), (4.0, -45), (4.4, 62, "back"),
          (4.6, -30, "back"), (9.4, -28), (9.65, -12, "back"), (14.2, -10), (14.5, 62), (16.9, 30, "back"),
          (20.2, 32), (20.35, -30, "back"), (22.0, -28), (22.25, -50, "back"), (22.9, -45), (23.25, -15, "back"),
          (27.0, -12), (27.3, 62), (38.62, 62), (39.08, -48, "back"), (41.43, -45), (41.73, -55, "back"),
          (44.43, -52), (44.68, -40, "back"), (45.63, -42), (45.78, -50, "back"), (46.52, -45), (46.67, -58),
          (46.82, -32), (46.97, -58), (47.12, -32), (DUR, 62))
HEAD = G((0, 0), (0.7, -6), (2.1, 8), (2.35, -8), (2.6, 8), (2.85, 0), (4.6, -6), (7.5, -7), (8.2, 6), (9.65, -5),
         (12.7, 8), (14.2, 0), (15.0, 6), (16.8, 0), (20.4, -6), (22.2, 9), (22.9, -4), (23.3, -6), (27.5, 7),
         (31.8, 6), (34.4, -8), (35.5, 6), (38.0, 4), (39.08, -6), (45.8, 0), (DUR, 0))


# ---------------------------------------------------------------- картинки вставок
def rec_tex(t):
    """Диктофон: мигающая точка REC и волна записи."""
    im = rounded((620, 400), 40, (18, 18, 26, 255), (80, 80, 100, 255), 4)
    d = ImageDraw.Draw(im)
    if int(t * 2.5) % 2 == 0:
        d.ellipse([40, 40, 80, 80], fill=RED_RGB + (255,))
    d.text((100, 34), "ЗАПИСЬ", font=mg.font(44), fill="white")
    d.text((420, 40), f"0:0{min(9, int(t * 2))}", font=mg.font(40), fill=(150, 150, 170))
    for i in range(36):
        x = 40 + i * 15
        a = 0.2 + 0.8 * abs(math.sin(i * 0.7 + t * 6)) * abs(math.sin(i * 0.23 + t * 2))
        h = int(110 * a)
        d.rounded_rectangle([x, 210 - h, x + 8, 210 + h], 4, fill=(255, 120, 120, 255))
    d.rounded_rectangle([40, 340, 580, 352], 6, fill=(44, 44, 58, 255))
    d.rounded_rectangle([40, 340, 40 + int(540 * min(1, t / 3)), 352], 6, fill=RED_RGB + (255,))
    return mg.premul(im)


def head_tex(pa, pb, ph):
    """Схема головы: путь по воздуху (синий, снаружи) и через кости (оранжевый, внутри). ph — бег волн."""
    im = rounded((780, 600), 34, (18, 18, 26, 255), (72, 72, 92, 255), 3)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([36, 30, 330, 92], 30, fill=GRAY_RGB + (255,))
    d.text((58, 38), "ДВА ПУТИ ЗВУКА", font=mg.font(36), fill="white")
    # голова в профиль (смотрит вправо)
    d.ellipse([200, 140, 480, 440], fill=(40, 40, 54, 255), outline=(120, 120, 140, 255), width=5)
    d.polygon([(450, 300), (520, 360), (470, 380)], fill=(40, 40, 54, 255))  # нос/подбородок
    d.rounded_rectangle([300, 270, 330, 330], 14, fill=(150, 150, 170, 255))  # ухо
    mouth, ear = (470, 400), (315, 300)
    d.ellipse([mouth[0] - 10, mouth[1] - 10, mouth[0] + 10, mouth[1] + 10], fill=(255, 255, 255, 255))
    # по воздуху: дуга снаружи — от рта вперёд, вниз и назад к уху
    if pa > 0:
        pts = []
        for s in np.linspace(0, pa, 40):
            a = math.radians(-20 + 230 * s)
            pts.append((340 + 230 * math.cos(a) * (1 - 0.15 * s), 300 + 200 * math.sin(a)))
        pts = [mouth] + pts + ([ear] if pa > 0.98 else [])
        for i in range(len(pts) - 1):
            if (i + int(ph * 10)) % 3 != 0:
                d.line([pts[i], pts[i + 1]], fill=(90, 150, 255, 255), width=9)
        d.text((560, 470), "по воздуху", font=mg.font(36), fill=(120, 170, 255))
    # через кости: прямая светящаяся дорожка внутри головы
    if pb > 0:
        x = mouth[0] + (ear[0] - mouth[0]) * pb
        y = mouth[1] + (ear[1] - mouth[1]) * pb
        d.line([mouth, (x, y)], fill=WARM_RGB + (255,), width=14)
        for k in range(3):
            q = ((ph * 1.5 + k / 3) % 1) * pb
            cx, cy = mouth[0] + (ear[0] - mouth[0]) * q, mouth[1] + (ear[1] - mouth[1]) * q
            d.ellipse([cx - 12, cy - 12, cx + 12, cy + 12], fill=(255, 220, 160, 255))
        d.text((40, 470), "через кости", font=mg.font(36), fill=(255, 180, 110))
    return mg.premul(im)


def eq_tex(boost):
    """Эквалайзер: кости усиливают низкие — левые столбики растут."""
    im = rounded((780, 520), 34, (18, 18, 26, 255), (72, 72, 92, 255), 3)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([36, 30, 420, 92], 30, fill=WARM_RGB + (255,))
    d.text((58, 38), "КОСТИ УСИЛИВАЮТ", font=mg.font(36), fill="white")
    base_v = [0.45, 0.5, 0.55, 0.5, 0.45, 0.4, 0.42, 0.38]
    for i, v in enumerate(base_v):
        add = boost * max(0.0, 0.45 - i * 0.11)
        h = int(280 * min(1.0, v + add))
        x = 60 + i * 85
        col = WARM_RGB if add > 0.02 else (90, 110, 150)
        d.rounded_rectangle([x, 420 - h, x + 60, 420], 18, fill=col + (255,))
    d.text((60, 440), "низкие", font=mg.font(34), fill=(255, 180, 110))
    d.text((590, 440), "высокие", font=mg.font(34), fill=(150, 150, 170))
    return mg.premul(im)


def mic_tex():
    im = Image.new("RGBA", (200, 300), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([30, 0, 170, 200], 70, fill=(70, 72, 86, 255), outline=(160, 160, 180, 255), width=6)
    for y in range(40, 180, 26):
        d.line([(50, y), (150, y)], fill=(110, 112, 130), width=5)
    d.rounded_rectangle([80, 200, 120, 300], 14, fill=(60, 60, 72, 255))
    return mg.premul(im)


def ring_tex(col):
    im = Image.new("RGBA", (300, 300), (0, 0, 0, 0))
    ImageDraw.Draw(im).ellipse([10, 10, 289, 289], outline=col + (255,), width=14)
    return mg.premul(im)


def rating_tex(stars, reveal):
    im = rounded((780, 470), 34, (18, 18, 26, 255), (72, 72, 92, 255), 3)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([36, 30, 200, 92], 30, fill=GREEN_RGB + (255,))
    d.text((58, 38), "ОПЫТ", font=mg.font(36), fill="white")
    d.text((40, 120), "чей-то голос на записи:", font=mg.font(42), fill=(200, 200, 215))
    for i in range(5):
        cx, cy, r = 110 + i * 140, 250, 54
        pts = []
        for k in range(10):
            a = -math.pi / 2 + k * math.pi / 5
            rr = r if k % 2 == 0 else r * 0.45
            pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
        d.polygon(pts, fill=GOLD_RGB + (255,) if i < stars else (50, 50, 64, 255))
    if reveal:
        d.rounded_rectangle([40, 350, 740, 430], 30, fill=(30, 60, 44, 255))
        d.text((64, 362), "а это был их собственный голос", font=mg.font(40), fill="white")
    return mg.premul(im)


def loop_tex(n, fill):
    im = rounded((660, 330), 34, (18, 26, 22, 255), GREEN_RGB + (255,), 5)
    d = ImageDraw.Draw(im)
    d.ellipse([40, 40, 160, 160], fill=GREEN_RGB + (255,))
    d.polygon([(85, 70), (85, 130), (135, 100)], fill="white")
    d.text((200, 50), f"слушаешь ×{n}", font=mg.font(64, tall=True), fill="white")
    d.text((40, 200), "мозг привыкает", font=mg.font(38), fill=(150, 220, 180))
    d.rounded_rectangle([40, 260, 620, 290], 15, fill=(40, 52, 46, 255))
    if fill > 0.03:
        d.rounded_rectangle([40, 260, 40 + int(580 * fill), 290], 15, fill=GREEN_RGB + (255,))
    return mg.premul(im)


# ---------------------------------------------------------------- сцена
class Scene(base.Scene):
    cfg = sys.modules[__name__]

    def __init__(self, voice):
        super().__init__(voice)
        T = self.T
        T["eww"] = stamp("ФУ!", RED_RGB)
        T["mic"] = mic_tex()
        T["ring_b"], T["ring_w"] = ring_tex((90, 150, 255)), ring_tex(WARM_RGB)
        T["only_air"] = chip("ТОЛЬКО ВОЗДУХ", BLUE_RGB)
        T["brain"] = card("МОЗГ", "ЧЕЙ ЭТО ГОЛОС?!", "звук не совпадает с привычным", BLUE_RGB)
        T["notme"] = stamp("ЭТО НЕ Я!", RED_RGB)
        T["term"] = card("У ПСИХОЛОГОВ", "ВСТРЕЧА С ГОЛОСОМ", "voice confrontation, 1960-е", BLUE_RGB)
        T["good"] = chip("ХОРОШАЯ НОВОСТЬ", GREEN_RGB)
        T["ok"] = stamp("ЗВУЧИШЬ ОК!", GREEN_RGB)
        T["you"], T["all"] = chip("ТАК СЛЫШИШЬ ТЫ", WARM_RGB), chip("ТАК СЛЫШАТ ВСЕ", BLUE_RGB)
        T["you_s"], T["all_s"] = base.label("ниже, бархатнее", (255, 190, 130), 44), base.label("выше, тоньше", (150, 180, 255), 44)
        self.cache = {}
        self.side = None
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
        m = np.array(SPLIT_TINT[self.side] if self.side is not None else track(TINT, self.t), np.float32)
        if np.abs(m - 1).max() < 1e-3:
            return frame
        return np.clip(frame.astype(np.float32) * m, 0, 255).astype(np.uint8)

    def draw(self, t):
        self.t = t
        if not (SPLIT[0] <= t < SPLIT[1]):
            return super().draw(t)
        # разделённый экран: два ракурса, из каждого берём середину
        halves = []
        self.cfg = self.nosub
        for s in (0, 1):
            self.side = s
            fr = super().draw(t)
            halves.append(fr[:, W // 4: W // 4 + W // 2].copy())
        self.side = None
        del self.cfg
        out = np.concatenate(halves, 1)
        self.split_overlay(out, t)
        mg.subtitles(out, t, SUBS)
        self.frame = out
        return out

    def split_overlay(self, out, t):
        T = self.T
        k = out_back(clamp((t - SPLIT[0]) / 0.35))
        cv2.line(out, (W // 2, 0), (W // 2, H), (230, 230, 240), 6, cv2.LINE_AA)
        env = self.env[min(len(self.env) - 1, int(t * FPS))]
        for s, (key, sub, col, lam, thick) in enumerate((("you", "you_s", WARM, 260, 12), ("all", "all_s", BLUE, 90, 6))):
            x0 = s * W // 2
            tex = T[key]
            sc = 0.8 * k
            if sc > 0.05:
                th = cv2.resize(tex, (max(2, int(tex.shape[1] * sc)), max(2, int(tex.shape[0] * sc))), interpolation=cv2.INTER_AREA)
                mg.over(out, th, x0 + (W // 2 - th.shape[1]) // 2, 250)
                ts = T[sub]
                mg.over(out, ts, x0 + (W // 2 - ts.shape[1]) // 2, 250 + th.shape[0] + 14)
            xs = np.arange(x0 + 40, x0 + W // 2 - 40, 4)
            amp = (25 + 70 * env) * k
            ys = 520 + amp * np.sin((xs - x0) * 2 * math.pi / lam - t * 9)
            pts = np.stack([xs, ys], -1).astype(np.int32)
            glow = np.zeros_like(out)
            cv2.polylines(glow, [pts], False, col, thick + 10, cv2.LINE_AA)
            cv2.add(out, cv2.GaussianBlur(glow, (0, 0), 8), out)
            cv2.polylines(out, [pts], False, col, thick, cv2.LINE_AA)

    def cam(self, t):
        if self.side is not None:
            yaw, dist, hgt, ty, fov = SPLIT_CAMS[self.side]
            yaw += 6 * (t - SPLIT[0]) * (1 if self.side == 0 else -1)
            return mg.Cam(yaw + 0.5 * wobble(t, 1), dist, hgt, ty + 0.015 * wobble(t, 3), fov=fov,
                          roll=0.5 * wobble(t * 0.8, 4))
        yaw, dist, hgt, ty, fov = track(CAMERA, t)
        off = track(OFFSET, t)
        for p in PUNCH:
            if p < t < p + 0.5:
                q = t - p
                fov *= 1 - 0.1 * (smooth(q / 0.06) if q < 0.06 else 1 - smooth((q - 0.06) / 0.44))
        a = math.radians(yaw)
        shake = 0.0
        for s0 in SHAKES:
            if s0 < t < s0 + 0.35:
                shake = (1 - (t - s0) / 0.35) * math.sin((t - s0) * 70)
        return mg.Cam(yaw + 0.5 * wobble(t, 1) + 1.2 * shake, dist, hgt + 0.02 * wobble(t, 2) + 0.03 * shake,
                      ty + 0.015 * wobble(t, 3), fov=fov, roll=0.5 * wobble(t * 0.8, 4) + 1.5 * shake,
                      tx=off * math.cos(a), tz=off * math.sin(a))

    def equalizer(self, it, cam, t, k):
        """Столбики-эквалайзер по кругу на полу, прыгают в такт голосу; только позади персонажа."""
        env = self.env[min(len(self.env) - 1, int(t * FPS))]
        cd = cam.depth((0.0, 0.3, 0.0))
        for i in range(22):
            a = 2 * math.pi * i / 22
            x, z = 0.95 * math.cos(a), 0.95 * math.sin(a)
            if cam.depth((x, 0.3, z)) < cd + 0.25:
                continue
            h = (0.05 + 0.5 * env * (0.4 + 0.6 * abs(math.sin(i * 1.7 + t * 5)))) * k
            col = tuple(int(c) for c in np.array(WARM) * (0.6 + 0.4 * env))
            self.add_box(it, cam, (x, z), (0.11, 0.11), h, col, radius=0.04)

    def items_for(self, t, cam, it):
        if self.side is not None:
            return
        T = self.T
        P = lambda y0, *a, **k: place(y0, *a, **k)  # noqa: E731

        # 1. Крючок (персонаж справа): диктофон пишет, «фу!»; эквалайзер на полу
        y0 = SEC["hook"]
        ke = pop(t, 0.2, 4.4, 0.5, 0.4)
        if ke > 0:
            self.equalizer(it, cam, t, ke)
        k = pop(t, 0.4, 2.95, 0.35, 0.3)
        if k > 0:
            tq = round(t * 15) / 15
            pos = P(y0, -0.75, 1.6) + [0, 0.02 * math.sin(t * 3), 0]
            self.add_panel(it, cam, pos, face(y0, -12), 0.7 * k, 0.452 * k, 0.05 * k, self.cached("rec", rec_tex, tq),
                           pitch=-5, side=(60, 40, 50), edge=(255, 200, 200))
        k = pop(t, 2.1, 2.95, 0.15, 0.25)
        if k > 0:
            ks = clamp((t - 2.1) / 0.15)
            self.add_bb(it, cam, P(y0, -0.55, 1.95, 0.1), 0.16 * (1 + 0.6 * (1 - ks)) * k, T["eww"], anchor=(0.5, 0.5))

        # 2. Заставка — карточка переворачивается к зрителю
        k = pop(t, 3.05, 4.5, 0.35, 0.3)
        if k > 0:
            flip = 180 * (1 - smooth((t - 3.05) / 0.45))
            self.add_panel(it, cam, place(0, 0.0, 1.95, 0.35), face(0, -6) + flip, 1.2 * k, 0.564 * k, 0.11 * k,
                           T["stinger"], pitch=-6, side=(30, 24, 120), edge=(200, 190, 255))

        # 3. Два пути звука (схема головы), эквалайзер «кости усиливают низкие»
        y0 = SEC["paths"]
        k = pop(t, 4.7, 9.55, 0.4, 0.3)
        if k > 0:
            pa = round(smooth((t - 7.45) / 0.8) * 20) / 20
            pb = round(smooth((t - 8.15) / 0.6) * 20) / 20
            ph = round((t * 1.2) % 1 * 12) / 12
            pos = P(y0, 0.72, 1.58) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 12), 0.9 * k, 0.692 * k, 0.06 * k,
                           self.cached("head", head_tex, pa, pb, ph), pitch=-5, side=(50, 50, 70), edge=(220, 220, 255))
        k = pop(t, 9.5, 14.45, 0.4, 0.3)
        if k > 0:
            b = round(smooth((t - 10.6) / 0.8) * 16) / 16
            pos = P(y0, 0.72, 1.55) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 12), 0.9 * k, 0.6 * k, 0.06 * k, self.cached("eq", eq_tex, b),
                           pitch=-5, side=(70, 50, 40), edge=(255, 210, 180))

        # 4. Микрофон ловит только воздух (персонаж справа)
        y0 = SEC["mic"]
        r_, c_ = basis(y0)
        k = pop(t, 14.75, SPLIT[0] + 0.1, 0.45, 0.2)
        if k > 0:
            mp = r_ * -0.95 + c_ * -0.05
            self.add_box(it, cam, (mp[0], mp[2]), (0.3, 0.3), 0.03, (60, 60, 72), radius=0.12)
            self.add_box(it, cam, (mp[0], mp[2]), (0.04, 0.04), 1.0 * k, (80, 80, 94), radius=0.015)
            self.add_bb(it, cam, mp + [0, 1.0 * k, 0], 0.32 * k, T["mic"], anchor=(0.5, 0.35))
            for i in range(4):  # синие волны летят от паучка к микрофону
                q = ((t - 14.9) * 0.8 + i / 4) % 1
                if t < 14.9 + i * 0.3:
                    continue
                pos = (r_ * -0.38 + c_ * -0.12 + [0, 1.05, 0]) * (1 - q) + (mp + [0, 1.08, 0]) * q
                self.add_bb(it, cam, pos, (0.1 + 0.15 * q) * k, T["ring_b"] * (1 - q), anchor=(0.5, 0.5))
            kc = pop(t, 15.7, SPLIT[0] + 0.1, 0.3, 0.2)
            if kc > 0:
                self.add_bb(it, cam, P(y0, -0.85, 1.95), 0.12 * kc, T["only_air"], anchor=(0.5, 0.5))

        # 5. Мозг не узнаёт и пугается; «встреча с собственным голосом»
        y0 = SEC["brain"]
        k = pop(t, SPLIT[1] + 0.1, 23.2, 0.4, 0.3)
        if k > 0:
            pos = P(y0, 0.72, 1.5) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 14), 0.9 * k, 0.48 * k, 0.06 * k, T["brain"], pitch=-6,
                           side=(40, 50, 90), edge=(200, 210, 255))
        k = pop(t, 22.24, 23.2, 0.15, 0.25)
        if k > 0:
            ks = clamp((t - 22.24) / 0.15)
            self.add_bb(it, cam, P(y0, 0.8, 1.95, 0.1), 0.16 * (1 + 0.6 * (1 - ks)) * k, T["notme"], anchor=(0.5, 0.5))
        k = pop(t, 23.25, 27.3, 0.4, 0.3)
        if k > 0:
            pos = P(y0, 0.72, 1.55) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 14), 0.9 * k, 0.48 * k, 0.06 * k, T["term"], pitch=-6,
                           side=(40, 50, 90), edge=(200, 210, 255))

        # 6. Хорошая новость (персонаж справа): звёзды, «звучишь ок!», «слушаешь ×3 — мозг привыкает»
        y0 = SEC["good"]
        ke = pop(t, 27.4, 38.6, 0.5, 0.4)
        if ke > 0:
            self.equalizer(it, cam, t, ke)
        k = pop(t, 27.45, 28.7, 0.3, 0.25)
        if k > 0:
            self.add_bb(it, cam, P(y0, -0.78, 1.95), 0.13 * k, T["good"], anchor=(0.5, 0.5))
        k = pop(t, 28.65, 35.4, 0.4, 0.3)
        if k > 0:
            stars = int(4 * smooth((t - 31.9) / 0.8) + 0.999) if t > 31.9 else 0
            pos = P(y0, -0.74, 1.55) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, -12), 0.9 * k, 0.542 * k, 0.06 * k,
                           self.cached("rate", rating_tex, stars, t > 33.0), pitch=-5, side=(40, 70, 50), edge=(200, 255, 220))
        k = pop(t, 34.44, 35.4, 0.15, 0.25)
        if k > 0:
            ks = clamp((t - 34.44) / 0.15)
            self.add_bb(it, cam, P(y0, -0.8, 1.05, 0.1), 0.16 * (1 + 0.6 * (1 - ks)) * k, T["ok"], anchor=(0.5, 0.5))
        k = pop(t, 35.4, 38.65, 0.4, 0.3)
        if k > 0:
            n = 1 + (t > 36.6) + (t > 37.0)
            fill = round(smooth((t - 37.4) / 0.7) * 12) / 12
            pos = P(y0, -0.74, 1.55) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, -12), 0.8 * k, 0.4 * k, 0.05 * k, self.cached("loop", loop_tex, n, fill),
                           pitch=-5, side=(40, 70, 50), edge=(200, 255, 220))

        # 7. Концовка с Telegram — общий шаблон
        self.outro_items(t, cam, it, OUTRO, SEC["tg"])


SFX = [(t, "whoosh") for t in (0.0, 2.95, 4.4, 14.5, SPLIT[0] - 0.05, SPLIT[1] - 0.05, 27.2, 38.6)]


def render(voice, out, frames):
    sc = Scene(voice)
    wr = mg.Writer(out, FPS)
    for n, i in enumerate(frames):
        t = i / FPS
        speed = abs(sc.cam(t + 0.5 / FPS).yaw - sc.cam(t - 0.5 / FPS).yaw) * FPS
        cut = any(abs(t - c) < 1.0 / FPS for c in SPLIT)
        sub = 5 if speed > math.radians(60) and not cut and not (SPLIT[0] <= t < SPLIT[1]) else 1
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
        for t in (0.1, 1.2, 2.2, 2.6, 3.4, 4.0, 6.0, 7.9, 8.8, 11.5, 13.4, 15.2, 16.4, 17.5, 19.6, 21.0, 22.4, 24.5,
                  26.5, 27.8, 30.5, 32.9, 34.6, 36.8, 37.9, 40.0, 45.0):
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
    base.mix_audio(tmp, voice, out, music=os.path.join(ROOT, "work", "voice", "music.wav"), sfx_list=SFX, dur=DUR,
                   sfx_vol=0.15)  # «вжух» лёгкий и тихий
    os.remove(tmp)


if __name__ == "__main__":
    main()
