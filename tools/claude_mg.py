"""Ролик №25: «Как появился Claude?» — история Anthropic (+ рекламный баннер в середине).

python tools/claude_mg.py work/claude/voice_clean.wav output/claude.mp4 [--preview] [--jobs 4]
затем: python tools/ad_banner.py insert output/claude.mp4 30.95 output/claude_ad.mp4 claude_mg
Общий движок, персонаж и концовка с Telegram — из memory_mg.py; параллельный рендер — как в money_mg.py.
Новое в монтаже: тёплая терракотовая гамма; настоящие логотипы Claude, Anthropic и OpenAI; знак Claude
медленно вращается и светится; стрелка «OpenAI → Anthropic»; свиток «конституции», в котором по одному
появляются правила; смена сцены «полосами» — кадр режется на горизонтальные ленты, которые разъезжаются
в разные стороны; чат с первым сообщением Claude; поток нулей и единиц у Клода Шеннона; три модели —
маленькая, средняя и большая; иконки «код, документы, этот ролик».
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
OUTRO = float(open(os.path.join(ROOT, "work", "claude", "outro_start.txt")).read())  # 51.79
DUR = 60.45
AD_CUT = 30.95                   # здесь вклеивается баннер (середина ролика с баннером ±5 с)
CLAY_RGB, CREAM_RGB, GREEN_RGB, GOLD_RGB, PURPLE_RGB = (217, 119, 87), (240, 230, 214), (46, 196, 110), (245, 190, 50), (150, 90, 230)


def logo(name, height):
    im = Image.open(os.path.join(ROOT, "assets", "logos", name)).convert("RGBA")
    return im.resize((int(im.width * height / im.height), height), Image.LANCZOS)


# ---------------------------------------------------------------- субтитры
T_, Cl_, G_, Go_, Pu_ = TG_RGB, CLAY_RGB, GREEN_RGB, GOLD_RGB, PURPLE_RGB
SUBS = [
    (0.169, "КАК ПОЯВИЛСЯ CLAUDE —", ("CLAUDE",), Cl_), (1.489, "ОДНА ИЗ", ()),
    (2.209, "САМЫХ УМНЫХ НЕЙРОСЕТЕЙ?", ("УМНЫХ",), Cl_), (3.5, "", ()),
    (5.429, "В 2021 ГОДУ", ("2021",), Go_), (7.129, "БРАТ И СЕСТРА", ()), (8.089, "ДАРИО И ДАНИЭЛА АМОДЕИ", ("АМОДЕИ",), Cl_),
    (9.709, "УШЛИ ИЗ OPENAI", ("OPENAI",)), (11.289, "И ВМЕСТЕ С КОЛЛЕГАМИ", ()),
    (12.669, "ОСНОВАЛИ ANTHROPIC", ("ANTHROPIC",), Cl_),
    (14.88, "ГЛАВНАЯ ИДЕЯ —", ()), (16.32, "СДЕЛАТЬ ИСКУССТВЕННЫЙ", ()), (17.24, "ИНТЕЛЛЕКТ БЕЗОПАСНЫМ", ("БЕЗОПАСНЫМ",), G_),
    (19.21, "ДЛЯ ЭТОГО ОНИ ПРИДУМАЛИ", ()), (20.43, "«КОНСТИТУЦИЮ»:", ("«КОНСТИТУЦИЮ»:",), Cl_),
    (21.63, "СВОД ПРАВИЛ, ПО КОТОРЫМ", ()), (22.93, "МОДЕЛЬ САМА УЧИТСЯ", ()),
    (24.19, "ОТВЕЧАТЬ ЧЕСТНО И ПОЛЕЗНО", ("ЧЕСТНО", "ПОЛЕЗНО"), G_),
    (26.37, "В МАРТЕ 2023 ГОДА", ("2023",), Go_), (29.044, "ВЫШЕЛ ПЕРВЫЙ CLAUDE", ("CLAUDE",), Cl_),
    (AD_CUT - 0.02, "", ()),
    (31.15, "НАЗВАЛИ ЕГО, ПО ОДНОЙ", ()), (32.744, "ИЗ ВЕРСИЙ, В ЧЕСТЬ", ()),
    (34.064, "КЛОДА ШЕННОНА —", ("ШЕННОНА",), Cl_), (35.394, "УЧЁНОГО, КОТОРЫЙ ЗАЛОЖИЛ", ()),
    (37.234, "ОСНОВЫ ТЕОРИИ ИНФОРМАЦИИ", ("ИНФОРМАЦИИ",), T_),
    (39.56, "ПОТОМ ПОЯВИЛИСЬ МОДЕЛИ", ()), (40.739, "С ПОЭТИЧНЫМИ НАЗВАНИЯМИ:", ()),
    (42.399, "ХАЙКУ, СОНЕТ И ОПУС", ("ХАЙКУ,", "СОНЕТ", "ОПУС"), Cl_),
    (44.699, "А СЕГОДНЯ CLAUDE", ("CLAUDE",), Cl_), (45.699, "ПИШЕТ КОД,", ("КОД,",), G_),
    (46.679, "АНАЛИЗИРУЕТ ДОКУМЕНТЫ", ("ДОКУМЕНТЫ",), G_), (47.979, "И ДАЖЕ ПОМОГАЕТ ДЕЛАТЬ", ()),
    (49.499, "ТАКИЕ РОЛИКИ, КАК ЭТОТ", ("ЭТОТ",), Cl_),
    (51.752, "А ЕЩЁ БОЛЬШЕ ИНТЕРЕСНОГО —", (), T_), (53.112, "В МОЁМ ТЕЛЕГРАМ-КАНАЛЕ", ("ТЕЛЕГРАМ-КАНАЛЕ",), T_),
    (54.442, "«ПАУЧЬЯ НОРКА»", ("«ПАУЧЬЯ", "НОРКА»"), T_), (55.562, "ТАМ ЭКСКЛЮЗИВ,", ("ЭКСКЛЮЗИВ",), T_),
    (56.262, "КОТОРОГО НЕТ В РОЛИКАХ", (), T_), (57.562, "ССЫЛКА В ПРОФИЛЕ —", ("ПРОФИЛЕ",), T_),
    (58.582, "ЗАЛЕТАЙ!", ("ЗАЛЕТАЙ!",), T_),
]

# ---------------------------------------------------------------- камера и сцены
STRIPS = (14.6, 31.0, 39.3)      # смена сцены «полосами»; склейка — на середине
C1, C2, C3 = (x + 0.15 for x in STRIPS)
CAMERA = [
    (0.0, (16.0, 4.4, 1.6, 0.95, 50.0)), (2.5, (6.0, 3.7, 1.2, 0.86, 50.0), "out"),
    (3.45, (5.0, 3.7, 1.2, 0.86, 50.0)),
    (3.7, (0.0, 3.9, 1.3, 0.98, 50.0), "inout"),              # заставка
    (5.0, (-1.0, 3.9, 1.3, 0.98, 50.0)),
    (5.3, (92.0, 3.6, 1.25, 0.86, 50.0), "expo"),             # основатели
    (C1 - 0.01, (99.0, 3.4, 1.25, 0.86, 50.0), "lin"),
    (C1 + 0.01, (182.0, 3.7, 1.45, 0.9, 50.0), "lin"),        # безопасность и конституция
    (26.1, (188.0, 3.6, 1.45, 0.9, 50.0), "lin"),
    (26.4, (270.0, 3.5, 0.8, 1.0, 50.0), "expo"),             # первый Claude — снизу
    (C2 - 0.01, (274.0, 3.4, 0.8, 1.0, 50.0), "lin"),
    (C2 + 0.01, (352.0, 3.6, 1.25, 0.86, 50.0), "lin"),       # Клод Шеннон
    (C3 - 0.01, (357.0, 3.5, 1.25, 0.86, 50.0), "lin"),
    (C3 + 0.01, (420.0, 3.8, 1.6, 0.92, 50.0), "lin"),        # Хайку, Сонет, Опус — чуть сверху
    (44.5, (424.0, 3.8, 1.6, 0.92, 50.0), "lin"),
    (44.8, (500.0, 3.6, 1.25, 0.86, 50.0), "expo"),           # сегодня
    (51.4, (505.0, 3.4, 1.25, 0.86, 49.0), "lin"),
    (51.75, (590.0, 4.3, 1.3, 1.12, 50.0), "expo"), (DUR, (591.0, 4.0, 1.3, 1.15, 50.0)),
]
OFFSET = [(0.0, -0.5), (3.45, -0.5), (3.7, 0.0, "inout"), (5.0, 0.0), (5.3, 0.55, "expo"), (C1 - 0.01, 0.55),
          (C1 + 0.01, -0.55, "lin"), (26.1, -0.55), (26.4, 0.55, "expo"), (C2 - 0.01, 0.55),
          (C2 + 0.01, -0.55, "lin"), (C3 - 0.01, -0.55), (C3 + 0.01, 0.55, "lin"), (44.5, 0.55),
          (44.8, -0.55, "expo"), (51.4, -0.55), (51.75, 0.0, "expo"), (DUR, 0.0)]
# тёплая гамма Claude (B, G, R)
TINT = [(0.0, (0.85, 0.98, 1.15)), (3.45, (0.85, 0.98, 1.15)), (3.7, (1.0, 1.0, 1.0)), (5.0, (1.0, 1.0, 1.0)),
        (5.3, (0.9, 1.0, 1.12)), (C1 - 0.01, (0.9, 1.0, 1.12)), (C1 + 0.01, (0.95, 1.1, 0.95), "lin"),
        (26.1, (0.95, 1.1, 0.95)), (26.4, (0.82, 0.95, 1.2)), (C2 - 0.01, (0.82, 0.95, 1.2)),
        (C2 + 0.01, (1.12, 1.02, 0.9), "lin"), (C3 - 0.01, (1.12, 1.02, 0.9)), (C3 + 0.01, (1.05, 0.92, 1.1), "lin"),
        (44.5, (1.05, 0.92, 1.1)), (44.8, (0.85, 0.98, 1.15)), (51.4, (0.85, 0.98, 1.15)), (51.8, (1.0, 1.0, 1.0)),
        (DUR, (1.0, 1.0, 1.0))]
SEC = {"hook": 6, "found": 95, "safe": 185, "first": 272, "shannon": 354, "models": 422, "today": 502, "tg": 590.5}
SHAKES = ()
FLASHES = ((29.9, 0.1), (51.8, 0.12))
JUMPS = [29.9]
ALARM = None
SPOT = None

G = lambda *k: list(k)  # noqa: E731
O = OUTRO - 48.035      # жесты концовки — как в monet_mg, со сдвигом
ARM_L = G((0, 62), (3.4, 62), (3.7, -50, "back"), (4.7, -45), (5.0, 62, "back"), (5.4, -30, "back"), (9.6, -28),
          (9.7, -45, "back"), (12.6, -42), (12.7, -55, "back"), (14.3, -52), (14.5, 62), (26.4, -30, "back"),
          (29.8, -28), (29.9, -55, "back"), (30.7, -52), (30.85, 62), (C3 + 0.05, 62), (39.7, -30, "back"),
          (42.3, -28), (42.4, -45, "back"), (44.4, -42), (44.6, 62),
          (54.94 + O, 62), (55.09 + O, -50, "back"), (55.69 + O, -45), (56.09 + O, 62, "back"), (DUR, 62))
ARM_R = G((0, 62), (0.3, -28, "back"), (3.3, -26), (3.5, 62), (3.7, -50, "back"), (4.7, -45), (5.0, 62, "back"),
          (C1 + 0.05, 62), (15.0, -30, "back"), (19.1, -28), (19.2, -45, "back"), (26.0, -42), (26.2, 62),
          (29.9, -55, "back"), (30.7, -52), (30.85, 62), (C2 + 0.05, 62), (31.4, -30, "back"), (39.1, -28),
          (39.25, 62), (45.0, -30, "back"), (47.9, -28), (48.0, -50, "back"), (51.2, -48), (51.5, 62),
          (47.79 + O, 62), (48.25 + O, -48, "back"), (50.6 + O, -45), (50.9 + O, -55, "back"), (53.6 + O, -52),
          (53.85 + O, -40, "back"), (54.8 + O, -42), (54.95 + O, -50, "back"), (55.69 + O, -45), (55.84 + O, -58),
          (55.99 + O, -32), (56.14 + O, -58), (56.29 + O, -32), (DUR, 62))
HEAD = G((0, 0), (0.6, 7), (1.9, -6), (3.0, 6), (3.7, 0), (5.4, -6), (7.1, 6), (8.7, -6), (9.7, 7), (11.3, -6),
         (12.7, 8), (14.5, 0), (14.9, -6), (16.3, 6), (17.9, -7), (19.2, 6), (20.4, -6), (21.6, 6), (23.3, -6),
         (24.7, 7), (26.4, -6), (27.0, 6), (29.0, -7), (29.9, 9), (30.8, 0), (31.2, 6), (32.9, -6), (34.1, 7),
         (35.4, -6), (37.2, 6), (39.3, 0), (39.6, -6), (40.8, 6), (42.4, -7), (43.8, 7), (44.7, -6), (45.7, 6),
         (46.7, -6), (48.0, 6), (49.5, -7), (51.3, 0), (52.0, -6), (58.8, 0), (DUR, 0))


# ---------------------------------------------------------------- картинки вставок
def person_tex(name, sub, col):
    im = rounded((380, 440), 34, (26, 20, 18, 255), col + (255,), 6)
    d = ImageDraw.Draw(im)
    d.ellipse([130, 50, 250, 170], fill=col + (255,))
    d.pieslice([70, 180, 310, 420], 180, 360, fill=col + (255,))
    f = mg.font(60, tall=True)
    tw = d.textbbox((0, 0), name, font=f)[2]
    d.rounded_rectangle([190 - tw / 2 - 24, 300, 190 + tw / 2 + 24, 380], 30, fill=(26, 20, 18, 255),
                        outline=col + (255,), width=4)
    d.text((190 - tw / 2, 308), name, font=f, fill="white")
    f2 = mg.font(32)
    tw = d.textbbox((0, 0), sub, font=f2)[2]
    d.text((190 - tw / 2, 392), sub, font=f2, fill=(210, 200, 190))
    return mg.premul(im)


def move_tex(p):
    """«OpenAI → Anthropic»: стрелка вырастает слева направо (p от 0 до 1)."""
    im = rounded((860, 300), 34, (24, 20, 18, 255), CLAY_RGB + (255,), 4)
    oa = logo("openai.png", 150)
    im.alpha_composite(oa, (50, 40))
    d = ImageDraw.Draw(im)
    d.text((40, 210), "OpenAI", font=mg.font(46, tall=True), fill=(220, 220, 230))
    x1 = 240 + int(200 * p)
    if p > 0.02:
        d.line([(240, 115), (x1, 115)], fill=CLAY_RGB + (255,), width=14)
        d.polygon([(x1, 85), (x1 + 44, 115), (x1, 145)], fill=CLAY_RGB + (255,))
    if p > 0.95:
        an = logo("anthropic.png", 46)
        an = an.resize((330, int(an.height * 330 / an.width)), Image.LANCZOS)
        im.alpha_composite(an, (500, 98))
        d.text((500, 210), "2021", font=mg.font(46, tall=True), fill=GOLD_RGB + (255,))
    return mg.premul(im)


def shield_tex():
    im = Image.new("RGBA", (360, 420), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    pts = [(180, 10), (340, 70), (320, 250), (180, 410), (40, 250), (20, 70)]
    d.polygon(pts, fill=(18, 30, 22, 255))
    d.line(pts + [pts[0]], fill=GREEN_RGB + (255,), width=14, joint="curve")
    d.line([(110, 210), (165, 265), (260, 150)], fill=GREEN_RGB + (255,), width=26)
    return mg.premul(im)


def scroll_tex(n):
    """Свиток «Конституция»: n правил уже написано."""
    im = Image.new("RGBA", (640, 640), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([60, 40, 580, 600], 20, fill=CREAM_RGB + (255,))
    for y in (20, 590):
        d.rounded_rectangle([30, y, 610, y + 40], 20, fill=(190, 160, 120, 255))
    f = mg.font(58, tall=True)
    tw = d.textbbox((0, 0), "КОНСТИТУЦИЯ", font=f)[2]
    d.text((320 - tw / 2, 80), "КОНСТИТУЦИЯ", font=f, fill=(90, 50, 30))
    rules = ("1. будь честным", "2. будь полезным", "3. не вреди")
    for i, s in enumerate(rules[:n]):
        d.text((110, 200 + i * 110), s, font=mg.font(48), fill=(60, 40, 30))
        d.line([(110, 262 + i * 110), (530, 262 + i * 110)], fill=(200, 180, 150), width=3)
    return mg.premul(im)


def chat_tex(n):
    """Окно чата: «Привет! Я Claude» печатается по буквам (n букв)."""
    im = rounded((760, 420), 34, (30, 26, 24, 255), CLAY_RGB + (255,), 4)
    star = logo("claude.png", 80)
    im.alpha_composite(star, (34, 26))
    d = ImageDraw.Draw(im)
    d.text((130, 40), "Claude", font=mg.font(52, tall=True), fill=CREAM_RGB + (255,))
    d.text((520, 52), "март 2023", font=mg.font(34), fill=(200, 180, 160))
    msg = "Привет! Я Claude. Чем помочь?"
    d.rounded_rectangle([34, 150, 726, 290], 30, fill=(50, 42, 38, 255))
    d.text((64, 196), msg[:n] + ("|" if n < len(msg) else ""), font=mg.font(42), fill="white")
    d.rounded_rectangle([34, 320, 726, 390], 30, outline=(110, 95, 85), width=3)
    d.text((64, 336), "Напиши сообщение…", font=mg.font(32), fill=(140, 125, 115))
    return mg.premul(im)


def bits_tex(t):
    """Поток нулей и единиц."""
    im = Image.new("RGBA", (520, 420), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    rng = np.random.default_rng(int(t * 8))
    f = mg.font(48, tall=True)
    for r in range(6):
        for c in range(9):
            if rng.random() < 0.7:
                d.text((20 + c * 54, 20 + r * 66), str(int(rng.integers(0, 2))), font=f,
                       fill=(TG_RGB if (r + c) % 3 else CLAY_RGB) + (int(120 + 135 * rng.random()),))
    return mg.premul(im)


def model_label(name, sub, col):
    im = rounded((340, 150), 30, (26, 20, 18, 255), col + (255,), 4)
    d = ImageDraw.Draw(im)
    f = mg.font(64, tall=True)
    tw = d.textbbox((0, 0), name, font=f)[2]
    d.text((170 - tw / 2, 14), name, font=f, fill="white")
    f2 = mg.font(30)
    tw = d.textbbox((0, 0), sub, font=f2)[2]
    d.text((170 - tw / 2, 100), sub, font=f2, fill=(210, 200, 190))
    return mg.premul(im)


def today_icon(kind):
    S = 260
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, S - 1, S - 1], 60, fill=(30, 26, 24, 255), outline=CLAY_RGB + (255,), width=8)
    c = S // 2
    if kind == "code":
        d.text((44, 70), "</>", font=mg.font(110, tall=True), fill=CLAY_RGB + (255,))
    elif kind == "doc":
        d.rounded_rectangle([75, 45, 185, 215], 14, fill=CREAM_RGB + (255,))
        for y in (85, 115, 145, 175):
            d.line([(95, y), (165, y)], fill=(150, 130, 110), width=8)
    else:
        d.rounded_rectangle([60, 70, 200, 200], 16, fill=(60, 52, 48, 255))
        d.polygon([(110, 105), (110, 165), (160, 135)], fill=CLAY_RGB + (255,))
        d.text((c - 70, 208), "РОЛИК", font=mg.font(34, tall=True), fill=CREAM_RGB + (255,))
    return mg.premul(im)


# ---------------------------------------------------------------- сцена
class Scene(base.Scene):
    cfg = sys.modules[__name__]

    def __init__(self, voice):
        super().__init__(voice)
        T = self.T
        T["star"] = logo("claude.png", 500)
        T["an_logo"] = mg.premul(logo("anthropic.png", 60))
        T["y2021"] = chip("2021", GOLD_RGB)
        T["dario"], T["daniela"] = person_tex("ДАРИО", "сейчас — CEO", CLAY_RGB), person_tex("ДАНИЭЛА", "президент", (230, 160, 120))
        T["sib"] = chip("БРАТ И СЕСТРА", CLAY_RGB)
        T["shield"] = shield_tex()
        T["safe"] = chip("БЕЗОПАСНЫЙ ИИ", GREEN_RGB)
        T["cai"] = chip("CONSTITUTIONAL AI · 2022", CLAY_RGB)
        T["shannon"] = card("УЧЁНЫЙ", "КЛОД ШЕННОН", "отец теории информации", TG_RGB)
        T["version"] = chip("ПО ОДНОЙ ИЗ ВЕРСИЙ", GRAY_RGB)
        T["haiku"], T["sonnet"], T["opus"] = (model_label("ХАЙКУ", "быстрая", (120, 180, 220)),
                                              model_label("СОНЕТ", "универсальная", CLAY_RGB),
                                              model_label("ОПУС", "самая мощная", PURPLE_RGB))
        T["code"], T["doc"], T["film"] = today_icon("code"), today_icon("doc"), today_icon("film")
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

    def star_rot(self, a):
        return mg.premul(self.T["star"].rotate(a, resample=Image.BICUBIC))

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
        for s0 in STRIPS:  # «полосы»: кадр режется на ленты, они разъезжаются в стороны и съезжаются уже новым
            p = (t - s0) / 0.3
            if 0 < p < 1:
                q = smooth(p / 0.5) if p < 0.5 else 1 - smooth((p - 0.5) / 0.5)
                out = np.full_like(fr, (22, 18, 16))
                n = 10
                hs = H // n
                for i in range(n):
                    dx = int(W * q * (1 if i % 2 else -1))
                    y0, y1 = i * hs, H if i == n - 1 else (i + 1) * hs
                    if abs(dx) >= W:
                        continue
                    if dx >= 0:
                        out[y0:y1, dx:] = fr[y0:y1, :W - dx]
                    else:
                        out[y0:y1, :W + dx] = fr[y0:y1, -dx:]
                fr = out
        return fr

    def items_for(self, t, cam, it):
        T = self.T
        P = lambda y0, *a, **k: place(y0, *a, **k)  # noqa: E731

        # 1. Крючок (персонаж справа): знак Claude медленно вращается
        y0 = SEC["hook"]
        k = pop(t, 0.15, 3.45, 0.5, 0.25)
        if k > 0:
            self.add_bb(it, cam, P(y0, -0.7, 1.75) + [0, 0.03 * math.sin(t * 2), 0], 0.6 * k,
                        self.cached("st", self.star_rot, int(t * 20) % 36 * 10), anchor=(0.5, 0.5))

        # 2. Заставка
        k = pop(t, 3.7, 5.0, 0.35, 0.3)
        if k > 0:
            self.add_panel(it, cam, place(0, 0.0, 1.95, 0.35), face(0, -6), 1.2 * k, 0.564 * k, 0.11 * k,
                           T["stinger"], pitch=-6, side=(30, 24, 120), edge=(200, 190, 255))

        # 3. Основатели (персонаж слева): 2021 → Дарио и Даниэла → «OpenAI → Anthropic»
        y0 = SEC["found"]
        k = pop(t, 5.35, 9.65, 0.4, 0.25)
        if k > 0:
            self.add_bb(it, cam, P(y0, 0.75, 2.25), 0.12 * k, T["y2021"], anchor=(0.5, 0.5))
            for i, (key, t0, sd) in enumerate((("dario", 7.7, 0.5), ("daniela", 8.6, 1.0))):
                kp = pop(t, t0, 9.65, 0.4, 0.25)
                if kp > 0:
                    self.add_panel(it, cam, P(y0, sd, 1.55), face(y0, 10 + 6 * i), 0.42 * kp, 0.486 * kp, 0.04 * kp,
                                   T[key], side=(60, 40, 30))
            ks = pop(t, 7.1, 9.65, 0.35, 0.25)
            if ks > 0:
                self.add_bb(it, cam, P(y0, 0.75, 1.0), 0.1 * ks, T["sib"], anchor=(0.5, 0.5))
        k = pop(t, 9.7, C1, 0.4, 0.2)
        if k > 0:
            p = round(smooth((t - 10.0) / 1.2) * 20) / 20 if t < 12.6 else 1.0
            pos = P(y0, 0.72, 1.6) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 14), 0.95 * k, 0.331 * k, 0.05 * k, self.cached("mv", move_tex, p),
                           pitch=-6, side=(70, 45, 35), edge=(255, 210, 190))

        # 4. Безопасность и конституция (персонаж справа): щит → свиток с правилами → Constitutional AI
        y0 = SEC["safe"]
        k = pop(t, C1 + 0.05, 19.1, 0.4, 0.25)
        if k > 0:
            self.add_bb(it, cam, P(y0, -0.72, 1.55) + [0, 0.02 * math.sin(t * 2), 0], 0.42 * k, T["shield"],
                        anchor=(0.5, 0.5))
            ks = pop(t, 17.2, 19.1, 0.35, 0.25)
            if ks > 0:
                self.add_bb(it, cam, P(y0, -0.72, 2.15), 0.11 * ks, T["safe"], anchor=(0.5, 0.5))
        k = pop(t, 19.2, 26.2, 0.4, 0.25)
        if k > 0:
            n = 0 if t < 21.6 else 1 + (t > 24.6) + (t > 25.3)
            self.add_bb(it, cam, P(y0, -0.72, 1.5) + [0, 0.02 * math.sin(t * 2), 0], 0.62 * k,
                        self.cached("sc", scroll_tex, n), anchor=(0.5, 0.5))
            self.add_bb(it, cam, P(y0, -0.72, 2.2), 0.09 * k, T["cai"], anchor=(0.5, 0.5))

        # 5. Первый Claude (персонаж слева, снизу): окно чата, сообщение печатается
        y0 = SEC["first"]
        k = pop(t, 26.45, AD_CUT + 0.3, 0.4, 0.2)
        if k > 0:
            n = int(clamp((t - 29.0) / 1.4) * 30)
            pos = P(y0, 0.72, 1.6) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 14), 0.9 * k, 0.497 * k, 0.06 * k, self.cached("ch", chat_tex, n),
                           pitch=6, side=(70, 45, 35), edge=(255, 210, 190))

        # 6. Клод Шеннон (персонаж справа): «по одной из версий» → карточка → поток нулей и единиц
        y0 = SEC["shannon"]
        k = pop(t, C2 + 0.05, 34.0, 0.35, 0.25)
        if k > 0:
            self.add_bb(it, cam, P(y0, -0.72, 2.0), 0.11 * k, T["version"], anchor=(0.5, 0.5))
        k = pop(t, 34.05, C3, 0.4, 0.2)
        if k > 0:
            pos = P(y0, -0.72, 1.85) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, -14), 0.9 * k, 0.427 * k, 0.06 * k, T["shannon"], pitch=-6,
                           side=(30, 60, 80), edge=(190, 230, 255))
            kb = pop(t, 37.2, C3, 0.35, 0.2)
            if kb > 0:
                self.add_bb(it, cam, P(y0, -0.72, 1.2), 0.4 * kb, self.cached("bits", bits_tex, round(t * 8) / 8),
                            anchor=(0.5, 0.5))

        # 7. Модели (персонаж слева, чуть сверху): Хайку, Сонет, Опус — маленькая, средняя, большая
        y0 = SEC["models"]
        for i, (key, sz, t0) in enumerate((("haiku", 0.35, 42.4), ("sonnet", 0.6, 43.0), ("opus", 0.9, 43.8))):
            kb = pop(t, t0, 44.55, 0.45, 0.2)
            if kb <= 0:
                continue
            q = P(y0, 0.45 + 0.42 * i, 0.0, -0.4)
            self.add_box(it, cam, (q[0], q[2]), (0.28, 0.28), sz * kb, (90, 120, 215) if i == 2 else
                         (87, 119, 217) if i == 1 else (220, 180, 120), radius=0.08, edge=(240, 230, 214))
            self.add_bb(it, cam, q + [0, sz * kb + 0.2, 0], 0.16 * kb, T[key], anchor=(0.5, 0.5))
        k = pop(t, 39.6, 42.35, 0.4, 0.25)
        if k > 0:
            self.add_bb(it, cam, P(y0, 0.75, 1.75) + [0, 0.03 * math.sin(t * 2), 0], 0.55 * k,
                        self.cached("st2", self.star_rot, int(t * 20) % 36 * 10), anchor=(0.5, 0.5))

        # 8. Сегодня (персонаж справа): код, документы, этот ролик
        y0 = SEC["today"]
        for i, (key, t0) in enumerate((("code", 45.7), ("doc", 46.7), ("film", 48.6))):
            kk = pop(t, t0, 51.45, 0.35, 0.25)
            if kk > 0:
                self.add_bb(it, cam, P(y0, -1.0 + 0.38 * i, 1.95), 0.26 * kk, T[key],
                            anchor=(0.5, 0.5))
        k = pop(t, 44.8, 51.45, 0.4, 0.25)
        if k > 0:
            self.add_bb(it, cam, P(y0, -0.62, 1.55), 0.06 * k, T["an_logo"], anchor=(0.5, 0.5))

        # 9. Концовка с Telegram — общий шаблон
        self.outro_items(t, cam, it, OUTRO, SEC["tg"])


SFX = [(t, "whoosh") for t in (0.0, 3.65, 5.2) + STRIPS + (26.3, 44.7, 51.65)]


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
        for t in (0.6, 2.5, 4.3, 6.5, 9.0, 12.8, 14.75, 17.5, 20.0, 25.6, 27.5, 30.6, 32.5, 36.0, 38.5, 40.5,
                  43.5, 46.0, 50.0, 55.0):
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
    base.mix_audio(tmp, voice, out, music=os.path.join(ROOT, "work", "claude", "music.wav"), sfx_list=SFX, dur=DUR,
                   sfx_vol=0.15)  # «вжух» лёгкий и тихий
    os.remove(tmp)


if __name__ == "__main__":
    main()
