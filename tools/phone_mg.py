"""Ролик №2: «Телефон подслушивает? Почему реклама угадывает мысли».

python tools/phone_mg.py work/phone/voice_clean.wav output/phone.mp4 [--preview]
Общий движок, персонаж и концовка с Telegram — из memory_mg.py (шаблон); здесь только сцены этого ролика.
Звуки: только тихий «вжух» на смене ракурса и появлении вставок (правило пользователя).
"""
import math
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memory_mg as base  # noqa: E402
import mg  # noqa: E402
from memory_mg import (BLUE, BLUE_RGB, GRAY_RGB, RED, RED_RGB, TG_RGB, basis, card, chip, face, place,  # noqa: E402,F401
                       pop, rounded, stamp)
from mg import H, W, clamp, out_back, smooth  # noqa: E402

FPS = 30
ROOT = mg.ROOT
OUTRO = float(open(os.path.join(ROOT, "work", "phone", "outro_start.txt")).read())  # 42.99
DUR = 51.7

# ---------------------------------------------------------------- субтитры
T_ = TG_RGB
SUBS = [
    (0.23, "ПОЧЕМУ ТЫ ТОЛЬКО ПОДУМАЛ", ()), (1.47, "О НОВЫХ КРОССОВКАХ,", ("КРОССОВКАХ",)),
    (2.55, "А ТЕЛЕФОН УЖЕ ПОКАЗЫВАЕТ", ()), (3.69, "ИХ В РЕКЛАМЕ?", ("РЕКЛАМЕ",)),
    (4.53, "ОН ЧТО, ПОДСЛУШИВАЕТ?", ("ПОДСЛУШИВАЕТ",)), (6.0, "", ()),
    (7.68, "УЧЁНЫЕ ПРОВЕРИЛИ", ()), (8.62, "БОЛЬШЕ 17 ТЫСЯЧ", ("17", "ТЫСЯЧ")), (9.66, "ПРИЛОЖЕНИЙ", ()),
    (10.52, "ТАЙНОЙ ЗАПИСИ РАЗГОВОРОВ", ()), (11.68, "ОНИ ТАК И НЕ НАШЛИ", ("НЕ", "НАШЛИ"), BLUE_RGB),
    (12.86, "ВСЁ ПРОЩЕ И СТРАШНЕЕ", ("СТРАШНЕЕ",)), (14.60, "ТЕЛЕФОН И ТАК ЗНАЕТ", ()),
    (15.48, "О ТЕБЕ ПОЧТИ ВСЁ", ("ВСЁ",)), (16.97, "ЧТО ТЫ ИЩЕШЬ,", ("ИЩЕШЬ",)), (17.85, "ГДЕ БЫВАЕШЬ,", ("БЫВАЕШЬ",)),
    (18.71, "НА ЧЁМ ЗАДЕРЖИВАЕШЬ ВЗГЛЯД", ("ВЗГЛЯД",)), (19.85, "И ЧЕМ ИНТЕРЕСУЮТСЯ ДРУЗЬЯ,", ("ДРУЗЬЯ",)),
    (21.37, "КОТОРЫЕ РЯДОМ С ТОБОЙ", ()), (22.91, "ИЗ ЭТИХ СЛЕДОВ", ()), (23.77, "АЛГОРИТМ СОБИРАЕТ", ()),
    (24.71, "ТВОЙ ПОРТРЕТ", ("ПОРТРЕТ",)), (25.39, "И УГАДЫВАЕТ ЖЕЛАНИЕ", ("ЖЕЛАНИЕ",)),
    (26.41, "РАНЬШЕ, ЧЕМ ТЫ САМ", ()), (27.35, "ЕГО ОСОЗНАЛ", ()),
    (28.59, "А ЕЩЁ РАБОТАЕТ", ()), (29.47, "ИЛЛЮЗИЯ ЧАСТОТЫ", ("ИЛЛЮЗИЯ", "ЧАСТОТЫ")),
    (30.85, "ТЫ ЗАМЕЧАЕШЬ РЕКЛАМУ", ()), (31.99, "ТОГО, О ЧЁМ ДУМАЛ,", ()),
    (32.93, "И НЕ ЗАМЕЧАЕШЬ", ()), (33.83, "СОТНИ ОСТАЛЬНЫХ", ("СОТНИ",)),
    (35.11, "ХОЧЕШЬ МЕНЬШЕ СЛЕЖКИ?", ("СЛЕЖКИ",), BLUE_RGB), (36.65, "ВЫКЛЮЧИ В НАСТРОЙКАХ", ()),
    (37.67, "ПЕРСОНАЛЬНУЮ РЕКЛАМУ", ("РЕКЛАМУ",), BLUE_RGB), (38.87, "И ЗАБЕРИ У ЛИШНИХ ПРИЛОЖЕНИЙ", ()),
    (40.41, "ДОСТУП К МИКРОФОНУ", ("МИКРОФОНУ",), BLUE_RGB), (41.49, "И ГЕОЛОКАЦИИ", ("ГЕОЛОКАЦИИ",), BLUE_RGB),
    (42.95, "А ЕЩЁ БОЛЬШЕ ИНТЕРЕСНОГО —", (), T_), (44.31, "В МОЁМ ТЕЛЕГРАМ-КАНАЛЕ", ("ТЕЛЕГРАМ-КАНАЛЕ",), T_),
    (45.65, "«ПАУЧЬЯ НОРКА»", ("«ПАУЧЬЯ", "НОРКА»"), T_), (46.77, "ТАМ ЭКСКЛЮЗИВ,", ("ЭКСКЛЮЗИВ",), T_),
    (47.47, "КОТОРОГО НЕТ В РОЛИКАХ", (), T_), (48.77, "ССЫЛКА В ПРОФИЛЕ —", ("ПРОФИЛЕ",), T_),
    (49.79, "ЗАЛЕТАЙ!", ("ЗАЛЕТАЙ!",), T_),
]

# ---------------------------------------------------------------- камера и сцены
CAMERA = [
    (0.0, (-28.0, 5.4, 1.6, 0.85, 50.0)), (0.7, (-12.0, 3.6, 1.15, 0.82, 50.0), "out"),
    (5.7, (-6.0, 3.5, 1.1, 0.82, 50.0)),
    (6.2, (110.0, 3.9, 1.35, 0.98, 50.0), "expo"), (7.2, (116.0, 3.8, 1.3, 0.98, 50.0)),
    (7.6, (100.0, 3.6, 1.2, 0.82, 50.0), "inout"), (12.8, (108.0, 3.7, 1.25, 0.84, 50.0)),
    (13.6, (104.0, 3.2, 1.1, 0.86, 50.0), "out"),     # «и страшнее» — наезд
    (14.15, (110.0, 3.7, 1.25, 0.84, 50.0)),
    (14.55, (200.0, 3.6, 1.2, 0.84, 50.0), "expo"), (27.9, (216.0, 3.8, 1.3, 0.86, 50.0)),
    (28.3, (270.0, 3.6, 1.2, 0.86, 50.0), "expo"), (34.6, (282.0, 3.7, 1.25, 0.86, 50.0)),
    (35.0, (350.0, 3.6, 1.2, 0.84, 50.0), "expo"), (42.55, (360.0, 3.7, 1.25, 0.84, 50.0)),
    (42.95, (444.0, 4.3, 1.3, 1.12, 50.0), "expo"), (DUR, (445.0, 4.0, 1.3, 1.15, 50.0)),
]
OFFSET = [(0.0, 0.5), (5.7, 0.5), (6.2, 0.0, "expo"), (7.2, 0.0), (7.6, 0.55, "inout"),
          (42.55, 0.55), (42.95, 0.0, "expo"), (DUR, 0.0)]
SEC = {"hook": -9, "study": 106, "know": 207, "freq": 276, "tips": 356, "tg": 444.5}
SHAKES = (6.2,)
FLASHES = ((5.95, 0.18), (14.35, 0.1), (28.1, 0.1), (34.8, 0.1), (42.75, 0.12))
JUMPS = [6.05, 49.9]
ALARM = (13.7, 14.45)  # короткая красная пульсация на «страшнее»
SPOT = None

G = lambda *k: list(k)  # noqa: E731
# L — рука слева на экране, R — справа (к вставкам)
ARM_L = G((0, 62), (4.5, 62), (4.7, 35, "back"), (5.6, 38), (5.9, -50, "back"), (6.8, -45), (7.3, 62, "back"),
          (12.8, 62), (13.05, 30, "back"), (14.2, 32), (14.5, 62), (28.4, 62), (28.6, 40, "back"), (29.6, 42),
          (29.9, 62), (34.9, 62), (35.1, 30, "back"), (36.4, 32), (36.6, 62), (49.7, 62), (49.85, -50, "back"),
          (50.6, -45), (51.0, 62, "back"), (DUR, 62))
ARM_R = G((0, 62), (1.0, 62), (1.2, -40, "back"), (2.4, -38), (2.6, -8, "back"), (4.4, -10), (4.6, 35, "back"),
          (5.6, 38), (5.9, -50, "back"), (6.8, -45), (7.3, 62, "back"), (7.7, -30, "back"), (10.4, -28),
          (10.6, -12, "back"), (12.7, -10), (13.05, 30, "back"), (14.2, 32), (14.5, 62), (16.9, 62),
          (17.1, -35, "back"), (22.8, -32), (23.0, -10, "back"), (27.8, -8), (28.4, 62), (28.6, 40, "back"),
          (29.6, 42), (29.9, -25, "back"), (34.6, -22), (34.9, 62), (35.1, 30, "back"), (36.4, 32),
          (36.6, -15, "back"), (42.4, -12), (42.8, 62), (43.2, -48, "back"), (45.6, -45), (45.9, -55, "back"),
          (48.6, -52), (48.85, -40, "back"), (49.7, -42), (49.85, -50, "back"), (50.6, -45), (50.75, -58),
          (50.9, -32), (51.05, -58), (51.2, -32), (51.35, -58), (DUR, 62))
HEAD = G((0, 0), (1.2, -7), (2.6, -5), (4.6, 8), (5.6, 6), (6.0, 0), (7.7, -6), (10.6, -5), (12.1, 5),
         (13.1, 9), (14.4, 0), (17.1, -7), (20.0, -6), (23.0, -8), (27.8, -4), (28.6, 7), (29.9, -6), (33.0, 5),
         (35.1, 7), (36.6, -6), (41.6, -5), (43.2, -6), (49.9, 0), (DUR, 0))


# ---------------------------------------------------------------- картинки вставок
def phone_tex(ad):
    """Объёмный телефон: экран с лентой, а при ad=True — с рекламой кроссовок."""
    Wp, Hp = 420, 860
    im = rounded((Wp, Hp), 64, (14, 14, 18, 255), (80, 80, 96, 255), 6)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([18, 18, Wp - 19, Hp - 19], 50, fill=(26, 27, 36, 255))
    d.rounded_rectangle([Wp // 2 - 55, 34, Wp // 2 + 55, 62], 14, fill=(8, 8, 10, 255))  # «чёлка»
    if not ad:
        for i in range(4):
            y = 110 + i * 175
            d.rounded_rectangle([44, y, Wp - 44, y + 150], 22, fill=(40, 42, 54, 255))
            d.rounded_rectangle([64, y + 20, 200, y + 40], 10, fill=(70, 72, 88, 255))
            d.rounded_rectangle([64, y + 56, Wp - 70, y + 72], 8, fill=(56, 58, 72, 255))
    else:
        d.rounded_rectangle([40, 100, Wp - 40, 640], 30, fill=(36, 20, 26, 255), outline=RED_RGB + (255,), width=4)
        d.rounded_rectangle([64, 124, 220, 170], 22, fill=RED_RGB + (255,))
        d.text((82, 128), "РЕКЛАМА", font=mg.font(30), fill="white")
        # кроссовок из простых фигур
        d.rounded_rectangle([86, 330, 330, 400], 30, fill=(240, 240, 245, 255))
        d.polygon([(110, 340), (180, 240), (260, 250), (300, 340)], fill=(240, 240, 245, 255))
        d.rounded_rectangle([86, 392, 336, 420], 12, fill=RED_RGB + (255,))
        for i in range(3):
            d.line([(190 + i * 22, 270 + i * 4), (210 + i * 22, 300 + i * 4)], fill=(30, 30, 40), width=6)
        d.text((64, 450), "КРОССОВКИ", font=mg.font(64, tall=True), fill="white")
        d.rounded_rectangle([64, 540, 190, 600], 26, fill=(255, 210, 60, 255))
        d.text((82, 548), "−30%", font=mg.font(40), fill=(20, 20, 20))
        d.rounded_rectangle([40, 670, Wp - 40, 760], 26, fill=(40, 42, 54, 255))
    return mg.premul(im)


def thought_tex():
    im = Image.new("RGBA", (760, 300), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([60, 0, 759, 210], 60, fill=(24, 24, 32, 255), outline=(90, 90, 110, 255), width=4)
    d.ellipse([20, 210, 80, 260], fill=(24, 24, 32, 255), outline=(90, 90, 110, 255), width=4)
    d.ellipse([0, 262, 30, 292], fill=(24, 24, 32, 255), outline=(90, 90, 110, 255), width=4)
    d.text((110, 30), "хочу новые", font=mg.font(56), fill=(170, 170, 190))
    d.text((110, 100), "КРОССОВКИ…", font=mg.font(78, tall=True), fill="white")
    return mg.premul(im)


def mic_tex():
    """Значок микрофона с вопросом — «подслушивает?»."""
    S = 300
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([0, 0, S - 1, S - 1], fill=RED_RGB + (255,))
    d.rounded_rectangle([118, 60, 182, 170], 32, fill="white")
    d.arc([92, 110, 208, 206], 0, 180, fill="white", width=12)
    d.line([(150, 206), (150, 238)], fill="white", width=12)
    d.line([(118, 240), (182, 240)], fill="white", width=12)
    d.text((205, 30), "?", font=mg.font(110), fill="white")
    return mg.premul(im)


def ring_tex():
    im = Image.new("RGBA", (400, 400), (0, 0, 0, 0))
    ImageDraw.Draw(im).ellipse([10, 10, 389, 389], outline=RED_RGB + (255,), width=10)
    return mg.premul(im)


def profile_tex():
    im = rounded((780, 560), 34, (18, 18, 26, 255), (72, 72, 92, 255), 3)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([36, 34, 336, 96], 30, fill=RED_RGB + (255,))
    d.text((58, 42), "ТВОЙ ПОРТРЕТ", font=mg.font(36), fill="white")
    d.ellipse([40, 130, 200, 290], fill=(44, 44, 58, 255))
    d.ellipse([90, 150, 150, 210], fill=(110, 110, 130, 255))
    d.pieslice([62, 215, 178, 335], 180, 360, fill=(110, 110, 130, 255))
    rows = [("Кроссовки", 0.92), ("Спорт", 0.74), ("Кофе", 0.61), ("Путешествия", 0.38)]
    for i, (k, v) in enumerate(rows):
        y = 130 + i * 62
        d.text((240, y), k, font=mg.font(34), fill=(200, 200, 215))
        d.rounded_rectangle([240, y + 42, 740, y + 52], 5, fill=(44, 44, 58, 255))
        d.rounded_rectangle([240, y + 42, 240 + int(500 * v), y + 52], 5, fill=RED_RGB + (255,))
        d.text((660, y), f"{int(v * 100)}%", font=mg.font(34), fill="white")
    d.text((40, 400), "собран из твоих следов в сети", font=mg.font(34), fill=(140, 140, 160))
    return mg.premul(im)


def tile_tex(text, hot):
    im = rounded((420, 200), 30, (36, 18, 24, 255) if hot else (28, 28, 34, 255),
                 RED_RGB + (255,) if hot else (64, 64, 76, 255), 5)
    d = ImageDraw.Draw(im)
    d.text((26, 20), "РЕКЛАМА", font=mg.font(26), fill=(255, 140, 140) if hot else (110, 110, 125))
    f = mg.font(58, tall=True)
    d.text((26, 74), text, font=f, fill="white" if hot else (140, 140, 155))
    return mg.premul(im)


def toggles_tex(states):
    """Карточка настроек: три переключателя; states — насколько каждый выключен (0 — включён, 1 — выключен)."""
    im = rounded((780, 470), 34, (18, 18, 26, 255), (72, 72, 92, 255), 3)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([36, 34, 300, 96], 30, fill=BLUE_RGB + (255,))
    d.text((58, 42), "НАСТРОЙКИ", font=mg.font(36), fill="white")
    rows = ["Персональная реклама", "Доступ к микрофону", "Геолокация"]
    for i, (k, s) in enumerate(zip(rows, states)):
        y = 130 + i * 110
        d.text((40, y + 14), k, font=mg.font(40), fill="white")
        on = tuple(int(a + (b - a) * s) for a, b in zip((52, 199, 89), (70, 70, 84)))  # зелёный → серый
        d.rounded_rectangle([600, y, 740, y + 70], 35, fill=on + (255,))
        x = int(705 - 70 * s)
        d.ellipse([x - 30, y + 5, x + 30, y + 65], fill="white")
    return mg.premul(im)


# ---------------------------------------------------------------- сцена
class Scene(base.Scene):
    cfg = sys.modules[__name__]

    def __init__(self, voice):
        super().__init__(voice)
        T = self.T
        T["phone"], T["phone_ad"] = phone_tex(False), phone_tex(True)
        T["thought"], T["mic"], T["ring"] = thought_tex(), mic_tex(), ring_tex()
        T["study"] = card("ИССЛЕДОВАНИЕ", "17 000+ ПРИЛОЖЕНИЙ", "Северо-Восточный университет, 2018", BLUE_RGB)
        T["secret"] = card("ИСКАЛИ", "ТАЙНУЮ ЗАПИСЬ", "разговоров через микрофон", BLUE_RGB)
        T["notfound"] = stamp("НЕ НАЙДЕНО", BLUE_RGB)
        T["scary"] = card("НА САМОМ ДЕЛЕ", "ВСЁ ПРОЩЕ", "и от этого страшнее")
        for key, txt in (("c_search", "ЧТО ИЩЕШЬ"), ("c_geo", "ГДЕ БЫВАЕШЬ"), ("c_eyes", "НА ЧЁМ ЗАЛИПАЕШЬ"),
                         ("c_friends", "ДРУЗЬЯ РЯДОМ")):
            T[key] = chip(txt, RED_RGB)
        T["profile"] = profile_tex()
        T["wants"] = stamp("ХОЧЕТ КРОССОВКИ")
        T["freq"] = card("ПСИХОЛОГИЯ", "ИЛЛЮЗИЯ ЧАСТОТЫ", "эффект Баадера-Майнхоф", h=300)
        ads = ["ПИЦЦА", "КРЕДИТ", "ОТПУСК", "ДИВАН", "КРОССОВКИ", "ФИТНЕС", "КОФЕ", "ИГРЫ", "ТАКСИ"]
        T["tiles"] = [(a, tile_tex(a, a == "КРОССОВКИ")) for a in ads]
        self.tog_cache = {}

    def toggles(self, st):
        key = tuple(round(x * 8) / 8 for x in st)
        if key not in self.tog_cache:
            self.tog_cache[key] = toggles_tex(key)
        return self.tog_cache[key]

    def items_for(self, t, cam, it):
        T = self.T
        P = lambda y0, *a, **k: place(y0, *a, **k)  # noqa: E731

        # 1. Крючок: мысль о кроссовках → реклама в телефоне → «подслушивает?»
        y0 = SEC["hook"]
        k = pop(t, 0.9, 5.95)
        if k > 0:
            pos = P(y0, 0.9, 0.95) + [0, 0.02 * math.sin(t * 2), 0]
            tex = T["phone_ad"] if t > 3.15 else T["phone"]
            sw = 1 - 0.12 * math.sin(math.pi * clamp((t - 3.05) / 0.2))
            self.add_panel(it, cam, pos, face(y0, 20 - 25 * (1 - k)), 0.42 * k * sw, 0.86 * k, 0.06 * k, tex,
                           side=(40, 40, 50), edge=(200, 200, 230), radius=0.42 * k * 0.15)
            if t > 4.6:  # волны «прослушки» вокруг телефона
                for i in range(3):
                    q = ((t - 4.6) * 0.9 + i / 3) % 1
                    self.add_bb(it, cam, pos + [0, 0, 0.05], 0.4 + 0.9 * q, T["ring"] * (1 - q) * 0.7 * k, anchor=(0.5, 0.5))
        k = pop(t, 1.2, 3.1, 0.4, 0.25)
        if k > 0:
            self.add_panel(it, cam, P(y0, 0.75, 1.75), face(y0, 10), 0.85 * k, 0.335 * k, 0.03 * k, T["thought"],
                           side=(30, 30, 40), radius=0.06)
        k = pop(t, 4.55, 5.95, 0.35, 0.25)
        if k > 0:
            self.add_bb(it, cam, P(y0, 0.95, 1.72, 0.1), 0.24 * k, T["mic"], anchor=(0.5, 0.5))

        # 2. Заставка «Паучок объясняет»
        k = pop(t, 5.95, 7.45, 0.4, 0.3)
        if k > 0:
            self.add_panel(it, cam, place(116, 0.0, 1.95, 0.35), face(116, -8), 1.35 * k, 0.635 * k, 0.12 * k,
                           T["stinger"], pitch=-6, side=(30, 24, 120), edge=(200, 190, 255))

        # 3. Исследование: 17 000 приложений, тайной записи не нашли; «всё проще и страшнее»
        y0 = SEC["study"]
        for key, a, b in (("study", 7.7, 10.6), ("secret", 10.5, 12.85), ("scary", 12.85, 14.45)):
            k = pop(t, a, b, 0.4, 0.25)
            if k > 0:
                pos = P(y0, 0.68, 1.5) + [0, 0.03 * math.sin(t * 2), 0]
                self.add_panel(it, cam, pos, face(y0, 14), 0.9 * k, 0.48 * k, 0.06 * k, T[key], pitch=-6,
                               side=(90, 50, 40) if key != "scary" else (60, 50, 90), edge=(255, 210, 190))
                if key == "secret" and t > 12.1:
                    ks = clamp((t - 12.1) / 0.18)
                    self.add_bb(it, cam, pos + place(y0, 0.05, -0.05, 0.12) - place(y0, 0, 0, 0), 0.15 * (1 + 0.5 * (1 - ks)) * k,
                                T["notfound"], anchor=(0.5, 0.5))

        # 4. Что знает телефон: следы → портрет → «хочет кроссовки»
        y0 = SEC["know"]
        kp = pop(t, 14.7, 22.95)
        if kp > 0:
            ppos = P(y0, 0.95, 1.0) + [0, 0.02 * math.sin(t * 2), 0]
            self.add_panel(it, cam, ppos, face(y0, 10), 0.42 * kp, 0.86 * kp, 0.06 * kp, T["phone"],
                           side=(40, 40, 50), edge=(200, 200, 230), radius=0.42 * kp * 0.15)
        chips = (("c_search", 17.41, 1.85), ("c_geo", 17.95, 1.55), ("c_eyes", 18.93, 1.25), ("c_friends", 20.85, 0.95))
        for i, (key, t0, hy) in enumerate(chips):
            if t0 < t < 24.0:
                fly = smooth((t - 22.95) / 0.8)  # летят в портрет
                k = out_back(clamp((t - t0) / 0.35)) * (1 - fly)
                pos = P(y0, 0.72, hy) * (1 - fly) + P(y0, 0.78, 1.4) * fly
                if k > 0.01:
                    self.add_bb(it, cam, pos, 0.12 * k, T[key], anchor=(0.5, 0.5))
        k = pop(t, 23.4, 28.05)
        if k > 0:
            pos = P(y0, 0.78, 1.4) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 12), 0.9 * k, 0.646 * k, 0.06 * k, T["profile"], pitch=-5,
                           side=(60, 50, 90), edge=(220, 200, 255))
            if t > 25.6:
                ks = clamp((t - 25.6) / 0.18)
                self.add_bb(it, cam, pos + place(y0, 0.0, -0.3, 0.12) - place(y0, 0, 0, 0), 0.14 * (1 + 0.6 * (1 - ks)) * k,
                            T["wants"], anchor=(0.5, 0.5))

        # 5. Иллюзия частоты: сетка реклам, замечаешь одну
        y0 = SEC["freq"]
        k = pop(t, 28.6, 34.85)
        if k > 0:
            self.add_panel(it, cam, P(y0, 0.75, 1.95), face(y0, 10), 0.8 * k, 0.31 * k, 0.05 * k, T["freq"],
                           pitch=-6, side=(60, 50, 90), edge=(220, 200, 255))
        if 30.7 < t < 34.85:
            dim = 1 - 0.65 * smooth((t - 33.0) / 0.6)  # «не замечаешь сотни остальных» — гаснут
            for i, (name, tex) in enumerate(T["tiles"]):
                kk = out_back(clamp((t - 30.8 - 0.05 * i) / 0.35)) * (1 - smooth((t - 34.5) / 0.3))
                if kk <= 0:
                    continue
                r, c = divmod(i, 3)
                pos = P(y0, 0.42 + 0.34 * c, 1.32 - 0.21 * r)
                hot = name == "КРОССОВКИ"
                s = 1.0 + (0.18 * smooth((t - 31.5) / 0.3) if hot else 0)
                self.add_bb(it, cam, pos, 0.17 * kk * s, tex * (1.0 if hot else dim), anchor=(0.5, 0.5))

        # 6. Как защититься: переключатели выключаются по словам
        y0 = SEC["tips"]
        k = pop(t, 35.25, 42.6)
        if k > 0:
            st = (smooth((t - 37.7) / 0.35), smooth((t - 40.45) / 0.35), smooth((t - 41.55) / 0.35))
            pos = P(y0, 0.78, 1.35) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 12), 0.9 * k, 0.543 * k, 0.06 * k, self.toggles(st), pitch=-5,
                           side=(90, 50, 40), edge=(255, 210, 190))

        # 7. Концовка с Telegram — общий шаблон
        self.outro_items(t, cam, it, OUTRO, SEC["tg"])


SFX = [(t, "whoosh") for t in (0.0, 0.9, 5.95, 7.7, 14.35, 17.4, 23.4, 28.1, 30.8, 34.8, 35.25, 42.75)]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    voice, out = args[0], args[1]
    only = None
    if "--preview" in sys.argv:
        only = [int(x * FPS) for x in (1.8, 3.6, 5.2, 6.6, 9.0, 12.4, 13.9, 18.0, 21.0, 24.0, 26.2, 29.4, 32.0,
                                       33.8, 36.0, 39.0, 41.9, 44.0, 47.2, 49.0, 50.2, 51.4)]
    sc = Scene(voice)
    tmp = out + ".video.mp4"
    wr = None if only else mg.Writer(tmp, FPS)
    n = int(DUR * FPS)
    for i in range(n) if not only else only:
        t = i / FPS
        speed = abs(sc.cam(t + 0.5 / FPS).yaw - sc.cam(t - 0.5 / FPS).yaw) * FPS
        sub = 1 if speed < math.radians(60) else 5
        if sub == 1:
            fr = sc.draw(t)
        else:
            acc = np.zeros((H, W, 3), np.float32)
            for k in range(sub):
                acc += sc.draw(t + (k / sub - 0.5) / FPS)
            fr = (acc / sub).astype(np.uint8)
        fr = sc.finish(fr, t)
        if only:
            cv2.imwrite(f"{out}_{i:04d}.jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        else:
            wr.write(fr)
            if i % 150 == 0:
                print(f"{i}/{n}", flush=True)
    if only:
        return
    wr.close()
    base.mix_audio(tmp, voice, out, music=os.path.join(ROOT, "work", "phone", "music.wav"), sfx_list=SFX, dur=DUR,
                   sfx_vol=0.15)  # «вжух» лёгкий и тихий
    os.remove(tmp)


if __name__ == "__main__":
    main()
