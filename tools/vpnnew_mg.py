"""Ролик №24: «Почему сейчас появляется так много новых VPN?» (+ рекламный баннер в середине).

python tools/vpnnew_mg.py work/vpnnew/voice_clean.wav output/vpnnew.mp4 [--preview] [--jobs 4]
затем: python tools/ad_banner.py insert output/vpnnew.mp4 29.65 output/vpnnew_ad.mp4 vpnnew_mg
Общий движок, персонаж и концовка с Telegram — из memory_mg.py; параллельный рендер — как в money_mg.py.
Новое в монтаже: щит-замок над планетой, камера медленно облетает; график спроса взлетает; серверная стойка
с мигающими лампочками и ценой аренды; окно с открытым кодом; настоящий логотип Telegram и чат с ботом;
сетка иконок-приложений, которые появляются и гаснут; смена сцены «плитками» — кадр рассыпается на
квадраты волной и собирается уже новым; карточка опыта CSIRO; три телефона, в одном — вирус; открытый замок;
штрихкод «товар — это ты»; чек-лист.
Логотип Telegram — assets/logos. Звуки: только тихий «вжух» на смене ракурса.
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
from money_mg import text_rub  # noqa: E402
from mg import H, W, clamp, out_back, smooth, track, wobble  # noqa: E402

FPS = 30
ROOT = mg.ROOT
OUTRO = float(open(os.path.join(ROOT, "work", "vpnnew", "outro_start.txt")).read())  # 51.45
DUR = 60.1
AD_CUT = 29.65                   # здесь вклеивается баннер (середина ролика с баннером ±5 с)
GREEN_RGB, GOLD_RGB, CYAN_RGB, PURPLE_RGB = (46, 196, 110), (245, 190, 50), (60, 220, 240), (150, 90, 230)

# ---------------------------------------------------------------- субтитры
T_, G_, Go_, C_, Pu_ = TG_RGB, GREEN_RGB, GOLD_RGB, CYAN_RGB, PURPLE_RGB
SUBS = [
    (0.155, "ПОЧЕМУ СЕЙЧАС ПОЯВЛЯЕТСЯ", ()), (1.495, "ТАК МНОГО НОВЫХ VPN?", ("VPN?",), C_), (3.0, "", ()),
    (4.655, "ВСЁ ПРОСТО:", ()), (5.595, "СПРОС ОГРОМНЫЙ,", ("ОГРОМНЫЙ,",), G_),
    (6.775, "А ЗАПУСТИТЬ СВОЙ СЕРВИС", ()), (7.895, "СТАЛО ОЧЕНЬ ЛЕГКО", ("ЛЕГКО",), C_),
    (9.455, "АРЕНДОВАТЬ СЕРВЕР СТОИТ", ()), (10.655, "НЕСКОЛЬКО СОТЕН РУБЛЕЙ", ("СОТЕН",), Go_),
    (11.675, "В МЕСЯЦ,", ()), (12.675, "А НУЖНЫЕ ПРОГРАММЫ", ()), (13.615, "ДАВНО ЛЕЖАТ", ()),
    (14.275, "В ОТКРЫТОМ ДОСТУПЕ", ("ОТКРЫТОМ",), G_),
    (15.6, "ПРОДАВАТЬ МОЖНО ПРЯМО", ()), (16.683, "ЧЕРЕЗ БОТА В ТЕЛЕГРАМЕ —", ("ТЕЛЕГРАМЕ",), T_),
    (18.283, "БЕЗ САЙТА И БЕЗ", ()), (19.503, "ПРИЛОЖЕНИЯ В МАГАЗИНЕ", ()),
    (21.163, "ПОЭТОМУ НОВЫЕ СЕРВИСЫ", ()), (22.543, "ПОЯВЛЯЮТСЯ КАЖДУЮ НЕДЕЛЮ,", ("КАЖДУЮ", "НЕДЕЛЮ,"), C_),
    (24.463, "А СТАРЫЕ ЧАСТО ПЕРЕСТАЮТ", ()), (26.003, "РАБОТАТЬ И ИСЧЕЗАЮТ", ("ИСЧЕЗАЮТ",)),
    (27.79, "НО БУДЬ ОСТОРОЖЕН", ("ОСТОРОЖЕН",)), (28.713, "С БЕСПЛАТНЫМИ", ("БЕСПЛАТНЫМИ",)),
    (AD_CUT - 0.02, "", ()),
    (29.79, "УЧЁНЫЕ ИЗ АВСТРАЛИИ", ("АВСТРАЛИИ",), C_), (30.993, "ПРОВЕРИЛИ ПОЧТИ 300", ("300",), Go_),
    (32.173, "БЕСПЛАТНЫХ VPN-ПРИЛОЖЕНИЙ:", ()), (34.293, "В КАЖДОМ ТРЕТЬЕМ НАШЛИ", ("ТРЕТЬЕМ",)),
    (35.633, "ВРЕДОНОСНЫЙ КОД,", ("ВРЕДОНОСНЫЙ",)), (36.933, "А МНОГИЕ ВООБЩЕ", ()),
    (37.853, "НЕ ШИФРОВАЛИ ДАННЫЕ", ("НЕ", "ШИФРОВАЛИ")),
    (39.41, "ПОМНИ ПРАВИЛО:", ("ПРАВИЛО:",), Go_), (40.601, "ЕСЛИ СЕРВИС БЕСПЛАТНЫЙ,", ("БЕСПЛАТНЫЙ,",)),
    (42.221, "ТО ТОВАР — ЭТО ТЫ", ("ТОВАР",), Pu_), (43.381, "И ТВОИ ДАННЫЕ", ("ДАННЫЕ",), Pu_),
    (44.861, "ВЫБИРАЙ ПРОВЕРЕННЫЕ", ("ПРОВЕРЕННЫЕ",), G_), (45.821, "ПЛАТНЫЕ СЕРВИСЫ,", ()),
    (47.141, "ЧИТАЙ ОТЗЫВЫ", ("ОТЗЫВЫ",), G_), (48.121, "И НЕ ВВОДИ ПАРОЛИ", ("ПАРОЛИ",)),
    (49.361, "ТАМ, ГДЕ СОМНЕВАЕШЬСЯ", ()),
    (51.413, "А ЕЩЁ БОЛЬШЕ ИНТЕРЕСНОГО —", (), T_), (52.773, "В МОЁМ ТЕЛЕГРАМ-КАНАЛЕ", ("ТЕЛЕГРАМ-КАНАЛЕ",), T_),
    (54.103, "«ПАУЧЬЯ НОРКА»", ("«ПАУЧЬЯ", "НОРКА»"), T_), (55.223, "ТАМ ЭКСКЛЮЗИВ,", ("ЭКСКЛЮЗИВ",), T_),
    (55.923, "КОТОРОГО НЕТ В РОЛИКАХ", (), T_), (57.223, "ССЫЛКА В ПРОФИЛЕ —", ("ПРОФИЛЕ",), T_),
    (58.243, "ЗАЛЕТАЙ!", ("ЗАЛЕТАЙ!",), T_),
]

# ---------------------------------------------------------------- камера и сцены
TILES = (15.35, 29.7, 39.25)     # смена сцены «плитками»; склейка — на середине
C1, C2, C3 = (x + 0.2 for x in TILES)
CAMERA = [
    (0.0, (-40.0, 3.9, 1.4, 0.9, 50.0)), (2.7, (6.0, 3.7, 1.2, 0.86, 50.0), "inout"),   # медленный облёт
    (2.95, (6.0, 3.7, 1.2, 0.86, 50.0)),
    (3.2, (0.0, 3.9, 1.3, 0.98, 50.0), "inout"),              # заставка
    (4.4, (-1.0, 3.9, 1.3, 0.98, 50.0)),
    (4.7, (92.0, 3.6, 1.25, 0.86, 50.0), "expo"),             # спрос, сервер, открытый код
    (C1 - 0.01, (99.0, 3.4, 1.25, 0.86, 50.0), "lin"),
    (C1 + 0.01, (185.0, 3.6, 1.3, 0.88, 50.0), "lin"),        # бот в Телеграме, сетка приложений
    (21.0, (188.0, 3.6, 1.3, 0.88, 50.0), "lin"),
    (21.35, (188.0, 4.6, 1.8, 1.0, 52.0), "out"),             # отъезд — видно всю сетку
    (C2 - 0.01, (190.0, 4.5, 1.8, 1.0, 52.0), "lin"),
    (C2 + 0.01, (278.0, 3.5, 0.85, 0.95, 50.0), "lin"),       # опыт CSIRO — чуть снизу
    (C3 - 0.01, (283.0, 3.4, 0.85, 0.95, 50.0), "lin"),
    (C3 + 0.01, (368.0, 3.6, 1.25, 0.86, 50.0), "lin"),       # правило и чек-лист
    (51.1, (374.0, 3.4, 1.25, 0.86, 49.0), "lin"),
    (51.45, (460.0, 4.3, 1.3, 1.12, 50.0), "expo"), (DUR, (461.0, 4.0, 1.3, 1.15, 50.0)),
]
OFFSET = [(0.0, -0.5), (2.95, -0.5), (3.2, 0.0, "inout"), (4.4, 0.0), (4.7, 0.55, "expo"), (C1 - 0.01, 0.55),
          (C1 + 0.01, -0.55, "lin"), (C2 - 0.01, -0.55), (C2 + 0.01, 0.55, "lin"), (C3 - 0.01, 0.55),
          (C3 + 0.01, -0.55, "lin"), (51.1, -0.55), (51.45, 0.0, "expo"), (DUR, 0.0)]
TINT = [(0.0, (1.15, 1.05, 0.88)), (2.95, (1.15, 1.05, 0.88)), (3.2, (1.0, 1.0, 1.0)), (4.4, (1.0, 1.0, 1.0)),
        (4.7, (1.12, 1.08, 0.88)), (C1 - 0.01, (1.12, 1.08, 0.88)), (C1 + 0.01, (1.2, 1.0, 0.9), "lin"),
        (C2 - 0.01, (1.2, 1.0, 0.9)), (C2 + 0.01, (0.9, 0.92, 1.18), "lin"), (C3 - 0.01, (0.9, 0.92, 1.18)),
        (C3 + 0.01, (0.95, 1.1, 0.95), "lin"), (51.1, (0.95, 1.1, 0.95)), (51.5, (1.0, 1.0, 1.0)),
        (DUR, (1.0, 1.0, 1.0))]
SEC = {"hook": -6, "make": 95, "sell": 188, "risk": 280, "rule": 371, "tg": 460.5}
SHAKES = (35.63,)
FLASHES = ((51.5, 0.12),)
JUMPS = [5.6]
ALARM = None
SPOT = None

G = lambda *k: list(k)  # noqa: E731
O = OUTRO - 48.035      # жесты концовки — как в monet_mg, со сдвигом
ARM_L = G((0, 62), (0.3, -30, "back"), (2.8, -28), (3.0, 62), (3.2, -50, "back"), (4.1, -45), (4.4, 62, "back"),
          (C1 + 0.05, 62), (15.8, -30, "back"), (20.9, -28), (21.2, -45, "back"), (27.4, -42), (27.7, 30, "back"),
          (29.3, 32), (29.5, 62), (C3 + 0.05, 62), (39.6, -30, "back"), (44.6, -28), (44.9, -40, "back"),
          (51.0, -38), (51.3, 62),
          (54.94 + O, 62), (55.09 + O, -50, "back"), (55.69 + O, -45), (56.09 + O, 62, "back"), (DUR, 62))
ARM_R = G((0, 62), (3.0, 62), (3.2, -50, "back"), (4.1, -45), (4.4, 62, "back"), (4.7, -30, "back"), (5.5, -28),
          (5.6, -55, "back"), (6.7, -52), (6.9, -30, "back"), (12.6, -28), (12.8, -40, "back"), (15.1, -38),
          (15.3, 62), (27.7, 30, "back"), (29.3, 32), (29.5, 62), (C2 + 0.05, 62), (30.0, -30, "back"),
          (34.2, -28), (34.3, -45, "back"), (36.8, -42), (37.0, -30, "back"), (39.0, -28), (39.2, 62),
          (47.79 + O, 62), (48.25 + O, -48, "back"), (50.6 + O, -45), (50.9 + O, -55, "back"), (53.6 + O, -52),
          (53.85 + O, -40, "back"), (54.8 + O, -42), (54.95 + O, -50, "back"), (55.69 + O, -45), (55.84 + O, -58),
          (55.99 + O, -32), (56.14 + O, -58), (56.29 + O, -32), (DUR, 62))
HEAD = G((0, 0), (0.6, -6), (1.5, 7), (2.3, -5), (3.2, 0), (4.7, 6), (5.6, -8), (6.8, 6), (7.9, -6), (9.5, 7),
         (10.7, -6), (12.7, 6), (14.3, -6), (15.6, 7), (16.7, -6), (18.3, 6), (19.5, -6), (21.2, 7), (22.5, -6),
         (24.5, 6), (26.0, -7), (27.8, 9), (28.7, -6), (29.5, 0), (29.8, 6), (31.0, -6), (32.2, 6), (34.3, -7),
         (35.6, 9), (37.0, -6), (37.9, 6), (39.4, -6), (40.6, 7), (42.2, -6), (43.4, 7), (44.9, -6), (46.3, 6),
         (47.1, -6), (48.4, 6), (49.4, -6), (51.0, 0), (51.7, -6), (58.6, 0), (DUR, 0))


# ---------------------------------------------------------------- картинки вставок
def shield_globe_tex(a):
    """Планета с меридианами и щит-замок поверх; a — поворот меридианов."""
    S = 620
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c, r = S / 2, 250
    d.ellipse([c - r, c - r, c + r, c + r], fill=(20, 40, 70, 255), outline=CYAN_RGB + (255,), width=8)
    for k in range(-2, 3):
        w = abs(r * math.cos(math.radians(a + k * 36)))
        d.ellipse([c - w, c - r, c + w, c + r], outline=(60, 140, 190), width=4)
    for k in (-2, -1, 0, 1, 2):
        y = c + k * r * 0.4
        w = math.sqrt(max(0, r * r - (y - c) ** 2))
        d.line([(c - w, y), (c + w, y)], fill=(60, 140, 190), width=4)
    pts = [(c, c - 150), (c + 120, c - 100), (c + 105, c + 40), (c, c + 150), (c - 105, c + 40), (c - 120, c - 100)]
    d.polygon(pts, fill=(16, 30, 26, 255))
    d.line(pts + [pts[0]], fill=GREEN_RGB + (255,), width=12, joint="curve")
    d.rounded_rectangle([c - 45, c - 20, c + 45, c + 60], 12, fill=GREEN_RGB + (255,))
    d.arc([c - 32, c - 70, c + 32, c - 6], 180, 360, fill=GREEN_RGB + (255,), width=12)
    return mg.premul(im)


def demand_tex(p):
    im = rounded((820, 420), 34, (18, 18, 26, 255), GREEN_RGB + (255,), 4)
    d = ImageDraw.Draw(im)
    d.text((40, 26), "СПРОС НА VPN", font=mg.font(52, tall=True), fill="white")
    for y in range(140, 380, 60):
        d.line([(60, y), (780, y)], fill=(40, 40, 52), width=2)
    n = max(2, int(30 * p))
    pts = [(60 + 700 * i / 29, 370 - 230 * ((i / 29) ** 2.4) - 10 * math.sin(i * 1.3)) for i in range(n)]
    d.line(pts, fill=GREEN_RGB + (255,), width=10, joint="curve")
    if p > 0.95:
        x, y = pts[-1]
        d.polygon([(x - 24, y + 10), (x + 14, y - 30), (x + 22, y + 18)], fill=GREEN_RGB + (255,))
    return mg.premul(im)


def rack_tex(t):
    """Серверная стойка с мигающими лампочками."""
    im = Image.new("RGBA", (360, 620), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([20, 10, 340, 610], 26, fill=(24, 26, 34, 255), outline=(110, 115, 135, 255), width=6)
    for i in range(6):
        y = 40 + i * 94
        d.rounded_rectangle([44, y, 316, y + 74], 12, fill=(40, 44, 58, 255))
        for j in range(4):
            on = (int(t * 6) + i * 3 + j * 5) % 7 < 4
            d.ellipse([64 + j * 30, y + 28, 82 + j * 30, y + 46], fill=GREEN_RGB + (255,) if on else (30, 70, 45))
        d.rounded_rectangle([200, y + 30, 296, y + 44], 6, fill=(70, 74, 92))
    return mg.premul(im)


def rent_tex():
    im = rounded((520, 200), 34, (18, 18, 26, 255), GOLD_RGB + (255,), 5)
    d = ImageDraw.Draw(im)
    d.text((40, 22), "АРЕНДА СЕРВЕРА", font=mg.font(40, tall=True), fill=(220, 220, 230))
    text_rub(d, (40, 78), "от 300 ₽", mg.font(84, tall=True), GOLD_RGB)
    d.text((340, 112), "/ мес", font=mg.font(44, tall=True), fill=(200, 200, 215))
    return mg.premul(im)


def code_tex(n):
    im = rounded((780, 440), 30, (14, 16, 22, 255), (72, 72, 92, 255), 3)
    d = ImageDraw.Draw(im)
    for i, c in enumerate((RED_RGB, GOLD_RGB, GREEN_RGB)):
        d.ellipse([30 + i * 40, 24, 56 + i * 40, 50], fill=c + (255,))
    d.text((180, 18), "ОТКРЫТЫЙ КОД", font=mg.font(40, tall=True), fill=(200, 200, 215))
    lines = ["$ git clone vpn-server", "$ install --quick", "$ start  :443", "", "  сервер готов ✓"]
    for i, s in enumerate(lines[:n]):
        col = GREEN_RGB if "готов" in s else (180, 200, 230)
        d.text((40, 100 + i * 62), s.replace("✓", ""), font=mg.font(40), fill=col + (255,))
    return mg.premul(im)


def bot_tex(n):
    """Чат с ботом в Телеграме: n появившихся сообщений."""
    im = rounded((620, 700), 40, (18, 26, 36, 255), TG_RGB + (255,), 5)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, 619, 100], 40, fill=(30, 44, 60, 255))
    d.ellipse([30, 22, 86, 78], fill=TG_RGB + (255,))
    d.text((106, 30), "VPN-бот", font=mg.font(44, tall=True), fill="white")
    msgs = [("bot", "Привет! Выбери тариф:"), ("btn", "1 месяц"), ("btn", "3 месяца"), ("me", "1 месяц"),
            ("bot", "Готово! Ключ отправлен")]
    y = 130
    for kind, s in msgs[:n]:
        f = mg.font(36)
        tw = d.textbbox((0, 0), s, font=f)[2]
        if kind == "bot":
            d.rounded_rectangle([30, y, 30 + tw + 50, y + 80], 26, fill=(40, 54, 72, 255))
            d.text((55, y + 18), s, font=f, fill="white")
        elif kind == "me":
            d.rounded_rectangle([590 - tw - 50, y, 590, y + 80], 26, fill=TG_RGB + (255,))
            d.text((590 - tw - 25, y + 18), s, font=f, fill="white")
        else:
            d.rounded_rectangle([30, y, 590, y + 80], 26, outline=TG_RGB + (255,), width=4)
            d.text((310 - tw / 2, y + 18), s, font=f, fill=TG_RGB + (255,))
        y += 100
    return mg.premul(im)


def app_tex(col, dead=False):
    im = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, 199, 199], 46, fill=(col if not dead else (60, 60, 70)) + (255,))
    pts = [(100, 36), (152, 58), (146, 118), (100, 164), (54, 118), (48, 58)]
    d.polygon(pts, fill=(255, 255, 255, 230) if not dead else (110, 110, 120, 255))
    if dead:
        d.line([(30, 30), (170, 170)], fill=RED_RGB + (255,), width=16)
    return mg.premul(im)


def free_tex():
    f = mg.font(96, tall=True)
    tw = int(ImageDraw.Draw(Image.new("RGBA", (10, 10))).textbbox((0, 0), "БЕСПЛАТНО", font=f)[2])
    Wt = 120 + tw + 130
    im = Image.new("RGBA", (Wt, 240), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.polygon([(0, 120), (90, 0), (Wt - 1, 0), (Wt - 1, 239), (90, 239)], fill=(30, 18, 22, 255))
    d.line([(0, 120), (90, 0), (Wt - 1, 0), (Wt - 1, 239), (90, 239), (0, 120)], fill=RED_RGB + (255,), width=8)
    d.text((110, 60), "БЕСПЛАТНО", font=f, fill="white")
    x = 120 + tw + 20
    d.polygon([(x + 45, 60), (x + 95, 160), (x - 5, 160)], fill=GOLD_RGB + (255,))
    d.text((x + 36, 84), "!", font=mg.font(64, tall=True), fill=(40, 30, 10))
    return mg.premul(im)


def phones_tex(bad):
    """Три телефона; при bad — в одном вирус."""
    im = Image.new("RGBA", (780, 460), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for i in range(3):
        x = 30 + i * 260
        evil = bad and i == 2
        d.rounded_rectangle([x, 20, x + 220, 440], 34, fill=(16, 16, 22, 255),
                            outline=(RED_RGB if evil else (150, 150, 165)) + (255,), width=8)
        d.rounded_rectangle([x + 18, 50, x + 202, 410], 22, fill=((70, 20, 26) if evil else (30, 36, 52)) + (255,))
        cx, cy = x + 110, 230
        if evil:
            for k in range(8):
                a = k * math.pi / 4
                d.line([(cx, cy), (cx + 66 * math.cos(a), cy + 66 * math.sin(a))], fill=(120, 230, 120), width=8)
            d.ellipse([cx - 44, cy - 44, cx + 44, cy + 44], fill=(80, 180, 90))
        else:
            pts = [(cx, cy - 56), (cx + 44, cy - 38), (cx + 40, cy + 14), (cx, cy + 56), (cx - 40, cy + 14),
                   (cx - 44, cy - 38)]
            d.polygon(pts, fill=GREEN_RGB + (255,))
    return mg.premul(im)


def lock_tex():
    im = Image.new("RGBA", (360, 420), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([40, 180, 320, 400], 30, fill=RED_RGB + (255,))
    d.arc([70, 30, 230, 230], 180, 330, fill=(200, 200, 215), width=30)
    d.line([(222, 92), (230, 140)], fill=(200, 200, 215), width=30)
    d.ellipse([160, 250, 200, 290], fill=(40, 10, 14))
    d.rectangle([172, 280, 188, 340], fill=(40, 10, 14))
    return mg.premul(im)


def barcode_tex():
    im = rounded((620, 320), 30, (240, 238, 232, 255), (180, 175, 165, 255), 3)
    d = ImageDraw.Draw(im)
    rng = np.random.default_rng(24)
    x = 40
    while x < 580:
        w = int(rng.integers(3, 14))
        d.rectangle([x, 40, x + w, 200], fill=(20, 20, 26))
        x += w + int(rng.integers(4, 12))
    d.text((40, 220), "ТОВАР — ЭТО ТЫ", font=mg.font(70, tall=True), fill=(20, 20, 26))
    return mg.premul(im)


def check_tex(n):
    im = rounded((820, 420), 34, (16, 24, 20, 255), GREEN_RGB + (255,), 4)
    d = ImageDraw.Draw(im)
    rows = ("проверенный платный", "читай отзывы", "пароли — где уверен")
    for i, s in enumerate(rows[:n]):
        y = 50 + i * 120
        d.rounded_rectangle([40, y, 120, y + 80], 22, fill=GREEN_RGB + (255,))
        d.line([(58, y + 42), (74, y + 58), (102, y + 24)], fill="white", width=10)
        d.text((150, y + 14), s, font=mg.font(48), fill="white")
    return mg.premul(im)


# ---------------------------------------------------------------- сцена
class Scene(base.Scene):
    cfg = sys.modules[__name__]

    def __init__(self, voice):
        super().__init__(voice)
        T = self.T
        tg = Image.open(os.path.join(ROOT, "assets", "logos", "telegram.png")).convert("RGBA")
        T["tglogo"] = mg.premul(tg.resize((320, int(tg.height * 320 / tg.width)), Image.LANCZOS))
        T["rent"] = rent_tex()
        T["apps"] = [app_tex(c) for c in (TG_RGB, GREEN_RGB, PURPLE_RGB, (240, 120, 60), CYAN_RGB, (230, 70, 120))]
        T["dead"] = app_tex((60, 60, 70), True)
        T["weekly"] = chip("НОВЫЕ — КАЖДУЮ НЕДЕЛЮ", CYAN_RGB)
        T["free"] = free_tex()
        T["csiro"] = card("ИССЛЕДОВАНИЕ", "CSIRO, АВСТРАЛИЯ", "283 бесплатных VPN-приложения", CYAN_RGB)
        T["third"] = stamp("1 ИЗ 3", RED_RGB)
        T["lock"] = lock_tex()
        T["noenc"] = chip("БЕЗ ШИФРОВАНИЯ", RED_RGB)
        T["barcode"] = barcode_tex()
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
        for x0 in TILES:  # «плитки»: квадраты волной сжимаются в точки, под ними склейка, потом вырастают
            p = (t - x0) / 0.4
            if 0 < p < 1:
                n = 9
                s = W // n
                out = np.full_like(fr, (20, 16, 24))
                for gy in range(0, H, s):
                    for gx in range(0, W, s):
                        wave = (gx / W + gy / H) / 2 * 0.4
                        q = clamp((p - wave) / 0.4) if p < 0.5 else clamp((p - 0.5 - wave) / 0.4)
                        k = 1 - smooth(q) if p < 0.5 else smooth(q)
                        if k < 0.02:
                            continue
                        w = max(1, int(s * k))
                        x1, y1 = gx + (s - w) // 2, gy + (s - w) // 2
                        out[y1:y1 + w, x1:x1 + w] = fr[y1:y1 + w, x1:x1 + w]
                fr = out
        return fr

    def items_for(self, t, cam, it):
        T = self.T
        P = lambda y0, *a, **k: place(y0, *a, **k)  # noqa: E731

        # 1. Крючок (персонаж справа, медленный облёт): планета и щит-замок
        y0 = SEC["hook"]
        k = pop(t, 0.15, 2.95, 0.5, 0.25)
        if k > 0:
            self.add_bb(it, cam, P(y0, -0.7, 1.75) + [0, 0.03 * math.sin(t * 2), 0], 0.62 * k,
                        self.cached("gl", shield_globe_tex, int(t * 40) % 36 * 5), anchor=(0.5, 0.5))

        # 2. Заставка
        k = pop(t, 3.2, 4.4, 0.35, 0.3)
        if k > 0:
            self.add_panel(it, cam, place(0, 0.0, 1.95, 0.35), face(0, -6), 1.2 * k, 0.564 * k, 0.11 * k,
                           T["stinger"], pitch=-6, side=(30, 24, 120), edge=(200, 190, 255))

        # 3. Как запускают (персонаж слева): график спроса → стойка сервера и аренда → открытый код
        y0 = SEC["make"]
        k = pop(t, 4.7, 9.4, 0.4, 0.25)
        if k > 0:
            pp = round(clamp((t - 5.4) / 1.3) * 30) / 30
            pos = P(y0, 0.72, 1.6) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 14), 0.9 * k, 0.461 * k, 0.06 * k, self.cached("dm", demand_tex, pp),
                           pitch=-6, side=(30, 70, 40), edge=(190, 255, 210))
        k = pop(t, 9.45, 12.65, 0.4, 0.25)
        if k > 0:
            self.add_bb(it, cam, P(y0, 0.95, 1.35), 0.75 * k, self.cached("rk", rack_tex, round(t * 6) / 6),
                        anchor=(0.5, 0.5))
            kr = pop(t, 10.6, 12.65, 0.35, 0.25)
            if kr > 0:
                self.add_bb(it, cam, P(y0, 0.5, 2.0), 0.16 * kr, T["rent"], anchor=(0.5, 0.5))
        k = pop(t, 12.7, C1, 0.4, 0.2)
        if k > 0:
            n = min(5, int((t - 12.9) / 0.4) + 1) if t > 12.9 else 0
            pos = P(y0, 0.72, 1.6) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 14), 0.9 * k, 0.508 * k, 0.06 * k, self.cached("cd", code_tex, n),
                           pitch=-6, side=(40, 40, 52), edge=(200, 200, 230))

        # 4. Продажа (персонаж справа): логотип Telegram и чат с ботом → сетка приложений появляется и гаснет
        y0 = SEC["sell"]
        r_, c_ = basis(y0)
        k = pop(t, C1 + 0.05, 21.05, 0.4, 0.25)
        if k > 0:
            n = min(5, int((t - 16.6) / 0.7) + 1) if t > 16.6 else 0
            self.add_panel(it, cam, P(y0, -0.72, 1.45) + [0, 0.02 * math.sin(t * 2), 0], face(y0, -12), 0.6 * k,
                           0.677 * k, 0.05 * k, self.cached("bot", bot_tex, n), side=(30, 50, 70))
            self.add_bb(it, cam, P(y0, -0.72, 2.25), 0.22 * k, T["tglogo"], anchor=(0.5, 0.5))
        k = pop(t, 21.1, AD_CUT + 0.3, 0.4, 0.2)
        if k > 0:
            for i in range(12):
                r, c = divmod(i, 4)
                t0 = 21.3 + 0.18 * i
                ka = pop(t, t0, AD_CUT + 0.3, 0.3, 0.2)
                if ka <= 0:
                    continue
                dead = t > 24.5 + 0.25 * ((i * 7) % 12) and i % 3 == 1
                q = P(y0, -1.35 + 0.32 * c, 2.25 - 0.36 * r)
                self.add_bb(it, cam, q, 0.26 * ka, T["dead"] if dead else T["apps"][i % 6], anchor=(0.5, 0.5))
            kw = pop(t, 22.5, 27.7, 0.35, 0.25)
            if kw > 0:
                self.add_bb(it, cam, P(y0, -0.85, 0.85), 0.1 * kw, T["weekly"], anchor=(0.5, 0.5))
            kf = pop(t, 27.8, AD_CUT + 0.3, 0.35, 0.2)
            if kf > 0:
                self.add_bb(it, cam, P(y0, -1.05, 0.95), 0.21 * kf, T["free"], anchor=(0.5, 0.5))

        # 5. Риск (персонаж слева, чуть снизу): опыт CSIRO → «1 из 3» с вирусом → открытый замок
        y0 = SEC["risk"]
        k = pop(t, C2 + 0.05, 34.25, 0.4, 0.25)
        if k > 0:
            pos = P(y0, 0.72, 1.6) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 14), 0.9 * k, 0.427 * k, 0.06 * k, T["csiro"], pitch=6,
                           side=(30, 60, 80), edge=(190, 230, 255))
        k = pop(t, 34.3, 36.9, 0.4, 0.25)
        if k > 0:
            self.add_bb(it, cam, P(y0, 0.75, 1.55), 0.5 * k, self.cached("ph", phones_tex, t > 35.6), anchor=(0.5, 0.5))
            if t > 35.63:
                ks = clamp((t - 35.63) / 0.15)
                self.add_bb(it, cam, P(y0, 0.75, 2.1, 0.1), 0.16 * (1 + 0.6 * (1 - ks)) * k, T["third"],
                            anchor=(0.5, 0.5))
        k = pop(t, 36.95, C3, 0.4, 0.2)
        if k > 0:
            self.add_bb(it, cam, P(y0, 0.75, 1.5) + [0, 0.02 * math.sin(t * 2), 0], 0.48 * k, T["lock"],
                        anchor=(0.5, 0.5))
            self.add_bb(it, cam, P(y0, 0.75, 2.1), 0.1 * k, T["noenc"], anchor=(0.5, 0.5))

        # 6. Правило (персонаж справа): штрихкод «товар — это ты» → чек-лист
        y0 = SEC["rule"]
        k = pop(t, C3 + 0.05, 44.8, 0.4, 0.25)
        if k > 0:
            self.add_panel(it, cam, P(y0, -0.72, 1.65) + [0, 0.02 * math.sin(t * 2), 0], face(y0, -12), 0.75 * k,
                           0.387 * k, 0.04 * k, T["barcode"], side=(150, 145, 135))
        k = pop(t, 44.85, 51.2, 0.4, 0.25)
        if k > 0:
            n = 1 + (t > 47.1) + (t > 48.1)
            pos = P(y0, -0.72, 1.6) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, -14), 0.9 * k, 0.461 * k, 0.06 * k, self.cached("ck", check_tex, n),
                           pitch=-6, side=(30, 70, 40), edge=(190, 255, 210))

        # 7. Концовка с Telegram — общий шаблон
        self.outro_items(t, cam, it, OUTRO, SEC["tg"])


SFX = [(t, "whoosh") for t in (0.0, 3.15, 4.6) + TILES + (21.2, 51.3)]


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
        for t in (0.6, 2.0, 3.8, 6.5, 11.5, 14.8, 15.55, 18.5, 20.5, 23.5, 26.5, 28.6, 29.9, 32.0, 35.8, 38.0,
                  39.45, 42.0, 48.5, 50.5, 55.0):
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
    base.mix_audio(tmp, voice, out, music=os.path.join(ROOT, "work", "vpnnew", "music.wav"), sfx_list=SFX, dur=DUR,
                   sfx_vol=0.15)  # «вжух» лёгкий и тихий
    os.remove(tmp)


if __name__ == "__main__":
    main()
