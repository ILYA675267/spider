"""Ролик №3: «Почему зарплата растёт, а денег всё равно не хватает?» (инфляция образа жизни).

python tools/money_mg.py work/money/voice_clean.wav output/money.mp4 [--preview] [--jobs 4]
Общий движок, персонаж и концовка с Telegram — из memory_mg.py (шаблон); здесь только сцены этого ролика.
Новое в монтаже (чтобы ролики не были однотипными): начало сверху с «краном» вниз, заставка с
«головокружительным» наездом (dolly zoom), наклон кадра на «ловушке», съёмка снизу, плавный облёт,
персонаж то слева, то справа, свой цвет у каждой сцены; вставки — печатающийся чек, живой график,
батарейка радости, карточка-перевёртыш, беговая дорожка, монета делится пополам, монеты сыплются в копилку.
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
from memory_mg import (BLUE_RGB, GRAY_RGB, RED, RED_RGB, TG_RGB, basis, card, chip, face, place, pop,  # noqa: E402,F401
                       rounded, stamp)
from mg import H, W, clamp, out_back, smooth, track, wobble  # noqa: E402

FPS = 30
ROOT = mg.ROOT
OUTRO = float(open(os.path.join(ROOT, "work", "money", "outro_start.txt")).read())  # 42.43
DUR = 51.1
GREEN_RGB, GOLD_RGB = (46, 196, 110), (240, 186, 60)
GREEN, GOLD = GREEN_RGB[::-1], GOLD_RGB[::-1]  # BGR

# ---------------------------------------------------------------- субтитры
T_, G_ = TG_RGB, GREEN_RGB
SUBS = [
    (0.22, "ПОЧЕМУ ЗАРПЛАТА РАСТЁТ,", ("ЗАРПЛАТА",), G_), (1.68, "А ДЕНЕГ ВСЁ РАВНО", ("ДЕНЕГ",)),
    (2.44, "НЕ ХВАТАЕТ?", ("ХВАТАЕТ?",)), (3.3, "", ()),
    (5.28, "ЭТО ЛОВУШКА,", ("ЛОВУШКА",)), (5.94, "КОТОРАЯ НАЗЫВАЕТСЯ", ()),
    (6.66, "ИНФЛЯЦИЯ ОБРАЗА ЖИЗНИ", ("ИНФЛЯЦИЯ",)), (8.4, "ПОЛУЧИЛ ПРИБАВКУ —", ("ПРИБАВКУ",), G_),
    (9.64, "И СРАЗУ КОФЕ ПОДОРОЖЕ", ("КОФЕ",)), (10.92, "ТАКСИ ВМЕСТО АВТОБУСА", ("ТАКСИ",)),
    (12.3, "НОВЫЕ ПОДПИСКИ", ("ПОДПИСКИ",)), (13.14, "И НОВЫЙ ТЕЛЕФОН", ("ТЕЛЕФОН",)),
    (14.3, "ТРАТЫ РАСТУТ РОВНО", ("ТРАТЫ",)), (15.36, "ВСЛЕД ЗА ДОХОДОМ", ("ДОХОДОМ",), G_),
    (16.74, "ЕЩЁ В ПРОШЛОМ ВЕКЕ", ()), (17.86, "ЭТО ЗАМЕТИЛ ИСТОРИК", ()),
    (18.94, "СИРИЛ ПАРКИНСОН", ("ПАРКИНСОН",), GOLD_RGB), (20.04, "РАСХОДЫ РАСТУТ", ("РАСХОДЫ",)),
    (20.98, "ДО УРОВНЯ ДОХОДОВ", ("ДОХОДОВ",), G_), (22.32, "А РАДОСТЬ ОТ НОВЫХ ВЕЩЕЙ", ("РАДОСТЬ",), GOLD_RGB),
    (23.54, "БЫСТРО ПРОХОДИТ", ("ПРОХОДИТ",)), (24.8, "ЧЕРЕЗ ПАРУ МЕСЯЦЕВ", ()),
    (25.82, "ДОРОГОЕ СТАНОВИТСЯ", ("ДОРОГОЕ",)), (26.42, "ОБЫЧНЫМ", ("ОБЫЧНЫМ",)),
    (27.6, "И ХОЧЕТСЯ ЕЩЁ", ("ЕЩЁ",)), (28.94, "ПСИХОЛОГИ НАЗЫВАЮТ ЭТО", ()),
    (30.18, "ГЕДОНИСТИЧЕСКОЙ", ("ГЕДОНИСТИЧЕСКОЙ",)), (30.9, "БЕГОВОЙ ДОРОЖКОЙ", ("ДОРОЖКОЙ",)),
    (32.08, "БЕЖИШЬ ВСЁ БЫСТРЕЕ,", ("БЫСТРЕЕ",)), (33.3, "А ОСТАЁШЬСЯ НА МЕСТЕ", ("НА", "МЕСТЕ")),
    (34.82, "КАК ВЫБРАТЬСЯ?", ("ВЫБРАТЬСЯ?",), G_), (35.82, "С КАЖДОЙ ПРИБАВКИ", ()),
    (36.56, "ПОЛОВИНУ СРАЗУ ОТКЛАДЫВАЙ", ("ПОЛОВИНУ",), G_), (38.26, "НАСТРОЙ АВТОПЕРЕВОД", ("АВТОПЕРЕВОД",), G_),
    (39.38, "В ДЕНЬ ЗАРПЛАТЫ", ()), (40.38, "И НАКОПЛЕНИЯ ПОЯВЯТСЯ САМИ", ("НАКОПЛЕНИЯ",), G_),
    (42.4, "А ЕЩЁ БОЛЬШЕ ИНТЕРЕСНОГО —", (), T_), (43.76, "В МОЁМ ТЕЛЕГРАМ-КАНАЛЕ", ("ТЕЛЕГРАМ-КАНАЛЕ",), T_),
    (45.1, "«ПАУЧЬЯ НОРКА»", ("«ПАУЧЬЯ", "НОРКА»"), T_), (46.22, "ТАМ ЭКСКЛЮЗИВ,", ("ЭКСКЛЮЗИВ",), T_),
    (46.92, "КОТОРОГО НЕТ В РОЛИКАХ", (), T_), (48.22, "ССЫЛКА В ПРОФИЛЕ —", ("ПРОФИЛЕ",), T_),
    (49.24, "ЗАЛЕТАЙ!", ("ЗАЛЕТАЙ!",), T_),
]

# ---------------------------------------------------------------- камера и сцены
# (yaw, dist, height, target_y, fov)
CAMERA = [
    (0.0, (-24.0, 3.3, 3.7, 0.45, 54.0)),                 # начало сверху…
    (1.7, (-8.0, 3.6, 1.15, 0.84, 50.0), "inout"),        # …кран опускается к персонажу
    (3.4, (-6.0, 3.6, 1.15, 0.9, 50.0)),
    (5.0, (-3.0, 6.9, 1.35, 1.0, 27.0), "inout"),         # заставка: dolly zoom (фон «наезжает»)
    (5.15, (-3.0, 6.9, 1.35, 1.0, 27.0)),
    (5.6, (96.0, 3.6, 1.2, 0.84, 50.0), "expo"),          # ловушка (кадр наклонён)
    (8.0, (102.0, 3.7, 1.25, 0.84, 50.0)),
    (8.45, (150.0, 4.1, 2.7, 0.9, 50.0), "expo"),         # переход «через верх»
    (8.95, (180.0, 3.6, 1.3, 1.02, 50.0), "out"),         # чек
    (13.9, (186.0, 3.65, 1.25, 0.86, 50.0)),              # камера сползает вниз за чеком
    (14.4, (190.0, 3.25, 1.15, 0.9, 50.0), "out"),        # график — наезд
    (16.3, (193.0, 3.3, 1.15, 0.9, 50.0)),
    (16.75, (268.0, 3.5, 0.42, 1.08, 52.0), "expo"),      # Паркинсон — снизу
    (22.0, (274.0, 3.4, 0.45, 1.06, 52.0)),
    (22.45, (350.0, 3.6, 1.2, 0.84, 50.0), "expo"),       # радость — медленный облёт
    (28.4, (368.0, 3.7, 1.3, 0.86, 50.0), "lin"),
    (28.85, (448.0, 4.3, 2.35, 0.72, 50.0), "expo"),      # беговая дорожка — сверху сбоку
    (34.4, (458.0, 4.2, 2.25, 0.74, 50.0)),
    (34.85, (530.0, 3.6, 1.2, 0.86, 50.0), "expo"),       # как выбраться
    (35.25, (532.0, 3.2, 1.15, 0.9, 50.0), "out"),       # «как выбраться?» — резкий наезд
    (36.0, (533.0, 3.6, 1.2, 0.86, 50.0), "inout"),
    (42.0, (540.0, 3.7, 1.25, 0.86, 50.0)),
    (42.45, (615.0, 4.3, 1.3, 1.12, 50.0), "expo"), (DUR, (616.0, 4.0, 1.3, 1.15, 50.0)),
]
# >0 — персонаж слева, вставки справа; <0 — наоборот
OFFSET = [(0.0, -0.5), (3.4, -0.5), (4.6, 0.0, "inout"), (5.15, 0.0), (5.6, 0.55, "expo"), (16.3, 0.55),
          (16.75, -0.55, "expo"), (22.0, -0.55), (22.45, 0.55, "expo"), (42.0, 0.55), (42.45, 0.0, "expo"), (DUR, 0.0)]
ROLL = [(0.0, 0.0), (5.6, 0.0), (5.95, -7.0, "back"), (7.9, -6.0), (8.3, 0.0), (DUR, 0.0)]
# цвет комнаты по сценам (множители BGR)
TINT = [(0.0, (1.0, 1.0, 1.0)), (5.3, (1.0, 1.0, 1.0)), (5.7, (0.82, 0.85, 1.18)), (8.0, (0.82, 0.85, 1.18)),
        (8.6, (1.0, 1.0, 1.0)), (16.4, (1.0, 1.0, 1.0)), (16.9, (0.8, 0.98, 1.15)), (22.1, (0.8, 0.98, 1.15)),
        (22.6, (1.0, 1.0, 1.0)), (24.4, (1.12, 0.98, 0.84), "inout"), (28.5, (1.12, 0.98, 0.84)),
        (28.9, (1.12, 0.86, 1.06)), (34.5, (1.12, 0.86, 1.06)), (34.9, (0.84, 1.12, 0.92)), (42.0, (0.84, 1.12, 0.92)),
        (42.4, (1.0, 1.0, 1.0)), (DUR, (1.0, 1.0, 1.0))]
SEC = {"hook": -8, "trap": 99, "bill": 184, "park": 271, "joy": 360, "run": 452, "fix": 534, "tg": 615.5}
SHAKES = (5.95,)
FLASHES = ((3.5, 0.18), (5.4, 0.08), (8.45, 0.1), (16.55, 0.1), (22.25, 0.1), (28.65, 0.1), (34.65, 0.1),
           (42.25, 0.12))
JUMPS = [3.45, 32.1, 32.62, 33.14]
ALARM = None
SPOT = None

G = lambda *k: list(k)  # noqa: E731
# L — рука слева на экране, R — справа. Когда персонаж справа (крючок, Паркинсон) — показывает левой.
ARM_L = G((0, 62), (0.5, 62), (0.75, -25, "back"), (2.3, -22), (2.5, 30, "back"), (3.2, 32), (3.4, -50, "back"),
          (4.6, -45), (5.0, 62, "back"), (5.4, 62), (5.6, 35, "back"), (7.8, 36), (8.1, 62), (17.0, 62),
          (17.25, -30, "back"), (19.0, -28), (19.3, -10, "back"), (21.9, -8), (22.2, 62), (32.0, 62),
          (32.1, 20, "back"), (33.2, 20), (33.45, 40, "back"), (34.3, 42), (34.6, 62), (34.9, 25, "back"),
          (35.7, 28), (36.0, 62), (49.2, 62), (49.35, -50, "back"), (50.1, -45), (50.5, 62, "back"), (DUR, 62))
ARM_R = G((0, 62), (2.3, 62), (2.5, 30, "back"), (3.2, 32), (3.4, -50, "back"), (4.6, -45), (5.0, 62, "back"),
          (5.4, 62), (5.6, 35, "back"), (7.8, 36), (8.1, 62), (8.6, 62), (8.9, -30, "back"), (11.0, -28),
          (11.2, -12, "back"), (13.9, -10), (14.2, -35, "back"), (16.2, -32), (16.5, 62), (22.4, 62),
          (22.6, -20, "back"), (24.6, -18), (24.8, 62), (25.7, 62), (25.9, -30, "back"), (27.4, -28),
          (27.55, 30, "back"), (28.4, 32), (28.7, 62), (29.0, -25, "back"), (31.9, -22), (32.1, 20, "back"),
          (33.2, 20), (33.45, 40, "back"), (34.3, 42), (34.6, 62), (34.9, 25, "back"), (35.7, 28), (35.9, -15, "back"),
          (38.1, -12), (38.3, -35, "back"), (40.3, -32), (40.5, -10, "back"), (41.9, -8), (42.2, 62),
          (42.65, -48, "back"), (45.0, -45), (45.3, -55, "back"), (48.0, -52), (48.25, -40, "back"), (49.2, -42),
          (49.35, -50, "back"), (50.1, -45), (50.25, -58), (50.4, -32), (50.55, -58), (50.7, -32), (DUR, 62))
HEAD = G((0, 0), (0.8, 6), (2.5, -6), (3.4, 0), (5.6, -7), (7.0, 5), (8.9, -6), (11.2, -5), (13.9, 4), (14.3, -6),
         (16.5, 0), (17.2, 7), (19.3, 6), (21.9, 4), (22.6, -6), (24.4, 6), (25.9, -6), (27.6, 8), (29.0, -6),
         (31.9, -4), (33.3, 7), (34.9, 8), (35.9, -6), (40.5, -5), (42.65, -6), (49.3, 0), (DUR, 0))


# ---------------------------------------------------------------- картинки вставок
def text_rub(d, xy, s, font, fill):
    """Текст со знаком рубля в конце: в шрифтах канала нет «₽», рисуем его сами — буква «Р» с чертой."""
    x, y = xy
    if not s.endswith("₽"):
        d.text(xy, s, font=font, fill=fill)
        return
    s = s[:-1] + "Р"
    d.text(xy, s, font=font, fill=fill)
    x0 = x + d.textlength(s[:-1], font=font)
    l, t_, r, b = font.getbbox("Р")
    w = r - l
    yy = y + t_ + (b - t_) * 0.76
    d.line([(x0 + l - w * 0.22, yy), (x0 + l + w * 0.62, yy)], fill=fill, width=max(2, int(w * 0.17)))


def rub_width(d, s, font):
    return d.textlength(s.replace("₽", "Р"), font=font)


def wallet_tex():
    im = rounded((600, 370), 34, (18, 18, 26, 255), (72, 72, 92, 255), 3)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([36, 34, 250, 96], 30, fill=RED_RGB + (255,))
    d.text((58, 42), "КОШЕЛЁК", font=mg.font(36), fill="white")
    text_rub(d, (36, 112), "0 ₽", mg.font(128, tall=True), (255, 255, 255))
    d.text((40, 290), "к концу месяца", font=mg.font(38), fill=(150, 150, 170))
    return mg.premul(im)
def receipt_tex(n_rows):
    """Тёмный чек: строки покупок появляются по одной, внизу итог."""
    Wr, Hr = 600, 900
    im = Image.new("RGBA", (Wr, Hr), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, Wr - 1, Hr - 1], 30, fill=(24, 24, 32, 255), outline=(80, 80, 98, 255), width=3)
    d.text((40, 30), "ЧЕК", font=mg.font(64, tall=True), fill="white")
    d.text((40, 112), "после прибавки", font=mg.font(34), fill=(140, 140, 160))
    for x in range(30, Wr - 30, 26):
        d.line([(x, 176), (x + 12, 176)], fill=(90, 90, 110), width=3)
    rows = [("Кофе подороже", "+350 ₽"), ("Такси", "+600 ₽"), ("Подписки", "+899 ₽"), ("Телефон", "+89 990 ₽")]
    for i, (a, b) in enumerate(rows[:n_rows]):
        y = 210 + i * 112
        d.rounded_rectangle([28, y, Wr - 28, y + 92], 22, fill=(34, 34, 44, 255))
        d.text((50, y + 22), a, font=mg.font(40), fill=(220, 220, 232))
        f = mg.font(40)
        text_rub(d, (Wr - 50 - rub_width(d, b, f), y + 22), b, f, (255, 120, 120))
    if n_rows >= 5:
        y = 210 + 4 * 112 + 20
        for x in range(30, Wr - 30, 26):
            d.line([(x, y), (x + 12, y)], fill=(90, 90, 110), width=3)
        d.rounded_rectangle([28, y + 30, Wr - 28, y + 150], 30, fill=RED_RGB + (255,))
        d.text((54, y + 52), "ИТОГО: −ПРИБАВКА", font=mg.font(56, tall=True), fill="white")
    return mg.premul(im)


def chart_tex(p):
    """График: доход (зелёный) и траты (красный) растут вместе; p — сколько нарисовано (0..1)."""
    Wc, Hc = 780, 560
    im = rounded((Wc, Hc), 34, (18, 18, 26, 255), (72, 72, 92, 255), 3)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([36, 30, 240, 92], 30, fill=GREEN_RGB + (255,))
    d.text((58, 38), "ДОХОД", font=mg.font(36), fill="white")
    d.rounded_rectangle([260, 30, 450, 92], 30, fill=RED_RGB + (255,))
    d.text((282, 38), "ТРАТЫ", font=mg.font(36), fill="white")
    x0, x1, y0, y1 = 60, Wc - 60, Hc - 60, 140
    for k in range(4):
        y = y0 - (y0 - y1) * k / 3
        d.line([(x0, y), (x1, y)], fill=(40, 40, 52), width=2)
    xs = np.linspace(0, 1, 60)
    inc = 0.1 + 0.8 * xs ** 1.3 + 0.03 * np.sin(xs * 9)
    exp = inc - 0.035 + 0.02 * np.sin(xs * 13 + 1)
    n = max(2, int(len(xs) * p))
    for vals, col in ((inc, GREEN_RGB), (exp, RED_RGB)):
        pts = [(x0 + (x1 - x0) * x, y0 - (y0 - y1) * v) for x, v in zip(xs[:n], vals[:n])]
        d.line(pts, fill=col + (255,), width=10, joint="curve")
        cx, cy = pts[-1]
        d.ellipse([cx - 12, cy - 12, cx + 12, cy + 12], fill=col + (255,))
    return mg.premul(im)


def battery_tex(level):
    """Батарейка «РАДОСТЬ»: заряд level (0..1), цвет от зелёного к красному."""
    Wb, Hb = 720, 330
    im = rounded((Wb, Hb), 34, (18, 18, 26, 255), (72, 72, 92, 255), 3)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([36, 30, 260, 92], 30, fill=GOLD_RGB + (255,))
    d.text((58, 38), "РАДОСТЬ", font=mg.font(36), fill=(30, 24, 10))
    d.rounded_rectangle([40, 130, 600, 290], 34, outline=(200, 200, 215, 255), width=8)
    d.rounded_rectangle([606, 175, 640, 245], 12, fill=(200, 200, 215, 255))
    col = tuple(int(a + (b - a) * (1 - level)) for a, b in zip(GREEN_RGB, RED_RGB))
    if level > 0.02:
        d.rounded_rectangle([58, 148, 58 + int(524 * level), 272], 22, fill=col + (255,))
    f = mg.font(60, tall=True)
    txt = f"{int(round(level * 100))}%"
    tw = d.textbbox((0, 0), txt, font=f)[2]
    d.text((320 - tw / 2, 170), txt, font=f, fill="white")
    return mg.premul(im)


def tag_tex(text, sub, color):
    im = rounded((640, 300), 34, (20, 20, 28, 255), color + (255,), 6)
    d = ImageDraw.Draw(im)
    d.ellipse([40, 40, 90, 90], outline=color + (255,), width=8)  # «дырочка» ценника
    f = mg.font(110, tall=True)
    tw = d.textbbox((0, 0), text, font=f)[2]
    d.text((320 - tw / 2, 70), text, font=f, fill="white")
    f2 = mg.font(38)
    tw = d.textbbox((0, 0), sub, font=f2)[2]
    d.text((320 - tw / 2, 214), sub, font=f2, fill=(160, 160, 180))
    return mg.premul(im)


def coin_tex(text="₽", size=300):
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([0, 0, size - 1, size - 1], fill=(196, 140, 30, 255))
    d.ellipse([size * 0.07, size * 0.07, size * 0.93, size * 0.93], fill=GOLD_RGB + (255,))
    f = mg.font(int(size * 0.5), tall=True)
    l, t_, r, b = d.textbbox((0, 0), text.replace("₽", "Р"), font=f)
    text_rub(d, ((size - (r - l)) / 2 - l + (size * 0.03 if text == "₽" else 0), (size - (b - t_)) / 2 - t_), text, f,
             (120, 80, 10))
    return mg.premul(im)


def half_tex(top, big, color):
    im = rounded((420, 250), 30, (20, 20, 28, 255), color + (255,), 5)
    d = ImageDraw.Draw(im)
    d.text((30, 26), top, font=mg.font(36), fill=(170, 170, 190))
    d.text((30, 90), big, font=mg.font(96, tall=True), fill="white")
    return mg.premul(im)


def auto_tex(on):
    """Карточка банка: «Автоперевод в копилку» с переключателем; on — насколько включён (0..1)."""
    im = rounded((780, 420), 34, (18, 18, 26, 255), (72, 72, 92, 255), 3)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([36, 34, 330, 96], 30, fill=GREEN_RGB + (255,))
    d.text((58, 42), "АВТОПЕРЕВОД", font=mg.font(36), fill="white")
    d.text((40, 130), "50% в копилку", font=mg.font(64, tall=True), fill="white")
    d.text((40, 220), "каждый день зарплаты", font=mg.font(40), fill=(150, 150, 170))
    col = tuple(int(a + (b - a) * on) for a, b in zip((70, 70, 84), GREEN_RGB))
    d.rounded_rectangle([40, 300, 180, 370], 35, fill=col + (255,))
    x = int(75 + 70 * on)
    d.ellipse([x - 30, 305, x + 30, 365], fill="white")
    d.text((210, 308), "ВКЛ" if on > 0.5 else "ВЫКЛ", font=mg.font(44), fill=(230, 230, 240) if on > 0.5 else (130, 130, 145))
    return mg.premul(im)


def savings_tex(amount):
    im = rounded((620, 240), 30, (18, 26, 22, 255), GREEN_RGB + (255,), 5)
    d = ImageDraw.Draw(im)
    d.text((34, 24), "НАКОПЛЕНИЯ", font=mg.font(38), fill=(150, 220, 180))
    text_rub(d, (34, 90), f"{amount:,} ₽".replace(",", " "), mg.font(100, tall=True), (255, 255, 255))
    return mg.premul(im)


# ---------------------------------------------------------------- сцена
class Scene(base.Scene):
    cfg = sys.modules[__name__]

    def __init__(self, voice):
        super().__init__(voice)
        T = self.T
        T["income"] = chip("ЗАРПЛАТА ↑", GREEN_RGB)
        T["wallet"] = wallet_tex()
        T["trap"] = card("ЛОВУШКА", "ИНФЛЯЦИЯ", "образа жизни")
        T["raise"] = chip("+ ПРИБАВКА", GREEN_RGB)
        T["parkinson"] = card("ИСТОРИК", "СИРИЛ ПАРКИНСОН", "Великобритания, XX век", GOLD_RGB)
        T["law"] = card("ЗАКОН ПАРКИНСОНА", "РАСХОДЫ = ДОХОДЫ", "расходы растут до уровня доходов", GOLD_RGB)
        T["l_inc"], T["l_exp"] = base.label("ДОХОД", (150, 240, 190), 60), base.label("РАСХОДЫ", (255, 140, 140), 60)
        T["new"] = tag_tex("НОВИНКА", "дорогое и радует", GOLD_RGB)
        T["usual"] = tag_tex("ОБЫЧНОЕ", "через 2 месяца", GRAY_RGB)
        T["more"] = stamp("ХОЧУ ЕЩЁ!", GOLD_RGB)
        T["mill"] = card("ПСИХОЛОГИЯ", "БЕГОВАЯ ДОРОЖКА", "гедонистическая адаптация", (150, 90, 230))
        T["place"] = stamp("0 МЕТРОВ", (150, 90, 230))
        T["coin"], T["coin_s"] = coin_tex("+", 300), coin_tex("₽", 160)
        T["half_a"] = half_tex("на жизнь", "50%", GRAY_RGB)
        T["half_b"] = half_tex("в копилку", "50%", GREEN_RGB)
        T["howout"] = chip("КАК ВЫБРАТЬСЯ?", GREEN_RGB)
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

    def coins_fall(self, it, cam, target, t, t0, t1, n=9):
        """Монетки падают сверху в точку target (с отскоком масштаба при «попадании»)."""
        rng = np.random.default_rng(5)
        for i in range(n):
            s = t0 + (t1 - t0) * i / n + rng.uniform(0, 0.1)
            p = (t - s) / 0.55
            if not 0 < p < 1:
                continue
            dx, dz = rng.uniform(-0.12, 0.12), rng.uniform(-0.05, 0.05)
            pos = target + [dx * (1 - p), 0.55 * (1 - p * p), dz]
            self.add_bb(it, cam, pos, 0.1 * (1 - 0.4 * max(0.0, p - 0.85) / 0.15), self.T["coin_s"], anchor=(0.5, 0.5))

    def treadmill(self, it, cam, y0, t, k):
        """Беговая дорожка на полу: корпус, лента с бегущими полосками, стойки и поручень."""
        r_, c_ = basis(y0)
        cen = r_ * 0.95 + c_ * 0.35
        L, Wd = 1.0 * k, 0.42 * k
        if k <= 0.01:
            return
        yaw = math.degrees(math.atan2(c_[2], c_[0]))  # длинная сторона — вдоль взгляда камеры сцены
        self.add_box(it, cam, (cen[0], cen[2]), (L, Wd), 0.1 * k, (52, 46, 58), yaw=yaw, radius=0.06)

        def belt():
            speed = 1.2 + 2.2 * smooth((t - 32.0) / 1.0)
            ph = (t * speed) % 0.16
            lay = np.zeros_like(self.frame)
            for j in range(7):
                u = -L / 2 + 0.08 + ((j * 0.16 + ph) % (L - 0.16))
                a = cen + c_ * u - r_ * Wd * 0.38 + [0, 0.105 * k, 0]
                b = cen + c_ * u + r_ * Wd * 0.38 + [0, 0.105 * k, 0]
                uv, z = cam.proj(np.array([a, b]))
                if (z > 0.3).all():
                    mg.lines(lay, uv, (150, 110, 230), 4, closed=False)
            cv2.add(self.frame, lay, self.frame)
        it.append((cam.depth(cen + [0, 0.1, 0]) - 0.05, belt))
        for sgn in (-1, 1):  # стойки у дальнего конца
            p = cen + c_ * (-L / 2 + 0.06) + r_ * sgn * Wd * 0.45
            self.add_box(it, cam, (p[0], p[2]), (0.05, 0.05), 0.75 * k, (80, 70, 95), radius=0.02)
        p = cen + c_ * (-L / 2 + 0.06)
        self.add_panel(it, cam, p + [0, 0.78 * k, 0], face(y0, 0), 0.34 * k, 0.06 * k, 0.04, self.cached("bar", rounded_bar),
                       side=(80, 70, 95), radius=0.03 * k)

    def items_for(self, t, cam, it):
        T = self.T
        P = lambda y0, *a, **k: place(y0, *a, **k)  # noqa: E731

        # 1. Крючок (персонаж справа): зарплата растёт ↑, а в кошельке 0
        y0 = SEC["hook"]
        k = pop(t, 0.55, 3.45)
        if k > 0:
            g = smooth((t - 0.6) / 1.6)
            for i, (side, col, lab) in enumerate(((-0.95, GREEN, "l_inc"), (-0.6, RED, "l_exp"))):
                h = (0.25 + 0.95 * g) * k * (1 if i == 0 else 0.97)
                p = P(y0, side, 0.0, 0.25)
                self.add_box(it, cam, (p[0], p[2]), (0.22, 0.22), h, col, radius=0.06)
                self.add_bb(it, cam, p + [0, h + 0.04, 0], 0.07 * k, T[lab], anchor=(0.5, 1.0))
        k = pop(t, 0.3, 1.9, 0.35, 0.25)
        if k > 0:
            self.add_bb(it, cam, P(y0, -0.75, 1.9), 0.13 * k, T["income"], anchor=(0.5, 0.5))
        k = pop(t, 1.8, 3.45, 0.35, 0.25)
        if k > 0:
            pos = P(y0, -0.75, 1.75) + [0, 0.02 * math.sin(t * 3), 0]
            self.add_panel(it, cam, pos, face(y0, -14), 0.62 * k, 0.382 * k, 0.05 * k, T["wallet"], pitch=-6,
                           side=(90, 40, 40), edge=(255, 200, 200))

        # 2. Заставка «Паучок объясняет» — во время dolly zoom
        k = pop(t, 3.5, 5.25, 0.4, 0.3)
        if k > 0:
            self.add_panel(it, cam, place(-3, 0.0, 1.95, 0.35), face(-3, -6), 1.35 * k, 0.635 * k, 0.12 * k,
                           T["stinger"], pitch=-6, side=(30, 24, 120), edge=(200, 190, 255))

        # 3. Ловушка (наклонённый кадр)
        y0 = SEC["trap"]
        k = pop(t, 5.75, 8.2, 0.4, 0.25)
        if k > 0:
            pos = P(y0, 0.72, 1.5) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 14), 0.9 * k, 0.48 * k, 0.06 * k, T["trap"], pitch=-6,
                           side=(90, 40, 40), edge=(255, 200, 200))

        # 4. Чек «печатается»: строки по словам, итог — «минус прибавка»
        y0 = SEC["bill"]
        k = pop(t, 8.35, 9.6, 0.35, 0.25)
        if k > 0:
            self.add_bb(it, cam, P(y0, 0.78, 1.95), 0.13 * k, T["raise"], anchor=(0.5, 0.5))
        k = pop(t, 9.2, 14.25, 0.4, 0.3)
        if k > 0:
            n = 0 + (t > 9.94) + (t > 10.92) + (t > 12.6) + (t > 13.46) + (t > 13.82)
            grow = 0.62 + 0.38 * smooth((t - 12.6) / 0.6)  # чек «вытягивается» вниз
            hh = 0.9 * grow
            pos = P(y0, 0.84, 2.0 - hh * 1.15 * 1.25 / 2) + [0, 0.02 * math.sin(t * 2), 0]
            tex = self.cached("receipt", receipt_tex, n)
            th = tex.shape[0]
            cut = int(th * grow)
            self.add_panel(it, cam, pos, face(y0, 12), 0.6 * 1.15 * k, hh * 1.15 * k, 0.03 * k,
                           np.ascontiguousarray(tex[:cut]), pitch=-4,
                           side=(50, 50, 62), edge=(220, 220, 240), radius=0.04)
        k = pop(t, 14.1, 16.65, 0.4, 0.3)
        if k > 0:
            p = round(smooth((t - 14.3) / 1.8) * 30) / 30
            pos = P(y0, 0.75, 1.42) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 12), 0.9 * k, 0.646 * k, 0.06 * k, self.cached("chart", chart_tex, p),
                           pitch=-5, side=(40, 70, 50), edge=(200, 255, 220))

        # 5. Паркинсон (съёмка снизу, персонаж справа): карточка, закон, два равных столбика
        y0 = SEC["park"]
        for key, a, b, hy in (("parkinson", 17.0, 20.0, 1.55), ("law", 19.95, 22.2, 1.6)):
            k = pop(t, a, b, 0.4, 0.25)
            if k > 0:
                pos = P(y0, -0.75, hy) + [0, 0.03 * math.sin(t * 2), 0]
                self.add_panel(it, cam, pos, face(y0, -14), 0.9 * k, 0.48 * k, 0.06 * k, T[key], pitch=6,
                               side=(70, 60, 30), edge=(255, 230, 170))
        k = pop(t, 20.3, 22.2)
        if k > 0:
            g = smooth((t - 20.4) / 1.4)
            for side, col, lab, d in ((-1.0, GREEN, "l_inc", 0.0), (-0.68, RED, "l_exp", 0.25)):
                h = (0.15 + 0.6 * smooth((t - 20.4 - d) / 1.2)) * k
                p = P(y0, side, 0.0, 0.3)
                self.add_box(it, cam, (p[0], p[2]), (0.2, 0.2), h, col, radius=0.06)
                self.add_bb(it, cam, p + [0, h + 0.04, 0], 0.07 * k * g, T[lab], anchor=(0.5, 1.0))

        # 6. Радость проходит: батарейка садится; ценник переворачивается «НОВИНКА» → «ОБЫЧНОЕ»; «хочу ещё!»
        y0 = SEC["joy"]
        k = pop(t, 22.6, 25.0, 0.4, 0.3)
        if k > 0:
            lvl = round((1 - 0.88 * smooth((t - 23.4) / 1.1)) * 25) / 25
            pos = P(y0, 0.72, 1.5) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 12), 0.84 * k, 0.385 * k, 0.06 * k,
                           self.cached("bat", battery_tex, lvl), pitch=-5, side=(70, 60, 30), edge=(255, 230, 170))
        k = pop(t, 24.9, 28.6, 0.4, 0.3)
        if k > 0:
            flip = 180 * smooth((t - 26.1) / 0.5)
            pos = P(y0, 0.72, 1.5) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 10) + flip, 0.75 * k, 0.35 * k, 0.05 * k, T["new"], pitch=-5,
                           side=(70, 60, 30) if flip < 90 else (50, 50, 60), back_tex=T["usual"], radius=0.05)
        k = pop(t, 27.6, 28.6, 0.2, 0.25)
        if k > 0:
            ks = clamp((t - 27.6) / 0.15)
            self.add_bb(it, cam, P(y0, 0.95, 1.12, 0.1), 0.15 * (1 + 0.6 * (1 - ks)) * k, T["more"], anchor=(0.5, 0.5))

        # 7. Гедонистическая беговая дорожка
        y0 = SEC["run"]
        k = pop(t, 29.0, 34.6, 0.5, 0.3)
        if k > 0:
            self.treadmill(it, cam, y0, t, k)
            pos = P(y0, 0.72, 1.82) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 12), 0.9 * k, 0.48 * k, 0.06 * k, T["mill"], pitch=-14,
                           side=(70, 40, 90), edge=(220, 190, 255))
        k = pop(t, 33.3, 34.6, 0.2, 0.25)
        if k > 0:
            ks = clamp((t - 33.3) / 0.15)
            self.add_bb(it, cam, P(y0, 0.85, 1.25, 0.1), 0.14 * (1 + 0.6 * (1 - ks)) * k, T["place"], anchor=(0.5, 0.5))

        # 8. Как выбраться: прибавка-монета делится пополам → автоперевод → копилка растёт, монетки падают
        y0 = SEC["fix"]
        k = pop(t, 34.9, 36.0, 0.3, 0.25)
        if k > 0:
            self.add_bb(it, cam, P(y0, 0.75, 1.85), 0.13 * k, T["howout"], anchor=(0.5, 0.5))
        k = pop(t, 35.85, 38.25, 0.35, 0.3)
        if k > 0:
            sp = smooth((t - 36.6) / 0.45)  # монета «раскалывается» на две половины
            c0 = P(y0, 0.75, 1.5)
            if sp < 0.98:
                self.add_panel(it, cam, c0, face(y0, 10 + 360 * smooth((t - 35.85) / 0.7)), 0.4 * k * (1 - sp),
                               0.4 * k * (1 - sp), 0.06 * k, T["coin"], side=(30, 110, 170), radius=0.2 * k * (1 - sp))
            if sp > 0.02:
                for key, dx, col in (("half_a", -0.31, (60, 60, 70)), ("half_b", 0.31, (60, 120, 80))):
                    self.add_panel(it, cam, c0 + basis(y0)[0] * dx * sp, face(y0, 10), 0.5 * sp * k,
                                   0.298 * sp * k, 0.05 * k, T[key], pitch=-5, side=col, radius=0.03)
        k = pop(t, 38.25, 40.4, 0.4, 0.3)
        if k > 0:
            on = round(smooth((t - 39.4) / 0.35) * 8) / 8
            pos = P(y0, 0.75, 1.5) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 12), 0.9 * k, 0.485 * k, 0.06 * k, self.cached("auto", auto_tex, on),
                           pitch=-5, side=(40, 70, 50), edge=(200, 255, 220))
        k = pop(t, 40.3, 42.2)
        if k > 0:
            g = smooth((t - 40.5) / 1.3)
            amount = int(round(10000 + 110000 * g, -3))
            p = P(y0, 0.95, 0.0, 0.25)
            h = (0.15 + 0.55 * g) * k
            self.add_box(it, cam, (p[0], p[2]), (0.26, 0.26), h, GREEN, radius=0.08, glow=(80, 200, 120))
            self.coins_fall(it, cam, p + [0, h + 0.05, 0], t, 40.45, 41.8)
            pos = P(y0, 0.72, 1.72) + [0, 0.02 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 12), 0.72 * k, 0.279 * k, 0.05 * k,
                           self.cached("sav", savings_tex, amount), pitch=-5, side=(40, 70, 50), edge=(200, 255, 220))

        # 9. Концовка с Telegram — общий шаблон
        self.outro_items(t, cam, it, OUTRO, SEC["tg"])


def rounded_bar():
    return mg.premul(rounded((340, 60), 30, (110, 100, 130, 255)))


SFX = [(t, "whoosh") for t in (0.0, 3.5, 5.15, 8.0, 14.1, 16.3, 22.0, 28.4, 34.4, 42.0)]


def render(voice, out, frames):
    sc = Scene(voice)
    wr = mg.Writer(out, FPS)
    for n, i in enumerate(frames):
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
        wr.write(sc.finish(fr, t))
        if n % 100 == 0:
            print(f"{out}: {n}/{len(frames)}", flush=True)
    wr.close()


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    voice, out = args[0], args[1]
    n = int(DUR * FPS)
    if "--chunk" in sys.argv:  # служебный режим: один кусок кадров (для параллельного рендера)
        a, b = int(args[2]), int(args[3])
        cv2.setNumThreads(1)
        render(voice, out, range(a, b))
        return
    if "--preview" in sys.argv:
        sc = Scene(voice)
        for t in (0.3, 1.2, 2.2, 3.0, 4.2, 4.9, 6.8, 9.0, 11.5, 13.95, 15.4, 18.0, 21.2, 23.6, 26.3, 26.9, 28.0, 30.5,
                  33.6, 35.4, 36.9, 37.6, 39.8, 41.4, 44.0, 49.0):
            fr = sc.finish(sc.draw(t), t)
            cv2.imwrite(f"{out}_{int(t * 100):05d}.jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return
    # параллельный рендер: кадры делятся на куски, каждый кусок — отдельный процесс (те же кадры, то же качество)
    jobs = 4
    if "--jobs" in sys.argv:
        jobs = int(sys.argv[sys.argv.index("--jobs") + 1])
        args = [a for a in args if a != str(jobs)]
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
    base.mix_audio(tmp, voice, out, music=os.path.join(ROOT, "work", "money", "music.wav"), sfx_list=SFX, dur=DUR,
                   sfx_vol=0.15)  # «вжух» лёгкий и тихий
    os.remove(tmp)


if __name__ == "__main__":
    main()
