"""Ролик №1: «Почему ты помнишь, как опозорился в школе, но не помнишь, что ел вчера?»

python3 tools/memory_mg.py work/memory/voice_full.wav output/memory.mp4 [--preview]

Структура (правила канала): вопрос-крючок → «Паучок объясняет» → объяснение по блокам
(у каждого блока свой ракурс камеры и объёмная вставка) → концовка с Telegram «Паучья Норка».
"""
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mg  # noqa: E402
from intro_mg import CHAR, VIGNETTE  # noqa: E402
from mg import H, W, clamp, out_back, smooth, track, wobble  # noqa: E402

FPS = 30
DUR = 59.9
ROOT = mg.ROOT
CHAR_H = 1.15

RED, BLUE, WHITE, DARK, GRAY = (46, 38, 226), (214, 84, 30), (238, 236, 240), (30, 24, 28), (90, 86, 96)  # BGR
TG = (227, 165, 38)  # BGR фирменный голубой Telegram
RED_RGB, BLUE_RGB, TG_RGB, GRAY_RGB = (226, 38, 46), (30, 84, 214), (38, 165, 227), (110, 110, 125)

# ---------------------------------------------------------------- субтитры (начало, текст, выделить)
SUBS = [
    (0.20, "ПОЧЕМУ ТЫ ДО СИХ ПОР", ()), (1.10, "ПОМНИШЬ, КАК ОПОЗОРИЛСЯ", ("ОПОЗОРИЛСЯ",)),
    (2.10, "В ШКОЛЕ,", ("ШКОЛЕ",)), (2.70, "НО НЕ ПОМНИШЬ,", ()), (3.34, "ЧТО ЕЛ ВЧЕРА НА ОБЕД?", ("ОБЕД",)),
    (4.60, "", ()),  # заставка «Паучок объясняет» — без субтитров
    (6.20, "МОЗГ НЕ ЗАПИСЫВАЕТ", ("МОЗГ",)), (7.20, "ВСЁ ПОДРЯД", ()), (8.24, "ОН СОХРАНЯЕТ ТОЛЬКО", ()),
    (9.16, "ТО, ЧТО СЧИТАЕТ ВАЖНЫМ", ("ВАЖНЫМ",)), (11.0, "ВЧЕРАШНИЙ ОБЕД —", ()), (11.96, "ОБЫЧНОЕ ДЕЛО", ()),
    (13.08, "МОЗГ РЕШИЛ:", ()), (13.78, "НЕ ПРИГОДИТСЯ —", ()), (14.76, "И СТЁР", ("СТЁР",)),
    (15.86, "А НЕЛОВКИЙ МОМЕНТ —", ()), (17.06, "ЭТО СИГНАЛ ОПАСНОСТИ", ("ОПАСНОСТИ",)),
    (18.66, "ДЛЯ ДРЕВНЕГО ЧЕЛОВЕКА", ()), (19.62, "ОПОЗОРИТЬСЯ ПЕРЕД СВОИМИ", ()),
    (20.84, "ОЗНАЧАЛО РИСК", ()), (21.64, "ОСТАТЬСЯ БЕЗ ПЛЕМЕНИ", ("ПЛЕМЕНИ",)),
    (23.09, "В ТАКИЕ МОМЕНТЫ", ()), (23.95, "В КРОВЬ ВЫБРАСЫВАЕТСЯ", ()), (24.87, "АДРЕНАЛИН", ("АДРЕНАЛИН",)),
    (25.71, "А ОСОБЫЙ ОТДЕЛ МОЗГА —", ()), (27.03, "МИНДАЛЕВИДНОЕ ТЕЛО —", ("МИНДАЛЕВИДНОЕ",)),
    (28.33, "СТАВИТ НА ВОСПОМИНАНИЕ", ()), (29.41, "МЕТКУ:", ()), (30.20, "«ОЧЕНЬ ВАЖНО!»", ("«ОЧЕНЬ", "ВАЖНО!»")),
    (31.73, "И ОНО ЗАПИСЫВАЕТСЯ", ()), (32.81, "НАМЕРТВО", ("НАМЕРТВО",)),
    (33.77, "А ДАЛЬШЕ ТЫ САМ", ()), (34.67, "ЕГО УКРЕПЛЯЕШЬ", ()), (35.85, "КАЖДЫЙ РАЗ, КОГДА", ()),
    (36.65, "ПРОКРУЧИВАЕШЬ ЭТОТ МОМЕНТ", ("ПРОКРУЧИВАЕШЬ",)), (37.77, "В ГОЛОВЕ,", ()),
    (38.55, "ПАМЯТЬ СТАНОВИТСЯ", ()), (39.13, "ТОЛЬКО ЯРЧЕ", ("ЯРЧЕ",)),
    (40.47, "НО ЕСТЬ ХОРОШАЯ НОВОСТЬ", ("НОВОСТЬ",), BLUE_RGB), (41.95, "УЧЁНЫЕ ВЫЯСНИЛИ:", ()),
    (43.20, "ЛЮДЯМ КАЖЕТСЯ,", ()), (44.05, "ЧТО ИХ ПРОМАХ ЗАМЕТИЛА", ()),
    (44.99, "ПОЛОВИНА ОКРУЖАЮЩИХ", ("ПОЛОВИНА",)), (46.30, "А НА САМОМ ДЕЛЕ —", ()),
    (47.05, "МЕНЬШЕ ЧЕТВЕРТИ", ("ЧЕТВЕРТИ",), BLUE_RGB), (48.47, "ЭТО НАЗЫВАЕТСЯ", ()),
    (49.30, "«ЭФФЕКТ ПРОЖЕКТОРА»", ("«ЭФФЕКТ", "ПРОЖЕКТОРА»")),
    (51.20, "А ЕЩЁ БОЛЬШЕ ИНТЕРЕСНОГО —", (), TG_RGB), (52.80, "В МОЁМ ТЕЛЕГРАМ-КАНАЛЕ", ("ТЕЛЕГРАМ-КАНАЛЕ",), TG_RGB),
    (53.88, "«ПАУЧЬЯ НОРКА»", ("«ПАУЧЬЯ", "НОРКА»"), TG_RGB), (55.00, "ТАМ ЭКСКЛЮЗИВ,", ("ЭКСКЛЮЗИВ",), TG_RGB),
    (55.72, "КОТОРОГО НЕТ В РОЛИКАХ", (), TG_RGB), (57.02, "ССЫЛКА В ПРОФИЛЕ —", ("ПРОФИЛЕ",), TG_RGB),
    (58.04, "ЗАЛЕТАЙ!", ("ЗАЛЕТАЙ!",), TG_RGB),
]

# ---------------------------------------------------------------- камера: (время, (облёт°, дистанция, высота, взгляд, обзор°))
CAMERA = [
    (0.0, (-28.0, 5.4, 1.6, 0.85, 50.0)), (0.7, (-12.0, 3.6, 1.15, 0.82, 50.0), "out"),
    (4.45, (-6.0, 3.5, 1.1, 0.82, 50.0)),
    (4.95, (110.0, 3.9, 1.35, 0.98, 50.0), "expo"),            # облёт на «Паучок объясняет»
    (5.95, (116.0, 3.8, 1.3, 0.98, 50.0)), (6.3, (100.0, 3.6, 1.2, 0.82, 50.0), "inout"),
    (15.45, (112.0, 3.8, 1.25, 0.84, 50.0)),
    (15.85, (200.0, 3.6, 1.2, 0.84, 50.0), "expo"), (22.65, (214.0, 3.7, 1.3, 0.86, 50.0)),
    (23.05, (262.0, 3.6, 1.2, 0.86, 50.0), "expo"), (33.35, (282.0, 3.8, 1.25, 0.86, 50.0)),
    (33.75, (350.0, 3.6, 1.2, 0.84, 50.0), "expo"), (40.1, (362.0, 3.7, 1.25, 0.84, 50.0)),
    (40.5, (420.0, 3.6, 1.2, 0.86, 50.0), "expo"), (48.0, (432.0, 3.6, 1.2, 0.86, 50.0)),
    (48.7, (436.0, 3.0, 1.75, 0.7, 50.0), "out"), (50.8, (441.0, 3.1, 1.75, 0.7, 50.0)),  # прожектор: сверху
    (51.2, (524.0, 4.3, 1.3, 1.12, 50.0), "expo"), (DUR, (525.0, 4.0, 1.3, 1.15, 50.0)),
]
OFFSET = [(0.0, 0.5), (4.45, 0.5), (4.95, 0.0, "expo"), (6.0, 0.0), (6.4, 0.55, "inout"),
          (48.2, 0.55), (48.7, 0.0, "out"), (50.8, 0.0), (51.2, 0.0, "expo"), (DUR, 0.0)]
# основной угол каждой сцены — вокруг него расставляются вставки
SEC = {"hook": -9, "brain": 106, "danger": 207, "adren": 272, "replay": 356, "study": 426, "spot": 438, "tg": 527}

# ---------------------------------------------------------------- жесты: L — рука слева на экране, R — справа (к вставкам)
G = lambda *k: list(k)  # noqa: E731
ARM_L = G((0, 62), (4.5, 62), (4.75, -50, "back"), (5.6, -45), (6.0, 62, "back"), (8.2, 62), (8.5, 40, "back"),
          (9.0, 62), (13.0, 62), (13.25, -15, "back"), (14.4, -10), (14.7, 62), (15.9, 62), (16.1, -25, "back"),
          (16.9, -20), (17.2, 55), (20.8, 55), (21.0, 32, "back"), (22.2, 35), (22.5, 62), (24.85, 62),
          (25.05, -45, "back"), (25.6, -40), (25.9, 62), (30.15, 62), (30.3, -50, "back"), (30.9, -45), (31.2, 62),
          (33.8, 62), (34.0, 42, "back"), (35.6, 45), (35.9, 62), (36.6, 62), (36.8, -10, "back"), (38.3, -12),
          (38.6, 62), (40.5, 62), (40.65, -42, "back"), (41.3, -40), (41.6, 62), (47.4, 62), (47.6, 32, "back"),
          (48.3, 35), (48.6, 58), (50.9, 58), (54.9, 62), (55.1, 40, "back"), (56.0, 42), (56.3, 62),
          (58.0, 62), (58.15, -50, "back"), (58.8, -45), (59.1, 62, "back"), (DUR, 62))
ARM_R = G((0, 62), (0.9, 62), (1.1, -30, "back"), (2.4, -28), (2.65, 22, "back"), (3.3, 25), (3.45, 35, "back"),
          (4.45, 35), (4.75, -50, "back"), (5.6, -45), (6.0, 62, "back"), (6.3, 62), (6.5, -12, "back"), (8.0, -10),
          (8.5, 40, "back"), (9.2, 15, "back"), (10.8, 12), (11.0, 25, "back"), (12.9, 25), (13.25, -15, "back"),
          (14.4, -10), (14.7, 30, "back"), (15.5, 32), (15.9, 62), (16.1, -25, "back"), (16.9, -20),
          (17.2, -35, "back"), (18.5, -33), (18.7, -5, "back"), (20.6, -5), (21.0, 32, "back"), (22.2, 35),
          (22.5, 62), (23.1, 62), (23.3, -40, "back"), (24.7, -38), (25.05, -45, "back"), (25.6, -40),
          (25.9, -10, "back"), (28.1, -8), (28.4, 10, "back"), (30.0, 12), (30.3, -50, "back"), (30.9, -45),
          (31.7, -5, "back"), (33.3, -5), (33.6, 62), (34.0, 42, "back"), (35.6, 45), (35.9, -20, "back"),
          (36.5, -18), (36.8, -10, "back"), (38.3, -12), (38.6, -45, "back"), (40.1, -42), (40.3, 62),
          (40.65, -42, "back"), (41.3, -40), (42.0, -15, "back"), (44.7, -15), (44.95, -40, "back"), (46.1, -38),
          (46.35, 15, "back"), (47.4, 15), (47.6, 32, "back"), (48.3, 35), (48.6, 58), (50.9, 58),
          (51.3, -48, "back"), (53.7, -45), (53.95, -55, "back"), (56.8, -52), (57.05, -40, "back"), (57.9, -42),
          (58.15, -50, "back"), (58.8, -45), (58.95, -58), (59.1, -32), (59.25, -58), (59.4, -32), (59.55, -58),
          (59.7, -40), (DUR, 62))
HEAD = G((0, 0), (1.1, -6), (2.6, 6), (3.4, 8), (4.5, 0), (6.3, -5), (9.2, -7), (11.0, -9), (13.2, 3),
         (14.8, -6), (15.9, 0), (16.2, 6), (17.2, -6), (18.7, -7), (21.0, 8), (22.5, 0), (23.3, -6), (25.0, 4),
         (25.9, -6), (28.4, -8), (30.3, 0), (31.7, -6), (33.6, 0), (34.0, 7), (36.0, -5), (38.6, -7), (40.6, 3),
         (42.0, -6), (45.0, -8), (46.4, -9), (47.6, 8), (48.6, 10), (50.9, 6), (51.3, -6), (54.0, -7), (55.1, 5),
         (57.0, -8), (58.2, 0), (59.0, -5), (DUR, 0))
JUMPS = [4.8, 30.3, 40.6, 58.2]


def char_motion(t):
    y, sx, sy = 0.0, 1.0, 1.0
    for j in JUMPS:
        if j < t < j + 0.42:
            p = (t - j) / 0.42
            y = 0.26 * math.sin(math.pi * p)
            sy = 1 + 0.05 * math.sin(math.pi * p)
        elif j + 0.42 <= t < j + 0.56:
            k = math.sin(math.pi * (t - j - 0.42) / 0.14)
            sy, sx = 1 - 0.08 * k, 1 + 0.07 * k
    return y, sx, sy * (1 + 0.008 * math.sin(t * 2 * math.pi / 2.4))


# ---------------------------------------------------------------- геометрия расстановки
def basis(yaw):
    a = math.radians(yaw)
    right = np.array([math.cos(a), 0.0, math.sin(a)])
    tocam = np.array([math.sin(a), 0.0, -math.cos(a)])
    return right, tocam


def place(yaw, side, up, fwd=0.0):
    r, c = basis(yaw)
    return r * side + c * (fwd + 0.22) + np.array([0.0, up, 0.0])  # вставки чуть ближе к камере, чем персонаж


def face(yaw, turn=0.0):
    """Угол карточки, чтобы она смотрела на камеру сцены (turn — доворот для объёма)."""
    return -yaw + turn


def pop(t, t_in, t_out, din=0.45, dout=0.3):
    if t < t_in or t > t_out:
        return 0.0
    return out_back((t - t_in) / din) * (1 - smooth((t - (t_out - dout)) / dout))


# ---------------------------------------------------------------- текстуры вставок
def rounded(size, radius, fill, outline=None, width=0):
    im = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(im).rounded_rectangle([0, 0, size[0] - 1, size[1] - 1], radius, fill=fill,
                                         outline=outline, width=width)
    return im


def fit_font(d, text, max_w, size, tall=True):
    while size > 30:
        f = mg.font(size, tall)
        if d.textbbox((0, 0), text, font=f)[2] <= max_w:
            return f
        size -= 4
    return mg.font(size, tall)


def card(head, big=None, sub=None, accent=RED_RGB, w=780, h=None, faded=False):
    h = h or (150 + (150 if big else 0) + (70 if sub else 0))
    base = (18, 18, 26, 255) if not faded else (28, 28, 34, 255)
    im = rounded((w, h), 34, base, (72, 72, 92, 255), 3)
    d = ImageDraw.Draw(im)
    fh = mg.font(36)
    tw = d.textbbox((0, 0), head, font=fh)[2]
    d.rounded_rectangle([36, 34, 36 + tw + 44, 96], 30, fill=accent + (255,))
    d.text((58, 42), head, font=fh, fill="white")
    y = 116
    if big:
        f = fit_font(d, big, w - 72, 128)
        d.text((36, y), big, font=f, fill=(255, 255, 255) if not faded else (150, 150, 160))
        y += 150
    if sub:
        d.text((40, y), sub, font=mg.font(38), fill=(150, 150, 170))
    return mg.premul(im)


def chip(text, accent):
    f = mg.font(54)
    tmp = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    tw = tmp.textbbox((0, 0), text, font=f)[2]
    im = rounded((tw + 80, 110), 30, (20, 20, 28, 255), accent + (255,), 5)
    ImageDraw.Draw(im).text((40, 22), text, font=f, fill="white")
    return mg.premul(im)


def stamp(text, color=RED_RGB):
    f = mg.font(120, tall=True)
    tmp = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    l, t_, r, b = tmp.textbbox((0, 0), text, font=f)
    im = Image.new("RGBA", (r - l + 90, b - t_ + 80), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([6, 6, im.width - 7, im.height - 7], 22, outline=color + (255,), width=12)
    d.text((45 - l, 40 - t_), text, font=f, fill=color + (255,))
    return mg.premul(im.rotate(-8, expand=True, resample=Image.BICUBIC))


def label(text, color=(255, 255, 255), size=70):
    return mg.premul(mg.text_rgba(text, size, color, tall=True))


def stinger_tex():
    im = rounded((1000, 470), 44, (16, 16, 24, 255), (90, 80, 110, 255), 4)
    d = ImageDraw.Draw(im)
    d.rectangle([60, 400, 330, 414], fill=RED_RGB + (255,))
    d.text((56, 30), "ПАУЧОК", font=mg.font(190, tall=True), fill="white")
    d.text((60, 250), "ОБЪЯСНЯЕТ", font=mg.font(120, tall=True), fill=RED_RGB + (255,))
    return mg.premul(im)


def danger_tex():
    im = rounded((780, 330), 34, (24, 10, 14, 255), (226, 38, 46, 255), 5)
    d = ImageDraw.Draw(im)
    d.polygon([(70, 250), (170, 70), (270, 250)], fill=(226, 38, 46, 255))
    d.rectangle([162, 125, 178, 200], fill="white")
    d.ellipse([160, 212, 180, 232], fill="white")
    d.text((310, 70), "СИГНАЛ", font=mg.font(92, tall=True), fill="white")
    d.text((310, 180), "ОПАСНОСТИ", font=mg.font(84, tall=True), fill=(255, 120, 120))
    return mg.premul(im)


def tg_card_tex():
    """Скриншот канала: увеличен качественно, со скруглёнными углами."""
    src = cv2.imread(os.path.join(ROOT, "assets", "tg_channel.png"))
    big = cv2.resize(src, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_LANCZOS4)
    blur = cv2.GaussianBlur(big, (0, 0), 2.0)
    big = cv2.addWeighted(big, 1.5, blur, -0.5, 0)
    h, w = big.shape[:2]
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w - 1, h - 1], 40, fill=255)
    a = np.asarray(mask, np.float32)[..., None] / 255
    return np.concatenate([big.astype(np.float32) / 255 * a, a], -1).astype(np.float32)


def logo_tex(size=512):
    im = cv2.imread(os.path.join(ROOT, "assets", "logos", "telegram.png"), cv2.IMREAD_UNCHANGED)
    im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    a = im[..., 3:4]
    return np.concatenate([im[..., :3] * a, a], -1).astype(np.float32)


def tg_button(pressed):
    im = rounded((760, 200), 100, (60, 60, 70, 255) if pressed else TG_RGB + (255,), (255, 255, 255, 120), 4)
    d = ImageDraw.Draw(im)
    txt = "ТЫ В НОРКЕ!" if pressed else "ПОДПИСАТЬСЯ"
    f = mg.font(100, tall=True)
    l, t_, r, b = d.textbbox((0, 0), txt, font=f)
    d.text(((760 - (r - l)) / 2 - l, (200 - (b - t_)) / 2 - t_), txt, font=f, fill="white")
    return mg.premul(im)


def tg_glass_card():
    """Карточка канала в стиле INSTA, тёмная: круглая аватарка в голубом ободке, круглый значок Telegram,
    название, подпись; кнопка — отдельная объёмная пилюля перед карточкой."""
    W_, H_ = 920, 600
    im = Image.new("RGBA", (W_, H_), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, W_ - 1, H_ - 1], 56, fill=(22, 24, 36, 196), outline=(80, 90, 120, 255), width=3)
    # мягкий голубой отсвет внутри стекла сверху-слева
    glow = Image.new("RGBA", (W_, H_), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse([-120, -160, 520, 300], fill=TG_RGB + (70,))
    from PIL import ImageFilter
    glow = glow.filter(ImageFilter.GaussianBlur(70))
    mask = Image.new("L", (W_, H_), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, W_ - 1, H_ - 1], 56, fill=255)
    im.paste(Image.alpha_composite(im, glow), (0, 0), mask)
    d = ImageDraw.Draw(im)
    # аватарка
    av = Image.open(os.path.join(ROOT, "output", "avatar.png")).convert("RGBA").resize((190, 190), Image.LANCZOS)
    m = Image.new("L", (190, 190), 0)
    ImageDraw.Draw(m).ellipse([0, 0, 189, 189], fill=255)
    cx = W_ // 2
    d.ellipse([cx - 108, 34, cx + 108, 250], fill=TG_RGB + (255,))
    d.ellipse([cx - 101, 41, cx + 101, 243], fill=(20, 22, 32, 255))
    im.paste(av, (cx - 95, 47), m)
    # круглый значок Telegram на аватарке
    logo = Image.open(os.path.join(ROOT, "assets", "logos", "telegram.png")).convert("RGBA").resize((74, 74), Image.LANCZOS)
    d.ellipse([cx + 50, 172, cx + 132, 254], fill=(20, 22, 32, 255))
    im.alpha_composite(logo, (cx + 54, 176))
    f1 = mg.font(74)
    t1 = "Паучья Норка"
    w1 = d.textbbox((0, 0), t1, font=f1)[2]
    d.text((cx - w1 / 2, 272), t1, font=f1, fill="white")
    f2 = mg.font(40)
    a_, b_ = "Telegram-канал · ", "t.me/pit_of_spider"
    wa, wb = d.textbbox((0, 0), a_, font=f2)[2], d.textbbox((0, 0), b_, font=f2)[2]
    x0 = cx - (wa + wb) / 2
    d.text((x0, 372), a_, font=f2, fill=(150, 155, 175))
    d.text((x0 + wa, 372), b_, font=f2, fill=TG_RGB)
    return mg.premul(im)


def tg_pill(text, pressed=False):
    """Кнопка-пилюля с градиентом и значком Telegram (как в INSTA)."""
    W_, H_ = 640, 150
    im = Image.new("RGBA", (W_, H_), (0, 0, 0, 0))
    grad = Image.new("RGBA", (W_, H_))
    top, mid, bot = ((70, 190, 245), TG_RGB, (31, 147, 209)) if not pressed else ((90, 90, 104), (64, 64, 76), (48, 48, 58))
    gd = ImageDraw.Draw(grad)
    for y in range(H_):
        k = y / H_
        c = [int(top[i] + (mid[i] - top[i]) * k / 0.6) if k < 0.6 else int(mid[i] + (bot[i] - mid[i]) * (k - 0.6) / 0.4) for i in range(3)]
        gd.line([(0, y), (W_, y)], fill=tuple(c) + (255,))
    m = Image.new("L", (W_, H_), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, W_ - 1, H_ - 1], H_ // 2, fill=255)
    im.paste(grad, (0, 0), m)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([18, 6, W_ - 19, 12], 3, fill=(255, 255, 255, 90))  # блик сверху
    # маленькая паутинка в правом верхнем углу кнопки — тематично
    web = Image.new("RGBA", (W_, H_), (0, 0, 0, 0))
    wd = ImageDraw.Draw(web)
    ox, oy, R = W_ - 30, 6, 78
    angs = [math.radians(a) for a in (95, 118, 141, 164, 187)]
    for a in angs:
        wd.line([(ox, oy), (ox + R * math.cos(a), oy + R * math.sin(a))], fill=(255, 255, 255, 120), width=2)
    for rr in (22, 42, 62, 76):
        pts = []
        for a0, a1 in zip(angs, angs[1:]):
            for i in range(7):  # нити между лучами слегка провисают к центру
                q = i / 6
                a = a0 + (a1 - a0) * q
                r = rr * (1 - 0.12 * math.sin(math.pi * q))
                pts.append((ox + r * math.cos(a), oy + r * math.sin(a)))
        wd.line(pts, fill=(255, 255, 255, 105), width=2)
    im.alpha_composite(Image.composite(web, Image.new("RGBA", (W_, H_), (0, 0, 0, 0)), m))
    f = mg.font(62)
    tw = d.textbbox((0, 0), text, font=f)[2]
    icon = 64
    x0 = (W_ - (icon + 22 + tw)) / 2
    # белый круг с голубым самолётиком (значок круглый)
    logo = Image.open(os.path.join(ROOT, "assets", "logos", "telegram.png")).convert("RGBA").resize((icon, icon), Image.LANCZOS)
    arr = np.asarray(logo).copy()
    blue = (arr[..., 2] > 150) & (arr[..., 0] < 120)
    arr[..., :3][blue] = 255
    arr[..., :3][~blue & (arr[..., 3] > 0)] = np.array(mid if not pressed else (64, 64, 76))
    im.alpha_composite(Image.fromarray(arr), (int(x0), (H_ - icon) // 2))
    d.text((x0 + icon + 22, (H_ - 76) / 2), text, font=f, fill="white")
    return mg.premul(im)


def neon(text, color=TG_RGB):
    """Неоновая вывеска: белые буквы с голубым свечением."""
    f = mg.font(130, tall=True)
    tmp = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    l, t_, r, b = tmp.textbbox((0, 0), text, font=f)
    pad = 60
    im = Image.new("RGBA", (r - l + 2 * pad, b - t_ + 2 * pad), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((pad - l, pad - t_), text, font=f, fill=(255, 255, 255, 255))
    a = np.asarray(im, np.float32)[..., 3] / 255
    glow = cv2.GaussianBlur(a, (0, 0), 18) * 1.6 + cv2.GaussianBlur(a, (0, 0), 6) * 0.8
    glow = np.clip(glow, 0, 1)
    col = np.array(color[::-1], np.float32) / 255  # в BGR
    rgb = col * glow[..., None] * (1 - a[..., None]) + a[..., None] * np.array([1.0, 0.97, 0.92])
    al = np.clip(np.maximum(a, glow * 0.85), 0, 1)
    return np.concatenate([rgb * (al[..., None] > 0), al[..., None]], -1).astype(np.float32)


def arrow_tex(color=TG_RGB):
    im = Image.new("RGBA", (220, 300), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.polygon([(110, 10), (210, 130), (145, 130), (145, 290), (75, 290), (75, 130), (10, 130)], fill=color + (255,))
    return mg.premul(im)


def plane_tex():
    im = Image.new("RGBA", (300, 240), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.polygon([(10, 110), (290, 10), (220, 230), (140, 150)], fill=(255, 255, 255, 255))
    d.polygon([(140, 150), (290, 10), (120, 200)], fill=(200, 225, 245, 255))
    return mg.premul(im)


# ---------------------------------------------------------------- сцена
class Scene:
    def __init__(self, voice):
        self.room = mg.Room(os.path.join(ROOT, "assets", "bg_grid.jpg"), radius=9.0, tiles=12)
        self.puppet = mg.Puppet(os.path.join(ROOT, "assets", "char_tpose.jpg"), CHAR)
        p = self.puppet
        self.px_to_m = CHAR_H / (p.feet[1] - p.top)
        self.anchor = (p.feet[0] / p.Wc, p.feet[1] / p.Hc)
        self.env = mg.voice_envelope(voice, FPS)
        T = self.T = {}
        T["school"] = card("ВОСПОМИНАНИЕ", "ПОЗОР В ШКОЛЕ", "5 класс · у доски")
        T["lunch"] = card("ВОСПОМИНАНИЕ", "ОБЕД ВЧЕРА", "что это было?..", GRAY_RGB, faded=True)
        T["stinger"] = stinger_tex()
        T["brain"] = card("МОЗГ", "СОРТИРОВКА", "что сохранить, а что стереть")
        T["chip_school"] = chip("ПОЗОР В ШКОЛЕ", RED_RGB)
        T["chip_lunch"] = chip("ОБЕД ВЧЕРА", GRAY_RGB)
        T["chip_keys"] = chip("ГДЕ КЛЮЧИ?", GRAY_RGB)
        T["imp"] = label("ВАЖНО", (255, 120, 120))
        T["trash"] = label("МУСОР", (170, 170, 185))
        T["erased"] = stamp("СТЁРТО")
        T["danger"] = danger_tex()
        T["tribe"] = label("ПЛЕМЯ", (200, 200, 215))
        T["alone"] = label("ОДИН", (255, 110, 110))
        T["adren"] = card("В КРОВИ", "АДРЕНАЛИН", "сердце колотится, всё запоминается")
        T["amyg"] = card("ОТДЕЛ МОЗГА", "МИНДАЛЕВИДНОЕ ТЕЛО", "отвечает за эмоции и страх")
        T["very"] = stamp("ОЧЕНЬ ВАЖНО!")
        T["forever"] = chip("ЗАПИСАНО НАВСЕГДА", RED_RGB)
        T["bright"] = card("ЯРКОСТЬ", "ВОСПОМИНАНИЯ", h=300)
        T["good"] = card("ХОРОШАЯ НОВОСТЬ", "ВЫДОХНИ", "всё не так страшно", BLUE_RGB)
        T["study"] = card("ИССЛЕДОВАНИЕ", "ГИЛОВИЧ, 2000", "Корнеллский университет, США", BLUE_RGB)
        T["spot"] = card("ПСИХОЛОГИЯ", "ЭФФЕКТ ПРОЖЕКТОРА", "кажется, что все смотрят на тебя", BLUE_RGB)
        T["tg"] = tg_card_tex()
        T["logo"] = logo_tex()
        T["logo_s"] = logo_tex(200)
        T["excl"] = chip("ЭКСКЛЮЗИВ", TG_RGB)
        T["btn"], T["btn2"] = tg_button(False), tg_button(True)
        T["plane"] = plane_tex()
        T["neon"] = neon("ПАУЧЬЯ НОРКА")
        T["tgcard"] = tg_glass_card()
        T["pill"], T["pill2"] = tg_pill("Подписаться"), tg_pill("Ты в норке!", True)
        fing = cv2.imread(os.path.join(ROOT, "assets", "finger_3d.png"), cv2.IMREAD_UNCHANGED)
        fing = cv2.resize(fing, (384, 384), interpolation=cv2.INTER_LANCZOS4).astype(np.float32) / 255
        T["finger"] = np.concatenate([fing[..., :3] * fing[..., 3:], fing[..., 3:]], -1).astype(np.float32)
        T["excl"] = stamp("ЭКСКЛЮЗИВ", TG_RGB)
        T["arrow"] = arrow_tex()
        self.txt = {}

    def text(self, s, size=90, color=(255, 255, 255)):
        k = (s, size, color)
        if k not in self.txt:
            self.txt[k] = label(s, color, size)
        return self.txt[k]

    def cam(self, t):
        yaw, dist, hgt, ty, fov = track(CAMERA, t)
        off = track(OFFSET, t)
        a = math.radians(yaw)
        shake = 0.0
        for s0 in (4.95, 30.2):  # удар камеры на заставке и на штампе
            if s0 < t < s0 + 0.35:
                shake = (1 - (t - s0) / 0.35) * math.sin((t - s0) * 70)
        return mg.Cam(yaw + 0.5 * wobble(t, 1) + 1.2 * shake, dist, hgt + 0.02 * wobble(t, 2) + 0.03 * shake,
                      ty + 0.015 * wobble(t, 3), fov=fov, roll=0.5 * wobble(t * 0.8, 4) + 1.5 * shake,
                      tx=off * math.cos(a), tz=off * math.sin(a))

    # --- обёртки, чтобы складывать объекты в список с глубиной
    def add_panel(self, items, cam, pos, yaw, w, h, th, tex, **kw):
        w, h, th = w * 1.25, h * 1.25, th * 1.25  # крупнее — текст на карточках читается
        items.append((cam.depth(pos), lambda: mg.panel(self.frame, cam, pos, yaw, w, h, th, tex, **kw)))

    def add_bb(self, items, cam, pos, height, tex, **kw):
        height *= 1.3
        items.append((cam.depth(pos), lambda: mg.billboard(self.frame, cam, pos, height, tex, **kw)))

    def add_box(self, items, cam, pos2, size, hgt, col, **kw):
        cen = np.array([pos2[0], hgt / 2 + kw.get("base_y", 0.0), pos2[1]])
        items.append((cam.depth(cen), lambda: mg.box(self.frame, cam, pos2, size, hgt, col, **kw)))

    def draw(self, t):
        cam = self.cam(t)
        frame = self.frame = self.room.render(cam)
        T = self.T
        # окраска комнаты: тревога (красная пульсация) и концовка (голубой Telegram)
        alarm = pop(t, 15.9, 22.8, 0.2, 0.5) * (0.55 + 0.45 * math.sin(t * 9))
        tgk = smooth((t - 51.1) / 0.6)
        spot = pop(t, 48.45, 50.95, 0.35, 0.3)
        f = frame.astype(np.float32)
        if alarm > 0:
            f *= np.array([1 - 0.35 * alarm, 1 - 0.35 * alarm, 1 + 0.6 * alarm], np.float32)
        if tgk > 0:
            f *= np.array([1 + 0.25 * tgk, 1 + 0.08 * tgk, 1 - 0.1 * tgk], np.float32)
        if spot > 0:
            f *= 1 - 0.5 * spot
        frame[...] = np.clip(f, 0, 255).astype(np.uint8)

        glow_col = (60, 50, 210) if tgk < 0.5 else (220, 150, 40)
        mg.floor_blob(frame, cam, (0, 0), 1.1, 1.1, 0.2 * (1 - spot), color=glow_col, blur=151, mode="light")
        mg.floor_grid(frame, cam, step=0.75, extent=9, base=110, fade=11, radius=8.9, mask=self.room.floor_mask)

        items = []
        self.items_for(t, cam, items)

        # прожектор: луч сверху и яркое пятно на полу
        if spot > 0:
            a = np.linspace(0, 2 * math.pi, 40)
            ring = np.stack([0.55 * np.cos(a), np.zeros_like(a), 0.55 * np.sin(a)], -1)
            uv, z = cam.proj(np.vstack([ring, [[0, 3.4, 0]]]))
            if (z > 0.3).all():
                lay = np.zeros((H, W), np.float32)
                mg.poly(lay, cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2), 1.0)
                lay = cv2.GaussianBlur(lay, (0, 0), 18) * 0.22 * spot
                frame[...] = np.clip(frame.astype(np.float32) + lay[..., None] * np.array([235, 245, 255]), 0, 255).astype(np.uint8)
            mg.floor_blob(frame, cam, (0, 0), 0.6, 0.6, 0.7 * spot, color=(230, 240, 255), blur=61, mode="light")

        # персонаж
        y, sx, sy = char_motion(t)
        cx, cz, vis = 0.0, 0.0, 1.0
        env = self.env[min(len(self.env) - 1, int(t * FPS))]
        p = self.puppet
        canvas = p.render(armL=track(ARM_L, t) + 3 * wobble(t * 1.7, 11), armR=track(ARM_R, t) + 3 * wobble(t * 1.5, 12),
                          head=track(HEAD, t) + 2.5 * wobble(t * 1.3, 7) + 3 * env * math.sin(t * 9),
                          jaw=clamp(env * 1.3), head_dy=-8 * env)
        # персонаж разворачивается к камере с небольшим запозданием, но не больше чем на 30° —
        # иначе на резком облёте он на миг становится «ребром» (тонкой полоской)
        now = math.degrees(cam.yaw)
        lagd = math.degrees(self.cam(max(0.0, t - 0.12)).yaw)
        facing = now + max(-30.0, min(30.0, lagd - now))
        hgt_m = p.Hc * self.px_to_m
        sk = clamp(1 - y / 1.5)
        if vis > 0.02:
            mg.floor_blob(frame, cam, (cx, cz), 0.34 * (0.6 + 0.4 * sk) * vis, 0.22 * (0.6 + 0.4 * sk) * vis,
                          0.6 * sk * vis, blur=51)
            self.char_args = ((cx, y, cz), hgt_m * vis, canvas, facing, (sx, sy), cam)
            items.append((-1e9 if getattr(self, "char_last", False) else cam.depth((cx, 0.6, cz)), lambda: mg.billboard(
                frame, cam, (cx, y, cz), hgt_m * vis, canvas, facing_yaw=facing, anchor=self.anchor,
                squash=(sx, sy))))
        for _, fn in sorted(items, key=lambda x: -x[0]):
            fn()
        mg.subtitles(frame, t, SUBS)
        return frame

    # ------------------------------------------------------------ вставки по сценам
    def items_for(self, t, cam, it):
        T, fr = self.T, self.frame
        sh = lambda pos2, rx, rz, k: mg.floor_blob(fr, cam, pos2, rx * k, rz * k, 0.45 * k, blur=51)  # noqa: E731

        # 1. Крючок: яркое воспоминание и бледное, которое рассыпается
        y0 = SEC["hook"]
        k = pop(t, 0.95, 4.55)
        if k > 0:
            pos = place(y0, 0.86, 1.36) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 24 - 20 * (1 - k)), 0.9 * k, 0.53 * k, 0.06 * k, T["school"],
                           pitch=-4, roll=-3, side=(60, 50, 90), edge=(220, 200, 255))
        k = pop(t, 2.75, 4.55)
        if k > 0:
            fade = 1 - 0.75 * smooth((t - 3.9) / 0.5)
            flick = 0.7 + 0.3 * math.sin(t * 40) if t > 3.85 else 1.0
            pos = place(y0, 0.9, 0.66) + [0, 0.02 * math.sin(t * 2.4), 0]
            self.add_panel(it, cam, pos, face(y0, 18), 0.8 * k, 0.47 * k, 0.04 * k, T["lunch"] * (fade * flick),
                           pitch=-6, roll=2, side=(40, 40, 46))

        # 2. Заставка «Паучок объясняет»
        k = pop(t, 4.65, 6.15, 0.4, 0.3)
        if k > 0:
            pos = place(118, 0.0, 1.95, 0.35)
            self.add_panel(it, cam, pos, face(118, -8), 1.35 * k, 0.635 * k, 0.12 * k, T["stinger"], pitch=-6,
                           side=(30, 24, 120), edge=(200, 190, 255))

        # 3. Мозг сортирует: «важно» и «мусор»
        y0 = SEC["brain"]
        k = pop(t, 6.3, 15.6)
        if k > 0:
            pos = place(y0, 0.78, 1.72) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 14), 1.0 * k, 0.48 * k, 0.06 * k, T["brain"], pitch=-6,
                           side=(60, 50, 90), edge=(220, 200, 255))
        kb = pop(t, 8.3, 15.6)
        bins = {"imp": place(y0, 0.66, 0, 0.2), "trash": place(y0, 1.22, 0, 0.05)}
        if kb > 0:
            for name, col, edge in (("imp", (40, 30, 120), (90, 80, 240)), ("trash", (40, 36, 42), (90, 86, 96))):
                b = bins[name]
                hh = 0.3 * kb
                if name == "trash" and 14.75 < t < 15.0:
                    hh *= 1 + 0.25 * math.sin(math.pi * (t - 14.75) / 0.25)
                self.add_box(it, cam, (b[0], b[2]), (0.42, 0.3), hh, col, edge=edge, yaw=-y0,
                             glow=(40, 30, 200) if name == "imp" else None)
                self.add_bb(it, cam, b + [0, hh + 0.05, 0], 0.11 * kb, T[name], anchor=(0.5, 1.0))
        for name, t0, dest in (("chip_school", 9.2, "imp"), ("chip_lunch", 11.0, "trash"), ("chip_keys", 12.4, "trash")):
            if t0 < t < t0 + 1.2:
                p = clamp((t - t0) / 0.8)
                start = place(y0, 0.95, 1.45)
                end = bins[dest] + [0, 0.34, 0]
                pos = start * (1 - p) + end * p + [0, 0.45 * math.sin(math.pi * p), 0]
                s = (1 - smooth((t - t0 - 0.85) / 0.3)) * out_back(clamp((t - t0) / 0.25))
                tex = T[name]
                self.add_bb(it, cam, pos, 0.13 * s, tex, anchor=(0.5, 0.5))
        k = pop(t, 14.78, 15.6, 0.25, 0.25)
        if k > 0:
            self.add_bb(it, cam, bins["trash"] + [0, 0.62, 0.05], 0.22 * k, T["erased"], anchor=(0.5, 0.5))

        # 4. Сигнал опасности и племя
        y0 = SEC["danger"]
        k = pop(t, 16.95, 22.9)
        if k > 0:
            pos = place(y0, 0.78, 1.7) + [0, 0.03 * math.sin(t * 3), 0]
            self.add_panel(it, cam, pos, face(y0, 14), 0.95 * k, 0.4 * k, 0.06 * k, T["danger"], pitch=-6,
                           side=(30, 20, 110), edge=(150, 140, 255))
        kt = pop(t, 18.65, 22.9)
        if kt > 0:
            away = smooth((t - 21.6) / 0.7)
            for i in range(6):
                gx = 0.82 + 0.2 * (i % 3) + 0.45 * away
                gz = -0.15 + 0.22 * (i // 3)
                pp = place(y0, gx, 0, gz)
                g = out_back(clamp((t - 18.65 - 0.06 * i) / 0.4))
                self.add_box(it, cam, (pp[0], pp[2]), (0.12, 0.1), 0.2 * g * kt, (60, 54, 64), edge=(110, 100, 120), yaw=-y0)
                self.add_box(it, cam, (pp[0], pp[2]), (0.1, 0.1), 0.1 * g * kt, (80, 74, 86), edge=(120, 110, 130),
                             yaw=-y0, base_y=0.22 * g * kt)
            me = place(y0, 0.5 - 0.05 * away, 0, 0.0)
            g = out_back(clamp((t - 18.9) / 0.4)) * kt
            self.add_box(it, cam, (me[0], me[2]), (0.12, 0.1), 0.2 * g, (40, 30, 170), edge=(80, 70, 255), yaw=-y0)
            self.add_box(it, cam, (me[0], me[2]), (0.1, 0.1), 0.1 * g, (50, 40, 200), edge=(90, 80, 255), yaw=-y0, base_y=0.22 * g)
            self.add_bb(it, cam, place(y0, 1.05 + 0.45 * away, 0.62), 0.1 * kt, T["tribe"], anchor=(0.5, 1.0))
            ka = pop(t, 21.7, 22.9, 0.3, 0.3)
            if ka > 0:
                self.add_bb(it, cam, me + [0, 0.5, 0], 0.1 * ka, T["alone"], anchor=(0.5, 1.0))

        # 5. Адреналин, миндалевидное тело, штамп «ОЧЕНЬ ВАЖНО!»
        y0 = SEC["adren"]
        k = pop(t, 23.95, 26.0)
        if k > 0:
            pos = place(y0, 0.8, 1.62) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 14), 1.0 * k, 0.48 * k, 0.06 * k, T["adren"], pitch=-6,
                           side=(60, 50, 90), edge=(220, 200, 255))
            it.append((cam.depth(pos) - 0.3, lambda: self.ekg(cam, place(y0, 0.8, 1.05), y0, t, k)))
        k = pop(t, 25.75, 28.4)
        if k > 0:
            pos = place(y0, 0.8, 1.62) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 14), 1.0 * k, 0.48 * k, 0.06 * k, T["amyg"], pitch=-6,
                           side=(60, 50, 90), edge=(220, 200, 255))
        k = pop(t, 28.35, 33.45)
        if k > 0:
            thick = 0.06 + 0.16 * smooth((t - 32.8) / 0.4)
            pos = place(y0, 0.9, 1.32) + [0, 0.03 * math.sin(t * 2), 0]
            if 30.2 < t < 30.45:
                pos = pos + [0, -0.05 * math.sin(math.pi * (t - 30.2) / 0.25), 0]
            self.add_panel(it, cam, pos, face(y0, 18), 0.95 * k, 0.56 * k, thick * k, T["school"], pitch=-5,
                           side=(40, 30, 140), edge=(220, 200, 255))
            ks = clamp((t - 30.15) / 0.18)
            if ks > 0:
                s = (1.0 + 0.6 * (1 - ks)) * k
                self.add_bb(it, cam, pos + place(y0, 0.05, -0.02, 0.12), 0.19 * s, T["very"], anchor=(0.5, 0.5))
            kf = pop(t, 32.85, 33.45, 0.35, 0.25)
            if kf > 0:
                self.add_bb(it, cam, place(y0, 1.0, 0.8, 0.1), 0.11 * kf, T["forever"], anchor=(0.5, 0.5))

        # 6. Прокручиваешь момент — кольцо вокруг персонажа и растущие столбики «яркости»
        y0 = SEC["replay"]
        kr = pop(t, 36.55, 38.6, 0.3, 0.3)
        if kr > 0:
            it.append((cam.depth((0, 0, 0)) + 0.6, lambda: self.loop_ring(cam, t, kr)))
        kb = pop(t, 37.6, 40.35)
        if kb > 0:
            pos = place(y0, 0.92, 1.75) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 12), 0.85 * kb, 0.33 * kb, 0.05 * kb, T["bright"], pitch=-6,
                           side=(60, 50, 90), edge=(220, 200, 255))
            for i, (name, hh) in enumerate((("1 РАЗ", 0.3), ("5 РАЗ", 0.7), ("20 РАЗ", 1.15))):
                g = out_back(clamp((t - 38.3 - 0.25 * i) / 0.5), s=1.2) * kb
                if g <= 0:
                    continue
                c = place(y0, 0.55 + 0.3 * i, 0, 0.1)
                col = tuple(int(v) for v in np.array(RED) * (0.55 + 0.25 * i))
                self.add_box(it, cam, (c[0], c[2]), (0.22, 0.22), hh * g, col, edge=(20, 20, 24), yaw=-y0,
                             glow=(40, 30, 220) if i == 2 else None)
                self.add_bb(it, cam, c + [0, hh * g + 0.04, 0], 0.075 * g, self.text(name, 70, (210, 210, 225)), anchor=(0.5, 1.0))

        # 7. Хорошая новость, исследование, столбики 50% / 23%, эффект прожектора
        y0 = SEC["study"]
        for key, a, b in (("good", 40.55, 42.0), ("study", 41.95, 43.3)):
            k = pop(t, a, b, 0.4, 0.25)
            if k > 0:
                pos = place(y0, 0.8, 1.6) + [0, 0.03 * math.sin(t * 2), 0]
                self.add_panel(it, cam, pos, face(y0, 14), 1.0 * k, 0.48 * k, 0.06 * k, T[key], pitch=-6,
                               side=(90, 50, 40), edge=(255, 210, 190))
        kb = pop(t, 43.3, 48.45)
        if kb > 0:
            for i, (name, val, hh, col, t0) in enumerate((("КАЖЕТСЯ", 50, 1.0, RED, 44.9), ("НА САМОМ ДЕЛЕ", 23, 0.46, BLUE, 47.0))):
                g = out_back(clamp((t - t0) / 0.6), s=1.2) * kb
                c = place(y0, 0.62 + 0.42 * i, 0, 0.1)
                if g <= 0:
                    continue
                self.add_box(it, cam, (c[0], c[2]), (0.3, 0.3), hh * g, col, edge=(20, 20, 24), yaw=-y0)
                num = int(val * clamp((t - t0) / 0.8))
                self.add_bb(it, cam, c + [0, hh * g + 0.04, 0], 0.13 * g, self.text(f"{num}%", 90), anchor=(0.5, 1.0))
                self.add_bb(it, cam, c + [0, hh * g + 0.2, 0], 0.065 * g, self.text(name, 60, (190, 190, 210)), anchor=(0.5, 1.0))
        k = pop(t, 49.3, 50.95, 0.4, 0.25)
        if k > 0:
            pos = place(SEC["spot"], 0.0, 1.5, 0.3)
            self.add_panel(it, cam, pos, face(SEC["spot"], 0), 0.84 * k, 0.41 * k, 0.07 * k, T["spot"], pitch=-18,
                           side=(90, 50, 40), edge=(255, 210, 190))

        # 8. Концовка как в INSTA, тёмная: крупная стеклянная карточка лицом к зрителю по центру, персонаж под ней
        y0 = SEC["tg"]
        k = clamp((t - 51.35) / 0.8)
        if k > 0:
            e = out_back(k, s=1.25)
            r_, c_ = basis(y0)
            s_ = 0.85 + 0.15 * e
            card = place(y0, 0.0, 1.95, -0.22) + [0, 0.9 * (1 - e) + 0.015 * math.sin(t * 1.5), 0]
            tilt = 3 + 2 * math.sin(t * 1.5)
            cw, ch = 1.4 * s_, 0.913 * s_   # (до общего увеличения ×1.25)
            it.append((cam.depth(card) + 0.3, lambda card=card, e=e: self.glow_behind(cam, card, y0, e)))
            self.add_panel(it, cam, card, face(y0, 0), cw, ch, 0.04 * s_, T["tgcard"],
                           pitch=-tilt, side=(70, 60, 50), edge=(255, 225, 180), radius=cw * 0.061)
            tap = math.sin(math.pi * (t - 58.0) / 0.3) if 58.0 < t < 58.3 else 0.0
            pulse = 1 + 0.03 * math.sin((t - 52.8) * 6) * (1 if t > 52.8 else 0)
            # кнопка во всю нижнюю часть карточки (как в образце), чуть впереди стекла
            bpos = card + c_ * 0.05 + [0, -0.375 * s_, 0]
            sb = s_ * pulse * (1 - 0.07 * tap)
            tex = T["pill2"] if t > 58.15 else T["pill"]
            self.add_panel(it, cam, bpos, face(y0, 0), 0.9 * sb, 0.211 * sb, 0.05 * sb, tex, pitch=-tilt,
                           side=(150, 100, 20) if t <= 58.15 else (50, 50, 60), radius=0.211 * sb / 2)
            ke = pop(t, 55.05, DUR + 1, 0.3)
            if ke > 0:  # штамп «ЭКСКЛЮЗИВ» на углу карточки
                self.add_bb(it, cam, card + r_ * 0.7 + c_ * 0.08 + [0, 0.46, 0],
                            0.1 * ke * (1 + 0.8 * (1 - clamp((t - 55.05) / 0.15))), T["excl"], anchor=(0.5, 0.5))
            kf = clamp((t - 57.1) / 0.6)  # палец подъезжает к правому краю кнопки и нажимает
            if kf > 0:
                ef = out_back(kf, s=1.1)
                fpos = bpos + r_ * (0.45 + 0.35 * (1 - ef)) + c_ * 0.1 + [0, -0.07 - 0.35 * (1 - ef) + 0.05 * tap, 0]
                self.add_bb(it, cam, fpos, 0.15 * ef, T["finger"], anchor=(0.35, 0.12))

    def glow_behind(self, cam, card, yaw, e):
        """Мягкие пятна света за стеклянной карточкой: голубое сверху-слева, тёплое снизу-справа."""
        r_, _ = basis(yaw)
        for off, col, rad in ((r_ * -0.62 + [0, 0.3, 0], TG, 0.55), (r_ * 0.62 + [0, -0.3, 0], (120, 160, 255), 0.5)):
            uv, z = cam.proj(np.array([card + off]))
            if z[0] < 0.3:
                continue
            pr = rad * cam.focal / z[0]
            lay = np.zeros((H, W), np.float32)
            cv2.circle(lay, (int(uv[0][0]), int(uv[0][1])), int(pr), 1.0, -1)
            lay = cv2.GaussianBlur(lay, (0, 0), pr * 0.5 + 1) * 0.45 * e
            self.frame[...] = np.clip(self.frame.astype(np.float32) + lay[..., None] * np.array(col, np.float32), 0, 255).astype(np.uint8)

    # ------------------------------------------------------------ особые эффекты
    def ekg(self, cam, center, yaw, t, k):
        """Кардиограмма, которая рисуется слева направо и светится."""
        r, _ = basis(yaw)
        xs = np.linspace(-0.55, 0.55, 160)
        ys = np.zeros_like(xs)
        for c in (-0.3, 0.12, 0.45):
            ys += 0.22 * np.exp(-((xs - c) / 0.018) ** 2) - 0.09 * np.exp(-((xs - c - 0.04) / 0.02) ** 2)
        prog = clamp((t - 24.0) / 1.2)
        n = max(2, int(len(xs) * prog))
        P = np.array([center + r * x * k + np.array([0, y * k, 0]) for x, y in zip(xs[:n], ys[:n])])
        uv, z = cam.proj(P)
        if (z < 0.3).any():
            return
        lay = np.zeros_like(self.frame)
        mg.lines(lay, uv, (60, 50, 255), 10, closed=False)
        lay = cv2.GaussianBlur(lay, (0, 0), 6)
        mg.lines(lay, uv, (210, 210, 255), 3, closed=False)
        cv2.add(self.frame, lay, self.frame)

    def loop_ring(self, cam, t, k):
        """Пунктирное кольцо на полу вокруг персонажа — «прокручиваешь в голове»."""
        lay = np.zeros_like(self.frame)
        for i in range(18):
            a0 = t * 2.4 + i * 2 * math.pi / 18
            a = np.linspace(a0, a0 + 0.22, 6)
            P = np.stack([0.62 * np.cos(a), np.full_like(a, 0.02), 0.62 * np.sin(a)], -1)
            uv, z = cam.proj(P)
            if (z > 0.3).all():
                mg.lines(lay, uv, (int(70 * k), int(60 * k), int(255 * k)), 6, closed=False)
        g = cv2.GaussianBlur(lay, (0, 0), 5)
        cv2.add(self.frame, g, self.frame)
        cv2.add(self.frame, lay, self.frame)

    def plane(self, cam, yaw, t):
        def at(tt):
            p = clamp((tt - 55.7) / 1.6)
            return place(yaw, -0.4 + 2.2 * p, 0.9 + 1.0 * p - 0.5 * math.sin(math.pi * p), 0.6 - 0.4 * p)
        trail = [at(tt) for tt in np.linspace(max(55.7, t - 0.8), t, 14)]
        uv, z = cam.proj(np.array(trail))
        if (z > 0.3).all():
            for i in range(0, len(uv) - 1, 2):
                mg.lines(self.frame, uv[i:i + 2], (240, 220, 200), 3, closed=False)
        mg.billboard(self.frame, cam, at(t), 0.16, self.T["plane"], anchor=(0.5, 0.5))

    def finish(self, frame, t):
        f = frame.astype(np.float32)
        flash = 0.0
        for s0, amp in ((4.7, 0.18), (15.6, 0.1), (22.8, 0.1), (33.5, 0.1), (40.2, 0.1), (50.95, 0.12)):
            if s0 < t < s0 + 0.25:
                flash = max(flash, math.sin(math.pi * (t - s0) / 0.25) * amp)
        if flash:
            f += flash * 255
        f *= VIGNETTE[..., None]
        f *= clamp(t / 0.25) * clamp((DUR - t) / 0.35)
        return np.clip(f, 0, 255).astype(np.uint8)


SFX = [(0.0, "whoosh"), (0.95, "pop"), (2.75, "pop"), (3.9, "whoosh"), (4.55, "whoosh"), (4.85, "boom"),
       (6.3, "pop"), (8.3, "pop"), (9.2, "whoosh"), (11.0, "whoosh"), (12.4, "whoosh"), (14.78, "click"),
       (15.5, "whoosh"), (16.0, "rumble"), (16.95, "pop"), (18.65, "pop"), (21.6, "whoosh"), (22.7, "whoosh"),
       (23.95, "pop"), (25.75, "pop"), (28.35, "pop"), (30.15, "boom"), (32.85, "pop"), (33.4, "whoosh"),
       (36.55, "whoosh"), (37.6, "pop"), (38.3, "pop"), (38.55, "pop"), (38.8, "pop"), (40.15, "whoosh"),
       (40.55, "pop"), (43.3, "pop"), (44.9, "pop"), (47.0, "pop"), (48.4, "rumble"), (49.3, "pop"),
       (50.9, "whoosh"), (51.4, "whoosh"), (55.05, "pop"), (57.1, "whoosh"), (58.05, "click")]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    voice, out = args[0], args[1]
    only = None
    if "--preview" in sys.argv:
        only = [int(x * FPS) for x in (1.5, 3.2, 5.2, 9.6, 11.6, 14.9, 17.5, 21.9, 24.8, 27.0, 30.4, 33.0, 37.2,
                                       39.6, 41.0, 46.0, 49.8, 52.0, 54.5, 56.2, 57.6, 58.6)]
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
    fx = mg.sfx(out + ".sfx.wav", SFX, DUR)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", tmp, "-i", voice, "-i", fx,
                    "-filter_complex", "[1:a]volume=1.0[v];[2:a]volume=0.4[s];[v][s]amix=inputs=2:normalize=0[a]",
                    "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", out],
                   check=True)
    os.remove(tmp)
    os.remove(fx)


if __name__ == "__main__":
    main()
