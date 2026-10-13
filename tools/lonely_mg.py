"""Ролик №4: «Почему вокруг миллионы людей, а поговорить не с кем?» (одиночество в большом городе).

python tools/lonely_mg.py work/lonely/voice_clean.wav output/lonely.mp4 [--preview] [--jobs 4]
Общий движок, персонаж и концовка с Telegram — из memory_mg.py; параллельный рендер — как в money_mg.py.
Новое в монтаже: начало крупным планом и отъезд, за которым открывается толпа с телефонами;
заставка с полным оборотом камеры вокруг паучка; боковое скольжение камеры вдоль вставок;
«город» сверху; плавное приближение; цвет от холодного синего (одиночество) к тёплому (дружба).
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
from memory_mg import (BLUE_RGB, GRAY_RGB, RED_RGB, TG, TG_RGB, basis, card, chip, face, place, pop,  # noqa: E402,F401
                       rounded, stamp)
from mg import H, W, clamp, out_back, smooth, track, wobble  # noqa: E402

FPS = 30
ROOT = mg.ROOT
OUTRO = float(open(os.path.join(ROOT, "work", "lonely", "outro_start.txt")).read())  # 40.34
DUR = 49.0
WARM_RGB, GREEN_RGB = (240, 160, 70), (46, 196, 110)
WARM, GREEN = WARM_RGB[::-1], GREEN_RGB[::-1]  # BGR

# ---------------------------------------------------------------- субтитры
T_, B_, W_ = TG_RGB, BLUE_RGB, WARM_RGB
SUBS = [
    (0.226, "ПОЧЕМУ ВОКРУГ", ()), (0.846, "МИЛЛИОНЫ ЛЮДЕЙ,", ("МИЛЛИОНЫ",), B_),
    (1.986, "А ПОГОВОРИТЬ НЕ С КЕМ?", ("НЕ", "С", "КЕМ?")), (3.15, "", ()),
    (4.906, "ОДИНОЧЕСТВО ИСПЫТЫВАЕТ", ("ОДИНОЧЕСТВО",), B_), (5.946, "КАЖДЫЙ ШЕСТОЙ", ("ШЕСТОЙ",), B_),
    (6.606, "ЧЕЛОВЕК НА ПЛАНЕТЕ", ()), (7.946, "И ДЛЯ ЗДОРОВЬЯ ЭТО ВРЕДНО", ("ВРЕДНО",)),
    (9.586, "ПОЧТИ КАК КУРЕНИЕ", ("КУРЕНИЕ",)), (10.966, "В ГОРОДЕ МЫ РАБОТАЕМ", ()),
    (11.586, "ИЗ ДОМА", ("ДОМА",), B_), (12.666, "И НЕ ЗНАЕМ СОСЕДЕЙ", ("СОСЕДЕЙ",)),
    (13.886, "А МОЗГ НЕ СЧИТАЕТ ЛАЙК", ("ЛАЙК",)), (15.066, "ЖИВЫМ РАЗГОВОРОМ", ("РАЗГОВОРОМ",), B_),
    (16.401, "УЧЁНЫЕ ПОСЧИТАЛИ:", ()), (17.621, "ЧТОБЫ ЗНАКОМЫЙ", ("ЗНАКОМЫЙ",), W_),
    (18.341, "СТАЛ ПРИЯТЕЛЕМ,", ("ПРИЯТЕЛЕМ",), W_), (19.241, "НУЖНО ОКОЛО 50 ЧАСОВ", ("50", "ЧАСОВ"), W_),
    (20.441, "ОБЩЕНИЯ", ()), (21.301, "А ЧТОБЫ БЛИЗКИМ ДРУГОМ —", ("ДРУГОМ",), W_),
    (22.441, "БОЛЬШЕ 200", ("200",), W_), (23.701, "А ЕЩЁ ПОСЛЕ РАЗГОВОРА", ()),
    (24.801, "НАМ КАЖЕТСЯ,", ()), (25.361, "ЧТО МЫ ПОНРАВИЛИСЬ МЕНЬШЕ,", ("МЕНЬШЕ",)),
    (26.441, "ЧЕМ НА САМОМ ДЕЛЕ", ("НА", "САМОМ", "ДЕЛЕ"), B_), (27.481, "ПОЭТОМУ МЫ БОИМСЯ", ("БОИМСЯ",)),
    (28.401, "ПИСАТЬ ПЕРВЫМИ", ("ПЕРВЫМИ",)), (29.705, "КАК БЫТЬ?", ("БЫТЬ?",), W_),
    (30.745, "ХОДИ РЕГУЛЯРНО В ОДНО МЕСТО:", ("РЕГУЛЯРНО",), W_), (32.385, "СЕКЦИЯ, КУРСЫ,", ()),
    (33.745, "КЛУБ ПО ИНТЕРЕСАМ", ("КЛУБ",), W_), (34.845, "ДРУЖБА РОЖДАЕТСЯ", ("ДРУЖБА",), W_),
    (35.745, "ИЗ ПОВТОРЯЮЩИХСЯ ВСТРЕЧ", ("ПОВТОРЯЮЩИХСЯ",), W_), (37.285, "И ПИШИ ПЕРВЫМ —", ("ПЕРВЫМ",), W_),
    (38.185, "ТЕБЕ РАДЫ БОЛЬШЕ,", ("РАДЫ",), W_), (39.045, "ЧЕМ КАЖЕТСЯ", (), W_),
    (40.31, "А ЕЩЁ БОЛЬШЕ ИНТЕРЕСНОГО —", (), T_), (41.67, "В МОЁМ ТЕЛЕГРАМ-КАНАЛЕ", ("ТЕЛЕГРАМ-КАНАЛЕ",), T_),
    (43.0, "«ПАУЧЬЯ НОРКА»", ("«ПАУЧЬЯ", "НОРКА»"), T_), (44.12, "ТАМ ЭКСКЛЮЗИВ,", ("ЭКСКЛЮЗИВ",), T_),
    (44.82, "КОТОРОГО НЕТ В РОЛИКАХ", (), T_), (46.12, "ССЫЛКА В ПРОФИЛЕ —", ("ПРОФИЛЕ",), T_),
    (47.14, "ЗАЛЕТАЙ!", ("ЗАЛЕТАЙ!",), T_),
]

# ---------------------------------------------------------------- камера и сцены
# (yaw, dist, height, target_y, fov)
CAMERA = [
    (0.0, (0.0, 1.75, 1.05, 0.95, 40.0)), (0.25, (0.0, 1.75, 1.05, 0.95, 40.0)),  # крупный план…
    (1.7, (8.0, 5.6, 2.7, 0.7, 50.0), "inout"),            # …отъезд: вокруг толпа
    (3.2, (12.0, 5.4, 2.6, 0.75, 50.0)),
    (4.35, (360.0, 4.2, 1.3, 0.98, 50.0), "expo"),         # заставка: полный оборот вокруг паучка
    (4.8, (361.0, 4.2, 1.3, 0.98, 50.0)),
    (5.2, (455.0, 3.6, 1.2, 0.84, 50.0), "expo"),          # каждый шестой
    (10.6, (460.0, 3.6, 1.2, 0.84, 50.0)),
    (11.0, (545.0, 4.4, 2.9, 0.7, 50.0), "expo"),          # город — сверху
    (16.0, (552.0, 4.1, 2.3, 0.78, 50.0)),
    (16.4, (635.0, 3.6, 1.15, 0.84, 50.0), "expo"),        # часы до дружбы — скольжение вдоль ступенек
    (23.25, (640.0, 3.7, 1.35, 0.9, 50.0)),
    (23.65, (725.0, 3.6, 1.2, 0.86, 50.0), "expo"),        # «понравились меньше» — плавное приближение
    (29.3, (728.0, 3.6, 1.2, 0.88, 44.0), "inout"),
    (29.7, (815.0, 3.6, 1.2, 0.84, 50.0), "expo"),         # как быть — медленный облёт
    (39.9, (833.0, 3.7, 1.25, 0.86, 50.0), "lin"),
    (40.3, (900.0, 4.3, 1.3, 1.12, 50.0), "expo"), (DUR, (901.0, 4.0, 1.3, 1.15, 50.0)),
]
# >0 — персонаж слева, вставки справа; <0 — наоборот
OFFSET = [(0.0, 0.0), (4.8, 0.0), (5.2, 0.68, "expo"), (10.6, 0.45), (11.0, -0.55, "expo"), (16.0, -0.55),
          (16.4, 0.42, "expo"), (23.25, 0.7), (23.65, -0.55, "expo"), (29.3, -0.5), (29.7, 0.55, "expo"),
          (39.9, 0.55), (40.3, 0.0, "expo"), (DUR, 0.0)]
ROLL = [(0.0, 0.0), (DUR, 0.0)]
# цвет комнаты (множители BGR): холодный синий → нейтральный → тёплый
TINT = [(0.0, (1.18, 1.0, 0.84)), (10.8, (1.2, 0.98, 0.8)), (16.0, (1.2, 0.98, 0.8)), (16.6, (1.0, 1.0, 1.0)),
        (23.4, (1.0, 1.0, 1.0)), (29.4, (0.96, 1.0, 1.04)), (29.9, (0.84, 1.0, 1.16)), (40.0, (0.84, 1.0, 1.16)),
        (40.4, (1.0, 1.0, 1.0)), (DUR, (1.0, 1.0, 1.0))]
SEC = {"hook": 0, "six": 455, "city": 545, "hours": 635, "gap": 725, "fix": 815, "tg": 900.5}
SHAKES = (4.35,)
FLASHES = ((4.3, 0.15), (5.0, 0.08), (10.85, 0.1), (16.25, 0.1), (23.5, 0.1), (29.55, 0.1), (40.15, 0.12))
JUMPS = [29.65]
ALARM = None
SPOT = None

G = lambda *k: list(k)  # noqa: E731
# L — рука слева на экране, R — справа
ARM_L = G((0, 62), (0.4, 62), (0.7, 30, "back"), (1.8, 32), (2.0, 38, "back"), (2.9, 36), (3.2, -50, "back"),
          (4.3, -45), (4.7, 62, "back"), (10.9, 62), (11.15, -25, "back"), (13.6, -22), (13.85, -10, "back"),
          (15.9, -8), (16.2, 62), (23.7, 62), (23.95, -25, "back"), (26.9, -22), (27.4, 36, "back"), (29.0, 38),
          (29.3, 62), (29.6, -40, "back"), (30.4, -38), (30.7, 62), (47.26, 62), (47.4, -50, "back"), (48.0, -45),
          (48.4, 62, "back"), (DUR, 62))
ARM_R = G((0, 62), (0.4, 62), (0.7, 30, "back"), (1.8, 32), (2.0, 38, "back"), (2.9, 36), (3.2, -50, "back"),
          (4.3, -45), (4.7, 62, "back"), (5.1, 62), (5.35, -30, "back"), (7.6, -28), (7.85, -12, "back"),
          (10.5, -10), (10.8, 62), (17.7, 62), (17.9, -12, "back"), (19.2, -12), (19.45, -28, "back"), (21.2, -26),
          (21.45, -45, "back"), (23.1, -43), (23.4, 62), (27.4, 36, "back"), (29.0, 38), (29.3, 62),
          (29.6, -40, "back"), (30.4, -38), (30.75, -22, "back"), (34.6, -20), (34.85, -12, "back"), (37.1, -10),
          (37.3, -32, "back"), (39.6, -30), (40.1, 62), (40.56, -48, "back"), (42.91, -45), (43.21, -55, "back"),
          (45.91, -52), (46.16, -40, "back"), (47.11, -42), (47.26, -50, "back"), (48.0, -45), (48.15, -58),
          (48.3, -32), (48.45, -58), (48.6, -32), (DUR, 62))
HEAD = G((0, 0), (0.6, -6), (2.0, 7), (2.9, 6), (3.3, 0), (5.3, -6), (7.9, -5), (9.6, 6), (11.15, 7),
         (13.85, 5), (15.4, -4), (17.9, -6), (21.4, -8), (23.0, -4), (23.95, 6), (26.4, 5), (27.4, -7),
         (29.0, -6), (29.7, 0), (30.75, -6), (34.85, -5), (37.3, -7), (39.6, -4), (40.56, -6), (47.3, 0), (DUR, 0))


# ---------------------------------------------------------------- картинки вставок
def person_tex(body=(62, 66, 84), glow=True):
    """Безликий человечек; glow — светящийся телефон в руке (все уткнулись в экраны)."""
    im = Image.new("RGBA", (220, 440), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([60, 0, 160, 100], fill=body + (255,))
    d.rounded_rectangle([20, 112, 200, 439], 70, fill=body + (255,))
    if glow:
        g = Image.new("RGBA", (220, 440), (0, 0, 0, 0))
        ImageDraw.Draw(g).ellipse([50, 110, 170, 230], fill=(90, 170, 255, 150))
        from PIL import ImageFilter
        im = Image.alpha_composite(im, g.filter(ImageFilter.GaussianBlur(18)))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle([88, 150, 132, 222], 10, fill=(140, 205, 255, 255))
    return mg.premul(im)


def like_tex():
    im = rounded((780, 360), 34, (18, 18, 26, 255), (72, 72, 92, 255), 3)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([36, 30, 160, 92], 30, fill=BLUE_RGB + (255,))
    d.text((58, 38), "МОЗГ", font=mg.font(36), fill="white")
    # сердечко-лайк
    cx, cy = 170, 220
    d.ellipse([cx - 90, cy - 70, cx, cy + 20], fill=RED_RGB + (255,))
    d.ellipse([cx, cy - 70, cx + 90, cy + 20], fill=RED_RGB + (255,))
    d.polygon([(cx - 86, cy - 2), (cx + 86, cy - 2), (cx, cy + 95)], fill=RED_RGB + (255,))
    d.text((318, 140), "=", font=mg.font(140), fill=(200, 200, 215))
    d.line([(392, 170), (346, 300)], fill=(200, 200, 215), width=12)  # «≠» — в шрифте его нет
    # облачко разговора
    d.rounded_rectangle([460, 130, 740, 290], 50, fill=(44, 46, 60, 255), outline=(150, 160, 200, 255), width=4)
    d.polygon([(500, 284), (540, 284), (490, 330)], fill=(44, 46, 60, 255))
    d.text((500, 180), "Привет!", font=mg.font(52), fill="white")
    return mg.premul(im)


def door_tex():
    im = rounded((600, 300), 34, (18, 18, 26, 255), (72, 72, 92, 255), 3)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([36, 30, 230, 92], 30, fill=BLUE_RGB + (255,))
    d.text((58, 38), "СОСЕДИ", font=mg.font(36), fill="white")
    for i in range(3):
        x = 40 + i * 180
        d.rounded_rectangle([x, 120, x + 160, 270], 22, fill=(36, 38, 50, 255))
        d.text((x + 58, 140), "?", font=mg.font(100, tall=True), fill=(130, 130, 150))
    return mg.premul(im)


def counter_tex(hours):
    im = rounded((620, 260), 34, (24, 18, 14, 255), WARM_RGB + (255,), 5)
    d = ImageDraw.Draw(im)
    d.text((36, 26), "ЧАСОВ ОБЩЕНИЯ", font=mg.font(40), fill=(240, 200, 150))
    d.text((36, 90), f"{hours}", font=mg.font(130, tall=True), fill="white")
    return mg.premul(im)


def meter_tex(a, b):
    """«Как, по-твоему, ты понравился» (a) против «на самом деле» (b), 0..1."""
    im = rounded((780, 420), 34, (18, 18, 26, 255), (72, 72, 92, 255), 3)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([36, 30, 400, 92], 30, fill=BLUE_RGB + (255,))
    d.text((58, 38), "ТЫ ПОНРАВИЛСЯ", font=mg.font(36), fill="white")
    for i, (txt, v, col) in enumerate((("тебе кажется", a, GRAY_RGB), ("на самом деле", b, GREEN_RGB))):
        y = 130 + i * 140
        d.text((40, y), txt, font=mg.font(40), fill=(200, 200, 215))
        d.rounded_rectangle([40, y + 58, 740, y + 100], 21, fill=(40, 40, 52, 255))
        if v > 0.02:
            d.rounded_rectangle([40, y + 58, 40 + int(700 * v), y + 100], 21, fill=col + (255,))
        d.text((640, y), f"{int(round(v * 10))}/10", font=mg.font(40), fill="white")
    return mg.premul(im)


def chat_tex(typed, sent=False, reply=False):
    """Переписка: поле ввода с набираемым текстом; sent — сообщение отправлено; reply — пришёл ответ."""
    Wc, Hc = 600, 520
    im = rounded((Wc, Hc), 40, (16, 16, 22, 255), (80, 80, 100, 255), 4)
    d = ImageDraw.Draw(im)
    d.ellipse([30, 26, 100, 96], fill=(80, 90, 130, 255))
    d.text((120, 30), "Аня", font=mg.font(44), fill="white")
    d.text((120, 78), "была недавно", font=mg.font(26), fill=(130, 130, 150))
    d.line([(20, 120), (Wc - 20, 120)], fill=(50, 50, 64), width=3)
    if sent:
        d.rounded_rectangle([170, 145, Wc - 26, 235], 34, fill=TG_RGB + (255,))
        d.text((198, 166), "Привет! Как ты?", font=mg.font(40), fill="white")
    if reply:
        d.rounded_rectangle([26, 255, 420, 345], 34, fill=(44, 46, 60, 255))
        d.text((54, 276), "Как здорово!", font=mg.font(40), fill="white")
    d.rounded_rectangle([24, Hc - 110, Wc - 120, Hc - 30], 36, fill=(34, 34, 46, 255))
    if typed:
        d.text((52, Hc - 94), typed + "|", font=mg.font(40), fill=(230, 230, 240))
    else:
        d.text((52, Hc - 92), "Сообщение…", font=mg.font(36), fill=(110, 110, 125))
    d.ellipse([Wc - 104, Hc - 112, Wc - 22, Hc - 30], fill=TG_RGB + (255,) if typed else (50, 50, 64, 255))
    d.polygon([(Wc - 80, Hc - 90), (Wc - 40, Hc - 71), (Wc - 80, Hc - 52)], fill="white")
    return mg.premul(im)


def calendar_tex(n):
    """Календарь на 4 недели: галочки на одном и том же дне каждую неделю (n — сколько уже отмечено)."""
    im = rounded((780, 560), 34, (24, 18, 14, 255), WARM_RGB + (255,), 4)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([36, 30, 380, 92], 30, fill=WARM_RGB + (255,))
    d.text((58, 38), "КАЖДУЮ СРЕДУ", font=mg.font(36), fill=(40, 24, 10))
    days = ["ПН", "ВТ", "СР", "ЧТ", "ПТ", "СБ", "ВС"]
    for c, dname in enumerate(days):
        d.text((52 + c * 100, 116), dname, font=mg.font(30), fill=(240, 200, 150) if c == 2 else (130, 120, 115))
    for r in range(4):
        for c in range(7):
            x, y = 40 + c * 100, 166 + r * 96
            hot = c == 2 and r < n
            d.rounded_rectangle([x, y, x + 84, y + 80], 18, fill=WARM_RGB + (255,) if hot else (40, 34, 30, 255))
            if hot:
                d.line([(x + 20, y + 42), (x + 36, y + 58), (x + 64, y + 24)], fill="white", width=9)
    return mg.premul(im)


def heart_tex(size=260):
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    s = size / 260
    d.ellipse([20 * s, 30 * s, 135 * s, 145 * s], fill=RED_RGB + (255,))
    d.ellipse([125 * s, 30 * s, 240 * s, 145 * s], fill=RED_RGB + (255,))
    d.polygon([(26 * s, 110 * s), (234 * s, 110 * s), (130 * s, 235 * s)], fill=RED_RGB + (255,))
    return mg.premul(im)


# толпа в начале: позиции на полу (x, z) — только сзади и по бокам персонажа, чтобы никого не загораживать
_rng = np.random.default_rng(7)
CROWD = []
while len(CROWD) < 16:
    x, z = _rng.uniform(-2.6, 2.6), _rng.uniform(-0.6, 2.8)
    if (z < 0.4 and abs(x) < 1.15) or any((x - a) ** 2 + (z - b) ** 2 < 0.36 for a, b, _ in CROWD):
        continue
    CROWD.append((x, z, _rng.uniform(0, 6.28)))


# ---------------------------------------------------------------- сцена
class Scene(base.Scene):
    cfg = sys.modules[__name__]

    def __init__(self, voice):
        super().__init__(voice)
        T = self.T
        T["person"], T["person_dim"] = person_tex(), person_tex((44, 46, 58), glow=False)
        T["person_hot"] = person_tex(TG_RGB, glow=False)
        T["who"] = chip("ВОЗ, 2025", BLUE_RGB)
        T["smoke"] = card("ВРЕД ДЛЯ ЗДОРОВЬЯ", "ПОЧТИ КАК КУРЕНИЕ", "по данным учёных")
        T["home"] = chip("РАБОТА ИЗ ДОМА", BLUE_RGB)
        T["door"], T["like"] = door_tex(), like_tex()
        T["s1"], T["s2"], T["s3"] = (base.label(s, c, 60) for s, c in (("ЗНАКОМЫЙ", (190, 190, 205)),
                                                                        ("ПРИЯТЕЛЬ", (255, 200, 140)),
                                                                        ("ДРУГ", (255, 170, 90))))
        T["h50"], T["h200"] = chip("50 ЧАСОВ", WARM_RGB), chip("200+ ЧАСОВ", WARM_RGB)
        T["fear"] = stamp("СТРАШНО ПИСАТЬ", GRAY_RGB)
        T["how"] = chip("КАК БЫТЬ?", WARM_RGB)
        T["c1"], T["c2"], T["c3"] = chip("СЕКЦИЯ", WARM_RGB), chip("КУРСЫ", WARM_RGB), chip("КЛУБ ПО ИНТЕРЕСАМ", WARM_RGB)
        T["heart"] = heart_tex()
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

    def items_for(self, t, cam, it):
        T = self.T
        P = lambda y0, *a, **k: place(y0, *a, **k)  # noqa: E731

        # 1. Крючок: толпа вокруг, все в телефонах; на «не с кем» гаснут
        k = pop(t, 0.2, 3.45, 0.5, 0.35)
        if k > 0:
            dim = smooth((t - 2.3) / 0.5)
            for i, (x, z, ph) in enumerate(CROWD):
                kk = out_back(clamp((t - 0.2 - 0.04 * i) / 0.45)) * (1 if t < 3.1 else k)
                if kk <= 0.01:
                    continue
                tex = T["person"] * (1 - dim) + T["person_dim"] * dim
                self.add_bb(it, cam, (x, 0.0, z), 0.62 * kk * (1 + 0.02 * math.sin(t * 2 + ph)), tex)
                mg_blob = (x, z)
                it.append((cam.depth((x, 0.0, z)) + 0.01, lambda p=mg_blob, kk=kk: mg.floor_blob(
                    self.frame, cam, p, 0.22 * kk, 0.14 * kk, 0.4, blur=31)))

        # 2. Заставка «Паучок объясняет» (появляется, когда камера заканчивает оборот)
        k = pop(t, 3.85, 4.95, 0.4, 0.3)
        if k > 0:
            self.add_panel(it, cam, place(360, 0.0, 1.95, 0.35), face(360, -6), 1.2 * k, 0.564 * k, 0.11 * k,
                           T["stinger"], pitch=-6, side=(30, 24, 120), edge=(200, 190, 255))

        # 3. Каждый шестой: шесть человечков, один подсвечен; «вредно как курение»
        y0 = SEC["six"]
        k = pop(t, 5.3, 8.0, 0.4, 0.3)
        if k > 0:
            hot = smooth((t - 6.2) / 0.3)
            for i in range(6):
                r, c = divmod(i, 3)
                kk = out_back(clamp((t - 5.3 - 0.06 * i) / 0.4)) * (k if t > 7.5 else 1)
                if kk <= 0.01:
                    continue
                pos = P(y0, 0.52 + 0.27 * c, 1.62 - 0.42 * r)
                if i == 4:
                    tex = T["person_dim"] * (1 - hot) + T["person_hot"] * hot
                    s = 1 + 0.15 * hot
                else:
                    tex = T["person_dim"] * (1 - 0.4 * hot)
                    s = 1
                self.add_bb(it, cam, pos, 0.3 * kk * s, tex, anchor=(0.5, 0.5))
        k = pop(t, 6.0, 8.0, 0.3, 0.3)
        if k > 0:
            self.add_bb(it, cam, P(y0, 0.78, 2.08), 0.11 * k, T["who"], anchor=(0.5, 0.5))
        k = pop(t, 7.95, 10.75, 0.4, 0.3)
        if k > 0:
            pos = P(y0, 0.72, 1.5) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 14), 0.9 * k, 0.48 * k, 0.06 * k, T["smoke"], pitch=-6,
                           side=(90, 40, 40), edge=(255, 200, 200))

        # 4. Город сверху (персонаж справа): дома-коробки с окнами, «соседи?», лайк ≠ разговор
        y0 = SEC["city"]
        k = pop(t, 10.95, 16.2, 0.6, 0.35)
        if k > 0:
            r_, c_ = basis(y0)
            for i, (sd, fw, hh) in enumerate(((-0.75, -0.9, 0.9), (-1.15, -0.55, 0.6), (-1.35, -1.25, 1.2),
                                              (-0.45, -1.5, 0.7), (-0.95, -1.8, 1.0))):
                kk = out_back(clamp((t - 10.95 - 0.08 * i) / 0.5)) * (k if t > 15.5 else 1)
                p = r_ * sd + c_ * fw
                lit = (int(170 + 60 * ((i * 37) % 3 == 0)), 120, 60)
                self.add_box(it, cam, (p[0], p[2]), (0.32, 0.32), hh * kk, (58, 50, 52), radius=0.05,
                             glow=lit if t > 11.6 else None)
        k = pop(t, 11.3, 12.7, 0.3, 0.25)
        if k > 0:
            self.add_bb(it, cam, P(y0, -0.8, 1.95), 0.12 * k, T["home"], anchor=(0.5, 0.5))
        k = pop(t, 12.6, 13.95, 0.35, 0.25)
        if k > 0:
            pos = P(y0, -0.76, 1.62) + [0, 0.02 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, -12), 0.7 * k, 0.35 * k, 0.05 * k, T["door"], pitch=-10,
                           side=(40, 50, 90), edge=(200, 210, 255))
        k = pop(t, 13.85, 16.2, 0.4, 0.3)
        if k > 0:
            pos = P(y0, -0.74, 1.6) + [0, 0.02 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, -12), 0.9 * k, 0.415 * k, 0.06 * k, T["like"], pitch=-10,
                           side=(40, 50, 90), edge=(200, 210, 255))

        # 5. Часы до дружбы: ступеньки «знакомый → приятель → друг» и счётчик часов
        y0 = SEC["hours"]
        r_, c_ = basis(y0)
        for i, (t0, sd, hh, lab) in enumerate(((17.9, 0.55, 0.22, "s1"), (18.45, 0.9, 0.5, "s2"),
                                               (21.6, 1.25, 0.85, "s3"))):
            k = pop(t, t0, 23.45, 0.45, 0.3)
            if k > 0:
                p = r_ * sd + c_ * 0.05
                h = hh * k
                self.add_box(it, cam, (p[0], p[2]), (0.26, 0.26), h, WARM if i == 2 else (70, 90, 120), radius=0.07,
                             glow=(70, 140, 240) if i == 2 else None)
                self.add_bb(it, cam, p + [0, h + 0.03, 0], 0.075 * k, T[lab], anchor=(0.5, 1.0))
        k = pop(t, 19.3, 23.45)
        if k > 0:
            hrs = int(round(50 * smooth((t - 19.5) / 0.7) + 150 * smooth((t - 22.4) / 0.8)))
            hrs = hrs // 5 * 5
            pos = P(y0, 0.9, 1.95) + [0, 0.02 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 12), 0.62 * k, 0.26 * k, 0.05 * k,
                           self.cached("cnt", counter_tex, hrs), pitch=-5, side=(40, 60, 90), edge=(170, 210, 255))

        # 6. «Понравились меньше, чем на самом деле» (персонаж справа) → страшно написать первым
        y0 = SEC["gap"]
        k = pop(t, 24.0, 27.5, 0.4, 0.3)
        if k > 0:
            a = 0.42 * smooth((t - 25.4) / 0.6)
            b = 0.82 * smooth((t - 26.5) / 0.7)
            a, b = round(a * 20) / 20, round(b * 20) / 20
            pos = P(y0, -0.74, 1.55) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, -12), 0.9 * k, 0.485 * k, 0.06 * k,
                           self.cached("meter", meter_tex, a, b), pitch=-5, side=(40, 50, 90), edge=(200, 210, 255))
        k = pop(t, 27.4, 29.5, 0.4, 0.3)
        if k > 0:
            msg = "Привет! Давно не"
            n = int(len(msg) * clamp((t - 27.7) / 0.6))
            n = int(n * (1 - clamp((t - 28.75) / 0.35)))  # стёр, так и не отправив
            pos = P(y0, -0.72, 1.45) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, -12), 0.6 * k, 0.52 * k, 0.05 * k,
                           self.cached("chat", chat_tex, msg[:n]), pitch=-5, side=(40, 50, 90), radius=0.05)
        k = pop(t, 28.45, 29.5, 0.2, 0.25)
        if k > 0:
            ks = clamp((t - 28.45) / 0.15)
            self.add_bb(it, cam, P(y0, -0.85, 0.95, 0.1), 0.13 * (1 + 0.6 * (1 - ks)) * k, T["fear"], anchor=(0.5, 0.5))

        # 7. Как быть: секция / курсы / клуб → календарь встреч → «пиши первым», ответ с сердечком
        y0 = SEC["fix"]
        k = pop(t, 29.75, 31.0, 0.3, 0.25)
        if k > 0:
            self.add_bb(it, cam, P(y0, 0.78, 1.85), 0.13 * k, T["how"], anchor=(0.5, 0.5))
        for key, t0, hy in (("c1", 32.39, 2.0), ("c2", 33.13, 1.72), ("c3", 33.75, 1.44)):
            k = pop(t, t0, 34.9, 0.35, 0.25)
            if k > 0:
                self.add_bb(it, cam, P(y0, 0.78, hy), 0.12 * k, T[key], anchor=(0.5, 0.5))
        k = pop(t, 34.85, 37.35, 0.4, 0.3)
        if k > 0:
            n = 0 + (t > 35.4) + (t > 35.8) + (t > 36.2) + (t > 36.6)
            pos = P(y0, 0.74, 1.5) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 12), 0.9 * k, 0.646 * k, 0.06 * k, self.cached("cal", calendar_tex, n),
                           pitch=-5, side=(60, 80, 110), edge=(170, 210, 255))
        k = pop(t, 37.3, 40.0, 0.4, 0.3)
        if k > 0:
            msg = "Привет! Как ты?"
            n = int(len(msg) * clamp((t - 37.45) / 0.45))
            sent = t > 38.0
            reply = t > 38.6
            tex = self.cached("chat2", chat_tex, "" if sent else msg[:n], sent, reply)
            pos = P(y0, 0.74, 1.45) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 12), 0.6 * k, 0.52 * k, 0.05 * k, tex, pitch=-5,
                           side=(60, 80, 110), radius=0.05)
            if reply:
                kh = out_back(clamp((t - 38.6) / 0.35))
                self.add_bb(it, cam, pos + basis(y0)[0] * 0.3 + basis(y0)[1] * 0.08 + [0, -0.05 + 0.03 * math.sin(t * 5), 0],
                            0.13 * kh * k, T["heart"], anchor=(0.5, 0.5))

        # 8. Концовка с Telegram — общий шаблон
        self.outro_items(t, cam, it, OUTRO, SEC["tg"])


SFX = [(t, "whoosh") for t in (0.3, 3.3, 4.95, 10.8, 16.2, 23.45, 29.5, 40.1)]


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
        cv2.setNumThreads(1)
        render(voice, out, range(int(args[2]), int(args[3])))
        return
    if "--preview" in sys.argv:
        sc = Scene(voice)
        for t in (0.15, 1.0, 1.9, 2.8, 3.7, 4.1, 4.6, 6.0, 7.0, 9.0, 11.8, 13.2, 15.0, 18.3, 20.3, 22.9, 25.0,
                  26.9, 28.2, 28.9, 30.3, 33.9, 36.4, 37.7, 39.2, 42.0, 47.0):
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
    base.mix_audio(tmp, voice, out, music=os.path.join(ROOT, "work", "lonely", "music.wav"), sfx_list=SFX, dur=DUR,
                   sfx_vol=0.15)  # «вжух» лёгкий и тихий
    os.remove(tmp)


if __name__ == "__main__":
    main()
