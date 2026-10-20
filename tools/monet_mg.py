"""Ролик №14: «Что такое монетизация?».

python tools/monet_mg.py work/monet/voice_clean.wav output/monet.mp4 [--preview] [--jobs 4]
Общий движок, персонаж и концовка с Telegram — из memory_mg.py; параллельный рендер — как в money_mg.py.
Новое в монтаже: переключатель «МОНЕТИЗАЦИЯ» щёлкает во «ВКЛ»; смена сцен огромной вращающейся монетой (за ней
склейка); схема «рекламодатель → платформа → автор», по которой бегут монетки; «разблокировка» как в игре;
1000 просмотров против одной монетки; карточки способов заработка раскрываются веером; живые комментарии.
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
from money_mg import text_rub  # noqa: E402
from mg import H, W, clamp, out_back, smooth, track, wobble  # noqa: E402

FPS = 30
ROOT = mg.ROOT
OUTRO = float(open(os.path.join(ROOT, "work", "monet", "outro_start.txt")).read())  # 48.04
DUR = 56.7
GOLD_RGB, GREEN_RGB, PINK_RGB = (245, 190, 50), (46, 196, 110), (240, 70, 110)
GOLD, GREEN = GOLD_RGB[::-1], GREEN_RGB[::-1]

# ---------------------------------------------------------------- субтитры
T_, Go_, G_, Pi_ = TG_RGB, GOLD_RGB, GREEN_RGB, PINK_RGB
SUBS = [
    (0.218, "ЧТО ТАКОЕ МОНЕТИЗАЦИЯ?", ("МОНЕТИЗАЦИЯ?",), Go_), (2.0, "", ()),
    (3.838, "ЭТО КОГДА ТВОИ РОЛИКИ", ()), (4.998, "НАЧИНАЮТ ПРИНОСИТЬ ДЕНЬГИ", ("ДЕНЬГИ",), Go_),
    (7.178, "ПЛАТФОРМА ПОКАЗЫВАЕТ", ()), (8.358, "РЕКЛАМУ РЯДОМ С ВИДЕО", ("РЕКЛАМУ",)),
    (9.738, "И ДЕЛИТСЯ С ТОБОЙ", ("ДЕЛИТСЯ",), G_), (11.018, "ЧАСТЬЮ ДЕНЕГ ОТ НЕЁ", ("ЧАСТЬЮ",), G_),
    (12.81, "НО ВКЛЮЧАЮТ ЕЁ НЕ СРАЗУ", ("НЕ", "СРАЗУ")), (14.63, "НА YOUTUBE, НАПРИМЕР, НУЖНА", ()),
    (16.21, "ТЫСЯЧА ПОДПИСЧИКОВ", ("ТЫСЯЧА",), Go_), (17.69, "И 10 МИЛЛИОНОВ ПРОСМОТРОВ", ("10", "МИЛЛИОНОВ"), Go_),
    (19.33, "ШОРТСОВ ЗА ТРИ МЕСЯЦА", ("ТРИ",)), (21.63, "А ЗА ТЫСЯЧУ ПРОСМОТРОВ", ()),
    (22.93, "КОРОТКИХ РОЛИКОВ ОБЫЧНО", ()), (24.41, "ПЛАТЯТ МЕНЬШЕ ДОЛЛАРА", ("МЕНЬШЕ",)),
    (26.542, "А В РОССИИ YOUTUBE", ()), (27.662, "С 2022 ГОДА", ("2022",)), (29.202, "АВТОРАМ НЕ ПЛАТИТ", ("НЕ", "ПЛАТИТ")),
    (30.602, "ПОЭТОМУ БЛОГЕРЫ", ()), (31.502, "ЗАРАБАТЫВАЮТ ИНАЧЕ:", ("ИНАЧЕ:",), G_),
    (32.982, "РЕКЛАМА БРЕНДОВ,", ("РЕКЛАМА",), Go_), (34.122, "ДОНАТЫ,", ("ДОНАТЫ,",), Pi_),
    (34.762, "СВОЙ ТЕЛЕГРАМ-КАНАЛ", ("ТЕЛЕГРАМ-КАНАЛ",), T_), (36.422, "НО ГЛАВНОЕ ПРАВИЛО", ()),
    (37.542, "ВЕЗДЕ ОДНО:", ()), (38.602, "СНАЧАЛА ЗРИТЕЛИ,", ("ЗРИТЕЛИ,",), G_), (39.822, "ПОТОМ ДЕНЬГИ", ("ДЕНЬГИ",), Go_),
    (41.073, "КСТАТИ, ЭТУ ИДЕЮ", ("ИДЕЮ",), Pi_), (42.273, "ПРЕДЛОЖИЛИ ПОДПИСЧИКИ", ("ПОДПИСЧИКИ",), Pi_),
    (43.573, "В КОММЕНТАРИЯХ", ()), (44.673, "НАПИШИ, НА КАКУЮ ТЕМУ", ()), (45.773, "СНЯТЬ СЛЕДУЮЩЕЕ ВИДЕО", ("СЛЕДУЮЩЕЕ",), Pi_),
    (48.0, "А ЕЩЁ БОЛЬШЕ ИНТЕРЕСНОГО —", (), T_), (49.36, "В МОЁМ ТЕЛЕГРАМ-КАНАЛЕ", ("ТЕЛЕГРАМ-КАНАЛЕ",), T_),
    (50.69, "«ПАУЧЬЯ НОРКА»", ("«ПАУЧЬЯ", "НОРКА»"), T_), (51.81, "ТАМ ЭКСКЛЮЗИВ,", ("ЭКСКЛЮЗИВ",), T_),
    (52.51, "КОТОРОГО НЕТ В РОЛИКАХ", (), T_), (53.81, "ССЫЛКА В ПРОФИЛЕ —", ("ПРОФИЛЕ",), T_),
    (54.83, "ЗАЛЕТАЙ!", ("ЗАЛЕТАЙ!",), T_),
]

# ---------------------------------------------------------------- камера и сцены
COINS = (12.55, 26.3)            # смена сцены вращающейся монетой во весь экран
C1, C2 = (c + 0.2 for c in COINS)
CAMERA = [
    (0.0, (-12.0, 4.1, 1.4, 0.88, 50.0)), (0.9, (-8.0, 3.6, 1.15, 0.84, 50.0), "out"),
    (1.9, (-6.0, 3.6, 1.15, 0.84, 50.0)),
    (2.3, (0.0, 3.9, 1.3, 0.98, 50.0), "inout"),              # заставка
    (3.3, (1.0, 3.9, 1.3, 0.98, 50.0)),
    (3.75, (95.0, 3.6, 1.2, 0.84, 50.0), "expo"),             # что это и как течёт деньги
    (C1 - 0.01, (104.0, 3.6, 1.25, 0.84, 50.0), "lin"),
    (C1 + 0.01, (190.0, 3.4, 0.7, 0.96, 48.0), "lin"),        # условия YouTube — снизу
    (C2 - 0.01, (197.0, 3.3, 0.75, 0.96, 48.0), "lin"),
    (C2 + 0.01, (285.0, 3.6, 1.25, 0.84, 50.0), "lin"),       # Россия и другие способы
    (40.6, (296.0, 3.7, 1.3, 0.86, 50.0), "lin"),
    (41.05, (372.0, 4.1, 1.55, 0.9, 50.0), "expo"),           # идею предложили подписчики
    (47.6, (378.0, 4.0, 1.5, 0.92, 50.0), "lin"),
    (48.0, (460.0, 4.3, 1.3, 1.12, 50.0), "expo"), (DUR, (461.0, 4.0, 1.3, 1.15, 50.0)),
]
OFFSET = [(0.0, -0.5), (1.9, -0.5), (2.3, 0.0, "inout"), (3.3, 0.0), (3.75, 0.55, "expo"), (C1 - 0.01, 0.55),
          (C1 + 0.01, -0.55, "lin"), (C2 - 0.01, -0.55), (C2 + 0.01, 0.55, "lin"), (40.6, 0.55),
          (41.05, 0.0, "expo"), (DUR, 0.0)]
TINT = [(0.0, (0.88, 1.04, 1.14)), (3.3, (0.88, 1.04, 1.14)), (3.75, (0.9, 1.06, 1.06)), (C1 - 0.01, (0.9, 1.06, 1.06)),
        (C1 + 0.01, (1.06, 0.98, 0.92), "lin"), (C2 - 0.01, (1.06, 0.98, 0.92)), (C2 + 0.01, (0.94, 1.08, 0.98), "lin"),
        (40.7, (0.94, 1.08, 0.98)), (41.1, (1.06, 0.94, 1.08)), (47.7, (1.06, 0.94, 1.08)), (48.1, (1.0, 1.0, 1.0)),
        (DUR, (1.0, 1.0, 1.0))]
SEC = {"hook": -8, "flow": 99, "rules": 193, "ru": 290, "idea": 375, "tg": 460.5}
SHAKES = (25.4,)
FLASHES = ((0.9, 0.1), (20.6, 0.12), (48.05, 0.12))
JUMPS = [20.6]
ALARM = None
SPOT = None

G = lambda *k: list(k)  # noqa: E731
ARM_L = G((0, 62), (0.3, -25, "back"), (1.8, -22), (2.0, 62), (2.2, -50, "back"), (3.2, -45), (3.5, 62, "back"),
          (C1 + 0.05, 62), (13.0, -25, "back"), (21.4, -22), (21.6, -12, "back"), (25.2, -10), (25.4, 36, "back"),
          (26.2, 38), (26.4, 62), (41.0, 62), (41.15, 30, "back"), (43.9, 32), (44.1, 62), (54.94, 62),
          (55.09, -50, "back"), (55.69, -45), (56.09, 62, "back"), (DUR, 62))
ARM_R = G((0, 62), (2.0, 62), (2.2, -50, "back"), (3.2, -45), (3.5, 62, "back"), (3.9, -30, "back"), (6.9, -28),
          (7.1, -40, "back"), (9.6, -38), (9.8, -15, "back"), (12.4, -12), (12.6, 62), (20.5, 62), (20.6, -50, "back"),
          (21.2, -45), (21.4, 62), (25.4, 36, "back"), (26.2, 38), (26.4, 62), (C2 + 0.05, 62), (26.7, -30, "back"),
          (30.4, -28), (30.6, -20, "back"), (36.2, -18), (36.4, -35, "back"), (40.8, -33), (41.0, 62),
          (41.15, 30, "back"), (43.9, 32), (44.6, -30, "back"), (47.6, -28), (47.79, 62), (48.25, -48, "back"),
          (50.6, -45), (50.9, -55, "back"), (53.6, -52), (53.85, -40, "back"), (54.8, -42), (54.95, -50, "back"),
          (55.69, -45), (55.84, -58), (55.99, -32), (56.14, -58), (56.29, -32), (DUR, 62))
HEAD = G((0, 0), (0.6, -6), (1.0, 7), (2.0, 0), (3.9, -6), (5.0, 6), (7.1, -7), (9.8, 6), (12.8, -5), (14.6, 6),
         (16.2, -6), (17.7, 7), (20.6, 0), (21.6, -6), (24.4, 7), (25.4, 9), (26.6, -6), (29.2, -8), (30.6, 5),
         (33.0, -6), (34.8, 6), (36.4, -5), (38.6, 7), (40.0, -6), (41.1, 6), (43.0, -5), (44.7, 7), (48.25, -6),
         (55.0, 0), (DUR, 0))


# ---------------------------------------------------------------- картинки вставок
def header(d, text, col, dark=False):
    f = mg.font(36)
    tw = d.textbbox((0, 0), text, font=f)[2]
    d.rounded_rectangle([36, 30, 36 + tw + 44, 92], 30, fill=col + (255,))
    d.text((58, 38), text, font=f, fill=(30, 24, 10) if dark else "white")


def toggle_tex(on):
    im = rounded((720, 300), 40, (18, 18, 26, 255), (90, 90, 110, 255), 4)
    d = ImageDraw.Draw(im)
    d.text((40, 40), "МОНЕТИЗАЦИЯ", font=mg.font(64, tall=True), fill="white")
    col = tuple(int(a + (b - a) * on) for a, b in zip((70, 70, 84), GREEN_RGB))
    d.rounded_rectangle([40, 170, 300, 270], 50, fill=col + (255,))
    x = int(90 + 160 * on)
    d.ellipse([x - 42, 178, x + 42, 262], fill="white")
    d.text((330, 186), "ВКЛ" if on > 0.5 else "ВЫКЛ", font=mg.font(60, tall=True),
           fill=GREEN_RGB + (255,) if on > 0.5 else (130, 130, 145))
    return mg.premul(im)


def coin_small():
    S = 160
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([0, 0, S - 1, S - 1], fill=(196, 140, 30, 255))
    d.ellipse([12, 12, S - 13, S - 13], fill=GOLD_RGB + (255,))
    text_rub(d, (52, 34), "₽", mg.font(80, tall=True), (130, 90, 10))
    return mg.premul(im)


def big_coin():
    """Огромная монета для перехода: диаметр больше диагонали кадра."""
    S = 2300
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([0, 0, S - 1, S - 1], fill=(170, 120, 20, 255))
    d.ellipse([90, 90, S - 91, S - 91], fill=GOLD_RGB + (255,))
    d.ellipse([260, 260, S - 261, S - 261], outline=(200, 145, 30, 255), width=40)
    text_rub(d, (790, 600), "₽", mg.font(1000, tall=True), (150, 100, 10))
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGBA2BGRA)


def node_tex(label, col, kind):
    S, W = 300, 420
    im = Image.new("RGBA", (W, S + 70), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([80, 0, W - 81, S - 41], fill=(22, 22, 30, 255), outline=col + (255,), width=10)
    cx, cy = W // 2, (S - 40) // 2
    if kind == "brand":
        d.rounded_rectangle([cx - 60, cy - 30, cx + 60, cy + 50], 14, fill=col + (255,))
        d.rounded_rectangle([cx - 25, cy - 55, cx + 25, cy - 25], 10, outline=col + (255,), width=10)
    elif kind == "play":
        d.rounded_rectangle([cx - 70, cy - 50, cx + 70, cy + 50], 24, fill=col + (255,))
        d.polygon([(cx - 18, cy - 28), (cx - 18, cy + 28), (cx + 28, cy)], fill="white")
    else:
        d.ellipse([cx - 30, cy - 60, cx + 30, cy], fill=col + (255,))
        d.pieslice([cx - 60, cy + 5, cx + 60, cy + 125], 180, 360, fill=col + (255,))
    f = mg.font(40)
    tw = d.textbbox((0, 0), label, font=f)[2]
    d.rounded_rectangle([W / 2 - tw / 2 - 16, S - 30, W / 2 + tw / 2 + 16, S + 30], 24, fill=col + (255,))
    d.text((W / 2 - tw / 2, S - 26), label, font=f, fill="white")
    return mg.premul(im)


def unlock_tex(a, b, opened):
    im = rounded((780, 520), 34, (18, 18, 26, 255), (72, 72, 92, 255), 3)
    d = ImageDraw.Draw(im)
    header(d, "УСЛОВИЯ YOUTUBE", RED_RGB)
    for i, (txt, v) in enumerate((("подписчики · 1 000", a), ("просмотры Shorts · 10 млн за 90 дней", b))):
        y = 130 + i * 120
        d.text((40, y), txt, font=mg.font(36), fill=(210, 210, 225))
        d.rounded_rectangle([40, y + 52, 740, y + 92], 20, fill=(40, 40, 52, 255))
        if v > 0.03:
            d.rounded_rectangle([40, y + 52, 40 + int(700 * v), y + 92], 20, fill=GOLD_RGB + (255,))
    x, y = 60, 390
    col = GREEN_RGB if opened else (120, 120, 135)
    d.rounded_rectangle([x, y + 30, x + 70, y + 100], 12, fill=col + (255,))
    if opened:
        d.arc([x + 30, y - 10, x + 100, y + 60], 180, 360, fill=col + (255,), width=12)
    else:
        d.arc([x + 5, y - 10, x + 65, y + 60], 180, 360, fill=col + (255,), width=12)
    d.text((170, y + 30), "МОНЕТИЗАЦИЯ ОТКРЫТА" if opened else "пока закрыто",
           font=mg.font(48, tall=True) if opened else mg.font(40), fill=GREEN_RGB + (255,) if opened else (130, 130, 145))
    return mg.premul(im)


def pay_tex(p):
    """1000 просмотров против одной маленькой монетки."""
    im = rounded((780, 460), 34, (18, 18, 26, 255), (72, 72, 92, 255), 3)
    d = ImageDraw.Draw(im)
    header(d, "1 000 ПРОСМОТРОВ", BLUE_RGB)
    n = int(50 * p)
    for i in range(50):
        r, c = divmod(i, 10)
        x, y = 50 + c * 42, 130 + r * 42
        col = (150, 180, 240, 255) if i < n else (44, 44, 56, 255)
        d.ellipse([x, y + 8, x + 30, y + 26], outline=col, width=4)
        d.ellipse([x + 10, y + 12, x + 20, y + 22], fill=col)
    d.text((40, 360), "каждый значок — 20 просмотров", font=mg.font(30), fill=(130, 130, 150))
    if p >= 1:
        d.text((510, 150), "=", font=mg.font(90), fill="white")
        d.ellipse([600, 165, 660, 225], fill=GOLD_RGB + (255,))
        d.text((540, 250), "меньше $1", font=mg.font(46, tall=True), fill=GOLD_RGB + (255,))
    return mg.premul(im)


def fan_tex(title, sub, col):
    im = rounded((420, 560), 40, (20, 20, 28, 255), col + (255,), 8)
    d = ImageDraw.Draw(im)
    d.ellipse([140, 60, 280, 200], fill=col + (255,))
    f = mg.font(54, tall=True)
    for k, line in enumerate(title.split("\n")):
        tw = d.textbbox((0, 0), line, font=f)[2]
        d.text((210 - tw / 2, 250 + k * 72), line, font=f, fill="white")
    f2 = mg.font(30)
    tw = d.textbbox((0, 0), sub, font=f2)[2]
    d.text((210 - tw / 2, 470), sub, font=f2, fill=(170, 170, 190))
    return mg.premul(im)


def rule_tex():
    im = rounded((780, 300), 34, (16, 24, 20, 255), GREEN_RGB + (255,), 5)
    d = ImageDraw.Draw(im)
    header(d, "ГЛАВНОЕ ПРАВИЛО", GREEN_RGB)
    f = mg.font(50, tall=True)
    d.rounded_rectangle([30, 130, 330, 250], 30, fill=GREEN_RGB + (255,))
    tw = d.textbbox((0, 0), "1. ЗРИТЕЛИ", font=f)[2]
    d.text((180 - tw / 2, 160), "1. ЗРИТЕЛИ", font=f, fill="white")
    d.polygon([(350, 175), (400, 175), (400, 150), (445, 190), (400, 230), (400, 205), (350, 205)], fill=(230, 230, 240, 255))
    d.rounded_rectangle([460, 130, 750, 250], 30, fill=GOLD_RGB + (255,))
    tw = d.textbbox((0, 0), "2. ДЕНЬГИ", font=f)[2]
    d.text((605 - tw / 2, 160), "2. ДЕНЬГИ", font=f, fill=(40, 30, 10))
    return mg.premul(im)


def idea_tex(text):
    im = rounded((700, 230), 40, (18, 16, 26, 255), PINK_RGB + (255,), 4)
    d = ImageDraw.Draw(im)
    header(d, "ТВОЯ ИДЕЯ", PINK_RGB)
    d.rounded_rectangle([36, 116, 664, 196], 36, fill=(34, 32, 46, 255))
    d.text((66, 132), (text + "|") if text else "Напиши тему для ролика…", font=mg.font(38),
           fill="white" if text else (120, 118, 135))
    return mg.premul(im)


IDEAS = ["про монетизацию!", "про деньги", "про сон", "про стримы", "про нейросети", "про отношения", "про учёбу"]


# ---------------------------------------------------------------- сцена
class Scene(base.Scene):
    cfg = sys.modules[__name__]

    def __init__(self, voice):
        super().__init__(voice)
        T = self.T
        T["coin"] = coin_small()
        T["bigcoin"] = big_coin()
        T["what"] = card("МОНЕТИЗАЦИЯ", "РОЛИКИ = ДЕНЬГИ", "реклама рядом с твоим видео", GOLD_RGB)
        T["brand"], T["plat"], T["author"] = (node_tex("рекламодатель", (240, 140, 60), "brand"),
                                              node_tex("платформа", RED_RGB, "play"),
                                              node_tex("автор", GREEN_RGB, "person"))
        T["part"] = chip("ЧАСТЬ ДЕНЕГ — ТЕБЕ", GREEN_RGB)
        T["only"] = stamp("ВСЕГО?!", RED_RGB)
        T["ru"] = card("В РОССИИ", "ВЫПЛАТ НЕТ", "выплаты остановлены в 2022", RED_RGB)
        T["fan"] = [fan_tex("РЕКЛАМА\nБРЕНДОВ", "интеграции", GOLD_RGB), fan_tex("ДОНАТЫ", "поддержка зрителей", PINK_RGB),
                    fan_tex("TELEGRAM-\nКАНАЛ", "свой канал", TG_RGB)]
        T["rule"] = rule_tex()
        T["ideas"] = [chip(s, PINK_RGB if i % 2 else (150, 90, 230)) for i, s in enumerate(IDEAS)]
        T["fromyou"] = chip("ИДЕЮ ПРЕДЛОЖИЛИ ВЫ", PINK_RGB)
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
        for c0 in COINS:  # монета поворачивается к зрителю, закрывая кадр, и уходит ребром
            if c0 < t < c0 + 0.4:
                q = 1 - abs(2 * (t - c0) / 0.4 - 1)
                w = int(2300 * math.sin(math.pi / 2 * q))
                if w < 4:
                    continue
                coin = cv2.resize(self.T["bigcoin"], (w, 2300), interpolation=cv2.INTER_AREA)
                x0, y0 = W // 2 - w // 2, H // 2 - 1150
                xa, xb = max(0, x0), min(W, x0 + w)
                ya, yb = max(0, y0), min(H, y0 + 2300)
                part = coin[ya - y0:yb - y0, xa - x0:xb - x0].astype(np.float32)
                a = part[..., 3:4] / 255.0
                shade = 0.75 + 0.25 * q
                fr = fr.copy()
                fr[ya:yb, xa:xb] = (fr[ya:yb, xa:xb] * (1 - a) + part[..., :3] * shade * a).astype(np.uint8)
        return fr

    def coin_path(self, it, cam, a, b, t, t0, t1, n, size=0.07):
        """Монетки бегут из точки a в точку b."""
        for i in range(n):
            s = t0 + (t1 - t0) * i / n
            p = (t - s) / 0.7
            if 0 < p < 1:
                pos = a * (1 - p) + b * p + [0, 0.12 * math.sin(math.pi * p), 0]
                self.add_bb(it, cam, pos, size, self.T["coin"], anchor=(0.5, 0.5))

    def items_for(self, t, cam, it):
        T = self.T
        P = lambda y0, *a, **k: place(y0, *a, **k)  # noqa: E731

        # 1. Крючок (персонаж справа): переключатель «МОНЕТИЗАЦИЯ» щёлкает во «ВКЛ», вылетают монеты
        y0 = SEC["hook"]
        k = pop(t, 0.2, 2.1, 0.35, 0.25)
        if k > 0:
            on = round(smooth((t - 0.8) / 0.2) * 8) / 8
            pos = P(y0, -0.74, 1.62) + [0, 0.02 * math.sin(t * 3), 0]
            self.add_panel(it, cam, pos, face(y0, -12), 0.72 * k, 0.3 * k, 0.05 * k, self.cached("tg", toggle_tex, on),
                           pitch=-5, side=(40, 60, 50), edge=(200, 255, 220))
            if 0.9 < t < 2.1:
                for i in range(8):
                    p = (t - 0.9 - 0.05 * i) / 0.9
                    if 0 < p < 1:
                        a = i * 0.8
                        q = pos + np.array([math.cos(a) * 0.5 * p, 0.35 * math.sin(math.pi * p) + 0.2, 0]) + \
                            basis(y0)[1] * 0.1
                        self.add_bb(it, cam, q, 0.07, T["coin"], anchor=(0.5, 0.5))

        # 2. Заставка
        k = pop(t, 2.15, 3.55, 0.35, 0.3)
        if k > 0:
            self.add_panel(it, cam, place(0, 0.0, 1.95, 0.35), face(0, -6), 1.2 * k, 0.564 * k, 0.11 * k,
                           T["stinger"], pitch=-6, side=(30, 24, 120), edge=(200, 190, 255))

        # 3. Что это (персонаж слева): «ролики = деньги» → схема рекламодатель → платформа → автор
        y0 = SEC["flow"]
        k = pop(t, 3.85, 7.1, 0.4, 0.3)
        if k > 0:
            pos = P(y0, 0.72, 1.6) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 14), 0.9 * k, 0.48 * k, 0.06 * k, T["what"], pitch=-6,
                           side=(70, 60, 30), edge=(255, 230, 170))
        k = pop(t, 7.1, C1, 0.4, 0.2)
        if k > 0:
            nb, npl, na = P(y0, 0.95, 2.1), P(y0, 0.95, 1.5), P(y0, 0.95, 0.9)
            for tex, pos in ((T["brand"], nb), (T["plat"], npl), (T["author"], na)):
                self.add_bb(it, cam, pos, 0.32 * k, tex, anchor=(0.5, 0.5))
            if k > 0.9:
                self.coin_path(it, cam, nb + [0, -0.12, 0], npl + [0, 0.12, 0], t, 7.5, 12.3, 14)
                self.coin_path(it, cam, npl + [0, -0.12, 0], na + [0, 0.12, 0], t, 9.9, 12.3, 5)
            kc = pop(t, 10.3, C1, 0.3, 0.2)
            if kc > 0:
                self.add_bb(it, cam, P(y0, 0.62, 1.4), 0.1 * kc, T["part"], anchor=(1.0, 0.5))

        # 4. Условия (персонаж справа, снизу): «разблокировка» → 1000 просмотров = меньше доллара
        y0 = SEC["rules"]
        r_, c_ = basis(y0)
        k = pop(t, C1 + 0.05, 21.6, 0.4, 0.3)
        if k > 0:
            a = round(smooth((t - 16.2) / 0.9) * 16) / 16
            b = round(smooth((t - 17.9) / 1.6) * 16) / 16
            pos = P(y0, -0.72, 1.65) + [0, 0.02 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, -12), 0.9 * k, 0.6 * k, 0.06 * k,
                           self.cached("ul", unlock_tex, a, b, t > 20.6), pitch=7, side=(80, 40, 40), edge=(255, 200, 200))
        k = pop(t, 21.6, C2, 0.4, 0.2)
        if k > 0:
            p = round(smooth((t - 22.0) / 2.2) * 25) / 25
            pos = P(y0, -0.72, 1.65) + [0, 0.02 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, -12), 0.9 * k, 0.531 * k, 0.06 * k, self.cached("pay", pay_tex, p),
                           pitch=7, side=(40, 50, 90), edge=(200, 210, 255))
            if t > 25.4:
                ks = clamp((t - 25.4) / 0.15)
                self.add_bb(it, cam, pos + c_ * 0.12 + [0, -0.42, 0], 0.15 * (1 + 0.6 * (1 - ks)) * k, T["only"],
                            anchor=(0.5, 0.5))

        # 5. Россия и другие способы (персонаж слева): «выплат нет» → веер карточек → «сначала зрители»
        y0 = SEC["ru"]
        r_, c_ = basis(y0)
        k = pop(t, C2 + 0.05, 30.6, 0.4, 0.3)
        if k > 0:
            pos = P(y0, 0.72, 1.6) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 14), 0.9 * k, 0.48 * k, 0.06 * k, T["ru"], pitch=-6,
                           side=(80, 40, 40), edge=(255, 200, 200))
        if 30.6 < t < 36.5:
            base_pos = P(y0, 0.82, 1.45)
            for i, t0 in enumerate((32.98, 34.12, 34.76)):
                kk = pop(t, t0 - 0.1, 36.5, 0.35, 0.3)
                if kk <= 0:
                    continue
                ang = (i - 1) * 16 * smooth((t - t0) / 0.4)
                a = math.radians(ang)
                pos = base_pos + r_ * 0.36 * math.sin(a) * 1.0 + np.array([0, -0.36 * (1 - math.cos(a)), 0]) + c_ * (0.02 * i)
                self.add_panel(it, cam, pos, face(y0, 8), 0.34 * kk, 0.453 * kk, 0.03, T["fan"][i], roll=-ang,
                               side=(40, 40, 52), radius=0.03)
        k = pop(t, 36.45, 41.0, 0.4, 0.3)
        if k > 0:
            pos = P(y0, 0.72, 1.62) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 12), 0.9 * k, 0.346 * k, 0.05 * k, T["rule"], pitch=-6,
                           side=(30, 60, 40), edge=(190, 255, 220))

        # 6. Идею предложили подписчики (персонаж в центре): летят идеи-комментарии, набирается своя
        y0 = SEC["idea"]
        r_, c_ = basis(y0)
        if 41.0 < t < 47.9:
            for i in range(14):
                s0 = 41.1 + 0.42 * i
                p = (t - s0) / 1.8
                if not 0 < p < 1:
                    continue
                side = (1 if i % 2 else -1) * (0.52 + 0.12 * ((i // 2) % 2))
                pos = r_ * side + c_ * -0.25 + [0, 0.4 + 1.6 * p, 0]
                kk = out_back(clamp(p / 0.15)) * (1 - smooth((p - 0.8) / 0.2))
                self.add_bb(it, cam, pos, 0.12 * kk, T["ideas"][i % len(T["ideas"])], anchor=(0.5, 0.5))
        k = pop(t, 41.15, 44.6, 0.3, 0.3)
        if k > 0:
            self.add_bb(it, cam, P(y0, 0.0, 2.3), 0.12 * k, T["fromyou"], anchor=(0.5, 0.5))
        k = pop(t, 44.6, 47.8, 0.4, 0.3)
        if k > 0:
            msg = "Сними про…"
            n = int(len(msg) * clamp((t - 45.4) / 1.0))
            pos = P(y0, 0.0, 2.2) + [0, 0.02 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 0), 0.75 * k, 0.246 * k, 0.04 * k, self.cached("idea", idea_tex, msg[:n]),
                           pitch=-6, side=(70, 30, 50), edge=(255, 200, 220))

        # 7. Концовка с Telegram — общий шаблон
        self.outro_items(t, cam, it, OUTRO, SEC["tg"])


SFX = [(t, "whoosh") for t in (0.0, 2.1, 3.3) + COINS + (40.6, 47.6)]


def render(voice, out, frames):
    sc = Scene(voice)
    wr = mg.Writer(out, FPS)
    for n, i in enumerate(frames):
        t = i / FPS
        speed = abs(sc.cam(t + 0.5 / FPS).yaw - sc.cam(t - 0.5 / FPS).yaw) * FPS
        cut = any(abs(t - c) < 1.0 / FPS for c in (C1, C2))
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
        for t in (0.6, 1.3, 2.8, 5.0, 8.4, 11.2, 12.75, 16.8, 20.9, 23.6, 25.6, 26.5, 28.5, 34.0, 35.5, 38.5, 42.4,
                  46.0, 50.0, 55.5):
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
    base.mix_audio(tmp, voice, out, music=os.path.join(ROOT, "work", "monet", "music.wav"), sfx_list=SFX, dur=DUR,
                   sfx_vol=0.15)  # «вжух» лёгкий и тихий
    os.remove(tmp)


if __name__ == "__main__":
    main()
