"""Ролик №19: «Что будет, если жить без часов и календаря?» (+ рекламный баннер в середине).

python tools/clock_mg.py work/clock/voice_clean.wav output/clock.mp4 [--preview] [--jobs 4]
затем: python tools/ad_banner.py insert output/clock.mp4 25.5 output/clock_ad.mp4 clock_mg
Общий движок, персонаж и концовка с Telegram — из memory_mg.py; параллельный рендер — как в money_mg.py.
Новое в монтаже: листки календаря срываются и улетают, стрелки часов крутятся; «пещера» — темнота и луч
фонарика, который ходит по кадру; камера опускается вниз, как в пещеру; смена сцены «стрелкой часов» —
кадр закрывается и открывается круговым движением; два календаря «думал / на самом деле» и штамп «−25 дней»;
циферблат на 24 часа, стрелка которого уходит за край; солнце встаёт из-за пола; шторы раздвигаются.
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
OUTRO = float(open(os.path.join(ROOT, "work", "clock", "outro_start.txt")).read())  # 38.25
DUR = 46.9
AD_CUT = 25.5                    # здесь вклеивается баннер (камера стоит спокойно)
GOLD_RGB, GREEN_RGB, PURPLE_RGB, SUN_RGB = (245, 190, 50), (46, 196, 110), (150, 90, 230), (255, 170, 60)

# ---------------------------------------------------------------- субтитры
T_, Go_, G_, Pu_, S_ = TG_RGB, GOLD_RGB, GREEN_RGB, PURPLE_RGB, SUN_RGB
SUBS = [
    (0.177, "ЧТО БУДЕТ, ЕСЛИ ЖИТЬ", ()), (1.317, "БЕЗ ЧАСОВ И КАЛЕНДАРЯ?", ("ЧАСОВ", "КАЛЕНДАРЯ?"), Go_),
    (2.85, "", ()),
    (4.377, "В 1962 ГОДУ", ("1962",), Go_), (6.497, "ФРАНЦУЗ МИШЕЛЬ СИФР", ("СИФР",), T_),
    (8.077, "ДВА МЕСЯЦА ПРОЖИЛ", ("ДВА", "МЕСЯЦА"), Go_), (9.397, "В ПЕЩЕРЕ БЕЗ ЧАСОВ", ("ПЕЩЕРЕ",), Pu_),
    (10.557, "И СОЛНЕЧНОГО СВЕТА", ()),
    (12.16, "КОГДА ЕГО ВЫВЕЛИ НАРУЖУ,", ()), (13.727, "ОН БЫЛ УВЕРЕН, ЧТО", ()), (14.527, "ПРОШЁЛ ВСЕГО МЕСЯЦ", ("МЕСЯЦ",), Go_),
    (16.307, "ОН ПОТЕРЯЛ ПОЧТИ", ()), (17.167, "25 ДНЕЙ!", ("25", "ДНЕЙ!")),
    (19.477, "ЕГО СУТКИ", ()), (20.157, "САМИ РАСТЯНУЛИСЬ:", ("РАСТЯНУЛИСЬ:",), Pu_),
    (21.537, "ВНУТРЕННИЕ ЧАСЫ ЧЕЛОВЕКА", ()), (22.917, "ИДУТ ЧУТЬ ДОЛЬШЕ", ("ДОЛЬШЕ",), Pu_), (24.237, "24 ЧАСОВ", ("24",), Go_),
    (AD_CUT - 0.02, "", ()),
    (25.93, "КАЖДОЕ УТРО СОЛНЕЧНЫЙ", ("СОЛНЕЧНЫЙ",), S_), (27.286, "СВЕТ ПОДВОДИТ ЭТИ ЧАСЫ,", ()),
    (28.566, "КАК БУДИЛЬНИК", ("БУДИЛЬНИК",), S_), (29.606, "ПОЭТОМУ БЕЗ СВЕТА", ()),
    (30.686, "И РАСПИСАНИЯ РЕЖИМ", ()), (31.786, "БЫСТРО ЛОМАЕТСЯ", ("ЛОМАЕТСЯ",)),
    (33.276, "СОВЕТ:", ("СОВЕТ:",), G_), (34.056, "УТРОМ ОТКРОЙ ШТОРЫ", ("ШТОРЫ",), S_),
    (35.116, "ИЛИ ВЫЙДИ НА УЛИЦУ", ()), (36.316, "ХОТЯ БЫ НА 10 МИНУТ", ("10", "МИНУТ"), G_),
    (38.213, "А ЕЩЁ БОЛЬШЕ ИНТЕРЕСНОГО —", (), T_), (39.573, "В МОЁМ ТЕЛЕГРАМ-КАНАЛЕ", ("ТЕЛЕГРАМ-КАНАЛЕ",), T_),
    (40.903, "«ПАУЧЬЯ НОРКА»", ("«ПАУЧЬЯ", "НОРКА»"), T_), (42.023, "ТАМ ЭКСКЛЮЗИВ,", ("ЭКСКЛЮЗИВ",), T_),
    (42.723, "КОТОРОГО НЕТ В РОЛИКАХ", (), T_), (44.023, "ССЫЛКА В ПРОФИЛЕ —", ("ПРОФИЛЕ",), T_),
    (45.043, "ЗАЛЕТАЙ!", ("ЗАЛЕТАЙ!",), T_),
]

# ---------------------------------------------------------------- камера и сцены
WIPES = (11.85, 25.55)           # смена сцены «стрелкой часов»; склейка — на середине
C1, C2 = (w + 0.2 for w in WIPES)
CAVE = (4.4, 11.9)               # пещера: темнота и луч фонарика
CAMERA = [
    (0.0, (12.0, 4.2, 1.45, 0.9, 50.0)), (0.8, (7.0, 3.6, 1.15, 0.84, 50.0), "out"),
    (2.7, (6.0, 3.6, 1.15, 0.84, 50.0)),
    (2.95, (0.0, 3.9, 1.3, 0.98, 50.0), "inout"),             # заставка
    (4.2, (-1.0, 3.9, 1.3, 0.98, 50.0)),
    (4.5, (-88.0, 3.6, 3.2, 1.4, 50.0), "expo"),              # пещера: камера опускается вниз
    (11.8, (-94.0, 3.6, 1.15, 0.86, 50.0), "inout"),
    (C1 - 0.01, (-94.0, 3.6, 1.15, 0.86, 50.0)),
    (C1 + 0.01, (-182.0, 3.6, 1.25, 0.86, 50.0), "lin"),      # потерял 25 дней + сутки растянулись
    (C2 - 0.01, (-188.0, 3.5, 1.25, 0.86, 50.0), "lin"),
    (C2 + 0.01, (-272.0, 3.6, 0.7, 1.0, 50.0), "lin"),        # утро: кран вверх с восходом
    (37.9, (-278.0, 3.7, 2.0, 1.0, 50.0), "inout"),
    (38.2, (-460.0, 4.3, 1.3, 1.12, 50.0), "expo"), (DUR, (-461.0, 4.0, 1.3, 1.15, 50.0)),
]
OFFSET = [(0.0, -0.5), (2.7, -0.5), (2.95, 0.0, "inout"), (4.2, 0.0), (4.5, 0.55, "expo"), (C1 - 0.01, 0.55),
          (C1 + 0.01, -0.55, "lin"), (C2 - 0.01, -0.55), (C2 + 0.01, 0.55, "lin"), (37.9, 0.55), (38.2, 0.0, "expo"),
          (DUR, 0.0)]
ROLL = [(0.0, 0.0), (C1 + 0.01, 0.0), (19.4, 0.0), (21.0, -5.0, "inout"), (24.9, -5.0), (25.4, 0.0, "inout"), (DUR, 0.0)]
TINT = [(0.0, (1.0, 1.04, 1.1)), (2.7, (1.0, 1.04, 1.1)), (2.95, (1.0, 1.0, 1.0)), (4.2, (1.0, 1.0, 1.0)),
        (4.5, (1.0, 0.95, 0.9)), (C1 - 0.01, (1.0, 0.95, 0.9)), (C1 + 0.01, (1.12, 0.92, 1.05), "lin"),
        (C2 - 0.01, (1.12, 0.92, 1.05)), (C2 + 0.01, (0.85, 0.98, 1.15), "lin"), (37.9, (0.8, 1.0, 1.2)),
        (38.25, (1.0, 1.0, 1.0)), (DUR, (1.0, 1.0, 1.0))]
SEC = {"hook": 8, "cave": -91, "lost": -185, "sun": -275, "tg": -460.5}
SHAKES = (17.17,)
FLASHES = ((38.27, 0.12),)
JUMPS = [17.2]
ALARM = None
SPOT = None

G = lambda *k: list(k)  # noqa: E731
O = OUTRO - 48.035      # жесты концовки — как в monet_mg, со сдвигом
ARM_L = G((0, 62), (2.7, 62), (2.95, -50, "back"), (3.9, -45), (4.2, 62, "back"),
          (C1 + 0.05, 62), (12.3, -30, "back"), (15.8, -28), (16.1, 25, "back"), (17.1, 28), (17.2, -55, "back"),
          (18.6, -52), (18.9, -30, "back"), (24.9, -28), (25.2, 62), (33.2, 62), (33.4, -40, "back"), (34.0, -38),
          (34.1, -25, "back"), (37.6, -22), (37.9, 62),
          (54.94 + O, 62), (55.09 + O, -50, "back"), (55.69 + O, -45), (56.09 + O, 62, "back"), (DUR, 62))
ARM_R = G((0, 62), (0.3, -28, "back"), (2.6, -26), (2.75, 62), (2.95, -50, "back"), (3.9, -45), (4.2, 62, "back"),
          (4.6, -30, "back"), (7.5, -28), (7.7, -40, "back"), (9.3, -38), (9.5, -25, "back"), (11.6, -22),
          (11.8, 62), (16.1, 25, "back"), (17.1, 28), (17.2, -55, "back"), (18.6, -52), (18.9, 62),
          (C2 + 0.05, 62), (26.0, -30, "back"), (29.4, -28), (29.6, 30, "back"), (32.9, 32), (33.4, -40, "back"),
          (34.0, -38), (34.1, 62),
          (47.79 + O, 62), (48.25 + O, -48, "back"), (50.6 + O, -45), (50.9 + O, -55, "back"), (53.6 + O, -52),
          (53.85 + O, -40, "back"), (54.8 + O, -42), (54.95 + O, -50, "back"), (55.69 + O, -45), (55.84 + O, -58),
          (55.99 + O, -32), (56.14 + O, -58), (56.29 + O, -32), (DUR, 62))
HEAD = G((0, 0), (0.6, -6), (1.3, 7), (2.1, -5), (3.0, 0), (4.4, 6), (6.5, -6), (8.1, 7), (9.4, -6), (10.6, 5),
         (12.2, -6), (13.7, 6), (15.1, -7), (16.3, 6), (17.2, 10), (18.9, -5), (20.2, 7), (21.5, -6), (22.9, 6),
         (24.2, -5), (25.3, 0), (26.0, 6), (27.3, -6), (28.6, 7), (29.6, -6), (31.8, 8), (33.3, -5), (34.1, 7),
         (35.1, -6), (36.3, 6), (37.8, 0), (38.5, -6), (45.3, 0), (DUR, 0))


# ---------------------------------------------------------------- картинки вставок
def cal_tex(top, big, col=RED_RGB):
    im = rounded((360, 400), 30, (240, 236, 228, 255), (180, 175, 165, 255), 3)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, 359, 110], 30, fill=col + (255,))
    d.rectangle([0, 60, 359, 110], fill=col + (255,))
    for x in (90, 270):
        d.ellipse([x - 14, -10, x + 14, 18], fill=(60, 60, 70))
    sz = 50
    while mg.font(sz, tall=True).getlength(top) > 320:
        sz -= 2
    f = mg.font(sz, tall=True)
    tw = d.textbbox((0, 0), top, font=f)[2]
    d.text((180 - tw / 2, 34 + (50 - sz) // 2), top, font=f, fill="white")
    sz = 200
    while mg.font(sz, tall=True).getlength(big) > 320:
        sz -= 6
    f2 = mg.font(sz, tall=True)
    l, t_, r, b = d.textbbox((0, 0), big, font=f2)
    d.text((180 - (r + l) / 2, 255 - (b + t_) / 2), big, font=f2, fill=(30, 30, 40))
    return mg.premul(im)


def dial_tex(ang1, ang2, n=12, over=0.0, broken=False):
    """Циферблат: n делений, стрелки под углами ang1/ang2 (градусы, 0 — вверх); over — «лишний» красный кусок."""
    S = 460
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c, r = S / 2, 200
    d.ellipse([c - r, c - r, c + r, c + r], fill=(22, 22, 30, 255), outline=(200, 200, 215, 255), width=10)
    if over > 0:
        d.arc([c - r + 14, c - r + 14, c + r - 14, c + r - 14], -90, -90 + 360 * over, fill=RED_RGB + (255,), width=22)
    for i in range(n):
        a = math.radians(360 * i / n - 90)
        l = 26 if i % (n // 4) == 0 else 14
        d.line([(c + (r - 26) * math.cos(a), c + (r - 26) * math.sin(a)),
                (c + (r - 26 - l) * math.cos(a), c + (r - 26 - l) * math.sin(a))], fill=(200, 200, 215), width=6)
    for ang, ln, w in ((ang1, 150, 12), (ang2, 110, 16)):
        a = math.radians(ang - 90)
        d.line([(c, c), (c + ln * math.cos(a), c + ln * math.sin(a))], fill=GOLD_RGB + (255,), width=w)
    d.ellipse([c - 16, c - 16, c + 16, c + 16], fill=GOLD_RGB + (255,))
    if broken:
        d.line([(c - 120, c - 160), (c - 40, c - 40), (c - 90, c + 40), (c + 10, c + 170)], fill=RED_RGB + (255,), width=10)
    return mg.premul(im)


def sun_tex():
    S = 600
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for k in range(12):
        a = k * math.pi / 6
        d.line([(300 + 190 * math.cos(a), 300 + 190 * math.sin(a)), (300 + 280 * math.cos(a), 300 + 280 * math.sin(a))],
               fill=SUN_RGB + (255,), width=26)
    d.ellipse([130, 130, 470, 470], fill=(255, 200, 80, 255))
    return mg.premul(im)


def window_tex(open_):
    """Окно со шторами: open_ от 0 (закрыто) до 1 (шторы раздвинуты, за окном солнце)."""
    im = rounded((600, 520), 30, (30, 30, 40, 255), (150, 150, 165, 255), 8)
    d = ImageDraw.Draw(im)
    d.rectangle([40, 40, 560, 480], fill=(120, 190, 255))
    d.ellipse([360, 90, 480, 210], fill=(255, 210, 90))
    d.rectangle([40, 360, 560, 480], fill=(90, 160, 90))
    d.line([(300, 40), (300, 480)], fill=(200, 200, 215), width=10)
    d.line([(40, 260), (560, 260)], fill=(200, 200, 215), width=10)
    w = int(260 * (1 - open_)) + 20
    for x0, x1 in ((20, 20 + w), (580 - w, 580)):
        d.rectangle([x0, 20, x1, 500], fill=(150, 60, 70))
        for x in range(x0 + 20, x1, 40):
            d.line([(x, 20), (x, 500)], fill=(120, 45, 55), width=6)
    d.rectangle([10, 10, 590, 34], fill=(100, 90, 90))
    return mg.premul(im)


# ---------------------------------------------------------------- сцена
class Scene(base.Scene):
    cfg = sys.modules[__name__]

    def __init__(self, voice):
        super().__init__(voice)
        T = self.T
        T["cal_month"] = [cal_tex("ИЮЛЬ", str(d)) for d in (14, 15, 16, 17, 18, 19)]
        T["siffre"] = card("ОПЫТ", "МИШЕЛЬ СИФР, 1962", "2 месяца в пещере, Франция", TG_RGB)
        T["deep"] = chip("130 М ПОД ЗЕМЛЁЙ", PURPLE_RGB)
        T["thought"] = cal_tex("ДУМАЛ", "20.08", GRAY_RGB)
        T["real"] = cal_tex("НА САМОМ ДЕЛЕ", "14.09", RED_RGB)
        T["minus"] = stamp("−25 ДНЕЙ", RED_RGB)
        T["inner"] = chip("ВНУТРЕННИЕ ЧАСЫ", PURPLE_RGB)
        T["more24"] = chip("ЧУТЬ ДОЛЬШЕ 24 Ч", RED_RGB)
        T["sun"] = sun_tex()
        T["alarm"] = chip("СВЕТ = БУДИЛЬНИК", GOLD_RGB)
        T["broken"] = chip("РЕЖИМ ЛОМАЕТСЯ", RED_RGB)
        T["tip"] = card("СОВЕТ", "10 МИНУТ СВЕТА", "утром — шторы или улица", GREEN_RGB)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        self.ang = (np.degrees(np.arctan2(xx - W / 2, -(yy - H / 2))) + 360) % 360   # 0 — вверх, по часовой
        self.yy, self.xx = yy, xx
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
                      ty + 0.015 * wobble(t, 3), fov=fov, roll=0.5 * wobble(t * 0.8, 4) + 1.5 * shake + track(ROLL, t),
                      tx=off * math.cos(a), tz=off * math.sin(a))

    def finish(self, frame, t):
        fr = super().finish(frame, t)
        kc = pop(t, CAVE[0], CAVE[1], 0.5, 0.4)
        if kc > 0:  # пещера: темнота, светло только в круге фонарика, который ходит по кадру
            cx = W * (0.5 + 0.25 * math.sin(t * 0.9))
            cy = H * (0.42 + 0.12 * math.sin(t * 1.3 + 1))
            d2 = ((self.xx - cx) ** 2 + (self.yy - cy) ** 2) / (520.0 ** 2)
            light = np.clip(1.25 - d2, 0.0, 1.0)
            m = (1 - kc) + kc * (0.18 + 0.82 * light)
            fr = (fr.astype(np.float32) * m[..., None]).astype(np.uint8)
            mg.subtitles(fr, t, SUBS)                  # субтитры поверх темноты — не затемняются
        for w0 in WIPES:  # «стрелка часов»: кадр закрывается по кругу, под ним склейка, потом открывается
            p = (t - w0) / 0.4
            if 0 < p < 1:
                if p < 0.5:
                    s = smooth(p / 0.5) * 360
                    m = self.ang < s
                else:
                    s = smooth((p - 0.5) / 0.5) * 360
                    m = self.ang >= s
                fr = fr.copy()
                fr[m] = (24, 18, 30)
                a = math.radians(s)
                cv2.line(fr, (W // 2, H // 2), (int(W / 2 + 2000 * math.sin(a)), int(H / 2 - 2000 * math.cos(a))),
                         (60, 190, 245), 14, cv2.LINE_AA)
                cv2.circle(fr, (W // 2, H // 2), 22, (60, 190, 245), -1, cv2.LINE_AA)
        return fr

    def items_for(self, t, cam, it):
        T = self.T
        P = lambda y0, *a, **k: place(y0, *a, **k)  # noqa: E731

        # 1. Крючок (персонаж справа): листки календаря срываются, стрелки часов крутятся
        y0 = SEC["hook"]
        r_, c_ = basis(y0)
        k = pop(t, 0.2, 2.85, 0.45, 0.25)
        if k > 0:
            base_pos = P(y0, -0.75, 1.75)
            n = len(T["cal_month"])
            cur = min(n - 1, int(max(0.0, t - 0.6) / 0.38))
            self.add_panel(it, cam, base_pos, face(y0, -10), 0.4 * k, 0.444 * k, 0.04 * k, T["cal_month"][cur],
                           side=(150, 145, 135))
            if cur > 0:   # сорванный листок улетает
                pf = (max(0.0, t - 0.6) % 0.38) / 0.38
                q = base_pos + r_ * (0.5 * pf) + np.array([0, 0.35 * pf, 0]) + c_ * 0.05
                self.add_panel(it, cam, q, face(y0, -10 + 40 * pf), 0.4 * k, 0.444 * k, 0.02, T["cal_month"][cur - 1],
                               roll=-60 * pf, side=(150, 145, 135))
            sp = t * 900
            self.add_bb(it, cam, P(y0, -0.75, 1.0), 0.3 * k, self.cached("d0", dial_tex, round(sp % 360 / 6) * 6,
                                                                        round(sp / 12 % 360 / 6) * 6), anchor=(0.5, 0.5))

        # 2. Заставка
        k = pop(t, 2.95, 4.2, 0.35, 0.3)
        if k > 0:
            self.add_panel(it, cam, place(0, 0.0, 1.95, 0.35), face(0, -6), 1.2 * k, 0.564 * k, 0.11 * k,
                           T["stinger"], pitch=-6, side=(30, 24, 120), edge=(200, 190, 255))

        # 3. Пещера (персонаж слева, темнота и фонарик, камера опускается): Сифр → «130 м под землёй»
        y0 = SEC["cave"]
        k = pop(t, 6.4, 11.85, 0.45, 0.25)
        if k > 0:
            pos = P(y0, 0.72, 1.6) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 14), 0.9 * k, 0.427 * k, 0.06 * k, T["siffre"], pitch=-6,
                           side=(30, 60, 80), edge=(190, 230, 255))
            kd = pop(t, 9.4, 11.85, 0.35, 0.25)
            if kd > 0:
                self.add_bb(it, cam, P(y0, 0.72, 1.12), 0.1 * kd, T["deep"], anchor=(0.5, 0.5))

        # 4. Потерял 25 дней + сутки растянулись (персонаж справа): два календаря → штамп → циферблат на 24 часа
        y0 = SEC["lost"]
        k = pop(t, C1 + 0.05, 19.3, 0.4, 0.25)
        if k > 0:
            self.add_panel(it, cam, P(y0, -0.95, 1.7), face(y0, -10), 0.4 * k, 0.444 * k, 0.04 * k, T["thought"],
                           side=(150, 145, 135))
            kr = pop(t, 14.5, 19.3, 0.4, 0.25)
            if kr > 0:
                self.add_panel(it, cam, P(y0, -0.45, 1.7), face(y0, -14), 0.4 * kr, 0.444 * kr, 0.04 * kr, T["real"],
                               side=(150, 145, 135))
            if t > 17.17:
                ks = clamp((t - 17.17) / 0.15)
                self.add_bb(it, cam, P(y0, -0.7, 1.28, 0.1), 0.16 * (1 + 0.6 * (1 - ks)) * k, T["minus"],
                            anchor=(0.5, 0.5))
        k = pop(t, 19.4, AD_CUT + 0.3, 0.45, 0.2)
        if k > 0:
            over = clamp((t - 23.0) / 1.4) * 0.05
            hand = 360 * clamp((t - 19.8) / 3.2) + 360 * over
            self.add_bb(it, cam, P(y0, -0.72, 1.6) + [0, 0.02 * math.sin(t * 2), 0], 0.5 * k,
                        self.cached("d24", dial_tex, round(hand / 3) * 3, 0, 24, round(over * 100) / 100), anchor=(0.5, 0.5))
            ki = pop(t, 21.5, AD_CUT + 0.3, 0.35, 0.2)
            if ki > 0:
                self.add_bb(it, cam, P(y0, -0.72, 2.05), 0.1 * ki, T["inner"] if t < 22.9 else T["more24"],
                            anchor=(0.5, 0.5))

        # 5. Утро (персонаж слева, кран вверх): солнце встаёт → «свет = будильник» → сломанный режим → шторы
        y0 = SEC["sun"]
        r_, c_ = basis(y0)
        k = pop(t, C2 + 0.05, 33.2, 0.5, 0.3)
        if k > 0:
            rise = smooth((t - 25.9) / 2.0)
            self.add_bb(it, cam, P(y0, 0.75, 0.3 + 1.6 * rise, -1.2), 0.75 * k, T["sun"], anchor=(0.5, 0.5))
            ka = pop(t, 27.3, 29.55, 0.35, 0.25)
            if ka > 0:
                self.add_bb(it, cam, P(y0, 0.72, 1.0), 0.1 * ka, T["alarm"], anchor=(0.5, 0.5))
            kb = pop(t, 29.6, 33.2, 0.35, 0.25)
            if kb > 0:
                self.add_bb(it, cam, P(y0, 0.72, 1.05), 0.36 * kb,
                            self.cached("db", dial_tex, 130, 300, 12, 0.0, t > 31.8), anchor=(0.5, 0.5))
                if t > 31.8:
                    self.add_bb(it, cam, P(y0, 0.72, 0.72), 0.09 * kb, T["broken"], anchor=(0.5, 0.5))
        k = pop(t, 33.25, 37.95, 0.4, 0.25)
        if k > 0:
            op = round(smooth((t - 34.1) / 1.0) * 10) / 10
            self.add_panel(it, cam, P(y0, 0.72, 1.75) + [0, 0.02 * math.sin(t * 2), 0], face(y0, 12), 0.62 * k,
                           0.537 * k, 0.05 * k, self.cached("win", window_tex, op), side=(60, 60, 70))
            kt = pop(t, 35.1, 37.95, 0.35, 0.25)
            if kt > 0:
                self.add_panel(it, cam, P(y0, 0.72, 1.05), face(y0, 12), 0.75 * kt, 0.356 * kt, 0.05 * kt, T["tip"],
                               pitch=-6, side=(30, 70, 40), edge=(190, 255, 210))

        # 6. Концовка с Telegram — общий шаблон
        self.outro_items(t, cam, it, OUTRO, SEC["tg"])


SFX = [(t, "whoosh") for t in (0.0, 2.9, 4.4) + WIPES + (38.1,)]


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
        for t in (0.6, 1.8, 3.5, 5.5, 8.0, 10.5, 11.95, 13.0, 15.5, 17.5, 20.5, 23.5, 25.0, 25.7, 27.0, 28.5, 31.0,
                  32.5, 34.5, 36.5, 41.0):
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
    base.mix_audio(tmp, voice, out, music=os.path.join(ROOT, "work", "clock", "music.wav"), sfx_list=SFX, dur=DUR,
                   sfx_vol=0.15)  # «вжух» лёгкий и тихий
    os.remove(tmp)


if __name__ == "__main__":
    main()
