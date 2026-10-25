"""Ролик №18: «Что будет, если все деньги мира раздать поровну?» (+ рекламный баннер в середине).

python tools/wealth_mg.py work/wealth/voice_clean.wav output/wealth.mp4 [--preview] [--jobs 4]
затем: python tools/ad_banner.py insert output/wealth.mp4 26.78 output/wealth_ad.mp4 wealth_mg
Общий движок, персонаж и концовка с Telegram — из memory_mg.py; параллельный рендер — как в money_mg.py.
Новое в монтаже: круглый «пирог» богатства режется на равные кусочки; бегущий счётчик «$450 трлн» и пример
«÷ 8 млрд = ~$55 000», который собирается по частям; камера скользит вбок, как по рельсам; смена сцены
«денежным дождём» — купюры сыплются и закрывают кадр; дом, земля и завод — 3D-блоки, а наличные — крошечная
пачка; завод разлетается на кусочки; ценник растёт; «вложил / потратил» и столбики, которые из равных снова
становятся разными.
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
OUTRO = float(open(os.path.join(ROOT, "work", "wealth", "outro_start.txt")).read())  # 41.92
DUR = 50.6
AD_CUT = 26.78                   # здесь вклеивается баннер (камера стоит спокойно)
GOLD_RGB, GREEN_RGB, PURPLE_RGB, MONEY_RGB = (245, 190, 50), (46, 196, 110), (150, 90, 230), (90, 170, 90)

# ---------------------------------------------------------------- субтитры
T_, Go_, G_, Pu_ = TG_RGB, GOLD_RGB, GREEN_RGB, PURPLE_RGB
SUBS = [
    (0.154, "ЧТО БУДЕТ, ЕСЛИ ВСЕ", ()), (1.278, "ДЕНЬГИ МИРА РАЗДАТЬ", ("ДЕНЬГИ",), Go_),
    (2.25, "ПОРОВНУ?", ("ПОРОВНУ?",), Go_), (2.95, "", ()),
    (4.421, "ВСЁ БОГАТСТВО ЛЮДЕЙ", ()), (5.392, "НА ЗЕМЛЕ — ПРИМЕРНО", ()), (6.383, "450 ТРИЛЛИОНОВ ДОЛЛАРОВ", ("450",), Go_),
    (9.01, "ЕСЛИ РАЗДЕЛИТЬ НА ВСЕХ,", ()),
    (10.478, "КАЖДОМУ ДОСТАЛОСЬ БЫ", ()), (11.43, "ОКОЛО 55 ТЫСЯЧ", ("55",), G_),
    (13.629, "НО НАЛИЧНЫХ ДЕНЕГ", ("НАЛИЧНЫХ",)), (14.674, "СРЕДИ ЭТОГО СОВСЕМ МАЛО", ("МАЛО",)),
    (16.465, "БОЛЬШАЯ ЧАСТЬ БОГАТСТВА —", ()), (17.97, "ЭТО ДОМА, ЗЕМЛЯ", ("ДОМА,", "ЗЕМЛЯ"), Pu_),
    (19.074, "И ЗАВОДЫ", ("ЗАВОДЫ",), Pu_), (20.274, "ПРИШЛОСЬ БЫ РАЗДАВАТЬ", ()),
    (21.208, "ЛЮДЯМ КУСОЧКИ ФАБРИК", ("ФАБРИК",), Pu_), (22.895, "А ЕСЛИ У ВСЕХ ВДРУГ", ()),
    (23.846, "СТАНЕТ БОЛЬШЕ ДЕНЕГ,", ("БОЛЬШЕ",), G_), (25.427, "ЦЕНЫ БЫСТРО ВЫРАСТУТ", ("ВЫРАСТУТ",)),
    (AD_CUT - 0.02, "", ()),
    (26.924, "И ГЛАВНОЕ —", ()), (27.865, "ЛЮДИ ТРАТЯТ ПО-РАЗНОМУ", ("ПО-РАЗНОМУ",), Go_),
    (29.465, "ОДИН ВЛОЖИТ ДЕНЬГИ В ДЕЛО,", ("ВЛОЖИТ",), G_),
    (31.027, "ДРУГОЙ ПОТРАТИТ ЗА МЕСЯЦ", ("ПОТРАТИТ",)), (32.762, "МНОГИЕ ЭКОНОМИСТЫ", ()),
    (33.789, "СЧИТАЮТ, ЧТО ЧЕРЕЗ", ()), (34.912, "НЕСКОЛЬКО ЛЕТ РАЗНИЦА", ("РАЗНИЦА",), Go_),
    (36.17, "МЕЖДУ БОГАТЫМИ И БЕДНЫМИ", ()), (37.484, "ВЕРНУЛАСЬ БЫ", ("ВЕРНУЛАСЬ",)),
    (38.636, "ДЕНЬГИ МОЖНО РАЗДАТЬ,", ("РАЗДАТЬ,",), G_), (40.103, "А ПРИВЫЧКИ — НЕТ", ("ПРИВЫЧКИ",), Pu_),
    (41.882, "А ЕЩЁ БОЛЬШЕ ИНТЕРЕСНОГО —", (), T_), (43.242, "В МОЁМ ТЕЛЕГРАМ-КАНАЛЕ", ("ТЕЛЕГРАМ-КАНАЛЕ",), T_),
    (44.572, "«ПАУЧЬЯ НОРКА»", ("«ПАУЧЬЯ", "НОРКА»"), T_), (45.692, "ТАМ ЭКСКЛЮЗИВ,", ("ЭКСКЛЮЗИВ",), T_),
    (46.392, "КОТОРОГО НЕТ В РОЛИКАХ", (), T_), (47.692, "ССЫЛКА В ПРОФИЛЕ —", ("ПРОФИЛЕ",), T_),
    (48.712, "ЗАЛЕТАЙ!", ("ЗАЛЕТАЙ!",), T_),
]

# ---------------------------------------------------------------- камера и сцены
RAIN = (13.4, 26.8)              # «денежный дождь»: купюры закрывают кадр, под ними склейка
C1, C2 = (r + 0.25 for r in RAIN)
CAMERA = [
    (0.0, (-10.0, 4.2, 1.45, 0.9, 50.0)), (0.8, (-6.0, 3.6, 1.15, 0.84, 50.0), "out"),
    (2.8, (-5.0, 3.6, 1.15, 0.84, 50.0)),
    (3.05, (0.0, 3.9, 1.3, 0.98, 50.0), "inout"),             # заставка
    (4.3, (1.0, 3.9, 1.3, 0.98, 50.0)),
    (4.6, (95.0, 3.7, 1.3, 0.86, 50.0), "expo"),              # цифры — камера скользит вбок (см. OFFSET)
    (C1 - 0.01, (97.0, 3.6, 1.3, 0.86, 50.0), "lin"),
    (C1 + 0.01, (190.0, 3.8, 1.9, 0.9, 50.0), "lin"),         # дома, земля, заводы — чуть сверху
    (C2 - 0.01, (198.0, 3.7, 1.8, 0.9, 50.0), "lin"),
    (C2 + 0.01, (282.0, 3.6, 1.25, 0.86, 50.0), "lin"),       # тратят по-разному
    (38.3, (288.0, 3.5, 1.25, 0.86, 50.0), "lin"),
    (38.65, (375.0, 3.7, 1.0, 0.95, 50.0), "expo"),           # финал
    (41.5, (379.0, 3.4, 1.0, 0.95, 49.0), "lin"),
    (41.9, (460.0, 4.3, 1.3, 1.12, 50.0), "expo"), (DUR, (461.0, 4.0, 1.3, 1.15, 50.0)),
]
OFFSET = [(0.0, -0.5), (2.8, -0.5), (3.05, 0.0, "inout"), (4.3, 0.0), (4.6, 0.75, "expo"),
          (C1 - 0.01, 0.4, "lin"),                            # скольжение вбок по «рельсам»
          (C1 + 0.01, -0.55, "lin"), (C2 - 0.01, -0.55), (C2 + 0.01, 0.55, "lin"), (38.3, 0.55), (38.65, 0.0, "expo"),
          (DUR, 0.0)]
TINT = [(0.0, (0.88, 1.06, 1.1)), (2.8, (0.88, 1.06, 1.1)), (3.05, (1.0, 1.0, 1.0)), (4.3, (1.0, 1.0, 1.0)),
        (4.6, (0.92, 1.1, 0.94)), (C1 - 0.01, (0.92, 1.1, 0.94)), (C1 + 0.01, (1.12, 0.95, 0.95), "lin"),
        (C2 - 0.01, (1.12, 0.95, 0.95)), (C2 + 0.01, (0.85, 0.98, 1.15), "lin"), (38.3, (0.85, 0.98, 1.15)),
        (38.65, (1.08, 0.92, 1.08)), (41.5, (1.08, 0.92, 1.08)), (41.95, (1.0, 1.0, 1.0)), (DUR, (1.0, 1.0, 1.0))]
SEC = {"hook": -6, "num": 96, "assets": 194, "habit": 285, "final": 377, "tg": 460.5}
SHAKES = (11.45,)
FLASHES = ((2.25, 0.08), (41.95, 0.12))
JUMPS = [11.5]
ALARM = None
SPOT = None

G = lambda *k: list(k)  # noqa: E731
O = OUTRO - 48.035      # жесты концовки — как в monet_mg, со сдвигом
ARM_L = G((0, 62), (0.3, -25, "back"), (2.7, -22), (2.9, 62), (3.1, -50, "back"), (4.0, -45), (4.3, 62, "back"),
          (13.4, 62), (13.7, -30, "back"), (16.2, -28), (16.5, -38, "back"), (20.0, -35), (20.3, -25, "back"),
          (22.7, -22), (22.9, 35, "back"), (25.2, 38), (25.4, -45, "back"), (26.4, -42), (26.6, 62),
          (38.6, 62), (38.8, -35, "back"), (40.0, -33), (40.2, 25, "back"), (41.6, 28), (41.8, 62),
          (54.94 + O, 62), (55.09 + O, -50, "back"), (55.69 + O, -45), (56.09 + O, 62, "back"), (DUR, 62))
ARM_R = G((0, 62), (2.9, 62), (3.1, -50, "back"), (4.0, -45), (4.3, 62, "back"), (4.7, -30, "back"), (6.2, -28),
          (6.4, -45, "back"), (8.8, -42), (9.0, -25, "back"), (11.3, -22), (11.45, -55, "back"), (12.6, -52),
          (12.9, 62), (22.9, 35, "back"), (25.2, 38), (25.4, 62), (C2 + 0.05, 62), (27.0, -30, "back"),
          (29.3, -28), (29.5, -45, "back"), (30.9, -42), (31.0, 30, "back"), (32.5, 32), (32.8, -30, "back"),
          (37.4, -28), (37.5, -40, "back"), (38.3, -38), (38.6, 62), (40.2, 25, "back"), (41.6, 28), (41.8, 62),
          (47.79 + O, 62), (48.25 + O, -48, "back"), (50.6 + O, -45), (50.9 + O, -55, "back"), (53.6 + O, -52),
          (53.85 + O, -40, "back"), (54.8 + O, -42), (54.95 + O, -50, "back"), (55.69 + O, -45), (55.84 + O, -58),
          (55.99 + O, -32), (56.14 + O, -58), (56.29 + O, -32), (DUR, 62))
HEAD = G((0, 0), (0.6, -6), (1.4, 7), (2.3, -5), (3.1, 0), (4.5, 6), (6.4, -7), (8.0, 6), (9.0, -5), (10.5, 6),
         (11.5, 9), (13.0, -4), (13.7, -6), (15.0, 6), (16.5, -6), (18.0, 7), (19.1, -6), (20.3, 6), (21.6, -7),
         (22.9, 6), (24.0, -5), (25.4, 9), (26.6, 0), (27.0, -6), (28.0, 7), (29.5, -6), (31.0, 8), (32.8, -5),
         (35.0, 7), (37.5, -7), (38.7, 6), (40.1, -8), (41.5, 0), (42.1, -6), (49.0, 0), (DUR, 0))


# ---------------------------------------------------------------- картинки вставок
def pie_tex(n, gap):
    """«Пирог» богатства: n равных кусочков, раздвинутых на gap."""
    S = 520
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    cols = [GOLD_RGB, (225, 165, 40), (250, 205, 90), (210, 150, 30)]
    r = 200
    for i in range(n):
        a0, a1 = 360 * i / n, 360 * (i + 1) / n
        am = math.radians((a0 + a1) / 2)
        cx, cy = S / 2 + gap * math.cos(am), S / 2 + gap * math.sin(am)
        d.pieslice([cx - r, cy - r, cx + r, cy + r], a0, a1, fill=cols[i % 4] + (255,), outline=(120, 80, 10, 255),
                   width=4)
    f = mg.font(110, tall=True)
    tw = d.textbbox((0, 0), "$", font=f)[2]
    d.text((S / 2 - tw / 2, S / 2 - 70), "$", font=f, fill=(120, 80, 10))
    return mg.premul(im)


def counter_tex(v):
    im = rounded((820, 300), 34, (16, 22, 18, 255), MONEY_RGB + (255,), 5)
    d = ImageDraw.Draw(im)
    d.text((40, 26), "ВСЁ БОГАТСТВО ЛЮДЕЙ", font=mg.font(40, tall=True), fill=(190, 220, 190))
    txt = f"${v} ТРЛН"
    f = mg.font(150, tall=True)
    tw = d.textbbox((0, 0), txt, font=f)[2]
    d.text((410 - tw / 2, 90), txt, font=f, fill=GOLD_RGB + (255,))
    return mg.premul(im)


def equation_tex(step):
    """Пример собирается по частям: $450 трлн → ÷ 8 млрд людей → = около $55 000."""
    im = rounded((820, 420), 34, (18, 18, 26, 255), (72, 72, 92, 255), 3)
    d = ImageDraw.Draw(im)
    f = mg.font(84, tall=True)
    d.text((40, 30), "$450 трлн", font=f, fill="white")
    if step >= 1:
        d.text((40, 150), "÷ 8 млрд людей", font=f, fill=(170, 200, 255))
    if step >= 2:
        d.rounded_rectangle([30, 270, 790, 390], 30, fill=GREEN_RGB + (255,))
        d.text((56, 284), "~ $55 000 КАЖДОМУ", font=mg.font(78, tall=True), fill="white")
    return mg.premul(im)


def note_tex():
    """Купюра для «денежного дождя»."""
    im = Image.new("RGBA", (300, 140), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, 299, 139], 18, fill=(70, 140, 80, 255), outline=(40, 90, 50, 255), width=6)
    d.ellipse([110, 30, 190, 110], outline=(40, 90, 50, 255), width=6)
    d.text((134, 36), "$", font=mg.font(56, tall=True), fill=(40, 90, 50))
    d.text((20, 12), "100", font=mg.font(28, tall=True), fill=(40, 90, 50))
    d.text((236, 96), "100", font=mg.font(28, tall=True), fill=(40, 90, 50))
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGBA2BGRA)


def icon_label(text, col):
    return mg.premul(mg.text_rgba(text, 70, col, tall=True))


def price_tex(v):
    im = Image.new("RGBA", (460, 260), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.polygon([(0, 130), (80, 0), (459, 0), (459, 259), (80, 259)], fill=(28, 26, 34, 255))
    d.line([(0, 130), (80, 0), (459, 0), (459, 259), (80, 259), (0, 130)], fill=RED_RGB + (255,), width=8)
    d.ellipse([40, 112, 76, 148], fill=(80, 80, 96))
    d.text((110, 26), "ЦЕНА", font=mg.font(44, tall=True), fill=(180, 180, 195))
    d.text((110, 90), str(v), font=mg.font(120, tall=True), fill="white")
    x = 110 + d.textbbox((0, 0), str(v), font=mg.font(120, tall=True))[2] + 18
    d.polygon([(x, 200), (x + 34, 140), (x + 68, 200)], fill=RED_RGB + (255,))
    d.rectangle([x + 22, 196, x + 46, 226], fill=RED_RGB + (255,))
    return mg.premul(im)


def two_ways_tex(p, q):
    """«ВЛОЖИЛ» — линия растёт, «ПОТРАТИЛ» — кошелёк пустеет (p, q — от 0 до 1)."""
    im = rounded((860, 420), 34, (18, 18, 26, 255), (72, 72, 92, 255), 3)
    d = ImageDraw.Draw(im)
    for x, title, col in ((30, "ВЛОЖИЛ", GREEN_RGB), (450, "ПОТРАТИЛ", RED_RGB)):
        d.rounded_rectangle([x, 30, x + 380, 390], 30, fill=(26, 26, 36, 255), outline=col + (255,), width=5)
        f = mg.font(60, tall=True)
        tw = d.textbbox((0, 0), title, font=f)[2]
        d.text((x + 190 - tw / 2, 50), title, font=f, fill=col + (255,))
    if p > 0:  # линия роста
        pts = [(70 + 300 * i / 20, 350 - 180 * (i / 20) ** 1.6 - 14 * math.sin(i)) for i in range(int(20 * p) + 1)]
        if len(pts) > 1:
            d.line(pts, fill=GREEN_RGB + (255,), width=10, joint="curve")
    v = 1 - q  # кошелёк и монеты, которые улетают
    d.rounded_rectangle([540, 210, 780, 340], 26, fill=(110, 80, 60, 255))
    d.rounded_rectangle([700, 250, 790, 300], 14, fill=(80, 56, 40, 255))
    for i in range(int(round(5 * v))):
        d.ellipse([560 + i * 34, 160 - (i % 2) * 14, 610 + i * 34, 210 - (i % 2) * 14], fill=GOLD_RGB + (255,))
    return mg.premul(im)


def final_tex():
    im = rounded((860, 380), 40, (22, 16, 26, 255), PURPLE_RGB + (255,), 5)
    d = ImageDraw.Draw(im)
    f = mg.font(84, tall=True)
    d.text((44, 40), "ДЕНЬГИ — РАЗДАТЬ", font=f, fill=GOLD_RGB + (255,))
    d.text((44, 150), "МОЖНО,", font=f, fill="white")
    l, t_, r, b = d.textbbox((0, 0), "ПРИВЫЧКИ — НЕТ", font=f)
    d.rounded_rectangle([40, 260, 40 + r - l + 52, 260 + b - t_ + 36], 30, fill=PURPLE_RGB + (255,))
    d.text((66 - l, 278 - t_), "ПРИВЫЧКИ — НЕТ", font=f, fill="white")
    return mg.premul(im)


# ---------------------------------------------------------------- сцена
class Scene(base.Scene):
    cfg = sys.modules[__name__]

    def __init__(self, voice):
        super().__init__(voice)
        T = self.T
        T["note"] = note_tex()
        T["cash"] = chip("НАЛИЧНЫЕ — ЧУТЬ-ЧУТЬ", GOLD_RGB)
        T["l_house"], T["l_land"], T["l_plant"] = (icon_label(s, (230, 230, 240)) for s in ("ДОМА", "ЗЕМЛЯ", "ЗАВОДЫ"))
        T["l_cash"] = icon_label("НАЛИЧНЫЕ", GOLD_RGB)
        T["piece"] = chip("ТЕБЕ — КУСОЧЕК ЗАВОДА", PURPLE_RGB)
        T["eco"] = card("МНЕНИЕ", "ЭКОНОМИСТОВ", "через несколько лет", GOLD_RGB)
        T["final"] = final_tex()
        rng = np.random.default_rng(18)
        self.rain = [(rng.uniform(-150, W + 150), rng.uniform(0, 1), rng.uniform(-40, 40), rng.uniform(0.7, 1.2))
                     for _ in range(150)]
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
        for r0 in RAIN:  # «денежный дождь»: купюры падают сверху, закрывают кадр и улетают вниз
            p = (t - r0) / 0.5
            if 0 < p < 1:
                fr = fr.copy()
                for i, (x, ph, ang, sc) in enumerate(self.rain):
                    y = -350 + (H + 700) * (1.8 * p - 0.8 * ph)
                    if -200 < y < H + 200:
                        w = int(300 * sc * 1.9)
                        note = cv2.resize(self.T["note"], (w, int(140 * sc * 1.9)))
                        M = cv2.getRotationMatrix2D((w / 2, note.shape[0] / 2), ang + 200 * p * (1 if i % 2 else -1), 1)
                        rot = cv2.warpAffine(note, M, (w, note.shape[0]))
                        x0, y0 = int(x - w / 2), int(y)
                        xa, xb = max(0, x0), min(W, x0 + w)
                        ya, yb = max(0, y0), min(H, y0 + rot.shape[0])
                        if xa < xb and ya < yb:
                            part = rot[ya - y0:yb - y0, xa - x0:xb - x0].astype(np.float32)
                            al = part[..., 3:4] / 255
                            fr[ya:yb, xa:xb] = (fr[ya:yb, xa:xb] * (1 - al) + part[..., :3] * al).astype(np.uint8)
        return fr

    def items_for(self, t, cam, it):
        T = self.T
        P = lambda y0, *a, **k: place(y0, *a, **k)  # noqa: E731

        # 1. Крючок (персонаж справа): «пирог» богатства режется на равные кусочки
        y0 = SEC["hook"]
        k = pop(t, 0.2, 2.95, 0.45, 0.25)
        if k > 0:
            n = 1 if t < 1.6 else 2 if t < 1.85 else 4 if t < 2.1 else 8
            gap = 0 if n == 1 else 26 * smooth((t - 1.6) / 0.6)
            self.add_bb(it, cam, P(y0, -0.7, 1.7) + [0, 0.02 * math.sin(t * 3), 0], 0.62 * k,
                        self.cached("pie", pie_tex, n, round(gap)), anchor=(0.5, 0.5))

        # 2. Заставка
        k = pop(t, 3.05, 4.3, 0.35, 0.3)
        if k > 0:
            self.add_panel(it, cam, place(0, 0.0, 1.95, 0.35), face(0, -6), 1.2 * k, 0.564 * k, 0.11 * k,
                           T["stinger"], pitch=-6, side=(30, 24, 120), edge=(200, 190, 255))

        # 3. Цифры (персонаж слева, камера скользит вбок): счётчик → пример по частям
        y0 = SEC["num"]
        k = pop(t, 4.5, 9.0, 0.4, 0.25)
        if k > 0:
            v = int(round(450 * smooth((t - 5.9) / 1.3)))
            pos = P(y0, 0.72, 1.62) + [0, 0.02 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 14), 0.9 * k, 0.329 * k, 0.06 * k, self.cached("cnt", counter_tex, v),
                           pitch=-6, side=(30, 60, 40), edge=(190, 255, 210))
        k = pop(t, 9.0, C1, 0.4, 0.2)
        if k > 0:
            step = 0 if t < 9.9 else 1 if t < 11.43 else 2
            pos = P(y0, 0.72, 1.62) + [0, 0.02 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 14), 0.9 * k, 0.461 * k, 0.06 * k, self.cached("eq", equation_tex, step),
                           pitch=-6, side=(50, 50, 70), edge=(220, 220, 255))

        # 4. Из чего богатство (персонаж справа, чуть сверху): наличные — крошка; дом, земля, завод — блоки;
        #    завод разлетается на кусочки; ценник растёт
        y0 = SEC["assets"]
        r_, c_ = basis(y0)
        if C1 < t < C2:
            kc = pop(t, C1 + 0.05, 20.2, 0.4, 0.3)
            if kc > 0:
                q = P(y0, -1.05, 0.0, -0.1)
                self.add_box(it, cam, (q[0], q[2]), (0.14, 0.08), 0.05 * kc, (70, 140, 80), radius=0.02)
                self.add_bb(it, cam, q + [0, 0.12, 0], 0.07 * kc, T["l_cash"], anchor=(0.5, 1.0))
            kt = pop(t, 13.9, 16.4, 0.35, 0.25)
            if kt > 0:
                self.add_bb(it, cam, P(y0, -0.7, 2.25), 0.1 * kt, T["cash"], anchor=(0.5, 0.5))
            for i, (sd, fw, w, hh, key, t0) in enumerate(((-0.95, -0.9, 0.5, 0.8, "l_house", 17.9),
                                                          (-0.45, -1.3, 0.7, 0.06, "l_land", 18.6),
                                                          (-1.1, -2.0, 0.6, 1.2, "l_plant", 19.2))):
                kb = pop(t, t0, 22.8, 0.45, 0.3)
                if kb <= 0:
                    continue
                q = P(y0, sd, 0.0, fw)
                col = ((70, 80, 120), (60, 110, 70), (90, 90, 100))[i]
                if key == "l_plant" and t > 20.9:   # завод разлетается на кусочки
                    sp = smooth((t - 20.9) / 0.7)
                    for j in range(6):
                        a = j * math.pi / 3
                        qq = q + r_ * (0.4 * sp * (math.cos(a) - 0.6)) + c_ * (0.35 * sp * (math.sin(a) - 0.5))
                        self.add_box(it, cam, (qq[0], qq[2]), (0.2, 0.2), hh * kb * (0.35 + 0.1 * (j % 2)),
                                     col, radius=0.03, edge=(200, 200, 220))
                else:
                    self.add_box(it, cam, (q[0], q[2]), (w, w * 0.8), max(0.02, hh * kb), col, radius=0.05,
                                 edge=(200, 200, 220))
                self.add_bb(it, cam, q + [0, max(0.06, hh) * kb + 0.12, 0], 0.08 * kb, T[key], anchor=(0.5, 1.0))
            kp = pop(t, 21.2, 22.8, 0.35, 0.25)
            if kp > 0:
                self.add_bb(it, cam, P(y0, -0.7, 2.25), 0.1 * kp, T["piece"], anchor=(0.5, 0.5))
            k = pop(t, 22.85, AD_CUT + 0.3, 0.4, 0.2)
            if k > 0:
                v = 100 + int(round(150 * smooth((t - 25.4) / 1.0) / 10)) * 10
                self.add_bb(it, cam, P(y0, -0.7, 1.75) + [0, 0.02 * math.sin(t * 2), 0], 0.36 * k,
                            self.cached("price", price_tex, v), anchor=(0.5, 0.5))

        # 5. Тратят по-разному (персонаж слева): «вложил / потратил» → мнение экономистов → столбики
        y0 = SEC["habit"]
        k = pop(t, C2 + 0.05, 32.7, 0.4, 0.25)
        if k > 0:
            p = round(clamp((t - 29.6) / 1.4) * 20) / 20
            q = round(clamp((t - 31.2) / 1.0) * 5) / 5
            pos = P(y0, 0.72, 1.6) + [0, 0.02 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 14), 0.95 * k, 0.464 * k, 0.06 * k, self.cached("tw", two_ways_tex, p, q),
                           pitch=-6, side=(50, 50, 70), edge=(220, 220, 255))
        k = pop(t, 32.75, 35.5, 0.4, 0.25)
        if k > 0:
            pos = P(y0, 0.72, 1.85) + [0, 0.02 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 14), 0.9 * k, 0.427 * k, 0.06 * k, T["eco"], pitch=-6,
                           side=(70, 60, 30), edge=(255, 230, 170))
        k = pop(t, 34.9, 38.6, 0.4, 0.25)
        if k > 0:
            e = smooth((t - 36.3) / 1.6)
            for i, hh in enumerate((0.45, 1.25, 0.25, 0.8, 0.35)):
                q = P(y0, 0.4 + 0.22 * i, 0.0, -0.6)
                h = (0.6 * (1 - e) + hh * e) * k
                self.add_box(it, cam, (q[0], q[2]), (0.16, 0.16), h, (60, 120, 70) if hh > 0.6 else (110, 70, 70),
                             radius=0.04, edge=(220, 230, 220))

        # 6. Финал (персонаж в центре): «ДЕНЬГИ — РАЗДАТЬ МОЖНО, ПРИВЫЧКИ — НЕТ»
        y0 = SEC["final"]
        k = pop(t, 38.7, 41.55, 0.45, 0.25)
        if k > 0:
            pos = place(y0, 0.0, 1.9, 0.2) + [0, 0.02 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 0), 0.85 * k, 0.376 * k, 0.06 * k, T["final"], pitch=-6,
                           side=(60, 40, 80), edge=(220, 200, 255))

        # 7. Концовка с Telegram — общий шаблон
        self.outro_items(t, cam, it, OUTRO, SEC["tg"])


SFX = [(t, "whoosh") for t in (0.0, 3.0, 4.5) + RAIN + (38.6, 41.8)]


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
        for t in (0.6, 2.4, 3.8, 6.0, 7.5, 10.5, 12.5, 13.65, 15.0, 18.5, 20.5, 21.6, 24.0, 26.3, 26.95, 28.5,
                  30.5, 32.0, 34.0, 37.8, 40.0, 45.0):
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
    base.mix_audio(tmp, voice, out, music=os.path.join(ROOT, "work", "wealth", "music.wav"), sfx_list=SFX, dur=DUR,
                   sfx_vol=0.15)  # «вжух» лёгкий и тихий
    os.remove(tmp)


if __name__ == "__main__":
    main()
