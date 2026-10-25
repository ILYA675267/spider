"""Ролик №17: «Что будет, если спать всего по 4 часа в сутки?» (+ рекламный баннер в середине).

python tools/sleep_mg.py work/sleep/voice_clean.wav output/sleep.mp4 [--preview] [--jobs 4]
затем: python tools/ad_banner.py insert output/sleep.mp4 28.6 output/sleep_ad.mp4 sleep_mg
Общий движок, персонаж и концовка с Telegram — из memory_mg.py; параллельный рендер — как в money_mg.py.
Новое в монтаже: ночная комната и будильник «04:00» с летящими «Z»; сравнение «14 дней по 4 часа = 2 ночи без сна»;
смена сцены «морганием» — сверху и снизу смыкаются веки; сонная камера плавно покачивается; два прибора
«кажется / на самом деле»; щит организма трескается, вирусов становится в 4 раза больше; батарейка заряжается,
а вторая — мигает на последних процентах.
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
OUTRO = float(open(os.path.join(ROOT, "work", "sleep", "outro_start.txt")).read())  # 41.74
DUR = 50.4
AD_CUT = 28.6                    # здесь вклеивается баннер (камера стоит спокойно)
GOLD_RGB, GREEN_RGB, PURPLE_RGB, NIGHT_RGB = (245, 190, 50), (46, 196, 110), (150, 90, 230), (90, 120, 255)

# ---------------------------------------------------------------- субтитры
T_, Go_, G_, Pu_, N_ = TG_RGB, GOLD_RGB, GREEN_RGB, PURPLE_RGB, NIGHT_RGB
SUBS = [
    (0.171, "ЧТО БУДЕТ, ЕСЛИ СПАТЬ", ()), (1.711, "ВСЕГО ПО 4 ЧАСА В СУТКИ?", ("4", "ЧАСА"), N_), (3.4, "", ()),
    (5.311, "УЧЁНЫЕ ПРОВЕРИЛИ ЭТО:", ()), (7.251, "ДВЕ НЕДЕЛИ ЛЮДИ", ()), (8.151, "СПАЛИ ПО 4 ЧАСА", ("4", "ЧАСА"), N_),
    (10.201, "В ИТОГЕ ИХ ВНИМАНИЕ", ()), (11.361, "УПАЛО ТАК ЖЕ, КАК", ("УПАЛО",)),
    (12.581, "ПОСЛЕ ДВУХ БЕССОННЫХ", ("ДВУХ",)), (14.021, "НОЧЕЙ ПОДРЯД", ()),
    (15.25, "САМОЕ ОПАСНОЕ —", ("ОПАСНОЕ",)), (16.515, "САМИ ЛЮДИ ПОЧТИ", ()), (17.375, "ЭТОГО НЕ ЗАМЕЧАЛИ", ("НЕ", "ЗАМЕЧАЛИ")),
    (18.875, "ИМ КАЗАЛОСЬ, ЧТО ОНИ", ()), (19.875, "ПРОСТО НЕМНОГО УСТАЛИ", ("УСТАЛИ",), Pu_),
    (21.58, "А ЕЩЁ НЕДОСЫП", ()), (22.455, "ОСЛАБЛЯЕТ ЗАЩИТУ", ("ОСЛАБЛЯЕТ",)), (23.015, "ОРГАНИЗМА", ()),
    (24.455, "КТО СПИТ МЕНЬШЕ 6 ЧАСОВ,", ("6",), N_), (26.335, "ПРОСТУЖАЕТСЯ", ()), (26.975, "В 4 РАЗА ЧАЩЕ", ("4", "РАЗА")),
    (AD_CUT - 0.02, "", ()),
    (28.83, "ХОРОШАЯ НОВОСТЬ —", ("ХОРОШАЯ",), G_), (29.985, "ОРГАНИЗМ МОЖНО", ()), (30.805, "ВЕРНУТЬ В НОРМУ", ("НОРМУ",), G_),
    (32.105, "ВЗРОСЛОМУ НУЖНО", ()), (32.985, "7–9 ЧАСОВ СНА", ("7–9",), G_),
    (34.95, "ТАК ЧТО СОН — ЭТО", ()), (36.025, "НЕ ПОТЕРЯННОЕ ВРЕМЯ,", ()), (37.405, "А ПОДЗАРЯДКА,", ("ПОДЗАРЯДКА,",), G_),
    (38.345, "БЕЗ КОТОРОЙ ТЫ РАБОТАЕШЬ", ()), (39.605, "НА ПОСЛЕДНИХ ПРОЦЕНТАХ", ("ПОСЛЕДНИХ",)),
    (41.708, "А ЕЩЁ БОЛЬШЕ ИНТЕРЕСНОГО —", (), T_), (43.068, "В МОЁМ ТЕЛЕГРАМ-КАНАЛЕ", ("ТЕЛЕГРАМ-КАНАЛЕ",), T_),
    (44.388, "«ПАУЧЬЯ НОРКА»", ("«ПАУЧЬЯ", "НОРКА»"), T_), (45.508, "ТАМ ЭКСКЛЮЗИВ,", ("ЭКСКЛЮЗИВ",), T_),
    (46.228, "КОТОРОГО НЕТ В РОЛИКАХ", (), T_), (47.528, "ССЫЛКА В ПРОФИЛЕ —", ("ПРОФИЛЕ",), T_),
    (48.548, "ЗАЛЕТАЙ!", ("ЗАЛЕТАЙ!",), T_),
]

# ---------------------------------------------------------------- камера и сцены
BLINKS = (15.0, 28.62)           # «моргание»: веки смыкаются, под ними склейка
C1, C2 = (b + 0.15 for b in BLINKS)
DROWSY = (15.3, 27.9)            # сонная камера: медленное покачивание
CAMERA = [
    (0.0, (8.0, 4.4, 1.6, 0.9, 50.0)), (1.0, (4.0, 3.7, 1.2, 0.86, 50.0), "out"),
    (3.2, (3.0, 3.7, 1.2, 0.86, 50.0)),
    (3.5, (0.0, 3.9, 1.3, 0.98, 50.0), "inout"),              # заставка
    (4.9, (-1.0, 3.9, 1.3, 0.98, 50.0)),
    (5.3, (85.0, 3.6, 1.25, 0.86, 50.0), "expo"),             # опыт
    (C1 - 0.01, (92.0, 3.5, 1.25, 0.86, 50.0), "lin"),
    (C1 + 0.01, (180.0, 3.7, 1.35, 0.88, 50.0), "lin"),       # не замечали + защита — сонная камера
    (C2 - 0.01, (186.0, 3.6, 1.35, 0.88, 50.0), "lin"),
    (C2 + 0.01, (272.0, 3.5, 0.65, 1.0, 50.0), "lin"),        # хорошая новость — снизу, наезд
    (41.3, (278.0, 3.35, 0.75, 1.0, 48.0), "lin"),
    (41.7, (460.0, 4.3, 1.3, 1.12, 50.0), "expo"), (DUR, (461.0, 4.0, 1.3, 1.15, 50.0)),
]
OFFSET = [(0.0, 0.55), (3.2, 0.55), (3.5, 0.0, "inout"), (4.9, 0.0), (5.3, -0.55, "expo"), (C1 - 0.01, -0.55),
          (C1 + 0.01, 0.55, "lin"), (C2 - 0.01, 0.55), (C2 + 0.01, -0.45, "lin"), (41.3, -0.45), (41.7, 0.0, "expo"),
          (DUR, 0.0)]
# окраска комнаты по сценам (B, G, R): ночь — синяя, опыт — фиолетовая, недосып — серо-зелёная, утро — тёплая
TINT = [(0.0, (1.25, 0.95, 0.75)), (3.2, (1.25, 0.95, 0.75)), (3.5, (1.0, 1.0, 1.0)), (4.9, (1.0, 1.0, 1.0)),
        (5.3, (1.15, 0.9, 1.0)), (C1 - 0.01, (1.15, 0.9, 1.0)), (C1 + 0.01, (0.95, 1.05, 0.92), "lin"),
        (C2 - 0.01, (0.95, 1.05, 0.92)), (C2 + 0.01, (0.85, 1.0, 1.15), "lin"), (41.3, (0.85, 1.0, 1.15)),
        (41.75, (1.0, 1.0, 1.0)), (DUR, (1.0, 1.0, 1.0))]
SEC = {"hook": 6, "exp": 88, "tired": 183, "good": 275, "tg": 460.5}
SHAKES = (26.98,)
FLASHES = ((41.75, 0.12),)
JUMPS = [37.45]
ALARM = None
SPOT = None

G = lambda *k: list(k)  # noqa: E731
O = OUTRO - 48.035      # жесты концовки — как в monet_mg, со сдвигом
ARM_L = G((0, 62), (3.3, 62), (3.5, -50, "back"), (4.5, -45), (4.8, 62, "back"), (5.4, -30, "back"), (9.8, -28),
          (10.2, -45, "back"), (12.4, -42), (12.6, -25, "back"), (14.6, -22), (14.9, 62), (18.8, 62),
          (19.0, 30, "back"), (21.3, 32), (21.5, 62), (24.4, 62), (24.6, 35, "back"), (26.3, 38), (26.5, 62),
          (28.8, -40, "back"), (30.6, -38), (31.0, -25, "back"), (34.7, -22), (35.0, -30, "back"), (37.3, -28),
          (37.45, -55, "back"), (38.2, -52), (38.4, -25, "back"), (41.3, -22), (41.6, 62),
          (54.94 + O, 62), (55.09 + O, -50, "back"), (55.69 + O, -45), (56.09 + O, 62, "back"), (DUR, 62))
ARM_R = G((0, 62), (0.3, -28, "back"), (3.1, -26), (3.3, 62), (3.5, -50, "back"), (4.5, -45), (4.8, 62, "back"),
          (15.2, 62), (15.4, -30, "back"), (18.6, -28), (19.0, 30, "back"), (21.3, 32), (21.6, -35, "back"),
          (24.2, -33), (24.6, 35, "back"), (26.3, 38), (26.6, -40, "back"), (28.0, -38), (28.4, 62),
          (37.3, 62), (37.45, -55, "back"), (38.2, -52), (38.4, 62),
          (47.79 + O, 62), (48.25 + O, -48, "back"), (50.6 + O, -45), (50.9 + O, -55, "back"), (53.6 + O, -52),
          (53.85 + O, -40, "back"), (54.8 + O, -42), (54.95 + O, -50, "back"), (55.69 + O, -45), (55.84 + O, -58),
          (55.99 + O, -32), (56.14 + O, -58), (56.29 + O, -32), (DUR, 62))
HEAD = G((0, 0), (0.6, 6), (1.7, -7), (2.7, 6), (3.5, 0), (5.3, -6), (7.2, 6), (8.6, -6), (10.2, 7), (11.4, -6),
         (12.9, 8), (14.4, -4), (15.3, -8), (17.6, 9), (19.0, -10), (20.4, 8), (21.6, -6), (23.4, 7), (24.5, -6),
         (27.0, 9), (28.4, 0), (28.9, 7), (30.8, -6), (33.0, 6), (35.0, -6), (37.4, 8), (39.6, -7), (41.4, 0),
         (42.0, -6), (48.8, 0), (DUR, 0))


# ---------------------------------------------------------------- картинки вставок
def clock_tex(blink):
    """Будильник «04:00» — светящиеся цифры, двоеточие мигает."""
    im = rounded((620, 300), 60, (14, 14, 22, 255), (60, 70, 120, 255), 6)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([40, 40, 580, 230], 30, fill=(8, 8, 14, 255))
    f = mg.font(170, tall=True)
    txt = "04:00" if blink else "04 00"
    tw = d.textbbox((0, 0), txt, font=f)[2]
    d.text((310 - tw / 2, 42), txt, font=f, fill=(255, 80, 80))
    d.text((60, 244), "БУДИЛЬНИК", font=mg.font(34), fill=(130, 140, 190))
    d.ellipse([520, 246, 556, 282], fill=RED_RGB + (255,))
    return mg.premul(im)


def z_tex():
    return mg.premul(mg.text_rgba("Z", 120, NIGHT_RGB, tall=True))


def compare_tex():
    im = rounded((860, 380), 34, (18, 18, 26, 255), (72, 72, 92, 255), 3)
    d = ImageDraw.Draw(im)
    for x, (a, b, col) in zip((40, 490), (("14 ДНЕЙ", "по 4 часа сна", NIGHT_RGB), ("2 НОЧИ", "совсем без сна", RED_RGB))):
        d.rounded_rectangle([x, 40, x + 330, 340], 30, fill=(28, 28, 40, 255), outline=col + (255,), width=6)
        f = mg.font(84, tall=True)
        tw = d.textbbox((0, 0), a, font=f)[2]
        d.text((x + 165 - tw / 2, 90), a, font=f, fill="white")
        f2 = mg.font(34)
        tw = d.textbbox((0, 0), b, font=f2)[2]
        d.text((x + 165 - tw / 2, 220), b, font=f2, fill=(170, 170, 190))
    d.text((395, 130), "=", font=mg.font(120, tall=True), fill=GOLD_RGB + (255,))
    return mg.premul(im)


def gauge_tex(title, v, col):
    """Прибор-стрелка: v от 0 до 1."""
    im = rounded((420, 330), 34, (18, 18, 26, 255), col + (255,), 5)
    d = ImageDraw.Draw(im)
    f = mg.font(40, tall=True)
    tw = d.textbbox((0, 0), title, font=f)[2]
    d.text((210 - tw / 2, 24), title, font=f, fill="white")
    cx, cy, r = 210, 270, 150
    d.arc([cx - r, cy - r, cx + r, cy + r], 180, 360, fill=(50, 50, 64), width=26)
    d.arc([cx - r, cy - r, cx + r, cy + r], 180, 180 + int(180 * v), fill=col + (255,), width=26)
    a = math.radians(180 + 180 * v)
    d.line([(cx, cy), (cx + (r - 30) * math.cos(a), cy + (r - 30) * math.sin(a))], fill="white", width=10)
    d.ellipse([cx - 16, cy - 16, cx + 16, cy + 16], fill="white")
    return mg.premul(im)


def shield_tex(crack):
    im = Image.new("RGBA", (360, 420), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    col = GREEN_RGB if crack < 0.5 else (200, 120, 60)
    pts = [(180, 10), (340, 70), (320, 250), (180, 410), (40, 250), (20, 70)]
    d.polygon(pts, fill=(20, 26, 22, 255), outline=col + (255,))
    d.line(pts + [pts[0]], fill=col + (255,), width=12, joint="curve")
    d.line([(150, 150), (210, 210), (170, 260), (220, 330)], fill="white", width=6)
    if crack > 0:
        n = int(crack * 4)
        segs = [[(180, 30), (150, 120), (200, 170), (160, 250)], [(160, 250), (210, 310), (180, 390)],
                [(200, 170), (290, 140)], [(150, 120), (60, 160)]]
        for s in segs[:max(1, n)]:
            d.line(s, fill=RED_RGB + (255,), width=8)
    return mg.premul(im)


def virus_tex():
    im = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for k in range(10):
        a = k * math.pi / 5
        d.line([(100, 100), (100 + 88 * math.cos(a), 100 + 88 * math.sin(a))], fill=(120, 220, 120), width=10)
        d.ellipse([100 + 88 * math.cos(a) - 12, 100 + 88 * math.sin(a) - 12, 100 + 88 * math.cos(a) + 12,
                   100 + 88 * math.sin(a) + 12], fill=(120, 220, 120))
    d.ellipse([40, 40, 160, 160], fill=(80, 170, 90))
    d.ellipse([70, 70, 95, 95], fill=(40, 110, 50))
    d.ellipse([110, 100, 130, 120], fill=(40, 110, 50))
    return mg.premul(im)


def battery_tex(v, label, blink=False):
    im = Image.new("RGBA", (380, 520), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([140, 20, 240, 60], 12, fill=(150, 150, 165, 255))
    d.rounded_rectangle([60, 50, 320, 440], 40, fill=(18, 18, 26, 255), outline=(150, 150, 165, 255), width=12)
    col = GREEN_RGB if v > 0.5 else GOLD_RGB if v > 0.2 else RED_RGB
    h = max(22, int(360 * v))
    if not blink:
        d.rounded_rectangle([82, 424 - h, 298, 424], 26, fill=col + (255,))
    f = mg.font(70, tall=True)
    txt = f"{int(round(v * 100))}%"
    tw = d.textbbox((0, 0), txt, font=f)[2]
    d.text((190 - tw / 2, 200), txt, font=f, fill="white")
    f2 = mg.font(44, tall=True)
    tw = d.textbbox((0, 0), label, font=f2)[2]
    d.text((190 - tw / 2, 460), label, font=f2, fill=(200, 200, 215))
    return mg.premul(im)


# ---------------------------------------------------------------- сцена
class Scene(base.Scene):
    cfg = sys.modules[__name__]

    def __init__(self, voice):
        super().__init__(voice)
        T = self.T
        T["z"] = z_tex()
        T["exp"] = card("ОПЫТ", "ПЕНСИЛЬВАНИЯ, 2003", "две недели по 4 часа сна", PURPLE_RGB)
        T["cmp"] = compare_tex()
        T["unnot"] = chip("ЛЮДИ ЭТОГО НЕ ЗАМЕЧАЛИ", RED_RGB)
        T["virus"] = virus_tex()
        T["x4"] = stamp("×4", RED_RGB)
        T["good"] = chip("ХОРОШАЯ НОВОСТЬ", GREEN_RGB)
        T["norm"] = card("ВЗРОСЛОМУ НУЖНО", "7–9 ЧАСОВ СНА", "каждую ночь", GREEN_RGB)
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
        dz = pop(t, DROWSY[0], DROWSY[1], 0.8, 0.6)             # сонная камера: плывёт и чуть кренится
        yaw += 3.0 * dz * math.sin((t - DROWSY[0]) * 1.1)
        hgt += 0.08 * dz * math.sin((t - DROWSY[0]) * 0.8)
        a = math.radians(yaw)
        shake = 0.0
        for s0 in SHAKES:
            if s0 < t < s0 + 0.35:
                shake = (1 - (t - s0) / 0.35) * math.sin((t - s0) * 70)
        return mg.Cam(yaw + 0.5 * wobble(t, 1) + 1.2 * shake, dist, hgt + 0.02 * wobble(t, 2) + 0.03 * shake,
                      ty + 0.015 * wobble(t, 3), fov=fov,
                      roll=0.5 * wobble(t * 0.8, 4) + 1.5 * shake + 4.0 * dz * math.sin((t - DROWSY[0]) * 0.9),
                      tx=off * math.cos(a), tz=off * math.sin(a))

    def finish(self, frame, t):
        fr = super().finish(frame, t)
        for b in BLINKS:  # «моргание»: веки (дуги) смыкаются сверху и снизу, под ними склейка, потом открываются
            p = (t - b) / 0.3
            if 0 < p < 1:
                q = smooth(p / 0.5) if p < 0.5 else smooth((1 - p) / 0.5)
                lid = np.zeros((H, W), np.uint8)
                yy = int(H / 2 * q * 1.08)
                sag = int(160 * q)
                cv2.ellipse(lid, (W // 2, yy - H // 2 - 40), (int(W * 0.95), H // 2 + sag), 0, 0, 180, 255, -1)
                cv2.ellipse(lid, (W // 2, H - yy + H // 2 + 40), (int(W * 0.95), H // 2 + sag), 0, 180, 360, 255, -1)
                cv2.rectangle(lid, (0, 0), (W, max(0, yy - 300)), 255, -1)
                cv2.rectangle(lid, (0, min(H, H - yy + 300)), (W, H), 255, -1)
                m = cv2.GaussianBlur(lid, (0, 0), 6).astype(np.float32)[..., None] / 255
                fr = (fr * (1 - m) + np.array([10, 8, 14], np.float32) * m).astype(np.uint8)
        return fr

    def items_for(self, t, cam, it):
        T = self.T
        P = lambda y0, *a, **k: place(y0, *a, **k)  # noqa: E731

        # 1. Крючок (персонаж слева, ночь): будильник «04:00», летят «Z»
        y0 = SEC["hook"]
        k = pop(t, 0.2, 3.3, 0.45, 0.25)
        if k > 0:
            pos = P(y0, 0.7, 1.6) + [0, 0.02 * math.sin(t * 3), 0]
            self.add_panel(it, cam, pos, face(y0, 12), 0.75 * k, 0.363 * k, 0.12 * k,
                           self.cached("clk", clock_tex, int(t * 2) % 2 == 0), pitch=-6, side=(40, 40, 70),
                           edge=(170, 180, 255))
            for i in range(5):
                p = ((t - 0.5 - 0.5 * i) % 2.5) / 2.5
                if t > 0.5 + 0.5 * i:
                    q = pos + np.array([0.0, 0.25 + 0.6 * p, 0.0]) + basis(y0)[0] * (0.15 * math.sin(p * 6 + i) - 0.1)
                    self.add_bb(it, cam, q, 0.14 * (0.6 + 0.6 * p) * k * (1 - p), T["z"], anchor=(0.5, 0.5))

        # 2. Заставка
        k = pop(t, 3.5, 4.9, 0.35, 0.3)
        if k > 0:
            self.add_panel(it, cam, place(0, 0.0, 1.95, 0.35), face(0, -6), 1.2 * k, 0.564 * k, 0.11 * k,
                           T["stinger"], pitch=-6, side=(30, 24, 120), edge=(200, 190, 255))

        # 3. Опыт (персонаж справа): карточка → «14 дней по 4 часа = 2 ночи без сна»
        y0 = SEC["exp"]
        k = pop(t, 5.3, 10.15, 0.4, 0.25)
        if k > 0:
            pos = P(y0, -0.72, 1.6) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, -14), 0.9 * k, 0.427 * k, 0.06 * k, T["exp"], pitch=-6,
                           side=(60, 40, 80), edge=(220, 200, 255))
        k = pop(t, 10.2, C1, 0.45, 0.2)
        if k > 0:
            pos = P(y0, -0.66, 1.6) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, -14), 0.95 * k, 0.419 * k, 0.06 * k, T["cmp"], pitch=-6,
                           side=(50, 50, 70), edge=(220, 220, 255))

        # 4. Не замечали + защита (персонаж слева, сонная камера): приборы → щит трескается → вирусы ×4
        y0 = SEC["tired"]
        r_, c_ = basis(y0)
        k = pop(t, 15.3, 21.5, 0.4, 0.25)
        if k > 0:
            self.add_bb(it, cam, P(y0, 0.75, 2.2), 0.1 * k, T["unnot"], anchor=(0.5, 0.5))
            kg = pop(t, 18.85, 21.5, 0.4, 0.25)
            if kg > 0:
                v1 = round(clamp((t - 19.0) / 0.8) * 0.8 * 20) / 20
                v2 = round(clamp((t - 19.6) / 0.8) * 0.3 * 20) / 20
                self.add_panel(it, cam, P(y0, 0.5, 1.55), face(y0, 10), 0.42 * kg, 0.33 * kg, 0.04 * kg,
                               self.cached("g1", gauge_tex, "КАЖЕТСЯ", v1, GOLD_RGB), side=(60, 50, 30))
                self.add_panel(it, cam, P(y0, 0.98, 1.55), face(y0, 14), 0.42 * kg, 0.33 * kg, 0.04 * kg,
                               self.cached("g2", gauge_tex, "НА САМОМ ДЕЛЕ", v2, RED_RGB), side=(70, 30, 30))
        k = pop(t, 21.55, AD_CUT + 0.3, 0.4, 0.2)
        if k > 0:
            crack = round(clamp((t - 22.5) / 0.9) * 4) / 4
            self.add_bb(it, cam, P(y0, 0.72, 1.95) + [0, 0.02 * math.sin(t * 2), 0], 0.4 * k,
                        self.cached("sh", shield_tex, crack), anchor=(0.5, 0.5))
            kv = pop(t, 24.45, AD_CUT + 0.3, 0.35, 0.2)
            if kv > 0:
                n = 1 if t < 26.95 else 4
                for i in range(n):
                    ki = pop(t, 24.45 if i == 0 else 26.95 + 0.08 * i, AD_CUT + 0.3, 0.3, 0.2)
                    ang = i * 1.6 + t * 0.6
                    q = P(y0, 0.72 + 0.32 * math.cos(ang), 1.2 + 0.18 * math.sin(ang), 0.05)
                    self.add_bb(it, cam, q, 0.16 * ki, T["virus"], anchor=(0.5, 0.5))
                if t > 26.95:
                    ks = clamp((t - 26.95) / 0.15)
                    self.add_bb(it, cam, P(y0, 1.0, 1.25, 0.1), 0.16 * (1 + 0.6 * (1 - ks)) * kv, T["x4"],
                                anchor=(0.5, 0.5))

        # 5. Хорошая новость (персонаж справа, снизу): норма сна → батарейки «со сном / без сна»
        y0 = SEC["good"]
        k = pop(t, C2 + 0.05, 32.0, 0.35, 0.25)
        if k > 0:
            self.add_bb(it, cam, P(y0, -0.7, 2.0), 0.14 * k * (1 + 0.4 * (1 - clamp((t - C2) / 0.2))), T["good"],
                        anchor=(0.5, 0.5))
        k = pop(t, 32.05, 34.9, 0.4, 0.25)
        if k > 0:
            pos = P(y0, -0.72, 1.7) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, -14), 0.9 * k, 0.427 * k, 0.06 * k, T["norm"], pitch=8,
                           side=(30, 70, 40), edge=(190, 255, 210))
        k = pop(t, 34.95, 41.35, 0.4, 0.25)
        if k > 0:
            v = round((0.08 + 0.92 * smooth((t - 37.0) / 1.2)) * 50) / 50
            self.add_bb(it, cam, P(y0, -0.95, 1.5), 0.55 * k, self.cached("bat", battery_tex, v, "СО СНОМ"),
                        anchor=(0.5, 0.5))
            kb = pop(t, 38.35, 41.35, 0.35, 0.25)
            if kb > 0:
                self.add_bb(it, cam, P(y0, -0.45, 1.5), 0.55 * kb,
                            self.cached("bat2", battery_tex, 0.03, "БЕЗ СНА", int(t * 4) % 2 == 0), anchor=(0.5, 0.5))

        # 6. Концовка с Telegram — общий шаблон
        self.outro_items(t, cam, it, OUTRO, SEC["tg"])


SFX = [(t, "whoosh") for t in (0.0, 3.45, 5.25) + BLINKS + (41.6,)]


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
        for t in (0.6, 2.2, 4.2, 7.0, 9.5, 12.0, 14.5, 15.1, 17.0, 20.0, 21.0, 23.5, 25.5, 27.5, 28.7, 30.0, 33.5,
                  36.5, 39.5, 41.0, 45.0):
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
    base.mix_audio(tmp, voice, out, music=os.path.join(ROOT, "work", "sleep", "music.wav"), sfx_list=SFX, dur=DUR,
                   sfx_vol=0.15)  # «вжух» лёгкий и тихий
    os.remove(tmp)


if __name__ == "__main__":
    main()
