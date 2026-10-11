"""Тестовый ролик 10 с: Паучок представляется. Объёмная моушн-графика, облёт камеры на 90°.

python3 tools/intro_mg.py work/intro/voice.wav output/intro_mg.mp4 [--preview]
"""
import math
import os
import random
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mg  # noqa: E402
from mg import H, W, clamp, out_back, track, wobble  # noqa: E402

FPS = 30
DUR = 10.0
ROOT = mg.ROOT

CHAR = dict(arm_y=(405, 550), cut_L=252, cut_R=556, shoulder_y=478, shoulder_r=42, neck_y=430, jaw_y=(338, 432), feet=(400, 1165), top_y=48,
            pivots={"armL": (268, 492), "armR": (540, 492), "head": (400, 428)})
CHAR_H = 1.6  # рост персонажа в «метрах» сцены

RED = (46, 38, 226)      # BGR
BLUE = (214, 84, 30)
WHITE = (238, 236, 240)
DARK = (30, 24, 28)

# ---------------------------------------------------------------- сценарий
SUBS = [  # (начало, конец, текст, выделенное слово)
    (0.45, 1.15, "ПРИВЕТ!", "ПРИВЕТ!"),
    (1.15, 1.75, "МЕНЯ ЗОВУТ", None),
    (1.75, 2.55, "ПАУЧОК", "ПАУЧОК"),
    (2.6, 3.5, "Я ЖИВУ НА КРЫШАХ", None),
    (3.5, 4.25, "БОЛЬШОГО ГОРОДА", "ГОРОДА"),
    (4.3, 5.0, "И ЗНАЮ КУЧУ", None),
    (5.0, 6.15, "ИНТЕРЕСНЫХ ИСТОРИЙ", "ИСТОРИЙ"),
    (6.25, 7.0, "ПОДПИСЫВАЙСЯ", "ПОДПИСЫВАЙСЯ"),
    (7.0, 8.1, "БУДЕТ ВЕСЕЛО!", "ВЕСЕЛО!"),
]

# камера: (время, (облёт°, расстояние, высота, точка взгляда по высоте, угол обзора°), плавность)
CAMERA = [
    (0.0, (-35.0, 7.0, 2.6, 1.0, 50.0)),
    (0.75, (-10.0, 3.5, 1.15, 0.9, 50.0), "out"),
    (1.3, (-12.0, 3.5, 1.15, 0.92, 50.0)),
    (2.55, (-20.0, 3.9, 1.25, 1.0, 50.0)),
    (3.9, (-34.0, 5.0, 1.9, 1.1, 52.0)),
    (4.28, (-30.0, 4.4, 1.5, 0.95, 52.0)),
    (4.62, (68.0, 3.9, 1.2, 0.95, 50.0), "expo"),   # резкий облёт на 90°
    (6.15, (74.0, 3.9, 1.2, 0.95, 50.0)),
    (6.6, (70.0, 3.2, 1.05, 1.0, 50.0), "out"),     # наезд на «Подписывайся»
    (8.0, (76.0, 3.1, 1.05, 1.0, 50.0)),
    (10.0, (90.0, 3.5, 1.15, 0.95, 50.0)),
]
# сдвиг кадра вбок (в метрах): персонаж уступает место вставке
OFFSET = [(0.0, 0.0), (1.25, 0.0), (1.6, -0.55, "inout"), (3.6, -0.5), (4.1, 0.0, "inout"), (4.6, 0.6, "expo"),
          (6.1, 0.6), (6.5, -0.4, "inout"), (7.9, -0.4), (8.4, 0.0, "inout"), (10.0, 0.0)]

ARM_L = [(0.0, -40), (0.38, -40), (0.6, 62, "back"), (1.3, 60), (1.5, -6, "back"), (2.5, -2), (2.75, 40),
         (3.05, -28, "back"), (3.8, -20), (4.1, 55), (4.4, -15, "out"), (4.75, 60, "back"), (6.2, 58),
         (6.4, 4, "back"), (6.95, 8), (7.1, -50, "back"), (7.8, -45), (8.15, 62, "back"), (10.0, 62)]
ARM_R = [(0.0, -40), (0.38, -40), (0.55, 62, "back"), (0.62, -58, "back"), (0.77, -32), (0.92, -62), (1.07, -32),
         (1.22, -58), (1.45, 60, "back"), (2.75, 60), (3.05, -28, "back"), (3.8, -20), (4.1, 55), (4.4, -15, "out"),
         (4.85, -6, "back"), (6.1, -2), (6.35, 60), (7.0, 60), (7.1, -50, "back"), (7.8, -45), (8.15, 55),
         (8.4, -58, "back"), (8.6, -32), (8.8, -60), (9.0, -32), (9.2, -60), (9.4, -34), (9.75, 60, "back"), (10.0, 62)]
HEAD = [(0.0, 0), (0.6, -5), (1.2, -6), (1.45, 8), (2.5, 7), (2.8, 0), (3.6, -3), (4.3, 2), (4.8, -8),
        (6.1, -7), (6.4, 6), (7.0, 5), (7.2, -4), (8.1, 0), (8.6, -6), (9.6, -4), (10.0, 0)]
TURN = [(0.0, 0), (1.4, 0), (1.6, -14), (2.5, -14), (2.75, 0), (4.8, 0), (5.0, 12), (6.1, 12), (6.3, 0), (10, 0)]


def char_motion(t):
    """Падение в кадр в начале, приседание при приземлении, прыжок на «весело»."""
    y = 0.0
    sx = sy = 1.0
    if t < 0.38:
        y = 1.8 * (1 - (t / 0.38) ** 2)
        sy, sx = 1.08, 0.94
    elif t < 0.62:
        k = math.sin(math.pi * (t - 0.38) / 0.24)
        sy, sx = 1 - 0.14 * k, 1 + 0.12 * k
    if 7.1 < t < 7.6:
        p = (t - 7.1) / 0.5
        y = 0.38 * math.sin(math.pi * p)
        sy = 1 + 0.06 * math.sin(math.pi * p)
    if 7.6 <= t < 7.75:
        k = math.sin(math.pi * (t - 7.6) / 0.15)
        sy, sx = 1 - 0.08 * k, 1 + 0.07 * k
    breathe = 1 + 0.008 * math.sin(t * 2 * math.pi / 2.4)
    return y, sx, sy * breathe


# ---------------------------------------------------------------- картинки для вставок
def rounded(size, radius, fill, outline=None, width=0):
    im = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(im).rounded_rectangle([0, 0, size[0] - 1, size[1] - 1], radius, fill=fill,
                                         outline=outline, width=width)
    return im


def card_passport():
    im = rounded((760, 480), 34, (20, 20, 28, 255), (70, 70, 90, 255), 3)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([36, 34, 352, 92], 29, fill=(226, 38, 46, 255))
    d.text((58, 42), "ПАСПОРТ ГЕРОЯ", font=mg.font(34), fill="white")
    d.text((36, 104), "ПАУЧОК", font=mg.font(132, tall=True), fill="white")
    rows = [("ГОРОД", "крыши мегаполиса"), ("СУПЕРСИЛА", "истории"), ("УРОВЕНЬ", "99")]
    for i, (k, v) in enumerate(rows):
        y = 300 + i * 56
        d.text((40, y), k, font=mg.font(30), fill=(140, 140, 160))
        d.text((300, y), v, font=mg.font(34), fill="white")
        if i < 2:
            d.line([(40, y + 47), (720, y + 47)], fill=(50, 50, 64), width=2)
    return mg.premul(im)


def card_title(text):
    im = rounded((760, 200), 34, (20, 20, 28, 255), (70, 70, 90, 255), 3)
    d = ImageDraw.Draw(im)
    d.rectangle([40, 170, 200, 180], fill=(226, 38, 46, 255))
    d.text((40, 22), text, font=mg.font(96, tall=True), fill="white")
    return mg.premul(im)


def button(pressed):
    if pressed:
        im = rounded((760, 220), 110, (40, 40, 48, 255), (90, 90, 110, 255), 4)
        txt, col = "ТЫ С НАМИ!", (230, 230, 240)
    else:
        im = rounded((760, 220), 110, (226, 38, 46, 255), (255, 120, 120, 255), 4)
        txt, col = "ПОДПИСАТЬСЯ", (255, 255, 255)
    d = ImageDraw.Draw(im)
    f = mg.font(112, tall=True)
    l, t, r, b = d.textbbox((0, 0), txt, font=f)
    d.text(((760 - (r - l)) / 2 - l, (220 - (b - t)) / 2 - t), txt, font=f, fill=col)
    return mg.premul(im)


TEXT_CACHE = {}


def text_tex(s, size, color=(255, 255, 255)):
    k = (s, size, color)
    if k not in TEXT_CACHE:
        TEXT_CACHE[k] = mg.premul(mg.text_rgba(s, size, color, tall=True))
    return TEXT_CACHE[k]


# ---------------------------------------------------------------- субтитры
def subtitles(frame, t):
    for j, (t0, t1, text, key) in enumerate(SUBS):
        nxt = SUBS[j + 1][0] if j + 1 < len(SUBS) else 1e9
        end = min(t1 + 0.08, nxt)
        if not (t0 <= t < end):
            continue
        k = out_back((t - t0) / 0.16)
        fade = clamp((end - t) / 0.06)
        size = 110 if len(text) < 12 else 88 if len(text) < 17 else 76
        f = mg.font(size)
        words = text.split()
        im = Image.new("RGBA", (W, 260), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        space = size * 0.28
        widths = [d.textbbox((0, 0), w_, font=f)[2] for w_ in words]
        total = sum(widths) + space * (len(words) - 1)
        x = (W - total) / 2
        for w_, ww in zip(words, widths):
            if key and w_ == key:
                d.rounded_rectangle([x - 18, 40, x + ww + 18, 60 + size * 1.12], 22, fill=(226, 38, 46, 255))
                d.text((x, 48), w_, font=f, fill="white")
            else:
                d.text((x + 5, 53), w_, font=f, fill=(0, 0, 0, 160))
                d.text((x, 48), w_, font=f, fill="white", stroke_width=3, stroke_fill=(0, 0, 0))
            x += ww + space
        arr = mg.premul(im) * fade
        sc = 0.55 + 0.45 * k
        arr = cv2.resize(arr, None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
        h, w = arr.shape[:2]
        mg.over(frame, arr, (W - w) // 2, int(1560 - h / 2 + (1 - k) * 40))


# ---------------------------------------------------------------- сцена
class Scene:
    def __init__(self, voice):
        self.room = mg.Room(os.path.join(ROOT, "assets", "bg_grid.jpg"), radius=9.0, tiles=12)
        self.puppet = mg.Puppet(os.path.join(ROOT, "assets", "char_tpose.jpg"), CHAR)
        p = self.puppet
        self.px_to_m = CHAR_H / (p.feet[1] - p.top)
        self.anchor = (p.feet[0] / p.Wc, p.feet[1] / p.Hc)
        self.env = mg.voice_envelope(voice, FPS)
        self.tex = {"passport": card_passport(), "title": card_title("МОИ ИСТОРИИ"),
                    "btn": button(False), "btn2": button(True)}
        rnd = random.Random(4)
        self.city = []
        for i in range(16):
            ang = rnd.uniform(-60, 60) + 180 - 20
            r = rnd.uniform(2.6, 5.5)
            x, z = r * math.sin(math.radians(ang)), -r * math.cos(math.radians(ang))
            self.city.append(dict(pos=(x, z), size=(rnd.uniform(0.45, 0.9), rnd.uniform(0.45, 0.9)),
                                  h=rnd.uniform(0.7, 2.6), delay=rnd.uniform(0, 0.55), yaw=rnd.uniform(-20, 20),
                                  glow=rnd.random() < 0.4))
        self.grain = np.random.default_rng(0).integers(-7, 8, size=(4, H // 2, W // 2), dtype=np.int16)

    def cam(self, t):
        yaw, dist, hgt, ty, fov = track(CAMERA, t)
        off = track(OFFSET, t)
        a = math.radians(yaw)
        # «ручная» камера: лёгкие покачивания
        return mg.Cam(yaw + 0.5 * wobble(t, 1), dist, hgt + 0.02 * wobble(t, 2), ty + 0.015 * wobble(t, 3),
                      fov=fov, roll=0.5 * wobble(t * 0.8, 4), tx=off * math.cos(a), tz=off * math.sin(a))

    def draw(self, t):
        cam = self.cam(t)
        frame = self.room.render(cam)
        # свет под персонажем и сетка пола
        mg.floor_blob(frame, cam, (0, 0), 1.3, 1.3, 0.22, color=(90, 70, 220), blur=151, mode="light")
        mg.floor_grid(frame, cam, step=0.75, extent=9, base=110, fade=11, radius=8.9, mask=self.room.floor_mask)
        # волна от приземления
        if 0.36 < t < 1.0:
            p = (t - 0.36) / 0.64
            a = np.linspace(0, 2 * math.pi, 64)
            r = 0.3 + 2.4 * mg.out_cubic(p)
            P = np.stack([r * np.cos(a), np.zeros_like(a), r * np.sin(a)], -1)
            uv, z = cam.proj(P)
            if (z > 0.3).all():
                lay = np.zeros_like(frame)
                mg.lines(lay, uv, tuple(int(c * (1 - p)) for c in (120, 110, 255)), 5)
                cv2.add(frame, cv2.GaussianBlur(lay, (9, 9), 0), frame)

        items = []  # (глубина, функция рисования)

        # персонаж
        y, sx, sy = char_motion(t)
        env = self.env[min(len(self.env) - 1, int(t * FPS))]
        p = self.puppet
        canvas = p.render(armL=track(ARM_L, t), armR=track(ARM_R, t),
                          head=track(HEAD, t) + 2.5 * wobble(t * 1.3, 7) + 3 * env * math.sin(t * 9),
                          jaw=clamp(env * 1.1), head_dy=-7 * env)
        lag_cam = self.cam(max(0.0, t - 0.12))
        facing = math.degrees(lag_cam.yaw) + track(TURN, t)
        hgt_m = p.Hc * self.px_to_m
        shadow_k = clamp(1 - y / 2.0)
        mg.floor_blob(frame, cam, (0, 0), 0.5 * (0.6 + 0.4 * shadow_k), 0.32 * (0.6 + 0.4 * shadow_k), 0.75 * shadow_k, blur=41)
        items.append((cam.depth((0, 0.8, 0)), lambda: mg.billboard(frame, cam, (0, y, 0), hgt_m, canvas,
                                                                     facing_yaw=facing, anchor=self.anchor, squash=(sx, sy))))

        # вставка 1: паспорт героя (слева от персонажа в первом ракурсе)
        if 1.25 < t < 4.4:
            k_in = out_back((t - 1.3) / 0.45)
            k_out = clamp((t - 4.0) / 0.35)
            s = max(0.001, k_in * (1 - mg.smooth(k_out)))
            r15 = np.array([math.cos(math.radians(-15)), 0, math.sin(math.radians(-15))])
            tc15 = np.array([math.sin(math.radians(-15)), 0, -math.cos(math.radians(-15))])
            base = -r15 * 1.0 + tc15 * 0.35 + np.array([0, 1.25, 0])
            pos = base + np.array([0, -0.6 * (1 - k_in) - 0.8 * k_out, 0]) + np.array([0, 0.03 * math.sin(t * 2.2), 0])
            yaw = 15 + 28 - 25 * (1 - k_in) + 40 * k_out
            items.append((cam.depth(pos), lambda pos=pos, s=s, yaw=yaw: (
                mg.floor_blob(frame, cam, (pos[0], pos[2]), 0.55 * s, 0.18 * s, 0.5 * s, blur=61),
                mg.panel(frame, cam, pos, yaw, 0.95 * s, 0.6 * s, 0.06 * s, self.tex["passport"],
                         pitch=-4, roll=-3 + 6 * (1 - k_in), side=(70, 60, 70), edge=(200, 200, 230)))))

        # город из коробок вырастает из пола
        if t > 2.65:
            for b in self.city:
                g = mg.out_back(clamp((t - 2.7 - b["delay"]) / 0.5), s=1.4)
                if g <= 0:
                    continue
                cen = np.array([b["pos"][0], b["h"] * g / 2, b["pos"][1]])
                items.append((cam.depth(cen), lambda b=b, g=g: mg.box(
                    frame, cam, b["pos"], b["size"], b["h"] * g, DARK, edge=(90, 70, 210) if b["glow"] else (70, 60, 80),
                    yaw=b["yaw"], glow=(40, 30, 200) if b["glow"] else None)))

        # вставка 2 (после облёта): объёмный график «Мои истории»
        if 4.6 < t < 6.6:
            gone = mg.smooth((t - 6.1) / 0.4)
            right = np.array([math.cos(math.radians(70)), 0, math.sin(math.radians(70))])
            back = np.array([-math.sin(math.radians(70)), 0, math.cos(math.radians(70))])
            vals = [("СМЕШНЫЕ", 120, 0.62, RED), ("СТРАШНЫЕ", 85, 0.45, BLUE), ("ПРО ГЕРОЕВ", 200, 1.05, WHITE)]
            for i, (name, num, hh, col) in enumerate(vals):
                c = right * (0.55 + 0.31 * i) - back * 0.15
                g = mg.out_back(clamp((t - 4.75 - 0.15 * i) / 0.55), s=1.2) * (1 - gone)
                if g <= 0:
                    continue
                cen = np.array([c[0], hh * g / 2, c[2]])

                def draw_bar(c=c, hh=hh, g=g, col=col, name=name, num=num, i=i):
                    mg.floor_blob(frame, cam, (c[0], c[2]), 0.22, 0.22, 0.6, blur=31)
                    mg.box(frame, cam, (c[0], c[2]), (0.26, 0.26), hh * g, col, yaw=70, edge=(20, 20, 24))
                    cnt = int(num * clamp((t - 4.75 - 0.15 * i) / 0.8))
                    mg.billboard(frame, cam, (c[0], hh * g + 0.04, c[2]), 0.13, text_tex(str(cnt), 90),
                                 anchor=(0.5, 1.0))
                    mg.billboard(frame, cam, (c[0], hh * g + 0.18, c[2]), 0.07, text_tex(name, 60, (180, 180, 200)),
                                 anchor=(0.5, 1.0))
                items.append((cam.depth(cen), draw_bar))
            k = out_back((t - 4.7) / 0.5) * (1 - gone)
            if k > 0.01:
                pos = right * 0.86 - back * 0.15 + np.array([0, 1.62 + 0.03 * math.sin(t * 2), 0])
                items.append((cam.depth(pos), lambda pos=pos, k=k: mg.panel(
                    frame, cam, pos, -70 + 12, 0.95 * k, 0.25 * k, 0.06 * k, self.tex["title"], pitch=-6, roll=2,
                    side=(70, 60, 70), edge=(200, 200, 230))))

        # вставка 3: кнопка «Подписаться» — нажимается
        if t > 6.25:
            right = np.array([math.cos(math.radians(72)), 0, math.sin(math.radians(72))])
            tocam = np.array([math.sin(math.radians(72)), 0, -math.cos(math.radians(72))])
            k = out_back((t - 6.3) / 0.4)
            press = 0.0
            if 7.3 < t < 7.5:
                press = math.sin(math.pi * (t - 7.3) / 0.2)
            pos = -right * 0.62 + tocam * 0.45 + np.array([0, 1.05 + 0.02 * math.sin(t * 3), 0])
            tex = self.tex["btn2"] if t > 7.4 else self.tex["btn"]
            s = k * (1 - 0.12 * press)
            items.append((cam.depth(pos), lambda pos=pos, s=s, tex=tex: (
                mg.floor_blob(frame, cam, (pos[0], pos[2]), 0.45 * s, 0.15 * s, 0.5 * s, blur=41),
                mg.panel(frame, cam, pos, -72 - 22, 0.8 * s, 0.23 * s, 0.12 * s, tex, pitch=-8,
                         side=(30, 24, 150) if t <= 7.4 else (40, 40, 50), edge=(160, 160, 255)))))

        for _, fn in sorted(items, key=lambda x: -x[0]):
            fn()

        subtitles(frame, t)
        return frame

    def finish(self, frame, t, i):
        # вспышка на облёте, виньетка, лёгкое «плёночное» зерно
        f = frame.astype(np.float32)
        flash = 0.0
        if 4.38 < t < 4.6:
            flash = math.sin(math.pi * (t - 4.38) / 0.22) * 0.12
        if t < 0.12:
            flash = 0.35 * (1 - t / 0.12)
        if flash:
            f = f + flash * 255
        f *= VIGNETTE[..., None]
        g = cv2.resize(self.grain[i % 4], (W, H), interpolation=cv2.INTER_NEAREST)
        f += g[..., None]
        fade = clamp((DUR - t) / 0.35)
        return np.clip(f * fade, 0, 255).astype(np.uint8)


def _vignette():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((xx - W / 2) / (W * 0.75)) ** 2 + ((yy - H / 2) / (H * 0.72)) ** 2)
    return np.clip(1.08 - 0.55 * d ** 2, 0.45, 1.0).astype(np.float32)


VIGNETTE = _vignette()


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    voice, out = args[0], args[1]
    only = None
    if "--preview" in sys.argv:  # несколько кадров-картинок для проверки
        only = [int(x * FPS) for x in (0.3, 0.7, 1.0, 1.9, 3.4, 4.45, 5.6, 6.8, 7.4, 9.0)]
    sc = Scene(voice)
    tmp = out + ".video.mp4"
    wr = None if only else mg.Writer(tmp, FPS)
    n = int(DUR * FPS)
    prev = None
    for i in range(n) if not only else only:
        t = i / FPS
        # размытие в движении на быстром облёте: несколько под-кадров
        speed = abs(sc.cam(t + 0.5 / FPS).yaw - sc.cam(t - 0.5 / FPS).yaw) * FPS
        sub = 1 if speed < math.radians(60) else 6
        if sub == 1:
            fr = sc.draw(t)
        else:
            acc = np.zeros((H, W, 3), np.float32)
            for k in range(sub):
                acc += sc.draw(t + (k / sub - 0.5) / FPS)
            fr = (acc / sub).astype(np.uint8)
        fr = sc.finish(fr, t, i)
        if only:
            cv2.imwrite(f"{out}_{i:03d}.jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        else:
            wr.write(fr)
            if i % 30 == 0:
                print(f"{i}/{n}", flush=True)
    if only:
        return
    wr.close()
    events = [(0.0, "whoosh"), (0.36, "boom"), (1.3, "pop"), (2.72, "rumble"), (4.22, "whoosh"),
              (4.75, "pop"), (4.9, "pop"), (5.05, "pop"), (4.72, "pop"), (6.3, "pop"), (7.32, "click"),
              (7.1, "whoosh")]
    fx = mg.sfx(out + ".sfx.wav", events, DUR)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", tmp, "-i", voice, "-i", fx,
                    "-filter_complex", "[1:a]volume=1.0[v];[2:a]volume=0.55[s];[v][s]amix=inputs=2:normalize=0[a]",
                    "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", out],
                   check=True)
    os.remove(tmp)
    os.remove(fx)


if __name__ == "__main__":
    main()
