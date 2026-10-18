"""Ролик №12: «Бывают ли настоящие друзья в интернете?» (по просьбе зрителя, со скриншотом его сообщения).

python tools/netfriends_mg.py work/netfriends/voice_clean.wav output/netfriends.mp4 [--preview] [--jobs 4]
Общий движок, персонаж и концовка с Telegram — из memory_mg.py; параллельный рендер — как в money_mg.py.
Голос укорочен до минуты: tools/tighten.py (паузы до 0,32 с + ускорение 6% без изменения высоты, концовка не тронута).
Новое в монтаже: 3D-телефон со скриншотом переписки влетает, вращаясь, просьба подсвечивается; смена сцен
«цифровым глитчем» (кадр рассыпается на пиксели); сигнал бежит между двумя людьми далеко друг от друга;
общие интересы соединяют двух людей; входящий звонок «Интернет-друг»; ролик «улетает» другу; набор комментария.
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
OUTRO = float(open(os.path.join(ROOT, "work", "netfriends", "outro_start.txt")).read())  # 51.29
DUR = 59.9
CYAN_RGB, PINK_RGB, GREEN_RGB, GOLD_RGB = (70, 200, 240), (240, 80, 130), (46, 196, 110), (245, 190, 50)
CYAN, PINK = CYAN_RGB[::-1], PINK_RGB[::-1]

# ---------------------------------------------------------------- субтитры
T_, C_, P_, G_ = TG_RGB, CYAN_RGB, PINK_RGB, GREEN_RGB
SUBS = [
    (0.154, "БЫВАЮТ ЛИ НАСТОЯЩИЕ ДРУЗЬЯ", ("ДРУЗЬЯ",), C_), (1.644, "В ИНТЕРНЕТЕ?", ("ИНТЕРНЕТЕ?",), C_),
    (2.7, "", ()),
    (4.201, "ЭТУ ТЕМУ НАМ ПРЕДЛОЖИЛ", ()), (5.541, "ЭТОТ ЗРИТЕЛЬ В СООБЩЕНИЯХ", ("ЗРИТЕЛЬ",), C_),
    (7.05, "СПАСИБО!", ("СПАСИБО!",), P_), (8.031, "ОТВЕТ УЧЁНЫХ:", ()), (9.276, "ДА, БЫВАЮТ", ("БЫВАЮТ",), G_),
    (10.575, "ПСИХОЛОГ ДЖОЗЕФ УОЛТЕР", ("УОЛТЕР",), BLUE_RGB), (11.89, "ЗАМЕТИЛ, ЧТО В ПЕРЕПИСКЕ", ()),
    (13.55, "ЛЮДИ ИНОГДА СБЛИЖАЮТСЯ", ("СБЛИЖАЮТСЯ",), C_), (14.758, "ДАЖЕ БЫСТРЕЕ, ЧЕМ ВЖИВУЮ", ("БЫСТРЕЕ,",), C_),
    (16.66, "ПОЧЕМУ ТАК?", ("ТАК?",)), (17.682, "В СЕТИ МЫ СМЕЛЕЕ:", ("СМЕЛЕЕ:",), C_),
    (18.852, "НЕ ВИДИМ ЛИЦА,", ("ЛИЦА,",)), (19.984, "МЕНЬШЕ БОИМСЯ ОСУЖДЕНИЯ", ("ОСУЖДЕНИЯ",)),
    (21.418, "И РАССКАЗЫВАЕМ О СЕБЕ БОЛЬШЕ", ("БОЛЬШЕ",), C_), (23.493, "А ДРУГА НАХОДИМ", ()),
    (24.361, "ПО ИНТЕРЕСАМ,", ("ИНТЕРЕСАМ,",), G_), (25.192, "А НЕ ПО СОСЕДСТВУ", ()),
    (26.361, "ТАМ НАХОДЯТ ДАЖЕ ЛЮБОВЬ", ("ЛЮБОВЬ",), P_), (28.009, "ПО ДАННЫМ СТЭНФОРДА,", ("СТЭНФОРДА,",), BLUE_RGB),
    (29.286, "ПОЧТИ 40% ПАР В США", ("40%",), P_), (31.116, "ПОЗНАКОМИЛИСЬ ОНЛАЙН", ("ОНЛАЙН",), C_),
    (32.925, "КАК СДЕЛАТЬ ТАКУЮ", ()), (33.859, "ДРУЖБУ КРЕПЧЕ?", ("КРЕПЧЕ?",), G_),
    (35.237, "ПОЗВОНИ ГОЛОСОМ", ("ГОЛОСОМ",), G_), (36.454, "В ОПЫТЕ УЧЁНЫХ ЛЮДИ БОЯЛИСЬ,", ("БОЯЛИСЬ,",)),
    (38.529, "ЧТО ЗВОНОК БУДЕТ НЕЛОВКИМ,", ("НЕЛОВКИМ,",)), (40.208, "А В ИТОГЕ ЧУВСТВОВАЛИ СЕБЯ", ()),
    (41.416, "БЛИЖЕ, ЧЕМ В ПЕРЕПИСКЕ", ("БЛИЖЕ,",), G_), (43.378, "ТАК ЧТО ЕСЛИ У ТЕБЯ", ()),
    (44.322, "ЕСТЬ ИНТЕРНЕТ-ДРУГ —", ("ИНТЕРНЕТ-ДРУГ",), C_), (45.793, "ОТПРАВЬ ЕМУ ЭТОТ РОЛИК", ("ОТПРАВЬ",), T_),
    (47.482, "ПИШИ В КОММЕНТАРИИ,", ("КОММЕНТАРИИ,",), P_), (48.633, "НА КАКУЮ ТЕМУ", ()),
    (49.463, "СНЯТЬ НОВЫЙ РОЛИК", ("НОВЫЙ",), P_),
    (51.25, "А ЕЩЁ БОЛЬШЕ ИНТЕРЕСНОГО —", (), T_), (52.61, "В МОЁМ ТЕЛЕГРАМ-КАНАЛЕ", ("ТЕЛЕГРАМ-КАНАЛЕ",), T_),
    (53.94, "«ПАУЧЬЯ НОРКА»", ("«ПАУЧЬЯ", "НОРКА»"), T_), (55.06, "ТАМ ЭКСКЛЮЗИВ,", ("ЭКСКЛЮЗИВ",), T_),
    (55.76, "КОТОРОГО НЕТ В РОЛИКАХ", (), T_), (57.06, "ССЫЛКА В ПРОФИЛЕ —", ("ПРОФИЛЕ",), T_),
    (58.08, "ЗАЛЕТАЙ!", ("ЗАЛЕТАЙ!",), T_),
]

# ---------------------------------------------------------------- камера и сцены
GLITCH = (16.35, 32.6)                # «цифровой глитч»: кадр рассыпается на пиксели, под этим склейка
C1, C2 = (g + 0.2 for g in GLITCH)
CAMERA = [
    (0.0, (-14.0, 4.0, 1.35, 0.86, 50.0)), (0.9, (-8.0, 3.6, 1.15, 0.84, 50.0), "out"),
    (2.6, (-5.0, 3.6, 1.15, 0.84, 50.0)),
    (3.0, (0.0, 3.9, 1.3, 0.98, 50.0), "inout"),             # заставка
    (3.85, (1.0, 3.9, 1.3, 0.98, 50.0)),
    (4.25, (95.0, 3.6, 1.2, 0.84, 50.0), "expo"),            # сообщение зрителя, «да, бывают», Уолтер
    (C1 - 0.01, (104.0, 3.5, 1.2, 0.86, 50.0), "lin"),
    (C1 + 0.01, (190.0, 3.6, 1.2, 0.86, 50.0), "lin"),       # почему так: смелее, интересы, любовь
    (C2 - 0.01, (198.0, 3.6, 1.25, 0.86, 50.0), "lin"),
    (C2 + 0.01, (285.0, 3.6, 1.2, 0.84, 50.0), "lin"),       # позвони, отправь, напиши
    (50.85, (299.0, 3.7, 1.25, 0.86, 50.0), "lin"),
    (51.25, (380.0, 4.3, 1.3, 1.12, 50.0), "expo"), (DUR, (381.0, 4.0, 1.3, 1.15, 50.0)),
]
OFFSET = [(0.0, -0.5), (2.6, -0.5), (3.0, 0.0, "inout"), (3.85, 0.0), (4.25, 0.55, "expo"), (C1 - 0.01, 0.55),
          (C1 + 0.01, -0.55, "lin"), (C2 - 0.01, -0.55), (C2 + 0.01, 0.55, "lin"), (50.85, 0.55),
          (51.25, 0.0, "expo"), (DUR, 0.0)]
TINT = [(0.0, (1.12, 1.02, 0.86)), (3.85, (1.12, 1.02, 0.86)), (4.3, (1.06, 1.0, 0.94)), (C1 - 0.01, (1.06, 1.0, 0.94)),
        (C1 + 0.01, (1.12, 0.92, 1.04), "lin"), (C2 - 0.01, (1.12, 0.92, 1.04)), (C2 + 0.01, (0.92, 1.08, 0.96), "lin"),
        (50.9, (0.92, 1.08, 0.96)), (51.3, (1.0, 1.0, 1.0)), (DUR, (1.0, 1.0, 1.0))]
SEC = {"hook": -8, "msg": 99, "why": 194, "call": 292, "tg": 380.5}
SHAKES = ()
FLASHES = ((4.2, 0.08), (9.3, 0.08), (51.3, 0.12))
JUMPS = [9.3]
ALARM = None
SPOT = None

G = lambda *k: list(k)  # noqa: E731
ARM_L = G((0, 62), (0.4, 62), (0.65, -30, "back"), (2.4, -28), (2.7, 62), (2.9, -50, "back"), (3.8, -45),
          (4.1, 62, "back"), (7.0, 62), (7.2, 30, "back"), (7.9, 32), (8.1, 62), (9.25, 62), (9.35, -50, "back"),
          (10.2, -45), (10.5, 62, "back"), (C1 + 0.05, 62), (16.75, -25, "back"), (23.3, -22), (23.5, -12, "back"),
          (26.2, -10), (26.4, -35, "back"), (32.4, -33), (32.7, 62), (58.19, 62), (58.34, -50, "back"), (58.94, -45),
          (59.34, 62, "back"), (DUR, 62))
ARM_R = G((0, 62), (2.7, 62), (2.9, -50, "back"), (3.8, -45), (4.1, 62, "back"), (4.3, -30, "back"), (7.0, -28),
          (7.2, 30, "back"), (7.9, 32), (8.1, -15, "back"), (9.2, -12), (9.35, -50, "back"), (10.2, -45),
          (10.5, -28, "back"), (16.2, -26), (16.4, 62), (C2 + 0.05, 62), (33.0, -30, "back"), (36.3, -28),
          (36.5, -15, "back"), (43.2, -12), (43.4, -40, "back"), (47.2, -38), (47.5, -20, "back"), (50.9, -18),
          (51.04, 62), (51.5, -48, "back"), (53.85, -45), (54.15, -55, "back"), (56.85, -52), (57.1, -40, "back"),
          (58.05, -42), (58.2, -50, "back"), (58.94, -45), (59.09, -58), (59.24, -32), (59.39, -58), (59.54, -32),
          (DUR, 62))
HEAD = G((0, 0), (0.6, -6), (1.8, 7), (2.7, 0), (4.3, -6), (6.0, -5), (7.05, 10), (7.6, 4), (8.1, -6), (9.3, 0),
         (10.6, -6), (13.5, -5), (15.0, 7), (16.4, 0), (16.8, 7), (18.8, 6), (21.4, -5), (23.5, 6), (26.4, -6),
         (29.3, 7), (31.9, 5), (33.0, -6), (35.2, 7), (36.5, -5), (40.2, 6), (43.4, -6), (45.8, 7), (47.5, -5),
         (51.5, -6), (58.2, 0), (DUR, 0))


# ---------------------------------------------------------------- картинки вставок
def header(d, text, col, dark=False, y=30):
    f = mg.font(36)
    tw = d.textbbox((0, 0), text, font=f)[2]
    d.rounded_rectangle([36, y, 36 + tw + 44, y + 62], 30, fill=col + (255,))
    d.text((58, y + 8), text, font=f, fill=(30, 24, 10) if dark else "white")


def heart(d, cx, cy, r, col):
    d.ellipse([cx - r, cy - r * 0.8, cx, cy + r * 0.2], fill=col)
    d.ellipse([cx, cy - r * 0.8, cx + r, cy + r * 0.2], fill=col)
    d.polygon([(cx - r * 0.96, cy - r * 0.1), (cx + r * 0.96, cy - r * 0.1), (cx, cy + r)], fill=col)


def phone_chat_tex(glow):
    """Телефон со скриншотом переписки зрителя; glow — подсветить просьбу «сними видео про инет друзей»."""
    shot = Image.open(os.path.join(ROOT, "assets", "viewer_chat.jpg")).convert("RGB")
    shot = shot.crop((0, 840, 1320, 2296))                     # только сообщения
    sw = 520
    sh = int(shot.height * sw / shot.width)
    shot = shot.resize((sw, sh), Image.LANCZOS)
    Wp, Hp = sw + 60, sh + 150
    im = rounded((Wp, Hp), 60, (14, 14, 18, 255), (90, 90, 110, 255), 6)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([Wp // 2 - 60, 24, Wp // 2 + 60, 50], 13, fill=(6, 6, 8, 255))
    m = Image.new("L", shot.size, 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, sw - 1, sh - 1], 28, fill=255)
    im.paste(shot, (30, 80), m)
    if glow:
        k = sw / 1320
        x0, y0, x1, y1 = 140 * k + 30, (1226 - 840) * k + 80, 900 * k + 30, (1402 - 840) * k + 80
        for i, a in enumerate((90, 160, 255)):
            d.rounded_rectangle([x0 - 12 + i * 4, y0 - 12 + i * 4, x1 + 12 - i * 4, y1 + 12 - i * 4], 26,
                                outline=GOLD_RGB + (a,), width=5)
    return mg.premul(im)


def avatar_tex(col, letter):
    S = 240
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([0, 0, S - 1, S - 1], fill=(24, 24, 32, 255), outline=col + (255,), width=10)
    d.ellipse([85, 45, 155, 115], fill=col + (255,))
    d.pieslice([55, 120, 185, 250], 180, 360, fill=col + (255,))
    return mg.premul(im)


def walther_tex():
    return card("ДЖОЗЕФ УОЛТЕР, 1996", "СБЛИЖАЮТСЯ БЫСТРЕЕ", "«гиперличное» общение в переписке", BLUE_RGB)


def brave_tex(n):
    """Почему в сети смелее: три пункта загораются по очереди."""
    im = rounded((780, 470), 34, (18, 18, 26, 255), (72, 72, 92, 255), 3)
    d = ImageDraw.Draw(im)
    header(d, "В СЕТИ МЫ СМЕЛЕЕ", CYAN_RGB, dark=True)
    rows = ["не видим лица", "меньше боимся осуждения", "рассказываем о себе больше"]
    for i, txt in enumerate(rows):
        y = 130 + i * 105
        on = i < n
        d.rounded_rectangle([36, y, 744, y + 86], 30, fill=(30, 48, 58, 255) if on else (28, 28, 36, 255))
        d.ellipse([56, y + 18, 106, y + 68], fill=CYAN_RGB + (255,) if on else (60, 60, 72, 255))
        if on:
            d.line([(68, y + 44), (79, y + 56), (96, y + 32)], fill=(10, 30, 40), width=7)
        d.text((128, y + 20), txt, font=mg.font(40), fill="white" if on else (110, 110, 125))
    return mg.premul(im)


def love_tex(p):
    im = rounded((760, 420), 34, (26, 14, 20, 255), PINK_RGB + (255,), 5)
    d = ImageDraw.Draw(im)
    header(d, "СТЭНФОРД, ДАННЫЕ 2017", PINK_RGB)
    heart(d, 170, 250, 110, (60, 30, 40, 255))
    # сердце заполняется снизу на 40%
    fill = Image.new("RGBA", im.size, (0, 0, 0, 0))
    heart(ImageDraw.Draw(fill), 170, 250, 110, PINK_RGB + (255,))
    top = int(360 - 220 * 0.4 * p)
    fill = fill.crop((0, top, im.width, im.height))
    im.paste(fill, (0, top), fill)
    d = ImageDraw.Draw(im)
    d.text((330, 140), "почти", font=mg.font(44), fill=(255, 190, 210))
    d.text((330, 190), f"{int(round(40 * p))}% пар", font=mg.font(96, tall=True), fill="white")
    d.text((330, 320), "познакомились онлайн", font=mg.font(34), fill=(255, 190, 210))
    return mg.premul(im)


def call_tex(state, ph):
    """Входящий звонок: state 0 — звонит, 1 — принят (идёт разговор, волна голоса)."""
    im = rounded((600, 520), 50, (16, 20, 18, 255), GREEN_RGB + (255,), 5)
    d = ImageDraw.Draw(im)
    d.ellipse([230, 40, 370, 180], fill=(40, 60, 90, 255), outline=CYAN_RGB + (255,), width=6)
    d.ellipse([275, 70, 325, 120], fill=CYAN_RGB + (255,))
    d.pieslice([250, 125, 350, 205], 180, 360, fill=CYAN_RGB + (255,))
    f = mg.font(46)
    t = "Интернет-друг"
    tw = d.textbbox((0, 0), t, font=f)[2]
    d.text((300 - tw / 2, 200), t, font=f, fill="white")
    if state == 0:
        s = "входящий звонок…"
        f2 = mg.font(34)
        tw = d.textbbox((0, 0), s, font=f2)[2]
        d.text((300 - tw / 2, 262), s, font=f2, fill=(150, 220, 180))
        r = 58 + int(8 * math.sin(ph * 2 * math.pi))
        d.ellipse([300 - r, 400 - r, 300 + r, 400 + r], fill=GREEN_RGB + (255,))
        d.rounded_rectangle([272, 380, 328, 420], 18, fill="white")
    else:
        for i in range(24):
            x = 60 + i * 20
            a = 0.25 + 0.75 * abs(math.sin(i * 0.8 + ph * 6))
            h = int(60 * a)
            d.rounded_rectangle([x, 380 - h, x + 10, 380 + h], 5, fill=GREEN_RGB + (255,))
        d.text((230, 268), "02:14", font=mg.font(40), fill=(150, 220, 180))
    return mg.premul(im)


def feel_tex(a, b):
    im = rounded((780, 440), 34, (18, 18, 26, 255), (72, 72, 92, 255), 3)
    d = ImageDraw.Draw(im)
    header(d, "ОПЫТ: ЗВОНОК ИЛИ ПЕРЕПИСКА", GREEN_RGB)
    for i, (txt, v, col) in enumerate((("ждали: будет неловко", a, GRAY_RGB), ("на деле: ближе, чем в чате", b, GREEN_RGB))):
        y = 130 + i * 150
        d.text((40, y), txt, font=mg.font(40), fill=(210, 210, 225))
        d.rounded_rectangle([40, y + 60, 740, y + 104], 22, fill=(40, 40, 52, 255))
        if v > 0.03:
            d.rounded_rectangle([40, y + 60, 40 + int(700 * v), y + 104], 22, fill=col + (255,))
    return mg.premul(im)


def comment_tex(text):
    im = rounded((760, 260), 40, (18, 18, 26, 255), PINK_RGB + (255,), 4)
    d = ImageDraw.Draw(im)
    header(d, "ТВОЙ КОММЕНТАРИЙ", PINK_RGB)
    d.rounded_rectangle([36, 120, 724, 216], 40, fill=(34, 34, 46, 255))
    d.text((70, 142), (text + "|") if text else "Добавить комментарий…", font=mg.font(42),
           fill="white" if text else (120, 120, 135))
    return mg.premul(im)


def share_tex():
    S = 220
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([0, 0, S - 1, S - 1], fill=TG_RGB + (255,))
    c = 110
    d.polygon([(c + 70, c - 10), (c + 5, c - 70), (c + 5, c - 35), (c - 70, c - 25), (c - 60, c + 60),
               (c + 5, c + 15), (c + 5, c + 50)], fill="white")
    return mg.premul(im)


def tags_tex(text):
    return chip(text, CYAN_RGB)


# ---------------------------------------------------------------- сцена
class Scene(base.Scene):
    cfg = sys.modules[__name__]

    def __init__(self, voice):
        super().__init__(voice)
        T = self.T
        T["chat"], T["chat_g"] = phone_chat_tex(False), phone_chat_tex(True)
        T["a1"], T["a2"] = avatar_tex(CYAN_RGB, "A"), avatar_tex(PINK_RGB, "B")
        T["thanks"] = stamp("СПАСИБО!", PINK_RGB)
        T["yes"] = stamp("ДА, БЫВАЮТ!", GREEN_RGB)
        T["walther"] = walther_tex()
        T["tags"] = [tags_tex(s) for s in ("игры", "музыка", "мемы", "аниме")]
        T["share"] = share_tex()
        T["sent"] = chip("ОТПРАВЛЕНО ДРУГУ", TG_RGB)
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
        return mg.Cam(yaw + 0.5 * wobble(t, 1), dist, hgt + 0.02 * wobble(t, 2), ty + 0.015 * wobble(t, 3), fov=fov,
                      roll=0.5 * wobble(t * 0.8, 4), tx=off * math.cos(a), tz=off * math.sin(a))

    def finish(self, frame, t):
        fr = super().finish(frame, t)
        for g0 in GLITCH:  # кадр рассыпается на крупные пиксели и цветные сдвиги, потом собирается
            if g0 < t < g0 + 0.4:
                q = 1 - abs(2 * (t - g0) / 0.4 - 1)
                b = int(2 + 46 * q)
                small = cv2.resize(fr, (max(1, W // b), max(1, H // b)), interpolation=cv2.INTER_AREA)
                fr = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)
                s = int(16 * q)
                if s:
                    fr = fr.copy()
                    fr[:, :, 2] = np.roll(fr[:, :, 2], s, 1)
                    fr[:, :, 0] = np.roll(fr[:, :, 0], -s, 1)
        return fr

    def link(self, cam, a, b, t, k, col=CYAN):
        """Светящаяся дуга между двумя точками, по ней бегут «пакеты» сигнала."""
        mid = (a + b) / 2 + [0, 0.35, 0]
        ss = np.linspace(0, 1, 32)
        pts = np.array([(1 - s) ** 2 * a + 2 * (1 - s) * s * mid + s * s * b for s in ss])
        uv, z = cam.proj(pts)
        if (z < 0.3).any():
            return
        lay = np.zeros_like(self.frame)
        for i in range(0, len(uv) - 1, 2):
            mg.lines(lay, uv[i:i + 2], tuple(int(c * 0.8 * k) for c in col), 5, closed=False)
        for j in range(3):
            q = (t * 0.9 + j / 3) % 1
            p = uv[int(q * (len(uv) - 1))]
            cv2.circle(lay, (int(p[0]), int(p[1])), 12, (230, 250, 255), -1, cv2.LINE_AA)
        cv2.add(self.frame, cv2.GaussianBlur(lay, (0, 0), 6), self.frame)
        cv2.add(self.frame, lay, self.frame)

    def items_for(self, t, cam, it):
        T = self.T
        P = lambda y0, *a, **k: place(y0, *a, **k)  # noqa: E731

        # 1. Крючок (персонаж справа): два человека далеко друг от друга, между ними бежит сигнал
        y0 = SEC["hook"]
        k = pop(t, 0.3, 2.85, 0.45, 0.3)
        if k > 0:
            a, b = P(y0, -1.0, 1.35, -0.3), P(y0, -0.42, 2.1, -0.3)
            self.add_bb(it, cam, a, 0.24 * k, T["a1"], anchor=(0.5, 0.5))
            self.add_bb(it, cam, b, 0.24 * k, T["a2"], anchor=(0.5, 0.5))
            it.append((cam.depth(a) + 0.05, lambda a=a, b=b, k=k: self.link(cam, a, b, t, k)))

        # 2. Заставка
        k = pop(t, 2.85, 4.2, 0.35, 0.3)
        if k > 0:
            self.add_panel(it, cam, place(0, 0.0, 1.95, 0.35), face(0, -6), 1.2 * k, 0.564 * k, 0.11 * k,
                           T["stinger"], pitch=-6, side=(30, 24, 120), edge=(200, 190, 255))

        # 3. Сообщение зрителя (персонаж слева): телефон влетает, вращаясь; просьба подсвечена; спасибо; да, бывают; Уолтер
        y0 = SEC["msg"]
        r_, c_ = basis(y0)
        k = pop(t, 4.3, 8.05, 0.6, 0.3)
        if k > 0:
            spin = 360 * (1 - smooth((t - 4.3) / 0.7))
            tex = T["chat_g"] if t > 5.6 else T["chat"]
            th, tw = tex.shape[:2]
            hh = 1.0
            pos = P(y0, 0.74, 1.5) + [0, 0.02 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 12) + spin, hh * tw / th * k, hh * k, 0.05 * k, tex,
                           side=(40, 40, 50), radius=0.06 * k)
            if t > 7.05:
                ks = clamp((t - 7.05) / 0.15)
                self.add_bb(it, cam, pos + c_ * 0.14 + [0, -0.45, 0], 0.15 * (1 + 0.6 * (1 - ks)) * k, T["thanks"],
                            anchor=(0.5, 0.5))
        k = pop(t, 9.3, 10.6, 0.15, 0.25)
        if k > 0:
            ks = clamp((t - 9.3) / 0.15)
            self.add_bb(it, cam, P(y0, 0.78, 1.8, 0.1), 0.17 * (1 + 0.6 * (1 - ks)) * k, T["yes"], anchor=(0.5, 0.5))
        k = pop(t, 10.6, C1, 0.4, 0.2)
        if k > 0:
            pos = P(y0, 0.72, 1.62) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 14), 0.9 * k, 0.48 * k, 0.06 * k, T["walther"], pitch=-6,
                           side=(40, 50, 90), edge=(200, 210, 255))

        # 4. Почему так (персонаж справа): смелее → интересы соединяют двоих → любовь онлайн
        y0 = SEC["why"]
        r_, c_ = basis(y0)
        k = pop(t, C1 + 0.05, 23.35, 0.4, 0.3)
        if k > 0:
            n = 1 + (t > 19.9) + (t > 21.4) if t > 18.8 else 0
            pos = P(y0, -0.72, 1.6) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, -12), 0.9 * k, 0.542 * k, 0.06 * k, self.cached("brave", brave_tex, n),
                           pitch=-5, side=(30, 60, 70), edge=(190, 240, 255))
        k = pop(t, 23.4, 26.3, 0.4, 0.3)
        if k > 0:
            a, b = P(y0, -1.12, 1.45, -0.2), P(y0, -0.36, 1.95, -0.2)
            self.add_bb(it, cam, a, 0.22 * k, T["a1"], anchor=(0.5, 0.5))
            self.add_bb(it, cam, b, 0.22 * k, T["a2"], anchor=(0.5, 0.5))
            it.append((cam.depth(a) + 0.05, lambda a=a, b=b, k=k: self.link(cam, a, b, t, k * smooth((t - 24.4) / 0.4))))
            for i, tex in enumerate(T["tags"]):
                kk = out_back(clamp((t - 24.3 - 0.15 * i) / 0.35)) * k
                if kk > 0.01:
                    q = (i + 0.5) / 4
                    pos = a * (1 - q) + b * q + [0, 0.26 + 0.05 * math.sin(t * 3 + i), 0]
                    self.add_bb(it, cam, pos, 0.085 * kk, tex, anchor=(0.5, 0.5))
        k = pop(t, 26.35, C2, 0.4, 0.2)
        if k > 0:
            p = round(smooth((t - 29.3) / 1.0) * 20) / 20
            pos = P(y0, -0.72, 1.62) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, -12), 0.9 * k, 0.497 * k, 0.06 * k, self.cached("love", love_tex, p),
                           pitch=-5, side=(70, 30, 50), edge=(255, 200, 220))

        # 5. Как укрепить (персонаж слева): звонок → опыт «неловко / ближе» → отправь другу → напиши комментарий
        y0 = SEC["call"]
        r_, c_ = basis(y0)
        k = pop(t, C2 + 0.1, 36.4, 0.4, 0.3)
        if k > 0:
            st = 1 if t > 35.5 else 0
            ph = round((t * 1.5) % 1 * 12) / 12
            pos = P(y0, 0.72, 1.62) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 12), 0.6 * k, 0.52 * k, 0.05 * k, self.cached("call", call_tex, st, ph),
                           pitch=-5, side=(30, 60, 40), radius=0.07)
        k = pop(t, 36.45, 43.3, 0.4, 0.3)
        if k > 0:
            a = round(0.7 * smooth((t - 37.4) / 0.8) * 20) / 20
            b = round(0.85 * smooth((t - 41.5) / 0.8) * 20) / 20
            pos = P(y0, 0.72, 1.6) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 12), 0.9 * k, 0.508 * k, 0.06 * k, self.cached("feel", feel_tex, a, b),
                           pitch=-5, side=(30, 60, 40), edge=(190, 255, 220))
        k = pop(t, 43.35, 47.45, 0.4, 0.3)
        if k > 0:
            friend = P(y0, 1.05, 1.95)
            self.add_bb(it, cam, friend, 0.24 * k, T["a2"], anchor=(0.5, 0.5))
            f = smooth((t - 45.8) / 0.6)
            start = P(y0, 0.5, 1.45)
            pos = start * (1 - f) + friend * f + [0, 0.3 * math.sin(math.pi * f), 0]
            if f < 0.97:
                self.add_bb(it, cam, pos, 0.2 * k * (1 - 0.4 * f), T["share"], anchor=(0.5, 0.5))
            if t > 46.4:
                self.add_bb(it, cam, P(y0, 0.8, 1.5), 0.11 * k, T["sent"], anchor=(0.5, 0.5))
        k = pop(t, 47.45, 51.2, 0.4, 0.3)
        if k > 0:
            msg = "Сними ролик про…"
            n = int(len(msg) * clamp((t - 48.6) / 1.4))
            pos = P(y0, 0.72, 1.62) + [0, 0.03 * math.sin(t * 2), 0]
            self.add_panel(it, cam, pos, face(y0, 12), 0.9 * k, 0.308 * k, 0.05 * k, self.cached("cm", comment_tex, msg[:n]),
                           pitch=-5, side=(70, 30, 50), edge=(255, 200, 220))

        # 6. Концовка с Telegram — общий шаблон
        self.outro_items(t, cam, it, OUTRO, SEC["tg"])


SFX = [(t, "whoosh") for t in (0.0, 2.85, 3.85) + GLITCH + (50.9,)]


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
        for t in (1.2, 2.4, 3.5, 4.6, 6.2, 7.4, 9.6, 12.5, 16.5, 19.0, 22.2, 25.2, 30.2, 32.75, 34.5, 36.0, 39.0,
                  42.3, 44.5, 46.6, 49.8, 54.0, 59.0):
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
    base.mix_audio(tmp, voice, out, music=os.path.join(ROOT, "work", "netfriends", "music.wav"), sfx_list=SFX, dur=DUR,
                   sfx_vol=0.15)  # «вжух» лёгкий и тихий
    os.remove(tmp)


if __name__ == "__main__":
    main()
