"""Мини-движок объёмной моушн-графики (2.5D) для вертикальных роликов 1080×1920.

Мир — настоящее 3D-пространство с камерой (облёт, наезды, «ручная» тряска):
пол-сетка в перспективе, объёмные карточки с толщиной и тенью, 3D-столбики, коробки-здания.
Персонаж — нарисованная картинка, разрезанная на части (голова, руки, тело) и стоящая
в мире как «щит», развёрнутый к камере. Всё рисуется кадр за кадром (numpy + OpenCV + Pillow)
и сразу отдаётся в ffmpeg.
"""
import math
import os
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1920
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT_BOLD = os.path.join(ROOT, "fonts", "Inter-ExtraBold.ttf")
FONT_TALL = os.path.join(ROOT, "fonts", "Oswald-Bold.ttf")


# ---------------------------------------------------------------- время и плавность
def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def smooth(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def out_cubic(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def in_out_cubic(x):
    x = clamp(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def out_back(x, s=1.9):
    x = clamp(x)
    return 1 + (s + 1) * (x - 1) ** 3 + s * (x - 1) ** 2


def in_out_expo(x):
    x = clamp(x)
    if x in (0.0, 1.0):
        return x
    return 2 ** (20 * x - 10) / 2 if x < 0.5 else (2 - 2 ** (-20 * x + 10)) / 2


EASE = {"smooth": smooth, "out": out_cubic, "inout": in_out_cubic, "back": out_back,
        "expo": in_out_expo, "lin": lambda x: clamp(x)}


def track(keys, t):
    """Ключи [(время, значение[, плавность])] → значение в момент t (числа или кортежи)."""
    if t <= keys[0][0]:
        return keys[0][1]
    for k0, k1 in zip(keys, keys[1:]):
        if t <= k1[0]:
            e = EASE[k1[2] if len(k1) > 2 else "smooth"]((t - k0[0]) / max(1e-6, k1[0] - k0[0]))
            a, b = k0[1], k1[1]
            if isinstance(a, tuple):
                return tuple(x + (y - x) * e for x, y in zip(a, b))
            return a + (b - a) * e
    return keys[-1][1]


def wobble(t, seed, freqs=(0.37, 0.71, 1.13)):
    """Плавный «шум» из синусов — для ручной камеры и живости."""
    return sum(math.sin(t * f * 2 * math.pi + seed * (i + 1) * 1.7) / (i + 1) for i, f in enumerate(freqs)) / 1.83


# ---------------------------------------------------------------- камера
class Cam:
    """Камера на орбите вокруг точки (0, target_y, 0). yaw — угол облёта в градусах."""

    def __init__(self, yaw, dist, height, target_y, fov=50.0, roll=0.0, tx=0.0, tz=0.0):
        a = math.radians(yaw)
        self.yaw = a
        self.C = np.array([tx + dist * math.sin(a), height, tz - dist * math.cos(a)])
        T = np.array([tx, target_y, tz])
        f = T - self.C
        f /= np.linalg.norm(f)
        r = np.cross([0.0, 1.0, 0.0], f)
        r /= np.linalg.norm(r)
        u = np.cross(f, r)
        cr, sr = math.cos(math.radians(roll)), math.sin(math.radians(roll))
        self.r, self.u = r * cr + u * sr, -r * sr + u * cr
        self.f = f
        self.focal = (H / 2) / math.tan(math.radians(fov) / 2)
        self.flat_right = np.array([math.cos(a), 0.0, math.sin(a)])  # «вправо» в плоскости пола

    def proj(self, P):
        P = np.asarray(P, dtype=np.float64) - self.C
        x, y, z = P @ self.r, P @ self.u, P @ self.f
        zc = np.maximum(z, 1e-3)
        return np.stack([W / 2 + self.focal * x / zc, H / 2 - self.focal * y / zc], -1), z

    def depth(self, P):
        return float((np.asarray(P) - self.C) @ self.f)


# ---------------------------------------------------------------- рисование
SHIFT = 4  # субпиксельная точность для cv2 (координаты × 16)


def _pts(uv):
    return np.round(np.asarray(uv) * (1 << SHIFT)).astype(np.int32)


def poly(img, uv, color, aa=True):
    cv2.fillConvexPoly(img, _pts(uv), color, cv2.LINE_AA if aa else cv2.LINE_8, SHIFT)


def lines(img, uv, color, thick=2, closed=True):
    cv2.polylines(img, [_pts(uv)], closed, color, thick, cv2.LINE_AA, SHIFT)


def over(frame, rgba, x0, y0):
    """Наложить RGBA (float 0..1, премультиплицированный) на кадр uint8 BGR с позиции (x0, y0)."""
    h, w = rgba.shape[:2]
    X0, Y0, X1, Y1 = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if X1 <= X0 or Y1 <= Y0:
        return
    src = rgba[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    dst = frame[Y0:Y1, X0:X1].astype(np.float32) / 255.0
    a = src[..., 3:4]
    out = dst * (1 - a) + src[..., :3]
    frame[Y0:Y1, X0:X1] = np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)


def warp_onto(frame, rgba, quad):
    """Натянуть картинку RGBA (премульт.) на четырёхугольник quad (TL, TR, BR, BL) в кадре."""
    quad = np.asarray(quad, dtype=np.float32)
    x0, y0 = np.floor(quad.min(0)).astype(int) - 2
    x1, y1 = np.ceil(quad.max(0)).astype(int) + 2
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    h, w = rgba.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src, quad - np.float32([x0, y0]))
    roi = cv2.warpPerspective(rgba, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    over(frame, roi, x0, y0)


def premul(img_rgba_u8):
    """PIL RGBA → float32 BGRA премультиплицированный (формат для over/warp_onto)."""
    a = np.asarray(img_rgba_u8, dtype=np.float32) / 255.0
    rgb = a[..., [2, 1, 0]] * a[..., 3:4]
    return np.concatenate([rgb, a[..., 3:4]], -1)


def font(size, tall=False):
    return ImageFont.truetype(FONT_TALL if tall else FONT_BOLD, size)


# ---------------------------------------------------------------- объекты мира
def floor_grid(frame, cam, step=0.5, extent=7.0, base=150, color=(1.0, 1.0, 1.0), fade=9.0, radius=None, mask=None):
    """Сетка на полу в перспективе — главный «объёмный» ориентир сцены."""
    layer = np.zeros((H, W), np.uint8)
    n = int(extent / step)
    seg = 28
    for k in range(-n, n + 1):
        c = k * step
        for axis in (0, 1):
            ts = np.linspace(-extent, extent, seg + 1)
            P = np.zeros((seg + 1, 3))
            P[:, axis] = c
            P[:, 1 - axis if axis == 1 else 2] = ts
            if axis == 0:
                P[:, 2] = ts
            else:
                P[:, 0] = ts
                P[:, 2] = c
            uv, z = cam.proj(P)
            d = np.linalg.norm(P - cam.C, axis=1)
            for i in range(seg):
                if z[i] < 0.3 or z[i + 1] < 0.3:
                    continue
                if radius is not None and (np.hypot(P[i, 0], P[i, 2]) > radius or np.hypot(P[i + 1, 0], P[i + 1, 2]) > radius):
                    continue
                v = int(base * clamp(1 - ((d[i] + d[i + 1]) / 2) / fade) ** 1.4)
                if v < 4:
                    continue
                cv2.line(layer, tuple(_pts(uv[i])), tuple(_pts(uv[i + 1])), v, 2, cv2.LINE_AA, SHIFT)
    if mask is not None:
        layer[~mask] = 0
    col = np.array(color, np.float32)
    add = (layer[..., None].astype(np.float32) * col).astype(np.uint8)
    cv2.add(frame, add, frame)


def floor_blob(frame, cam, center, rx, rz, strength, color=(0, 0, 0), blur=41, mode="shadow"):
    """Тень (или световое пятно) на полу: эллипс в перспективе, размытый."""
    a = np.linspace(0, 2 * math.pi, 48, endpoint=False)
    P = np.stack([center[0] + rx * np.cos(a), np.zeros_like(a), center[1] + rz * np.sin(a)], -1)
    uv, z = cam.proj(P)
    if (z < 0.3).any():
        return
    x0, y0 = np.floor(uv.min(0)).astype(int) - blur
    x1, y1 = np.ceil(uv.max(0)).astype(int) + blur
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if x1 <= x0 or y1 <= y0:
        return
    m = np.zeros((y1 - y0, x1 - x0), np.float32)
    poly(m, uv - [x0, y0], 1.0)
    k = blur | 1
    m = cv2.GaussianBlur(m, (k, k), 0) * strength
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    if mode == "shadow":
        roi *= (1 - m[..., None])
    else:
        roi += m[..., None] * np.array(color, np.float32)
    frame[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)


def box(frame, cam, center, size, height, color, base_y=0.0, edge=None, yaw=0.0, glow=None):
    """Объёмный параллелепипед (столбик графика, здание) со светом по граням."""
    cx, cz = center
    w, d = size
    ca, sa = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    loc = [(-w / 2, -d / 2), (w / 2, -d / 2), (w / 2, d / 2), (-w / 2, d / 2)]
    ring = [(cx + x * ca - z * sa, cz + x * sa + z * ca) for x, z in loc]
    V = np.array([(x, base_y, z) for x, z in ring] + [(x, base_y + height, z) for x, z in ring])
    faces = [((4, 5, 6, 7), (0, 1, 0), 1.25),   # верх — светлее
             ((0, 1, 5, 4), None, 1.0), ((1, 2, 6, 5), None, 0.78),
             ((2, 3, 7, 6), None, 0.62), ((3, 0, 4, 7), None, 0.85)]
    uv, z = cam.proj(V)
    if (z < 0.3).any() or height <= 1e-4:
        return
    drawn = []
    for idx, n, shade in faces:
        Pf = V[list(idx)]
        cen = Pf.mean(0)
        if n is None:
            e1, e2 = Pf[1] - Pf[0], Pf[3] - Pf[0]
            n = np.cross(e2, e1)
        if np.dot(n, cam.C - cen) <= 0:
            continue
        # свет сверху-слева относительно камеры: грань, смотрящая на «ключевой свет», ярче
        col = tuple(float(min(255, c * shade)) for c in color)
        poly(frame, uv[list(idx)], col)
        drawn.append(idx)
    if edge is not None:
        for idx in drawn:
            lines(frame, uv[list(idx)], edge, 2)
    if glow is not None:
        top = uv[[4, 5, 6, 7]]
        g = np.zeros((H, W, 3), np.uint8)
        lines(g, top, glow, 6)
        g = cv2.GaussianBlur(g, (31, 31), 0)
        cv2.add(frame, g, frame)


def panel(frame, cam, center, yaw, w, h, thick, tex, pitch=0.0, roll=0.0, side=(40, 40, 50), edge=None,
          back_tex=None):
    """Объёмная карточка: картинка-лицо + толщина-бортик. tex — float BGRA премульт."""
    a, b, c = math.radians(yaw), math.radians(pitch), math.radians(roll)
    Ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    Rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    R = Ry @ Rx @ Rz
    loc = np.array([(-w / 2, h / 2), (w / 2, h / 2), (w / 2, -h / 2), (-w / 2, -h / 2)])
    front = np.array([(x, y, -thick / 2) for x, y in loc]) @ R.T + center
    back = np.array([(x, y, thick / 2) for x, y in loc]) @ R.T + center
    uvf, zf = cam.proj(front)
    uvb, zb = cam.proj(back)
    if (zf < 0.3).any() or (zb < 0.3).any():
        return
    normal = R @ np.array([0, 0, -1.0])
    facing = np.dot(normal, cam.C - center) > 0
    hull = cv2.convexHull(np.concatenate([uvf, uvb]).astype(np.float32)).reshape(-1, 2)
    poly(frame, hull, side)
    if edge is not None:
        lines(frame, uvf[:2], edge, 3, closed=False)  # светлая кромка сверху
    if facing:
        warp_onto(frame, tex, uvf)
    else:
        if back_tex is not None:
            warp_onto(frame, back_tex, uvb[[1, 0, 3, 2]])
        else:
            poly(frame, uvb, tuple(c * 0.7 for c in side))


def billboard(frame, cam, pos, height, tex, facing_yaw=None, anchor=(0.5, 1.0), squash=(1.0, 1.0)):
    """Плоская картинка в мире, развёрнутая к камере (или под углом facing_yaw)."""
    hh, ww = tex.shape[:2]
    d = max(0.3, cam.depth(pos))
    px = height * squash[1] * cam.focal / d  # примерная высота на экране в пикселях
    if px < hh * 0.85:
        k = max(px * 1.15 / hh, 0.05)
        tex = cv2.resize(tex, (max(2, int(ww * k)), max(2, int(hh * k))), interpolation=cv2.INTER_AREA)
        hh, ww = tex.shape[:2]
    a = cam.yaw if facing_yaw is None else math.radians(facing_yaw)
    right = np.array([math.cos(a), 0.0, math.sin(a)])
    up = np.array([0.0, 1.0, 0.0])
    hw = height * ww / hh
    sx, sy = squash
    o = np.asarray(pos, dtype=np.float64)
    ax, ay = anchor
    # углы: anchor — какая точка картинки стоит в pos (по умолчанию середина низа)
    TL = o - right * ax * hw * sx + up * (ay * height * sy)
    TR = o + right * (1 - ax) * hw * sx + up * (ay * height * sy)
    BR = o + right * (1 - ax) * hw * sx - up * ((1 - ay) * height * sy)
    BL = o - right * ax * hw * sx - up * ((1 - ay) * height * sy)
    uv, z = cam.proj(np.array([TL, TR, BR, BL]))
    if (z < 0.3).any():
        return
    warp_onto(frame, tex, uv)


def text_rgba(text, size, color=(255, 255, 255), tall=False, stroke=0, stroke_color=(0, 0, 0), pad=10):
    f = font(size, tall)
    l, t, r, b = f.getbbox(text, stroke_width=stroke)
    im = Image.new("RGBA", (r - l + 2 * pad, b - t + 2 * pad), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((pad - l, pad - t), text, font=f, fill=color + (255,),
                            stroke_width=stroke, stroke_fill=stroke_color + (255,))
    return im


# ---------------------------------------------------------------- фон
class Backdrop:
    """Зацикленный фон-панорама вокруг сцены: при облёте камеры плавно прокручивается."""

    def __init__(self, path, dim=0.75, tiles_per_turn=4):
        img = cv2.imread(path)
        s = (H * 1.2) / img.shape[0]
        img = cv2.resize(img, (int(img.shape[1] * s), int(H * 1.2)), interpolation=cv2.INTER_CUBIC)
        self.img = (img.astype(np.float32) * dim).astype(np.uint8)
        self.tile = self.img.shape[1]
        self.per_rad = self.tile * tiles_per_turn / (2 * math.pi)

    def render(self, cam):
        off = int(round(cam.yaw * self.per_rad)) % self.tile
        y = int(clamp((H * 0.1) - (cam.C[1] - 1.0) * 60, 0, H * 0.2))
        reps = W // self.tile + 2
        strip = np.concatenate([self.img] * reps, 1)
        return strip[y:y + H, off:off + W].copy()


class Room:
    """Круглая «комната» вокруг сцены: стена — зацикленная картинка, натянутая на цилиндр
    радиуса R, пол — сплошная плоскость. Стык стены и пола получается геометрически точным:
    каждый пиксель кадра — луч из камеры, который попадает либо в пол, либо в стену."""

    def __init__(self, path, radius=9.0, tiles=12, dim=0.72, floor_color=(16, 13, 18), upscale=3):
        src = cv2.imread(path)
        if upscale > 1:  # аккуратное увеличение один раз — линии сетки на стене не мылятся
            src = cv2.resize(src, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_LANCZOS4)
            blur = cv2.GaussianBlur(src, (0, 0), 1.2 * upscale)
            src = cv2.addWeighted(src, 1.6, blur, -0.6, 0)
        tex = src.astype(np.float32) * dim
        self.tex = tex
        self.th, self.tw = tex.shape[:2]
        self.R = radius
        self.tiles = tiles
        self.tile_w = 2 * math.pi * radius / tiles
        self.tile_h = self.tile_w * self.th / self.tw
        self.floor_color = np.array(floor_color, np.float32)
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
        self.px, self.py = xs - W / 2, -(ys - H / 2)
        self.floor_mask = None

    def render(self, cam):
        f = cam.focal
        dx, dy = self.px / f, self.py / f
        D = [dx * cam.r[i] + dy * cam.u[i] + cam.f[i] for i in range(3)]
        Cx, Cy, Cz = cam.C
        a = D[0] ** 2 + D[2] ** 2 + 1e-9
        b = 2 * (Cx * D[0] + Cz * D[2])
        c = Cx * Cx + Cz * Cz - self.R ** 2
        tw = (-b + np.sqrt(np.maximum(b * b - 4 * a * c, 0))) / (2 * a)
        with np.errstate(divide="ignore", invalid="ignore"):
            tf = np.where(D[1] < -1e-6, -Cy / D[1], np.inf)
        floor = tf < tw
        self.floor_mask = floor
        # стена
        hx, hy, hz = Cx + tw * D[0], Cy + tw * D[1], Cz + tw * D[2]
        theta = np.arctan2(hx, -hz)
        mapx = ((theta / (2 * math.pi)) * self.tiles * self.tw) % self.tw
        mapy = ((self.tile_h - hy) / self.tile_h * self.th) % self.th
        wall = cv2.remap(self.tex, mapx.astype(np.float32), mapy.astype(np.float32), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_WRAP)
        ao_wall = 0.5 + 0.5 * np.clip(hy / 0.9, 0, 1)              # угол у пола темнее
        top = np.clip(1.15 - hy / (self.tile_h * 0.9), 0.55, 1.0)   # к потолку темнее
        wall *= (ao_wall * top)[..., None]
        # пол
        fx, fz = Cx + tf * D[0], Cz + tf * D[2]
        r = np.sqrt(np.where(floor, fx * fx + fz * fz, 0))
        ao_floor = 0.45 + 0.55 * np.clip((self.R - r) / 1.2, 0, 1)
        floor_col = self.floor_color[None, None, :] * ao_floor[..., None]
        out = np.where(floor[..., None], floor_col, wall)
        return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- персонаж-«кукла»
class Puppet:
    """Рисунок персонажа в позе «Т», разрезанный на части: руки, тело, голова (+ челюсть маски).

    Координаты разреза подобраны под assets/char_tpose.jpg (832×1216).
    """
    M = 90  # поля вокруг рисунка, чтобы поднятые руки не обрезались

    def __init__(self, path, cfg):
        self.cfg = cfg
        img = cv2.imread(path)
        alpha = self._alpha(img)
        img, alpha, pad = self._cap_hands(img, alpha)
        h, w = img.shape[:2]
        M = self.M
        ox, oy = M + pad, M  # где на холсте оказывается точка (0, 0) исходного рисунка
        self.Wc, self.Hc = w + 2 * M, h + 2 * M
        canvas = np.zeros((self.Hc, self.Wc, 3), np.uint8)
        canvas[M:M + h, M:M + w] = img
        a = np.zeros((self.Hc, self.Wc), np.float32)
        a[M:M + h, M:M + w] = alpha
        yy, xx = np.mgrid[0:self.Hc, 0:self.Wc]
        X, Y = xx - ox, yy - oy
        c = cfg
        arm_band = (Y > c["arm_y"][0]) & (Y < c["arm_y"][1])
        masks = {
            "armL": arm_band & (X < c["cut_L"]),
            "armR": arm_band & (X > c["cut_R"]),
            "head": Y < c["neck_y"],
        }
        masks["body"] = (Y >= c["neck_y"] - 14) & (Y < c["feet"][1] + 12) & ~masks["armL"] & ~masks["armR"]
        # круглые «плечи» у тела: срез руки становится круглым и с контуром — щели при взмахе не видно
        body_rgb = canvas.copy()
        for cx in (c["cut_L"] + 12, c["cut_R"] - 12):
            d = np.hypot(X - cx, Y - c["shoulder_y"])
            r = c["shoulder_r"]
            disk = (d < r) & (a > 0.5)
            masks["body"] |= disk
            ring = disk & (d > r - 5) & ((X < c["cut_L"]) | (X > c["cut_R"]))
            body_rgb[ring] = (20, 20, 20)
        self.parts = {}
        for name, m in masks.items():
            pa = a * m
            self.parts[name] = self._shade(body_rgb if name == "body" else canvas, pa)
        self.pivot = {k: (float(v[0] + ox), float(v[1] + oy)) for k, v in c["pivots"].items()}
        self.jaw = (c["jaw_y"][0] + oy, c["jaw_y"][1] + oy)
        self.feet = (c["feet"][0] + ox, c["feet"][1] + oy)
        self.top = c["top_y"] + oy

    @staticmethod
    def _alpha(img):
        h, w = img.shape[:2]
        white = (img.min(axis=2) > 205).astype(np.uint8)
        ff = white.copy()
        for x in range(0, w, 6):
            for y in (0, h - 1):
                if ff[y, x] == 1:
                    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (x, y), 2)
        for y in range(0, h, 6):
            for x in (0, w - 1):
                if ff[y, x] == 1:
                    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (x, y), 2)
        a = (ff != 2).astype(np.uint8) * 255
        a = cv2.erode(a, np.ones((3, 3), np.uint8))
        return cv2.GaussianBlur(a, (3, 3), 0).astype(np.float32) / 255.0

    @staticmethod
    def _cap_hands(img, alpha):
        """Кисти на рисунке обрезаны краем картинки — дорисовываем круглые кулачки."""
        h, w = img.shape[:2]
        pad = 40
        big = np.full((h, w + 2 * pad, 3), 255, np.uint8)
        big[:, pad:pad + w] = img
        ab = np.zeros((h, w + 2 * pad), np.float32)
        ab[:, pad:pad + w] = alpha
        for col, sgn in ((0, -1), (w - 1, 1)):
            rows = np.where(alpha[:, col] > 0.5)[0]
            if len(rows) < 6:
                continue
            y0, y1 = rows.min(), rows.max()
            yc, ry = (y0 + y1) / 2, (y1 - y0) / 2
            xs = col + pad
            reds = img[y0:y1, (col + 8 if sgn < 0 else col - 8)]
            reds = reds[(reds[:, 2] > 150) & (reds[:, 1] < 110)]
            red = np.median(reds, 0) if len(reds) else np.array([40, 40, 220])
            cap = np.zeros_like(ab)
            cv2.ellipse(cap, (int(xs), int(yc)), (int(ry * 0.75), int(ry)), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
            side = np.zeros_like(ab)
            if sgn < 0:
                side[:, :xs + 1] = 1
            else:
                side[:, xs:] = 1
            cap *= side
            fill = big.copy()
            cv2.ellipse(fill, (int(xs), int(yc)), (int(ry * 0.75), int(ry)), 0, 0, 360,
                        tuple(int(v) for v in red), -1, cv2.LINE_AA)
            cv2.ellipse(fill, (int(xs), int(yc)), (int(ry * 0.75), int(ry)), 0, 0, 360, (20, 20, 20), 5, cv2.LINE_AA)
            sel = cap > 0.01
            big[sel] = (fill[sel] * cap[sel][:, None] + big[sel] * (1 - cap[sel][:, None])).astype(np.uint8)
            ab = np.maximum(ab, cap)
        return big, ab, pad

    def _shade(self, canvas, a):
        """«Подушечный» объём: края части чуть темнее, центр светлее — рисунок выглядит выпуклым."""
        k = 61
        soft = cv2.GaussianBlur(a, (k, k), 0)
        shade = 0.70 + 0.38 * np.clip(soft / 0.75, 0, 1)
        rgb = canvas.astype(np.float32) / 255.0 * shade[..., None]
        rgb = np.clip(rgb, 0, 1)
        return np.concatenate([rgb * a[..., None], a[..., None]], -1).astype(np.float32)

    def _rot(self, part, center, deg, dx=0.0, dy=0.0):
        Mx = cv2.getRotationMatrix2D(center, deg, 1.0)
        Mx[:, 2] += (dx, dy)
        return cv2.warpAffine(part, Mx, (self.Wc, self.Hc), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    def render(self, armL=62.0, armR=62.0, head=0.0, jaw=0.0, head_dy=0.0, rim=1.0):
        """armL/armR — насколько опущена рука от горизонтали (градусы, минус — вверх).
        head — наклон головы, jaw 0..1 — открытие нижней части маски."""
        out = np.zeros((self.Hc, self.Wc, 4), np.float32)

        def put(layer):
            a = layer[..., 3:4]
            out[...] = out * (1 - a) + layer

        put(self._rot(self.parts["armL"], self.pivot["armL"], armL))
        put(self._rot(self.parts["armR"], self.pivot["armR"], -armR))
        put(self.parts["body"])
        hd = self.parts["head"]
        if jaw > 0.01:
            y0, y1 = self.jaw
            s = 1.0 + 0.2 * jaw
            region = hd[y0 - 2:y1 + 40]
            hgt = region.shape[0]
            mapy = (np.arange(int(hgt * s)) / s).astype(np.float32)
            mapy = np.clip(mapy, 0, hgt - 1)
            mx, my = np.meshgrid(np.arange(self.Wc, dtype=np.float32), mapy)
            stretched = cv2.remap(region, mx, my, cv2.INTER_LINEAR)
            hd = hd.copy()
            hd[y0 - 2:] = 0
            end = min(self.Hc, y0 - 2 + stretched.shape[0])
            hd[y0 - 2:end] = stretched[:end - (y0 - 2)]
        put(self._rot(hd, self.pivot["head"], head, 0, head_dy))
        if rim > 0:
            # контровой свет: тонкая светлая кромка справа-сверху — отделяет персонажа от фона
            a = out[..., 3]
            sh = np.zeros_like(a)
            sh[6:, :-7] = a[:-6, 7:]
            edge = np.clip(a - sh, 0, 1) * rim
            edge = cv2.GaussianBlur(edge, (7, 7), 0) * a
            out[..., :3] += edge[..., None] * 0.18
        return out


# ---------------------------------------------------------------- субтитры
def subtitles(frame, t, subs, y=1560, accent=(226, 38, 46)):
    """subs: [(начало, текст, выделенные слова)] — фраза держится до начала следующей.
    Фраза «выпрыгивает», ключевые слова — в цветной плашке (accent — RGB)."""
    for j, item in enumerate(subs):
        t0, text = item[0], item[1]
        keys = item[2] if len(item) > 2 else ()
        acc = item[3] if len(item) > 3 else accent
        end = subs[j + 1][0] if j + 1 < len(subs) else t0 + 1.4
        if not (t0 <= t < end) or not text:
            continue
        k = out_back((t - t0) / 0.16)
        fade = clamp((end - t) / 0.06)
        size = 104
        words = text.split()
        im = Image.new("RGBA", (W, 260), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        while True:  # подбираем размер, чтобы фраза целиком влезала в кадр с полями
            f = font(size)
            space = size * 0.28
            widths = [d.textbbox((0, 0), w_, font=f)[2] for w_ in words]
            total = sum(widths) + space * (len(words) - 1) + 32 * sum(
                1 for w_ in words if w_.strip("«»,.!?:—-") in keys or w_ in keys)
            if total <= W - 110 or size <= 56:
                break
            size -= 4
        x = (W - total) / 2
        for w_, ww in zip(words, widths):
            if w_.strip("«»,.!?:—-") in keys or w_ in keys:
                d.rounded_rectangle([x - 16, 40, x + ww + 16, 60 + size * 1.12], 22, fill=tuple(acc) + (255,))
                d.text((x, 48), w_, font=f, fill="white")
            else:
                d.text((x + 5, 53), w_, font=f, fill=(0, 0, 0, 170))
                d.text((x, 48), w_, font=f, fill="white", stroke_width=3, stroke_fill=(0, 0, 0))
            x += ww + space
        arr = premul(im) * fade
        sc = 0.55 + 0.45 * k
        arr = cv2.resize(arr, None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
        h, w = arr.shape[:2]
        over(frame, arr, (W - w) // 2, int(y - h / 2 + (1 - k) * 40))


# ---------------------------------------------------------------- звук
def voice_envelope(wav, fps):
    """Громкость голоса по кадрам (0..1) — для челюсти маски и кивков."""
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", wav, "-ac", "1", "-ar", "16000",
                          "-f", "s16le", "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    hop = 16000 // fps
    n = len(x) // hop
    rms = np.sqrt(np.mean(x[:n * hop].reshape(n, hop) ** 2, 1))
    rms = rms / (np.percentile(rms, 95) + 1e-6)
    return np.clip(np.convolve(rms, [0.25, 0.5, 0.25], "same"), 0, 1.3)


def sfx(path, events, dur, sr=44100):
    """Звуки моушн-графики (свист пролёта, «поп», щелчок, гул) — синтезируются, без чужих файлов."""
    rng = np.random.default_rng(1)
    out = np.zeros(int(dur * sr), np.float32)
    for t, kind in events:
        i = int(t * sr)
        if kind == "whoosh":
            n = int(0.55 * sr)
            noise = rng.standard_normal(n).astype(np.float32)
            env = np.sin(np.linspace(0, math.pi, n)) ** 2
            # фильтр, «едущий» по частоте: сглаживание с меняющимся окном
            y = np.zeros(n, np.float32)
            acc = 0.0
            for j in range(n):
                k = 0.02 + 0.25 * (j / n)
                acc += k * (noise[j] - acc)
                y[j] = acc
            s = y * env * 0.9
        elif kind == "pop":
            n = int(0.18 * sr)
            tt = np.arange(n) / sr
            s = np.sin(2 * math.pi * (300 + 900 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 22) * 0.55
        elif kind == "click":
            n = int(0.08 * sr)
            tt = np.arange(n) / sr
            s = (np.sin(2 * math.pi * 1800 * tt) * np.exp(-tt * 90)) * 0.5
        elif kind == "rumble":
            n = int(1.0 * sr)
            tt = np.arange(n) / sr
            s = (np.sin(2 * math.pi * 55 * tt) + 0.5 * np.sin(2 * math.pi * 82 * tt)) * np.sin(np.linspace(0, math.pi, n)) * 0.35
        elif kind == "boom":
            n = int(0.7 * sr)
            tt = np.arange(n) / sr
            s = np.sin(2 * math.pi * (50 + 60 * np.exp(-tt * 8)) * tt) * np.exp(-tt * 5) * 0.8
        else:
            continue
        s = s.astype(np.float32)
        out[i:i + len(s)] += s[:max(0, len(out) - i)]
    out = np.clip(out, -1, 1)
    pcm = (out * 32767).astype(np.int16).tobytes()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", "-", path],
                   input=pcm, check=True)
    return path


class Writer:
    """Кадры → ffmpeg (без промежуточных картинок на диске)."""

    def __init__(self, path, fps):
        self.p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                   "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "medium",
                                   "-crf", "15", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)

    def write(self, frame):
        self.p.stdin.write(frame.tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()
