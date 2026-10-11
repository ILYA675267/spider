"""Позы и жесты персонажа + анимация по ключевым кадрам.

Поза — словарь {шарнир: (X, Y, Z) в градусах}. Для рук задаётся одна сторона
("L"/"R"), зеркалирование делает mirror(). Жесты ставятся на время в секундах:

    anim = Animator(P, fps=24)
    anim.pose(0.0, "idle")
    anim.pose(1.2, "wave_R")
    ...
"""
import math
import random

import bpy

# ---------------------------------------------------------------- руки
# плечо (ZYX): Z — поворот руки вокруг своей оси, Y — отвести в сторону, X — вперёд(-)/назад(+)
# локоть: X — согнуть (минус = кисть вперёд/вверх)

ARM = {
    "down":      {"shoulder": (4, -8, 0),     "elbow": (-12, 0, 0), "wrist": (0, 0, 0),  "hand": "relaxed"},
    "hip":       {"shoulder": (20, -48, -75), "elbow": (-95, 0, 0), "wrist": (0, 0, 30), "hand": "fist"},
    "point_side": {"shoulder": (0, -85, 0),   "elbow": (-8, 0, 0),  "wrist": (0, 0, -90), "hand": "point"},
    "point_up":  {"shoulder": (-15, -150, 0), "elbow": (-10, 0, 0), "wrist": (0, 0, -90), "hand": "point"},
    "point_cam": {"shoulder": (-80, -15, 0),  "elbow": (-15, 0, 0), "wrist": (0, 0, 0),  "hand": "point"},
    "wave_a":    {"shoulder": (0, -70, 0),    "elbow": (0, -85, 0), "wrist": (0, 0, 0),  "hand": "open"},
    "wave_b":    {"shoulder": (0, -70, 0),    "elbow": (0, -55, 0), "wrist": (0, 0, 0),  "hand": "open"},
    "open":      {"shoulder": (-35, -30, 0),  "elbow": (-55, 0, 0), "wrist": (0, 0, -70), "hand": "open"},
    "chest":     {"shoulder": (-30, -25, -80),   "elbow": (-110, 0, 0), "wrist": (0, 0, 0), "hand": "open"},
    "cheer":     {"shoulder": (-10, -140, 0), "elbow": (-25, 0, 0), "wrist": (0, 0, 0),  "hand": "fist"},
    "think":     {"shoulder": (-50, -10, 0),  "elbow": (-140, 0, 0), "wrist": (0, 0, 0), "hand": "fist"},
}

HAND = {  # (указательный, три пальца, большой) — угол сгиба
    "relaxed": (-25, -35, 20),
    "fist":    (-100, -105, 60),
    "point":   (0, -110, 55),
    "open":    (0, 0, 35),
}


def arm(side, name):
    """Поза руки для стороны side ("L"/"R"). Значения в ARM заданы для левой руки."""
    a = ARM[name]
    s = 1 if side == "L" else -1
    out = {}
    sx, sy, sz = a["shoulder"]
    out["shoulder_" + side] = (sx, sy * s, sz * s)
    ex, ey, ez = a["elbow"]
    out["elbow_" + side] = (ex, ey * s, ez * s)
    wx, wy, wz = a["wrist"]
    out["wrist_" + side] = (wx, wy * s, wz * s)
    ix, fx, tx = HAND[a["hand"]]
    out["index_" + side] = (ix, 0, 0)
    for i in range(3):
        out[f"finger{i}_{side}"] = (fx, 0, 0)
    out["thumb_" + side] = (tx, 0, 0)
    return out


def body(spine=(0, 0, 0), head=(0, 0, 0), hips=(0, 0, 0), legs="stand"):
    out = {"spine": spine, "head": head, "hips": hips}
    if legs == "stand":
        out.update({"hip_L": (0, 0, 0), "hip_R": (0, 0, 0), "knee_L": (0, 0, 0), "knee_R": (0, 0, 0)})
    elif legs == "contrapposto":  # вес на одной ноге, вторая чуть вперёд — как на картинке
        out.update({"hip_L": (-8, 0, 4), "hip_R": (2, 0, -3), "knee_L": (12, 0, 0), "knee_R": (0, 0, 0)})
    elif legs == "crouch":
        out.update({"hip_L": (-35, 0, 0), "hip_R": (-35, 0, 0), "knee_L": (60, 0, 0), "knee_R": (60, 0, 0)})
    return out


def pose(left="down", right="down", **kw):
    p = body(**kw)
    p.update(arm("L", left))
    p.update(arm("R", right))
    return p


# ---------------------------------------------------------------- анимация
class Animator:
    def __init__(self, P, fps=24):
        self.P = P
        self.fps = fps

    def f(self, t):
        return round(t * self.fps) + 1

    # Как у живого человека: взгляд/голова начинают движение раньше руки,
    # а локоть, кисть и пальцы догоняют плечо с небольшим запозданием (инерция).
    LAG = {"head": -0.07, "elbow": 0.05, "wrist": 0.09, "index": 0.11, "finger": 0.12, "thumb": 0.11,
           "knee": 0.04}

    def pose(self, t, p):
        for name, rot in p.items():
            lag = next((v for k, v in self.LAG.items() if name.startswith(k)), 0.0)
            ob = self.P[name]
            ob.rotation_euler = [math.radians(a) for a in rot]
            ob.keyframe_insert("rotation_euler", frame=self.f(max(0.0, t + lag)))

    def loc(self, t, name, xyz):
        ob = self.P[name]
        ob.location = xyz
        ob.keyframe_insert("location", frame=self.f(t))

    def talk(self, t0, t1, seed=1, amp=14.0):
        """Челюсть «говорит»: слоги ~5 в секунду со случайной силой, на ударных — кивок головой.

        Позже здесь будет громкость настоящего голоса вместо случайных слогов.
        """
        jaw = self.P["jaw"]
        neck = self.P["neck"]
        rnd = random.Random(seed)
        t = t0
        jaw.rotation_euler = (0, 0, 0)
        jaw.keyframe_insert("rotation_euler", frame=self.f(t0))
        while t < t1 - 0.15:
            syl = rnd.uniform(0.13, 0.24)
            open_t = t + syl * 0.45
            a = amp * rnd.uniform(0.45, 1.0)
            if rnd.random() < 0.12:  # короткая пауза между фразами
                a = 0.0
            jaw.rotation_euler = (math.radians(a), 0, 0)
            jaw.keyframe_insert("rotation_euler", frame=self.f(open_t))
            if a > amp * 0.85 and rnd.random() < 0.6:  # ударение: лёгкий кивок / наклон
                tilt = rnd.uniform(-4, 4)
                neck.rotation_euler = (math.radians(5), 0, math.radians(tilt * 0.5))
                neck.keyframe_insert("rotation_euler", frame=self.f(open_t))
                neck.rotation_euler = (math.radians(-1), math.radians(tilt), 0)
                neck.keyframe_insert("rotation_euler", frame=self.f(open_t + 0.28))
            jaw.rotation_euler = (math.radians(a * 0.15), 0, 0)
            jaw.keyframe_insert("rotation_euler", frame=self.f(t + syl))
            t += syl
        jaw.rotation_euler = (0, 0, 0)
        jaw.keyframe_insert("rotation_euler", frame=self.f(t1))

    def blink(self, t, length=0.16, amount=0.12):
        """Моргание: линзы маски «прищуриваются» и раскрываются обратно."""
        for nm in ("eye_L", "eye_R"):
            e = self.P[nm]
            for dt, k in ((-0.01, None), (length * 0.4, amount), (length, 1.0)):
                if k is None:
                    k = 1.0
                e.scale = (1.0, 1.0, k)
                e.keyframe_insert("scale", frame=self.f(max(0.0, t + dt)))

    def eyes(self, t, k, hold=None):
        """Выражение глаз: k<1 — прищур (думает, хитрит), k>1 — широко (удивлён)."""
        for nm in ("eye_L", "eye_R"):
            e = self.P[nm]
            e.scale = (1.0 + (k - 1) * 0.3, 1.0, k)
            e.keyframe_insert("scale", frame=self.f(t))
            if hold:
                e.keyframe_insert("scale", frame=self.f(t + hold))

    def life(self, t0, t1, seed=0):
        """Живость поверх жестов: голова и корпус никогда не замирают, вес переносится с ноги на ногу."""
        rnd = random.Random(seed)
        noise = [  # (шарнир, канал, ось, сила, скорость в кадрах)
            ("head", "rotation_euler", 0, math.radians(2.5), 38),
            ("head", "rotation_euler", 1, math.radians(2.5), 46),
            ("head", "rotation_euler", 2, math.radians(4.0), 52),
            ("spine", "rotation_euler", 1, math.radians(1.6), 60),
            ("spine", "rotation_euler", 2, math.radians(2.0), 70),
            ("hips", "location", 0, 0.025, 90),
            ("hips", "rotation_euler", 1, math.radians(1.5), 90),
        ]
        for name, path, axis, strength, scale in noise:
            ob = self.P[name]
            if not (ob.animation_data and ob.animation_data.action and
                    any(fc.data_path == path and fc.array_index == axis for fc in _fcurves(ob.animation_data.action))):
                ob.keyframe_insert(path, index=axis, frame=self.f(t0))
            fc = next(fc for fc in _fcurves(ob.animation_data.action)
                      if fc.data_path == path and fc.array_index == axis)
            m = fc.modifiers.new("NOISE")
            m.strength = strength * 2
            m.scale = scale
            m.phase = rnd.uniform(0, 100)
            m.depth = 1
        # моргания через неравные промежутки, как у людей
        t = t0 + rnd.uniform(0.6, 1.4)
        while t < t1 - 0.3:
            self.blink(t)
            t += rnd.uniform(1.8, 3.6)

    def breathe(self, t0, t1, period=2.6):
        """Лёгкое «дыхание» и покачивание — чтобы персонаж никогда не замирал."""
        sp = self.P["spine"]
        n = int((t1 - t0) / (period / 2)) + 1
        for i in range(n + 1):
            t = t0 + i * period / 2
            k = 1.0 + (0.012 if i % 2 else 0.0)
            sp.scale = (1.0, 1.0, k)
            sp.keyframe_insert("scale", frame=self.f(t))

    def finish(self):
        """Плавные переходы между позами (мягкий старт и остановка)."""
        for ob in bpy.data.objects:
            ad = ob.animation_data
            if not ad or not ad.action:
                continue
            for fc in _fcurves(ad.action):
                for kp in fc.keyframe_points:
                    kp.interpolation = "BEZIER"
                    kp.handle_left_type = kp.handle_right_type = "AUTO_CLAMPED"


def _fcurves(action):
    if hasattr(action, "fcurves") and action.fcurves is not None:
        try:
            return list(action.fcurves)
        except TypeError:
            pass
    out = []  # Blender 5: слоистые действия
    for layer in action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                out.extend(bag.fcurves)
    return out
