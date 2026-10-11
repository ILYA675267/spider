"""Проба движений: 10 секунд, без голоса и монтажа.

Рендер кадров (Blender как модуль Python, pip install bpy pillow):
    python tools/probe.py work/probe/frames [ширина высота сэмплы]
(сэмплы 0 — быстрый черновик без света и паутины)
Потом сборка видео: python3 tools/compose.py work/probe/frames assets/background.png output/probe.mp4
"""
import math
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import character  # noqa: E402
import poses  # noqa: E402
import scene  # noqa: E402
from poses import pose  # noqa: E402

FPS = 24
DUR = 10.0


def choreography(A):
    """Сценарий жестов: (время, поза). Голова смотрит туда, куда показывает рука."""
    hip = dict(legs="contrapposto")
    A.pose(0.0, pose("hip", "hip", **hip))
    A.pose(0.9, pose("hip", "hip", head=(4, 0, 0), **hip))
    # помахать правой рукой
    A.pose(1.25, pose("hip", "wave_a", head=(0, 0, -6), spine=(0, 3, 0), **hip))
    A.pose(1.55, pose("hip", "wave_b", head=(0, 0, -6), spine=(0, 3, 0), **hip))
    A.pose(1.85, pose("hip", "wave_a", head=(0, 0, -6), spine=(0, 3, 0), **hip))
    A.pose(2.15, pose("hip", "wave_b", head=(0, 0, -6), spine=(0, 3, 0), **hip))
    A.pose(2.45, pose("hip", "wave_a", head=(0, 0, -4), spine=(0, 2, 0), **hip))
    # указать в сторону левой рукой и посмотреть туда
    A.pose(2.9, pose("point_side", "down", head=(0, 0, 28), spine=(0, -3, 12)))
    A.pose(3.9, pose("point_side", "down", head=(-4, 0, 30), spine=(0, -3, 14)))
    A.pose(4.3, pose("point_side", "down", head=(2, 0, 4), spine=(0, -2, 6)))
    # объясняет: обе руки раскрыты
    A.pose(4.8, pose("open", "open", head=(-5, 0, 0)))
    A.pose(5.3, pose("open", "open", head=(4, 0, -6), spine=(4, 0, 0)))
    A.pose(5.8, pose("open", "open", head=(-3, 0, 5)))
    # указать вверх правой рукой
    A.pose(6.3, pose("down", "point_up", head=(-18, 0, -12), spine=(0, 4, -6)))
    A.pose(7.0, pose("down", "point_up", head=(-20, 0, -14), spine=(0, 4, -6)))
    # задумался
    A.pose(7.5, pose("think", "hip", head=(8, -10, 8), spine=(0, -3, 0), **hip))
    A.pose(8.2, pose("think", "hip", head=(6, -12, 12), spine=(0, -3, 0), **hip))
    # указать на зрителя
    A.pose(8.6, pose("hip", "point_cam", head=(6, 0, 0), spine=(-6, 0, 0)))
    A.pose(9.25, pose("hip", "point_cam", head=(4, 0, 0), spine=(-6, 0, 0)))
    # вернуться в позу с картинки
    A.pose(9.7, pose("hip", "hip", **hip))
    A.pose(DUR, pose("hip", "hip", head=(3, 0, 0), **hip))

    # подпрыгнуть от радости, когда показывает вверх
    root = A.P["root"]
    for t, z in ((6.05, 0.0), (6.25, -0.04), (6.45, 0.12), (6.65, 0.0), (6.75, -0.02), (6.9, 0.0)):
        A.loc(t, "root", (0, 0, z))

    A.talk(0.2, 2.6, seed=3)
    A.talk(2.9, 6.1, seed=5)
    A.talk(7.4, 9.6, seed=8)
    A.breathe(0.0, DUR)
    A.life(0.0, DUR, seed=2)
    A.eyes(2.7, 1.0)                   # перед «смотри туда» — глаза чуть шире
    A.eyes(2.95, 1.12, hold=0.9)
    A.eyes(4.2, 1.0)
    A.eyes(7.35, 1.0)                  # задумался — прищур
    A.eyes(7.6, 0.72, hold=0.55)
    A.eyes(8.45, 1.0)
    A.finish()
    return root


def camera_move(cam):
    """Медленный «наезд» камеры и лёгкое покачивание — кадр живой."""
    s = bpy.context.scene
    for t, y, x, z in ((0.0, -4.6, 0.0, 1.02), (5.0, -4.2, -0.06, 1.06), (DUR, -3.9, 0.04, 1.1)):
        cam.location = (x, y, z)
        cam.keyframe_insert("location", frame=round(t * FPS) + 1)
    for fc in poses._fcurves(cam.animation_data.action):
        for kp in fc.keyframe_points:
            kp.interpolation = "SINE"
            kp.easing = "EASE_IN_OUT"
    s.frame_start = 1
    s.frame_end = round(DUR * FPS)


def main():
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    out = args[0] if args else "work/probe/frames"
    w, h = (int(args[1]), int(args[2])) if len(args) > 2 else (1080, 1920)
    samples = int(args[3]) if len(args) > 3 else 12
    frames = [int(x) for x in args[4].split(",")] if len(args) > 4 else None

    bpy.ops.wm.read_factory_settings(use_empty=True)
    P = character.build()
    cam = scene.setup(width=w, height=h, fps=FPS, samples=samples, preview=(samples == 0))
    A = poses.Animator(P, fps=FPS)
    choreography(A)
    camera_move(cam)

    s = bpy.context.scene
    os.makedirs(out, exist_ok=True)
    todo = frames or list(range(s.frame_start, s.frame_end + 1))
    for f in todo:
        path = os.path.join(out, f"{f:04d}.png")
        if os.path.exists(path) and frames is None:
            continue  # можно продолжить прерванный рендер
        s.frame_set(f)
        s.render.filepath = path
        bpy.ops.render.render(write_still=True)
        print(f"frame {f}/{s.frame_end}", flush=True)


if __name__ == "__main__":
    main()
