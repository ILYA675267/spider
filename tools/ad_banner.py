"""Рекламный баннер в середине ролика (правило пользователя).

В середине ролика камера просто меняет ракурс (облёт в той же комнате, без склейки и растворения): паучок
крупно говорит «Чуть не забыл!» (голос пользователя assets/voice/chut_ne_zabyl.mp3, рука вверх, субтитр
сверху), затем по центру экрана идёт баннер рекламодателя (зелёный фон убирается, звук баннера из его файла,
не приглушать), а паучок под баннером спокойно жестикулирует и покачивает головой. Ролик под баннером не идёт;
после баннера камера тем же облётом возвращается в исходный ракурс, и ролик продолжается с того же места.
Вставка рисуется сценой самого ролика (комната, цвет, вставки на месте замирают), поэтому стыков не видно.

python tools/ad_banner.py insert <ролик.mp4> <секунда_склейки | mid> <выход.mp4> <модуль_ролика> [секунда_в_сцене]
   [баннер.mp4] [ширина=1150]
Секунда склейки — пауза между предложениями в середине, когда камера стоит спокойно (не во время облёта).
mid — баннер строго посередине итогового ролика (правило пользователя): склейка в паузе между словами
(work/<тема>/words.json), ближайшей к точке, где середина баннера совпадает с серединой ролика.
Баннер — не меньше 25% экрана (правило пользователя): ширина 1150, края срезаются до 1080 → 1080×488 = 25,4%.
Секунда в сцене — то же время по сценарию модуля (по умолчанию равна секунде склейки; отличается только
для пробы на вырезанном куске). Вставка добавляет к ролику ≈ 9 с (фраза + баннер 7,2 с) — учитывать в длине.
"""
import importlib
import json
import math
import os
import subprocess
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memory_mg as base  # noqa: E402
import mg  # noqa: E402
from mg import track, wobble  # noqa: E402

ROOT = mg.ROOT
AD = os.path.join(ROOT, "assets", "ads", "natria_vpn.mp4")
PHRASE = os.path.join(ROOT, "assets", "voice", "chut_ne_zabyl.mp3")
FPS = 30
FLY = 0.6                        # облёт камеры туда и обратно
LEAD = 0.4                       # фраза начинается к концу облёта
DUR = 1.4                        # от начала вставки до баннера (фраза ≈ 0,85 с)
TURN = -80.0                     # на сколько градусов облетает камера (в сторону от вставок сцены)
VIEW = (4.2, 1.85, 1.77, 46.0)   # ракурс вставки: расстояние, высота, точка взгляда, угол зрения (паучок внизу)


def duration(p):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p],
                                capture_output=True, text=True, check=True).stdout)


AD_DUR = duration(AD)
AD_GAIN = 2.0                    # баннер громче в 2 раза (+6 дБ) — просьба пользователя
TOTAL = DUR + AD_DUR + FLY       # вся вставка: облёт + фраза + баннер + облёт обратно


def make_scene(modname, t0, wav):
    """Сцена ролика, «замороженная» в момент t0: та же комната и вставки, своя камера и жесты паучка."""
    M = importlib.import_module(modname)
    yaw0, d0, h0, ty0, fov0 = track(M.CAMERA, t0)
    off0 = track(M.OFFSET, t0)
    aL, aR, hd = track(M.ARM_L, t0), track(M.ARM_R, t0), track(M.HEAD, t0)
    E = TOTAL - FLY                                    # начало облёта обратно
    yaw1 = yaw0 + TURN
    cfg = types.SimpleNamespace(**vars(M))
    cfg.SUBS, cfg.JUMPS, cfg.SHAKES, cfg.FLASHES, cfg.ALARM, cfg.SPOT = [], [], (), (), None, None
    cfg.OUTRO, cfg.DUR = 999.0, TOTAL
    cfg.CAMERA = [(0.0, (yaw0, d0, h0, ty0, fov0, off0)), (FLY, (yaw1,) + VIEW + (0.0,), "expo"),
                  (E, (yaw1 + 4,) + VIEW + (0.0,), "lin"), (TOTAL, (yaw0, d0, h0, ty0, fov0, off0), "expo")]
    # «ой!» — рука вверх на фразе; под баннером спокойно жестикулирует; к концу — снова поза из ролика
    cfg.ARM_R = [(0, aR), (LEAD + 0.05, -78, "back"), (DUR - 0.1, -76), (DUR + 0.3, -35, "back"), (DUR + 1.6, -38),
                 (DUR + 2.0, 20, "back"), (DUR + 3.1, 25), (DUR + 3.5, -30, "back"), (DUR + 4.8, -33),
                 (DUR + 5.2, 15, "back"), (E - 0.2, 20), (TOTAL - 0.1, aR), (TOTAL, aR)]
    cfg.ARM_L = [(0, aL), (LEAD + 0.15, 20, "back"), (DUR, 22), (DUR + 0.6, 45, "back"), (DUR + 2.0, 40),
                 (DUR + 2.4, -10, "back"), (DUR + 3.4, -5), (DUR + 3.8, 50, "back"), (DUR + 5.4, 48),
                 (DUR + 5.8, 0, "back"), (E - 0.2, 5), (TOTAL - 0.1, aL), (TOTAL, aL)]
    cfg.HEAD = ([(0, hd), (LEAD + 0.05, -10, "back"), (LEAD + 0.5, 8), (DUR, 6)] +
                [(DUR + 0.8 * k + 0.4, (5 if k % 2 else -5) + (2 if k % 3 == 0 else 0))
                 for k in range(int(AD_DUR / 0.8) + 1)] + [(TOTAL - 0.1, hd), (TOTAL, hd)])

    class AdScene(M.Scene):
        def draw(self, t):
            self.t = t0                                # цвет комнаты — как в момент склейки
            return base.Scene.draw(self, t)

        def items_for(self, t, cam, it):
            # вставки ролика замирают на месте и прячутся на самом быстром месте облёта (смазано — не заметно),
            # чтобы в ракурсе вставки ничего не стояло за паучком; на облёте обратно появляются так же
            if t < FLY / 2 or t > TOTAL - FLY / 2:
                M.Scene.items_for(self, t0, cam, it)

        def cam(self, t):
            yaw, dist, hgt, ty, fov, off = track(cfg.CAMERA, t)
            a = math.radians(yaw)
            return mg.Cam(yaw + 0.5 * wobble(t0, 1), dist, hgt + 0.02 * wobble(t0, 2), ty + 0.015 * wobble(t0, 3),
                          fov=fov, roll=0.5 * wobble(t0 * 0.8, 4), tx=off * math.cos(a), tz=off * math.sin(a))

        def finish(self, frame, t):
            f = frame.astype(np.float32) * base.VIGNETTE[..., None]
            fr = np.clip(f, 0, 255).astype(np.uint8)
            if t < 0.3 or t > TOTAL - 0.3:             # субтитр ролика в момент склейки — чтобы не мигал
                mg.subtitles(fr, t0, M.SUBS)
            mg.subtitles(fr, t, [(LEAD, "ЧУТЬ НЕ ЗАБЫЛ!", ("ЗАБЫЛ!",)), (DUR, "", ())], y=430)   # над баннером
            return fr

    AdScene.cfg = cfg
    return AdScene(wav)


def shot(out, modname, t0):
    """Вставка с паучком (облёт → фраза → жесты под баннером → облёт обратно), со звуком фразы; без баннера."""
    wav = out + ".voice.wav"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", PHRASE, "-af",
                    f"silenceremove=start_periods=1:start_threshold=-45dB,adelay={int(LEAD * 1000)},apad",
                    "-t", str(TOTAL), "-ac", "1", "-ar", "44100", wav], check=True)
    sc = make_scene(modname, t0, wav)
    vid = out + ".video.mp4"
    wr = mg.Writer(vid, FPS)
    for i in range(1, round(TOTAL * FPS)):             # кадр 0 и последний — это кадры самого ролика
        t = i / FPS
        if t < FLY or t > TOTAL - FLY:                 # размытие в движении на быстром облёте, как в роликах
            acc = np.zeros((mg.H, mg.W, 3), np.float32)
            for k in range(5):
                acc += sc.draw(t + (k / 5 - 0.5) / FPS)
            fr = (acc / 5).astype(np.uint8)
        else:
            fr = sc.draw(t)
        wr.write(sc.finish(fr, t))
    wr.close()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", vid, "-i", wav, "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-af", f"atrim=start={1 / FPS},asetpts=PTS-STARTPTS,pan=stereo|c0=c0|c1=c0",
                    "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-shortest", out],
                   check=True)  # голос в оба канала без потери громкости
    os.remove(wav)
    os.remove(vid)


def insert(video, t, out, modname, t0=None, ad=AD, width=1150):
    """Ролик до t → облёт к паучку, «Чуть не забыл!» → баннер по центру (его звук), паучок под ним
    жестикулирует → облёт обратно → ролик с t."""
    t0 = t if t0 is None else t0
    t = round(t * FPS) / FPS                           # склейка ровно по кадру
    sh = out + ".shot.mp4"
    shot(sh, modname, t0)
    st = "fps=30,format=yuv420p,setsar=1"
    au = "aformat=sample_rates=48000:channel_layouts=stereo"
    bd = DUR - 1 / FPS                                 # баннер во вставке (без её нулевого кадра)
    graph = (f"[0:v]split[va][vb];[0:a]asplit[aa][ab];"
             f"[va]trim=0:{t},setpts=PTS-STARTPTS,{st}[v1];"
             f"[aa]atrim=0:{t},asetpts=PTS-STARTPTS,{au},afade=t=out:st={t - 0.12}:d=0.12[a1];"
             f"[vb]trim=start={t},setpts=PTS-STARTPTS,{st}[v4];"
             f"[ab]atrim=start={t},asetpts=PTS-STARTPTS,{au},afade=t=in:d=0.12[a4];"
             f"[2:v]chromakey=0x00FF00:0.25:0.05,despill=green,scale={width}:-2,crop=min(iw\\,{mg.W}):ih,setpts=PTS-STARTPTS+{bd}/TB[ad];"
             f"[1:v]{st}[sv0];[sv0][ad]overlay=(W-w)/2:(H-h)/2:eof_action=pass,{st}[sv];"
             f"[2:a]{au},volume={AD_GAIN},adelay={int(bd * 1000)}:all=1[ba];[1:a]{au}[pa];"
             f"[pa][ba]amix=inputs=2:normalize=0:duration=first[sa];"
             f"[v1][a1][sv][sa][v4][a4]concat=n=3:v=1:a=1[v][a]")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", video, "-i", sh, "-i", ad,
                    "-filter_complex", graph, "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "15",
                    "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", out], check=True)
    os.remove(sh)


def middle(video):
    """Секунда склейки, при которой баннер стоит посередине итогового ролика: пауза между словами (≥ 0,25 с,
    до концовки), ближайшая к расчётной точке. Время ролика и слов совпадает (ролик собран по words.json)."""
    topic = os.path.splitext(os.path.basename(video))[0]
    folder = os.path.join(ROOT, "work", topic)
    words = json.load(open(os.path.join(folder, "words.json")))
    outro = float(open(os.path.join(folder, "outro_start.txt")).read())
    bd = DUR - 1 / FPS
    target = (duration(video) + TOTAL) / 2 - bd - AD_DUR / 2
    gaps = [((a["e"] + b["s"]) / 2) for a, b in zip(words, words[1:]) if b["s"] - a["e"] >= 0.25 and b["s"] < outro - 1]
    t = min(gaps, key=lambda g: abs(g - target))
    print(f"середина: расчёт {target:.2f} с, пауза {t:.2f} с")
    return t


if __name__ == "__main__":
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == "insert":
        t = middle(a[0]) if a[1] == "mid" else float(a[1])
        insert(a[0], t, a[2], a[3], t0=float(a[4]) if len(a) > 4 else None,
               ad=a[5] if len(a) > 5 else AD, width=int(a[6]) if len(a) > 6 else 1150)
