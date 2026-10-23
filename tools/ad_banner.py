"""Рекламный баннер в середине ролика (правило пользователя): паучок говорит «Чуть не забыл!» → по центру
экрана идёт баннер (видео на зелёном фоне, зелёный убирается), потом рассказ продолжается.

1) Пауза в голосе под баннер — после voice_prep.py / tighten.py, до рендера сцен:
   python tools/ad_banner.py gap <папка_ролика> <секунда_конца_«забыл»> [баннер.mp4]
   Вставляет тишину длиной баннера (+0,3 с) в voice_full.wav и voice_clean.wav, сдвигает words.json и
   outro_start.txt, пишет ad_start.txt (с какой секунды баннер). В скрипте ролика в это время
   вставок не ставить, субтитры убрать ((t, "", ())), проверку закрытия персонажа это окно пропускать.
2) Наложение на готовый ролик — после рендера, до версии для телефона:
   python tools/ad_banner.py overlay <ролик.mp4> <старт_с> <выход.mp4> [баннер.mp4] [ширина=1040]
   Баннер по центру, почти во всю ширину (меньше 25% экрана нельзя), фон под ним слегка затемняется,
   звук баннера подмешивается, музыка ролика на это время тише.
"""
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from voice_prep import ROOT, SR, load  # noqa: E402

AD = os.path.join(ROOT, "assets", "ads", "natria_vpn.mp4")


def duration(p):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p],
                                capture_output=True, text=True, check=True).stdout)


def gap(folder, t, ad=AD):
    L = round(duration(ad) + 0.3, 3)
    t0 = t + 0.15                      # баннер начинается сразу после «Чуть не забыл!»
    for name in ("voice_full.wav", "voice_clean.wav"):
        p = os.path.join(folder, name)
        x = load(p)
        k = int(t0 * SR)
        y = np.concatenate([x[:k], np.zeros(int(L * SR), np.float32), x[k:]])
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-", p],
                       input=y.tobytes(), check=True)
    wp = os.path.join(folder, "words.json")
    words = json.load(open(wp))
    for w in words:
        if w["s"] >= t0:
            w["s"], w["e"] = round(w["s"] + L, 3), round(w["e"] + L, 3)
    json.dump(words, open(wp, "w"), ensure_ascii=False, indent=0)
    op = os.path.join(folder, "outro_start.txt")
    outro = round(float(open(op).read()) + L, 3)
    open(op, "w").write(str(outro))
    open(os.path.join(folder, "ad_start.txt"), "w").write(f"{t0:.3f} {L - 0.3:.3f}")
    print(f"баннер с {t0:.2f} с, {L - 0.3:.2f} с; концовка теперь с {outro} с; длина {len(y) / SR:.2f} с")


def overlay(video, t0, out, ad=AD, width=1040):
    d = duration(ad)
    t1 = t0 + d
    on = f"between(t\\,{t0}\\,{t1})"
    graph = (f"[0:v]drawbox=c=black@0.45:t=fill:enable={on}[bg];"          # фон под баннером чуть темнее
             f"[1:v]chromakey=0x00FF00:0.25:0.05,despill=green,scale={width}:-1,"
             f"setpts=PTS-STARTPTS+{t0}/TB[ad];"
             f"[bg][ad]overlay=(W-w)/2:(H-h)/2:eof_action=pass:enable={on}[v];"
             f"[0:a]volume=0.35:enable={on}[a0];"
             f"[1:a]aformat=channel_layouts=stereo,adelay={int(t0 * 1000)}:all=1[a1];"
             f"[a0][a1]amix=inputs=2:normalize=0:duration=first[a]")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", video, "-i", ad, "-filter_complex", graph,
                    "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "15", "-preset", "medium",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", out], check=True)


if __name__ == "__main__":
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == "gap":
        gap(a[0], float(a[1]), *a[2:3])
    else:
        overlay(a[0], float(a[1]), a[2], *(a[3:4] or [AD]), *[int(x) for x in a[4:5]])
