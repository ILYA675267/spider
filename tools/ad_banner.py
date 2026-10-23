"""Рекламный баннер в середине ролика (правило пользователя): паучок говорит «Чуть не забыл!» → ролик
останавливается, отдельным кадром идёт серый экран с баннером по центру (зелёный фон баннера убирается,
звук баннера свой) → ролик продолжается с того же места. Параллельно баннеру ролик НЕ идёт.

1) Фраза в голос — после voice_prep.py / tighten.py, до рендера сцен (между предложениями середины):
   python tools/ad_banner.py phrase <папка_ролика> <секунда_вставки>
   Вставляет шаблон assets/voice/chut_ne_zabyl.mp3 (не озвучивать заново) с паузой 0,32 с перед ним,
   сдвигает words.json и outro_start.txt, добавляет слова фразы, пишет ad_cut.txt — секунду склейки
   (0,3 с после «забыл!»). В скрипте ролика на фразу — субтитр «ЧУТЬ НЕ ЗАБЫЛ!» и жест (рука ко лбу).
2) Вклейка баннера в готовый ролик — после рендера, до версии для телефона:
   python tools/ad_banner.py insert <ролик.mp4> <секунда_склейки> <выход.mp4> [баннер.mp4] [ширина=1040]
"""
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from voice_prep import ROOT, SR, load  # noqa: E402

AD = os.path.join(ROOT, "assets", "ads", "natria_vpn.mp4")
PHRASE = os.path.join(ROOT, "assets", "voice", "chut_ne_zabyl.mp3")
PHRASE_WORDS = [("Чуть", 0.0, 0.32), ("не", 0.32, 0.44), ("забыл!", 0.44, 0.84)]   # распознано заранее
GREY = "0x4b4c52"


def duration(p):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p],
                                capture_output=True, text=True, check=True).stdout)


def phrase(folder, t):
    ph = load(PHRASE)
    db = 20 * np.log10(np.abs(ph) + 1e-9)
    loud = np.where(db > -40)[0]
    a = max(0, loud[0] - int(0.03 * SR))
    ph = ph[a:loud[-1] + int(0.25 * SR)]                    # без тишины по краям, хвост 0,25 с
    lead = a / SR
    ins = np.concatenate([np.zeros(int(0.32 * SR), np.float32), ph, np.zeros(int(0.05 * SR), np.float32)])
    L = len(ins) / SR
    for name in ("voice_full.wav", "voice_clean.wav"):
        p = os.path.join(folder, name)
        x = load(p)
        k = int(t * SR)
        y = np.concatenate([x[:k], ins, x[k:]])
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-", p],
                       input=y.tobytes(), check=True)
    wp = os.path.join(folder, "words.json")
    words = json.load(open(wp))
    for w in words:
        if w["s"] >= t:
            w["s"], w["e"] = round(w["s"] + L, 3), round(w["e"] + L, 3)
    s0 = t + 0.32 - lead
    words += [{"w": w, "s": round(s0 + s, 3), "e": round(s0 + e, 3)} for w, s, e in PHRASE_WORDS]
    words.sort(key=lambda w: w["s"])
    json.dump(words, open(wp, "w"), ensure_ascii=False, indent=0)
    op = os.path.join(folder, "outro_start.txt")
    outro = round(float(open(op).read()) + L, 3)
    open(op, "w").write(str(outro))
    cut = round(s0 + PHRASE_WORDS[-1][2] + 0.3, 3)
    open(os.path.join(folder, "ad_cut.txt"), "w").write(str(cut))
    print(f"«Чуть не забыл!» с {s0:.2f} с; склейка с баннером {cut} с; концовка с {outro} с; "
          f"длина {len(y) / SR:.2f} с (+ баннер {duration(AD):.1f} с)")


def insert(video, t, out, ad=AD, width=1040):
    """Ролик до t → серый кадр с баннером (свой звук) → ролик с t."""
    d = duration(ad)
    graph = (f"[0:v]split[va][vb];[0:a]asplit[aa][ab];"
             f"[va]trim=0:{t},setpts=PTS-STARTPTS[v1];[aa]atrim=0:{t},asetpts=PTS-STARTPTS[a1];"
             f"[vb]trim=start={t},setpts=PTS-STARTPTS[v3];[ab]atrim=start={t},asetpts=PTS-STARTPTS[a3];"
             f"color=c={GREY}:s=1080x1920:r=30:d={d},format=yuv420p[g];"
             f"[1:v]chromakey=0x00FF00:0.25:0.05,despill=green,scale={width}:-1,setpts=PTS-STARTPTS[ad];"
             f"[g][ad]overlay=(W-w)/2:(H-h)/2:eof_action=pass,fps=30,format=yuv420p,setsar=1[v2];"
             f"[1:a]aformat=sample_rates=48000:channel_layouts=stereo,apad,atrim=0:{d}[a2];"
             f"[v1]fps=30,format=yuv420p,setsar=1[v1f];[v3]fps=30,format=yuv420p,setsar=1[v3f];"
             f"[a1]aformat=sample_rates=48000:channel_layouts=stereo[a1f];"
             f"[a3]aformat=sample_rates=48000:channel_layouts=stereo[a3f];"
             f"[v1f][a1f][v2][a2][v3f][a3f]concat=n=3:v=1:a=1[v][a]")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", video, "-i", ad, "-filter_complex", graph,
                    "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "15", "-preset", "medium",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", out], check=True)


if __name__ == "__main__":
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == "phrase":
        phrase(a[0], float(a[1]))
    else:
        insert(a[0], float(a[1]), a[2], *(a[3:4] or [AD]), *[int(x) for x in a[4:5]])
