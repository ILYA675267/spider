"""Рекламный баннер в середине ролика (правило пользователя).

Между двумя кадрами (сценами) в середине ролика вклеивается ОТДЕЛЬНАЯ вставка: паучок крупно говорит
«Чуть не забыл!» (шаблон голоса assets/voice/chut_ne_zabyl.mp3, хлопает себя по лбу) и застывает —
на застывшем кадре по центру экрана идёт баннер рекламодателя (зелёный фон убирается, звук баннера из его
файла, не приглушать). Ролик под баннером не идёт; после баннера ролик продолжается с того же места.

python tools/ad_banner.py insert <ролик.mp4> <секунда_склейки> <выход.mp4> [баннер.mp4] [ширина=1040]
Секунда склейки — пауза между предложениями/сценами в середине (лучше на смене сцены).
Вставка добавляет к ролику ≈1,4 с + длину баннера (7,2 с) — учитывать в длине ролика (≤ 1 мин).
python tools/ad_banner.py shot <выход.mp4>   — только вставка с паучком (проверить)
"""
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memory_mg as base  # noqa: E402
import mg  # noqa: E402
from mg import track, wobble  # noqa: E402

ROOT = mg.ROOT
AD = os.path.join(ROOT, "assets", "ads", "natria_vpn.mp4")
PHRASE = os.path.join(ROOT, "assets", "voice", "chut_ne_zabyl.mp3")
FPS = 30
LEAD = 0.15                      # тишина перед фразой во вставке
DUR = 1.45                       # длина вставки до застывания (фраза ≈ 0,85 с + реакция)

# настройки вставки для общего движка (как у роликов)
OUTRO, SHAKES, FLASHES, JUMPS, ALARM, SPOT, SUBS = 99.0, (), ((0.0, 0.1),), [], None, None, []
CAMERA = [(0.0, (-4.0, 4.6, 1.9, 1.95, 48.0)), (0.35, (2.0, 4.2, 1.85, 1.77, 46.0), "expo"),
          (DUR, (3.0, 4.15, 1.85, 1.77, 46.0))]
OFFSET = [(0.0, 0.0), (DUR, 0.0)]
ARM_R = [(0, 62), (0.18, -78, "back"), (DUR, -76)]          # рука ко лбу: «ой!»
ARM_L = [(0, 62), (0.3, 20, "back"), (DUR, 22)]             # вторая рука в сторону
HEAD = [(0, 0), (0.2, -10, "back"), (0.7, 8), (DUR, 6)]


class Scene(base.Scene):
    cfg = sys.modules[__name__]

    def items_for(self, t, cam, it):
        pass

    def cam(self, t):
        yaw, dist, hgt, ty, fov = track(CAMERA, t)
        return mg.Cam(yaw + 0.4 * wobble(t, 1), dist, hgt, ty, fov=fov, roll=0.4 * wobble(t * 0.8, 4))

    def finish(self, frame, t):
        f = frame.astype(np.float32) * base.VIGNETTE[..., None]
        fr = np.clip(f, 0, 255).astype(np.uint8)
        mg.subtitles(fr, t, [(LEAD, "ЧУТЬ НЕ ЗАБЫЛ!", ("ЗАБЫЛ!",))], y=430)   # над баннером, не на паучке
        return fr


def duration(p):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p],
                                capture_output=True, text=True, check=True).stdout)


def shot(out):
    """Вставка: паучок говорит «Чуть не забыл!» (со звуком) → mp4; последний кадр → out.png (для застывания)."""
    wav = out + ".voice.wav"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", PHRASE, "-af",
                    f"silenceremove=start_periods=1:start_threshold=-45dB,adelay={int(LEAD * 1000)},apad",
                    "-t", str(DUR), "-ac", "1", "-ar", "44100", wav], check=True)
    sc = Scene(wav)
    vid = out + ".video.mp4"
    wr = mg.Writer(vid, FPS)
    for i in range(int(DUR * FPS)):
        fr = sc.finish(sc.draw(i / FPS), i / FPS)
        wr.write(fr)
    wr.close()
    cv2.imwrite(out + ".png", fr)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", vid, "-i", wav, "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", out], check=True)
    os.remove(wav)
    os.remove(vid)


def insert(video, t, out, ad=AD, width=1040):
    """Ролик до t → вставка «Чуть не забыл!» → застывший кадр + баннер (его звук) → ролик с t."""
    sh = out + ".shot.mp4"
    shot(sh)
    d = duration(ad)
    st = "fps=30,format=yuv420p,setsar=1"
    au = "aformat=sample_rates=48000:channel_layouts=stereo"
    graph = (f"[0:v]split[va][vb];[0:a]asplit[aa][ab];"
             f"[va]trim=0:{t},setpts=PTS-STARTPTS,{st}[v1];[aa]atrim=0:{t},asetpts=PTS-STARTPTS,{au}[a1];"
             f"[vb]trim=start={t},setpts=PTS-STARTPTS,{st}[v4];[ab]atrim=start={t},asetpts=PTS-STARTPTS,{au}[a4];"
             f"[1:v]{st}[v2];[1:a]{au}[a2];"
             f"[2:v]loop=loop=-1:size=1,trim=0:{d},setpts=PTS-STARTPTS,{st}[fz];"
             f"[3:v]chromakey=0x00FF00:0.25:0.05,despill=green,scale={width}:-1,setpts=PTS-STARTPTS[ad];"
             f"[fz][ad]overlay=(W-w)/2:(H-h)/2:eof_action=pass,{st}[v3];"
             f"[3:a]{au},apad,atrim=0:{d}[a3];"
             f"[v1][a1][v2][a2][v3][a3][v4][a4]concat=n=4:v=1:a=1[v][a]")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", video, "-i", sh, "-i", sh + ".png", "-i", ad,
                    "-filter_complex", graph, "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "15",
                    "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", out], check=True)
    os.remove(sh)
    os.remove(sh + ".png")


if __name__ == "__main__":
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == "shot":
        shot(a[0])
    else:
        insert(a[0], float(a[1]), a[2], *(a[3:4] or [AD]), *[int(x) for x in a[4:5]])
