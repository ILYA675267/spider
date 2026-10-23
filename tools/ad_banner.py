"""Рекламный баннер в середине ролика (правило пользователя).

Между двумя сценами в середине ролика — плавный переход (растворение) в ОТДЕЛЬНУЮ вставку: паучок крупно
говорит «Чуть не забыл!» (голос пользователя assets/voice/chut_ne_zabyl.mp3, рука вверх, субтитр сверху),
затем по центру экрана идёт баннер рекламодателя (зелёный фон убирается, звук баннера из его файла, не
приглушать), а паучок под баннером спокойно жестикулирует и покачивает головой. Ролик под баннером не идёт;
после баннера — плавный переход обратно, ролик продолжается с того же места.

python tools/ad_banner.py insert <ролик.mp4> <секунда_склейки> <выход.mp4> [баннер.mp4] [ширина=1040]
Секунда склейки — пауза между предложениями/сценами в середине (лучше на смене сцены).
Вставка добавляет к ролику ≈ 8,5 с (фраза + баннер 7,2 с) — учитывать в длине ролика (≤ 1 мин).
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
XF = 0.4                         # плавный переход (растворение) в вставку и обратно
LEAD = 0.4                       # фраза начинается, когда переход почти закончился
DUR = 1.7                        # от начала вставки до баннера (фраза ≈ 0,85 с + реакция)


def duration(p):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p],
                                capture_output=True, text=True, check=True).stdout)


AD_DUR = duration(AD)
TOTAL = DUR + AD_DUR + XF        # вся вставка: фраза + баннер + выход

# настройки вставки для общего движка (как у роликов)
OUTRO, SHAKES, FLASHES, JUMPS, ALARM, SPOT, SUBS = 99.0, (), (), [], None, None, []
CAMERA = [(0.0, (-4.0, 4.6, 1.9, 1.95, 48.0)), (0.6, (2.0, 4.2, 1.85, 1.77, 46.0), "expo"),
          (DUR, (3.0, 4.15, 1.85, 1.77, 46.0)), (TOTAL, (-3.0, 4.05, 1.85, 1.77, 46.0), "inout")]
OFFSET = [(0.0, 0.0), (TOTAL, 0.0)]
# «ой!» — рука вверх на фразе; пока идёт баннер — спокойно жестикулирует: показывает на баннер, разводит руками
ARM_R = [(0, 62), (LEAD + 0.05, -78, "back"), (DUR - 0.1, -76), (DUR + 0.3, -35, "back"), (DUR + 1.6, -38),
         (DUR + 2.0, 20, "back"), (DUR + 3.1, 25), (DUR + 3.5, -30, "back"), (DUR + 4.8, -33), (DUR + 5.2, 15, "back"),
         (DUR + 6.2, 20), (DUR + 6.6, -25, "back"), (TOTAL, -28)]
ARM_L = [(0, 62), (LEAD + 0.15, 20, "back"), (DUR, 22), (DUR + 0.6, 45, "back"), (DUR + 2.0, 40),
         (DUR + 2.4, -10, "back"), (DUR + 3.4, -5), (DUR + 3.8, 50, "back"), (DUR + 5.4, 48), (DUR + 5.8, 0, "back"),
         (DUR + 6.8, 5), (TOTAL, 40, "back")]
HEAD = [(0, 0), (LEAD + 0.05, -10, "back"), (LEAD + 0.5, 8), (DUR, 6)] + \
       [(DUR + 0.8 * k + 0.4, (5 if k % 2 else -5) + (2 if k % 3 == 0 else 0)) for k in range(int(AD_DUR / 0.8) + 1)] + \
       [(TOTAL, 0)]


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
        mg.subtitles(fr, t, [(LEAD, "ЧУТЬ НЕ ЗАБЫЛ!", ("ЗАБЫЛ!",)), (DUR, "", ())], y=430)   # над баннером
        return fr


def shot(out):
    """Вставка с паучком (фраза → он жестикулирует, пока идёт баннер), со звуком фразы; без баннера."""
    wav = out + ".voice.wav"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", PHRASE, "-af",
                    f"silenceremove=start_periods=1:start_threshold=-45dB,adelay={int(LEAD * 1000)},apad",
                    "-t", str(TOTAL), "-ac", "1", "-ar", "44100", wav], check=True)
    sc = Scene(wav)
    vid = out + ".video.mp4"
    wr = mg.Writer(vid, FPS)
    for i in range(round(TOTAL * FPS)):
        wr.write(sc.finish(sc.draw(i / FPS), i / FPS))
    wr.close()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", vid, "-i", wav, "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-af", "pan=stereo|c0=c0|c1=c0", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", out],
                   check=True)  # голос в оба канала без потери громкости
    os.remove(wav)
    os.remove(vid)


def insert(video, t, out, ad=AD, width=1040):
    """Ролик до t ⇢ (растворение) вставка «Чуть не забыл!» → баннер по центру (его звук), паучок под ним
    жестикулирует ⇢ (растворение) ролик с t."""
    sh = out + ".shot.mp4"
    shot(sh)
    st = "fps=30,format=yuv420p,setsar=1"
    au = "aformat=sample_rates=48000:channel_layouts=stereo"
    o1 = round(t - XF, 3)
    o2 = round(o1 + TOTAL - XF, 3)
    graph = (f"[0:v]split[va][vb];[0:a]asplit[aa][ab];"
             f"[va]trim=0:{t},setpts=PTS-STARTPTS,{st}[v1];[aa]atrim=0:{t},asetpts=PTS-STARTPTS,{au}[a1];"
             f"[vb]trim=start={t},setpts=PTS-STARTPTS,{st}[v4];[ab]atrim=start={t},asetpts=PTS-STARTPTS,{au}[a4];"
             f"[2:v]chromakey=0x00FF00:0.25:0.05,despill=green,scale={width}:-1,setpts=PTS-STARTPTS+{DUR}/TB[ad];"
             f"[1:v]{st}[sv0];[sv0][ad]overlay=(W-w)/2:(H-h)/2:eof_action=pass,{st}[sv];"
             f"[2:a]{au},adelay={int(DUR * 1000)}:all=1[ba];[1:a]{au}[pa];"
             f"[pa][ba]amix=inputs=2:normalize=0:duration=first[sa];"
             f"[v1][sv]xfade=transition=fade:duration={XF}:offset={o1}[x1];"
             f"[x1][v4]xfade=transition=fade:duration={XF}:offset={o2}[v];"
             f"[a1][sa]acrossfade=d={XF}[y1];[y1][a4]acrossfade=d={XF}[a]")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", video, "-i", sh, "-i", ad,
                    "-filter_complex", graph, "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "15",
                    "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", out], check=True)
    os.remove(sh)


if __name__ == "__main__":
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == "shot":
        shot(a[0])
    else:
        insert(a[0], float(a[1]), a[2], *(a[3:4] or [AD]), *[int(x) for x in a[4:5]])
