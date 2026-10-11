"""Сборка видео: кадры персонажа (с прозрачностью) поверх фона → mp4 1080×1920.

python3 tools/compose.py <папка кадров> <фон.png> <выход.mp4> [аудио.wav]
"""
import subprocess
import sys


def compose(frames, background, out, audio=None, fps=24):
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-loop", "1", "-framerate", str(fps), "-i", background,
           "-framerate", str(fps), "-i", f"{frames}/%04d.png"]
    if audio:
        cmd += ["-i", audio]
    vf = ("[0:v]scale=1080:1920,setsar=1[bg];"
          "[1:v]scale=1080:1920:flags=lanczos[fg];"
          "[bg][fg]overlay=0:0:shortest=1,format=yuv420p[v]")
    cmd += ["-filter_complex", vf, "-map", "[v]"]
    if audio:
        cmd += ["-map", "2:a", "-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", "17", "-movflags", "+faststart", out]
    subprocess.run(cmd, check=True)
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    compose(a[0], a[1], a[2], a[3] if len(a) > 3 else None)
