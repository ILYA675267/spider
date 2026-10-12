"""Подготовка голоса к ролику: части озвучки → распознавание слов → склейка с шаблоном концовки → чистка пауз.

python tools/voice_prep.py <папка_ролика> <часть1.mp3> <часть2.mp3> ... [--asr <python с faster-whisper>]

Создаёт в папке ролика: voice/partN.mp3, voice/partN.words.json, voice_full.wav, words.json (общая шкала),
voice_clean.wav (паузы приглушены по ГРОМКОСТИ с удержанием 0,25 с — окончания слов не срезаются),
outro_start.txt (секунда, где начинается концовка про Telegram).
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTRO = os.path.join(ROOT, "assets", "voice", "outro_telegram.mp3")
SR = 44100


def load(p, sr=SR):
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", p, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def transcribe(asr_python, mp3, out_json):
    raw = os.path.splitext(out_json)[0] + ".raw"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", mp3, "-ar", "16000", "-ac", "1", "-f", "f32le", raw], check=True)
    r = subprocess.run([asr_python, "-I", os.path.join(ROOT, "tools", "asr.py"), raw, out_json],
                       capture_output=True, text=True, check=True)
    os.remove(raw)
    return r.stdout.strip().splitlines()[-1]


def main():
    args = sys.argv[1:]
    asr = "python3"
    if "--asr" in args:
        i = args.index("--asr")
        asr = args[i + 1]
        del args[i:i + 2]
    folder, parts = args[0], args[1:]
    vdir = os.path.join(folder, "voice")
    os.makedirs(vdir, exist_ok=True)
    files = []
    for i, p in enumerate(parts + [OUTRO], 1):
        dst = os.path.join(vdir, f"part{i}.mp3")
        shutil.copy(p, dst)
        wj = os.path.join(vdir, f"part{i}.words.json")
        if not os.path.exists(wj):
            print(f"часть {i}:", transcribe(asr, dst, wj))
        files.append((dst, wj))

    # склейка: обрезаем тишину по краям частей, между частями пауза 0,32 с (перед концовкой 0,45 с)
    out = [np.zeros(int(0.25 * SR), np.float32)]
    t = 0.25
    words, outro_start = [], None
    for i, (a, wj) in enumerate(files):
        x = load(a)
        idx = np.where(np.abs(x) > 10 ** (-42 / 20))[0]
        s0, s1 = max(0, idx[0] - int(0.03 * SR)), min(len(x), idx[-1] + int(0.15 * SR))
        seg = x[s0:s1].copy()
        n = int(0.01 * SR)
        seg[:n] *= np.linspace(0, 1, n)
        seg[-n:] *= np.linspace(1, 0, n)
        off = t - s0 / SR
        if i == len(files) - 1:
            outro_start = round(t, 3)
        for w in json.load(open(wj)):
            words.append({"w": w["w"], "s": round(w["s"] + off, 3), "e": round(w["e"] + off, 3), "part": i + 1})
        out.append(seg)
        t += len(seg) / SR
        gap = 0.32 if i < len(files) - 2 else 0.45
        out.append(np.zeros(int(gap * SR), np.float32))
        t += gap
    out.append(np.zeros(int(0.6 * SR), np.float32))
    y = np.concatenate(out)
    full = os.path.join(folder, "voice_full.wav")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-",
                    "-af", "loudnorm=I=-14:TP=-1.2:LRA=9", "-ar", str(SR), full], input=y.tobytes(), check=True)
    json.dump(words, open(os.path.join(folder, "words.json"), "w"), ensure_ascii=False, indent=0)
    open(os.path.join(folder, "outro_start.txt"), "w").write(str(outro_start))

    # чистка пауз по громкости (с удержанием), проверка, что речь не тронута
    x = load(full)
    hop = 441
    n = len(x) // hop
    db = 20 * np.log10(np.sqrt(np.mean(x[:n * hop].reshape(n, hop) ** 2, 1)) + 1e-9)
    keep = db > -48
    for a in words:
        keep[max(0, int((a["s"] - 0.1) * 100)):int((a["e"] + 0.35) * 100)] = True
    k2 = keep.copy()
    for i in range(1, 26):
        k2[i:] |= keep[:-i]
    k2[:-8] |= keep[8:]
    g = np.repeat(np.where(k2, 1.0, 10 ** (-40 / 20)).astype(np.float32), hop)
    g = np.concatenate([g, np.full(len(x) - len(g), g[-1])])
    f = int(0.06 * SR)
    g = np.convolve(g, np.ones(f) / f, "same").astype(np.float32)
    yc = x * g
    clean = os.path.join(folder, "voice_clean.wav")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-", clean],
                   input=yc.tobytes(), check=True)
    dby = 20 * np.log10(np.sqrt(np.mean(yc[:n * hop].reshape(n, hop) ** 2, 1)) + 1e-9)
    loud = db > -45
    print(f"длина {len(x) / SR:.2f} с, концовка с {outro_start} с, ослабление речи макс. {float((db - dby)[loud].max()):.2f} дБ")


if __name__ == "__main__":
    main()
