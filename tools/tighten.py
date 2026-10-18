"""Укоротить длинные паузы в озвучке (если ролик выходит длиннее минуты).

python tools/tighten.py <папка_ролика> [макс_пауза_с=0.4] [порог_тишины_дБ=-45] [ускорение=1.0]
Работает после voice_prep.py: берёт voice_full.wav и words.json, паузы между словами длиннее заданной
укорачивает (по ГРОМКОСТИ: после конца звука остаётся не меньше 0,25 с, перед началом — 0,08 с),
концовку-шаблон не трогает. Ускорение (без изменения высоты голоса, ffmpeg atempo) — только для части до концовки.
Пересчитывает время слов, outro_start.txt и чистит паузы заново (voice_clean.wav).
"""
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from voice_prep import SR, load  # noqa: E402


def main():
    folder = sys.argv[1]
    maxgap = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
    thr = float(sys.argv[3]) if len(sys.argv) > 3 else -45.0
    speed = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    full = os.path.join(folder, "voice_full.wav")
    x = load(full)
    words = json.load(open(os.path.join(folder, "words.json")))
    outro = float(open(os.path.join(folder, "outro_start.txt")).read())
    hop = 441
    n = len(x) // hop
    db = 20 * np.log10(np.sqrt(np.mean(x[:n * hop].reshape(n, hop) ** 2, 1)) + 1e-9)
    loud = db > thr
    # тишина: подряд идущие тихие кадры; укорачиваем только паузы до концовки
    cuts = []  # (начало вырезки, длина) в секундах
    i = 0
    while i < n:
        if loud[i]:
            i += 1
            continue
        j = i
        while j < n and not loud[j]:
            j += 1
        s, e = i / 100, j / 100
        if i > 0 and j < n and e < outro - 0.05 and e - s > maxgap:
            keep_a, keep_b = 0.27, 0.08          # хвост после звука ≥ 0,25 с и чуть-чуть перед началом
            extra = (e - s) - max(maxgap, keep_a + keep_b)
            if extra > 0.02:
                cuts.append((s + keep_a, extra))
        i = j
    # вырезаем
    keep = np.ones(len(x), bool)
    for a, L in cuts:
        keep[int(a * SR):int((a + L) * SR)] = False
    y = x[keep]

    def shift(t):
        return t - sum(L for a, L in cuts if a + L <= t) - sum(t - a for a, L in cuts if a < t < a + L)

    for w in words:
        w["s"], w["e"] = round(shift(w["s"]), 3), round(shift(w["e"]), 3)
    outro2 = round(shift(outro), 3)
    if speed != 1.0:  # ускоряем только речь до концовки, концовка-шаблон остаётся как есть
        k = int(outro2 * SR)
        r = subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-",
                            "-af", f"atempo={speed}", "-f", "f32le", "-"], input=y[:k].astype(np.float32).tobytes(),
                           capture_output=True, check=True)
        head = np.frombuffer(r.stdout, np.float32)
        y = np.concatenate([head, y[k:]])
        new_outro = len(head) / SR
        for w in words:
            if w["s"] < outro2:
                w["s"], w["e"] = round(w["s"] / speed, 3), round(min(w["e"] / speed, new_outro), 3)
            else:
                w["s"], w["e"] = round(w["s"] - outro2 + new_outro, 3), round(w["e"] - outro2 + new_outro, 3)
        outro2 = round(new_outro, 3)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-", full],
                   input=y.astype(np.float32).tobytes(), check=True)
    json.dump(words, open(os.path.join(folder, "words.json"), "w"), ensure_ascii=False, indent=0)
    open(os.path.join(folder, "outro_start.txt"), "w").write(str(outro2))

    # чистка пауз по громкости (как в voice_prep.py) и проверка, что речь не ослаблена
    x = y
    n = len(x) // hop
    db = 20 * np.log10(np.sqrt(np.mean(x[:n * hop].reshape(n, hop) ** 2, 1)) + 1e-9)
    keepf = db > -48
    for a in words:
        keepf[max(0, int((a["s"] - 0.1) * 100)):int((a["e"] + 0.35) * 100)] = True
    k2 = keepf.copy()
    for i in range(1, 26):
        k2[i:] |= keepf[:-i]
    k2[:-8] |= keepf[8:]
    g = np.repeat(np.where(k2, 1.0, 10 ** (-40 / 20)).astype(np.float32), hop)
    g = np.concatenate([g, np.full(len(x) - len(g), g[-1])])
    f = int(0.06 * SR)
    g = np.convolve(g, np.ones(f) / f, "same").astype(np.float32)
    yc = x * g
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-",
                    os.path.join(folder, "voice_clean.wav")], input=yc.tobytes(), check=True)
    dby = 20 * np.log10(np.sqrt(np.mean(yc[:n * hop].reshape(n, hop) ** 2, 1)) + 1e-9)
    loud = db > -45
    print(f"вырезано пауз: {len(cuts)}, всего {sum(L for _, L in cuts):.2f} с; длина {len(x) / SR:.2f} с, "
          f"концовка с {outro2} с, ослабление речи макс. {float((db - dby)[loud].max()):.2f} дБ")


if __name__ == "__main__":
    main()
