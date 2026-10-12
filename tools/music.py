"""Фоновая музыка: спокойный, но бодрый лоу-фай бит, полностью синтезируется (никаких чужих треков — без жалоб).

python3 tools/music.py <выход.wav> <длительность_с> [bpm] [seed]

Аккорды Am – F – C – G, мягкий пэд, бас, «перебор», лёгкие ударные. Начало — плавное нарастание, конец — затухание.
"""
import math
import subprocess
import sys

import numpy as np

SR = 44100


def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12)


def onepole(x, cutoff):
    """Однополюсный ФНЧ (мягкий фильтр) через свёртку с экспоненциальным окном."""
    a = math.exp(-2 * math.pi * cutoff / SR)
    n = int(6 / (1 - a)) + 1
    k = (1 - a) * a ** np.arange(n)
    return np.convolve(x, k)[:len(x)]


def saw(f, t, detune=0.0):
    ph = (f * (1 + detune)) * t
    return 2 * (ph - np.floor(ph + 0.5))


def env_adsr(n, a, d, s, r, sr=SR):
    e = np.ones(n) * s
    ia, idd, ir = int(a * sr), int(d * sr), int(r * sr)
    ia = min(ia, n)
    e[:ia] = np.linspace(0, 1, ia, endpoint=False)
    e[ia:ia + idd] = np.linspace(1, s, len(e[ia:ia + idd]), endpoint=False)
    if ir > 0:
        e[-ir:] *= np.linspace(1, 0, min(ir, n))
    return e


def make(dur, bpm=104, seed=3):
    rng = np.random.default_rng(seed)
    beat = 60 / bpm
    bar = 4 * beat
    n = int(dur * SR)
    pad = np.zeros(n)
    bass = np.zeros(n)
    pluck = np.zeros(n)
    drums = np.zeros(n)
    prog = [(57, [57, 60, 64]), (53, [53, 57, 60]), (48, [55, 60, 64]), (55, [55, 59, 62])]  # Am F C G
    nbars = int(math.ceil(dur / bar))

    for b in range(nbars):
        root, chord = prog[b % 4]
        t0 = b * bar
        i0 = int(t0 * SR)
        L = min(int(bar * SR), n - i0)
        if L <= 0:
            break
        t = np.arange(L) / SR
        # пэд: три немного расстроенных пилы, мягкий фильтр; аккорды перетекают друг в друга без провалов
        Lp = min(L + int(0.6 * SR), n - i0)
        if Lp < 2:
            continue
        tp = np.arange(Lp) / SR
        p = np.zeros(Lp)
        for m in chord:
            f = midi(m)
            p += saw(f, tp, -0.004) + saw(f, tp, 0.004) + 0.5 * saw(f * 2, tp, 0.002)
        e = np.ones(Lp)
        ia = min(int(0.3 * SR), Lp)
        e[:ia] = np.sin(np.linspace(0, math.pi / 2, ia)) ** 2
        if Lp > L:
            e[L:] = np.cos(np.linspace(0, math.pi / 2, Lp - L)) ** 2
        p = onepole(p, 900) * e * 0.05
        pad[i0:i0 + Lp] += p
        # бас: восьмые, корень и октава, тёплый
        for k in range(8):
            s0 = int(k * beat / 2 * SR)
            ln = int(beat / 2 * 0.85 * SR)
            if s0 >= L:
                break
            ln = min(ln, L - s0)
            tt = np.arange(ln) / SR
            note = root - 12 + (12 if k in (3, 7) else 0)
            f = midi(note)
            v = (np.sin(2 * math.pi * f * tt) + 0.35 * saw(f, tt)) * env_adsr(ln, 0.005, 0.08, 0.6, 0.05)
            bass[i0 + s0:i0 + s0 + ln] += onepole(v, 500) * 0.22
        # перебор: шестнадцатые по нотам аккорда вверх-вниз, с затуханием и «эхом»
        arp = chord + [chord[1] + 12, chord[2] + 12, chord[1] + 12]
        for k in range(16):
            if rng.random() < 0.18:
                continue  # пропуски — живее
            s0 = int(k * beat / 4 * SR)
            ln = min(int(0.32 * SR), L - s0)
            if ln <= 0:
                continue
            tt = np.arange(ln) / SR
            f = midi(arp[k % len(arp)] + 12)
            tone = np.sin(2 * math.pi * f * tt) + 0.3 * np.sin(4 * math.pi * f * tt)
            pluck[i0 + s0:i0 + s0 + ln] += tone * np.exp(-tt * 14) * (0.055 if k % 4 == 0 else 0.04)
        # ударные: бочка на каждую долю (мягко), хлопок на 2 и 4, хэты восьмыми
        for k in range(4):
            s0 = i0 + int(k * beat * SR)
            if s0 >= n - 10:
                break
            ln = min(int(0.35 * SR), n - s0)
            tt = np.arange(ln) / SR
            kick = np.sin(2 * math.pi * (45 + 75 * np.exp(-tt * 28)) * tt) * np.exp(-tt * 9)
            drums[s0:s0 + ln] += kick * 0.42
            if k in (1, 3):
                ln2 = min(int(0.22 * SR), n - s0)
                nz = rng.standard_normal(ln2)
                clap = (nz - onepole(nz, 1200)) * np.exp(-np.arange(ln2) / SR * 18)
                drums[s0:s0 + ln2] += onepole(clap, 6000) * 0.16
        for k in range(8):
            s0 = i0 + int(k * beat / 2 * SR)
            ln = min(int(0.06 * SR), n - s0)
            if ln <= 1:
                continue
            nz = rng.standard_normal(ln)
            hat = (nz - onepole(nz, 6000)) * np.exp(-np.arange(ln) / SR * 60)
            drums[s0:s0 + ln] += hat * (0.07 if k % 2 else 0.045)

    # простое «эхо» на переборе
    d = int(beat * 0.75 * SR)
    echo = np.zeros(n)
    echo[d:] = pluck[:-d] * 0.35
    out = pad + bass + pluck + echo + drums
    # мягкое нарастание в начале (2 такта: без ударных громкость ниже) и затухание в конце
    fade_in = np.clip(np.arange(n) / (SR * 1.5), 0, 1)
    fade_out = np.clip((n - np.arange(n)) / (SR * 2.0), 0, 1)
    out = out * fade_in * fade_out
    out = np.tanh(out * 1.3) / 1.3  # мягкое ограничение пиков
    out /= np.max(np.abs(out)) + 1e-9
    return (out * 0.9).astype(np.float32)


if __name__ == "__main__":
    path, dur = sys.argv[1], float(sys.argv[2])
    bpm = float(sys.argv[3]) if len(sys.argv) > 3 else 104
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    y = make(dur, bpm, seed)
    stereo = np.stack([y, np.roll(y, int(0.012 * SR)) * 0.96], 1)  # лёгкая ширина
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-",
                    "-af", "loudnorm=I=-16:TP=-1.5", path], input=stereo.tobytes(), check=True)
