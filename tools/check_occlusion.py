"""Проверка: не закрывают ли вставки персонажа. Каждый момент рисуется дважды — как обычно и с персонажем
поверх всего; если в области персонажа кадры отличаются больше чем на 2% — его что-то закрывает.
python tools/check_occlusion.py   (печатает [(секунда, % закрыто)], пустой список — всё чисто)"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2, numpy as np, mg, memory_mg as M
sc = M.Scene(os.path.join(M.ROOT, "work", "memory", "voice_clean.wav"))
bad = []
TS = [i / 4 for i in range(int(M.DUR * 4))]
for t in TS:
    sc.char_last = False; A = sc.draw(t).astype(np.int16)
    sc.char_last = True; B = sc.draw(t).astype(np.int16)
    pos, h, canvas, facing, sq, cam = sc.char_args
    white = canvas.copy(); white[..., :3] = white[..., 3:4]
    m = np.zeros((mg.H, mg.W, 3), np.uint8)
    mg.billboard(m, cam, pos, h, white, facing_yaw=facing, anchor=sc.anchor, squash=sq)
    mask = m[..., 0] > 128
    if mask.sum() < 50:
        continue
    diff = (np.abs(A - B).max(-1) > 30) & mask
    frac = diff.sum() / mask.sum()
    if frac > 0.02:
        bad.append((t, round(float(frac) * 100, 1)))
print(bad)
