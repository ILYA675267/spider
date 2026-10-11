"""Фон по умолчанию: крыша на закате с силуэтами города (мягко размыт).

python3 tools/background.py assets/background.png
Фон можно заменить любой своей картинкой 1080×1920 с тем же именем.
"""
import random
import sys

from PIL import Image, ImageDraw, ImageFilter

W, H = 1080, 1920


def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def make(path):
    rnd = random.Random(7)
    im = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(im)
    # небо: тёмно-синий → фиолетовый → оранжевый у горизонта
    stops = [(0, (28, 30, 82)), (0.38, (98, 58, 128)), (0.62, (236, 120, 92)), (0.75, (255, 190, 120))]
    for y in range(H):
        t = y / H
        for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
            if t0 <= t <= t1:
                d.line([(0, y), (W, y)], fill=lerp(c0, c1, (t - t0) / (t1 - t0)))
                break
        else:
            d.line([(0, y), (W, y)], fill=stops[-1][1])
    # солнце
    sun = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(sun).ellipse([620, 1060, 900, 1340], fill=(255, 225, 160, 255))
    im.paste(sun.filter(ImageFilter.GaussianBlur(12)), (0, 0), sun.filter(ImageFilter.GaussianBlur(12)))

    # три слоя города: дальние светлее и сильнее размыты
    horizon = 1330
    layers = [((150, 92, 140), 0.55, 9, 260), ((98, 56, 104), 0.8, 6, 380), ((52, 30, 66), 1.0, 3, 520)]
    for color, k, blur, hmax in layers:
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ld = ImageDraw.Draw(lay)
        x = -40
        while x < W:
            bw = rnd.randint(70, 150)
            bh = rnd.randint(int(hmax * 0.35), hmax)
            top = horizon - bh
            ld.rectangle([x, top, x + bw, H], fill=color + (255,))
            if rnd.random() < 0.25:  # антенна
                ld.rectangle([x + bw // 2 - 3, top - 60, x + bw // 2 + 3, top], fill=color + (255,))
            # окна
            for wy in range(top + 18, horizon - 10, 34):
                for wx in range(x + 12, x + bw - 16, 26):
                    if rnd.random() < 0.33 * k:
                        lit = lerp((255, 200, 110), (255, 235, 170), rnd.random())
                        ld.rectangle([wx, wy, wx + 11, wy + 16], fill=lit + (255,))
            x += bw + rnd.randint(-10, 12)
        lay = lay.filter(ImageFilter.GaussianBlur(blur))
        im.paste(lay, (0, 0), lay)

    # крыша, на которой стоит персонаж
    roof_top = 1390
    roof = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    rd = ImageDraw.Draw(roof)
    for y in range(roof_top, H):
        t = (y - roof_top) / (H - roof_top)
        rd.line([(0, y), (W, y)], fill=lerp((120, 86, 104), (66, 44, 62), t) + (255,))
    rd.rectangle([0, roof_top - 46, W, roof_top], fill=(84, 56, 76, 255))       # бортик
    rd.rectangle([0, roof_top - 46, W, roof_top - 38], fill=(150, 104, 112, 255))
    # водонапорная башня — узнаваемая деталь нью-йоркских крыш
    rd.rectangle([60, 1060, 250, 1250], fill=(70, 44, 60, 255))
    rd.polygon([(48, 1062), (155, 990), (262, 1062)], fill=(62, 38, 54, 255))
    for lx in (80, 150, 220):
        rd.rectangle([lx, 1250, lx + 10, roof_top - 46], fill=(56, 34, 50, 255))
    roof = roof.filter(ImageFilter.GaussianBlur(2.5))
    im.paste(roof, (0, 0), roof)

    # мягкая виньетка
    vig = Image.new("L", (W, H), 0)
    ImageDraw.Draw(vig).ellipse([-300, -200, W + 300, H + 200], fill=255)
    vig = vig.filter(ImageFilter.GaussianBlur(220))
    dark = Image.new("RGB", (W, H), (14, 8, 24))
    im = Image.composite(im, dark, vig)
    im.save(path)
    return path


if __name__ == "__main__":
    make(sys.argv[1] if len(sys.argv) > 1 else "assets/background.png")
