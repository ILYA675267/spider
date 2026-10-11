"""Озвучка текста через Fish Audio API.

python3 tools/fish_tts.py <текст.txt> <выход.mp3> <id_голоса> [модель]

Ключ берётся из переменной окружения FISH_AUDIO_API_KEY (кладётся в настройки облачной среды,
в код и в чат его не писать). id голоса — часть ссылки на голос в библиотеке Fish Audio:
https://fish.audio/m/<id_голоса>/
"""
import json
import os
import sys
import urllib.error
import urllib.request

API = "https://api.fish.audio/v1/tts"


def tts(text, out, voice_id, model="s1"):
    key = os.environ.get("FISH_AUDIO_API_KEY")
    if not key:
        sys.exit("Нет ключа: добавьте FISH_AUDIO_API_KEY в настройки облачной среды и откройте новую сессию.")
    body = json.dumps({"text": text, "reference_id": voice_id, "format": "mp3",
                       "mp3_bitrate": 192, "normalize": True, "latency": "normal"}).encode()
    req = urllib.request.Request(API, data=body, method="POST", headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json", "model": model})
    try:
        with urllib.request.urlopen(req, timeout=180) as r, open(out, "wb") as f:
            f.write(r.read())
    except urllib.error.HTTPError as e:
        msg = e.read().decode(errors="replace")[:300]
        sys.exit(f"Fish Audio ответил ошибкой {e.code}: {msg}")
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    text = open(a[0], encoding="utf-8").read().strip()
    print(tts(text, a[1], a[2], a[3] if len(a) > 3 else "s1"))
