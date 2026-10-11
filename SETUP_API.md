# Подключение нейросетей, чтобы Claude делал ролики сам

Ключи **никогда не присылать в чат** — только в настройки облачной среды.
Настройки начинают работать **в новой сессии**.

## 1. Fish Audio (голос)
1. fish.audio → «Разработчик» → создать API-ключ.
2. «Настройки команды» → вкладка «Кошелёк» → пополнить ($5–10 хватит надолго).

## 2. fal.ai (оживление персонажа: картинка + голос → видео)
Один ключ даёт доступ к OmniHuman (говорит и жестикулирует под голос), Kling, Hailuo и др.
Цена OmniHuman ≈ $0,14–0,16 за секунду видео (ролик 30 с ≈ $4–5, проба 10 с ≈ $1,5).
1. Зарегистрироваться на fal.ai → Dashboard → **Keys** → создать ключ.
2. **Billing** → пополнить баланс (карта; российские карты, скорее всего, не пройдут).

## 3. Настройки облака (меню среды вверху сессии → Edit)
**Network access → Allowed domains** — добавить:
```
api.fish.audio
fal.run
queue.fal.run
rest.alpha.fal.ai
fal.media
v3.fal.media
v3b.fal.media
storage.googleapis.com
```
(галочку «Allow package managers» не снимать)

**Ключи** — в разделе Network secrets, а если его нет — в Environment variables:
```
FISH_AUDIO_API_KEY=ключ_от_fish_audio
FAL_KEY=ключ_от_fal
```

## 4. Что прислать Claude в новой сессии
- ссылку на выбранный голос Fish Audio;
- картинку персонажа (лучше в полный рост, стоит прямо).

Подробно о настройках среды: https://code.claude.com/docs/en/cloud-environments#network-access
