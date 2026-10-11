# Как подключить API-ключ Fish Audio

Ключ **никогда не присылать в чат** — только в настройки облачной среды.

1. Вверху сессии Claude Code нажать на название облачной среды (меню среды) → **Edit**.
2. **Network access** → в **Allowed domains** добавить: `api.fish.audio`
   (галочку «Allow package managers» не снимать).
3. Ключ:
   - если есть раздел **Network secrets** — добавить ключ туда для домена `api.fish.audio`;
   - если нет — в **Environment variables** добавить строку `FISH_AUDIO_API_KEY=ваш_ключ`.
4. Сохранить. Настройки начинают работать **в новой сессии**.
5. Пополнить баланс: fish.audio → «Настройки команды» → вкладка **«Кошелёк»** (API тратит деньги оттуда,
   а не кредиты бесплатного плана).
6. Выбрать голос в библиотеке Fish Audio и прислать Claude ссылку на него (это не секрет).

Подробно: https://code.claude.com/docs/en/cloud-environments#network-access
