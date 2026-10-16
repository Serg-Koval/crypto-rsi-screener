# Market Heat Scanner (crypto-rsi-screener)

Telegram-скринер перегріву і «short watch» на USDT-перпетуалах **Bybit** (лише крипта: акції, ETF і товари, які Bybit теж торгує, відсікаються за `symbolType`). Раз на годину серед монет з найбільшим зростанням за 24 год з обігом ≥ 5 млн (ріст порогом не є) знаходить ті, де RSI у зоні перегріву, та показує, де є підстави дивитись на short.

Це список спостереження (watchlist), а не сигнали на вхід. Бот **не торгує**. Рішення приймає людина.

## Як це працює

```
Cloudflare Worker (ladovka-screener-dispatcher)
   ├─ cron раз на годину ──────┐
   └─ Telegram-команда /run ───┤
                               ▼
        GitHub Actions: workflow "Ladovka RSI Screener" (screener.yml)
                               ▼
        python main.py  →  запити до Bybit ЧЕРЕЗ проксі у Worker → аналіз → звіт у Telegram
```

1. **Cloudflare Worker** за розкладом (cron) або за командою `/run` у Telegram запускає workflow `screener.yml` у GitHub Actions.
2. **GitHub Actions** виконує `python main.py`. Весь аналіз (RSI, фактори, класифікація) робить Python.
3. **Дані Bybit** приходять через **проксі в тому ж Worker** (`GET /bybit/*`). Напряму з GitHub Actions до Bybit звернутись не можна: Bybit відповідає HTTP 403 (геоблок). Прямих запитів до `api.bybit.com` у коді немає.
4. **Звіт** надходить у Telegram одним повідомленням.

### Проксі Bybit у Worker

- Шлях: `GET /bybit/*`. Захист: заголовок `X-Proxy-Secret`. Без секрету Worker відповідає 401, на чужий шлях 404.
- Дозволені лише чотири шляхи Bybit: `/v5/market/tickers`, `/v5/market/kline`, `/v5/market/open-interest`, `/v5/market/instruments-info`.
- **У Worker увімкнено Placement = EU2 (Європа).** Не змінюйте це: інакше Worker виконується в США (поруч із GitHub-раннером), і Bybit знову дає 403.
- Код Worker (`worker.js`) живе в Cloudflare, у цьому репозиторії його немає.

## Секрети

Значення секретів ніколи не записуються в код, логи чи коміти.

**GitHub** (Settings → Secrets and variables → Actions):

| Секрет | Для чого |
|---|---|
| `TELEGRAM_BOT_TOKEN` | токен Telegram-бота |
| `TELEGRAM_CHAT_ID` | чат, куди йде звіт |
| `BYBIT_PROXY_URL` | адреса Worker (без слеша в кінці) |
| `BYBIT_PROXY_SECRET` | секрет проксі (той самий, що `PROXY_SECRET` у Cloudflare) |

**Cloudflare Worker**: `PROXY_SECRET` (секрет проксі) та власні секрети Worker для запуску workflow і Telegram-вебхука.

## Як запустити

- **З Telegram:** надішліть боту команду `/run`.
- **З GitHub:** Actions → **Ladovka RSI Screener** → Run workflow.
- **За розкладом:** раз на годину запускає Worker.

Workflow не має власного розкладу в GitHub: розклад задано в Cloudflare. Одночасно виконується лише один запуск (`concurrency`), ліміт часу 15 хвилин.

## Що показує звіт

Для кожної монети: статус (`HIGH PRIORITY SHORT WATCH`, `SHORT WATCH`, `OVERHEAT WATCH`), символ (наприклад `BTCUSDT.P`), ціну, зміну за 24 год, обіг, RSI 1H / 4H, контекст Open Interest, дивергенції, рівні відкриття (D/W/M/Y), liquidity sweep і rejection candle. Повний опис логіки, порогів і формул: [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md).

### Попередження про неповні дані

Неповні дані не маскуються під «ринок спокійний»:

- якщо частину монет не вдалося проаналізувати (не більше половини), у звіті є рядок `⚠️ Проаналізовано X з Y кандидатів, N помилок`;
- якщо не вдалося проаналізувати **більше половини** кандидатів, замість звіту приходить `⚠️ Дані неповні...`, а запуск у GitHub завершується з помилкою;
- якщо не вдалося отримати список монет або тікери з Bybit, приходить окреме попередження, запуск теж завершується з помилкою.

## Перевірки

```
python main.py --self-test      # самотести (усі мають проходити)
python tests/golden_test.py     # «золотий» регресійний тест: має показати «без відмінностей»
```

Золотий тест фіксує результати аналізу на 1177 сценаріях. Його очікуваний файл (`tests/golden_expected.json`) перезаписується (`--update`) лише свідомо, коли змінюється логіка аналізу.

Залежності закріплені в `requirements.txt` (Python 3.11).

## Файли

| Файл | Призначення |
|---|---|
| `main.py` | лише точка входу: `python main.py` запускає скринер, `python main.py --self-test` самотести |
| `screener/config.py` | усі константи й пороги, `SCRIPT_VERSION` |
| `screener/common.py` | HTTP-сесія, RSI, форматування, допоміжні функції для свічок і факторів |
| `screener/oi_context.py` | історія й метрики Open Interest, підписи OI-контексту |
| `screener/factors/sweep.py` | фактор liquidity sweep |
| `screener/factors/divergence.py` | RSI- і OI-дивергенції |
| `screener/factors/open_levels.py` | рівні відкриття D/W/M/Y і їхній контекст |
| `screener/factors/rejection.py` | фактор rejection candle |
| `screener/factors/analysis.py` | збирає п'ять факторів в один результат (`analyze_short_factors`) |
| `screener/classification.py` | оцінки, дерево класифікації, ранги сигналів |
| `screener/bybit.py` | запити до Bybit через проксі, universe, prefilter, аналіз кандидата |
| `screener/telegram.py` | облікові дані Telegram, розбиття довгих повідомлень, відправка |
| `screener/report.py` | таблиці сигналів, текст Telegram-звіту, діагностичні логи |
| `screener/runner.py` | `run_screener` і контроль повноти даних |
| `.github/workflows/screener.yml` | workflow запуску скринера |
| `requirements.txt` | закріплені версії бібліотек |
| `tests/cases.py` | синтетичні свічки й фактори для тестів (`make_*`) |
| `tests/self_tests.py`, `tests/self_tests_*.py` | самотести; `run_all()` запускає їх усі |
| `tests/golden_test.py`, `tests/golden_expected.json` | «золотий» регресійний тест |
| `CLAUDE.md` | правила роботи Claude Code в проєкті |
| `PROJECT_CONTEXT.md` | повний опис логіки, порогів і поточного стану |

## Важливо

Проєкт призначений для спостереження за ринком і ручного прийняття рішень. Він не виконує угод і не є фінансовою порадою.
