# crypto-rsi-screener

OKX USDT perpetual RSI screener with Telegram alerts.

## Current MVP

- Fetches OKX USDT perpetual market data
- Filters by 24h change and 24h volume
- Calculates TradingView-compatible Wilder RSI
- Checks RSI 1H / 4H overheating
- Sends compact Telegram alerts

## Required GitHub Secrets

Repository settings → Secrets and variables → Actions:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

## Manual run

Actions → OKX RSI Screener → Run workflow

## Notes

This project is for learning, demo scanning, and manual decision support only.
It does not execute trades.
