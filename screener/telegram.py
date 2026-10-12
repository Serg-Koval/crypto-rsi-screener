import os
import time
import requests

from screener.config import (
    TELEGRAM_SEND_MAX_ATTEMPTS,
    TELEGRAM_SEND_RETRY_SLEEP_SECONDS,
    TELEGRAM_SEND_TIMEOUT_SECONDS,
)


def get_telegram_credentials():
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not bot_token:
        raise Exception("TELEGRAM_BOT_TOKEN is missing.")

    if not chat_id:
        raise Exception("TELEGRAM_CHAT_ID is missing.")

    return bot_token, chat_id


def redact_telegram_token(text):
    """Hide the bot token in text that goes to logs.

    GitHub masks only the exact secret value, but exception messages from
    requests contain the full request URL (/bot<token>/sendMessage).
    """

    text = str(text)
    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()

    if not token:
        return text

    return text.replace(f"bot{token}", "***").replace(token, "***")


def telegram_signal_label(signal_level):
    labels = {
        "HIGH_PRIORITY_SHORT_WATCH": "🔴 HIGH PRIORITY SHORT WATCH",
        "SHORT_WATCH": "🟠 SHORT WATCH",
        "OVERHEAT_WATCH": "🟡 OVERHEAT WATCH",
        "PUMP_WATCH": "🟡 PUMP WATCH",
        "NO_SIGNAL": "⚪ NO SIGNAL",
    }

    return labels.get(signal_level, signal_level)


def split_long_message(text, max_length=3900):
    if len(text) <= max_length:
        return [text]

    parts = []
    current_part = ""

    for line in text.split("\n"):
        if len(current_part) + len(line) + 1 > max_length:
            parts.append(current_part)
            current_part = line
        else:
            current_part += "\n" + line if current_part else line

    if current_part:
        parts.append(current_part)

    return parts


def send_telegram_message(text):
    """Send a Telegram message without crashing the screener on Telegram outages.

    r33 failed after the analysis was completed because Telegram sendMessage
    timed out and the shared SESSION retried the POST several times. For the
    report delivery step we prefer a short, best-effort request: log the error
    and let GitHub Actions finish successfully so the run results remain usable.
    """

    bot_token, chat_id = get_telegram_credentials()

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    for attempt in range(1, int(TELEGRAM_SEND_MAX_ATTEMPTS) + 1):
        try:
            # Use requests.post directly instead of the shared retrying SESSION.
            # This avoids long urllib3 POST retries when api.telegram.org is slow.
            response = requests.post(
                url,
                json=payload,
                timeout=int(TELEGRAM_SEND_TIMEOUT_SECONDS),
            )
        except requests.exceptions.RequestException as e:
            print(
                "Telegram sendMessage failed (non-fatal): "
                f"attempt={attempt}/{TELEGRAM_SEND_MAX_ATTEMPTS} error={redact_telegram_token(repr(e))}"
            )
            if attempt < int(TELEGRAM_SEND_MAX_ATTEMPTS):
                time.sleep(float(TELEGRAM_SEND_RETRY_SLEEP_SECONDS))
                continue
            return False

        if response.status_code == 200:
            print("Telegram message sent.")
            return True

        print("Telegram error response (non-fatal):")
        print(redact_telegram_token(response.text[:1000]))

        if attempt < int(TELEGRAM_SEND_MAX_ATTEMPTS):
            time.sleep(float(TELEGRAM_SEND_RETRY_SLEEP_SECONDS))
            continue

        return False

    return False


def send_telegram_message_safe(text):
    all_sent = True
    parts = split_long_message(text)

    for index, part in enumerate(parts):
        if index > 0:
            part = f"Part {index + 1}\n\n" + part

        sent = send_telegram_message(part)
        all_sent = bool(all_sent and sent)

        if index < len(parts) - 1:
            time.sleep(1)

    if not all_sent:
        print("WARNING: Telegram delivery failed or was incomplete. Screener analysis completed; this is a delivery-layer issue.")

    return all_sent
