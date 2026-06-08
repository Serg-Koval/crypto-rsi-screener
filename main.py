import os
import time
import html
import requests
import pandas as pd
import numpy as np

from datetime import datetime
from zoneinfo import ZoneInfo
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============================================================
# GLOBAL CONFIG
# ============================================================

BASE_URL = "https://www.okx.com"

INST_TYPE = "SWAP"          # OKX perpetual contracts
SETTLE_CCY = "USDT"         # USDT-margined perpetuals only

RSI_PERIOD = 14

# Signal logic
RSI_4H_ALERT = 80
RSI_4H_EXTREME = 85
RSI_1H_ALERT = 82

# Prefilter:
# price_change_24h >= 4%
# AND volume_usd_24h_est >= 5M
MIN_PRICE_CHANGE_24H = 4
MIN_VOLUME_USD_24H = 5_000_000

# Scanner limits
PRE_FILTER_TOP_N = 40
FINAL_TOP_N = 30

# More history for stable TradingView-like RSI
CANDLE_LIMIT_1H = 500
CANDLE_LIMIT_4H = 500

REQUEST_DELAY_SECONDS = 0.15

# Telegram
TELEGRAM_ENABLED = True
SEND_MESSAGE_IF_NO_SIGNALS = True

# Timezone
KYIV_TZ = ZoneInfo("Europe/Kyiv")

# Output controls
DEBUG_SHOW_MARKET_SAMPLE = False
DEBUG_SHOW_PREFILTERED_CANDIDATES = False
VERBOSE_PROGRESS = False


# ============================================================
# HTTP SESSION WITH RETRY
# ============================================================

def create_http_session():
    session = requests.Session()

    retry_strategy = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"]
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update({
        "User-Agent": "Mozilla/5.0 crypto-rsi-screener/1.0"
    })

    return session


SESSION = create_http_session()


# ============================================================
# SAFE REQUEST HELPER
# ============================================================

def safe_get_json(endpoint, params=None, show_debug=False):
    if params is None:
        params = {}

    url = BASE_URL + endpoint

    response = SESSION.get(url, params=params, timeout=20)

    if show_debug:
        print("=" * 100)
        print("Request URL:", response.url)
        print("Status code:", response.status_code)
        print("Content-Type:", response.headers.get("Content-Type"))

    if response.status_code != 200:
        print("HTTP error response:")
        print(response.text[:1000])
        raise Exception(f"HTTP error: {response.status_code}")

    try:
        data = response.json()
    except Exception as e:
        print("JSON parse error.")
        print("First 1000 characters of response:")
        print(response.text[:1000])
        raise e

    if data.get("code") != "0":
        print("OKX API returned error:")
        print(data)
        raise Exception(f"OKX API error: {data.get('msg')}")

    return data


# ============================================================
# OKX API FUNCTIONS
# ============================================================

def get_all_instruments(inst_type=INST_TYPE):
    endpoint = "/api/v5/public/instruments"
    data = safe_get_json(endpoint, {"instType": inst_type})
    return data["data"]


def get_all_tickers(inst_type=INST_TYPE):
    endpoint = "/api/v5/market/tickers"
    data = safe_get_json(endpoint, {"instType": inst_type})
    return data["data"]


def get_candles(inst_id, bar, limit):
    endpoint = "/api/v5/market/candles"

    params = {
        "instId": inst_id,
        "bar": bar,
        "limit": limit
    }

    data = safe_get_json(endpoint, params)
    return data


# ============================================================
# DATAFRAME HELPERS
# ============================================================

def instruments_to_dataframe(instruments):
    df = pd.DataFrame(instruments)

    required_columns = ["instType", "settleCcy", "state", "instId"]

    for col in required_columns:
        if col not in df.columns:
            raise Exception(f"Missing instrument column: {col}")

    df = df[
        (df["instType"] == INST_TYPE) &
        (df["settleCcy"] == SETTLE_CCY) &
        (df["state"] == "live") &
        (df["instId"].str.endswith("-USDT-SWAP"))
    ].copy()

    return df.reset_index(drop=True)


def tickers_to_dataframe(tickers):
    df = pd.DataFrame(tickers)

    numeric_columns = [
        "last",
        "open24h",
        "high24h",
        "low24h",
        "vol24h",
        "volCcy24h",
        "volCcyQuote24h"
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "ts" in df.columns:
        df["timestamp"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms")

    return df


def candles_to_dataframe(raw_candles):
    rows = raw_candles["data"]

    df = pd.DataFrame(
        rows,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "volume_currency",
            "quote_volume",
            "confirm"
        ]
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "volume_currency",
        "quote_volume"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms")
    df["confirm"] = df["confirm"].astype(int)

    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


# ============================================================
# TRADINGVIEW-COMPATIBLE RSI
# ============================================================

def calculate_wilder_rma(series, period):
    values = series.astype(float).to_numpy()
    rma = np.full(len(values), np.nan)

    valid_indexes = np.where(~np.isnan(values))[0]

    if len(valid_indexes) < period:
        return pd.Series(rma, index=series.index)

    seed_indexes = valid_indexes[:period]
    seed_end_index = seed_indexes[-1]

    seed_value = np.nanmean(values[seed_indexes])
    rma[seed_end_index] = seed_value

    previous_value = seed_value

    for i in range(seed_end_index + 1, len(values)):
        current_value = values[i]

        if np.isnan(current_value):
            rma[i] = previous_value
            continue

        current_rma = ((previous_value * (period - 1)) + current_value) / period
        rma[i] = current_rma
        previous_value = current_rma

    return pd.Series(rma, index=series.index)


def calculate_rsi(df, period=RSI_PERIOD):
    df = df.copy()

    delta = df["close"].diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    gain.iloc[0] = np.nan
    loss.iloc[0] = np.nan

    avg_gain = calculate_wilder_rma(gain, period)
    avg_loss = calculate_wilder_rma(loss, period)

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.where(avg_loss != 0, 100)
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50)

    df["rsi"] = rsi

    return df


# ============================================================
# INDICATOR HELPERS
# ============================================================

def get_last_closed_candle(df):
    closed_df = df[df["confirm"] == 1].copy()

    if closed_df.empty:
        raise Exception("No closed candles found.")

    return closed_df.iloc[-1]


def calculate_closed_24h_quote_volume_from_1h(df_1h):
    closed_df = df_1h[df_1h["confirm"] == 1].copy()

    if len(closed_df) < 24:
        return None

    return float(closed_df.tail(24)["quote_volume"].sum())


def calculate_volume_change_24h_from_1h(df_1h):
    closed_df = df_1h[df_1h["confirm"] == 1].copy()

    if len(closed_df) < 48:
        return None

    prev_24 = closed_df.iloc[-48:-24]
    last_24 = closed_df.iloc[-24:]

    prev_volume = prev_24["quote_volume"].sum()
    last_volume = last_24["quote_volume"].sum()

    if prev_volume == 0:
        return None

    return float(((last_volume - prev_volume) / prev_volume) * 100)


# ============================================================
# FORMATTING HELPERS
# ============================================================

def format_large_number(value):
    if value is None or pd.isna(value):
        return "N/A"

    value = float(value)

    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"

    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"

    if abs(value) >= 1_000:
        return f"{value / 1_000:.2f}K"

    return f"{value:.2f}"


def format_price_2(value):
    if value is None or pd.isna(value):
        return "N/A"

    return f"{float(value):.2f}"


def format_percent_2(value, show_plus=True):
    if value is None or pd.isna(value):
        return "N/A"

    value = float(value)

    if show_plus:
        return f"{value:+.2f}"

    return f"{value:.2f}"


def parse_float_from_value(value):
    if value is None:
        return None

    try:
        text = str(value).replace("%", "").replace("+", "").strip()

        if text.upper() == "N/A":
            return None

        return float(text)

    except Exception:
        return None


def make_report_symbol(inst_id):
    symbol = inst_id.replace("-SWAP", "")
    symbol = symbol.replace("-", "")
    symbol = symbol + ".P"

    return symbol


def print_df(df, title=None, max_rows=50):
    if title:
        print("\n" + "=" * 120)
        print(title)
        print("=" * 120)

    if df is None or df.empty:
        print("Empty table.")
        return

    print(df.head(max_rows).to_string(index=False))


def prepare_active_output_table(df_active):
    if df_active is None or df_active.empty:
        return df_active

    df_out = df_active.copy()

    df_out["price"] = df_out["price"].apply(format_price_2)
    df_out["rsi_1h"] = df_out["rsi_1h"].round(2)
    df_out["rsi_4h"] = df_out["rsi_4h"].round(2)
    df_out["chg_24h_%"] = df_out["price_change_24h_percent"].apply(format_percent_2)
    df_out["vol_24h"] = df_out["volume_usd_24h_exact"].apply(format_large_number)
    df_out["vol_chg_24h_%"] = df_out["volume_change_24h_percent"].apply(format_percent_2)

    columns = [
        "signal_level",
        "symbol",
        "price",
        "rsi_1h",
        "rsi_4h",
        "chg_24h_%",
        "vol_24h",
        "vol_chg_24h_%",
        "reason"
    ]

    return df_out[columns]


# ============================================================
# SCREENER LOGIC
# ============================================================

def build_market_universe():
    instruments = get_all_instruments(inst_type=INST_TYPE)
    df_instruments = instruments_to_dataframe(instruments)

    tickers = get_all_tickers(inst_type=INST_TYPE)
    df_tickers = tickers_to_dataframe(tickers)

    df = df_instruments.merge(
        df_tickers,
        on="instId",
        how="inner",
        suffixes=("_instrument", "_ticker")
    )

    df["price_change_24h_percent"] = (
        (df["last"] - df["open24h"]) / df["open24h"]
    ) * 100

    if "volCcyQuote24h" in df.columns and df["volCcyQuote24h"].notna().any():
        df["volume_usd_24h_est"] = df["volCcyQuote24h"]
    else:
        df["volume_usd_24h_est"] = df["volCcy24h"] * df["last"]

    df = df.dropna(
        subset=[
            "last",
            "open24h",
            "price_change_24h_percent",
            "volume_usd_24h_est"
        ]
    ).copy()

    return df


def prefilter_candidates(df_market):
    df = df_market.copy()

    df = df[
        (df["price_change_24h_percent"] >= MIN_PRICE_CHANGE_24H) &
        (df["volume_usd_24h_est"] >= MIN_VOLUME_USD_24H)
    ].copy()

    df = df.sort_values(
        by=[
            "price_change_24h_percent",
            "volume_usd_24h_est"
        ],
        ascending=[
            False,
            False
        ]
    ).reset_index(drop=True)

    return df.head(PRE_FILTER_TOP_N)


def classify_signal(rsi_1h, rsi_4h, exact_volume_24h, price_change_24h):
    volume_ok = (
        exact_volume_24h is not None and
        exact_volume_24h >= MIN_VOLUME_USD_24H
    )

    price_change_ok = price_change_24h >= MIN_PRICE_CHANGE_24H

    if not volume_ok or not price_change_ok:
        return {
            "signal_level": "NO_SIGNAL",
            "volume_condition": volume_ok,
            "price_change_condition": price_change_ok,
            "rsi_4h_condition": False,
            "rsi_1h_condition": False,
            "combined_condition": False,
            "reason": "Filters not passed"
        }

    rsi_4h_condition = rsi_4h > RSI_4H_ALERT
    rsi_4h_extreme_condition = rsi_4h > RSI_4H_EXTREME
    rsi_1h_condition = rsi_1h > RSI_1H_ALERT
    combined_condition = rsi_4h_condition and rsi_1h_condition

    if combined_condition:
        signal_level = "COMBINED_OVERHEAT"
        reason = f"RSI 1H > {RSI_1H_ALERT} and RSI 4H > {RSI_4H_ALERT}"

    elif rsi_4h_extreme_condition:
        signal_level = "EXTREME_4H"
        reason = f"RSI 4H > {RSI_4H_EXTREME}"

    elif rsi_4h_condition:
        signal_level = "STRONG_4H"
        reason = f"RSI 4H > {RSI_4H_ALERT}"

    elif rsi_1h_condition:
        signal_level = "EXTREME_1H"
        reason = f"RSI 1H > {RSI_1H_ALERT}"

    else:
        signal_level = "NO_SIGNAL"
        reason = "RSI filters not passed"

    return {
        "signal_level": signal_level,
        "volume_condition": volume_ok,
        "price_change_condition": price_change_ok,
        "rsi_4h_condition": rsi_4h_condition,
        "rsi_1h_condition": rsi_1h_condition,
        "combined_condition": combined_condition,
        "reason": reason
    }


def analyze_candidate(inst_id, ticker_row):
    raw_1h = get_candles(
        inst_id=inst_id,
        bar="1H",
        limit=CANDLE_LIMIT_1H
    )

    df_1h = candles_to_dataframe(raw_1h)
    df_1h = calculate_rsi(df_1h, period=RSI_PERIOD)

    raw_4h = get_candles(
        inst_id=inst_id,
        bar="4H",
        limit=CANDLE_LIMIT_4H
    )

    df_4h = candles_to_dataframe(raw_4h)
    df_4h = calculate_rsi(df_4h, period=RSI_PERIOD)

    last_1h = get_last_closed_candle(df_1h)
    last_4h = get_last_closed_candle(df_4h)

    exact_volume_24h = calculate_closed_24h_quote_volume_from_1h(df_1h)
    volume_change_24h = calculate_volume_change_24h_from_1h(df_1h)

    rsi_1h = float(last_1h["rsi"])
    rsi_4h = float(last_4h["rsi"])
    price = float(last_1h["close"])
    price_change_24h = float(ticker_row["price_change_24h_percent"])

    classification = classify_signal(
        rsi_1h=rsi_1h,
        rsi_4h=rsi_4h,
        exact_volume_24h=exact_volume_24h,
        price_change_24h=price_change_24h
    )

    return {
        "inst_id": inst_id,
        "symbol": make_report_symbol(inst_id),
        "price": price,

        "rsi_1h": rsi_1h,
        "rsi_4h": rsi_4h,

        "last_1h_candle_time": last_1h["timestamp"],
        "last_4h_candle_time": last_4h["timestamp"],

        "volume_usd_24h_est": float(ticker_row["volume_usd_24h_est"]),
        "volume_usd_24h_exact": exact_volume_24h,
        "volume_change_24h_percent": volume_change_24h,

        "price_change_24h_percent": price_change_24h,

        "volume_condition": classification["volume_condition"],
        "price_change_condition": classification["price_change_condition"],
        "rsi_4h_condition": classification["rsi_4h_condition"],
        "rsi_1h_condition": classification["rsi_1h_condition"],
        "combined_condition": classification["combined_condition"],

        "signal_level": classification["signal_level"],
        "reason": classification["reason"]
    }


def run_screener():
    print("Loading OKX market universe...")

    df_market = build_market_universe()
    total_universe_count = len(df_market)

    df_candidates = prefilter_candidates(df_market)
    prefiltered_count = len(df_candidates)

    results = []

    for index, row in df_candidates.iterrows():
        inst_id = row["instId"]

        if VERBOSE_PROGRESS:
            print(f"[{index + 1}/{len(df_candidates)}] Analyzing {inst_id}...")

        try:
            result = analyze_candidate(inst_id, row)
            results.append(result)

        except Exception as e:
            print(f"Error while analyzing {inst_id}: {e}")

        time.sleep(REQUEST_DELAY_SECONDS)

    df_results = pd.DataFrame(results)

    if df_results.empty:
        return df_results, total_universe_count, prefiltered_count, 0

    signal_rank = {
        "COMBINED_OVERHEAT": 4,
        "EXTREME_4H": 3,
        "STRONG_4H": 2,
        "EXTREME_1H": 1,
        "NO_SIGNAL": 0
    }

    df_results["signal_rank"] = df_results["signal_level"].map(signal_rank)

    df_results = df_results.sort_values(
        by=[
            "signal_rank",
            "rsi_4h",
            "rsi_1h",
            "price_change_24h_percent",
            "volume_usd_24h_exact"
        ],
        ascending=[
            False,
            False,
            False,
            False,
            False
        ]
    ).reset_index(drop=True)

    active_count = len(df_results[df_results["signal_level"] != "NO_SIGNAL"])

    return df_results, total_universe_count, prefiltered_count, active_count


# ============================================================
# TELEGRAM FUNCTIONS
# ============================================================

def get_telegram_credentials():
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    return bot_token, chat_id


def validate_telegram_config():
    if not TELEGRAM_ENABLED:
        print("Telegram disabled.")
        return None, None

    bot_token, chat_id = get_telegram_credentials()

    if not bot_token:
        raise Exception("TELEGRAM_BOT_TOKEN is missing. Add it to GitHub Secrets.")

    if not chat_id:
        raise Exception("TELEGRAM_CHAT_ID is missing. Add it to GitHub Secrets.")

    return bot_token, chat_id


def telegram_signal_label(signal_level):
    labels = {
        "COMBINED_OVERHEAT": "🔴🔴 COMBINED OVERHEAT",
        "EXTREME_4H": "🔴 EXTREME 4H",
        "STRONG_4H": "🟠 STRONG 4H",
        "EXTREME_1H": "🟡 EXTREME 1H",
        "NO_SIGNAL": "⚪ NO SIGNAL"
    }

    return labels.get(signal_level, signal_level)


def telegram_signal_badges(row):
    badges = []

    chg_24h = parse_float_from_value(row.get("chg_24h_%"))
    vol_chg = parse_float_from_value(row.get("vol_chg_24h_%"))

    if chg_24h is not None and chg_24h >= 20:
        badges.append("🚀 Strong pump")

    if vol_chg is not None and vol_chg >= 100:
        badges.append("🔥 Volume spike")

    if vol_chg is not None and vol_chg < 0:
        badges.append("⚠️ Volume fading")

    return badges


def send_telegram_message(text):
    if not TELEGRAM_ENABLED:
        print("Telegram disabled.")
        return

    bot_token, chat_id = validate_telegram_config()

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }

    response = SESSION.post(url, json=payload, timeout=20)

    if response.status_code != 200:
        print("Telegram error response:")
        print(response.text)
        raise Exception(f"Telegram sendMessage error: {response.status_code}")

    print("Telegram message sent.")


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


def send_telegram_message_safe(text):
    parts = split_long_message(text)

    for index, part in enumerate(parts):
        if len(parts) > 1:
            header = f"Part {index + 1}/{len(parts)}\n\n"
            send_telegram_message(header + part)
        else:
            send_telegram_message(part)

        time.sleep(1)


def format_active_signals_for_telegram(
    df_active_output,
    total_universe_count,
    prefiltered_count,
    active_count
):
    now_kyiv = datetime.now(KYIV_TZ).strftime("%Y-%m-%d %H:%M Kyiv")

    lines = []

    lines.append("🚨 <b>OKX RSI Screener</b>")
    lines.append(f"🕒 <code>{html.escape(now_kyiv)}</code>")
    lines.append("")

    if df_active_output is None or df_active_output.empty:
        lines.append("✅ Активних сигналів немає.")
        return "\n".join(lines)

    for idx, row in df_active_output.iterrows():
        signal_label = telegram_signal_label(row["signal_level"])

        symbol = html.escape(str(row["symbol"]))
        price = html.escape(str(row["price"]))
        rsi_1h = html.escape(str(row["rsi_1h"]))
        rsi_4h = html.escape(str(row["rsi_4h"]))
        chg_24h = html.escape(str(row["chg_24h_%"]))
        vol_24h = html.escape(str(row["vol_24h"]))
        vol_chg = html.escape(str(row["vol_chg_24h_%"]))
        reason = html.escape(str(row["reason"]))

        lines.append(f"{idx + 1}) {signal_label}")
        lines.append(f"📌 <b>{symbol}</b>")
        lines.append(f"💵 Price: <code>{price}</code>")
        lines.append(f"📊 RSI 1H / 4H: <code>{rsi_1h} / {rsi_4h}</code>")
        lines.append(f"📈 24h: <code>{chg_24h}%</code>")
        lines.append(f"💰 Vol: <code>{vol_24h}</code>")
        lines.append(f"🔥 Vol chg: <code>{vol_chg}%</code>")

        badges = telegram_signal_badges(row)

        if badges:
            lines.append("")
            for badge in badges:
                lines.append(badge)

        lines.append("")
        lines.append(f"Reason: <i>{reason}</i>")

        if idx != len(df_active_output) - 1:
            lines.append("")
            lines.append("────────────")
            lines.append("")

    return "\n".join(lines)


def run_screener_once_and_send_telegram():
    started_at = datetime.now(KYIV_TZ)

    print("\n" + "=" * 120)
    print("RUNNING GITHUB ACTIONS SCREENER")
    print("=" * 120)
    print("Started at:", started_at.strftime("%Y-%m-%d %H:%M:%S Kyiv"))

    df_results, total_universe_count, prefiltered_count, active_count = run_screener()

    if df_results is not None and not df_results.empty:
        df_active = df_results[df_results["signal_level"] != "NO_SIGNAL"].copy()
    else:
        df_active = pd.DataFrame()

    if df_active.empty:
        print("No active signals.")

        if SEND_MESSAGE_IF_NO_SIGNALS:
            message = format_active_signals_for_telegram(
                df_active_output=pd.DataFrame(),
                total_universe_count=total_universe_count,
                prefiltered_count=prefiltered_count,
                active_count=active_count
            )

            send_telegram_message_safe(message)

        return df_results

    df_active_output = prepare_active_output_table(df_active)

    print_df(
        df_active_output,
        title="ACTIVE SIGNALS ONLY",
        max_rows=FINAL_TOP_N
    )

    message = format_active_signals_for_telegram(
        df_active_output=df_active_output,
        total_universe_count=total_universe_count,
        prefiltered_count=prefiltered_count,
        active_count=active_count
    )

    send_telegram_message_safe(message)

    finished_at = datetime.now(KYIV_TZ)
    print("Finished at:", finished_at.strftime("%Y-%m-%d %H:%M:%S Kyiv"))

    return df_results


def main():
    validate_telegram_config()
    run_screener_once_and_send_telegram()


if __name__ == "__main__":
    main()
