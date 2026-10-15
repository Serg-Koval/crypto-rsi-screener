import os
import time
import pandas as pd

from urllib.parse import urlparse

from screener.common import (
    calculate_rsi,
    safe_get_json,
)
from screener.config import (
    BYBIT_CATEGORY,
    BYBIT_CONTRACT_TYPE,
    BYBIT_CRYPTO_SYMBOL_TYPE,
    BYBIT_INTERVALS,
    BYBIT_OI_HISTORY_LIMIT,
    BYBIT_OI_INTERVAL,
    BYBIT_PROXY_PATH_PREFIX,
    BYBIT_PROXY_SECRET_HEADER,
    BYBIT_QUOTE_COIN,
    BYBIT_STATUS,
    CANDLE_LIMIT_1D,
    CANDLE_LIMIT_1H,
    CANDLE_LIMIT_4H,
    MIN_PRICE_CHANGE_24H,
    MIN_VOLUME_USD_24H,
    PRE_FILTER_TOP_N,
    REQUEST_DELAY_SECONDS,
    RSI_PERIOD,
)
from screener.classification import (
    classify_signal,
    get_signal_rank,
)
from screener.factors.analysis import analyze_short_factors
from screener.oi_context import (
    build_open_interest_metrics,
    normalize_oi_history_rows,
)


def get_bybit_proxy_config():
    """Read the proxy URL and secret from the environment.

    GitHub Actions cannot reach api.bybit.com directly (HTTP 403, geo block),
    so every Bybit request goes through the Cloudflare Worker proxy. Values are
    never printed: the error message only names the missing variables.
    """

    base_url = (os.getenv("BYBIT_PROXY_URL") or "").strip().rstrip("/")
    secret = (os.getenv("BYBIT_PROXY_SECRET") or "").strip()

    missing = []

    if not base_url:
        missing.append("BYBIT_PROXY_URL")

    if not secret:
        missing.append("BYBIT_PROXY_SECRET")

    if missing:
        raise RuntimeError(
            "Bybit proxy is not configured. Missing environment variables: "
            + ", ".join(missing)
        )

    return base_url, secret


def redact_proxy_details(text):
    """Hide the proxy URL, its host and the secret in text that goes to logs.

    GitHub masks only the exact secret value, but exception messages from
    requests can contain a part of it (for example the host name).
    """

    text = str(text)
    hidden = []

    url = (os.getenv("BYBIT_PROXY_URL") or "").strip().rstrip("/")
    secret = (os.getenv("BYBIT_PROXY_SECRET") or "").strip()

    if url:
        hidden.append(url)

        try:
            parsed = urlparse(url)
            hidden.extend([parsed.netloc, parsed.hostname or ""])
        except ValueError:
            pass

    if secret:
        hidden.append(secret)

    for value in sorted({v for v in hidden if v}, key=len, reverse=True):
        text = text.replace(value, "***")

    return text


def bybit_success(data):
    return str(data.get("retCode")) == "0"


def bybit_get(endpoint, params, label="Bybit"):
    base_url, secret = get_bybit_proxy_config()

    data = safe_get_json(
        base_url=base_url + BYBIT_PROXY_PATH_PREFIX,
        endpoint=endpoint,
        params=params,
        provider_name=label,
        headers={BYBIT_PROXY_SECRET_HEADER: secret},
    )

    if not bybit_success(data):
        raise Exception(f"{label} API error: retCode={data.get('retCode')} retMsg={data.get('retMsg')}")

    return data


def bybit_result_list(data):
    result = data.get("result") or {}
    rows = result.get("list")

    return rows if isinstance(rows, list) else []


def bybit_get_instruments():
    data = bybit_get(
        "/v5/market/instruments-info",
        {"category": BYBIT_CATEGORY, "limit": 1000},
        label="Bybit instruments",
    )

    if (data.get("result") or {}).get("nextPageCursor"):
        print("WARNING: Bybit instruments-info has more pages; only the first 1000 instruments are used.")

    return bybit_result_list(data)


def bybit_get_tickers():
    data = bybit_get(
        "/v5/market/tickers",
        {"category": BYBIT_CATEGORY},
        label="Bybit tickers",
    )

    return bybit_result_list(data)


def bybit_get_candles(symbol, interval, limit):
    data = bybit_get(
        "/v5/market/kline",
        {
            "category": BYBIT_CATEGORY,
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        },
        label="Bybit kline",
    )

    return bybit_result_list(data)


def bybit_instruments_to_dataframe(instruments):
    df = pd.DataFrame(instruments)

    if df.empty:
        return df

    for column in ["symbol", "contractType", "quoteCoin", "status"]:
        if column not in df.columns:
            raise Exception(f"Bybit instruments response has no {column} column.")

    df = df[
        (df["contractType"] == BYBIT_CONTRACT_TYPE)
        & (df["quoteCoin"] == BYBIT_QUOTE_COIN)
        & (df["status"] == BYBIT_STATUS)
    ].copy()

    if "symbolType" in df.columns:
        symbol_type = df["symbolType"].fillna("").astype(str).str.strip()
        df = df[symbol_type == BYBIT_CRYPTO_SYMBOL_TYPE].copy()
    else:
        print("WARNING: Bybit instruments response has no symbolType column; TradFi filter skipped.")

    df["symbol"] = df["symbol"].astype(str)

    return df.reset_index(drop=True)


def bybit_tickers_to_dataframe(tickers):
    df = pd.DataFrame(tickers)

    if df.empty:
        return df

    for column in ["symbol", "lastPrice", "price24hPcnt", "turnover24h"]:
        if column not in df.columns:
            raise Exception(f"Bybit tickers response has no {column} column.")

    df["symbol"] = df["symbol"].astype(str)
    df["price"] = pd.to_numeric(df["lastPrice"], errors="coerce")
    # Bybit returns the 24h change as a ratio: 0.1273 = 12.73%.
    df["price_change_24h_percent"] = pd.to_numeric(df["price24hPcnt"], errors="coerce") * 100
    # turnover24h is already in USDT for linear contracts.
    df["volume_usd_24h_est"] = pd.to_numeric(df["turnover24h"], errors="coerce")

    return df[["symbol", "price", "price_change_24h_percent", "volume_usd_24h_est"]].reset_index(drop=True)


def bybit_build_market_universe():
    instruments = bybit_get_instruments()
    tickers = bybit_get_tickers()

    if not instruments:
        raise Exception("Bybit returned an empty instruments list.")

    if not tickers:
        raise Exception("Bybit returned an empty tickers list.")

    df_instruments = bybit_instruments_to_dataframe(instruments)
    df_tickers = bybit_tickers_to_dataframe(tickers)

    df = df_instruments[["symbol"]].merge(df_tickers, on="symbol", how="inner")

    df = df.dropna(
        subset=[
            "price",
            "price_change_24h_percent",
            "volume_usd_24h_est",
        ]
    ).copy()

    if df.empty:
        raise Exception("Bybit market universe is empty after merging instruments and tickers.")

    return df.reset_index(drop=True)


def bybit_prefilter_candidates(df_market):
    df = df_market.copy()

    df = df[
        (df["price_change_24h_percent"] >= MIN_PRICE_CHANGE_24H)
        & (df["volume_usd_24h_est"] >= MIN_VOLUME_USD_24H)
    ].copy()

    df = df.sort_values(
        by=["price_change_24h_percent", "volume_usd_24h_est"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return df.head(PRE_FILTER_TOP_N)


def bybit_candles_to_dataframe(rows):
    """Convert Bybit kline rows into the dataframe used by the analysis.

    Bybit returns candles from newest to oldest as
    [startTime ms, open, high, low, close, volume, turnover]. The dataframe is
    sorted oldest to newest. There is no confirm flag: the last row is the live
    candle and the previous row is the last closed candle (the analysis helpers
    already treat dataframes without a confirm column this way).
    """

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    if df.shape[1] < 7:
        raise Exception(f"Bybit: unexpected candle format. Columns: {df.shape[1]}")

    df = df.iloc[:, :7].copy()
    df.columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
    ]

    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms")
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def bybit_get_last_closed_candle(df):
    # Bybit has no confirm field: the previous candle is the last closed one.
    if df is None or df.empty:
        raise Exception("Bybit: empty candles dataframe.")

    if len(df) >= 2:
        return df.iloc[-2]

    return df.iloc[-1]


def bybit_get_last_live_candle(df):
    # Bybit has no confirm field: the latest row is the live candle.
    if df is None or df.empty:
        raise Exception("Bybit: empty candles dataframe.")

    return df.iloc[-1]


def bybit_closed_24h_quote_volume(df_1h):
    if df_1h is None or df_1h.empty or "quote_volume" not in df_1h.columns:
        return None

    closed_df = df_1h.iloc[:-1].copy()

    if len(closed_df) < 24:
        return None

    return float(closed_df.tail(24)["quote_volume"].sum())


def bybit_volume_change_24h(df_1h):
    if df_1h is None or df_1h.empty or "quote_volume" not in df_1h.columns:
        return None

    closed_df = df_1h.iloc[:-1].copy()

    if len(closed_df) < 48:
        return None

    prev_24 = closed_df.iloc[-48:-24]["quote_volume"].sum()
    last_24 = closed_df.iloc[-24:]["quote_volume"].sum()

    if prev_24 == 0:
        return None

    return float(((last_24 - prev_24) / prev_24) * 100)


def bybit_get_open_interest_history(symbol):
    data = bybit_get(
        "/v5/market/open-interest",
        {
            "category": BYBIT_CATEGORY,
            "symbol": symbol,
            "intervalTime": BYBIT_OI_INTERVAL,
            "limit": BYBIT_OI_HISTORY_LIMIT,
        },
        label="Bybit OI",
    )

    # Rows are newest first: {"openInterest": "...", "timestamp": "ms"}.
    # normalize_oi_history_rows sorts them oldest to newest.
    return normalize_oi_history_rows(bybit_result_list(data))


def bybit_get_open_interest_metrics(symbol):
    try:
        history_df = bybit_get_open_interest_history(symbol)
    except Exception as e:
        print(f"Bybit OI unavailable for {symbol}: {redact_proxy_details(e)}")
        history_df = pd.DataFrame(columns=["timestamp", "open_interest"])

    current_oi = None

    if not history_df.empty:
        current_oi = float(history_df.iloc[-1]["open_interest"])

    return build_open_interest_metrics(
        history_df=history_df,
        current_oi=current_oi,
        provider="Bybit",
    )


def bybit_analyze_candidate(symbol, ticker_row):
    raw_1h = bybit_get_candles(symbol=symbol, interval=BYBIT_INTERVALS["1H"], limit=CANDLE_LIMIT_1H)
    df_1h = bybit_candles_to_dataframe(raw_1h)
    df_1h = calculate_rsi(df_1h, period=RSI_PERIOD)

    raw_4h = bybit_get_candles(symbol=symbol, interval=BYBIT_INTERVALS["4H"], limit=CANDLE_LIMIT_4H)
    df_4h = bybit_candles_to_dataframe(raw_4h)
    df_4h = calculate_rsi(df_4h, period=RSI_PERIOD)

    raw_1d = bybit_get_candles(symbol=symbol, interval=BYBIT_INTERVALS["1D"], limit=CANDLE_LIMIT_1D)
    df_1d = bybit_candles_to_dataframe(raw_1d)

    last_1h_live = bybit_get_last_live_candle(df_1h)
    last_1h_closed = bybit_get_last_closed_candle(df_1h)
    last_4h_live = bybit_get_last_live_candle(df_4h)

    exact_volume_24h = bybit_closed_24h_quote_volume(df_1h)
    volume_change_24h = bybit_volume_change_24h(df_1h)

    rsi_1h_live = float(last_1h_live["rsi"])
    rsi_1h_closed = float(last_1h_closed["rsi"])
    rsi_4h_live = float(last_4h_live["rsi"])
    price = float(last_1h_live["close"])
    price_change_24h = float(ticker_row["price_change_24h_percent"])

    oi_metrics = bybit_get_open_interest_metrics(symbol)

    short_analysis = analyze_short_factors(
        df_1h,
        df_4h,
        df_1d=df_1d,
        current_price=price,
        oi_history=oi_metrics.get("oi_history_df"),
        provider="Bybit",
    )

    classification = classify_signal(
        rsi_1h_live=rsi_1h_live,
        rsi_1h_closed=rsi_1h_closed,
        rsi_4h_live=rsi_4h_live,
        exact_volume_24h=exact_volume_24h,
        volume_change_24h=volume_change_24h,
        price_change_24h=price_change_24h,
        short_setup_score=short_analysis["score"],
        short_factors=short_analysis["factors"],
    )

    return {
        "exchange": "Bybit",
        "provider": "Bybit",
        "raw_symbol": symbol,
        "symbol": f"{symbol}.P",
        "price": price,
        "rsi_1h_live": rsi_1h_live,
        "rsi_1h_closed": rsi_1h_closed,
        "rsi_4h_live": rsi_4h_live,
        "volume_usd_24h_exact": exact_volume_24h,
        "volume_change_24h_percent": volume_change_24h,
        "price_change_24h_percent": price_change_24h,
        "oi_current": oi_metrics.get("oi_current"),
        "oi_change_1h_percent": oi_metrics.get("oi_change_1h_percent"),
        "oi_change_4h_percent": oi_metrics.get("oi_change_4h_percent"),
        "oi_context": oi_metrics.get("oi_context", "N/A"),
        "oi_status": oi_metrics.get("oi_status", "not_available"),
        "oi_source": oi_metrics.get("oi_source", "Bybit"),
        "oi_history_points": oi_metrics.get("oi_history_points", 0),
        "oi_history_latest_ts": oi_metrics.get("oi_history_latest_ts"),
        "signal_level": classification["signal_level"],
        "reason": classification["reason"],
        "pump_score": classification["pump_score"],
        "rsi_score": classification["rsi_score"],
        "volume_score": classification["volume_score"],
        "short_setup_score": classification["short_setup_score"],
        "final_score": classification["final_score"],
        "location_score": classification.get("location_score", 0),
        "trigger_count": classification.get("trigger_count", 0),
        "setup_status": classification.get("setup_status", "N/A"),
        "short_factors": short_analysis["factors"],
        "confirmed_short_factors_count": short_analysis["confirmed_count"],
        "total_short_factors_count": short_analysis["total_count"],
    }


def run_bybit_screener():
    """Fetch the Bybit universe and analyze every prefiltered candidate.

    Returns (df_results, stats). A failure to get the instruments or tickers
    raises an exception; a failure on a single symbol is counted in
    stats["errors"] so the report can warn about incomplete data.
    """

    print("\n" + "=" * 120)
    print("RUNNING BYBIT PROVIDER")
    print("=" * 120)

    df_market = bybit_build_market_universe()
    total_universe_count = len(df_market)

    df_candidates = bybit_prefilter_candidates(df_market)
    prefiltered_count = len(df_candidates)

    print("Bybit total universe:", total_universe_count)
    print("Bybit prefiltered:", prefiltered_count)

    results = []
    error_count = 0

    for index, row in df_candidates.iterrows():
        symbol = row["symbol"]

        try:
            result = bybit_analyze_candidate(symbol, row)
            results.append(result)

        except Exception as e:
            error_count += 1
            print(f"Bybit error while analyzing {symbol}: {redact_proxy_details(e)}")

        time.sleep(REQUEST_DELAY_SECONDS)

    df_results = pd.DataFrame(results)

    stats = {
        "total_universe": total_universe_count,
        "prefiltered": prefiltered_count,
        "analyzed": len(results),
        "errors": error_count,
        "active": 0,
    }

    if df_results.empty:
        return df_results, stats

    df_results["signal_rank"] = df_results["signal_level"].apply(get_signal_rank)

    stats["active"] = len(df_results[df_results["signal_level"] != "NO_SIGNAL"])

    return df_results, stats
