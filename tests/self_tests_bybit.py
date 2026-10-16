import os
import pandas as pd
import numpy as np

from screener.config import SCRIPT_VERSION
from screener.oi_context import normalize_oi_history_rows
from screener.bybit import (
    bybit_analyze_candidate,
    bybit_build_market_universe,
    bybit_candles_to_dataframe,
    bybit_closed_24h_quote_volume,
    bybit_get,
    bybit_get_last_closed_candle,
    bybit_get_last_live_candle,
    bybit_instruments_to_dataframe,
    bybit_prefilter_candidates,
    bybit_tickers_to_dataframe,
    bybit_volume_change_24h,
    get_bybit_proxy_config,
    redact_proxy_details,
    run_bybit_screener,
)
from screener.report import format_telegram_report
from screener.runner import (
    classify_data_quality,
    format_data_warning_line,
    run_screener,
)


def run_bybit_self_tests():
    print("RUNNING BYBIT SELF TESTS")
    print("SCRIPT VERSION:", SCRIPT_VERSION)

    import contextlib
    import io

    @contextlib.contextmanager
    def patched(**replacements):
        # Each replaced name must be patched in the module whose functions read it.
        import screener.bybit as bybit_module
        import screener.common as common_module
        import screener.runner as runner_module
        owner_modules = {
            "SESSION": common_module,
            "bybit_get_candles": bybit_module,
            "bybit_get_open_interest_history": bybit_module,
            "bybit_get_instruments": bybit_module,
            "bybit_get_tickers": bybit_module,
            "bybit_build_market_universe": bybit_module,
            "bybit_prefilter_candidates": bybit_module,
            "bybit_analyze_candidate": bybit_module,
            "REQUEST_DELAY_SECONDS": bybit_module,
            "run_bybit_screener": runner_module,
            "send_telegram_message_safe": runner_module,
        }

        targets = {}
        for name in replacements:
            if name not in owner_modules:
                raise KeyError(
                    f"patched(): no owner module configured for '{name}'. "
                    "Add it to owner_modules in run_bybit_self_tests."
                )
            targets[name] = owner_modules[name]

        saved = {name: getattr(targets[name], name) for name in replacements}
        for name, value in replacements.items():
            setattr(targets[name], name, value)
        try:
            yield
        finally:
            for name, value in saved.items():
                setattr(targets[name], name, value)

    @contextlib.contextmanager
    def proxy_env(url="https://proxy.example.com/", secret="test-secret-value"):
        names = ["BYBIT_PROXY_URL", "BYBIT_PROXY_SECRET"]
        saved = {name: os.environ.get(name) for name in names}
        for name, value in zip(names, [url, secret]):
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        try:
            yield
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    class FakeResponse:
        def __init__(self, payload, status_code=200):
            self.payload = payload
            self.status_code = status_code
            self.text = str(payload)

        def json(self):
            return self.payload

    class FakeSession:
        def __init__(self, payload, status_code=200):
            self.payload = payload
            self.status_code = status_code
            self.calls = []

        def get(self, url, params=None, headers=None, timeout=None):
            self.calls.append({"url": url, "params": params, "headers": headers})
            return FakeResponse(self.payload, self.status_code)

    def quiet(func, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return func(*args, **kwargs)

    def raises(func, *args, **kwargs):
        try:
            quiet(func, *args, **kwargs)
        except BaseException as exc:
            return exc
        return None

    # Bybit kline rows: newest first, strings, [start ms, o, h, l, c, volume, turnover].
    kline_rows = [
        ["1781532000000", "10.5", "11.0", "10.0", "10.8", "100", "1080"],
        ["1781528400000", "10.0", "10.6", "9.9", "10.5", "90", "950"],
        ["1781524800000", "9.8", "10.1", "9.7", "10.0", "80", "790"],
    ]
    df_k = bybit_candles_to_dataframe(kline_rows)

    # Synthetic API payloads for the full candidate pipeline.
    def to_bybit_rows(df):
        rows = []
        for _, r in df.iloc[::-1].iterrows():
            rows.append([
                str(int(r["timestamp"].timestamp() * 1000)),
                str(r["open"]), str(r["high"]), str(r["low"]), str(r["close"]),
                str(r["volume"]), str(r["quote_volume"]),
            ])
        return rows

    rng = np.random.default_rng(7)
    n = 400
    closes = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.004, size=n)))
    closes[-30:] *= np.linspace(1.0, 1.25, 30)
    base_df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="1h"),
        "open": np.concatenate([[100.0], closes[:-1]]),
        "high": closes * 1.004,
        "low": closes * 0.996,
        "close": closes,
        "volume": 1000.0,
        "quote_volume": 1000.0 * closes,
    })
    market_1h = base_df
    market_4h = base_df.iloc[::4].reset_index(drop=True)
    market_1d = base_df.iloc[::24].reset_index(drop=True)
    market_oi = [
        {"openInterest": str(5000 + i), "timestamp": str(int(ts.timestamp() * 1000))}
        for i, ts in list(enumerate(base_df["timestamp"].tail(200)))[::-1]
    ]

    def fake_candles(symbol, interval, limit):
        source = {"60": market_1h, "240": market_4h, "D": market_1d}[interval]
        return to_bybit_rows(source)

    def fake_oi_history(symbol):
        return normalize_oi_history_rows(market_oi)

    ticker_row = pd.Series({"symbol": "TESTUSDT", "price_change_24h_percent": 25.0})

    with patched(bybit_get_candles=fake_candles, bybit_get_open_interest_history=fake_oi_history):
        candidate = quiet(bybit_analyze_candidate, "TESTUSDT", ticker_row)

    instruments = [
        {"symbol": "AAAUSDT", "contractType": "LinearPerpetual", "quoteCoin": "USDT", "status": "Trading"},
        {"symbol": "BBBUSDT", "contractType": "LinearFutures", "quoteCoin": "USDT", "status": "Trading"},
        {"symbol": "CCCUSDC", "contractType": "LinearPerpetual", "quoteCoin": "USDC", "status": "Trading"},
        {"symbol": "DDDUSDT", "contractType": "LinearPerpetual", "quoteCoin": "USDT", "status": "Closed"},
        {"symbol": "EEEUSDT", "contractType": "LinearPerpetual", "quoteCoin": "USDT", "status": "Trading"},
    ]
    tickers = [
        {"symbol": "AAAUSDT", "lastPrice": "2.5", "price24hPcnt": "0.1273", "turnover24h": "74340000"},
        {"symbol": "EEEUSDT", "lastPrice": "1.0", "price24hPcnt": "0.05", "turnover24h": "90000000"},
        {"symbol": "BBBUSDT", "lastPrice": "1.0", "price24hPcnt": "0.50", "turnover24h": "90000000"},
    ]
    df_inst = bybit_instruments_to_dataframe(instruments)
    df_tick = bybit_tickers_to_dataframe(tickers)

    with patched(bybit_get_instruments=lambda: instruments, bybit_get_tickers=lambda: tickers):
        df_universe = bybit_build_market_universe()
        df_pre = bybit_prefilter_candidates(df_universe)

    # Prefilter on a hand-made frame: growth is only a sort key, volume is the filter.
    def make_pre(rows):
        return pd.DataFrame([
            {"symbol": s, "price": 1.0, "price_change_24h_percent": ch, "volume_usd_24h_est": vol}
            for s, ch, vol in rows
        ])

    pre_small = bybit_prefilter_candidates(make_pre([
        ("LOWUSDT", 3.0, 6_000_000.0),
        ("NEGUSDT", -2.0, 6_000_000.0),
        ("THINUSDT", 50.0, 4_990_000.0),
        ("EDGEUSDT", 1.0, 5_000_000.0),
        ("TIEAUSDT", 7.0, 8_000_000.0),
        ("TIEBUSDT", 7.0, 9_000_000.0),
    ]))
    pre_many = bybit_prefilter_candidates(make_pre(
        [(f"C{i:02d}USDT", float(i), 6_000_000.0) for i in range(45)]
    ))

    # Data quality / error accounting.
    def run_with_errors(failing_symbols):
        candidates = pd.DataFrame({
            "symbol": [f"S{i}USDT" for i in range(10)],
            "price": 1.0,
            "price_change_24h_percent": 20.0,
            "volume_usd_24h_est": 10_000_000.0,
        })

        def fake_analyze(symbol, row):
            if symbol in failing_symbols:
                raise Exception("boom")
            return {"signal_level": "NO_SIGNAL", "symbol": symbol}

        with patched(
            bybit_build_market_universe=lambda: candidates,
            bybit_prefilter_candidates=lambda df: df,
            bybit_analyze_candidate=fake_analyze,
            REQUEST_DELAY_SECONDS=0,
        ):
            return quiet(run_bybit_screener)

    _, stats_none = run_with_errors(set())
    _, stats_three = run_with_errors({"S0USDT", "S1USDT", "S2USDT"})

    sent_messages = []

    def fake_send(text):
        sent_messages.append(text)
        return True

    def run_screener_with(result=None, error=None):
        sent_messages.clear()

        def fake_run():
            if error is not None:
                raise error
            return result

        with patched(run_bybit_screener=fake_run, send_telegram_message_safe=fake_send):
            return raises(run_screener)

    def stats(analyzed, errors):
        return {"total_universe": 100, "prefiltered": analyzed + errors, "analyzed": analyzed, "errors": errors, "active": 0}

    exit_incomplete = run_screener_with(result=(pd.DataFrame(), stats(4, 6)))
    incomplete_messages = list(sent_messages)
    exit_fetch_error = run_screener_with(error=Exception("Bybit HTTP error: 403"))
    fetch_error_messages = list(sent_messages)
    exit_partial = run_screener_with(result=(pd.DataFrame(), stats(8, 2)))
    partial_messages = list(sent_messages)
    exit_ok = run_screener_with(result=(pd.DataFrame(), stats(10, 0)))
    ok_messages = list(sent_messages)

    def call_bybit_with(session):
        with patched(SESSION=session), proxy_env():
            return bybit_get("/v5/market/kline", {"category": "linear", "symbol": "XUSDT"}, label="Bybit kline")

    ok_session = FakeSession({"retCode": 0, "retMsg": "OK", "result": {"list": [1]}})
    call_bybit_with(ok_session)

    with proxy_env():
        configured_url, configured_secret = get_bybit_proxy_config()
    with proxy_env(url="https://my-worker.example.workers.dev", secret="s3cret-xyz"):
        redacted_text = redact_proxy_details(
            "failed https://my-worker.example.workers.dev/bybit/v5 host=my-worker.example.workers.dev token=s3cret-xyz"
        )
    with proxy_env(url=None, secret=None):
        missing_both = raises(get_bybit_proxy_config)
    with proxy_env(url="https://leak-check.example.com", secret=None):
        missing_secret = raises(get_bybit_proxy_config)

    tests = [
        # --- configuration and requests
        ("config: both variables missing -> clear error", isinstance(missing_both, RuntimeError)
            and "BYBIT_PROXY_URL" in str(missing_both) and "BYBIT_PROXY_SECRET" in str(missing_both)),
        ("config: error does not show configured values", isinstance(missing_secret, RuntimeError)
            and "leak-check" not in str(missing_secret) and "BYBIT_PROXY_SECRET" in str(missing_secret)),
        ("config: values are read, trailing slash is removed", configured_url == "https://proxy.example.com" and configured_secret == "test-secret-value"),
        ("request: URL goes through proxy /bybit prefix", ok_session.calls[0]["url"] == "https://proxy.example.com/bybit/v5/market/kline"),
        ("request: X-Proxy-Secret header is sent", ok_session.calls[0]["headers"] == {"X-Proxy-Secret": "test-secret-value"}),
        ("request: query params are passed", ok_session.calls[0]["params"] == {"category": "linear", "symbol": "XUSDT"}),
        ("request: retCode != 0 raises", raises(call_bybit_with, FakeSession({"retCode": 10001, "retMsg": "bad", "result": {}})) is not None),
        ("request: HTTP 401 raises", raises(call_bybit_with, FakeSession({"error": "unauthorized"}, status_code=401)) is not None),
        ("redaction: URL, host and secret are hidden in log text", redacted_text == "failed ***/bybit/v5 host=*** token=***"),
        ("request: errors never contain the secret", "test-secret-value" not in str(raises(call_bybit_with, FakeSession({"retCode": 10001, "retMsg": "bad", "result": {}})))),

        # --- candles
        ("candles: sorted oldest first", list(df_k["close"]) == [10.0, 10.5, 10.8]),
        ("candles: columns", list(df_k.columns) == ["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]),
        ("candles: no confirm column", "confirm" not in df_k.columns),
        ("candles: quote_volume is turnover", list(df_k["quote_volume"]) == [790.0, 950.0, 1080.0]),
        ("candles: numeric dtypes", all(str(df_k[c].dtype).startswith("float") for c in ["open", "high", "low", "close", "volume", "quote_volume"])),
        ("candles: timestamp from ms", df_k["timestamp"].iloc[0] == pd.to_datetime(1781524800000, unit="ms")),
        ("candles: last row is live", float(bybit_get_last_live_candle(df_k)["close"]) == 10.8),
        ("candles: previous row is last closed", float(bybit_get_last_closed_candle(df_k)["close"]) == 10.5),
        ("candles: too few columns raises", raises(bybit_candles_to_dataframe, [["1781532000000", "1", "2", "3", "4", "5"]]) is not None),
        ("candles: empty rows give empty dataframe", bybit_candles_to_dataframe([]).empty),
        ("volume: 24h sum excludes live candle", bybit_closed_24h_quote_volume(pd.DataFrame({"quote_volume": [1.0] * 24 + [999.0]})) == 24.0),
        ("volume: change vs previous 24h", abs(bybit_volume_change_24h(pd.DataFrame({"quote_volume": [1.0] * 24 + [2.0] * 24 + [999.0]})) - 100.0) < 1e-9),
        ("volume: not enough candles -> None", bybit_closed_24h_quote_volume(pd.DataFrame({"quote_volume": [1.0] * 10})) is None),

        # --- universe
        ("instruments: only LinearPerpetual USDT Trading", list(df_inst["symbol"]) == ["AAAUSDT", "EEEUSDT"]),
        ("tickers: 24h change ratio -> percent", abs(float(df_tick.loc[df_tick["symbol"] == "AAAUSDT", "price_change_24h_percent"].iloc[0]) - 12.73) < 1e-9),
        ("tickers: turnover is USDT volume", float(df_tick.loc[df_tick["symbol"] == "AAAUSDT", "volume_usd_24h_est"].iloc[0]) == 74340000.0),
        ("universe: dated futures are excluded", "BBBUSDT" not in list(df_universe["symbol"])),
        ("prefilter: volume >= 5M only, sorted by growth desc", list(df_pre["symbol"]) == ["AAAUSDT", "EEEUSDT"]),
        ("prefilter: +3% with volume >= 5M passes", "LOWUSDT" in list(pre_small["symbol"])),
        ("prefilter: -2% with volume >= 5M passes", "NEGUSDT" in list(pre_small["symbol"])),
        ("prefilter: volume 4.99M fails even with +50%", "THINUSDT" not in list(pre_small["symbol"])),
        ("prefilter: volume exactly 5M passes", "EDGEUSDT" in list(pre_small["symbol"])),
        ("prefilter: sort by growth desc, then volume desc",
            list(pre_small["symbol"]) == ["TIEBUSDT", "TIEAUSDT", "LOWUSDT", "EDGEUSDT", "NEGUSDT"]),
        ("prefilter: PRE_FILTER_TOP_N limit applies (45 -> 40)",
            len(pre_many) == 40 and pre_many["symbol"].iloc[0] == "C44USDT" and pre_many["symbol"].iloc[-1] == "C05USDT"),

        # --- full candidate pipeline on fake API data
        ("candidate: exchange and provider are Bybit", candidate["exchange"] == "Bybit" and candidate["provider"] == "Bybit"),
        ("candidate: report symbol has .P suffix", candidate["symbol"] == "TESTUSDT.P"),
        ("candidate: OI source is Bybit with history", candidate["oi_source"] == "Bybit" and candidate["oi_history_points"] == 200),
        ("candidate: all five factors analyzed", candidate["total_short_factors_count"] == 5),
        ("candidate: price change comes from ticker", candidate["price_change_24h_percent"] == 25.0),

        # --- 2.5 data quality
        ("quality: no errors -> ok", classify_data_quality(10, 0) == "ok"),
        ("quality: no candidates -> ok", classify_data_quality(0, 0) == "ok"),
        ("quality: 1 of 10 failed -> partial", classify_data_quality(10, 1) == "partial"),
        ("quality: exactly half failed -> partial", classify_data_quality(10, 5) == "partial"),
        ("quality: more than half failed -> incomplete", classify_data_quality(10, 6) == "incomplete"),
        ("quality: all failed -> incomplete", classify_data_quality(5, 5) == "incomplete"),
        ("runner: no errors counted", stats_none["errors"] == 0 and stats_none["analyzed"] == 10),
        ("runner: errors are counted per symbol", stats_three["errors"] == 3 and stats_three["analyzed"] == 7),
        ("warning line text", format_data_warning_line(7, 10, 3) == "⚠️ Проаналізовано 7 з 10 кандидатів, 3 помилок"),
        ("report: warning line is shown", "⚠️ Проаналізовано 7 з 10 кандидатів, 3 помилок" in format_telegram_report([], data_warning=format_data_warning_line(7, 10, 3))),
        ("report: no warning without errors", "⚠️" not in format_telegram_report([])),
        ("run: >half failed -> exit code 1", isinstance(exit_incomplete, SystemExit) and exit_incomplete.code == 1),
        ("run: >half failed -> 'data incomplete' message, not 'No signals'",
            len(incomplete_messages) == 1 and "Дані неповні" in incomplete_messages[0] and "No overheat" not in incomplete_messages[0]),
        ("run: market data failure -> exit code 1 and warning",
            isinstance(exit_fetch_error, SystemExit) and exit_fetch_error.code == 1
            and len(fetch_error_messages) == 1 and "список монет" in fetch_error_messages[0]),
        ("run: some errors -> normal report with warning line, exit 0",
            exit_partial is None and len(partial_messages) == 1
            and "Проаналізовано 8 з 10 кандидатів, 2 помилок" in partial_messages[0]
            and "No overheat or short-watch signals" in partial_messages[0]),
        ("run: no errors -> normal report without warning",
            exit_ok is None and len(ok_messages) == 1 and "⚠️" not in ok_messages[0]),
    ]

    failed = 0
    for name, passed in tests:
        status = "PASS" if passed else "FAIL"
        print(f"{status} | {name}")

        if not passed:
            failed += 1

    if failed:
        raise AssertionError(f"Bybit self-tests failed: {failed}/{len(tests)}")

    print(f"BYBIT SELF TESTS PASSED: {len(tests)}/{len(tests)}")


def run_tradfi_filter_self_tests():
    print("\n" + "=" * 120)
    print("RUNNING TRADFI FILTER SELF TESTS")
    print("SCRIPT VERSION:", SCRIPT_VERSION)
    print("=" * 120)

    import contextlib
    import io
    import screener.bybit as bybit_module

    def inst(symbol, symbol_type="", **overrides):
        row = {"symbol": symbol, "contractType": "LinearPerpetual", "quoteCoin": "USDT", "status": "Trading"}
        if symbol_type is not None:
            row["symbolType"] = symbol_type
        row.update(overrides)
        return row

    def convert(instruments):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            df = bybit_instruments_to_dataframe(instruments)
        return df, buffer.getvalue()

    def kept(instruments):
        df, _ = convert(instruments)
        return sorted(df["symbol"].tolist())

    row_missing_key = inst("MISSINGUSDT", symbol_type=None)  # column exists elsewhere, key absent here
    row_null = inst("NULLUSDT", symbol_type=None)
    row_null["symbolType"] = None

    mixed = [
        inst("BTCUSDT", ""),
        inst("SOXSUSDT", "ETF"),
        inst("BSPUSDT", "stock"),
        inst("XAUUSDT", "commodity"),
        inst("NEWUSDT", "index"),
        row_missing_key,
        row_null,
        inst("SPACEUSDT", "  "),
        inst("TRADFICLOSEDUSDT", "stock", status="Closed"),
        inst("CRYPTOFUTURESUSDT", "", contractType="LinearFutures"),
    ]

    # No symbolType column at all (API change): nothing is dropped by type, one warning line.
    no_column = [
        {"symbol": "AAAUSDT", "contractType": "LinearPerpetual", "quoteCoin": "USDT", "status": "Trading"},
        {"symbol": "BBBUSDT", "contractType": "LinearPerpetual", "quoteCoin": "USDT", "status": "Trading"},
    ]
    df_no_column, log_no_column = convert(no_column)
    warning = "WARNING: Bybit instruments response has no symbolType column; TradFi filter skipped."
    _, log_with_column = convert(mixed)

    # End to end: same numbers for a TradFi and a crypto symbol.
    instruments = [inst("TRADFIUSDT", "stock"), inst("CRYPTOUSDT", "")]
    tickers = [
        {"symbol": "TRADFIUSDT", "lastPrice": "10", "price24hPcnt": "0.30", "turnover24h": "90000000"},
        {"symbol": "CRYPTOUSDT", "lastPrice": "10", "price24hPcnt": "0.30", "turnover24h": "90000000"},
    ]
    saved = (bybit_module.bybit_get_instruments, bybit_module.bybit_get_tickers)
    bybit_module.bybit_get_instruments = lambda: instruments
    bybit_module.bybit_get_tickers = lambda: tickers
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            df_universe = bybit_build_market_universe()
            df_pre = bybit_prefilter_candidates(df_universe)
    finally:
        bybit_module.bybit_get_instruments, bybit_module.bybit_get_tickers = saved

    tests = [
        ("empty symbolType is kept", "BTCUSDT" in kept(mixed)),
        ("stock / ETF / commodity / new type are dropped",
            not {"SOXSUSDT", "BSPUSDT", "XAUUSDT", "NEWUSDT"} & set(kept(mixed))),
        ("null or missing symbolType in a row (column present) is kept as crypto",
            {"MISSINGUSDT", "NULLUSDT"} <= set(kept(mixed))),
        ("whitespace-only symbolType is kept", "SPACEUSDT" in kept(mixed)),
        ("TradFi with status Closed is dropped", "TRADFICLOSEDUSDT" not in kept(mixed)),
        ("crypto symbolType but LinearFutures is dropped", "CRYPTOFUTURESUSDT" not in kept(mixed)),
        ("exact kept set", kept(mixed) == ["BTCUSDT", "MISSINGUSDT", "NULLUSDT", "SPACEUSDT"]),
        ("no warning when the column exists", warning not in log_with_column),
        ("no symbolType column: nothing dropped, no exception", df_no_column["symbol"].tolist() == ["AAAUSDT", "BBBUSDT"]),
        ("no symbolType column: exactly one warning line", log_no_column.count(warning) == 1 and log_no_column.strip() == warning),
        ("empty instruments still returns an empty frame", convert([])[0].empty),
        ("universe: TradFi symbol is not in df_universe", df_universe["symbol"].tolist() == ["CRYPTOUSDT"]),
        ("prefilter: TradFi symbol is not a candidate, crypto is", df_pre["symbol"].tolist() == ["CRYPTOUSDT"]),
    ]

    failed = 0
    for name, passed in tests:
        print(f"{'PASS' if passed else 'FAIL'} | {name}")
        if not passed:
            failed += 1

    if failed:
        raise AssertionError(f"TradFi filter self-tests failed: {failed}/{len(tests)}")

    print(f"TRADFI FILTER SELF TESTS PASSED: {len(tests)}/{len(tests)}")
