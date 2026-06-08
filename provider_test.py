import json
import requests


def print_separator():
    print("=" * 100)


def safe_get(url, params=None, title=None):
    if params is None:
        params = {}

    print_separator()

    if title:
        print(title)

    print("Request URL:", url)
    print("Params:", params)

    try:
        response = requests.get(url, params=params, timeout=20)

        print("Final URL:", response.url)
        print("Status code:", response.status_code)
        print("Content-Type:", response.headers.get("Content-Type"))

        text_preview = response.text[:1500]

        if response.status_code != 200:
            print("HTTP error response preview:")
            print(text_preview)
            return None

        try:
            data = response.json()
        except Exception as e:
            print("JSON parse error:")
            print(e)
            print("Response preview:")
            print(text_preview)
            return None

        print("JSON response preview:")
        print(json.dumps(data, indent=2, ensure_ascii=False)[:2500])

        return data

    except Exception as e:
        print("Request failed:")
        print(e)
        return None


# ============================================================
# BITGET TESTS
# ============================================================

def test_bitget():
    print_separator()
    print("TESTING BITGET API")
    print_separator()

    base_url = "https://api.bitget.com"

    # 1. Bitget USDT futures contracts
    contracts = safe_get(
        url=f"{base_url}/api/v2/mix/market/contracts",
        params={
            "productType": "usdt-futures"
        },
        title="BITGET: USDT futures contracts"
    )

    if contracts:
        data = contracts.get("data", [])
        print_separator()
        print("BITGET CONTRACTS SUMMARY")
        print("Contracts count:", len(data))

        sample_symbols = [item.get("symbol") for item in data[:10]]
        print("Sample symbols:", sample_symbols)

        velvet_matches = [
            item for item in data
            if str(item.get("symbol", "")).upper() == "VELVETUSDT"
        ]

        if velvet_matches:
            print("VELVETUSDT FOUND on Bitget USDT futures.")
            print(json.dumps(velvet_matches[0], indent=2, ensure_ascii=False)[:2000])
        else:
            print("VELVETUSDT NOT FOUND on Bitget USDT futures contracts list.")

    # 2. Bitget all tickers
    safe_get(
        url=f"{base_url}/api/v2/mix/market/tickers",
        params={
            "productType": "usdt-futures"
        },
        title="BITGET: all USDT futures tickers"
    )

    # 3. BTCUSDT ticker
    safe_get(
        url=f"{base_url}/api/v2/mix/market/ticker",
        params={
            "productType": "usdt-futures",
            "symbol": "BTCUSDT"
        },
        title="BITGET: BTCUSDT ticker"
    )

    # 4. BTCUSDT 1H candles
    safe_get(
        url=f"{base_url}/api/v2/mix/market/candles",
        params={
            "productType": "usdt-futures",
            "symbol": "BTCUSDT",
            "granularity": "1H",
            "limit": "10"
        },
        title="BITGET: BTCUSDT 1H candles"
    )

    # 5. VELVETUSDT ticker
    safe_get(
        url=f"{base_url}/api/v2/mix/market/ticker",
        params={
            "productType": "usdt-futures",
            "symbol": "VELVETUSDT"
        },
        title="BITGET: VELVETUSDT ticker"
    )

    # 6. VELVETUSDT 4H candles
    safe_get(
        url=f"{base_url}/api/v2/mix/market/candles",
        params={
            "productType": "usdt-futures",
            "symbol": "VELVETUSDT",
            "granularity": "4H",
            "limit": "10"
        },
        title="BITGET: VELVETUSDT 4H candles"
    )


# ============================================================
# KRAKEN TESTS
# ============================================================

def test_kraken():
    print_separator()
    print("TESTING KRAKEN API")
    print_separator()

    # Kraken Futures API
    futures_base = "https://futures.kraken.com"

    # 1. Kraken futures instruments
    futures_instruments = safe_get(
        url=f"{futures_base}/derivatives/api/v3/instruments",
        params={},
        title="KRAKEN FUTURES: instruments"
    )

    if futures_instruments:
        instruments = futures_instruments.get("instruments", [])
        print_separator()
        print("KRAKEN FUTURES INSTRUMENTS SUMMARY")
        print("Instruments count:", len(instruments))

        sample_symbols = [item.get("symbol") for item in instruments[:10]]
        print("Sample symbols:", sample_symbols)

        velvet_matches = [
            item for item in instruments
            if "VELVET" in str(item.get("symbol", "")).upper()
        ]

        if velvet_matches:
            print("VELVET FOUND on Kraken Futures.")
            print(json.dumps(velvet_matches[:3], indent=2, ensure_ascii=False)[:2500])
        else:
            print("VELVET NOT FOUND on Kraken Futures instruments list.")

    # 2. Kraken futures tickers
    safe_get(
        url=f"{futures_base}/derivatives/api/v3/tickers",
        params={},
        title="KRAKEN FUTURES: tickers"
    )

    # 3. Kraken futures BTC perpetual candles
    # Typical Kraken futures BTC perpetual symbol: PI_XBTUSD
    safe_get(
        url=f"{futures_base}/api/charts/v1/trade/PI_XBTUSD/1h",
        params={},
        title="KRAKEN FUTURES: PI_XBTUSD 1H candles"
    )

    # Kraken Spot API
    spot_base = "https://api.kraken.com"

    # 4. Kraken spot asset pairs
    spot_pairs = safe_get(
        url=f"{spot_base}/0/public/AssetPairs",
        params={},
        title="KRAKEN SPOT: asset pairs"
    )

    if spot_pairs:
        result = spot_pairs.get("result", {})
        pair_names = list(result.keys())

        print_separator()
        print("KRAKEN SPOT PAIRS SUMMARY")
        print("Spot pairs count:", len(pair_names))
        print("Sample pairs:", pair_names[:10])

        velvet_pairs = [
            pair for pair in pair_names
            if "VELVET" in pair.upper()
        ]

        if velvet_pairs:
            print("VELVET FOUND on Kraken Spot.")
            print("VELVET pairs:", velvet_pairs)
        else:
            print("VELVET NOT FOUND on Kraken Spot pairs list.")

    # 5. Kraken spot BTC/USD OHLC
    safe_get(
        url=f"{spot_base}/0/public/OHLC",
        params={
            "pair": "XBTUSD",
            "interval": "60"
        },
        title="KRAKEN SPOT: XBTUSD 1H OHLC"
    )


def main():
    print("Starting provider connectivity test from GitHub Actions...")
    test_bitget()
    test_kraken()
    print_separator()
    print("Provider test finished.")


if __name__ == "__main__":
    main()
