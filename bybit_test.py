import json
import requests


BASE_URLS = [
    "https://api.bybit.com",
    "https://api.bytick.com",
]


def print_separator():
    print("=" * 100)


def safe_request(base_url, endpoint, params):
    url = base_url + endpoint

    print_separator()
    print("Request URL:", url)
    print("Params:", params)

    try:
        response = requests.get(url, params=params, timeout=20)

        print("Final URL:", response.url)
        print("Status code:", response.status_code)
        print("Content-Type:", response.headers.get("Content-Type"))

        text_preview = response.text[:1000]

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
        print(json.dumps(data, indent=2)[:2000])

        return data

    except Exception as e:
        print("Request failed:")
        print(e)
        return None


def test_bybit_base_url(base_url):
    print_separator()
    print("TESTING BYBIT BASE URL:", base_url)
    print_separator()

    # 1. Basic instruments test
    instruments_data = safe_request(
        base_url=base_url,
        endpoint="/v5/market/instruments-info",
        params={
            "category": "linear",
            "limit": 5,
        }
    )

    # 2. BTC ticker test
    safe_request(
        base_url=base_url,
        endpoint="/v5/market/tickers",
        params={
            "category": "linear",
            "symbol": "BTCUSDT",
        }
    )

    # 3. BTC kline test
    safe_request(
        base_url=base_url,
        endpoint="/v5/market/kline",
        params={
            "category": "linear",
            "symbol": "BTCUSDT",
            "interval": "60",
            "limit": 5,
        }
    )

    # 4. Check VELVETUSDT instrument
    velvet_data = safe_request(
        base_url=base_url,
        endpoint="/v5/market/instruments-info",
        params={
            "category": "linear",
            "symbol": "VELVETUSDT",
        }
    )

    if velvet_data is not None:
        result = velvet_data.get("result", {})
        instruments = result.get("list", [])

        print_separator()
        print("VELVETUSDT CHECK RESULT")

        if instruments:
            print("VELVETUSDT FOUND on Bybit linear market.")
            print(json.dumps(instruments[0], indent=2)[:2000])
        else:
            print("VELVETUSDT NOT FOUND on Bybit linear market.")

    # 5. Check VELVETUSDT kline
    safe_request(
        base_url=base_url,
        endpoint="/v5/market/kline",
        params={
            "category": "linear",
            "symbol": "VELVETUSDT",
            "interval": "240",
            "limit": 5,
        }
    )


def main():
    print("Starting Bybit API connectivity test from GitHub Actions...")

    for base_url in BASE_URLS:
        test_bybit_base_url(base_url)

    print_separator()
    print("Bybit API test finished.")


if __name__ == "__main__":
    main()
