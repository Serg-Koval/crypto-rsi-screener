name: bybit-test
on: workflow_dispatch
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: |
          B=https://api.bybit.com/v5/market
          for p in "kline?category=linear&symbol=ATOMUSDT&interval=60&limit=3" \
                   "open-interest?category=linear&symbol=ATOMUSDT&intervalTime=1h&limit=3" \
                   "funding/history?category=linear&symbol=ATOMUSDT&limit=3" \
                   "account-ratio?category=linear&symbol=ATOMUSDT&period=1h&limit=3" \
                   "recent-trade?category=linear&symbol=ATOMUSDT&limit=3"; do
            echo "== $p"; curl -s -o /tmp/r -w "HTTP %{http_code}\n" "$B/$p"; head -c 400 /tmp/r; echo
          done
