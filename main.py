"""Entry point of Market Heat Scanner.

All analysis logic lives in the screener/ package; tests live in tests/.
`python main.py` runs the screener, `python main.py --self-test` runs the self-tests.
"""

import sys

from screener.bybit import get_bybit_proxy_config
from screener.telegram import (
    get_telegram_credentials,
)
from screener.runner import run_screener


def main():
    if "--self-test" in sys.argv:
        from tests.self_tests import run_all
        run_all()
        return

    get_telegram_credentials()

    try:
        get_bybit_proxy_config()
    except RuntimeError as e:
        raise SystemExit(f"ERROR: {e}")

    run_screener()


if __name__ == "__main__":
    main()
