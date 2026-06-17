"""Study 228 — Pre-Earnings Runup: do stocks drift UP in the days before their earnings announcement?

The claim (distinct from PEAD in study 34, which is POST-event): there is a systematic pre-announcement
price runup as informed traders and option-market participants position ahead of earnings. Buy a few days
before the scheduled announcement date, exit just before the release. A well-known but increasingly
arbitraged pattern; expected to be fragile or a mirage on a liquid large-cap universe.

**Data.** Real run: yfinance prices and earnings dates for a fixed large-cap universe (S&P 500 proxy
tickers). Fully-offline synthetic control (a panel where a planted drift occurs in the N days before each
scheduled event, plus a null) backs the test-suite.
"""

from . import data, strategy  # noqa: F401
