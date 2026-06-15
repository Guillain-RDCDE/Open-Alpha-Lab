"""Study 210 — Crypto-Trend.

Trend-following on Bitcoin: does the simple 200-day (or 10-month) moving-average timing
rule on BTC-USD dramatically cut the brutal crypto drawdowns while keeping most of the
upside? The Faber rule applied to crypto.

Package layout::

    crypto_trend/
        data.py      — synthetic generator + real BTC-USD loader
        strategy.py  — MA timing rule, buy-and-hold, random-timing control, HAC inference
"""
