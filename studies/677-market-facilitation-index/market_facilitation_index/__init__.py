"""Study 677 — Market Facilitation Index (Bill Williams' BW-MFI).

``market_facilitation_index.data``     — real-tape loader (yfinance, cache-first) + a
                                          deterministic synthetic world with a tunable
                                          planted continuation/reversal effect.
``market_facilitation_index.strategy`` — the four-state classifier (green/fade/fake/squat),
                                          forward-return and continuation-score inference,
                                          and the state-conditioned timers.
"""
