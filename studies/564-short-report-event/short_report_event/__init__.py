"""Study 564 — Short-Report-Event: abnormal returns after activist short-seller reports.

The short-side cousin of Study 390 (Activist-13D). When a famous activist *short* seller
(Muddy Waters, Hindenburg, Citron, ...) publishes a public hit piece, does the target
actually fall on the day — and, more importantly, *keep* falling in a way you could trade?

Engine in ``strategy``; data (a transparent hardcoded event table + cache-first yfinance
prices + a deterministic synthetic control) in ``data``.
"""
