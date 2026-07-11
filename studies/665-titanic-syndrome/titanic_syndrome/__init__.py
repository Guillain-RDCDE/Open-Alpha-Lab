"""Study 665 — Titanic Syndrome.

Package layout:

* :mod:`titanic_syndrome.data` — real tape (Dow-30 breadth panel + ^GSPC/SPY, yfinance,
  cached) and a deterministic synthetic panel with a tunable planted effect.
* :mod:`titanic_syndrome.strategy` — the Ohama (1965) breadth signal, forward-return
  and false-alarm tests, the "exit on signal" timer, and the inference primitives.
"""
