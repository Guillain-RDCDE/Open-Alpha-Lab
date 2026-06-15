"""Study 182 — Vortex-Indicator.

The Vortex Indicator (Botes & Siepman, 2010) distils two directional movement lines
— VI+ (upward vortex) and VI– (downward vortex) — from the high/low range and the
prior bar's range.  A VI+/VI– crossover is used as a trend-change signal: go long on
a VI+ > VI– crossover, short on VI– > VI+.

This package implements:
  - ``data``     — synthetic daily OHLCV generator (AR(1) momentum knob) + real yfinance loader.
  - ``strategy`` — Vortex Indicator computation, crossover entries, random-direction control,
                   fixed-horizon forward-return backtest, and HAC-inference summary.

See ``docs/results.md`` for the real-tape verdict.
"""
