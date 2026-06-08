"""Study 08 — True-Strength: is the TSI a *truer* strength gauge than MACD/RSI,
or the same momentum trade repainted?

The package is small and split the desk's usual way:
  * :mod:`true_strength.oscillators` — TSI / MACD / RSI, the param grid, normalisation.
  * :mod:`true_strength.data`        — the cached real universe + an offline synthetic one.
  * :mod:`true_strength.backtest`    — oscillator -> position -> costed daily P&L.
  * :mod:`true_strength.collinearity`— the gauntlet: are the three the same signal?
"""
