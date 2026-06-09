"""Study 22 — Crystal-Ball: the HP-filter strategy that 'sees the future'.

The fifth study mined from Kakushadze & Serur's *151 Trading Strategies* (strategy 8.1, moving averages
with an HP filter). It is the desk's first study about a *backtest trap* rather than a market effect.
The steelman: detrend a price with a Hodrick-Prescott filter, then trade the mean reversion of the
cycle (long when price is below trend) -- a spectacular backtest. The catch: the classic HP filter is
**two-sided**, so the trend at time t uses the *entire* series, future included; a strategy on that
cycle is quietly using tomorrow's data. We show the edge is pure look-ahead and the honest causal
version has nothing. The reusable pieces:

    * :mod:`data` — a synthetic log-price = slow trend + an AR(1) cycle: ``revert_rho<1`` mean-reverts
      (a real signal an honest filter finds), ``==1`` is a pure random walk (nothing to find), plus a
      cache-only ETF reader.
    * :mod:`hp` — the load-bearing engine: the **two-sided** HP trend (uses all data -- the trap) and a
      **one-sided** causal HP trend (trailing-window endpoint -- tradable), and the cycle from each.
    * :mod:`strategy` — the cycle-mean-reversion book ``w = -sign(cycle)``, built two ways (``causal``
      flag) so the only difference is what the filter is allowed to see.
    * :mod:`decompose` — the inference: the **look-ahead Sharpe gap** with HAC t on each book, the
      **future-leakage** smoking gun (the two-sided cycle correlates with *future* returns), and the
      one-sided book's **honest edge**. The verdict it lands: Signal `NONE`, Tradability `MIRAGE`,
      Look-ahead bias `BUSTED` -- a fabricated edge, real only on paper.
    * :mod:`extension` — the beat-7 worked complement: the bias **survives every robustness check**
      (cost, smoothing ``lam``) except the one that matters -- using only past data.
"""
