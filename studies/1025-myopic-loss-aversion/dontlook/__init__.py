"""Study 1025 — Don't Look.

Benartzi & Thaler (1995) explained the equity premium with two ingredients: investors feel a
loss about twice as sharply as an equal gain (loss aversion), and they *count* their gains and
losses often (myopia). Look at a stock portfolio every day and you see a loss on almost half of
them; look once a decade and you almost never do. Put Tversky-Kahneman (1992) preferences on
the historical distribution of returns and an investor becomes indifferent between stocks and
bonds at an evaluation horizon of about a year — so the premium, on this reading, is the price
of looking too often. The folk version is advice: *check your portfolio less and you will stay
invested*.

This study rebuilds the calculation on frozen, fingerprinted tapes and asks two separate
questions, as the desk always does:

- **Signal** — does the horizon really flip a loss-averse investor's preference, and is the
  flip point pinned near one year once the sampling noise of ninety years of returns is put on
  it?
- **Tradability** — "looking less" changes what you *see*, not what you *earn*. The only money
  in the claim is the cost of the behaviour it is meant to prevent: an investor who flees to
  bills after seeing a loss. How much does that behaviour actually cost, net, risk-adjusted?

- :mod:`dontlook.data` — the real tapes (``quantlab.bundled``: Fama-French total return,
  Moody's AAA yields, core CPI, daily S&P 500 price index) and the deterministic synthetic
  generator used by the whole test-suite.
- :mod:`dontlook.strategy` — cumulative prospect theory on empirical horizon distributions,
  the break-even horizon, the block bootstrap, the sweeps and the myopic switcher.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
