"""Study 386 — NFCI-Conditions (does a financial-conditions index time stocks?).

The Chicago Fed's **National Financial Conditions Index (NFCI)** is a weekly z-score of
~100 risk, credit and leverage measures: positive => *tighter* than average financial
conditions, negative => *looser*. The folklore (repeated across macro-Twitter and the
financial press) says it is a regime switch for equities: **when the index turns tight,
step out of stocks**; when it loosens, get back in.

The true NFCI is published only on FRED (``NFCI``/``ANFCI``), which is **not reachable from
this environment** (the St. Louis Fed endpoint times out / resets here). So we CONSTRUCT a
transparent **financial-conditions proxy** entirely from yfinance instruments — equity
volatility (``^VIX``), rates volatility (``^MOVE``), a corporate-credit-spread proxy
(investment-grade ``LQD`` vs duration-matched Treasuries ``IEF``) and a broad-dollar momentum
leg (``UUP``) — each z-scored and bundled into one weekly index where **high = tight**, the
same sign convention as NFCI. We then test whether tightness predicts forward S&P 500 returns
and drawdowns, and whether a "step out when tight" timing rule beats buy-and-hold net of costs.

The decisive findings are (a) the proxy index is **contemporaneously** glued to stocks
(tight weeks *are* falling-market weeks — that is mechanical, not predictive), and (b) once
you act on the signal with a one-week execution lag and one-way costs, the *forward* edge is
small, regime-concentrated (one '08 crisis), and does not clear the desk's t >= 2 bar on the
real tape.

See :mod:`nfci_conditions.data` (real proxy loader + deterministic synthetic control) and
:mod:`nfci_conditions.strategy` (tight/loose regime split, forward-return inference, placebo
null, the timing backtest with execution lag and costs)."""

from . import data, strategy

__all__ = ["data", "strategy"]
