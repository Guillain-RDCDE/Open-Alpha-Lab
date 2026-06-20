"""Reproduce the Study 338 (Preferred-Stocks) headline run on the real total-return tapes.

    PYTHONIOENCODING=utf-8 python examples/verify.py

Pins PFF (iShares Preferred & Income Securities ETF) against SPY (equity) and IEF
(7-10y Treasuries) on yield/return, volatility, drawdown, and — the decisive test —
who PFF moves with into an equity crash. As-of 2026-05-31 (last complete month);
match the fingerprints to confirm you hold the same tapes.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from preferred_stocks import data, strategy  # noqa: E402

AS_OF = "2026-05-31"


def _hac_beta_t(p, x):
    """(beta, HAC t) of p on x via the influence series whose mean is the OLS slope."""
    import numpy as np
    p = p.to_numpy(dtype=float); x = x.to_numpy(dtype=float)
    xc = x - x.mean()
    infl = (p - p.mean()) * xc / (xc @ xc) * len(xc)
    return float(infl.mean()), strategy.hac_tstat(infl)


def main() -> int:
    prices = data.load_real(("PFF", "SPY", "IEF")).loc[:AS_OF]
    rets = strategy.to_returns(prices)
    print(f"panel: {len(prices):,} rows  {prices.index[0].date()} -> "
          f"{prices.index[-1].date()}  fingerprint={data.fingerprint(prices)}")

    print("\n--- the three arms (total return) ---")
    for c in ("PFF", "SPY", "IEF"):
        s = strategy.stats(rets[c])
        print(f"{c}: CAGR {s['cagr']*100:6.2f}%  vol {s['vol']*100:5.1f}%  "
              f"Sharpe {s['sharpe']:.3f}  maxDD {s['max_dd']*100:6.1f}%")

    print("\n--- who does PFF move with? ---")
    b_spy, t_spy = _hac_beta_t(rets["PFF"], rets["SPY"])
    b_ief, t_ief = _hac_beta_t(rets["PFF"], rets["IEF"])
    print(f"beta PFF~SPY = {b_spy:+.3f}  HAC t = {t_spy:+.2f}")
    print(f"beta PFF~IEF = {b_ief:+.3f}  HAC t = {t_ief:+.2f}")
    print(f"downside beta to SPY (worst 10% SPY days) = "
          f"{strategy.downside_beta(rets['PFF'], rets['SPY']):.3f}")
    print(f"downside beta to IEF (worst 10% SPY days) = "
          f"{strategy.downside_beta(rets['PFF'], rets['IEF']):.3f}")

    print("\n--- every >10% equity crash: did PFF cushion (like a bond) or pile on? ---")
    eps = strategy.equity_drawdowns(rets[["SPY", "PFF", "IEF"]], "SPY", thresh=-0.10)
    rows = [{"peak": e["peak"].date(), "trough": e["trough"].date(),
             "SPY %": e["stock_loss"]*100, "PFF %": e["others"]["PFF"]*100,
             "IEF %": e["others"]["IEF"]*100} for e in eps]
    print(pd.DataFrame(rows).round(1).to_string(index=False))
    pff_fell = sum(1 for e in eps if e["others"]["PFF"] < 0)
    ief_rose = sum(1 for e in eps if e["others"]["IEF"] > 0)
    print(f"PFF FELL in {pff_fell}/{len(eps)} equity crashes; "
          f"IEF (bonds) ROSE in {ief_rose}/{len(eps)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
