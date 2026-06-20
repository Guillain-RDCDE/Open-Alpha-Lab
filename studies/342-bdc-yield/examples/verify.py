"""Reproduce the Study 342 (BDC-Yield) headline run on the real total-return tapes.

    PYTHONIOENCODING=utf-8 python examples/verify.py

Pins BIZD (VanEck BDC Income ETF) against SPY (equity) and IEF (7-10y Treasuries) on
yield/total-return, volatility, drawdown, and — the decisive test — who BIZD moves with
into an equity/credit crash. As-of 2026-05-31 (last complete month); match the
fingerprints to confirm you hold the same tapes.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from bdc_yield import data, strategy  # noqa: E402

AS_OF = "2026-05-31"
QUOTED_YIELD = 0.10  # BIZD's headline distribution rate, ~10%


def _hac_beta_t(p, x):
    """(beta, HAC t) of p on x via the influence series whose mean is the OLS slope."""
    import numpy as np
    p = p.to_numpy(dtype=float); x = x.to_numpy(dtype=float)
    xc = x - x.mean()
    infl = (p - p.mean()) * xc / (xc @ xc) * len(xc)
    return float(infl.mean()), strategy.hac_tstat(infl)


def main() -> int:
    prices = data.load_real(("BIZD", "SPY", "IEF")).loc[:AS_OF]
    rets = strategy.to_returns(prices)
    print(f"panel: {len(prices):,} rows  {prices.index[0].date()} -> "
          f"{prices.index[-1].date()}  fingerprint={data.fingerprint(prices)}")

    print("\n--- the three arms (total return) ---")
    bizd_cagr = None
    for c in ("BIZD", "SPY", "IEF"):
        s = strategy.stats(rets[c])
        if c == "BIZD":
            bizd_cagr = s["cagr"]
        print(f"{c}: CAGR {s['cagr']*100:6.2f}%  vol {s['vol']*100:5.1f}%  "
              f"Sharpe {s['sharpe']:.3f}  maxDD {s['max_dd']*100:6.1f}%")

    print("\n--- the headline-yield vs realised total-return gap ---")
    hv = strategy.headline_vs_realised(bizd_cagr, QUOTED_YIELD)
    print(f"quoted distribution yield = {hv['quoted_yield']*100:.1f}%  "
          f"realised total-return CAGR = {hv['realised_cagr']*100:.2f}%  "
          f"gap = {hv['gap']*100:.2f} pts ({hv['fraction_phantom']*100:.0f}% of the headline)")

    print("\n--- who does BIZD move with? ---")
    b_spy, t_spy = _hac_beta_t(rets["BIZD"], rets["SPY"])
    b_ief, t_ief = _hac_beta_t(rets["BIZD"], rets["IEF"])
    print(f"beta BIZD~SPY = {b_spy:+.3f}  HAC t = {t_spy:+.2f}")
    print(f"beta BIZD~IEF = {b_ief:+.3f}  HAC t = {t_ief:+.2f}")
    print(f"downside beta to SPY (worst 10% SPY days) = "
          f"{strategy.downside_beta(rets['BIZD'], rets['SPY']):.3f}")
    print(f"downside beta to IEF (worst 10% SPY days) = "
          f"{strategy.downside_beta(rets['BIZD'], rets['IEF']):.3f}")

    boot = strategy.bootstrap_downside_beta_diff(
        rets["BIZD"], rets["SPY"], rets["IEF"], n_boot=2000, seed=342)
    print(f"downside-beta diff (SPY - IEF) = {boot['point']:+.3f}  "
          f"95% CI [{boot['ci95'][0]:+.3f}, {boot['ci95'][1]:+.3f}]  "
          f"frac SPY>IEF = {boot['frac_spy_wins']*100:.0f}%")

    print("\n--- every >10% equity crash: did BIZD cushion (like a bond) or pile on? ---")
    eps = strategy.equity_drawdowns(rets[["SPY", "BIZD", "IEF"]], "SPY", thresh=-0.10)
    rows = [{"peak": e["peak"].date(), "trough": e["trough"].date(),
             "SPY %": e["stock_loss"]*100, "BIZD %": e["others"]["BIZD"]*100,
             "IEF %": e["others"]["IEF"]*100} for e in eps]
    print(pd.DataFrame(rows).round(1).to_string(index=False))
    bizd_fell = sum(1 for e in eps if e["others"]["BIZD"] < 0)
    ief_rose = sum(1 for e in eps if e["others"]["IEF"] > 0)
    print(f"BIZD FELL in {bizd_fell}/{len(eps)} equity crashes; "
          f"IEF (bonds) ROSE in {ief_rose}/{len(eps)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
