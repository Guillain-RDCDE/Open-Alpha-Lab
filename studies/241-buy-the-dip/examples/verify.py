"""Real-tape verification — Study 241 (Buy-the-Dip). Regenerates docs/results.md numbers.

Fetches (or reads from cache) SPY daily bars, runs systematic dip-buying across a
grid of drawdown thresholds (2%, 5%, 10%, 15%, 20%), compares to always-invested
buy-and-hold and monthly DCA, and computes HAC t-stats on the daily excess returns.

    python studies/241-buy-the-dip/examples/verify.py            # cache-only
    python studies/241-buy-the-dip/examples/verify.py --fetch    # refresh the tape
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from buy_the_dip import data, strategy as st  # noqa: E402


def hac_tstat(series: pd.Series) -> float:
    """Newey-West HAC t-stat for the mean of a series."""
    x = series.dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 6:
        return float("nan")
    mu = x.mean()
    e = x - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def main(fetch: bool) -> None:
    cache_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "_cache")
    )
    bars = data.fetch_daily("SPY", start="1993-01-01", fetch=fetch, cache_dir=cache_dir)
    fp = data.fingerprint(bars)
    n_years = len(bars) / 252.0

    print(f"SPY tape: {bars.index[0].date()} to {bars.index[-1].date()}, "
          f"{len(bars)} days ({n_years:.1f} years)")
    print(f"Fingerprint: {fp}\n")

    # Buy-and-hold benchmark
    bh = st.buy_and_hold_equity(bars)
    bh_daily = bh.pct_change().dropna()
    bh_cagr = (float(bh.iloc[-1]) ** (1.0 / n_years) - 1) * 100
    bh_sharpe = float(bh_daily.mean()) / float(bh_daily.std()) * np.sqrt(252)
    bh_maxdd = float(((bh - bh.cummax()) / bh.cummax()).min()) * 100
    print(f"=== BUY-AND-HOLD BENCHMARK ===")
    print(f"CAGR={bh_cagr:.2f}%  Sharpe={bh_sharpe:.3f}  MaxDD={bh_maxdd:.1f}%\n")

    # Threshold sweep
    print(f"=== THRESHOLD SWEEP (1 bp one-way cost, 0% cash rate) ===")
    print(f"{'Thr':>5}  {'Buys':>5}  {'Frac':>6}  {'CAGR':>7}  {'BH CAGR':>8}  {'vs BH':>7}  {'HAC t':>7}")
    for thr in st.THRESHOLDS:
        eq, meta = st.dip_buyer_equity(bars, threshold=thr, cost_bps=1.0, cash_rate_annual=0.0)
        s = st.summarize_equity(eq, bh)
        ret_strat = eq.pct_change().dropna()
        diff = ret_strat - bh_daily
        t = hac_tstat(diff)
        print(f"{int(thr*100):>4}%  {meta['n_buys']:>5}  {meta['fraction_invested']:>6.3f}"
              f"  {s['cagr']*100:>6.2f}%  {s['bh_cagr']*100:>7.2f}%  "
              f"{s['cagr_vs_bh']*100:>+6.2f}%  {t:>+6.2f}")

    # DCA comparison
    print()
    print(f"=== DCA COMPARISON ===")
    dca = st.dca_equity(bars, freq="ME")
    dca_cagr = (float(dca.iloc[-1]) ** (1.0 / n_years) - 1) * 100
    print(f"DCA CAGR: {dca_cagr:.2f}%  BH CAGR: {bh_cagr:.2f}%  "
          f"DCA vs BH: {dca_cagr - bh_cagr:+.2f}%")

    # Sensitivity: 4% cash rate
    print()
    print(f"=== SENSITIVITY: 4% ANNUAL CASH RATE ===")
    print(f"{'Thr':>5}  {'CAGR':>7}  {'vs BH':>7}")
    for thr in st.THRESHOLDS:
        eq, meta = st.dip_buyer_equity(bars, threshold=thr, cost_bps=1.0, cash_rate_annual=0.04)
        s = st.summarize_equity(eq, bh)
        print(f"{int(thr*100):>4}%  {s['cagr']*100:>6.2f}%  {s['cagr_vs_bh']*100:>+6.2f}%")

    print(f"\nquantlab.repro.fingerprint: {fp}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
