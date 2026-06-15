"""Real-tape verification — Study 175 (Crypto-Weekend). Regenerates docs/results.md numbers.

Fetches (or reads from cache) the BTC-USD daily tape, runs all three sub-claims
(mean return, volatility, timing rule), tests the pre/post-2023 regime split, and
shows the day-of-week breakdown. Network is touched only with --fetch.

    python studies/175-crypto-weekend/examples/verify.py            # cache-only
    python studies/175-crypto-weekend/examples/verify.py --fetch    # refresh the tape
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from crypto_weekend import data, strategy as st  # noqa: E402

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
COST_BPS = 5.0


def main(fetch: bool) -> None:
    bars = data.fetch_daily(fetch=fetch)
    if fetch:
        print(f"BTC-USD {bars.index[0].date()} .. {bars.index[-1].date()} "
              f"n={len(bars)} fp={data.fingerprint(bars)}")
        print()

    daily = st.daily_returns(bars)
    rc = st.return_comparison(daily)
    vc = st.volatility_comparison(daily)
    tr_gross = st.weekend_timing_rule(bars, cost_bps=0.0)
    tr_net = st.weekend_timing_rule(bars, cost_bps=COST_BPS)
    tr_10 = st.weekend_timing_rule(bars, cost_bps=10.0)
    rs = st.regime_split(daily)

    print("=== Claim 1: mean return — weekend vs weekday ===")
    print(f"Weekend  n={rc['weekend']['n']} mean={rc['weekend']['mean_pct']:+.4f}% "
          f"sharpe_ann={rc['weekend']['sharpe_ann']:.3f}")
    print(f"Weekday  n={rc['weekday']['n']} mean={rc['weekday']['mean_pct']:+.4f}% "
          f"sharpe_ann={rc['weekday']['sharpe_ann']:.3f}")
    print(f"Premium: {rc['premium_pct']:+.4f}%  HAC t = {rc['diff_tstat']:+.3f}")
    print(f"Bonferroni-significant (|t|>=2.39)? {rc['bonferroni_sig']}")
    print()

    print("=== Claim 2: volatility — weekend vs weekday ===")
    print(f"Ann. vol weekend: {vc['vol_ann_weekend']:.4f}  weekday: {vc['vol_ann_weekday']:.4f}")
    print(f"Ratio (wknd/wkday): {vc['vol_ratio']:.4f}  Levene HAC t: {vc['levene_t']:+.3f}")
    print()

    print("=== Claim 3: weekend-only timing rule ===")
    print(f"Gross (cost=0):    mean={tr_gross['mean_pct']:+.4f}%  t={tr_gross['tstat']:+.3f}")
    print(f"Net (cost={COST_BPS}bps): mean={tr_net['net_mean_pct']:+.4f}%  t={tr_net['net_tstat']:+.3f}")
    print(f"Net (cost=10bps): mean={tr_10['net_mean_pct']:+.4f}%  t={tr_10['net_tstat']:+.3f}")
    print()

    print("=== Day-of-week breakdown ===")
    for dow in range(7):
        sub = daily[daily["dow"] == dow]["log_ret"]
        print(f"  {DAY_NAMES[dow]}: n={len(sub)} mean={sub.mean()*100:+.4f}% std={sub.std(ddof=1):.4f}")
    print()

    print("=== Regime split (pre/post 2023-01-01) ===")
    for regime, d in rs.items():
        print(f"  {regime}: n_wknd={d['n_weekend']} n_wkday={d['n_weekday']} "
              f"premium={d['premium_pct']:+.4f}% t={d['diff_tstat']:+.3f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
