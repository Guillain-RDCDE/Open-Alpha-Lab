"""Real-tape verification — Study 324 (Bitcoin-Treasury). Regenerates docs/results.md.

Fetches (or reads from cache) MSTR and BTC-USD daily closes, decomposes MSTR into its
Bitcoin beta + residual alpha (HAC t on the intercept — the inference-bar number), races
holding MSTR against a synthetic levered-BTC replica at MSTR's own beta (net of borrow,
excess-of-cash Sharpe), and tests whether timing the NAV premium pays. Network is touched
only with --fetch.

    python studies/324-bitcoin-treasury/examples/verify.py            # cache-only
    python studies/324-bitcoin-treasury/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bitcoin_treasury import data, strategy as st  # noqa: E402

# Annual rates for the carry/cash assumptions, converted to per-day.
RF_ANNUAL = 0.03           # cash rate, excess-of-cash Sharpe
BORROW_ANNUAL = 0.08       # financing on MSTR's borrowed (beta-1) notional
RF_DAILY = RF_ANNUAL / 365
BORROW_BPS_DAY = BORROW_ANNUAL / 365 * 1e4


def main(fetch: bool) -> None:
    if fetch:
        for t in ("MSTR", "BTC-USD"):
            s = data.fetch_close(t, start="2014-01-01", fetch=True)
            print(f"{t:9s} {s.index[0].date()}..{s.index[-1].date()} n={len(s)}")
        print()

    full = data.fetch_pair(fetch=False)
    # The "Bitcoin treasury" era begins 2020-08-11 (Saylor's first BTC purchase).
    # Pre-2020 MSTR was a no-growth software company; including it corrupts the beta.
    TREASURY_START = "2020-08-11"
    frame = full.loc[full.index >= TREASURY_START]
    print(f"Full tape : {full.index[0].date()} to {full.index[-1].date()}, "
          f"{len(full)} common days")
    print(f"Treasury era (>= {TREASURY_START}): {frame.index[0].date()} to "
          f"{frame.index[-1].date()}, {len(frame)} days")
    print(f"Fingerprint (treasury era): {data.fingerprint(frame)}\n")

    dec = st.beta_decomposition(frame)
    print("=== MSTR decomposed onto BTC (simple daily returns) ===")
    print(f"  beta to BTC : {dec['beta']:.2f}")
    print(f"  R^2         : {dec['r2']*100:.1f}%  (share of MSTR variance that is just BTC)")
    print(f"  alpha       : {dec['alpha_daily']*1e4:+.2f} bps/day "
          f"({dec['alpha_ann']*100:+.1f}%/yr)")
    print(f"  HAC t(alpha): {dec['t_alpha']:+.2f}\n")

    race = st.race_mstr_vs_proxy(frame, rf_daily=RF_DAILY,
                                 financing_bps_per_day=BORROW_BPS_DAY)
    print("=== Race: hold MSTR vs levered-BTC replica (net borrow, excess-Sharpe) ===")
    print(f"  leverage used    : {race['leverage']:.2f}x")
    print(f"  MSTR  CAGR {race['mstr_cagr']*100:+7.1f}%  Sharpe {race['mstr_sharpe']:+.2f}  "
          f"MDD {race['mstr_mdd']*100:6.1f}%")
    print(f"  BTC   CAGR {race['btc_cagr']*100:+7.1f}%  Sharpe {race['btc_sharpe']:+.2f}  "
          f"MDD {race['btc_mdd']*100:6.1f}%")
    print(f"  proxy CAGR {race['proxy_cagr']*100:+7.1f}%  Sharpe {race['proxy_sharpe']:+.2f}  "
          f"MDD {race['proxy_mdd']*100:6.1f}%")
    print(f"  MSTR - proxy daily diff: {race['diff_mean_bps']:+.2f} bps  "
          f"HAC t {race['diff_t']:+.2f}\n")

    tim = st.premium_timing(frame, cost_bps=5.0)
    print("=== NAV-premium timing (buy MSTR only when its price-premium is cheap) ===")
    print(f"  timed  CAGR {tim['timed_cagr']*100:+7.1f}%  Sharpe {tim['timed_sharpe']:+.2f}")
    print(f"  always CAGR {tim['always_cagr']*100:+7.1f}%  Sharpe {tim['always_sharpe']:+.2f}")
    print(f"  timed - always daily diff: {tim['diff_mean_bps']:+.2f} bps  "
          f"HAC t {tim['diff_t']:+.2f}  (in market {tim['pct_in_market']*100:.0f}% of days)")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
