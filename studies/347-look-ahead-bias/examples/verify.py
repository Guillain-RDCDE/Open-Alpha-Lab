"""Real-tape verification — Study 347 (Look-Ahead-Bias). Regenerates docs/results.md.

Loads (or fetches) a daily close series, runs the SAME momentum signal under the
look-ahead ("peek", same-bar fill) and the correct one-bar-lag ("lag1")
conventions, and reports the manufactured Sharpe and HAC t of the peek versus the
honest lag — i.e. the look-ahead tax on real prices. Also prints the synthetic
positive control (a planted, lag-tradable reversion edge the harness banks).

    python studies/347-look-ahead-bias/examples/verify.py           # cache-only
    python studies/347-look-ahead-bias/examples/verify.py --fetch   # refresh the tape
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from look_ahead_bias import data, strategy as st  # noqa: E402

TICKER = "SPY"
LOOKBACK = 20


def _as_of_full_month(s: pd.Series) -> pd.Series:
    """Drop the in-progress final month so a stamped run never includes a partial bar."""
    today = pd.Timestamp.today().normalize()
    month_start = today.to_period("M").to_timestamp()
    asof = month_start - pd.Timedelta(days=1)
    return s[s.index <= asof]


def main(fetch: bool) -> None:
    s = data.load_real(TICKER, fetch=fetch)
    s = _as_of_full_month(s)
    print(f"tape: {TICKER}  {s.index[0].date()} .. {s.index[-1].date()}  "
          f"({len(s)} trading days)")
    print(f"fingerprint: {data.fingerprint(s)}\n")

    for cost in (0.0, 5.0):
        r = st.compare(s, lookback=LOOKBACK, cost_bps=cost, signal="momentum")
        tag = "gross" if cost == 0.0 else f"net {cost:.0f}bps/side"
        print(f"=== Momentum signal, {tag} ===")
        print(f"  peek (look-ahead) : Sharpe {r['summary_peek']['sharpe']:+6.2f}  "
              f"HAC t={r['test_peek']['tstat']:+6.2f}  CAGR={r['summary_peek']['cagr']*100:+7.1f}%")
        print(f"  lag1 (correct)    : Sharpe {r['summary_lag1']['sharpe']:+6.2f}  "
              f"HAC t={r['test_lag1']['tstat']:+6.2f}  CAGR={r['summary_lag1']['cagr']*100:+7.1f}%")
        print(f"  Sharpe inflation  : {r['sharpe_inflation']:+.2f}\n")

    # Synthetic positive control — a planted, lag-tradable reversion edge.
    print("=== Synthetic positive control (planted lag-tradable reversion edge) ===")
    for edge in (0.0, 0.002):
        p, _ = data.synthetic_prices(n_days=3000, edge=edge, seed=347)
        rc = st.compare(p, lookback=LOOKBACK, cost_bps=0.0, signal="reversion")
        label = "null (edge=0)" if edge == 0.0 else f"edge={edge}"
        print(f"  {label:14s}: lag1 Sharpe {rc['summary_lag1']['sharpe']:+6.2f} "
              f"(t={rc['test_lag1']['tstat']:+6.2f})  "
              f"peek Sharpe {rc['summary_peek']['sharpe']:+6.2f} "
              f"(t={rc['test_peek']['tstat']:+6.2f})")
    print("\nThe harness banks the planted edge ONLY with the correct lag; the peek "
          "is anti-correlated noise. On the null the lagged Sharpe is ~0 — a clean "
          "negative control.")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
