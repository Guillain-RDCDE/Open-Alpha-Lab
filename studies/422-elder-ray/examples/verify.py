"""Real-tape verification — Study 422 (Elder Ray). Regenerates docs/results.md numbers.

Loads (cache-first) the daily tapes, runs the Elder-Ray long/flat timing rule against
buy-and-hold and the two simpler benchmarks (200-day SMA filter, EMA13 cross), sweeps
costs, runs the rotation placebo, and prints the synthetic positive control. Network is
touched only with --fetch (or on a cache miss).

    python studies/422-elder-ray/examples/verify.py            # cache-first (offline)
    python studies/422-elder-ray/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from elder_ray import data, strategy as st  # noqa: E402

ASOF = "2026-05-31"
PANEL = ["SPY", "QQQ", "IWM", "EFA", "GLD"]
CACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_cache"))


def load(t: str, fetch: bool):
    return data.load_real(t, cache_dir=CACHE, fetch=fetch).loc[:ASOF]


def main(fetch: bool) -> None:
    print(f"=== Data stamp (as-of {ASOF}) ===")
    for t in PANEL:
        b = load(t, fetch)
        print(f"{t:4s} {b.index[0].date()}..{b.index[-1].date()} "
              f"n={len(b)} fp={data.fingerprint(b)}")

    spy = load("SPY", False)
    res = st.run_experiment(spy, span=13, cost_bps=2.0)
    e = res["elder"]

    print("\n=== SPY — Elder-Ray long/flat (net, cost=2bps) vs buy-and-hold ===")
    print(f"  Elder Sharpe {e['strat_sharpe']:+.3f}  CAGR {e['strat_cagr']*100:+.2f}%  "
          f"MDD {e['strat_mdd']*100:.1f}%  HAC t {e['strat_hac_t']:+.3f}  "
          f"TIM {e['time_in_market']*100:.1f}%  turnover {e['turnover_oneway_per_yr']:.1f}/yr")
    print(f"  BuyHold Sharpe {e['bh_sharpe']:+.3f}  CAGR {e['bh_cagr']*100:+.2f}%  "
          f"MDD {e['bh_mdd']*100:.1f}%  HAC t {e['bh_hac_t']:+.3f}")
    print(f"  Sharpe delta (Elder - hold): {e['sharpe_delta']:+.3f}   placebo p = {e['perm_p']:.3f}")

    print("\n=== Simpler benchmarks on SPY (net, cost=2bps) ===")
    print(f"  200-day SMA filter  Sharpe {res['sma200']['strat_sharpe']:+.3f}  "
          f"CAGR {res['sma200']['strat_cagr']*100:+.2f}%  HAC t {res['sma200']['strat_hac_t']:+.3f}")
    print(f"  EMA13 cross         Sharpe {res['emacross']['strat_sharpe']:+.3f}  "
          f"CAGR {res['emacross']['strat_cagr']*100:+.2f}%  HAC t {res['emacross']['strat_hac_t']:+.3f}")

    print("\n=== Long/short variant (SPY) ===")
    ls = st.run_experiment(spy, span=13, cost_bps=2.0, long_short=True)["elder"]
    print(f"  Sharpe {ls['strat_sharpe']:+.3f}  CAGR {ls['strat_cagr']*100:+.2f}%  "
          f"HAC t {ls['strat_hac_t']:+.3f}  TIM {ls['time_in_market']*100:.1f}%")

    print("\n=== Cost sweep (Elder long/flat, SPY) ===")
    for c in (0.0, 1.0, 2.0, 5.0):
        r = st.run_experiment(spy, span=13, cost_bps=c)["elder"]
        print(f"  cost={c:.1f}bps  Sharpe {r['strat_sharpe']:+.3f}  HAC t {r['strat_hac_t']:+.3f}")

    print("\n=== Panel (Elder long/flat net cost=2) ===")
    for t in PANEL:
        b = load(t, False)
        r = st.run_experiment(b, span=13, cost_bps=2.0)["elder"]
        print(f"  {t:4s} Elder {r['strat_sharpe']:+.3f}  hold {r['bh_sharpe']:+.3f}  "
              f"delta {r['sharpe_delta']:+.3f}  HAC t {r['strat_hac_t']:+.3f}")

    print("\n=== Synthetic positive control (n=6000) ===")
    for edge in (0.0, 0.20):
        b, _ = data.synthetic_panel(n_days=6000, edge=edge, seed=422)
        r = st.run_experiment(b, span=13, cost_bps=2.0)["elder"]
        print(f"  edge={edge:+.2f}  Elder {r['strat_sharpe']:+.3f}  hold {r['bh_sharpe']:+.3f}  "
              f"delta {r['sharpe_delta']:+.3f}  HAC t {r['strat_hac_t']:+.3f}  placebo p {r['perm_p']:.3f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
