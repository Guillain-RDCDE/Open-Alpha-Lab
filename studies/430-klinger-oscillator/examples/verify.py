"""Real-tape verification — Study 430 (Klinger Volume Oscillator).

Reproduces every number in docs/results.md: the headline Klinger long/flat race vs
buy-and-hold / 200d SMA / MACD on SPY (net Sharpe, excess-of-cash, 1-day lag), the HAC
t of the daily difference, the block-permutation placebo, the rule variants, the cost
sweep, the 6-ETF panel, and the synthetic positive control. Cache-first: network is
touched only with --fetch on a cache miss.

    python studies/430-klinger-oscillator/examples/verify.py            # cache-only
    python studies/430-klinger-oscillator/examples/verify.py --fetch    # refresh tapes
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from klinger_oscillator import data, strategy as st  # noqa: E402

ASOF = "2026-05-31"   # drop the partial current month (house rule)


def _spy(fetch: bool):
    b = data.fetch_panel("SPY", fetch=fetch)
    return b[b.index <= ASOF]


def main(fetch: bool) -> None:
    if fetch:
        for t in data.PANEL:
            b = data.fetch_panel(t, fetch=True)
            print(f"{t:5s} {b.index[0].date()}..{b.index[-1].date()} n={len(b)} fp={data.fingerprint(b)}")
        print()

    spy = _spy(fetch)
    print(f"SPY {spy.index[0].date()}..{spy.index[-1].date()} n={len(spy)} "
          f"fp={data.fingerprint(spy)}  as-of {ASOF}  years={len(spy)/252:.1f}")

    # --- headline race (zero mode, long/flat, 1bp) ---
    r = st.run_experiment(spy, mode="zero", long_short=False, cost_bps=1.0,
                          with_placebo=True, n_draws=3000)
    k = r["klinger"]
    print("\n=== headline race (KVO>0 long/flat, 1bp, excess-of-cash) ===")
    print(f"Klinger   Sharpe={k['sharpe']:+.3f} CAGR={k['cagr']:+.3f} maxDD={k['mdd']:+.3f} "
          f"TiM={k['tim']:.2f} turnover/yr={k['turnover_yr']:.1f}")
    print(f"buy&hold  Sharpe={r['asset_sharpe']:+.3f} CAGR={r['asset_cagr']:+.3f} maxDD={r['asset_mdd']:+.3f}")
    print(f"200d SMA  Sharpe={r['sma']['sharpe']:+.3f} CAGR={r['sma']['cagr']:+.3f} maxDD={r['sma']['mdd']:+.3f}")
    print(f"MACD      Sharpe={r['macd']['sharpe']:+.3f}")
    print(f"HAC t  vs buy-hold={r['t_vs_hold']:+.2f}  vs SMA={r['t_vs_sma']:+.2f}  vs MACD={r['t_vs_macd']:+.2f}")
    print(f"placebo p(random schedule >= Klinger) = {r['placebo_p']:.3f}  (obs Sharpe {r['placebo_obs']:.3f})")

    # --- variants ---
    print("\n=== rule variants (1bp) ===")
    for mode, ls, name in [("zero", False, "KVO>0 L/F"), ("zero", True, "KVO>0 L/S"),
                           ("signal", False, "KVO>sig L/F"), ("signal", True, "KVO>sig L/S")]:
        rr = st.run_experiment(spy, mode=mode, long_short=ls, cost_bps=1.0, with_placebo=False)
        print(f"{name:12s} Sharpe={rr['klinger']['sharpe']:+.3f}  t_vs_hold={rr['t_vs_hold']:+.2f}")

    # --- cost sweep ---
    print("\n=== cost sweep (KVO>0 long/flat) ===")
    for c in (0.0, 1.0, 2.0, 5.0):
        rr = st.run_experiment(spy, mode="zero", cost_bps=c, with_placebo=False)
        print(f"cost={c:4.1f}bps  Sharpe={rr['klinger']['sharpe']:+.3f}  t_vs_hold={rr['t_vs_hold']:+.2f}")

    # --- panel ---
    print("\n=== panel (KVO>0 long/flat, 1bp) ===")
    for t in data.PANEL:
        if not data.have_real(t):
            continue
        bb = data.load_real(t)
        bb = bb[bb.index <= ASOF]
        rr = st.run_experiment(bb, mode="zero", cost_bps=1.0, with_placebo=False)
        print(f"{t:5s} Klinger={rr['klinger']['sharpe']:+.3f}  buy-hold={rr['asset_sharpe']:+.3f}  "
              f"t_vs_hold={rr['t_vs_hold']:+.2f}")

    # --- synthetic positive control ---
    print("\n=== synthetic control (KVO>0 long/flat) ===")
    for edge in (0.0, 3.0):
        bars, _ = data.synthetic_panel(edge=edge, seed=430)
        rr = st.run_experiment(bars, mode="zero", cost_bps=1.0, with_placebo=True, n_draws=2000)
        print(f"edge={edge:.1f}: Klinger={rr['klinger']['sharpe']:+.3f}  buy-hold={rr['asset_sharpe']:+.3f}  "
              f"t_vs_hold={rr['t_vs_hold']:+.2f}  placebo_p={rr['placebo_p']:.3f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
