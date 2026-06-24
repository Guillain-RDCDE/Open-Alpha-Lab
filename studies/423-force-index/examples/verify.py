"""Real-tape verification — Study 423 (Force Index). Regenerates docs/results.md numbers.

Reads (cache-first) six daily ETF tapes, turns Elder's FI(13) zero-cross into a long/flat
(and long/short) timing rule, and races its NET Sharpe against buy-and-hold (excess-vs-
excess) with a HAC t on the daily return difference, a sign-permutation placebo, a period
sweep, a cost sweep, and the simpler SMA/RSI benchmarks. Network is touched only with
--fetch (otherwise offline from cache). The partial as-of+1 bar is dropped.

    python studies/423-force-index/examples/verify.py            # cache-only
    python studies/423-force-index/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from force_index import data, strategy as st  # noqa: E402

ASOF = "2026-06-23"
TICKERS = ["SPY", "QQQ", "DIA", "IWM", "XLE", "GLD"]
CACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_cache"))


def _load(t: str, fetch: bool):
    b = data.load_real(t, fetch=fetch, cache_dir=CACHE)
    return b[b.index <= ASOF]


def main(fetch: bool) -> None:
    if fetch:
        for t in TICKERS:
            b = data.load_real(t, fetch=True, cache_dir=CACHE)
            print(f"{t:5s} {b.index[0].date()}..{b.index[-1].date()} fp={data.fingerprint(b)}")
        print()

    print(f"=== Data stamp (as-of {ASOF}) ===")
    for t in TICKERS:
        b = _load(t, False)
        print(f"{t:5s} {b.index[0].date()}..{b.index[-1].date()} "
              f"n={len(b):5d} fp={data.fingerprint(b)}")

    print("\n=== FI(13) zero-cross long/flat vs buy-and-hold, NET 1bp ===")
    print(f"{'tkr':5s} {'n':>5s} {'flips/y':>7s} {'FI_shp':>6s} "
          f"{'BH_shp':>6s} {'diff_bps':>8s} {'HAC_t':>6s}")
    for t in TICKERS:
        r = st.run_experiment(_load(t, False), period=13, cost_bps=1.0)
        print(f"{t:5s} {r['fi']['n_days']:5d} {r['fi_flips_per_yr']:7.1f} "
              f"{r['fi_sharpe']:6.2f} {r['bh_sharpe']:6.2f} "
              f"{r['fi']['mean_diff_bps']:+8.3f} {r['fi_hac_t']:+6.2f}")

    spy = _load("SPY", False)

    print("\n=== FI period sweep (SPY, long/flat, NET 1bp) ===")
    for p in (2, 13, 50):
        r = st.run_experiment(spy, period=p, cost_bps=1.0)
        print(f"FI({p:2d}) flips/y={r['fi_flips_per_yr']:5.1f} "
              f"net_Sharpe={r['fi_sharpe']:.2f} HAC_t={r['fi_hac_t']:+.2f}")

    print("\n=== Benchmark races (SPY, long/flat, NET 1bp) ===")
    r = st.run_experiment(spy, period=13, cost_bps=1.0)
    for name, key in (("FI(13)", "fi"), ("SMA 50/200", "sma"), ("RSI(14)>50", "rsi")):
        d = r[key]
        print(f"{name:11s} Sharpe={d['sharpe_strat']:.2f} "
              f"diff_bps={d['mean_diff_bps']:+.3f} HAC_t={d['hac_t_diff']:+.2f}")
    print(f"{'Buy & hold':11s} Sharpe={r['bh_sharpe']:.2f}")

    print("\n=== Cost sweep (SPY FI(13) long/flat) ===")
    for c in (0.0, 1.0, 2.0, 5.0):
        r = st.run_experiment(spy, period=13, cost_bps=c)
        print(f"cost={c:.1f}bp net_Sharpe={r['fi_sharpe']:.2f} HAC_t={r['fi_hac_t']:+.2f}")

    print("\n=== Long/SHORT FI(13) SPY (NET 1bp, borrow 50bp/yr) ===")
    rls = st.run_experiment(spy, period=13, cost_bps=1.0, long_short=True)
    print(f"net_Sharpe={rls['fi_sharpe']:+.2f} BH={rls['bh_sharpe']:.2f} "
          f"diff_bps={rls['fi']['mean_diff_bps']:+.3f} HAC_t={rls['fi_hac_t']:+.2f}")

    print("\n=== Sign-permutation placebo (SPY FI(13) long/flat 1bp, 1000 draws) ===")
    pv = st.permutation_pvalue(spy, period=13, cost_bps=1.0, n_perm=1000)
    print(f"real Sharpe gap={pv['real_gap']:+.3f}  p={pv['p_value']:.3f}")

    print("\n=== Synthetic positive control (FI13 long/short, NET 1bp, n=4000) ===")
    for pe in (0.0, 0.1, 0.2, 0.4, 0.7):
        b, _ = data.synthetic_panel(n_days=4000, planted_edge=pe, seed=423)
        r = st.run_experiment(b, period=13, cost_bps=1.0, long_short=True)
        print(f"planted={pe:.2f} net_Sharpe={r['fi_sharpe']:+.2f} "
              f"BH={r['bh_sharpe']:+.2f} HAC_t={r['fi_hac_t']:+.2f}")
    bn, _ = data.synthetic_panel(n_days=4000, planted_edge=0.0, seed=423)
    bp, _ = data.synthetic_panel(n_days=4000, planted_edge=0.4, seed=423)
    print("perm p (planted=0):  "
          f"{st.permutation_pvalue(bn, 13, 1.0, n_perm=500, long_short=True)['p_value']:.3f}")
    print("perm p (planted=0.4):"
          f"{st.permutation_pvalue(bp, 13, 1.0, n_perm=500, long_short=True)['p_value']:.3f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
