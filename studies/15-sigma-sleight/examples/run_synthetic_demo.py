"""Offline demo — the whole machine on the synthetic tape, no network.

Generates a toy daily close with a *known*, single-horizon mean reversion, then runs the same
teardown the real script runs:

    * the sigma<->RSI map is a strictly-monotone relabel (within a length, an adaptive zone IS a
      constant RSI band -- identical trades, max position diff 0);
    * the adaptive-zone cheat-sheet is pure arithmetic in n (RSI(14)=70 <=> +1.53 sigma; zones
      compress toward 50 as length grows);
    * the cost-charged horse race: fixed 70/30 vs adaptive-sigma vs a re-optimised constant
      (adaptive can't beat re-optimising, because it's just one constant);
    * Rescaled RSI is a cross-length relabel: its rank IC equals raw RSI(long)'s, so the
      "horizon translation" adds no information the raw long RSI didn't already carry.

    python examples/run_synthetic_demo.py

This is the reproducible core: it proves the decomposition recovers what we *baked in* (a real
oversold edge, and sigma-machinery that only relabels), so the real-data run (`verify.py`) is a
measurement, not a hope.
"""

import os
import sys

_STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STUDY)
sys.path.insert(0, os.path.abspath(os.path.join(_STUDY, "..", "..")))

from sigma_sleight import data, decompose


def main():
    close, truth = data.synthetic_prices(seed=15)
    print(f"synthetic tape: {truth.n_bars} bars, baked-in mean reversion kappa = {truth.kappa} "
          f"(one ~{truth.horizon:.0f}-bar horizon)")

    print("\n[1 . the sigma-transform is a monotone relabel]")
    for n in (2, 14, 200):
        chk = decompose.monotone_check(n)
        print(f"  len {n:>3}: strictly increasing = {chk['strictly_increasing']}, "
              f"round-trip err {chk['max_roundtrip_err']:.1e}")
    ident = decompose.crossing_identity(close, length=2, lower_sigma=-1.732)
    print(f"  RSI(2) adaptive -sqrt3 sigma band == constant RSI {ident['implied_lower_rsi']:.1f}/"
          f"{ident['implied_exit_rsi']:.0f}: max position diff = {ident['max_position_diff']:.1f} "
          f"({ident['n_entries']} entries, identical = {ident['identical']})")

    print("\n[2 . the cheat-sheet is arithmetic, not calibration]")
    z = decompose.zone_arithmetic([2, 5, 14, 50, 200])
    print(z["table"].round(2).to_string())
    print(f"  RSI(14)=70 <=> {z['sigma_at_rsi70_len14']:.3f} sigma "
          f"(the '1.53 sigma' landmark = {z['rsi70_is_1p53_sigma']}); "
          f"compresses toward 50 = {z['compresses_toward_50']}")

    print("\n[3 . horse race: fixed 70/30 vs adaptive-sigma vs re-optimised constant]")
    cmp = decompose.strategy_compare(close, length=2, cost_bps=1.0)
    f, a, r = cmp["fixed"], cmp["adaptive"], cmp["reopt"]
    print(f"  fixed 30/50   : net {f['net_ann_return']:+.1%}/yr, Sharpe {f['sharpe']:+.2f}, "
          f"{f['n_entries']} entries")
    print(f"  adaptive band : net {a['net_ann_return']:+.1%}/yr, Sharpe {a['sharpe']:+.2f}, "
          f"{a['n_entries']} entries  (= constant RSI {cmp['adaptive_implied_lower_rsi']:.1f}/50)")
    print(f"  reopt const   : net {r['net_ann_return']:+.1%}/yr, Sharpe {r['sharpe']:+.2f}, "
          f"best lower = {r['lower']:.0f}")
    print(f"  -> adaptive minus re-optimised Sharpe = {cmp['adaptive_vs_reopt_sharpe']:+.2f} "
          f"(<= 0: sigma-calibration is one of the constants the grid already searches)")

    print("\n[4 . Rescaled RSI is a cross-length relabel]")
    out = decompose.rescale_increment(close, target_length=14, long_length=70, horizon=5)
    print(f"  IC raw RSI(70) = {out['ic_native_long']:+.3f}, "
          f"IC rescaled(70->14) = {out['ic_rescaled_long']:+.3f}, "
          f"gap = {out['rescale_ic_gap']:+.1e} (rank-invariant)")
    print(f"  the longer window does add info over RSI(14) "
          f"(partial IC {out['incremental_ic_window_over_native']:+.3f}) -- but it's already in "
          f"*raw* RSI(70); the sigma-rescaling contributes none of it.")


if __name__ == "__main__":
    main()
