"""Real-tape verification — Study 936 (Tolerance Bands). Regenerates docs/results.md.

Reads cached SPY / IEF / GLD / BIL daily total-return closes from the shared desk cache
and races four rebalancing policies (drift, annual, quarterly, 5/25 bands) on two books:
a 60/40 SPY/IEF and a 50/30/20 SPY/IEF/GLD. Prints the excess-of-cash Sharpe race,
turnover and cost drag, the HAC *t* on the daily return difference, block-bootstrap CIs
on the Sharpe difference, an era cut, a cost sweep, a band-width sweep, and a longer
2002-2026 cross-check. Network only on ``--fetch``.

    python studies/936-rebalance-bands/examples/verify.py            # cache-only
    python studies/936-rebalance-bands/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from rebal_bands import data, strategy as st  # noqa: E402

BOOK_2 = (("SPY", "IEF"), (0.60, 0.40))
BOOK_3 = (("SPY", "IEF", "GLD"), (0.50, 0.30, 0.20))
COST_BPS = 5.0
SPLIT = "2016-01-01"
# The longer SPY/IEF cross-check is pinned to a fixed start so it cannot creep if the
# shared cache is later refreshed with a longer SPY history.
LONG_START = "2003-01-02"


def _race_block(rets, cols, target, cash, title):
    r = st.race(rets[list(cols)], target, cash=cash, cost_bps=COST_BPS)
    print(f"\n=== {title} ===")
    print(f"{'policy':>10} {'exSharpe':>9} {'CAGR':>8} {'vol':>7} {'maxDD':>8} "
          f"{'traded/yr':>10} {'rebals':>7} {'drag bps/yr':>12}")
    for p, s in r["stats"].items():
        print(f"{p:>10} {s['sharpe']:+9.4f} {s['cagr']:+8.2%} {s['vol_ann']:7.2%} "
              f"{r['dd_abs'][p]:+8.2%} {r['traded_per_year'][p]:10.3f} "
              f"{r['n_rebals'][p]:7d} {r['cost_drag_bps'][p]:12.2f}")
    for k, v in r["pairs"].items():
        print(f"  {k:24s} Sharpe diff {v['sharpe_diff']:+.4f}  HAC t(return diff) "
              f"{v['t_diff']:+.2f}  CAGR diff {v['cagr_diff_pp']:+.3f} pp")
    return r


def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    px = data.load_prices()
    common = px.dropna(how="any")
    rets = data.daily_returns(px)
    cash = rets["BIL"]

    print(f"SPY/IEF/GLD/BIL common window: {common.index[0].date()} -> {common.index[-1].date()}"
          f"  n={len(common)}  fp(returns)={data.fingerprint(rets)}"
          f"  fp(levels)={data.fingerprint(common)}")
    print("    (the RETURNS fingerprint is the reproducibility key: adjusted-close LEVELS")
    print("     are rescaled by every new dividend, so their fingerprint churns on refetch")
    print("     while the returns — and every number below — do not)")
    print(f"as-of {data.AS_OF}  |  one-way cost {COST_BPS:.0f} bps  |  band 5/25 "
          f"(min(5pp absolute, 25% relative))")

    r2 = _race_block(rets, *BOOK_2, cash, "Book A — 60/40 SPY/IEF, excess-of-cash")
    r3 = _race_block(rets, *BOOK_3, cash, "Book B — 50/30/20 SPY/IEF/GLD, excess-of-cash")

    print("\n=== block-bootstrap CI on the Sharpe DIFFERENCE (2,000 draws, 21-day blocks) ===")
    for tag, r in (("60/40", r2), ("50/30/20", r3)):
        for a, b in (("bands", "annual"), ("bands", "quarterly"), ("quarterly", "annual")):
            ci = st.bootstrap_diff_ci(r["excess"][a], r["excess"][b], seed=936)
            print(f"  {tag:9s} {a:>9} - {b:<9}: {ci['point']:+.4f}  "
                  f"95% CI [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]  share<0 {ci['frac_negative']:.3f}")

    print(f"\n=== era cut (split {SPLIT}), 60/40 ===")
    for tag, e in st.era_cut(rets[list(BOOK_2[0])], BOOK_2[1], cash=cash,
                             split=SPLIT, cost_bps=COST_BPS).items():
        if e is None:
            continue
        print(f"  {tag:5s} {e['start']} -> {e['end']} (n={e['n_days']}): "
              f"annual {e['sharpe_annual']:+.3f} / quarterly {e['sharpe_quarterly']:+.3f} / "
              f"bands {e['sharpe_bands']:+.3f} / drift {e['sharpe_drift']:+.3f}  |  "
              f"bands-annual {e['diff_bands_annual']:+.4f} (t={e['t_bands_annual']:+.2f})  |  "
              f"traded/yr {e['traded_bands']:.3f} vs {e['traded_annual']:.3f}")

    print("\n=== cost sweep (one-way bps), 60/40 ===")
    for row in st.cost_sweep(rets[list(BOOK_2[0])], BOOK_2[1], cash=cash):
        print(f"  {row['cost_bps']:5.1f} bps: annual {row['sharpe_annual']:+.4f}  "
              f"quarterly {row['sharpe_quarterly']:+.4f}  bands {row['sharpe_bands']:+.4f}  "
              f"| bands-annual {row['diff_bands_annual']:+.4f} (t={row['t_bands_annual']:+.2f})")

    print("\n=== band-width sweep (the 5/25 rule is an ASSUMPTION), 60/40 ===")
    for row in st.band_sweep(rets[list(BOOK_2[0])], BOOK_2[1], cash=cash, cost_bps=COST_BPS):
        print(f"  {row['band_abs']:.0%}/{row['band_rel']:.0%}: Sharpe {row['sharpe_bands']:+.4f}  "
              f"rebals {row['n_rebals']:3d}  traded/yr {row['traded_per_year']:.3f}  "
              f"| vs annual {row['diff_vs_annual']:+.4f} (t={row['t_vs_annual']:+.2f})")

    print("\n=== weight discipline (what the schedule actually buys) ===")
    for p in ("drift", "annual", "quarterly", "bands"):
        w = r2["runs"][p]["weights"]["SPY"]
        print(f"  {p:>10}: SPY weight min {w.min():.3f}  max {w.max():.3f}  "
              f"final {w.iloc[-1]:.3f}  mean |dev from 60%| {(w - 0.60).abs().mean():.4f}")

    print("\n=== like-for-like check: the SAME headline window run GROSS-of-cash ===")
    print("    The cash leg cancels EXACTLY in the daily return difference (so the HAC t is")
    print("    identical either way), but a Sharpe RATIO is nonlinear, so the Sharpe")
    print("    DIFFERENCE is not cash-invariant. This block prints the conversion factor so")
    print("    the gross-of-cash long window below is compared against a gross-of-cash")
    print("    number, never against the excess-of-cash headline.")
    r2_gross = st.race(rets[list(BOOK_2[0])], BOOK_2[1], cash=None, cost_bps=COST_BPS)
    for k in r2["pairs"]:
        ex, gr = r2["pairs"][k], r2_gross["pairs"][k]
        print(f"  {k:24s} excess {ex['sharpe_diff']:+.4f} (t {ex['t_diff']:+.2f})   "
              f"gross {gr['sharpe_diff']:+.4f} (t {gr['t_diff']:+.2f})   "
              f"level shift {gr['sharpe_diff'] - ex['sharpe_diff']:+.4f}")

    print(f"\n=== longer-window cross-check: SPY/IEF from {LONG_START}, GROSS of cash ===")
    print("    (BIL does not exist before 2007, so this window has no tradable cash leg;")
    print("     compare its numbers to the GROSS column printed just above, not to the")
    print("     excess-of-cash headline)")
    long_px = px[["SPY", "IEF"]].dropna(how="any").loc[LONG_START:]
    long_rets = long_px.pct_change().dropna()
    rl = st.race(long_rets, BOOK_2[1], cash=None, cost_bps=COST_BPS)
    print(f"  window {long_rets.index[0].date()} -> {long_rets.index[-1].date()} "
          f"n={len(long_rets)}  fp(returns)={data.fingerprint(long_rets)}"
          f"  fp(levels)={data.fingerprint(long_px)}")
    for p, s in rl["stats"].items():
        print(f"    {p:>10}: Sharpe(gross-of-cash) {s['sharpe']:+.4f}  CAGR {s['cagr']:+.2%}  "
              f"traded/yr {rl['traded_per_year'][p]:.3f}  rebals {rl['n_rebals'][p]}")
    for k, v in rl["pairs"].items():
        print(f"    {k:24s} Sharpe diff {v['sharpe_diff']:+.4f}  HAC t {v['t_diff']:+.2f}")

    print("\n=== synthetic control (machinery proof — never supports the stamp) ===")
    import numpy as np
    for tag, ss in (("planted (mean-reverting)", 1.0), ("null (random walk)", 0.0)):
        diffs, ts = [], []
        for s in range(8):
            prices, _ = data.synthetic_panel(signal_strength=ss, seed=936 + s)
            d = st.synthetic_detect(prices)
            diffs.append(d["diff_bands_annual"])
            ts.append(d["t_bands_annual"])
        diffs, ts = np.array(diffs), np.array(ts)
        print(f"  {tag:26s}: bands-annual Sharpe diff mean {diffs.mean():+.4f} "
              f"(sd {diffs.std(ddof=1):.4f}), HAC t >= 2 in {(ts >= 2).sum()}/8 seeds")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
