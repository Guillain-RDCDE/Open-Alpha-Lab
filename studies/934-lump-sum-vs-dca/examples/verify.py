"""Real-tape verification — Study 934 (Lump Sum vs DCA). Prints the run behind docs/results.md.

Reads cached SPY / IEF / BIL daily total-return closes from the shared desk cache and
rolls the twelve-month lump-sum vs DCA race over every start month: win rate, mean and
median terminal-wealth gap, the full dispersion, the overlap-corrected HAC *t*, a
block-bootstrap CI, the **exposure-matched** race (the test that says whether the gap
is timing or just beta), the conditional cuts (starting valuation stretch, starting
drawdown), an era cut, a cost sweep, a tranche sweep, the zero-cash variant, the
long-history extension and the bond-heavy variant. Network only on ``--fetch``.
``docs/results.md`` is written by hand from this output, not by this script.

    python studies/934-lump-sum-vs-dca/examples/verify.py            # cache-only
    python studies/934-lump-sum-vs-dca/examples/verify.py --fetch    # refresh the tapes

``studies/_cache`` is SHARED with every other study on the desk, and other studies
re-pull the same tickers with their own start dates. The long-history extension is
therefore floored at ``LONG_START`` rather than at "whatever SPY history happens to be
in the cache today", so the number reproduces whether the cached SPY tape begins in
1993 or in 2000. The printed data stamp names the cache's own first date so any
mismatch is visible rather than silent.
"""

from __future__ import annotations

import os
import sys

import numpy as np  # noqa: F401
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from lump_vs_dca import data, strategy as st  # noqa: E402

COST_BPS = 1.0
# Floor for the long-history (0%-cash) extension. Pinned so the number does not move
# when another study refreshes the SHARED SPY parquet with a different start date.
LONG_START = "2000-01-03"


def show(tag: str, s: dict) -> None:
    print(f"\n=== {tag} ===")
    print(f"  windows {s['n_windows']}  |  lump-sum win rate {s['win_rate']:.1%} "
          f"[{s['win_ci_low']:.1%}, {s['win_ci_high']:.1%}]")
    print(f"  gap (lump - DCA), cents per $1: mean {s['mean_gap_cents']:+.2f}  "
          f"median {s['median_gap_cents']:+.2f}  sd {s['sd_gap_cents']:.2f}")
    print(f"  dispersion of the gap: p5 {s['p05_gap_cents']:+.2f}  p95 {s['p95_gap_cents']:+.2f}  "
          f"worst {s['worst_gap_cents']:+.2f}  best {s['best_gap_cents']:+.2f}")
    print(f"  HAC t (12 lags) {s['t_hac']:+.2f}  |  non-overlapping t {s['t_nonoverlap']:+.2f}  |  "
          f"boot 95% CI [{s['boot_ci_low']:+.2f}, {s['boot_ci_high']:+.2f}] "
          f"(share<0 {s['boot_frac_negative']:.3f})")
    print(f"  terminal wealth: lump mean {s['mean_w_lump']:+.2%} worst {s['worst_w_lump']:+.2%} "
          f"sd {s['sd_w_lump']:.4f}  |  DCA mean {s['mean_w_dca']:+.2%} worst {s['worst_w_dca']:+.2%} "
          f"sd {s['sd_w_dca']:.4f}  |  dispersion ratio {s['dispersion_ratio']:.3f}")


def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    px = data.load_prices()
    bil = px["BIL"].dropna()
    print(f"shared cache depth: SPY from {px['SPY'].dropna().index[0].date()}, "
          f"IEF from {px['IEF'].dropna().index[0].date()}, BIL from {bil.index[0].date()} "
          f"(long-history extension floored at {LONG_START})")

    for asset_tk in ("SPY", "IEF"):
        a = px[asset_tk].dropna()
        common = a.index.intersection(bil.index)
        a, c = a.loc[common], bil.loc[common]
        print(f"\n################ {asset_tk} vs BIL ################")
        print(f"{common[0].date()} -> {common[-1].date()}  n={len(common)}  "
              f"fp={data.fingerprint(px[[asset_tk, 'BIL']].loc[common])}  as-of {data.AS_OF}")

        res = st.race(a, c, cost_bps=COST_BPS)
        exp = res.pop("exp")
        show(f"{asset_tk}: 12-month lump sum vs 12 monthly tranches (1 bp one-way)", res)
        print(f"  first start {exp.index[0].date()} -> last end {exp['end'].iloc[-1].date()}")

        # Is the gap TIMING or just EXPOSURE? Race DCA against a static portfolio
        # holding DCA's own average exposure (analytic (n+1)/2n, no hindsight).
        em = st.exposure_matched_race(exp)
        dm = st.exposure_matched_race(exp, weight=st.dispersion_matched_weight(exp))
        print(f"\n=== {asset_tk}: exposure-matched race (the beta control) ===")
        print(f"  risk-adjusted, excess of the SAME cash leg: lump {em['mean_excess_lump']:+.2%} "
              f"per unit dispersion {em['reward_disp_lump']:.3f}  |  DCA {em['mean_excess_dca']:+.2%} "
              f"per unit dispersion {em['reward_disp_dca']:.3f}")
        for tag, d in [("analytic exposure weight (no hindsight)", em),
                       ("dispersion-matched weight (IN-SAMPLE fit)", dm)]:
            print(f"  {tag}: w={d['weight']:.4f}  mean gap {d['mean_gap_cents']:+.2f}c  "
                  f"win {d['win_rate']:.1%}  HAC t {d['t_hac']:+.2f}  "
                  f"non-overlap t {d['t_nonoverlap']:+.2f}  "
                  f"boot CI [{d['boot_ci_low']:+.2f}, {d['boot_ci_high']:+.2f}]  "
                  f"(sd matched {d['sd_matched']:.4f} vs DCA {d['sd_dca']:.4f})")

        if asset_tk != "SPY":
            continue

        print("\n=== conditional cuts (SPY) ===")
        for tag, row in st.conditional_cuts(exp).items():
            if row.get("n", 0) < 5:
                print(f"  {tag:16s}: n={row.get('n', 0)} (too few)")
                continue
            print(f"  {tag:16s}: n={row['n']:3d}  win {row['win_rate']:.1%}  "
                  f"mean gap {row['mean_gap_cents']:+.2f}c  median {row['median_gap_cents']:+.2f}c  "
                  f"t={row['t_hac']:+.2f}  mean 12m asset return {row['mean_asset_ret']:+.1%}")

        print("\n=== era cut (split 2017-01-01, by start month) ===")
        for tag, row in st.era_cut(exp).items():
            print(f"  {tag:6s}: n={row['n']:3d}  win {row['win_rate']:.1%}  "
                  f"mean gap {row['mean_gap_cents']:+.2f}c  t={row['t_hac']:+.2f}")

        print("\n=== cost sweep (one-way bps; DCA pays it 12x, lump once) ===")
        for row in st.cost_sweep(a, c):
            print(f"  {row['cost_bps']:5.1f} bps: win {row['win_rate']:.1%}  "
                  f"mean gap {row['mean_gap_cents']:+.2f}c  t={row['t_hac']:+.2f}")

        print("\n=== tranche sweep (how long you take to average in) ===")
        for row in st.tranche_sweep(a, c):
            print(f"  {row['n_tranches']:2d} tranches: n={row['n_windows']:3d}  "
                  f"win {row['win_rate']:.1%}  mean gap {row['mean_gap_cents']:+.2f}c  "
                  f"t={row['t_hac']:+.2f}  dispersion ratio {row['dispersion_ratio']:.3f}")

        print("\n=== fixed-ticket sweep (ASSUMPTION: $10,000 windfall, DCA pays 12 tickets) ===")
        for row in st.ticket_sweep(a, c):
            print(f"  ${row['ticket_usd']:5.2f}/trade: adds {row['extra_gap_cents']:+.2f}c to the gap "
                  f"-> win {row['win_rate']:.1%}  mean gap {row['mean_gap_cents']:+.2f}c  t={row['t_hac']:+.2f}")

        z = st.zero_cash_variant(a, c)
        print("\n=== zero-cash variant (ASSUMPTION: idle cash earns 0%) ===")
        print(f"  win {z['win_rate']:.1%}  mean gap {z['mean_gap_cents']:+.2f}c  t={z['t_hac']:+.2f}"
              f"   (vs {res['mean_gap_cents']:+.2f}c with real BIL)")

        # Long-history check. BIL only exists from 2007, so extending the sample back
        # forces the 0% cash ASSUMPTION on the whole window. Floored at LONG_START so
        # the read does not depend on how deep the shared cache happens to be today.
        spy_full = px["SPY"].dropna()
        spy_full = spy_full[spy_full.index >= pd.Timestamp(LONG_START)]
        flat = pd.Series(1.0, index=spy_full.index)
        exp_long = st.run_experiment(spy_full, flat, cost_bps=COST_BPS)
        s_long = st.summarise(exp_long, n_boot=1000)
        print(f"\n=== long-history check: SPY {exp_long.index[0].date()} -> "
              f"{exp_long['end'].iloc[-1].date()} (0% cash ASSUMPTION, floored at {LONG_START}) ===")
        print(f"  windows {s_long['n_windows']}  win {s_long['win_rate']:.1%}  "
              f"mean gap {s_long['mean_gap_cents']:+.2f}c  median {s_long['median_gap_cents']:+.2f}c  "
              f"HAC t {s_long['t_hac']:+.2f}  non-overlap t {s_long['t_nonoverlap']:+.2f}")
        for tag, row in st.era_cut(exp_long, split="2010-01-01").items():
            print(f"    {tag:6s}: n={row['n']:3d}  win {row['win_rate']:.1%}  "
                  f"mean gap {row['mean_gap_cents']:+.2f}c  t={row['t_hac']:+.2f}")
        print("    decade by decade (start month):")
        for lo, hi in (("2000-01-01", "2010-01-01"), ("2010-01-01", "2020-01-01"),
                       ("2020-01-01", "2030-01-01")):
            m = (exp_long.index >= pd.Timestamp(lo)) & (exp_long.index < pd.Timestamp(hi))
            row = st._cut_row(exp_long[m], 12)
            if row.get("n", 0) >= 5:
                print(f"      {lo[:4]}s: n={row['n']:3d}  win {row['win_rate']:.1%}  "
                      f"mean gap {row['mean_gap_cents']:+.2f}c  t={row['t_hac']:+.2f}  "
                      f"mean 12m SPY {row['mean_asset_ret']:+.1%}")
        em_long = st.exposure_matched_race(exp_long)
        print(f"    exposure-matched on the long tape: {em_long['mean_gap_cents']:+.2f}c  "
              f"HAC t {em_long['t_hac']:+.2f}  "
              f"boot CI [{em_long['boot_ci_low']:+.2f}, {em_long['boot_ci_high']:+.2f}]")

    print("\n=== synthetic control, 12 seeds each (machinery proof; never supports the stamp) ===")
    for ss, label in [(1.0, "planted premium"), (0.0, "null"), (-1.0, "falling market")]:
        d = st.synthetic_control(ss)
        print(f"  ss={ss:+.1f} ({label:16s}): mean gap {d['mean_gap_cents']:+.2f}c "
              f"(sd {d['sd_gap_cents']:.2f}, se {d['se_gap_cents']:.2f})  "
              f"mean win rate {d['mean_win_rate']:.1%}  "
              f"lump wins on {d['frac_seeds_lump_wins']*100:.0f}% of seeds")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
