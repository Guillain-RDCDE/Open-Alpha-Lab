"""Real-tape verification — Study 921 (Bill Ladder vs ETF). Regenerates docs/results.md.

Reads the cached ^IRX 13-week bill quote and the three cash ETFs (BIL, SGOV, SHV), builds
a rolling 13-rung held-to-maturity 3-month bill ladder, and races it against each fund's
total return. Prints the annualised gap in bps/yr with naive and HAC *t*, the block
bootstrap CI, the fee attribution, the era and rate-regime cuts, the discount-basis check,
the rung-count check, and the two friction sweeps. Network only on ``--fetch``.

    python studies/921-bill-ladder-vs-etf/examples/verify.py            # cache-only
    python studies/921-bill-ladder-vs-etf/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import numpy as np  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from bill_ladder import data, strategy as st  # noqa: E402

PRIMARY = "BIL"


def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    px = data.load_prices()
    irx = px[data.RATE_SYMBOL].dropna()
    etf = px[PRIMARY].dropna()

    head = st.race(irx, etf)
    fp = data.fingerprint(px[[PRIMARY, data.RATE_SYMBOL]].loc[head["frame"].index])
    print(f"as-of {data.AS_OF}   ladder = 13 rungs x 91-day bills, weekly roll, one-day lag")
    print(f"{PRIMARY} vs ladder: {head['start'].date()} -> {head['end'].date()}  "
          f"n={head['n_days']}  rolls={head['n_rolls']}  fp={fp}")

    print("\n=== the race (total return vs total return; cash is the numeraire) ===")
    for tk in data.ETFS:
        r = st.race(irx, px[tk].dropna())
        print(f"  {tk:5s} {r['start'].date()}->{r['end'].date()} n={r['n_days']:5d}  "
              f"ladder CAGR {r['cagr_ladder']:.4%} vs ETF {r['cagr_etf']:.4%}  "
              f"gap {r['gap_bps']:+6.2f} bps/yr  HAC t {r['t_hac']:+.2f}  "
              f"(naive t {r['t_naive']:+.2f})")
    print(f"  vol: ladder {head['vol_ladder']:.4%} vs {PRIMARY} {head['vol_etf']:.4%} "
          f"-- the ladder's calm is amortised-cost accounting, NOT less risk")

    print("\n=== is the gap the fee? (gross-of-fee attribution; ER is a PROXY) ===")
    for tk in data.ETFS:
        a = st.fee_attribution(irx, px[tk].dropna(), data.EXPENSE_RATIO_BPS[tk])
        print(f"  {tk:5s} ladder {a['cagr_ladder_bps']:7.1f} bps/yr vs ETF net "
              f"{a['cagr_etf_bps']:7.1f} + ER {a['expense_ratio_bps']:5.2f} = gross "
              f"{a['gross_etf_bps']:7.1f}  ->  residual {a['residual_bps']:+6.1f} bps/yr")

    print(f"\n=== block bootstrap CI on the gap vs {PRIMARY} (2000 draws, 21-day blocks) ===")
    ci = st.bootstrap_gap_ci(head["diff"], seed=921)
    print(f"  gap {ci['gap_bps']:+.2f} bps/yr  95% CI [{ci['ci_low']:+.2f}, {ci['ci_high']:+.2f}]  "
          f"share<0 {ci['frac_negative']:.4f}")
    print("  block-length sensitivity (the bootstrap's own knob):")
    for b in (5, 10, 21, 63):
        c = st.bootstrap_gap_ci(head["diff"], block=b, seed=921)
        print(f"    block {b:3d}d  CI [{c['ci_low']:+6.2f}, {c['ci_high']:+6.2f}]  "
              f"share<0 {c['frac_negative']:.4f}")

    # ------------------------------------------------------------------ #
    # The inference audit. HAC RAISES this study's t (+2.75 vs a naive +1.15), so the
    # bandwidth cannot be left unexamined and the significance cannot rest on a knob.
    # ------------------------------------------------------------------ #
    print("\n=== inference audit: is the significance real or is it the bandwidth? ===")
    print(f"  lag-1 autocorrelation of the daily difference: {head['acf1']:+.3f}  "
          "(negative = Roll-1984 bid-offer bounce in the ETF close -> naive SE too big)")
    print("  HAC t vs bandwidth (disclosed because HAC helps us here):")
    for row in st.hac_bandwidth_scan(head["diff"]):
        print(f"    {row['label']:14s} lags {row['lags']:4d}  t {row['t']:+.2f}")
    print("  NON-OVERLAPPING period sums -- ordinary t, no bandwidth, no block length:")
    for row in st.horizon_check(head["diff"]):
        print(f"    {row['freq']:2s}  n={row['n_periods']:4d} periods  "
              f"mean {row['mean_bps']:+.3f} bps/period  t {row['t']:+.2f}")
    print("    -> the knob-free test agrees with HAC, not with the naive t. THIS is the")
    print("       evidence the Real stamp rests on; HAC and the bootstrap merely concur.")
    print("  same test on the CONSERVATIVE raw-quote convention:")
    raw_diff = st.race(irx, etf, basis="raw")["diff"]
    for row in st.horizon_check(raw_diff):
        print(f"    {row['freq']:2s}  n={row['n_periods']:4d} periods  t {row['t']:+.2f}")

    print(f"\n=== era cut vs {PRIMARY} ===")
    for tag, e in st.era_cut(irx, etf).items():
        if e is None:
            continue
        print(f"  {tag:26s} n={e['n_days']:5d} {e['start'].date()}->{e['end'].date()}  "
              f"mean 13w quote {e['mean_rate_pct']:.2f}%  gap {e['gap_bps']:+6.2f} "
              f"(HAC t={e['t_hac']:+.2f}, knob-free monthly t={e['t_month']:+.2f})")
    print("  NB: the sign is positive in all three, but only eras 2 and 3 are individually")
    print("      significant -- era 1 (2007-2015) is directionally right and nothing more.")

    print("\n=== rate-regime cut (13-week quote below / above 1.00%) ===")
    for tag, e in st.rate_regime_cut(irx, etf).items():
        if e is None:
            continue
        print(f"  {tag:12s} n={e['n_days']:5d}  mean quote {e['mean_rate_pct']:.2f}%  "
              f"gap {e['gap_bps']:+6.2f} (t={e['t_hac']:+.2f})")

    print("\n=== rate-convention check (the discount -> bond-equivalent ASSUMPTION) ===")
    for basis, b in st.basis_check(irx, etf).items():
        print(f"  {basis:9s} ladder CAGR {b['cagr_ladder']:.4%}  gap {b['gap_bps']:+6.2f} "
              f"(t={b['t_hac']:+.2f})")

    print("\n=== rung-count check ===")
    for row in st.rung_check(irx, etf):
        print(f"  {row['n_rungs']:3d} rungs ({row['n_rolls']:4d} rolls)  gap {row['gap_bps']:+6.2f} "
              f"(t={row['t_hac']:+.2f})")

    print("\n=== per-auction friction sweep (PROXY; ~4x cost_bps of annual drag) ===")
    for row in st.friction_sweep(irx, etf):
        print(f"  {row['cost_bps']:5.1f} bps/roll  gap {row['gap_bps']:+6.2f} "
              f"(t={row['t_hac']:+.2f})  ladder CAGR {row['cagr_ladder']:.4%}")

    print("\n=== reinvestment idle-day sweep (PROXY) ===")
    for row in st.idle_sweep(irx, etf):
        print(f"  {row['idle_days']:4.1f} idle days  gap {row['gap_bps']:+6.2f} "
              f"(t={row['t_hac']:+.2f})")

    print("\n=== synthetic control (machinery proof; never supports the stamp) ===")
    pl = np.array([st.synthetic_detect(
        data.synthetic_daily(signal_strength=1.0, seed=921 + s)[0])["gap_bps"] for s in range(8)])
    nl = np.array([st.synthetic_detect(
        data.synthetic_daily(signal_strength=0.0, seed=921 + s)[0])["gap_bps"] for s in range(8)])
    print(f"  planted 13.5 bps fee x8: recovered {pl.mean():+.2f} (sd {pl.std(ddof=1):.2f})")
    print(f"  free-ETF null       x8: recovered {nl.mean():+.2f} (sd {nl.std(ddof=1):.2f}), "
          f"|gap|>=4 bps in {(np.abs(nl) >= 4).sum()}/8")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
