"""Real-tape verification — Study 922 (Floating-Rate Front End). Regenerates docs/results.md.

Reads cached USFR, TFLO, BIL, SHY total-return closes and the ^IRX 13-week bill rate,
races the floating-rate funds against fixed bills and 1-3y Treasuries, cuts the race by
the direction of ^IRX and by the rate cycle, and prints the drawdown table, the
bootstrap CIs and every sweep. Network only on ``--fetch``.

    python studies/922-frn-vs-fixed-front-end/examples/verify.py            # cache-only
    python studies/922-frn-vs-fixed-front-end/examples/verify.py --fetch    # refresh
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from frn_front import data, strategy as st  # noqa: E402
from quantlab import repro  # noqa: E402

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)


def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    px = data.load_prices()
    rets = st.daily_returns(px)
    irx = px["IRX"]
    cash = st.cash_leg(irx)

    print(repro.data_stamp("USFR/TFLO/BIL/SHY + ^IRX", px, asof=data.AS_OF))
    print(f"raw-array fingerprint {data.fingerprint(px)}  |  total-return closes "
          f"(auto_adjust=True); ^IRX is a price-only yield index, not investable")

    print("\n=== per-fund levels (excess-of-cash vs the ^IRX/252 accrual proxy) ===")
    lvl = st.level_table(rets, cash)
    for c, row in lvl.iterrows():
        print(f"  {c:5s} total return {row['ann_return']*100:+.2f}%/yr  vol {row['vol_ann']*100:.2f}%  "
              f"maxDD {row['max_drawdown']*100:+.2f}%  |  excess-of-cash {row['excess_ann']*100:+.3f}%/yr  "
              f"Sharpe {row['excess_sharpe']:+.2f} (HAC t {row['excess_t']:+.2f})")

    print("\n=== the pairwise race (annualised difference, %; HAC t) ===")
    tbl = st.race_table(rets)
    for pair, row in tbl.iterrows():
        print(f"  {pair:12s} {row['ann_diff_gross']*100:+.3f}%/yr  t={row['tstat']:+.2f}  "
              f"tracking vol {row['vol_ann']*100:.2f}%  hit {row['hit_rate']:.1%}")

    print("\n=== liquidity-era cut (USFR/TFLO launched Feb-2014 and barely traded until ~2018) ===")
    for lbl, sub in [("2014-2017", rets[rets.index < st.LIQUIDITY_SPLIT]),
                     ("2018-2026", rets[rets.index >= st.LIQUIDITY_SPLIT])]:
        r = st.race_table(sub)
        print(f"  {lbl}: " + "  ".join(
            f"{p} {r.loc[p, 'ann_diff_gross']*100:+.3f}% (t={r.loc[p, 'tstat']:+.2f})"
            for p in ("USFR-BIL", "TFLO-BIL", "USFR-SHY")))

    print("\n=== regime cut on the direction of ^IRX (63-day change, +/-0.25 pp dead band) ===")
    reg = st.irx_regime(irx)
    rt = st.regime_table(rets, reg)
    for r_, row in rt.iterrows():
        print(f"  {r_:8s} n={int(row['n_days']):5d}  "
              f"USFR {row['ret_USFR']:+.2f}%  BIL {row['ret_BIL']:+.2f}%  SHY {row['ret_SHY']:+.2f}%  |  "
              f"USFR-SHY {row['USFR-SHY']:+.2f}% (t={row['t_USFR-SHY']:+.2f})  "
              f"USFR-BIL {row['USFR-BIL']:+.2f}% (t={row['t_USFR-BIL']:+.2f})")

    print("\n=== headline test: does the rate direction flip the ranking? (HAC-OLS contrast) ===")
    for a, b in [("USFR", "SHY"), ("BIL", "SHY"), ("USFR", "BIL")]:
        c = st.regime_contrast(rets, a, b, reg)
        print(f"  {a}-{b}: flat {c['flat_diff']:+.2f}% (t={c['flat_t']:+.2f})  "
              f"rising +{c['rising_extra']:.2f} (t={c['rising_t']:+.2f})  "
              f"falling {c['falling_extra']:+.2f} (t={c['falling_t']:+.2f})  ->  "
              f"contrast {c['contrast']:+.2f} pp/yr (t={c['contrast_t']:+.2f})")

    print("\n=== rate-cycle windows (declared ASSUMPTION: a hardcoded Fed calendar) ===")
    ct = st.cycle_table(rets)
    for w, row in ct.iterrows():
        print(f"  {w:20s} n={int(row['n_days']):4d}  USFR {row['ret_USFR']:+.2f}%  "
              f"TFLO {row['ret_TFLO']:+.2f}%  BIL {row['ret_BIL']:+.2f}%  SHY {row['ret_SHY']:+.2f}%  |  "
              f"USFR-SHY {row['USFR-SHY']:+.2f}% (t={row['t_USFR-SHY']:+.2f})  "
              f"USFR-BIL {row['USFR-BIL']:+.2f}% (t={row['t_USFR-BIL']:+.2f})")

    print("\n=== drawdown table (absolute, lived) ===")
    print(st.drawdown_table(rets).round(2))

    print("\n=== block-bootstrap CIs on the annualised difference (2,000 draws, 21-day blocks) ===")
    for a, b in [("USFR", "BIL"), ("USFR", "SHY"), ("BIL", "SHY")]:
        ci = st.block_bootstrap_ci(rets[a] - rets[b])
        print(f"  {a}-{b}: {ci['point']*100:+.3f}%/yr  95% CI [{ci['ci_low']*100:+.3f}, "
              f"{ci['ci_high']*100:+.3f}]  share<0 {ci['frac_negative']:.1%}")

    print("\n=== cost sweep (round-trip spread on a HELD sleeve, amortised) ===")
    print(st.cost_sweep(rets, "USFR", "SHY").round(3).to_string(index=False))

    print("\n=== borrow sweep (the same difference read as a long/short pair) ===")
    print(st.borrow_sweep(rets, "USFR", "SHY").round(3).to_string(index=False))

    print("\n=== cash-proxy sweep (does the convention change the ranking?) ===")
    print(st.cash_proxy_sweep(rets, irx, bil=rets["BIL"]).round(3))

    print("\n=== regime-classifier sweep (window / dead band) ===")
    print(st.regime_param_sweep(rets, irx, "USFR", "SHY").round(3).to_string(index=False))

    print("\n=== duration attribution: USFR-SHY gap vs D x d(rate) "
          "(ASSUMPTION: SHY duration ~1.85y) ===")
    print(st.duration_attribution(rets, irx).round(2))

    print("\n=== synthetic control (machinery proof only - never supports the stamp) ===")
    pl = st.synthetic_detect(data.synthetic_panel(signal_strength=1.0, seed=922)[0])
    print(f"  planted (fixed leg has 1.85y duration): contrast {pl['contrast']:+.2f} pp/yr "
          f"(t={pl['contrast_t']:+.1f}); rising +{pl['rising_extra']:.2f}, "
          f"falling {pl['falling_extra']:+.2f}")
    nulls = np.array([st.synthetic_detect(
        data.synthetic_panel(signal_strength=0.0, seed=s)[0])["contrast"] for s in range(922, 930)])
    tn = np.array([st.synthetic_detect(
        data.synthetic_panel(signal_strength=0.0, seed=s)[0])["contrast_t"] for s in range(922, 930)])
    print(f"  null x8 (fixed leg has no duration): contrast mean {nulls.mean():+.2f} "
          f"(sd {nulls.std(ddof=1):.2f}), |t|>=2 in {(np.abs(tn) >= 2).sum()}/8")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
