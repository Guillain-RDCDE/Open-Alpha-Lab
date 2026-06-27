# -*- coding: utf-8 -*-
"""Real-tape verification -- Study 531 (Enterprise-Multiple). Prints every number.

Runs the monthly EV/EBITDA long-cheap/short-expensive sort on a fixed large-cap survivor
basket (yfinance fundamentals + prices, cached under ../_cache/). Prints the hedge stats,
the label-shuffle placebo p-value, gross vs net (costs + borrow), the monthly IC, and the
seed-averaged synthetic positive control. These are the numbers mirrored in docs/results.md.

    python studies/531-enterprise-multiple/examples/verify.py            # cache-only
    python studies/531-enterprise-multiple/examples/verify.py --fetch    # (re)download tape
"""

from __future__ import annotations

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from enterprise_multiple import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    em, fwd = data.fetch_panel(fetch=fetch)
    if em.empty:
        print("No cached data found. Re-run with --fetch to download the tape.")
        return

    print(f"EV/EBITDA panel: {em.shape}  months {em.index.min()}-{em.index.max()}")
    print(f"Names used: {len(em.columns)} of {len(data.BASKET)} requested")
    print(f"Fingerprints: EM={data.fingerprint(em)}  fwd={data.fingerprint(fwd)}")
    print()

    # --- Long-short hedge ---
    h = st.long_short_hedge(em, fwd)
    s = st.summary(h["hedge"])
    s_long = st.summary(h["long"])
    s_short = st.summary(h["short"])
    s_mkt = st.summary(h["market"])
    print("=== Monthly long-cheap / short-expensive hedge (low EV/EBITDA = long) ===")
    print(f"  months used:   {s['n']}")
    print(f"  long  (cheap): mean={s_long['mean']:+.3f}/yr  Sharpe={s_long['sharpe']:+.2f}")
    print(f"  short (exp.):  mean={s_short['mean']:+.3f}/yr  Sharpe={s_short['sharpe']:+.2f}")
    print(f"  market (EW):   mean={s_mkt['mean']:+.3f}/yr  Sharpe={s_mkt['sharpe']:+.2f}")
    print(f"  HEDGE:         mean={s['mean']:+.3f}/yr  vol={s['vol']:.3f}  Sharpe={s['sharpe']:+.2f}")
    print(f"                 t_simple={s['t_simple']:+.2f}  t_HAC={s['t_hac']:+.2f}  "
          f"hit={s['hit_rate']:.0%}  maxDD={s['max_drawdown']:.1%}")
    print(f"  avg monthly turnover (both legs): {h['turnover'].mean():.3f}")

    # --- Placebo ---
    pl = st.placebo_null(em, fwd, n_shuffles=500)
    print()
    print("=== Placebo (within-month signal shuffle, 500 draws) ===")
    print(f"  real |t_HAC| = {abs(pl['real_t']):.2f}   placebo p-value = {pl['p_value']:.3f}  "
          f"(n_valid={pl['n_shuffles']})")

    # --- Costs ---
    hc = st.apply_costs(h, one_way_bps=5.0, borrow_bps_annual=50.0)
    s_net = st.summary(hc["net"])
    print()
    print("=== Costs: 5 bps/leg x turnover + 50 bps/yr borrow on shorts ===")
    print(f"  gross hedge: mean={s['mean']:+.3f}/yr  t_HAC={s['t_hac']:+.2f}")
    print(f"  net hedge:   mean={s_net['mean']:+.3f}/yr  t_HAC={s_net['t_hac']:+.2f}  "
          f"Sharpe={s_net['sharpe']:+.2f}")
    print(f"  avg monthly cost: {hc['cost'].mean()*1e4:.1f} bps")

    # --- IC ---
    ic = st.monthly_ic(em, fwd)
    s_ic = st.summary(ic)
    print()
    print("=== Monthly IC: Spearman(-EV/EBITDA, next-month return) ===")
    print(f"  mean IC={ic.mean():+.4f}  t_HAC={s_ic['t_hac']:+.2f}  months={len(ic)}")

    # --- Synthetic positive control (seed-averaged) ---
    print()
    print("=== Synthetic positive control (seed-averaged HAC t over 20 seeds) ===")
    ctrl = st.synthetic_control()
    for prem, row in ctrl.iterrows():
        print(f"  premium={prem:.3f}  mean={row['mean_pct_yr']:+.2f}%/yr  "
              f"t_HAC(mean of {int(row['n_seeds'])} seeds)={row['t_hac_mean']:+.2f}")

    print()
    print("--- HEADLINE for docs/results.md ---")
    print(f"hedge_mean_pct={s['mean']*100:+.2f}  hedge_t_hac={s['t_hac']:+.2f}  "
          f"placebo_p={pl['p_value']:.3f}  net_mean_pct={s_net['mean']*100:+.2f}  "
          f"net_t={s_net['t_hac']:+.2f}  ic_mean={ic.mean():+.4f}  ic_t={s_ic['t_hac']:+.2f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
