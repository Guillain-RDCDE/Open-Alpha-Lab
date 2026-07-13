"""Reproduce the real-data headline run (docs/results.md) — guacamole-bowl seasonality.

    python examples/verify.py            # cache-only (offline)
    python examples/verify.py --fetch    # download daily PEP + SPY + ^IRX from Yahoo, then run

The tradable leg is PEP (Frito-Lay's Super-Bowl chip-and-dip complex — a LABELLED PROXY, because the
pure-play avocado name CVGW is unavailable on the current Yahoo feed); the benchmark is SPY; the cash
leg is the rolled 13-week T-bill (^IRX). The avocado-price seasonal is a hardcoded, cited, APPROXIMATE
proxy — shape only, never a Signal stamp. The sample is pinned to the desk's as-of (quantlab.repro).
"""

from __future__ import annotations

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from guacamole_bowl import data, strategy as st  # noqa: E402
from quantlab import repro  # noqa: E402


def main(fetch: bool) -> None:
    # --- the avocado price seasonal (labelled proxy, shape only) -------------------------------
    av = data.avocado_window_vs_year()
    print("# Avocado wholesale-price seasonal — HARDCODED, CITED, APPROXIMATE (a proxy, shape only)")
    print(f"guac window (Jan-Feb) mean {av['window_mean']:.1f}  vs annual mean {av['year_mean']:.1f}  "
          f"=> gap {av['gap']:+.1f} index pts (a NEGATIVE gap undercuts the 'surge' before the tape)\n")

    d = data.fetch_data(fetch=fetch)
    if d.empty:
        print("No cached real data. Re-run with --fetch (needs network).")
        return
    d = repro.as_of(d)
    rf = d["tbill"]

    print(f"Guacamole-bowl seasonality, {d.index.min().date()}..{d.index.max().date()} "
          f"({len(d)} months, PEP/SPY daily closes resampled to month-end)\n")

    ms = st.month_stats(d["pep"])
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    print("PEP per-month mean returns (one-sample t vs 0; naive and HAC):")
    for m in range(1, 13):
        row = ms.loc[m]
        star = "  <- guac window" if m in st.GUAC_MONTHS else ""
        print(f"  {month_names[m-1]:4s}: mean {row['mean']*100:+6.2f}%  "
              f"t={row['tstat']:+.2f}  t_HAC={row['tstat_hac']:+.2f}  n={int(row['n'])}{star}")

    ws = st.window_spread_tstat(d["pep"])
    print(f"\nGuac window (Jan-Feb) mean: {ws['window_mean']*100:+.2f}%  n={ws['n_window']}")
    print(f"Rest of year mean:          {ws['rest_mean']*100:+.2f}%  n={ws['n_rest']}")
    print(f"Window vs rest spread:      {ws['spread']*100:+.2f}%  t={ws['tstat']:+.2f}  (WRONG SIGN)")

    pb = st.placebo_pairs(d["pep"])
    print(f"\nPlacebo across all {pb['n_pairs']} month-pairs: Jan-Feb spread {pb['thesis_spread']*100:+.2f}% "
          f"ranks {pb['rank']}/{pb['n_pairs']} (1=lowest, pct {pb['pct']*100:.0f}%), z={pb['z']:+.2f}")
    print(f"  most POSITIVE pairs (off-thesis): {pb['most_positive']}")

    ci = st.spread_bootstrap_ci(d["pep"], n_boot=5000, seed=723)
    print(f"Block-bootstrap 95% CI on window spread: [{ci['lo']*100:.2f}%, {ci['hi']*100:.2f}%]  "
          f"(point {ci['point']*100:.2f}%, n_boot={ci['n_boot']})")

    timer = st.seasonal_timer(d["pep"], tbill=rf)
    net = st.apply_costs(timer, n_trades_per_year=2, cost_bps_one_way=5)
    bh_pep = st.buy_hold(d["pep"])
    bh_spy = st.buy_hold(d["spy"])
    print("\nJan-Feb timer (long PEP Jan-Feb, T-bill otherwise) vs buy-and-hold. Sharpe = excess of T-bill:")
    for nm, r in [("guac timer (gross)", timer), ("guac timer (net 5bp/leg)", net),
                  ("buy & hold PEP", bh_pep), ("buy & hold SPY", bh_spy)]:
        s = st.summary(r, rf=rf)
        print(f"  {nm:26s}  CAGR {s['cagr']:+.1%}  Sharpe {s['sharpe']:+.2f}  maxDD {s['max_drawdown']:.0%}")

    nw = st.newey_west_alpha_t(d["pep"], d["spy"], lags=6)
    print(f"\nPEP vs SPY (Newey-West 6-lag): alpha {nw['alpha_ann']*100:+.2f}%/yr  beta {nw['beta']:.2f}  "
          f"t={nw['t_alpha']:+.2f}  (|t|<2 -> no harvestable alpha, and wrong instrument anyway)")

    print(f"\n{repro.data_stamp('PEP/SPY monthly', d)}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
