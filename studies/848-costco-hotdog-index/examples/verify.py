"""Reproducible headline run for Study 848 — "Costco Hot-Dog Index" (COST vs CPI vs SPY).

Prints every number quoted in docs/results.md and frozen into
notebooks/build_notebooks.py (the ``R`` dict). COST/SPY/XLP come from the cached month-end
yfinance pull under ``_cache/``; CPI is the **real** FRED/BLS ``CPIAUCSL`` series (BLS
``CUSR0000SA0``) fetched from the BLS public API and cached (embedded real values are the
offline fallback). Deterministic & offline once cached.

    python examples/verify.py            # cache-only (offline)
    python examples/verify.py --fetch    # download COST+SPY+XLP (Yahoo) + CPI (BLS), then run
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

from hotdog_index import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    panel = data.build_panel(asof=data.AS_OF)
    if panel.empty:
        print("No cached market data. Re-run with --fetch (needs network).")
        return
    cpi = data.load_cpi(asof=data.AS_OF)

    print(f"\nCostco Hot-Dog Index — COST vs CPI vs SPY, "
          f"{panel.index.min().date()}..{panel.index.max().date()} ({len(panel)} months)")
    print(f"Fingerprint [cost,spy,cpi] = {data.fingerprint(panel)}  (as-of {data.AS_OF})\n")

    print("# Half 1 — the real-price identity ($1.50 combo, 1985 dollars)")
    er = st.erosion_stats(cpi)
    print(f"  price level since 1985 : {er['cpi_multiple']:.2f}x")
    print(f"  real price today       : ${er['real_now']:.2f}  (eroded {er['eroded_pct']:.1f}%)")
    print(f"  to hold 1985 value     : ${er['inflation_adjusted']:.2f}  (giveaway ${er['giveaway']:.2f}/combo)")

    print("\n# Half 2 — total-return race (total return; CPI as deflator)")
    race = st.total_return_race(panel)
    for k, lab in [("cost", "COST"), ("spy", "SPY "), ("cpi", "CPI ")]:
        r = race[k]
        print(f"  {lab}: {r['mult']:5.1f}x  CAGR {r['cagr']*100:+5.1f}%  real {r['real_mult']:5.1f}x  "
              f"$10k -> ${10000*r['mult']:,.0f}")
    print(f"  ({race['years']:.1f} years)")

    print("\n# Half 3 — inflation-surprise beta (HAC t): is COST a distinctive hedge?")
    sig = st.signal_summary(panel)
    for key, lab in [("cost_contemp", "COST ~ surprise (same month)"),
                     ("spy_contemp", "SPY  ~ surprise"),
                     ("xlp_contemp", "XLP  ~ surprise (staples)"),
                     ("cost_pred", "COST ~ surprise (next month)")]:
        r = sig[key]
        print(f"  {lab:32s}: slope {r['slope']:+.3f}  t_nw {r['t_nw']:+.2f}  R2 {r['r2']*100:.2f}%")
    g = sig["gap"]
    print(f"  COST - XLP gap (beta_COST - beta_staples): slope {g['cost_minus_xlp']['slope']:+.3f}  "
          f"t_nw {g['cost_minus_xlp']['t_nw']:+.2f}")
    print(f"  COST - SPY gap                           : slope {g['cost_minus_spy']['slope']:+.3f}  "
          f"t_nw {g['cost_minus_spy']['t_nw']:+.2f}")

    print("\n# Half 4 — placebo + sub-era cut")
    pl = st.placebo_pvalue(panel, "cost", n_perm=3000)
    print(f"  block-rotation placebo: obs slope {pl['obs_slope']:+.3f}  "
          f"placebo mean {pl['placebo_mean']:+.3f}  two-sided p {pl['p_value']:.3f}")
    print(st.era_cut(panel).to_string())

    print("\n# Half 5 — inflation-timing overlay, net of costs (excess-of-cash Sharpe)")
    tm = st.timer_stats(panel, cost_bps=10.0)
    print(f"  overlay (net 10bp) {tm['timer_sharpe']:.2f} | buy-hold COST {tm['cost_bh_sharpe']:.2f} | "
          f"SPY {tm['spy_sharpe']:.2f}")
    print(f"  ({tm['switches_per_yr']:.1f} switches/yr, invested {tm['invested_frac']*100:.0f}% of months)")

    print("\n# Synthetic positive control — planted inflation beta (edge=8) vs null (edge=0)")
    sm8 = st.synthetic_mean_t(data, edge=8.0, n_seeds=20)
    sm0 = st.synthetic_mean_t(data, edge=0.0, n_seeds=20)
    print(f"  planted: mean slope {sm8['mean_slope']:+.2f}  mean t {sm8['mean_t']:+.2f}  "
          f"gap t {sm8['mean_gap_t']:+.2f}  fires {sm8['fire_frac']*100:.0f}%")
    print(f"  null   : mean slope {sm0['mean_slope']:+.2f}  mean t {sm0['mean_t']:+.2f}  "
          f"fires {sm0['fire_frac']*100:.0f}%")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
