"""Reproducible headline run for Study 885 — Ultra-Short Credit Pickup.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached ETF panel under
``_cache/`` if present (the real-tape numbers), and always runs the synthetic
control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))  # repo root (quantlab)

from ultra_short import data, strategy as st
from quantlab import repro


def main() -> None:
    print("# Ultra-Short Credit Pickup — JPST/ICSH/MINT vs BIL/SHV, yfinance total return")
    if data.have_real():
        px = data.load_prices()                      # sliced to the frozen as-of
        print(repro.data_stamp("ETF panel (total-return closes)", px, asof=data.AS_OF))
        print(f"  fingerprint {repro.fingerprint(px)}")
        rets = st.daily_returns(px)

        # Common sample across the whole panel (starts at the youngest, JPST 2017).
        common = st.align_common(rets, data.TICKERS)
        print(f"\ncommon daily sample (all 5): {len(common)} days, "
              f"{common.index.min():%Y-%m-%d} -> {common.index.max():%Y-%m-%d}")

        print("\n# The reward-per-risk race — annualised EXCESS-of-BIL Sharpe (rf = BIL)")
        for tk in data.TICKERS:
            e = st.excess(common, tk, "BIL")
            print(f"  {tk:<5}: excess Sharpe {st.ann_sharpe(e):+6.2f}  "
                  f"ann ret {st.ann_return(common[tk]):5.2f}%/yr  "
                  f"ann vol {st.ann_vol(common[tk]):5.2f}%")

        print("\n# Sleeve vs cash — the pickup (equal-weight credit sleeve minus BIL)")
        sleeve = common[data.CREDIT].mean(axis=1)
        pickup = (sleeve - common["BIL"]).dropna()
        h = st.hac_mean(pickup)
        print(f"  sleeve - BIL: {h['mean_bps_yr']:+.1f} bps/yr  HAC t = {h['t_nw']:+.2f}  "
              f"(n={h['n']}, lags={h['lags']})")
        print(f"  sleeve excess Sharpe {st.ann_sharpe(pickup):+.2f} vs "
              f"BIL-vs-BIL {0.0:+.2f} (definitionally 0)")
        # each credit name minus BIL
        for tk in data.CREDIT:
            hp = st.hac_mean((common[tk] - common["BIL"]).dropna())
            print(f"    {tk:<5} - BIL: {hp['mean_bps_yr']:+.1f} bps/yr  HAC t = {hp['t_nw']:+.2f}")

        print("\n# Bootstrap CI on the sleeve excess Sharpe (circular block, 2000 draws)")
        ci = st.sharpe_ci(pickup)
        print(f"  sharpe {ci['sharpe']:+.2f}  95% CI [{ci['ci_low']:+.2f}, {ci['ci_high']:+.2f}]  "
              f"frac<0 = {ci['frac_negative']:.3f}  (block {ci['block_size']})")

        print("\n# Excess-vs-excess Sharpe race — sleeve vs SHV (both excess of BIL)")
        shv_ex = st.excess(common, "SHV", "BIL")
        print(f"  SHV  - BIL: {st.hac_mean(shv_ex)['mean_bps_yr']:+.1f} bps/yr  "
              f"excess Sharpe {st.ann_sharpe(shv_ex):+.2f}")
        print(f"  sleeve-BIL: {h['mean_bps_yr']:+.1f} bps/yr  excess Sharpe {st.ann_sharpe(pickup):+.2f}")

        print("\n# Sub-eras — pre/post 2019 (is the pickup stable?)")
        ec = st.era_cut(pickup, "2019-01-01")
        for tag, r in (("<2019", ec["early"]), (">=2019", ec["late"])):
            print(f"  {tag:>7}: {r['mean_bps_yr']:+.1f} bps/yr  HAC t = {r['t_nw']:+.2f}  (n={r['n']})")
        print(f"  Welch t (early vs late daily): {ec['welch_t']:+.2f}")

        print("\n# Drawdowns — daily total-return, common sample")
        for tk in ["BIL"] + data.CREDIT:
            dd = st.max_drawdown(px[tk].loc[common.index.min():])
            print(f"  {tk:<5}: max DD {dd['depth_pct']:6.2f}%  (peak {dd['peak']} -> trough {dd['trough']})")

        print("\n# Calendar years — total return %/yr (2020 COVID & 2022 hikes in view)")
        cyt = st.calendar_year_table(rets, ["BIL"] + data.CREDIT)
        print(cyt.round(2).to_string())

        print("\n# Costed sleeve — one-way spread x NAV x turnover/yr, long-only (no borrow)")
        for cost in (1.0, 2.0, 5.0):
            nc = st.net_of_cost_excess(pickup, cost_bps_oneway=cost, turnover_yr=1.0)
            print(f"  {cost:>4.1f} bp one-way x1/yr: gross {nc['gross_bps_yr']:+.1f} -> "
                  f"net {nc['net_bps_yr']:+.1f} bps/yr (drag {nc['drag_bps_yr']:.1f}, "
                  f"net Sharpe {nc['net_sharpe']:+.2f})")
    else:
        print("(no _cache/usc_prices.csv — run data.fetch_prices() once to build the cache)")

    print("\n# Synthetic control — machinery proof (deterministic, no network; never evidence)")
    print("  the excess-Sharpe / HAC-mean pipeline must recover a PLANTED pickup and NOT invent one from 0.")
    for planted in (0.0, 40.0):
        w = data.synthetic_world(pickup_bps_yr=planted, seed=885)
        ex = (w["CREDIT"] - w["CASH"]).dropna()
        hm = st.hac_mean(ex)
        print(f"  planted={planted:+6.1f} bps/yr: excess mean {hm['mean_bps_yr']:+.1f} bps/yr  "
              f"HAC t = {hm['t_nw']:+.2f}  excess Sharpe {st.ann_sharpe(ex):+.2f}")


if __name__ == "__main__":
    main()
