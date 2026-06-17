"""Real-data verification -- Study 271 (Cardboard-Box). Regenerates docs/results.md numbers.

Loads the cached ^GSPC annual price returns (this study's _cache/gspc_annual.parquet),
joins the hardcoded box/rail-freight growth table with a one-year forecast lag, and
runs the full analysis: Newey-West predictive regression (box & rail), the coincident
control, and the cost-charged timing backtest.

The ^GSPC parquet is cache-only by default (no network). To (re)populate it once::

    python -c "import sys; sys.path.insert(0,'studies/271-cardboard-box'); \
        from cardboard_box import data; data.fetch_gspc_annual(fetch=True)"

Then run from anywhere::

    python studies/271-cardboard-box/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cardboard_box import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 271 -- Cardboard-Box -- real-data verification\n")
    print("Data: ^GSPC annual price returns (cache: _cache/gspc_annual.parquet)")
    print("      Box/rail-freight growth: hardcoded in data.py (1970-2024)\n")

    gspc = data.fetch_gspc_annual(fetch=False)  # cache-only
    freight = data.freight_table()

    frame = data.build_forecast_frame(freight, gspc, predictor="box_growth")
    fp = data.fingerprint(frame)
    print(f"Forecast frame: ret years {frame['ret_year'].min()}-{frame['ret_year'].max()}, "
          f"n={len(frame)}, fingerprint={fp}\n")

    s = st.summarize(freight, gspc, cost_bps=10.0, lags=1)

    print("=== Forecast regression: box growth_t -> S&P return_{t+1} ===")
    print(f"  slope beta = {s['box_beta']:+.4f}   HAC t = {s['box_t']:+.3f}")
    print(f"  R^2 = {s['box_r2']:.4f}   Pearson r = {s['box_pearson_r']:+.4f} "
          f"(p = {s['box_pearson_p']:.3f})\n")

    print("=== Forecast regression: rail growth_t -> S&P return_{t+1} ===")
    print(f"  slope beta = {s['rail_beta']:+.4f}   HAC t = {s['rail_t']:+.3f}   "
          f"R^2 = {s['rail_r2']:.4f}\n")

    print("=== Coincident control: box growth_t -> S&P return_t ===")
    print(f"  HAC t = {s['coin_box_t']:+.3f}   R^2 = {s['coin_box_r2']:.4f}   "
          f"r = {s['coin_box_pearson_r']:+.4f}\n")

    print("=== Timing strategy (long next year iff box growth>0, 10 bps one-way) ===")
    print(f"  Unconditional up-rate: {s['uncond_up_pct']:.1f}%")
    print(f"  Timer hit-rate:        {s['hit_rate_pct']:.1f}%")
    print(f"  Turnover:              {s['turnover']:.4f} flips/yr  "
          f"(invested {s['frac_invested']*100:.1f}% of years)")
    print(f"  Buy & hold:            {s['bah_ann_pct']:+.2f}%/yr (price-only)")
    print(f"  Box timer GROSS:       {s['gross_ann_pct']:+.2f}%/yr")
    print(f"  Box timer NET:         {s['net_ann_pct']:+.2f}%/yr")
    print(f"  Net minus buy-and-hold:{s['net_minus_bah_pct']:+.2f}pp/yr\n")

    print("Verdict: Signal=NONE, Tradability=MIRAGE -- a coincident GDP gauge, "
          "not a market forecast.")


if __name__ == "__main__":
    main()
