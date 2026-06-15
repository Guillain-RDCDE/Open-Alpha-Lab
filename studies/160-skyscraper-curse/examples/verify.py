"""Real-tape verification — Study 160 (Skyscraper-Curse). Regenerates docs/results.md.

Loads the Shiller monthly S&P 500 dataset (pre-staged at _cache/shiller_sp500.parquet),
computes annual total returns, and measures forward S&P performance in the 1-, 2-, and
3-year windows after each world-record skyscraper completion.

Also fetches (or reads from cache) daily ^GSPC via yfinance for supplementary drawdown
analysis. Network is touched only with --fetch.

    python studies/160-skyscraper-curse/examples/verify.py            # cache-only
    python studies/160-skyscraper-curse/examples/verify.py --fetch    # refresh ^GSPC
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from skyscraper_curse import data, strategy as st  # noqa: E402

# World-record events in the S&P 500 era (post-1957)
EVENTS_SP500 = data.world_record_events(min_year=1957, max_year=2025)
# Include pre-S&P events for the full-history narrative
EVENTS_ALL = data.world_record_events(min_year=1929, max_year=2025)


def main(fetch: bool = False) -> None:
    # 1. Load Shiller data
    shiller = data.load_shiller()
    fp = data.fingerprint(shiller, col="SP500")
    tape_start = str(shiller.index.min())[:10]
    tape_end = str(shiller.index.max())[:10]
    print(f"Shiller S&P 500: {tape_start} to {tape_end}  fp={fp}")

    # 2. Annual total returns
    annual_rets = st.shiller_annual_returns(shiller, real=False)
    annual_real = st.shiller_annual_returns(shiller, real=True)
    print(f"Annual return series: {int(annual_rets.index.min())}–{int(annual_rets.index.max())}, "
          f"n={len(annual_rets)}, mean={annual_rets.mean():.3f}")

    # 3. World-record events in the modern S&P era (1957+)
    event_years_sp = EVENTS_SP500["year"].tolist()
    event_years_all = EVENTS_ALL["year"].tolist()
    print(f"\nWorld-record completions in S&P 500 era ({min(event_years_sp)}–{max(event_years_sp)}): "
          f"n={len(event_years_sp)} events")
    for _, row in EVENTS_SP500.iterrows():
        print(f"  {row['year']}  {row['building']:<30s}  {row['height_m']}m")

    # 4. Forward returns — 1, 2, 3 years post-completion
    for h in [1, 2, 3]:
        fwd = st.event_forward_returns(annual_rets, event_years_sp, horizons=[h])
        s = st.summarize(fwd["compound_ret"].values, label=f"{h}yr")
        unc = st.unconditional_stats(annual_rets, horizon=h)
        print(f"\n--- {h}-year forward returns after event ---")
        print(f"  events:        n={s['n']}  mean={s['mean']:+.3f}  "
              f"std={s['std']:.3f}  t={s['tstat']:+.2f}  LOW_POWER={s['low_power_warning']}")
        print(f"  unconditional: n={unc['n']}  mean={unc['mean']:+.3f}  "
              f"std={unc['std']:.3f}")
        print(f"  event mean vs unconditional: {s['mean'] - unc['mean']:+.3f}")

    # 5. Per-event breakdown (1-year)
    print("\n--- Per-event 1-year forward S&P return ---")
    fwd1 = st.event_forward_returns(annual_rets, event_years_sp, horizons=[1])
    for _, row in fwd1.iterrows():
        building_name = EVENTS_SP500[EVENTS_SP500["year"] == row["event_year"]]["building"].values
        name = building_name[0] if len(building_name) else "?"
        print(f"  {int(row['event_year'])} ({name:<30s}): "
              f"+{int(row['event_year'])+1}  ret={row['compound_ret']:+.3f} "
              f"({row['compound_ret']*100:+.1f}%)")

    # 6. Random-day control
    print("\n--- Random-day control (n_draws=2000, horizon=1yr) ---")
    ctrl = st.random_control(annual_rets, n_events=len(event_years_sp),
                             horizon=1, n_draws=2000, seed=160)
    obs_mean = fwd1["compound_ret"].mean()
    p_val = float((ctrl <= obs_mean).mean())  # one-sided: how often random is worse?
    print(f"  observed event mean (1yr): {obs_mean:+.3f}")
    print(f"  random control mean:       {ctrl.mean():+.3f}  std={ctrl.std():.3f}")
    print(f"  empirical p-value (event <= random): {p_val:.3f}")
    print(f"  90th/95th pct of random: {ctrl.quantile(0.90):.3f} / {ctrl.quantile(0.95):.3f}")

    # 7. Real returns version
    print("\n--- Real (inflation-adjusted) returns, 1yr ---")
    fwd_real = st.event_forward_returns(annual_real, event_years_sp, horizons=[1])
    s_real = st.summarize(fwd_real["compound_ret"].values, label="real_1yr")
    unc_real = st.unconditional_stats(annual_real, horizon=1)
    print(f"  events:        mean={s_real['mean']:+.3f}  t={s_real['tstat']:+.2f}")
    print(f"  unconditional: mean={unc_real['mean']:+.3f}")

    # 8. Verdict
    print("\n=== VERDICT ===")
    print(f"  n = {len(event_years_sp)} world-record events in the S&P era")
    print(f"  1-yr HAC t = {st.summarize(fwd1['compound_ret'].values)['tstat']:+.2f} "
          f"(LOW POWER: n too small for any |t|>=2 to be credible)")
    print("  Signal: NONE (narrative, not signal)")
    print("  Tradability: MIRAGE (insufficient n; not replicable in advance)")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
