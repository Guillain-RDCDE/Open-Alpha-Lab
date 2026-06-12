"""Real-tape verification — Study 85 (Dr-Copper). Regenerates docs/results.md numbers.

Fetches (or reads from cache) the four daily tapes, runs the copper/gold ratio
predictive regressions at weekly and monthly horizons, computes OOS R², and
compares contemporaneous vs predictive correlations. Network is touched only
with --fetch.

    python studies/85-dr-copper/examples/verify.py            # cache-only
    python studies/85-dr-copper/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dr_copper import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "_cache")

    if fetch:
        print("=== Fetching real data and stamping fingerprints ===")
        for t in data.TICKERS:
            b = data.fetch_daily(t, fetch=True, cache_dir=cache_dir)
            print(f"{t:8s} {b.index[0].date()}..{b.index[-1].date()} rows={len(b)}")
        print()

    panel = data.load_panel(fetch=False, cache_dir=cache_dir)
    daily = data.build_ratio_frame(panel)
    print(f"Panel: {len(daily)} aligned daily rows, "
          f"{daily.index[0].date()} .. {daily.index[-1].date()}")
    fp = data.fingerprint(daily)
    print(f"Equity fingerprint: {fp}")
    print()

    for horizon, label in [("W", "WEEKLY"), ("ME", "MONTHLY")]:
        res = st.summarize(daily, horizon=horizon)
        print(f"=== {label} REGRESSION (horizon={horizon}) ===")
        print(f"Periods: {res['n_periods']}")

        print("\nIn-sample: copper/gold delta → fwd equity")
        is_e = res["is_equity"]
        print(f"  beta={is_e['beta']:+.4f}  HAC t={is_e['tstat']:+.3f}  R²={is_e['r2']:.4f}"
              f"  (naive t={is_e['tstat_naive']:+.3f})  n={is_e['n']}")

        print("\nIn-sample: copper/gold delta → fwd yield change")
        is_y = res["is_yield"]
        print(f"  beta={is_y['beta']:+.4f}  HAC t={is_y['tstat']:+.3f}  R²={is_y['r2']:.4f}"
              f"  (naive t={is_y['tstat_naive']:+.3f})  n={is_y['n']}")

        print("\nContemporaneous: copper/gold delta ~ same-period equity ret")
        c = res["contemp"]
        print(f"  beta={c['beta']:+.4f}  HAC t={c['tstat']:+.3f}  R²={c['r2']:.4f}")

        print("\nOOS R² vs historical mean (equity):")
        oos_e = res["oos_equity"]
        print(f"  OOS R²={oos_e['r2_oos']:+.5f}  DM t={oos_e['dm_tstat']:+.3f}"
              f"  n_oos={oos_e['n_oos']}")

        print("\nOOS R² vs historical mean (yield):")
        oos_y = res["oos_yield"]
        print(f"  OOS R²={oos_y['r2_oos']:+.5f}  DM t={oos_y['dm_tstat']:+.3f}"
              f"  n_oos={oos_y['n_oos']}")
        print()


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
