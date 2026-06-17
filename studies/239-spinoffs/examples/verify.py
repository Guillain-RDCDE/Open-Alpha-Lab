"""Real-tape verification — Study 239 (Spinoffs). Regenerates docs/results.md numbers.

Reads from study-local cache (under studies/239-spinoffs/_cache/) by default.
Use --fetch to refresh from Yahoo! Finance.

    python studies/239-spinoffs/examples/verify.py
    python studies/239-spinoffs/examples/verify.py --fetch

Notes
-----
The spin-off event table is hardcoded in ``data.SPINOFF_TABLE``.  The table is
curated (notable large-cap spin-offs widely discussed at the time) — this is named
explicitly as a selection-bias caveat throughout the analysis.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spinoffs import data, strategy as st  # noqa: E402

_CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "_cache")
)


def main(fetch: bool) -> None:
    if fetch:
        print("Fetching spin-off prices from Yahoo Finance...")
        events, prices = data.fetch_spinoff_panel(fetch=True, cache_dir=_CACHE_DIR)
    else:
        events, prices = data.fetch_spinoff_panel(fetch=False, cache_dir=_CACHE_DIR)

    print(f"=== Spin-off events (n={len(events)}) ===")
    print(events[["child", "parent", "ex_date", "name"]].to_string(index=False))
    print()

    print(f"Data fingerprint: {data.fingerprint(events)}")
    print(f"Date range: {events['ex_date'].min().date()} to {events['ex_date'].max().date()}")
    print()

    print("=== Alpha (child vs SPY) by horizon ===")
    for col, lbl, h in zip(
        ("alpha_6m", "alpha_12m", "alpha_18m", "alpha_24m"),
        ("6m", "12m", "18m", "24m"),
        (126, 252, 378, 504),
    ):
        s = st.summarize_alpha(events, col=col)
        print(
            f"  {lbl}: n={s['n']}, child={s['mean_child_pct']:+.2f}%, "
            f"SPY={s['mean_spy_pct']:+.2f}%, alpha={s['mean_alpha_pct']:+.2f}%, "
            f"median={s['median_alpha_pct']:+.2f}%, win={s['win_rate']:.1%}, "
            f"HAC t={s['tstat']:+.2f}"
        )

    print()
    comp = st.compare_horizons(events)
    print("=== Full horizon comparison ===")
    print(comp.to_string(index=False))

    print()
    print("=== Cost-adjusted annualised alpha (12m horizon) ===")
    gross_12m = events["alpha_12m"].dropna().mean()
    for cost in (0.0, 5.0, 10.0, 20.0):
        ann = st.cost_adjusted_return(float(gross_12m), 252, cost_bps=cost)
        print(f"  cost={cost:.0f}bps: gross_alpha_12m={gross_12m*100:+.2f}%, net_ann={ann*100:+.1f}%/yr")

    print()
    print("=== VERDICT ===")
    s12 = st.summarize_alpha(events, col="alpha_12m")
    s6 = st.summarize_alpha(events, col="alpha_6m")
    print(f"  6m alpha={s6['mean_alpha_pct']:+.2f}%, HAC t={s6['tstat']:+.2f}")
    print(f"  12m alpha={s12['mean_alpha_pct']:+.2f}%, HAC t={s12['tstat']:+.2f}")
    print(f"  Signal: {'WEAK (barely clears |t|>=2 at 6/12m, curated table, n=14)' if abs(s12['tstat']) >= 2 else 'NONE'}")
    print(f"  Tradability: FRAGILE — n=14, cherry-picked table, extreme outliers, illiquid early days")
    print(f"  Fingerprint: {data.fingerprint(events)}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
