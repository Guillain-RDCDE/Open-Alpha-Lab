"""Real-tape verification — Study 319 (Lockup-Expiry). Regenerates docs/results.md numbers.

Builds the event-time abnormal-return panel for the hardcoded IPO basket (cache-first),
computes the average abnormal return at the lock-up-expiry bar and cumulative abnormal
returns over windows around it, with HAC t-stats and block-bootstrap CIs, then runs the
believers' market-hedged short net of two-leg costs.

    python studies/319-lockup-expiry/examples/verify.py            # cache-only
    python studies/319-lockup-expiry/examples/verify.py --fetch    # refresh from yfinance

Caveats: survivorship bias (basket = tickers still on yfinance), beta=1 market control,
small mixed-vintage n, and an event date that is calendar-known in advance (anticipatable).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lockup_expiry import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    panel = data.fetch_event_panel(fetch=fetch)
    n_events = panel["event"].nunique()
    print(f"=== Event panel: {n_events} IPOs, tau in [{panel['tau'].min()},{panel['tau'].max()}] ===")
    print(f"Events: {sorted(panel['event'].unique())}")
    print()

    a = st.aar(panel)
    row0 = a[a["tau"] == 0].iloc[0]
    print("=== Expiry-bar abnormal return (tau=0) ===")
    print(f"  AAR(0) = {row0['aar']*1e4:+.1f} bps   t = {row0['tstat']:+.2f}   n = {int(row0['n'])}")
    print()

    print("=== Cumulative abnormal return over windows ===")
    print(f"  {'window':<10} {'n':>3} {'mean_bps':>9} {'med_bps':>9} {'sag%':>5} {'HAC_t':>7} {'CI_bps':>16}")
    for lo, hi in [(-5, -1), (-1, 1), (0, 5), (0, 10), (-5, 5)]:
        s = st.car(panel, lo=lo, hi=hi, n_boot=3000)
        print(f"  {s['window']:<10} {s['n']:>3} {s['mean_car_bps']:>+9.1f} "
              f"{s['median_car_bps']:>+9.1f} {s['win_rate']*100:>4.0f}% {s['tstat']:>+7.2f} "
              f"[{s['ci_lo_bps']:>+5.0f},{s['ci_hi_bps']:>+5.0f}]")
    print()

    print("=== Believers' market-hedged short, post-expiry window [0,+5] ===")
    for c in (0.0, 10.0, 30.0):
        sh = st.tradable_short(panel, lo=0, hi=5, cost_bps=c)
        print(f"  cost={c:>4.0f}bps/leg  gross={sh['gross_bps']:>+8.1f}  net={sh['net_bps']:>+8.1f}  "
              f"t_net={sh['tstat_net']:>+5.2f}  win={sh['win_rate']:.2f}")
    print()

    print("=== Verdict ===")
    sag_windows = [(-5, -1), (-1, 1), (0, 5)]
    real_sag = any(
        (lambda s: s["mean_car_bps"] < 0 and s["tstat"] <= -2.0)(st.car(panel, lo=lo, hi=hi, n_boot=1000))
        for lo, hi in sag_windows
    )
    print(f"  Signal: {'REAL sag' if real_sag else 'NONE — no window clears t<=-2 with the predicted sign'}")
    print(f"  Tradability: Mirage — folklore short loses gross & worse net")
    print(f"  Supply-shock sag? Not supported on this 2019-2024 basket (anticipated/arbitraged)")
    print()
    print(f"  Panel fingerprint: {data.fingerprint(panel)}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
