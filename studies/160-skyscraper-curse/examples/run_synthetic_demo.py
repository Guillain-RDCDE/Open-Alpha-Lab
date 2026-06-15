"""Offline, deterministic demo — Study 160 (Skyscraper-Curse).

No network. Uses the hardcoded event table and a synthetic annual S&P series to show
the study's spine in one screen: the crash knob shifts post-event returns only when
we deliberately plant a crash — on a null tape (crash_boost=0) the engine reads near
zero, confirming it is a faithful detector and that any real-tape result is a statement
about the data, not the method.

Run:
    python studies/160-skyscraper-curse/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from skyscraper_curse import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 160 — Skyscraper-Curse — synthetic spine demo\n")

    # The event table
    events_world = data.world_record_events(min_year=1930, max_year=2020)
    print("World-record completions in the demo window:")
    for _, row in events_world.iterrows():
        print(f"  {row['year']}  {row['building']:<30s}  {row['height_m']}m  ({row['city']})")
    print(f"\n  n = {len(events_world)} world-record events\n")

    event_years = events_world["year"].tolist()

    print(f"{'crash_boost':>12} | {'1yr mean':>10} {'3yr mean':>10} {'tstat(1yr)':>11} | "
          f"{'uncondit 1yr':>13} | beats unconditional?")
    print("-" * 80)

    for boost in (0.00, -0.15, -0.30, -0.50):
        df, truth = data.synthetic_annual(
            n_years=80, crash_boost=boost, seed=160,
            event_years=event_years, start_year=1930,
        )
        rets = pd.Series(df["log_ret"].values, index=df["year"], name="log_ret")

        fwd1 = st.event_forward_returns(rets, event_years, horizons=[1])
        fwd3 = st.event_forward_returns(rets, event_years, horizons=[3])

        s1 = st.summarize(fwd1["compound_ret"].values, label="1yr")
        s3 = st.summarize(fwd3["compound_ret"].values, label="3yr")
        unc = st.unconditional_stats(rets, horizon=1)

        beats = "YES" if s1["mean"] < unc["mean"] - 0.03 else "no"
        print(
            f"{boost:>12.2f} | {s1['mean']:>+10.3f} {s3['mean']:>+10.3f} "
            f"{s1['tstat']:>+11.2f} | {unc['mean']:>+13.3f} | {beats}"
        )

    print()
    print("The engine is a faithful detector: with planted crash_boost=-0.30+,")
    print("post-event returns drop below the unconditional baseline.")
    print("With crash_boost=0 (null) the effect is absent — the real-tape verdict")
    print("is a statement about the market, not the method.")
    print()
    print(
        f"INFERENCE BAR WARNING: n = {len(event_years)} world-record events "
        "spanning ~90 years means\n"
        "  any |t| >= 2 claim is based on ~6–9 data points — far below the power floor.\n"
        "  Even a 'big' observed effect is just sampling noise at this n.\n"
        "  Verdict: NONE (narrative, not signal)."
    )


if __name__ == "__main__":
    main()
